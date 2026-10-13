"""End-to-end training: cohort → features → plan → develop → evaluate → artifact.

Order of operations, and why it is this order
----------------------------------------------
1. **Plan the split before anything is fitted.** :func:`trs.validation.protocol.
   plan_training` fixes a development partition, a locked evaluation partition
   and the forward-chaining folds inside development, and writes a manifest
   whose hash covers every row id, boundary and eligibility rule. The protocol
   it settles on is named in the artifact, together with every protocol it
   rejected and the measured reason.
2. **Develop without the evaluation partition.** Hyperparameter selection,
   family selection and calibration all live in :mod:`trs.pipeline.develop` and
   consume development rows only. Each fold fits its own feature transformer on
   its own permitted training rows, so imputation constants and categorical
   vocabularies are fold-local rather than shared across the whole training
   partition.
3. **Select on development out-of-fold predictions.** The previous version
   passed the *test* labels to :func:`trs.pipeline.select.choose_family` and
   then reported the winner on those same rows, which made the test partition a
   selection partition and every headline figure a selection-set figure. The
   selection criterion is now pooled forward-chaining out-of-fold log loss over
   development.
4. **Evaluate once.** Exactly one function reads the evaluation labels, after
   the artifact's family, hyperparameters, coefficients and calibrator are all
   fixed. Diagnostics that refit — the domain ablation, the interval check, the
   target-grouped sensitivity — run on development, so the locked partition is
   read a single time.
5. **Derive a status, do not assume one.** :mod:`trs.pipeline.status` turns the
   evaluation into a status against versioned criteria. Registration and
   activation are separate: a synthetic-trained model is registered and capped
   at ``demo_only`` whatever its metrics say.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.cohort.outcomes import PRIMARY_ENDPOINT, EndpointDefinition
from trs.db.base import utcnow
from trs.db.models.analytics import CohortEntry, CohortVersion, FeatureSnapshot
from trs.db.models.core import Program, ProgramIndication
from trs.features.spec import FEATURE_SCHEMA_VERSION
from trs.modeling.calibration import select_calibrator
from trs.modeling.logistic import L2LogisticRisk
from trs.pipeline.artifacts import (
    ModelBundle,
    capture_git_commit,
    capture_library_versions,
    next_version,
    register_artifact,
)
from trs.pipeline.develop import (
    DEFAULT_L2_GRID,
    apply_fit,
    calibration_scale_audit,
    cross_fitted_calibrated,
    default_hyperparameters,
    fit_on_development,
    grouped_oof_predictions,
    out_of_fold_predictions,
    tune_group_precision_on_folds,
    tune_l2_on_folds,
)
from trs.pipeline.features import materialise_cohort_snapshots, snapshot_rows
from trs.pipeline.status import PROMOTION_CRITERIA, ModelStatus, derive_status
from trs.scoring.contract import PREDICTION_CONTRACT_VERSION, PRIMARY_FUNCTIONAL
from trs.validation.metrics import (
    deployable_reference_prediction,
    evaluate,
    log_loss,
)
from trs.validation.protocol import (
    Protocol,
    ProtocolError,
    TrainingPlan,
    plan_training,
)
from trs.validation.report import (
    assemble_report,
    coefficient_sign_audit,
    completeness_tertiles,
    domain_ablation,
    era_labels,
    interval_agreement,
    subgroup_metrics,
)
from trs.validation.splitters import (
    GROUP_SCOPE_KEYS,
    ConnectedComponentGrouper,
    GroupScope,
    build_grouping_keys,
    grouped_kfold,
    leakage_audit,
)


def _dataset_hash(X: np.ndarray, y: np.ndarray, columns: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(list(columns)).encode())
    h.update(np.ascontiguousarray(np.round(X, 10)).tobytes())
    h.update(np.ascontiguousarray(y.astype(np.int8)).tobytes())
    return h.hexdigest()


def train(
    session: Session,
    *,
    cohort_version_id: str | None = None,
    endpoint: EndpointDefinition = PRIMARY_ENDPOINT,
    development_fraction: float = 0.76,
    data_cutoff: dt.date | None = None,
    n_development_folds: int = 4,
    protocol_preference: Sequence[Protocol] = (
        Protocol.STRICT_TEMPORAL,
        Protocol.TEMPORAL_OUTER_MATURITY,
        Protocol.RETROSPECTIVE,
    ),
    candidate_families: Sequence[str] = ("l2_logistic", "hierarchical_logistic", "gbdt"),
    n_bootstrap: int = 600,
    random_state: int = 20240917,
    make_active: bool = True,
    run_interval_check: bool = True,
    run_ablation: bool = True,
    register_candidates: bool = True,
    verbose: bool = True,
) -> ModelBundle:
    """Train, validate and register a model for one endpoint.

    ``make_active`` is a *request*, not a decision. An artifact is activated
    only if its derived status permits serving; a synthetic-trained model is
    capped at ``demo_only`` and is activated so the demonstration surfaces have
    something to show, while the eligibility policy in
    :mod:`trs.pipeline.status` still refuses to score real evidence with it.
    """
    started = time.monotonic()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ---- 1. Cohort ---------------------------------------------------------
    if cohort_version_id:
        cohort = session.get(CohortVersion, cohort_version_id)
    else:
        cohort = session.scalars(
            select(CohortVersion).order_by(CohortVersion.built_at.desc()).limit(1)
        ).first()
    if cohort is None:
        raise RuntimeError("No cohort version available; build one first.")

    entries = list(
        session.scalars(
            select(CohortEntry)
            .where(
                CohortEntry.cohort_version_id == cohort.id,
                CohortEntry.label_primary.is_not(None),
            )
            .order_by(CohortEntry.as_of_date, CohortEntry.id)
        )
    )
    if len(entries) < 40:
        raise RuntimeError(
            f"Cohort {cohort.name}/{cohort.version} has only {len(entries)} labelled "
            "entries. Training below 40 would produce a model whose validation "
            "statistics are not interpretable."
        )
    log(f"Cohort {cohort.name}/{cohort.version}: {len(entries)} labelled entries")

    # ---- 2. Features -------------------------------------------------------
    snapshots = materialise_cohort_snapshots(session, entries, progress=verbose)
    by_track = {(s.program_indication_id, s.as_of_date): s for s in snapshots}
    ordered: list[FeatureSnapshot] = [
        by_track[(e.program_indication_id, e.as_of_date)] for e in entries
    ]
    rows = snapshot_rows(ordered)
    y = np.asarray([e.label_primary for e in entries], dtype=int)
    dates = [e.as_of_date for e in entries]
    completeness = [s.completeness_score for s in ordered]

    # Compound ids are part of the identity scope: the same molecule under two
    # program records is the same asset, whatever it is called.
    compound_by_track = dict(
        session.execute(
            select(ProgramIndication.id, Program.primary_compound_id).join(
                Program, Program.id == ProgramIndication.program_id
            )
        ).all()
    )
    keys = build_grouping_keys(
        entries, [compound_by_track.get(e.program_indication_id) for e in entries]
    )
    grouper = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY])
    components = grouper.fit(keys)
    comp_summary = grouper.summary(components)
    comp_summary["scope"] = str(GroupScope.IDENTITY)
    log(
        f"Grouping ({GroupScope.IDENTITY}): {comp_summary['n_components']} "
        f"connected components from {comp_summary['n_rows']} rows "
        f"(largest {comp_summary['largest_component']})"
    )

    # ---- 3. Plan the split, before anything is fitted ----------------------
    # Label availability is a separate question from feature availability, and
    # the predecessor enforced only the second. A programme's outcome date is
    # when its label became knowable; a fit that claims a historical origin may
    # not use labels that resolved after it.
    outcome_date_by_track = dict(
        session.execute(
            select(ProgramIndication.id, ProgramIndication.outcome_date)
        ).all()
    )
    label_available_on = [
        outcome_date_by_track.get(e.program_indication_id) for e in entries
    ]

    try:
        plan: TrainingPlan = plan_training(
            [e.id for e in entries],
            dates,
            label_available_on,
            components,
            preference=protocol_preference,
            development_fraction=development_fraction,
            data_cutoff=data_cutoff,
            grouping_scope=GroupScope.IDENTITY,
            n_folds=n_development_folds,
        )
    except ProtocolError as exc:
        raise RuntimeError(
            f"No declared evaluation protocol fits this cohort.\n{exc}\n"
            "This is a statement about the data, not a bug. Do not lower the "
            "thresholds in trs.validation.protocol to make it pass."
        ) from exc

    manifest = plan.manifest
    dev, ev = manifest.development_index, manifest.evaluation_index
    log(
        f"Protocol {manifest.protocol}: development {dev.size}, "
        f"evaluation {ev.size}, origin {manifest.dates.fitting_origin}, "
        f"manifest {manifest.manifest_hash[:12]}"
    )
    for rejected in plan.protocol_trace:
        if "rejected_because" in rejected:
            log(f"  ✗ {rejected['protocol']}: {rejected['rejected_because'][:160]}")
    log(f"  purged: {manifest.purge_counts}")
    log(f"  folds: {[(f.index, f.train.size, f.test.size) for f in plan.folds]}")
    if not manifest.permits_deployment_claim:
        log(
            "  ⚠ this protocol may NOT describe its numbers as deployment "
            "performance; see the limitations block"
        )

    audit = leakage_audit(components, dev, ev, keys, scope=GroupScope.IDENTITY)
    if not audit["clean"]:
        raise RuntimeError(
            f"Leakage audit failed across the development/evaluation "
            f"boundary: {audit}"
        )

    mech = np.asarray([k.get("mechanism") or "unknown" for k in keys])

    # The deployable reference: an intercept-only model fitted on development
    # labels. Not the evaluation set's own prevalence, which is the best
    # constant in hindsight and so not a competitor anyone could ship.
    reference_p = deployable_reference_prediction(y[dev])
    log(
        f"Deployable reference (development rate): {reference_p:.4f}; "
        f"evaluation prevalence {float(np.mean(y[ev])):.4f}"
    )

    # ---- 4. Development: tune on forward-chaining folds -------------------
    best_l2, tune_trace = tune_l2_on_folds(
        rows, y, plan.folds, grid=DEFAULT_L2_GRID, random_state=random_state
    )
    tune_summary = next((r for r in tune_trace if "selected_l2_penalty" in r), {})
    log(f"Selected L2 penalty {best_l2} on fold-local forward-chaining folds")
    if tune_summary.get("optimum_at_grid_boundary"):
        log(f"  ⚠ {tune_summary['warning']}")

    # ---- 5. Development: out-of-fold predictions, calibration, selection --
    candidate_results: list[dict[str, Any]] = []
    developed: dict[str, dict[str, Any]] = {}

    for family in candidate_families:
        hp = default_hyperparameters(family, best_l2, random_state)
        gp_trace: list[dict[str, Any]] = []
        if family == "hierarchical_logistic":
            best_gp, gp_trace = tune_group_precision_on_folds(
                rows, y, plan.folds, mech, best_l2, random_state=random_state
            )
            hp["group_precision"] = best_gp

        try:
            # Two out-of-fold pools, for two different jobs.
            #
            # Forward-chaining: every prediction is genuinely forward in time,
            # which is what selection and drift diagnosis need. But its folds
            # only test rows after the first training block, so the pool
            # over-weights the later development years and its base rate is not
            # the development base rate.
            #
            # Grouped: covers every development row, so its level is correct by
            # construction, which is what fitting a recalibration *intercept*
            # needs. It is not forward in time and cannot speak to drift.
            oof = out_of_fold_predictions(
                family, rows, y, plan.folds, mech, hp, random_state=random_state
            )
            grouped_oof = grouped_oof_predictions(
                family, rows, y, dev, components, mech, hp,
                random_state=random_state,
            )
            fit = fit_on_development(
                family, rows, y, dev, mech, hp, random_state=random_state
            )
            final_dev_logit = apply_fit(fit, rows, dev, mech)

            # Choose the calibration pool on measured scale consistency rather
            # than on principle. The audit checks that the pool the calibrator
            # is fitted on has the level and the logit scale of the predictions
            # it will be applied to; a pool that fails leaves the recalibration
            # intercept fitted to the wrong thing, which cost 0.19 nats of
            # evaluation log loss when the forward-chaining pool was used alone.
            pools = {
                "forward_chaining_out_of_fold": oof,
                "grouped_out_of_fold_full_coverage": grouped_oof,
            }
            audits = {
                name: calibration_scale_audit(
                    pool.logit, y[pool.index], final_dev_logit, y[dev]
                )
                for name, pool in pools.items()
            }
            chosen_pool_name = min(
                pools,
                key=lambda n: (
                    audits[n]["verdict"] != "consistent",
                    abs(audits[n]["pool_rate_minus_development_rate"]),
                    abs(audits[n]["mean_logit_shift"]),
                ),
            )
            cal_pool = pools[chosen_pool_name]
            calibrator, cal_record = select_calibrator(
                cal_pool.logit, y[cal_pool.index], random_state=random_state
            )
            cal_record = {
                **cal_record,
                "design": chosen_pool_name,
                "design_chosen_by": (
                    "calibration scale audit: the pool whose observed rate and "
                    "logit scale most nearly match the predictions the "
                    "calibrator will be applied to"
                ),
                "n_out_of_fold_predictions": cal_pool.n,
                "n_development_rows": int(dev.size),
                "rows_without_out_of_fold_prediction": int(dev.size - cal_pool.n),
                "scale_audit": audits[chosen_pool_name],
                "rejected_designs": {
                    n: audits[n] for n in pools if n != chosen_pool_name
                },
            }
            # Selection metrics come from a *cross-fitted* calibration: the
            # pooled calibrator above is what will be deployed, but applying it
            # to the rows it was fitted on forces the calibration slope to 1.0
            # for every candidate and disables the selection gate's calibration
            # band. These predictions measure; the pooled one serves.
            p_sel, sel_pos, xfit_record = cross_fitted_calibrated(
                oof.logit, y[oof.index], components[oof.index],
                random_state=random_state,
            )
            sel_index = oof.index[sel_pos]
            cal_record["selection_calibration"] = xfit_record
            dev_metrics = evaluate(
                y[sel_index], p_sel, components=components[sel_index],
                n_bootstrap=min(n_bootstrap, 300), random_state=random_state,
                reference_prediction=reference_p,
            )
        except Exception as exc:  # noqa: BLE001
            candidate_results.append(
                {"family": family, "error": str(exc), "eligible": False}
            )
            log(f"  {family}: FAILED ({exc})")
            continue

        developed[family] = {
            "fit": fit,
            "calibrator": calibrator,
            "oof": oof,
            "selection_index": sel_index,
            "selection_predictions": p_sel,
            "calibration_record": cal_record,
        }
        candidate_results.append({
            "family": family,
            "eligible": True,
            "hyperparameters": hp,
            "tuning_trace": gp_trace,
            "calibration": cal_record,
            "selection_basis": "development_forward_chaining_out_of_fold",
            "log_loss": dev_metrics["accuracy"]["log_loss"]["value"],
            "brier": dev_metrics["accuracy"]["brier"]["value"],
            "scaled_brier": dev_metrics["accuracy"]["scaled_brier"]["value"],
            "auroc": dev_metrics["discrimination"]["auroc"]["value"],
            "calibration_slope": dev_metrics["calibration"]["calibration_slope"]["value"],
            "supports_exact_attribution": fit.model.supports_exact_attribution,
            "supports_analytic_posterior": fit.model.supports_analytic_posterior,
        })
        log(
            f"  {family}: development OOF log-loss "
            f"{dev_metrics['accuracy']['log_loss']['value']:.4f}  "
            f"AUROC {dev_metrics['discrimination']['auroc']['value']}  "
            f"slope {dev_metrics['calibration']['calibration_slope']['value']}"
        )

    if not developed:
        raise RuntimeError(
            "Every candidate family failed during development:\n"
            + "\n".join(f"  {c['family']}: {c.get('error')}" for c in candidate_results)
        )

    # Selection consumes development out-of-fold predictions on the rows all
    # surviving families actually predicted. The evaluation partition is not an
    # argument to this call and cannot become one.
    shared_index = developed[next(iter(developed))]["selection_index"]
    for spec in developed.values():
        shared_index = np.intersect1d(shared_index, spec["selection_index"])
    # Every family's selection index is sorted ascending, so masking to the
    # shared rows preserves alignment without a per-row search.
    selection_predictions = {
        family: spec["selection_predictions"][
            np.isin(spec["selection_index"], shared_index)
        ]
        for family, spec in developed.items()
    }
    for family, preds in selection_predictions.items():
        if preds.shape != shared_index.shape:
            raise RuntimeError(
                f"family {family!r} contributed {preds.size} selection "
                f"predictions for {shared_index.size} shared rows; the "
                "comparison would be misaligned"
            )

    from trs.pipeline.select import choose_family

    selection = choose_family(
        candidate_results,
        y[shared_index],
        selection_predictions,
        components[shared_index],
        random_state=random_state,
    )
    selection["selection_partition"] = {
        "basis": (
            "development forward-chaining out-of-fold predictions, with "
            "cross-fitted calibration so no row is scored by a calibrator "
            "fitted on it"
        ),
        "n_rows": int(shared_index.size),
        "reads_evaluation_labels": False,
        "note": (
            "Family selection, hyperparameter selection and calibration all "
            "read development labels only. The evaluation partition is read "
            "once, after this point, by a single call."
        ),
    }
    chosen = selection["chosen_family"]
    log(f"Selected family: {chosen} — {selection['rationale']}")

    chosen_spec = developed[chosen]
    fit = chosen_spec["fit"]
    model = fit.model
    transformer = fit.transformer
    columns = fit.columns
    calibrator = chosen_spec["calibrator"]
    chosen_record = next(c for c in candidate_results if c["family"] == chosen)

    # The development-fitted encoding, used for diagnostics that refit. It is
    # applied to evaluation rows below, which is correct: evaluation
    # contributed nothing to fitting it.
    X = transformer.transform(rows)

    # ---- 6. Evaluate, once ------------------------------------------------
    manifest.verify(dates, label_available_on)
    eval_logit = apply_fit(fit, rows, ev, mech)
    p_eval = calibrator.transform(eval_logit)
    final_metrics = evaluate(
        y[ev], p_eval, components=components[ev],
        n_bootstrap=n_bootstrap, random_state=random_state,
        reference_prediction=reference_p,
    )
    final_metrics["evaluated_functional"] = str(PRIMARY_FUNCTIONAL)
    final_metrics["protocol"] = manifest.protocol
    final_metrics["permits_deployment_claim"] = manifest.permits_deployment_claim
    log(
        f"Evaluation ({ev.size} rows, read once): log-loss "
        f"{final_metrics['accuracy']['log_loss']['value']:.4f}  "
        f"AUROC {final_metrics['discrimination']['auroc']['value']}  "
        f"slope {final_metrics['calibration']['calibration_slope']['value']}"
    )
    ref_block = final_metrics.get("reference") or {}
    log(
        f"  vs deployable reference {reference_p:.4f}: Brier skill "
        f"{ref_block.get('brier_skill')}, better = "
        f"{ref_block.get('model_better_than_reference')}"
    )

    # ---- 7. Diagnostics ----------------------------------------------------
    # Subgroup metrics are a decomposition of the single evaluation pass, not a
    # fresh look: they re-partition `p_eval`, which is already computed, and
    # refit nothing.
    subgroups = subgroup_metrics(
        y[ev], p_eval, components[ev],
        {
            "modality_group": [rows[i].get("modality_group") for i in ev],
            "target_class_group": [rows[i].get("target_class_group") for i in ev],
            "evidence_completeness": [
                completeness_tertiles(completeness)[i] for i in ev
            ],
            "era": [era_labels(dates)[i] for i in ev],
        },
        n_bootstrap=min(n_bootstrap, 200),
    )

    sign_audit = (
        coefficient_sign_audit(model.coefficient_table(), transformer)
        if hasattr(model, "coefficient_table") else {}
    )
    if sign_audit.get("n_discordant_credible"):
        log(
            f"  ⚠ {sign_audit['n_discordant_credible']} coefficient(s) credibly "
            "oppose their pre-registered direction — flagged for review"
        )

    # Both of the diagnostics below refit models, so both run on development
    # rows only. Probing the evaluation partition 12 times for an ablation and
    # 40 times for an interval check would turn a single look into 52.
    interval_check: dict[str, Any] = {}
    if run_interval_check and model.family == "l2_logistic":
        probe = X[dev][: min(30, dev.size)]
        interval_check = interval_agreement(
            lambda: L2LogisticRisk(l2_penalty=best_l2),
            transformer, None, X[dev], y[dev], components[dev], probe,
            n_refits=40, random_state=random_state,
        )
        log(f"  interval agreement (development): {interval_check.get('verdict')}")

    ablation: dict[str, Any] = {}
    if run_ablation:
        ablation = _run_ablation(
            rows, y, plan, columns, components, best_l2, random_state
        )

    # ---- 7b. Target-grouped sensitivity analysis ---------------------------
    # The headline split enforces identity only, because an analyst at a
    # historical decision point genuinely knew other sponsors' outcomes on the
    # same target. This second evaluation removes that knowledge entirely, so
    # the gap between the two quantifies how much of the model's performance is
    # target memorisation rather than evidence reading.
    target_grouped = _target_grouped_sensitivity(
        X, y, keys, columns, best_l2, random_state, n_bootstrap=min(n_bootstrap, 300)
    )
    if target_grouped.get("available"):
        log(
            f"  target-grouped sensitivity: AUROC "
            f"{target_grouped['auroc']} vs headline "
            f"{final_metrics['discrimination']['auroc']['value']}"
        )

    # ---- 8. Bundle ---------------------------------------------------------
    version = next_version(session, endpoint.name)
    dataset_hash = _dataset_hash(X, y, columns)
    trained_on_synthetic = any(s.is_synthetic for s in ordered) or bool(cohort.contains_synthetic)

    report = assemble_report(
        split_description=(
            f"Protocol {manifest.protocol}: development rows assessed on or "
            f"before {manifest.dates.assessment_end.isoformat()} with outcomes "
            f"knowable by {manifest.dates.label_cutoff.isoformat()}; evaluation "
            "rows assessed strictly after, read once. Connected components kept "
            "whole."
        ),
        split_sizes={
            "development": int(dev.size),
            "evaluation": int(ev.size),
            "purged": len(manifest.purged),
            "purged_by_reason": manifest.purge_counts,
            "n_selection_rows": int(shared_index.size),
        },
        leakage_audit_result=audit,
        component_summary=comp_summary,
        test_metrics=final_metrics,
        calibration_selection=chosen_record["calibration"],
        subgroups=subgroups,
        sign_audit=sign_audit,
        interval_check=interval_check,
        ablation=ablation,
        model_comparison=selection,
        cohort_summary={
            "name": cohort.name,
            "version": cohort.version,
            "endpoint": endpoint.to_dict(),
            "n_entries": cohort.n_entries,
            "n_positive": cohort.n_positive,
            "n_negative": cohort.n_negative,
            "n_excluded": cohort.n_excluded,
            "inclusion_summary": cohort.inclusion_summary,
            "as_of_rule": cohort.as_of_rule,
            "contains_synthetic": cohort.contains_synthetic,
        },
        limitations=_limitations(
            trained_on_synthetic, comp_summary, manifest, interval_check
        ),
    )
    report["target_grouped_sensitivity"] = target_grouped
    report["hyperparameter_selection"] = {
        "l2_grid": list(DEFAULT_L2_GRID),
        "design": (
            "Expanding-window folds inside the training partition only. Each "
            "fold trains on everything before a cut date and evaluates on the "
            "next slice, mirroring deployment. Hyperparameter selection never "
            "sees the calibration or test partitions."
        ),
        "trace": tune_trace,
    }
    report["usefulness"] = _usefulness_verdict(final_metrics)
    report["protocol"] = plan.to_dict()
    report["prediction_contract"] = {
        "version": PREDICTION_CONTRACT_VERSION,
        "evaluated_functional": str(PRIMARY_FUNCTIONAL),
        "note": (
            "Every metric in this artifact was computed on the calibrated "
            "plug-in probability, which is also the quantity the API returns as "
            "`probability`. The predictive mean over imputation and parameter "
            "draws is reported beside it under its own name and is never "
            "substituted for it."
        ),
    }
    report["calibration_audit"] = chosen_record["calibration"].get("scale_audit", {})
    report["reference_model"] = {
        "kind": "intercept_only_fitted_on_development",
        "prediction": reference_p,
        "n_development_rows": int(dev.size),
        "evaluation_prevalence": float(np.mean(y[ev])),
        "why": (
            "A constant fitted on development labels is a model that could "
            "actually have been shipped. `scaled_brier` normalises against the "
            "evaluation set's own prevalence instead, which is the best "
            "constant in hindsight and therefore a harder but non-deployable "
            "bar. Both are reported; they answer different questions."
        ),
    }

    status = derive_status(
        trained_on_synthetic=trained_on_synthetic,
        protocol=manifest.protocol,
        permits_deployment_claim=manifest.permits_deployment_claim,
        n_development=int(dev.size),
        n_evaluation=int(ev.size),
        log_loss=final_metrics["accuracy"]["log_loss"]["value"],
        calibration_slope=final_metrics["calibration"]["calibration_slope"]["value"],
        reference_comparison=final_metrics.get("reference"),
    )
    report["status"] = status.to_dict()
    report["promotion_criteria"] = PROMOTION_CRITERIA
    log(f"Derived status: {status.status} — {status.rationale[:140]}")

    bundle = ModelBundle(
        version=version,
        endpoint=endpoint.name,
        family=chosen,
        transformer=transformer,
        model=model,
        calibrator=calibrator,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        cohort_version_id=cohort.id,
        dataset_hash=dataset_hash,
        trained_at=utcnow(),
        metrics=report,
        selection_record=selection,
        calibration_record=chosen_record["calibration"],
        limitations=report["limitations"],
        trained_on_synthetic=trained_on_synthetic,
        library_versions=capture_library_versions(),
        git_commit=capture_git_commit(),
        n_train=int(dev.size),
        n_calibration=0,
        n_test=int(ev.size),
        group_key_name="mechanism" if chosen == "hierarchical_logistic" else None,
        protocol=manifest.protocol,
        split_manifest=manifest.to_dict(),
        status=status.status,
        status_record=status.to_dict(),
        prediction_functional=str(PRIMARY_FUNCTIONAL),
        prediction_contract_version=PREDICTION_CONTRACT_VERSION,
    )

    # Activation is gated on the derived status, not on the caller's wish. A
    # status outside SERVING_PERMITTED cannot be activated at all: there would
    # be nothing it is allowed to serve.
    activate = bool(make_active and status.may_serve)
    if make_active and not activate:
        log(
            f"  ⚠ activation refused: status {status.status} does not permit "
            "serving. The artifact is registered so the attempt is on the "
            "record."
        )
    if activate and status.status == str(ModelStatus.DEMO_ONLY):
        log(
            "  activated as demo_only: scoring is available for synthetic "
            "evidence in an explicitly selected demonstration context, and is "
            "refused for real evidence."
        )

    row = register_artifact(
        session, bundle, make_active=activate,
        description=(
            f"{endpoint.display_name}; {chosen}; trained "
            f"{utcnow().date().isoformat()} on cohort {cohort.name}/{cohort.version}."
        ),
    )
    row.training_duration_seconds = time.monotonic() - started
    session.flush()
    log(
        f"Registered {version} ({time.monotonic() - started:.1f}s), "
        f"active={activate}, status={status.status}"
    )

    # The losing candidates are kept as inactive artifacts, not discarded.
    #
    # They are genuine fits on the same cohort and split, so scoring a program
    # with each of them measures how much the estimate depends on the *choice
    # of model form* rather than on the evidence — which is exactly what
    # `specification_sensitivity` reports beside every score. Without them that
    # panel has nothing to compare against and silently reports "unavailable".
    #
    # They are never activated, so they cannot be served by accident.
    if register_candidates:
        for family, spec in developed.items():
            if family == chosen:
                continue
            cand_fit = spec["fit"]
            cand_model = cand_fit.model
            cand_cal = spec["calibrator"]
            rec = next(c for c in candidate_results if c["family"] == family)
            sibling = ModelBundle(
                version=f"{version}-alt-{family}",
                endpoint=endpoint.name,
                family=family,
                transformer=cand_fit.transformer,
                model=cand_model,
                calibrator=cand_cal,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                cohort_version_id=cohort.id,
                dataset_hash=dataset_hash,
                trained_at=utcnow(),
                metrics={"primary_metrics": {}, "note": (
                    "Candidate model retained for specification-sensitivity "
                    "comparison only. Not eligible for serving; see the model "
                    "comparison table in the validation report of "
                    f"{version} for why it was not adopted."
                )},
                selection_record={"outcome": "not_adopted", "verdict": rec.get("verdict")},
                calibration_record=rec["calibration"],
                limitations=[
                    "Candidate artifact retained for comparison. It did not clear "
                    "the model-selection gate and must not be served."
                ],
                trained_on_synthetic=trained_on_synthetic,
                library_versions=capture_library_versions(),
                git_commit=capture_git_commit(),
                n_train=int(dev.size), n_calibration=0, n_test=int(ev.size),
                group_key_name="mechanism" if family == "hierarchical_logistic" else None,
                protocol=manifest.protocol,
                split_manifest=manifest.to_dict(),
                status=str(ModelStatus.EVALUATION_FAILED)
                if rec.get("error") else str(ModelStatus.RESEARCH_ONLY),
                status_record={
                    "status": "candidate_not_adopted",
                    "may_serve": False,
                    "rationale": (
                        "Candidate retained for specification sensitivity. It "
                        "did not clear the selection gate and is never "
                        "activated."
                    ),
                },
                prediction_functional=str(PRIMARY_FUNCTIONAL),
                prediction_contract_version=PREDICTION_CONTRACT_VERSION,
            )
            register_artifact(
                session, sibling, make_active=False,
                display_name=f"{family} (candidate, not adopted)",
                description=(
                    f"Candidate fit from the {version} training run, retained so "
                    "specification sensitivity can be reported. Never served."
                ),
            )
        log(
            f"Retained {len(developed) - 1} candidate artifact(s) for "
            "specification sensitivity"
        )

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation(
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    plan: TrainingPlan,
    columns: Sequence[str],
    components: np.ndarray,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    """Domain ablation on development forward-chaining folds.

    Measured on pooled out-of-fold development predictions rather than on the
    evaluation partition. Twelve domains × one refit each would otherwise be
    twelve additional reads of a partition that is supposed to be read once,
    and the ablation's purpose — ranking which evidence domains carry signal —
    does not need the locked partition to serve it.

    Encoding is fold-local, so the column subset is recomputed per fold from
    that fold's own vocabulary rather than sliced out of a shared matrix.
    """
    from trs.pipeline.develop import fold_design
    from trs.validation.metrics import auroc as _auroc

    def fit_and_score(excluded_features: frozenset[str]) -> dict[str, Any]:
        idx: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for fold in plan.folds:
            if len(np.unique(y[fold.train])) < 2:
                continue
            d = fold_design(rows, fold, random_state=seed)
            keep = [
                i for i, c in enumerate(d.columns)
                if not any(
                    c == f or c.startswith(f"{f}=") or c == f"{f}__missing"
                    for f in excluded_features
                )
            ]
            if len(keep) < 2:
                return {"log_loss": None, "auroc": None}
            cols = [d.columns[i] for i in keep]
            m = L2LogisticRisk(l2_penalty=l2).fit(
                d.X_train[:, keep], y[fold.train], columns=cols
            )
            idx.append(fold.test)
            preds.append(np.asarray(m.predict_logit(d.X_test[:, keep])).ravel())

        if not idx:
            return {"log_loss": None, "auroc": None}
        index = np.concatenate(idx)
        logits = np.concatenate(preds)
        order = np.argsort(index, kind="stable")
        index, logits = index[order], logits[order]
        cal, _ = select_calibrator(logits, y[index], random_state=seed)
        p = cal.transform(logits)
        return {"log_loss": log_loss(y[index], p), "auroc": _auroc(y[index], p)}

    result = domain_ablation(fit_and_score)
    if isinstance(result, dict):
        result["measured_on"] = "development_forward_chaining_out_of_fold"
    return result


def _limitations(
    synthetic: bool,
    comp_summary: dict[str, Any],
    manifest: Any,
    interval_check: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    # Protocol limitations first: they govern how every number below reads.
    items.extend(manifest.limitations)
    if not manifest.permits_deployment_claim:
        items.append(
            f"The accepted split protocol is {manifest.protocol!r}, which is "
            "NOT permitted to describe its numbers as temporal deployment "
            "performance. Read the held-out metrics as retrospective "
            "discrimination and calibration under the stated label-availability "
            "policy, not as an estimate of how the model would behave if "
            "deployed at the fitting origin."
        )
    if synthetic:
        items.append(
            "This model was trained on synthetic demonstration data. Its "
            "coefficients, metrics and predictions describe a simulated "
            "data-generating process and carry NO scientific interpretation "
            "about real therapeutic programs."
        )
    items.append(
        f"Validation rests on {comp_summary['n_components']} independent "
        f"connected components, not {comp_summary['n_rows']} rows. Confidence "
        "intervals are computed at the component level for this reason."
    )
    if comp_summary["largest_component"] > 0.15 * max(comp_summary["n_rows"], 1):
        items.append(
            f"The largest connected component contains "
            f"{comp_summary['largest_component']} rows "
            f"({comp_summary['largest_component'] / comp_summary['n_rows']:.0%} of the "
            "cohort), so fold composition is coarse and fold-to-fold variation "
            "is large."
        )
    purges = manifest.purge_counts
    if purges.get("component_straddled_assessment_boundary"):
        items.append(
            f"{purges['component_straddled_assessment_boundary']} rows were "
            "purged because their connected component straddled the assessment "
            "boundary; they contributed to neither development nor evaluation."
        )
    if purges.get("label_immature_at_fitting_origin"):
        items.append(
            f"{purges['label_immature_at_fitting_origin']} rows were excluded "
            "from development because their outcome had not resolved by the "
            f"fitting origin {manifest.dates.fitting_origin.isoformat()}. This "
            "is the dominant cost of enforcing label maturity and it is the "
            "reason the development partition is far smaller than the cohort."
        )
    if purges.get("label_unresolved_at_data_cutoff"):
        items.append(
            f"{purges['label_unresolved_at_data_cutoff']} rows were excluded "
            "entirely because their outcome resolves after the corpus data "
            f"cutoff {manifest.dates.data_cutoff.isoformat()} — they are the "
            "future, not labelled data."
        )
    if interval_check.get("verdict") == "analytic_intervals_too_narrow":
        items.append(
            "The analytic Laplace intervals are narrower than a bootstrap refit "
            "suggests at this sample size. Published intervals should be read as "
            "a lower bound on true uncertainty."
        )
    items.append(
        "Historical class-performance features are computed as of the cutoff "
        "and exclude the focal program, but other programs in the same "
        "cross-validation fold do contribute to them. This reflects what an "
        "analyst genuinely knew at the decision point; its effect is quantified "
        "by the historical-class-performance row of the domain ablation."
    )
    items.append(
        "Publicly disclosed programs are not a random sample of all programs. "
        "Assets abandoned before an IND are systematically under-represented, so "
        "estimates conditional on entering human development are better "
        "supported than unconditional attrition estimates."
    )
    return items


def _target_grouped_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    keys: list[dict[str, str | None]],
    columns: list[str],
    l2: float,
    seed: int,
    *,
    n_bootstrap: int = 300,
) -> dict[str, Any]:
    """Grouped k-fold under the STRICT scope, pooling out-of-fold predictions.

    Not a temporal split: under strict grouping a target's programs span the
    whole time range, so every component straddles every temporal boundary and
    the split degenerates. That incompatibility is inherent, and reporting this
    as a separate, clearly-labelled cross-sectional analysis is more honest than
    forcing the two constraints together and quietly dropping most of the data.
    """
    grouper = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.STRICT])
    comps = grouper.fit(keys)
    summary = grouper.summary(comps)
    if summary["n_components"] < 5:
        return {
            "available": False,
            "reason": (
                f"Only {summary['n_components']} target-grouped components; "
                "cross-validation would not be meaningful."
            ),
            "grouping": summary,
        }

    oof = np.full(y.shape, np.nan, dtype=float)
    for tr, te in grouped_kfold(comps, n_splits=5, y=y, random_state=seed):
        if len(np.unique(y[tr])) < 2:
            continue
        m = L2LogisticRisk(l2_penalty=l2).fit(X[tr], y[tr], columns=columns)
        # Calibrate inside the fold on a held-out slice of its own training
        # data, so the out-of-fold probabilities are calibrated without ever
        # touching the fold's test rows.
        inner = max(20, int(0.2 * tr.size))
        cal_idx, fit_idx = tr[-inner:], tr[:-inner]
        if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[cal_idx])) < 2:
            oof[te] = m.predict_proba(X[te])
            continue
        m2 = L2LogisticRisk(l2_penalty=l2).fit(X[fit_idx], y[fit_idx], columns=columns)
        cal, _ = select_calibrator(m2.predict_logit(X[cal_idx]), y[cal_idx], random_state=seed)
        oof[te] = cal.transform(m2.predict_logit(X[te]))

    mask = ~np.isnan(oof)
    if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
        return {"available": False, "reason": "Too few out-of-fold predictions.", "grouping": summary}

    metrics = evaluate(
        y[mask], oof[mask], components=comps[mask],
        n_bootstrap=n_bootstrap, random_state=seed, thresholds=(0.5,),
    )
    return {
        "available": True,
        "design": (
            "5-fold cross-validation over connected components formed from "
            "program, compound AND target. Cross-sectional, not temporal."
        ),
        "grouping": summary,
        "n_scored": int(mask.sum()),
        "auroc": metrics["discrimination"]["auroc"]["value"],
        "auroc_ci": [
            metrics["discrimination"]["auroc"]["ci_low"],
            metrics["discrimination"]["auroc"]["ci_high"],
        ],
        "brier": metrics["accuracy"]["brier"]["value"],
        "log_loss": metrics["accuracy"]["log_loss"]["value"],
        "scaled_brier": metrics["accuracy"]["scaled_brier"]["value"],
        "calibration_slope": metrics["calibration"]["calibration_slope"]["value"],
        "interpretation": (
            "Performance when no program sharing a target with a test program "
            "is allowed in training. A large drop against the headline temporal "
            "split would indicate the model relies on target-level memorisation "
            "rather than on the evidence for the individual program."
        ),
    }


#: Relative log-loss margin by which a calibration method must beat logistic
#: recalibration to be preferred over it. See `_usefulness_verdict` for the
#: companion base-rate check.
def _usefulness_verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Does the model actually beat quoting the cohort base rate?

    This is the question a portfolio committee should ask first and the one an
    AUROC headline hides. A model can discriminate slightly while producing
    probabilities that are, on average, further from the truth than the base
    rate — and in that state it must not be presented as decision support.

    The verdict is written into the artifact and surfaced on the Methods page
    and in every API score response.
    """
    sb = metrics["accuracy"]["scaled_brier"]
    value, lo, hi = sb.get("value"), sb.get("ci_low"), sb.get("ci_high")

    if value is None:
        verdict, headline = "indeterminate", "Skill against the base rate could not be estimated."
    elif lo is None or hi is None:
        # No interval was estimated — the bootstrap was switched off, or the
        # test partition was too small for component resampling to produce
        # enough usable draws. Without an interval the comparison against the
        # base rate cannot be adjudicated in either direction, and claiming
        # otherwise from a point estimate is exactly the over-reading this
        # verdict exists to prevent.
        verdict = "indeterminate"
        headline = (
            f"Brier skill against the cohort base rate is {value:+.1%}, but no "
            "bootstrap interval was estimated for it, so the model cannot be "
            "said to beat the base rate. Treat individual estimates as weakly "
            "informative at best."
        )
    elif lo > 0:
        verdict = "beats_base_rate"
        headline = (
            f"The model is better than quoting the cohort base rate: Brier skill "
            f"{value:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}), interval above zero."
        )
    elif hi < 0:
        verdict = "worse_than_base_rate"
        headline = (
            f"The model is WORSE than quoting the cohort base rate on held-out "
            f"data: Brier skill {value:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}). "
            "It must not be used for decisions."
        )
    else:
        verdict = "not_distinguishable_from_base_rate"
        headline = (
            f"The model is not distinguishable from quoting the cohort base "
            f"rate: Brier skill {value:+.1%} (95% CI {lo:+.1%} to {hi:+.1%}), "
            "interval includes zero. Treat individual estimates as weakly "
            "informative at best."
        )

    return {
        "verdict": verdict,
        "headline": headline,
        "scaled_brier": sb,
        "reference": "Cohort base rate (no-covariate model).",
        "why_this_matters": (
            "Discrimination metrics are invariant to monotone distortions of "
            "probability. A model can rank programs correctly and still produce "
            "probabilities less accurate than the base rate. Because this "
            "product's output is a probability, skill against the base rate is "
            "the threshold question."
        ),
    }
