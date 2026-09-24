"""End-to-end training: cohort → features → split → tune → fit → calibrate →
validate → artifact.

Order of operations, and why it is this order
----------------------------------------------
1. **Split first, on time and on connected components.** Everything fitted
   afterwards — imputation medians, one-hot levels, standardisation constants,
   hyperparameters, the calibrator — is fitted inside the training partition
   only. Splitting after any of those steps leaks.
2. **Tune on expanding-window folds inside train.** Hyperparameter selection is
   a form of fitting. Using the test set for it, or even using k-fold that
   allows later programs to inform earlier ones, would make the reported
   performance optimistic.
3. **Calibrate on a partition neither trained nor tested on.** A three-way
   split costs sample size and buys the only calibration numbers worth
   reporting.
4. **Evaluate once, on test.** No iterating against the test partition. When a
   candidate loses, it loses.
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
from trs.features.transform import FeatureTransformer
from trs.modeling import MODEL_FAMILIES
from trs.modeling.base import RiskModel
from trs.modeling.calibration import select_calibrator
from trs.modeling.logistic import L2LogisticRisk
from trs.pipeline.artifacts import (
    ModelBundle,
    capture_git_commit,
    capture_library_versions,
    next_version,
    register_artifact,
)
from trs.pipeline.calibrate import (
    MIN_DEDICATED_CALIBRATION_N,
    cross_validated_calibration,
)
from trs.pipeline.features import materialise_cohort_snapshots, snapshot_rows
from trs.validation.metrics import evaluate, log_loss
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
    SplitAssignment,
    build_grouping_keys,
    grouped_kfold,
    leakage_audit,
    nested_temporal_folds,
    temporal_split,
)

#: L2 penalty grid, log-spaced over four orders of magnitude.
#:
#: The range is wide on purpose. With ~110 encoded columns and a few hundred
#: training rows, the optimum sits at heavy shrinkage — an early version of this
#: grid stopped at 64 and the tuner pinned itself to that endpoint, which
#: underfitted and left the model over-confident out of sample. A grid whose
#: argmin lands on an endpoint has not selected anything, so
#: :func:`tune_l2` now detects that case and records it.
DEFAULT_L2_GRID = (
    0.25, 1.0, 4.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
)

#: Group-effect precisions for the hierarchical family. Larger = more pooling.
DEFAULT_GROUP_PRECISION_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _split_fractions(
    dates: Sequence[dt.date], train_frac: float, calib_frac: float
) -> tuple[dt.date, dt.date]:
    arr = np.sort(np.asarray([d.toordinal() for d in dates]))
    train_end = dt.date.fromordinal(int(np.quantile(arr, train_frac)))
    calib_end = dt.date.fromordinal(int(np.quantile(arr, train_frac + calib_frac)))
    return train_end, calib_end


def _dataset_hash(X: np.ndarray, y: np.ndarray, columns: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(list(columns)).encode())
    h.update(np.ascontiguousarray(np.round(X, 10)).tobytes())
    h.update(np.ascontiguousarray(y.astype(np.int8)).tobytes())
    return h.hexdigest()


def tune_l2(
    X: np.ndarray,
    y: np.ndarray,
    dates: Sequence[dt.date],
    components: np.ndarray,
    columns: list[str],
    *,
    grid: Sequence[float] = DEFAULT_L2_GRID,
) -> tuple[float, list[dict[str, Any]]]:
    """Select the ridge penalty on expanding-window folds inside training data."""
    folds = nested_temporal_folds(dates, components, n_folds=4)
    trace: list[dict[str, Any]] = []

    if not folds:
        # Too few rows to build forward folds; fall back to a defensible
        # default rather than tuning on the data we are about to fit.
        trace.append({
            "note": "Insufficient rows for expanding-window tuning; used default penalty.",
        })
        return 4.0, trace

    for lam in grid:
        losses = []
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = L2LogisticRisk(l2_penalty=lam).fit(X[tr], y[tr], columns=columns)
            losses.append(log_loss(y[te], m.predict_proba(X[te])))
        if losses:
            trace.append({
                "l2_penalty": lam,
                "mean_log_loss": float(np.mean(losses)),
                "n_folds": len(losses),
            })

    scored = [r for r in trace if "mean_log_loss" in r]
    if not scored:
        return 4.0, trace
    best = min(scored, key=lambda r: r["mean_log_loss"])

    # An argmin at an endpoint means the grid, not the data, chose the penalty.
    # Recording it makes the failure visible in the artifact instead of silently
    # shipping an underfitted or unregularised model.
    at_boundary = best["l2_penalty"] in (grid[0], grid[-1])
    trace.append({
        "selected_l2_penalty": float(best["l2_penalty"]),
        "selected_mean_log_loss": best["mean_log_loss"],
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "optimum_at_grid_boundary": at_boundary,
        "warning": (
            "The cross-validated optimum lies at a grid endpoint; the grid "
            "should be widened before these results are relied upon."
            if at_boundary else None
        ),
    })
    return float(best["l2_penalty"]), trace


def _fit_family(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    groups: np.ndarray | None,
    hyperparameters: dict[str, Any],
) -> RiskModel:
    cls = MODEL_FAMILIES[family]
    model = cls(**hyperparameters)
    kwargs: dict[str, Any] = {"columns": columns}
    if family in ("hierarchical_logistic", "gbdt"):
        kwargs["groups"] = groups
    return model.fit(X, y, **kwargs)


def _predict(model: RiskModel, X: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    if model.family == "hierarchical_logistic" and groups is not None:
        return np.asarray(model.predict_logit(X, groups=groups))
    return np.asarray(model.predict_logit(X))


def train(
    session: Session,
    *,
    cohort_version_id: str | None = None,
    endpoint: EndpointDefinition = PRIMARY_ENDPOINT,
    train_fraction: float = 0.76,
    calibration_fraction: float = 0.00,
    candidate_families: Sequence[str] = ("l2_logistic", "hierarchical_logistic", "gbdt"),
    n_bootstrap: int = 600,
    random_state: int = 20240917,
    make_active: bool = True,
    run_interval_check: bool = True,
    run_ablation: bool = True,
    register_candidates: bool = True,
    verbose: bool = True,
) -> ModelBundle:
    """Train, validate and register a model for one endpoint."""
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

    # ---- 3. Split ----------------------------------------------------------
    train_end, calib_end = _split_fractions(dates, train_fraction, calibration_fraction)
    split = temporal_split(dates, components, train_end=train_end, calibration_end=calib_end)
    log(f"Split: {split.sizes()} — {split.description}")
    if split.train.size < 30 or split.test.size < 10:
        raise RuntimeError(
            f"Temporal split left train={split.train.size}, test={split.test.size}. "
            "Too small to validate honestly."
        )

    audit = leakage_audit(
        components, split.train, split.test, keys, scope=GroupScope.IDENTITY
    )
    if not audit["clean"]:
        raise RuntimeError(f"Leakage audit failed on the training split: {audit}")

    # ---- 4. Encoder (fit on TRAIN only) ------------------------------------
    transformer = FeatureTransformer().fit(
        [rows[i] for i in split.train], random_state=random_state
    )
    X = transformer.transform(rows)
    columns = transformer.columns
    mech = np.asarray([k.get("mechanism") or "unknown" for k in keys])

    tr, ca, te = split.train, split.calibration, split.test

    # A dedicated calibration slice is only worth its cost when it is large
    # enough to fit a stable calibrator. Below the threshold, the slice is
    # folded back into training and the calibrator is fitted on pooled
    # out-of-fold predictions across the whole training partition instead. See
    # trs.pipeline.calibrate for the reasoning and the accepted bias.
    use_cv_calibration = ca.size < MIN_DEDICATED_CALIBRATION_N
    if use_cv_calibration:
        log(
            f"Dedicated calibration slice has {ca.size} rows (< "
            f"{MIN_DEDICATED_CALIBRATION_N}); folding it back into training and "
            "calibrating on pooled out-of-fold predictions."
        )
        tr = np.sort(np.concatenate([tr, ca]))
        ca = np.asarray([], dtype=int)

    # ---- 5. Tune -----------------------------------------------------------
    best_l2, tune_trace = tune_l2(
        X[tr], y[tr], [dates[i] for i in tr], components[tr], columns
    )
    tune_summary = next((r for r in tune_trace if "selected_l2_penalty" in r), {})
    log(f"Selected L2 penalty {best_l2} from expanding-window folds")
    if tune_summary.get("optimum_at_grid_boundary"):
        log(f"  ⚠ {tune_summary['warning']}")

    # ---- 6. Fit candidates, calibrate, evaluate ----------------------------
    candidate_results: list[dict[str, Any]] = []
    fitted: dict[str, tuple[RiskModel, Any, np.ndarray]] = {}

    for family in candidate_families:
        hp = _default_hyperparameters(family, best_l2, random_state)
        if family == "hierarchical_logistic":
            best_gp, gp_trace = _tune_group_precision(
                X[tr], y[tr], mech[tr], [dates[i] for i in tr], components[tr], columns, best_l2
            )
            hp["group_precision"] = best_gp
        else:
            gp_trace = []

        try:
            if use_cv_calibration:
                model, calibrator, cal_record = cross_validated_calibration(
                    lambda f=family, h=hp: MODEL_FAMILIES[f](**h),
                    X[tr], y[tr],
                    components=components[tr], columns=columns,
                    groups=mech[tr], random_state=random_state,
                )
            else:
                model = _fit_family(family, X[tr], y[tr], columns, mech[tr], hp)
                cal_logit = _predict(model, X[ca], mech[ca])
                calibrator, cal_record = select_calibrator(
                    cal_logit, y[ca], random_state=random_state
                )
        except Exception as exc:  # noqa: BLE001
            candidate_results.append({"family": family, "error": str(exc), "eligible": False})
            log(f"  {family}: FAILED ({exc})")
            continue

        test_logit = _predict(model, X[te], mech[te])
        p_test = calibrator.transform(test_logit)
        metrics = evaluate(
            y[te], p_test, components=components[te],
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        fitted[family] = (model, calibrator, p_test)
        candidate_results.append({
            "family": family,
            "eligible": True,
            "hyperparameters": hp,
            "tuning_trace": gp_trace,
            "calibration": cal_record,
            "log_loss": metrics["accuracy"]["log_loss"]["value"],
            "brier": metrics["accuracy"]["brier"]["value"],
            "scaled_brier": metrics["accuracy"]["scaled_brier"]["value"],
            "auroc": metrics["discrimination"]["auroc"]["value"],
            "calibration_slope": metrics["calibration"]["calibration_slope"]["value"],
            "supports_exact_attribution": model.supports_exact_attribution,
            "supports_analytic_posterior": model.supports_analytic_posterior,
        })
        log(
            f"  {family}: log-loss {metrics['accuracy']['log_loss']['value']:.4f}  "
            f"AUROC {metrics['discrimination']['auroc']['value']}  "
            f"slope {metrics['calibration']['calibration_slope']['value']}"
        )

    from trs.pipeline.select import choose_family

    selection = choose_family(
        candidate_results, y[te], {f: v[2] for f, v in fitted.items()}, components[te],
        random_state=random_state,
    )
    chosen = selection["chosen_family"]
    log(f"Selected family: {chosen} — {selection['rationale']}")

    model, calibrator, p_test = fitted[chosen]
    chosen_record = next(c for c in candidate_results if c["family"] == chosen)
    final_metrics = evaluate(
        y[te], p_test, components=components[te],
        n_bootstrap=n_bootstrap, random_state=random_state,
    )

    # ---- 7. Diagnostics ----------------------------------------------------
    subgroups = subgroup_metrics(
        y[te], p_test, components[te],
        {
            "modality_group": [rows[i].get("modality_group") for i in te],
            "target_class_group": [rows[i].get("target_class_group") for i in te],
            "evidence_completeness": [
                completeness_tertiles(completeness)[i] for i in te
            ],
            "era": [era_labels(dates)[i] for i in te],
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

    interval_check: dict[str, Any] = {}
    if run_interval_check and model.family == "l2_logistic":
        probe = X[te][: min(30, te.size)]
        interval_check = interval_agreement(
            lambda: L2LogisticRisk(l2_penalty=best_l2),
            transformer, None, X[tr], y[tr], components[tr], probe,
            n_refits=40, random_state=random_state,
        )
        log(f"  interval agreement: {interval_check.get('verdict')}")

    ablation: dict[str, Any] = {}
    if run_ablation:
        ablation = _run_ablation(
            X, y, tr, ca, te, columns, transformer, components, best_l2, random_state
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
        split_description=split.description,
        split_sizes={**split.sizes(), "train_final": int(tr.size), "calibration_final": int(ca.size)},
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
        limitations=_limitations(trained_on_synthetic, comp_summary, split, interval_check),
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
        n_train=int(tr.size),
        n_calibration=int(ca.size),
        n_test=int(te.size),
        group_key_name="mechanism" if chosen == "hierarchical_logistic" else None,
    )

    row = register_artifact(
        session, bundle, make_active=make_active,
        description=(
            f"{endpoint.display_name}; {chosen}; trained "
            f"{utcnow().date().isoformat()} on cohort {cohort.name}/{cohort.version}."
        ),
    )
    row.training_duration_seconds = time.monotonic() - started
    session.flush()
    log(f"Registered {version} ({time.monotonic() - started:.1f}s), active={make_active}")

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
        for family, (cand_model, cand_cal, _) in fitted.items():
            if family == chosen:
                continue
            rec = next(c for c in candidate_results if c["family"] == family)
            sibling = ModelBundle(
                version=f"{version}-alt-{family}",
                endpoint=endpoint.name,
                family=family,
                transformer=transformer,
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
                n_train=int(tr.size), n_calibration=int(ca.size), n_test=int(te.size),
                group_key_name="mechanism" if family == "hierarchical_logistic" else None,
            )
            register_artifact(
                session, sibling, make_active=False,
                display_name=f"{family} (candidate, not adopted)",
                description=(
                    f"Candidate fit from the {version} training run, retained so "
                    "specification sensitivity can be reported. Never served."
                ),
            )
        log(f"Retained {len(fitted) - 1} candidate artifact(s) for specification sensitivity")

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_hyperparameters(family: str, l2: float, seed: int) -> dict[str, Any]:
    match family:
        case "l2_logistic":
            return {"l2_penalty": l2}
        case "hierarchical_logistic":
            return {"l2_penalty": l2, "group_precision": 4.0}
        case "gbdt":
            return {
                "learning_rate": 0.05, "max_iter": 150, "max_depth": 3,
                "min_samples_leaf": 15, "l2_regularization": 1.0,
                "ensemble_size": 40, "random_state": seed,
            }
        case _:
            return {}


def _tune_group_precision(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    dates: Sequence[dt.date],
    components: np.ndarray,
    columns: list[str],
    l2: float,
) -> tuple[float, list[dict[str, Any]]]:
    from trs.modeling.hierarchical import HierarchicalLogisticRisk

    folds = nested_temporal_folds(dates, components, n_folds=3)
    trace: list[dict[str, Any]] = []
    if not folds:
        return 4.0, trace
    for gp in DEFAULT_GROUP_PRECISION_GRID:
        losses = []
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = HierarchicalLogisticRisk(l2_penalty=l2, group_precision=gp).fit(
                X[tr], y[tr], columns=columns, groups=groups[tr]
            )
            from trs.modeling.base import sigmoid

            losses.append(log_loss(y[te], sigmoid(m.predict_logit(X[te], groups=groups[te]))))
        if losses:
            trace.append({"group_precision": gp, "mean_log_loss": float(np.mean(losses))})
    if not trace:
        return 4.0, trace
    return float(min(trace, key=lambda r: r["mean_log_loss"])["group_precision"]), trace


def _run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    ca: np.ndarray,
    te: np.ndarray,
    columns: list[str],
    transformer: FeatureTransformer,
    components: np.ndarray,
    l2: float,
    seed: int,
) -> dict[str, Any]:

    col_index = {c: i for i, c in enumerate(columns)}

    def fit_and_score(excluded_features: frozenset[str]) -> dict[str, Any]:
        keep = [
            i for c, i in col_index.items()
            if not any(
                c == f or c.startswith(f"{f}=") or c == f"{f}__missing"
                for f in excluded_features
            )
        ]
        if len(keep) < 2:
            return {"log_loss": None, "auroc": None}
        Xs = X[:, keep]
        cols = [columns[i] for i in keep]
        if ca.size >= 20:
            m = L2LogisticRisk(l2_penalty=l2).fit(Xs[tr], y[tr], columns=cols)
            cal, _ = select_calibrator(m.predict_logit(Xs[ca]), y[ca], random_state=seed)
        else:
            m, cal, _ = cross_validated_calibration(
                lambda: L2LogisticRisk(l2_penalty=l2),
                Xs[tr], y[tr], components=components[tr], columns=cols,
                random_state=seed,
            )
        p = cal.transform(m.predict_logit(Xs[te]))
        from trs.validation.metrics import auroc as _auroc

        return {"log_loss": log_loss(y[te], p), "auroc": _auroc(y[te], p)}

    return domain_ablation(fit_and_score)


def _limitations(
    synthetic: bool,
    comp_summary: dict[str, Any],
    split: SplitAssignment,
    interval_check: dict[str, Any],
) -> list[str]:
    items: list[str] = []
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
    items.append(
        f"{split.dropped.get('component_straddled_boundary', 0)} rows were moved "
        "to a later partition because their connected component straddled a "
        "temporal boundary; they did not contribute to training."
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
