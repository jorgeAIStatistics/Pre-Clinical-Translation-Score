"""Regression tests for the development/evaluation protocol.

These exist because of specific defects, not for coverage. Each test names the
defect it would have caught, so a future change that reintroduces one fails
with an explanation rather than a diff.

The headline test is :func:`test_evaluation_labels_cannot_influence_development`.
Everything else in this file supports it: a locked evaluation partition is only
a claim until something demonstrates that corrupting it changes nothing about
the fitted model.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import trs.db.models  # noqa: F401  (registers the mappers)
from trs.cohort.build import build_cohort
from trs.db.base import Base
from trs.db.models.analytics import CohortEntry, CohortVersion
from trs.db.models.core import Program, ProgramIndication
from trs.pipeline.develop import (
    calibration_scale_audit,
    cross_fitted_calibrated,
    fold_design,
)
from trs.pipeline.status import (
    ModelStatus,
    RefusalCode,
    check_eligibility,
    derive_status,
)
from trs.validation.metrics import (
    calibration_intercept_slope,
    deployable_reference_prediction,
    scaled_brier,
    skill_against_reference,
)
from trs.validation.protocol import (
    DEPLOYMENT_CLAIM_PERMITTED,
    MIN_USABLE_FOLDS,
    Protocol,
    ProtocolError,
    forward_chaining_folds,
    plan_split,
    plan_training,
)
from trs.validation.splitters import (
    GROUP_SCOPE_KEYS,
    ConnectedComponentGrouper,
    GroupScope,
    build_grouping_keys,
)

#: Enough programmes that `temporal_outer_maturity` is satisfiable after label
#: maturity purges roughly half the eligible development rows. See
#: tests/integration/test_api.py for the measurements behind this number.
N_PROGRAMS = 700


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort_session(tmp_path_factory: pytest.TempPathFactory) -> Session:
    """A seeded in-memory database with one built cohort."""
    from trs.seed.generator import DemoDataGenerator

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False, future=True
    )()
    DemoDataGenerator(session, seed=4242).run(n_programs=N_PROGRAMS)
    session.commit()
    build_cohort(session)
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _planning_inputs(session: Session) -> tuple[list, list, list, np.ndarray, np.ndarray]:
    """Row ids, as-of dates, label-availability dates, components, labels."""
    cohort = session.scalars(
        select(CohortVersion).order_by(CohortVersion.built_at.desc()).limit(1)
    ).first()
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
    outcome_date = dict(
        session.execute(
            select(ProgramIndication.id, ProgramIndication.outcome_date)
        ).all()
    )
    compound = dict(
        session.execute(
            select(ProgramIndication.id, Program.primary_compound_id).join(
                Program, Program.id == ProgramIndication.program_id
            )
        ).all()
    )
    keys = build_grouping_keys(
        entries, [compound.get(e.program_indication_id) for e in entries]
    )
    components = ConnectedComponentGrouper(
        GROUP_SCOPE_KEYS[GroupScope.IDENTITY]
    ).fit(keys)
    return (
        [e.id for e in entries],
        [e.as_of_date for e in entries],
        [outcome_date.get(e.program_indication_id) for e in entries],
        components,
        np.asarray([e.label_primary for e in entries], dtype=int),
    )


# ---------------------------------------------------------------------------
# The headline property
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_evaluation_labels_cannot_influence_development(
    cohort_session: Session, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """If this breaks, the evaluation partition is a selection partition again.

    This is the defect the protocol rewrite exists to fix. The predecessor
    passed ``y[te]`` to ``choose_family`` and then reported the winner on those
    same rows, so its headline AUROC, log loss and Brier skill were
    selection-set figures presented as held-out performance.

    The test corrupts the evaluation labels — permutes them, which preserves
    their marginal distribution and so cannot be detected by any aggregate
    check — and retrains. Everything chosen or fitted during development must
    be bit-identical: the split manifest hash, the selected family, the
    hyperparameters, the encoder state, the coefficients and the calibrator.

    Evaluation *metrics* must of course change; an assertion below requires
    that they do, otherwise the test would pass vacuously on a pipeline that
    ignored the evaluation partition entirely.
    """
    from trs.config import settings
    from trs.pipeline.artifacts import clear_bundle_cache
    from trs.pipeline.train import train

    saved_dir = settings.artifact_dir
    settings.artifact_dir = tmp_path_factory.mktemp("protocol-artifacts")
    clear_bundle_cache()

    kwargs = dict(
        candidate_families=("l2_logistic", "hierarchical_logistic"),
        n_bootstrap=0,
        run_interval_check=False,
        run_ablation=False,
        register_candidates=False,
        make_active=False,
        verbose=False,
        data_cutoff=dt.date(2026, 10, 13),
    )

    try:
        baseline = train(cohort_session, **kwargs)
        cohort_session.commit()

        # Permute the labels of the evaluation rows only.
        row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
        manifest = plan_training(
            row_ids, as_of, label_on, components,
            data_cutoff=dt.date(2026, 10, 13), development_fraction=0.76,
        ).manifest
        assert manifest.manifest_hash == baseline.split_manifest["manifest_hash"], (
            "the test planned a different split from the one training used; "
            "the comparison below would be meaningless"
        )

        eval_ids = [manifest.row_ids[i] for i in manifest.evaluation]
        assert len(eval_ids) >= 60
        rows = {
            e.id: e
            for e in cohort_session.scalars(
                select(CohortEntry).where(CohortEntry.id.in_(eval_ids))
            )
        }
        original = {rid: rows[rid].label_primary for rid in eval_ids}
        permuted = list(original.values())
        rng = np.random.default_rng(99)
        rng.shuffle(permuted)
        assert permuted != list(original.values()), "permutation was a no-op"
        for rid, lab in zip(eval_ids, permuted, strict=True):
            rows[rid].label_primary = int(lab)
        cohort_session.flush()
        cohort_session.commit()

        clear_bundle_cache()
        corrupted = train(cohort_session, **kwargs)
        cohort_session.commit()

        # --- nothing developed may differ -------------------------------
        assert corrupted.split_manifest["manifest_hash"] == (
            baseline.split_manifest["manifest_hash"]
        ), "the split itself moved when evaluation labels changed"
        assert corrupted.protocol == baseline.protocol
        assert corrupted.family == baseline.family, (
            "family selection responded to evaluation labels"
        )
        assert (
            corrupted.metrics["model_comparison"]["chosen_family"]
            == baseline.metrics["model_comparison"]["chosen_family"]
        )
        base_hp = baseline.metrics["hyperparameter_selection"]
        corr_hp = corrupted.metrics["hyperparameter_selection"]
        assert corr_hp["trace"][-1]["selected_l2_penalty"] == (
            base_hp["trace"][-1]["selected_l2_penalty"]
        ), "hyperparameter selection responded to evaluation labels"

        assert (
            corrupted.transformer.state.to_dict() == baseline.transformer.state.to_dict()
        ), "the feature encoder was fitted using evaluation rows"

        b_state, c_state = baseline.model.to_state(), corrupted.model.to_state()
        assert c_state["columns"] == b_state["columns"]
        np.testing.assert_allclose(
            np.asarray(c_state["coef"], dtype=float),
            np.asarray(b_state["coef"], dtype=float),
            rtol=0, atol=0,
            err_msg="model coefficients changed when only evaluation labels did",
        )
        assert c_state["intercept"] == b_state["intercept"], (
            "the intercept moved when only evaluation labels did"
        )

        b_cal, c_cal = baseline.calibrator.to_dict(), corrupted.calibrator.to_dict()
        for key in ("method", "intercept", "slope"):
            assert c_cal.get(key) == b_cal.get(key), (
                f"calibrator {key} changed when only evaluation labels did; "
                "the calibrator saw the evaluation partition"
            )

        # --- and the evaluation metrics must move -----------------------
        b_eval = baseline.metrics["primary_metrics"]["accuracy"]["log_loss"]["value"]
        c_eval = corrupted.metrics["primary_metrics"]["accuracy"]["log_loss"]["value"]
        assert b_eval != pytest.approx(c_eval), (
            "evaluation log loss was identical under permuted evaluation "
            "labels, so this test is not exercising the evaluation partition "
            "at all"
        )
    finally:
        # Restore the labels so later tests in this module see a clean cohort.
        for rid, lab in original.items():
            rows[rid].label_primary = lab
        cohort_session.commit()
        settings.artifact_dir = saved_dir
        clear_bundle_cache()


# ---------------------------------------------------------------------------
# Manifest invariants
# ---------------------------------------------------------------------------


def test_manifest_accounts_for_every_row(cohort_session: Session) -> None:
    """If this breaks, rows vanish between the cohort and the split."""
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )
    assert len(m.development) + len(m.evaluation) + len(m.purged) == m.n_rows
    assert not set(m.development) & set(m.evaluation)
    assert m.purge_counts  # something was purged, so the accounting is tested
    m.verify(as_of, label_on)


def test_no_component_spans_the_evaluation_boundary(cohort_session: Session) -> None:
    """If this breaks, the same asset appears on both sides of the split."""
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )
    comps = np.asarray(m.components)
    dev_comps = set(comps[m.development_index].tolist())
    ev_comps = set(comps[m.evaluation_index].tolist())
    assert not dev_comps & ev_comps
    assert dev_comps and ev_comps


def test_strict_temporal_headline_contains_no_pre_boundary_evaluation_rows(
    cohort_session: Session,
) -> None:
    """If this breaks, a 'temporal' metric silently includes the past.

    ``verify`` enforces this, so the test corrupts a manifest and requires the
    check to fire. A guard that is never exercised is a comment.
    """
    import dataclasses

    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )
    moved = m.development[0]
    smuggled = dataclasses.replace(
        m,
        development=tuple(i for i in m.development if i != moved),
        evaluation=tuple(sorted(set(m.evaluation) | {moved})),
    )
    with pytest.raises(ProtocolError, match="at or before the assessment boundary"):
        smuggled.verify(as_of, label_on)


def test_label_maturity_is_enforced_and_costly(cohort_session: Session) -> None:
    """If this breaks, training uses outcomes that had not happened yet.

    Also pins the *cost*: label maturity is not a free correctness win, it
    removes a large share of otherwise-eligible development rows, and a change
    that makes it look cheap has probably stopped enforcing it.
    """
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    strict = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.TEMPORAL_OUTER_MATURITY,
        data_cutoff=dt.date(2026, 10, 13),
    )
    loose = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )

    assert strict.label_maturity == "enforced_at_fitting_origin"
    assert loose.label_maturity == "not_enforced"

    cutoff = strict.dates.label_cutoff
    for i in strict.development:
        assert label_on[i] is not None and label_on[i] <= cutoff

    immature = strict.purge_counts.get("label_immature_at_fitting_origin", 0)
    assert immature > 0, "no row was purged for label immaturity"
    assert len(strict.development) < len(loose.development)
    assert immature >= 0.3 * len(loose.development), (
        f"only {immature} rows were purged for immaturity against "
        f"{len(loose.development)} retrospective development rows. Either the "
        "cohort's outcome lags changed or maturity stopped being enforced."
    )


def test_future_labels_are_excluded_by_the_data_cutoff(cohort_session: Session) -> None:
    """If this breaks, outcomes dated after the corpus cutoff become training data."""
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    cutoff = dt.date(2020, 1, 1)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=cutoff,
    )
    assert m.purge_counts.get("label_unresolved_at_data_cutoff", 0) > 0
    for i in list(m.development) + list(m.evaluation):
        assert label_on[i] is not None and label_on[i] <= cutoff


def test_manifest_hash_is_stable_under_reordering_and_sensitive_to_membership(
    cohort_session: Session,
) -> None:
    """If this breaks, two runs cannot be compared by their manifest hash."""
    import dataclasses

    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )
    # Reordering the *recorded* partitions must not change the hash, because
    # the hash is taken over sorted row ids.
    shuffled = dataclasses.replace(m, development=tuple(reversed(m.development)))
    assert shuffled.manifest_hash == m.manifest_hash

    moved = dataclasses.replace(m, development=tuple(m.development[:-1]))
    assert moved.manifest_hash != m.manifest_hash


# ---------------------------------------------------------------------------
# No silent fallback
# ---------------------------------------------------------------------------


def test_protocol_rejection_is_recorded_not_hidden(cohort_session: Session) -> None:
    """If this breaks, a weaker protocol substitutes itself without saying so.

    On this cohort ``strict_temporal`` is genuinely infeasible: its inner folds
    must also respect label maturity, and at an early fold origin almost
    nothing has resolved. The point of this test is not that it fails — it is
    that the failure is reported, with its measured reason, in the trace that
    travels into the artifact.
    """
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    plan = plan_training(
        row_ids, as_of, label_on, components,
        data_cutoff=dt.date(2026, 10, 13), development_fraction=0.76,
    )
    rejected = [t for t in plan.protocol_trace if "rejected_because" in t]
    accepted = [t for t in plan.protocol_trace if t.get("accepted") == "yes"]

    assert len(accepted) == 1
    assert accepted[0]["protocol"] == plan.protocol
    assert rejected, "nothing was rejected, so the trace is untested here"
    for entry in rejected:
        assert entry["rejected_because"].strip(), "a rejection carried no reason"
    assert any(
        "fold" in t["rejected_because"] for t in rejected
    ), "fold feasibility was never the reason for any rejection"

    # And the accepted protocol must declare whether it may claim deployment
    # performance, rather than leaving the reader to guess.
    assert plan.permits_deployment_claim == (
        plan.protocol in DEPLOYMENT_CLAIM_PERMITTED
    )
    if not plan.permits_deployment_claim:
        assert plan.manifest.limitations, (
            "a protocol that may not claim deployment performance recorded no "
            "limitation explaining that"
        )


def test_strict_protocol_fold_maturity_is_actually_enforced(
    cohort_session: Session,
) -> None:
    """If this breaks, per-fold label maturity silently stops applying."""
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )
    enforced, diag_on = forward_chaining_folds(
        as_of, label_on, components, m.development_index,
        enforce_label_maturity=True,
    )
    relaxed, diag_off = forward_chaining_folds(
        as_of, label_on, components, m.development_index,
        enforce_label_maturity=False,
    )
    assert diag_on["label_maturity_enforced"] is True
    assert diag_off["label_maturity_enforced"] is False
    assert sum(f.train.size for f in enforced) < sum(f.train.size for f in relaxed)

    for fold in enforced:
        assert fold.dropped.get("label_immature_at_fold_origin", 0) >= 0
        for i in fold.train:
            assert label_on[i] is not None
            assert label_on[i] <= fold.origin, (
                f"fold at origin {fold.origin} trained on a row whose outcome "
                f"resolved {label_on[i]}"
            )


def test_folds_never_train_on_a_component_in_their_test_window(
    cohort_session: Session,
) -> None:
    """If this breaks, the same asset trains and tests inside one fold."""
    row_ids, as_of, label_on, components, _ = _planning_inputs(cohort_session)
    m = plan_split(
        row_ids, as_of, label_on, components,
        protocol=Protocol.RETROSPECTIVE, data_cutoff=dt.date(2026, 10, 13),
    )
    folds, _ = forward_chaining_folds(
        as_of, label_on, components, m.development_index,
        enforce_label_maturity=False,
    )
    assert len(folds) >= MIN_USABLE_FOLDS
    comps = np.asarray(components)
    for fold in folds:
        assert not set(comps[fold.train].tolist()) & set(comps[fold.test].tolist())
        assert max(as_of[i] for i in fold.train) < min(
            as_of[i] for i in fold.test
        ), "a fold trained on rows assessed after its own test window"


# ---------------------------------------------------------------------------
# Fold-local preprocessing
# ---------------------------------------------------------------------------


def test_fold_preprocessing_is_fold_local() -> None:
    """If this breaks, a fold's held-out rows shape their own encoding.

    Constructed so the two folds have clearly different medians. A shared
    transformer would impute both folds with the same constant; a fold-local
    one must not.
    """
    from trs.validation.protocol import Fold

    feature = "animal_n_invivo_studies"
    rows = [{feature: v} for v in [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0]]
    rows.append({feature: None})  # the row that gets imputed

    fold_a = Fold(
        index=0, origin=dt.date(2010, 1, 1),
        train=np.array([0, 1, 2, 3]), test=np.array([8]), dropped={},
    )
    fold_b = Fold(
        index=1, origin=dt.date(2015, 1, 1),
        train=np.array([4, 5, 6, 7]), test=np.array([8]), dropped={},
    )

    design_a = fold_design(rows, fold_a)
    design_b = fold_design(rows, fold_b)

    med_a = design_a.transformer.state.medians.get(feature)
    med_b = design_b.transformer.state.medians.get(feature)
    assert med_a is not None and med_b is not None
    assert med_a != pytest.approx(med_b), (
        "both folds learned the same imputation median, so preprocessing is "
        "not fold-local"
    )
    # And the imputed value handed to the model must differ as a result.
    #
    # Compared on the *raw* scale deliberately: standardisation centres each
    # fold on its own mean, and a row imputed at its fold's median therefore
    # lands at ~0 in both folds. On the standardised scale this test would pass
    # whether or not preprocessing were fold-local, which makes it worthless.
    col = design_a.columns.index(feature)
    raw_a = design_a.transformer.transform_raw([rows[8]])[0, col]
    raw_b = design_b.transformer.transform_raw([rows[8]])[0, col]
    assert raw_a == pytest.approx(med_a)
    assert raw_b == pytest.approx(med_b)
    assert raw_a != pytest.approx(raw_b), (
        "the imputed value was identical across folds despite different "
        "learned medians"
    )


# ---------------------------------------------------------------------------
# Calibration honesty
# ---------------------------------------------------------------------------


def test_cross_fitted_calibration_does_not_force_slope_one() -> None:
    """If this breaks, the selection gate's calibration band stops working.

    Fitting a calibrator on a pool and then scoring that same pool drives the
    calibration slope to 1.0 by construction. That is exactly what happened on
    the first run of the rewritten pipeline: all three candidate families
    reported a slope of 1.0 to sixteen significant figures, and the gate's
    [0.80, 1.25] band silently admitted everything.
    """
    from trs.modeling.calibration import select_calibrator

    rng = np.random.default_rng(7)
    n = 400
    z = rng.normal(0.0, 1.5, size=n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.4 * z - 0.2))))
    components = np.arange(n)  # all independent

    in_sample_cal, _ = select_calibrator(z, y, random_state=0)
    in_sample_slope = calibration_intercept_slope(y, in_sample_cal.transform(z))["slope"]
    assert in_sample_slope == pytest.approx(1.0, abs=0.02), (
        "the premise of this test no longer holds: in-sample recalibration is "
        "supposed to produce a slope of ~1.0"
    )

    p_xfit, positions, diag = cross_fitted_calibrated(
        z, y, components, random_state=0
    )
    assert diag["n_calibrated"] == positions.size > 0
    xfit_slope = calibration_intercept_slope(y[positions], p_xfit)["slope"]
    assert abs(xfit_slope - 1.0) > 1e-6, (
        "cross-fitted calibration still produced a slope indistinguishable "
        "from 1.0, which means the calibrator saw the rows it scored"
    )


def test_calibration_scale_audit_detects_a_mismatched_pool() -> None:
    """If this breaks, a calibrator fitted on the wrong scale ships silently.

    This is the defect that cost 0.19 nats of evaluation log loss: the
    forward-chaining pool's base rate was 0.664 against the development
    partition's 0.631, and the fold models' mean logit sat 0.33 below the final
    model's, so the recalibration intercept was fitted to lift flat logits onto
    the wrong level.
    """
    rng = np.random.default_rng(3)
    pool_logit = rng.normal(0.28, 0.61, size=120)
    pool_y = rng.binomial(1, 0.664, size=120)
    final_logit = rng.normal(0.62, 0.77, size=190)
    dev_y = rng.binomial(1, 0.631, size=190)

    bad = calibration_scale_audit(pool_logit, pool_y, final_logit, dev_y)
    assert bad["verdict"] == "scale_mismatch"
    assert bad["problems"]
    assert any("different scale" in p for p in bad["problems"])

    # A pool drawn from the same distribution as the final predictions passes.
    good_logit = rng.normal(0.62, 0.77, size=190)
    good = calibration_scale_audit(good_logit, dev_y, final_logit, dev_y)
    assert good["verdict"] == "consistent", good["problems"]


# ---------------------------------------------------------------------------
# The reference model
# ---------------------------------------------------------------------------


def test_deployable_reference_is_not_the_evaluation_prevalence() -> None:
    """If this breaks, 'skill' is measured against a hindsight-optimal constant.

    The two quantities answer different questions and must not collapse into
    one. ``scaled_brier`` normalises by the best constant *on the rows being
    scored*; the deployable reference is a constant fitted on development
    labels, which is what could actually have been shipped.
    """
    y_dev = np.array([1] * 63 + [0] * 37)          # development rate 0.63
    y_eval = np.array([1] * 55 + [0] * 45)         # evaluation prevalence 0.55
    p = np.full(y_eval.shape, 0.63)

    ref = deployable_reference_prediction(y_dev)
    assert ref == pytest.approx(0.63, abs=0.01)
    assert ref != pytest.approx(float(np.mean(y_eval)), abs=0.01)

    descriptive = scaled_brier(y_eval, p)
    against_ref = skill_against_reference(y_eval, p, ref, components=np.arange(100))

    # Predicting the development rate exactly equals the deployable reference,
    # so skill against it is zero — while the descriptive figure is negative,
    # because the hindsight-optimal constant is a harder bar.
    assert against_ref["brier_skill"] == pytest.approx(0.0, abs=1e-9)
    assert descriptive < 0.0
    assert against_ref["reference_value"] == pytest.approx(ref)


def test_reference_requires_development_labels() -> None:
    """If this breaks, a reference can be fitted on nothing."""
    with pytest.raises(ValueError, match="zero development rows"):
        deployable_reference_prediction(np.array([]))


# ---------------------------------------------------------------------------
# Status and eligibility
# ---------------------------------------------------------------------------


def _status_kwargs(**over: object) -> dict:
    base = dict(
        trained_on_synthetic=False,
        protocol=str(Protocol.STRICT_TEMPORAL),
        permits_deployment_claim=True,
        n_development=200,
        n_evaluation=120,
        log_loss=0.60,
        calibration_slope=1.0,
        reference_comparison={
            "brier_skill": 0.1,
            "model_better_than_reference": True,
            "brier_difference_model_minus_reference": {
                "excludes_zero": True, "ci_high": -0.01, "ci_low": -0.05,
            },
        },
    )
    base.update(over)
    return base


def test_synthetic_training_caps_status_at_demo_only() -> None:
    """If this breaks, a simulation's opinion is promotable on its own metrics."""
    decision = derive_status(**_status_kwargs(trained_on_synthetic=True))
    assert decision.status == str(ModelStatus.DEMO_ONLY)
    assert "synthetic" in decision.rationale.lower()
    # Every other criterion passed, which is the point: excellent simulated
    # metrics must not promote.
    assert decision.criteria["beats_deployable_reference"]["passed"] is True
    assert decision.criteria["calibrated_in_band"]["passed"] is True
    assert decision.criteria["real_evidence"]["passed"] is False


def test_failing_the_reference_comparison_yields_research_only() -> None:
    """If this breaks, a model no better than a constant is marked approved."""
    decision = derive_status(
        **_status_kwargs(
            reference_comparison={
                "brier_skill": -0.04,
                "model_better_than_reference": False,
                "brier_difference_model_minus_reference": {"excludes_zero": False},
            }
        )
    )
    assert decision.status == str(ModelStatus.RESEARCH_ONLY)
    assert decision.may_serve is True  # research use is allowed; promotion is not


def test_degenerate_fit_yields_evaluation_failed() -> None:
    """If this breaks, a confidently-wrong model is servable."""
    decision = derive_status(**_status_kwargs(log_loss=1.45))
    assert decision.status == str(ModelStatus.EVALUATION_FAILED)
    assert decision.may_serve is False


def test_small_sample_yields_insufficient_data() -> None:
    """If this breaks, a claim is made on a sample too small to support one."""
    decision = derive_status(**_status_kwargs(n_development=50, n_evaluation=20))
    assert decision.status == str(ModelStatus.INSUFFICIENT_DATA)
    assert decision.may_serve is False


def test_approval_is_possible_but_is_not_clinical_validation() -> None:
    """If this breaks, either nothing can ever pass, or approval overclaims."""
    decision = derive_status(**_status_kwargs())
    assert decision.status == str(ModelStatus.APPROVED_FOR_DEFINED_RESEARCH_USE)
    assert decision.may_serve is True
    payload = decision.to_dict()
    assert "not clinical validation" in payload["note"]
    assert "not clinical validation" in decision.rationale


def test_synthetic_model_refuses_real_evidence() -> None:
    """If this breaks, a simulation scores a real programme.

    The refusal must be a refusal, not a warning attached to a number.
    """
    decision = check_eligibility(
        status=str(ModelStatus.DEMO_ONLY),
        trained_on_synthetic=True,
        evidence_is_synthetic=False,
        demo_context_selected=True,
    )
    assert decision.permitted is False
    assert decision.refusal_code == str(RefusalCode.SYNTHETIC_MODEL_REAL_EVIDENCE)
    assert "not a meaningful quantity" in decision.reason


def test_demo_scores_require_an_explicit_demo_context() -> None:
    """If this breaks, a demonstration number appears without being asked for."""
    refused = check_eligibility(
        status=str(ModelStatus.DEMO_ONLY),
        trained_on_synthetic=True,
        evidence_is_synthetic=True,
        demo_context_selected=False,
    )
    assert refused.permitted is False
    assert refused.refusal_code == str(RefusalCode.DEMO_CONTEXT_NOT_SELECTED)

    allowed = check_eligibility(
        status=str(ModelStatus.DEMO_ONLY),
        trained_on_synthetic=True,
        evidence_is_synthetic=True,
        demo_context_selected=True,
    )
    assert allowed.permitted is True
    assert allowed.is_demonstration is True


def test_unknown_status_is_not_treated_as_approved() -> None:
    """If this breaks, a legacy artifact serves numbers nobody has vetted."""
    for status in (None, "unknown", str(ModelStatus.UNKNOWN)):
        decision = check_eligibility(
            status=status,
            trained_on_synthetic=False,
            evidence_is_synthetic=False,
        )
        assert decision.permitted is False
        assert decision.refusal_code == str(RefusalCode.STATUS_UNKNOWN)


def test_non_serving_statuses_are_refused() -> None:
    """If this breaks, default-deny becomes default-allow."""
    for status in (ModelStatus.INSUFFICIENT_DATA, ModelStatus.EVALUATION_FAILED):
        decision = check_eligibility(
            status=str(status),
            trained_on_synthetic=False,
            evidence_is_synthetic=False,
        )
        assert decision.permitted is False
        assert decision.refusal_code == str(RefusalCode.STATUS_NOT_PERMITTED)
