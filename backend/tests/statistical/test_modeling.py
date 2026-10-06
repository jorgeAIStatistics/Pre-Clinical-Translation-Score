"""Statistical correctness of the modelling layer.

These are not smoke tests. Every assertion below pins a numerical property that
the product's published output depends on: a coefficient that is actually the
penalised MAP, an explanation whose parts add up to the number on screen, an
interval that widens for the right reason, a split that does not leak, and a
selection gate that can both reject a flattering challenger and demote a broken
incumbent.

Nothing here touches the network or the database. Data is simulated with fixed
seeds so that a failure is a real change in behaviour rather than a new draw.

Where a test compares against `sklearn`, that is a cross-implementation check of
a hand-written optimiser, not an endorsement of sklearn's answer: the two solve
the same convex problem, so if they disagree, one of them has a bug and the
hand-written IRLS is the one this platform ships.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from trs.features.registry import FEATURES
from trs.features.spec import FeatureKind
from trs.features.transform import FeatureTransformer
from trs.modeling.base import sigmoid
from trs.modeling.calibration import (
    MIN_N_FOR_ISOTONIC,
    SLOPE_CORRECTION_PREFERENCE,
    IdentityCalibration,
    IsotonicCalibration,
    LogisticRecalibration,
    select_calibrator,
)
from trs.modeling.hierarchical import HierarchicalLogisticRisk
from trs.modeling.logistic import _INTERCEPT_PRECISION, L2LogisticRisk
from trs.modeling.uncertainty import predict_with_uncertainty
from trs.pipeline.calibrate import cross_validated_calibration
from trs.pipeline.select import (
    ATTRIBUTION_LOSS_SURCHARGE,
    CALIBRATION_SLOPE_BAND,
    MAX_ACCEPTABLE_LOG_LOSS,
    MIN_RELATIVE_IMPROVEMENT,
    POSTERIOR_LOSS_SURCHARGE,
    choose_family,
    required_margin,
)
from trs.validation.metrics import (
    calibration_intercept_slope,
    decision_curve,
    paired_bootstrap_difference,
    scaled_brier,
)
from trs.validation.splitters import (
    GROUP_SCOPE_KEYS,
    ConnectedComponentGrouper,
    GroupScope,
    grouped_kfold,
    leakage_audit,
    temporal_split,
)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers. Fixed seeds throughout; no fixture draws a fresh sample.
# ─────────────────────────────────────────────────────────────────────────────

def simulate_logistic(n: int, beta: np.ndarray, intercept: float, seed: int):
    """Design matrix and Bernoulli outcomes from a known logistic model."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, beta.size))
    p = sigmoid(X @ beta + intercept)
    y = (rng.random(n) < p).astype(float)
    return X, y


TRUE_BETA = np.array([1.5, -1.0, 0.75, 0.0, -0.5, 0.25])
TRUE_INTERCEPT = -0.4


# ─────────────────────────────────────────────────────────────────────────────
# 1. L2LogisticRisk — is the optimiser solving the problem it claims to solve?
# ─────────────────────────────────────────────────────────────────────────────

def test_l2_logistic_recovers_known_coefficients():
    """If this fails, every coefficient, odds ratio and driver the product shows
    is a number the optimiser invented rather than the maximum-a-posteriori fit
    to the evidence — the whole explanation layer becomes fiction."""
    X, y = simulate_logistic(6000, TRUE_BETA, TRUE_INTERCEPT, seed=5)
    model = L2LogisticRisk(l2_penalty=1e-3, max_iter=2000, tol=1e-12).fit(X, y)

    assert model.fit_report is not None
    assert model.fit_report.converged is True
    assert model.fit_report.n_rows == 6000
    assert model.fit_report.n_features == TRUE_BETA.size
    assert model.fit_report.n_positive == int(y.sum())

    # With n = 6000 and essentially no penalty, the MAP is the MLE and the
    # sampling error on each coefficient is well under 0.1.
    assert np.abs(model.coef_ - TRUE_BETA).max() < 0.15
    assert model.intercept_ == pytest.approx(TRUE_INTERCEPT, abs=0.15)
    # The zero coefficient must stay near zero: a fitter that smears signal
    # across columns would put a spurious driver in front of a user.
    assert abs(model.coef_[3]) < 0.1


@pytest.mark.parametrize("lam", [0.5, 1.0, 10.0, 128.0])
def test_l2_logistic_matches_sklearn_to_tight_tolerance(lam):
    """The IRLS here is hand-written so that the Hessian feeding the Laplace
    posterior is provably the matrix the optimiser converged on. That is only
    defensible if it lands on the same optimum as a mature implementation of the
    identical objective. Divergence means the shipped coefficients — and the
    posterior built from them — are wrong at the third decimal place or worse.

    λ = 128 is the penalty the current demonstration run actually selected, so
    the operating point is covered, not just round numbers.
    """
    X, y = simulate_logistic(800, TRUE_BETA, TRUE_INTERCEPT, seed=11)

    ours = L2LogisticRisk(l2_penalty=lam, max_iter=2000, tol=1e-14).fit(X, y)
    # sklearn minimises 0.5‖w‖² + C·Σ nll, i.e. Σ nll + (1/2C)‖w‖²; the module
    # here minimises Σ nll + (λ/2)‖β‖². So C = 1/λ. lbfgs leaves the intercept
    # unpenalised, matching _INTERCEPT_PRECISION = 1e-6 to within 1e-6.
    theirs = LogisticRegression(
        C=1.0 / lam, solver="lbfgs", max_iter=20000, tol=1e-12
    ).fit(X, y)

    assert np.abs(ours.coef_ - theirs.coef_.ravel()).max() < 1e-6
    assert ours.intercept_ == pytest.approx(float(theirs.intercept_[0]), abs=1e-6)
    assert np.abs(ours.predict_proba(X) - theirs.predict_proba(X)[:, 1]).max() < 1e-6


def test_l2_logistic_rejects_use_before_fit():
    """An unfitted model must refuse rather than return a silent zero-logit
    (p = 0.5), which would look like a real estimate in the API response."""
    with pytest.raises(RuntimeError, match="before fit"):
        L2LogisticRisk().predict_logit(np.zeros((1, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attribution — the explanation must be the prediction, not a story about it
# ─────────────────────────────────────────────────────────────────────────────

def test_attribution_sums_exactly_to_the_predicted_logit():
    """The product promises that the drivers it lists ARE the score: every bar
    in the explanation is a term of ``b + Σ βⱼzⱼ``. If the terms stop summing to
    the prediction, the platform is showing a post-hoc rationalisation while
    claiming exactness — the single most damaging thing it could do."""
    X, y = simulate_logistic(600, TRUE_BETA, TRUE_INTERCEPT, seed=11)
    model = L2LogisticRisk(l2_penalty=2.0).fit(X, y)

    rng = np.random.default_rng(1234)
    X_new = rng.standard_normal((200, TRUE_BETA.size)) * 3.0  # deliberately wide

    contributions = model.attribution(X_new)
    assert contributions.shape == X_new.shape

    reconstructed = contributions.sum(axis=1) + model.intercept_at_reference()
    direct = model.predict_logit(X_new)
    # Only floating-point summation order separates the two expressions.
    assert np.abs(reconstructed - direct).max() < 1e-12

    # The reference row (cohort average in standardised space) contributes
    # nothing, so the intercept alone is the average program's log-odds.
    assert model.attribution(np.zeros((1, TRUE_BETA.size))).sum() == 0.0
    assert model.intercept_at_reference() == model.intercept_
    assert model.supports_exact_attribution is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Laplace posterior — the basis for every published interval
# ─────────────────────────────────────────────────────────────────────────────

def test_posterior_draws_centre_on_the_point_estimate():
    """Every interval the product publishes is a quantile of these draws. If the
    draws are not centred on the point estimate, the interval is off-centre from
    the number displayed beside it and the two contradict each other."""
    X, y = simulate_logistic(600, TRUE_BETA, TRUE_INTERCEPT, seed=11)
    model = L2LogisticRisk(l2_penalty=2.0).fit(X, y)

    rng = np.random.default_rng(7)
    X_new = np.random.default_rng(21).standard_normal((8, TRUE_BETA.size))
    draws = model.posterior_logit_samples(X_new, 20_000, rng)

    assert draws.shape == (20_000, 8)
    assert np.abs(draws.mean(axis=0) - model.predict_logit(X_new)).max() < 0.01
    # The draws must actually move; a degenerate posterior would publish an
    # interval of zero width and assert certainty the model does not have.
    assert draws.std(axis=0).min() > 1e-3


def test_posterior_covariance_is_the_inverse_hessian_at_the_map():
    """The published interval rests on the claim that the posterior is the
    Laplace approximation N(β̂, H⁻¹) with H = XᵀWX + Λ evaluated at the MAP. This
    re-derives both H and the MAP condition from scratch. If either drifts, the
    interval is a Gaussian around the right point with the wrong width, and
    "honest uncertainty" becomes a decorative claim.

    A non-PSD covariance additionally means the Cholesky factor is silently
    repaired and the "posterior" draws are no longer draws from anything.
    """
    lam = 1.0
    X, y = simulate_logistic(400, TRUE_BETA, TRUE_INTERCEPT, seed=3)
    model = L2LogisticRisk(l2_penalty=lam).fit(X, y)

    cov = np.asarray(model.to_state()["posterior_cov"], dtype=float)
    assert cov.shape == (TRUE_BETA.size + 1, TRUE_BETA.size + 1)
    assert np.array_equal(cov, cov.T), "covariance must be exactly symmetric"

    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > 0.0, "H⁻¹ must be positive definite, not merely PSD"

    # Rebuild the penalised objective's gradient and Hessian independently.
    beta = np.concatenate([[model.intercept_], model.coef_])
    Xa = np.hstack([np.ones((X.shape[0], 1)), X])
    mu = sigmoid(Xa @ beta)
    prior = np.concatenate([[_INTERCEPT_PRECISION], np.full(TRUE_BETA.size, lam)])

    gradient = Xa.T @ (mu - y) + prior * beta
    assert np.abs(gradient).max() < 1e-6, "fit did not converge to a stationary point"

    hessian = Xa.T @ (Xa * (mu * (1 - mu))[:, None]) + np.diag(prior)
    assert np.abs(cov @ hessian - np.eye(hessian.shape[0])).max() < 1e-8

    # Posterior standard deviations surfaced in the coefficient table must be
    # the square roots of this matrix's diagonal, not a separate estimate.
    table = model.coefficient_table()
    diag_sd = np.sqrt(np.diag(cov))[1:]
    assert [r["posterior_sd"] for r in table] == pytest.approx(diag_sd.tolist())
    for row, sd in zip(table, diag_sd, strict=True):
        assert row["ci_low"] < row["coefficient"] < row["ci_high"]
        assert row["ci_high"] - row["ci_low"] == pytest.approx(2 * 1.959964 * sd)


def test_model_state_round_trip_is_bit_identical():
    """Artifacts are serialised to JSON and reloaded to serve scores. If a
    reload changes a prediction at all, a program's published risk depends on
    which process answered the request, and scores stop being reproducible or
    auditable."""
    X, y = simulate_logistic(500, TRUE_BETA, TRUE_INTERCEPT, seed=11)
    model = L2LogisticRisk(l2_penalty=4.0).fit(X, y, columns=[f"f{i}" for i in range(6)])

    # Through JSON, exactly as the artifact store does it.
    reloaded = L2LogisticRisk.from_state(json.loads(json.dumps(model.to_state())))

    X_new = np.random.default_rng(99).standard_normal((50, TRUE_BETA.size))
    assert reloaded.columns == model.columns
    assert np.array_equal(reloaded.predict_logit(X_new), model.predict_logit(X_new))
    assert np.array_equal(reloaded.predict_proba(X_new), model.predict_proba(X_new))
    assert np.array_equal(reloaded.attribution(X_new), model.attribution(X_new))
    assert reloaded.intercept_at_reference() == model.intercept_at_reference()

    # The posterior must survive too, or reloaded artifacts would publish
    # different intervals from the process that trained them.
    before = model.posterior_logit_samples(X_new, 64, np.random.default_rng(3))
    after = reloaded.posterior_logit_samples(X_new, 64, np.random.default_rng(3))
    assert np.array_equal(before, after)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hierarchical logistic — partial pooling
# ─────────────────────────────────────────────────────────────────────────────

def _hierarchical_data(n_per_group: int = 120, seed: int = 99):
    rng = np.random.default_rng(seed)
    offsets = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    groups = np.repeat([f"g{i}" for i in range(offsets.size)], n_per_group)
    n = offsets.size * n_per_group
    X = rng.standard_normal((n, 3))
    beta = np.array([0.8, -0.5, 0.3])
    p = sigmoid(X @ beta + np.repeat(offsets, n_per_group))
    y = (rng.random(n) < p).astype(float)
    return X, y, groups, offsets, beta


def test_hierarchical_recovers_group_offsets():
    """The hierarchical family exists to stop one mechanism class's base rate
    being forced onto another. If the group offsets are not recovered, the
    family adds parameters and noise without adding the structure it was
    admitted for."""
    X, y, groups, offsets, beta = _hierarchical_data()
    model = HierarchicalLogisticRisk(l2_penalty=1.0, group_precision=0.25).fit(
        X, y, groups=groups
    )

    assert model.fit_report is not None
    assert model.fit_report.diagnostics["n_groups_modelled"] == offsets.size
    assert model.fit_report.diagnostics["n_groups_observed"] == offsets.size

    estimated = np.array([model.group_effects_[f"g{i}"] for i in range(offsets.size)])
    assert np.corrcoef(estimated, offsets)[0, 1] > 0.95
    assert np.abs(estimated - offsets).max() < 0.35
    # Fixed effects must not be distorted by the group structure.
    assert np.abs(model.coef_ - beta).max() < 0.2

    # Group effects genuinely move predictions.
    X_new = np.zeros((1, 3))
    lowest = model.predict_logit(X_new, groups=np.array(["g0"]))[0]
    highest = model.predict_logit(X_new, groups=np.array(["g5"]))[0]
    assert highest - lowest == pytest.approx(
        model.group_effects_["g5"] - model.group_effects_["g0"]
    )
    assert highest > lowest


def test_hierarchical_unseen_group_falls_back_to_population_mean():
    """A program in a mechanism class the model never saw must be scored at the
    population mean. Anything else invents a group effect from no observations —
    the product would report a class-specific risk it has no evidence for."""
    X, y, groups, _offsets, _beta = _hierarchical_data()
    model = HierarchicalLogisticRisk(l2_penalty=1.0, group_precision=1.0).fit(
        X, y, groups=groups
    )
    # Every training group clears min_group_size, so there is no pooled
    # "small groups" level for an unseen group to inherit.
    assert "__small__" not in model.group_levels_

    X_new = np.random.default_rng(4).standard_normal((5, 3))
    population = model.predict_logit(X_new)
    unseen = model.predict_logit(X_new, groups=np.array(["never_seen"] * 5))
    assert np.array_equal(unseen, population), "unseen group effect must be exactly 0"

    # And a seen group must differ, or the fallback would be indistinguishable
    # from the model ignoring groups entirely.
    seen = model.predict_logit(X_new, groups=np.array(["g0"] * 5))
    assert not np.allclose(seen, population)


def test_hierarchical_rare_groups_are_pooled_not_given_their_own_column():
    """Groups below ``min_group_size`` must share one pooled level. Fitting an
    intercept from two programs would hand a user a mechanism-class effect
    estimated on two observations and present it like any other."""
    X, y, groups, _offsets, _beta = _hierarchical_data()
    groups = groups.copy()
    groups[:3] = "rare_class"  # 3 rows, below the default min_group_size of 5
    model = HierarchicalLogisticRisk(l2_penalty=1.0, group_precision=1.0).fit(
        X, y, groups=groups
    )
    assert "rare_class" not in model.group_levels_
    assert "__small__" in model.group_levels_

    X_new = np.zeros((2, 3))
    rare = model.predict_logit(X_new, groups=np.array(["rare_class"] * 2))
    pooled = model.predict_logit(X_new, groups=np.array(["__small__"] * 2))
    assert np.array_equal(rare, pooled)


def test_higher_group_precision_shrinks_group_effects_monotonically():
    """``group_precision`` is 1/τ² — the prior precision on group effects. If
    raising it did not shrink the effects toward the population mean, the
    hyperparameter grid searched during training would be searching nothing and
    the "amount of pooling is learned" claim would be false."""
    X, y, groups, _offsets, _beta = _hierarchical_data()

    magnitudes = []
    for precision in (0.25, 1.0, 4.0, 16.0, 64.0, 256.0):
        model = HierarchicalLogisticRisk(
            l2_penalty=1.0, group_precision=precision
        ).fit(X, y, groups=groups)
        effects = np.array([model.group_effects_[lv] for lv in model.group_levels_])
        magnitudes.append(float(np.abs(effects).max()))
        assert model.fit_report.diagnostics["implied_tau"] == pytest.approx(
            1.0 / np.sqrt(precision)
        )

    assert all(
        later < earlier
        for earlier, later in zip(magnitudes[:-1], magnitudes[1:], strict=True)
    ), f"group effects must shrink monotonically, got {magnitudes}"
    # Shrinkage must be substantial at the tight end, not a rounding effect.
    assert magnitudes[-1] < 0.2 * magnitudes[0]


def test_hierarchical_requires_groups():
    """Silently fitting a pooled model when groups are absent would mean the
    artifact says "hierarchical" while the fit is not."""
    X, y, _groups, _offsets, _beta = _hierarchical_data(n_per_group=10)
    with pytest.raises(ValueError, match="requires `groups`"):
        HierarchicalLogisticRisk().fit(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Calibration
# ─────────────────────────────────────────────────────────────────────────────

def test_logistic_recalibration_recovers_a_known_slope():
    """A deliberately over-confident model is the failure mode a temporally
    shifted deployment reliably produces. If recalibration cannot recover the
    slope, the product publishes "85%" for programs that fail 70% of the time
    and a committee acts on it."""
    n = 50_000
    rng = np.random.default_rng(2024)
    true_logit = rng.normal(0.0, 1.4, n)
    y = (rng.random(n) < sigmoid(true_logit)).astype(float)

    true_slope = 0.6
    overconfident = true_logit / true_slope  # too extreme by construction

    cal = LogisticRecalibration().fit(overconfident, y)
    assert cal.slope == pytest.approx(true_slope, abs=0.03)
    assert cal.intercept == pytest.approx(0.0, abs=0.03)
    assert cal.n_fit == n
    assert cal.supports_parameter_uncertainty is True

    # Recalibration must actually pull the probabilities in.
    raw_p = sigmoid(overconfident)
    cal_p = cal.transform(overconfident)
    assert np.abs(cal_p - 0.5).max() < np.abs(raw_p - 0.5).max()
    assert calibration_intercept_slope(y, cal_p)["slope"] == pytest.approx(1.0, abs=0.05)

    # Parameter covariance must be a usable 2×2 with positive variances.
    cov = np.asarray(cal.covariance, dtype=float)
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) > 0)
    assert np.linalg.eigvalsh(cov).min() > 0

    # Round-trip through the artifact payload.
    reloaded = LogisticRecalibration._load(json.loads(json.dumps(cal.to_dict())))
    assert np.array_equal(reloaded.transform_logit(overconfident[:100]),
                          cal.transform_logit(overconfident[:100]))


def test_isotonic_output_is_bounded_by_the_rule_of_succession():
    """An unbounded isotonic fit whose endpoint bin is pure asserts p = 0, i.e.
    an infinite log-odds. One confidently-wrong test row then dominates every
    aggregate metric — during development this drove held-out log loss from 0.70
    to 1.89. The bound is the only thing standing between the product and that."""
    n = 400
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 2.0, n)
    y = (rng.random(n) < sigmoid(z)).astype(float)

    cal = IsotonicCalibration().fit(z, y)
    bound = 1.0 / (n + 2.0)
    assert cal.n_fit == n
    assert cal.probability_bound == pytest.approx(bound)
    assert cal.supports_parameter_uncertainty is False

    # The underlying isotonic fit really does saturate at 0 and 1 here, so the
    # clip is load-bearing rather than incidental.
    assert cal._iso.predict(np.array([-1e6, 1e6])).tolist() == [0.0, 1.0]

    extreme = cal.transform(np.array([-1e6, 1e6]))
    assert extreme[0] == pytest.approx(bound)
    assert extreme[1] == pytest.approx(1.0 - bound)
    assert extreme[0] > 0.0 and extreme[1] < 1.0
    assert np.all(np.isfinite(cal.transform_logit(np.array([-1e6, 1e6]))))

    # No input anywhere on the real line may escape the bound.
    sweep = cal.transform(np.linspace(-50.0, 50.0, 2001))
    assert sweep.min() >= bound - 1e-12
    assert sweep.max() <= 1.0 - bound + 1e-12

    # Shape is preserved for an (S, M) draw block, which the uncertainty layer
    # passes in whole.
    block = cal.transform_logit(np.zeros((4, 7)))
    assert block.shape == (4, 7)


def test_select_calibrator_excludes_isotonic_below_the_sample_size_floor():
    """Below the floor, isotonic overfits the calibration split and produces
    degenerate endpoint bins. If the exclusion regresses, a noisy
    cross-validation can hand the product a step-function calibrator on 200
    points and the published probabilities become unstable."""
    rng = np.random.default_rng(31)
    small = MIN_N_FOR_ISOTONIC - 100
    z_small = rng.normal(0.0, 1.5, small)
    y_small = (rng.random(small) < sigmoid(z_small)).astype(float)

    chosen, record = select_calibrator(z_small, y_small)
    assert record["n_calibration"] == small
    assert "isotonic" in record["excluded_candidates"]
    assert str(MIN_N_FOR_ISOTONIC) in record["excluded_candidates"]["isotonic"]
    assert "isotonic" not in record["candidate_scores"]
    assert chosen.method != "isotonic"

    # Above the floor it must be admitted again — otherwise the "floor" is an
    # unconditional ban wearing a threshold's clothes.
    large = MIN_N_FOR_ISOTONIC + 100
    z_large = rng.normal(0.0, 1.5, large)
    y_large = (rng.random(large) < sigmoid(z_large)).astype(float)
    _, record_large = select_calibrator(z_large, y_large)
    assert "isotonic" not in record_large["excluded_candidates"]
    assert "isotonic" in record_large["candidate_scores"]


def test_slope_correction_tie_break_prefers_logistic_within_its_margin():
    """The tie-break stops a 0.01-nat cross-validation fluke selecting
    ``identity`` and leaving an over-confident model uncorrected — which is
    exactly what happened during development. It must fire when logistic is
    within the preference margin and must NOT fire when logistic is genuinely
    worse, or the rule would become "always pick logistic"."""
    applied_seen = False
    genuinely_worse_seen = False

    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 60
        # Already well calibrated and high-signal: identity is the truth, and
        # logistic's two extra parameters cost out-of-fold variance.
        z = rng.normal(0.0, 5.0, n)
        y = (rng.random(n) < sigmoid(z)).astype(float)

        chosen, record = select_calibrator(
            z, y, candidates=(LogisticRecalibration, IdentityCalibration)
        )
        scores = record["candidate_scores"]
        if set(scores) != {"logistic", "identity"}:
            continue

        best_method = min(scores, key=scores.__getitem__)
        within_margin = (
            scores["logistic"] <= scores[best_method] * (1.0 + SLOPE_CORRECTION_PREFERENCE)
        )

        if within_margin:
            assert record["method"] == "logistic"
            assert isinstance(chosen, LogisticRecalibration)
            if best_method != "logistic":
                assert record["slope_correction_tie_break_applied"] is True
                assert "slope-correction rule" in record["reason"]
                applied_seen = True
            else:
                assert record["slope_correction_tie_break_applied"] is False
        else:
            # Logistic lost by more than the margin: the winner stands.
            assert record["method"] == best_method != "logistic"
            assert record["slope_correction_tie_break_applied"] is False
            genuinely_worse_seen = True

    assert applied_seen, "no seed exercised the tie-break branch"
    assert genuinely_worse_seen, "no seed exercised the margin-exceeded branch"


def test_select_calibrator_falls_back_when_a_class_is_too_rare():
    """With one or two positives there is nothing to cross-validate. Failing
    loudly or picking isotonic here would break training on a small or heavily
    skewed cohort — the situation this platform is built for."""
    z = np.linspace(-2.0, 2.0, 40)
    y = np.zeros(40)
    y[:1] = 1.0  # a single positive
    chosen, record = select_calibrator(z, y)
    assert isinstance(chosen, LogisticRecalibration)
    assert record["candidate_scores"] == {}
    assert "Insufficient class balance" in record["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. predict_with_uncertainty — the interval the product publishes
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_SPECS = [s for s in FEATURES if not s.is_nominal]
_NOMINAL_SPECS = [s for s in FEATURES if s.is_nominal]


def _feature_row(rng: np.random.Generator) -> dict:
    """A fully observed feature row, valid under the declared schema."""
    row: dict = {}
    for spec in _NUMERIC_SPECS:
        if spec.kind is FeatureKind.BINARY:
            row[spec.name] = float(rng.integers(0, 2))
        elif spec.kind is FeatureKind.COUNT:
            row[spec.name] = float(rng.integers(0, 12))
        elif spec.kind is FeatureKind.ORDINAL:
            lo = int(spec.valid_min if spec.valid_min is not None else 0)
            hi = int(spec.valid_max if spec.valid_max is not None else 3)
            row[spec.name] = float(rng.integers(lo, hi + 1))
        else:
            lo = spec.valid_min if spec.valid_min is not None else -2.0
            hi = spec.valid_max if spec.valid_max is not None else 2.0
            row[spec.name] = float(rng.uniform(lo, hi))
    for spec in _NOMINAL_SPECS:
        row[spec.name] = str(rng.choice(list(spec.levels)))
    return row


@pytest.fixture(scope="module")
def scoring_stack():
    """A fitted transformer, model and calibrator over simulated feature rows."""
    rng = np.random.default_rng(0)
    rows = [_feature_row(rng) for _ in range(400)]
    transformer = FeatureTransformer().fit(rows, random_state=0)
    X = transformer.transform(rows)

    weights = np.random.default_rng(1).normal(0.0, 0.25, X.shape[1])
    y = (np.random.default_rng(2).random(len(rows)) < sigmoid(X @ weights)).astype(float)

    model = L2LogisticRisk(l2_penalty=64.0).fit(X, y, columns=transformer.columns)
    calibrator = LogisticRecalibration().fit(model.predict_logit(X), y)
    return transformer, model, calibrator, X, y


def test_uncertainty_interval_is_a_valid_ordered_probability_interval(scoring_stack):
    """These three numbers are the product's headline output. A bound outside
    [0, 1] or an estimate sitting outside its own interval is an immediately
    visible contradiction in the UI and the API."""
    transformer, model, calibrator, _X, _y = scoring_stack
    row = _feature_row(np.random.default_rng(42))

    result = predict_with_uncertainty(
        model, calibrator, transformer, row,
        rng=np.random.default_rng(7), n_imputations=25, n_param_samples=200,
    )

    assert 0.0 <= result.interval_low <= result.probability <= result.interval_high <= 1.0
    assert result.interval_level == 0.95
    assert result.interval_width == pytest.approx(
        result.interval_high - result.interval_low
    )
    assert result.n_imputations == 25
    assert result.n_param_samples == 200
    assert set(result.components_sd_logit) == {"parameter", "imputation", "calibration"}
    assert all(v >= 0.0 for v in result.components_sd_logit.values())
    assert sum(result.variance_share.values()) == pytest.approx(1.0)
    assert result.dominant_source in result.components_sd_logit
    assert result.method.startswith("laplace_posterior")
    assert "calibration_covariance" in result.method

    payload = result.to_dict()
    assert payload["interval_low"] == result.interval_low
    assert payload["dominant_source"] == result.dominant_source

    # A tighter interval level must produce a narrower band.
    narrow = predict_with_uncertainty(
        model, calibrator, transformer, row,
        rng=np.random.default_rng(7), n_imputations=25, n_param_samples=200, level=0.5,
    )
    assert narrow.interval_width < result.interval_width


def test_missing_features_strictly_widen_the_imputation_component(scoring_stack):
    """The central honesty claim of the interval is that it widens because of
    what is *not known*. If a sparse evidence package scored no wider than a
    complete one, the platform would be quoting equal confidence for unequal
    evidence."""
    transformer, model, calibrator, _X, _y = scoring_stack
    complete = _feature_row(np.random.default_rng(42))

    sparse = dict(complete)
    for spec in _NUMERIC_SPECS[:12]:
        sparse[spec.name] = None
    sparser = dict(complete)
    for spec in _NUMERIC_SPECS[:30]:
        sparser[spec.name] = None

    def score(row):
        return predict_with_uncertainty(
            model, calibrator, transformer, row,
            rng=np.random.default_rng(7), n_imputations=40, n_param_samples=200,
        )

    r_complete, r_sparse, r_sparser = score(complete), score(sparse), score(sparser)

    # A complete row has nothing to impute, so the component is exactly zero.
    assert r_complete.components_sd_logit["imputation"] == 0.0
    assert r_sparse.components_sd_logit["imputation"] > 0.0
    assert (
        r_sparse.components_sd_logit["imputation"]
        < r_sparser.components_sd_logit["imputation"]
    )
    assert r_complete.variance_share["imputation"] == 0.0

    # And the user must be told, with the right count.
    assert not any("unobserved" in c for c in r_complete.caveats)
    assert any("12 of " in c and "unobserved" in c for c in r_sparse.caveats)
    assert any("30 of " in c and "unobserved" in c for c in r_sparser.caveats)


def test_calibration_component_is_exactly_zero_without_parameter_uncertainty(
    scoring_stack,
):
    """When isotonic is selected there is no parameter covariance to propagate.
    Reporting a non-zero calibration component would be fabricating a number;
    reporting zero without the caveat would hide that the interval is too
    narrow. Both must hold."""
    transformer, model, _cal, X, y = scoring_stack
    isotonic = IsotonicCalibration().fit(model.predict_logit(X), y)
    assert isotonic.supports_parameter_uncertainty is False

    row = _feature_row(np.random.default_rng(42))
    result = predict_with_uncertainty(
        model, isotonic, transformer, row,
        rng=np.random.default_rng(7), n_imputations=10, n_param_samples=120,
    )

    assert result.components_sd_logit["calibration"] == 0.0
    assert result.variance_share["calibration"] == 0.0
    assert "calibration_covariance" not in result.method
    assert any(
        "isotonic" in c and "not included" in c for c in result.caveats
    ), result.caveats

    # The logistic calibrator, by contrast, must contribute something.
    logistic_result = predict_with_uncertainty(
        model, _cal, transformer, row,
        rng=np.random.default_rng(7), n_imputations=10, n_param_samples=120,
    )
    assert logistic_result.components_sd_logit["calibration"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Splitters — leakage control
# ─────────────────────────────────────────────────────────────────────────────

def test_connected_components_unite_rows_transitively():
    """Grouping on the program id alone misses one asset carrying two
    development codes. Without the transitive closure, a near-duplicate of a
    test row sits in training and the held-out AUROC the product publishes is
    measuring memorisation."""
    keys = [
        {"program": "P1", "compound": None, "target": "T1"},   # 0  A
        {"program": "P1", "compound": "C1", "target": "T2"},   # 1  B — A via program
        {"program": "P2", "compound": "C1", "target": "T3"},   # 2  C — B via compound
        {"program": "P3", "compound": "C2", "target": "T1"},   # 3  D — shares only target
        {"program": None, "compound": None, "target": None},   # 4  E — no keys at all
    ]

    identity = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.IDENTITY]).fit(keys)
    assert identity[0] == identity[1] == identity[2], "A–B–C must close transitively"
    assert identity[3] != identity[0], "a shared target is not identity leakage"
    assert identity[4] != identity[0] and identity[4] != identity[3]
    assert len(set(identity.tolist())) == 3

    strict = ConnectedComponentGrouper(GROUP_SCOPE_KEYS[GroupScope.STRICT]).fit(keys)
    assert strict[0] == strict[3], "STRICT must also unite on target"
    assert len(set(strict.tolist())) == 2

    summary = ConnectedComponentGrouper.summary(identity)
    assert summary["n_rows"] == 5
    assert summary["n_components"] == 3
    assert summary["largest_component"] == 3
    assert summary["singleton_components"] == 2


def test_temporal_split_never_leaves_a_component_spanning_partitions():
    """The headline held-out metric is the temporal split. If a component
    straddles the boundary, a program's later sibling trains the model that
    scores its earlier self, and the reported AUROC is inflated by leakage the
    report claims to have removed."""
    dates = [dt.date(2015 + (i % 8), 1 + (i % 12), 1) for i in range(120)]
    # Component 0 deliberately spans the entire date range.
    components = np.array([0 if i % 17 == 0 else 1 + i % 25 for i in range(120)])

    split = temporal_split(
        dates, components,
        train_end=dt.date(2018, 12, 31), calibration_end=dt.date(2020, 12, 31),
    )

    partition = np.empty(len(dates), dtype=object)
    partition[split.train] = "train"
    partition[split.calibration] = "calibration"
    partition[split.test] = "test"

    for comp in np.unique(components):
        assigned = set(partition[components == comp].tolist())
        assert len(assigned) == 1, f"component {comp} spans partitions {assigned}"

    # Partitions are disjoint and exhaustive; the splitter reassigns straddling
    # rows to the later side rather than deleting them, and reports how many.
    all_idx = np.concatenate([split.train, split.calibration, split.test])
    assert sorted(all_idx.tolist()) == list(range(120))
    assert split.dropped["component_straddled_boundary"] > 0
    assert split.sizes()["train"] + split.sizes()["calibration"] + split.sizes()[
        "test"
    ] == 120

    # A straddling component must land in the LATEST partition it touches —
    # the conservative direction.
    assert set(components[split.train].tolist()).isdisjoint({0})
    assert partition[components == 0][0] == "test"

    # Dates must still respect the boundary for non-straddling train rows.
    train_dates = [dates[i] for i in split.train]
    assert max(train_dates) <= dt.date(2018, 12, 31)


def test_grouped_kfold_folds_are_disjoint_and_cover_every_row():
    """Hyperparameter selection runs on these folds. A row appearing in two test
    folds is counted twice; a row in none is silently excluded — either way the
    selected penalty is chosen against a cohort that does not exist."""
    rng = np.random.default_rng(3)
    components = rng.integers(0, 40, size=300)
    y = (rng.random(300) < 0.4).astype(int)

    folds = list(grouped_kfold(components, n_splits=5, y=y, random_state=0))
    assert len(folds) == 5

    seen: list[int] = []
    for train_idx, test_idx in folds:
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
        assert sorted(train_idx.tolist() + test_idx.tolist()) == list(range(300))
        # No component may appear on both sides of a fold.
        assert set(components[train_idx].tolist()).isdisjoint(
            components[test_idx].tolist()
        )
        # Balancing must leave every fold with both classes.
        assert len(np.unique(y[test_idx])) == 2
        seen.extend(test_idx.tolist())

    assert sorted(seen) == list(range(300)), "test folds must partition the rows"


def test_leakage_audit_distinguishes_a_clean_split_from_a_corrupted_one():
    """This audit is what lets the report *assert* rather than assume the split
    is clean. An audit that cannot detect deliberate contamination gives the
    validation report a rubber stamp."""
    keys = [
        {"program": f"P{i}", "compound": f"C{i}", "target": f"T{i % 5}",
         "sponsor": f"S{i % 3}", "mechanism": f"M{i % 4}"}
        for i in range(40)
    ]
    components = ConnectedComponentGrouper(
        GROUP_SCOPE_KEYS[GroupScope.IDENTITY]
    ).fit(keys)

    train_idx = np.arange(0, 30)
    test_idx = np.arange(30, 40)

    clean = leakage_audit(components, train_idx, test_idx, keys)
    assert clean["clean"] is True
    assert clean["component"] == 0
    assert clean["program"] == 0
    assert clean["compound"] == 0
    assert clean["scope"] == str(GroupScope.IDENTITY)
    # Target/sponsor/mechanism overlap is reported as context, not failure.
    assert clean["informational_overlap"]["target"] > 0
    assert clean["informational_overlap"]["sponsor"] > 0

    # Deliberate contamination: one test program also appears in training.
    corrupted_keys = [dict(k) for k in keys]
    corrupted_keys[0]["program"] = corrupted_keys[35]["program"]
    corrupted_components = ConnectedComponentGrouper(
        GROUP_SCOPE_KEYS[GroupScope.IDENTITY]
    ).fit(corrupted_keys)

    dirty = leakage_audit(corrupted_components, train_idx, test_idx, corrupted_keys)
    assert dirty["clean"] is False
    assert dirty["component"] == 1
    assert dirty["program"] == 1


def test_calibrator_is_never_fitted_on_a_score_from_a_model_that_saw_that_row():
    """If this fails, the calibration curve is fitted on in-sample scores.

    ``cross_validated_calibration`` is the one place a leak would not show up as
    a split violation: the model and the calibrator are both fitted on the
    training partition, so ``leakage_audit`` sees nothing wrong. The protection
    is entirely internal — each pooled score must come from a fold model that
    did not see the row it is scoring. Fit the fold models on all the rows
    instead and the pooled scores become in-sample, the calibrator learns to
    correct over-confidence the final model does not have, and every published
    probability is skewed with no diagnostic anywhere reporting it.

    The fold models are spied on rather than inspected indirectly: each row of
    the design matrix is a distinct continuous draw, so a fitted matrix can be
    mapped back to the exact row indices it contained.
    """
    rng = np.random.default_rng(5)
    n = 300
    X = rng.standard_normal((n, 4))
    components = rng.integers(0, 60, size=n)
    y = (rng.random(n) < sigmoid(X @ np.array([1.2, -0.8, 0.5, 0.0]))).astype(int)
    assert len(np.unique(y)) == 2

    row_of = {X[i].tobytes(): i for i in range(n)}
    assert len(row_of) == n, "rows must be uniquely identifiable for the spy"
    fitted_rows: list[set[int]] = []

    class SpyLogistic(L2LogisticRisk):
        def fit(self, X_fit, y_fit, **kwargs):
            fitted_rows.append({row_of[r.tobytes()] for r in np.asarray(X_fit)})
            return super().fit(X_fit, y_fit, **kwargs)

    _model, _cal, record = cross_validated_calibration(
        lambda: SpyLogistic(l2_penalty=4.0),
        X, y,
        components=components,
        columns=[f"f{i}" for i in range(4)],
        n_folds=5,
        random_state=0,
    )

    # The calibrator must have been fitted out of fold, not in sample. The
    # fallback path is legitimate on tiny data but not here, and it would make
    # every assertion below meaningless.
    assert record["design"] == "cross_validated_out_of_fold", record
    assert record["n_out_of_fold_predictions"] == n

    # grouped_kfold is deterministic in random_state, so the folds the
    # implementation used are re-derivable here.
    folds = [
        (tr, te) for tr, te in grouped_kfold(components, n_splits=5, y=y, random_state=0)
        if len(np.unique(y[tr])) >= 2
    ]
    assert folds
    assert len(fitted_rows) == len(folds) + 1, (
        "expected one fit per usable fold plus the final full-data refit; got "
        f"{len(fitted_rows)} fits for {len(folds)} folds"
    )

    # The final model is refitted on everything — that is intended, and it is
    # not what the calibrator was fitted on.
    assert fitted_rows[-1] == set(range(n))

    scored_rows: set[int] = set()
    for (train_idx, test_idx), seen in zip(folds, fitted_rows[:-1], strict=True):
        assert seen == set(train_idx.tolist())
        # The invariant: the rows this fold model scored are rows it never saw.
        assert seen.isdisjoint(test_idx.tolist())
        scored_rows |= set(test_idx.tolist())

    # ...and every row's calibration input came from exactly one such model.
    assert scored_rows == set(range(n))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _calibration_data(n: int = 4000, seed: int = 5):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, n)
    p_true = sigmoid(z)
    y = (rng.random(n) < p_true).astype(float)
    return y, z, p_true


def test_calibration_slope_detects_over_confidence():
    """The calibration slope is the diagnostic the model card and the selection
    gate both read. If it cannot tell a calibrated model from an over-confident
    one, the gate stops disqualifying broken candidates."""
    y, z, p_true = _calibration_data()

    perfect = calibration_intercept_slope(y, p_true)
    assert perfect["slope"] == pytest.approx(1.0, abs=0.1)
    assert perfect["intercept"] == pytest.approx(0.0, abs=0.1)
    assert perfect["slope_se"] > 0 and perfect["intercept_se"] > 0

    over_confident = calibration_intercept_slope(y, sigmoid(z / 0.6))
    assert over_confident["slope"] < 1.0
    assert over_confident["slope"] == pytest.approx(0.6, abs=0.1)

    under_confident = calibration_intercept_slope(y, sigmoid(z * 0.6))
    assert under_confident["slope"] > 1.0

    # Systematically low predictions must show up as a positive intercept.
    too_low = calibration_intercept_slope(y, sigmoid(z - 1.0))
    assert too_low["intercept"] == pytest.approx(1.0, abs=0.15)

    # Degenerate inputs must return None rather than a fabricated slope.
    assert calibration_intercept_slope(np.ones(50), np.full(50, 0.6))["slope"] is None
    assert calibration_intercept_slope(np.array([0, 1]), np.array([0.2, 0.8]))[
        "slope"
    ] is None


def test_scaled_brier_is_zero_for_a_base_rate_predictor():
    """Scaled Brier is how the platform decides whether it beats simply quoting
    the cohort attrition rate — the "not_distinguishable_from_base_rate" verdict
    depends on it. If the base-rate reference were mis-computed, the platform
    could claim skill it does not have."""
    y, _z, p_true = _calibration_data()

    base_rate = np.full(y.size, float(y.mean()))
    assert scaled_brier(y, base_rate) == pytest.approx(0.0, abs=1e-12)

    # A real signal must score positive; a deliberately inverted one, negative.
    assert scaled_brier(y, p_true) > 0.1
    assert scaled_brier(y, 1.0 - p_true) < 0.0

    assert scaled_brier(np.array([]), np.array([])) is None
    # A constant outcome has zero base-rate Brier: no skill score is definable.
    assert scaled_brier(np.ones(20), np.full(20, 0.5)) is None


def test_paired_bootstrap_difference_detects_a_real_difference():
    """This is the statistical evidence the selection gate requires before
    adopting a more complex model. A test that cannot separate a real
    improvement from noise either blocks every upgrade or waves through noise."""
    y, _z, p_true = _calibration_data()
    components = np.arange(y.size) // 2  # two rows per component
    base_rate = np.full(y.size, float(y.mean()))

    real = paired_bootstrap_difference(
        y, p_true, base_rate, components, metric="log_loss",
        n_bootstrap=300, random_state=0,
    )
    assert real["observed_difference"] < 0, "the better model must have lower log loss"
    assert real["ci_low"] < real["ci_high"] < 0
    assert real["excludes_zero"] is True
    assert real["n_bootstrap"] > 250
    assert real["ci_low"] <= real["observed_difference"] <= real["ci_high"]

    # Comparing a model with itself must produce exactly no difference.
    none = paired_bootstrap_difference(
        y, p_true, p_true.copy(), components, metric="log_loss",
        n_bootstrap=300, random_state=0,
    )
    assert none["observed_difference"] == 0.0
    assert none["excludes_zero"] is False

    # Too few usable replicates must report None rather than a 2-draw interval.
    tiny = paired_bootstrap_difference(
        y[:6], p_true[:6], base_rate[:6], np.arange(6), metric="log_loss",
        n_bootstrap=50, random_state=0,
    )
    assert tiny["excludes_zero"] is None
    assert tiny["ci_low"] is None


def test_decision_curve_net_benefit_behaves_at_the_prevalence_threshold():
    """Net benefit is the number a portfolio committee actually acts on. At a
    threshold equal to prevalence, "flag everything" must be worth exactly zero
    — the same as flagging nothing. If that identity breaks, the reference
    strategies the curve is read against are wrong and the chart misleads."""
    y, _z, p_true = _calibration_data()
    prevalence = float(y.mean())

    curve = decision_curve(y, p_true, thresholds=np.array([prevalence]))
    assert curve["prevalence"] == pytest.approx(prevalence)
    assert curve["treat_all"][0] == pytest.approx(0.0, abs=1e-12)
    assert curve["treat_none"][0] == 0.0
    # A model with real signal must beat both reference strategies there.
    assert curve["model"][0] > 0.02

    # A useless (constant) score flags everyone above its own value and
    # therefore cannot beat treat_all.
    useless = decision_curve(
        y, np.full(y.size, prevalence), thresholds=np.array([prevalence])
    )
    assert useless["model"][0] == pytest.approx(0.0, abs=1e-12)

    # Net benefit is bounded above by prevalence at any threshold.
    full = decision_curve(y, p_true)
    assert len(full["thresholds"]) == len(full["model"]) == len(full["treat_all"])
    assert max(full["model"]) <= prevalence + 1e-12
    # treat_all must fall monotonically as the threshold rises.
    assert all(
        b <= a + 1e-12
        for a, b in zip(full["treat_all"][:-1], full["treat_all"][1:], strict=True)
    )

    assert decision_curve(np.array([]), np.array([]))["model"] == []


# ─────────────────────────────────────────────────────────────────────────────
# 9. The selection gate
# ─────────────────────────────────────────────────────────────────────────────

def _candidate(family: str, log_loss: float, slope: float, **caps):
    return {
        "family": family,
        "eligible": True,
        "log_loss": log_loss,
        "auroc": 0.63,
        "brier": 0.238,
        "scaled_brier": 0.035,
        "calibration_slope": slope,
        "supports_exact_attribution": caps.get("attribution", True),
        "supports_analytic_posterior": caps.get("posterior", True),
    }


def _gate_inputs(n: int = 400, seed: int = 17):
    """Outcomes, per-family predictions and components for the paired test."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.2, n)
    y = (rng.random(n) < sigmoid(z)).astype(int)
    components = np.arange(n) // 2
    strong = sigmoid(z)                     # tracks the truth
    weak = np.full(n, float(y.mean()))      # base rate only
    return y, components, strong, weak


def test_required_margin_adds_a_surcharge_for_lost_capability():
    """The surcharges encode a stated product judgement: an uninterrogable model
    must be clearly better, not marginally better. If they stop applying, a
    black box can displace the explainable default on a coin-flip improvement."""
    full = _candidate("hierarchical_logistic", 0.60, 1.0)
    margin, reasons = required_margin(full)
    assert margin == pytest.approx(MIN_RELATIVE_IMPROVEMENT)
    assert len(reasons) == 1

    black_box = _candidate("gbdt", 0.60, 1.0, attribution=False, posterior=False)
    margin_bb, reasons_bb = required_margin(black_box)
    assert margin_bb == pytest.approx(
        MIN_RELATIVE_IMPROVEMENT + ATTRIBUTION_LOSS_SURCHARGE + POSTERIOR_LOSS_SURCHARGE
    )
    assert len(reasons_bb) == 3
    assert any("additive explanation" in r for r in reasons_bb)
    assert any("analytic posterior" in r for r in reasons_bb)


def test_gate_retains_the_baseline_when_the_margin_is_not_cleared():
    """Baseline privilege is what stops noise in a 149-row test set choosing the
    architecture. Without it the deployed family would churn between runs and
    nothing about the product would be reproducible."""
    y, components, strong, _weak = _gate_inputs()
    # A challenger that is better, but only just.
    challenger_preds = np.clip(strong + 0.001, 1e-6, 1 - 1e-6)

    candidates = [
        _candidate("l2_logistic", 0.600, 1.00),
        _candidate("gbdt", 0.595, 1.00, attribution=False, posterior=False),
    ]
    result = choose_family(
        candidates, y,
        {"l2_logistic": strong, "gbdt": challenger_preds},
        components, random_state=0,
    )

    assert result["chosen_family"] == "l2_logistic"
    assert result["baseline_family"] == "l2_logistic"
    assert result["baseline_demoted"] is None

    gbdt = next(c for c in result["comparisons"] if c["family"] == "gbdt")
    assert gbdt["adopted"] is False
    assert gbdt["required_margin"] == pytest.approx(
        MIN_RELATIVE_IMPROVEMENT + ATTRIBUTION_LOSS_SURCHARGE + POSTERIOR_LOSS_SURCHARGE
    )
    assert gbdt["relative_log_loss_improvement"] < gbdt["required_margin"]
    assert "does not clear the required" in gbdt["verdict"]
    assert "Retained the l2_logistic baseline" in result["rationale"]


def test_gate_disqualifies_a_challenger_outside_the_calibration_band():
    """The product's output is a probability. A model that ranks brilliantly but
    is systematically over-confident is worse than useless here, because users
    act on the number rather than the ranking — so discrimination must not be
    able to buy its way past calibration."""
    y, components, strong, weak = _gate_inputs()
    candidates = [
        _candidate("l2_logistic", 0.600, 1.00),
        # A large log-loss improvement, but a broken calibration slope.
        _candidate("gbdt", 0.400, 0.50, attribution=False, posterior=False),
    ]
    result = choose_family(
        candidates, y, {"l2_logistic": weak, "gbdt": strong}, components, random_state=0,
    )

    assert result["chosen_family"] == "l2_logistic"
    gbdt = next(c for c in result["comparisons"] if c["family"] == "gbdt")
    assert gbdt["adopted"] is False
    assert "disqualified" in gbdt["verdict"]
    assert str(CALIBRATION_SLOPE_BAND) in gbdt["verdict"]
    # Disqualification must short-circuit before the margin is even computed.
    assert "relative_log_loss_improvement" not in gbdt
    # 0.50 is inside the wider sanity band, so this is an adoption refusal, not
    # a "broken fit" refusal — the two must stay distinguishable.
    assert result["sanity_checks"]["gbdt"]["passed"] is True


def test_gate_promotes_a_passing_candidate_when_the_baseline_is_broken():
    """An early version of this gate kept a baseline whose calibrator had
    degenerated and whose held-out log loss was 1.89, purely because it was the
    incumbent. Baseline privilege must not become baseline immunity, or the
    platform can register and serve a demonstrably broken model."""
    y, components, strong, weak = _gate_inputs()
    broken_ll = MAX_ACCEPTABLE_LOG_LOSS + 0.69  # the 1.89 that actually happened
    candidates = [
        _candidate("l2_logistic", broken_ll, 1.00),
        _candidate("hierarchical_logistic", 0.620, 1.00),
    ]
    result = choose_family(
        candidates, y,
        {"l2_logistic": weak, "hierarchical_logistic": strong},
        components, random_state=0,
    )

    assert result["chosen_family"] == "hierarchical_logistic"
    assert result["baseline_family"] == "hierarchical_logistic"
    assert result["baseline_demoted"] is not None
    assert "failed its sanity check" in result["baseline_demoted"]
    assert result["sanity_checks"]["l2_logistic"]["passed"] is False
    assert "sanity ceiling" in result["sanity_checks"]["l2_logistic"]["reason"]
    assert result["sanity_checks"]["hierarchical_logistic"]["passed"] is True
    assert result["rationale"].startswith("The l2_logistic baseline failed")

    promoted = next(
        c for c in result["comparisons"] if c["family"] == "hierarchical_logistic"
    )
    assert promoted["is_baseline"] is True
    assert promoted["adopted"] is True


def test_gate_adopts_a_challenger_that_clears_every_bar():
    """The mirror image of the rejection tests: a gate that never promotes
    anything is not a gate, it is a hard-coded choice. A genuinely better,
    calibrated, fully-capable family must be able to win."""
    y, components, strong, weak = _gate_inputs()
    candidates = [
        _candidate("l2_logistic", 0.690, 1.00),
        _candidate("hierarchical_logistic", 0.500, 1.05),  # 27.5% better
    ]
    result = choose_family(
        candidates, y,
        {"l2_logistic": weak, "hierarchical_logistic": strong},
        components, random_state=0,
    )

    assert result["chosen_family"] == "hierarchical_logistic"
    assert result["baseline_family"] == "l2_logistic"
    assert result["baseline_demoted"] is None

    winner = next(
        c for c in result["comparisons"] if c["family"] == "hierarchical_logistic"
    )
    assert winner["adopted"] is True
    assert winner["required_margin"] == pytest.approx(MIN_RELATIVE_IMPROVEMENT)
    assert winner["relative_log_loss_improvement"] > MIN_RELATIVE_IMPROVEMENT
    assert winner["paired_bootstrap"]["excludes_zero"] is True
    assert winner["paired_bootstrap"]["observed_difference"] < 0
    assert "Adopted hierarchical_logistic" in result["rationale"]


def test_gate_refuses_to_choose_when_nothing_is_eligible():
    """Registering a model after every candidate failed would ship an untested
    artifact. Training must fail loudly instead."""
    with pytest.raises(RuntimeError, match="No candidate model fitted"):
        choose_family([], np.array([0, 1]), {}, np.array([0, 1]))

    y, components, strong, _weak = _gate_inputs()
    all_broken = [
        _candidate("l2_logistic", 5.0, 1.0),
        _candidate("gbdt", 4.0, 1.0, attribution=False, posterior=False),
    ]
    with pytest.raises(RuntimeError, match="No candidate model passed the sanity"):
        choose_family(
            all_broken, y, {"l2_logistic": strong, "gbdt": strong}, components
        )
