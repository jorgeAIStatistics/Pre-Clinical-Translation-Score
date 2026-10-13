"""Why the estimate is what it is — derived from the model, never narrated.

The rule this module exists to enforce
---------------------------------------
Every driver shown to a user is a term in the model's own arithmetic. The
deployed family is linear on the log-odds scale, so

    logit(p) = a + b·(β₀ + Σⱼ βⱼ zⱼ)

where ``a`` and ``b`` are the calibration intercept and slope. Each feature's
contribution is exactly ``b·βⱼ·zⱼ``, and the contributions sum — to floating
point — to the log-odds of the number on screen. :func:`explain` asserts that
identity rather than assuming it.

This is the difference between explanation and rationalisation. A language model
shown a score can produce fluent reasons for it; those reasons are generated
*from the score*, not from the mechanism that produced it, and they will sound
equally convincing if the score is wrong. Nothing in this platform works that
way. A language model may later re-word the structured output below, but the
drivers, their directions and their magnitudes all originate here.

Because the design matrix is standardised on the training mean, ``zⱼ = 0`` is
the cohort-average program. So each contribution reads as "how this program
differs from the average program in the cohort, in log-odds" — which is the
comparison a user is implicitly making anyway.

Aggregation
-----------
One feature can occupy several design columns: a value column, a
``__missing`` indicator, and one one-hot column per level. Those are summed
back into a single per-feature contribution before display, because a user
thinks in features, not design columns. The missing-indicator contribution is
kept separately as well, since "we don't know this" is a distinct and
reportable reason for a program's position.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from trs.features.registry import FEATURES_BY_NAME
from trs.features.spec import DOMAIN_LABELS, FeatureDomain
from trs.features.transform import FeatureTransformer
from trs.modeling.base import RiskModel
from trs.modeling.calibration import Calibrator, LogisticRecalibration
from trs.modeling.contract import FUNCTIONAL_DESCRIPTIONS, PRIMARY_FUNCTIONAL
from trs.ontology.enums import MissingnessReason

#: Contributions below this absolute log-odds magnitude are pooled into an
#: "other factors" bucket. 0.02 log-odds is under half a percentage point of
#: probability near the middle of the range — below the noise floor of the
#: underlying evidence, and not worth a user's attention.
MIN_REPORTABLE_CONTRIBUTION = 0.02


@dataclasses.dataclass(slots=True)
class Driver:
    feature: str
    label: str
    domain: str
    domain_label: str
    #: Signed contribution to the calibrated log-odds of failure.
    contribution: float
    #: The program's value, in original units.
    value: float | str | None
    #: Cohort mean for this feature, in original units.
    cohort_mean: float | None
    #: Standard deviations from the cohort mean.
    z_score: float | None
    direction: str          # "increases_risk" | "decreases_risk"
    is_missing: bool
    missing_reason: str | None
    #: Evidence rows that produced the value.
    evidence_ids: list[str]
    #: Approximate effect on the probability, holding everything else fixed.
    probability_delta: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(slots=True)
class Explanation:
    baseline_logit: float
    total_logit: float
    probability: float
    #: Probability of the cohort-average program under this model.
    baseline_probability: float
    drivers: list[Driver]
    increasing: list[Driver]
    decreasing: list[Driver]
    by_domain: list[dict[str, Any]]
    other_factors_contribution: float
    #: Internal check: |sum(contributions) + baseline - total| must be ~0. Both
    #: sides derive from the same terms, so this alone proves little.
    reconstruction_error: float
    method: str
    #: External check: ``|served − reconstructed|`` when the caller supplied the
    #: number it is going to display, else ``None``. ``None`` means the check
    #: did not run, which is not the same as it having passed — and it is the
    #: only one of the two that can catch explaining the wrong quantity.
    served_reconstruction_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_logit": self.baseline_logit,
            "baseline_probability": self.baseline_probability,
            "total_logit": self.total_logit,
            "probability": self.probability,
            "drivers": [d.to_dict() for d in self.drivers],
            "increasing_risk": [d.to_dict() for d in self.increasing],
            "decreasing_risk": [d.to_dict() for d in self.decreasing],
            "by_domain": self.by_domain,
            "other_factors_contribution": self.other_factors_contribution,
            "reconstruction_error": self.reconstruction_error,
            "served_reconstruction_error": self.served_reconstruction_error,
            "external_check_ran": self.served_reconstruction_error is not None,
            "explained_quantity": str(PRIMARY_FUNCTIONAL),
            "explained_quantity_description": FUNCTIONAL_DESCRIPTIONS[
                str(PRIMARY_FUNCTIONAL)
            ],
            "calibration_scale": "log_odds_affine",
            "method": self.method,
            "statement": (
                "Each contribution is a term in the model's own linear predictor "
                "on the log-odds scale. They sum exactly to the difference "
                "between this program's log-odds and the cohort-average "
                "program's. No part of this explanation is generated after the "
                "fact."
            ),
        }


class ExplanationUnavailable(RuntimeError):
    """Raised when the deployed family admits no exact attribution."""


def explain(
    model: RiskModel,
    calibrator: Calibrator,
    transformer: FeatureTransformer,
    feature_row: dict[str, Any],
    *,
    missingness: dict[str, str] | None = None,
    evidence_map: dict[str, list[str]] | None = None,
    top_k: int = 8,
    served_probability: float | None = None,
    served_tolerance: float = 1e-9,
) -> Explanation:
    """Decompose one prediction into exact per-feature contributions.

    ``served_probability`` is the number the caller is actually going to
    display. When supplied, it is compared against the probability this
    decomposition reconstructs, and a disagreement raises.

    That external comparison is the point. The predecessor checked only that
    ``baseline_logit + Σ contributions + other`` equalled ``total_logit`` — but
    both sides were built from the same vector of scaled terms, so the check
    verified that a partition of a sum sums to the sum. It could not detect
    that the decomposition was explaining the *wrong quantity*, and it did: the
    interface displayed a Monte-Carlo draw-mean while this function
    reconstructed the plug-in value. An internal consistency check cannot catch
    a mismatch with something it never looks at.
    """
    if not model.supports_exact_attribution:
        raise ExplanationUnavailable(
            f"Model family {model.family!r} admits no exact additive "
            "decomposition. Rather than present an approximation beside exact "
            "numbers elsewhere in the interface, no driver breakdown is shown."
        )

    missingness = missingness or {}
    evidence_map = evidence_map or {}
    state = transformer.state

    X = transformer.transform([feature_row])              # (1, n_columns)
    column_contrib = model.attribution(X)[0]               # β ⊙ z
    intercept = model.intercept_at_reference()

    # Calibration composes linearly on the log-odds scale for the logistic
    # recalibrator, so contributions scale by the slope exactly. For any other
    # calibrator the decomposition is reported on the *uncalibrated* scale and
    # labelled as such, rather than silently misattributing.
    if isinstance(calibrator, LogisticRecalibration):
        scale, shift = calibrator.slope, calibrator.intercept
        method = "exact_additive_logodds_calibrated"
    else:
        scale, shift = 1.0, 0.0
        method = "exact_additive_logodds_uncalibrated"

    scaled = column_contrib * scale
    baseline_logit = shift + scale * intercept
    total_logit = baseline_logit + float(scaled.sum())

    # ---- fold design columns back into features ---------------------------
    per_feature: dict[str, float] = {}
    per_feature_missing: dict[str, float] = {}
    for col, value in zip(state.columns, scaled, strict=True):
        base, _ = transformer.column_origin(col)
        if col.endswith("__missing"):
            per_feature_missing[base] = per_feature_missing.get(base, 0.0) + float(value)
        per_feature[base] = per_feature.get(base, 0.0) + float(value)

    from trs.modeling.base import sigmoid

    probability = float(sigmoid(total_logit))
    baseline_probability = float(sigmoid(baseline_logit))

    drivers: list[Driver] = []
    other = 0.0
    for name, contribution in per_feature.items():
        spec = FEATURES_BY_NAME.get(name)
        if spec is None:
            other += contribution
            continue
        if abs(contribution) < MIN_REPORTABLE_CONTRIBUTION:
            other += contribution
            continue

        raw_value = feature_row.get(name)
        is_missing = raw_value is None
        cohort_mean = None
        z = None
        if not spec.is_nominal:
            cohort_mean = state.means.get(name)
            scale_j = state.scales.get(name, 1.0)
            if raw_value is not None and cohort_mean is not None and scale_j:
                from trs.features.transform import _apply_transform

                transformed = _apply_transform(spec, raw_value)
                if transformed is not None:
                    z = float((transformed - cohort_mean) / scale_j)

        # Effect on probability of removing this contribution, holding the rest.
        delta = probability - float(sigmoid(total_logit - contribution))

        drivers.append(Driver(
            feature=name,
            label=spec.label,
            domain=str(spec.domain),
            domain_label=DOMAIN_LABELS[spec.domain],
            contribution=float(contribution),
            value=raw_value,
            cohort_mean=cohort_mean,
            z_score=z,
            direction="increases_risk" if contribution > 0 else "decreases_risk",
            is_missing=is_missing,
            missing_reason=(
                missingness.get(name) if is_missing
                and missingness.get(name) != MissingnessReason.PRESENT else None
            ),
            evidence_ids=list(evidence_map.get(name, [])),
            probability_delta=float(delta),
        ))

    drivers.sort(key=lambda d: abs(d.contribution), reverse=True)

    # ---- integrity checks --------------------------------------------------
    # (a) Internal: the terms partition the linear predictor. Necessary, and on
    #     its own worth little — both sides derive from the same terms.
    reconstructed = baseline_logit + sum(d.contribution for d in drivers) + other
    reconstruction_error = abs(reconstructed - total_logit)
    if reconstruction_error > 1e-8:  # pragma: no cover - guards a real defect
        raise AssertionError(
            "Attribution does not reconstruct the prediction "
            f"(error {reconstruction_error:.3e}). The decomposition is wrong and "
            "must not be shown."
        )

    # (b) External: the reconstructed probability equals the number the caller
    #     will display. This is the check that catches explaining the wrong
    #     functional, which no amount of internal consistency can detect.
    served_error: float | None = None
    if served_probability is not None:
        served_error = abs(float(served_probability) - probability)
        if served_error > served_tolerance:
            raise AssertionError(
                "The driver decomposition does not reconstruct the quantity "
                f"being served: served {served_probability!r}, decomposition "
                f"reconstructs {probability!r} (error {served_error:.3e}, "
                f"tolerance {served_tolerance:.1e}). Either the served "
                "functional changed without the explainer following, or the "
                "explainer is reading a different feature vector. Showing "
                "drivers for a number nobody can see is worse than showing "
                "none."
            )

    # ---- domain roll-up ----------------------------------------------------
    domain_totals: dict[str, float] = {}
    domain_counts: dict[str, int] = {}
    for d in drivers:
        domain_totals[d.domain] = domain_totals.get(d.domain, 0.0) + d.contribution
        domain_counts[d.domain] = domain_counts.get(d.domain, 0) + 1
    by_domain = sorted(
        (
            {
                "domain": dom,
                "domain_label": DOMAIN_LABELS[FeatureDomain(dom)],
                "contribution": total,
                "n_drivers": domain_counts[dom],
                "direction": "increases_risk" if total > 0 else "decreases_risk",
            }
            for dom, total in domain_totals.items()
        ),
        key=lambda r: abs(r["contribution"]),
        reverse=True,
    )

    increasing = [d for d in drivers if d.contribution > 0][:top_k]
    decreasing = [d for d in drivers if d.contribution < 0][:top_k]

    return Explanation(
        baseline_logit=baseline_logit,
        total_logit=total_logit,
        probability=probability,
        baseline_probability=baseline_probability,
        drivers=drivers,
        increasing=increasing,
        decreasing=decreasing,
        by_domain=by_domain,
        other_factors_contribution=float(other),
        reconstruction_error=float(reconstruction_error),
        served_reconstruction_error=served_error,
        method=method,
    )
