"""The prediction contract: which number is *the* number.

The defect this fixes
---------------------
Three parts of the predecessor computed three different quantities and called
them all "the probability":

* ``trs.modeling.uncertainty.predict_with_uncertainty`` returned
  ``float(np.mean(probs))`` — a Monte-Carlo mean over the product of imputation
  draws, coefficient draws and calibration-parameter draws.
* ``trs.scoring.service`` served that value as ``probability``.
* Training, validation and every metric in the model artifact used
  ``calibrator.transform(model.predict_logit(X_point))`` — a plug-in point
  prediction on median-imputed features.

These differ systematically, not by Monte-Carlo noise. :math:`E[\\sigma(Z)]
\\neq \\sigma(E[Z])` whenever :math:`Z` has spread, and the gap widens with
imputation variance — that is, it is largest for exactly the programs with the
least evidence, where the published number matters most. The explanation panel
reconstructed the third quantity while the interface displayed the first.

The resolution
--------------
One functional is declared primary. It is the one the metrics are computed on,
the one the API returns as ``probability``, and the one the attribution
reconstructs. The other remains available under its own name and is never
substituted for it.

Why the plug-in probability is primary
--------------------------------------
Three reasons, in order of weight:

1. **It is the quantity that can be explained exactly.** For a linear model
   under affine recalibration, ``logit(p) = a + b·(β₀ + Σ βⱼzⱼ)`` decomposes
   additively with zero residual. The draw-mean has no such decomposition: any
   additive attribution for it is an approximation, and presenting an
   approximation as an exact driver list is the kind of quiet dishonesty this
   codebase exists to avoid.
2. **It is reproducible without a random seed.** A historical prediction must be
   recreatable; a Monte-Carlo mean is recreatable only given the seed, the draw
   count and the identical RNG implementation.
3. **It is what the metrics measure.** Evaluating one functional and serving
   another makes the calibration curve a statement about a number nobody sees.

The cost, stated: the plug-in probability ignores imputation uncertainty in its
*location*, not only in its spread. For a program with substantial missing
evidence the predictive mean is pulled toward the base rate relative to the
plug-in value. That difference is reported as
:attr:`PredictionContract.functional_gap` rather than hidden, and when it is
material the interface says so instead of inventing a biological driver to
explain it.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any

#: Bumped when the meaning of a served number changes. Carried in every
#: artifact and every score payload so two numbers can be compared only when
#: they are the same quantity.
PREDICTION_CONTRACT_VERSION = "pcx-1.0.0"


class Functional(StrEnum):
    """The quantities this system can compute, named so they cannot be confused."""

    #: σ(a + b·η(x̃)) where x̃ is the median-imputed feature vector, η is the
    #: model's point logit at posterior-mean parameters, and (a, b) are the
    #: fitted recalibration parameters. **Primary.** Exactly decomposable.
    CALIBRATED_PLUG_IN = "calibrated_plug_in"

    #: E[σ(a + b·η(X))] under the joint draw distribution of imputations,
    #: coefficients and calibration parameters. Secondary, reported, never
    #: substituted. Not exactly decomposable.
    PREDICTIVE_MEAN = "predictive_mean"


PRIMARY_FUNCTIONAL = Functional.CALIBRATED_PLUG_IN

#: What the published interval actually estimates. Named explicitly because the
#: previous wording ("uncertainty interval") invited reading it as a prediction
#: interval for an individual binary outcome, which it is not and cannot be: a
#: Bernoulli outcome's own 95% interval is {0, 1} for any probability in
#: (0.025, 0.975).
INTERVAL_ESTIMAND = (
    "A central interval for the model's estimated probability for this "
    "programme, propagating coefficient uncertainty, imputation uncertainty "
    "for unreported evidence, and recalibration-parameter uncertainty. It is "
    "NOT a prediction interval for the individual binary outcome, and it does "
    "not carry a coverage guarantee for the true probability."
)

#: Above this absolute difference on the probability scale, the gap between the
#: primary and secondary functionals is surfaced to the reader rather than only
#: recorded. Chosen to be just above the noise floor of the default draw count.
MATERIAL_FUNCTIONAL_GAP = 0.02


@dataclasses.dataclass(frozen=True)
class PredictionContract:
    """What a served number is, attached to the number.

    Every score payload carries one of these. A consumer that does not
    understand a field is better off than one handed a bare float.
    """

    #: The functional actually returned as ``probability``.
    functional: str
    primary_probability: float
    #: The secondary quantity, for comparison. ``None`` when uncertainty
    #: propagation was unavailable for this family.
    predictive_mean_probability: float | None
    #: ``predictive_mean − primary``. Positive means the draw-mean sits above
    #: the plug-in value.
    functional_gap: float | None
    calibration_scale: str
    calibration_method: str
    interval_estimand: str
    contract_version: str = PREDICTION_CONTRACT_VERSION

    @property
    def gap_is_material(self) -> bool:
        return (
            self.functional_gap is not None
            and abs(self.functional_gap) >= MATERIAL_FUNCTIONAL_GAP
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contract_version": self.contract_version,
            "functional": self.functional,
            "functional_description": FUNCTIONAL_DESCRIPTIONS[self.functional],
            "probability": self.primary_probability,
            "predictive_mean_probability": self.predictive_mean_probability,
            "functional_gap": self.functional_gap,
            "gap_is_material": self.gap_is_material,
            "calibration_scale": self.calibration_scale,
            "calibration_method": self.calibration_method,
            "interval_estimand": self.interval_estimand,
            "evaluated_functional": str(PRIMARY_FUNCTIONAL),
        }
        if self.gap_is_material:
            out["gap_note"] = (
                f"The predictive mean ({self.predictive_mean_probability:.3f}) "
                f"differs from the served plug-in probability "
                f"({self.primary_probability:.3f}) by "
                f"{self.functional_gap:+.3f}. This is an arithmetic consequence "
                "of averaging a non-linear function over imputation draws for "
                "unreported evidence — it is not a biological signal and no "
                "driver in the attribution corresponds to it."
            )
        return out


FUNCTIONAL_DESCRIPTIONS: dict[str, str] = {
    str(Functional.CALIBRATED_PLUG_IN): (
        "Calibrated plug-in probability: the recalibration map applied to the "
        "model's point log-odds at median-imputed feature values. This is the "
        "quantity the model's reported metrics were computed on and the one the "
        "driver attribution reconstructs exactly."
    ),
    str(Functional.PREDICTIVE_MEAN): (
        "Predictive mean probability: the average of the calibrated probability "
        "over draws of the coefficients, the imputations for unreported "
        "evidence, and the recalibration parameters. Reported for comparison. "
        "It has no exact additive decomposition, and the model's metrics were "
        "not computed on it."
    ),
}


def build_contract(
    *,
    primary_probability: float,
    predictive_mean_probability: float | None,
    calibration_method: str,
    calibration_scale: str = "log_odds_affine",
) -> PredictionContract:
    """Assemble the contract for one served prediction."""
    gap = (
        None
        if predictive_mean_probability is None
        else float(predictive_mean_probability - primary_probability)
    )
    return PredictionContract(
        functional=str(PRIMARY_FUNCTIONAL),
        primary_probability=float(primary_probability),
        predictive_mean_probability=(
            None if predictive_mean_probability is None
            else float(predictive_mean_probability)
        ),
        functional_gap=gap,
        calibration_scale=calibration_scale,
        calibration_method=calibration_method,
        interval_estimand=INTERVAL_ESTIMAND,
    )
