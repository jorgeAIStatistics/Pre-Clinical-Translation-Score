"""Does the published interval cover anything?

The question, and why it is answerable here
-------------------------------------------
Every score carries an interval. :mod:`trs.modeling.contract` states what that
interval is: a central interval for *the model's estimated probability* for this
programme, propagating coefficient, imputation and recalibration-parameter
uncertainty — explicitly **not** a prediction interval for the binary outcome,
and explicitly without a coverage guarantee for the true probability.

That disclaimer is the honest default. It is also untested, and an untested
disclaimer is a guess about one's own software. On a synthetic cohort the true
per-programme probability is known, because the generator wrote it down, so the
guess can be replaced with a measurement: how often does a nominal 95% interval
contain the probability the simulation actually used?

Three outcomes are possible and all three are informative:

* **Coverage near 95%** — the interval behaves like a confidence interval for
  the true probability on this data-generating process. That would be more than
  the contract claims, and the contract would stay as it is: one generator is
  not a guarantee.
* **Coverage well below 95%** — the interval is too narrow, and the three
  uncertainty sources it propagates do not account for the dominant error. The
  most likely missing term is *model misspecification*: the fitted linear model
  is not the generator, and no amount of parameter uncertainty around the wrong
  model recovers the right answer.
* **Coverage well above 95%** — the interval is conservative. Useful to know,
  and not a licence to narrow it: the existing `interval_agreement` diagnostic
  already reports the analytic intervals as ~1.9× wider than a bootstrap refit,
  and nothing here should be tuned to make a documentation sentence come true.

What this does **not** establish
--------------------------------
Nothing about real programmes. The probability being covered is a quantity that
exists only inside the simulation. A well-covered interval on synthetic data
says the implementation propagates the uncertainty it claims to propagate under
a known generator — which is worth knowing, and is not evidence of
translational validity.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class CoverageResult:
    """Measured interval behaviour against a known truth."""

    n: int
    level: float
    #: Fraction of intervals containing the true probability.
    coverage: float
    #: Wilson interval on that fraction, so a small n cannot read as precision.
    coverage_ci_low: float
    coverage_ci_high: float
    #: Mean (predicted − true). Positive means the model over-predicts risk.
    mean_bias: float
    mean_absolute_error: float
    median_interval_width: float
    #: How often the truth sits below / above the interval. A lopsided miss
    #: pattern is a bias; a symmetric one is width.
    missed_low: int
    missed_high: int
    verdict: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "nominal_level": self.level,
            "coverage": self.coverage,
            "coverage_ci": [self.coverage_ci_low, self.coverage_ci_high],
            "mean_bias_predicted_minus_true": self.mean_bias,
            "mean_absolute_error": self.mean_absolute_error,
            "median_interval_width": self.median_interval_width,
            "missed_low": self.missed_low,
            "missed_high": self.missed_high,
            "verdict": self.verdict,
            "interpretation": self.interpretation,
            "scope": (
                "Coverage of the simulation's own per-programme probability. "
                "Establishes that the implementation propagates the uncertainty "
                "it claims to under a known generator. It is not evidence about "
                "real programmes."
            ),
        }


def _wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval — behaves at the ends, unlike the normal one."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


#: Bands for the verdict. Deliberately wide: with ~130 programmes the Wilson
#: interval on a coverage estimate is itself ±8 points, so a tighter band would
#: be reporting noise as a finding.
NOMINAL_TOLERANCE = 0.10

#: Median interval width at or above which coverage stops being evidence about
#: the estimate. A probability interval half the width of the entire [0, 1]
#: scale covers almost any truth; reporting that as "well calibrated
#: uncertainty" is the same error as reporting an AUROC without a reference
#: model. The first run of this diagnostic measured 0.66 and the verdict logic
#: called it reassuring, which is exactly the flattery this codebase exists to
#: prevent.
WIDTH_UNINFORMATIVE = 0.40

#: Absolute mean bias at or above which the point estimate is reported as
#: systematically off, whatever the coverage.
BIAS_MATERIAL = 0.05


def interval_coverage(
    predicted: Sequence[float],
    interval_low: Sequence[float],
    interval_high: Sequence[float],
    true_probability: Sequence[float],
    *,
    level: float = 0.95,
) -> CoverageResult:
    """Measure how often the published interval contains the true probability."""
    p = np.asarray(predicted, dtype=float)
    lo = np.asarray(interval_low, dtype=float)
    hi = np.asarray(interval_high, dtype=float)
    truth = np.asarray(true_probability, dtype=float)
    if not (p.shape == lo.shape == hi.shape == truth.shape):
        raise ValueError("predicted, interval bounds and truth must align")
    n = int(p.size)
    if n == 0:
        raise ValueError("no rows to measure coverage on")

    inside = (truth >= lo) & (truth <= hi)
    k = int(inside.sum())
    coverage = k / n
    ci_low, ci_high = _wilson(k, n)

    below = int((truth < lo).sum())
    above = int((truth > hi).sum())
    bias = float(np.mean(p - truth))

    width = float(np.median(hi - lo))
    # Coverage alone is trivially gameable: an interval of [0, 1] covers
    # everything. Two qualifiers decide whether near-nominal coverage is a
    # property of the interval or an artefact of its width and a one-sided
    # bias, and both are checked before the reassuring verdict is allowed.
    wide = width >= WIDTH_UNINFORMATIVE
    lopsided = (below + above) > 0 and min(below, above) == 0 and (below + above) >= 3
    biased = abs(bias) >= BIAS_MATERIAL

    if abs(coverage - level) <= NOMINAL_TOLERANCE and (wide or lopsided or biased):
        verdict = "nominal_but_uninformative"
        parts = [
            f"{coverage:.1%} of intervals contained the simulation's true "
            f"probability against a nominal {level:.0%} — but that number "
            "should not be read as reassurance."
        ]
        if wide:
            parts.append(
                f"The median interval is {width:.2f} wide on a scale of 0 to 1. "
                "An interval that broad covers almost any truth, so coverage "
                "here is substantially a property of the width rather than of "
                "the estimate."
            )
        if biased:
            parts.append(
                f"The mean predicted probability exceeds the true probability "
                f"by {bias:+.3f}; the point estimate is systematically "
                f"{'high' if bias > 0 else 'low'}."
            )
        if lopsided:
            parts.append(
                f"Every one of the {below + above} misses falls on the same "
                f"side ({below} below, {above} above), which is the signature "
                "of a shifted level rather than a width problem."
            )
        interpretation = " ".join(parts)
    elif abs(coverage - level) <= NOMINAL_TOLERANCE:
        verdict = "near_nominal"
        interpretation = (
            f"{coverage:.1%} of intervals contained the simulation's true "
            f"probability against a nominal {level:.0%}, with a median width of "
            f"{width:.2f}, a mean bias of {bias:+.3f} and misses on both sides. "
            "On this generator the interval behaves roughly as a confidence "
            "interval for the true probability — more than the published "
            "contract claims, which is left unchanged because one generator is "
            "not a guarantee."
        )
    elif coverage < level - NOMINAL_TOLERANCE:
        verdict = "under_covers"
        interpretation = (
            f"{coverage:.1%} coverage against a nominal {level:.0%}: the "
            "interval is too narrow for the true probability. It propagates "
            "coefficient, imputation and recalibration uncertainty, and none of "
            "those is model-misspecification error — the fitted linear model is "
            "not the generator, and parameter uncertainty around the wrong "
            f"model does not recover the right answer. Misses are {below} low "
            f"and {above} high."
        )
    else:
        verdict = "over_covers"
        interpretation = (
            f"{coverage:.1%} coverage against a nominal {level:.0%}: the "
            "interval is conservative for the true probability. That is a "
            "finding, not a defect to tune away — intervals are not narrowed "
            "here to make a target number appear."
        )

    return CoverageResult(
        n=n,
        level=level,
        coverage=coverage,
        coverage_ci_low=ci_low,
        coverage_ci_high=ci_high,
        mean_bias=bias,
        mean_absolute_error=float(np.mean(np.abs(p - truth))),
        median_interval_width=float(np.median(hi - lo)),
        missed_low=below,
        missed_high=above,
        verdict=verdict,
        interpretation=interpretation,
    )
