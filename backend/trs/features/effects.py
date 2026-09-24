"""Putting heterogeneous preclinical effect sizes on one scale.

The problem
-----------
A preclinical package reports 78% tumour-growth inhibition here, a hazard ratio
of 0.42 there, a 3.1-fold change in a pharmacodynamic marker somewhere else.
These are not comparable as written, and averaging them is meaningless.

The approach
------------
Each metric is mapped to an approximate standardised effect, oriented so that
**larger means a stronger effect in favour of the intervention** and **zero
means no effect**. The mappings are deliberately simple, documented, monotone
transformations — not attempts at exact statistical equivalence, which the
source data do not support.

What this is and is not
-----------------------
This is a *comparability device* so that "how large were the animal effects"
can be a feature at all. It is not a meta-analytic pooled estimate: the inputs
lack the variance information a real meta-analysis needs, and the studies are
not exchangeable. The resulting feature is therefore interpreted ordinally by
the model — the only interpretation the transformation supports. This is
restated as a limitation in MODEL_CARD.md.

Conversion families
-------------------
Two families cover everything usable:

1. **Ratio-of-remaining** — for metrics expressing how much of an undesirable
   quantity was eliminated (tumour growth inhibition, response fraction).
   ``d = -ln(1 - p) * SCALE`` where ``p`` is the eliminated fraction. This is
   zero at no effect, monotone increasing, and unbounded above, which matches
   the underlying quantity. ``SCALE`` is calibrated so 50% elimination maps to
   ≈0.90 and 90% to ≈2.99 — the same range the odds-ratio family produces for
   moderate and large effects.

2. **Log-ratio** — for hazard, risk and odds ratios and fold changes. Uses the
   Hasselblad–Hedges result ``d = ln(OR) * sqrt(3) / pi`` to place log-odds on
   a standardised-mean-difference scale. Applying it to hazard and risk ratios
   is an approximation, defensible for the moderate effect sizes seen here.

Orientation
-----------
Hazard and risk ratios below 1 favour treatment; odds ratios and fold changes
above 1 favour treatment. Those defaults are wrong for some endpoints, so
``EvidenceItem.effect_favours_treatment`` overrides them: when set, the
computed magnitude is kept and the sign is taken from the curator's assertion.
"""

from __future__ import annotations

import math

from trs.ontology.enums import EffectMetric

#: Calibrates the ratio-of-remaining family. 1.3 places 50% elimination at
#: ≈0.90 and 90% at ≈2.99 standardised units.
_REMAINING_SCALE = 1.3

#: Floor on the remaining fraction. 100% tumour-growth inhibition is reported
#: routinely and is not literally infinite evidence; 0.02 caps a single
#: complete-response study at ≈5.1 before the global cap applies.
_REMAINING_FLOOR = 0.02

#: Cap applied after conversion. Preclinical effect reporting has a long right
#: tail driven by near-complete responses in small groups; uncapped, one such
#: study would dominate any median or maximum.
_SMD_CAP = 8.0

_LOG_OR_TO_SMD = math.sqrt(3.0) / math.pi  # ≈ 0.5513


def _cap(value: float) -> float:
    return max(-_SMD_CAP, min(_SMD_CAP, value))


def _ratio_of_remaining(eliminated_fraction: float) -> float:
    """``-ln(remaining) * SCALE``; zero at no effect, negative if worse."""
    remaining = 1.0 - eliminated_fraction
    remaining = max(remaining, _REMAINING_FLOOR)
    return -math.log(remaining) * _REMAINING_SCALE


def standardise_effect(
    metric: str | None,
    value: float | None,
    *,
    favours_treatment: bool | None = None,
) -> float | None:
    """Convert a reported effect to an approximate standardised effect size.

    Returns ``None`` when the metric carries no efficacy interpretation or the
    inputs are unusable. ``None`` propagates as missing and is never silently
    coerced to zero.
    """
    if metric is None or value is None or not math.isfinite(value):
        return None

    m = metric if isinstance(metric, EffectMetric) else EffectMetric(metric)

    match m:
        case EffectMetric.PERCENT_TGI:
            # Percentage reduction in tumour growth relative to control.
            signed = _ratio_of_remaining(value / 100.0)

        case EffectMetric.RESPONSE_RATE:
            # An absolute response fraction in [0, 1]; the null is 0, not 0.5.
            if not 0.0 <= value <= 1.0:
                return None
            signed = _ratio_of_remaining(value)

        case EffectMetric.PERCENT_CHANGE:
            # Signed percentage change in an endpoint where reduction is good.
            signed = _ratio_of_remaining(value / 100.0)

        case EffectMetric.HAZARD_RATIO | EffectMetric.RISK_RATIO:
            if value <= 0:
                return None
            signed = -math.log(value) * _LOG_OR_TO_SMD  # <1 favours treatment

        case EffectMetric.ODDS_RATIO:
            if value <= 0:
                return None
            signed = math.log(value) * _LOG_OR_TO_SMD   # >1 favours treatment

        case EffectMetric.FOLD_CHANGE:
            if value <= 0:
                return None
            signed = math.log2(value) * 0.5             # >1 favours treatment

        case EffectMetric.STANDARDISED_MEAN_DIFFERENCE:
            signed = value

        case EffectMetric.ABSOLUTE_DIFFERENCE:
            # Units are endpoint-specific and not comparable across studies.
            return None

        case (
            EffectMetric.IC50_NM
            | EffectMetric.EC50_NM
            | EffectMetric.KI_NM
            | EffectMetric.KD_NM
        ):
            # Potency, not efficacy. Enters the model via the PK/PD bridge.
            return None

        case _:
            return None

    if favours_treatment is not None:
        # The curator asserted the direction; keep the magnitude, take the sign.
        signed = abs(signed) if favours_treatment else -abs(signed)

    return _cap(signed)


def effect_precision(
    ci_low: float | None,
    ci_high: float | None,
    se: float | None,
    n_per_arm: int | None,
) -> float | None:
    """A crude precision weight in (0, 1], used only for ordering evidence.

    Prefers a reported standard error, falls back to confidence-interval width,
    and finally to group size. Returns ``None`` when nothing is reported —
    precision is never assumed.
    """
    if se is not None and se > 0:
        return min(1.0, 1.0 / (1.0 + se))
    if ci_low is not None and ci_high is not None and ci_high > ci_low:
        return min(1.0, 1.0 / (1.0 + (ci_high - ci_low)))
    if n_per_arm is not None and n_per_arm > 0:
        return min(1.0, math.sqrt(n_per_arm) / 10.0)
    return None
