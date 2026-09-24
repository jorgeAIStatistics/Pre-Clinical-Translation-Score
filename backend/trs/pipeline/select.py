"""The model-selection gate.

A more flexible model is not adopted because it scored better. It is adopted
because it scored *meaningfully* better, by an amount that survives a paired
bootstrap at the component level, while remaining calibrated — and, when it
costs the product a capability, by a larger margin still.

The rules, and why each exists
-------------------------------

**Baseline privilege, but not baseline immunity.** ``l2_logistic`` is the
incumbent and wins ties; the burden of proof is on the challenger. Without that
rule, noise in a small test set decides the architecture. But the incumbent
must still pass the same sanity checks as everyone else: an early version of
this gate retained a baseline whose calibrator had degenerated and whose
held-out log loss was 1.89, purely because it was the incumbent. If the
baseline fails, the best *passing* candidate is promoted; if nothing passes,
training fails loudly rather than registering a broken model.

**Calibration gate.** A candidate whose calibration slope falls outside
[0.80, 1.25] is disqualified regardless of discrimination. The product's output
is a probability; a model that ranks well but is systematically over-confident
is worse than useless here, because users act on the number.

**Material margin.** A challenger must reduce test log loss by at least
:data:`MIN_RELATIVE_IMPROVEMENT`. Log loss is the criterion rather than AUROC
because it is a proper scoring rule and is sensitive to calibration; AUROC is
invariant to any monotone distortion of the probabilities, which is precisely
the failure mode that matters most here.

**Statistical evidence.** The 95% paired component-bootstrap interval on the
log-loss difference must exclude zero. Paired, because both models see the same
programs; component-level, because rows within a component are not independent.

**Capability surcharge.** A challenger that cannot produce an exact additive
attribution must clear :data:`ATTRIBUTION_LOSS_SURCHARGE` on top of the normal
margin. This is a deliberate product judgement, stated rather than hidden: a
translational risk estimate a scientist cannot interrogate is a weaker product
than a slightly less accurate one they can. The surcharge is a threshold, not a
prohibition — a genuinely large gain still wins.

**Posterior surcharge.** Likewise for a family that cannot produce a real
posterior, since the honesty of the published interval depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from trs.validation.metrics import paired_bootstrap_difference

BASELINE_FAMILY = "l2_logistic"

#: Held-out log loss above which a fit is treated as broken rather than weak.
#: A no-covariate model predicting the base rate scores at most ln(2) ≈ 0.693
#: for a balanced outcome, so anything near 1.2 signals confidently-wrong
#: predictions — a degenerate calibrator, not a model with little signal.
MAX_ACCEPTABLE_LOG_LOSS = 1.20

#: Sanity band on the calibration slope, wider than the adoption band. Inside
#: this but outside CALIBRATION_SLOPE_BAND means "poorly calibrated"; outside
#: this means "the fit is broken".
SANITY_SLOPE_BAND = (0.15, 6.0)

#: Minimum relative reduction in test log loss for a challenger to be adopted.
MIN_RELATIVE_IMPROVEMENT = 0.02

#: Extra relative improvement demanded of a family with no exact attribution.
ATTRIBUTION_LOSS_SURCHARGE = 0.05

#: Extra relative improvement demanded of a family with no analytic posterior.
POSTERIOR_LOSS_SURCHARGE = 0.03

#: Acceptable calibration-slope band. Outside this, a candidate is disqualified.
CALIBRATION_SLOPE_BAND = (0.80, 1.25)


def required_margin(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    """Relative log-loss improvement this candidate must clear, and why."""
    margin = MIN_RELATIVE_IMPROVEMENT
    reasons = [f"base margin {MIN_RELATIVE_IMPROVEMENT:.0%}"]
    if not candidate.get("supports_exact_attribution", False):
        margin += ATTRIBUTION_LOSS_SURCHARGE
        reasons.append(
            f"+{ATTRIBUTION_LOSS_SURCHARGE:.0%} because this family cannot produce "
            "an exact additive explanation of its output"
        )
    if not candidate.get("supports_analytic_posterior", False):
        margin += POSTERIOR_LOSS_SURCHARGE
        reasons.append(
            f"+{POSTERIOR_LOSS_SURCHARGE:.0%} because this family has no analytic "
            "posterior, so its intervals rest on a bootstrap approximation"
        )
    return margin, reasons


def choose_family(
    candidates: Sequence[dict[str, Any]],
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    components: np.ndarray,
    *,
    random_state: int = 0,
) -> dict[str, Any]:
    """Apply the gate and return a full, auditable selection record."""
    eligible = [c for c in candidates if c.get("eligible")]
    if not eligible:
        raise RuntimeError("No candidate model fitted successfully.")

    by_family = {c["family"]: c for c in eligible}

    def passes_sanity(cand: dict[str, Any]) -> tuple[bool, str]:
        """Minimum bar every family must clear, incumbent included."""
        slope = cand.get("calibration_slope")
        ll = cand.get("log_loss")
        if ll is None or not np.isfinite(ll):
            return False, "held-out log loss is unavailable or not finite"
        if ll > MAX_ACCEPTABLE_LOG_LOSS:
            return False, (
                f"held-out log loss {ll:.3f} exceeds the sanity ceiling "
                f"{MAX_ACCEPTABLE_LOG_LOSS}, which indicates a degenerate "
                "calibrator or a broken fit rather than a merely weak model"
            )
        if slope is None or not np.isfinite(slope):
            return False, "calibration slope is unavailable or not finite"
        if not (SANITY_SLOPE_BAND[0] <= slope <= SANITY_SLOPE_BAND[1]):
            return False, (
                f"calibration slope {slope:.3g} is outside the sanity band "
                f"{SANITY_SLOPE_BAND}"
            )
        return True, ""

    sanity: dict[str, tuple[bool, str]] = {c["family"]: passes_sanity(c) for c in eligible}

    baseline = by_family.get(BASELINE_FAMILY)
    baseline_demoted: str | None = None
    if baseline is None or not sanity[baseline["family"]][0]:
        failing = baseline["family"] if baseline else None
        survivors = [c for c in eligible if sanity[c["family"]][0]]
        if not survivors:
            raise RuntimeError(
                "No candidate model passed the sanity checks. Reasons: "
                + "; ".join(f"{f}: {why}" for f, (ok, why) in sanity.items() if not ok)
            )
        if failing:
            baseline_demoted = (
                f"The {failing} baseline failed its sanity check "
                f"({sanity[failing][1]}); the best passing candidate was "
                "promoted in its place."
            )
        baseline = min(survivors, key=lambda c: c.get("log_loss") or float("inf"))

    baseline_family = baseline["family"]
    baseline_ll = baseline.get("log_loss")

    comparisons: list[dict[str, Any]] = []
    winner = baseline_family
    winner_margin = 0.0

    for cand in eligible:
        family = cand["family"]
        entry: dict[str, Any] = {
            "family": family,
            "log_loss": cand.get("log_loss"),
            "auroc": cand.get("auroc"),
            "brier": cand.get("brier"),
            "scaled_brier": cand.get("scaled_brier"),
            "calibration_slope": cand.get("calibration_slope"),
        }
        if family == baseline_family:
            entry.update({
                "is_baseline": True, "adopted": True,
                "verdict": "incumbent baseline",
            })
            comparisons.append(entry)
            continue

        entry["is_baseline"] = False
        slope = cand.get("calibration_slope")
        if slope is None or not (
            CALIBRATION_SLOPE_BAND[0] <= slope <= CALIBRATION_SLOPE_BAND[1]
        ):
            entry.update({
                "adopted": False,
                "verdict": (
                    f"disqualified: calibration slope {slope} outside the "
                    f"acceptable band {CALIBRATION_SLOPE_BAND}"
                ),
            })
            comparisons.append(entry)
            continue

        ll = cand.get("log_loss")
        if ll is None or baseline_ll in (None, 0):
            entry.update({"adopted": False, "verdict": "log loss unavailable"})
            comparisons.append(entry)
            continue

        relative = (baseline_ll - ll) / abs(baseline_ll)
        margin, margin_reasons = required_margin(cand)
        entry["relative_log_loss_improvement"] = float(relative)
        entry["required_margin"] = float(margin)
        entry["margin_rationale"] = margin_reasons

        if family in predictions and baseline_family in predictions:
            paired = paired_bootstrap_difference(
                y_test, predictions[family], predictions[baseline_family],
                components, metric="log_loss", random_state=random_state,
            )
            entry["paired_bootstrap"] = paired
            statistically_supported = bool(
                paired.get("excludes_zero") and (paired.get("observed_difference") or 0) < 0
            )
        else:
            entry["paired_bootstrap"] = None
            statistically_supported = False

        if relative < margin:
            entry.update({
                "adopted": False,
                "verdict": (
                    f"improvement {relative:+.1%} does not clear the required "
                    f"{margin:.0%} margin"
                ),
            })
        elif not statistically_supported:
            entry.update({
                "adopted": False,
                "verdict": (
                    "improvement clears the margin but the paired "
                    "component-bootstrap interval on the difference includes zero"
                ),
            })
        else:
            entry.update({"adopted": True, "verdict": "clears margin and is statistically supported"})
            if relative > winner_margin:
                winner = family
                winner_margin = relative
        comparisons.append(entry)

    if winner == baseline_family:
        rationale = (
            f"Retained the {baseline_family} baseline. No candidate cleared both "
            "its required improvement margin and the paired component-bootstrap "
            "test. The baseline additionally provides exact additive attribution "
            "and an analytic posterior, which the product depends on."
        )
    else:
        w = next(c for c in comparisons if c["family"] == winner)
        rationale = (
            f"Adopted {winner}: test log loss improved "
            f"{w['relative_log_loss_improvement']:+.1%} against the "
            f"{baseline_family} baseline (required {w['required_margin']:.0%}), "
            "the paired component-bootstrap interval on the difference excludes "
            "zero, and calibration slope "
            f"{w['calibration_slope']:.2f} is within the acceptable band."
        )

    if baseline_demoted:
        rationale = baseline_demoted + " " + rationale

    return {
        "chosen_family": winner,
        "baseline_family": baseline_family,
        "baseline_demoted": baseline_demoted,
        "sanity_checks": {
            f: {"passed": ok, "reason": why or "passed"} for f, (ok, why) in sanity.items()
        },
        "rationale": rationale,
        "comparisons": comparisons,
        "gate": {
            "criterion": "test log loss (proper scoring rule)",
            "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "attribution_surcharge": ATTRIBUTION_LOSS_SURCHARGE,
            "posterior_surcharge": POSTERIOR_LOSS_SURCHARGE,
            "calibration_slope_band": list(CALIBRATION_SLOPE_BAND),
            "sanity_slope_band": list(SANITY_SLOPE_BAND),
            "max_acceptable_log_loss": MAX_ACCEPTABLE_LOG_LOSS,
            "statistical_test": "paired component-level bootstrap, 95%, must exclude zero",
            "note": (
                "AUROC is reported but is not the selection criterion: it is "
                "invariant to monotone distortions of probability, which is the "
                "failure this product most needs to avoid."
            ),
        },
    }
