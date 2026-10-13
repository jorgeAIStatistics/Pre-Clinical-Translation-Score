"""Model status, promotion criteria and the eligibility policy that enforces them.

The gap this closes
-------------------
The predecessor computed an honest verdict about itself —
``not_distinguishable_from_base_rate`` — recorded it in the artifact, printed it
on the Methods page, and then served a probability for every program anyway.
``trained_on_synthetic`` was likewise recorded, displayed, and allowed to gate
nothing. The system told the truth and behaved as though it had not.

So three things are separated here that were previously one bit
(``is_active``):

**Numerical validity** — did the fit converge, is the calibrator non-degenerate,
is the design non-singular. A failure here means the artifact is broken.

**Empirical performance** — does the model beat a reference model anyone could
have shipped, by a margin whose paired interval excludes zero. A failure here
means the artifact is sound and uninformative, which is a legitimate state and
the one this demonstration is actually in.

**Authorisation for a use** — whether *this* model may produce *this* number for
*this* evidence. A synthetic-trained model scoring real evidence is refused
regardless of how good its simulated metrics are, because the metrics describe
a simulation.

Default deny
------------
:data:`SERVING_PERMITTED` lists the statuses that may produce a score.
Everything else, including :attr:`ModelStatus.UNKNOWN`, may not. An artifact
whose status cannot be determined is not treated as approved — which is the
opposite of the usual default and the entire point.

Promotion criteria are versioned
--------------------------------
:data:`PROMOTION_CRITERIA_VERSION` and :data:`PROMOTION_CRITERIA` are fixed
before the evaluation partition is read, recorded in the artifact, and changed
by editing this file and bumping the version — not by passing a kinder
threshold at the call site. The thresholds are product judgements stated as
such, not universal scientific constants, and each says why it is where it is.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any


class ModelStatus(StrEnum):
    """What an artifact is allowed to be used for."""

    #: Trained on synthetic data. May be explored against synthetic evidence in
    #: an explicitly selected demonstration context and nowhere else. No
    #: quantity it produces carries scientific meaning.
    DEMO_ONLY = "demo_only"

    #: The cohort, the development partition or the evaluation partition was
    #: too small for the declared protocol. Registered so the attempt is on the
    #: record; never served.
    INSUFFICIENT_DATA = "insufficient_data"

    #: The artifact is numerically broken: non-convergence, a degenerate
    #: calibrator, a calibration slope far outside any defensible band. Never
    #: served, and the reason is recorded.
    EVALUATION_FAILED = "evaluation_failed"

    #: Numerically sound, trained on real evidence, but not shown to beat a
    #: deployable reference model. May be used for evidence review and for
    #: clearly-labelled research exploration. Individual estimates must carry
    #: the verdict beside them.
    RESEARCH_ONLY = "research_only"

    #: Numerically sound, beats a deployable reference with a paired interval
    #: excluding zero, calibrated within band, on a protocol permitted to make
    #: a temporal claim. This is a deployment-policy decision about a defined
    #: research use. **It is not clinical validation and not regulatory
    #: qualification**, and nothing in this codebase can confer either.
    APPROVED_FOR_DEFINED_RESEARCH_USE = "approved_for_defined_research_use"

    #: Status could not be determined — an artifact from before this module, or
    #: one whose metrics block is unreadable. Treated as not permitted.
    UNKNOWN = "unknown"


#: Statuses that may produce a numeric score at all. Default deny: anything
#: absent from this set is refused, including UNKNOWN.
SERVING_PERMITTED: frozenset[str] = frozenset({
    ModelStatus.DEMO_ONLY,                      # synthetic evidence only
    ModelStatus.RESEARCH_ONLY,
    ModelStatus.APPROVED_FOR_DEFINED_RESEARCH_USE,
})

#: Statuses whose numbers must always be presented as demonstration output.
DEMONSTRATION_ONLY: frozenset[str] = frozenset({ModelStatus.DEMO_ONLY})

PROMOTION_CRITERIA_VERSION = "pc-1.0.0"

#: The criteria, fixed before the evaluation partition is read. Each entry is a
#: statement of what is required and why that is the requirement.
PROMOTION_CRITERIA: dict[str, dict[str, Any]] = {
    "numerically_sound": {
        "requirement": (
            "The fit converged, the calibrator is non-degenerate, and the "
            "evaluation log loss is below 1.20."
        ),
        "why": (
            "A no-covariate model on a balanced outcome scores at most "
            "ln(2) ≈ 0.693. A held-out log loss near 1.20 is not a weak model; "
            "it is confidently wrong, which in practice means a degenerate "
            "calibrator."
        ),
        "threshold": {"max_log_loss": 1.20},
    },
    "calibrated_in_band": {
        "requirement": "Evaluation calibration slope within [0.80, 1.25].",
        "why": (
            "The product's output is a probability that users act on. A model "
            "that ranks well while being systematically over-confident is worse "
            "than useless here. The band is a product judgement, not a law."
        ),
        "threshold": {"slope_band": [0.80, 1.25]},
    },
    "beats_deployable_reference": {
        "requirement": (
            "Lower Brier score than an intercept-only model fitted on "
            "development labels, with a paired component-bootstrap interval on "
            "the difference that excludes zero."
        ),
        "why": (
            "Discrimination metrics are invariant to monotone distortions of "
            "probability, so AUROC cannot detect a model whose probabilities "
            "are worse than a constant. The reference is fitted on development "
            "data because a hindsight-optimal constant is not a competitor "
            "anyone could have shipped."
        ),
        "threshold": {"paired_interval_excludes_zero": True},
    },
    "adequate_sample": {
        "requirement": (
            "At least 120 development rows and 60 evaluation rows, counted as "
            "rows, with the independent-unit count reported separately."
        ),
        "why": (
            "Rows within a connected component are not independent. 465 "
            "components for 552 rows on the demonstration cohort is the number "
            "that governs interval width, and it is reported beside every "
            "metric for that reason."
        ),
        "threshold": {"min_development_rows": 120, "min_evaluation_rows": 60},
    },
    "protocol_permits_claim": {
        "requirement": (
            "The split protocol is one permitted to describe its numbers as "
            "temporal deployment performance."
        ),
        "why": (
            "A retrospective split that ignores label maturity measures "
            "something real, but not deployment. Promotion to a defined "
            "research use should not rest on a number that answers a different "
            "question."
        ),
        "threshold": {"protocols": ["strict_temporal"]},
    },
    "real_evidence": {
        "requirement": "The training cohort is not synthetic.",
        "why": (
            "Excellent metrics on simulated data establish that the "
            "implementation recovers a known data-generating process. They say "
            "nothing about translational biology. No synthetic-trained model is "
            "promotable, whatever its numbers."
        ),
        "threshold": {"trained_on_synthetic": False},
    },
}


# ---------------------------------------------------------------------------
# Deriving a status
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StatusDecision:
    """The assigned status, and the evaluation of every criterion behind it."""

    status: str
    criteria_version: str
    #: criterion name → (passed, observed value or reason)
    criteria: dict[str, dict[str, Any]]
    rationale: str

    @property
    def may_serve(self) -> bool:
        return self.status in SERVING_PERMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "may_serve": self.may_serve,
            "criteria_version": self.criteria_version,
            "criteria": self.criteria,
            "rationale": self.rationale,
            "note": (
                "A status of approved_for_defined_research_use is a deployment "
                "policy decision about a defined research use. It is not "
                "clinical validation and not regulatory qualification."
            ),
        }


def derive_status(
    *,
    trained_on_synthetic: bool,
    protocol: str,
    permits_deployment_claim: bool,
    n_development: int,
    n_evaluation: int,
    log_loss: float | None,
    calibration_slope: float | None,
    reference_comparison: dict[str, Any] | None,
    numerical_failure: str | None = None,
) -> StatusDecision:
    """Assign a status from the declared criteria. Deny by default.

    Order matters: a numerical failure outranks everything, inadequate sample
    outranks performance (a model cannot be shown to work on data too small to
    show anything), and synthetic training caps the result at ``demo_only``
    however good the rest looks.
    """
    criteria: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, observed: Any) -> bool:
        criteria[name] = {
            "passed": bool(passed),
            "observed": observed,
            "requirement": PROMOTION_CRITERIA[name]["requirement"],
        }
        return bool(passed)

    sound = record(
        "numerically_sound",
        numerical_failure is None
        and log_loss is not None
        and log_loss < PROMOTION_CRITERIA["numerically_sound"]["threshold"]["max_log_loss"],
        numerical_failure or {"log_loss": log_loss},
    )
    lo, hi = PROMOTION_CRITERIA["calibrated_in_band"]["threshold"]["slope_band"]
    in_band = record(
        "calibrated_in_band",
        calibration_slope is not None and lo <= calibration_slope <= hi,
        {"calibration_slope": calibration_slope, "band": [lo, hi]},
    )
    thr = PROMOTION_CRITERIA["adequate_sample"]["threshold"]
    adequate = record(
        "adequate_sample",
        n_development >= thr["min_development_rows"]
        and n_evaluation >= thr["min_evaluation_rows"],
        {"n_development": n_development, "n_evaluation": n_evaluation},
    )
    beats = record(
        "beats_deployable_reference",
        bool((reference_comparison or {}).get("model_better_than_reference")),
        {
            "brier_skill": (reference_comparison or {}).get("brier_skill"),
            "paired_difference": (reference_comparison or {}).get(
                "brier_difference_model_minus_reference"
            ),
        },
    )
    claimable = record(
        "protocol_permits_claim", permits_deployment_claim, {"protocol": protocol}
    )
    real = record("real_evidence", not trained_on_synthetic,
                  {"trained_on_synthetic": trained_on_synthetic})

    if not sound:
        return StatusDecision(
            status=str(ModelStatus.EVALUATION_FAILED),
            criteria_version=PROMOTION_CRITERIA_VERSION,
            criteria=criteria,
            rationale=(
                numerical_failure
                or f"Evaluation log loss {log_loss} is at or above the 1.20 "
                   "threshold that separates a weak model from a confidently "
                   "wrong one."
            ),
        )
    if not adequate:
        return StatusDecision(
            status=str(ModelStatus.INSUFFICIENT_DATA),
            criteria_version=PROMOTION_CRITERIA_VERSION,
            criteria=criteria,
            rationale=(
                f"{n_development} development and {n_evaluation} evaluation rows "
                "are below the declared minimums; no performance claim can be "
                "resolved on a sample this size."
            ),
        )
    if not real:
        return StatusDecision(
            status=str(ModelStatus.DEMO_ONLY),
            criteria_version=PROMOTION_CRITERIA_VERSION,
            criteria=criteria,
            rationale=(
                "Trained on synthetic demonstration data. Its coefficients and "
                "metrics describe a simulated data-generating process, so no "
                "quantity it produces carries scientific meaning about real "
                "programmes — however favourable those metrics are. Capped at "
                "demo_only by criterion real_evidence."
            ),
        )
    if beats and in_band and claimable:
        return StatusDecision(
            status=str(ModelStatus.APPROVED_FOR_DEFINED_RESEARCH_USE),
            criteria_version=PROMOTION_CRITERIA_VERSION,
            criteria=criteria,
            rationale=(
                "Every declared criterion passed, including a paired "
                "component-bootstrap interval on the Brier difference against a "
                "development-fitted reference that excludes zero. This "
                "authorises a defined research use and is not clinical "
                "validation."
            ),
        )
    unmet = [k for k, v in criteria.items() if not v["passed"]]
    return StatusDecision(
        status=str(ModelStatus.RESEARCH_ONLY),
        criteria_version=PROMOTION_CRITERIA_VERSION,
        criteria=criteria,
        rationale=(
            "Numerically sound on an adequate sample, but these criteria were "
            f"not met: {', '.join(unmet)}. Usable for evidence review and "
            "clearly-labelled exploration; individual estimates must be "
            "presented with the unmet criteria beside them."
        ),
    )


# ---------------------------------------------------------------------------
# Enforcing eligibility at the point of use
# ---------------------------------------------------------------------------


class RefusalCode(StrEnum):
    """Machine-readable reasons a score was not produced."""

    SYNTHETIC_MODEL_REAL_EVIDENCE = "synthetic_model_real_evidence"
    DEMO_CONTEXT_NOT_SELECTED = "demo_context_not_selected"
    STATUS_NOT_PERMITTED = "status_not_permitted"
    STATUS_UNKNOWN = "status_unknown"


@dataclasses.dataclass(frozen=True)
class EligibilityDecision:
    """Whether a numeric score may be produced here, and how it must be labelled."""

    permitted: bool
    status: str
    refusal_code: str | None
    reason: str
    #: True when the number, if produced, is demonstration output.
    is_demonstration: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "permitted": self.permitted,
            "model_status": self.status,
            "refusal_code": self.refusal_code,
            "reason": self.reason,
            "is_demonstration": self.is_demonstration,
        }


def check_eligibility(
    *,
    status: str | None,
    trained_on_synthetic: bool,
    evidence_is_synthetic: bool,
    demo_context_selected: bool = False,
) -> EligibilityDecision:
    """Decide whether this model may score this evidence.

    The rule that does the real work: a synthetic-trained model may not score
    real evidence. Not with a warning, not with a disclaimer — the number is not
    produced, and the caller receives a refusal code it can render. The reverse
    direction (a real-trained model scoring synthetic evidence) is permitted but
    marked as demonstration, because the *evidence* is then the fiction.
    """
    resolved = status or str(ModelStatus.UNKNOWN)

    if resolved == str(ModelStatus.UNKNOWN):
        return EligibilityDecision(
            permitted=False,
            status=resolved,
            refusal_code=str(RefusalCode.STATUS_UNKNOWN),
            reason=(
                "This model artifact carries no determinable status — it "
                "predates the status policy or its metrics block is unreadable. "
                "Unknown is not treated as approved."
            ),
            is_demonstration=False,
        )

    if resolved not in SERVING_PERMITTED:
        return EligibilityDecision(
            permitted=False,
            status=resolved,
            refusal_code=str(RefusalCode.STATUS_NOT_PERMITTED),
            reason=(
                f"Model status {resolved!r} does not permit producing a score. "
                "Evidence review, completeness assessment and provenance remain "
                "available."
            ),
            is_demonstration=False,
        )

    if trained_on_synthetic and not evidence_is_synthetic:
        return EligibilityDecision(
            permitted=False,
            status=resolved,
            refusal_code=str(RefusalCode.SYNTHETIC_MODEL_REAL_EVIDENCE),
            reason=(
                "This model was trained entirely on synthetic demonstration "
                "data, and the evidence for this programme is not synthetic. A "
                "score would be a simulation's opinion about a real programme, "
                "which is not a meaningful quantity at any confidence. Evidence "
                "review, gap and contradiction analysis, completeness and "
                "provenance are unaffected."
            ),
            is_demonstration=False,
        )

    if trained_on_synthetic and not demo_context_selected:
        return EligibilityDecision(
            permitted=False,
            status=resolved,
            refusal_code=str(RefusalCode.DEMO_CONTEXT_NOT_SELECTED),
            reason=(
                "This is a demonstration model on demonstration evidence. Its "
                "scores are available only in an explicitly selected "
                "demonstration context, so that a number from a simulation "
                "cannot be mistaken for a finding."
            ),
            is_demonstration=True,
        )

    return EligibilityDecision(
        permitted=True,
        status=resolved,
        refusal_code=None,
        reason=(
            "Demonstration score on demonstration evidence, in an explicitly "
            "selected demonstration context."
            if trained_on_synthetic
            else f"Model status {resolved!r} permits this use."
        ),
        is_demonstration=bool(trained_on_synthetic or evidence_is_synthetic),
    )
