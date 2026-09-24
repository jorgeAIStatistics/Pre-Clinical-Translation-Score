"""Endpoint definitions: turning observed development histories into labels.

The central scientific decision in this file
---------------------------------------------
"Did the program fail?" is not one question. A program can stop because the
biology did not work, because the molecule was toxic, because exposure could
not be achieved, because the sponsor ran out of money, or because a portfolio
committee preferred a different asset. Collapsing these into a single "failure"
label produces a model that predicts *corporate behaviour* while claiming to
predict *biology*, and it is the most common way this kind of analysis goes
wrong.

So each endpoint below states explicitly which observed outcomes count as
positive, which count as negative, and which are **excluded**. Exclusions are
counted and reported: a cohort that discards 40% of its candidates is telling
you something, and the Methods page shows that number.

Competing risks
---------------
The primary endpoint is *cause-specific*. A program discontinued for toxicity
never got the chance to demonstrate efficacy, so it is neither a positive nor a
negative example for the efficacy question; it is removed. The resulting
estimand is therefore:

    P(fails to show sufficient Phase II efficacy | the program is not removed
      from development for a non-efficacy reason first)

That conditional is stated verbatim on the Methods page and in the API
response, because a user who reads the number as unconditional will
under-estimate total attrition. The composite secondary endpoint
``ph2_any_scientific_failure`` exists precisely so the unconditional scientific
attrition question can also be asked, and ``trs.modeling.competing_risks``
provides the cumulative-incidence view when the data support it.

Follow-up and censoring
-----------------------
A program that entered the clinic recently has not yet had the opportunity to
fail. Labelling it "did not fail" would bias every rate downward. Each endpoint
declares ``min_followup_years``; a track with an unresolved outcome and less
follow-up than that is excluded as censored rather than counted as a success.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from trs.db.models.core import ProgramIndication
from trs.ontology.enums import DiscontinuationReason as R
from trs.ontology.enums import OutcomeLabel as O

#: Reasons that are *not* statements about the biology. A track discontinued
#: for one of these is excluded from every scientific-failure endpoint.
NON_SCIENTIFIC_REASONS: frozenset[str] = frozenset({
    R.COMMERCIAL_STRATEGIC,
    R.PORTFOLIO_REPRIORITISATION,
    R.SPONSOR_INSOLVENCY,
    R.COMPETITIVE_LANDSCAPE,
    R.ENROLMENT_FEASIBILITY,
    R.SUPERSEDED_BY_BACKUP,
    R.MANUFACTURING_CMC,
})


class LabelDecision(dict):
    """Result of applying an endpoint to one track: label plus its reasoning."""


@dataclasses.dataclass(frozen=True, slots=True)
class EndpointDefinition:
    """A fully declarative supervised endpoint."""

    name: str
    display_name: str
    #: The estimand, in words, shown verbatim in the UI and API.
    estimand: str
    positive_outcomes: frozenset[str]
    negative_outcomes: frozenset[str]
    #: Outcomes removed from the cohort entirely, with the reason recorded.
    excluded_outcomes: frozenset[str]
    #: When true, a track discontinued for a non-scientific reason is excluded
    #: regardless of which outcome label it carries.
    exclude_non_scientific_reasons: bool = True
    #: Minimum observation time after the as-of date before an unresolved track
    #: may be treated as a negative example.
    min_followup_years: float = 5.0
    #: When true, only tracks that actually entered human development qualify.
    requires_human_entry: bool = True
    is_primary: bool = False
    notes: str = ""

    # ---- application -------------------------------------------------------

    def label(
        self,
        track: ProgramIndication,
        *,
        as_of: dt.date,
        data_cutoff: dt.date,
    ) -> LabelDecision:
        """Classify one track. Never raises; always explains itself."""
        outcome = track.outcome_label
        reason = track.discontinuation_reason

        if self.requires_human_entry and track.ind_or_cta_on is None and track.first_in_human_on is None:
            return LabelDecision(
                label=None, excluded=True, exclusion_reason="never_entered_human_development",
            )

        if self.exclude_non_scientific_reasons and reason in NON_SCIENTIFIC_REASONS:
            return LabelDecision(
                label=None, excluded=True,
                exclusion_reason=f"non_scientific_discontinuation:{reason}",
            )

        if outcome in self.excluded_outcomes:
            return LabelDecision(
                label=None, excluded=True, exclusion_reason=f"outcome_excluded:{outcome}",
            )

        if outcome in self.positive_outcomes:
            return LabelDecision(
                label=1, excluded=False, exclusion_reason=None, source_outcome=outcome,
            )

        if outcome in self.negative_outcomes:
            # A negative requires enough observation time to be credible.
            horizon = track.status_verified_on or track.outcome_date or data_cutoff
            if track.outcome_date is None:
                elapsed = (horizon - as_of).days / 365.25
                if elapsed < self.min_followup_years:
                    return LabelDecision(
                        label=None, excluded=True,
                        exclusion_reason=(
                            f"insufficient_followup:{elapsed:.1f}y"
                            f"<{self.min_followup_years:.1f}y"
                        ),
                    )
            return LabelDecision(
                label=0, excluded=False, exclusion_reason=None, source_outcome=outcome,
            )

        return LabelDecision(
            label=None, excluded=True, exclusion_reason=f"unclassifiable_outcome:{outcome}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "estimand": self.estimand,
            "positive_outcomes": sorted(self.positive_outcomes),
            "negative_outcomes": sorted(self.negative_outcomes),
            "excluded_outcomes": sorted(self.excluded_outcomes),
            "exclude_non_scientific_reasons": self.exclude_non_scientific_reasons,
            "min_followup_years": self.min_followup_years,
            "requires_human_entry": self.requires_human_entry,
            "is_primary": self.is_primary,
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# The endpoint catalogue
# ─────────────────────────────────────────────────────────────────────────────

PH2_EFFICACY_FAILURE = EndpointDefinition(
    name="ph2_efficacy_failure",
    display_name="Phase II efficacy failure",
    estimand=(
        "Probability that a program entering human development fails to "
        "demonstrate sufficient efficacy to progress successfully beyond "
        "Phase II, conditional on not being withdrawn first for safety, "
        "pharmacokinetic, manufacturing or commercial reasons."
    ),
    positive_outcomes=frozenset({O.PHASE_2_EFFICACY_FAILURE}),
    negative_outcomes=frozenset({
        O.PROGRESSED_PAST_PHASE_2,
        O.PHASE_3_FAILURE,      # reached Phase III ⇒ cleared the Phase II bar
        O.REGULATORY_APPROVAL,
    }),
    excluded_outcomes=frozenset({
        O.FAILED_BEFORE_HUMAN_DOSING,
        O.PHASE_1_SAFETY_FAILURE,
        O.PHASE_1_PKPD_FAILURE,
        O.PHASE_2_SAFETY_FAILURE,
        O.ACTIVE_NO_OUTCOME_YET,
        O.CENSORED_INSUFFICIENT_FOLLOWUP,
        O.UNKNOWN,
    }),
    is_primary=True,
    notes=(
        "DISCONTINUED_THROUGH_PHASE_2 is resolved by discontinuation reason: "
        "efficacy_insufficient counts as positive, other scientific reasons are "
        "excluded as competing events, non-scientific reasons are excluded "
        "outright. See resolve_composite_discontinuation()."
    ),
)

PH2_ANY_SCIENTIFIC_FAILURE = EndpointDefinition(
    name="ph2_any_scientific_failure",
    display_name="Any scientific failure through Phase II",
    estimand=(
        "Probability that a program entering human development stops for a "
        "scientific reason — insufficient efficacy, unacceptable safety, or "
        "inadequate pharmacokinetics — at or before Phase II. Commercial and "
        "strategic discontinuations are excluded."
    ),
    positive_outcomes=frozenset({
        O.PHASE_2_EFFICACY_FAILURE,
        O.PHASE_2_SAFETY_FAILURE,
        O.PHASE_1_SAFETY_FAILURE,
        O.PHASE_1_PKPD_FAILURE,
    }),
    negative_outcomes=frozenset({
        O.PROGRESSED_PAST_PHASE_2, O.PHASE_3_FAILURE, O.REGULATORY_APPROVAL,
    }),
    excluded_outcomes=frozenset({
        O.FAILED_BEFORE_HUMAN_DOSING,
        O.ACTIVE_NO_OUTCOME_YET,
        O.CENSORED_INSUFFICIENT_FOLLOWUP,
        O.UNKNOWN,
    }),
    notes="The composite endpoint; fewer exclusions than the primary, so a larger cohort.",
)

PH1_SAFETY_FAILURE = EndpointDefinition(
    name="ph1_safety_failure",
    display_name="Phase I safety/tolerability failure",
    estimand=(
        "Probability that a program entering human development stops in "
        "Phase I for safety or tolerability reasons."
    ),
    positive_outcomes=frozenset({O.PHASE_1_SAFETY_FAILURE}),
    negative_outcomes=frozenset({
        O.PHASE_2_EFFICACY_FAILURE, O.PHASE_2_SAFETY_FAILURE,
        O.PROGRESSED_PAST_PHASE_2, O.PHASE_3_FAILURE, O.REGULATORY_APPROVAL,
    }),
    excluded_outcomes=frozenset({
        O.FAILED_BEFORE_HUMAN_DOSING, O.ACTIVE_NO_OUTCOME_YET,
        O.CENSORED_INSUFFICIENT_FOLLOWUP, O.UNKNOWN,
    }),
    min_followup_years=2.0,
)

PH1_PKPD_FAILURE = EndpointDefinition(
    name="ph1_pkpd_failure",
    display_name="Phase I PK/PD failure",
    estimand=(
        "Probability that a program entering human development stops in "
        "Phase I because the required exposure or target engagement could not "
        "be achieved."
    ),
    positive_outcomes=frozenset({O.PHASE_1_PKPD_FAILURE}),
    negative_outcomes=frozenset({
        O.PHASE_2_EFFICACY_FAILURE, O.PHASE_2_SAFETY_FAILURE,
        O.PROGRESSED_PAST_PHASE_2, O.PHASE_3_FAILURE, O.REGULATORY_APPROVAL,
    }),
    excluded_outcomes=frozenset({
        O.FAILED_BEFORE_HUMAN_DOSING, O.ACTIVE_NO_OUTCOME_YET,
        O.CENSORED_INSUFFICIENT_FOLLOWUP, O.UNKNOWN,
    }),
    min_followup_years=2.0,
)

PRECLINICAL_ATTRITION = EndpointDefinition(
    name="preclinical_attrition",
    display_name="Failure before human dosing",
    estimand=(
        "Probability that a preclinical program is abandoned before any human "
        "is dosed, for a scientific reason."
    ),
    positive_outcomes=frozenset({O.FAILED_BEFORE_HUMAN_DOSING}),
    negative_outcomes=frozenset({
        O.PHASE_1_SAFETY_FAILURE, O.PHASE_1_PKPD_FAILURE,
        O.PHASE_2_EFFICACY_FAILURE, O.PHASE_2_SAFETY_FAILURE,
        O.PROGRESSED_PAST_PHASE_2, O.PHASE_3_FAILURE, O.REGULATORY_APPROVAL,
    }),
    excluded_outcomes=frozenset({
        O.ACTIVE_NO_OUTCOME_YET, O.CENSORED_INSUFFICIENT_FOLLOWUP, O.UNKNOWN,
    }),
    requires_human_entry=False,
    min_followup_years=3.0,
    notes=(
        "Severely affected by survivorship bias in public data: programs that "
        "die before an IND are frequently never disclosed. Interpret with the "
        "ascertainment caveat in MODEL_CARD.md."
    ),
)

EVENTUAL_APPROVAL = EndpointDefinition(
    name="eventual_approval",
    display_name="Eventual regulatory approval",
    estimand=(
        "Probability that a program entering human development is eventually "
        "approved by a major regulator in this indication."
    ),
    positive_outcomes=frozenset({O.REGULATORY_APPROVAL}),
    negative_outcomes=frozenset({
        O.PHASE_1_SAFETY_FAILURE, O.PHASE_1_PKPD_FAILURE,
        O.PHASE_2_EFFICACY_FAILURE, O.PHASE_2_SAFETY_FAILURE, O.PHASE_3_FAILURE,
    }),
    excluded_outcomes=frozenset({
        O.FAILED_BEFORE_HUMAN_DOSING, O.ACTIVE_NO_OUTCOME_YET,
        O.PROGRESSED_PAST_PHASE_2,  # progressed but not yet resolved either way
        O.CENSORED_INSUFFICIENT_FOLLOWUP, O.UNKNOWN,
    }),
    min_followup_years=8.0,
    notes="Long horizon; in practice only older cohorts contribute.",
)

ENDPOINTS: dict[str, EndpointDefinition] = {
    e.name: e
    for e in (
        PH2_EFFICACY_FAILURE,
        PH2_ANY_SCIENTIFIC_FAILURE,
        PH1_SAFETY_FAILURE,
        PH1_PKPD_FAILURE,
        PRECLINICAL_ATTRITION,
        EVENTUAL_APPROVAL,
    )
}

PRIMARY_ENDPOINT = PH2_EFFICACY_FAILURE


def resolve_composite_discontinuation(
    track: ProgramIndication, endpoint: EndpointDefinition
) -> LabelDecision | None:
    """Resolve ``DISCONTINUED_THROUGH_PHASE_2`` using the discontinuation reason.

    This outcome label is deliberately ambiguous in the ontology — it records
    "stopped somewhere at or before Phase II" without asserting why. The reason
    field disambiguates it, and doing so here keeps the mapping in one place
    rather than spread across endpoint definitions.

    Returns ``None`` when the track does not carry this outcome, so the caller
    falls through to the normal endpoint logic.
    """
    if track.outcome_label != O.DISCONTINUED_THROUGH_PHASE_2:
        return None

    reason = track.discontinuation_reason
    if reason in NON_SCIENTIFIC_REASONS:
        return LabelDecision(
            label=None, excluded=True,
            exclusion_reason=f"non_scientific_discontinuation:{reason}",
        )

    efficacy_positive = endpoint.name in {
        "ph2_efficacy_failure", "ph2_any_scientific_failure",
    }
    match reason:
        case R.EFFICACY_INSUFFICIENT if efficacy_positive:
            return LabelDecision(
                label=1, excluded=False, exclusion_reason=None,
                source_outcome=f"{O.DISCONTINUED_THROUGH_PHASE_2}:{reason}",
            )
        case R.SAFETY_TOXICITY | R.PK_PD_INADEQUATE if endpoint.name == "ph2_any_scientific_failure":
            return LabelDecision(
                label=1, excluded=False, exclusion_reason=None,
                source_outcome=f"{O.DISCONTINUED_THROUGH_PHASE_2}:{reason}",
            )
        case R.SAFETY_TOXICITY | R.PK_PD_INADEQUATE:
            return LabelDecision(
                label=None, excluded=True,
                exclusion_reason=f"competing_event:{reason}",
            )
        case R.EFFICACY_INSUFFICIENT:
            # Efficacy failure, but this endpoint does not treat it as positive
            # (e.g. the Phase I endpoints). It is a negative for those.
            return LabelDecision(
                label=0, excluded=False, exclusion_reason=None,
                source_outcome=f"{O.DISCONTINUED_THROUGH_PHASE_2}:{reason}",
            )
        case _:
            return LabelDecision(
                label=None, excluded=True,
                exclusion_reason=f"undisclosed_discontinuation_reason:{reason}",
            )


def label_track(
    track: ProgramIndication,
    endpoint: EndpointDefinition,
    *,
    as_of: dt.date,
    data_cutoff: dt.date,
) -> LabelDecision:
    """Apply an endpoint to a track, handling the composite outcome first."""
    composite = resolve_composite_discontinuation(track, endpoint)
    if composite is not None:
        return composite
    return endpoint.label(track, as_of=as_of, data_cutoff=data_cutoff)
