"""Historical comparators: programs that made a similar bet, and how they fared.

What this is
------------
A retrieval over structured facets — target, pathway, mechanism class, modality,
indication, disease stage, biomarker strategy, preclinical model construction —
returning earlier programs that resemble the focal one, with their observed
outcomes and the reason each is considered comparable.

What this is not
----------------
It is **not** evidence that the focal program will share their fate. Similarity
on a handful of recorded facets is not exchangeability: two KRAS inhibitors can
differ in the ways that decide the outcome and agree on everything this system
records. The API and the interface therefore present comparators as *context
for judgement*, never as a prediction, and the response carries that statement
rather than leaving it to be inferred.

Temporal rule
-------------
Only programs whose outcome was already known at the focal program's cutoff are
returned. A 2020 failure is not a comparator for a 2014 decision; showing it
would let a user read the score as if it had been informed by information that
did not exist. This mirrors the constraint on the model's own features.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.db.models.core import Indication, Program, ProgramIndication, Sponsor, Target
from trs.features.spec import MODALITY_GROUPS, TARGET_CLASS_GROUPS
from trs.ontology import enums as E

MATCHER_VERSION = "cmp-1.0.0"

#: Facet weights. Sum to 1. Target identity dominates because sharing a target
#: means sharing the target-validation evidence outright; modality and
#: indication follow because they determine how that biology is engaged and in
#: whom it is tested.
FACET_WEIGHTS: dict[str, float] = {
    "target": 0.32,
    "mechanism_class": 0.16,
    "modality": 0.16,
    "indication": 0.18,
    "disease_stage": 0.06,
    "biomarker_strategy": 0.07,
    "model_construction": 0.05,
}

FACET_LABELS: dict[str, str] = {
    "target": "Same molecular target",
    "mechanism_class": "Same target class and modality group",
    "modality": "Same therapeutic modality",
    "indication": "Same indication",
    "disease_stage": "Same disease stage",
    "biomarker_strategy": "Same patient-selection strategy",
    "model_construction": "Comparable preclinical model construction",
}


@dataclasses.dataclass(slots=True)
class Comparator:
    track_id: str
    program_id: str
    program_name: str
    sponsor: str | None
    target_symbol: str | None
    indication: str
    modality: str
    similarity: float
    matched_facets: list[dict[str, Any]]
    rationale: str
    outcome_label: str
    outcome_display: str
    outcome_date: dt.date | None
    highest_phase_reached: str
    discontinuation_reason: str
    outcome_confidence: str
    ind_or_cta_on: dt.date | None
    is_synthetic: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


OUTCOME_DISPLAY: dict[str, str] = {
    E.OutcomeLabel.PHASE_2_EFFICACY_FAILURE: "Failed in Phase II on efficacy",
    E.OutcomeLabel.PHASE_2_SAFETY_FAILURE: "Failed in Phase II on safety",
    E.OutcomeLabel.PHASE_1_SAFETY_FAILURE: "Failed in Phase I on safety",
    E.OutcomeLabel.PHASE_1_PKPD_FAILURE: "Failed in Phase I on PK/PD",
    E.OutcomeLabel.PROGRESSED_PAST_PHASE_2: "Progressed beyond Phase II",
    E.OutcomeLabel.PHASE_3_FAILURE: "Reached Phase III, then failed",
    E.OutcomeLabel.REGULATORY_APPROVAL: "Approved",
    E.OutcomeLabel.DISCONTINUED_THROUGH_PHASE_2: "Discontinued at or before Phase II",
    E.OutcomeLabel.FAILED_BEFORE_HUMAN_DOSING: "Abandoned before first human dose",
    E.OutcomeLabel.ACTIVE_NO_OUTCOME_YET: "Still in development",
    E.OutcomeLabel.CENSORED_INSUFFICIENT_FOLLOWUP: "Outcome not yet observable",
    E.OutcomeLabel.UNKNOWN: "Outcome not established",
}


def _mechanism_class(program: Program, target: Target | None) -> str:
    tc = TARGET_CLASS_GROUPS.get(getattr(target, "target_class", "unknown") or "unknown", "other")
    mg = MODALITY_GROUPS.get(program.modality, "other")
    return f"{tc}|{mg}"


class _FacetScorer:
    """Accumulates facet matches for one candidate comparator.

    A small class rather than a closure defined inside the candidate loop.
    The closure version worked — it was defined and called within the same
    iteration — but it captured loop-local accumulators by reference, which is
    the shape of a real late-binding bug even when this instance is not one.
    Holding the accumulators on an object makes the state explicit and removes
    the hazard rather than suppressing the warning about it.
    """

    __slots__ = ("matched", "score")

    def __init__(self) -> None:
        self.matched: list[dict[str, Any]] = []
        self.score: float = 0.0

    def award(self, facet: str, condition: bool, detail: str) -> None:
        """Credit a facet at its full weight, if the condition holds."""
        if condition:
            self.award_partial(facet, 1.0, FACET_LABELS[facet], detail)

    def award_partial(
        self, facet: str, fraction: float, label: str, detail: str
    ) -> None:
        """Credit a fraction of a facet's weight — for a weaker kind of match."""
        w = FACET_WEIGHTS[facet] * fraction
        self.score += w
        self.matched.append(
            {"facet": facet, "label": label, "weight": w, "detail": detail}
        )


def find_comparators(
    session: Session,
    track: ProgramIndication,
    *,
    as_of: dt.date,
    limit: int = 10,
    min_similarity: float = 0.20,
    require_resolved_outcome: bool = True,
) -> list[Comparator]:
    """Retrieve historical programs comparable to ``track``, as of ``as_of``."""
    focal_program = session.get(Program, track.program_id)
    focal_target = (
        session.get(Target, focal_program.primary_target_id)
        if focal_program.primary_target_id else None
    )
    focal_indication = session.get(Indication, track.indication_id)
    focal_mechanism = _mechanism_class(focal_program, focal_target)
    focal_modality_group = MODALITY_GROUPS.get(focal_program.modality, "other")

    rows = session.execute(
        select(ProgramIndication, Program)
        .join(Program, Program.id == ProgramIndication.program_id)
        .where(ProgramIndication.program_id != track.program_id)
    ).all()

    results: list[Comparator] = []
    for other, other_program in rows:
        # Temporal gate: the outcome must have been known at the cutoff.
        if require_resolved_outcome:
            if other.outcome_date is None or other.outcome_date > as_of:
                continue
            if other.outcome_label in (
                E.OutcomeLabel.ACTIVE_NO_OUTCOME_YET,
                E.OutcomeLabel.CENSORED_INSUFFICIENT_FOLLOWUP,
                E.OutcomeLabel.UNKNOWN,
            ):
                continue

        other_target = (
            session.get(Target, other_program.primary_target_id)
            if other_program.primary_target_id else None
        )
        other_indication = session.get(Indication, other.indication_id)

        scorer = _FacetScorer()
        award = scorer.award

        award(
            "target",
            focal_target is not None and other_program.primary_target_id == focal_target.id,
            f"Both target {getattr(focal_target, 'symbol', '?')}",
        )
        award(
            "mechanism_class",
            _mechanism_class(other_program, other_target) == focal_mechanism,
            f"Both {focal_mechanism.replace('|', ' / ').replace('_', ' ')}",
        )
        award(
            "modality",
            MODALITY_GROUPS.get(other_program.modality, "other") == focal_modality_group,
            f"Both {focal_modality_group.replace('_', ' ')}",
        )
        same_indication = other.indication_id == track.indication_id
        same_area = (
            other_indication is not None and focal_indication is not None
            and other_indication.therapeutic_area == focal_indication.therapeutic_area
        )
        if same_indication:
            award("indication", True, f"Both in {focal_indication.name}")
        elif same_area:
            # Partial credit: same therapeutic area is weaker than same disease.
            scorer.award_partial(
                "indication", 0.35, "Same therapeutic area",
                f"Both in {focal_indication.therapeutic_area}",
            )
        award(
            "disease_stage",
            other.disease_stage == track.disease_stage
            and track.disease_stage != E.DiseaseStage.UNKNOWN,
            f"Both {track.disease_stage.replace('_', ' ')}",
        )
        award(
            "biomarker_strategy",
            other.biomarker_selection == track.biomarker_selection
            and track.biomarker_selection != E.BiomarkerSelectionStrategy.UNKNOWN,
            f"Both {track.biomarker_selection.replace('_', ' ')}",
        )

        score, matched = scorer.score, scorer.matched
        if score < min_similarity:
            continue

        sponsor = (
            session.get(Sponsor, other_program.current_sponsor_id)
            if other_program.current_sponsor_id else None
        )
        top = sorted(matched, key=lambda m: -m["weight"])[:3]
        rationale = "; ".join(m["detail"] for m in top) if top else "Weak structural overlap"

        results.append(Comparator(
            track_id=other.id,
            program_id=other.program_id,
            program_name=other_program.name,
            sponsor=sponsor.name if sponsor else None,
            target_symbol=getattr(other_target, "symbol", None),
            indication=other_indication.name if other_indication else "—",
            modality=other_program.modality,
            similarity=round(min(score, 1.0), 4),
            matched_facets=matched,
            rationale=rationale,
            outcome_label=other.outcome_label,
            outcome_display=OUTCOME_DISPLAY.get(other.outcome_label, other.outcome_label),
            outcome_date=other.outcome_date,
            highest_phase_reached=other.highest_phase_reached,
            discontinuation_reason=other.discontinuation_reason,
            outcome_confidence=other.outcome_confidence,
            ind_or_cta_on=other.ind_or_cta_on,
            is_synthetic=bool(other.is_synthetic or other_program.is_synthetic),
        ))

    results.sort(key=lambda c: (-c.similarity, c.outcome_date or dt.date.min))
    return results[:limit]


def summarise_comparators(comparators: list[Comparator]) -> dict[str, Any]:
    """Aggregate outcomes across comparators, with the causal caveat attached."""
    if not comparators:
        return {
            "n": 0,
            "caveat": (
                "No historical comparators met the similarity threshold and were "
                "resolved before this program's cutoff date."
            ),
        }

    counts: dict[str, int] = {}
    for c in comparators:
        counts[c.outcome_display] = counts.get(c.outcome_display, 0) + 1

    efficacy_failures = sum(
        1 for c in comparators if c.outcome_label == E.OutcomeLabel.PHASE_2_EFFICACY_FAILURE
    )
    progressed = sum(
        1 for c in comparators
        if c.outcome_label in (
            E.OutcomeLabel.PROGRESSED_PAST_PHASE_2, E.OutcomeLabel.REGULATORY_APPROVAL,
            E.OutcomeLabel.PHASE_3_FAILURE,
        )
    )
    resolved = efficacy_failures + progressed

    return {
        "n": len(comparators),
        "outcome_counts": counts,
        "n_efficacy_failures": efficacy_failures,
        "n_progressed_past_phase_2": progressed,
        "observed_failure_share": (efficacy_failures / resolved) if resolved else None,
        "mean_similarity": round(
            sum(c.similarity for c in comparators) / len(comparators), 4
        ),
        "caveat": (
            "These programs resemble the focal program on recorded structural "
            "facets. Similarity is not exchangeability: two programs can match "
            "on every facet this system records and differ in whatever actually "
            "decided their outcomes. The observed share is descriptive of this "
            "comparator set only — it is not a prediction, not a base rate, and "
            "not evidence of a causal relationship."
        ),
    }
