"""Cohort construction: development tracks → labelled, dated analysis units.

Produces a frozen :class:`~trs.db.models.analytics.CohortVersion` plus one
:class:`~trs.db.models.analytics.CohortEntry` per included track, and records
every exclusion with its reason so the Methods page can show the full
attrition waterfall rather than a final N.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.cohort.outcomes import ENDPOINTS, PRIMARY_ENDPOINT, EndpointDefinition, label_track
from trs.db.base import utcnow
from trs.db.models.analytics import CohortEntry, CohortVersion
from trs.db.models.core import Program, ProgramIndication
from trs.features.historical import mechanism_key_for

#: How the as-of date is chosen for each track.
AS_OF_RULES = {
    "ind_or_cta_on": (
        "The regulatory filing that authorises first human dosing — the point "
        "at which the go/no-go decision this product supports is actually made."
    ),
    "first_in_human_on": "Date of first human dose.",
    "phase_2_start_on": "Start of the first Phase II study.",
}


@dataclasses.dataclass(slots=True)
class CohortBuildResult:
    cohort_version: CohortVersion
    entries: list[CohortEntry]
    exclusions: Counter
    n_candidates: int

    @property
    def summary(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_included": len(self.entries),
            "n_excluded": sum(self.exclusions.values()),
            "exclusions": dict(self.exclusions.most_common()),
        }


def _as_of_for(track: ProgramIndication, rule: str) -> dt.date | None:
    """Resolve the cutoff date, falling back down the milestone chain.

    Falling back is safe in one direction only: an earlier milestone is always
    a valid (more conservative) cutoff, because it admits strictly less
    information. Falling *forward* — using Phase II start when the IND date is
    unknown — would admit Phase I results into a preclinical feature vector, so
    it is never done.
    """
    primary = getattr(track, rule, None)
    if primary is not None:
        return primary
    if rule == "ind_or_cta_on":
        # An IND precedes first dosing; using the dosing date would be
        # permissive, so subtract nothing and accept the small conservatism of
        # a slightly later cutoff only when no IND date exists at all.
        return track.first_in_human_on
    return None


def build_cohort(
    session: Session,
    *,
    name: str = "oncology_ph2_efficacy",
    version: str = "v1",
    endpoint: EndpointDefinition = PRIMARY_ENDPOINT,
    as_of_rule: str = "ind_or_cta_on",
    data_cutoff: dt.date | None = None,
    therapeutic_areas: tuple[str, ...] | None = None,
    description: str | None = None,
    persist: bool = True,
) -> CohortBuildResult:
    """Build one cohort version for one primary endpoint.

    Secondary endpoints are evaluated for every included track and stored in
    ``CohortEntry.labels_secondary``, so a single cohort supports several
    supervised problems without re-deriving membership.
    """
    if as_of_rule not in AS_OF_RULES:
        raise ValueError(f"Unknown as_of_rule {as_of_rule!r}; expected one of {list(AS_OF_RULES)}")

    data_cutoff = data_cutoff or dt.date.today()
    exclusions: Counter = Counter()
    entries: list[CohortEntry] = []

    stmt = select(ProgramIndication, Program).join(
        Program, Program.id == ProgramIndication.program_id
    )
    candidates = list(session.execute(stmt))

    for track, program in candidates:
        as_of = _as_of_for(track, as_of_rule)
        if as_of is None:
            exclusions["no_as_of_date"] += 1
            continue
        if as_of > data_cutoff:
            exclusions["as_of_after_data_cutoff"] += 1
            continue

        if therapeutic_areas:
            ta = getattr(track.indication, "therapeutic_area", None)
            if ta not in therapeutic_areas:
                exclusions[f"therapeutic_area_not_in_scope:{ta}"] += 1
                continue

        decision = label_track(track, endpoint, as_of=as_of, data_cutoff=data_cutoff)
        if decision.get("excluded"):
            exclusions[decision.get("exclusion_reason") or "unspecified"] += 1
            continue

        secondary: dict[str, int | None] = {}
        for other_name, other in ENDPOINTS.items():
            if other_name == endpoint.name:
                continue
            d = label_track(track, other, as_of=as_of, data_cutoff=data_cutoff)
            secondary[other_name] = None if d.get("excluded") else d.get("label")

        time_to_event = None
        if track.outcome_date is not None:
            time_to_event = (track.outcome_date - as_of).days
        elif track.status_verified_on is not None:
            time_to_event = (track.status_verified_on - as_of).days

        entries.append(
            CohortEntry(
                program_indication_id=track.id,
                as_of_date=as_of,
                label_primary=decision.get("label"),
                labels_secondary=secondary,
                label_source_outcome=decision.get("source_outcome"),
                label_confidence=track.outcome_confidence,
                exclusion_reason=None,
                time_to_event_days=time_to_event,
                event_observed=track.outcome_date is not None,
                event_type=track.discontinuation_reason,
                group_program_id=track.program_id,
                group_target_id=program.primary_target_id,
                group_sponsor_id=program.current_sponsor_id or program.originator_sponsor_id,
                group_mechanism_key=mechanism_key_for(program),
                is_synthetic=bool(track.is_synthetic or program.is_synthetic),
            )
        )

    n_pos = sum(1 for e in entries if e.label_primary == 1)
    n_neg = sum(1 for e in entries if e.label_primary == 0)

    definition = {
        "endpoint": endpoint.to_dict(),
        "as_of_rule": as_of_rule,
        "as_of_rule_description": AS_OF_RULES[as_of_rule],
        "data_cutoff": data_cutoff.isoformat(),
        "therapeutic_areas": list(therapeutic_areas) if therapeutic_areas else None,
    }
    dataset_hash = hashlib.sha256(
        json.dumps(
            {
                "definition": definition,
                "members": sorted(
                    (e.program_indication_id, e.as_of_date.isoformat(), e.label_primary)
                    for e in entries
                ),
            },
            sort_keys=True, default=str,
        ).encode()
    ).hexdigest()

    cv = CohortVersion(
        name=name,
        version=version,
        description=description or (
            f"{endpoint.display_name} cohort, as-of rule '{as_of_rule}', "
            f"data cutoff {data_cutoff.isoformat()}."
        ),
        definition=definition,
        inclusion_summary={
            "n_candidates": len(candidates),
            "n_included": len(entries),
            "exclusions": dict(exclusions.most_common()),
        },
        n_entries=len(entries),
        n_positive=n_pos,
        n_negative=n_neg,
        n_excluded=sum(exclusions.values()),
        as_of_rule=as_of_rule,
        built_at=utcnow(),
        dataset_hash=dataset_hash,
        contains_synthetic=any(e.is_synthetic for e in entries),
    )

    if persist:
        session.add(cv)
        session.flush()
        for e in entries:
            e.cohort_version_id = cv.id
        session.add_all(entries)
        session.flush()

    return CohortBuildResult(
        cohort_version=cv,
        entries=entries,
        exclusions=exclusions,
        n_candidates=len(candidates),
    )
