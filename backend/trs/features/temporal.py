"""Temporal integrity: the single chokepoint for as-of filtering.

The property this module defends
--------------------------------
For a development track whose decision point is date ``T``, the feature vector
must be a function only of facts knowable at ``T``. If any later fact leaks in,
every validation metric computed downstream is optimistic by an unknown amount
and the product is worthless.

Rather than trusting every call site to remember a ``WHERE available_at <= T``
clause, all evidence retrieval goes through :class:`AsOfContext`. Call sites
cannot obtain evidence rows any other way; ``trs.features.builder`` takes a
context, not a session. The leakage test suite
(``tests/statistical/test_temporal_leakage.py``) exercises the invariant
directly: it builds a snapshot at ``T``, inserts post-``T`` evidence designed
to move every feature, rebuilds, and asserts bit-identical output.

Why ``available_at`` and not ``published_on``
---------------------------------------------
They differ, and the difference matters. A 2015 paper reporting a 2011
experiment became *knowable* in 2015. A trial registered in 2013 whose "why
stopped" field was populated in 2018 yields two facts with two different
availability dates. ``available_at`` is always the earlier of "when a diligent
analyst could have found this" — never the date of the underlying event.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, TypeVar

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from trs.db.models.core import ProgramIndication
from trs.db.models.evidence import (
    EvidenceItem,
    PkPdBridge,
    ProgramBiomarker,
    SafetyFinding,
    TargetAssessment,
)

T = TypeVar("T")


class TemporalIntegrityError(RuntimeError):
    """Raised when a query would admit information from after the cutoff."""


@dataclasses.dataclass(frozen=True, slots=True)
class AsOfContext:
    """A cutoff date plus the session to read through.

    Every accessor applies ``available_at <= as_of`` before returning rows.
    ``strict`` additionally verifies the returned rows, which is cheap and
    catches a mis-specified filter rather than trusting the SQL to be right.
    """

    session: Session
    as_of: dt.date
    strict: bool = True

    # ---- generic guard ----------------------------------------------------

    def _verify(self, rows: list[Any], attr: str = "available_at") -> list[Any]:
        if not self.strict:
            return rows
        for row in rows:
            value = getattr(row, attr, None)
            if value is not None and value > self.as_of:
                raise TemporalIntegrityError(
                    f"{type(row).__name__} {getattr(row, 'id', '?')} has {attr}="
                    f"{value} which is after the as-of cutoff {self.as_of}. "
                    "A filter is missing or wrong."
                )
        return rows

    def guard(self, stmt: Select, column) -> Select:  # noqa: ANN001
        """Apply the cutoff predicate to an arbitrary select."""
        return stmt.where(column <= self.as_of)

    # ---- evidence accessors ----------------------------------------------

    def evidence_for_track(
        self,
        track_id: str,
        *,
        evidence_classes: tuple[str, ...] | None = None,
        include_program_level: bool = True,
    ) -> list[EvidenceItem]:
        """Evidence attached to a development track.

        ``include_program_level`` also returns evidence attached to the parent
        program without an indication, which is how target-level and
        chemistry-level findings are usually recorded.
        """
        track = self.session.get(ProgramIndication, track_id)
        if track is None:
            return []

        conditions = [EvidenceItem.program_indication_id == track_id]
        if include_program_level:
            conditions.append(
                and_(
                    EvidenceItem.program_id == track.program_id,
                    EvidenceItem.program_indication_id.is_(None),
                )
            )

        stmt = select(EvidenceItem).where(
            EvidenceItem.available_at <= self.as_of,
            _or_(conditions),
        )
        if evidence_classes:
            stmt = stmt.where(EvidenceItem.evidence_class.in_(evidence_classes))
        stmt = stmt.order_by(EvidenceItem.available_at, EvidenceItem.id)
        return self._verify(list(self.session.scalars(stmt)))

    def latest_pkpd_bridge(self, track_id: str) -> PkPdBridge | None:
        """Most recent PK/PD bridge assembled at or before the cutoff.

        "Most recent at or before" — never "most recent overall". A program's
        exposure argument improves over time; scoring a 2014 decision with a
        2019 bridge is exactly the leak this system exists to prevent.
        """
        stmt = (
            select(PkPdBridge)
            .where(
                PkPdBridge.program_indication_id == track_id,
                PkPdBridge.available_at <= self.as_of,
            )
            .order_by(PkPdBridge.available_at.desc(), PkPdBridge.id.desc())
            .limit(1)
        )
        row = self.session.scalars(stmt).first()
        return self._verify([row])[0] if row is not None else None

    def latest_target_assessment(
        self, target_id: str | None, indication_id: str | None = None
    ) -> TargetAssessment | None:
        """Latest target assessment at or before the cutoff.

        Prefers an assessment specific to the indication; falls back to a
        target-level assessment with no indication.
        """
        if target_id is None:
            return None
        base = select(TargetAssessment).where(
            TargetAssessment.target_id == target_id,
            TargetAssessment.available_at <= self.as_of,
        )
        if indication_id is not None:
            stmt = (
                base.where(TargetAssessment.indication_id == indication_id)
                .order_by(TargetAssessment.available_at.desc(), TargetAssessment.id.desc())
                .limit(1)
            )
            row = self.session.scalars(stmt).first()
            if row is not None:
                return self._verify([row])[0]
        stmt = (
            base.where(TargetAssessment.indication_id.is_(None))
            .order_by(TargetAssessment.available_at.desc(), TargetAssessment.id.desc())
            .limit(1)
        )
        row = self.session.scalars(stmt).first()
        return self._verify([row])[0] if row is not None else None

    def biomarkers_for_track(self, track_id: str) -> list[ProgramBiomarker]:
        """Latest readiness state per biomarker at or before the cutoff.

        A biomarker can appear several times as it matures. Only the most
        recent state on or before the cutoff counts.
        """
        stmt = (
            select(ProgramBiomarker)
            .where(
                ProgramBiomarker.program_indication_id == track_id,
                ProgramBiomarker.available_at <= self.as_of,
            )
            .order_by(ProgramBiomarker.available_at, ProgramBiomarker.id)
        )
        rows = self._verify(list(self.session.scalars(stmt)))
        latest: dict[str, ProgramBiomarker] = {}
        for row in rows:
            key = row.biomarker_id or f"{row.biomarker_type}:{row.intended_use_in_trial}"
            latest[key] = row  # ordered ascending, so the last write wins
        return list(latest.values())

    def safety_findings(self, track_id: str, program_id: str | None = None) -> list[SafetyFinding]:
        conditions = [SafetyFinding.program_indication_id == track_id]
        if program_id:
            conditions.append(
                and_(
                    SafetyFinding.program_id == program_id,
                    SafetyFinding.program_indication_id.is_(None),
                )
            )
        stmt = (
            select(SafetyFinding)
            .where(SafetyFinding.available_at <= self.as_of, _or_(conditions))
            .order_by(SafetyFinding.available_at, SafetyFinding.id)
        )
        return self._verify(list(self.session.scalars(stmt)))


def _or_(conditions: list[Any]) -> Any:
    """``OR`` over a list, collapsing the single-element case."""
    if len(conditions) == 1:
        return conditions[0]
    from sqlalchemy import or_

    return or_(*conditions)


def years_between(start: dt.date | None, end: dt.date | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).days / 365.25
