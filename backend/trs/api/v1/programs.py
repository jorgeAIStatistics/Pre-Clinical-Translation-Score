"""Program, track and evidence endpoints."""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trs.api.deps import get_db, get_principal
from trs.api.schemas import (
    EvidenceMatrix,
    EvidenceMatrixCell,
    EvidenceOut,
    IndicationSummary,
    ProgramAliasOut,
    ProgramDetail,
    SourceOut,
    SponsorSummary,
    TargetSummary,
    TrackSummary,
)
from trs.api.security import Principal
from trs.db.models.core import (
    Indication,
    Program,
    ProgramAlias,
    ProgramIndication,
    Sponsor,
    Target,
)
from trs.db.models.evidence import EvidenceItem, PkPdBridge, SafetyFinding, Source, TargetAssessment
from trs.features.effects import standardise_effect
from trs.features.temporal import AsOfContext

router = APIRouter(prefix="/programs", tags=["programs"])


def _track_payload(session: Session, track: ProgramIndication, n_evidence: int = 0) -> dict[str, Any]:
    program = session.get(Program, track.program_id)
    target = (
        session.get(Target, program.primary_target_id)
        if program.primary_target_id else None
    )
    indication = session.get(Indication, track.indication_id)
    sponsor = (
        session.get(Sponsor, program.current_sponsor_id)
        if program.current_sponsor_id else None
    )
    return {
        "id": track.id,
        "program_id": program.id,
        "program_name": program.name,
        "program_code": program.code,
        "target": TargetSummary.model_validate(target) if target else None,
        "indication": IndicationSummary.model_validate(indication),
        "sponsor": SponsorSummary.model_validate(sponsor) if sponsor else None,
        "modality": program.modality,
        "mechanism_of_action": program.mechanism_of_action,
        "disease_stage": track.disease_stage,
        "line_of_therapy": track.line_of_therapy,
        "biomarker_selection": track.biomarker_selection,
        "unmet_need": track.unmet_need,
        "combination_partner": track.combination_partner,
        "ind_or_cta_on": track.ind_or_cta_on,
        "first_in_human_on": track.first_in_human_on,
        "highest_phase_reached": track.highest_phase_reached,
        "outcome_label": track.outcome_label,
        "outcome_date": track.outcome_date,
        "discontinuation_reason": track.discontinuation_reason,
        "outcome_confidence": track.outcome_confidence,
        "is_synthetic": bool(track.is_synthetic or program.is_synthetic),
        "n_evidence_items": n_evidence,
    }


@router.get("", response_model=list[TrackSummary])
def list_programs(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    q: str | None = Query(default=None, description="Free-text match on program name, code or alias."),
    target: str | None = Query(default=None, description="Target symbol."),
    modality: str | None = None,
    therapeutic_area: str | None = None,
    outcome: str | None = None,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
) -> list[TrackSummary]:
    """List development tracks visible to the caller's organisation."""
    stmt = (
        select(ProgramIndication)
        .join(Program, Program.id == ProgramIndication.program_id)
        .join(Indication, Indication.id == ProgramIndication.indication_id)
        .where(
            (Program.organization_id == principal.organization_id)
            | (Program.organization_id.is_(None))
        )
    )
    if q:
        pattern = f"%{q.lower()}%"
        alias_match = select(ProgramAlias.program_id).where(
            func.lower(ProgramAlias.alias_normalised).like(pattern)
        )
        stmt = stmt.where(
            func.lower(Program.name).like(pattern)
            | func.lower(func.coalesce(Program.code, "")).like(pattern)
            | Program.id.in_(alias_match)
        )
    if target:
        stmt = stmt.join(Target, Target.id == Program.primary_target_id).where(
            func.lower(Target.symbol) == target.lower()
        )
    if modality:
        stmt = stmt.where(Program.modality == modality)
    if therapeutic_area:
        stmt = stmt.where(Indication.therapeutic_area == therapeutic_area)
    if outcome:
        stmt = stmt.where(ProgramIndication.outcome_label == outcome)

    tracks = list(
        session.scalars(
            stmt.order_by(Program.name, Indication.name).offset(offset).limit(limit)
        )
    )
    counts = dict(
        session.execute(
            select(EvidenceItem.program_indication_id, func.count())
            .where(EvidenceItem.program_indication_id.in_([t.id for t in tracks] or [""]))
            .group_by(EvidenceItem.program_indication_id)
        ).all()
    )
    return [
        TrackSummary.model_validate(_track_payload(session, t, counts.get(t.id, 0)))
        for t in tracks
    ]


@router.get("/{track_id}", response_model=ProgramDetail)
def get_program(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> ProgramDetail:
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")

    n_evidence = session.scalar(
        select(func.count()).select_from(EvidenceItem).where(
            EvidenceItem.program_indication_id == track_id
        )
    ) or 0
    payload = _track_payload(session, track, n_evidence)

    program = session.get(Program, track.program_id)
    payload["first_disclosure_on"] = program.first_disclosure_on
    payload["standard_of_care_note"] = track.standard_of_care_note
    payload["aliases"] = [
        ProgramAliasOut.model_validate(a)
        for a in session.scalars(
            select(ProgramAlias).where(ProgramAlias.program_id == program.id)
        )
    ]
    # Sibling tracks: the same asset pursued in other indications. Shown because
    # a reader must know the program is not a single-indication bet.
    payload["all_tracks"] = [
        {
            "track_id": t.id,
            "indication": session.get(Indication, t.indication_id).name,
            "outcome_label": t.outcome_label,
            "ind_or_cta_on": t.ind_or_cta_on.isoformat() if t.ind_or_cta_on else None,
            "is_current": t.id == track_id,
        }
        for t in session.scalars(
            select(ProgramIndication).where(ProgramIndication.program_id == program.id)
        )
    ]
    return ProgramDetail.model_validate(payload)


@router.get("/{track_id}/evidence", response_model=list[EvidenceOut])
def get_evidence(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = Query(
        default=None,
        description=(
            "Restrict to evidence available on or before this date. Defaults to "
            "the program's IND/CTA date, which is the cutoff the score uses."
        ),
    ),
    evidence_class: str | None = None,
    species: str | None = None,
    direction: str | None = None,
    include_post_cutoff: bool = Query(
        default=False,
        description=(
            "Include evidence that postdates the cutoff. Such items are visible "
            "for context but never contribute to a score."
        ),
    ),
) -> list[EvidenceOut]:
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")

    cutoff = as_of or track.ind_or_cta_on or track.first_in_human_on or dt.date.today()

    if include_post_cutoff:
        stmt = select(EvidenceItem).where(
            EvidenceItem.program_indication_id == track_id
        ).order_by(EvidenceItem.available_at.desc())
        items = list(session.scalars(stmt))
    else:
        ctx = AsOfContext(session=session, as_of=cutoff, strict=True)
        items = ctx.evidence_for_track(track_id)

    if evidence_class:
        items = [e for e in items if e.evidence_class == evidence_class]
    if species:
        items = [e for e in items if e.species == species]
    if direction:
        items = [e for e in items if e.direction == direction]

    out: list[EvidenceOut] = []
    for e in items:
        payload = {c.name: getattr(e, c.name) for c in EvidenceItem.__table__.columns}
        payload["standardised_effect"] = standardise_effect(
            e.effect_metric, e.effect_value, favours_treatment=e.effect_favours_treatment
        )
        src = session.get(Source, e.source_id) if e.source_id else None
        payload["source"] = SourceOut.model_validate(src) if src else None
        out.append(EvidenceOut.model_validate(payload))
    return out


@router.get("/{track_id}/evidence-matrix", response_model=EvidenceMatrix)
def get_evidence_matrix(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = None,
) -> EvidenceMatrix:
    """Evidence class × biological system, with endpoint detail per cell.

    The value of this view is negative space: a matrix with an empty
    human-derived row says more about a program than any populated cell does.
    """
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")

    cutoff = as_of or track.ind_or_cta_on or track.first_in_human_on or dt.date.today()
    ctx = AsOfContext(session=session, as_of=cutoff, strict=True)
    items = ctx.evidence_for_track(track_id)

    from trs.ontology.enums import DEFAULT_EVIDENCE_CLASS_ORDER, BiologicalSystem

    rows = [str(c) for c in DEFAULT_EVIDENCE_CLASS_ORDER]
    columns = [
        str(BiologicalSystem.CELL_FREE_BIOCHEMICAL),
        str(BiologicalSystem.IMMORTALISED_CELL_LINE),
        str(BiologicalSystem.PRIMARY_CELL),
        str(BiologicalSystem.ORGANOID),
        str(BiologicalSystem.EX_VIVO_TISSUE),
        str(BiologicalSystem.IN_VIVO_WHOLE_ANIMAL),
        str(BiologicalSystem.HUMAN_SUBJECT),
        str(BiologicalSystem.HUMAN_POPULATION),
    ]

    buckets: dict[tuple[str, str, str], list[EvidenceItem]] = {}
    for e in items:
        buckets.setdefault(
            (e.evidence_class, e.biological_system, e.endpoint_category), []
        ).append(e)

    cells = [
        EvidenceMatrixCell(
            evidence_class=ec,
            system=sys_,
            endpoint=ep,
            n_total=len(group),
            n_supporting=sum(1 for g in group if g.direction == "supports"),
            n_refuting=sum(1 for g in group if g.direction == "refutes"),
            evidence_ids=[g.id for g in group],
        )
        for (ec, sys_, ep), group in sorted(buckets.items())
    ]
    occupied = {(c.evidence_class, c.system) for c in cells}
    total_cells = len(rows) * len(columns)

    return EvidenceMatrix(
        rows=rows,
        columns=columns,
        cells=cells,
        n_evidence=len(items),
        empty_cells=total_cells - len(occupied),
        coverage=round(len(occupied) / total_cells, 4) if total_cells else 0.0,
        note=(
            f"Evidence available at the {cutoff.isoformat()} cutoff. Empty cells "
            "are gaps in the evidence base, not zeros: they mean no study of "
            "that type is on record, which is different from a study that found "
            "nothing."
        ),
    )


@router.get("/{track_id}/pkpd", tags=["evidence"])
def get_pkpd(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = None,
) -> dict[str, Any]:
    """The PK/PD translational bridge available at the cutoff, plus its history."""
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")
    cutoff = as_of or track.ind_or_cta_on or track.first_in_human_on or dt.date.today()

    all_bridges = list(session.scalars(
        select(PkPdBridge).where(PkPdBridge.program_indication_id == track_id)
        .order_by(PkPdBridge.available_at.desc())
    ))
    current = next((b for b in all_bridges if b.available_at <= cutoff), None)

    def serialise(b: PkPdBridge) -> dict[str, Any]:
        return {c.name: getattr(b, c.name) for c in PkPdBridge.__table__.columns}

    return {
        "track_id": track_id,
        "as_of_date": cutoff.isoformat(),
        "current": serialise(current) if current else None,
        "history": [serialise(b) for b in all_bridges],
        "available_at_cutoff": current is not None,
        "note": (
            "The bridge shown is the most recent assembled on or before the "
            "cutoff. Later revisions are listed in history for context but do "
            "not contribute to the score."
            if current else
            "No PK/PD bridge was on record at the cutoff. The exposure features "
            "are imputed, and the prediction interval widens accordingly."
        ),
    }


@router.get("/{track_id}/target-assessment", tags=["evidence"])
def get_target_assessment(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = None,
) -> dict[str, Any]:
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")
    program = session.get(Program, track.program_id)
    cutoff = as_of or track.ind_or_cta_on or track.first_in_human_on or dt.date.today()

    ctx = AsOfContext(session=session, as_of=cutoff, strict=True)
    current = ctx.latest_target_assessment(program.primary_target_id, track.indication_id)
    history = list(session.scalars(
        select(TargetAssessment)
        .where(TargetAssessment.target_id == program.primary_target_id)
        .order_by(TargetAssessment.available_at.desc())
    )) if program.primary_target_id else []

    def serialise(a: TargetAssessment) -> dict[str, Any]:
        return {c.name: getattr(a, c.name) for c in TargetAssessment.__table__.columns}

    return {
        "track_id": track_id,
        "as_of_date": cutoff.isoformat(),
        "current": serialise(current) if current else None,
        "history": [serialise(a) for a in history],
        "note": (
            "Target knowledge accumulates. The assessment shown is the latest "
            "published on or before the cutoff — never the most recent overall."
        ),
    }


@router.get("/{track_id}/safety", tags=["evidence"])
def get_safety(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = None,
) -> dict[str, Any]:
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")
    cutoff = as_of or track.ind_or_cta_on or track.first_in_human_on or dt.date.today()
    ctx = AsOfContext(session=session, as_of=cutoff, strict=True)
    findings = ctx.safety_findings(track_id, track.program_id)
    return {
        "track_id": track_id,
        "as_of_date": cutoff.isoformat(),
        "findings": [
            {c.name: getattr(f, c.name) for c in SafetyFinding.__table__.columns}
            for f in findings
        ],
        "n_findings": len(findings),
    }
