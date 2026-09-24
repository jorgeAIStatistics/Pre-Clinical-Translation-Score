"""Portfolio view: many programs, compared.

Deliberately not a ranking. Rows carry the estimate, its interval, the evidence
confidence, the completeness and the leading risk driver together, because a
list sorted by probability alone invites exactly the decision this product
exists to prevent — cutting the program with the highest number without looking
at how much evidence that number rests on.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trs.api.deps import get_db, get_principal
from trs.api.schemas import PortfolioResponse, PortfolioRow
from trs.api.security import Principal
from trs.cohort.outcomes import ENDPOINTS
from trs.db.models.analytics import ModelArtifact
from trs.db.models.core import Indication, Program, ProgramIndication, Sponsor, Target
from trs.db.models.evidence import EvidenceItem
from trs.db.models.tenancy import Project
from trs.scoring.service import ScoringError, ScoringService

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioResponse)
def get_portfolio(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    project_id: str | None = None,
    endpoint: str = "ph2_efficacy_failure",
    therapeutic_area: str | None = None,
    modality: str | None = None,
    target: str | None = None,
    limit: int = Query(default=200, le=500),
    score: bool = Query(
        default=True,
        description="Compute a live estimate per row. Disable for a fast listing.",
    ),
) -> PortfolioResponse:
    stmt = (
        select(ProgramIndication, Program, Indication)
        .join(Program, Program.id == ProgramIndication.program_id)
        .join(Indication, Indication.id == ProgramIndication.indication_id)
        .where(
            (Program.organization_id == principal.organization_id)
            | (Program.organization_id.is_(None))
        )
        .where(ProgramIndication.ind_or_cta_on.is_not(None))
    )
    if project_id:
        stmt = stmt.where(Program.project_id == project_id)
    if therapeutic_area:
        stmt = stmt.where(Indication.therapeutic_area == therapeutic_area)
    if modality:
        stmt = stmt.where(Program.modality == modality)
    if target:
        stmt = stmt.join(Target, Target.id == Program.primary_target_id).where(
            func.lower(Target.symbol) == target.lower()
        )

    records = list(session.execute(stmt.order_by(Program.name).limit(limit)))
    track_ids = [t.id for t, _, _ in records]
    evidence_counts = dict(
        session.execute(
            select(EvidenceItem.program_indication_id, func.count())
            .where(EvidenceItem.program_indication_id.in_(track_ids or [""]))
            .group_by(EvidenceItem.program_indication_id)
        ).all()
    )

    artifact = session.scalars(
        select(ModelArtifact).where(
            ModelArtifact.endpoint == endpoint, ModelArtifact.is_active.is_(True)
        )
    ).first()

    service = ScoringService(session) if (score and artifact) else None
    rows: list[PortfolioRow] = []
    any_synthetic = False

    for track, program, indication in records:
        target_obj = (
            session.get(Target, program.primary_target_id)
            if program.primary_target_id else None
        )
        sponsor = (
            session.get(Sponsor, program.current_sponsor_id)
            if program.current_sponsor_id else None
        )
        synthetic = bool(track.is_synthetic or program.is_synthetic)
        any_synthetic = any_synthetic or synthetic

        row: dict[str, Any] = {
            "track_id": track.id,
            "program_id": program.id,
            "program_name": program.name,
            "target_symbol": getattr(target_obj, "symbol", None),
            "indication": indication.name,
            "therapeutic_area": indication.therapeutic_area,
            "modality": program.modality,
            "sponsor": sponsor.name if sponsor else None,
            "stage": track.highest_phase_reached,
            "as_of_date": track.ind_or_cta_on,
            "n_evidence_items": evidence_counts.get(track.id, 0),
            "outcome_label": track.outcome_label,
            "is_synthetic": synthetic,
        }

        if service is not None:
            try:
                # Lighter sampling than a single-program score: the portfolio
                # needs comparable point estimates and honest interval widths,
                # not the precision a detail page warrants.
                result = service.score_track(
                    track.id, endpoint=endpoint, persist=False,
                    n_imputations=8, n_param_samples=150, compare_families=False,
                )
                top = None
                if result.explanation and result.explanation["increasing_risk"]:
                    top = result.explanation["increasing_risk"][0]
                row.update({
                    "probability": result.probability,
                    "interval_low": result.interval_low,
                    "interval_high": result.interval_high,
                    "evidence_confidence": result.evidence_confidence,
                    "completeness": result.completeness["overall"],
                    "top_risk_driver": top["label"] if top else None,
                    "top_risk_driver_domain": top["domain_label"] if top else None,
                    "last_scored_at": result.computed_at,
                })
            except ScoringError:
                pass

        rows.append(PortfolioRow.model_validate(row))

    project = session.get(Project, project_id) if project_id else None
    endpoint_def = ENDPOINTS[endpoint]
    is_demo = any_synthetic or bool(artifact and artifact.trained_on_synthetic)

    return PortfolioResponse(
        project_id=project_id,
        project_name=project.name if project else None,
        model_version=artifact.version if artifact else None,
        endpoint=endpoint,
        estimand=endpoint_def.estimand,
        rows=rows,
        total=len(rows),
        is_demonstration=is_demo,
        demonstration_note=(
            "DEMONSTRATION PORTFOLIO — programs, evidence and outcomes are "
            "synthetic. Estimates shown have no scientific interpretation."
            if is_demo else None
        ),
        model_usefulness=(artifact.metrics or {}).get("usefulness", {}) if artifact else {},
    )


@router.get("/projects")
def list_projects(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(Project).where(
            Project.organization_id == principal.organization_id,
            Project.archived_at.is_(None),
        )
    ).all()
    return [
        {
            "id": p.id, "name": p.name, "slug": p.slug,
            "description": p.description, "is_private": p.is_private,
            "n_programs": session.scalar(
                select(func.count()).select_from(Program).where(Program.project_id == p.id)
            ) or 0,
        }
        for p in rows
    ]
