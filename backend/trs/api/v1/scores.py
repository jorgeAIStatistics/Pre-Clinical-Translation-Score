"""Scoring, scenario and comparator endpoints."""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from trs.api.deps import get_db, get_principal, require_analyst
from trs.api.schemas import (
    ComparatorOut,
    ComparatorResponse,
    ScenarioComparison,
    ScenarioRequest,
    ScoreOut,
    ScoreRequest,
)
from trs.api.security import Principal, record_audit
from trs.db.models.analytics import Scenario
from trs.db.models.core import ProgramIndication
from trs.scoring.comparators import find_comparators, summarise_comparators
from trs.scoring.scenarios import (
    ScenarioValidationError,
    lever_catalogue,
)
from trs.scoring.service import ScoringError, ScoringService

router = APIRouter(tags=["scoring"])


def _service(session: Session) -> ScoringService:
    return ScoringService(session)


@router.post("/programs/{track_id}/score", response_model=ScoreOut)
def score_program(
    track_id: str,
    payload: ScoreRequest | None = None,
    request: Request = None,  # type: ignore[assignment]
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> ScoreOut:
    """Compute a calibrated translational-risk estimate for one track.

    The response always carries the interval, the decomposition of its width,
    the evidence-confidence grade and the model version — a bare probability is
    never returned, because a bare probability cannot be acted on responsibly.
    """
    payload = payload or ScoreRequest()
    try:
        result = _service(session).score_track(
            track_id,
            as_of=payload.as_of,
            model_version=payload.model_version,
            interval_level=payload.interval_level,
            persist=payload.persist,
            demo_context=payload.demo_context,
            user_id=principal.user_id,
            request_id=request.headers.get("x-request-id") if request else None,
        )
    except ScoringError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    record_audit(
        session, principal=principal, action="score.compute",
        resource_type="program_indication", resource_id=track_id, request=request,
        detail={
            "model_version": result.model_version,
            "as_of": result.as_of_date.isoformat(),
            "probability": result.probability,
            "score_record_id": result.score_record_id,
            "model_status": result.eligibility.get("model_status"),
            "score_permitted": result.eligibility.get("permitted"),
            "refusal_code": result.eligibility.get("refusal_code"),
            "demo_context_requested": payload.demo_context,
        },
    )
    return ScoreOut.model_validate(result.to_dict())


@router.get("/programs/{track_id}/score", response_model=ScoreOut)
def get_score(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = None,
    demo_context: bool = False,
) -> ScoreOut:
    """Convenience read-only scoring, for the UI's initial page load.

    ``demo_context`` carries the same meaning as on the POST form: without it,
    a model trained on synthetic data returns a refusal rather than a number.
    """
    try:
        result = _service(session).score_track(
            track_id, as_of=as_of, persist=False, demo_context=demo_context,
        )
    except ScoringError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ScoreOut.model_validate(result.to_dict())


@router.get("/scenarios/levers")
def get_levers(principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """The catalogue of named scenario levers."""
    return {
        "levers": lever_catalogue(),
        "note": (
            "A lever describes an evidence state a team could bring about. "
            "Applying one recomputes the model under that state. It is a "
            "sensitivity analysis of the estimate, not a prediction that the "
            "experiment would succeed."
        ),
    }


@router.post("/programs/{track_id}/scenarios", response_model=ScenarioComparison)
def run_scenario(
    track_id: str,
    payload: ScenarioRequest,
    request: Request = None,  # type: ignore[assignment]
    session: Session = Depends(get_db),
    principal: Principal = Depends(require_analyst),
) -> ScenarioComparison:
    """Recompute the estimate under a hypothetical evidence state."""
    if not payload.lever_keys and not payload.overrides:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Supply at least one lever key or feature override.",
        )
    service = _service(session)
    try:
        current = service.score_track(
            track_id, as_of=payload.as_of, persist=False,
            demo_context=payload.demo_context,
        )
        scenario = service.score_track(
            track_id,
            as_of=payload.as_of,
            lever_keys=payload.lever_keys,
            overrides=payload.overrides,
            persist=False,
            demo_context=payload.demo_context,
        )
    except ScenarioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ScoringError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if payload.persist_scenario:
        session.add(Scenario(
            organization_id=principal.organization_id,
            program_indication_id=track_id,
            name=payload.name or "Untitled scenario",
            overrides=(scenario.scenario or {}).get("overrides", {}),
            lever_keys={"keys": payload.lever_keys},
            created_by_user_id=principal.user_id,
        ))
        session.flush()

    record_audit(
        session, principal=principal, action="scenario.run",
        resource_type="program_indication", resource_id=track_id, request=request,
        detail={
            "levers": payload.lever_keys,
            "current": current.probability,
            "scenario": scenario.probability,
        },
    )

    # A scenario is a comparison of two numbers. When the eligibility policy
    # declines to produce them, there is no comparison to return — and
    # returning zeros, or nulls inside a shape that promises a delta, would
    # invite a client to render "no change" where the truth is "no estimate".
    if current.probability is None or scenario.probability is None:
        refusal = (
            current.eligibility
            if current.probability is None
            else scenario.eligibility
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "scenario_unavailable",
                "reason": refusal.get("reason"),
                "refusal_code": refusal.get("refusal_code"),
                "model_status": refusal.get("model_status"),
                "note": (
                    "Scenario analysis compares model output under two "
                    "evidence states. No score may be produced for this "
                    "programme under this model, so there is nothing to "
                    "compare. Evidence review, gaps and provenance remain "
                    "available."
                ),
            },
        )

    return ScenarioComparison(
        current=ScoreOut.model_validate(current.to_dict()),
        scenario=ScoreOut.model_validate(scenario.to_dict()),
        delta_probability=scenario.probability - current.probability,
        delta_interval_width=(
            (scenario.interval_high - scenario.interval_low)
            - (current.interval_high - current.interval_low)
        ),
        changes=(scenario.scenario or {}).get("changes", []),
        disclaimer=(
            "This compares the model's output under two evidence states. It "
            "does not predict the result of the experiment that would produce "
            "the second state, and it is not a forecast that the risk will "
            "fall if the work is commissioned."
        ),
    )


@router.get("/programs/{track_id}/comparators", response_model=ComparatorResponse)
def get_comparators(
    track_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    as_of: dt.date | None = None,
    limit: int = Query(default=10, le=50),
    min_similarity: float = Query(default=0.20, ge=0.0, le=1.0),
) -> ComparatorResponse:
    """Historical programs that made a structurally similar bet."""
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found.")
    cutoff = as_of or track.ind_or_cta_on or track.first_in_human_on or dt.date.today()

    comparators = find_comparators(
        session, track, as_of=cutoff, limit=limit, min_similarity=min_similarity
    )
    return ComparatorResponse(
        track_id=track_id,
        as_of_date=cutoff,
        comparators=[ComparatorOut.model_validate(c.to_dict()) for c in comparators],
        summary=summarise_comparators(comparators),
    )
