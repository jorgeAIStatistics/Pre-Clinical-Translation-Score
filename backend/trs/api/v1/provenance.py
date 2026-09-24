"""Provenance endpoints: where every fact came from.

The chain a user can walk: score → feature snapshot → contributing evidence →
source → raw connector payload. Each link is addressable, which is what makes
"trace any number to its origin" a capability rather than a claim.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.api.deps import get_db, get_principal
from trs.api.schemas import ConnectorHealthOut
from trs.api.security import Principal
from trs.config import settings
from trs.db.base import utcnow
from trs.db.models.analytics import ConnectorRun, FeatureSnapshot, ScoreRecord
from trs.db.models.evidence import EvidenceItem, RawRecord, Source
from trs.features.registry import FEATURES_BY_NAME

router = APIRouter(prefix="/provenance", tags=["provenance"])


@router.get("/sources/{source_id}")
def get_source(
    source_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    source = session.get(Source, source_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found.")
    raw = session.get(RawRecord, source.raw_record_id) if source.raw_record_id else None
    return {
        "source": {c.name: getattr(source, c.name) for c in Source.__table__.columns},
        "raw_record": (
            {
                "id": raw.id, "connector": raw.connector,
                "connector_version": raw.connector_version,
                "endpoint": raw.endpoint, "http_status": raw.http_status,
                "content_hash": raw.content_hash,
                "fetched_at": raw.fetched_at.isoformat(),
                "from_cache": raw.from_cache,
            }
            if raw else None
        ),
        "note": (
            "The raw record is the verbatim payload the connector received, "
            "retained so any derived value can be re-checked against exactly "
            "what produced it."
            if raw else
            "No raw payload is attached. Synthetic demonstration records and "
            "hand-curated entries have no upstream API response."
        ),
    }


@router.get("/evidence/{evidence_id}")
def get_evidence_provenance(
    evidence_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    item = session.get(EvidenceItem, evidence_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found.")
    source = session.get(Source, item.source_id) if item.source_id else None
    return {
        "evidence_id": item.id,
        "available_at": item.available_at.isoformat(),
        "observed_on": item.observed_on.isoformat() if item.observed_on else None,
        "extraction_date": item.extraction_date.isoformat() if item.extraction_date else None,
        "provenance_method": item.provenance_method,
        "extractor_version": item.extractor_version,
        "extraction_confidence": item.extraction_confidence,
        "source_snippet": item.source_snippet,
        "source_locator": item.source_locator,
        "field_missingness": item.field_missingness,
        "is_synthetic": item.is_synthetic,
        "synthetic_note": item.synthetic_note,
        "source": (
            {c.name: getattr(source, c.name) for c in Source.__table__.columns}
            if source else None
        ),
        "field_provenance": [
            {
                "field_name": fp.field_name, "value_repr": fp.value_repr,
                "method": fp.method, "confidence": fp.confidence,
                "snippet": fp.snippet,
                "extracted_at": fp.extracted_at.isoformat(),
                "extracted_by": fp.extracted_by,
            }
            for fp in item.field_provenance
        ],
    }


@router.get("/scores/{score_id}/lineage")
def get_score_lineage(
    score_id: str,
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    """The full reconstruction chain for one persisted score."""
    record = session.get(ScoreRecord, score_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Score not found.")
    snapshot = (
        session.get(FeatureSnapshot, record.feature_snapshot_id)
        if record.feature_snapshot_id else None
    )
    evidence_ids = (snapshot.contributing_evidence_ids or {}).get("ids", []) if snapshot else []
    evidence = list(session.scalars(
        select(EvidenceItem).where(EvidenceItem.id.in_(evidence_ids or [""]))
    ))
    return {
        "score": {
            "id": record.id, "probability": record.probability,
            "interval": [record.interval_low, record.interval_high],
            "interval_level": record.interval_level,
            "model_version": record.model_version,
            "as_of_date": record.as_of_date.isoformat(),
            "computed_at": record.computed_at.isoformat(),
            "evidence_confidence": record.evidence_confidence,
            "is_demonstration": record.is_demonstration,
        },
        "feature_snapshot": (
            {
                "id": snapshot.id,
                "feature_schema_version": snapshot.feature_schema_version,
                "builder_version": snapshot.builder_version,
                "vector_hash": snapshot.vector_hash,
                "built_at": snapshot.built_at.isoformat(),
                "completeness_score": snapshot.completeness_score,
                "n_evidence_items": snapshot.n_evidence_items,
                "features": snapshot.features,
                "missingness": snapshot.missingness,
                "feature_evidence_map": snapshot.feature_evidence_map,
            }
            if snapshot else None
        ),
        "contributing_evidence": [
            {
                "id": e.id, "title": e.title, "evidence_class": e.evidence_class,
                "direction": e.direction, "species": e.species,
                "available_at": e.available_at.isoformat(),
                "source_id": e.source_id, "is_synthetic": e.is_synthetic,
            }
            for e in evidence
        ],
        "chain": [
            "score_record", "feature_snapshot", "evidence_item", "source", "raw_record",
        ],
        "note": (
            "Every link is persisted, so this score can be reconstructed after "
            "the model has been retrained and the evidence base has moved on."
        ),
    }


@router.get("/features/{feature_name}")
def get_feature_provenance(
    feature_name: str,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    spec = FEATURES_BY_NAME.get(feature_name)
    if spec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown feature.")
    from trs.features.spec import DOMAIN_LABELS

    return {
        "name": spec.name, "label": spec.label,
        "domain": str(spec.domain), "domain_label": DOMAIN_LABELS[spec.domain],
        "description": spec.description, "derivation": spec.derivation,
        "kind": str(spec.kind), "transform": str(spec.transform),
        "missing_policy": str(spec.missing_policy),
        "expected_direction": str(spec.expected_direction),
        "valid_min": spec.valid_min, "valid_max": spec.valid_max,
        "unit": spec.unit, "completeness_weight": spec.completeness_weight,
        "scenario_adjustable": spec.scenario_adjustable,
    }


#: Per-probe ceiling for a health check, independent of the ingestion timeout.
#:
#: Ingestion can afford to wait 30s for a slow but working endpoint; a status
#: page cannot. Probed serially at the ingestion timeout, six connectors could
#: block a page load for three minutes — and in practice held it for nine
#: seconds even when everything was healthy.
HEALTH_PROBE_TIMEOUT_SECONDS = 5.0

#: Wall-clock ceiling for the whole probe set. Probes run concurrently, so this
#: is the page's worst case regardless of how many connectors are registered.
HEALTH_PROBE_DEADLINE_SECONDS = 8.0


@router.get("/connectors", response_model=list[ConnectorHealthOut])
def get_connector_health(
    principal: Principal = Depends(get_principal),
) -> list[ConnectorHealthOut]:
    """Live reachability of every ingestion source, probed concurrently.

    Each connector is probed in its own thread with a short timeout. The probes
    are independent network calls, so total latency is the slowest single
    source rather than their sum, and one hanging endpoint cannot hold the page
    hostage — it reports ``unreachable`` and the rest still render.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FuturesTimeout

    from trs.connectors.base import available_connectors, get_connector

    names = available_connectors()

    def probe(name: str) -> ConnectorHealthOut:
        try:
            import httpx

            client = httpx.Client(
                timeout=HEALTH_PROBE_TIMEOUT_SECONDS,
                headers={"User-Agent": settings.connector_user_agent, "Accept": "application/json"},
                follow_redirects=True,
            )
            try:
                with get_connector(name, client=client, max_retries=1) as c:
                    return ConnectorHealthOut.model_validate(c.health().to_dict())
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            return ConnectorHealthOut(
                connector=name, status="unreachable",
                checked_at=utcnow().isoformat(), detail=str(exc)[:300],
            )

    results: dict[str, ConnectorHealthOut] = {}
    # A hard wall-clock deadline on the whole set. `as_completed` without a
    # timeout waits for every future, so a single hanging endpoint would still
    # decide the response time no matter how short each individual timeout is.
    pool = ThreadPoolExecutor(max_workers=min(8, len(names) or 1))
    try:
        futures = {pool.submit(probe, n): n for n in names}
        try:
            for fut in as_completed(futures, timeout=HEALTH_PROBE_DEADLINE_SECONDS):
                name = futures[fut]
                try:
                    results[name] = fut.result(timeout=0)
                except Exception as exc:  # noqa: BLE001
                    results[name] = ConnectorHealthOut(
                        connector=name, status="unreachable",
                        checked_at=utcnow().isoformat(),
                        detail=f"Health probe failed: {type(exc).__name__}",
                    )
        except FuturesTimeout:
            pass  # whatever finished is reported; the rest are marked below
        for name in names:
            if name not in results:
                results[name] = ConnectorHealthOut(
                    connector=name, status="unreachable",
                    checked_at=utcnow().isoformat(),
                    detail=(
                        f"Did not respond within the {HEALTH_PROBE_DEADLINE_SECONDS:.0f}s "
                        "health-check deadline. This reflects the probe, not necessarily "
                        "the source: ingestion uses a longer timeout and retries."
                    ),
                )
    finally:
        # Do not block the response on stragglers; they are already reported
        # as unreachable and their threads will finish on their own.
        pool.shutdown(wait=False, cancel_futures=True)

    # Stable ordering, independent of which probe finished first.
    return [results[n] for n in names]


@router.get("/connectors/catalogue")
def get_connector_catalogue(principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    from trs.connectors.base import connector_catalogue

    return {
        "connectors": connector_catalogue(),
        "note": (
            "Each connector retains the verbatim payload it receives before any "
            "interpretation, so every derived value remains checkable against "
            "its source."
        ),
    }


@router.get("/ingestion-runs")
def get_ingestion_runs(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    limit: int = 50,
) -> list[dict[str, Any]]:
    return [
        {
            "id": r.id, "connector": r.connector, "operation": r.operation,
            "status": r.status, "started_at": r.started_at.isoformat(),
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "n_requests": r.n_requests, "n_records_raw": r.n_records_raw,
            "n_records_normalised": r.n_records_normalised, "n_errors": r.n_errors,
        }
        for r in session.scalars(
            select(ConnectorRun).order_by(ConnectorRun.started_at.desc()).limit(limit)
        )
    ]
