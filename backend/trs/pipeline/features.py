"""Materialising feature snapshots for a cohort.

Snapshots are persisted rather than recomputed on demand for three reasons:
they are what a historical prediction is reconstructed from; recomputation
after an evidence update would silently change an old score; and the
``(track, as_of, schema_version)`` uniqueness constraint makes accidental
divergence impossible to miss.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.cohort.outcomes import PRIMARY_ENDPOINT, EndpointDefinition
from trs.db.base import utcnow
from trs.db.models.analytics import CohortEntry, FeatureSnapshot
from trs.db.models.core import ProgramIndication
from trs.features.builder import BUILDER_VERSION, build_features
from trs.features.historical import HistoricalRateIndex, mechanism_key_for
from trs.features.missingness import compute_completeness
from trs.features.spec import FEATURE_SCHEMA_VERSION
from trs.features.temporal import AsOfContext


def build_history_index(
    session: Session, endpoint: EndpointDefinition = PRIMARY_ENDPOINT
) -> HistoricalRateIndex:
    """Index of resolved outcomes for the as-of historical-rate features."""
    return HistoricalRateIndex.from_session(
        session,
        failure_labels=frozenset(endpoint.positive_outcomes),
        success_labels=frozenset(endpoint.negative_outcomes),
        mechanism_key_fn=mechanism_key_for,
    )


def materialise_snapshot(
    session: Session,
    track_id: str,
    as_of: dt.date,
    *,
    history: HistoricalRateIndex | None = None,
    overwrite: bool = False,
) -> FeatureSnapshot:
    """Build and persist one feature snapshot, reusing an existing one if present."""
    existing = session.scalars(
        select(FeatureSnapshot).where(
            FeatureSnapshot.program_indication_id == track_id,
            FeatureSnapshot.as_of_date == as_of,
            FeatureSnapshot.feature_schema_version == FEATURE_SCHEMA_VERSION,
        )
    ).first()
    if existing is not None and not overwrite:
        return existing

    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise KeyError(f"No development track {track_id!r}")

    ctx = AsOfContext(session=session, as_of=as_of, strict=True)
    built = build_features(ctx, track, history=history)
    completeness = compute_completeness(built.missingness)

    target = existing or FeatureSnapshot(
        program_indication_id=track_id,
        as_of_date=as_of,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    target.features = built.values
    target.missingness = built.missingness
    target.completeness_score = completeness.overall
    target.domain_completeness = completeness.by_domain
    target.contributing_evidence_ids = {"ids": built.contributing_evidence_ids}
    target.feature_evidence_map = built.evidence_map
    target.n_evidence_items = built.n_evidence_items
    target.built_at = utcnow()
    target.builder_version = BUILDER_VERSION
    target.vector_hash = built.vector_hash()
    target.is_synthetic = built.contains_synthetic

    if existing is None:
        session.add(target)
    session.flush()
    return target


def materialise_cohort_snapshots(
    session: Session,
    entries: Sequence[CohortEntry],
    *,
    overwrite: bool = False,
    progress: bool = False,
) -> list[FeatureSnapshot]:
    """Build snapshots for every entry in a cohort, sharing one history index."""
    history = build_history_index(session)
    out: list[FeatureSnapshot] = []
    for i, entry in enumerate(entries, start=1):
        out.append(
            materialise_snapshot(
                session, entry.program_indication_id, entry.as_of_date,
                history=history, overwrite=overwrite,
            )
        )
        if progress and i % 25 == 0:
            print(f"  … {i}/{len(entries)} snapshots")
    return out


def snapshot_rows(snapshots: Sequence[FeatureSnapshot]) -> list[dict[str, Any]]:
    """Extract raw feature dicts in snapshot order, for the transformer."""
    return [dict(s.features) for s in snapshots]
