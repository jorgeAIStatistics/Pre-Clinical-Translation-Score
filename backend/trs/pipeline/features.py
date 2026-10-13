"""Materialising feature snapshots for a cohort.

Snapshots are persisted rather than recomputed on demand because they are what
a historical prediction is reconstructed from. That only works if they are
immutable.

What was wrong before
---------------------
Identity was ``(track, as_of, feature_schema_version)``, and
:func:`materialise_snapshot` took an ``overwrite`` flag that rewrote
``features``, ``missingness``, ``vector_hash`` and ``built_at`` on the existing
row. Any :class:`~trs.db.models.analytics.ScoreRecord` already pointing at that
snapshot then referenced numbers that had silently changed — so the old
assessment became unreproducible, which is precisely the failure the table
exists to prevent. The module docstring claimed the uniqueness constraint made
"accidental divergence impossible to miss"; overwrite made it impossible to
notice.

Identity now includes the evidence revision and the endpoint:

**Evidence revision.** A SHA-256 over the content identity of every evidence
row that contributed — its id, its last modification time and its availability
date. Correcting an item, retracting one, or adding one that changes an
aggregate all change the revision, so the rebuild creates a *new* snapshot at
the same scientific cutoff and marks the previous one superseded. The previous
row's values are never touched.

Because the revision is computed from *contributing* evidence rather than from
all eligible evidence, a new item that changes nothing produces no new
snapshot. That is the correct behaviour: if an item does not enter any
aggregate, the vector is identical and a second identical row would be noise.

**Endpoint.** The historical-rate features are built from resolved outcomes
under a specific endpoint definition. ``build_history_index`` defaulted to
``PRIMARY_ENDPOINT`` and ``materialise_cohort_snapshots`` took no endpoint
argument at all, so training any of the other five declared endpoints silently
used the primary endpoint's labels as predictors. The endpoint is now threaded
through and is part of snapshot identity.

Known limitation, stated rather than hidden
-------------------------------------------
The revision covers the focal track's own contributing evidence. The
historical-rate features also depend on *other* programmes' resolved outcomes
as of the cutoff; a correction to one of those does not change this track's
revision. Those features are already flagged in the model artifact's
limitations for a related reason, and the dependency is recorded on each
snapshot's provenance under ``history_index_endpoint``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.cohort.outcomes import PRIMARY_ENDPOINT, EndpointDefinition
from trs.db.base import utcnow
from trs.db.models.analytics import CohortEntry, FeatureSnapshot, ScoreRecord
from trs.db.models.core import ProgramIndication
from trs.db.models.evidence import EvidenceItem
from trs.features.builder import BUILDER_VERSION, build_features
from trs.features.historical import HistoricalRateIndex, mechanism_key_for
from trs.features.missingness import compute_completeness
from trs.features.spec import FEATURE_SCHEMA_VERSION
from trs.features.temporal import AsOfContext

#: Recorded on rows written before snapshots carried an evidence revision.
#: Deliberately not a computed hash: back-filling one would imply the evidence
#: behind those rows had been checked, which it had not.
LEGACY_REVISION = "legacy-unversioned"


class SnapshotImmutabilityError(RuntimeError):
    """Raised on an attempt to alter a snapshot a score already references."""


def build_history_index(
    session: Session, endpoint: EndpointDefinition = PRIMARY_ENDPOINT
) -> HistoricalRateIndex:
    """Index of resolved outcomes for the as-of historical-rate features.

    The endpoint decides which observed outcomes count as failures and which as
    successes, so an index built for one endpoint is not reusable for another.
    """
    return HistoricalRateIndex.from_session(
        session,
        failure_labels=frozenset(endpoint.positive_outcomes),
        success_labels=frozenset(endpoint.negative_outcomes),
        mechanism_key_fn=mechanism_key_for,
    )


def evidence_revision(
    session: Session,
    evidence_ids: Sequence[str],
    *,
    endpoint: EndpointDefinition,
) -> str:
    """Content identity of the evidence behind one snapshot.

    Hashes each contributing item's ``(id, updated_at, available_at)`` — so a
    corrected extraction changes the revision even when the item's id does not
    — together with the endpoint, the builder version and the schema version,
    each of which changes what the same evidence produces.
    """
    h = hashlib.sha256()
    h.update(f"endpoint={endpoint.name}\n".encode())
    h.update(f"builder={BUILDER_VERSION}\n".encode())
    h.update(f"schema={FEATURE_SCHEMA_VERSION}\n".encode())

    ids = sorted({str(e) for e in evidence_ids if e})
    if ids:
        rows = session.execute(
            select(
                EvidenceItem.id, EvidenceItem.updated_at, EvidenceItem.available_at
            ).where(EvidenceItem.id.in_(ids))
        ).all()
        found = {r[0]: r for r in rows}
        for eid in ids:
            row = found.get(eid)
            if row is None:
                # A contributing id with no row is itself a state worth
                # hashing: it means evidence was deleted under a snapshot.
                h.update(f"{eid}|MISSING\n".encode())
                continue
            _, updated_at, available_at = row
            h.update(
                f"{eid}|{updated_at.isoformat() if updated_at else ''}"
                f"|{available_at.isoformat() if available_at else ''}\n".encode()
            )
    else:
        h.update(b"no-contributing-evidence\n")
    return h.hexdigest()


def _is_referenced(session: Session, snapshot_id: str) -> bool:
    return (
        session.scalars(
            select(ScoreRecord.id)
            .where(ScoreRecord.feature_snapshot_id == snapshot_id)
            .limit(1)
        ).first()
        is not None
    )


def materialise_snapshot(
    session: Session,
    track_id: str,
    as_of: dt.date,
    *,
    endpoint: EndpointDefinition = PRIMARY_ENDPOINT,
    history: HistoricalRateIndex | None = None,
    rebuild: bool = False,
) -> FeatureSnapshot:
    """Build and persist one feature snapshot, reusing an identical one.

    Returns an existing row when the evidence revision matches, because nothing
    about the vector can have changed. When the revision differs, a **new** row
    is created and the previous one is marked superseded; its stored values are
    not modified, so any score referencing it stays reproducible.

    ``rebuild=True`` forces the features to be recomputed rather than trusting
    a revision match. It still cannot mutate an existing row: if the recomputed
    vector disagrees with a stored snapshot that a score references, that is a
    builder-version bug and :class:`SnapshotImmutabilityError` is raised rather
    than quietly reconciled.
    """
    track = session.get(ProgramIndication, track_id)
    if track is None:
        raise KeyError(f"No development track {track_id!r}")

    ctx = AsOfContext(session=session, as_of=as_of, strict=True)
    built = build_features(ctx, track, history=history)
    revision = evidence_revision(
        session, built.contributing_evidence_ids, endpoint=endpoint
    )
    vector_hash = built.vector_hash()

    siblings = list(
        session.scalars(
            select(FeatureSnapshot).where(
                FeatureSnapshot.program_indication_id == track_id,
                FeatureSnapshot.as_of_date == as_of,
                FeatureSnapshot.feature_schema_version == FEATURE_SCHEMA_VERSION,
                FeatureSnapshot.endpoint == endpoint.name,
            )
        )
    )
    exact = next((s for s in siblings if s.evidence_revision == revision), None)
    if exact is not None:
        if rebuild and exact.vector_hash != vector_hash:
            referenced = _is_referenced(session, exact.id)
            raise SnapshotImmutabilityError(
                f"Snapshot {exact.id} has evidence revision {revision} but its "
                f"stored vector hash {exact.vector_hash} does not match the "
                f"freshly built {vector_hash}. Identical evidence produced a "
                "different vector, which means the builder changed without "
                f"BUILDER_VERSION changing (currently {BUILDER_VERSION!r}). "
                "Bump the builder version rather than overwriting history."
                + (
                    f" This snapshot is referenced by at least one score "
                    f"record, so overwriting it would retroactively change a "
                    f"delivered assessment."
                    if referenced
                    else " No score references this snapshot yet, but the "
                    "builder discrepancy is still a defect."
                )
            )
        return exact

    # A legacy row has an unknown revision, so it can neither be matched nor
    # trusted to differ. Supersede it: the new row records what was actually
    # built, and the old one keeps whatever a past score read from it.
    completeness = compute_completeness(built.missingness)
    snapshot = FeatureSnapshot(
        program_indication_id=track_id,
        as_of_date=as_of,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        endpoint=endpoint.name,
        evidence_revision=revision,
        features=built.values,
        missingness=built.missingness,
        completeness_score=completeness.overall,
        domain_completeness=completeness.by_domain,
        contributing_evidence_ids={
            "ids": built.contributing_evidence_ids,
            "history_index_endpoint": endpoint.name,
        },
        feature_evidence_map=built.evidence_map,
        n_evidence_items=built.n_evidence_items,
        built_at=utcnow(),
        builder_version=BUILDER_VERSION,
        vector_hash=vector_hash,
        is_synthetic=built.contains_synthetic,
    )
    session.add(snapshot)
    session.flush()

    for old in siblings:
        if old.superseded_by_id is None:
            old.superseded_by_id = snapshot.id
    session.flush()
    return snapshot


def supersede_chain(session: Session, snapshot_id: str) -> list[FeatureSnapshot]:
    """Walk the supersession pointers forward from one snapshot.

    Lets a reviewer reading an old score see what the current evidence would
    produce, without the old score's inputs having moved.
    """
    chain: list[FeatureSnapshot] = []
    seen: set[str] = set()
    current = session.get(FeatureSnapshot, snapshot_id)
    while current is not None and current.id not in seen:
        chain.append(current)
        seen.add(current.id)
        current = (
            session.get(FeatureSnapshot, current.superseded_by_id)
            if current.superseded_by_id else None
        )
    return chain


def materialise_cohort_snapshots(
    session: Session,
    entries: Sequence[CohortEntry],
    *,
    endpoint: EndpointDefinition = PRIMARY_ENDPOINT,
    rebuild: bool = False,
    progress: bool = False,
) -> list[FeatureSnapshot]:
    """Build snapshots for every entry in a cohort, sharing one history index.

    The endpoint is threaded through rather than defaulted per call, so a
    cohort for a secondary endpoint cannot be built against the primary
    endpoint's historical labels.
    """
    history = build_history_index(session, endpoint)
    out: list[FeatureSnapshot] = []
    for i, entry in enumerate(entries, start=1):
        out.append(
            materialise_snapshot(
                session, entry.program_indication_id, entry.as_of_date,
                endpoint=endpoint, history=history, rebuild=rebuild,
            )
        )
        if progress and i % 25 == 0:
            print(f"  … {i}/{len(entries)} snapshots")
    return out


def snapshot_rows(snapshots: Sequence[FeatureSnapshot]) -> list[dict[str, Any]]:
    """Extract raw feature dicts in snapshot order, for the transformer."""
    return [dict(s.features) for s in snapshots]
