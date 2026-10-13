"""Snapshot identity: evidence revision, endpoint and supersession lineage.

Revision ID: b1c7e4f29a30
Revises: 05c2a99460a1

Why this migration exists
-------------------------
``feature_snapshot`` was unique on ``(program_indication_id, as_of_date,
feature_schema_version)``, and the builder offered an ``overwrite`` flag that
rewrote the feature values of the matching row. A score already referencing
that snapshot then pointed at numbers that had silently changed — so a
historical assessment could not be reproduced, which is the one property the
snapshot table exists to provide.

Identity now includes the evidence revision and the endpoint. Corrected
evidence produces a new snapshot at the same scientific cutoff, the previous
row is marked superseded without being altered, and the old score remains
reproducible from it.

Backfill policy
---------------
Existing rows get ``evidence_revision = 'legacy-unversioned'`` and
``endpoint = 'ph2_efficacy_failure'``.

The endpoint value is correct rather than assumed: the predecessor's
``build_history_index`` defaulted to ``PRIMARY_ENDPOINT`` and
``materialise_cohort_snapshots`` took no endpoint argument, so every snapshot
ever written used the primary endpoint's labels.

The revision sentinel is deliberately *not* a computed hash. Back-filling one
would imply the evidence behind those rows had been checked at the time, which
it was not. An unknown provenance recorded as unknown is more useful than a
plausible-looking value.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b1c7e4f29a30"
down_revision = "05c2a99460a1"
branch_labels = None
depends_on = None

LEGACY_REVISION = "legacy-unversioned"
LEGACY_ENDPOINT = "ph2_efficacy_failure"


def upgrade() -> None:
    with op.batch_alter_table("feature_snapshot") as batch:
        batch.add_column(
            sa.Column(
                "evidence_revision",
                sa.String(length=64),
                nullable=False,
                server_default=LEGACY_REVISION,
            )
        )
        batch.add_column(
            sa.Column(
                "endpoint",
                sa.String(length=64),
                nullable=False,
                server_default=LEGACY_ENDPOINT,
            )
        )
        batch.add_column(
            sa.Column("superseded_by_id", sa.String(length=32), nullable=True)
        )

    # The old uniqueness is now too weak: it would forbid exactly the second
    # snapshot that corrected evidence is supposed to create.
    with op.batch_alter_table("feature_snapshot") as batch:
        batch.create_index(
            "ix_feature_snapshot_evidence_revision", ["evidence_revision"]
        )
        batch.create_index("ix_feature_snapshot_endpoint", ["endpoint"])
        batch.create_index(
            "ix_feature_snapshot_revision",
            ["program_indication_id", "as_of_date", "endpoint", "evidence_revision"],
        )
        batch.create_unique_constraint(
            "uq_feature_snapshot_identity",
            [
                "program_indication_id",
                "as_of_date",
                "feature_schema_version",
                "endpoint",
                "evidence_revision",
            ],
        )


def downgrade() -> None:
    # Downgrading loses the ability to distinguish two snapshots that differ
    # only by evidence revision. Any such pair would violate the restored
    # constraint, so the downgrade fails loudly rather than deleting one.
    conn = op.get_bind()
    duplicates = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM ("
            "  SELECT program_indication_id, as_of_date, feature_schema_version"
            "  FROM feature_snapshot"
            "  GROUP BY program_indication_id, as_of_date, feature_schema_version"
            "  HAVING COUNT(*) > 1"
            ")"
        )
    ).scalar_one()
    if duplicates:
        raise RuntimeError(
            f"{duplicates} (track, cutoff, schema) group(s) hold more than one "
            "snapshot revision. Downgrading would require discarding "
            "snapshots that existing scores may reference. Resolve by hand."
        )

    with op.batch_alter_table("feature_snapshot") as batch:
        batch.drop_constraint("uq_feature_snapshot_identity", type_="unique")
        batch.drop_index("ix_feature_snapshot_revision")
        batch.drop_index("ix_feature_snapshot_endpoint")
        batch.drop_index("ix_feature_snapshot_evidence_revision")
        batch.drop_column("superseded_by_id")
        batch.drop_column("endpoint")
        batch.drop_column("evidence_revision")
