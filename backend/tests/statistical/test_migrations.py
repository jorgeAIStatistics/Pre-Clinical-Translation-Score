"""Migrations are tested against migrated schema, not model-created schema.

Why this file exists
--------------------
Every other test in this suite builds its database with
``Base.metadata.create_all``, which constructs tables from the *current* model
definitions. That is fast and right for behavioural tests, and it is blind to
an entire class of migration defect: anything the model no longer declares
cannot appear in the test schema, however stubbornly it survives in a real
database.

Migration ``b1c7e4f29a30`` added a five-column unique constraint to
``feature_snapshot`` and failed to drop the three-column constraint it
replaced. On a model-created schema there was no old constraint, so 286 tests
passed. Applying the migration to a copy of the real database exercised the DDL
and inserted nothing, so that passed too. The defect surfaced only when
training tried to write a second snapshot for a corrected programme and SQLite
rejected it — which is to say, at the first moment the feature was used.

So these tests run Alembic against a fresh file-backed database and then
*write rows*. A schema that cannot be written to is not a schema that works.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


@pytest.fixture
def migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fresh database with every migration applied, in order.

    ``trs/db/migrations/env.py`` sets ``sqlalchemy.url`` from
    ``trs.config.settings`` on every run, overriding whatever the caller put in
    the Alembic config — a deliberate design choice so that migrations and the
    application can never disagree about which database they are touching. It
    also means a test cannot select a database by configuring Alembic; it has
    to patch the setting. Getting this wrong points the migration at the
    developer's real database, which is exactly what happened the first time
    this fixture was written.
    """
    from trs.config import settings

    db = tmp_path / "migrated.db"
    url = f"sqlite+pysqlite:///{db}"
    monkeypatch.setattr(settings, "database_url", url, raising=True)

    root = Path(__file__).resolve().parents[2]  # backend/
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "trs" / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, "head")
    engine = create_engine(url, future=True)
    assert db.exists(), "migrations did not create the temporary database"
    try:
        yield engine, cfg
    finally:
        engine.dispose()
    

def test_migrations_apply_to_an_empty_database(migrated) -> None:
    """If this breaks, a fresh clone cannot create its schema."""
    engine, _cfg = migrated
    tables = set(inspect(engine).get_table_names())
    # 40 application tables plus Alembic's own bookkeeping.
    assert "feature_snapshot" in tables
    assert "model_artifact" in tables
    assert "alembic_version" in tables
    assert len(tables) >= 41, f"only {len(tables)} tables were created"


def _seed_track(session) -> str:
    """Minimal parent rows for a feature snapshot, built through the ORM.

    Using the ORM rather than hand-written INSERTs is deliberate: it checks
    that the mapped models and the *migrated* schema agree, which is the other
    half of the defect this file exists for. A column the model declares and
    the migration forgot fails here.
    """
    from trs.db.models.core import Indication, Program, ProgramIndication
    from trs.db.models.tenancy import Organization

    org = Organization(name="Test org", slug="test-org")
    session.add(org)
    session.flush()
    indication = Indication(name="Test indication", therapeutic_area="oncology")
    program = Program(name="Test programme", modality="small_molecule")
    session.add_all([indication, program])
    session.flush()
    track = ProgramIndication(
        program_id=program.id, indication_id=indication.id
    )
    session.add(track)
    session.flush()
    return track.id


def _snapshot(track_id: str, revision: str, **over):
    from trs.db.base import utcnow
    from trs.db.models.analytics import FeatureSnapshot

    kwargs = dict(
        program_indication_id=track_id,
        as_of_date=dt.date(2015, 6, 1),
        feature_schema_version="fs-1.0.0",
        endpoint="ph2_efficacy_failure",
        evidence_revision=revision,
        features={},
        missingness={},
        completeness_score=0.5,
        n_evidence_items=3,
        built_at=utcnow(),
        builder_version="fb-1.0.0",
    )
    kwargs.update(over)
    return FeatureSnapshot(**kwargs)


def test_snapshot_identity_admits_two_revisions_of_one_cutoff(migrated) -> None:
    """If this breaks, corrected evidence cannot be recorded at all.

    This is the test the superseded-constraint bug needed. Two snapshots for
    the same track, cutoff, schema and endpoint that differ only by evidence
    revision must both insert. Under the old three-column constraint the
    second is rejected, so every rebuild after an evidence correction fails
    with an IntegrityError — which is how the bug was eventually found, during
    a real retraining run rather than by any test.
    """
    from sqlalchemy.orm import sessionmaker

    import trs.db.models  # noqa: F401  (registers the mappers)

    engine, _cfg = migrated
    session = sessionmaker(bind=engine, future=True)()
    try:
        track_id = _seed_track(session)
        session.add(_snapshot(track_id, "revision-a"))
        session.flush()
        session.add(_snapshot(track_id, "revision-b"))
        session.flush()
        session.commit()

        from sqlalchemy import func, select

        from trs.db.models.analytics import FeatureSnapshot

        n = session.scalar(
            select(func.count()).select_from(FeatureSnapshot)
            .where(FeatureSnapshot.program_indication_id == track_id)
        )
        assert n == 2, (
            "the second evidence revision was not stored, so a corrected "
            "extraction cannot be recorded at this cutoff"
        )
    finally:
        session.close()


def test_duplicate_identity_is_still_rejected(migrated) -> None:
    """If this breaks, identity stops being identity.

    Dropping the superseded constraint must not have left the table with no
    uniqueness at all: two snapshots identical in every identity column are a
    bug, not a correction.
    """
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    import trs.db.models  # noqa: F401

    engine, _cfg = migrated
    session = sessionmaker(bind=engine, future=True)()
    try:
        track_id = _seed_track(session)
        session.add(_snapshot(track_id, "same-revision"))
        session.flush()
        session.add(_snapshot(track_id, "same-revision"))
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.rollback()
        session.close()


def test_superseded_constraint_is_gone(migrated) -> None:
    """If this breaks, the replaced constraint is still enforcing the old rule."""
    engine, _cfg = migrated
    names = {
        c["name"]
        for c in inspect(engine).get_unique_constraints("feature_snapshot")
        if c.get("name")
    }
    assert "uq_feature_snapshot_identity" in names
    assert (
        "uq_feature_snapshot_program_indication_id_as_of_date_feature_schema_version"
        not in names
    ), "the superseded three-column unique constraint survived the migration"


def test_downgrade_restores_the_previous_schema(migrated) -> None:
    """If this breaks, the migration is one-way and cannot be backed out."""
    engine, cfg = migrated
    command.downgrade(cfg, "-1")

    insp = inspect(engine)
    columns = {c["name"] for c in insp.get_columns("feature_snapshot")}
    assert "evidence_revision" not in columns
    assert "endpoint" not in columns
    assert "superseded_by_id" not in columns

    names = {
        c["name"] for c in insp.get_unique_constraints("feature_snapshot")
        if c.get("name")
    }
    assert (
        "uq_feature_snapshot_program_indication_id_as_of_date_feature_schema_version"
        in names
    ), "the downgrade dropped the new constraint without restoring the old one"
    assert "uq_feature_snapshot_identity" not in names

    command.upgrade(cfg, "head")
