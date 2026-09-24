"""Shared pytest fixtures.

Tests run entirely offline against an in-memory database and recorded connector
payloads. No test touches the network: a suite that needs the internet to pass
is a suite that fails when the internet does, and CI would then teach the team
to ignore it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TRS_ENV", "test")
os.environ.setdefault("TRS_DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("TRS_ALLOW_LIVE_NETWORK", "false")

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONNECTOR_FIXTURES = FIXTURE_DIR / "connectors"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


def load_connector_fixture(name: str) -> dict:
    """Load a recorded connector payload, stripping the provenance marker."""
    payload = json.loads((CONNECTOR_FIXTURES / name).read_text())
    if isinstance(payload, dict):
        payload.pop("_fixture_note", None)
    return payload


@pytest.fixture
def db_session():
    """A fresh in-memory database with the full schema, per test."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    import trs.db.models  # noqa: F401
    from trs.db.base import Base

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN001
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def seeded_session(db_session):
    """A small seeded universe — enough structure, small enough to stay fast."""
    from trs.seed.generator import DemoDataGenerator

    gen = DemoDataGenerator(db_session, seed=4242)
    result = gen.run(n_programs=60)
    db_session.commit()
    return db_session, result


@pytest.fixture
def today() -> dt.date:
    return dt.date(2026, 1, 1)
