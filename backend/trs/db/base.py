"""Declarative base, portable column types and shared mixins.

Portability
-----------
The production target is PostgreSQL 15+; local development runs on SQLite so
the platform has no external service dependency. To keep one set of models and
one migration history working on both:

* Primary keys are ULID-like 26-character strings, not native ``UUID``.
  They sort lexicographically by creation time, which gives stable, index-
  friendly ordering on both engines.
* Structured payloads use :class:`JSONVariant`, which compiles to ``JSONB`` on
  PostgreSQL and ``JSON`` (TEXT-backed) on SQLite.
* Timestamps are always timezone-aware UTC. SQLite does not enforce this, so
  :func:`utcnow` is the single source of "now" and the ORM never relies on
  database-side ``CURRENT_TIMESTAMP``.
* No engine-specific DDL (partial indexes, array columns, ``ENUM`` types).
  Controlled vocabularies are stored as ``VARCHAR`` and validated in Pydantic
  and by CHECK-free application-level enforcement, so adding a vocabulary term
  does not require a database migration on either engine.
"""

from __future__ import annotations

import datetime as dt
import os
import secrets
from typing import Any

from sqlalchemy import DateTime, MetaData, String, TypeDecorator
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

# Deterministic constraint naming makes Alembic autogenerate stable and lets
# migrations drop constraints by name on both engines.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_N_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id(prefix: str = "") -> str:
    """Generate a lexicographically time-sortable 26-char identifier.

    This is a ULID: 48 bits of millisecond timestamp followed by 80 bits of
    randomness, Crockford base-32 encoded. Time-sortable primary keys keep
    B-tree inserts append-only and make ``ORDER BY id`` a valid proxy for
    creation order without an extra index.
    """
    ts = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ts << 80) | rand
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    ulid = "".join(reversed(out))
    return f"{prefix}{ulid}" if prefix else ulid


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def utcnow() -> dt.datetime:
    """Timezone-aware UTC now. The only clock the ORM layer uses."""
    return dt.datetime.now(dt.UTC)


class JSONVariant(TypeDecorator):
    """``JSONB`` on PostgreSQL, portable ``JSON`` elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):  # noqa: ANN001, ANN201
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.JSONB())
        return dialect.type_descriptor(JSON())


class UTCDateTime(TypeDecorator):
    """Timezone-aware datetime that round-trips correctly on SQLite.

    SQLite stores naive strings. Without this decorator a value written as
    aware UTC comes back naive, which silently breaks every ``available_at``
    comparison in the temporal-integrity layer — the single most important
    correctness property in this system.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):  # noqa: ANN001, ANN201
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: dt.datetime | None, dialect):  # noqa: ANN001, ANN201
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "Naive datetime written to a UTCDateTime column. All timestamps "
                "must be timezone-aware; use trs.db.base.utcnow()."
            )
        return value.astimezone(dt.UTC)

    def process_result_value(self, value: dt.datetime | None, dialect):  # noqa: ANN001, ANN201
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {dict[str, Any]: JSONVariant, dt.datetime: UTCDateTime}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk!r}>"


class IdMixin:
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(
        UTCDateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


class SyntheticFlagMixin:
    """Marks records that are demonstration data, not observed evidence.

    Every table that can hold demonstration content carries this flag, and it
    propagates into feature snapshots, scores and API responses. The UI refuses
    to present a score as validated when any contributing record is synthetic.
    """

    is_synthetic: Mapped[bool] = mapped_column(default=False, nullable=False, index=True)
    synthetic_note: Mapped[str | None] = mapped_column(String(512), default=None)
