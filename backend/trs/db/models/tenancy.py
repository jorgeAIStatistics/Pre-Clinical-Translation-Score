"""Multi-tenancy, identity, access control and audit.

Isolation model
---------------
``Organization`` is the tenant boundary. Every scientific record that can be
private carries ``organization_id``; the API's dependency layer scopes all
queries by the caller's organization and never trusts a client-supplied id.

``Project`` is a workspace inside an organization: a portfolio, a diligence
engagement, a therapeutic-area team. Programs belong to exactly one project.

Reference data (targets, indications, pathways, published sources) is shared
and has ``organization_id IS NULL`` — it is global, read-only to tenants, and
maintained by the ingestion pipelines.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trs.db.base import Base, IdMixin, JSONVariant, TimestampMixin, UTCDateTime


class Organization(Base, IdMixin, TimestampMixin):
    __tablename__ = "organization"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    settings: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    projects: Mapped[list[Project]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "app_user"

    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200), default=None)
    # Argon2/bcrypt hash. Never a plaintext or reversible value.
    password_hash: Mapped[str | None] = mapped_column(String(256), default=None)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_service_account: Mapped[bool] = mapped_column(default=False, nullable=False)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Membership(Base, IdMixin, TimestampMixin):
    """A user's role within an organization."""

    __tablename__ = "membership"
    __table_args__ = (UniqueConstraint("organization_id", "user_id"),)

    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organization.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("app_user.id", ondelete="CASCADE"), index=True
    )
    # trs.ontology.enums.OrgRole
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="viewer")

    organization: Mapped[Organization] = relationship(back_populates="memberships")
    user: Mapped[User] = relationship(back_populates="memberships")


class Project(Base, IdMixin, TimestampMixin):
    """A workspace: a portfolio, diligence engagement or therapeutic-area team."""

    __tablename__ = "project"
    __table_args__ = (UniqueConstraint("organization_id", "slug"),)

    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organization.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    is_private: Mapped[bool] = mapped_column(default=True, nullable=False)
    archived_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)

    organization: Mapped[Organization] = relationship(back_populates="projects")


class ApiKey(Base, IdMixin, TimestampMixin):
    """Machine credential for programmatic API access.

    Only the hash is stored; the plaintext is shown once at creation time.
    """

    __tablename__ = "api_key"

    organization_id: Mapped[str] = mapped_column(
        ForeignKey("organization.id", ondelete="CASCADE"), index=True
    )
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("app_user.id", ondelete="SET NULL"), default=None
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="viewer")
    expires_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)


class AuditEvent(Base, IdMixin):
    """Append-only record of consequential actions.

    Scoring is an auditable act: a portfolio committee that saw a number must
    be able to learn, later, exactly which model version and which evidence
    snapshot produced it. Audit rows are never updated or deleted.
    """

    __tablename__ = "audit_event"
    __table_args__ = (
        Index("ix_audit_event_org_occurred", "organization_id", "occurred_at"),
    )

    organization_id: Mapped[str | None] = mapped_column(String(32), default=None)
    actor_user_id: Mapped[str | None] = mapped_column(String(32), default=None)
    actor_api_key_id: Mapped[str | None] = mapped_column(String(32), default=None)
    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(80), default=None)
    resource_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)
    ip_address: Mapped[str | None] = mapped_column(String(64), default=None)
    detail: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
