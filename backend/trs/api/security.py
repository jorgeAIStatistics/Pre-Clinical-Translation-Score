"""Authentication, authorisation and rate limiting.

Posture
-------
The MVP ships working foundations rather than a stub: password hashing, JWT
bearer tokens, hashed API keys, organisation-scoped access, role checks, an
append-only audit trail and a token-bucket rate limiter.

One deliberate development affordance, and its guard rail
----------------------------------------------------------
When ``TRS_ENV`` is ``development`` or ``test``, an unauthenticated request is
resolved to the demonstration organisation so the UI runs without a login
flow. That is a convenience, and conveniences of this kind are how systems ship
with authentication effectively disabled. It is therefore hard-gated:
:func:`assert_production_safety` refuses to start the application in production
if the fallback is reachable or the secret key is still the development
default. The check runs at import of the app, not at first request.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import secrets
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.config import settings
from trs.db.base import utcnow
from trs.db.models.tenancy import ApiKey, AuditEvent
from trs.ontology.enums import OrgRole

ALGORITHM = "HS256"
API_KEY_PREFIX_LENGTH = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#: Role ordering for permission checks. Higher index implies every lower power.
ROLE_ORDER: tuple[str, ...] = (
    OrgRole.VIEWER, OrgRole.ANALYST, OrgRole.ADMIN, OrgRole.OWNER,
)


class Principal:
    """The authenticated caller, whatever the credential type."""

    __slots__ = ("user_id", "organization_id", "role", "api_key_id", "is_development_fallback")

    def __init__(
        self,
        organization_id: str,
        role: str,
        *,
        user_id: str | None = None,
        api_key_id: str | None = None,
        is_development_fallback: bool = False,
    ) -> None:
        self.organization_id = organization_id
        self.role = role
        self.user_id = user_id
        self.api_key_id = api_key_id
        self.is_development_fallback = is_development_fallback

    def has_role(self, minimum: str) -> bool:
        try:
            return ROLE_ORDER.index(self.role) >= ROLE_ORDER.index(minimum)
        except ValueError:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "organization_id": self.organization_id,
            "role": self.role,
            "user_id": self.user_id,
            "api_key_id": self.api_key_id,
            "is_development_fallback": self.is_development_fallback,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Passwords and tokens
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(
    *, user_id: str, organization_id: str, role: str, expires_minutes: int | None = None
) -> str:
    ttl = expires_minutes or settings.access_token_ttl_minutes
    now = utcnow()
    payload = {
        "sub": user_id,
        "org": organization_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=ttl)).timestamp()),
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# API keys
# ─────────────────────────────────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str, str]:
    """Return ``(plaintext, prefix, hash)``. The plaintext is shown once."""
    raw = f"trs_{secrets.token_urlsafe(32)}"
    return raw, raw[:API_KEY_PREFIX_LENGTH], hash_api_key(raw)


def hash_api_key(raw: str) -> str:
    # Keyed hash: a leaked database of key hashes is not offline-crackable
    # without the application secret.
    return hmac.new(settings.secret_key.encode(), raw.encode(), hashlib.sha256).hexdigest()


def resolve_api_key(session: Session, raw: str) -> Principal | None:
    digest = hash_api_key(raw)
    row = session.scalars(
        select(ApiKey).where(
            ApiKey.prefix == raw[:API_KEY_PREFIX_LENGTH],
            ApiKey.revoked_at.is_(None),
        )
    ).first()
    if row is None or not hmac.compare_digest(row.key_hash, digest):
        return None
    if row.expires_at is not None and row.expires_at < utcnow():
        return None
    row.last_used_at = utcnow()
    return Principal(
        organization_id=row.organization_id, role=row.role, api_key_id=row.id
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class TokenBucketLimiter:
    """In-process token bucket, keyed by principal or client address.

    Adequate for a single-process deployment and for making the limit visible
    in the API contract. A multi-process deployment needs a shared store
    (Redis); the interface here is the one that would be swapped, and that is
    noted in ARCHITECTURE.md rather than discovered later.
    """

    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self.rate = rate_per_minute / 60.0
        self.burst = float(burst)
        self._buckets: dict[str, tuple[float, float]] = defaultdict(
            lambda: (self.burst, time.monotonic())
        )

    def check(self, key: str) -> tuple[bool, float]:
        tokens, last = self._buckets[key]
        now = time.monotonic()
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False, (1.0 - tokens) / self.rate
        self._buckets[key] = (tokens - 1.0, now)
        return True, 0.0


limiter = TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)


def enforce_rate_limit(request: Request, principal: Principal | None) -> None:
    key = (
        principal.api_key_id or principal.user_id or principal.organization_id
        if principal else (request.client.host if request.client else "anonymous")
    )
    allowed, retry_after = limiter.check(str(key))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit of {settings.rate_limit_per_minute} requests per "
                "minute exceeded."
            ),
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────────────

def record_audit(
    session: Session,
    *,
    principal: Principal | None,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    request: Request | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append an audit row. Never raises into the request path.

    Scoring is an auditable act: a committee that saw a number must later be
    able to learn which model version and which evidence snapshot produced it.
    """
    try:
        session.add(AuditEvent(
            organization_id=principal.organization_id if principal else None,
            actor_user_id=principal.user_id if principal else None,
            actor_api_key_id=principal.api_key_id if principal else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            occurred_at=utcnow(),
            request_id=(
                request.headers.get("x-request-id") if request is not None else None
            ),
            ip_address=(
                request.client.host if request is not None and request.client else None
            ),
            detail=detail,
        ))
        session.flush()
    except Exception:  # noqa: BLE001 - auditing must not break the request
        session.rollback()


# ─────────────────────────────────────────────────────────────────────────────
# Production guard
# ─────────────────────────────────────────────────────────────────────────────

def development_fallback_enabled() -> bool:
    return settings.env in ("development", "test")


def assert_production_safety() -> None:
    """Refuse to start a production process with development affordances live."""
    if not settings.is_production:
        return
    problems: list[str] = []
    if development_fallback_enabled():
        problems.append(
            "the unauthenticated development fallback is reachable in production"
        )
    if settings.secret_key == "dev-only-insecure-key-change-me":
        problems.append("TRS_SECRET_KEY is still the development default")
    if "*" in settings.cors_origin_list:
        problems.append("CORS is configured to allow every origin")
    if problems:
        raise RuntimeError(
            "Refusing to start in production: " + "; ".join(problems) + ". "
            "See .env.example."
        )


def require_role(principal: Principal, minimum: str) -> None:
    if not principal.has_role(minimum):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This action requires the {minimum!r} role or higher.",
        )
