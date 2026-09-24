"""FastAPI dependencies: session, principal, tenancy scoping."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.api.security import (
    Principal,
    decode_access_token,
    development_fallback_enabled,
    enforce_rate_limit,
    resolve_api_key,
)
from trs.db.models.tenancy import Organization
from trs.db.session import get_session_factory
from trs.ontology.enums import OrgRole


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _development_principal(session: Session) -> Principal:
    """Resolve to the demonstration organisation. Development and test only."""
    org = session.scalars(
        select(Organization).order_by(Organization.created_at).limit(1)
    ).first()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No organisation exists yet. Run `python -m trs.cli pipeline` "
                "to create the demonstration dataset."
            ),
        )
    return Principal(
        organization_id=org.id, role=OrgRole.ANALYST, is_development_fallback=True
    )


def get_principal(
    request: Request,
    session: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Principal:
    """Resolve the caller from a bearer token or an API key.

    In development and test, an unauthenticated request resolves to the
    demonstration organisation so the UI runs without a login flow. The
    resulting principal is flagged ``is_development_fallback`` and every
    response carries that flag, so a deployment running with the fallback
    active cannot do so invisibly. :func:`assert_production_safety` prevents it
    reaching production at all.
    """
    if x_api_key:
        principal = resolve_api_key(session, x_api_key)
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
            )
        enforce_rate_limit(request, principal)
        return principal

    if authorization and authorization.lower().startswith("bearer "):
        payload = decode_access_token(authorization.split(" ", 1)[1])
        principal = Principal(
            organization_id=payload["org"], role=payload["role"], user_id=payload["sub"]
        )
        enforce_rate_limit(request, principal)
        return principal

    if development_fallback_enabled():
        principal = _development_principal(session)
        enforce_rate_limit(request, principal)
        return principal

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Supply a bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_analyst(principal: Principal = Depends(get_principal)) -> Principal:
    from trs.api.security import require_role

    require_role(principal, OrgRole.ANALYST)
    return principal


def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    from trs.api.security import require_role

    require_role(principal, OrgRole.ADMIN)
    return principal
