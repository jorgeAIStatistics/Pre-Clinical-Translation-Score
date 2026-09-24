"""FastAPI application.

    uvicorn trs.api.main:app --reload --port 8000

OpenAPI is served at ``/api/docs``. The schema is generated from the Pydantic
models in :mod:`trs.api.schemas`, so the contract cannot drift from the code
that serves it.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from trs import __version__
from trs.api.schemas import HealthOut
from trs.api.security import assert_production_safety, development_fallback_enabled
from trs.api.v1 import api_v1
from trs.config import settings
from trs.features.spec import FEATURE_SCHEMA_VERSION

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("trs.api")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    # Fail fast rather than serving with development affordances live.
    assert_production_safety()
    if development_fallback_enabled():
        logger.warning(
            "Unauthenticated development fallback is ACTIVE (TRS_ENV=%s). "
            "Requests without credentials resolve to the first organisation. "
            "This is refused in production.",
            settings.env,
        )
    yield


DESCRIPTION = """
A decision-support API that estimates and explains the risk that a preclinical
therapeutic program fails to translate into human clinical efficacy.

**What every estimate carries.** No endpoint returns a bare probability. A score
always arrives with its uncertainty interval, a decomposition of where that
interval's width comes from, an evidence-completeness assessment, an
evidence-confidence grade, the exact model-derived driver attribution, and the
model version that produced it.

**Temporal integrity.** Every score is computed as of a stated cutoff date and
uses only evidence that was knowable on or before it. This is enforced in one
place and tested directly, because a violation invalidates every performance
figure the platform reports.

**Explanations come from the model.** Driver contributions are terms in the
model's own linear predictor and sum exactly to the reported log-odds. Nothing
in an explanation is generated after the fact.

**Demonstration data.** A deployment seeded with the demonstration dataset
flags every affected response with `is_demonstration`. Those estimates describe
a documented simulation and carry no scientific interpretation.
""".strip()

app = FastAPI(
    title="Preclinical Translational Risk Scorer",
    description=DESCRIPTION,
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):  # noqa: ANN001, ANN201
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: ANN201
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    # Internals are logged, never returned: a stack trace in a response body is
    # an information-disclosure bug.
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error.",
            "request_id": request.headers.get("x-request-id"),
        },
    )


app.include_router(api_v1)


@app.get("/api/v1/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    from sqlalchemy import select

    from trs.db.models.analytics import ModelArtifact
    from trs.db.session import session_scope

    active_model: str | None = None
    demo = False
    db_status = "ok"
    try:
        with session_scope() as session:
            artifact = session.scalars(
                select(ModelArtifact).where(ModelArtifact.is_active.is_(True)).limit(1)
            ).first()
            if artifact:
                active_model = artifact.version
                demo = artifact.trained_on_synthetic
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {type(exc).__name__}"

    return HealthOut(
        status="ok" if db_status == "ok" else "degraded",
        version=__version__,
        environment=settings.env,
        database=db_status,
        active_model=active_model,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        is_demonstration_deployment=demo,
        development_auth_fallback=development_fallback_enabled(),
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "name": "Preclinical Translational Risk Scorer",
        "version": __version__,
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }
