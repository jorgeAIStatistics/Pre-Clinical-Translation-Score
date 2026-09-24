"""API version 1.

Versioned from the first release. A breaking change to a payload ships as
``/api/v2``; ``/api/v1`` keeps working. Model versions and feature-schema
versions are carried *inside* payloads rather than in the URL, because a
retrained model is not a new API.
"""

from fastapi import APIRouter

from trs.api.v1 import model, portfolio, programs, provenance, scores

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(programs.router)
api_v1.include_router(scores.router)
api_v1.include_router(model.router)
api_v1.include_router(portfolio.router)
api_v1.include_router(provenance.router)

__all__ = ["api_v1"]
