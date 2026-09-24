"""Reproducible pipeline: features → cohort → training → artifacts → serving."""

from trs.pipeline.artifacts import (
    ModelBundle,
    activate_artifact,
    get_active_artifact,
    load_bundle,
    register_artifact,
)

__all__ = [
    "ModelBundle", "activate_artifact", "get_active_artifact", "load_bundle",
    "register_artifact",
]
