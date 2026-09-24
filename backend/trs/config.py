"""Application settings.

Configuration is environment-driven (12-factor). Nothing secret is ever
committed: see ``.env.example`` at the repository root for the full template.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration, read from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="TRS_",
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Core -------------------------------------------------------------
    env: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"

    # ---- Database ---------------------------------------------------------
    # Production target is PostgreSQL. SQLite is the zero-dependency default so
    # the platform is runnable immediately; the ORM layer and migrations are
    # written to be portable across both (see ARCHITECTURE.md § Persistence).
    database_url: str = f"sqlite+pysqlite:///{BACKEND_ROOT / 'trs.db'}"
    database_echo: bool = False

    # ---- Security ---------------------------------------------------------
    secret_key: str = "dev-only-insecure-key-change-me"
    access_token_ttl_minutes: int = 720
    cors_origins: str = "http://localhost:3000"

    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 40

    # ---- Model artifacts --------------------------------------------------
    artifact_dir: Path = BACKEND_ROOT / "artifacts"
    active_model_version: str = ""

    # ---- Connectors -------------------------------------------------------
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    ncbi_tool: str = "preclinical-translational-risk-scorer"
    openalex_mailto: str = ""
    crossref_mailto: str = ""
    opentargets_url: str = "https://api.platform.opentargets.org/api/v4/graphql"
    chembl_url: str = "https://www.ebi.ac.uk/chembl/api/data"
    ctgov_url: str = "https://clinicaltrials.gov/api/v2"

    connector_cache_dir: Path = BACKEND_ROOT / ".cache"
    connector_cache_ttl_hours: int = 168
    connector_timeout_seconds: float = 30.0
    connector_max_retries: int = 3
    connector_user_agent: str = (
        "PreclinicalTranslationalRiskScorer/0.1 "
        "(+https://example.org/trs; research use)"
    )

    # ---- Behaviour flags --------------------------------------------------
    allow_live_network: bool = True
    seed_random_state: int = 20240917

    @field_validator("artifact_dir", "connector_cache_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


settings = get_settings()
