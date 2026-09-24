"""Ingestion connector contract.

Pipeline shape (identical for every source)
-------------------------------------------
``fetch`` → ``RawRecord`` → ``normalise`` → domain objects → entity resolution
→ evidence

``fetch`` is the only method permitted to touch the network. It always writes a
:class:`~trs.db.models.evidence.RawRecord` containing the verbatim payload
before anything interprets it, so every downstream value can be re-derived and
every extraction bug can be diagnosed against exactly the bytes that produced
it. Raw records are never rewritten and never deleted.

``normalise`` is a pure function from a raw payload to normalised
:class:`NormalisedRecord` objects. Being pure, it is testable against stored
fixtures with no network, which is how the connector test-suite runs in CI.

Why adapters rather than direct calls
--------------------------------------
Every public biomedical source is unreliable in a different way: rate limits,
silent schema drift, IP-based blocking, partial outages of specific endpoints.
Commercial sources will be added later with entirely different auth and shape.
Isolating each behind this interface means an outage degrades one connector
instead of the platform, and a licensed trial-intelligence feed can be dropped
in as a sibling class without touching feature engineering.

Reachability is a first-class state
------------------------------------
:meth:`Connector.health` probes the source and returns a
:class:`ConnectorHealth`. A connector that cannot reach its source reports
``unreachable`` and, if a cached payload exists, serves it while marking every
derived record ``from_cache``. It never fabricates, and it never silently
returns an empty result that a caller could mistake for "no evidence found".
"""

from __future__ import annotations

import abc
import dataclasses
import datetime as dt
import hashlib
import json
import time
from enum import StrEnum
from typing import Any, ClassVar

import httpx
from sqlalchemy.orm import Session

from trs.config import settings
from trs.connectors.cache import ResponseCache
from trs.db.base import utcnow
from trs.db.models.analytics import ConnectorRun
from trs.db.models.evidence import RawRecord


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    NOT_CONFIGURED = "not_configured"
    DISABLED = "disabled"


@dataclasses.dataclass(slots=True)
class ConnectorHealth:
    connector: str
    status: HealthStatus
    checked_at: dt.datetime
    latency_ms: float | None = None
    detail: str = ""
    endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "status": str(self.status),
            "checked_at": self.checked_at.isoformat(),
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "endpoint": self.endpoint,
        }


@dataclasses.dataclass(slots=True)
class RawPayload:
    """What ``fetch`` returns, before any interpretation."""

    endpoint: str
    params: dict[str, Any]
    status_code: int
    media_type: str
    content: bytes
    fetched_at: dt.datetime
    from_cache: bool = False

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


@dataclasses.dataclass(slots=True)
class NormalisedRecord:
    """A source-agnostic normalised record.

    ``kind`` tells the loader which domain table it belongs to
    (``source`` / ``trial`` / ``target_assessment`` / ``compound`` / ``evidence``).
    ``available_at`` is mandatory: a record whose availability date cannot be
    established is not admissible, because it cannot be placed in time and
    therefore cannot be used without risking leakage.
    """

    kind: str
    external_id: str
    available_at: dt.date
    payload: dict[str, Any]
    provenance: dict[str, Any]
    #: Confidence in the normalisation itself, not in the underlying science.
    confidence: str = "high"
    warnings: list[str] = dataclasses.field(default_factory=list)


class ConnectorError(RuntimeError):
    """Recoverable connector failure; the run is recorded and the caller informed."""


class Connector(abc.ABC):
    """Base class for all ingestion adapters."""

    #: Stable registry key, stored on every RawRecord and Source.
    name: ClassVar[str]
    #: Bump when ``normalise`` changes output for unchanged input.
    version: ClassVar[str] = "1.0.0"
    #: Human-readable source name for provenance display.
    display_name: ClassVar[str] = ""
    #: Canonical landing page, shown in the provenance UI.
    homepage: ClassVar[str] = ""
    #: Licence / terms-of-use statement shown alongside ingested records.
    license_note: ClassVar[str] = ""
    #: Minimum seconds between requests, to respect published rate limits.
    min_request_interval: ClassVar[float] = 0.35

    def __init__(
        self,
        *,
        cache: ResponseCache | None = None,
        client: httpx.Client | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.cache = cache or ResponseCache(settings.connector_cache_dir, self.name)
        self._client = client
        self._owns_client = client is None
        self._last_request_at = 0.0
        #: Retry budget for this instance. Ingestion wants the configured
        #: retries with backoff; a health probe wants exactly one attempt —
        #: retrying a dead endpoint three times with exponential backoff turns
        #: a 6-second timeout into a 20-second wait, and six connectors into a
        #: 40-second page load. Callers that need a fast answer pass 1.
        self.max_retries = (
            settings.connector_max_retries if max_retries is None else max(1, int(max_retries))
        )

    # ---- required ----------------------------------------------------------

    @abc.abstractmethod
    def base_url(self) -> str: ...

    @abc.abstractmethod
    def health(self) -> ConnectorHealth: ...

    @abc.abstractmethod
    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        """Pure transformation. Must not perform I/O."""

    # ---- HTTP --------------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=settings.connector_timeout_seconds,
                headers={
                    "User-Agent": settings.connector_user_agent,
                    "Accept": "application/json",
                },
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Connector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_at = time.monotonic()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        use_cache: bool = True,
        allow_stale_on_error: bool = True,
    ) -> RawPayload:
        """Perform one request with caching, throttling and bounded retries.

        On a network failure, a cached payload is served if one exists and
        ``allow_stale_on_error`` is set — flagged ``from_cache`` so provenance
        records that the data did not come fresh from the source.
        """
        url = path if path.startswith("http") else f"{self.base_url().rstrip('/')}/{path.lstrip('/')}"
        key = self.cache.key(method, url, params, json_body)

        if use_cache:
            cached = self.cache.get(key, ttl_hours=settings.connector_cache_ttl_hours)
            if cached is not None:
                return RawPayload(
                    endpoint=url, params=params or {}, status_code=cached["status_code"],
                    media_type=cached["media_type"], content=cached["content"],
                    fetched_at=dt.datetime.fromisoformat(cached["fetched_at"]),
                    from_cache=True,
                )

        if not settings.allow_live_network:
            stale = self.cache.get(key, ttl_hours=None)
            if stale is not None:
                return RawPayload(
                    endpoint=url, params=params or {}, status_code=stale["status_code"],
                    media_type=stale["media_type"], content=stale["content"],
                    fetched_at=dt.datetime.fromisoformat(stale["fetched_at"]),
                    from_cache=True,
                )
            raise ConnectorError(
                f"{self.name}: live network disabled and no cached response for {url}"
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self.client.request(method, url, params=params, json=json_body)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = ConnectorError(
                    f"{self.name}: HTTP {response.status_code} from {url}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
                continue

            payload = RawPayload(
                endpoint=url,
                params=params or {},
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
                content=response.content,
                fetched_at=utcnow(),
            )
            if response.is_success and use_cache:
                self.cache.put(key, payload)
            if not response.is_success:
                raise ConnectorError(
                    f"{self.name}: HTTP {response.status_code} from {url} "
                    f"({response.text[:200]})"
                )
            return payload

        if allow_stale_on_error:
            stale = self.cache.get(key, ttl_hours=None)
            if stale is not None:
                return RawPayload(
                    endpoint=url, params=params or {}, status_code=stale["status_code"],
                    media_type=stale["media_type"], content=stale["content"],
                    fetched_at=dt.datetime.fromisoformat(stale["fetched_at"]),
                    from_cache=True,
                )
        raise ConnectorError(f"{self.name}: exhausted retries for {url}") from last_error

    # ---- persistence -------------------------------------------------------

    def persist_raw(self, session: Session, payload: RawPayload) -> RawRecord:
        """Store the verbatim payload, deduplicating on content hash."""
        from sqlalchemy import select

        digest = payload.content_hash
        existing = session.scalars(
            select(RawRecord).where(RawRecord.content_hash == digest)
        ).first()
        if existing is not None:
            return existing

        content_json = None
        if "json" in (payload.media_type or ""):
            try:
                content_json = payload.json()
            except (ValueError, UnicodeDecodeError):
                content_json = None

        record = RawRecord(
            connector=self.name,
            connector_version=self.version,
            endpoint=payload.endpoint,
            request_params=payload.params,
            http_status=payload.status_code,
            media_type=payload.media_type,
            content_hash=digest,
            content_bytes=None if content_json is not None else payload.content,
            content_json=content_json,
            fetched_at=payload.fetched_at,
            from_cache=payload.from_cache,
        )
        session.add(record)
        session.flush()
        return record

    def start_run(self, session: Session, operation: str, parameters: dict) -> ConnectorRun:
        run = ConnectorRun(
            connector=self.name, connector_version=self.version,
            operation=operation, parameters=parameters,
            started_at=utcnow(), status="running",
        )
        session.add(run)
        session.flush()
        return run

    @staticmethod
    def finish_run(
        session: Session, run: ConnectorRun, *, status: str = "succeeded", **counters: Any
    ) -> ConnectorRun:
        run.finished_at = utcnow()
        run.status = status
        for k, v in counters.items():
            if hasattr(run, k):
                setattr(run, k, v)
        session.flush()
        return run

    # ---- provenance --------------------------------------------------------

    def provenance_stub(self, payload: RawPayload) -> dict[str, Any]:
        return {
            "connector": self.name,
            "connector_version": self.version,
            "source_name": self.display_name or self.name,
            "endpoint": payload.endpoint,
            "retrieved_at": payload.fetched_at.isoformat(),
            "from_cache": payload.from_cache,
            "content_hash": payload.content_hash,
            "license_note": self.license_note,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[Connector]] = {}


def register_connector(cls: type[Connector]) -> type[Connector]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a `name`")
    _REGISTRY[cls.name] = cls
    return cls


def get_connector(name: str, **kwargs: Any) -> Connector:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown connector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_connectors() -> list[str]:
    return sorted(_REGISTRY)


def connector_catalogue() -> list[dict[str, Any]]:
    """Metadata for every registered connector, for the provenance UI."""
    return [
        {
            "name": cls.name,
            "version": cls.version,
            "display_name": cls.display_name or cls.name,
            "homepage": cls.homepage,
            "license_note": cls.license_note,
        }
        for cls in sorted(_REGISTRY.values(), key=lambda c: c.name)
    ]
