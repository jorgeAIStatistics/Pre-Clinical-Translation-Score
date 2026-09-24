"""OpenAlex connector.

What OpenAlex adds over PubMed
------------------------------
Citation counts, an explicit ``is_retracted`` flag, open-access status,
institutional affiliations resolved to ROR identifiers, and coverage of the
preprint and non-MEDLINE literature that PubMed does not index. It is the
connector that populates :attr:`trs.db.models.evidence.Source.is_retracted` for
the general case; PubMed only knows about retractions it has indexed a notice
for.

Field selection is part of the contract
---------------------------------------
A full OpenAlex work is ~35 KB, most of it ``referenced_works`` and topic
scaffolding the platform does not read. Every request pins an explicit
``select`` list, which keeps raw payloads small enough to store verbatim
forever and makes schema drift visible: a field that disappears from the
response is a field we asked for and did not get, rather than one we silently
stopped noticing.

Dating
------
``publication_date`` is a full ISO date on essentially every work and is used
directly. Where only ``publication_year`` survives, the year resolves to 1
January (the shared first-day rule) and the record carries a warning, because
a year-precision date is an eleven-month window, not a date.

Abstracts arrive as an inverted index (token → positions) for copyright
reasons; reconstructing it is a pure rearrangement of the payload we were
given, not an inference, so it happens in ``normalise``.

Polite pool
-----------
OpenAlex serves anonymous traffic from a shared, slower pool. Supplying a
contact address via ``TRS_OPENALEX_MAILTO`` moves requests to the polite pool.
The connector reports ``DEGRADED`` when no address is configured, because the
source is usable but the platform is being a bad citizen and will be throttled
first under load.
"""

from __future__ import annotations

from typing import Any, ClassVar

from trs.config import settings
from trs.connectors import first_day_of_period, iso_date, parse_partial_date
from trs.connectors.base import (
    Connector,
    ConnectorError,
    ConnectorHealth,
    HealthStatus,
    NormalisedRecord,
    RawPayload,
    register_connector,
)
from trs.db.base import utcnow
from trs.ontology.enums import (
    ConfidenceLevel,
    ProvenanceMethod,
    PublicationStatus,
    SourceType,
)

#: Exactly the fields the platform reads. See the module docstring.
WORK_FIELDS: tuple[str, ...] = (
    "id", "doi", "ids", "title", "display_name", "publication_date",
    "publication_year", "type", "cited_by_count", "is_retracted", "is_paratext",
    "open_access", "concepts", "referenced_works_count", "authorships",
    "primary_location", "language", "abstract_inverted_index",
)

#: OpenAlex work types that a journal or conference venue makes peer-reviewed.
#: Anything outside this set is left UNKNOWN rather than assumed.
_JOURNAL_WORK_TYPES: frozenset[str] = frozenset(
    {"article", "review", "letter", "editorial", "erratum"}
)


@register_connector
class OpenAlexConnector(Connector):
    """Adapter for the OpenAlex works API."""

    name: ClassVar[str] = "openalex"
    version: ClassVar[str] = "1.0.0"
    display_name: ClassVar[str] = "OpenAlex"
    homepage: ClassVar[str] = "https://openalex.org/"
    license_note: ClassVar[str] = (
        "OpenAlex data are released into the public domain under CC0 1.0. "
        "Abstracts are served as an inverted index and remain subject to the "
        "original publisher's rights. See https://docs.openalex.org/"
    )
    #: OpenAlex asks for <=10 requests/second and 100k/day on the polite pool.
    min_request_interval: ClassVar[float] = 0.11

    _BASE: ClassVar[str] = "https://api.openalex.org"

    def base_url(self) -> str:
        return self._BASE

    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {"select": ",".join(WORK_FIELDS)}
        if settings.openalex_mailto:
            params["mailto"] = settings.openalex_mailto
        return params

    def health(self) -> ConnectorHealth:
        if not settings.allow_live_network:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DISABLED, checked_at=utcnow(),
                detail="live network disabled by configuration (TRS_ALLOW_LIVE_NETWORK)",
                endpoint=self._BASE,
            )
        endpoint = f"{self._BASE}/works"
        started = utcnow()
        try:
            payload = self.request(
                "GET", endpoint,
                params={**self._common_params(), "per-page": 1},
                use_cache=False, allow_stale_on_error=False,
            )
            body = payload.json()
        except (ConnectorError, ValueError) as exc:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.UNREACHABLE,
                checked_at=utcnow(), detail=str(exc), endpoint=endpoint,
            )
        latency = (utcnow() - started).total_seconds() * 1000
        count = ((body.get("meta") or {}).get("count"))
        if settings.openalex_mailto:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.OK, checked_at=utcnow(),
                latency_ms=latency, detail=f"works index reachable ({count} works)",
                endpoint=endpoint,
            )
        return ConnectorHealth(
            connector=self.name, status=HealthStatus.DEGRADED, checked_at=utcnow(),
            latency_ms=latency,
            detail=(
                f"works index reachable ({count} works) but no TRS_OPENALEX_MAILTO is "
                "configured, so requests use the shared anonymous pool and will be "
                "throttled before polite-pool traffic"
            ),
            endpoint=endpoint,
        )

    # ---- fetch -------------------------------------------------------------

    def search_works(
        self,
        query: str,
        *,
        max_results: int = 25,
        filters: str | None = None,
        page: int = 1,
    ) -> RawPayload:
        """Full-text search over titles, abstracts and full text.

        ``filters`` is passed through verbatim as OpenAlex's ``filter``
        expression (for example
        ``from_publication_date:2015-01-01,is_retracted:false``). It is not
        parsed here: OpenAlex's filter grammar is richer than anything this
        connector should try to model, and mis-modelling it would silently
        change which evidence the platform sees.
        """
        params: dict[str, Any] = {
            **self._common_params(),
            "search": query,
            "per-page": max(1, min(max_results, 200)),
            "page": max(1, page),
        }
        if filters:
            params["filter"] = filters
        return self.request("GET", f"{self._BASE}/works", params=params)

    def get_work(self, work_id: str) -> RawPayload:
        """One work by OpenAlex id, ``doi:10.x/y``, ``pmid:123`` or full URL."""
        identifier = work_id.strip()
        if identifier.startswith("http"):
            identifier = identifier.rstrip("/").rsplit("/", 1)[-1]
        if not identifier:
            raise ConnectorError(f"{self.name}: empty work identifier")
        return self.request(
            "GET", f"{self._BASE}/works/{identifier}", params=self._common_params()
        )

    # ---- normalise ---------------------------------------------------------

    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        try:
            body = payload.json()
        except ValueError:
            return []
        if isinstance(body, dict) and "results" in body:
            works = [w for w in body.get("results") or [] if isinstance(w, dict)]
        elif isinstance(body, dict) and body.get("id"):
            works = [body]
        else:
            return []

        records: list[NormalisedRecord] = []
        for index, work in enumerate(works):
            record = self._work_record(payload, work, index)
            if record is not None:
                records.append(record)
        return records

    def _work_record(
        self, payload: RawPayload, work: dict[str, Any], index: int
    ) -> NormalisedRecord | None:
        openalex_id = _short_id(work.get("id"))
        if not openalex_id:
            return None

        warnings: list[str] = []
        available_at = parse_partial_date(work.get("publication_date"))
        if available_at is None:
            year = work.get("publication_year")
            available_at = first_day_of_period(year if isinstance(year, int) else None)
            if available_at is None:
                # Neither a date nor a year: unplaceable in time, so not emitted.
                return None
            warnings.append(
                "publication_date absent; available_at derived from publication_year "
                f"{year} and is therefore precise only to the year."
            )

        primary = work.get("primary_location") or {}
        venue = primary.get("source") or {}
        work_type = (work.get("type") or "").lower()
        status = _publication_status(work_type, venue)
        if status is PublicationStatus.UNKNOWN:
            warnings.append(
                f"publication_status left unknown for OpenAlex type {work_type!r} in venue "
                f"type {venue.get('type')!r}"
            )

        ids = work.get("ids") or {}
        body: dict[str, Any] = {
            "openalex_id": openalex_id,
            "pmid": _strip_prefix(ids.get("pmid"), "https://pubmed.ncbi.nlm.nih.gov/"),
            "pmcid": _strip_prefix(ids.get("pmcid"), "https://www.ncbi.nlm.nih.gov/pmc/articles/"),
            "doi": _strip_prefix(work.get("doi"), "https://doi.org/"),
            "title": work.get("title") or work.get("display_name"),
            "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
            "journal": venue.get("display_name"),
            "authors": _authors(work.get("authorships")),
            "publication_types": [work.get("type")] if work.get("type") else [],
            "published_on": iso_date(parse_partial_date(work.get("publication_date"))),
            "mesh_terms": [],
            "publication_status": str(status),
            "source_type": str(_source_type(status, venue)),
            "url": f"https://openalex.org/{openalex_id}",
            # Straight onto Source.is_retracted. OpenAlex is the platform's
            # authority for this flag because it tracks retractions for works
            # PubMed never indexed.
            "is_retracted": bool(work.get("is_retracted")),
            "is_paratext": bool(work.get("is_paratext")),
            "cited_by_count": work.get("cited_by_count"),
            "open_access": work.get("open_access") or {},
            "concepts": [
                {
                    "id": _short_id(c.get("id")),
                    "display_name": c.get("display_name"),
                    "level": c.get("level"),
                    "score": c.get("score"),
                }
                for c in work.get("concepts") or []
                if isinstance(c, dict)
            ],
            "referenced_works_count": work.get("referenced_works_count"),
            "institutions": _institutions(work.get("authorships")),
            "language": work.get("language"),
        }

        return NormalisedRecord(
            kind="source",
            external_id=f"openalex:{openalex_id}",
            available_at=available_at,
            payload=body,
            provenance={
                **self.provenance_stub(payload),
                "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
                "record_index": index,
                "available_at_basis": (
                    "publication_date" if work.get("publication_date") else "publication_year"
                ),
            },
            confidence=str(
                ConfidenceLevel.HIGH if work.get("publication_date") else ConfidenceLevel.MODERATE
            ),
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short_id(value: Any) -> str | None:
    """``https://openalex.org/W123`` → ``W123``; other shapes pass through."""
    if not isinstance(value, str) or not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def _strip_prefix(value: Any, prefix: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.removeprefix(prefix).strip("/")
    return text or None


def _reconstruct_abstract(inverted: Any) -> str | None:
    """Rebuild running text from OpenAlex's token → positions index.

    Purely a rearrangement of data already in the payload. Gaps (a position no
    token claims) are left as-is rather than filled, so a sparse index produces
    a shorter abstract instead of an invented one.
    """
    if not isinstance(inverted, dict) or not inverted:
        return None
    positions: dict[int, str] = {}
    for token, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for position in indexes:
            if isinstance(position, int):
                positions[position] = token
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _authors(authorships: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in authorships or []:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author") or {}
        name = author.get("display_name")
        if not name:
            continue
        record: dict[str, Any] = {"name": name}
        institutions = [
            i.get("display_name") for i in entry.get("institutions") or []
            if isinstance(i, dict) and i.get("display_name")
        ]
        if institutions:
            # Source.authors stores one affiliation string per author; the full
            # institution list is preserved separately under "institutions".
            record["affiliation"] = "; ".join(institutions)
        if entry.get("author_position"):
            record["position"] = entry["author_position"]
        if author.get("orcid"):
            record["orcid"] = _strip_prefix(author["orcid"], "https://orcid.org/")
        out.append(record)
    return out


def _institutions(authorships: Any) -> list[dict[str, Any]]:
    """Distinct institutions across all authorships, first-seen order."""
    seen: dict[str, dict[str, Any]] = {}
    for entry in authorships or []:
        if not isinstance(entry, dict):
            continue
        for institution in entry.get("institutions") or []:
            if not isinstance(institution, dict):
                continue
            key = institution.get("id") or institution.get("display_name")
            if not key or key in seen:
                continue
            seen[key] = {
                "id": _short_id(institution.get("id")),
                "display_name": institution.get("display_name"),
                "ror": institution.get("ror"),
                "country_code": institution.get("country_code"),
                "type": institution.get("type"),
            }
    return list(seen.values())


def _publication_status(work_type: str, venue: dict[str, Any]) -> PublicationStatus:
    """OpenAlex type + venue type → publication status.

    OpenAlex labels preprints two ways: a ``preprint`` work type, and an
    ordinary article whose primary location is a repository. Both are treated
    as preprints. A journal or conference venue makes the listed work types
    peer-reviewed; everything else stays ``UNKNOWN``.
    """
    venue_type = (venue.get("type") or "").lower()
    if work_type == "preprint" or venue_type == "repository":
        return PublicationStatus.PREPRINT
    if work_type in _JOURNAL_WORK_TYPES and venue_type in {"journal", "conference"}:
        return PublicationStatus.PEER_REVIEWED
    return PublicationStatus.UNKNOWN


def _source_type(status: PublicationStatus, venue: dict[str, Any]) -> SourceType:
    if status is PublicationStatus.PREPRINT:
        return SourceType.PREPRINT
    if (venue.get("type") or "").lower() in {"journal", "conference"}:
        return SourceType.JOURNAL_ARTICLE
    return SourceType.UNKNOWN


__all__ = ["OpenAlexConnector", "WORK_FIELDS"]