"""Crossref connector.

Crossref is the DOI registration agency's own metadata: it is authoritative for
what a DOI resolves to, for the licence a publisher attached to a work, and for
the *update graph* that records corrections and retractions.

Retraction detection — a correction to the obvious reading
-----------------------------------------------------------
Crossref models a retraction as a directed edge between two DOIs:

* the **retraction notice** carries ``update-to``, pointing at the work it
  retracts;
* the **retracted work** carries ``updated-by``, pointing back at the notice.

Reading ``update-to`` alone therefore flags the wrong object. It marks the
notice — a perfectly sound editorial document — as retracted, and leaves the
discredited paper clean. Checked against the live API on the canonical case
(DOI ``10.1016/S0140-6736(97)11096-0``, Wakefield 1998), the retracted article
returns ``update-to: null`` and an ``updated-by`` entry of type ``retraction``
dated 2010-02-06; a pipeline keyed on ``update-to`` would have missed it
entirely.

This connector therefore sets ``is_retracted`` from ``updated-by`` (and from
``relation.is-retracted-by``, which some records use instead), and reports
``update-to`` separately as ``is_retraction_notice``. Both directions are kept
in the payload so the loader can build the edge either way.

Dating
------
``available_at`` is the minimum over ``issued``, ``published``,
``published-online``, ``published-print`` and ``created``. ``created`` is the
DOI registration timestamp, which for back-filled content can be years after
publication (the 1998 Lancet paper was registered in 2002) — including it in a
minimum is harmless, using it alone would not be. Crossref date-parts are
frequently year+month only and resolve to the first of the month under the
shared rule.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar

from trs.config import settings
from trs.connectors import earliest_date, first_day_of_period, iso_date, parse_partial_date
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

#: Crossref ``type`` → platform source type. Types outside this map stay
#: UNKNOWN; Crossref has ~30 of them and guessing at the long tail buys nothing.
_SOURCE_TYPES: dict[str, SourceType] = {
    "journal-article": SourceType.JOURNAL_ARTICLE,
    "posted-content": SourceType.PREPRINT,
    "proceedings-article": SourceType.CONFERENCE_ABSTRACT,
    "report": SourceType.UNKNOWN,
}

#: Update types Crossref uses for a withdrawal. ``removal`` and ``withdrawal``
#: are retractions by another name for our purposes: the work should not be
#: relied on.
_RETRACTION_UPDATE_TYPES: frozenset[str] = frozenset(
    {"retraction", "removal", "withdrawal"}
)


@register_connector
class CrossrefConnector(Connector):
    """Adapter for the Crossref REST API."""

    name: ClassVar[str] = "crossref"
    version: ClassVar[str] = "1.0.0"
    display_name: ClassVar[str] = "Crossref"
    homepage: ClassVar[str] = "https://www.crossref.org/"
    license_note: ClassVar[str] = (
        "Crossref metadata are made available under the Crossref REST API terms "
        "and may be used freely with attribution; per-work licence URLs in the "
        "payload govern the underlying content, not the metadata. "
        "See https://www.crossref.org/documentation/retrieve-metadata/rest-api/"
    )
    #: Crossref publishes no hard anonymous limit but asks for restraint; the
    #: polite pool guidance is ~50 requests/second, which we do not need.
    min_request_interval: ClassVar[float] = 0.2

    _BASE: ClassVar[str] = "https://api.crossref.org"

    def base_url(self) -> str:
        return self._BASE

    def _common_params(self) -> dict[str, str]:
        # Crossref routes requests carrying a contact address to the polite
        # pool, which has materially better availability under load.
        return {"mailto": settings.crossref_mailto} if settings.crossref_mailto else {}

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
                "GET", endpoint, params={**self._common_params(), "rows": 0},
                use_cache=False, allow_stale_on_error=False,
            )
            body = payload.json()
        except (ConnectorError, ValueError) as exc:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.UNREACHABLE,
                checked_at=utcnow(), detail=str(exc), endpoint=endpoint,
            )
        latency = (utcnow() - started).total_seconds() * 1000
        total = ((body.get("message") or {}).get("total-results"))
        if settings.crossref_mailto:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.OK, checked_at=utcnow(),
                latency_ms=latency, detail=f"works index reachable ({total} works)",
                endpoint=endpoint,
            )
        return ConnectorHealth(
            connector=self.name, status=HealthStatus.DEGRADED, checked_at=utcnow(),
            latency_ms=latency,
            detail=(
                f"works index reachable ({total} works) but no TRS_CROSSREF_MAILTO is "
                "configured, so requests use the anonymous pool with no availability "
                "guarantee"
            ),
            endpoint=endpoint,
        )

    # ---- fetch -------------------------------------------------------------

    def get_work(self, doi: str) -> RawPayload:
        """One work by DOI. Accepts a bare DOI or any ``doi.org`` URL form."""
        identifier = _normalise_doi(doi)
        if not identifier:
            raise ConnectorError(f"{self.name}: empty or unparseable DOI {doi!r}")
        return self.request(
            "GET", f"{self._BASE}/works/{identifier}", params=self._common_params()
        )

    def search_works(self, query: str, *, rows: int = 20) -> RawPayload:
        """Bibliographic search, used to recover a DOI from a citation string."""
        return self.request(
            "GET", f"{self._BASE}/works",
            params={
                **self._common_params(),
                "query.bibliographic": query,
                "rows": max(1, min(rows, 100)),
            },
        )

    # ---- normalise ---------------------------------------------------------

    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        try:
            body = payload.json()
        except ValueError:
            return []
        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, dict):
            return []

        if body.get("message-type") == "work-list" or "items" in message:
            works = [w for w in message.get("items") or [] if isinstance(w, dict)]
        else:
            works = [message]

        records: list[NormalisedRecord] = []
        for index, work in enumerate(works):
            record = self._work_record(payload, work, index)
            if record is not None:
                records.append(record)
        return records

    def _work_record(
        self, payload: RawPayload, work: dict[str, Any], index: int
    ) -> NormalisedRecord | None:
        doi = _normalise_doi(work.get("DOI"))
        if not doi:
            return None

        warnings: list[str] = []
        dates = {
            "issued": _date_parts(work.get("issued")),
            "published": _date_parts(work.get("published")),
            "published_online": _date_parts(work.get("published-online")),
            "published_print": _date_parts(work.get("published-print")),
            "created": _date_parts(work.get("created")),
        }
        available_at = earliest_date(*dates.values())
        if available_at is None:
            # No Crossref date at all is rare and means the record cannot be
            # placed in time; it is dropped rather than dated by fiat.
            return None

        retracted_by = _retraction_edges(work.get("updated-by"))
        relation_retractions = _relation_retractions(work.get("relation"))
        retracts = _retraction_edges(work.get("update-to"))
        is_retracted = bool(retracted_by or relation_retractions)

        work_type = (work.get("type") or "").lower()
        subtype = (work.get("subtype") or "").lower()
        status = _publication_status(work_type, subtype)
        if status is PublicationStatus.UNKNOWN:
            warnings.append(
                f"publication_status left unknown for Crossref type {work_type!r}"
            )
        if retracts and not is_retracted:
            warnings.append(
                "This record is a retraction notice (it carries update-to), not a "
                "retracted work; is_retracted refers to the work itself and is False."
            )

        licences = [
            {
                "url": entry.get("URL"),
                "content_version": entry.get("content-version"),
                "start": iso_date(_date_parts(entry.get("start"))),
            }
            for entry in work.get("license") or []
            if isinstance(entry, dict)
        ]

        body: dict[str, Any] = {
            "doi": doi,
            "title": _first_string(work.get("title")),
            "container_title": _first_string(work.get("container-title")),
            "authors": _authors(work.get("author")),
            "published_on": iso_date(
                earliest_date(dates["issued"], dates["published_online"], dates["published_print"])
            ),
            "type": work.get("type"),
            "subtype": work.get("subtype") or None,
            "license": licences[0]["url"] if licences else None,
            "licenses": licences,
            "is_retracted": is_retracted,
            "is_retraction_notice": bool(retracts),
            "retraction_note": _retraction_note(retracted_by, relation_retractions),
            "retracted_by": retracted_by,
            "retracts": retracts,
            "publication_status": str(status),
            "source_type": str(_SOURCE_TYPES.get(work_type, SourceType.UNKNOWN)),
            "publisher": work.get("publisher"),
            "issn": list(work.get("ISSN") or []),
            "reference_count": work.get("reference-count"),
            "is_referenced_by_count": work.get("is-referenced-by-count"),
            "url": work.get("URL") or f"https://doi.org/{doi}",
            "dates": {key: iso_date(value) for key, value in dates.items()},
        }

        return NormalisedRecord(
            kind="source",
            external_id=f"doi:{doi}",
            available_at=available_at,
            payload=body,
            provenance={
                **self.provenance_stub(payload),
                "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
                "record_index": index,
                "available_at_basis": "min(" + ", ".join(
                    key for key, value in dates.items() if value == available_at
                ) + ")",
            },
            confidence=str(ConfidenceLevel.HIGH),
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_doi(value: Any) -> str | None:
    """Lower-cased bare DOI. Crossref is case-insensitive on lookup and
    lower-cases on output, so canonicalising here keeps joins stable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.strip().strip("/")
    return text.lower() or None


def _date_parts(node: Any) -> dt.date | None:
    """Crossref ``{"date-parts": [[1998, 2]]}`` → 1998-02-01.

    Crossref also supplies ``date-time`` on some nodes; ``date-parts`` is
    preferred because it states its own precision, whereas ``date-time``
    fabricates midnight on the first of the month for a month-precision date
    and gives no way to tell that apart from a real day.
    """
    if not isinstance(node, dict):
        return None
    parts = node.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list):
        values = [p for p in parts[0] if isinstance(p, int)]
        if values:
            return first_day_of_period(*values[:3])
    return parse_partial_date(node.get("date-time"))


def _first_string(value: Any) -> str | None:
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _authors(authors: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in authors or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or " ".join(
            p for p in (entry.get("given"), entry.get("family")) if p
        )
        if not name:
            continue
        record: dict[str, Any] = {"name": name.strip()}
        affiliations = [
            a.get("name") for a in entry.get("affiliation") or []
            if isinstance(a, dict) and a.get("name")
        ]
        if affiliations:
            record["affiliation"] = "; ".join(affiliations)
        if entry.get("ORCID"):
            record["orcid"] = str(entry["ORCID"]).rsplit("/", 1)[-1]
        if entry.get("sequence"):
            record["position"] = entry["sequence"]
        out.append(record)
    return out


def _retraction_edges(entries: Any) -> list[dict[str, Any]]:
    """Retraction-flavoured entries from ``updated-by`` or ``update-to``."""
    out: list[dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        if (entry.get("type") or "").lower() not in _RETRACTION_UPDATE_TYPES:
            continue
        out.append(
            {
                "doi": _normalise_doi(entry.get("DOI")),
                "type": entry.get("type"),
                "label": entry.get("label"),
                "updated_on": iso_date(_date_parts(entry.get("updated"))),
                "source": entry.get("source"),
            }
        )
    return out


def _relation_retractions(relation: Any) -> list[dict[str, Any]]:
    """``relation.is-retracted-by`` is the alternate encoding some records use."""
    if not isinstance(relation, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in relation.get("is-retracted-by") or []:
        if isinstance(entry, dict) and entry.get("id"):
            out.append({"doi": _normalise_doi(entry.get("id")), "type": "retraction",
                        "label": "Retraction", "updated_on": None, "source": "relation"})
    return out


def _retraction_note(
    retracted_by: list[dict[str, Any]], relation_retractions: list[dict[str, Any]]
) -> str | None:
    edges = retracted_by + relation_retractions
    if not edges:
        return None
    parts = [
        f"{edge.get('label') or edge.get('type')} by {edge.get('doi')}"
        + (f" on {edge['updated_on']}" if edge.get("updated_on") else "")
        for edge in edges
    ]
    return "; ".join(parts)


def _publication_status(work_type: str, subtype: str) -> PublicationStatus:
    if work_type == "posted-content":
        # Crossref's subtype distinguishes preprints from other posted content
        # (letters, comments). Only "preprint" is claimed as such.
        return PublicationStatus.PREPRINT if subtype in {"preprint", ""} else PublicationStatus.UNKNOWN
    if work_type == "journal-article":
        return PublicationStatus.PEER_REVIEWED
    if work_type == "proceedings-article":
        return PublicationStatus.CONFERENCE_ABSTRACT
    return PublicationStatus.UNKNOWN


__all__ = ["CrossrefConnector"]
