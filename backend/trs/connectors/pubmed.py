"""PubMed / NCBI E-utilities connector.

Endpoints used
--------------
``esearch`` resolves a query to PMIDs, ``efetch`` returns the full MEDLINE XML
record, and ``esummary`` returns a compact JSON summary. ``efetch`` is the
primary path because it is the only one carrying the abstract, the MeSH
headings, author affiliations and — critically — the ``<History>`` block that
dates the record's entry into PubMed. ``esummary`` is normalised too, so a
caller that already holds summaries is not forced into a second round trip,
but the resulting record is explicitly marked as lower-confidence.

Dating a PubMed record
----------------------
A single article carries up to five dates and they routinely disagree by
months. PMID 33138715, one of this connector's fixtures, was published
electronically on 2020-11-02 and appeared in the January 2021 print issue: an
``available_at`` of 2021-01-27 would hide it from every cutoff in the two
months during which it was, in fact, on the internet and citable.

``available_at`` is therefore the **minimum** over every date on this record
that denotes public availability:

* ``<PubmedData><History>`` entries with ``PubStatus`` in
  ``entrez`` / ``pubmed`` / ``epublish`` / ``aheadofprint`` / ``ecollection``;
* ``<Article><ArticleDate DateType="Electronic">``;
* the journal issue ``<PubDate>``.

``received``, ``accepted``, ``revised`` and ``submitted`` history entries are
deliberately excluded. They are earlier, and using them would be the classic
leak: a manuscript under review is not public, and no analyst could have acted
on it.

The journal issue date participates in the minimum rather than serving only as
a last-resort fallback. For back-filled older records the Entrez date is the
date NLM digitised the citation, which can be decades after the issue was on
library shelves; taking the minimum keeps the 1975 paper dated 1975.

Partial dates resolve to the first day of the period they name — see
:mod:`trs.connectors` for the rule and its residual risk.

Rate limits
-----------
NCBI publishes 3 requests/second without an API key and 10/second with one, and
enforces it (this connector was developed against a live 429). The throttle
interval is chosen per instance from :data:`trs.config.settings`, because the
limit is a property of the runtime credentials, not of the class.
"""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Any, ClassVar

from trs.config import settings
from trs.connectors import (
    earliest_date,
    first_day_of_period,
    iso_date,
    month_number,
    parse_partial_date,
)
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

#: History statuses that denote *public* availability. Anything describing the
#: manuscript's journey through peer review is excluded on purpose; see the
#: module docstring.
PUBLIC_HISTORY_STATUSES: frozenset[str] = frozenset(
    {"entrez", "pubmed", "epublish", "aheadofprint", "ecollection"}
)

#: PubMed publication types that determine :class:`PublicationStatus`. Anything
#: not listed leaves the status to the journal-presence fallback rather than
#: being forced into a bucket.
_PREPRINT_TYPES: frozenset[str] = frozenset({"Preprint"})
_ABSTRACT_TYPES: frozenset[str] = frozenset({"Congress", "Congresses"})

#: efetch is a GET; NCBI tolerates long identifier lists but a bounded batch
#: keeps URLs and cached payloads a sane size.
_EFETCH_BATCH = 200


@register_connector
class PubMedConnector(Connector):
    """Adapter for NCBI E-utilities over the PubMed database."""

    name: ClassVar[str] = "pubmed"
    version: ClassVar[str] = "1.0.0"
    display_name: ClassVar[str] = "PubMed (NCBI E-utilities)"
    homepage: ClassVar[str] = "https://pubmed.ncbi.nlm.nih.gov/"
    license_note: ClassVar[str] = (
        "PubMed/MEDLINE citation records are produced by the U.S. National "
        "Library of Medicine and are free of copyright restrictions; abstracts "
        "remain the copyright of their publishers. See "
        "https://www.nlm.nih.gov/databases/download/terms_and_conditions.html"
    )
    min_request_interval: ClassVar[float] = 0.34

    _NO_KEY_INTERVAL: ClassVar[float] = 0.34  # 3 requests/second
    _WITH_KEY_INTERVAL: ClassVar[float] = 0.11  # 10 requests/second
    _BASE: ClassVar[str] = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The published ceiling depends on whether a key is configured, so the
        # interval is an instance decision rather than a class constant.
        self.min_request_interval = (  # type: ignore[misc]
            self._WITH_KEY_INTERVAL if settings.ncbi_api_key else self._NO_KEY_INTERVAL
        )

    # ---- plumbing ----------------------------------------------------------

    def base_url(self) -> str:
        return self._BASE

    def _identity_params(self) -> dict[str, str]:
        """NCBI asks every client to identify itself; the key also lifts the
        rate limit. Both are optional, and the connector works without them."""
        params: dict[str, str] = {"tool": settings.ncbi_tool}
        if settings.ncbi_email:
            params["email"] = settings.ncbi_email
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key
        return params

    def health(self) -> ConnectorHealth:
        if not settings.allow_live_network:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DISABLED, checked_at=utcnow(),
                detail="live network disabled by configuration (TRS_ALLOW_LIVE_NETWORK)",
                endpoint=self._BASE,
            )
        endpoint = f"{self._BASE}/esearch.fcgi"
        started = utcnow()
        try:
            payload = self.request(
                "GET", endpoint,
                params={**self._identity_params(), "db": "pubmed", "term": "1[uid]",
                        "retmax": 0, "retmode": "json"},
                use_cache=False, allow_stale_on_error=False,
            )
        except ConnectorError as exc:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.UNREACHABLE,
                checked_at=utcnow(), detail=str(exc), endpoint=endpoint,
            )
        latency = (utcnow() - started).total_seconds() * 1000
        detail = "esearch responded"
        status = HealthStatus.OK
        if not settings.ncbi_api_key:
            # Usable, but at a third of the throughput and more 429-prone.
            status = HealthStatus.DEGRADED
            detail += "; no TRS_NCBI_API_KEY configured, throttled to 3 requests/second"
        _ = payload
        return ConnectorHealth(
            connector=self.name, status=status, checked_at=utcnow(),
            latency_ms=latency, detail=detail, endpoint=endpoint,
        )

    # ---- fetch -------------------------------------------------------------

    def search(
        self,
        term: str,
        *,
        max_results: int = 50,
        mindate: str | dt.date | None = None,
        maxdate: str | dt.date | None = None,
    ) -> list[RawPayload]:
        """Resolve ``term`` to PMIDs and fetch the full records for them.

        Returns the esearch payload followed by one efetch payload per batch.
        Every payload is returned, including esearch's, because the raw
        landing zone stores what was actually requested — a later audit needs
        to see the query that produced the identifier list, not only the
        records it happened to yield.

        ``mindate``/``maxdate`` filter on publication date (``datetype=pdat``).
        They are a server-side convenience, **not** a temporal-integrity
        control: as-of filtering happens on ``available_at`` after
        normalisation, because the registry's idea of a publication date is not
        the platform's idea of when a fact became knowable.
        """
        params: dict[str, Any] = {
            **self._identity_params(),
            "db": "pubmed",
            "term": term,
            "retmax": max(0, min(max_results, 10000)),
            "retmode": "json",
            "sort": "date",
        }
        if mindate is not None or maxdate is not None:
            params["datetype"] = "pdat"
            if mindate is not None:
                params["mindate"] = mindate.isoformat() if isinstance(mindate, dt.date) else mindate
            if maxdate is not None:
                params["maxdate"] = maxdate.isoformat() if isinstance(maxdate, dt.date) else maxdate

        search_payload = self.request("GET", f"{self._BASE}/esearch.fcgi", params=params)
        try:
            pmids = list(search_payload.json()["esearchresult"]["idlist"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ConnectorError(f"{self.name}: unparseable esearch response for {term!r}") from exc

        payloads = [search_payload]
        for start in range(0, len(pmids), _EFETCH_BATCH):
            payloads.append(self.fetch_articles(pmids[start : start + _EFETCH_BATCH]))
        return payloads

    def fetch_articles(self, pmids: Sequence[str]) -> RawPayload:
        """Full MEDLINE XML for up to :data:`_EFETCH_BATCH` PMIDs."""
        if not pmids:
            raise ConnectorError(f"{self.name}: fetch_articles called with no PMIDs")
        if len(pmids) > _EFETCH_BATCH:
            raise ConnectorError(
                f"{self.name}: {len(pmids)} PMIDs exceeds the {_EFETCH_BATCH} per-request batch"
            )
        return self.request(
            "GET", f"{self._BASE}/efetch.fcgi",
            params={**self._identity_params(), "db": "pubmed",
                    "id": ",".join(str(p) for p in pmids), "retmode": "xml"},
        )

    def fetch_summaries(self, pmids: Sequence[str]) -> RawPayload:
        """Compact JSON summaries. Cheaper than efetch, but no abstract, no
        MeSH headings and no history block — so the records it yields are
        marked ``moderate`` confidence."""
        if not pmids:
            raise ConnectorError(f"{self.name}: fetch_summaries called with no PMIDs")
        return self.request(
            "GET", f"{self._BASE}/esummary.fcgi",
            params={**self._identity_params(), "db": "pubmed",
                    "id": ",".join(str(p) for p in pmids), "retmode": "json"},
        )

    # ---- normalise ---------------------------------------------------------

    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        """Pure: dispatches on the payload's shape, never on the request."""
        text = payload.text().lstrip()
        if text.startswith("<"):
            return self._normalise_efetch(payload)
        try:
            body = payload.json()
        except ValueError:
            return []
        if "esearchresult" in body:
            # An identifier list is not a citable object; it carries no date of
            # its own and must not become a record.
            return []
        if "result" in body:
            return self._normalise_esummary(payload, body)
        return []

    # -- efetch XML --

    def _normalise_efetch(self, payload: RawPayload) -> list[NormalisedRecord]:
        try:
            root = ET.fromstring(payload.text())
        except ET.ParseError as exc:
            raise ConnectorError(f"{self.name}: malformed efetch XML ({exc})") from exc

        records: list[NormalisedRecord] = []
        for index, article in enumerate(root.iter("PubmedArticle")):
            record = self._article_record(payload, article, index)
            if record is not None:
                records.append(record)
        return records

    def _article_record(
        self, payload: RawPayload, article: ET.Element, index: int
    ) -> NormalisedRecord | None:
        pmid = _text(article.find("./MedlineCitation/PMID"))
        if not pmid:
            return None

        warnings: list[str] = []
        medline = article.find("./MedlineCitation")
        art = article.find("./MedlineCitation/Article")
        if medline is None or art is None:
            return None

        history = _history_dates(article)
        public_history = [
            date for status, date in history.items() if status in PUBLIC_HISTORY_STATUSES
        ]
        article_date = _element_date(art.find("./ArticleDate"))
        journal_date = _journal_pub_date(art)

        available_at = earliest_date(*public_history, article_date, journal_date)
        if available_at is None:
            # No date at all means the record cannot be placed in time, and a
            # record that cannot be placed in time is not admissible.
            return None

        published_on = journal_date or article_date
        pub_types = [
            t for t in (_text(e) for e in art.findall("./PublicationTypeList/PublicationType")) if t
        ]
        status = _publication_status(pub_types, has_journal=art.find("./Journal") is not None)
        if status is PublicationStatus.UNKNOWN:
            warnings.append(
                "publication_status left unknown: no recognised PublicationType and no "
                f"<Journal> element (types seen: {pub_types or 'none'})"
            )

        doi = _article_id(article, "doi") or _elocation_doi(art)
        pmcid = _article_id(article, "pmc")

        body: dict[str, Any] = {
            "pmid": pmid,
            "doi": doi,
            "pmcid": pmcid,
            "title": _text(art.find("./ArticleTitle")),
            "abstract": _abstract(art),
            "journal": _text(art.find("./Journal/Title")),
            "journal_abbrev": _text(art.find("./Journal/ISOAbbreviation")),
            "authors": _authors(art),
            "publication_types": pub_types,
            "published_on": iso_date(published_on),
            "mesh_terms": [
                t for t in (
                    _text(e) for e in medline.findall("./MeshHeadingList/MeshHeading/DescriptorName")
                ) if t
            ],
            "publication_status": str(status),
            "source_type": str(
                SourceType.PREPRINT if status is PublicationStatus.PREPRINT
                else SourceType.JOURNAL_ARTICLE
            ),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            # "Retracted Publication" marks the article that *was* retracted;
            # "Retraction of Publication" marks the notice that retracted it.
            # Conflating them would flag the notice as bad science.
            "is_retracted": "Retracted Publication" in pub_types,
            "is_retraction_notice": "Retraction of Publication" in pub_types,
            "history_dates": {k: iso_date(v) for k, v in sorted(history.items())},
            "article_date_electronic": iso_date(article_date),
            "journal_pub_date": iso_date(journal_date),
        }

        provenance = {
            **self.provenance_stub(payload),
            "provenance_method": str(ProvenanceMethod.RULE_BASED_PARSE),
            "record_index": index,
            "available_at_basis": _availability_basis(
                available_at, history, article_date, journal_date
            ),
        }
        return NormalisedRecord(
            kind="source",
            external_id=f"pmid:{pmid}",
            available_at=available_at,
            payload=body,
            provenance=provenance,
            confidence=str(ConfidenceLevel.HIGH),
            warnings=warnings,
        )

    # -- esummary JSON --

    def _normalise_esummary(
        self, payload: RawPayload, body: dict[str, Any]
    ) -> list[NormalisedRecord]:
        result = body.get("result") or {}
        uids = result.get("uids") or []
        records: list[NormalisedRecord] = []

        for index, uid in enumerate(uids):
            entry = result.get(uid)
            if not isinstance(entry, dict):
                continue
            # esummary exposes only the electronic and issue dates; there is no
            # history block, so the minimum is taken over a smaller candidate
            # set and the record is marked down accordingly.
            epub = parse_partial_date(entry.get("epubdate"))
            pub = parse_partial_date(entry.get("pubdate"))
            sort_pub = parse_partial_date(entry.get("sortpubdate"))
            available_at = earliest_date(epub, pub, sort_pub)
            if available_at is None:
                continue

            doi = entry.get("elocationid") or ""
            if isinstance(doi, str) and doi.lower().startswith("doi:"):
                doi = doi[4:].strip()
            else:
                doi = next(
                    (a.get("value") for a in entry.get("articleids") or []
                     if a.get("idtype") == "doi"),
                    None,
                )

            pub_types = list(entry.get("pubtype") or [])
            status = _publication_status(pub_types, has_journal=bool(entry.get("source")))
            body_out: dict[str, Any] = {
                "pmid": str(uid),
                "doi": doi,
                "title": entry.get("title"),
                "abstract": None,
                "journal": entry.get("fulljournalname") or entry.get("source"),
                "authors": [
                    {"name": a.get("name")} for a in entry.get("authors") or [] if a.get("name")
                ],
                "publication_types": pub_types,
                "published_on": iso_date(pub or sort_pub),
                "mesh_terms": [],
                "publication_status": str(status),
                "source_type": str(
                    SourceType.PREPRINT if status is PublicationStatus.PREPRINT
                    else SourceType.JOURNAL_ARTICLE
                ),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            }
            records.append(
                NormalisedRecord(
                    kind="source",
                    external_id=f"pmid:{uid}",
                    available_at=available_at,
                    payload=body_out,
                    provenance={
                        **self.provenance_stub(payload),
                        "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
                        "record_index": index,
                        "available_at_basis": "esummary epubdate/pubdate minimum",
                    },
                    confidence=str(ConfidenceLevel.MODERATE),
                    warnings=[
                        "Derived from esummary: no abstract, no MeSH headings and no Entrez "
                        "history block, so available_at may be later than the efetch record "
                        "for the same PMID. Prefer efetch for evidence extraction."
                    ],
                )
            )
        return records


# ─────────────────────────────────────────────────────────────────────────────
# XML helpers (module-level and pure, so they are unit-testable in isolation)
# ─────────────────────────────────────────────────────────────────────────────

def _text(element: ET.Element | None) -> str | None:
    """Flattened text of an element including inline markup (``<i>``, ``<sup>``)."""
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def _element_date(element: ET.Element | None) -> dt.date | None:
    """Parse a ``<Year>/<Month>/<Day>`` triple, tolerating month names."""
    if element is None:
        return None
    year_text = _text(element.find("Year"))
    if not year_text or not year_text.isdigit():
        return None
    month = month_number(_text(element.find("Month")))
    day_text = _text(element.find("Day"))
    day = int(day_text) if day_text and day_text.isdigit() else None
    return first_day_of_period(int(year_text), month, day)


def _history_dates(article: ET.Element) -> dict[str, dt.date]:
    """Every ``<PubMedPubDate>`` keyed by its ``PubStatus``, public or not.

    The full set is retained in the payload for audit; only the public subset
    feeds ``available_at``.
    """
    out: dict[str, dt.date] = {}
    for entry in article.findall("./PubmedData/History/PubMedPubDate"):
        status = (entry.get("PubStatus") or "").strip().lower()
        parsed = _element_date(entry)
        if status and parsed is not None:
            out[status] = parsed
    return out


def _journal_pub_date(article: ET.Element) -> dt.date | None:
    """Journal issue date, handling ``<Season>`` and free-text ``<MedlineDate>``."""
    pub_date = article.find("./Journal/JournalIssue/PubDate")
    if pub_date is None:
        return None
    medline_date = _text(pub_date.find("MedlineDate"))
    if medline_date:
        # Free text such as "1998 Dec-1999 Jan"; the shared parser takes the
        # first year and month it names.
        return parse_partial_date(medline_date)
    year_text = _text(pub_date.find("Year"))
    if not year_text or not year_text.isdigit():
        return None
    month = month_number(_text(pub_date.find("Month")) or _text(pub_date.find("Season")))
    day_text = _text(pub_date.find("Day"))
    day = int(day_text) if day_text and day_text.isdigit() else None
    return first_day_of_period(int(year_text), month, day)


def _article_id(article: ET.Element, id_type: str) -> str | None:
    for entry in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if (entry.get("IdType") or "").lower() == id_type:
            return _text(entry)
    return None


def _elocation_doi(article: ET.Element) -> str | None:
    for entry in article.findall("./ELocationID"):
        if (entry.get("EIdType") or "").lower() == "doi":
            return _text(entry)
    return None


def _abstract(article: ET.Element) -> str | None:
    """Structured abstracts keep their section labels; losing them would make
    "Results" indistinguishable from "Background" downstream."""
    parts: list[str] = []
    for node in article.findall("./Abstract/AbstractText"):
        chunk = _text(node)
        if not chunk:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {chunk}" if label else chunk)
    return "\n".join(parts) or None


def _authors(article: ET.Element) -> list[dict[str, Any]]:
    authors: list[dict[str, Any]] = []
    for node in article.findall("./AuthorList/Author"):
        collective = _text(node.find("CollectiveName"))
        if collective:
            authors.append({"name": collective, "is_collective": True})
            continue
        last = _text(node.find("LastName"))
        fore = _text(node.find("ForeName")) or _text(node.find("Initials"))
        if not last and not fore:
            continue
        name = " ".join(p for p in (fore, last) if p)
        entry: dict[str, Any] = {"name": name}
        affiliation = _text(node.find("./AffiliationInfo/Affiliation"))
        if affiliation:
            entry["affiliation"] = affiliation
        orcid = next(
            (_text(i) for i in node.findall("Identifier") if (i.get("Source") or "") == "ORCID"),
            None,
        )
        if orcid:
            entry["orcid"] = orcid
        authors.append(entry)
    return authors


def _publication_status(pub_types: Sequence[str], *, has_journal: bool) -> PublicationStatus:
    """Map PubMed publication types onto the platform vocabulary.

    Deliberately narrow. PubMed indexes preprints explicitly, so that case is
    certain. Everything carrying a ``<Journal>`` element is indexed from a
    journal and treated as peer-reviewed. Anything else is left ``UNKNOWN``
    rather than assigned a plausible status, because ``UNKNOWN`` and a wrong
    status are not equally recoverable downstream.
    """
    types = set(pub_types)
    if types & _PREPRINT_TYPES:
        return PublicationStatus.PREPRINT
    if types & _ABSTRACT_TYPES:
        return PublicationStatus.CONFERENCE_ABSTRACT
    if has_journal:
        return PublicationStatus.PEER_REVIEWED
    return PublicationStatus.UNKNOWN


def _availability_basis(
    chosen: dt.date,
    history: dict[str, dt.date],
    article_date: dt.date | None,
    journal_date: dt.date | None,
) -> str:
    """Name the field(s) that produced ``available_at``, for the provenance UI."""
    sources = [
        f"history:{status}" for status, date in sorted(history.items())
        if status in PUBLIC_HISTORY_STATUSES and date == chosen
    ]
    if article_date == chosen:
        sources.append("article_date_electronic")
    if journal_date == chosen:
        sources.append("journal_pub_date")
    return "min(" + ", ".join(sources) + ")" if sources else "unattributed"
