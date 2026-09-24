"""Ingestion connectors for the public biomedical sources the platform reads.

Importing this package imports every connector module, which is what populates
the registry in :mod:`trs.connectors.base`. Callers therefore reach a connector
through ``get_connector("pubmed")`` rather than by importing a class, so a
source can be swapped or retired without touching call sites.

Why the date primitives live in the package root
------------------------------------------------
``available_at`` is the one field every connector must agree on: it is the
single input to as-of filtering (:mod:`trs.features.temporal`), and a
disagreement between two connectors about what a date *means* is a leakage bug
that no individual connector's tests would catch. The primitives are therefore
defined once, here, above the connector imports. Each connector module does
``from trs.connectors import parse_partial_date``; by the time a connector's
module body runs, this module is already in ``sys.modules`` with these names
bound, which is what makes the arrangement safe.

The first-day rule
------------------
Sources report dates at mixed precision: ClinicalTrials.gov gives ``2015-03``,
PubMed gives ``2015 Mar`` or a bare ``1975``, Crossref gives ``[[1998, 2]]``.
A partial date names a *period*, not an instant. We resolve it to the **first
day** of that period — ``2015 Mar`` becomes ``2015-03-01``, ``1975`` becomes
``1975-01-01``.

The rationale is that ``available_at`` answers "from when could a diligent
analyst have known this?", and the earliest instant consistent with the source
is the earliest answer the source supports. Resolving to a later day (the last
day of the period, say) would claim the fact was unknowable during weeks when
it may well have been public, and would wrongly exclude it from a cutoff it
genuinely predates.

*Residual risk, stated plainly:* the rule is anti-conservative in the other
direction. For a bare year it can place a fact up to eleven months earlier than
it was truly knowable, which is a leak of at most that size for the coarsest
records. The exposure is bounded by date precision, not by the rule, so the
mitigation is to prefer precise date fields where a source offers several —
which every connector here does, and documents where it does not.

Seasons
-------
PubMed emits ``<Season>Spring</Season>`` and ranges like ``Mar-Apr``. A range
resolves to its first month. A season resolves to the first month of the
northern-hemisphere season a publisher means by it, except ``Winter``, which
straddles a year boundary and is resolved to January — again the earliest day
consistent with the label.
"""

from __future__ import annotations

import datetime as dt
import re

__all__ = [
    "MONTHS",
    "SEASON_FIRST_MONTH",
    "earliest_date",
    "first_day_of_period",
    "iso_date",
    "month_number",
    "parse_partial_date",
]


MONTHS: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

#: First month of each season label. ``winter`` straddles a year boundary, so
#: it resolves to January — the earliest day the label can mean within the
#: stated year.
SEASON_FIRST_MONTH: dict[str, int] = {
    "winter": 1,
    "spring": 3,
    "summer": 6,
    "fall": 9,
    "autumn": 9,
}

_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?:[T ]|$)")
_NUMERIC = re.compile(r"^(\d{4})[/.](\d{1,2})(?:[/.](\d{1,2}))?")
_YEAR_FIRST = re.compile(
    r"^(\d{4})\s+([A-Za-z]+)(?:\s*[-–/]\s*[A-Za-z]+)?(?:\s+(\d{1,2}))?"
)
_MONTH_FIRST = re.compile(r"^([A-Za-z]+)\.?\s+(?:(\d{1,2})\s*,?\s+)?(\d{4})\b")
_YEAR_ONLY = re.compile(r"^(\d{4})\b")


def month_number(token: str | None) -> int | None:
    """Month index for a month name, month range or season label.

    Returns ``None`` for anything unrecognised rather than guessing, so a
    caller can fall back to the first month of the year explicitly instead of
    silently inheriting a wrong month.
    """
    if not token:
        return None
    key = token.strip().lower().rstrip(".")
    # A range such as "Mar-Apr" or "Nov-Dec" names a period; take its start.
    key = re.split(r"[-–/]", key)[0].strip()
    if key.isdigit():
        value = int(key)
        return value if 1 <= value <= 12 else None
    return MONTHS.get(key) or SEASON_FIRST_MONTH.get(key)


def first_day_of_period(
    year: int | None, month: int | None = None, day: int | None = None
) -> dt.date | None:
    """Resolve a (year, month?, day?) triple to the first day it can denote.

    An impossible day-of-month (sources do emit ``Feb 30``) degrades to the
    first of that month rather than raising, because a malformed day is not a
    reason to discard an otherwise usable year and month.
    """
    if year is None or not (1000 <= year <= 9999):
        return None
    try:
        return dt.date(year, month or 1, day or 1)
    except ValueError:
        try:
            return dt.date(year, month or 1, 1)
        except ValueError:
            return dt.date(year, 1, 1)


def parse_partial_date(value: str | int | None) -> dt.date | None:
    """Parse any date string these sources emit, at whatever precision.

    Handles ISO dates and datetimes (``2015-03-04``, ``2015-03``,
    ``2015-03-04T12:00:00Z``), PubMed's ``2015 Mar 04`` / ``2015 Mar`` /
    ``2015`` / ``1975 Mar-Apr`` / ``1975 Spring``, Entrez's ``2015/03/04
    00:00``, and month-first forms. Returns ``None`` — never a guess and never
    "today" — when nothing recognisable is present.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return first_day_of_period(value)

    text = value.strip()
    if not text:
        return None

    match = _ISO_DATE.match(text)
    if match:
        year, month, day = match.groups()
        return first_day_of_period(int(year), int(month), int(day) if day else None)

    match = _NUMERIC.match(text)
    if match:
        year, month, day = match.groups()
        return first_day_of_period(int(year), int(month), int(day) if day else None)

    match = _YEAR_FIRST.match(text)
    if match:
        year, month_token, day = match.groups()
        return first_day_of_period(
            int(year), month_number(month_token), int(day) if day else None
        )

    match = _MONTH_FIRST.match(text)
    if match:
        month_token, day, year = match.groups()
        return first_day_of_period(
            int(year), month_number(month_token), int(day) if day else None
        )

    match = _YEAR_ONLY.match(text)
    if match:
        return first_day_of_period(int(match.group(1)))

    return None


def earliest_date(*candidates: dt.date | None) -> dt.date | None:
    """Earliest non-null date, or ``None`` when every candidate is missing.

    ``available_at`` is defined as the earliest date on which a fact was
    public, so connectors that can observe several public dates for one record
    (PubMed offers up to five) take their minimum rather than picking one by
    priority. Picking by priority would report a later date whenever the
    preferred field happened to be the later one.
    """
    known = [c for c in candidates if c is not None]
    return min(known) if known else None


def iso_date(value: dt.date | None) -> str | None:
    """ISO-8601 rendering for JSON payloads, preserving ``None``."""
    return value.isoformat() if value is not None else None


# Connector modules are imported last, so that the primitives above are already
# bound when each module body executes. Importing them here is also what
# registers every connector; see the module docstring.
from trs.connectors.chembl import ChemblConnector  # noqa: E402
from trs.connectors.crossref import CrossrefConnector  # noqa: E402
from trs.connectors.ctgov import ClinicalTrialsGovConnector  # noqa: E402
from trs.connectors.openalex import OpenAlexConnector  # noqa: E402
from trs.connectors.opentargets import OpenTargetsConnector  # noqa: E402
from trs.connectors.pubmed import PubMedConnector  # noqa: E402

__all__ += [
    "ChemblConnector",
    "ClinicalTrialsGovConnector",
    "CrossrefConnector",
    "OpenAlexConnector",
    "OpenTargetsConnector",
    "PubMedConnector",
]
