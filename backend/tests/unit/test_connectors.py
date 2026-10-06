"""Connector normalisation tests.

Every assertion here runs offline against payloads recorded from the live APIs
(or, for ClinicalTrials.gov, against fixtures whose shape follows the published
schema — that API is unreachable from the build network and the fixture files
say so in their own ``_fixture_note``).

The properties under test are the ones that would silently corrupt the platform
if they regressed:

* ``normalise`` is pure — no network, no clock, no database.
* Every emitted record carries a resolvable ``available_at``.
* ``available_at`` reflects when a fact became *knowable*, which for a trial's
  stop reason is the last-update date, not the first-posted date.
* Every categorical value is a member of its declared controlled vocabulary.
* Open Targets records are flagged with a temporal caveat, because their scores
  cannot be back-dated.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from tests.conftest import CONNECTOR_FIXTURES
from trs.connectors.base import (
    Connector,
    NormalisedRecord,
    RawPayload,
    available_connectors,
    connector_catalogue,
    get_connector,
)
from trs.ontology import enums as E

ALL_CONNECTORS = ("pubmed", "openalex", "crossref", "opentargets", "chembl", "ctgov")


def make_payload(fixture: str) -> RawPayload:
    """Rehydrate a recorded fixture into the RawPayload the connector saw.

    Fixtures are stored as complete request/response envelopes — endpoint,
    params, status, media type, retrieval timestamp and body — rather than bare
    response bodies. That matters: ``normalise`` legitimately reads
    ``fetched_at`` (a score that cannot be back-dated is dated to retrieval) and
    ``endpoint``, so replaying only the body would test a different function
    from the one that runs in production.
    """
    envelope = json.loads((CONNECTOR_FIXTURES / fixture).read_text())
    if "content_text" in envelope:
        body = envelope["content_text"].encode()
    else:
        body = json.dumps(envelope["content"]).encode()
    return RawPayload(
        endpoint=envelope["endpoint"],
        params=envelope.get("params") or {},
        status_code=envelope.get("status_code", 200),
        media_type=envelope.get("media_type", "application/json"),
        content=body,
        fetched_at=dt.datetime.fromisoformat(envelope["fetched_at"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def test_all_six_connectors_are_registered() -> None:
    assert set(available_connectors()) == set(ALL_CONNECTORS)


def test_catalogue_declares_provenance_metadata() -> None:
    catalogue = connector_catalogue()
    assert len(catalogue) == len(ALL_CONNECTORS), (
        "an empty or shrunken catalogue would make the per-entry checks below "
        f"pass vacuously; got {[e['name'] for e in catalogue]}"
    )
    for entry in catalogue:
        assert entry["display_name"], f"{entry['name']} has no display name"
        assert entry["homepage"], f"{entry['name']} has no homepage for the provenance UI"
        assert entry["version"]


@pytest.mark.parametrize("name", ALL_CONNECTORS)
def test_connector_is_constructible_and_declares_base_url(name: str) -> None:
    with get_connector(name) as c:
        assert isinstance(c, Connector)
        assert c.base_url().startswith("http")


# ─────────────────────────────────────────────────────────────────────────────
# Purity and the available_at contract
# ─────────────────────────────────────────────────────────────────────────────

FIXTURE_BY_CONNECTOR = {
    "pubmed": "pubmed_efetch_articles.json",
    "openalex": "openalex_works_search.json",
    "crossref": "crossref_work.json",
    "opentargets": "opentargets_target_disease_association.json",
    "chembl": "chembl_molecule.json",
    "ctgov": "ctgov_studies_search.json",
}


@pytest.mark.parametrize("name,fixture", sorted(FIXTURE_BY_CONNECTOR.items()))
def test_normalise_emits_records_with_resolvable_availability(name: str, fixture: str) -> None:
    """Every record must be placeable in time, or it must not be emitted.

    A record without ``available_at`` cannot be filtered by the as-of engine,
    so admitting one would open a silent leakage path.
    """
    with get_connector(name) as c:
        records = c.normalise(make_payload(fixture))
    assert records, f"{name} produced no records from {fixture}"
    for r in records:
        assert isinstance(r, NormalisedRecord)
        assert isinstance(r.available_at, dt.date), (
            f"{name}: record {r.external_id} has a non-date available_at"
        )
        assert r.kind, f"{name}: record {r.external_id} has no kind"
        assert r.provenance.get("connector") == name


@pytest.mark.parametrize("name,fixture", sorted(FIXTURE_BY_CONNECTOR.items()))
def test_normalise_is_deterministic(name: str, fixture: str) -> None:
    """Pure function: same payload in, identical records out.

    Catches accidental clock reads and dict-ordering dependence, both of which
    would break reproduction of a historical prediction.
    """
    with get_connector(name) as c:
        first = c.normalise(make_payload(fixture))
        second = c.normalise(make_payload(fixture))
    assert [(r.kind, r.external_id, r.available_at) for r in first] == [
        (r.kind, r.external_id, r.available_at) for r in second
    ]


@pytest.mark.parametrize("name,fixture", sorted(FIXTURE_BY_CONNECTOR.items()))
def test_normalise_does_not_read_the_wall_clock(name: str, fixture: str) -> None:
    """``available_at`` must derive from the payload, never from "now".

    Testing this by asserting "not equal to today" gives a false positive for
    records that are *legitimately* dated to retrieval — an Open Targets
    association score cannot be back-dated, so it is correctly stamped with
    ``fetched_at``, which in a freshly recorded fixture is today.

    The real question is whether the date tracks the payload or the clock. So
    the same fixture is replayed with its ``fetched_at`` shifted back a decade.
    A payload-derived date must not move. A retrieval-derived date must move
    *with fetched_at* — and must also be flagged with a temporal caveat, since
    dating a fact to retrieval is a statement that it cannot be placed earlier.
    """
    original = make_payload(fixture)
    shifted = RawPayload(
        endpoint=original.endpoint,
        params=original.params,
        status_code=original.status_code,
        media_type=original.media_type,
        content=original.content,
        fetched_at=original.fetched_at - dt.timedelta(days=3650),
    )

    with get_connector(name) as c:
        before = {r.external_id: r for r in c.normalise(original)}
        after = {r.external_id: r for r in c.normalise(shifted)}

    assert before.keys() == after.keys()
    for key, rec in before.items():
        moved = after[key].available_at != rec.available_at
        if not moved:
            continue
        # The date tracked fetched_at. That is only defensible for a record
        # that declares it cannot be back-dated.
        assert after[key].available_at == shifted.fetched_at.date(), (
            f"{name}: {key} moved with the retrieval timestamp but not to it"
        )
        assert rec.provenance.get("temporal_caveat") is True, (
            f"{name}: {key} is dated to retrieval but carries no temporal "
            "caveat; a retrieval-dated record must declare that it cannot be "
            "used for an earlier cutoff"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Controlled vocabularies
# ─────────────────────────────────────────────────────────────────────────────

ENUM_FIELDS = {
    "publication_status": E.PublicationStatus,
    "phase": E.DevelopmentPhase,
    "modality": E.Modality,
    "genetic_evidence_tier": E.GeneticEvidenceTier,
    "causal_support": E.CausalSupport,
    "source_type": E.SourceType,
}


@pytest.mark.parametrize("name,fixture", sorted(FIXTURE_BY_CONNECTOR.items()))
def test_emitted_categoricals_are_valid_enum_members(name: str, fixture: str) -> None:
    with get_connector(name) as c:
        records = c.normalise(make_payload(fixture))
    assert records, f"{name} produced no records from {fixture}"
    valid = {field: {str(m) for m in enum} for field, enum in ENUM_FIELDS.items()}
    checked = 0
    for r in records:
        for field, allowed in valid.items():
            value = r.payload.get(field)
            if value is None:
                continue
            checked += 1
            assert str(value) in allowed, (
                f"{name}: {field}={value!r} is not a member of "
                f"{ENUM_FIELDS[field].__name__}"
            )
    # Without this the test passes for a connector that stopped emitting
    # categoricals altogether, which is exactly when it should fail.
    assert checked, (
        f"{name}: no categorical field was checked. Either the connector "
        f"stopped populating {sorted(ENUM_FIELDS)} or the field names moved."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source-specific temporal semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_ctgov_stop_reason_is_dated_to_the_update_not_the_posting() -> None:
    """A terminated trial's *reason* became knowable when it was posted.

    Dating the stop reason to the trial's first-posted date would make a 2018
    failure appear knowable in 2013 — a textbook look-ahead leak, and one that
    would inflate every historical-performance feature.
    """
    with get_connector("ctgov") as c:
        records = c.normalise(make_payload("ctgov_study_terminated.json"))

    trials = [r for r in records if r.kind == "trial"]
    updates = [r for r in records if r.kind == "trial_status_update"]
    assert trials, "expected a trial record"
    assert updates, "a terminated trial with why_stopped must emit a status-update record"

    trial, update = trials[0], updates[0]
    assert update.payload.get("why_stopped"), "status update must carry the stop reason"
    assert update.available_at > trial.available_at, (
        "the stop reason must become available strictly after the registry "
        f"record was first posted (trial={trial.available_at}, "
        f"update={update.available_at})"
    )


def test_opentargets_records_carry_a_temporal_caveat() -> None:
    """Open Targets scores are computed from *current* evidence.

    They cannot be back-dated, so a record used for a historical cutoff earlier
    than its retrieval date would leak. The connector must flag this.
    """
    with get_connector("opentargets") as c:
        records = c.normalise(make_payload("opentargets_target_disease_association.json"))
    assert records
    for r in records:
        assert r.provenance.get("temporal_caveat") is True, (
            "Open Targets records must set provenance['temporal_caveat']"
        )
        assert r.warnings, "Open Targets records must carry an explicit warning"
        assert any("back-dat" in w or "retrieval" in w for w in r.warnings)


def test_opentargets_available_at_is_the_retrieval_date() -> None:
    payload = make_payload("opentargets_target_disease_association.json")
    with get_connector("opentargets") as c:
        records = c.normalise(payload)
    assert records, "no records to check — the loop below would pass vacuously"
    for r in records:
        assert r.available_at == payload.fetched_at.date(), (
            "an un-back-datable score must be dated to retrieval, not earlier"
        )


def test_pubmed_partial_dates_resolve_to_the_earliest_possible_day() -> None:
    """'2015' and '2015 Mar' must resolve to the first day of the known period.

    Resolving to a later day would understate how long a fact has been
    knowable and could wrongly exclude it from a cutoff it actually predates.
    """
    with get_connector("pubmed") as c:
        records = c.normalise(make_payload("pubmed_efetch_dates.json"))
    assert records
    for r in records:
        assert isinstance(r.available_at, dt.date)
    # Any record whose source month/day was unknown must land on a period start.
    for r in records:
        basis = r.provenance.get("availability_basis", "")
        if "year_only" in str(basis):
            assert (r.available_at.month, r.available_at.day) == (1, 1)
        elif "month_only" in str(basis):
            assert r.available_at.day == 1


def test_retraction_is_carried_through_normalisation() -> None:
    """A retracted paper must not silently enter the evidence base as normal."""
    with get_connector("openalex") as c:
        records = c.normalise(make_payload("openalex_work_retracted.json"))
    assert records
    assert any(r.payload.get("is_retracted") for r in records), (
        "the retracted-work fixture must normalise with is_retracted set"
    )


def test_chembl_molecule_normalises_to_a_compound_with_a_modality() -> None:
    with get_connector("chembl") as c:
        records = c.normalise(make_payload("chembl_molecule.json"))
    compounds = [r for r in records if r.kind == "compound"]
    assert compounds
    c0 = compounds[0]
    assert c0.payload.get("chembl_id", "").startswith("CHEMBL")
    assert str(c0.payload.get("modality")) in {str(m) for m in E.Modality}


# ─────────────────────────────────────────────────────────────────────────────
# Health reporting
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_CONNECTORS)
def test_health_reports_a_status_without_raising(name: str) -> None:
    """Health must degrade, never explode.

    With live network disabled in the test environment every connector should
    report a status rather than raise — an unreachable source is a state the
    platform models, not an error it crashes on.
    """
    with get_connector(name) as c:
        health = c.health()
    assert health.connector == name
    assert str(health.status) in {
        "ok", "degraded", "unreachable", "not_configured", "disabled",
    }
    assert isinstance(health.to_dict(), dict)
