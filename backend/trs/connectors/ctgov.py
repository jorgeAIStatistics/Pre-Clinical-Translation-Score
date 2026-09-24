"""ClinicalTrials.gov API v2 connector.

Reachability from this deployment
---------------------------------
Every ``/api/v2`` path — including ``/version`` — returns HTTP 403 from this
machine's network egress. The block is at the edge, not in the API: the same
requests are served normally from other networks. :meth:`health` therefore
reports :attr:`HealthStatus.UNREACHABLE` with the observed status and a note
about egress, rather than pretending the source is merely slow. The connector
is written against the documented v2 schema and its ``normalise`` path is fully
exercised offline against fixtures, so it will work unchanged the moment
egress is fixed.

═══════════════════════════════════════════════════════════════════════════════
The registry's temporal trap
═══════════════════════════════════════════════════════════════════════════════
A ClinicalTrials.gov study record is a **mutable document with one URL**. The
API returns its current state. It does not return the state as of any earlier
date. That makes a naive ``available_at`` a leak generator, because the record
mixes two very different kinds of field:

*Registration-time* fields — the registry id, titles, phase, study type,
allocation, masking, planned enrolment, conditions, interventions, primary
outcomes, lead sponsor, planned start — were present when the record was first
posted and are rarely revised in substance.

*Current-state* fields — ``overallStatus``, ``whyStopped``, actual enrolment,
actual completion dates, ``hasResults`` — were written later, some of them
years later.

``whyStopped`` is the sharpest case and the reason this connector emits two
records. A trial registered in 2013 and terminated for futility in 2018 has a
single API record carrying both the 2013 registration and the 2018 stop
reason. Attaching "terminated: interim analysis showed no efficacy benefit" to
``available_at = first_posted_on`` would hand a 2014 feature vector the answer
to the question the platform exists to ask. It is the single most damaging
leak available in this data source, and it is easy to write by accident.

So:

* ``kind="trial"`` is dated ``first_posted_on`` — the day the registry record
  became public.
* ``kind="trial_status_update"`` is dated ``last_update_on`` — the day the
  registry *posted* the update, which is the first day the stop reason was
  knowable. It carries ``why_stopped``, the current ``overall_status`` and
  ``has_posted_results``.

The ``trial`` payload still carries the current-state fields, because
:class:`~trs.db.models.evidence.Trial` has columns for them and the loader
needs to populate the row. They are listed explicitly in
``provenance["current_state_fields"]`` alongside
``provenance["current_state_available_at"]``, and the record carries a warning
naming them. **The loader must not let those columns reach a feature vector
whose cutoff precedes ``last_update_on``.** Nothing in this connector can
enforce that; flagging it is all a connector can do.

Dates
-----
v2 emits ``YYYY-MM`` or ``YYYY-MM-DD``. Month precision resolves to the first
of the month under the shared rule.

``studyFirstPostDateStruct`` is the public date. ``studyFirstSubmitDate`` is
the sponsor's submission, which is *not* public and is typically days to weeks
earlier; it is used only as a fallback, and only with a warning, because a
record with no post date at all would otherwise be dropped entirely. A record
with neither is not emitted.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar

from trs.config import settings
from trs.connectors import parse_partial_date
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
    DevelopmentPhase,
    MissingnessReason,
    ProvenanceMethod,
    SourceType,
)

#: v2 phase tokens → platform phase. A ``PHASE4`` trial is post-marketing, so
#: the programme has by definition reached approval. ``EARLY_PHASE1`` (what
#: used to be called phase 0) is still first-in-human dosing and maps to
#: PHASE_1; ``NA`` means the study is not drug-phase classified at all.
PHASE_MAP: dict[tuple[str, ...], DevelopmentPhase] = {
    ("EARLY_PHASE1",): DevelopmentPhase.PHASE_1,
    ("PHASE1",): DevelopmentPhase.PHASE_1,
    ("PHASE1", "PHASE2"): DevelopmentPhase.PHASE_1_2,
    ("PHASE2",): DevelopmentPhase.PHASE_2,
    ("PHASE2", "PHASE3"): DevelopmentPhase.PHASE_2_3,
    ("PHASE3",): DevelopmentPhase.PHASE_3,
    ("PHASE4",): DevelopmentPhase.APPROVED,
}

#: Payload keys whose values describe the record's state today, not its state
#: when it was first posted. Surfaced in provenance; see the module docstring.
CURRENT_STATE_FIELDS: tuple[str, ...] = (
    "overall_status",
    "why_stopped",
    "enrollment",
    "enrollment_type",
    "primary_completion_on",
    "completion_on",
    "last_update_on",
    "has_posted_results",
)


@register_connector
class ClinicalTrialsGovConnector(Connector):
    """Adapter for the ClinicalTrials.gov REST API v2."""

    name: ClassVar[str] = "ctgov"
    version: ClassVar[str] = "1.0.0"
    display_name: ClassVar[str] = "ClinicalTrials.gov"
    homepage: ClassVar[str] = "https://clinicaltrials.gov/"
    license_note: ClassVar[str] = (
        "ClinicalTrials.gov records are US Government works in the public "
        "domain; the NLM asks that reuse not imply endorsement. See "
        "https://clinicaltrials.gov/about-site/terms-conditions"
    )
    #: No published limit; the API documentation asks for reasonable use.
    min_request_interval: ClassVar[float] = 0.35

    def base_url(self) -> str:
        return settings.ctgov_url

    def health(self) -> ConnectorHealth:
        if not settings.allow_live_network:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DISABLED, checked_at=utcnow(),
                detail="live network disabled by configuration (TRS_ALLOW_LIVE_NETWORK)",
                endpoint=f"{self.base_url()}/version",
            )
        endpoint = f"{self.base_url()}/version"
        started = utcnow()
        try:
            body = self.request(
                "GET", endpoint, use_cache=False, allow_stale_on_error=False
            ).json()
        except (ConnectorError, ValueError) as exc:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.UNREACHABLE, checked_at=utcnow(),
                latency_ms=(utcnow() - started).total_seconds() * 1000, endpoint=endpoint,
                detail=(
                    f"{exc} — ClinicalTrials.gov API v2 is not reachable from this network "
                    "egress; every /api/v2 path including /version has been observed "
                    "returning HTTP 403. The block is at the network edge, not in the API. "
                    "Trial ingestion is unavailable until egress is changed or a proxy is "
                    "configured; normalise() remains covered by offline fixtures."
                ),
            )
        return ConnectorHealth(
            connector=self.name, status=HealthStatus.OK, checked_at=utcnow(),
            latency_ms=(utcnow() - started).total_seconds() * 1000, endpoint=endpoint,
            detail=f"API v2 {body.get('apiVersion')}, data timestamp {body.get('dataTimestamp')}",
        )

    # ---- fetch -------------------------------------------------------------

    def search_studies(
        self,
        *,
        query_term: str | None = None,
        query_cond: str | None = None,
        query_intr: str | None = None,
        fields: Sequence[str] | None = None,
        page_size: int = 20,
        page_token: str | None = None,
        filter_overall_status: Sequence[str] | None = None,
    ) -> RawPayload:
        """Search studies.

        ``fields`` restricts the response to named v2 field paths. Left unset,
        the full study document is returned — which is what evidence
        extraction needs, and what the raw landing zone should hold.
        """
        params: dict[str, Any] = {
            "format": "json",
            "pageSize": max(1, min(page_size, 1000)),
            "countTotal": "true",
        }
        if query_term:
            params["query.term"] = query_term
        if query_cond:
            params["query.cond"] = query_cond
        if query_intr:
            params["query.intr"] = query_intr
        if fields:
            params["fields"] = ",".join(fields)
        if page_token:
            params["pageToken"] = page_token
        if filter_overall_status:
            params["filter.overallStatus"] = ",".join(filter_overall_status)
        if not (query_term or query_cond or query_intr):
            raise ConnectorError(
                f"{self.name}: search_studies needs at least one of query_term, "
                "query_cond or query_intr; an unbounded search would pull the "
                "whole registry"
            )
        return self.request("GET", "studies", params=params)

    def get_study(self, nct_id: str) -> RawPayload:
        identifier = nct_id.strip().upper()
        if not identifier.startswith("NCT"):
            raise ConnectorError(f"{self.name}: {nct_id!r} is not an NCT identifier")
        return self.request("GET", f"studies/{identifier}", params={"format": "json"})

    # ---- normalise ---------------------------------------------------------

    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        try:
            body = payload.json()
        except ValueError:
            return []
        if not isinstance(body, dict):
            return []
        if "studies" in body:
            studies = [s for s in body.get("studies") or [] if isinstance(s, dict)]
        elif "protocolSection" in body:
            studies = [body]
        else:
            # /version and error documents carry no studies.
            return []

        records: list[NormalisedRecord] = []
        for index, study in enumerate(studies):
            records.extend(self._study_records(payload, study, index))
        return records

    def _study_records(
        self, payload: RawPayload, study: dict[str, Any], index: int
    ) -> list[NormalisedRecord]:
        protocol = study.get("protocolSection") or {}
        identification = protocol.get("identificationModule") or {}
        status_module = protocol.get("statusModule") or {}
        design = protocol.get("designModule") or {}
        design_info = design.get("designInfo") or {}
        sponsors = protocol.get("sponsorCollaboratorsModule") or {}

        nct_id = identification.get("nctId")
        if not nct_id:
            return []

        warnings: list[str] = []
        first_posted_on = _struct_date(status_module.get("studyFirstPostDateStruct"))
        available_at = first_posted_on
        basis = "studyFirstPostDateStruct"
        if available_at is None:
            # The submission date is not public, but it bounds the post date
            # from below by days; using it beats discarding the trial, provided
            # the approximation is stated on the record.
            available_at = parse_partial_date(status_module.get("studyFirstSubmitDate"))
            basis = "studyFirstSubmitDate (approximation)"
            if available_at is None:
                # No anchor at all: the record cannot be placed in time.
                return []
            warnings.append(
                "studyFirstPostDateStruct absent; available_at approximated from "
                "studyFirstSubmitDate, which is the sponsor's submission and not a public "
                "date. The true public date is a few days later."
            )

        last_update_on = _struct_date(status_module.get("lastUpdatePostDateStruct"))
        why_stopped = (status_module.get("whyStopped") or "").strip() or None
        phases = tuple(design.get("phases") or [])
        phase, phase_warning = map_phase(phases)
        if phase_warning:
            warnings.append(phase_warning)

        enrollment_info = design.get("enrollmentInfo") or {}
        masking_info = design_info.get("maskingInfo") or {}
        lead_sponsor = sponsors.get("leadSponsor") or {}

        body: dict[str, Any] = {
            "registry": "clinicaltrials_gov",
            "registry_id": nct_id,
            "brief_title": identification.get("briefTitle"),
            "official_title": identification.get("officialTitle"),
            "phase": str(phase),
            "phases_raw": list(phases),
            "overall_status": status_module.get("overallStatus"),
            "why_stopped": why_stopped,
            "study_type": design.get("studyType"),
            "allocation": design_info.get("allocation"),
            "intervention_model": design_info.get("interventionModel"),
            "masking": masking_info.get("masking"),
            "who_masked": list(masking_info.get("whoMasked") or []),
            "enrollment": enrollment_info.get("count"),
            "enrollment_type": enrollment_info.get("type"),
            "first_posted_on": _iso(first_posted_on),
            "start_on": _iso(_struct_date(status_module.get("startDateStruct"))),
            "primary_completion_on": _iso(
                _struct_date(status_module.get("primaryCompletionDateStruct"))
            ),
            "completion_on": _iso(_struct_date(status_module.get("completionDateStruct"))),
            "last_update_on": _iso(last_update_on),
            "lead_sponsor": {
                "name": lead_sponsor.get("name"),
                "class": lead_sponsor.get("class"),
            },
            "conditions": list((protocol.get("conditionsModule") or {}).get("conditions") or []),
            "interventions": [
                {"type": i.get("type"), "name": i.get("name"),
                 "other_names": list(i.get("otherNames") or [])}
                for i in (protocol.get("armsInterventionsModule") or {}).get("interventions") or []
                if isinstance(i, dict)
            ],
            "primary_outcomes": [
                {"measure": o.get("measure"), "description": o.get("description"),
                 "time_frame": o.get("timeFrame")}
                for o in (protocol.get("outcomesModule") or {}).get("primaryOutcomes") or []
                if isinstance(o, dict)
            ],
            "has_posted_results": bool(study.get("hasResults")),
            "source_type": str(SourceType.CLINICAL_TRIAL_REGISTRY),
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
            "field_missingness": {},
            "retrieved_on": payload.fetched_at.date().isoformat(),
        }
        if phase is DevelopmentPhase.UNKNOWN:
            body["field_missingness"]["phase"] = str(
                MissingnessReason.NOT_APPLICABLE
                if phases in ((), ("NA",))
                else MissingnessReason.NOT_REPORTED
            )

        populated_current_state = [
            field for field in CURRENT_STATE_FIELDS if body.get(field) not in (None, "", [], False)
        ]
        if populated_current_state:
            warnings.append(
                "Payload mixes registration-time and current-state fields. "
                f"available_at ({available_at.isoformat()}) dates the registration only; "
                f"{', '.join(populated_current_state)} were written to the registry later "
                f"and were knowable no earlier than "
                f"{last_update_on.isoformat() if last_update_on else 'an unknown later date'}. "
                "Do not admit those columns to a feature vector cut before that date."
            )

        records = [
            NormalisedRecord(
                kind="trial",
                external_id=f"nct:{nct_id}",
                available_at=available_at,
                payload=body,
                provenance={
                    **self.provenance_stub(payload),
                    "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
                    "record_index": index,
                    "available_at_basis": basis,
                    "current_state_fields": list(CURRENT_STATE_FIELDS),
                    "current_state_available_at": _iso(last_update_on),
                },
                confidence=str(
                    ConfidenceLevel.HIGH if first_posted_on else ConfidenceLevel.MODERATE
                ),
                warnings=warnings,
            )
        ]

        status_record = self._status_update_record(
            payload, body, index, why_stopped, last_update_on, available_at
        )
        if status_record is not None:
            records.append(status_record)
        elif why_stopped:
            records[0].warnings.append(
                "why_stopped is populated but lastUpdatePostDateStruct is absent, so the "
                "date on which the stop reason became knowable cannot be established and no "
                "trial_status_update record was emitted. The stop reason must not be used."
            )

        return records

    def _status_update_record(
        self,
        payload: RawPayload,
        trial: dict[str, Any],
        index: int,
        why_stopped: str | None,
        last_update_on: dt.date | None,
        first_available_at: dt.date,
    ) -> NormalisedRecord | None:
        """The correctly-dated half of a terminated trial.

        Emitted only when there is a stop reason *and* a posted update date to
        hang it on. Its ``available_at`` is deliberately the update date, never
        the first-posted date; see the module docstring for why that difference
        is the whole point of this connector.
        """
        if not why_stopped or last_update_on is None:
            return None

        warnings: list[str] = []
        if last_update_on < first_available_at:
            # Registry data errors of this shape exist. Reporting the recorded
            # date and flagging it beats silently "correcting" it.
            warnings.append(
                f"lastUpdatePostDate {last_update_on.isoformat()} precedes the study's "
                f"first public date {first_available_at.isoformat()}; the registry dates are "
                "internally inconsistent and this record's available_at is suspect."
            )

        return NormalisedRecord(
            kind="trial_status_update",
            external_id=f"nct:{trial['registry_id']}@{last_update_on.isoformat()}",
            available_at=last_update_on,
            payload={
                "registry": "clinicaltrials_gov",
                "registry_id": trial["registry_id"],
                "why_stopped": why_stopped,
                "overall_status": trial.get("overall_status"),
                "has_posted_results": trial.get("has_posted_results"),
                "last_update_on": trial.get("last_update_on"),
                "primary_completion_on": trial.get("primary_completion_on"),
                "completion_on": trial.get("completion_on"),
                "enrollment": trial.get("enrollment"),
                "enrollment_type": trial.get("enrollment_type"),
                "source_type": str(SourceType.CLINICAL_TRIAL_REGISTRY),
                "url": trial.get("url"),
            },
            provenance={
                **self.provenance_stub(payload),
                "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
                "record_index": index,
                "available_at_basis": "lastUpdatePostDateStruct",
                "supersedes_registration_available_at": first_available_at.isoformat(),
            },
            confidence=str(ConfidenceLevel.HIGH),
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _struct_date(struct: Any) -> dt.date | None:
    """v2 ``{"date": "2015-03", "type": "ACTUAL"}`` → 2015-03-01."""
    if not isinstance(struct, dict):
        return None
    return parse_partial_date(struct.get("date"))


def _iso(value: dt.date | None) -> str | None:
    return value.isoformat() if value is not None else None


def map_phase(phases: Sequence[str]) -> tuple[DevelopmentPhase, str | None]:
    """v2 ``phases`` list → (phase, warning).

    An unrecognised combination yields ``UNKNOWN`` and a warning rather than a
    nearest-neighbour guess: phase is a headline feature, and a wrong phase is
    worse than a missing one.
    """
    key = tuple(sorted(str(p).strip().upper() for p in phases if p))
    if not key or key == ("NA",):
        return DevelopmentPhase.UNKNOWN, None
    mapped = PHASE_MAP.get(key)
    if mapped is not None:
        return mapped, None
    return DevelopmentPhase.UNKNOWN, (
        f"ClinicalTrials.gov phases {list(phases)} do not match any entry in "
        "trs.connectors.ctgov.PHASE_MAP; phase left unknown."
    )


__all__ = [
    "CURRENT_STATE_FIELDS",
    "PHASE_MAP",
    "ClinicalTrialsGovConnector",
    "map_phase",
]
