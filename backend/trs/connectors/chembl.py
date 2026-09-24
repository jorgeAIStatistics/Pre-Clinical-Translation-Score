"""ChEMBL connector (EMBL-EBI REST).

What it supplies
----------------
Compound identity (preferred name, InChIKey, SMILES, ATC), development stage
(``max_phase``, ``first_approval``) and curated mechanism of action linked to a
ChEMBL target. InChIKey is the strongest available join key for small
molecules, so this connector is the backbone of compound entity resolution.

Endpoint reliability
--------------------
Direct resource fetches (``/molecule/{id}.json``, ``/mechanism.json?...``) are
the reliable path. The name-search endpoints (``/molecule/search.json``,
filtered ``/molecule.json``) have been observed returning HTTP 500 from EBI for
extended periods. :meth:`ChemblConnector.search_molecules` therefore tries the
Lucene search endpoint, falls back to an exact-name filter query, and raises a
single clear error only when both fail; :meth:`ChemblConnector.health` probes
both paths independently and reports ``DEGRADED`` — not ``UNREACHABLE`` — when
direct fetches work and search does not, because in that state the connector is
still fully usable for every workflow that already holds a ChEMBL id.

Dating: why these records are retrieval-dated
---------------------------------------------
A ChEMBL molecule record has no "as of" date and no creation date. It does
carry ``first_approval`` and ``usan_year``, and it is tempting to date the
record from one of those — but the record also carries ``max_phase``,
``withdrawn_flag`` and the mechanism annotations, all of which are
**current-state** fields that ChEMBL revises release by release. Dating the
record at ``first_approval`` would assert that today's ``max_phase`` was
knowable on the day of approval, which is exactly backwards: ``max_phase`` is
the strongest possible outcome leak for a model predicting whether a programme
advances.

So, as with Open Targets, ``available_at`` is the retrieval date and every
record carries :data:`TEMPORAL_CAVEAT` plus ``provenance["temporal_caveat"]``.
``first_approval`` is still surfaced — as ``first_approval_on``, resolved to 1
January of that year under the shared first-day rule — so the loader can build
a properly-dated approval fact from it. It is a dated fact *inside* an undated
record, and the two must not be conflated.

Modality
--------
``molecule_type`` is mapped to :class:`~trs.ontology.enums.Modality` only where
the mapping is unambiguous. ChEMBL's ``Antibody`` covers monoclonals,
bispecifics and antibody-drug conjugates without distinguishing them, and its
``Oligonucleotide`` covers antisense and siRNA alike; the platform ontology has
a specific member for each of those but none for the generic parent. Rather
than pick the modal case and be wrong for the rest, those map to
:attr:`Modality.OTHER` with a warning and the verbatim ``molecule_type``
retained, so a curator or a mechanism-aware pass can refine them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar

from trs.config import settings
from trs.connectors import first_day_of_period, iso_date
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
    Modality,
    ProvenanceMethod,
    SourceType,
)

TEMPORAL_CAVEAT: str = (
    "ChEMBL molecule and mechanism records describe the current data release: "
    "max_phase, withdrawn_flag and mechanism annotations are revised between "
    "releases and cannot be back-dated. available_at is the retrieval date; "
    "using this record for an earlier cutoff would leak development outcome."
)

#: Unambiguous ``molecule_type`` → modality mappings only. See the docstring
#: for why ``Antibody`` and ``Oligonucleotide`` are absent.
MOLECULE_TYPE_MODALITY: dict[str, Modality] = {
    "small molecule": Modality.SMALL_MOLECULE,
    "protein": Modality.THERAPEUTIC_PROTEIN,
    "enzyme": Modality.THERAPEUTIC_PROTEIN,
    "gene": Modality.GENE_THERAPY,
    "cell": Modality.CELL_THERAPY_OTHER,
}

#: ChEMBL types the platform ontology cannot represent without narrowing an
#: unstated distinction. Mapped to OTHER, with the verbatim type preserved.
AMBIGUOUS_MOLECULE_TYPES: dict[str, str] = {
    "antibody": (
        "ChEMBL molecule_type 'Antibody' does not distinguish monoclonal, "
        "bispecific and antibody-drug conjugate, each of which is a separate "
        "Modality member; mapped to 'other' pending curation."
    ),
    "oligonucleotide": (
        "ChEMBL molecule_type 'Oligonucleotide' does not distinguish antisense "
        "from siRNA, each of which is a separate Modality member; mapped to "
        "'other' pending curation."
    ),
    "oligosaccharide": (
        "ChEMBL molecule_type 'Oligosaccharide' has no Modality member; mapped "
        "to 'other'."
    ),
}

#: ChEMBL max_phase is a number, not a phase label. 0.5 is "early phase 1".
#: -1 and 0 mean "no clinical data", which is absence of information, not a
#: preclinical assertion, so they map to UNKNOWN rather than DISCOVERY.
MAX_PHASE_TO_ENUM: dict[float, DevelopmentPhase] = {
    0.5: DevelopmentPhase.PHASE_1,
    1.0: DevelopmentPhase.PHASE_1,
    2.0: DevelopmentPhase.PHASE_2,
    3.0: DevelopmentPhase.PHASE_3,
    4.0: DevelopmentPhase.APPROVED,
}

#: Server-side field projection for target lookups; see search_targets.
TARGET_LOOKUP_FIELDS: tuple[str, ...] = (
    "target_chembl_id", "pref_name", "target_type", "organism", "tax_id",
)

#: Stable, tiny record used to probe direct-fetch availability (aspirin).
_HEALTH_CANARY = "CHEMBL25"


@register_connector
class ChemblConnector(Connector):
    """Adapter for the ChEMBL REST API."""

    name: ClassVar[str] = "chembl"
    version: ClassVar[str] = "1.0.0"
    display_name: ClassVar[str] = "ChEMBL (EMBL-EBI)"
    homepage: ClassVar[str] = "https://www.ebi.ac.uk/chembl/"
    license_note: ClassVar[str] = (
        "ChEMBL data are released by EMBL-EBI under CC BY-SA 3.0; derived works "
        "must carry the same licence and attribute ChEMBL. See "
        "https://chembl.gitbook.io/chembl-interface-documentation/about"
    )
    #: EBI asks for courteous use of the public REST service.
    min_request_interval: ClassVar[float] = 0.35

    def base_url(self) -> str:
        return settings.chembl_url

    def health(self) -> ConnectorHealth:
        """Probe the two paths separately, because they fail independently."""
        if not settings.allow_live_network:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DISABLED, checked_at=utcnow(),
                detail="live network disabled by configuration (TRS_ALLOW_LIVE_NETWORK)",
                endpoint=self.base_url(),
            )
        started = utcnow()
        direct_error: str | None = None
        try:
            self.request(
                "GET", f"molecule/{_HEALTH_CANARY}.json",
                use_cache=False, allow_stale_on_error=False,
            )
        except ConnectorError as exc:
            direct_error = str(exc)

        search_error: str | None = None
        try:
            self.request(
                "GET", "molecule/search.json", params={"q": "aspirin", "limit": 1},
                use_cache=False, allow_stale_on_error=False,
            )
        except ConnectorError as exc:
            search_error = str(exc)

        latency = (utcnow() - started).total_seconds() * 1000
        if direct_error is not None:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.UNREACHABLE, checked_at=utcnow(),
                latency_ms=latency, endpoint=self.base_url(),
                detail=f"direct resource fetch failed: {direct_error}",
            )
        if search_error is not None:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DEGRADED, checked_at=utcnow(),
                latency_ms=latency, endpoint=self.base_url(),
                detail=(
                    "direct resource fetch by ChEMBL id works; name search is unavailable "
                    f"({search_error}). Workflows that already hold a ChEMBL id are "
                    "unaffected; name-based lookup will fall back to an exact-name filter."
                ),
            )
        return ConnectorHealth(
            connector=self.name, status=HealthStatus.OK, checked_at=utcnow(),
            latency_ms=latency, endpoint=self.base_url(),
            detail="direct resource fetch and name search both responding",
        )

    # ---- fetch -------------------------------------------------------------

    def get_molecule(self, chembl_id: str) -> RawPayload:
        identifier = chembl_id.strip().upper()
        if not identifier:
            raise ConnectorError(f"{self.name}: empty ChEMBL molecule id")
        return self.request("GET", f"molecule/{identifier}.json")

    def get_mechanisms(self, molecule_chembl_id: str, *, limit: int = 20) -> RawPayload:
        identifier = molecule_chembl_id.strip().upper()
        if not identifier:
            raise ConnectorError(f"{self.name}: empty ChEMBL molecule id")
        return self.request(
            "GET", "mechanism.json",
            params={"molecule_chembl_id": identifier, "limit": max(1, min(limit, 100))},
        )

    def search_molecules(self, name: str, *, limit: int = 5) -> RawPayload:
        """Name lookup, degrading through two endpoints before giving up.

        The Lucene ``search`` endpoint is preferred (it handles synonyms and
        misspellings). When EBI has it down — which has been observed for
        extended periods — an exact case-insensitive ``pref_name`` filter still
        answers the common case of a known INN. If both fail the caller gets
        one error naming both attempts, rather than a bare 500 from whichever
        endpoint was tried last.
        """
        query = name.strip()
        if not query:
            raise ConnectorError(f"{self.name}: empty molecule name")
        bounded = max(1, min(limit, 100))
        try:
            return self.request("GET", "molecule/search.json", params={"q": query, "limit": bounded})
        except ConnectorError as search_error:
            try:
                return self.request(
                    "GET", "molecule.json",
                    params={"pref_name__iexact": query.upper(), "limit": bounded},
                )
            except ConnectorError as filter_error:
                raise ConnectorError(
                    f"{self.name}: name lookup for {query!r} failed on both endpoints — "
                    f"search: {search_error}; pref_name filter: {filter_error}"
                ) from filter_error

    def search_targets(
        self, query: str, *, limit: int = 10, only: Sequence[str] | None = TARGET_LOOKUP_FIELDS
    ) -> RawPayload:
        """ChEMBL target lookup by free text (gene symbol, protein name).

        ``only`` is ChEMBL's server-side field projection. It defaults to
        :data:`TARGET_LOOKUP_FIELDS` because a full target record is ~60 KB of
        ``target_components`` (GO slims, cross-references) that this connector
        does not read, and raw payloads are retained forever. Pass ``only=None``
        to store the complete record when a caller genuinely needs it.
        """
        text = query.strip()
        if not text:
            raise ConnectorError(f"{self.name}: empty target query")
        bounded = max(1, min(limit, 100))
        projection = {"only": ",".join(only)} if only else {}
        try:
            return self.request(
                "GET", "target/search.json", params={"q": text, "limit": bounded, **projection}
            )
        except ConnectorError as search_error:
            try:
                return self.request(
                    "GET", "target.json",
                    params={"target_synonym__icontains": text, "limit": bounded, **projection},
                )
            except ConnectorError as filter_error:
                raise ConnectorError(
                    f"{self.name}: target lookup for {text!r} failed on both endpoints — "
                    f"search: {search_error}; synonym filter: {filter_error}"
                ) from filter_error

    # ---- normalise ---------------------------------------------------------

    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        try:
            body = payload.json()
        except ValueError:
            return []
        if not isinstance(body, dict):
            return []

        retrieved_on = payload.fetched_at.date()

        if body.get("molecule_chembl_id") and "mechanism_of_action" not in body:
            return [self._compound_record(payload, body, retrieved_on, 0)]
        if "molecules" in body:
            return [
                self._compound_record(payload, m, retrieved_on, i)
                for i, m in enumerate(body.get("molecules") or [])
                if isinstance(m, dict) and m.get("molecule_chembl_id")
            ]
        if "mechanisms" in body:
            records = []
            for index, mechanism in enumerate(body.get("mechanisms") or []):
                if isinstance(mechanism, dict):
                    record = self._mechanism_record(payload, mechanism, retrieved_on, index)
                    if record is not None:
                        records.append(record)
            return records
        # Target payloads carry no date and no domain record kind of their own;
        # they are fetched to resolve identifiers and are read from the raw
        # payload by the resolution layer.
        return []

    def _compound_record(
        self, payload: RawPayload, molecule: dict[str, Any], retrieved_on: dt.date, index: int
    ) -> NormalisedRecord:
        chembl_id = molecule.get("molecule_chembl_id")
        structures = molecule.get("molecule_structures") or {}
        molecule_type = molecule.get("molecule_type")
        modality, modality_warning = map_modality(molecule_type)
        max_phase = _as_float(molecule.get("max_phase"))
        first_approval = _as_int(molecule.get("first_approval"))

        warnings = [TEMPORAL_CAVEAT]
        if modality_warning:
            warnings.append(modality_warning)

        missingness: dict[str, str] = {}
        if not structures.get("standard_inchi_key"):
            # A biologic legitimately has no InChIKey; that is "not applicable",
            # not "we failed to read it".
            missingness["inchikey"] = str(
                MissingnessReason.NOT_APPLICABLE
                if modality is not Modality.SMALL_MOLECULE
                else MissingnessReason.NOT_REPORTED
            )
        if first_approval is None:
            missingness["first_approval"] = str(MissingnessReason.NOT_REPORTED)

        body: dict[str, Any] = {
            "chembl_id": chembl_id,
            "preferred_name": molecule.get("pref_name"),
            "inchikey": structures.get("standard_inchi_key"),
            "canonical_smiles": structures.get("canonical_smiles"),
            "standard_inchi": structures.get("standard_inchi"),
            "molecule_type": molecule_type,
            "modality": str(modality),
            "max_phase": max_phase,
            "max_phase_enum": str(map_max_phase(max_phase)),
            "first_approval": first_approval,
            # A year resolves to its first day under the shared rule; this is a
            # dated fact inside an undated record (see the module docstring).
            "first_approval_on": iso_date(first_day_of_period(first_approval)),
            "usan_year": _as_int(molecule.get("usan_year")),
            "atc": list(molecule.get("atc_classifications") or []),
            "withdrawn_flag": bool(molecule.get("withdrawn_flag")),
            "black_box_warning": bool(molecule.get("black_box_warning")),
            "oral": molecule.get("oral"),
            "parenteral": molecule.get("parenteral"),
            "synonyms": sorted(
                {
                    s["molecule_synonym"]
                    for s in molecule.get("molecule_synonyms") or []
                    if isinstance(s, dict) and s.get("molecule_synonym")
                }
            ),
            "source_type": str(SourceType.STRUCTURED_DATABASE),
            "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
            "field_missingness": missingness,
            "retrieved_on": retrieved_on.isoformat(),
        }

        return NormalisedRecord(
            kind="compound",
            external_id=f"chembl:{chembl_id}",
            available_at=retrieved_on,
            payload=body,
            provenance=self._provenance(payload, index),
            confidence=str(ConfidenceLevel.HIGH),
            warnings=warnings,
        )

    def _mechanism_record(
        self, payload: RawPayload, mechanism: dict[str, Any], retrieved_on: dt.date, index: int
    ) -> NormalisedRecord | None:
        molecule_id = mechanism.get("molecule_chembl_id")
        if not molecule_id:
            return None
        max_phase = _as_float(mechanism.get("max_phase"))
        target_id = mechanism.get("target_chembl_id")
        body: dict[str, Any] = {
            "molecule_chembl_id": molecule_id,
            "target_chembl_id": target_id,
            # ChEMBL's action_type is its own controlled vocabulary
            # (INHIBITOR / AGONIST / ...) with no platform enum counterpart, so
            # it is carried verbatim rather than squeezed into an unrelated one.
            "action_type": mechanism.get("action_type"),
            "mechanism_of_action": mechanism.get("mechanism_of_action"),
            "mechanism_comment": mechanism.get("mechanism_comment"),
            "direct_interaction": _as_bool(mechanism.get("direct_interaction")),
            "disease_efficacy": _as_bool(mechanism.get("disease_efficacy")),
            "molecular_mechanism": _as_bool(mechanism.get("molecular_mechanism")),
            "max_phase": max_phase,
            "max_phase_enum": str(map_max_phase(max_phase)),
            "mec_id": mechanism.get("mec_id"),
            "references": [
                {"type": r.get("ref_type"), "id": r.get("ref_id"), "url": r.get("ref_url")}
                for r in mechanism.get("mechanism_refs") or []
                if isinstance(r, dict)
            ],
            "source_type": str(SourceType.STRUCTURED_DATABASE),
            "retrieved_on": retrieved_on.isoformat(),
        }
        return NormalisedRecord(
            kind="mechanism",
            external_id=f"chembl:{molecule_id}:{target_id}:{mechanism.get('mec_id')}",
            available_at=retrieved_on,
            payload=body,
            provenance=self._provenance(payload, index),
            confidence=str(ConfidenceLevel.HIGH),
            warnings=[TEMPORAL_CAVEAT],
        )

    def _provenance(self, payload: RawPayload, index: int) -> dict[str, Any]:
        return {
            **self.provenance_stub(payload),
            "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
            "record_index": index,
            "temporal_caveat": True,
            "available_at_basis": "retrieval timestamp (source has no as-of semantics)",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def map_modality(molecule_type: Any) -> tuple[Modality, str | None]:
    """``molecule_type`` → (modality, warning). See the module docstring."""
    if not isinstance(molecule_type, str) or not molecule_type.strip():
        return Modality.UNKNOWN, None
    key = molecule_type.strip().lower()
    mapped = MOLECULE_TYPE_MODALITY.get(key)
    if mapped is not None:
        return mapped, None
    ambiguous = AMBIGUOUS_MOLECULE_TYPES.get(key)
    if ambiguous is not None:
        return Modality.OTHER, ambiguous
    if key in {"unknown", "unclassified"}:
        return Modality.UNKNOWN, None
    return Modality.UNKNOWN, (
        f"ChEMBL molecule_type {molecule_type!r} is not in "
        "trs.connectors.chembl.MOLECULE_TYPE_MODALITY; modality left unknown."
    )


def map_max_phase(max_phase: float | None) -> DevelopmentPhase:
    """ChEMBL's numeric max_phase → :class:`DevelopmentPhase`.

    0 and -1 mean ChEMBL holds no clinical record, which is an absence of
    information rather than evidence that the compound is preclinical, so they
    become ``UNKNOWN``.
    """
    if max_phase is None:
        return DevelopmentPhase.UNKNOWN
    return MAX_PHASE_TO_ENUM.get(float(max_phase), DevelopmentPhase.UNKNOWN)


def _as_float(value: Any) -> float | None:
    """ChEMBL returns max_phase as a string in some responses and a number in
    others; both must land on the same value."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    """ChEMBL encodes booleans as 0/1 integers, and sometimes omits them."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


__all__ = [
    "AMBIGUOUS_MOLECULE_TYPES",
    "MAX_PHASE_TO_ENUM",
    "MOLECULE_TYPE_MODALITY",
    "TARGET_LOOKUP_FIELDS",
    "TEMPORAL_CAVEAT",
    "ChemblConnector",
    "map_max_phase",
    "map_modality",
]
