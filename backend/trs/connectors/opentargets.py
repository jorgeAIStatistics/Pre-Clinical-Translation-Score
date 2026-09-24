"""Open Targets Platform connector (GraphQL).

═══════════════════════════════════════════════════════════════════════════════
TEMPORAL WARNING — read this before using any record this connector emits
═══════════════════════════════════════════════════════════════════════════════
Open Targets association scores are recomputed from scratch at every data
release, from *all* evidence known at that release, using that release's
scoring model and disease ontology. A score is a statement about today. It
carries no as-of semantics whatsoever: there is no field in the API that says
"this is what the score would have been in 2014", and the score for a
target-disease pair routinely moves by a wide margin between releases as new
GWAS, ClinVar submissions and trial records land.

Consequently every record from this connector is dated ``payload.fetched_at``
— the retrieval date — never an earlier date derived from the underlying
evidence. Each record also carries:

* the fixed warning string :data:`TEMPORAL_CAVEAT` on
  :attr:`NormalisedRecord.warnings`; and
* ``provenance["temporal_caveat"] = True``.

:class:`~trs.db.models.evidence.TargetAssessment` rows are picked by
"latest ``available_at`` at or before the cutoff", so a back-dated Open Targets
score would be silently selected into historical feature vectors and would
leak the outcome the platform is trying to predict — a 2024 score for a
target-disease pair partly *reflects* the trials that succeeded or failed on
it. Dating at retrieval means these rows simply do not exist for historical
cutoffs, which is the correct answer.

*Noted for future work, not used here:* the API exposes
``Target.associationTimeSeries``, which yields a per-year association score.
That is still a retrospective reconstruction — current evidence re-binned by
publication year with the current scoring model and ontology — so it is a
better approximation than a flat current score but is **not** a historical
snapshot and must not be treated as one. Using it would need its own
validation and its own version bump.

═══════════════════════════════════════════════════════════════════════════════
Genetic evidence tier mapping
═══════════════════════════════════════════════════════════════════════════════
:class:`~trs.ontology.enums.GeneticEvidenceTier` asks *what kind* of human
genetic evidence exists. The aggregate ``genetic_association`` datatype score
cannot answer that: a 0.8 from ClinVar pathogenic variants in a Mendelian
disorder and a 0.8 from a common-variant GWAS locus are different claims about
the world and imply different translational risk. The tier is therefore
derived from **which datasources contributed**, never from the score alone,
and the score is carried separately in ``association_score``.

The table (strongest match wins, in this order):

==============================  ==========================================
Tier                            Contributing Open Targets datasources
==============================  ==========================================
``mendelian_causal``            ``eva`` (ClinVar germline), ``clingen``,
                                ``gene2phenotype``, ``genomics_england``,
                                ``orphanet``, ``uniprot_literature``,
                                ``uniprot_variants``
``rare_coding_burden``          ``gene_burden``
``gwas_fine_mapped``            ``gwas_credible_sets``,
                                ``ot_genetics_portal``
``gwas_locus_only``             ``gwas_catalog`` (legacy releases)
``somatic_driver``              ``cancer_gene_census``, ``intogen``,
                                ``eva_somatic``
==============================  ==========================================

Justification for each row:

* **mendelian_causal** — every listed source is a curated gene-disease
  assertion from clinical genetics (ClinVar germline pathogenic submissions,
  ClinGen validity curation, DDG2P, Genomics England PanelApp, Orphanet,
  UniProt disease variants). Each asserts that variation in this gene causes
  this disease in humans. That is exactly the tier's definition.
* **rare_coding_burden** — ``gene_burden`` is exome/genome-wide rare-variant
  burden testing (Genebass, AstraZeneca PheWAS and similar). Directionally
  informative human coding evidence, but statistical rather than a curated
  causal assertion, so it sits below Mendelian.
* **gwas_fine_mapped** — Open Targets Genetics / credible-set evidence is
  locus-to-gene scored over fine-mapped credible sets, i.e. the gene
  assignment is probabilistic but the causal variant set is resolved. That is
  the definition of ``gwas_fine_mapped`` and is strictly stronger than a raw
  locus.
* **gwas_locus_only** — a bare GWAS Catalog association assigns the signal to
  the nearest gene with no fine-mapping. Retained for older releases; current
  releases do not serve it.
* **somatic_driver** — Cancer Gene Census, IntOGen and ClinVar-somatic are
  driver catalogues: recurrent somatic alteration with evidence of a driver
  role. Ranked last among positives not because it is weak but because it is a
  claim about tumour genomes rather than germline predisposition, and the
  enum's own ordering places it there.

Deliberate conservatism:

* If a ``genetic_association``/``genetic_literature`` datatype score is present
  but **no** contributing datasource is recognised, the tier is
  ``not_assessed`` and the record carries a warning naming the unrecognised
  datasources. A new Open Targets datasource must be classified by a human
  before it can move a tier. Guessing from the score would silently
  mis-stratify every target that source touches.
* ``expression_qtl_only`` is never emitted. Open Targets files eQTL/expression
  evidence under the ``rna_expression`` datatype, not under genetic
  association, so this connector has no basis for that tier. It is left for a
  future eQTL-specific connector.
* ``no_human_genetic_signal`` is emitted **only** when an association row
  exists for the pair and contributes no genetic datatype at all. It is a
  negative finding scoped to Open Targets' integrated sources, and the
  ``source_note`` says so. When the pair has no association row at all, the
  tier is ``not_assessed``: we learned nothing, which is not the same as
  learning there is nothing.

:class:`~trs.ontology.enums.CausalSupport` follows from the tier plus the
non-genetic datatypes present:

* Mendelian / rare-burden / fine-mapped GWAS / somatic driver →
  ``human_genetic_causal``.
* GWAS locus only → ``correlational_only`` (nearest-gene assignment is an
  association, not a causal claim about the gene).
* No genetic tier but the ``animal_model`` datatype present → Open Targets'
  animal-model evidence is IMPC targeted mouse perturbation, i.e. an
  intervention → ``animal_interventional``.
* No genetic tier, no animal model, but literature / pathway / expression
  evidence present → ``correlational_only``.
* Nothing usable → ``not_assessed``.

``human_tissue_expression_evidence``
------------------------------------
:class:`~trs.db.models.evidence.TargetAssessment` declares no enum for this
column and defaults it to ``"unknown"``. This connector always emits that
sentinel. Open Targets does carry baseline expression, but the column asks
whether the target is expressed in the tissue *relevant to this indication*,
and resolving an EFO/MONDO disease term to its relevant tissue is a curation
judgement this connector has no basis to make. A missingness reason is
recorded so the feature layer can tell "not looked at" from "looked at and
absent".
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar

from trs.config import settings
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
    CausalSupport,
    ConfidenceLevel,
    GeneticEvidenceTier,
    MissingnessReason,
    PathwayRedundancy,
    ProvenanceMethod,
)

#: Verbatim text attached to every record this connector emits. Asserted by the
#: test-suite; changing it is a behaviour change and needs a version bump.
TEMPORAL_CAVEAT: str = (
    "Open Targets association scores reflect evidence available at retrieval time "
    "and cannot be back-dated; using this record for a historical cutoff before "
    "the retrieval date would leak future information."
)

#: TargetAssessment.human_tissue_expression_evidence has no declared enum; this
#: is the column's own default sentinel. See the module docstring.
HUMAN_TISSUE_EXPRESSION_UNKNOWN: str = "unknown"

#: Datatypes Open Targets uses for human genetic evidence. The 26.x releases
#: split curated literature-derived gene-disease assertions into
#: ``genetic_literature``; both are human genetics for our purposes.
GENETIC_DATATYPES: frozenset[str] = frozenset({"genetic_association", "genetic_literature"})

#: Datasource → tier. See the module docstring for the justification of each
#: entry. Anything absent from this map is unrecognised *by design*.
DATASOURCE_TIERS: dict[str, GeneticEvidenceTier] = {
    "eva": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "clingen": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "gene2phenotype": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "genomics_england": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "orphanet": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "uniprot_literature": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "uniprot_variants": GeneticEvidenceTier.MENDELIAN_CAUSAL,
    "gene_burden": GeneticEvidenceTier.RARE_CODING_BURDEN,
    "gwas_credible_sets": GeneticEvidenceTier.GWAS_FINE_MAPPED,
    "ot_genetics_portal": GeneticEvidenceTier.GWAS_FINE_MAPPED,
    "gwas_catalog": GeneticEvidenceTier.GWAS_LOCUS_ONLY,
    "cancer_gene_census": GeneticEvidenceTier.SOMATIC_DRIVER,
    "intogen": GeneticEvidenceTier.SOMATIC_DRIVER,
    "eva_somatic": GeneticEvidenceTier.SOMATIC_DRIVER,
}

#: Strongest-first precedence when several datasources contribute.
TIER_PRECEDENCE: tuple[GeneticEvidenceTier, ...] = (
    GeneticEvidenceTier.MENDELIAN_CAUSAL,
    GeneticEvidenceTier.RARE_CODING_BURDEN,
    GeneticEvidenceTier.GWAS_FINE_MAPPED,
    GeneticEvidenceTier.GWAS_LOCUS_ONLY,
    GeneticEvidenceTier.SOMATIC_DRIVER,
)

#: Tiers that constitute a human causal genetic claim.
_CAUSAL_TIERS: frozenset[GeneticEvidenceTier] = frozenset(
    {
        GeneticEvidenceTier.MENDELIAN_CAUSAL,
        GeneticEvidenceTier.RARE_CODING_BURDEN,
        GeneticEvidenceTier.GWAS_FINE_MAPPED,
        GeneticEvidenceTier.SOMATIC_DRIVER,
    }
)

_CORRELATIONAL_DATATYPES: frozenset[str] = frozenset(
    {"literature", "affected_pathway", "rna_expression", "known_drug"}
)

SEARCH_QUERY = """
query TrsTargetSearch($queryString: String!, $size: Int!) {
  search(queryString: $queryString, entityNames: ["target"], page: {index: 0, size: $size}) {
    total
    hits {
      id
      name
      entity
      object { ... on Target { id approvedSymbol approvedName biotype } }
    }
  }
}
""".strip()

ASSOCIATION_QUERY = """
query TrsTargetDiseaseAssociation($ensemblId: String!, $diseaseIds: [String!]) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    associatedDiseases(Bs: $diseaseIds, page: {index: 0, size: 1}) {
      count
      rows {
        score
        disease { id name }
        datatypeScores { id score }
        datasourceScores { id score }
      }
    }
  }
}
""".strip()

DETAILS_QUERY = """
query TrsTargetDetails($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    targetClass { id label level }
    tractability { label modality value }
    pathways { pathwayId pathway topLevelTerm }
    drugAndClinicalCandidates { count }
  }
}
""".strip()

HEALTH_QUERY = "query TrsHealth { meta { name apiVersion { x y z } dataVersion { year month } } }"


@register_connector
class OpenTargetsConnector(Connector):
    """Adapter for the Open Targets Platform GraphQL API."""

    name: ClassVar[str] = "opentargets"
    version: ClassVar[str] = "1.0.0"
    display_name: ClassVar[str] = "Open Targets Platform"
    homepage: ClassVar[str] = "https://platform.opentargets.org/"
    license_note: ClassVar[str] = (
        "Open Targets Platform data are released under CC0 1.0; individual "
        "contributing datasources retain their own terms (ChEMBL CC BY-SA, "
        "Orphanet CC BY 4.0, and others). See "
        "https://platform-docs.opentargets.org/licence"
    )
    #: No published limit; this is self-imposed restraint on a free public API.
    min_request_interval: ClassVar[float] = 0.35

    def base_url(self) -> str:
        return settings.opentargets_url

    def health(self) -> ConnectorHealth:
        if not settings.allow_live_network:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DISABLED, checked_at=utcnow(),
                detail="live network disabled by configuration (TRS_ALLOW_LIVE_NETWORK)",
                endpoint=self.base_url(),
            )
        started = utcnow()
        try:
            body = self.request(
                "POST", self.base_url(), json_body={"query": HEALTH_QUERY},
                use_cache=False, allow_stale_on_error=False,
            ).json()
        except (ConnectorError, ValueError) as exc:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.UNREACHABLE,
                checked_at=utcnow(), detail=str(exc), endpoint=self.base_url(),
            )
        latency = (utcnow() - started).total_seconds() * 1000
        meta = ((body.get("data") or {}).get("meta") or {})
        version = meta.get("apiVersion") or {}
        if not meta:
            return ConnectorHealth(
                connector=self.name, status=HealthStatus.DEGRADED, checked_at=utcnow(),
                latency_ms=latency, endpoint=self.base_url(),
                detail=f"GraphQL responded without a meta block: {body.get('errors')}",
            )
        data_version = meta.get("dataVersion") or {}
        return ConnectorHealth(
            connector=self.name, status=HealthStatus.OK, checked_at=utcnow(),
            latency_ms=latency, endpoint=self.base_url(),
            detail=(
                f"apiVersion {version.get('x')}.{version.get('y')}.{version.get('z')}, "
                f"dataVersion {data_version.get('year')}.{data_version.get('month')} — "
                "scores are current-release only and are not back-datable"
            ),
        )

    # ---- fetch -------------------------------------------------------------

    def _graphql(self, query: str, variables: dict[str, Any]) -> RawPayload:
        return self.request(
            "POST", self.base_url(), json_body={"query": query, "variables": variables}
        )

    def search_target(self, symbol: str, *, max_results: int = 5) -> RawPayload:
        """Resolve a gene symbol to Ensembl gene ids and approved names."""
        if not symbol.strip():
            raise ConnectorError(f"{self.name}: empty target symbol")
        return self._graphql(SEARCH_QUERY, {"queryString": symbol.strip(),
                                            "size": max(1, min(max_results, 25))})

    def target_disease_association(self, ensembl_id: str, efo_id: str) -> RawPayload:
        """Overall association score and per-datatype/per-datasource breakdown.

        ``efo_id`` is the Open Targets disease identifier; current releases use
        MONDO ids for many diseases even though the argument is conventionally
        called an EFO id, and the API's own filter argument is ``Bs``. The
        identifier is passed through untouched — remapping it here would be a
        silent ontology decision.
        """
        return self._graphql(
            ASSOCIATION_QUERY, {"ensemblId": ensembl_id.strip(), "diseaseIds": [efo_id.strip()]}
        )

    def target_details(self, ensembl_id: str) -> RawPayload:
        """Biotype, target class, tractability, pathways and known-drug count."""
        return self._graphql(DETAILS_QUERY, {"ensemblId": ensembl_id.strip()})

    # ---- normalise ---------------------------------------------------------

    def normalise(self, payload: RawPayload) -> list[NormalisedRecord]:
        try:
            body = payload.json()
        except ValueError:
            return []
        if not isinstance(body, dict):
            return []
        data = body.get("data")
        if data is None:
            raise ConnectorError(
                f"{self.name}: GraphQL response carried no data ({body.get('errors')})"
            )
        if "search" in data:
            # A search resolves identifiers. It asserts nothing about a target
            # that could be assessed, so it yields no records.
            return []
        target = data.get("target")
        if not isinstance(target, dict):
            return []

        # ``fetched_at`` is the only admissible date for anything here; see the
        # module docstring. It comes off the payload, so normalise stays pure.
        retrieved_on = payload.fetched_at.date()

        if "associatedDiseases" in target:
            return self._association_records(payload, target, retrieved_on)
        if any(key in target for key in ("tractability", "targetClass", "pathways")):
            return [self._target_record(payload, target, retrieved_on)]
        return []

    # -- target × disease --

    def _association_records(
        self, payload: RawPayload, target: dict[str, Any], retrieved_on: dt.date
    ) -> list[NormalisedRecord]:
        rows = (target.get("associatedDiseases") or {}).get("rows") or []
        records: list[NormalisedRecord] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            disease = row.get("disease") or {}
            disease_id = disease.get("id")
            if not disease_id:
                continue

            datatypes = _scores(row.get("datatypeScores"))
            datasources = _scores(row.get("datasourceScores"))
            tier, tier_note, tier_warnings = classify_genetic_tier(datatypes, datasources)
            causal = classify_causal_support(tier, datatypes)

            body: dict[str, Any] = {
                "target_ensembl_id": target.get("id"),
                "target_symbol": target.get("approvedSymbol"),
                "target_name": target.get("approvedName"),
                "biotype": target.get("biotype"),
                "disease_id": disease_id,
                "disease_name": disease.get("name"),
                "genetic_evidence_tier": str(tier),
                "association_score": row.get("score"),
                "causal_support": str(causal),
                "pathway_redundancy": str(PathwayRedundancy.NOT_ASSESSED),
                "human_tissue_expression_evidence": HUMAN_TISSUE_EXPRESSION_UNKNOWN,
                "datatype_scores": datatypes,
                "datasource_scores": datasources,
                "source_note": (
                    f"Open Targets Platform association for {target.get('approvedSymbol')} "
                    f"({target.get('id')}) x {disease.get('name')} ({disease_id}), retrieved "
                    f"{retrieved_on.isoformat()}. {tier_note} Scores are computed from the "
                    "current data release and are not a historical snapshot."
                ),
                "field_missingness": {
                    "human_tissue_expression_evidence": str(MissingnessReason.NOT_EXTRACTED),
                    "pathway_redundancy": str(MissingnessReason.NOT_EXTRACTED),
                    "prior_clinical_programs_on_target": str(MissingnessReason.NOT_EXTRACTED),
                },
                "retrieved_on": retrieved_on.isoformat(),
            }

            records.append(
                NormalisedRecord(
                    kind="target_assessment",
                    external_id=f"opentargets:{target.get('id')}:{disease_id}",
                    available_at=retrieved_on,
                    payload=body,
                    provenance=self._provenance(payload, index),
                    confidence=str(
                        ConfidenceLevel.HIGH if tier is not GeneticEvidenceTier.NOT_ASSESSED
                        else ConfidenceLevel.MODERATE
                    ),
                    warnings=[TEMPORAL_CAVEAT, *tier_warnings],
                )
            )
        return records

    # -- target only --

    def _target_record(
        self, payload: RawPayload, target: dict[str, Any], retrieved_on: dt.date
    ) -> NormalisedRecord:
        """Target-level assessment with no indication attached.

        Tractability and known-drug counts are properties of the target, not of
        a target-disease pair, so this record carries no association score and
        an explicitly ``not_assessed`` tier rather than a tier inferred from
        target-level facts.
        """
        tractability = [
            {"label": t.get("label"), "modality": t.get("modality"), "value": t.get("value")}
            for t in target.get("tractability") or []
            if isinstance(t, dict)
        ]
        body: dict[str, Any] = {
            "target_ensembl_id": target.get("id"),
            "target_symbol": target.get("approvedSymbol"),
            "target_name": target.get("approvedName"),
            "biotype": target.get("biotype"),
            "disease_id": None,
            "disease_name": None,
            "genetic_evidence_tier": str(GeneticEvidenceTier.NOT_ASSESSED),
            "association_score": None,
            "causal_support": str(CausalSupport.NOT_ASSESSED),
            "pathway_redundancy": str(PathwayRedundancy.NOT_ASSESSED),
            "human_tissue_expression_evidence": HUMAN_TISSUE_EXPRESSION_UNKNOWN,
            "target_class": [
                {"id": c.get("id"), "label": c.get("label"), "level": c.get("level")}
                for c in target.get("targetClass") or []
                if isinstance(c, dict)
            ],
            "tractability": tractability,
            "tractable_modalities": sorted(
                {t["modality"] for t in tractability if t.get("value") and t.get("modality")}
            ),
            "known_drugs_count": (target.get("drugAndClinicalCandidates") or {}).get("count"),
            "pathways": [
                {"id": p.get("pathwayId"), "name": p.get("pathway"),
                 "top_level_term": p.get("topLevelTerm")}
                for p in target.get("pathways") or []
                if isinstance(p, dict)
            ],
            "source_note": (
                f"Open Targets Platform target record for {target.get('approvedSymbol')} "
                f"({target.get('id')}), retrieved {retrieved_on.isoformat()}. No indication "
                "attached, so no genetic evidence tier is asserted. Tractability and "
                "known-drug counts describe the current release."
            ),
            "field_missingness": {
                "genetic_evidence_tier": str(MissingnessReason.NOT_APPLICABLE),
                "association_score": str(MissingnessReason.NOT_APPLICABLE),
                "human_tissue_expression_evidence": str(MissingnessReason.NOT_EXTRACTED),
            },
            "retrieved_on": retrieved_on.isoformat(),
        }
        return NormalisedRecord(
            kind="target_assessment",
            external_id=f"opentargets:{target.get('id')}",
            available_at=retrieved_on,
            payload=body,
            provenance=self._provenance(payload, 0),
            confidence=str(ConfidenceLevel.HIGH),
            warnings=[TEMPORAL_CAVEAT],
        )

    def _provenance(self, payload: RawPayload, index: int) -> dict[str, Any]:
        return {
            **self.provenance_stub(payload),
            "provenance_method": str(ProvenanceMethod.API_STRUCTURED),
            "record_index": index,
            # The flag the loader and the leakage tests key on.
            "temporal_caveat": True,
            "available_at_basis": "retrieval timestamp (source has no as-of semantics)",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure mapping functions (module level so the mapping is testable on its own)
# ─────────────────────────────────────────────────────────────────────────────

def _scores(entries: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("id") is not None:
            score = entry.get("score")
            if isinstance(score, (int, float)):
                out[str(entry["id"])] = float(score)
    return out


def classify_genetic_tier(
    datatype_scores: dict[str, float], datasource_scores: dict[str, float]
) -> tuple[GeneticEvidenceTier, str, list[str]]:
    """Datasource composition → tier, a human-readable note and any warnings.

    The full justification is in the module docstring. The short version: the
    tier names a *kind* of evidence, so it is read off which datasources
    contributed, and anything unrecognised produces ``not_assessed`` plus a
    warning rather than a plausible-looking guess.
    """
    contributing = {
        source: score for source, score in datasource_scores.items() if score and score > 0
    }
    matched: dict[GeneticEvidenceTier, list[str]] = {}
    unrecognised: list[str] = []
    for source in sorted(contributing):
        tier = DATASOURCE_TIERS.get(source)
        if tier is not None:
            matched.setdefault(tier, []).append(source)

    genetic_present = any(
        datatype_scores.get(datatype, 0.0) > 0 for datatype in GENETIC_DATATYPES
    )
    somatic_present = datatype_scores.get("somatic_mutation", 0.0) > 0

    if not matched:
        if genetic_present or somatic_present:
            # The datatype says there is human genetic evidence, but no
            # datasource we know about supplied it. Refuse to guess.
            unrecognised = sorted(contributing)
            return (
                GeneticEvidenceTier.NOT_ASSESSED,
                "Genetic evidence is present but no contributing datasource is classified "
                "by this connector, so no tier is asserted.",
                [
                    "Open Targets reported genetic evidence from unclassified datasource(s) "
                    f"{unrecognised}; genetic_evidence_tier left not_assessed. Classify the "
                    "datasource in trs.connectors.opentargets.DATASOURCE_TIERS before relying "
                    "on the tier for this target."
                ],
            )
        if datatype_scores:
            # An association row exists and contributes no genetic evidence at
            # all. That is a negative finding, scoped to Open Targets.
            return (
                GeneticEvidenceTier.NO_HUMAN_GENETIC_SIGNAL,
                "No genetic or somatic-driver datatype contributes to this association in "
                "Open Targets' integrated sources; this is a negative finding scoped to "
                "those sources, not to the literature as a whole.",
                [],
            )
        return (
            GeneticEvidenceTier.NOT_ASSESSED,
            "The association row carries no datatype scores, so nothing was learned.",
            [],
        )

    for tier in TIER_PRECEDENCE:
        if tier in matched:
            sources = ", ".join(matched[tier])
            others = sorted(
                s for t, group in matched.items() if t is not tier for s in group
            )
            note = f"Tier {tier} assigned from datasource(s): {sources}."
            if others:
                note += f" Also contributing: {', '.join(others)}."
            return tier, note, []

    # Unreachable while TIER_PRECEDENCE covers DATASOURCE_TIERS' value set; kept
    # so that adding a tier to the map without adding it to the precedence
    # ordering fails loudly instead of silently picking an arbitrary one.
    return (
        GeneticEvidenceTier.NOT_ASSESSED,
        "Matched datasources map to tiers outside the declared precedence ordering.",
        [f"Tier precedence ordering does not cover {sorted(str(t) for t in matched)}"],
    )


def classify_causal_support(
    tier: GeneticEvidenceTier, datatype_scores: dict[str, float]
) -> CausalSupport:
    """Tier plus non-genetic datatypes → causal support. See module docstring."""
    if tier in _CAUSAL_TIERS:
        return CausalSupport.HUMAN_GENETIC_CAUSAL
    if tier is GeneticEvidenceTier.GWAS_LOCUS_ONLY:
        return CausalSupport.CORRELATIONAL_ONLY
    if datatype_scores.get("animal_model", 0.0) > 0:
        # Open Targets' animal-model evidence is IMPC targeted perturbation:
        # an intervention on the gene, not an observed correlation.
        return CausalSupport.ANIMAL_INTERVENTIONAL
    if any(datatype_scores.get(d, 0.0) > 0 for d in _CORRELATIONAL_DATATYPES):
        return CausalSupport.CORRELATIONAL_ONLY
    return CausalSupport.NOT_ASSESSED


__all__ = [
    "DATASOURCE_TIERS",
    "GENETIC_DATATYPES",
    "HUMAN_TISSUE_EXPRESSION_UNKNOWN",
    "OpenTargetsConnector",
    "TEMPORAL_CAVEAT",
    "TIER_PRECEDENCE",
    "classify_causal_support",
    "classify_genetic_tier",
]
