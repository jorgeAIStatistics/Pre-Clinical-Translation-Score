"""Evidence, sources and provenance.

The provenance chain
--------------------
``RawRecord`` → ``Source`` → ``EvidenceItem`` → ``EvidenceFieldProvenance``

``RawRecord`` is the immutable landing zone: the verbatim bytes a connector
received, keyed by a content hash. It is never rewritten and never deleted, so
any downstream value can be re-derived and any extraction bug can be diagnosed
against exactly the payload that produced it.

``Source`` is the citable object (an article, a registry entry, a label).

``EvidenceItem`` is one structured, model-relevant claim.

``EvidenceFieldProvenance`` records, per field, how that particular number came
to be — including the verbatim snippet it was read from.

Temporal integrity
------------------
Every evidence-bearing row carries ``available_at``: the date from which a
reasonable analyst could have known this fact. Feature construction for an
as-of date ``T`` uses only rows with ``available_at <= T``. This is enforced in
one place (:mod:`trs.features.temporal`) and asserted by a dedicated leakage
test suite, because a violation here silently invalidates every reported
performance number.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trs.db.base import (
    Base,
    IdMixin,
    JSONVariant,
    SyntheticFlagMixin,
    TimestampMixin,
    UTCDateTime,
)

# ─────────────────────────────────────────────────────────────────────────────
# Raw landing zone
# ─────────────────────────────────────────────────────────────────────────────

class RawRecord(Base, IdMixin):
    """Immutable verbatim connector payload. Append-only."""

    __tablename__ = "raw_record"
    __table_args__ = (
        UniqueConstraint("content_hash"),
        Index("ix_raw_record_connector_fetched", "connector", "fetched_at"),
    )

    connector: Mapped[str] = mapped_column(String(48), nullable=False)
    connector_version: Mapped[str] = mapped_column(String(32), nullable=False, default="0")
    endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    request_params: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    http_status: Mapped[int | None] = mapped_column(Integer, default=None)
    media_type: Mapped[str | None] = mapped_column(String(80), default=None)
    #: SHA-256 of the response body; the deduplication and integrity key.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    content_json: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    fetched_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    #: Set when the payload was replayed from an on-disk cache rather than the network.
    from_cache: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

class Source(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A citable object that evidence can be attributed to."""

    __tablename__ = "source"
    __table_args__ = (
        Index("ix_source_ids", "pmid", "doi", "nct_id"),
    )

    #: trs.ontology.enums.SourceType
    source_type: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    title: Mapped[str | None] = mapped_column(Text, default=None)
    container_title: Mapped[str | None] = mapped_column(String(400), default=None)
    authors: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    pmid: Mapped[str | None] = mapped_column(String(24), default=None, index=True)
    pmcid: Mapped[str | None] = mapped_column(String(24), default=None)
    doi: Mapped[str | None] = mapped_column(String(200), default=None, index=True)
    nct_id: Mapped[str | None] = mapped_column(String(24), default=None, index=True)
    openalex_id: Mapped[str | None] = mapped_column(String(48), default=None, index=True)
    chembl_id: Mapped[str | None] = mapped_column(String(32), default=None)
    other_identifier: Mapped[str | None] = mapped_column(String(200), default=None)
    url: Mapped[str | None] = mapped_column(Text, default=None)

    #: Date the source itself was published / first posted.
    published_on: Mapped[dt.date | None] = mapped_column(Date, default=None, index=True)
    #: Date our pipeline retrieved it.
    retrieved_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)
    retrieved_by_connector: Mapped[str | None] = mapped_column(String(48), default=None)
    raw_record_id: Mapped[str | None] = mapped_column(
        ForeignKey("raw_record.id", ondelete="SET NULL"), default=None
    )

    #: trs.ontology.enums.PublicationStatus
    publication_status: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    is_retracted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retraction_note: Mapped[str | None] = mapped_column(Text, default=None)
    license: Mapped[str | None] = mapped_column(String(80), default=None)
    abstract: Mapped[str | None] = mapped_column(Text, default=None)
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    raw_record: Mapped[RawRecord | None] = relationship()


class Trial(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A registered clinical study. Registry-specific fields live here rather
    than on :class:`Source` so registry semantics stay explicit."""

    __tablename__ = "trial"

    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None, index=True
    )
    registry: Mapped[str] = mapped_column(String(32), nullable=False, default="clinicaltrials_gov")
    registry_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    brief_title: Mapped[str | None] = mapped_column(Text, default=None)
    official_title: Mapped[str | None] = mapped_column(Text, default=None)
    #: trs.ontology.enums.DevelopmentPhase
    phase: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    overall_status: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    why_stopped: Mapped[str | None] = mapped_column(Text, default=None)
    study_type: Mapped[str | None] = mapped_column(String(48), default=None)
    allocation: Mapped[str | None] = mapped_column(String(48), default=None)
    masking: Mapped[str | None] = mapped_column(String(48), default=None)
    enrollment: Mapped[int | None] = mapped_column(Integer, default=None)
    enrollment_type: Mapped[str | None] = mapped_column(String(24), default=None)
    first_posted_on: Mapped[dt.date | None] = mapped_column(Date, default=None, index=True)
    start_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    primary_completion_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    completion_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    last_update_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    lead_sponsor_id: Mapped[str | None] = mapped_column(
        ForeignKey("sponsor.id", ondelete="SET NULL"), default=None
    )
    program_id: Mapped[str | None] = mapped_column(
        ForeignKey("program.id", ondelete="SET NULL"), default=None, index=True
    )
    program_indication_id: Mapped[str | None] = mapped_column(
        ForeignKey("program_indication.id", ondelete="SET NULL"), default=None, index=True
    )
    conditions: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    interventions: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    primary_outcomes: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    has_posted_results: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)


# ─────────────────────────────────────────────────────────────────────────────
# The central evidence fact table
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceItem(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """One structured, translationally-relevant claim.

    An evidence item is narrow on purpose: one system, one endpoint, one
    effect. A paper reporting three models and two endpoints yields six items.
    Aggregation into features is the feature layer's job, not the extractor's,
    so the aggregation rule is versioned and testable.
    """

    __tablename__ = "evidence_item"
    __table_args__ = (
        Index("ix_evidence_track_available", "program_indication_id", "available_at"),
        Index("ix_evidence_program_class", "program_id", "evidence_class"),
        Index("ix_evidence_target_available", "target_id", "available_at"),
    )

    organization_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)

    # ---- What the evidence is about ---------------------------------------
    program_id: Mapped[str | None] = mapped_column(
        ForeignKey("program.id", ondelete="CASCADE"), default=None, index=True
    )
    program_indication_id: Mapped[str | None] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), default=None, index=True
    )
    target_id: Mapped[str | None] = mapped_column(
        ForeignKey("target.id", ondelete="SET NULL"), default=None, index=True
    )
    compound_id: Mapped[str | None] = mapped_column(
        ForeignKey("compound.id", ondelete="SET NULL"), default=None
    )
    indication_id: Mapped[str | None] = mapped_column(
        ForeignKey("indication.id", ondelete="SET NULL"), default=None
    )

    # ---- Classification ----------------------------------------------------
    #: trs.ontology.enums.EvidenceClass
    evidence_class: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    #: trs.ontology.enums.EvidenceDirection
    direction: Mapped[str] = mapped_column(String(24), nullable=False, default="unknown")
    title: Mapped[str | None] = mapped_column(Text, default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)

    # ---- Experimental system ----------------------------------------------
    #: trs.ontology.enums.Species
    species: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    #: trs.ontology.enums.BiologicalSystem
    biological_system: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: trs.ontology.enums.ModelType
    model_type: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: trs.ontology.enums.OncologyModelSubtype
    model_subtype: Mapped[str] = mapped_column(String(48), nullable=False, default="not_applicable")
    model_identifier: Mapped[str | None] = mapped_column(String(200), default=None)
    #: trs.ontology.enums.ImmuneCompetence
    immune_competence: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: trs.ontology.enums.ConstructValidity — curated, with assessor recorded.
    construct_validity: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    construct_validity_assessor: Mapped[str | None] = mapped_column(String(120), default=None)
    #: Qualitative similarity of the model's disease biology to the human target
    #: population, on the same scale as construct validity.
    disease_similarity: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")

    # ---- Endpoint and effect ------------------------------------------------
    #: trs.ontology.enums.EndpointCategory
    endpoint_category: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    endpoint_description: Mapped[str | None] = mapped_column(Text, default=None)
    #: trs.ontology.enums.EffectMetric
    effect_metric: Mapped[str | None] = mapped_column(String(48), default=None)
    effect_value: Mapped[float | None] = mapped_column(Float, default=None)
    effect_ci_low: Mapped[float | None] = mapped_column(Float, default=None)
    effect_ci_high: Mapped[float | None] = mapped_column(Float, default=None)
    effect_se: Mapped[float | None] = mapped_column(Float, default=None)
    effect_p_value: Mapped[float | None] = mapped_column(Float, default=None)
    #: True when the reported effect favours the intervention, irrespective of
    #: whether the metric's scale is increasing or decreasing.
    effect_favours_treatment: Mapped[bool | None] = mapped_column(Boolean, default=None)

    # ---- Dose and exposure --------------------------------------------------
    dose_value: Mapped[float | None] = mapped_column(Float, default=None)
    dose_unit: Mapped[str | None] = mapped_column(String(48), default=None)
    dose_schedule: Mapped[str | None] = mapped_column(String(120), default=None)
    route: Mapped[str | None] = mapped_column(String(48), default=None)
    exposure_auc: Mapped[float | None] = mapped_column(Float, default=None)
    exposure_cmax: Mapped[float | None] = mapped_column(Float, default=None)
    exposure_unit: Mapped[str | None] = mapped_column(String(48), default=None)
    unbound_fraction: Mapped[float | None] = mapped_column(Float, default=None)

    # ---- Design and power ---------------------------------------------------
    study_duration_days: Mapped[int | None] = mapped_column(Integer, default=None)
    n_per_arm: Mapped[int | None] = mapped_column(Integer, default=None)
    n_arms: Mapped[int | None] = mapped_column(Integer, default=None)
    #: trs.ontology.enums.RandomizationLevel
    randomization: Mapped[str] = mapped_column(String(48), nullable=False, default="not_reported")
    #: trs.ontology.enums.BlindingLevel
    blinding: Mapped[str] = mapped_column(String(48), nullable=False, default="not_reported")
    sample_size_justified: Mapped[bool | None] = mapped_column(Boolean, default=None)
    multiplicity_controlled: Mapped[bool | None] = mapped_column(Boolean, default=None)
    prospective_protocol: Mapped[bool | None] = mapped_column(Boolean, default=None)
    preregistered: Mapped[bool | None] = mapped_column(Boolean, default=None)
    preregistration_id: Mapped[str | None] = mapped_column(String(120), default=None)

    # ---- Replication and reporting integrity --------------------------------
    #: trs.ontology.enums.ReplicationScope
    replication_scope: Mapped[str] = mapped_column(String(48), nullable=False, default="not_reported")
    independent_replications: Mapped[int | None] = mapped_column(Integer, default=None)
    #: Stable identifier for the performing laboratory/site, used to detect
    #: single-lab clusters masquerading as independent replication.
    lab_identifier: Mapped[str | None] = mapped_column(String(120), default=None, index=True)
    #: trs.ontology.enums.SelectiveReportingRisk
    selective_reporting_risk: Mapped[str] = mapped_column(
        String(32), nullable=False, default="not_assessed"
    )
    #: trs.ontology.enums.ConflictOfInterest
    conflict_of_interest: Mapped[str] = mapped_column(
        String(48), nullable=False, default="not_reported"
    )
    #: trs.ontology.enums.PublicationStatus
    publication_status: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")

    # ---- Provenance and time ------------------------------------------------
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None, index=True
    )
    #: THE temporal-integrity field. Earliest date this fact was knowable.
    available_at: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    #: When the underlying study was conducted/reported, if distinct.
    observed_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    extraction_date: Mapped[dt.date | None] = mapped_column(Date, default=None)
    #: trs.ontology.enums.ProvenanceMethod
    provenance_method: Mapped[str] = mapped_column(String(48), nullable=False, default="human_curation")
    extractor_version: Mapped[str | None] = mapped_column(String(48), default=None)
    #: trs.ontology.enums.ConfidenceLevel
    extraction_confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="not_assessed")
    source_snippet: Mapped[str | None] = mapped_column(Text, default=None)
    source_locator: Mapped[str | None] = mapped_column(String(200), default=None)

    #: Per-field missingness reasons: {field_name: MissingnessReason}
    field_missingness: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)

    source: Mapped[Source | None] = relationship()
    field_provenance: Mapped[list[EvidenceFieldProvenance]] = relationship(
        back_populates="evidence", cascade="all, delete-orphan"
    )


class EvidenceFieldProvenance(Base, IdMixin):
    """Per-field provenance for values whose origin differs from the item's.

    Populated when a single evidence item is assembled from more than one
    source, or when one field was human-verified while others were not.
    """

    __tablename__ = "evidence_field_provenance"
    __table_args__ = (UniqueConstraint("evidence_item_id", "field_name"),)

    evidence_item_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_item.id", ondelete="CASCADE"), index=True
    )
    field_name: Mapped[str] = mapped_column(String(80), nullable=False)
    value_repr: Mapped[str | None] = mapped_column(Text, default=None)
    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None
    )
    #: trs.ontology.enums.ProvenanceMethod
    method: Mapped[str] = mapped_column(String(48), nullable=False)
    #: trs.ontology.enums.ConfidenceLevel
    confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="not_assessed")
    snippet: Mapped[str | None] = mapped_column(Text, default=None)
    extracted_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    extracted_by: Mapped[str | None] = mapped_column(String(120), default=None)

    evidence: Mapped[EvidenceItem] = relationship(back_populates="field_provenance")


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific assessment tables
# ─────────────────────────────────────────────────────────────────────────────

class TargetAssessment(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """Time-stamped view of target-validation evidence.

    Target biology knowledge accumulates. A target that had no human genetic
    support in 2012 may have strong support by 2019. Assessments are therefore
    versioned by ``available_at`` and the feature layer picks the latest one at
    or before the cutoff — never the newest one overall.
    """

    __tablename__ = "target_assessment"
    __table_args__ = (Index("ix_target_assessment_lookup", "target_id", "available_at"),)

    target_id: Mapped[str] = mapped_column(ForeignKey("target.id", ondelete="CASCADE"), index=True)
    indication_id: Mapped[str | None] = mapped_column(
        ForeignKey("indication.id", ondelete="CASCADE"), default=None, index=True
    )
    available_at: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)

    #: trs.ontology.enums.GeneticEvidenceTier
    genetic_evidence_tier: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: Open Targets-style association score in [0, 1] when available.
    association_score: Mapped[float | None] = mapped_column(Float, default=None)
    #: trs.ontology.enums.CausalSupport
    causal_support: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: trs.ontology.enums.PathwayRedundancy
    pathway_redundancy: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    compensatory_mechanism_note: Mapped[str | None] = mapped_column(Text, default=None)

    #: Evidence that the target is expressed in the relevant human tissue.
    human_tissue_expression_evidence: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )
    target_conservation_note: Mapped[str | None] = mapped_column(Text, default=None)

    #: Prior clinical experience with this target across all sponsors,
    #: counted as of ``available_at`` only.
    prior_clinical_programs_on_target: Mapped[int | None] = mapped_column(Integer, default=None)
    prior_approved_drugs_on_target: Mapped[int | None] = mapped_column(Integer, default=None)
    prior_ph2_failures_on_target: Mapped[int | None] = mapped_column(Integer, default=None)

    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None
    )
    #: trs.ontology.enums.ProvenanceMethod
    provenance_method: Mapped[str] = mapped_column(String(48), nullable=False, default="api_structured")
    field_missingness: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)


class PkPdBridge(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """The exposure/target-engagement argument for a development track."""

    __tablename__ = "pkpd_bridge"
    __table_args__ = (Index("ix_pkpd_track_available", "program_indication_id", "available_at"),)

    program_indication_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    available_at: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)

    #: Exposure associated with efficacy in the pivotal preclinical model.
    preclinical_efficacious_auc: Mapped[float | None] = mapped_column(Float, default=None)
    preclinical_efficacious_cmax: Mapped[float | None] = mapped_column(Float, default=None)
    exposure_unit: Mapped[str | None] = mapped_column(String(48), default=None)
    preclinical_species: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")

    #: Human exposure predicted (or, later, observed) at a tolerated dose.
    projected_human_auc: Mapped[float | None] = mapped_column(Float, default=None)
    projected_human_cmax: Mapped[float | None] = mapped_column(Float, default=None)
    human_exposure_observed: Mapped[bool | None] = mapped_column(Boolean, default=None)

    #: Ratio of achievable human exposure to the efficacious preclinical
    #: exposure, both unbound where possible. >1 means the efficacious exposure
    #: appears reachable. Stored rather than derived because the correct
    #: unbound correction is a curation judgement.
    exposure_margin_ratio: Mapped[float | None] = mapped_column(Float, default=None)
    #: Ratio of the exposure at the no-observed-adverse-effect level to the
    #: efficacious exposure — the therapeutic window.
    therapeutic_index_ratio: Mapped[float | None] = mapped_column(Float, default=None)

    #: trs.ontology.enums.ExposureBridgeQuality
    exposure_bridge_quality: Mapped[str] = mapped_column(
        String(48), nullable=False, default="not_assessed"
    )
    #: trs.ontology.enums.TargetEngagementEvidence
    target_engagement_evidence: Mapped[str] = mapped_column(
        String(48), nullable=False, default="not_assessed"
    )
    target_engagement_percent: Mapped[float | None] = mapped_column(Float, default=None)
    #: Whether a monotone exposure–response relationship was demonstrated.
    exposure_response_established: Mapped[bool | None] = mapped_column(Boolean, default=None)
    #: Whether PK scaling was supported by more than one species.
    pk_scaling_species_count: Mapped[int | None] = mapped_column(Integer, default=None)
    pk_model_type: Mapped[str | None] = mapped_column(String(80), default=None)

    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None
    )
    provenance_method: Mapped[str] = mapped_column(String(48), nullable=False, default="human_curation")
    field_missingness: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)


class ProgramBiomarker(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A biomarker attached to a development track, with readiness over time."""

    __tablename__ = "program_biomarker"
    __table_args__ = (
        Index("ix_program_biomarker_lookup", "program_indication_id", "available_at"),
    )

    program_indication_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    biomarker_id: Mapped[str | None] = mapped_column(
        ForeignKey("biomarker.id", ondelete="SET NULL"), default=None
    )
    available_at: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)

    #: trs.ontology.enums.BiomarkerType
    biomarker_type: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: trs.ontology.enums.BiomarkerReadiness
    readiness: Mapped[str] = mapped_column(String(48), nullable=False, default="not_assessed")
    #: Does the marker behave equivalently in the preclinical species and human?
    cross_species_concordance: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )
    assay_validated_in_human: Mapped[bool | None] = mapped_column(Boolean, default=None)
    intended_use_in_trial: Mapped[str | None] = mapped_column(String(200), default=None)

    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None
    )
    provenance_method: Mapped[str] = mapped_column(String(48), nullable=False, default="human_curation")
    notes: Mapped[str | None] = mapped_column(Text, default=None)


class SafetyFinding(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """Preclinical toxicology / safety-pharmacology observations.

    Kept separate from efficacy evidence because safety translation is a
    distinct question with its own outcome (``phase_1_safety_failure``).
    """

    __tablename__ = "safety_finding"
    __table_args__ = (Index("ix_safety_track_available", "program_indication_id", "available_at"),)

    program_indication_id: Mapped[str | None] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), default=None, index=True
    )
    program_id: Mapped[str | None] = mapped_column(
        ForeignKey("program.id", ondelete="CASCADE"), default=None, index=True
    )
    target_id: Mapped[str | None] = mapped_column(
        ForeignKey("target.id", ondelete="SET NULL"), default=None
    )
    available_at: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)

    #: trs.ontology.enums.Species
    species: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    organ_system: Mapped[str | None] = mapped_column(String(120), default=None)
    finding: Mapped[str | None] = mapped_column(Text, default=None)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    reversible: Mapped[bool | None] = mapped_column(Boolean, default=None)
    monitorable_in_human: Mapped[bool | None] = mapped_column(Boolean, default=None)
    #: Whether the finding is attributed to the intended pharmacology.
    on_target: Mapped[bool | None] = mapped_column(Boolean, default=None)
    noael_dose: Mapped[float | None] = mapped_column(Float, default=None)
    noael_unit: Mapped[str | None] = mapped_column(String(48), default=None)
    #: Known class effect for this target or modality.
    is_known_class_effect: Mapped[bool | None] = mapped_column(Boolean, default=None)

    source_id: Mapped[str | None] = mapped_column(
        ForeignKey("source.id", ondelete="SET NULL"), default=None
    )
    provenance_method: Mapped[str] = mapped_column(String(48), nullable=False, default="human_curation")
    notes: Mapped[str | None] = mapped_column(Text, default=None)
