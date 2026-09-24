"""First-class biomedical entities and their alias graphs.

Why entity resolution is a first-class schema concern
-----------------------------------------------------
A therapeutic program is referred to by a discovery code, then a development
code, then an INN, then a brand name, and it changes sponsor once or twice on
the way. Publications cite the discovery code; the trial registry cites the
development code; the approval cites the INN. Treating those as four programs
would (a) inflate the cohort, (b) let the *same* biology appear in both the
training and the test fold, and (c) destroy the historical comparator feature.

So every nameable entity has an explicit alias table with the alias's source
and the method and confidence by which it was linked. Nothing is merged
silently: :mod:`trs.resolution` writes a resolution decision for every link,
and grouped cross-validation keys on the canonical entity id.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, Float, ForeignKey, Index, String, Text, UniqueConstraint
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
# Alias mixin
# ─────────────────────────────────────────────────────────────────────────────

class AliasMixin(TimestampMixin):
    """Shared shape for every alias table."""

    alias: Mapped[str] = mapped_column(String(300), nullable=False)
    #: Case/punctuation-folded form used for blocking and exact lookup.
    alias_normalised: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    alias_type: Mapped[str] = mapped_column(String(48), nullable=False, default="synonym")
    #: trs.ontology.enums.EntityMatchMethod
    match_method: Mapped[str] = mapped_column(String(32), nullable=False)
    match_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_id: Mapped[str | None] = mapped_column(String(32), default=None)
    #: Point in time from which this alias was publicly knowable.
    available_at: Mapped[dt.date | None] = mapped_column(Date, default=None)
    is_primary: Mapped[bool] = mapped_column(default=False, nullable=False)


# ─────────────────────────────────────────────────────────────────────────────
# Targets and pathways
# ─────────────────────────────────────────────────────────────────────────────

class Target(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A molecular target, canonicalised on HGNC symbol where one exists."""

    __tablename__ = "target"

    symbol: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(300), default=None)
    hgnc_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    ensembl_gene_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    uniprot_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    chembl_target_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    #: trs.ontology.enums.TargetClass
    target_class: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: Cross-species conservation of the target ortholog, 0–1 where measured.
    ortholog_identity_mouse: Mapped[float | None] = mapped_column(Float, default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    aliases: Mapped[list[TargetAlias]] = relationship(
        back_populates="target", cascade="all, delete-orphan"
    )


class TargetAlias(Base, IdMixin, AliasMixin):
    __tablename__ = "target_alias"
    __table_args__ = (UniqueConstraint("target_id", "alias_normalised"),)

    target_id: Mapped[str] = mapped_column(
        ForeignKey("target.id", ondelete="CASCADE"), index=True
    )
    target: Mapped[Target] = relationship(back_populates="aliases")


class Pathway(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    __tablename__ = "pathway"

    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    source_vocabulary: Mapped[str | None] = mapped_column(String(48), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)


class TargetPathway(Base, IdMixin, TimestampMixin):
    __tablename__ = "target_pathway"
    __table_args__ = (UniqueConstraint("target_id", "pathway_id"),)

    target_id: Mapped[str] = mapped_column(ForeignKey("target.id", ondelete="CASCADE"), index=True)
    pathway_id: Mapped[str] = mapped_column(ForeignKey("pathway.id", ondelete="CASCADE"), index=True)
    role: Mapped[str | None] = mapped_column(String(48), default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Compounds and sponsors
# ─────────────────────────────────────────────────────────────────────────────

class Compound(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A chemical or biological entity. ``inchikey`` is the strongest join key
    for small molecules; biologics rely on sequence/registry identifiers."""

    __tablename__ = "compound"

    preferred_name: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    inn: Mapped[str | None] = mapped_column(String(200), default=None, index=True)
    chembl_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    pubchem_cid: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    unii: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    inchikey: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    canonical_smiles: Mapped[str | None] = mapped_column(Text, default=None)
    #: trs.ontology.enums.Modality
    modality: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    first_disclosed_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    aliases: Mapped[list[CompoundAlias]] = relationship(
        back_populates="compound", cascade="all, delete-orphan"
    )


class CompoundAlias(Base, IdMixin, AliasMixin):
    __tablename__ = "compound_alias"
    __table_args__ = (UniqueConstraint("compound_id", "alias_normalised"),)

    compound_id: Mapped[str] = mapped_column(
        ForeignKey("compound.id", ondelete="CASCADE"), index=True
    )
    compound: Mapped[Compound] = relationship(back_populates="aliases")


class Sponsor(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A developing organisation. Distinct from :class:`~trs.db.models.tenancy.Organization`,
    which is a *tenant of this platform*; a Sponsor is a subject of study."""

    __tablename__ = "sponsor"

    name: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    name_normalised: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    sponsor_type: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    country: Mapped[str | None] = mapped_column(String(64), default=None)
    ultimate_parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("sponsor.id", ondelete="SET NULL"), default=None
    )
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    aliases: Mapped[list[SponsorAlias]] = relationship(
        back_populates="sponsor", cascade="all, delete-orphan"
    )


class SponsorAlias(Base, IdMixin, AliasMixin):
    __tablename__ = "sponsor_alias"
    __table_args__ = (UniqueConstraint("sponsor_id", "alias_normalised"),)

    sponsor_id: Mapped[str] = mapped_column(
        ForeignKey("sponsor.id", ondelete="CASCADE"), index=True
    )
    sponsor: Mapped[Sponsor] = relationship(back_populates="aliases")


# ─────────────────────────────────────────────────────────────────────────────
# Indications
# ─────────────────────────────────────────────────────────────────────────────

class Indication(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    __tablename__ = "indication"

    name: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    #: trs.ontology.enums.TherapeuticArea
    therapeutic_area: Mapped[str] = mapped_column(String(48), nullable=False, default="other")
    #: e.g. "EFO:0000305", "MONDO:0007254", "ICD10:C50"
    ontology_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    ontology_source: Mapped[str | None] = mapped_column(String(32), default=None)
    parent_indication_id: Mapped[str | None] = mapped_column(
        ForeignKey("indication.id", ondelete="SET NULL"), default=None
    )
    #: Free-text subtype descriptor, e.g. "KRAS G12C-mutant".
    molecular_subtype: Mapped[str | None] = mapped_column(String(200), default=None)
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    aliases: Mapped[list[IndicationAlias]] = relationship(
        back_populates="indication", cascade="all, delete-orphan"
    )


class IndicationAlias(Base, IdMixin, AliasMixin):
    __tablename__ = "indication_alias"
    __table_args__ = (UniqueConstraint("indication_id", "alias_normalised"),)

    indication_id: Mapped[str] = mapped_column(
        ForeignKey("indication.id", ondelete="CASCADE"), index=True
    )
    indication: Mapped[Indication] = relationship(back_populates="aliases")


class Biomarker(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    __tablename__ = "biomarker"

    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    #: trs.ontology.enums.BiomarkerType
    biomarker_type: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    analyte: Mapped[str | None] = mapped_column(String(200), default=None)
    matrix: Mapped[str | None] = mapped_column(String(80), default=None)
    target_id: Mapped[str | None] = mapped_column(
        ForeignKey("target.id", ondelete="SET NULL"), default=None
    )
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Programs
# ─────────────────────────────────────────────────────────────────────────────

class Program(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A therapeutic program: one asset pursued against one mechanism.

    A program persists across renaming, re-sponsoring and re-indication. It is
    the grouping key for leakage-resistant cross-validation.
    """

    __tablename__ = "program"
    __table_args__ = (Index("ix_program_org_project", "organization_id", "project_id"),)

    organization_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    project_id: Mapped[str | None] = mapped_column(
        ForeignKey("project.id", ondelete="SET NULL"), default=None, index=True
    )

    name: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    code: Mapped[str | None] = mapped_column(String(80), default=None, index=True)
    primary_compound_id: Mapped[str | None] = mapped_column(
        ForeignKey("compound.id", ondelete="SET NULL"), default=None, index=True
    )
    primary_target_id: Mapped[str | None] = mapped_column(
        ForeignKey("target.id", ondelete="SET NULL"), default=None, index=True
    )
    originator_sponsor_id: Mapped[str | None] = mapped_column(
        ForeignKey("sponsor.id", ondelete="SET NULL"), default=None, index=True
    )
    current_sponsor_id: Mapped[str | None] = mapped_column(
        ForeignKey("sponsor.id", ondelete="SET NULL"), default=None, index=True
    )
    #: trs.ontology.enums.Modality
    modality: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: Short, factual mechanism statement, e.g. "covalent KRAS G12C inhibitor".
    mechanism_of_action: Mapped[str | None] = mapped_column(Text, default=None)
    #: Earliest public disclosure of the asset; anchors the temporal window.
    first_disclosure_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    is_private: Mapped[bool] = mapped_column(default=False, nullable=False)
    external_refs: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    aliases: Mapped[list[ProgramAlias]] = relationship(
        back_populates="program", cascade="all, delete-orphan"
    )
    tracks: Mapped[list[ProgramIndication]] = relationship(
        back_populates="program", cascade="all, delete-orphan"
    )
    compound: Mapped[Compound | None] = relationship(foreign_keys=[primary_compound_id])
    target: Mapped[Target | None] = relationship(foreign_keys=[primary_target_id])
    current_sponsor: Mapped[Sponsor | None] = relationship(foreign_keys=[current_sponsor_id])
    originator_sponsor: Mapped[Sponsor | None] = relationship(foreign_keys=[originator_sponsor_id])


class ProgramAlias(Base, IdMixin, AliasMixin):
    """Development codes, INNs, brand names and sponsor-era designations."""

    __tablename__ = "program_alias"
    __table_args__ = (UniqueConstraint("program_id", "alias_normalised"),)

    program_id: Mapped[str] = mapped_column(
        ForeignKey("program.id", ondelete="CASCADE"), index=True
    )
    #: The sponsor under which this designation was used, when known.
    sponsor_id: Mapped[str | None] = mapped_column(
        ForeignKey("sponsor.id", ondelete="SET NULL"), default=None
    )
    program: Mapped[Program] = relationship(back_populates="aliases")


class ProgramTarget(Base, IdMixin, TimestampMixin):
    """Secondary / off-target / combination targets beyond the primary."""

    __tablename__ = "program_target"
    __table_args__ = (UniqueConstraint("program_id", "target_id", "relation"),)

    program_id: Mapped[str] = mapped_column(ForeignKey("program.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("target.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(32), nullable=False, default="primary")
    potency_nm: Mapped[float | None] = mapped_column(Float, default=None)
    available_at: Mapped[dt.date | None] = mapped_column(Date, default=None)


class ProgramIndication(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A development *track*: one program pursued in one indication.

    This is half of the analysis unit. The other half is the as-of date, which
    is not stored here because the same track is scored at many cutoffs — see
    :class:`~trs.db.models.analytics.CohortEntry` and
    :class:`~trs.db.models.analytics.FeatureSnapshot`.
    """

    __tablename__ = "program_indication"
    __table_args__ = (
        UniqueConstraint("program_id", "indication_id", "line_of_therapy"),
        Index("ix_program_indication_outcome", "outcome_label", "outcome_date"),
    )

    program_id: Mapped[str] = mapped_column(ForeignKey("program.id", ondelete="CASCADE"), index=True)
    indication_id: Mapped[str] = mapped_column(
        ForeignKey("indication.id", ondelete="CASCADE"), index=True
    )

    # ---- Intended clinical context (as designed at first human study) ------
    #: trs.ontology.enums.DiseaseStage
    disease_stage: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    line_of_therapy: Mapped[str] = mapped_column(String(48), nullable=False, default="unspecified")
    #: trs.ontology.enums.BiomarkerSelectionStrategy
    biomarker_selection: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    #: trs.ontology.enums.UnmetNeed
    unmet_need: Mapped[str] = mapped_column(String(48), nullable=False, default="not_assessed")
    standard_of_care_note: Mapped[str | None] = mapped_column(Text, default=None)
    combination_partner: Mapped[str | None] = mapped_column(String(200), default=None)

    # ---- Development milestones (each dated, for temporal filtering) -------
    preclinical_start_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    #: The canonical "decision point": entry into human development.
    ind_or_cta_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    first_in_human_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    phase_2_start_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    phase_3_start_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    approval_on: Mapped[dt.date | None] = mapped_column(Date, default=None)

    # ---- Observed outcome (the supervised target; never a feature) ---------
    #: trs.ontology.enums.DevelopmentPhase — highest phase actually reached.
    highest_phase_reached: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )
    #: trs.ontology.enums.OutcomeLabel
    outcome_label: Mapped[str] = mapped_column(String(48), nullable=False, default="unknown")
    outcome_date: Mapped[dt.date | None] = mapped_column(Date, default=None)
    #: trs.ontology.enums.DiscontinuationReason
    discontinuation_reason: Mapped[str] = mapped_column(
        String(48), nullable=False, default="unknown"
    )
    #: trs.ontology.enums.ConfidenceLevel — how well-attested the outcome is.
    outcome_confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="not_assessed")
    outcome_source_id: Mapped[str | None] = mapped_column(String(32), default=None)
    outcome_note: Mapped[str | None] = mapped_column(Text, default=None)

    #: Last date on which development status was verified. Anything after this
    #: is unobserved, not "no event".
    status_verified_on: Mapped[dt.date | None] = mapped_column(Date, default=None)

    program: Mapped[Program] = relationship(back_populates="tracks")
    indication: Mapped[Indication] = relationship()


class ProgramSponsorHistory(Base, IdMixin, TimestampMixin):
    """Who owned the asset, when. Prevents sponsor churn creating duplicates
    and lets sponsor-grouped splits use the owner *at the cutoff*."""

    __tablename__ = "program_sponsor_history"

    program_id: Mapped[str] = mapped_column(ForeignKey("program.id", ondelete="CASCADE"), index=True)
    sponsor_id: Mapped[str] = mapped_column(ForeignKey("sponsor.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(48), nullable=False, default="developer")
    start_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    end_on: Mapped[dt.date | None] = mapped_column(Date, default=None)
    source_id: Mapped[str | None] = mapped_column(String(32), default=None)


class EntityResolutionDecision(Base, IdMixin):
    """Audit record for every automated or human entity-linking decision.

    Written by :mod:`trs.resolution`. Nothing is merged without a row here, so
    a spurious merge can always be traced and reversed.
    """

    __tablename__ = "entity_resolution_decision"
    __table_args__ = (
        Index("ix_eres_entity", "entity_type", "canonical_id"),
    )

    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    canonical_id: Mapped[str] = mapped_column(String(32), nullable=False)
    candidate_key: Mapped[str] = mapped_column(String(300), nullable=False)
    #: trs.ontology.enums.EntityMatchMethod
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    accepted: Mapped[bool] = mapped_column(nullable=False, default=True)
    decided_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    decided_by: Mapped[str] = mapped_column(String(80), nullable=False, default="auto")
    evidence: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
