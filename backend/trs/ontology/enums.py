"""Closed controlled vocabularies.

Conventions
-----------
* Every enum is a ``str`` enum so values serialise transparently to JSON and
  store as ``VARCHAR`` in both PostgreSQL and SQLite.
* Values are lowercase ``snake_case`` and are **stable identifiers**: they are
  persisted in the database and embedded in model artifacts. Renaming a value
  is a breaking change requiring a feature-schema version bump and a migration.
* ``UNKNOWN`` is present wherever the real world permits genuine ignorance.
  ``UNKNOWN`` means "we do not know", which is *not* the same as a negative
  finding, and the feature layer keeps the two distinct (see
  ``trs.features.missingness``).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "EvidenceClass", "EvidenceDirection", "Species", "BiologicalSystem",
    "ModelType", "OncologyModelSubtype", "ImmuneCompetence", "ConstructValidity",
    "EndpointCategory", "EffectMetric", "Modality", "TargetClass",
    "GeneticEvidenceTier", "CausalSupport", "PathwayRedundancy",
    "ExposureBridgeQuality", "TargetEngagementEvidence", "BiomarkerType",
    "BiomarkerReadiness", "BlindingLevel", "RandomizationLevel",
    "ReplicationScope", "PublicationStatus", "SelectiveReportingRisk",
    "ConflictOfInterest", "TherapeuticArea", "DiseaseStage",
    "BiomarkerSelectionStrategy", "UnmetNeed", "DevelopmentPhase",
    "OutcomeLabel", "DiscontinuationReason", "SourceType", "ProvenanceMethod",
    "ConfidenceLevel", "MissingnessReason", "EntityMatchMethod", "OrgRole",
    "DEFAULT_EVIDENCE_CLASS_ORDER",
]


# ─────────────────────────────────────────────────────────────────────────────
# Evidence taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceClass(StrEnum):
    """What *kind* of study produced an evidence item.

    Ordered conceptually from most-removed to least-removed from human
    clinical reality, but the ordering carries **no hard-coded weight**: the
    relative informativeness of each class is estimated by the model, not
    asserted by the ontology. See ``DEFAULT_EVIDENCE_CLASS_ORDER`` for the
    display ordering only.
    """

    IN_SILICO = "in_silico"
    BIOCHEMICAL = "biochemical"
    CELL_LINE = "cell_line"
    PRIMARY_HUMAN_CELL = "primary_human_cell"
    ORGANOID = "organoid"
    HUMAN_EX_VIVO = "human_ex_vivo"
    ANIMAL_IN_VIVO = "animal_in_vivo"
    ANIMAL_TOXICOLOGY = "animal_toxicology"
    PK_PD_STUDY = "pk_pd_study"
    HUMAN_GENETIC = "human_genetic"
    HUMAN_OBSERVATIONAL = "human_observational"
    HUMAN_TISSUE_MOLECULAR = "human_tissue_molecular"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY_DOCUMENT = "regulatory_document"
    REAL_WORLD_EVIDENCE = "real_world_evidence"


DEFAULT_EVIDENCE_CLASS_ORDER: tuple[EvidenceClass, ...] = (
    EvidenceClass.IN_SILICO,
    EvidenceClass.BIOCHEMICAL,
    EvidenceClass.CELL_LINE,
    EvidenceClass.ANIMAL_IN_VIVO,
    EvidenceClass.ANIMAL_TOXICOLOGY,
    EvidenceClass.PK_PD_STUDY,
    EvidenceClass.ORGANOID,
    EvidenceClass.PRIMARY_HUMAN_CELL,
    EvidenceClass.HUMAN_EX_VIVO,
    EvidenceClass.HUMAN_TISSUE_MOLECULAR,
    EvidenceClass.HUMAN_GENETIC,
    EvidenceClass.HUMAN_OBSERVATIONAL,
    EvidenceClass.CLINICAL_TRIAL,
    EvidenceClass.REAL_WORLD_EVIDENCE,
    EvidenceClass.REGULATORY_DOCUMENT,
)


class EvidenceDirection(StrEnum):
    """Direction of an evidence item relative to the program's hypothesis."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Biological systems
# ─────────────────────────────────────────────────────────────────────────────

class Species(StrEnum):
    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    DOG = "dog"
    NON_HUMAN_PRIMATE = "non_human_primate"
    RABBIT = "rabbit"
    PIG = "pig"
    ZEBRAFISH = "zebrafish"
    NON_MAMMALIAN_OTHER = "non_mammalian_other"
    CELL_FREE = "cell_free"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class BiologicalSystem(StrEnum):
    """The experimental system, orthogonal to species."""

    IN_VIVO_WHOLE_ANIMAL = "in_vivo_whole_animal"
    EX_VIVO_TISSUE = "ex_vivo_tissue"
    ORGANOID = "organoid"
    PRIMARY_CELL = "primary_cell"
    IMMORTALISED_CELL_LINE = "immortalised_cell_line"
    CELL_FREE_BIOCHEMICAL = "cell_free_biochemical"
    COMPUTATIONAL = "computational"
    HUMAN_SUBJECT = "human_subject"
    HUMAN_POPULATION = "human_population"
    UNKNOWN = "unknown"


class ModelType(StrEnum):
    """Disease-model construction, therapeutic-area agnostic."""

    XENOGRAFT = "xenograft"
    SYNGENEIC = "syngeneic"
    GENETICALLY_ENGINEERED = "genetically_engineered"
    PATIENT_DERIVED_XENOGRAFT = "patient_derived_xenograft"
    ORTHOTOPIC = "orthotopic"
    CHEMICALLY_INDUCED = "chemically_induced"
    SURGICAL_INDUCED = "surgical_induced"
    SPONTANEOUS_DISEASE = "spontaneous_disease"
    HUMANISED = "humanised"
    HEALTHY_ANIMAL = "healthy_animal"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class OncologyModelSubtype(StrEnum):
    """Oncology-specific refinement of :class:`ModelType`.

    Kept separate so other therapeutic areas can add their own subtype
    vocabularies without polluting a single global enum.
    """

    CELL_LINE_DERIVED_XENOGRAFT_SUBQ = "cdx_subcutaneous"
    CELL_LINE_DERIVED_XENOGRAFT_ORTHO = "cdx_orthotopic"
    PDX_SUBCUTANEOUS = "pdx_subcutaneous"
    PDX_ORTHOTOPIC = "pdx_orthotopic"
    SYNGENEIC_SUBCUTANEOUS = "syngeneic_subcutaneous"
    SYNGENEIC_ORTHOTOPIC = "syngeneic_orthotopic"
    GEMM_AUTOCHTHONOUS = "gemm_autochthonous"
    HUMANISED_IMMUNE_MOUSE = "humanised_immune_mouse"
    METASTASIS_MODEL = "metastasis_model"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ImmuneCompetence(StrEnum):
    """Immune status of the host system.

    Material for immuno-oncology programs: an immunodeficient host cannot
    interrogate an immune-mediated mechanism.
    """

    FULLY_IMMUNOCOMPETENT = "fully_immunocompetent"
    PARTIALLY_RECONSTITUTED = "partially_reconstituted"
    HUMANISED_IMMUNE_SYSTEM = "humanised_immune_system"
    IMMUNODEFICIENT = "immunodeficient"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ConstructValidity(StrEnum):
    """Expert-assessed fidelity of the model to the human disease mechanism.

    This is a **judgement field**. It must be sourced from a named assessor
    (human curator or documented rubric) and is always accompanied by the
    assessor identity in provenance. It is never inferred by a language model.
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints and effects
# ─────────────────────────────────────────────────────────────────────────────

class EndpointCategory(StrEnum):
    TUMOUR_GROWTH_INHIBITION = "tumour_growth_inhibition"
    TUMOUR_REGRESSION = "tumour_regression"
    SURVIVAL = "survival"
    OBJECTIVE_RESPONSE = "objective_response"
    PROGRESSION_FREE_SURVIVAL = "progression_free_survival"
    OVERALL_SURVIVAL = "overall_survival"
    TARGET_ENGAGEMENT = "target_engagement"
    PHARMACODYNAMIC_MARKER = "pharmacodynamic_marker"
    BIOMARKER_MODULATION = "biomarker_modulation"
    HISTOPATHOLOGY = "histopathology"
    FUNCTIONAL_BEHAVIOURAL = "functional_behavioural"
    CELL_VIABILITY = "cell_viability"
    BINDING_AFFINITY = "binding_affinity"
    ENZYME_INHIBITION = "enzyme_inhibition"
    SAFETY_TOLERABILITY = "safety_tolerability"
    EXPOSURE_PK = "exposure_pk"
    OTHER = "other"
    UNKNOWN = "unknown"


class EffectMetric(StrEnum):
    """Units in which a reported effect size is expressed."""

    PERCENT_TGI = "percent_tgi"
    HAZARD_RATIO = "hazard_ratio"
    ODDS_RATIO = "odds_ratio"
    RISK_RATIO = "risk_ratio"
    FOLD_CHANGE = "fold_change"
    STANDARDISED_MEAN_DIFFERENCE = "standardised_mean_difference"
    ABSOLUTE_DIFFERENCE = "absolute_difference"
    PERCENT_CHANGE = "percent_change"
    IC50_NM = "ic50_nm"
    EC50_NM = "ec50_nm"
    KI_NM = "ki_nm"
    KD_NM = "kd_nm"
    RESPONSE_RATE = "response_rate"
    OTHER = "other"


# ─────────────────────────────────────────────────────────────────────────────
# Therapeutic programs
# ─────────────────────────────────────────────────────────────────────────────

class Modality(StrEnum):
    SMALL_MOLECULE = "small_molecule"
    MONOCLONAL_ANTIBODY = "monoclonal_antibody"
    BISPECIFIC_ANTIBODY = "bispecific_antibody"
    ANTIBODY_DRUG_CONJUGATE = "antibody_drug_conjugate"
    PEPTIDE = "peptide"
    THERAPEUTIC_PROTEIN = "therapeutic_protein"
    OLIGONUCLEOTIDE_ASO = "oligonucleotide_aso"
    SIRNA = "sirna"
    MRNA_THERAPY = "mrna_therapy"
    GENE_THERAPY = "gene_therapy"
    GENE_EDITING = "gene_editing"
    CELL_THERAPY_CAR_T = "cell_therapy_car_t"
    CELL_THERAPY_OTHER = "cell_therapy_other"
    ONCOLYTIC_VIRUS = "oncolytic_virus"
    THERAPEUTIC_VACCINE = "therapeutic_vaccine"
    PROPHYLACTIC_VACCINE = "prophylactic_vaccine"
    RADIOPHARMACEUTICAL = "radiopharmaceutical"
    PROTEIN_DEGRADER = "protein_degrader"
    MICROBIOME = "microbiome"
    COMBINATION_REGIMEN = "combination_regimen"
    OTHER = "other"
    UNKNOWN = "unknown"


class TargetClass(StrEnum):
    KINASE = "kinase"
    GPCR = "gpcr"
    ION_CHANNEL = "ion_channel"
    NUCLEAR_RECEPTOR = "nuclear_receptor"
    PROTEASE = "protease"
    EPIGENETIC_REGULATOR = "epigenetic_regulator"
    TRANSCRIPTION_FACTOR = "transcription_factor"
    GTPASE = "gtpase"
    PHOSPHATASE = "phosphatase"
    CELL_SURFACE_ANTIGEN = "cell_surface_antigen"
    IMMUNE_CHECKPOINT = "immune_checkpoint"
    CYTOKINE_OR_RECEPTOR = "cytokine_or_receptor"
    TRANSPORTER = "transporter"
    STRUCTURAL_PROTEIN = "structural_protein"
    RNA_TARGET = "rna_target"
    PROTEIN_PROTEIN_INTERACTION = "protein_protein_interaction"
    MULTI_TARGET = "multi_target"
    PHENOTYPIC_UNDEFINED = "phenotypic_undefined"
    OTHER = "other"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Target validation
# ─────────────────────────────────────────────────────────────────────────────

class GeneticEvidenceTier(StrEnum):
    """Strength of *human* genetic support linking target to disease.

    Tiers follow the logic used by large-scale target-validation efforts: a
    coding variant with a directionally consistent effect in humans is the
    strongest routinely available non-clinical evidence that modulating a
    target changes disease biology.
    """

    MENDELIAN_CAUSAL = "mendelian_causal"
    RARE_CODING_BURDEN = "rare_coding_burden"
    GWAS_FINE_MAPPED = "gwas_fine_mapped"
    GWAS_LOCUS_ONLY = "gwas_locus_only"
    SOMATIC_DRIVER = "somatic_driver"
    EXPRESSION_QTL_ONLY = "expression_qtl_only"
    NO_HUMAN_GENETIC_SIGNAL = "no_human_genetic_signal"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


class CausalSupport(StrEnum):
    """Whether the target–disease link rests on causal or associative data."""

    INTERVENTIONAL_HUMAN = "interventional_human"
    HUMAN_GENETIC_CAUSAL = "human_genetic_causal"
    MENDELIAN_RANDOMISATION = "mendelian_randomisation"
    ANIMAL_INTERVENTIONAL = "animal_interventional"
    CORRELATIONAL_ONLY = "correlational_only"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


class PathwayRedundancy(StrEnum):
    """Extent to which compensatory biology can bypass target inhibition."""

    NO_KNOWN_REDUNDANCY = "no_known_redundancy"
    LIMITED_REDUNDANCY = "limited_redundancy"
    SUBSTANTIAL_REDUNDANCY = "substantial_redundancy"
    DOCUMENTED_RESISTANCE_BYPASS = "documented_resistance_bypass"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# PK/PD translational bridge
# ─────────────────────────────────────────────────────────────────────────────

class ExposureBridgeQuality(StrEnum):
    """Quality of the argument that efficacious preclinical exposure is
    achievable and tolerable in humans."""

    HUMAN_EXPOSURE_DEMONSTRATED = "human_exposure_demonstrated"
    ALLOMETRIC_WITH_PK_MODEL = "allometric_with_pk_model"
    ALLOMETRIC_SIMPLE = "allometric_simple"
    ASSUMED_NOT_MODELLED = "assumed_not_modelled"
    EXPOSURE_GAP_IDENTIFIED = "exposure_gap_identified"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


class TargetEngagementEvidence(StrEnum):
    HUMAN_DIRECT = "human_direct"
    HUMAN_INDIRECT_PD = "human_indirect_pd"
    ANIMAL_DIRECT = "animal_direct"
    ANIMAL_INDIRECT_PD = "animal_indirect_pd"
    IN_VITRO_ONLY = "in_vitro_only"
    NONE_DEMONSTRATED = "none_demonstrated"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


class BiomarkerType(StrEnum):
    PHARMACODYNAMIC = "pharmacodynamic"
    TARGET_ENGAGEMENT = "target_engagement"
    PREDICTIVE_SELECTION = "predictive_selection"
    PROGNOSTIC = "prognostic"
    SAFETY = "safety"
    SURROGATE_ENDPOINT = "surrogate_endpoint"
    MONITORING = "monitoring"
    OTHER = "other"
    UNKNOWN = "unknown"


class BiomarkerReadiness(StrEnum):
    """How ready a biomarker is to be used in a human trial at the cutoff."""

    CLINICALLY_QUALIFIED = "clinically_qualified"
    VALIDATED_ASSAY_HUMAN = "validated_assay_human"
    FIT_FOR_PURPOSE_HUMAN = "fit_for_purpose_human"
    PRECLINICAL_ONLY = "preclinical_only"
    CONCEPTUAL = "conceptual"
    NONE_AVAILABLE = "none_available"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Study quality / reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class BlindingLevel(StrEnum):
    OUTCOME_ASSESSOR_BLINDED = "outcome_assessor_blinded"
    FULLY_BLINDED = "fully_blinded"
    PARTIALLY_BLINDED = "partially_blinded"
    NOT_BLINDED = "not_blinded"
    NOT_REPORTED = "not_reported"
    NOT_APPLICABLE = "not_applicable"


class RandomizationLevel(StrEnum):
    RANDOMISED_CONCEALED = "randomised_concealed"
    RANDOMISED = "randomised"
    SYSTEMATIC_ALLOCATION = "systematic_allocation"
    NOT_RANDOMISED = "not_randomised"
    NOT_REPORTED = "not_reported"
    NOT_APPLICABLE = "not_applicable"


class ReplicationScope(StrEnum):
    """Breadth of replication behind an evidence item or a claim."""

    INDEPENDENT_LABS_MULTIPLE = "independent_labs_multiple"
    INDEPENDENT_LAB_SINGLE = "independent_lab_single"
    SAME_LAB_MULTIPLE_STUDIES = "same_lab_multiple_studies"
    SINGLE_STUDY = "single_study"
    NOT_REPORTED = "not_reported"
    UNKNOWN = "unknown"


class PublicationStatus(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    CONFERENCE_ABSTRACT = "conference_abstract"
    REGULATORY_FILING = "regulatory_filing"
    PATENT = "patent"
    COMPANY_DISCLOSURE = "company_disclosure"
    INTERNAL_UNPUBLISHED = "internal_unpublished"
    UNKNOWN = "unknown"


class SelectiveReportingRisk(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    NOT_ASSESSED = "not_assessed"


class ConflictOfInterest(StrEnum):
    NONE_DECLARED = "none_declared"
    ACADEMIC_ONLY = "academic_only"
    SPONSOR_AFFILIATED = "sponsor_affiliated"
    SPONSOR_FUNDED_INDEPENDENT = "sponsor_funded_independent"
    UNDISCLOSED = "undisclosed"
    NOT_REPORTED = "not_reported"


# ─────────────────────────────────────────────────────────────────────────────
# Disease context
# ─────────────────────────────────────────────────────────────────────────────

class TherapeuticArea(StrEnum):
    ONCOLOGY = "oncology"
    HAEMATOLOGY = "haematology"
    IMMUNOLOGY_INFLAMMATION = "immunology_inflammation"
    NEUROLOGY = "neurology"
    PSYCHIATRY = "psychiatry"
    CARDIOVASCULAR = "cardiovascular"
    METABOLIC = "metabolic"
    INFECTIOUS_DISEASE = "infectious_disease"
    RESPIRATORY = "respiratory"
    RENAL = "renal"
    HEPATOLOGY = "hepatology"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    MUSCULOSKELETAL = "musculoskeletal"
    RARE_GENETIC = "rare_genetic"
    OTHER = "other"


class DiseaseStage(StrEnum):
    PREVENTION = "prevention"
    EARLY_STAGE = "early_stage"
    LOCALLY_ADVANCED = "locally_advanced"
    METASTATIC = "metastatic"
    RELAPSED_REFRACTORY = "relapsed_refractory"
    MAINTENANCE = "maintenance"
    ADJUVANT = "adjuvant"
    NEOADJUVANT = "neoadjuvant"
    CHRONIC = "chronic"
    ACUTE = "acute"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class BiomarkerSelectionStrategy(StrEnum):
    """How the intended trial population is selected."""

    REQUIRED_GENOMIC_ALTERATION = "required_genomic_alteration"
    REQUIRED_PROTEIN_EXPRESSION = "required_protein_expression"
    ENRICHED_NOT_REQUIRED = "enriched_not_required"
    ALL_COMERS = "all_comers"
    HISTOLOGY_ONLY = "histology_only"
    NOT_DEFINED = "not_defined"
    UNKNOWN = "unknown"


class UnmetNeed(StrEnum):
    NO_APPROVED_THERAPY = "no_approved_therapy"
    LIMITED_EFFICACY_SOC = "limited_efficacy_soc"
    EFFECTIVE_SOC_TOLERABILITY_GAP = "effective_soc_tolerability_gap"
    CROWDED_EFFECTIVE_SOC = "crowded_effective_soc"
    NOT_ASSESSED = "not_assessed"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Development outcomes
# ─────────────────────────────────────────────────────────────────────────────

class DevelopmentPhase(StrEnum):
    DISCOVERY = "discovery"
    LEAD_OPTIMISATION = "lead_optimisation"
    IND_ENABLING = "ind_enabling"
    PHASE_1 = "phase_1"
    PHASE_1_2 = "phase_1_2"
    PHASE_2 = "phase_2"
    PHASE_2_3 = "phase_2_3"
    PHASE_3 = "phase_3"
    FILED = "filed"
    APPROVED = "approved"
    UNKNOWN = "unknown"


class OutcomeLabel(StrEnum):
    """Mutually exclusive terminal states used to derive supervised labels.

    The MVP primary endpoint (``ph2_efficacy_failure``) is derived from these
    states by :mod:`trs.cohort.outcomes`, which applies an explicit, documented
    inclusion/exclusion rule rather than collapsing every non-success into
    "failure".
    """

    FAILED_BEFORE_HUMAN_DOSING = "failed_before_human_dosing"
    PHASE_1_SAFETY_FAILURE = "phase_1_safety_failure"
    PHASE_1_PKPD_FAILURE = "phase_1_pkpd_failure"
    PHASE_2_EFFICACY_FAILURE = "phase_2_efficacy_failure"
    PHASE_2_SAFETY_FAILURE = "phase_2_safety_failure"
    DISCONTINUED_THROUGH_PHASE_2 = "discontinued_through_phase_2"
    PROGRESSED_PAST_PHASE_2 = "progressed_past_phase_2"
    PHASE_3_FAILURE = "phase_3_failure"
    REGULATORY_APPROVAL = "regulatory_approval"
    ACTIVE_NO_OUTCOME_YET = "active_no_outcome_yet"
    CENSORED_INSUFFICIENT_FOLLOWUP = "censored_insufficient_followup"
    UNKNOWN = "unknown"


class DiscontinuationReason(StrEnum):
    """Why development stopped.

    ``COMMERCIAL_STRATEGIC`` and ``SPONSOR_INSOLVENCY`` are **not** biological
    failures. The cohort builder excludes them from the efficacy-failure label
    by default and reports how many programs were removed on that basis.
    """

    EFFICACY_INSUFFICIENT = "efficacy_insufficient"
    SAFETY_TOXICITY = "safety_toxicity"
    PK_PD_INADEQUATE = "pk_pd_inadequate"
    MANUFACTURING_CMC = "manufacturing_cmc"
    COMMERCIAL_STRATEGIC = "commercial_strategic"
    PORTFOLIO_REPRIORITISATION = "portfolio_reprioritisation"
    SPONSOR_INSOLVENCY = "sponsor_insolvency"
    COMPETITIVE_LANDSCAPE = "competitive_landscape"
    ENROLMENT_FEASIBILITY = "enrolment_feasibility"
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    SUPERSEDED_BY_BACKUP = "superseded_by_backup"
    NOT_DISCONTINUED = "not_discontinued"
    UNDISCLOSED = "undisclosed"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────

class SourceType(StrEnum):
    JOURNAL_ARTICLE = "journal_article"
    PREPRINT = "preprint"
    CLINICAL_TRIAL_REGISTRY = "clinical_trial_registry"
    REGULATORY_LABEL = "regulatory_label"
    REGULATORY_REVIEW = "regulatory_review"
    CONFERENCE_ABSTRACT = "conference_abstract"
    PATENT = "patent"
    COMPANY_PRESENTATION = "company_presentation"
    PRESS_RELEASE = "press_release"
    STRUCTURED_DATABASE = "structured_database"
    EXPERT_CURATION = "expert_curation"
    INTERNAL_EXPERIMENT = "internal_experiment"
    SYNTHETIC_DEMONSTRATION = "synthetic_demonstration"
    UNKNOWN = "unknown"


class ProvenanceMethod(StrEnum):
    """How a value got from its source into our database.

    ``LLM_ASSISTED_EXTRACTION`` is permitted only for locating and normalising
    values that a human or a deterministic rule then confirms; the resulting
    record retains the verbatim source snippet. No value may originate from a
    language model's unsupported generation — see
    :data:`ProvenanceMethod.MODEL_GENERATED_UNVERIFIED`, which the validation
    layer rejects from any model-ready dataset.
    """

    API_STRUCTURED = "api_structured"
    RULE_BASED_PARSE = "rule_based_parse"
    REGEX_EXTRACTION = "regex_extraction"
    LLM_ASSISTED_EXTRACTION = "llm_assisted_extraction"
    HUMAN_CURATION = "human_curation"
    HUMAN_VERIFIED_LLM = "human_verified_llm"
    DERIVED_COMPUTATION = "derived_computation"
    SYNTHETIC_GENERATOR = "synthetic_generator"
    MODEL_GENERATED_UNVERIFIED = "model_generated_unverified"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    NOT_ASSESSED = "not_assessed"


class MissingnessReason(StrEnum):
    """Why a field has no value. Distinguishing these is load-bearing.

    ``NOT_REPORTED`` (the study exists but is silent) is informative about
    study quality. ``NOT_APPLICABLE`` (the question does not arise) is not.
    ``NOT_YET_AVAILABLE`` means the information exists but postdates the
    as-of cutoff and was therefore withheld to preserve temporal integrity.
    """

    NOT_REPORTED = "not_reported"
    NOT_APPLICABLE = "not_applicable"
    NOT_YET_AVAILABLE = "not_yet_available"
    NOT_EXTRACTED = "not_extracted"
    SOURCE_INACCESSIBLE = "source_inaccessible"
    CONFLICTING_SOURCES = "conflicting_sources"
    PRESENT = "present"


class EntityMatchMethod(StrEnum):
    EXACT_IDENTIFIER = "exact_identifier"
    NORMALISED_NAME = "normalised_name"
    REGISTERED_ALIAS = "registered_alias"
    STRUCTURE_INCHIKEY = "structure_inchikey"
    CROSS_REFERENCE = "cross_reference"
    FUZZY_STRING = "fuzzy_string"
    HUMAN_ASSERTED = "human_asserted"
    UNRESOLVED = "unresolved"


class OrgRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
