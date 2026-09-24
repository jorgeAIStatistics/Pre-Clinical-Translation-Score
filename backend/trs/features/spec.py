"""The versioned feature schema.

This module is the contract between evidence and model. It declares, for every
feature: what it means, which evidence produces it, how it is encoded, what
"missing" means for it, and whether a user may vary it in scenario analysis.

Design commitments
------------------
**Declarative, not implicit.** A feature that is not declared here cannot enter
a model matrix. ``FEATURE_SCHEMA_VERSION`` is embedded in every feature
snapshot and every model artifact; a model can only score a snapshot built
under the schema version it was trained on.

**Ordinal encodings are assumptions, and are written down.** Mapping an ordered
category to an integer asserts equal spacing between adjacent levels. With the
sample sizes realistically available for translational cohorts (hundreds, not
millions) that assumption buys a large variance reduction over one-hot coding,
and it is the right trade — but it is a trade. Each ordinal map below carries
the scale it encodes. Where the spacing assumption is not defensible
(modality, target class) the feature is nominal and one-hot encoded.

**Direction hypotheses are recorded but never enforced.** ``expected_direction``
states the prior belief about a feature's association with failure. It is used
only to flag sign-discordant coefficients in the validation report for human
review. It never constrains fitting: a model that silently enforces the
analyst's priors cannot surprise the analyst, and surprising the analyst is the
point.

**Missingness is a modelled quantity, not a default.** See
:class:`MissingPolicy`. "Not reported" and "not applicable" are different
facts and are encoded differently.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum

from trs.ontology import enums as E

#: Bump on ANY change to feature names, encodings, ordinal maps or aggregation
#: rules. Models and snapshots are keyed on this; mismatches are hard errors.
FEATURE_SCHEMA_VERSION = "fs-1.0.0"


class FeatureDomain(StrEnum):
    """Evidence domains. These are the rows of the translational scorecard."""

    PROGRAM_CONTEXT = "program_context"
    ANIMAL_MODEL_VALIDITY = "animal_model_validity"
    CROSS_SPECIES_CONCORDANCE = "cross_species_concordance"
    PKPD_TRANSLATION = "pkpd_translation"
    HUMAN_BIOLOGICAL_SUPPORT = "human_biological_support"
    TARGET_VALIDATION = "target_validation"
    DISEASE_MODEL_RELEVANCE = "disease_model_relevance"
    REPRODUCIBILITY = "reproducibility"
    SAFETY_TRANSLATION = "safety_translation"
    BIOMARKER_READINESS = "biomarker_readiness"
    HISTORICAL_CLASS_PERFORMANCE = "historical_class_performance"
    EVIDENCE_VOLUME = "evidence_volume"


DOMAIN_LABELS: dict[FeatureDomain, str] = {
    FeatureDomain.PROGRAM_CONTEXT: "Program context",
    FeatureDomain.ANIMAL_MODEL_VALIDITY: "Animal-model validity",
    FeatureDomain.CROSS_SPECIES_CONCORDANCE: "Cross-species concordance",
    FeatureDomain.PKPD_TRANSLATION: "PK/PD translation",
    FeatureDomain.HUMAN_BIOLOGICAL_SUPPORT: "Human biological support",
    FeatureDomain.TARGET_VALIDATION: "Target validation",
    FeatureDomain.DISEASE_MODEL_RELEVANCE: "Disease-model relevance",
    FeatureDomain.REPRODUCIBILITY: "Reproducibility & study quality",
    FeatureDomain.SAFETY_TRANSLATION: "Safety translation",
    FeatureDomain.BIOMARKER_READINESS: "Biomarker readiness",
    FeatureDomain.HISTORICAL_CLASS_PERFORMANCE: "Historical class performance",
    FeatureDomain.EVIDENCE_VOLUME: "Evidence volume & recency",
}

DOMAIN_DESCRIPTIONS: dict[FeatureDomain, str] = {
    FeatureDomain.PROGRAM_CONTEXT: (
        "Modality and target class. Translational performance is not uniform "
        "across modalities, so these enter the model explicitly rather than "
        "being averaged away."
    ),
    FeatureDomain.ANIMAL_MODEL_VALIDITY: (
        "Whether the animal experiments that motivated the program were "
        "constructed to interrogate the human disease mechanism, and how "
        "strong and how precise the observed effects were."
    ),
    FeatureDomain.CROSS_SPECIES_CONCORDANCE: (
        "Whether the effect reproduces across species and systems, or rests on "
        "a single model. Species are not assumed equally informative; their "
        "relative weight is estimated."
    ),
    FeatureDomain.PKPD_TRANSLATION: (
        "Whether the exposure associated with preclinical efficacy is "
        "plausibly achievable and tolerable in humans, and whether target "
        "engagement has been demonstrated."
    ),
    FeatureDomain.HUMAN_BIOLOGICAL_SUPPORT: (
        "Evidence generated in human material or human populations: tissue, "
        "primary cells, organoids, genetics, prior clinical experience."
    ),
    FeatureDomain.TARGET_VALIDATION: (
        "Strength and causal character of the target–disease link, and the "
        "extent to which compensatory biology can bypass the target."
    ),
    FeatureDomain.DISEASE_MODEL_RELEVANCE: (
        "Fit between the intended trial population and the biology the "
        "preclinical package interrogates."
    ),
    FeatureDomain.REPRODUCIBILITY: (
        "Internal-validity features of the supporting studies: randomisation, "
        "blinding, replication across independent laboratories, "
        "pre-registration, reporting integrity."
    ),
    FeatureDomain.SAFETY_TRANSLATION: (
        "Preclinical safety findings that constrain the dose, and therefore "
        "the exposure, achievable in humans."
    ),
    FeatureDomain.BIOMARKER_READINESS: (
        "Availability of a pharmacodynamic or patient-selection biomarker that "
        "is ready to be deployed in the first human studies."
    ),
    FeatureDomain.HISTORICAL_CLASS_PERFORMANCE: (
        "Observed outcomes of earlier programs sharing the target, mechanism "
        "class or indication — counted strictly as of the cutoff date."
    ),
    FeatureDomain.EVIDENCE_VOLUME: (
        "How much evidence exists and how current it is. Confounded with "
        "publication and disclosure behaviour; see MODEL_CARD.md."
    ),
}


class FeatureKind(StrEnum):
    NUMERIC = "numeric"
    ORDINAL = "ordinal"
    BINARY = "binary"
    NOMINAL = "nominal"       # expanded to one-hot columns
    COUNT = "count"           # non-negative integer, log1p-transformed


class Transform(StrEnum):
    IDENTITY = "identity"
    LOG1P = "log1p"
    LOG2 = "log2"             # for ratios; requires strictly positive input
    LOGIT = "logit"           # for proportions in (0, 1), clipped
    NONE = "none"


class MissingPolicy(StrEnum):
    """How a missing value is handled when building a model matrix.

    ``IMPUTE_WITH_INDICATOR``
        Fill with the training-set median (numeric) or mode (categorical) and
        add a companion ``{feature}__missing`` column. The model can then learn
        an effect of *not knowing*, separately from the effect of the value.
        This is the default for anything where silence is informative.

    ``IMPUTE_ONLY``
        Fill, no indicator. Used where missingness is near-universal or
        near-absent, so an indicator would be collinear or degenerate.

    ``ZERO_IS_ABSENCE``
        Missing genuinely means zero — e.g. "count of non-human-primate
        studies" is 0, not unknown, when no such study exists in a complete
        evidence sweep. Used only where the evidence sweep is exhaustive by
        construction.

    ``EXPLICIT_LEVEL``
        For nominal features: "unknown" is a legitimate level with its own
        one-hot column.
    """

    IMPUTE_WITH_INDICATOR = "impute_with_indicator"
    IMPUTE_ONLY = "impute_only"
    ZERO_IS_ABSENCE = "zero_is_absence"
    EXPLICIT_LEVEL = "explicit_level"


class Direction(StrEnum):
    """Prior belief about the association with the *failure* outcome."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    UNCERTAIN = "uncertain"


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    domain: FeatureDomain
    kind: FeatureKind
    label: str
    description: str
    #: Plain-language statement of what evidence produces this value.
    derivation: str
    transform: Transform = Transform.IDENTITY
    missing_policy: MissingPolicy = MissingPolicy.IMPUTE_WITH_INDICATOR
    expected_direction: Direction = Direction.UNCERTAIN
    #: Inclusive bounds used for validation, not for clipping model input.
    valid_min: float | None = None
    valid_max: float | None = None
    #: Levels for NOMINAL features; the first is the reference (dropped) level.
    levels: tuple[str, ...] = ()
    #: Weight in the completeness score. Features that materially move the
    #: estimate are weighted higher, so "80% complete" means something.
    completeness_weight: float = 1.0
    #: True when a user may vary this in scenario analysis.
    scenario_adjustable: bool = False
    unit: str | None = None

    @property
    def is_nominal(self) -> bool:
        return self.kind is FeatureKind.NOMINAL


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal encodings
#
# Each map states the scale it represents. Integers are ordered; spacing is
# assumed uniform unless the map deliberately skips values to encode a larger
# qualitative gap (e.g. human-derived vs animal-derived target engagement).
# ─────────────────────────────────────────────────────────────────────────────

#: Qualitative strength scale shared by construct validity, disease similarity,
#: tissue-expression evidence and biomarker concordance. 0 = contested/absent.
QUALITATIVE_STRENGTH: dict[str, float] = {
    "contested": 0.0,
    "low": 1.0,
    "moderate": 2.0,
    "high": 3.0,
}

#: Human genetic support. The gap between any human genetic signal and none is
#: encoded as larger than the gaps between adjacent tiers, because the
#: qualitative distinction (does human biology speak at all?) is the one that
#: matters most for target validation.
GENETIC_EVIDENCE_ORDINAL: dict[str, float] = {
    E.GeneticEvidenceTier.NO_HUMAN_GENETIC_SIGNAL: 0.0,
    E.GeneticEvidenceTier.EXPRESSION_QTL_ONLY: 1.5,
    E.GeneticEvidenceTier.GWAS_LOCUS_ONLY: 2.0,
    E.GeneticEvidenceTier.GWAS_FINE_MAPPED: 3.0,
    E.GeneticEvidenceTier.SOMATIC_DRIVER: 3.5,
    E.GeneticEvidenceTier.RARE_CODING_BURDEN: 4.0,
    E.GeneticEvidenceTier.MENDELIAN_CAUSAL: 5.0,
}

CAUSAL_SUPPORT_ORDINAL: dict[str, float] = {
    E.CausalSupport.CORRELATIONAL_ONLY: 0.0,
    E.CausalSupport.ANIMAL_INTERVENTIONAL: 1.0,
    E.CausalSupport.MENDELIAN_RANDOMISATION: 2.0,
    E.CausalSupport.HUMAN_GENETIC_CAUSAL: 3.0,
    E.CausalSupport.INTERVENTIONAL_HUMAN: 4.0,
}

#: Higher = more redundancy = more ways for the biology to route around the
#: drug. Oriented so that larger values mean greater risk.
PATHWAY_REDUNDANCY_ORDINAL: dict[str, float] = {
    E.PathwayRedundancy.NO_KNOWN_REDUNDANCY: 0.0,
    E.PathwayRedundancy.LIMITED_REDUNDANCY: 1.0,
    E.PathwayRedundancy.SUBSTANTIAL_REDUNDANCY: 2.0,
    E.PathwayRedundancy.DOCUMENTED_RESISTANCE_BYPASS: 3.0,
}

EXPOSURE_BRIDGE_ORDINAL: dict[str, float] = {
    E.ExposureBridgeQuality.EXPOSURE_GAP_IDENTIFIED: 0.0,
    E.ExposureBridgeQuality.ASSUMED_NOT_MODELLED: 1.0,
    E.ExposureBridgeQuality.ALLOMETRIC_SIMPLE: 2.0,
    E.ExposureBridgeQuality.ALLOMETRIC_WITH_PK_MODEL: 3.0,
    E.ExposureBridgeQuality.HUMAN_EXPOSURE_DEMONSTRATED: 4.0,
}

#: The jump from any animal demonstration to a human demonstration is encoded
#: as two steps: engagement in humans answers a question animal data cannot.
TARGET_ENGAGEMENT_ORDINAL: dict[str, float] = {
    E.TargetEngagementEvidence.NONE_DEMONSTRATED: 0.0,
    E.TargetEngagementEvidence.IN_VITRO_ONLY: 1.0,
    E.TargetEngagementEvidence.ANIMAL_INDIRECT_PD: 2.0,
    E.TargetEngagementEvidence.ANIMAL_DIRECT: 3.0,
    E.TargetEngagementEvidence.HUMAN_INDIRECT_PD: 5.0,
    E.TargetEngagementEvidence.HUMAN_DIRECT: 6.0,
}

BIOMARKER_READINESS_ORDINAL: dict[str, float] = {
    E.BiomarkerReadiness.NONE_AVAILABLE: 0.0,
    E.BiomarkerReadiness.CONCEPTUAL: 1.0,
    E.BiomarkerReadiness.PRECLINICAL_ONLY: 2.0,
    E.BiomarkerReadiness.FIT_FOR_PURPOSE_HUMAN: 3.0,
    E.BiomarkerReadiness.VALIDATED_ASSAY_HUMAN: 4.0,
    E.BiomarkerReadiness.CLINICALLY_QUALIFIED: 5.0,
}

REPLICATION_ORDINAL: dict[str, float] = {
    E.ReplicationScope.SINGLE_STUDY: 0.0,
    E.ReplicationScope.SAME_LAB_MULTIPLE_STUDIES: 1.0,
    E.ReplicationScope.INDEPENDENT_LAB_SINGLE: 2.5,
    E.ReplicationScope.INDEPENDENT_LABS_MULTIPLE: 4.0,
}

SELECTIVE_REPORTING_ORDINAL: dict[str, float] = {
    E.SelectiveReportingRisk.LOW: 0.0,
    E.SelectiveReportingRisk.MODERATE: 1.0,
    E.SelectiveReportingRisk.HIGH: 2.0,
}

#: Later lines and refractory settings are harder places to show efficacy, but
#: also settings where a modest effect is more visible against a weak control.
#: The direction is genuinely uncertain, so the model estimates it.
DISEASE_STAGE_ORDINAL: dict[str, float] = {
    E.DiseaseStage.PREVENTION: 0.0,
    E.DiseaseStage.EARLY_STAGE: 1.0,
    E.DiseaseStage.ADJUVANT: 1.5,
    E.DiseaseStage.NEOADJUVANT: 1.5,
    E.DiseaseStage.LOCALLY_ADVANCED: 2.0,
    E.DiseaseStage.MAINTENANCE: 2.5,
    E.DiseaseStage.METASTATIC: 3.0,
    E.DiseaseStage.RELAPSED_REFRACTORY: 4.0,
    E.DiseaseStage.CHRONIC: 2.0,
    E.DiseaseStage.ACUTE: 2.0,
}

UNMET_NEED_ORDINAL: dict[str, float] = {
    E.UnmetNeed.CROWDED_EFFECTIVE_SOC: 0.0,
    E.UnmetNeed.EFFECTIVE_SOC_TOLERABILITY_GAP: 1.0,
    E.UnmetNeed.LIMITED_EFFICACY_SOC: 2.0,
    E.UnmetNeed.NO_APPROVED_THERAPY: 3.0,
}

SEVERITY_ORDINAL: dict[str, float] = {
    "none": 0.0, "minimal": 1.0, "mild": 2.0,
    "moderate": 3.0, "severe": 4.0, "life_threatening": 5.0,
}

#: Modality groups. Individual modalities are pooled into classes with shared
#: translational behaviour so that rare modalities do not each get a coefficient
#: estimated from two observations. The mapping is a modelling choice and is
#: restated in MODELING_PLAN.md.
MODALITY_GROUPS: dict[str, str] = {
    E.Modality.SMALL_MOLECULE: "small_molecule",
    E.Modality.PROTEIN_DEGRADER: "small_molecule",
    E.Modality.MONOCLONAL_ANTIBODY: "antibody",
    E.Modality.BISPECIFIC_ANTIBODY: "antibody",
    E.Modality.ANTIBODY_DRUG_CONJUGATE: "adc",
    E.Modality.RADIOPHARMACEUTICAL: "adc",
    E.Modality.PEPTIDE: "peptide_protein",
    E.Modality.THERAPEUTIC_PROTEIN: "peptide_protein",
    E.Modality.OLIGONUCLEOTIDE_ASO: "nucleic_acid",
    E.Modality.SIRNA: "nucleic_acid",
    E.Modality.MRNA_THERAPY: "nucleic_acid",
    E.Modality.GENE_THERAPY: "advanced_therapy",
    E.Modality.GENE_EDITING: "advanced_therapy",
    E.Modality.CELL_THERAPY_CAR_T: "cell_therapy",
    E.Modality.CELL_THERAPY_OTHER: "cell_therapy",
    E.Modality.ONCOLYTIC_VIRUS: "immunotherapy_other",
    E.Modality.THERAPEUTIC_VACCINE: "immunotherapy_other",
    E.Modality.PROPHYLACTIC_VACCINE: "immunotherapy_other",
    E.Modality.MICROBIOME: "other",
    E.Modality.COMBINATION_REGIMEN: "other",
    E.Modality.OTHER: "other",
    E.Modality.UNKNOWN: "unknown",
}

MODALITY_GROUP_LEVELS: tuple[str, ...] = (
    "small_molecule",       # reference level: the most common, best-characterised
    "antibody",
    "adc",
    "peptide_protein",
    "nucleic_acid",
    "cell_therapy",
    "advanced_therapy",
    "immunotherapy_other",
    "other",
    "unknown",
)

TARGET_CLASS_GROUPS: dict[str, str] = {
    E.TargetClass.KINASE: "kinase",
    E.TargetClass.PHOSPHATASE: "kinase",
    E.TargetClass.GPCR: "gpcr",
    E.TargetClass.ION_CHANNEL: "channel_transporter",
    E.TargetClass.TRANSPORTER: "channel_transporter",
    E.TargetClass.NUCLEAR_RECEPTOR: "nuclear_receptor",
    E.TargetClass.PROTEASE: "enzyme_other",
    E.TargetClass.GTPASE: "enzyme_other",
    E.TargetClass.EPIGENETIC_REGULATOR: "epigenetic",
    E.TargetClass.TRANSCRIPTION_FACTOR: "transcription_factor",
    E.TargetClass.CELL_SURFACE_ANTIGEN: "surface_antigen",
    E.TargetClass.IMMUNE_CHECKPOINT: "immune",
    E.TargetClass.CYTOKINE_OR_RECEPTOR: "immune",
    E.TargetClass.STRUCTURAL_PROTEIN: "other",
    E.TargetClass.RNA_TARGET: "other",
    E.TargetClass.PROTEIN_PROTEIN_INTERACTION: "ppi",
    E.TargetClass.MULTI_TARGET: "other",
    E.TargetClass.PHENOTYPIC_UNDEFINED: "phenotypic",
    E.TargetClass.OTHER: "other",
    E.TargetClass.UNKNOWN: "unknown",
}

TARGET_CLASS_GROUP_LEVELS: tuple[str, ...] = (
    "kinase",               # reference level
    "gpcr",
    "channel_transporter",
    "nuclear_receptor",
    "enzyme_other",
    "epigenetic",
    "transcription_factor",
    "surface_antigen",
    "immune",
    "ppi",
    "phenotypic",
    "other",
    "unknown",
)

ORDINAL_MAPS: dict[str, dict[str, float]] = {
    "qualitative_strength": QUALITATIVE_STRENGTH,
    "genetic_evidence": GENETIC_EVIDENCE_ORDINAL,
    "causal_support": CAUSAL_SUPPORT_ORDINAL,
    "pathway_redundancy": PATHWAY_REDUNDANCY_ORDINAL,
    "exposure_bridge": EXPOSURE_BRIDGE_ORDINAL,
    "target_engagement": TARGET_ENGAGEMENT_ORDINAL,
    "biomarker_readiness": BIOMARKER_READINESS_ORDINAL,
    "replication": REPLICATION_ORDINAL,
    "selective_reporting": SELECTIVE_REPORTING_ORDINAL,
    "disease_stage": DISEASE_STAGE_ORDINAL,
    "unmet_need": UNMET_NEED_ORDINAL,
    "severity": SEVERITY_ORDINAL,
}


def ordinal(map_name: str, value: str | None) -> float | None:
    """Look up an ordinal encoding, returning ``None`` for unknown/absent.

    Crucially, an unrecognised or ``unknown``/``not_assessed`` value maps to
    ``None`` (missing), never to 0. Encoding ignorance as the lowest level is
    the single most common way an evidence-scoring system silently penalises
    programs for incomplete documentation.
    """
    if value is None:
        return None
    return ORDINAL_MAPS[map_name].get(str(value))
