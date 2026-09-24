"""The declared feature set for :data:`FEATURE_SCHEMA_VERSION`.

Reading order: features are grouped by :class:`FeatureDomain`, and the domains
are the rows of the translational scorecard. Each entry states how it is
derived from evidence, because the derivation *is* the scientific claim.
"""

from __future__ import annotations

from trs.features.spec import (
    MODALITY_GROUP_LEVELS,
    TARGET_CLASS_GROUP_LEVELS,
    Direction,
)
from trs.features.spec import (
    FeatureDomain as D,
)
from trs.features.spec import (
    FeatureKind as K,
)
from trs.features.spec import (
    FeatureSpec as F,
)
from trs.features.spec import (
    MissingPolicy as M,
)
from trs.features.spec import (
    Transform as T,
)

FEATURES: tuple[F, ...] = (
    # ─────────────────────────────────────────────────────────────────────
    # Program context
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="modality_group",
        domain=D.PROGRAM_CONTEXT,
        kind=K.NOMINAL,
        label="Modality class",
        description=(
            "Therapeutic modality, pooled into classes that share translational "
            "behaviour. Small molecule is the reference level."
        ),
        derivation="Program.modality mapped through MODALITY_GROUPS.",
        levels=MODALITY_GROUP_LEVELS,
        missing_policy=M.EXPLICIT_LEVEL,
        expected_direction=Direction.UNCERTAIN,
        completeness_weight=2.0,
    ),
    F(
        name="target_class_group",
        domain=D.PROGRAM_CONTEXT,
        kind=K.NOMINAL,
        label="Target class",
        description="Molecular class of the primary target. Kinase is the reference level.",
        derivation="Target.target_class mapped through TARGET_CLASS_GROUPS.",
        levels=TARGET_CLASS_GROUP_LEVELS,
        missing_policy=M.EXPLICIT_LEVEL,
        expected_direction=Direction.UNCERTAIN,
        completeness_weight=1.5,
    ),
    F(
        name="is_first_in_class",
        domain=D.PROGRAM_CONTEXT,
        kind=K.BINARY,
        label="First-in-class",
        description=(
            "No drug acting on this target had been approved anywhere as of the "
            "cutoff date. First-in-class programs carry mechanism risk that "
            "followers do not."
        ),
        derivation=(
            "TargetAssessment.prior_approved_drugs_on_target == 0 at the latest "
            "assessment with available_at <= cutoff."
        ),
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        completeness_weight=2.0,
    ),
    F(
        name="is_combination_regimen",
        domain=D.PROGRAM_CONTEXT,
        kind=K.BINARY,
        label="Combination regimen",
        description="The program is developed as a combination rather than monotherapy.",
        derivation="ProgramIndication.combination_partner is populated.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.UNCERTAIN,
        completeness_weight=0.5,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Animal-model validity
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="animal_n_invivo_studies",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.COUNT,
        label="In vivo efficacy studies",
        description="Number of distinct in vivo animal efficacy experiments on record.",
        derivation=(
            "Count of EvidenceItem where evidence_class = animal_in_vivo and "
            "endpoint_category is an efficacy endpoint, available_at <= cutoff."
        ),
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=2.0,
    ),
    F(
        name="animal_positive_fraction",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.NUMERIC,
        label="Fraction of animal studies supporting",
        description=(
            "Of the in vivo efficacy experiments with an interpretable "
            "direction, the proportion that support the hypothesis. A value "
            "below 1 means the record contains discordant experiments."
        ),
        derivation="n(direction=supports) / n(direction in {supports, refutes}).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="animal_n_refuting_studies",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.COUNT,
        label="Contradictory animal studies",
        description="In vivo experiments whose result refutes the efficacy hypothesis.",
        derivation="Count of animal_in_vivo EvidenceItem with direction = refutes.",
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0,
        completeness_weight=1.5,
        scenario_adjustable=True,
    ),
    F(
        name="animal_best_construct_validity",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.ORDINAL,
        label="Best model construct validity",
        description=(
            "Highest curator-assessed construct validity among the supporting "
            "in vivo models: does the model reproduce the mechanism of the "
            "human disease, or merely the phenotype?"
        ),
        derivation="max(QUALITATIVE_STRENGTH[construct_validity]) over supporting in vivo items.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=3,
        completeness_weight=3.0,
        scenario_adjustable=True,
    ),
    F(
        name="animal_mean_disease_similarity",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.ORDINAL,
        label="Mean disease similarity",
        description="Average similarity of the animal disease state to the human target population.",
        derivation="mean(QUALITATIVE_STRENGTH[disease_similarity]) over supporting in vivo items.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=3,
        completeness_weight=2.0,
    ),
    F(
        name="animal_uses_high_fidelity_model",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.BINARY,
        label="PDX / GEMM / orthotopic model used",
        description=(
            "At least one supporting experiment used a patient-derived "
            "xenograft, a genetically engineered autochthonous model, or an "
            "orthotopic implantation — model constructions that preserve more "
            "of the human tumour or its microenvironment than a subcutaneous "
            "cell-line xenograft."
        ),
        derivation=(
            "Any supporting in vivo item with model_type in {patient_derived_xenograft, "
            "genetically_engineered, orthotopic} or an orthotopic/GEMM/PDX model_subtype."
        ),
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=2.0,
        scenario_adjustable=True,
    ),
    F(
        name="animal_immunocompetent_fraction",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.NUMERIC,
        label="Immunocompetent host fraction",
        description=(
            "Proportion of supporting in vivo experiments performed in a host "
            "with an intact or humanised immune system. Material whenever the "
            "mechanism is immune-dependent: an immunodeficient host cannot "
            "test an immune-mediated hypothesis."
        ),
        derivation=(
            "n(immune_competence in {fully_immunocompetent, humanised_immune_system, "
            "partially_reconstituted}) / n(items with a recorded immune status)."
        ),
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.5,
    ),
    F(
        name="animal_median_effect_z",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.NUMERIC,
        label="Median standardised effect",
        description=(
            "Median effect size across supporting in vivo experiments, mapped "
            "onto a common standardised scale so that percent tumour-growth "
            "inhibition, hazard ratios and fold changes are comparable."
        ),
        derivation=(
            "median over supporting in vivo items of "
            "trs.features.effects.standardise_effect(metric, value)."
        ),
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=-5.0, valid_max=8.0,
        completeness_weight=2.5,
    ),
    F(
        name="animal_max_effect_z",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.NUMERIC,
        label="Largest standardised effect",
        description=(
            "The single best reported effect. Held alongside the median because "
            "a large gap between the two is itself a signal about selective "
            "reporting or model cherry-picking."
        ),
        derivation="max over supporting in vivo items of the standardised effect.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=-5.0, valid_max=8.0,
        completeness_weight=1.5,
    ),
    F(
        name="animal_median_n_per_arm",
        domain=D.ANIMAL_MODEL_VALIDITY,
        kind=K.COUNT,
        label="Median group size",
        description="Median animals per treatment arm; a direct proxy for statistical precision.",
        derivation="median(n_per_arm) over supporting in vivo items.",
        transform=T.LOG1P,
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=1.5,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Cross-species concordance
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="species_count_supporting",
        domain=D.CROSS_SPECIES_CONCORDANCE,
        kind=K.COUNT,
        label="Species with supporting evidence",
        description="Number of distinct species in which a supporting in vivo effect was observed.",
        derivation="Distinct EvidenceItem.species among supporting animal_in_vivo items.",
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=2.0,
        scenario_adjustable=True,
    ),
    F(
        name="species_concordance_fraction",
        domain=D.CROSS_SPECIES_CONCORDANCE,
        kind=K.NUMERIC,
        label="Species concordance",
        description=(
            "Of the species in which the hypothesis was tested in vivo, the "
            "proportion in which it was supported. Distinguishes 'tested in "
            "three species, worked in three' from 'tested in three, worked in one'."
        ),
        derivation="n(species with a supporting item) / n(species tested in vivo).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="has_nonrodent_efficacy",
        domain=D.CROSS_SPECIES_CONCORDANCE,
        kind=K.BINARY,
        label="Non-rodent efficacy evidence",
        description="Supporting in vivo efficacy in a dog, pig, rabbit or non-human primate.",
        derivation="Any supporting in vivo item with species not in {mouse, rat, zebrafish}.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=1.0,
        scenario_adjustable=True,
    ),
    F(
        name="n_distinct_model_types",
        domain=D.CROSS_SPECIES_CONCORDANCE,
        kind=K.COUNT,
        label="Distinct model constructions",
        description=(
            "Number of structurally different disease models used. Replication "
            "across model constructions is stronger evidence than replication "
            "within one construction."
        ),
        derivation="Distinct (model_type, model_subtype) among supporting in vivo items.",
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=1.5,
        scenario_adjustable=True,
    ),
    F(
        name="has_contradictory_species",
        domain=D.CROSS_SPECIES_CONCORDANCE,
        kind=K.BINARY,
        label="Species-discordant result",
        description="At least one species supports the hypothesis while another refutes it.",
        derivation="Intersection of supporting and refuting species sets is empty and both are non-empty.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.INCREASES_RISK,
        completeness_weight=1.5,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # PK/PD translation
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="exposure_margin_log2",
        domain=D.PKPD_TRANSLATION,
        kind=K.NUMERIC,
        label="Exposure margin (log2)",
        description=(
            "log2 of the ratio of achievable human exposure to the exposure "
            "associated with preclinical efficacy. Zero means the efficacious "
            "exposure is exactly reachable; negative means it is not."
        ),
        derivation="log2(PkPdBridge.exposure_margin_ratio) at the latest bridge <= cutoff.",
        transform=T.IDENTITY,
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=-8.0, valid_max=8.0,
        completeness_weight=3.0,
        scenario_adjustable=True,
        unit="log2 ratio",
    ),
    F(
        name="therapeutic_index_log2",
        domain=D.PKPD_TRANSLATION,
        kind=K.NUMERIC,
        label="Therapeutic index (log2)",
        description=(
            "log2 of the ratio between the exposure at the no-observed-adverse-"
            "effect level and the efficacious exposure. A narrow window forces "
            "dosing below the efficacious exposure."
        ),
        derivation="log2(PkPdBridge.therapeutic_index_ratio) at the latest bridge <= cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=-6.0, valid_max=10.0,
        completeness_weight=2.5,
        scenario_adjustable=True,
        unit="log2 ratio",
    ),
    F(
        name="exposure_bridge_quality_ord",
        domain=D.PKPD_TRANSLATION,
        kind=K.ORDINAL,
        label="Exposure bridge quality",
        description=(
            "How the human exposure projection was arrived at, from 'assumed' "
            "through simple allometry to a demonstrated human exposure."
        ),
        derivation="EXPOSURE_BRIDGE_ORDINAL[PkPdBridge.exposure_bridge_quality].",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=4,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="target_engagement_ord",
        domain=D.PKPD_TRANSLATION,
        kind=K.ORDINAL,
        label="Target engagement evidence",
        description=(
            "Strongest demonstration that the drug engages its target, scaled "
            "so that a human demonstration counts substantially above any "
            "animal demonstration."
        ),
        derivation="TARGET_ENGAGEMENT_ORDINAL[PkPdBridge.target_engagement_evidence].",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=6,
        completeness_weight=3.0,
        scenario_adjustable=True,
    ),
    F(
        name="exposure_response_established",
        domain=D.PKPD_TRANSLATION,
        kind=K.BINARY,
        label="Exposure–response established",
        description="A monotone relationship between exposure and effect was demonstrated preclinically.",
        derivation="PkPdBridge.exposure_response_established.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=2.0,
        scenario_adjustable=True,
    ),
    F(
        name="pk_scaling_species_count",
        domain=D.PKPD_TRANSLATION,
        kind=K.COUNT,
        label="Species in PK scaling",
        description="Number of species contributing to the human PK projection.",
        derivation="PkPdBridge.pk_scaling_species_count.",
        transform=T.LOG1P,
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=1.5,
        scenario_adjustable=True,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Human biological support
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="human_evidence_count",
        domain=D.HUMAN_BIOLOGICAL_SUPPORT,
        kind=K.COUNT,
        label="Human-derived evidence items",
        description=(
            "Count of evidence generated in human material or human "
            "populations: patient tissue, primary human cells, organoids, "
            "human ex vivo systems, human genetics, human observational data."
        ),
        derivation=(
            "Count of EvidenceItem with evidence_class in the human-derived set "
            "and available_at <= cutoff."
        ),
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="human_evidence_supporting_fraction",
        domain=D.HUMAN_BIOLOGICAL_SUPPORT,
        kind=K.NUMERIC,
        label="Human evidence concordance",
        description="Proportion of directional human-derived evidence that supports the hypothesis.",
        derivation="n(human supporting) / n(human supporting + refuting).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="has_patient_tissue_evidence",
        domain=D.HUMAN_BIOLOGICAL_SUPPORT,
        kind=K.BINARY,
        label="Patient tissue evidence",
        description="Supporting evidence from human tissue or human ex vivo material.",
        derivation="Any supporting item with evidence_class in {human_tissue_molecular, human_ex_vivo}.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=2.0,
        scenario_adjustable=True,
    ),
    F(
        name="has_organoid_evidence",
        domain=D.HUMAN_BIOLOGICAL_SUPPORT,
        kind=K.BINARY,
        label="Human organoid evidence",
        description="Supporting evidence from a patient-derived organoid or comparable 3D human system.",
        derivation="Any supporting item with evidence_class = organoid and species = human.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=1.5,
        scenario_adjustable=True,
    ),
    F(
        name="has_prior_human_clinical_on_pathway",
        domain=D.HUMAN_BIOLOGICAL_SUPPORT,
        kind=K.BINARY,
        label="Prior human clinical readout on pathway",
        description=(
            "A different agent modulating this target or pathway has already "
            "produced an interpretable efficacy readout in humans. Whether "
            "that readout was positive or negative is captured separately by "
            "the historical-performance features."
        ),
        derivation=(
            "TargetAssessment.prior_clinical_programs_on_target > 0 at the "
            "latest assessment <= cutoff."
        ),
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=2.0,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Target validation
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="genetic_evidence_ord",
        domain=D.TARGET_VALIDATION,
        kind=K.ORDINAL,
        label="Human genetic support",
        description=(
            "Strongest tier of human genetic evidence linking the target to the "
            "disease, from no signal through GWAS loci to Mendelian causality."
        ),
        derivation="GENETIC_EVIDENCE_ORDINAL[TargetAssessment.genetic_evidence_tier] at cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=5,
        completeness_weight=3.0,
        scenario_adjustable=True,
    ),
    F(
        name="target_association_score",
        domain=D.TARGET_VALIDATION,
        kind=K.NUMERIC,
        label="Target–disease association score",
        description=(
            "Aggregated target–disease association on a 0–1 scale, as published "
            "by the source knowledge base at or before the cutoff."
        ),
        derivation="TargetAssessment.association_score at the latest assessment <= cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.0,
    ),
    F(
        name="causal_support_ord",
        domain=D.TARGET_VALIDATION,
        kind=K.ORDINAL,
        label="Causal support",
        description=(
            "Whether the target–disease link rests on interventional or causal-"
            "inference evidence, or only on correlation."
        ),
        derivation="CAUSAL_SUPPORT_ORDINAL[TargetAssessment.causal_support] at cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=4,
        completeness_weight=2.5,
    ),
    F(
        name="pathway_redundancy_ord",
        domain=D.TARGET_VALIDATION,
        kind=K.ORDINAL,
        label="Pathway redundancy",
        description=(
            "How readily compensatory signalling can bypass inhibition of this "
            "target. Oriented so larger values mean more redundancy."
        ),
        derivation="PATHWAY_REDUNDANCY_ORDINAL[TargetAssessment.pathway_redundancy] at cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0, valid_max=3,
        completeness_weight=2.5,
    ),
    F(
        name="human_tissue_expression_ord",
        domain=D.TARGET_VALIDATION,
        kind=K.ORDINAL,
        label="Human tissue expression evidence",
        description="Strength of evidence that the target is expressed in the relevant human tissue.",
        derivation="QUALITATIVE_STRENGTH[TargetAssessment.human_tissue_expression_evidence] at cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=3,
        completeness_weight=1.5,
    ),
    F(
        name="target_ortholog_identity_mouse",
        domain=D.TARGET_VALIDATION,
        kind=K.NUMERIC,
        label="Mouse ortholog identity",
        description=(
            "Sequence identity between the human target and its mouse ortholog. "
            "Low identity undermines the relevance of mouse pharmacology."
        ),
        derivation="Target.ortholog_identity_mouse.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.0,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Disease-model relevance
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="biomarker_selected_population",
        domain=D.DISEASE_MODEL_RELEVANCE,
        kind=K.BINARY,
        label="Biomarker-selected population",
        description=(
            "The intended trial enrols only patients carrying a required "
            "genomic alteration or protein expression level."
        ),
        derivation=(
            "ProgramIndication.biomarker_selection in "
            "{required_genomic_alteration, required_protein_expression}."
        ),
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="disease_stage_ord",
        domain=D.DISEASE_MODEL_RELEVANCE,
        kind=K.ORDINAL,
        label="Disease stage",
        description=(
            "Stage of the intended population, from prevention through "
            "relapsed/refractory. Direction is genuinely uncertain and is "
            "estimated rather than assumed."
        ),
        derivation="DISEASE_STAGE_ORDINAL[ProgramIndication.disease_stage].",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.UNCERTAIN,
        valid_min=0, valid_max=4,
        completeness_weight=1.5,
    ),
    F(
        name="unmet_need_ord",
        domain=D.DISEASE_MODEL_RELEVANCE,
        kind=K.ORDINAL,
        label="Unmet need",
        description=(
            "Strength of the standard-of-care gap. Higher unmet need generally "
            "means a weaker comparator and a lower efficacy bar."
        ),
        derivation="UNMET_NEED_ORDINAL[ProgramIndication.unmet_need].",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=3,
        completeness_weight=1.5,
    ),
    F(
        name="model_indication_match",
        domain=D.DISEASE_MODEL_RELEVANCE,
        kind=K.NUMERIC,
        label="Model–indication match",
        description=(
            "Proportion of supporting in vivo experiments performed in a model "
            "of the same indication the program is being developed in, rather "
            "than a convenient surrogate disease."
        ),
        derivation="n(supporting in vivo items with indication_id == track indication) / n(supporting in vivo items).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.0,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Reproducibility and study quality
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="replication_scope_ord",
        domain=D.REPRODUCIBILITY,
        kind=K.ORDINAL,
        label="Replication scope",
        description=(
            "Broadest replication behind the supporting evidence, from a single "
            "study to multiple independent laboratories."
        ),
        derivation="max(REPLICATION_ORDINAL[replication_scope]) over supporting items.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=4,
        completeness_weight=3.0,
        scenario_adjustable=True,
    ),
    F(
        name="n_independent_labs",
        domain=D.REPRODUCIBILITY,
        kind=K.COUNT,
        label="Independent laboratories",
        description=(
            "Distinct laboratories contributing supporting evidence. Counted "
            "from a stable lab identifier, so ten papers from one group count "
            "once."
        ),
        derivation="Distinct EvidenceItem.lab_identifier among supporting items.",
        transform=T.LOG1P,
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0,
        completeness_weight=3.0,
        scenario_adjustable=True,
    ),
    F(
        name="single_lab_dependence",
        domain=D.REPRODUCIBILITY,
        kind=K.BINARY,
        label="Single-laboratory dependence",
        description=(
            "All supporting evidence originates from one laboratory. The "
            "canonical reproducibility failure mode."
        ),
        derivation="n_independent_labs <= 1 while at least two supporting items exist.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        completeness_weight=2.0,
        scenario_adjustable=True,
    ),
    F(
        name="randomised_fraction",
        domain=D.REPRODUCIBILITY,
        kind=K.NUMERIC,
        label="Randomised fraction",
        description="Proportion of supporting in vivo studies that report randomised allocation.",
        derivation="n(randomization in {randomised, randomised_concealed}) / n(in vivo items).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.0,
    ),
    F(
        name="blinded_fraction",
        domain=D.REPRODUCIBILITY,
        kind=K.NUMERIC,
        label="Blinded fraction",
        description="Proportion of supporting in vivo studies with at least blinded outcome assessment.",
        derivation="n(blinding in {fully_blinded, outcome_assessor_blinded, partially_blinded}) / n(in vivo items).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=2.0,
    ),
    F(
        name="prespecified_fraction",
        domain=D.REPRODUCIBILITY,
        kind=K.NUMERIC,
        label="Prespecified / preregistered fraction",
        description="Proportion of supporting studies run under a prospective protocol or preregistration.",
        derivation="n(prospective_protocol or preregistered) / n(supporting items).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.5,
    ),
    F(
        name="selective_reporting_ord",
        domain=D.REPRODUCIBILITY,
        kind=K.ORDINAL,
        label="Selective reporting risk",
        description="Worst assessed selective-reporting risk across supporting studies.",
        derivation="max(SELECTIVE_REPORTING_ORDINAL[selective_reporting_risk]) over supporting items.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0, valid_max=2,
        completeness_weight=2.0,
    ),
    F(
        name="sponsor_affiliated_fraction",
        domain=D.REPRODUCIBILITY,
        kind=K.NUMERIC,
        label="Sponsor-affiliated evidence fraction",
        description=(
            "Proportion of supporting evidence authored by the sponsor or a "
            "sponsor-affiliated group. Not a judgement of integrity; a measure "
            "of how much of the case is externally corroborated."
        ),
        derivation="n(conflict_of_interest = sponsor_affiliated) / n(items with a COI statement).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.5,
    ),
    F(
        name="peer_reviewed_fraction",
        domain=D.REPRODUCIBILITY,
        kind=K.NUMERIC,
        label="Peer-reviewed fraction",
        description="Proportion of supporting evidence that appeared in peer-reviewed form.",
        derivation="n(publication_status = peer_reviewed) / n(supporting items).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.0,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Safety translation
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="n_safety_findings",
        domain=D.SAFETY_TRANSLATION,
        kind=K.COUNT,
        label="Preclinical safety findings",
        description="Distinct adverse findings recorded in preclinical toxicology.",
        derivation="Count of SafetyFinding with available_at <= cutoff.",
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0,
        completeness_weight=1.5,
    ),
    F(
        name="max_safety_severity_ord",
        domain=D.SAFETY_TRANSLATION,
        kind=K.ORDINAL,
        label="Worst safety severity",
        description="Severity of the most serious preclinical finding.",
        derivation="max(SEVERITY_ORDINAL[severity]) over SafetyFinding rows <= cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0, valid_max=5,
        completeness_weight=2.0,
    ),
    F(
        name="has_irreversible_on_target_finding",
        domain=D.SAFETY_TRANSLATION,
        kind=K.BINARY,
        label="Irreversible on-target toxicity",
        description=(
            "A severe, irreversible finding attributed to the intended "
            "pharmacology. On-target toxicity cannot be engineered away and "
            "directly caps the achievable exposure."
        ),
        derivation="Any SafetyFinding with on_target, reversible is false, and severity >= severe.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.INCREASES_RISK,
        completeness_weight=2.0,
    ),
    F(
        name="safety_monitorable_fraction",
        domain=D.SAFETY_TRANSLATION,
        kind=K.NUMERIC,
        label="Monitorable findings fraction",
        description="Proportion of findings judged monitorable in a human trial.",
        derivation="n(monitorable_in_human) / n(findings with an assessment).",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.0,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Biomarker readiness
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="biomarker_readiness_ord",
        domain=D.BIOMARKER_READINESS,
        kind=K.ORDINAL,
        label="Best biomarker readiness",
        description=(
            "Readiness of the most mature pharmacodynamic or target-engagement "
            "biomarker available at the cutoff."
        ),
        derivation="max(BIOMARKER_READINESS_ORDINAL[readiness]) over PD/TE biomarkers <= cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=5,
        completeness_weight=2.5,
        scenario_adjustable=True,
    ),
    F(
        name="has_predictive_selection_biomarker",
        domain=D.BIOMARKER_READINESS,
        kind=K.BINARY,
        label="Predictive selection biomarker",
        description="A biomarker exists that can select likely responders for the first efficacy study.",
        derivation="Any ProgramBiomarker with biomarker_type = predictive_selection and readiness >= fit_for_purpose_human.",
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.DECREASES_RISK,
        completeness_weight=2.0,
        scenario_adjustable=True,
    ),
    F(
        name="biomarker_species_concordance_ord",
        domain=D.BIOMARKER_READINESS,
        kind=K.ORDINAL,
        label="Biomarker cross-species concordance",
        description="Whether the biomarker behaves equivalently in the preclinical species and in humans.",
        derivation="max(QUALITATIVE_STRENGTH[cross_species_concordance]) over biomarkers <= cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.DECREASES_RISK,
        valid_min=0, valid_max=3,
        completeness_weight=1.5,
        scenario_adjustable=True,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Historical class performance  (computed strictly as-of; see
    # trs.features.historical for the leakage-control logic)
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="target_prior_failure_rate",
        domain=D.HISTORICAL_CLASS_PERFORMANCE,
        kind=K.NUMERIC,
        label="Prior failure rate on this target",
        description=(
            "Empirical-Bayes shrunk Phase II failure rate among earlier "
            "programs against the same target whose outcome was already known "
            "at the cutoff. The focal program is always excluded."
        ),
        derivation=(
            "Beta-binomial posterior mean over prior tracks with the same "
            "target_id and outcome_date <= cutoff, shrunk toward the cohort "
            "base rate; see trs.features.historical.shrunk_rate."
        ),
        missing_policy=M.IMPUTE_ONLY,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.5,
    ),
    F(
        name="target_prior_n_programs",
        domain=D.HISTORICAL_CLASS_PERFORMANCE,
        kind=K.COUNT,
        label="Prior programs on this target",
        description="Number of earlier resolved programs on the same target, as of the cutoff.",
        derivation="Count of prior tracks with the same target_id and a known outcome <= cutoff.",
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.UNCERTAIN,
        valid_min=0,
        completeness_weight=1.0,
    ),
    F(
        name="mechanism_class_prior_failure_rate",
        domain=D.HISTORICAL_CLASS_PERFORMANCE,
        kind=K.NUMERIC,
        label="Prior failure rate for this mechanism class",
        description=(
            "Shrunk Phase II failure rate among earlier programs sharing target "
            "class and modality, resolved before the cutoff."
        ),
        derivation="As target_prior_failure_rate, grouped on (target_class_group, modality_group).",
        missing_policy=M.IMPUTE_ONLY,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.5,
    ),
    F(
        name="indication_prior_failure_rate",
        domain=D.HISTORICAL_CLASS_PERFORMANCE,
        kind=K.NUMERIC,
        label="Prior failure rate in this indication",
        description="Shrunk Phase II failure rate among earlier programs in the same indication.",
        derivation="As target_prior_failure_rate, grouped on indication_id.",
        missing_policy=M.IMPUTE_ONLY,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0.0, valid_max=1.0,
        completeness_weight=1.5,
    ),

    # ─────────────────────────────────────────────────────────────────────
    # Evidence volume and recency
    # ─────────────────────────────────────────────────────────────────────
    F(
        name="evidence_total_count",
        domain=D.EVIDENCE_VOLUME,
        kind=K.COUNT,
        label="Total evidence items",
        description=(
            "Size of the evidence base at the cutoff. Confounded with "
            "disclosure behaviour — a well-published academic program and a "
            "quiet industrial one are not equally documented for reasons "
            "unrelated to their biology. Interpreted with that caveat."
        ),
        derivation="Count of all EvidenceItem rows for the track with available_at <= cutoff.",
        transform=T.LOG1P,
        missing_policy=M.ZERO_IS_ABSENCE,
        expected_direction=Direction.UNCERTAIN,
        valid_min=0,
        completeness_weight=0.5,
    ),
    F(
        name="evidence_span_years",
        domain=D.EVIDENCE_VOLUME,
        kind=K.NUMERIC,
        label="Evidence span (years)",
        description="Years between the earliest and latest evidence item available at the cutoff.",
        derivation="(max(available_at) - min(available_at)) / 365.25 over evidence <= cutoff.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.UNCERTAIN,
        valid_min=0.0, valid_max=40.0,
        completeness_weight=0.5,
        unit="years",
    ),
    F(
        name="evidence_recency_years",
        domain=D.EVIDENCE_VOLUME,
        kind=K.NUMERIC,
        label="Evidence recency (years)",
        description="Years between the most recent available evidence item and the cutoff date.",
        derivation="(cutoff - max(available_at)) / 365.25.",
        missing_policy=M.IMPUTE_WITH_INDICATOR,
        expected_direction=Direction.INCREASES_RISK,
        valid_min=0.0, valid_max=40.0,
        completeness_weight=0.5,
        unit="years",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry access
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_BY_NAME: dict[str, F] = {f.name: f for f in FEATURES}

_duplicates = len(FEATURES) - len(FEATURES_BY_NAME)
if _duplicates:  # pragma: no cover - guards a developer error at import time
    raise RuntimeError(f"Duplicate feature names in registry ({_duplicates} collisions)")


def features_in_domain(domain: D) -> tuple[F, ...]:
    return tuple(f for f in FEATURES if f.domain is domain)


def scenario_adjustable_features() -> tuple[F, ...]:
    return tuple(f for f in FEATURES if f.scenario_adjustable)


def feature_names() -> tuple[str, ...]:
    """Raw (pre-encoding) feature names, in declaration order."""
    return tuple(f.name for f in FEATURES)
