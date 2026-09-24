"""API response models.

Two conventions run through every schema here.

**Uncertainty travels with the number.** No endpoint returns a bare
probability. ``probability`` is always accompanied by its interval, the
interval's level, and the decomposition of where the width comes from.

**Provenance travels with the fact.** Evidence and score payloads carry their
source, extraction method and availability date, so a consumer of the API has
the same traceability the interface does.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ─────────────────────────────────────────────────────────────────────────────
# Programs
# ─────────────────────────────────────────────────────────────────────────────

class TargetSummary(ApiModel):
    id: str
    symbol: str
    name: str | None = None
    target_class: str
    ortholog_identity_mouse: float | None = None


class IndicationSummary(ApiModel):
    id: str
    name: str
    therapeutic_area: str
    molecular_subtype: str | None = None
    ontology_id: str | None = None


class SponsorSummary(ApiModel):
    id: str
    name: str
    sponsor_type: str


class ProgramAliasOut(ApiModel):
    alias: str
    alias_type: str
    match_method: str
    match_confidence: float
    available_at: dt.date | None = None
    is_primary: bool


class TrackSummary(ApiModel):
    """A development track: program × indication. The analysis unit's subject."""

    id: str
    program_id: str
    program_name: str
    program_code: str | None = None
    target: TargetSummary | None = None
    indication: IndicationSummary
    sponsor: SponsorSummary | None = None
    modality: str
    mechanism_of_action: str | None = None
    disease_stage: str
    line_of_therapy: str
    biomarker_selection: str
    unmet_need: str
    combination_partner: str | None = None
    ind_or_cta_on: dt.date | None = None
    first_in_human_on: dt.date | None = None
    highest_phase_reached: str
    outcome_label: str
    outcome_date: dt.date | None = None
    discontinuation_reason: str
    outcome_confidence: str
    is_synthetic: bool
    n_evidence_items: int = 0


class ProgramDetail(TrackSummary):
    aliases: list[ProgramAliasOut] = Field(default_factory=list)
    first_disclosure_on: dt.date | None = None
    standard_of_care_note: str | None = None
    all_tracks: list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────────────────────

class SourceOut(ApiModel):
    id: str
    source_type: str
    title: str | None = None
    container_title: str | None = None
    pmid: str | None = None
    doi: str | None = None
    nct_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    published_on: dt.date | None = None
    retrieved_at: dt.datetime | None = None
    retrieved_by_connector: str | None = None
    publication_status: str
    is_retracted: bool = False
    is_synthetic: bool = False


class EvidenceOut(ApiModel):
    id: str
    evidence_class: str
    direction: str
    title: str | None = None
    summary: str | None = None
    species: str
    biological_system: str
    model_type: str
    model_subtype: str
    immune_competence: str
    construct_validity: str
    disease_similarity: str
    endpoint_category: str
    endpoint_description: str | None = None
    effect_metric: str | None = None
    effect_value: float | None = None
    effect_ci_low: float | None = None
    effect_ci_high: float | None = None
    #: Effect mapped onto the common standardised scale, where the metric permits.
    standardised_effect: float | None = None
    n_per_arm: int | None = None
    study_duration_days: int | None = None
    dose_value: float | None = None
    dose_unit: str | None = None
    randomization: str
    blinding: str
    replication_scope: str
    independent_replications: int | None = None
    lab_identifier: str | None = None
    selective_reporting_risk: str
    conflict_of_interest: str
    publication_status: str
    available_at: dt.date
    observed_on: dt.date | None = None
    provenance_method: str
    extraction_confidence: str
    source_snippet: str | None = None
    source: SourceOut | None = None
    is_synthetic: bool


class EvidenceMatrixCell(ApiModel):
    evidence_class: str
    system: str
    endpoint: str
    n_total: int
    n_supporting: int
    n_refuting: int
    evidence_ids: list[str]


class EvidenceMatrix(ApiModel):
    """Evidence type × system × endpoint. Gaps are the point of this view."""

    rows: list[str]
    columns: list[str]
    cells: list[EvidenceMatrixCell]
    n_evidence: int
    empty_cells: int
    coverage: float
    note: str


# ─────────────────────────────────────────────────────────────────────────────
# Scores
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyOut(ApiModel):
    probability: float
    interval_low: float
    interval_high: float
    interval_level: float
    interval_width: float
    components_sd_logit: dict[str, float]
    variance_share: dict[str, float]
    dominant_source: str
    total_sd_logit: float
    n_imputations: int
    n_param_samples: int
    caveats: list[str]
    method: str


class DriverOut(ApiModel):
    feature: str
    label: str
    domain: str
    domain_label: str
    contribution: float
    value: Any = None
    cohort_mean: float | None = None
    z_score: float | None = None
    direction: str
    is_missing: bool
    missing_reason: str | None = None
    evidence_ids: list[str]
    probability_delta: float


class ExplanationOut(ApiModel):
    baseline_logit: float
    baseline_probability: float
    total_logit: float
    probability: float
    increasing_risk: list[DriverOut]
    decreasing_risk: list[DriverOut]
    by_domain: list[dict[str, Any]]
    other_factors_contribution: float
    reconstruction_error: float
    method: str
    statement: str


class ScoreOut(ApiModel):
    track_id: str
    program_id: str
    program_name: str
    as_of_date: dt.date
    endpoint: str
    estimand: str
    model_version: str
    model_family: str
    probability: float
    interval_low: float
    interval_high: float
    interval_level: float
    uncertainty: UncertaintyOut
    explanation: ExplanationOut | None = None
    explanation_unavailable_reason: str | None = None
    completeness: dict[str, Any]
    evidence_confidence: str
    confidence_reasons: list[str]
    feature_snapshot_id: str | None = None
    n_evidence_items: int
    is_demonstration: bool
    demonstration_note: str | None = None
    model_usefulness: dict[str, Any]
    specification_sensitivity: dict[str, Any]
    scenario: dict[str, Any] | None = None
    score_record_id: str | None = None
    computed_at: dt.datetime


class ScoreRequest(ApiModel):
    as_of: dt.date | None = Field(
        default=None,
        description=(
            "Decision cutoff. Only evidence available on or before this date is "
            "used. Defaults to the program's IND/CTA date."
        ),
    )
    model_version: str | None = None
    interval_level: float = Field(default=0.95, ge=0.5, le=0.999)
    persist: bool = True


class ScenarioRequest(ApiModel):
    lever_keys: list[str] = Field(
        default_factory=list,
        description="Named scenario levers to apply. See GET /scenarios/levers.",
    )
    overrides: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Raw feature overrides. Only features declared scenario-adjustable "
            "may be set."
        ),
    )
    as_of: dt.date | None = None
    name: str | None = None
    persist_scenario: bool = False


class ScenarioComparison(ApiModel):
    current: ScoreOut
    scenario: ScoreOut
    delta_probability: float
    delta_interval_width: float
    changes: list[dict[str, Any]]
    disclaimer: str


# ─────────────────────────────────────────────────────────────────────────────
# Comparators
# ─────────────────────────────────────────────────────────────────────────────

class ComparatorOut(ApiModel):
    track_id: str
    program_id: str
    program_name: str
    sponsor: str | None = None
    target_symbol: str | None = None
    indication: str
    modality: str
    similarity: float
    matched_facets: list[dict[str, Any]]
    rationale: str
    outcome_label: str
    outcome_display: str
    outcome_date: dt.date | None = None
    highest_phase_reached: str
    discontinuation_reason: str
    outcome_confidence: str
    ind_or_cta_on: dt.date | None = None
    is_synthetic: bool


class ComparatorResponse(ApiModel):
    track_id: str
    as_of_date: dt.date
    comparators: list[ComparatorOut]
    summary: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioRow(ApiModel):
    track_id: str
    program_id: str
    program_name: str
    target_symbol: str | None = None
    indication: str
    therapeutic_area: str
    modality: str
    sponsor: str | None = None
    stage: str
    as_of_date: dt.date
    probability: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    evidence_confidence: str | None = None
    completeness: float | None = None
    top_risk_driver: str | None = None
    top_risk_driver_domain: str | None = None
    n_evidence_items: int
    outcome_label: str
    is_synthetic: bool
    last_scored_at: dt.datetime | None = None


class PortfolioResponse(ApiModel):
    project_id: str | None = None
    project_name: str | None = None
    model_version: str | None = None
    endpoint: str
    estimand: str
    rows: list[PortfolioRow]
    total: int
    is_demonstration: bool
    demonstration_note: str | None = None
    model_usefulness: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Model / methods
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersionOut(ApiModel):
    version: str
    family: str
    endpoint: str
    display_name: str | None = None
    description: str | None = None
    feature_schema_version: str
    cohort_version_id: str | None = None
    dataset_hash: str
    trained_at: dt.datetime
    n_train: int
    n_calibration: int
    n_test: int
    is_active: bool
    trained_on_synthetic: bool
    git_commit: str | None = None
    library_versions: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] | None = None


class ValidationReportOut(ApiModel):
    model_version: str
    generated_at: str | None = None
    evaluation_design: dict[str, Any]
    cohort: dict[str, Any]
    primary_metrics: dict[str, Any]
    calibration_selection: dict[str, Any]
    subgroup_performance: dict[str, Any]
    coefficient_sign_audit: dict[str, Any]
    interval_agreement: dict[str, Any]
    domain_ablation: dict[str, Any]
    model_comparison: dict[str, Any]
    target_grouped_sensitivity: dict[str, Any] = Field(default_factory=dict)
    hyperparameter_selection: dict[str, Any] = Field(default_factory=dict)
    usefulness: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str]
    oracle_benchmark: dict[str, Any] | None = None


class CoefficientOut(ApiModel):
    column: str
    feature: str | None = None
    label: str | None = None
    domain: str | None = None
    coefficient: float
    odds_ratio_per_sd: float
    posterior_sd: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    expected_direction: str | None = None
    sign_matches_expectation: bool | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Provenance / meta
# ─────────────────────────────────────────────────────────────────────────────

class ConnectorHealthOut(ApiModel):
    connector: str
    status: str
    checked_at: str
    latency_ms: float | None = None
    detail: str = ""
    endpoint: str | None = None


class FeatureSpecOut(ApiModel):
    name: str
    label: str
    domain: str
    domain_label: str
    kind: str
    description: str
    derivation: str
    expected_direction: str
    missing_policy: str
    scenario_adjustable: bool
    unit: str | None = None
    valid_min: float | None = None
    valid_max: float | None = None
    completeness_weight: float


class HealthOut(ApiModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: str
    active_model: str | None = None
    feature_schema_version: str
    is_demonstration_deployment: bool
    development_auth_fallback: bool
