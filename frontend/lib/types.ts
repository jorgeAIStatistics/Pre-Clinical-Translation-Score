/**
 * Types mirroring the API contract in `trs/api/schemas.py`.
 *
 * Hand-written rather than generated, deliberately: the frontend consumes a
 * subset of each payload, and a hand-written type documents that subset. Where
 * the backend is authoritative the field names match exactly, so a mismatch
 * surfaces at compile time rather than as an undefined at runtime.
 */

export type ConfidenceLevel = "high" | "moderate" | "low" | "very_low" | "not_assessed";

export interface TargetSummary {
  id: string;
  symbol: string;
  name: string | null;
  target_class: string;
  ortholog_identity_mouse: number | null;
}

export interface IndicationSummary {
  id: string;
  name: string;
  therapeutic_area: string;
  molecular_subtype: string | null;
  ontology_id: string | null;
}

export interface SponsorSummary {
  id: string;
  name: string;
  sponsor_type: string;
}

export interface TrackSummary {
  id: string;
  program_id: string;
  program_name: string;
  program_code: string | null;
  target: TargetSummary | null;
  indication: IndicationSummary;
  sponsor: SponsorSummary | null;
  modality: string;
  mechanism_of_action: string | null;
  disease_stage: string;
  line_of_therapy: string;
  biomarker_selection: string;
  unmet_need: string;
  combination_partner: string | null;
  ind_or_cta_on: string | null;
  first_in_human_on: string | null;
  highest_phase_reached: string;
  outcome_label: string;
  outcome_date: string | null;
  discontinuation_reason: string;
  outcome_confidence: string;
  is_synthetic: boolean;
  n_evidence_items: number;
}

export interface ProgramDetail extends TrackSummary {
  aliases: Array<{
    alias: string;
    alias_type: string;
    match_method: string;
    match_confidence: number;
    available_at: string | null;
    is_primary: boolean;
  }>;
  first_disclosure_on: string | null;
  standard_of_care_note: string | null;
  all_tracks: Array<{
    track_id: string;
    indication: string;
    outcome_label: string;
    ind_or_cta_on: string | null;
    is_current: boolean;
  }>;
}

export interface SourceOut {
  id: string;
  source_type: string;
  title: string | null;
  container_title: string | null;
  pmid: string | null;
  doi: string | null;
  nct_id: string | null;
  url: string | null;
  published_on: string | null;
  retrieved_at: string | null;
  retrieved_by_connector: string | null;
  publication_status: string;
  is_retracted: boolean;
  is_synthetic: boolean;
}

export interface EvidenceOut {
  id: string;
  evidence_class: string;
  direction: "supports" | "refutes" | "mixed" | "neutral" | "unknown";
  title: string | null;
  summary: string | null;
  species: string;
  biological_system: string;
  model_type: string;
  model_subtype: string;
  immune_competence: string;
  construct_validity: string;
  disease_similarity: string;
  endpoint_category: string;
  endpoint_description: string | null;
  effect_metric: string | null;
  effect_value: number | null;
  standardised_effect: number | null;
  n_per_arm: number | null;
  study_duration_days: number | null;
  dose_value: number | null;
  dose_unit: string | null;
  randomization: string;
  blinding: string;
  replication_scope: string;
  independent_replications: number | null;
  lab_identifier: string | null;
  selective_reporting_risk: string;
  conflict_of_interest: string;
  publication_status: string;
  available_at: string;
  observed_on: string | null;
  provenance_method: string;
  extraction_confidence: string;
  source_snippet: string | null;
  source: SourceOut | null;
  is_synthetic: boolean;
}

export interface Driver {
  feature: string;
  label: string;
  domain: string;
  domain_label: string;
  contribution: number;
  value: number | string | null;
  cohort_mean: number | null;
  z_score: number | null;
  direction: "increases_risk" | "decreases_risk";
  is_missing: boolean;
  missing_reason: string | null;
  evidence_ids: string[];
  probability_delta: number;
}

export interface Explanation {
  baseline_logit: number;
  baseline_probability: number;
  total_logit: number;
  probability: number;
  increasing_risk: Driver[];
  decreasing_risk: Driver[];
  by_domain: Array<{
    domain: string;
    domain_label: string;
    contribution: number;
    n_drivers: number;
    direction: string;
  }>;
  other_factors_contribution: number;
  reconstruction_error: number;
  method: string;
  statement: string;
}

export interface Uncertainty {
  probability: number;
  interval_low: number;
  interval_high: number;
  interval_level: number;
  interval_width: number;
  components_sd_logit: Record<string, number>;
  variance_share: Record<string, number>;
  dominant_source: string;
  total_sd_logit: number;
  n_imputations: number;
  n_param_samples: number;
  caveats: string[];
  method: string;
}

export interface CompletenessReport {
  overall: number;
  by_domain: Record<string, number>;
  n_present: number;
  n_total: number;
  n_not_applicable: number;
  critical_gaps: string[];
  reason_counts: Record<string, number>;
}

export interface Eligibility {
  permitted: boolean;
  model_status: string;
  refusal_code: string | null;
  reason: string;
  is_demonstration: boolean;
}

export interface PredictionContract {
  contract_version: string;
  functional: string;
  functional_description: string;
  probability: number;
  predictive_mean_probability: number | null;
  functional_gap: number | null;
  gap_is_material: boolean;
  gap_note?: string;
  calibration_scale: string;
  calibration_method: string;
  interval_estimand: string;
  evaluated_functional: string;
}

export interface ModelStatusRecord {
  status?: string;
  may_serve?: boolean;
  criteria_version?: string;
  rationale?: string;
  note?: string;
  criteria?: Record<string, { passed: boolean; observed: unknown; requirement: string }>;
}

export interface ScoreOut {
  track_id: string;
  program_id: string;
  program_name: string;
  as_of_date: string;
  endpoint: string;
  estimand: string;
  model_version: string;
  model_family: string;
  /**
   * `null` when the eligibility policy declined to produce a score. A refusal
   * is the absence of a number, not a zero and not a wide interval — check
   * `eligibility.permitted` before rendering anything numeric.
   */
  probability: number | null;
  interval_low: number | null;
  interval_high: number | null;
  interval_level: number;
  uncertainty: Uncertainty | null;
  eligibility: Eligibility;
  prediction_contract: PredictionContract | null;
  model_status: ModelStatusRecord;
  explanation: Explanation | null;
  explanation_unavailable_reason: string | null;
  completeness: CompletenessReport;
  evidence_confidence: ConfidenceLevel;
  confidence_reasons: string[];
  feature_snapshot_id: string | null;
  n_evidence_items: number;
  is_demonstration: boolean;
  demonstration_note: string | null;
  model_usefulness: {
    verdict?: string;
    headline?: string;
    why_this_matters?: string;
    scaled_brier?: { value: number | null; ci_low: number | null; ci_high: number | null };
  };
  specification_sensitivity: {
    available: boolean;
    alternatives?: Record<string, number>;
    max_absolute_difference: number | null;
    interpretation?: string;
  };
  scenario: {
    lever_keys: string[];
    overrides: Record<string, number>;
    changes: Array<{
      feature: string;
      label: string;
      domain_label: string;
      from_value: number | null;
      to_value: number;
      was_missing: boolean;
    }>;
    disclaimer: string;
  } | null;
  score_record_id: string | null;
  computed_at: string;
}

export interface ScenarioLever {
  key: string;
  label: string;
  description: string;
  action: string;
  domain: string;
  domain_label: string;
  direction: "improvement" | "adverse" | "absolute";
  overrides: Record<string, number>;
  implies: string[];
  affected_features: Array<{ name: string; label: string }>;
}

export interface ScenarioComparison {
  current: ScoreOut;
  scenario: ScoreOut;
  delta_probability: number;
  delta_interval_width: number;
  changes: ScoreOut["scenario"] extends null ? never : NonNullable<ScoreOut["scenario"]>["changes"];
  disclaimer: string;
}

export interface Comparator {
  track_id: string;
  program_id: string;
  program_name: string;
  sponsor: string | null;
  target_symbol: string | null;
  indication: string;
  modality: string;
  similarity: number;
  matched_facets: Array<{ facet: string; label: string; weight: number; detail: string }>;
  rationale: string;
  outcome_label: string;
  outcome_display: string;
  outcome_date: string | null;
  highest_phase_reached: string;
  discontinuation_reason: string;
  outcome_confidence: string;
  ind_or_cta_on: string | null;
  is_synthetic: boolean;
}

export interface ComparatorResponse {
  track_id: string;
  as_of_date: string;
  comparators: Comparator[];
  summary: {
    n: number;
    outcome_counts?: Record<string, number>;
    n_efficacy_failures?: number;
    n_progressed_past_phase_2?: number;
    observed_failure_share?: number | null;
    mean_similarity?: number;
    caveat: string;
  };
}

export interface PortfolioRow {
  /** null when scoring was not requested; false when it was refused. */
  score_permitted?: boolean | null;
  refusal_code?: string | null;
  track_id: string;
  program_id: string;
  program_name: string;
  target_symbol: string | null;
  indication: string;
  therapeutic_area: string;
  modality: string;
  sponsor: string | null;
  stage: string;
  as_of_date: string;
  probability: number | null;
  interval_low: number | null;
  interval_high: number | null;
  evidence_confidence: ConfidenceLevel | null;
  completeness: number | null;
  top_risk_driver: string | null;
  top_risk_driver_domain: string | null;
  n_evidence_items: number;
  outcome_label: string;
  is_synthetic: boolean;
  last_scored_at: string | null;
}

export interface PortfolioResponse {
  project_id: string | null;
  project_name: string | null;
  model_version: string | null;
  endpoint: string;
  estimand: string;
  rows: PortfolioRow[];
  /** Rows on this page. */
  n_returned: number;
  /** Programmes matching the active filters — the portfolio-level denominator. */
  total_matching: number;
  page_offset: number;
  page_limit: number;
  has_more: boolean;
  /** Summaries over this page's rows only. Never portfolio-wide. */
  page_summary: {
    scope: string;
    n_rows_on_page: number;
    n_scored_on_page: number;
    n_refused_on_page: number;
    mean_probability: number | null;
    mean_interval_width: number | null;
    n_low_confidence_on_page: number;
    note: string;
  };
  is_demonstration: boolean;
  demonstration_note: string | null;
  model_usefulness: { verdict?: string; headline?: string };
}

export interface EvidenceMatrixCell {
  evidence_class: string;
  system: string;
  endpoint: string;
  n_total: number;
  n_supporting: number;
  n_refuting: number;
  evidence_ids: string[];
}

export interface EvidenceMatrix {
  rows: string[];
  columns: string[];
  cells: EvidenceMatrixCell[];
  n_evidence: number;
  empty_cells: number;
  coverage: number;
  note: string;
}

export interface MetricEstimate {
  value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n_bootstrap: number;
}

export interface ValidationReport {
  model_version: string;
  generated_at: string | null;
  evaluation_design: {
    split: string;
    sizes: Record<string, number>;
    leakage_audit: Record<string, unknown>;
    grouping: Record<string, unknown>;
    statement: string;
  };
  cohort: Record<string, any>;
  primary_metrics: {
    n: number;
    n_positive: number;
    prevalence: number | null;
    discrimination: { auroc: MetricEstimate; auprc: MetricEstimate };
    accuracy: { brier: MetricEstimate; log_loss: MetricEstimate; scaled_brier: MetricEstimate };
    calibration: {
      calibration_intercept: MetricEstimate;
      calibration_slope: MetricEstimate;
      ece: number | null;
      mce: number | null;
      bins: Array<{
        lower: number; upper: number; n: number;
        mean_predicted: number; observed_rate: number;
        observed_ci_low: number; observed_ci_high: number;
      }>;
    };
    thresholds: Array<Record<string, number | null>>;
    decision_curve: {
      thresholds: number[]; model: number[]; treat_all: number[]; treat_none: number[];
      prevalence: number; interpretation: string;
    };
  };
  calibration_selection: Record<string, any>;
  subgroup_performance: Record<string, { levels: any[]; min_n_for_reporting: number }>;
  coefficient_sign_audit: Record<string, any>;
  interval_agreement: Record<string, any>;
  domain_ablation: { baseline?: any; by_domain?: any[]; interpretation?: string };
  model_comparison: Record<string, any>;
  target_grouped_sensitivity: Record<string, any>;
  hyperparameter_selection: Record<string, any>;
  usefulness: { verdict?: string; headline?: string; why_this_matters?: string };
  limitations: string[];
  oracle_benchmark: Record<string, any> | null;
}

export interface Coefficient {
  column: string;
  feature: string | null;
  label: string | null;
  domain: string | null;
  coefficient: number;
  odds_ratio_per_sd: number;
  posterior_sd: number | null;
  ci_low: number | null;
  ci_high: number | null;
  expected_direction: string | null;
  sign_matches_expectation: boolean | null;
}

export interface ConnectorHealth {
  connector: string;
  status: "ok" | "degraded" | "unreachable" | "not_configured" | "disabled";
  checked_at: string;
  latency_ms: number | null;
  detail: string;
  endpoint: string | null;
}

export interface FeatureSpec {
  name: string;
  label: string;
  domain: string;
  domain_label: string;
  kind: string;
  description: string;
  derivation: string;
  expected_direction: string;
  missing_policy: string;
  scenario_adjustable: boolean;
  unit: string | null;
  valid_min: number | null;
  valid_max: number | null;
  completeness_weight: number;
}

export interface HealthOut {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  database: string;
  active_model: string | null;
  feature_schema_version: string;
  is_demonstration_deployment: boolean;
  development_auth_fallback: boolean;
}
