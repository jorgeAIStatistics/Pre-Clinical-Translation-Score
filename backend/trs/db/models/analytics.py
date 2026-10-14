"""Analytical tables: cohorts, feature snapshots, model artifacts, scores.

These tables exist so a prediction can be *reconstructed*. Given a score id you
can recover the exact feature vector, the exact cohort definition, the exact
model artifact and the exact evidence ids that produced it — which is the
difference between a decision-support tool and a number on a slide.
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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from trs.db.base import (
    Base,
    IdMixin,
    JSONVariant,
    SyntheticFlagMixin,
    TimestampMixin,
    UTCDateTime,
)

# ─────────────────────────────────────────────────────────────────────────────
# Cohorts
# ─────────────────────────────────────────────────────────────────────────────

class CohortVersion(Base, IdMixin, TimestampMixin):
    """A frozen, named cohort definition.

    Changing an inclusion rule creates a new cohort version rather than
    mutating the old one, so a historical model artifact keeps pointing at the
    cohort it was actually trained on.
    """

    __tablename__ = "cohort_version"
    __table_args__ = (UniqueConstraint("name", "version"),)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    #: Which OutcomeLabel values map to positive / negative / excluded, plus
    #: the as-of rule and any filters. Fully declarative and serialisable.
    definition: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    #: Counts by inclusion decision, written at build time.
    inclusion_summary: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    n_entries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_positive: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_negative: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_excluded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    as_of_rule: Mapped[str] = mapped_column(String(80), nullable=False, default="ind_or_cta_on")
    built_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    dataset_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    contains_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class CohortEntry(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """One supervised example: a development track observed at a cutoff date."""

    __tablename__ = "cohort_entry"
    __table_args__ = (
        UniqueConstraint("cohort_version_id", "program_indication_id", "as_of_date"),
        Index("ix_cohort_entry_version_split", "cohort_version_id", "split"),
    )

    cohort_version_id: Mapped[str] = mapped_column(
        ForeignKey("cohort_version.id", ondelete="CASCADE"), index=True
    )
    program_indication_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    #: THE cutoff. Features may use only evidence with available_at <= this.
    as_of_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)

    # ---- Label -------------------------------------------------------------
    #: Primary MVP endpoint: 1 = failed to show sufficient efficacy through Ph2.
    label_primary: Mapped[int | None] = mapped_column(Integer, default=None)
    #: Secondary labels, keyed by endpoint name (see trs.cohort.outcomes).
    labels_secondary: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    label_source_outcome: Mapped[str | None] = mapped_column(String(48), default=None)
    label_confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="not_assessed")
    #: Recorded when the track is excluded from the primary label (e.g. a
    #: strategic discontinuation, or insufficient follow-up).
    exclusion_reason: Mapped[str | None] = mapped_column(String(80), default=None)

    # ---- Time-to-event support ---------------------------------------------
    #: Days from as_of_date to the observed event or to censoring.
    time_to_event_days: Mapped[int | None] = mapped_column(Integer, default=None)
    event_observed: Mapped[bool | None] = mapped_column(Boolean, default=None)
    #: Competing-risk event type (efficacy / safety / pkpd / strategic / none).
    event_type: Mapped[str | None] = mapped_column(String(48), default=None)

    # ---- Grouping keys for leakage-resistant splitting ----------------------
    group_program_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    group_target_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    group_sponsor_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    #: Coarser key uniting programs that share target *and* modality, which is
    #: how "the same bet, made twice" usually shows up in a pipeline database.
    group_mechanism_key: Mapped[str | None] = mapped_column(String(120), default=None, index=True)

    #: Assigned by trs.validation.splitters: train / calibration / test / holdout.
    split: Mapped[str | None] = mapped_column(String(24), default=None, index=True)
    split_fold: Mapped[int | None] = mapped_column(Integer, default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSnapshot(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A materialised, model-ready feature vector for one analysis unit."""

    __tablename__ = "feature_snapshot"
    __table_args__ = (
        # Identity includes the evidence revision and the endpoint, not just
        # the track, cutoff and schema.
        #
        # The predecessor keyed on (track, as_of, schema) and offered an
        # `overwrite` flag that rewrote `features`, `missingness`,
        # `vector_hash` and `built_at` in place. Any score already referencing
        # that row then pointed at different numbers, which is the opposite of
        # what the docstring claimed the constraint guaranteed. Corrected
        # evidence must produce a *new* snapshot at the same scientific cutoff,
        # so that an old assessment stays reproducible and a new one is
        # available.
        UniqueConstraint(
            "program_indication_id", "as_of_date", "feature_schema_version",
            "endpoint", "evidence_revision",
            name="uq_feature_snapshot_identity",
        ),
        Index("ix_feature_snapshot_lookup", "program_indication_id", "as_of_date"),
        Index(
            "ix_feature_snapshot_revision",
            "program_indication_id", "as_of_date", "endpoint", "evidence_revision",
        ),
    )

    program_indication_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    as_of_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    feature_schema_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    #: {feature_name: float | None} — the raw, pre-transform feature values.
    features: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    #: {feature_name: MissingnessReason} for every feature that is None.
    missingness: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    #: Fraction of schema fields with an observed value, weighted by the
    #: schema's declared importance weights. See trs.features.missingness.
    completeness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    #: Per-domain completeness, e.g. {"pkpd": 0.4, "target_biology": 1.0}.
    domain_completeness: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    #: Ids of every evidence row that contributed, for full traceability.
    contributing_evidence_ids: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: {feature_name: [evidence_id, ...]}
    feature_evidence_map: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    n_evidence_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    built_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    builder_version: Mapped[str] = mapped_column(String(32), nullable=False, default="0")
    #: SHA-256 over the sorted feature dict; detects silent drift.
    vector_hash: Mapped[str | None] = mapped_column(String(64), default=None)

    #: SHA-256 over the content identity of every evidence row that
    #: contributed: its id, its last modification time and its availability
    #: date. Correcting one item, retracting one, or adding one that changes an
    #: aggregate all change this value, and therefore produce a new snapshot
    #: rather than mutating this one.
    #:
    #: Legacy rows written before this column existed carry
    #: ``"legacy-unversioned"``: their evidence revision is genuinely unknown,
    #: and recording that is more useful than back-filling a hash that would
    #: imply it had been checked.
    evidence_revision: Mapped[str] = mapped_column(
        String(64), nullable=False, default="legacy-unversioned", index=True
    )
    #: The endpoint whose resolved outcomes fed the historical-rate features.
    #: Those features are endpoint-specific predictors, so a snapshot built for
    #: one endpoint is not interchangeable with another's — which the
    #: predecessor allowed, because it always used the primary endpoint's
    #: labels whatever endpoint was being trained.
    endpoint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="ph2_efficacy_failure", index=True
    )
    #: Set when a later snapshot for the same track, cutoff, schema and
    #: endpoint supersedes this one because the evidence changed. The row's
    #: feature values are never altered; only this pointer is.
    superseded_by_id: Mapped[str | None] = mapped_column(
        ForeignKey("feature_snapshot.id", ondelete="SET NULL"), default=None
    )


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class ModelArtifact(Base, IdMixin, TimestampMixin):
    """A trained, calibrated, validated model — the deployable unit."""

    __tablename__ = "model_artifact"
    __table_args__ = (UniqueConstraint("version"),)

    #: Human-meaningful semantic version, e.g. "ptrs-0.3.0".
    version: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    #: trs.modeling registry key, e.g. "l2_logistic", "hier_logistic", "gbdt".
    family: Mapped[str] = mapped_column(String(48), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False, default="ph2_efficacy_failure")
    display_name: Mapped[str | None] = mapped_column(String(160), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    feature_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    cohort_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("cohort_version.id", ondelete="SET NULL"), default=None
    )
    #: SHA-256 over the serialised training matrix + labels.
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    hyperparameters: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: Point estimates and posterior summaries for interpretable families.
    coefficients: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: Calibration method and fitted parameters.
    calibration: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: Full validation report (see trs.validation.report).
    metrics: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: Subgroup performance tables.
    subgroup_metrics: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: Model-selection record: which candidates were compared and why this won.
    selection_record: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    trained_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    training_duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
    n_train: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_calibration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Relative path of the serialised artifact under settings.artifact_dir.
    artifact_path: Mapped[str | None] = mapped_column(String(400), default=None)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), default=None)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    #: True when any training row was synthetic. Propagates to every score.
    trained_on_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: Free-text limitations shown verbatim on the Methods page.
    limitations: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    git_commit: Mapped[str | None] = mapped_column(String(64), default=None)
    library_versions: Mapped[dict | None] = mapped_column(JSONVariant, default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Scores and scenarios
# ─────────────────────────────────────────────────────────────────────────────

class Scenario(Base, IdMixin, TimestampMixin):
    """A hypothetical modification of a track's evidence state.

    Scenarios never write evidence. They carry feature-level overrides that are
    applied to a copy of the snapshot at scoring time, so the underlying
    evidence base cannot be contaminated by speculation.
    """

    __tablename__ = "scenario"

    organization_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    program_indication_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    #: {feature_name: new_value}. Validated against the feature schema.
    overrides: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    #: Which named scenario levers produced these overrides, for display.
    lever_keys: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    created_by_user_id: Mapped[str | None] = mapped_column(String(32), default=None)
    is_shared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class ScoreRecord(Base, IdMixin, TimestampMixin, SyntheticFlagMixin):
    """A persisted prediction, fully reconstructable."""

    __tablename__ = "score_record"
    __table_args__ = (
        Index("ix_score_lookup", "program_indication_id", "as_of_date", "model_version"),
    )

    organization_id: Mapped[str | None] = mapped_column(String(32), default=None, index=True)
    program_indication_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    feature_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("feature_snapshot.id", ondelete="SET NULL"), default=None
    )
    scenario_id: Mapped[str | None] = mapped_column(
        ForeignKey("scenario.id", ondelete="SET NULL"), default=None, index=True
    )
    as_of_date: Mapped[dt.date] = mapped_column(Date, nullable=False)

    model_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_artifact.id", ondelete="SET NULL"), default=None
    )
    model_version: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False, default="ph2_efficacy_failure")

    #: Calibrated probability of the endpoint.
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    interval_low: Mapped[float] = mapped_column(Float, nullable=False)
    interval_high: Mapped[float] = mapped_column(Float, nullable=False)
    interval_level: Mapped[float] = mapped_column(Float, nullable=False, default=0.95)
    #: {"parameter": w, "imputation": w, "total_sd_logit": s, ...}
    uncertainty_decomposition: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    #: Exact additive log-odds decomposition; see trs.scoring.explanation.
    contributions: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    #: Domain-level roll-up of the same decomposition.
    domain_contributions: Mapped[dict | None] = mapped_column(JSONVariant, default=None)

    completeness_score: Mapped[float | None] = mapped_column(Float, default=None)
    #: Overall evidence-confidence grade shown next to the number.
    evidence_confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="not_assessed")
    #: True when the model, the evidence, or both include synthetic content.
    is_demonstration: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    computed_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    computed_by_user_id: Mapped[str | None] = mapped_column(String(32), default=None)
    request_id: Mapped[str | None] = mapped_column(String(64), default=None)


class ComparatorLink(Base, IdMixin, TimestampMixin):
    """A cached similarity edge between two development tracks."""

    __tablename__ = "comparator_link"
    __table_args__ = (
        UniqueConstraint("source_track_id", "comparator_track_id", "as_of_date"),
        Index("ix_comparator_lookup", "source_track_id", "similarity"),
    )

    source_track_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    comparator_track_id: Mapped[str] = mapped_column(
        ForeignKey("program_indication.id", ondelete="CASCADE"), index=True
    )
    as_of_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    #: Per-facet similarity contributions, for the "why is this comparable"
    #: rationale shown in the UI.
    components: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    computed_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    matcher_version: Mapped[str] = mapped_column(String(32), nullable=False, default="0")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline observability
# ─────────────────────────────────────────────────────────────────────────────

class ConnectorRun(Base, IdMixin):
    """One execution of an ingestion connector, successful or not."""

    __tablename__ = "connector_run"
    __table_args__ = (Index("ix_connector_run_lookup", "connector", "started_at"),)

    connector: Mapped[str] = mapped_column(String(48), nullable=False)
    connector_version: Mapped[str] = mapped_column(String(32), nullable=False, default="0")
    operation: Mapped[str] = mapped_column(String(80), nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
    started_at: Mapped[dt.datetime] = mapped_column(UTCDateTime, nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(UTCDateTime, default=None)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="running")
    n_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_records_raw: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_records_normalised: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_detail: Mapped[dict | None] = mapped_column(JSONVariant, default=None)
