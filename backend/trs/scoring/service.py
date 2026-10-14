"""Scoring orchestration: evidence → snapshot → model → explained estimate.

One entry point, :meth:`ScoringService.score_track`, does the whole chain and
returns a result that is complete enough to be audited: the probability, its
interval and the decomposition of that interval, the exact driver attribution,
the completeness assessment, the evidence-confidence grade, the model version,
and the identifiers needed to reconstruct every one of them later.

Guarantees this layer enforces
------------------------------
* **The as-of date governs.** Features are always rebuilt through
  :class:`~trs.features.temporal.AsOfContext`, never read from a snapshot built
  at a different cutoff.
* **Schema agreement is checked, not assumed.** A model trained under one
  feature-schema version cannot score a snapshot built under another; the
  columns would misalign silently. The artifact loader refuses.
* **Synthetic status propagates.** If the model, the evidence, or both include
  demonstration data, the result is flagged ``is_demonstration`` and the
  confidence grade is forced to its lowest level. Nothing downstream can
  present it as a validated estimate.
* **No interval without a basis.** If the deployed family cannot produce a
  posterior, the service raises rather than returning a fabricated range.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.db.base import utcnow
from trs.db.models.analytics import ModelArtifact, Scenario, ScoreRecord
from trs.db.models.core import Program, ProgramIndication, Target
from trs.features.missingness import compute_completeness, grade_evidence_confidence
from trs.features.spec import MODALITY_GROUPS, TARGET_CLASS_GROUPS
from trs.modeling.contract import build_contract
from trs.modeling.uncertainty import (
    DEFAULT_N_IMPUTATIONS,
    DEFAULT_N_PARAM_SAMPLES,
    predict_with_uncertainty,
    specification_sensitivity,
)
from trs.pipeline.artifacts import ModelBundle, get_active_artifact, load_bundle
from trs.pipeline.features import build_history_index, materialise_snapshot
from trs.pipeline.status import check_eligibility
from trs.scoring.explanation import ExplanationUnavailable, explain
from trs.scoring.scenarios import apply_overrides, resolve_levers, validate_overrides


class ScoringError(RuntimeError):
    pass


@dataclasses.dataclass(slots=True)
class ScoreResult:
    track_id: str
    program_id: str
    program_name: str
    as_of_date: dt.date
    endpoint: str
    estimand: str
    model_version: str
    model_family: str

    #: ``None`` when the eligibility policy refused to produce a score. A
    #: refusal is not a zero and not a wide interval; it is the absence of a
    #: number, with a machine-readable reason in :attr:`eligibility`.
    probability: float | None
    interval_low: float | None
    interval_high: float | None
    interval_level: float
    uncertainty: dict[str, Any] | None
    #: The eligibility decision from :mod:`trs.pipeline.status`. Always present,
    #: whether or not a score was produced.
    eligibility: dict[str, Any]
    #: What the served number *is*. See :mod:`trs.modeling.contract`.
    prediction_contract: dict[str, Any] | None
    #: Model status and the criteria behind it, carried beside every estimate.
    model_status: dict[str, Any]

    explanation: dict[str, Any] | None
    explanation_unavailable_reason: str | None

    completeness: dict[str, Any]
    evidence_confidence: str
    confidence_reasons: list[str]

    feature_snapshot_id: str | None
    n_evidence_items: int
    is_demonstration: bool
    demonstration_note: str | None
    model_usefulness: dict[str, Any]
    specification_sensitivity: dict[str, Any]
    scenario: dict[str, Any] | None
    score_record_id: str | None
    computed_at: dt.datetime

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["as_of_date"] = self.as_of_date.isoformat()
        d["computed_at"] = self.computed_at.isoformat()
        return d


class ScoringService:
    """Scores development tracks against the active model artifact."""

    def __init__(self, session: Session, *, random_state: int = 12345) -> None:
        self.session = session
        self.random_state = random_state
        self._bundle_cache: dict[str, ModelBundle] = {}

    # ---- artifact ----------------------------------------------------------

    def active_artifact(self, endpoint: str) -> ModelArtifact:
        artifact = get_active_artifact(self.session, endpoint)
        if artifact is None:
            raise ScoringError(
                f"No active model artifact for endpoint {endpoint!r}. Train one "
                "with `python -m trs.cli train` before scoring."
            )
        return artifact

    def bundle_for(self, version: str) -> ModelBundle:
        if version not in self._bundle_cache:
            self._bundle_cache[version] = load_bundle(self.session, version)
        return self._bundle_cache[version]

    # ---- scoring -----------------------------------------------------------

    def score_track(
        self,
        track_id: str,
        *,
        as_of: dt.date | None = None,
        endpoint: str = "ph2_efficacy_failure",
        model_version: str | None = None,
        scenario_id: str | None = None,
        lever_keys: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
        interval_level: float = 0.95,
        n_imputations: int = DEFAULT_N_IMPUTATIONS,
        n_param_samples: int = DEFAULT_N_PARAM_SAMPLES,
        persist: bool = True,
        user_id: str | None = None,
        request_id: str | None = None,
        compare_families: bool = True,
        demo_context: bool = False,
    ) -> ScoreResult:
        """Score one development track, or refuse and say why.

        ``demo_context`` must be set explicitly by the caller to obtain a score
        from a model trained on synthetic data. There is no default-on and no
        server-side override: a demonstration number should exist only because
        someone asked for one in a context that says so. A synthetic-trained
        model scoring *real* evidence is refused regardless of this flag.
        """
        track = self.session.get(ProgramIndication, track_id)
        if track is None:
            raise ScoringError(f"No development track {track_id!r}")
        program = self.session.get(Program, track.program_id)

        # The decision point defaults to entry into human development, which is
        # the moment this product exists to support. Falling back to "today"
        # would quietly change the question being asked.
        as_of = as_of or track.ind_or_cta_on or track.first_in_human_on
        if as_of is None:
            raise ScoringError(
                f"Track {track_id!r} has no IND/CTA or first-in-human date, so "
                "there is no defensible as-of cutoff. Supply one explicitly."
            )

        artifact = (
            self.session.scalars(
                select(ModelArtifact).where(ModelArtifact.version == model_version)
            ).first()
            if model_version else self.active_artifact(endpoint)
        )
        if artifact is None:
            raise ScoringError(f"No model artifact {model_version!r}")
        bundle = self.bundle_for(artifact.version)

        # ---- features ------------------------------------------------------
        # The endpoint comes from the artifact, not from the request default:
        # the historical-rate features are built from resolved outcomes under a
        # specific endpoint definition, and scoring a track with a snapshot
        # built under a different one would mix predictors.
        from trs.cohort.outcomes import ENDPOINTS as _ENDPOINTS

        endpoint_def_for_features = _ENDPOINTS[artifact.endpoint]
        history = build_history_index(self.session, endpoint_def_for_features)
        snapshot = materialise_snapshot(
            self.session, track_id, as_of,
            endpoint=endpoint_def_for_features, history=history,
        )
        base_row = dict(snapshot.features)
        missingness = dict(snapshot.missingness)

        # ---- scenario ------------------------------------------------------
        scenario_payload: dict[str, Any] | None = None
        feature_row = base_row
        if scenario_id or lever_keys or overrides:
            resolved: dict[str, float] = {}
            applied_levers: list[str] = []
            if scenario_id:
                stored = self.session.get(Scenario, scenario_id)
                if stored is None:
                    raise ScoringError(f"No scenario {scenario_id!r}")
                resolved.update(validate_overrides(stored.overrides or {}))
                applied_levers.extend((stored.lever_keys or {}).get("keys", []))
            if lever_keys:
                resolved.update(resolve_levers(lever_keys, current=base_row))
                applied_levers.extend(lever_keys)
            if overrides:
                resolved.update(validate_overrides(overrides))

            feature_row, changes = apply_overrides(base_row, resolved)
            for name in resolved:
                missingness[name] = "present"
            scenario_payload = {
                "scenario_id": scenario_id,
                "lever_keys": applied_levers,
                "overrides": resolved,
                "changes": changes,
                "disclaimer": (
                    "A scenario recomputes this model under a hypothetical "
                    "evidence state. It is a sensitivity analysis of the "
                    "estimate, not a prediction that the experiment would "
                    "succeed or that it would produce this result."
                ),
            }

        # ---- eligibility ---------------------------------------------------
        # Decided before any number is computed, so a refusal cannot be a
        # probability with a warning stapled to it. Evidence review,
        # completeness and provenance are unaffected and are returned either
        # way: declining to score is not declining to be useful.
        report = compute_completeness(missingness)
        contains_synthetic = bool(snapshot.is_synthetic or bundle.trained_on_synthetic)
        status_record = (artifact.metrics or {}).get("status", {})
        eligibility = check_eligibility(
            status=bundle.status,
            trained_on_synthetic=bundle.trained_on_synthetic,
            evidence_is_synthetic=bool(snapshot.is_synthetic),
            demo_context_selected=demo_context,
        )

        from trs.cohort.outcomes import ENDPOINTS

        endpoint_def = ENDPOINTS[artifact.endpoint]
        usefulness = (artifact.metrics or {}).get("usefulness", {})

        if not eligibility.permitted:
            grade, reasons = grade_evidence_confidence(
                report,
                n_evidence_items=snapshot.n_evidence_items,
                interval_width=None,
                contains_synthetic=contains_synthetic,
            )
            return ScoreResult(
                track_id=track_id,
                program_id=track.program_id,
                program_name=program.name,
                as_of_date=as_of,
                endpoint=artifact.endpoint,
                estimand=endpoint_def.estimand,
                model_version=artifact.version,
                model_family=artifact.family,
                probability=None,
                interval_low=None,
                interval_high=None,
                interval_level=interval_level,
                uncertainty=None,
                eligibility=eligibility.to_dict(),
                prediction_contract=None,
                model_status=status_record,
                explanation=None,
                explanation_unavailable_reason=(
                    "No score was produced, so there is nothing to explain. "
                    f"Reason: {eligibility.reason}"
                ),
                completeness=report.to_dict(),
                evidence_confidence=str(grade),
                confidence_reasons=reasons,
                feature_snapshot_id=snapshot.id,
                n_evidence_items=snapshot.n_evidence_items,
                is_demonstration=contains_synthetic,
                demonstration_note=None,
                model_usefulness=usefulness,
                specification_sensitivity={
                    "available": False,
                    "reason": "No score was produced for this programme.",
                },
                scenario=None,
                score_record_id=None,
                computed_at=utcnow(),
            )

        # ---- prediction ----------------------------------------------------
        rng = np.random.default_rng(self.random_state)
        mech_group = self._mechanism_key(program)
        uncertainty = predict_with_uncertainty(
            bundle.model, bundle.calibrator, bundle.transformer, feature_row,
            rng=rng, n_imputations=n_imputations, n_param_samples=n_param_samples,
            level=interval_level,
            groups=mech_group if bundle.group_key_name else None,
        )

        # ---- explanation ---------------------------------------------------
        explanation_payload: dict[str, Any] | None = None
        explanation_reason: str | None = None
        try:
            explanation_payload = explain(
                bundle.model, bundle.calibrator, bundle.transformer, feature_row,
                missingness=missingness,
                evidence_map=snapshot.feature_evidence_map or {},
                # Hand the explainer the number this response will carry, so
                # the decomposition is checked against what the reader sees
                # rather than only against itself.
                served_probability=uncertainty.probability,
            ).to_dict()
        except ExplanationUnavailable as exc:
            explanation_reason = str(exc)

        # ---- completeness and confidence -----------------------------------
        grade, reasons = grade_evidence_confidence(
            report,
            n_evidence_items=snapshot.n_evidence_items,
            interval_width=uncertainty.interval_width,
            contains_synthetic=contains_synthetic,
        )

        # ---- model-specification sensitivity -------------------------------
        alternatives: dict[str, float] = {}
        if compare_families:
            alternatives = self._alternative_family_predictions(
                artifact.endpoint, artifact.version, feature_row, mech_group
            )
        spec_sensitivity = specification_sensitivity(uncertainty.probability, alternatives)

        # ---- persist -------------------------------------------------------
        record_id: str | None = None
        if persist:
            record = ScoreRecord(
                organization_id=program.organization_id,
                program_indication_id=track_id,
                feature_snapshot_id=snapshot.id,
                scenario_id=scenario_id,
                as_of_date=as_of,
                model_artifact_id=artifact.id,
                model_version=artifact.version,
                endpoint=artifact.endpoint,
                probability=uncertainty.probability,
                interval_low=uncertainty.interval_low,
                interval_high=uncertainty.interval_high,
                interval_level=interval_level,
                uncertainty_decomposition=uncertainty.to_dict(),
                contributions=(explanation_payload or {}).get("drivers"),
                domain_contributions=(explanation_payload or {}).get("by_domain"),
                completeness_score=report.overall,
                evidence_confidence=str(grade),
                is_demonstration=contains_synthetic,
                computed_at=utcnow(),
                computed_by_user_id=user_id,
                request_id=request_id,
                is_synthetic=contains_synthetic,
            )
            self.session.add(record)
            self.session.flush()
            record_id = record.id

        return ScoreResult(
            track_id=track_id,
            program_id=track.program_id,
            program_name=program.name,
            as_of_date=as_of,
            endpoint=artifact.endpoint,
            estimand=endpoint_def.estimand,
            model_version=artifact.version,
            model_family=artifact.family,
            probability=uncertainty.probability,
            interval_low=uncertainty.interval_low,
            interval_high=uncertainty.interval_high,
            interval_level=interval_level,
            uncertainty=uncertainty.to_dict(),
            eligibility=eligibility.to_dict(),
            prediction_contract=build_contract(
                primary_probability=uncertainty.probability,
                predictive_mean_probability=uncertainty.predictive_mean_probability,
                calibration_method=bundle.calibrator.method,
            ).to_dict(),
            model_status=status_record,
            explanation=explanation_payload,
            explanation_unavailable_reason=explanation_reason,
            completeness=report.to_dict(),
            evidence_confidence=str(grade),
            confidence_reasons=reasons,
            feature_snapshot_id=snapshot.id,
            n_evidence_items=snapshot.n_evidence_items,
            is_demonstration=contains_synthetic,
            demonstration_note=(
                "DEMONSTRATION OUTPUT — this estimate depends on synthetic data "
                "and/or a model trained on it. It has no scientific "
                "interpretation and must not inform a real decision."
                if contains_synthetic else None
            ),
            model_usefulness=usefulness,
            specification_sensitivity=spec_sensitivity,
            scenario=scenario_payload,
            score_record_id=record_id,
            computed_at=utcnow(),
        )

    # ---- helpers -----------------------------------------------------------

    def _mechanism_key(self, program: Program) -> str:
        target = (
            self.session.get(Target, program.primary_target_id)
            if program.primary_target_id else None
        )
        tc = TARGET_CLASS_GROUPS.get(getattr(target, "target_class", "unknown") or "unknown", "other")
        mg = MODALITY_GROUPS.get(program.modality, "other")
        return f"{tc}|{mg}"

    def _alternative_family_predictions(
        self, endpoint: str, deployed_version: str, feature_row: dict, mech: str
    ) -> dict[str, float]:
        """Score the same row with artifacts of *other model families*.

        Reported beside the estimate as specification sensitivity, never folded
        into the interval — averaging across families would assert a prior over
        specifications that nobody has stated.

        Only one artifact per family, and never another version of the deployed
        family. Comparing the current model against an older run of the same
        family measures training-run variation, which is a different quantity
        and a misleading one to label "specification sensitivity": superseded
        artifacts may have been retired precisely because they were wrong, and
        their disagreement says nothing about how much the *choice of model
        form* matters.
        """
        out: dict[str, float] = {}
        deployed = self.session.scalars(
            select(ModelArtifact).where(ModelArtifact.version == deployed_version)
        ).first()
        deployed_family = deployed.family if deployed else None

        candidates = self.session.scalars(
            select(ModelArtifact).where(
                ModelArtifact.endpoint == endpoint,
                ModelArtifact.version != deployed_version,
                ModelArtifact.family != deployed_family,
            ).order_by(ModelArtifact.trained_at.desc())
        ).all()
        seen_families: set[str] = set()
        rows = []
        for row in candidates:
            if row.family in seen_families:
                continue
            seen_families.add(row.family)
            rows.append(row)
        for row in rows:
            try:
                alt = self.bundle_for(row.version)
                X = alt.transformer.transform([feature_row])
                logit = (
                    alt.model.predict_logit(X, groups=np.asarray([mech]))
                    if alt.group_key_name else alt.model.predict_logit(X)
                )
                out[f"{row.family}:{row.version}"] = float(
                    alt.calibrator.transform(np.atleast_1d(logit))[0]
                )
            except Exception:  # noqa: BLE001 - an unloadable artifact is skipped, not fatal
                continue
        return out

    # ---- batch -------------------------------------------------------------

    def score_portfolio(
        self, project_id: str | None = None, *, endpoint: str = "ph2_efficacy_failure",
        limit: int = 500, persist: bool = False,
    ) -> list[ScoreResult]:
        """Score every track in a project. Used by the portfolio view."""
        stmt = (
            select(ProgramIndication)
            .join(Program, Program.id == ProgramIndication.program_id)
            .where(ProgramIndication.ind_or_cta_on.is_not(None))
        )
        if project_id:
            stmt = stmt.where(Program.project_id == project_id)
        results: list[ScoreResult] = []
        for track in self.session.scalars(stmt.limit(limit)):
            try:
                results.append(
                    self.score_track(
                        track.id, endpoint=endpoint, persist=persist,
                        n_imputations=10, n_param_samples=200,
                        compare_families=False,
                    )
                )
            except ScoringError:
                continue
        return results
