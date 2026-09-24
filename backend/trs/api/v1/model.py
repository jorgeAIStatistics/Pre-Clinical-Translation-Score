"""Model registry, validation report and methods endpoints.

These back the Methods page, which is the page that decides whether a
scientifically literate user trusts the product. It therefore exposes the
unflattering numbers — the ablations, the sign audit, the interval-agreement
check, the limitations, and the verdict on whether the model beats the cohort
base rate at all — rather than a curated selection.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.api.deps import get_db, get_principal
from trs.api.schemas import (
    CoefficientOut,
    FeatureSpecOut,
    ModelVersionOut,
    ValidationReportOut,
)
from trs.api.security import Principal
from trs.cohort.outcomes import ENDPOINTS
from trs.db.models.analytics import CohortVersion, ModelArtifact
from trs.features.registry import FEATURES, FEATURES_BY_NAME
from trs.features.spec import DOMAIN_LABELS, FEATURE_SCHEMA_VERSION, FeatureDomain

router = APIRouter(prefix="/model", tags=["model"])


def _active(session: Session, endpoint: str) -> ModelArtifact:
    artifact = session.scalars(
        select(ModelArtifact).where(
            ModelArtifact.endpoint == endpoint, ModelArtifact.is_active.is_(True)
        )
    ).first()
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active model for endpoint {endpoint!r}.",
        )
    return artifact


@router.get("/version", response_model=ModelVersionOut)
def get_version(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    endpoint: str = "ph2_efficacy_failure",
) -> ModelVersionOut:
    return ModelVersionOut.model_validate(_active(session, endpoint))


@router.get("/versions", response_model=list[ModelVersionOut])
def list_versions(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[ModelVersionOut]:
    return [
        ModelVersionOut.model_validate(a)
        for a in session.scalars(
            select(ModelArtifact).order_by(ModelArtifact.trained_at.desc())
        )
    ]


@router.get("/validation", response_model=ValidationReportOut)
def get_validation(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    endpoint: str = "ph2_efficacy_failure",
    version: str | None = None,
) -> ValidationReportOut:
    """The full validation report for a model version."""
    artifact = (
        session.scalars(select(ModelArtifact).where(ModelArtifact.version == version)).first()
        if version else _active(session, endpoint)
    )
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")

    metrics = artifact.metrics or {}
    oracle = None
    if artifact.trained_on_synthetic:
        # On demonstration data the attainable ceiling is computable, and
        # reporting a metric without it invites a reader to over-interpret.
        try:
            from trs.seed.diagnostics import oracle_benchmark

            oracle = oracle_benchmark(session, cohort_version_id=artifact.cohort_version_id)
        except Exception:  # noqa: BLE001
            oracle = None

    return ValidationReportOut(
        model_version=artifact.version,
        generated_at=metrics.get("generated_at"),
        evaluation_design=metrics.get("evaluation_design", {}),
        cohort=metrics.get("cohort", {}),
        primary_metrics=metrics.get("primary_metrics", {}),
        calibration_selection=metrics.get("calibration_selection", {}),
        subgroup_performance=metrics.get("subgroup_performance", {}),
        coefficient_sign_audit=metrics.get("coefficient_sign_audit", {}),
        interval_agreement=metrics.get("interval_agreement", {}),
        domain_ablation=metrics.get("domain_ablation", {}),
        model_comparison=metrics.get("model_comparison", {}),
        target_grouped_sensitivity=metrics.get("target_grouped_sensitivity", {}),
        hyperparameter_selection=metrics.get("hyperparameter_selection", {}),
        usefulness=metrics.get("usefulness", {}),
        limitations=metrics.get("limitations", []),
        oracle_benchmark=oracle,
    )


@router.get("/coefficients", response_model=list[CoefficientOut])
def get_coefficients(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    endpoint: str = "ph2_efficacy_failure",
    version: str | None = None,
) -> list[CoefficientOut]:
    """Fitted coefficients with posterior intervals, auditable one by one."""
    artifact = (
        session.scalars(select(ModelArtifact).where(ModelArtifact.version == version)).first()
        if version else _active(session, endpoint)
    )
    if artifact is None or not artifact.coefficients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This model family does not expose interpretable coefficients.",
        )

    out: list[CoefficientOut] = []
    for row in artifact.coefficients.get("columns", []):
        column = row["column"]
        base = column.split("=")[0].replace("__missing", "")
        spec = FEATURES_BY_NAME.get(base)
        expected = str(spec.expected_direction) if spec else None
        matches = None
        if spec and expected and expected != "uncertain" and not column.endswith("__missing") and "=" not in column:
            matches = (row["coefficient"] > 0) == (expected == "increases_risk")
        out.append(CoefficientOut(
            column=column,
            feature=base if spec else None,
            label=(
                f"{spec.label} — not reported" if column.endswith("__missing") and spec
                else (spec.label if spec else column)
            ),
            domain=str(spec.domain) if spec else None,
            coefficient=row["coefficient"],
            odds_ratio_per_sd=row["odds_ratio_per_sd"],
            posterior_sd=row.get("posterior_sd"),
            ci_low=row.get("ci_low"),
            ci_high=row.get("ci_high"),
            expected_direction=expected,
            sign_matches_expectation=matches,
        ))
    out.sort(key=lambda c: abs(c.coefficient), reverse=True)
    return out


@router.get("/features", response_model=list[FeatureSpecOut])
def get_feature_schema(principal: Principal = Depends(get_principal)) -> list[FeatureSpecOut]:
    """The declared feature schema — the contract between evidence and model."""
    return [
        FeatureSpecOut(
            name=f.name, label=f.label, domain=str(f.domain),
            domain_label=DOMAIN_LABELS[f.domain], kind=str(f.kind),
            description=f.description, derivation=f.derivation,
            expected_direction=str(f.expected_direction),
            missing_policy=str(f.missing_policy),
            scenario_adjustable=f.scenario_adjustable,
            unit=f.unit, valid_min=f.valid_min, valid_max=f.valid_max,
            completeness_weight=f.completeness_weight,
        )
        for f in FEATURES
    ]


@router.get("/endpoints")
def get_endpoints(principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    """Every supported supervised endpoint and its estimand, stated in words."""
    return {
        "endpoints": [e.to_dict() for e in ENDPOINTS.values()],
        "primary": "ph2_efficacy_failure",
        "note": (
            "Failure modes are kept distinct. The primary endpoint is "
            "cause-specific: programs withdrawn for safety, PK/PD or commercial "
            "reasons are excluded rather than counted as efficacy failures."
        ),
    }


@router.get("/cohorts")
def get_cohorts(
    session: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> list[dict[str, Any]]:
    """Cohort versions with their full inclusion/exclusion waterfall."""
    return [
        {
            "id": c.id, "name": c.name, "version": c.version,
            "description": c.description, "definition": c.definition,
            "inclusion_summary": c.inclusion_summary,
            "n_entries": c.n_entries, "n_positive": c.n_positive,
            "n_negative": c.n_negative, "n_excluded": c.n_excluded,
            "as_of_rule": c.as_of_rule, "built_at": c.built_at.isoformat(),
            "dataset_hash": c.dataset_hash,
            "contains_synthetic": c.contains_synthetic,
        }
        for c in session.scalars(
            select(CohortVersion).order_by(CohortVersion.built_at.desc())
        )
    ]


@router.get("/schema-version")
def get_schema_version(principal: Principal = Depends(get_principal)) -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_features": len(FEATURES),
        "domains": [
            {
                "key": str(d), "label": DOMAIN_LABELS[d],
                "n_features": sum(1 for f in FEATURES if f.domain is d),
            }
            for d in FeatureDomain
        ],
    }
