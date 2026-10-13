"""Authoritative fact sheet, derived from the live database and model artifact.

Why this exists
---------------
Documentation that quotes metrics goes stale the moment a model is retrained,
and hand-transcribed numbers go wrong even when they are fresh. Every
quantitative claim in the project's Markdown is supposed to be checkable, so
there has to be one command that prints the checkable version.

``python -m trs.cli facts`` regenerates it. Nothing here is written by hand: if
a number appears in a document and not in this output, that number has no
source.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trs.cohort.outcomes import ENDPOINTS
from trs.db.base import Base
from trs.db.models.analytics import CohortVersion, ModelArtifact
from trs.db.models.core import Indication, Program, ProgramIndication, Sponsor, Target
from trs.db.models.evidence import (
    EvidenceItem,
    PkPdBridge,
    ProgramBiomarker,
    SafetyFinding,
    Source,
    TargetAssessment,
)
from trs.features.registry import FEATURES, scenario_adjustable_features
from trs.features.spec import FEATURE_SCHEMA_VERSION, FeatureDomain
from trs.scoring.scenarios import LEVERS

_COUNTED = [
    ("programs", Program), ("development_tracks", ProgramIndication),
    ("evidence_items", EvidenceItem), ("sources", Source),
    ("targets", Target), ("indications", Indication), ("sponsors", Sponsor),
    ("pkpd_bridges", PkPdBridge), ("target_assessments", TargetAssessment),
    ("program_biomarkers", ProgramBiomarker), ("safety_findings", SafetyFinding),
]


def fact_sheet(session: Session) -> dict[str, Any]:
    """Every checkable number, as structured data."""
    import trs.db.models  # noqa: F401  (registers tables on Base.metadata)

    facts: dict[str, Any] = {
        "dataset": {
            name: session.scalar(select(func.count()).select_from(model))
            for name, model in _COUNTED
        },
        "schema": {
            "database_tables": len(Base.metadata.tables),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "declared_features": len(FEATURES),
            "evidence_domains": len(list(FeatureDomain)),
            "scenario_adjustable_features": len(scenario_adjustable_features()),
            "scenario_levers": len(LEVERS),
            "supervised_endpoints": len(ENDPOINTS),
            "features_by_domain": {
                str(d): sum(1 for f in FEATURES if f.domain is d) for d in FeatureDomain
            },
        },
    }

    cohort = session.scalars(
        select(CohortVersion).order_by(CohortVersion.built_at.desc()).limit(1)
    ).first()
    if cohort is not None:
        facts["cohort"] = {
            "name": f"{cohort.name}/{cohort.version}",
            "candidates": (cohort.inclusion_summary or {}).get("n_candidates"),
            "included": cohort.n_entries,
            "positive": cohort.n_positive,
            "negative": cohort.n_negative,
            "excluded": cohort.n_excluded,
            "positive_rate": (
                cohort.n_positive / cohort.n_entries if cohort.n_entries else None
            ),
            "as_of_rule": cohort.as_of_rule,
            "dataset_hash": cohort.dataset_hash,
            "exclusions": (cohort.inclusion_summary or {}).get("exclusions", {}),
        }

    artifact = session.scalars(
        select(ModelArtifact).where(ModelArtifact.is_active.is_(True)).limit(1)
    ).first()
    if artifact is not None:
        m = artifact.metrics or {}
        pm = m.get("primary_metrics", {})
        ed = m.get("evaluation_design", {})
        facts["model"] = {
            "version": artifact.version,
            "family": artifact.family,
            "trained_at": artifact.trained_at.isoformat(),
            "hyperparameters": artifact.hyperparameters,
            "n_train": artifact.n_train,
            "n_calibration": artifact.n_calibration,
            "n_test": artifact.n_test,
            "dataset_hash": artifact.dataset_hash,
            "trained_on_synthetic": artifact.trained_on_synthetic,
            "grouping": ed.get("grouping", {}),
            "split_sizes": ed.get("sizes", {}),
            # The protocol governs how every metric below may be read, so it
            # is reported before them rather than in a footnote.
            "protocol": {
                "name": (m.get("protocol", {}).get("split_manifest", {})
                         .get("protocol")),
                "permits_deployment_claim": (
                    m.get("protocol", {}).get("split_manifest", {})
                    .get("permits_deployment_claim")
                ),
                "manifest_hash": (m.get("protocol", {}).get("split_manifest", {})
                                  .get("manifest_hash")),
                "fitting_origin": (
                    m.get("protocol", {}).get("split_manifest", {})
                    .get("dates", {}).get("fitting_origin")
                ),
                "label_cutoff": (
                    m.get("protocol", {}).get("split_manifest", {})
                    .get("dates", {}).get("label_cutoff")
                ),
                "data_cutoff": (
                    m.get("protocol", {}).get("split_manifest", {})
                    .get("dates", {}).get("data_cutoff")
                ),
                "label_maturity": (
                    m.get("protocol", {}).get("split_manifest", {})
                    .get("label_maturity")
                ),
                "rejected": [
                    t.get("protocol")
                    for t in m.get("protocol", {}).get("protocol_trace", [])
                    if "rejected_because" in t
                ],
                "n_folds": len(m.get("protocol", {}).get("folds", [])),
            },
            "status": {
                k: m.get("status", {}).get(k)
                for k in ("status", "may_serve", "criteria_version")
            },
            "prediction_contract": m.get("prediction_contract", {}),
            "reference_model": {
                k: m.get("reference_model", {}).get(k)
                for k in ("kind", "prediction", "evaluation_prevalence")
            },
            "skill_vs_deployable_reference": {
                k: (pm.get("reference") or {}).get(k)
                for k in (
                    "reference_value", "brier_skill", "log_loss_skill",
                    "model_better_than_reference",
                )
            },
            "calibration_scale_audit": {
                k: m.get("calibration_audit", {}).get(k)
                for k in (
                    "verdict", "pool_observed_rate", "development_observed_rate",
                    "mean_logit_shift",
                )
            },
            "calibration": {
                k: m.get("calibration_selection", {}).get(k)
                for k in ("method", "design", "n_out_of_fold_predictions")
            },
            "metrics": {
                "auroc": pm.get("discrimination", {}).get("auroc"),
                "auprc": pm.get("discrimination", {}).get("auprc"),
                "brier": pm.get("accuracy", {}).get("brier"),
                "log_loss": pm.get("accuracy", {}).get("log_loss"),
                "scaled_brier": pm.get("accuracy", {}).get("scaled_brier"),
                "calibration_slope": pm.get("calibration", {}).get("calibration_slope"),
                "calibration_intercept": pm.get("calibration", {}).get("calibration_intercept"),
                "ece": pm.get("calibration", {}).get("ece"),
                "n_test": pm.get("n"),
                "n_events": pm.get("n_positive"),
                "prevalence": pm.get("prevalence"),
            },
            "usefulness": m.get("usefulness", {}),
            "candidates": [
                {
                    "family": c.get("family"), "log_loss": c.get("log_loss"),
                    "auroc": c.get("auroc"), "slope": c.get("calibration_slope"),
                    "verdict": c.get("verdict"),
                }
                for c in m.get("model_comparison", {}).get("comparisons", [])
            ],
            "target_grouped_sensitivity_auroc": m.get(
                "target_grouped_sensitivity", {}
            ).get("auroc"),
            "interval_agreement": {
                k: m.get("interval_agreement", {}).get(k)
                for k in ("verdict", "median_width_ratio_analytic_over_bootstrap", "n_refits")
            },
            "sign_audit": {
                k: m.get("coefficient_sign_audit", {}).get(k)
                for k in ("n_checked", "n_discordant", "n_discordant_credible")
            },
            "domain_ablation": [
                {"domain": r.get("domain_label"), "delta_log_loss": r.get("delta_log_loss_vs_full")}
                for r in m.get("domain_ablation", {}).get("by_domain", [])
            ],
            "limitations": m.get("limitations", []),
        }

    try:
        from trs.seed.diagnostics import oracle_benchmark

        ob = oracle_benchmark(session)
        if ob.get("available"):
            facts["oracle"] = ob
    except Exception:  # noqa: BLE001 — absent on a non-demonstration deployment
        pass

    return facts


def render_fact_sheet(facts: dict[str, Any]) -> str:
    """Markdown rendering, for pasting into a document review."""
    out: list[str] = [
        "# Authoritative fact sheet",
        "",
        "Generated by `python -m trs.cli facts`. Every quantitative claim in this",
        "project's documentation must be checkable against this output. A number",
        "that appears in a document and not here has no source.",
        "",
    ]

    def block(title: str, payload: Any, indent: int = 0) -> None:
        pad = "  " * indent
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, (dict, list)):
                    out.append(f"{pad}- **{k}**:")
                    block("", v, indent + 1)
                else:
                    out.append(f"{pad}- **{k}**: {v}")
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    out.append(f"{pad}- " + ", ".join(f"{k}={v}" for k, v in item.items()))
                else:
                    out.append(f"{pad}- {item}")
        else:
            out.append(f"{pad}{payload}")

    for section, payload in facts.items():
        out.append(f"## {section}")
        block("", payload)
        out.append("")
    return "\n".join(out)


def fact_sheet_json(session: Session) -> str:
    return json.dumps(fact_sheet(session), indent=2, default=str)
