"""Diagnostics for the demonstration dataset — including the oracle ceiling.

Why an oracle benchmark matters
--------------------------------
Reporting "AUROC 0.72 on demonstration data" is close to meaningless on its own:
the reader cannot tell whether that is the pipeline recovering most of an
attainable signal, or a pipeline failing on an easy problem.

Because the simulation knows each track's *true* failure probability, the
maximum attainable discrimination can be computed directly: score every track
by its true probability and measure AUROC. Nothing can beat that, because
nothing can know more than the generator did. The remaining gap between the
oracle and the model is the combined cost of measurement noise in the evidence,
missing disclosure, finite sample size and model misspecification.

The ceiling is well below 1.0 by construction, because outcomes are Bernoulli
draws. That is the point: a demonstration dataset on which a model scored 0.97
would be advertising a capability the product does not have.

Ground truth lives on disk, not in the database. A real deployment has no
ground-truth column, and adding one for the demo would create a column the
serving code could accidentally learn to read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.config import settings
from trs.db.models.analytics import CohortEntry, CohortVersion

GROUND_TRUTH_FILENAME = "demo_ground_truth.json"


def save_ground_truth(ground_truth: dict[str, float], *, root: Path | None = None) -> Path:
    path = Path(root or settings.artifact_dir) / GROUND_TRUTH_FILENAME
    path.write_text(json.dumps({
        "_warning": (
            "Simulation ground truth for the synthetic demonstration dataset. "
            "Diagnostic use only. Never an input to feature engineering, "
            "training or scoring."
        ),
        "true_failure_probability": ground_truth,
    }, indent=2))
    return path


def load_ground_truth(root: Path | None = None) -> dict[str, float]:
    path = Path(root or settings.artifact_dir) / GROUND_TRUTH_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("true_failure_probability", {})


def oracle_benchmark(
    session: Session, *, cohort_version_id: str | None = None, root: Path | None = None
) -> dict[str, Any]:
    """Maximum attainable discrimination on the demonstration cohort."""
    from sklearn.metrics import roc_auc_score

    from trs.validation.metrics import brier_score, log_loss

    truth = load_ground_truth(root)
    if not truth:
        return {"available": False, "reason": "No ground-truth file; re-run seeding."}

    if cohort_version_id:
        cohort = session.get(CohortVersion, cohort_version_id)
    else:
        cohort = session.scalars(
            select(CohortVersion).order_by(CohortVersion.built_at.desc()).limit(1)
        ).first()
    if cohort is None:
        return {"available": False, "reason": "No cohort version."}

    entries = list(session.scalars(
        select(CohortEntry).where(
            CohortEntry.cohort_version_id == cohort.id,
            CohortEntry.label_primary.is_not(None),
        )
    ))
    pairs = [
        (truth[e.program_indication_id], e.label_primary)
        for e in entries
        if e.program_indication_id in truth
    ]
    if len(pairs) < 30:
        return {"available": False, "reason": f"Only {len(pairs)} matched tracks."}

    p = np.asarray([a for a, _ in pairs], dtype=float)
    y = np.asarray([b for _, b in pairs], dtype=int)
    if len(np.unique(y)) < 2:
        return {"available": False, "reason": "Single outcome class."}

    return {
        "available": True,
        "n": len(pairs),
        "oracle_auroc": float(roc_auc_score(y, p)),
        "oracle_brier": float(brier_score(y, p)),
        "oracle_log_loss": float(log_loss(y, p)),
        "base_rate_brier": float(brier_score(y, np.full_like(p, y.mean()))),
        "base_rate_log_loss": float(log_loss(y, np.full_like(p, y.mean()))),
        "mean_true_probability": float(p.mean()),
        "sd_true_probability": float(p.std()),
        "interpretation": (
            "The oracle scores each track by the probability the simulation "
            "actually used. No model can exceed it. The gap between a fitted "
            "model and this ceiling is the combined cost of measurement noise "
            "in the evidence, non-disclosure, finite sample size and model "
            "misspecification — not a defect of the pipeline."
        ),
    }
