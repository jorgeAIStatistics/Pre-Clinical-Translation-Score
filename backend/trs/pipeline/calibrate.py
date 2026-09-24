"""Cross-validated calibration on the training partition.

The problem with a dedicated calibration split
-----------------------------------------------
The textbook three-way split — train / calibrate / test — is clean but
expensive. On a cohort of a few hundred programs, carving out a calibration
slice large enough to fit a stable calibrator leaves too little for training,
and carving out a small one produces a calibrator fitted on ~100 points whose
slope estimate is dominated by noise.

It is worse than noisy when the split is *temporal*. A dedicated calibration
slice is a narrow window of years, so the calibrator learns that window's
idiosyncrasies and is then applied to a different window. During development of
this pipeline that produced a held-out calibration slope of 0.53 — badly
over-confident — even though recalibration had nominally been applied.

Cross-validated calibration
---------------------------
Instead: partition the *whole* training set into K grouped folds; for each fold,
fit the model on the other K−1 and predict the held-out fold; pool the K sets of
out-of-fold predictions and fit the calibrator on all of them. Then refit the
model on the entire training set and pair it with that calibrator.

Every prediction the calibrator sees is out-of-sample, so there is no leakage,
and the calibrator is fitted on the full training sample across the full
training time range rather than on one slice of it.

The cost, stated
----------------
Out-of-fold predictions come from models trained on (K−1)/K of the data, which
are marginally less sharp than the final model trained on all of it. The
calibrator therefore corrects for slightly more over-confidence than the final
model actually exhibits. The bias is small at K = 5 and shrinks as K grows; it
is the standard trade and is accepted here deliberately. The alternative — a
calibrator fitted on 110 points from a three-year window — was measurably worse.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from trs.modeling.base import RiskModel
from trs.modeling.calibration import Calibrator, select_calibrator
from trs.validation.splitters import grouped_kfold

#: Below this, a dedicated calibration split is too small to be trusted and
#: cross-validated calibration is used instead.
MIN_DEDICATED_CALIBRATION_N = 200

DEFAULT_CALIBRATION_FOLDS = 5


def cross_validated_calibration(
    model_factory: Callable[[], RiskModel],
    X: np.ndarray,
    y: np.ndarray,
    *,
    components: np.ndarray,
    columns: Sequence[str],
    groups: np.ndarray | None = None,
    n_folds: int = DEFAULT_CALIBRATION_FOLDS,
    random_state: int = 0,
) -> tuple[RiskModel, Calibrator, dict[str, Any]]:
    """Fit a model on all of ``X`` and a calibrator on pooled out-of-fold scores."""
    y = np.asarray(y, dtype=int)
    oof = np.full(y.shape, np.nan, dtype=float)
    n_used_folds = 0

    for tr, te in grouped_kfold(components, n_splits=n_folds, y=y, random_state=random_state):
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_factory()
        kwargs: dict[str, Any] = {"columns": list(columns)}
        if groups is not None and m.family in ("hierarchical_logistic", "gbdt"):
            kwargs["groups"] = groups[tr]
        m.fit(X[tr], y[tr], **kwargs)
        oof[te] = (
            m.predict_logit(X[te], groups=groups[te])
            if m.family == "hierarchical_logistic" and groups is not None
            else m.predict_logit(X[te])
        )
        n_used_folds += 1

    mask = ~np.isnan(oof)
    final = model_factory()
    kwargs = {"columns": list(columns)}
    if groups is not None and final.family in ("hierarchical_logistic", "gbdt"):
        kwargs["groups"] = groups
    final.fit(X, y, **kwargs)

    if mask.sum() < 40 or len(np.unique(y[mask])) < 2:
        # Not enough usable out-of-fold predictions; fall back to an in-sample
        # Platt fit and say so loudly. An in-sample calibrator is optimistic,
        # and the record makes that visible rather than implying otherwise.
        from trs.modeling.calibration import LogisticRecalibration

        cal = LogisticRecalibration().fit(
            final.predict_logit(X)
            if final.family != "hierarchical_logistic" or groups is None
            else final.predict_logit(X, groups=groups),
            y,
        )
        return final, cal, {
            "method": cal.method,
            "design": "in_sample_fallback",
            "reason": (
                f"Only {int(mask.sum())} usable out-of-fold predictions from "
                f"{n_used_folds} folds; the calibrator was fitted in-sample and "
                "is therefore optimistic."
            ),
            "n_calibration": int(y.size),
            "warning": "IN-SAMPLE CALIBRATION — held-out calibration metrics may be optimistic.",
        }

    cal, record = select_calibrator(oof[mask], y[mask], random_state=random_state)
    record.update({
        "design": "cross_validated_out_of_fold",
        "n_folds": n_used_folds,
        "n_out_of_fold_predictions": int(mask.sum()),
        "note": (
            "The calibrator was fitted on out-of-fold predictions pooled across "
            f"{n_used_folds} grouped folds of the training partition, then paired "
            "with a model refitted on the whole training partition. No test-"
            "partition information was used at any point."
        ),
        "known_bias": (
            "Out-of-fold predictions come from models trained on a fraction of "
            "the data and are marginally less sharp than the final model, so the "
            "calibrator corrects for slightly more over-confidence than the final "
            "model exhibits. Small at this number of folds; documented rather "
            "than corrected."
        ),
    })
    return final, cal, record
