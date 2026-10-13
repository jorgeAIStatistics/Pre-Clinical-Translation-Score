"""The development phase: everything fitted or chosen without the evaluation set.

This module exists to make one property structurally true rather than
aspirational: **no function in here receives the evaluation partition at all.**
Hyperparameter selection, family selection and calibration all consume
development rows and forward-chaining folds over them. The evaluation partition
is a separate argument to a separate function in
:mod:`trs.pipeline.train`, read once.

Fold-local pipelines
--------------------
The predecessor fitted one :class:`~trs.features.transform.FeatureTransformer`
on the whole outer-training partition and then handed the resulting matrix to
the tuner and the calibrator. Nothing leaked to the test set, but every
inner-fold number was optimistic: a fold's held-out rows had already
contributed their values to the imputation medians, the standardisation
constants and the categorical vocabularies that encoded them.

Here each fold fits its own transformer on its own permitted training rows.
This has a consequence worth stating because it looks like a bug the first time
it happens: **the encoded design matrix differs in width between folds.** The
transformer only emits one-hot columns for nominal levels it actually observed,
so a fold whose training window never saw ``modality = oligonucleotide`` has no
column for it. Models are fitted and scored entirely within a fold, so this is
correct; it only means fold coefficient vectors are not directly comparable,
which is why nothing here tries to average them.

Pooled out-of-fold predictions
------------------------------
Family selection and calibration both consume predictions pooled across the
forward-chaining folds. Every such prediction is made by a model that saw only
rows assessed before that fold's origin, so the pool is a set of genuinely
forward predictions — which is the property the previous random grouped k-fold
calibration did not have, despite being leakage-free in the grouping sense.

The pool does not cover every development row. Expanding-window folds test
disjoint windows after the first training block, so rows in that initial block
never appear in any test window and receive no out-of-fold prediction. The
count is reported rather than papered over.

One reuse is disclosed
----------------------
The L2 penalty is selected on the same folds later used for family selection.
Both are development-only, so the evaluation partition is untouched either way,
but the family comparison is mildly optimistic about the penalty being right for
each family. Nesting a second layer of forward-chaining folds inside training
windows of 62–150 rows would produce tuning sets of a dozen rows, which is
worse. The trade is recorded in the artifact as
``penalty_and_family_share_folds``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from trs.features.transform import FeatureTransformer
from trs.modeling import MODEL_FAMILIES
from trs.modeling.base import RiskModel
from trs.validation.metrics import log_loss
from trs.validation.protocol import Fold

#: L2 penalty grid, log-spaced over four orders of magnitude. The range is wide
#: on purpose: an early version stopped at 64 and the tuner pinned itself to
#: that endpoint. A grid whose argmin lands on an endpoint has not selected
#: anything, which :func:`tune_l2_on_folds` detects and records.
DEFAULT_L2_GRID: tuple[float, ...] = (
    0.25, 1.0, 4.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
)

#: Group-effect precisions for the hierarchical family. Larger = more pooling.
DEFAULT_GROUP_PRECISION_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


@dataclasses.dataclass(frozen=True, slots=True)
class FoldDesign:
    """One fold's own encoding of its own rows."""

    transformer: FeatureTransformer
    X_train: np.ndarray
    X_test: np.ndarray
    columns: list[str]


def fold_design(
    rows: Sequence[dict[str, Any]],
    fold: Fold,
    *,
    random_state: int = 0,
) -> FoldDesign:
    """Fit a transformer on this fold's training rows and encode both sides.

    The transformer never sees ``fold.test``. That is the whole point of the
    function existing separately from the caller.
    """
    train_rows = [dict(rows[i]) for i in fold.train]
    test_rows = [dict(rows[i]) for i in fold.test]
    tf = FeatureTransformer().fit(train_rows, random_state=random_state)
    return FoldDesign(
        transformer=tf,
        X_train=tf.transform(train_rows),
        X_test=tf.transform(test_rows),
        columns=list(tf.columns),
    )


def _fit(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    groups: np.ndarray | None,
    hyperparameters: dict[str, Any],
) -> RiskModel:
    model = MODEL_FAMILIES[family](**hyperparameters)
    kwargs: dict[str, Any] = {"columns": columns}
    if family in ("hierarchical_logistic", "gbdt"):
        kwargs["groups"] = groups
    return model.fit(X, y, **kwargs)


def _logit(model: RiskModel, X: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    if model.family == "hierarchical_logistic" and groups is not None:
        return np.asarray(model.predict_logit(X, groups=groups))
    return np.asarray(model.predict_logit(X))


def default_hyperparameters(family: str, l2: float, seed: int) -> dict[str, Any]:
    match family:
        case "l2_logistic":
            return {"l2_penalty": l2}
        case "hierarchical_logistic":
            return {"l2_penalty": l2, "group_precision": 4.0}
        case "gbdt":
            return {
                "learning_rate": 0.05, "max_iter": 150, "max_depth": 3,
                "min_samples_leaf": 15, "l2_regularization": 1.0,
                "ensemble_size": 40, "random_state": seed,
            }
        case _:
            return {}


# ---------------------------------------------------------------------------
# Hyperparameter selection
# ---------------------------------------------------------------------------


def tune_l2_on_folds(
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    grid: Sequence[float] = DEFAULT_L2_GRID,
    random_state: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the ridge penalty by pooled forward-chaining log loss.

    Each fold encodes its own rows, so the penalty is selected under the same
    fold-local preprocessing the family comparison will use. Folds in which
    either side is single-class are skipped and counted — a fold with no
    positives makes log loss meaningless, and silently averaging over it would
    bias the choice toward whichever penalty happened to predict the constant.
    """
    designs = [fold_design(rows, f, random_state=random_state) for f in folds]
    trace: list[dict[str, Any]] = []

    for lam in grid:
        losses: list[float] = []
        skipped = 0
        for f, d in zip(folds, designs, strict=True):
            if len(np.unique(y[f.train])) < 2 or len(np.unique(y[f.test])) < 2:
                skipped += 1
                continue
            m = _fit("l2_logistic", d.X_train, y[f.train], d.columns, None,
                     {"l2_penalty": float(lam)})
            losses.append(log_loss(y[f.test], m.predict_proba(d.X_test)))
        trace.append({
            "l2_penalty": float(lam),
            "mean_fold_log_loss": float(np.mean(losses)) if losses else None,
            "n_folds_used": len(losses),
            "n_folds_skipped_single_class": skipped,
        })

    usable = [t for t in trace if t["mean_fold_log_loss"] is not None]
    if not usable:
        raise RuntimeError(
            "no L2 penalty could be evaluated: every forward-chaining fold was "
            "single-class on one side. The development partition cannot tune a "
            "hyperparameter."
        )
    best = min(usable, key=lambda t: t["mean_fold_log_loss"])
    best_l2 = float(best["l2_penalty"])

    at_boundary = best_l2 in (float(grid[0]), float(grid[-1]))
    summary: dict[str, Any] = {
        "selected_l2_penalty": best_l2,
        "selection_log_loss": best["mean_fold_log_loss"],
        "grid": [float(g) for g in grid],
        "n_folds": len(folds),
        "optimum_at_grid_boundary": at_boundary,
        "preprocessing": "fold_local",
    }
    if at_boundary:
        summary["warning"] = (
            f"The selected penalty {best_l2} sits at an endpoint of the search "
            "grid, so the grid rather than the data chose it. The true optimum "
            "may lie outside the range."
        )
    trace.append(summary)
    return best_l2, trace


def tune_group_precision_on_folds(
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    folds: Sequence[Fold],
    groups: np.ndarray,
    l2: float,
    *,
    grid: Sequence[float] = DEFAULT_GROUP_PRECISION_GRID,
    random_state: int = 0,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the hierarchical model's pooling strength on the same folds."""
    designs = [fold_design(rows, f, random_state=random_state) for f in folds]
    trace: list[dict[str, Any]] = []
    for gp in grid:
        losses: list[float] = []
        for f, d in zip(folds, designs, strict=True):
            if len(np.unique(y[f.train])) < 2 or len(np.unique(y[f.test])) < 2:
                continue
            m = _fit("hierarchical_logistic", d.X_train, y[f.train], d.columns,
                     groups[f.train],
                     {"l2_penalty": l2, "group_precision": float(gp)})
            p = 1.0 / (1.0 + np.exp(-_logit(m, d.X_test, groups[f.test])))
            losses.append(log_loss(y[f.test], p))
        trace.append({
            "group_precision": float(gp),
            "mean_fold_log_loss": float(np.mean(losses)) if losses else None,
            "n_folds_used": len(losses),
        })
    usable = [t for t in trace if t["mean_fold_log_loss"] is not None]
    if not usable:
        return 4.0, trace
    best = min(usable, key=lambda t: t["mean_fold_log_loss"])
    return float(best["group_precision"]), trace


# ---------------------------------------------------------------------------
# Out-of-fold predictions per family
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OutOfFoldPredictions:
    """Pooled forward-chaining predictions for one family.

    ``index`` holds original-space row indices, so a caller can line these up
    with labels, components and dates without tracking fold membership.
    """

    family: str
    index: np.ndarray
    logit: np.ndarray
    per_fold: tuple[dict[str, Any], ...]
    hyperparameters: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.index.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_out_of_fold": self.n,
            "per_fold": [dict(f) for f in self.per_fold],
            "hyperparameters": dict(self.hyperparameters),
        }


def out_of_fold_predictions(
    family: str,
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    folds: Sequence[Fold],
    groups: np.ndarray,
    hyperparameters: dict[str, Any],
    *,
    random_state: int = 0,
) -> OutOfFoldPredictions:
    """Fit ``family`` once per fold and collect its held-out logits.

    Each fold's model is fitted on that fold's own encoding of its own training
    rows. Nothing is averaged across folds except the predictions, which refer
    to disjoint rows.
    """
    idx: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    per_fold: list[dict[str, Any]] = []

    for f in folds:
        if len(np.unique(y[f.train])) < 2:
            per_fold.append({
                "fold": f.index, "origin": f.origin.isoformat(),
                "skipped": "training window is single-class",
            })
            continue
        d = fold_design(rows, f, random_state=random_state)
        model = _fit(family, d.X_train, y[f.train], d.columns, groups[f.train],
                     hyperparameters)
        z = _logit(model, d.X_test, groups[f.test])
        idx.append(f.test)
        logits.append(np.asarray(z, dtype=float).ravel())
        per_fold.append({
            "fold": f.index,
            "origin": f.origin.isoformat(),
            "n_train": int(f.train.size),
            "n_test": int(f.test.size),
            "n_encoded_columns": len(d.columns),
        })

    if not idx:
        raise RuntimeError(
            f"family {family!r} produced no out-of-fold predictions: every "
            "forward-chaining fold had a single-class training window."
        )

    order = np.argsort(np.concatenate(idx), kind="stable")
    return OutOfFoldPredictions(
        family=family,
        index=np.concatenate(idx)[order],
        logit=np.concatenate(logits)[order],
        per_fold=tuple(per_fold),
        hyperparameters=dict(hyperparameters),
    )


def grouped_oof_predictions(
    family: str,
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    development: np.ndarray,
    components: np.ndarray,
    groups: np.ndarray,
    hyperparameters: dict[str, Any],
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> OutOfFoldPredictions:
    """Out-of-fold predictions covering **every** development row.

    Why this exists alongside :func:`out_of_fold_predictions`
    ---------------------------------------------------------
    Expanding-window folds only ever test rows *after* the first training
    block, so their pooled predictions are systematically drawn from the later
    part of the development window. Measured on the demonstration cohort: the
    forward-chaining pool's observed failure rate is 0.664 while the
    development partition's is 0.631, because the cohort's base rate rises
    through 2013–2017 and the pool over-weights those years.

    A calibrator fitted on that pool inherits the pool's level, not the
    development partition's. That is a silent bias in the *intercept* of the
    recalibration map, and it was worth 0.19 nats of evaluation log loss here.

    Grouped k-fold over all development rows gives a pool whose base rate is
    the development base rate by construction. The cost, stated plainly: a
    grouped out-of-fold prediction is **not** a forward-in-time prediction. A
    fold's training rows may be assessed later than its test rows, so this pool
    cannot support a claim about forward calibration drift. It is used for
    fitting the recalibration map, where full coverage matters more; the
    forward-chaining pool is retained for selection and both are audited.
    """
    from trs.validation.splitters import grouped_kfold

    development = np.asarray(development, dtype=int)
    idx: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    per_fold: list[dict[str, Any]] = []

    for k, (tr_local, te_local) in enumerate(
        grouped_kfold(
            components[development], n_splits=n_splits,
            y=y[development], random_state=random_state,
        )
    ):
        tr = development[tr_local]
        te = development[te_local]
        if len(np.unique(y[tr])) < 2:
            per_fold.append({"fold": k, "skipped": "training fold is single-class"})
            continue
        train_rows = [dict(rows[i]) for i in tr]
        tf = FeatureTransformer().fit(train_rows, random_state=random_state)
        model = _fit(
            family, tf.transform(train_rows), y[tr], list(tf.columns),
            groups[tr], hyperparameters,
        )
        z = _logit(model, tf.transform([dict(rows[i]) for i in te]), groups[te])
        idx.append(te)
        logits.append(np.asarray(z, dtype=float).ravel())
        per_fold.append({
            "fold": k, "n_train": int(tr.size), "n_test": int(te.size),
            "n_encoded_columns": len(tf.columns),
        })

    if not idx:
        raise RuntimeError(
            f"family {family!r} produced no grouped out-of-fold predictions."
        )
    order = np.argsort(np.concatenate(idx), kind="stable")
    return OutOfFoldPredictions(
        family=family,
        index=np.concatenate(idx)[order],
        logit=np.concatenate(logits)[order],
        per_fold=tuple(per_fold),
        hyperparameters=dict(hyperparameters),
    )


def calibration_scale_audit(
    pool_logit: np.ndarray,
    pool_y: np.ndarray,
    final_development_logit: np.ndarray,
    development_y: np.ndarray,
    *,
    logit_shift_tolerance: float = 0.15,
    rate_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Check that a calibrator is being fitted on the scale it will be applied to.

    Cross-validated calibration pairs a calibrator fitted on fold models'
    predictions with a final model fitted on more data. The standard framing
    treats the resulting bias as small and accepts it. It is not always small,
    and nothing in the predecessor measured it.

    Two mismatches are checked:

    **Logit scale.** Fold models see fewer rows and are shrunk harder, so their
    predictions are flatter. If the final model's mean development logit sits
    well above the pool's, the calibrator's intercept — fitted to lift flat
    logits onto the pool's base rate — over-corrects every served prediction.

    **Pool level.** If the pool's observed outcome rate differs from the
    development partition's, the calibrator has learned the wrong level
    outright.

    Returns the measurements and a verdict. A failing verdict becomes a
    recorded limitation, not a silent correction.
    """
    pool_logit = np.asarray(pool_logit, dtype=float).ravel()
    final_development_logit = np.asarray(final_development_logit, dtype=float).ravel()
    pool_rate = float(np.mean(pool_y))
    dev_rate = float(np.mean(development_y))
    shift = float(np.mean(final_development_logit) - np.mean(pool_logit))
    rate_gap = float(pool_rate - dev_rate)

    problems: list[str] = []
    if abs(shift) > logit_shift_tolerance:
        problems.append(
            f"the final model's mean development logit ({np.mean(final_development_logit):+.3f}) "
            f"differs from the calibration pool's ({np.mean(pool_logit):+.3f}) by "
            f"{shift:+.3f}, so the recalibration intercept was fitted on a "
            "different scale from the one it is applied to"
        )
    if abs(rate_gap) > rate_tolerance:
        problems.append(
            f"the calibration pool's observed rate ({pool_rate:.3f}) differs "
            f"from the development partition's ({dev_rate:.3f}) by "
            f"{rate_gap:+.3f}, so the calibrator learned a level that is not "
            "the development level"
        )

    return {
        "pool_n": int(pool_logit.size),
        "pool_observed_rate": pool_rate,
        "development_observed_rate": dev_rate,
        "pool_rate_minus_development_rate": rate_gap,
        "pool_mean_logit": float(np.mean(pool_logit)),
        "pool_sd_logit": float(np.std(pool_logit)),
        "final_mean_development_logit": float(np.mean(final_development_logit)),
        "final_sd_development_logit": float(np.std(final_development_logit)),
        "mean_logit_shift": shift,
        "tolerances": {
            "logit_shift": logit_shift_tolerance, "rate": rate_tolerance,
        },
        "verdict": "scale_mismatch" if problems else "consistent",
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Final refit on the whole development partition
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DevelopmentFit:
    """The model that will face the evaluation partition, and how it was made."""

    family: str
    transformer: FeatureTransformer
    model: RiskModel
    columns: list[str]
    hyperparameters: dict[str, Any]
    n_development_rows: int


def fit_on_development(
    family: str,
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    development: np.ndarray,
    groups: np.ndarray,
    hyperparameters: dict[str, Any],
    *,
    random_state: int = 0,
) -> DevelopmentFit:
    """Refit on every development row, with a transformer fitted on those rows.

    This is the only fit whose transformer spans the whole development
    partition, and it is correct for it to: the partition it will be applied to
    — evaluation — contributed nothing to it.
    """
    dev_rows = [dict(rows[i]) for i in development]
    tf = FeatureTransformer().fit(dev_rows, random_state=random_state)
    X = tf.transform(dev_rows)
    model = _fit(family, X, y[development], list(tf.columns),
                 groups[development], hyperparameters)
    return DevelopmentFit(
        family=family,
        transformer=tf,
        model=model,
        columns=list(tf.columns),
        hyperparameters=dict(hyperparameters),
        n_development_rows=int(development.size),
    )


def apply_fit(
    fit: DevelopmentFit,
    rows: Sequence[dict[str, Any]],
    index: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Uncalibrated logits for ``index`` under a development fit."""
    X = fit.transformer.transform([dict(rows[i]) for i in index])
    return _logit(fit.model, X, groups[index])


def model_factory(family: str, hyperparameters: dict[str, Any]) -> Callable[[], RiskModel]:
    """A zero-argument constructor, for helpers that refit repeatedly."""
    return lambda: MODEL_FAMILIES[family](**hyperparameters)


# ---------------------------------------------------------------------------
# Honest calibrated development predictions
# ---------------------------------------------------------------------------


def cross_fitted_calibrated(
    oof_logit: np.ndarray,
    y_oof: np.ndarray,
    components_oof: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrated development predictions whose calibrator never saw them.

    Returns ``(probabilities, positions, diagnostics)`` where ``positions`` are
    offsets *into the out-of-fold pool* (not original row indices) for the rows
    that received a cross-fitted prediction.

    Why this is needed, and what goes wrong without it
    --------------------------------------------------
    The deployed calibrator must be fitted on the whole out-of-fold pool — that
    is the point of pooling. But if the *selection* metrics are then computed by
    applying that calibrator to the same pool, the calibration slope is 1.0 by
    construction, for every candidate family, and the selection gate's
    calibration band silently stops doing anything.

    That is exactly what happened on the first run of the rewritten pipeline:
    all three families reported a development calibration slope of 1.0 to
    sixteen significant figures. The number was not wrong so much as vacuous —
    it measured the recalibration's own arithmetic rather than the model.

    So: partition the out-of-fold pool into grouped folds, fit a calibrator on
    each fold's complement, apply it to the fold, and pool. Every calibrated
    prediction returned here comes from a calibrator that did not see that row.
    The deployed calibrator is still fitted on everything; these predictions
    exist to *measure*, not to serve.
    """
    from trs.modeling.calibration import select_calibrator
    from trs.validation.splitters import grouped_kfold

    oof_logit = np.asarray(oof_logit, dtype=float).ravel()
    y_oof = np.asarray(y_oof, dtype=int).ravel()
    out = np.full(oof_logit.shape, np.nan, dtype=float)
    used = np.zeros(oof_logit.shape, dtype=bool)
    methods: list[str] = []

    for fit_idx, apply_idx in grouped_kfold(
        components_oof, n_splits=n_splits, y=y_oof, random_state=random_state
    ):
        if len(np.unique(y_oof[fit_idx])) < 2:
            continue
        cal, record = select_calibrator(
            oof_logit[fit_idx], y_oof[fit_idx], random_state=random_state
        )
        out[apply_idx] = cal.transform(oof_logit[apply_idx])
        used[apply_idx] = True
        methods.append(str(record.get("method", "unknown")))

    diagnostics = {
        "design": "cross_fitted_calibration_over_out_of_fold_pool",
        "n_splits": n_splits,
        "n_calibrated": int(used.sum()),
        "n_uncalibrated": int((~used).sum()),
        "methods_per_fold": methods,
        "why": (
            "Selection metrics must not be computed with a calibrator fitted on "
            "the rows being scored; doing so forces the calibration slope to "
            "1.0 for every candidate and disables the selection gate's "
            "calibration band."
        ),
    }
    if not used.any():
        raise RuntimeError(
            "cross-fitted calibration produced no usable predictions: every "
            "fold of the out-of-fold pool was single-class."
        )
    return out[used], np.flatnonzero(used), diagnostics
