"""Performance, calibration and clinical-utility metrics.

Discrimination is reported because it is expected, but it is not the metric
this product lives or dies by. A translational risk score is consumed as a
*probability* — a committee comparing 35% against 70% is reading the numbers,
not the ranking — so calibration and net benefit carry more weight than AUROC.
The report is ordered accordingly, and no single number is presented as
sufficient.

Every interval here is a **component-level** bootstrap: replicates resample
connected components (see :mod:`trs.validation.splitters`), not rows. Rows
within a component are not independent, and resampling them as if they were
produces intervals that are too narrow by roughly the square root of the mean
component size.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from trs.modeling.base import logit, sigmoid
from trs.validation.splitters import component_bootstrap_indices

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Point metrics
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def auprc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def scaled_brier(y: np.ndarray, p: np.ndarray) -> float | None:
    """Brier skill score against the no-covariate base-rate model.

    ``1 - BS/BS_base``. Positive means the model beats simply quoting the
    cohort's attrition rate to everyone — a low bar that a surprising number of
    risk scores fail, and one AUROC cannot detect.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return None
    base = float(np.mean(y))
    bs_base = float(np.mean((base - y) ** 2))
    if bs_base <= 0:
        return None
    return float(1.0 - brier_score(y, p) / bs_base)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    """Cox calibration: regress outcomes on the predicted log-odds.

    ``slope = 1`` and ``intercept = 0`` is perfect. ``slope < 1`` means
    predictions are too extreme (over-confident); ``intercept > 0`` means they
    are systematically too low.

    The intercept is estimated with the slope *fixed at 1* (calibration-in-the-
    large), which is the standard definition; estimating both jointly conflates
    two distinct failure modes.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(logit(np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)), dtype=float)
    if len(np.unique(y)) < 2 or z.size < 5:
        return {"intercept": None, "slope": None, "intercept_se": None, "slope_se": None}

    slope, slope_se = _fit_logistic_1d(z, y, with_intercept=True)
    intercept, intercept_se = _fit_offset_intercept(z, y)
    return {
        "intercept": intercept,
        "slope": slope,
        "intercept_se": intercept_se,
        "slope_se": slope_se,
    }


def _fit_logistic_1d(z: np.ndarray, y: np.ndarray, *, with_intercept: bool) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(z), z]) if with_intercept else z.reshape(-1, 1)
    params = np.zeros(X.shape[1])
    if with_intercept:
        params[1] = 1.0
    for _ in range(100):
        mu = sigmoid(X @ params)
        W = np.clip(mu * (1 - mu), 1e-10, None)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, X.T @ (mu - y))
        except np.linalg.LinAlgError:
            break
        params = params - step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = sigmoid(X @ params)
    W = np.clip(mu * (1 - mu), 1e-10, None)
    cov = np.linalg.pinv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]))
    idx = -1
    return float(params[idx]), float(np.sqrt(max(cov[idx, idx], 0.0)))


def _fit_offset_intercept(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Intercept-only logistic fit with the predicted log-odds as an offset."""
    a = 0.0
    for _ in range(100):
        mu = sigmoid(a + z)
        W = np.clip(mu * (1 - mu), 1e-10, None)
        grad = float(np.sum(mu - y))
        hess = float(np.sum(W)) + 1e-8
        step = grad / hess
        a -= step
        if abs(step) < 1e-12:
            break
    mu = sigmoid(a + z)
    W = np.clip(mu * (1 - mu), 1e-10, None)
    return float(a), float(np.sqrt(1.0 / max(float(np.sum(W)), 1e-8)))


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, *, strategy: str = "quantile"
) -> dict[str, Any]:
    """Binned calibration error plus the curve the UI plots.

    Quantile binning by default: equal-width bins leave most bins nearly empty
    when predictions concentrate, producing a calibration plot dominated by
    noise in regions with three observations.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = y.size
    if n == 0:
        return {"ece": None, "mce": None, "bins": []}

    n_bins = max(2, min(n_bins, n // 5 if n >= 10 else 2))
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))
        # Wilson interval: behaves sensibly at 0/1, unlike the normal
        # approximation, and calibration bins frequently sit at 0 or 1.
        lo, hi = _wilson_interval(int(y[mask].sum()), count)
        gap = abs(mean_pred - observed)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append({
            "bin_index": b,
            "lower": float(edges[b]),
            "upper": float(edges[b + 1]),
            "n": count,
            "mean_predicted": mean_pred,
            "observed_rate": observed,
            "observed_ci_low": lo,
            "observed_ci_high": hi,
        })
    return {"ece": float(ece), "mce": float(mce), "bins": bins, "strategy": strategy}


def _wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p, dtype=float) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    def _safe(num: int, den: int) -> float | None:
        return float(num / den) if den else None

    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "ppv": _safe(tp, tp + fp),
        "npv": _safe(tn, tn + fn),
        "flagged_fraction": _safe(tp + fp, len(y)),
    }


def decision_curve(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray | None = None
) -> dict[str, Any]:
    """Net benefit across decision thresholds.

    The right question for a portfolio committee is not "is the AUROC good" but
    "if I deprioritise every program above X% predicted risk, am I better off
    than deprioritising all of them or none of them?". Net benefit answers
    exactly that, with the threshold encoding the committee's own exchange rate
    between a wrongly killed good program and a wrongly funded bad one:

        NB = TP/n − (FP/n) · (pt / (1 − pt))

    ``pt`` is the threshold probability. A model is useful over the range where
    its curve sits above both reference strategies.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = y.size
    if n == 0:
        return {"thresholds": [], "model": [], "treat_all": [], "treat_none": []}

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    prevalence = float(np.mean(y))
    model_nb, all_nb, none_nb = [], [], []
    for pt in thresholds:
        odds = pt / (1 - pt)
        pred = p >= pt
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        model_nb.append(float(tp / n - (fp / n) * odds))
        all_nb.append(float(prevalence - (1 - prevalence) * odds))
        none_nb.append(0.0)

    return {
        "thresholds": [float(t) for t in thresholds],
        "model": model_nb,
        "treat_all": all_nb,
        "treat_none": none_nb,
        "prevalence": prevalence,
        "interpretation": (
            "Net benefit of acting on the model's flag, relative to flagging "
            "every program (treat_all) or none (treat_none). The threshold is "
            "the risk level at which a decision-maker becomes indifferent."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Assembly with bootstrap intervals
# ─────────────────────────────────────────────────────────────────────────────

METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float | None]] = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier_score,
    "log_loss": log_loss,
    "scaled_brier": scaled_brier,
}


@dataclasses.dataclass(slots=True)
class MetricEstimate:
    value: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    n_bootstrap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_bootstrap": self.n_bootstrap,
        }


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    *,
    components: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 0,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    level: float = 0.95,
) -> dict[str, Any]:
    """Full metric panel with component-level bootstrap intervals."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = y.size

    point = {name: fn(y, p) for name, fn in METRIC_FUNCTIONS.items()}
    cal = calibration_intercept_slope(y, p)

    estimates: dict[str, MetricEstimate] = {
        name: MetricEstimate(value=v) for name, v in point.items()
    }
    cal_estimates = {
        "calibration_intercept": MetricEstimate(value=cal["intercept"]),
        "calibration_slope": MetricEstimate(value=cal["slope"]),
    }

    if n_bootstrap and components is not None and n >= 20:
        rng = np.random.default_rng(random_state)
        draws: dict[str, list[float]] = {k: [] for k in list(point) + list(cal_estimates)}
        for _ in range(n_bootstrap):
            idx = component_bootstrap_indices(components, rng)
            if idx.size < 10 or len(np.unique(y[idx])) < 2:
                continue
            yb, pb = y[idx], p[idx]
            for name, fn in METRIC_FUNCTIONS.items():
                v = fn(yb, pb)
                if v is not None and np.isfinite(v):
                    draws[name].append(float(v))
            cb = calibration_intercept_slope(yb, pb)
            for key, src in (
                ("calibration_intercept", "intercept"), ("calibration_slope", "slope"),
            ):
                v = cb[src]
                if v is not None and np.isfinite(v):
                    draws[key].append(float(v))

        alpha = (1 - level) / 2
        for name, vals in draws.items():
            if len(vals) < 20:
                continue
            lo, hi = np.quantile(vals, [alpha, 1 - alpha])
            target = estimates if name in estimates else cal_estimates
            target[name] = MetricEstimate(
                value=target[name].value, ci_low=float(lo), ci_high=float(hi),
                n_bootstrap=len(vals),
            )

    return {
        "n": int(n),
        "n_positive": int(y.sum()),
        "prevalence": float(np.mean(y)) if n else None,
        "discrimination": {k: estimates[k].to_dict() for k in ("auroc", "auprc")},
        "accuracy": {
            k: estimates[k].to_dict() for k in ("brier", "log_loss", "scaled_brier")
        },
        "calibration": {
            **{k: v.to_dict() for k, v in cal_estimates.items()},
            **expected_calibration_error(y, p),
        },
        "thresholds": [threshold_metrics(y, p, t) for t in thresholds],
        "decision_curve": decision_curve(y, p),
        "bootstrap": {
            "n_requested": n_bootstrap,
            "unit": "connected component" if components is not None else "none",
            "level": level,
        },
    }


def paired_bootstrap_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    components: np.ndarray,
    *,
    metric: str = "log_loss",
    n_bootstrap: int = 1000,
    random_state: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Paired component bootstrap of ``metric(A) − metric(B)``.

    Paired because both models see the same programs: comparing two independent
    intervals would ignore that shared variation and badly overstate how
    uncertain the *difference* is. This is the test :mod:`trs.pipeline.select`
    uses to decide whether a complex model has actually earned deployment.
    """
    fn = METRIC_FUNCTIONS[metric]
    y = np.asarray(y, dtype=int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    observed_a, observed_b = fn(y, p_a), fn(y, p_b)
    observed = (
        None if observed_a is None or observed_b is None else float(observed_a - observed_b)
    )

    rng = np.random.default_rng(random_state)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = component_bootstrap_indices(components, rng)
        if idx.size < 10 or len(np.unique(y[idx])) < 2:
            continue
        va, vb = fn(y[idx], p_a[idx]), fn(y[idx], p_b[idx])
        if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
            diffs.append(float(va - vb))

    if len(diffs) < 20:
        return {
            "metric": metric, "observed_difference": observed,
            "ci_low": None, "ci_high": None, "n_bootstrap": len(diffs),
            "excludes_zero": None,
        }

    alpha = (1 - level) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return {
        "metric": metric,
        "observed_difference": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap": len(diffs),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "level": level,
    }
