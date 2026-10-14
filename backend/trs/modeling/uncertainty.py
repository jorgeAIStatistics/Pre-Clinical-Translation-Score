"""Composing an honest prediction interval.

Three sources of uncertainty are represented, and a fourth is measured
separately when it can be:

``parameter``
    Uncertainty about the model's coefficients given the training data. Drawn
    from the Laplace posterior for the linear families, or from a grouped
    bootstrap ensemble for the tree family — which is a different and narrower
    quantity, and is labelled as such.

``imputation``
    Uncertainty from features this program simply does not have. Each of ``M``
    imputations fills the gaps with an independent draw from the training
    distribution, so the spread across imputations is exactly the cost of not
    knowing. A program with a complete evidence package contributes zero here;
    a sparse one contributes a lot. This is what makes the interval widen for
    the right reasons.

``calibration``
    Uncertainty in the mapping from model score to probability. Propagated for
    logistic recalibration; excluded — and reported as excluded — for isotonic.

``specification`` (optional)
    Disagreement between candidate model families scored on the same row. This
    is not part of the interval: it is a separate diagnostic, because averaging
    over model families would imply a prior over them that nobody has stated.

Decomposition method
--------------------
Components are reported by **ablation on the log-odds scale**: each source is
varied alone, with the other two held at their point estimates, and the
standard deviation of the resulting log-odds is recorded.

The log-odds scale is used rather than the probability scale for a specific
reason. The logistic link compresses probability differences near 0 and 1, so
the same amount of underlying ignorance produces a much narrower
probability-scale band for a program estimated at 5% than for one estimated at
50%. Ablation widths measured in probability units are therefore not comparable
between programs, and not even comparable between components for the *same*
program when the ablated point estimates sit at different places on the curve.
Log-odds standard deviations are comparable in both directions.

``variance_share`` normalises the squared component standard deviations to sum
to one. It is an approximation: the sources interact, so the components do not
decompose the total variance exactly. The total is therefore reported
separately and is never reconstructed from the shares.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from trs.features.transform import FeatureTransformer
from trs.modeling.base import RiskModel, UncertaintyUnsupported, sigmoid
from trs.modeling.calibration import Calibrator
from trs.modeling.contract import MATERIAL_FUNCTIONAL_GAP

#: Defaults chosen so a single score takes well under a second while the
#: Monte-Carlo error on an interval endpoint stays below ~0.5 percentage points.
DEFAULT_N_IMPUTATIONS = 25
DEFAULT_N_PARAM_SAMPLES = 400


@dataclasses.dataclass(slots=True)
class UncertaintyResult:
    #: **The served and evaluated quantity**: the calibrated plug-in
    #: probability at median-imputed features and posterior-mean parameters.
    #: See :mod:`trs.modeling.contract` for why this functional is primary.
    probability: float
    #: The Monte-Carlo mean of the calibrated probability over imputation,
    #: coefficient and calibration-parameter draws. Reported for comparison and
    #: never substituted for ``probability``: it differs systematically rather
    #: than noisily (Jensen), most for the least-evidenced programmes, and it
    #: has no exact additive decomposition.
    predictive_mean_probability: float
    interval_low: float
    interval_high: float
    interval_level: float
    #: Ablation components as log-odds standard deviations (comparable units).
    components_sd_logit: dict[str, float]
    #: Approximate share of total variance per source; sums to 1 by
    #: construction, not by decomposition. See the module docstring.
    variance_share: dict[str, float]
    #: Standard deviation of the calibrated log-odds across all draws.
    total_sd_logit: float
    n_imputations: int
    n_param_samples: int
    #: Honest statements about what the interval does and does not include.
    caveats: list[str]
    method: str

    @property
    def interval_width(self) -> float:
        return self.interval_high - self.interval_low

    @property
    def dominant_source(self) -> str:
        return max(self.components_sd_logit, key=self.components_sd_logit.__getitem__)

    @property
    def functional_gap(self) -> float:
        """``predictive_mean − probability``. Arithmetic, not biology."""
        return self.predictive_mean_probability - self.probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "predictive_mean_probability": self.predictive_mean_probability,
            "functional_gap": self.functional_gap,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "interval_level": self.interval_level,
            "interval_width": self.interval_width,
            "components_sd_logit": self.components_sd_logit,
            "variance_share": self.variance_share,
            "dominant_source": self.dominant_source,
            "total_sd_logit": self.total_sd_logit,
            "n_imputations": self.n_imputations,
            "n_param_samples": self.n_param_samples,
            "caveats": self.caveats,
            "method": self.method,
        }


def _quantile_interval(probs: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(probs, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _sd(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def predict_with_uncertainty(
    model: RiskModel,
    calibrator: Calibrator,
    transformer: FeatureTransformer,
    feature_row: dict[str, Any],
    *,
    rng: np.random.Generator,
    n_imputations: int = DEFAULT_N_IMPUTATIONS,
    n_param_samples: int = DEFAULT_N_PARAM_SAMPLES,
    level: float = 0.95,
    groups: str | None = None,
) -> UncertaintyResult:
    """Score one program with a decomposed, defensible interval."""
    caveats: list[str] = []

    # ---- 1. Imputation draws ----------------------------------------------
    X_imp = transformer.transform_with_imputations(feature_row, n_imputations, rng)
    X_point = transformer.transform([feature_row])          # median-filled
    M = X_imp.shape[0]

    # ---- 2. Parameter draws ------------------------------------------------
    supports_posterior = getattr(model, "supports_analytic_posterior", False)
    try:
        kwargs = {"groups": np.repeat(groups, M)} if groups is not None and _accepts_groups(model) else {}
        raw_draws = model.posterior_logit_samples(X_imp, n_param_samples, rng, **kwargs)
    except UncertaintyUnsupported:
        raise
    if not supports_posterior:
        caveats.append(
            "Parameter uncertainty is estimated from a bootstrap ensemble rather "
            "than a posterior distribution; it is likely to under-cover."
        )

    # ---- 3. Calibration draws ---------------------------------------------
    if calibrator.supports_parameter_uncertainty:
        cal_draws = calibrator.apply_with_parameter_draws(raw_draws, rng)
    else:
        cal_draws = calibrator.transform_logit(raw_draws)
        caveats.append(
            f"The selected calibration method ({calibrator.method}) has no "
            "parameter covariance, so calibration uncertainty is not included "
            "in this interval."
        )

    probs = sigmoid(cal_draws).ravel()
    predictive_mean = float(np.mean(probs))
    lo, hi = _quantile_interval(probs, level)
    total_sd = float(np.std(cal_draws.ravel()))

    # The *served* quantity: the calibrated plug-in probability. Computed from
    # the point design matrix and the point calibration, with no draws, so it
    # is reproducible without a seed and is exactly the functional the model's
    # reported metrics were computed on.
    #
    # The predecessor returned `np.mean(probs)` here and called it
    # `probability`, while training, validation and the driver attribution all
    # used the plug-in value. Three quantities, one name.
    plug_in_logit = _point_logit(model, X_point, groups)
    plug_in = float(sigmoid(calibrator.transform_logit(plug_in_logit)).ravel()[0])

    # ---- 4. Ablation components (log-odds standard deviations) -------------
    point_kwargs = (
        {"groups": np.asarray([groups])} if groups is not None and _accepts_groups(model) else {}
    )

    # (a) parameter only: median-filled features, point calibration
    raw_param_only = model.posterior_logit_samples(
        X_point, n_param_samples, rng, **point_kwargs
    )
    sd_param = _sd(calibrator.transform_logit(raw_param_only))

    # (b) imputation only: posterior-mean parameters, point calibration
    logit_per_imputation = _point_logit(model, X_imp, groups)
    sd_imp = _sd(calibrator.transform_logit(logit_per_imputation))

    # (c) calibration only: point features and point parameters
    logit_point = _point_logit(model, X_point, groups)
    if calibrator.supports_parameter_uncertainty:
        block = np.repeat(np.atleast_2d(logit_point), n_param_samples, axis=0)
        sd_cal = _sd(calibrator.apply_with_parameter_draws(block, rng))
    else:
        sd_cal = 0.0

    components = {
        "parameter": float(sd_param),
        "imputation": float(sd_imp),
        "calibration": float(sd_cal),
    }
    var_total = sum(v * v for v in components.values())
    variance_share = (
        {k: (v * v) / var_total for k, v in components.items()}
        if var_total > 0
        else {k: 0.0 for k in components}
    )

    n_missing = sum(
        1 for name in transformer.state.numeric_features
        if feature_row.get(name) is None
    )
    if n_missing:
        caveats.append(
            f"{n_missing} of {len(transformer.state.numeric_features)} numeric "
            "features are unobserved for this program; the interval includes "
            "the cost of imputing them."
        )

    if abs(predictive_mean - plug_in) >= MATERIAL_FUNCTIONAL_GAP:
        caveats.append(
            f"The predictive mean over draws ({predictive_mean:.3f}) differs "
            f"from the served plug-in probability ({plug_in:.3f}) by "
            f"{predictive_mean - plug_in:+.3f}. This is an arithmetic "
            "consequence of averaging a non-linear function over imputation "
            "draws for unreported evidence, not a biological signal, and no "
            "driver in the attribution corresponds to it."
        )

    return UncertaintyResult(
        probability=plug_in,
        predictive_mean_probability=predictive_mean,
        interval_low=lo,
        interval_high=hi,
        interval_level=level,
        components_sd_logit=components,
        variance_share=variance_share,
        total_sd_logit=total_sd,
        n_imputations=M,
        n_param_samples=n_param_samples,
        caveats=caveats,
        method=(
            "laplace_posterior" if supports_posterior else "bootstrap_ensemble"
        )
        + "+multiple_imputation"
        + ("+calibration_covariance" if calibrator.supports_parameter_uncertainty else ""),
    )


def specification_sensitivity(
    primary: float, alternatives: dict[str, float]
) -> dict[str, Any]:
    """Disagreement across model families, reported beside the interval.

    Deliberately *not* folded into the interval: combining families would
    require a prior over model specifications, and asserting one silently would
    be exactly the kind of unearned precision this platform exists to avoid.
    """
    if not alternatives:
        return {"available": False, "alternatives": {}, "max_absolute_difference": None}
    diffs = {k: abs(v - primary) for k, v in alternatives.items()}
    return {
        "available": True,
        "alternatives": alternatives,
        "differences": diffs,
        "max_absolute_difference": max(diffs.values()),
        "interpretation": (
            "Absolute difference in estimated probability between the deployed "
            "model and each candidate family scored on the same evidence. Large "
            "disagreement means the estimate depends on modelling choices, not "
            "only on the evidence."
        ),
    }


def _accepts_groups(model: RiskModel) -> bool:
    return model.family == "hierarchical_logistic"


def _point_logit(model: RiskModel, X: np.ndarray, groups: str | None) -> np.ndarray:
    if _accepts_groups(model) and groups is not None:
        return np.asarray(model.predict_logit(X, groups=np.repeat(groups, X.shape[0])))
    return np.asarray(model.predict_logit(X))
