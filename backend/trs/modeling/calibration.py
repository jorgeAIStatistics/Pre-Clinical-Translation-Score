"""Calibration: turning a model score into a probability that means something.

The product's core output is a probability. A discrimination metric says
nothing about whether "70%" means seventy percent — a model can rank perfectly
and still be systematically over- or under-confident. So calibration is
mandatory here, not optional polish.

Rules enforced by this module and its callers
---------------------------------------------
1. **Never fit and assess calibration on the same rows.** The calibrator is fit
   on a dedicated calibration split, disjoint from both train and test. The
   reported calibration metrics come from the test split, which the calibrator
   has never seen.
2. **Choose the method out-of-sample too.** Picking between Platt and isotonic
   on the calibration set and then reporting that set's fit would be the same
   error one level up. :func:`select_calibrator` uses cross-validation *within*
   the calibration split to choose, then refits the winner on the whole split.
3. **Propagate what can be propagated.** Logistic recalibration has two
   parameters and an analytic covariance, so its uncertainty flows into the
   prediction interval. Isotonic regression has no such parameterisation; when
   it is selected, :attr:`Calibrator.supports_parameter_uncertainty` is False
   and the scoring layer reports that calibration uncertainty is not included.

Why logistic recalibration is usually the right choice here
-----------------------------------------------------------
Isotonic regression is more flexible and, on a few hundred calibration points,
that flexibility mostly buys overfitting and a step function with visible
plateaus. Logistic recalibration ``logit_cal = a + b·logit_raw`` fits two
parameters, composes cleanly with the additive log-odds decomposition (every
contribution simply scales by ``b``), and yields the calibration intercept and
slope directly as interpretable diagnostics: ``a = 0, b = 1`` is perfect
calibration, ``b < 1`` means over-confident.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import numpy as np
from scipy import linalg
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from trs.modeling.base import logit, sigmoid


class Calibrator(abc.ABC):
    method: ClassVar[str]
    supports_parameter_uncertainty: ClassVar[bool] = False

    @abc.abstractmethod
    def fit(self, raw_logit: np.ndarray, y: np.ndarray) -> Calibrator: ...

    @abc.abstractmethod
    def transform_logit(self, raw_logit: np.ndarray) -> np.ndarray: ...

    def transform(self, raw_logit: np.ndarray) -> np.ndarray:
        return sigmoid(self.transform_logit(raw_logit))

    def sample_transform_logit(
        self, raw_logit: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``(n_samples, n_rows)`` calibrated log-odds including calibration
        parameter uncertainty. The default repeats the point estimate."""
        base = self.transform_logit(np.atleast_1d(raw_logit))
        return np.repeat(base[None, :], n_samples, axis=0)

    def apply_with_parameter_draws(
        self, raw: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply calibration to an ``(S, M)`` block of raw log-odds, using an
        independent calibration-parameter draw for each of the ``S`` rows.

        The default ignores calibration uncertainty (there is none to
        represent); families that can represent it override. The scoring layer
        reads :attr:`supports_parameter_uncertainty` to decide whether to tell
        the user that this component is excluded.
        """
        return self.transform_logit(np.asarray(raw, dtype=float))

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> Calibrator:
        method = payload["method"]
        for cls in (LogisticRecalibration, IsotonicCalibration, IdentityCalibration):
            if cls.method == method:
                return cls._load(payload)
        raise ValueError(f"Unknown calibration method {method!r}")


class IdentityCalibration(Calibrator):
    """No transformation. Used only when recalibration demonstrably hurts."""

    method: ClassVar[str] = "identity"
    supports_parameter_uncertainty: ClassVar[bool] = True

    def fit(self, raw_logit: np.ndarray, y: np.ndarray) -> IdentityCalibration:
        return self

    def transform_logit(self, raw_logit: np.ndarray) -> np.ndarray:
        return np.asarray(raw_logit, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}

    @classmethod
    def _load(cls, payload: dict[str, Any]) -> IdentityCalibration:
        return cls()


@dataclasses.dataclass
class LogisticRecalibration(Calibrator):
    """``logit_cal = a + b · logit_raw`` — Platt scaling on the log-odds scale.

    ``a`` is the calibration intercept and ``b`` the calibration slope, the two
    standard diagnostics. The 2×2 parameter covariance from the fit is retained
    so that calibration uncertainty enters the prediction interval rather than
    being quietly ignored.
    """

    method: ClassVar[str] = "logistic"
    supports_parameter_uncertainty: ClassVar[bool] = True

    intercept: float = 0.0
    slope: float = 1.0
    covariance: list[list[float]] | None = None
    n_fit: int = 0

    def fit(self, raw_logit: np.ndarray, y: np.ndarray) -> LogisticRecalibration:
        z = np.asarray(raw_logit, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        X = np.column_stack([np.ones_like(z), z])

        params = np.array([0.0, 1.0])
        for _ in range(100):
            eta = X @ params
            mu = sigmoid(eta)
            W = np.clip(mu * (1 - mu), 1e-10, None)
            grad = X.T @ (mu - y)
            H = X.T @ (X * W[:, None])
            # A whisper of ridge keeps the fit finite on a perfectly separated
            # calibration split, which happens with small, skewed cohorts.
            H = H + 1e-6 * np.eye(2)
            try:
                step = linalg.solve(H, grad, assume_a="pos")
            except linalg.LinAlgError:
                break
            params = params - step
            if np.max(np.abs(step)) < 1e-10:
                break

        self.intercept, self.slope = float(params[0]), float(params[1])
        eta = X @ params
        W = np.clip(sigmoid(eta) * (1 - sigmoid(eta)), 1e-10, None)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        self.covariance = np.linalg.inv(H).tolist()
        self.n_fit = int(z.size)
        return self

    def transform_logit(self, raw_logit: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(raw_logit, dtype=float)

    def sample_transform_logit(
        self, raw_logit: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        z = np.atleast_1d(np.asarray(raw_logit, dtype=float))
        if self.covariance is None:
            return super().sample_transform_logit(z, n_samples, rng)
        mean = np.array([self.intercept, self.slope])
        cov = np.asarray(self.covariance, dtype=float)
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            return super().sample_transform_logit(z, n_samples, rng)
        draws = mean[None, :] + rng.standard_normal((n_samples, 2)) @ L.T
        return draws[:, 0:1] + draws[:, 1:2] * z[None, :]

    def apply_with_parameter_draws(
        self, raw: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        raw = np.atleast_2d(np.asarray(raw, dtype=float))
        if self.covariance is None:
            return self.transform_logit(raw)
        mean = np.array([self.intercept, self.slope])
        cov = np.asarray(self.covariance, dtype=float)
        try:
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            return self.transform_logit(raw)
        draws = mean[None, :] + rng.standard_normal((raw.shape[0], 2)) @ L.T
        return draws[:, 0:1] + draws[:, 1:2] * raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "intercept": self.intercept,
            "slope": self.slope,
            "covariance": self.covariance,
            "n_fit": self.n_fit,
        }

    @classmethod
    def _load(cls, payload: dict[str, Any]) -> LogisticRecalibration:
        return cls(
            intercept=payload["intercept"],
            slope=payload["slope"],
            covariance=payload.get("covariance"),
            n_fit=payload.get("n_fit", 0),
        )


class IsotonicCalibration(Calibrator):
    """Monotone non-parametric calibration, bounded by its own sample size.

    Flexible, but with three costs the platform must state.

    **It cannot extrapolate.** Predictions outside the calibration range are
    clipped to the observed endpoints.

    **It carries no parameter covariance**, so calibration uncertainty is
    excluded from the prediction interval when this method is selected.

    **Its endpoint bins assert certainty it does not have.** An isotonic fit
    whose lowest bin happens to contain only negatives predicts exactly zero.
    Taken literally that is an infinite log-odds, and one confidently-wrong
    test prediction then dominates every aggregate metric — exactly what
    happened during development of this pipeline, where an unbounded isotonic
    fit on 110 calibration points drove held-out log loss from 0.70 to 1.89.

    The fix is a sample-size bound rather than an arbitrary epsilon: from ``n``
    observations one cannot justify a probability more extreme than roughly
    ``1/(n+2)`` — the Laplace rule-of-succession bound. Output is clipped
    there, so a 110-point calibration set may assert at most 0.9%, not
    0.0001%.
    """

    method: ClassVar[str] = "isotonic"
    supports_parameter_uncertainty: ClassVar[bool] = False

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self.n_fit = 0

    @property
    def probability_bound(self) -> float:
        """Most extreme probability this fit's sample size can justify."""
        return 1.0 / (max(self.n_fit, 1) + 2.0)

    def fit(self, raw_logit: np.ndarray, y: np.ndarray) -> IsotonicCalibration:
        z = np.asarray(raw_logit, dtype=float).ravel()
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(z, np.asarray(y, dtype=float).ravel())
        self.n_fit = int(z.size)
        return self

    def transform_logit(self, raw_logit: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("Calibrator used before fit().")
        z = np.atleast_1d(np.asarray(raw_logit, dtype=float))
        # IsotonicRegression accepts only 1-D input; preserve the caller's
        # shape so callers can pass an (S, M) draw block like any other
        # calibrator.
        shape = z.shape
        bound = self.probability_bound
        p = np.clip(self._iso.predict(z.ravel()), bound, 1.0 - bound)
        return np.asarray(logit(p), dtype=float).reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        if self._iso is None:
            raise RuntimeError("Calibrator used before fit().")
        return {
            "method": self.method,
            "x_thresholds": np.asarray(self._iso.X_thresholds_).tolist(),
            "y_thresholds": np.asarray(self._iso.y_thresholds_).tolist(),
            "n_fit": self.n_fit,
            "probability_bound": self.probability_bound,
        }

    @classmethod
    def _load(cls, payload: dict[str, Any]) -> IsotonicCalibration:
        obj = cls()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        x = np.asarray(payload["x_thresholds"], dtype=float)
        yv = np.asarray(payload["y_thresholds"], dtype=float)
        iso.fit(x, yv)
        obj._iso = iso
        obj.n_fit = payload.get("n_fit", 0)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────

#: A non-slope-correcting method must beat logistic recalibration by more than
#: this relative margin to be selected. See `select_calibrator`.
SLOPE_CORRECTION_PREFERENCE = 0.03

#: Minimum calibration-set size before isotonic regression is admitted as a
#: candidate at all. Isotonic fits one parameter per distinct step; on a few
#: hundred points it overfits the calibration split, generalises worse than a
#: two-parameter Platt fit, and produces degenerate endpoint bins. The
#: threshold is a rule of thumb from the calibration literature, made explicit
#: here rather than left to whichever method wins a noisy cross-validation.
MIN_N_FOR_ISOTONIC = 300


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_calibrator(
    raw_logit: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 0,
    candidates: tuple[type[Calibrator], ...] = (
        LogisticRecalibration, IsotonicCalibration, IdentityCalibration,
    ),
) -> tuple[Calibrator, dict[str, Any]]:
    """Choose a calibration method by cross-validation *inside* the calibration
    split, then refit the winner on the whole split.

    Returns the fitted calibrator and a selection record for the artifact, so
    the Methods page can show what was compared and why one won.
    """
    z = np.asarray(raw_logit, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    usable_splits = min(n_splits, n_pos, n_neg)
    scores: dict[str, float] = {}

    if usable_splits < 2:
        # Too few of one class to cross-validate honestly. Fall back to the
        # two-parameter method, which is the most robust of the three at small
        # n, and say so in the record.
        chosen = LogisticRecalibration().fit(z, y)
        return chosen, {
            "method": chosen.method,
            "reason": (
                f"Insufficient class balance for cross-validated selection "
                f"({n_pos} positive, {n_neg} negative); defaulted to logistic "
                "recalibration."
            ),
            "candidate_scores": {},
            "n_calibration": int(z.size),
            "excluded_candidates": {},
        }

    excluded: dict[str, str] = {}
    admitted: list[type[Calibrator]] = []
    for cls in candidates:
        if cls is IsotonicCalibration and z.size < MIN_N_FOR_ISOTONIC:
            excluded[cls.method] = (
                f"Excluded: {z.size} calibration observations is below the "
                f"{MIN_N_FOR_ISOTONIC} required for a non-parametric fit to "
                "generalise reliably."
            )
            continue
        admitted.append(cls)
    candidates = tuple(admitted)

    cv = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=random_state)
    for cls in candidates:
        fold_losses = []
        for tr, te in cv.split(z.reshape(-1, 1), y):
            if len(np.unique(y[tr])) < 2:
                continue
            cal = cls().fit(z[tr], y[tr])
            fold_losses.append(_log_loss(y[te], cal.transform(z[te])))
        if fold_losses:
            scores[cls.method] = float(np.mean(fold_losses))

    if not scores:
        chosen = LogisticRecalibration().fit(z, y)
        return chosen, {
            "method": chosen.method,
            "reason": "No usable cross-validation folds; defaulted to logistic recalibration.",
            "candidate_scores": {},
            "n_calibration": int(z.size),
            "excluded_candidates": {},
        }

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best_method = ranked[0][0]

    # Slope-correction preference.
    #
    # A calibration split of a few hundred rows cannot distinguish two methods
    # separated by a fraction of a nat, and the ranking flips on resampling. In
    # that regime the tie-break should not be arbitrary: logistic recalibration
    # is preferred because it is the only candidate that corrects a calibration
    # *slope*, and slope error is the failure mode a temporally-shifted
    # deployment reliably produces. Selecting `identity` on a 0.01-nat edge
    # leaves an over-confident model uncorrected, which is exactly what
    # happened during development of this pipeline.
    tie_break_applied = False
    if best_method != LogisticRecalibration.method and LogisticRecalibration.method in scores:
        logistic_score = scores[LogisticRecalibration.method]
        best_score = scores[best_method]
        if logistic_score <= best_score * (1.0 + SLOPE_CORRECTION_PREFERENCE):
            best_method = LogisticRecalibration.method
            tie_break_applied = True

    best_cls = next(c for c in candidates if c.method == best_method)
    chosen = best_cls().fit(z, y)

    reason = (
        f"Lowest cross-validated log loss on the calibration split "
        f"({ranked[0][0]}: {ranked[0][1]:.4f}"
        + (f"; next best {ranked[1][0]}: {ranked[1][1]:.4f}" if len(ranked) > 1 else "")
        + ")."
    )
    if tie_break_applied:
        reason += (
            f" Logistic recalibration was preferred over {ranked[0][0]} under the "
            f"slope-correction rule: it is within {SLOPE_CORRECTION_PREFERENCE:.0%} "
            "of the best score and is the only candidate that corrects a "
            "calibration slope, which is the error a temporally-shifted "
            "deployment reliably introduces."
        )
    return chosen, {
        "method": chosen.method,
        "reason": reason,
        "candidate_scores": scores,
        "n_calibration": int(z.size),
        "supports_parameter_uncertainty": chosen.supports_parameter_uncertainty,
        "slope_correction_tie_break_applied": tie_break_applied,
        "excluded_candidates": excluded,
    }
