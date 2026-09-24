"""Regularised logistic regression with a Laplace posterior — the baseline.

Why this is the default deployed model
--------------------------------------
Three properties that matter more than a marginal AUROC gain:

* **Exact attribution.** The log-odds is ``b + Σ βⱼ zⱼ``. Every driver shown to
  a user is a term in that sum and the terms add up to the number on screen.
  No surrogate model, no sampling approximation, no post-hoc rationalisation.
* **A defensible posterior.** The L2 penalty *is* a Gaussian prior
  ``β ~ N(0, λ⁻¹I)``, so the penalised fit is a MAP estimate and the Laplace
  approximation ``N(β̂, H⁻¹)`` with ``H = XᵀWX + λI`` is the standard
  second-order approximation to the posterior, not an invention.
* **Auditability.** A biostatistician can read the coefficients, check their
  signs against prior expectation, and say whether the model is behaving.

The Laplace approximation is asymptotic. At the cohort sizes realistic here
(hundreds of programs, ~100 design columns after encoding) it is adequate but
not exact, particularly in the tails. The validation report therefore
cross-checks its interval widths against a nonparametric bootstrap
(:mod:`trs.validation.report`), and a material disagreement is surfaced rather
than buried.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from scipy import linalg

from trs.modeling.base import FitReport, RiskModel, _sigmoid

#: Weak prior on the intercept. Present only so the Hessian is invertible in
#: the intercept direction; it is loose enough not to shrink the base rate.
_INTERCEPT_PRECISION = 1e-6

#: Jitter ladder applied to the Hessian if the Cholesky factorisation fails.
_JITTER_LADDER = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)


class L2LogisticRisk(RiskModel):
    """L2-penalised logistic regression with a Laplace posterior.

    Parameters
    ----------
    l2_penalty:
        The ``λ`` in ``(λ/2)‖β‖²``, i.e. the prior precision. Selected by
        grouped/temporal cross-validation in :mod:`trs.pipeline.train`, never
        hard-coded.
    class_weight_balanced:
        Reweight classes to equal total weight. Off by default: the cohort's
        class balance is a real property of drug development and rebalancing it
        distorts the calibration the product depends on. Available for
        diagnostic comparison.
    """

    family: ClassVar[str] = "l2_logistic"
    supports_analytic_posterior: ClassVar[bool] = True
    supports_exact_attribution: ClassVar[bool] = True

    def __init__(
        self,
        l2_penalty: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-8,
        class_weight_balanced: bool = False,
    ) -> None:
        super().__init__(
            l2_penalty=l2_penalty,
            max_iter=max_iter,
            tol=tol,
            class_weight_balanced=class_weight_balanced,
        )
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._posterior_cov: np.ndarray | None = None
        self._posterior_chol: np.ndarray | None = None

    # ---- fitting -----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        columns: list[str] | None = None,
        groups: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> L2LogisticRisk:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.columns = list(columns) if columns else [f"x{i}" for i in range(p)]

        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if self.hyperparameters["class_weight_balanced"]:
            n_pos, n_neg = float(y.sum()), float((1 - y).sum())
            if n_pos > 0 and n_neg > 0:
                w = w * np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))

        lam_param = self.hyperparameters["l2_penalty"]
        lam = (
            float(lam_param) if np.isscalar(lam_param)
            else np.asarray(lam_param, dtype=float)
        )
        beta, converged, n_iter = _fit_penalised_logistic(
            X, y, w, lam,
            max_iter=int(self.hyperparameters["max_iter"]),
            tol=float(self.hyperparameters["tol"]),
        )
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:].copy()

        # Laplace posterior: H = Xᵀ W X + Λ, evaluated at the MAP.
        Xa = np.hstack([np.ones((n, 1)), X])
        eta = Xa @ beta
        mu = _sigmoid(eta)
        W = w * mu * (1 - mu)
        H = Xa.T @ (Xa * W[:, None])
        prior_precision = np.empty(p + 1)
        prior_precision[0] = _INTERCEPT_PRECISION
        prior_precision[1:] = np.asarray(lam, dtype=float)
        H = H + np.diag(prior_precision)
        self._posterior_cov, self._posterior_chol = _invert_and_factor(H)

        self.fit_report = FitReport(
            n_rows=n,
            n_features=p,
            n_positive=int(y.sum()),
            converged=converged,
            hyperparameters=dict(self.hyperparameters),
            diagnostics={
                "n_iter": n_iter,
                "hessian_condition_number": float(np.linalg.cond(H)),
                "posterior_sd_max": float(np.sqrt(np.diag(self._posterior_cov)).max()),
                "effective_degrees_of_freedom": float(
                    np.trace(H @ self._posterior_cov) - 0.0
                ),
            },
        )
        return self

    # ---- prediction --------------------------------------------------------

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model used before fit().")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def posterior_logit_samples(
        self, X: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw ``β ~ N(β̂, H⁻¹)`` and return ``(n_samples, n_rows)`` log-odds."""
        if self._posterior_chol is None:
            raise RuntimeError("Model used before fit().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xa = np.hstack([np.ones((X.shape[0], 1)), X])
        beta_hat = np.concatenate([[self.intercept_], self.coef_])
        z = rng.standard_normal((n_samples, beta_hat.size))
        draws = beta_hat[None, :] + z @ self._posterior_chol.T
        return draws @ Xa.T

    def attribution(self, X: np.ndarray) -> np.ndarray:
        """Exact per-column log-odds contributions ``βⱼ · zⱼ``.

        Rows sum to ``predict_logit(X) - intercept_``. Because the design
        matrix is standardised on the training mean, the intercept *is* the
        log-odds of the average program in the cohort, which makes each
        contribution read as "how this program differs from the cohort
        average" — the comparison a user actually wants.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X * self.coef_[None, :]

    def intercept_at_reference(self) -> float:
        return float(self.intercept_)

    # ---- introspection -----------------------------------------------------

    def coefficient_table(self) -> list[dict[str, Any]]:
        sd = np.sqrt(np.diag(self._posterior_cov))[1:] if self._posterior_cov is not None else None
        rows = []
        for i, name in enumerate(self.columns):
            row: dict[str, Any] = {
                "column": name,
                "coefficient": float(self.coef_[i]),
                "odds_ratio_per_sd": float(np.exp(self.coef_[i])),
            }
            if sd is not None:
                row["posterior_sd"] = float(sd[i])
                row["ci_low"] = float(self.coef_[i] - 1.959964 * sd[i])
                row["ci_high"] = float(self.coef_[i] + 1.959964 * sd[i])
            rows.append(row)
        return rows

    # ---- serialisation -----------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "hyperparameters": self.hyperparameters,
            "columns": self.columns,
            "intercept": self.intercept_,
            "coef": self.coef_.tolist() if self.coef_ is not None else None,
            "posterior_cov": (
                self._posterior_cov.tolist() if self._posterior_cov is not None else None
            ),
            "fit_report": (
                {
                    "n_rows": self.fit_report.n_rows,
                    "n_features": self.fit_report.n_features,
                    "n_positive": self.fit_report.n_positive,
                    "converged": self.fit_report.converged,
                    "hyperparameters": self.fit_report.hyperparameters,
                    "diagnostics": self.fit_report.diagnostics,
                }
                if self.fit_report else None
            ),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> L2LogisticRisk:
        model = cls(**state["hyperparameters"])
        model.columns = list(state["columns"])
        model.intercept_ = float(state["intercept"])
        model.coef_ = np.asarray(state["coef"], dtype=float)
        if state.get("posterior_cov") is not None:
            cov = np.asarray(state["posterior_cov"], dtype=float)
            model._posterior_cov = cov
            model._posterior_chol = _safe_cholesky(cov)
        fr = state.get("fit_report")
        if fr:
            model.fit_report = FitReport(**fr)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Numerics
# ─────────────────────────────────────────────────────────────────────────────

def _fit_penalised_logistic(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    lam: float | np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, bool, int]:
    """Newton–IRLS for the L2-penalised logistic MAP.

    Implemented directly rather than delegated so that the Hessian used for the
    Laplace posterior is provably the same matrix the optimiser converged on,
    and so the intercept's (near-absent) penalty is explicit rather than a
    library convention that could change under us.

    Falls back to a damped step when the full Newton step increases the
    objective, which happens on separable or near-separable folds.
    """
    n, p = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])
    # ``lam`` may be a scalar (one ridge penalty for every coefficient) or a
    # length-p vector giving a per-coefficient precision, which is how the
    # hierarchical family applies a separate variance component to its group
    # effects. The intercept is always near-unpenalised.
    prior_precision = np.empty(p + 1)
    prior_precision[0] = _INTERCEPT_PRECISION
    prior_precision[1:] = np.asarray(lam, dtype=float)

    beta = np.zeros(p + 1)
    # Start the intercept at the empirical log-odds: a better basin and one
    # fewer iteration on every fit.
    pos_rate = float(np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6))
    beta[0] = np.log(pos_rate / (1 - pos_rate))

    def objective(b: np.ndarray) -> float:
        eta = Xa @ b
        # log(1 + exp(eta)) computed stably.
        ll = np.sum(w * (y * eta - np.logaddexp(0.0, eta)))
        return -ll + 0.5 * float(prior_precision @ (b * b))

    prev = objective(beta)
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):  # noqa: B007 — read after the loop
        eta = Xa @ beta
        mu = _sigmoid(eta)
        W = w * mu * (1 - mu)
        grad = Xa.T @ (w * (mu - y)) + prior_precision * beta
        H = Xa.T @ (Xa * W[:, None]) + np.diag(prior_precision)

        try:
            step = linalg.solve(H, grad, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            step = np.linalg.lstsq(H, grad, rcond=None)[0]

        # Backtracking line search; guarantees monotone descent.
        t = 1.0
        new_obj = None
        for _ in range(30):
            candidate = beta - t * step
            new_obj = objective(candidate)
            if new_obj <= prev:
                break
            t *= 0.5
        beta = beta - t * step

        if new_obj is not None and abs(prev - new_obj) < tol * (1.0 + abs(prev)):
            prev = new_obj
            converged = True
            break
        prev = new_obj if new_obj is not None else prev

    return beta, converged, n_iter


def _invert_and_factor(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert the Hessian and return ``(cov, chol(cov))`` with jitter fallback."""
    for jitter in _JITTER_LADDER:
        try:
            Hj = H + jitter * np.eye(H.shape[0]) if jitter else H
            c, low = linalg.cho_factor(Hj, lower=True, check_finite=False)
            cov = linalg.cho_solve((c, low), np.eye(H.shape[0]), check_finite=False)
            cov = 0.5 * (cov + cov.T)  # enforce symmetry against round-off
            return cov, _safe_cholesky(cov)
        except linalg.LinAlgError:
            continue
    cov = np.linalg.pinv(H)
    cov = 0.5 * (cov + cov.T)
    return cov, _safe_cholesky(cov)


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Lower-triangular factor of a covariance, repaired if not quite PSD."""
    for jitter in _JITTER_LADDER:
        try:
            Cj = cov + jitter * np.eye(cov.shape[0]) if jitter else cov
            return np.linalg.cholesky(Cj)
        except np.linalg.LinAlgError:
            continue
    # Last resort: clip negative eigenvalues. Records nothing silently — the
    # caller's diagnostics carry the condition number that led here.
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(np.sqrt(vals))
