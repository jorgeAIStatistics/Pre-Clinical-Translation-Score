"""Hierarchical logistic regression with partial pooling on mechanism groups.

The scientific motivation
-------------------------
Translational performance is not homogeneous. Antibody-drug conjugates against
surface antigens behave differently from small molecules against kinases, and a
model with a single global intercept forces both to share a base rate. But
per-group intercepts fitted independently are hopeless with a handful of
programs in most groups: each would chase noise.

Partial pooling is the standard answer. Group intercepts are treated as draws
from ``N(0, τ²)``, so a group with many programs is allowed to move away from
the population mean and a group with two is pulled back toward it. The amount
of pooling is *learned* from the data rather than chosen.

Implementation and its honest limitation
-----------------------------------------
This is a penalised-likelihood (MAP) fit, not a fully Bayesian one: the design
matrix is augmented with group indicators carrying prior precision
``λ_g = 1/τ²``, and ``τ²`` is selected on a grid by grouped cross-validated log
loss rather than integrated over. Shrinkage is therefore captured correctly,
but **uncertainty in the variance component itself is not propagated** into
prediction intervals. Intervals from this family are consequently slightly too
narrow, by an amount that grows as the number of groups shrinks.

That limitation is why this family is a *candidate*, not the default, and why
:mod:`trs.pipeline.select` requires it to clear a meaningful margin over the
pooled baseline before it can be promoted. If the platform ever needs proper
variance-component uncertainty, the honest upgrade is a genuine MCMC fit
(PyMC), declared as an optional dependency in MODELING_PLAN.md — not a wider
interval asserted here without a basis.

Unseen groups at prediction time
--------------------------------
A program whose mechanism group was absent from training gets a group effect of
exactly zero, i.e. the population mean. That is the correct behaviour under the
model: with no observations the posterior for that group's deviation is its
prior, centred at zero.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from trs.modeling.base import FitReport, RiskModel, _sigmoid
from trs.modeling.logistic import (
    _INTERCEPT_PRECISION,
    _fit_penalised_logistic,
    _invert_and_factor,
    _safe_cholesky,
)

#: Minimum observations for a group to get its own indicator column. Below
#: this, members are folded into a shared "small groups" level rather than each
#: contributing a column estimated from one or two rows.
MIN_GROUP_SIZE = 5

_SMALL_GROUP = "__small__"


class HierarchicalLogisticRisk(RiskModel):
    """Logistic regression with partially pooled group intercepts."""

    family: ClassVar[str] = "hierarchical_logistic"
    supports_analytic_posterior: ClassVar[bool] = True
    supports_exact_attribution: ClassVar[bool] = True

    def __init__(
        self,
        l2_penalty: float = 1.0,
        group_precision: float = 4.0,
        max_iter: int = 500,
        tol: float = 1e-8,
        min_group_size: int = MIN_GROUP_SIZE,
    ) -> None:
        super().__init__(
            l2_penalty=l2_penalty,
            group_precision=group_precision,
            max_iter=max_iter,
            tol=tol,
            min_group_size=min_group_size,
        )
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.group_levels_: list[str] = []
        self.group_effects_: dict[str, float] = {}
        self._posterior_cov: np.ndarray | None = None
        self._posterior_chol: np.ndarray | None = None

    # ---- helpers -----------------------------------------------------------

    def _group_matrix(self, groups: np.ndarray) -> np.ndarray:
        levels = self.group_levels_
        Z = np.zeros((len(groups), len(levels)), dtype=float)
        index = {lv: i for i, lv in enumerate(levels)}
        for r, g in enumerate(groups):
            key = str(g) if str(g) in index else _SMALL_GROUP
            j = index.get(key)
            if j is not None:
                Z[r, j] = 1.0
        return Z

    # ---- fitting -----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        columns: list[str] | None = None,
        groups: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> HierarchicalLogisticRisk:
        if groups is None:
            raise ValueError(
                "HierarchicalLogisticRisk requires `groups`; use L2LogisticRisk "
                "when no grouping structure is available."
            )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        groups = np.asarray([str(g) for g in groups])
        n, p = X.shape
        self.columns = list(columns) if columns else [f"x{i}" for i in range(p)]

        unique, counts = np.unique(groups, return_counts=True)
        min_size = int(self.hyperparameters["min_group_size"])
        kept = [g for g, c in zip(unique, counts, strict=True) if c >= min_size]
        has_small = bool(set(unique) - set(kept))
        self.group_levels_ = sorted(kept) + ([_SMALL_GROUP] if has_small else [])

        Z = self._group_matrix(groups)
        design = np.hstack([X, Z])

        lam = float(self.hyperparameters["l2_penalty"])
        lam_g = float(self.hyperparameters["group_precision"])
        prior = np.concatenate([np.full(p, lam), np.full(Z.shape[1], lam_g)])

        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        beta, converged, n_iter = _fit_penalised_logistic(
            design, y, w, prior,
            max_iter=int(self.hyperparameters["max_iter"]),
            tol=float(self.hyperparameters["tol"]),
        )
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1 : 1 + p].copy()
        self.group_effects_ = {
            lv: float(v) for lv, v in zip(self.group_levels_, beta[1 + p :], strict=True)
        }

        Da = np.hstack([np.ones((n, 1)), design])
        mu = _sigmoid(Da @ beta)
        W = w * mu * (1 - mu)
        H = Da.T @ (Da * W[:, None])
        full_prior = np.concatenate([[_INTERCEPT_PRECISION], prior])
        H = H + np.diag(full_prior)
        self._posterior_cov, self._posterior_chol = _invert_and_factor(H)

        shrunk = np.asarray(list(self.group_effects_.values()))
        self.fit_report = FitReport(
            n_rows=n, n_features=p, n_positive=int(y.sum()),
            converged=converged,
            hyperparameters=dict(self.hyperparameters),
            diagnostics={
                "n_iter": n_iter,
                "n_groups_modelled": len(self.group_levels_),
                "n_groups_observed": int(unique.size),
                "group_effect_sd": float(shrunk.std()) if shrunk.size else 0.0,
                "group_effect_range": (
                    [float(shrunk.min()), float(shrunk.max())] if shrunk.size else [0.0, 0.0]
                ),
                "implied_tau": float(1.0 / np.sqrt(lam_g)),
            },
        )
        return self

    # ---- prediction --------------------------------------------------------

    def predict_logit(self, X: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model used before fit().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        base = X @ self.coef_ + self.intercept_
        if groups is None:
            return base
        offsets = np.asarray([
            self.group_effects_.get(
                str(g), self.group_effects_.get(_SMALL_GROUP, 0.0)
            )
            for g in groups
        ])
        return base + offsets

    def posterior_logit_samples(
        self,
        X: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._posterior_chol is None:
            raise RuntimeError("Model used before fit().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n_rows, p = X.shape
        Z = (
            self._group_matrix(np.asarray([str(g) for g in groups]))
            if groups is not None
            else np.zeros((n_rows, len(self.group_levels_)))
        )
        Da = np.hstack([np.ones((n_rows, 1)), X, Z])
        beta_hat = np.concatenate([
            [self.intercept_], self.coef_,
            np.asarray([self.group_effects_[lv] for lv in self.group_levels_]),
        ])
        z = rng.standard_normal((n_samples, beta_hat.size))
        draws = beta_hat[None, :] + z @ self._posterior_chol.T
        return draws @ Da.T

    def attribution(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X * self.coef_[None, :]

    def intercept_at_reference(self) -> float:
        return float(self.intercept_)

    # ---- serialisation -----------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "hyperparameters": self.hyperparameters,
            "columns": self.columns,
            "intercept": self.intercept_,
            "coef": self.coef_.tolist() if self.coef_ is not None else None,
            "group_levels": self.group_levels_,
            "group_effects": self.group_effects_,
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
    def from_state(cls, state: dict[str, Any]) -> HierarchicalLogisticRisk:
        model = cls(**state["hyperparameters"])
        model.columns = list(state["columns"])
        model.intercept_ = float(state["intercept"])
        model.coef_ = np.asarray(state["coef"], dtype=float)
        model.group_levels_ = list(state["group_levels"])
        model.group_effects_ = dict(state["group_effects"])
        if state.get("posterior_cov") is not None:
            cov = np.asarray(state["posterior_cov"], dtype=float)
            model._posterior_cov = cov
            model._posterior_chol = _safe_cholesky(cov)
        fr = state.get("fit_report")
        if fr:
            model.fit_report = FitReport(**fr)
        return model
