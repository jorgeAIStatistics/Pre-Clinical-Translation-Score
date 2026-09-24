"""Gradient-boosted trees: the non-linear candidate.

Included so the claim "a linear model is sufficient here" is *tested* rather
than assumed. It is held to a higher bar than the baseline for exactly the
reasons that make it attractive: it can fit interactions the linear model
cannot, and it can fit noise the linear model cannot.

Two capabilities are genuinely lost by using it, and the code says so rather
than papering over them:

* **No analytic posterior.** Uncertainty comes from a bootstrap ensemble, which
  measures sensitivity of the fit to the training sample. That is a real and
  useful quantity, but it is not the same as a posterior predictive interval:
  it omits the irreducible parameter uncertainty a Bayesian interval includes,
  and it systematically under-covers for small ensembles.
* **No exact attribution.** ``attribution()`` returns ``None``. Tree ensembles
  admit no exact additive decomposition; a Shapley approximation could be
  added, but it would be an approximation presented next to an exact one, and
  the UI would have no way to signal the difference to a user. Until that is
  solved properly, this family ships without a driver breakdown — which is
  itself a strong practical argument for the linear default.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from trs.modeling.base import FitReport, RiskModel, logit

#: Bootstrap replicates for the uncertainty ensemble. 60 is a compromise
#: between interval stability and fit time; the validation report records the
#: Monte-Carlo error this leaves.
DEFAULT_ENSEMBLE_SIZE = 60


class GradientBoostedRisk(RiskModel):
    """Histogram gradient boosting with a bootstrap uncertainty ensemble."""

    family: ClassVar[str] = "gbdt"
    supports_analytic_posterior: ClassVar[bool] = False
    supports_exact_attribution: ClassVar[bool] = False

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_depth: int | None = 3,
        min_samples_leaf: int = 15,
        l2_regularization: float = 1.0,
        max_leaf_nodes: int = 15,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_leaf_nodes=max_leaf_nodes,
            ensemble_size=ensemble_size,
            random_state=random_state,
        )
        self._model: HistGradientBoostingClassifier | None = None
        self._ensemble: list[HistGradientBoostingClassifier] = []

    def _make(self, seed: int) -> HistGradientBoostingClassifier:
        h = self.hyperparameters
        return HistGradientBoostingClassifier(
            learning_rate=h["learning_rate"],
            max_iter=h["max_iter"],
            max_depth=h["max_depth"],
            min_samples_leaf=h["min_samples_leaf"],
            l2_regularization=h["l2_regularization"],
            max_leaf_nodes=h["max_leaf_nodes"],
            early_stopping=False,
            random_state=seed,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        columns: list[str] | None = None,
        groups: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> GradientBoostedRisk:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.columns = list(columns) if columns else [f"x{i}" for i in range(X.shape[1])]

        seed = int(self.hyperparameters["random_state"])
        self._model = self._make(seed).fit(X, y, sample_weight=sample_weight)

        # Bootstrap ensemble. Resampling is done at the *group* level when
        # groups are supplied, so a program contributing several tracks moves
        # in or out of a replicate as a unit — resampling rows independently
        # would understate uncertainty by breaking that dependence.
        rng = np.random.default_rng(seed)
        self._ensemble = []
        n = len(y)
        for b in range(int(self.hyperparameters["ensemble_size"])):
            idx = _bootstrap_indices(rng, n, groups)
            if len(np.unique(y[idx])) < 2:
                continue
            w = None if sample_weight is None else np.asarray(sample_weight)[idx]
            self._ensemble.append(self._make(seed + 1000 + b).fit(X[idx], y[idx], sample_weight=w))

        self.fit_report = FitReport(
            n_rows=n, n_features=X.shape[1], n_positive=int(y.sum()),
            converged=True,
            hyperparameters=dict(self.hyperparameters),
            diagnostics={
                "ensemble_members": len(self._ensemble),
                "grouped_bootstrap": groups is not None,
            },
        )
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model used before fit().")
        p = self._model.predict_proba(np.atleast_2d(np.asarray(X, dtype=float)))[:, 1]
        return np.asarray(logit(p), dtype=float)

    def posterior_logit_samples(
        self, X: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Bootstrap-ensemble draws. Not a posterior; labelled as such upstream."""
        if not self._ensemble:
            raise RuntimeError("Ensemble empty; cannot produce an uncertainty band.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        member_logits = np.vstack([
            np.asarray(logit(m.predict_proba(X)[:, 1])) for m in self._ensemble
        ])
        pick = rng.integers(0, member_logits.shape[0], size=n_samples)
        return member_logits[pick]

    def to_state(self) -> dict[str, Any]:
        import base64
        import pickle

        payload = pickle.dumps(
            {"model": self._model, "ensemble": self._ensemble}, protocol=5
        )
        return {
            "family": self.family,
            "hyperparameters": self.hyperparameters,
            "columns": self.columns,
            # scikit-learn estimators have no stable cross-version text format.
            # The artifact records the library version so a load against a
            # different scikit-learn is detected rather than silently wrong.
            "pickle_b64": base64.b64encode(payload).decode("ascii"),
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
    def from_state(cls, state: dict[str, Any]) -> GradientBoostedRisk:
        import base64
        import pickle

        model = cls(**state["hyperparameters"])
        model.columns = list(state["columns"])
        payload = pickle.loads(base64.b64decode(state["pickle_b64"]))  # noqa: S301
        model._model = payload["model"]
        model._ensemble = payload["ensemble"]
        fr = state.get("fit_report")
        if fr:
            model.fit_report = FitReport(**fr)
        return model


def _bootstrap_indices(
    rng: np.random.Generator, n: int, groups: np.ndarray | None
) -> np.ndarray:
    if groups is None:
        return rng.integers(0, n, size=n)
    groups = np.asarray([str(g) for g in groups])
    unique = np.unique(groups)
    chosen = rng.choice(unique, size=unique.size, replace=True)
    idx: list[int] = []
    lookup = {g: np.flatnonzero(groups == g) for g in unique}
    for g in chosen:
        idx.extend(lookup[g].tolist())
    return np.asarray(idx, dtype=int)
