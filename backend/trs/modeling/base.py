"""Model interface shared by every candidate family.

Every model in this package must be able to do four things:

1. **Fit** on a design matrix and binary labels.
2. **Predict a log-odds**, not just a probability — the log-odds scale is where
   calibration and the additive explanation both live.
3. **Draw from a posterior (or an approximation of one)** over the log-odds for
   a given row. This is what makes an honest interval possible. A family that
   cannot support this must say so, and the scoring layer then refuses to
   publish an interval for it rather than inventing one.
4. **Describe itself** well enough to be serialised, reloaded and audited.

The third requirement is the reason the deployed default is a linear model.
A regularised logistic regression admits an analytic Gaussian approximation to
its posterior; a boosted-tree ensemble does not, and its bootstrap-ensemble
substitute measures a different and narrower thing. That is a real argument for
the simple model, not a concession.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, ClassVar

import numpy as np


class UncertaintyUnsupported(RuntimeError):
    """Raised when an interval is requested from a family that cannot give one."""


@dataclasses.dataclass(slots=True)
class FitReport:
    """What happened during fitting. Stored on the artifact."""

    n_rows: int
    n_features: int
    n_positive: int
    converged: bool
    hyperparameters: dict[str, Any]
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)


class RiskModel(abc.ABC):
    """Common interface for translational-risk models."""

    #: Registry key; must be unique across families.
    family: ClassVar[str]
    #: Whether :meth:`posterior_logit_samples` returns a genuine posterior
    #: approximation rather than a resampling heuristic.
    supports_analytic_posterior: ClassVar[bool] = False
    #: Whether the model admits an exact additive decomposition of its log-odds.
    supports_exact_attribution: ClassVar[bool] = False

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters = hyperparameters
        self.fit_report: FitReport | None = None
        self.columns: list[str] = []

    # ---- required ----------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        columns: list[str] | None = None,
        groups: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> RiskModel: ...

    @abc.abstractmethod
    def predict_logit(self, X: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def to_state(self) -> dict[str, Any]: ...

    @classmethod
    @abc.abstractmethod
    def from_state(cls, state: dict[str, Any]) -> RiskModel: ...

    # ---- provided ----------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logit(X))

    def posterior_logit_samples(
        self, X: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``(n_samples, n_rows)`` draws of the log-odds.

        The default refuses rather than fabricating. Families override.
        """
        raise UncertaintyUnsupported(
            f"Model family {self.family!r} does not support posterior sampling. "
            "The scoring layer must not publish an interval for it."
        )

    def attribution(self, X: np.ndarray) -> np.ndarray | None:
        """``(n_rows, n_features)`` exact additive log-odds contributions.

        ``None`` when the family admits no exact decomposition; the scoring
        layer then reports no driver breakdown rather than a post-hoc
        approximation dressed up as one.
        """
        return None

    def intercept_at_reference(self) -> float:
        """Log-odds at the training-mean row (the origin in standardised space)."""
        return float(self.predict_logit(np.zeros((1, len(self.columns))))[0])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid(z):  # noqa: ANN001, ANN201
    """Public stable sigmoid accepting scalars or arrays."""
    arr = np.asarray(z, dtype=float)
    result = _sigmoid(arr)
    return float(result) if np.isscalar(z) or arr.ndim == 0 else result


def logit(p, eps: float = 1e-12):  # noqa: ANN001, ANN201
    arr = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    result = np.log(arr / (1 - arr))
    return float(result) if np.isscalar(p) or np.asarray(p).ndim == 0 else result
