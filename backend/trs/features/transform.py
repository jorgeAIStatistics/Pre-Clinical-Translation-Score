"""Encoding, imputation and standardisation: raw features → design matrix.

Everything fitted here is fitted on the **training fold only** and then applied
unchanged to calibration, test and production rows. Imputation statistics,
one-hot level sets and standardisation constants are all leakage vectors if
fitted on the full dataset, and all three are handled the same way: fit once on
train, serialise into the model artifact, reuse verbatim thereafter.

Missing-indicator policy
------------------------
A feature declared ``IMPUTE_WITH_INDICATOR`` gains a companion
``{name}__missing`` column, but only when its training missing rate lies within
:data:`INDICATOR_RATE_BAND`. Outside that band the indicator is (near-)constant
and contributes nothing but a penalised coefficient. The band is recorded in
the artifact so the decision is auditable rather than emergent.

Multiple imputation
-------------------
For uncertainty propagation, :meth:`FeatureTransformer.transform_with_imputations`
produces ``M`` draws of a single row, filling each missing numeric feature with
a random draw from that feature's observed training distribution rather than
its median. Pooling predictions across the draws yields Rubin's
between-imputation variance directly, which is what lets the reported interval
widen honestly when a program's evidence base has holes. Marginal (rather than
conditional) draws are used: with the sample sizes available a conditional
imputation model would itself be unstable, and marginal draws are the
conservative choice because they ignore correlations that would tighten the
imputed distribution.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from trs.features.registry import FEATURES, FEATURES_BY_NAME
from trs.features.spec import (
    FEATURE_SCHEMA_VERSION,
    FeatureSpec,
    MissingPolicy,
    Transform,
)

#: Missing-rate band within which a missing indicator is informative.
INDICATOR_RATE_BAND = (0.02, 0.98)

#: Floor on a column's standard deviation before it is treated as constant.
_SD_FLOOR = 1e-8


def _apply_transform(spec: FeatureSpec, value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return float(value)
    v = float(value)
    if not math.isfinite(v):
        return None
    match spec.transform:
        case Transform.LOG1P:
            return math.log1p(max(v, 0.0))
        case Transform.LOG2:
            return math.log2(v) if v > 0 else None
        case Transform.LOGIT:
            p = min(max(v, 1e-6), 1 - 1e-6)
            return math.log(p / (1 - p))
        case _:
            return v


@dataclasses.dataclass
class TransformerState:
    """Everything needed to reproduce the encoding. Fully JSON-serialisable."""

    schema_version: str
    #: Design-matrix column names, in order.
    columns: list[str]
    #: Raw numeric/ordinal/binary/count feature names, in order.
    numeric_features: list[str]
    #: Nominal feature → the non-reference levels that got a column.
    nominal_levels: dict[str, list[str]]
    #: Reference (dropped) level per nominal feature.
    nominal_reference: dict[str, str]
    #: Post-transform training median per numeric feature (imputation fill).
    medians: dict[str, float]
    #: Numeric features that received a missing indicator.
    indicators: list[str]
    #: Training missing rate per numeric feature (for the artifact/report).
    missing_rates: dict[str, float]
    #: Standardisation constants per design-matrix column.
    means: dict[str, float]
    scales: dict[str, float]
    #: Observed post-transform training values per numeric feature, used for
    #: multiple-imputation draws. Subsampled to bound artifact size.
    imputation_pool: dict[str, list[float]]
    n_fit_rows: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TransformerState:
        return cls(**payload)


class FeatureTransformer:
    """Fit-on-train encoder for the declared feature schema."""

    #: Cap on stored imputation-pool size per feature.
    POOL_CAP = 500

    def __init__(self, state: TransformerState | None = None) -> None:
        self.state = state

    # ---- fitting -----------------------------------------------------------

    def fit(self, rows: list[dict[str, Any]], *, random_state: int = 0) -> FeatureTransformer:
        rng = np.random.default_rng(random_state)
        numeric: list[str] = []
        nominal_levels: dict[str, list[str]] = {}
        nominal_reference: dict[str, str] = {}

        for spec in FEATURES:
            if spec.is_nominal:
                # The reference level is the registry's first declared level,
                # chosen as the most common / best-characterised category so
                # every other coefficient reads as a contrast against it.
                nominal_reference[spec.name] = spec.levels[0]
                observed = {
                    str(r.get(spec.name)) for r in rows if r.get(spec.name) is not None
                }
                levels = [lv for lv in spec.levels[1:] if lv in observed]
                nominal_levels[spec.name] = levels
            else:
                numeric.append(spec.name)

        transformed: dict[str, list[float | None]] = {}
        for name in numeric:
            spec = FEATURES_BY_NAME[name]
            transformed[name] = [_apply_transform(spec, r.get(name)) for r in rows]

        medians: dict[str, float] = {}
        missing_rates: dict[str, float] = {}
        pool: dict[str, list[float]] = {}
        indicators: list[str] = []
        n = max(len(rows), 1)

        for name in numeric:
            spec = FEATURES_BY_NAME[name]
            vals = [v for v in transformed[name] if v is not None]
            rate = 1.0 - (len(vals) / n)
            missing_rates[name] = rate
            medians[name] = float(np.median(vals)) if vals else 0.0
            if vals:
                sample = vals if len(vals) <= self.POOL_CAP else list(
                    rng.choice(np.asarray(vals), size=self.POOL_CAP, replace=False)
                )
                pool[name] = [float(v) for v in sample]
            if (
                spec.missing_policy is MissingPolicy.IMPUTE_WITH_INDICATOR
                and INDICATOR_RATE_BAND[0] <= rate <= INDICATOR_RATE_BAND[1]
            ):
                indicators.append(name)

        columns: list[str] = list(numeric)
        for name in numeric:
            if name in indicators:
                columns.append(f"{name}__missing")
        for fname, levels in nominal_levels.items():
            columns.extend(f"{fname}={lv}" for lv in levels)

        state = TransformerState(
            schema_version=FEATURE_SCHEMA_VERSION,
            columns=columns,
            numeric_features=numeric,
            nominal_levels=nominal_levels,
            nominal_reference=nominal_reference,
            medians=medians,
            indicators=indicators,
            missing_rates=missing_rates,
            means={c: 0.0 for c in columns},
            scales={c: 1.0 for c in columns},
            imputation_pool=pool,
            n_fit_rows=len(rows),
        )
        self.state = state

        # Standardisation constants are computed from the encoded training
        # matrix (median-imputed), so they reflect exactly what the model sees.
        raw = self._encode(rows, standardise=False)
        means = raw.mean(axis=0)
        scales = raw.std(axis=0, ddof=0)
        scales = np.where(scales < _SD_FLOOR, 1.0, scales)
        state.means = {c: float(m) for c, m in zip(columns, means, strict=True)}
        state.scales = {c: float(s) for c, s in zip(columns, scales, strict=True)}
        return self

    # ---- transformation ----------------------------------------------------

    def _require_state(self) -> TransformerState:
        if self.state is None:
            raise RuntimeError("FeatureTransformer used before fit().")
        return self.state

    def _encode_row(
        self,
        row: dict[str, Any],
        st: TransformerState,
        *,
        fill: dict[str, float] | None = None,
    ) -> list[float]:
        fill = fill or st.medians
        out: list[float] = []
        missing_flags: dict[str, float] = {}

        for name in st.numeric_features:
            spec = FEATURES_BY_NAME[name]
            v = _apply_transform(spec, row.get(name))
            if v is None:
                missing_flags[name] = 1.0
                v = fill.get(name, st.medians.get(name, 0.0))
            else:
                missing_flags[name] = 0.0
            out.append(float(v))

        for name in st.numeric_features:
            if name in st.indicators:
                out.append(missing_flags[name])

        for fname, levels in st.nominal_levels.items():
            value = row.get(fname)
            value = str(value) if value is not None else st.nominal_reference[fname]
            for lv in levels:
                out.append(1.0 if value == lv else 0.0)

        return out

    def _encode(self, rows: list[dict[str, Any]], *, standardise: bool = True) -> np.ndarray:
        st = self._require_state()
        matrix = np.asarray([self._encode_row(r, st) for r in rows], dtype=float)
        if matrix.size == 0:
            return matrix.reshape(0, len(st.columns))
        if standardise:
            mean = np.asarray([st.means[c] for c in st.columns])
            scale = np.asarray([st.scales[c] for c in st.columns])
            matrix = (matrix - mean) / scale
        return matrix

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Standardised design matrix, median-imputed."""
        return self._encode(rows, standardise=True)

    def transform_raw(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Design matrix in original (unstandardised) units."""
        return self._encode(rows, standardise=False)

    def transform_with_imputations(
        self,
        row: dict[str, Any],
        n_imputations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """``(n_imputations, n_columns)`` draws for one row.

        Each draw fills every missing numeric feature with an independent draw
        from that feature's observed training distribution. Features that are
        present are identical across draws, so between-draw spread measures
        exactly the contribution of what is *not* known.
        """
        st = self._require_state()
        missing = [
            name for name in st.numeric_features
            if _apply_transform(FEATURES_BY_NAME[name], row.get(name)) is None
        ]
        if not missing or n_imputations <= 1:
            return np.repeat(self.transform([row]), max(n_imputations, 1), axis=0)

        draws = []
        for _ in range(n_imputations):
            fill = dict(st.medians)
            for name in missing:
                pool = st.imputation_pool.get(name)
                if pool:
                    fill[name] = float(rng.choice(np.asarray(pool)))
            encoded = np.asarray(self._encode_row(row, st, fill=fill), dtype=float)
            draws.append(encoded)

        matrix = np.vstack(draws)
        mean = np.asarray([st.means[c] for c in st.columns])
        scale = np.asarray([st.scales[c] for c in st.columns])
        return (matrix - mean) / scale

    # ---- introspection -----------------------------------------------------

    @property
    def columns(self) -> list[str]:
        return list(self._require_state().columns)

    def reference_row_standardised(self) -> np.ndarray:
        """The cohort-average row in standardised space: the zero vector.

        Standardisation centres every column on the training mean, so the
        training-average program is exactly the origin. That makes the origin
        the natural comparison point for the additive log-odds decomposition
        in :mod:`trs.scoring.explanation`.
        """
        return np.zeros(len(self._require_state().columns), dtype=float)

    def column_origin(self, column: str) -> tuple[str, str]:
        """Map a design-matrix column back to (feature name, human label)."""
        self._require_state()  # guard: refuse to interpret columns before fit()
        if column.endswith("__missing"):
            base = column[: -len("__missing")]
            spec = FEATURES_BY_NAME[base]
            return base, f"{spec.label} — not reported"
        if "=" in column:
            base, level = column.split("=", 1)
            spec = FEATURES_BY_NAME[base]
            return base, f"{spec.label}: {level.replace('_', ' ')}"
        spec = FEATURES_BY_NAME.get(column)
        return column, spec.label if spec else column

    def unstandardise_delta(self, column: str, standardised_delta: float) -> float:
        st = self._require_state()
        return standardised_delta * st.scales.get(column, 1.0)
