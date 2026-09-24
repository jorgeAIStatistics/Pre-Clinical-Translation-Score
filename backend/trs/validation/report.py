"""Assembling the validation report shown on the Methods page.

The report is written to be read by someone looking for a reason not to trust
the model. It therefore leads with the things most likely to be wrong —
calibration, subgroup behaviour, the honesty of the uncertainty intervals, the
size of the effective sample — and treats discrimination as one line item among
several rather than the headline.

Four checks here exist specifically to catch self-deception:

``coefficient_sign_audit``
    Flags coefficients whose sign contradicts the prior stated in the feature
    registry. A sign flip is not automatically wrong — it may be a real
    finding, or confounding, or a data bug — but it must be seen, not averaged
    into an aggregate.

``interval_agreement``
    Re-derives prediction-interval widths by nonparametric bootstrap refit and
    compares them with the analytic Laplace widths. The Laplace approximation
    is asymptotic; this is the check on whether that asymptotics has arrived at
    this sample size.

``ablation``
    Refits with whole feature domains removed. If performance is unchanged
    without the PK/PD domain, then the PK/PD narrative in the UI is decoration.
    The historical-performance domain is ablated specifically because it is the
    one most exposed to residual leakage.

``subgroup``
    Performance by modality, target class, evidence completeness and era. An
    aggregate metric can hide a model that works for small molecules and is
    noise for cell therapies.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from trs.features.registry import FEATURES_BY_NAME, features_in_domain
from trs.features.spec import DOMAIN_LABELS, Direction, FeatureDomain
from trs.validation.metrics import evaluate
from trs.validation.splitters import component_bootstrap_indices

#: Minimum rows before a subgroup metric is reported at all. Below this the
#: estimate is noise and publishing it invites over-reading.
MIN_SUBGROUP_N = 15


def coefficient_sign_audit(
    coefficients: Sequence[dict[str, Any]], transformer: Any
) -> dict[str, Any]:
    """Compare fitted coefficient signs with the registry's prior directions.

    Only ``UNCERTAIN`` features are exempt. Missing-indicator and one-hot
    columns carry no directional prior and are skipped.
    """
    discordant: list[dict[str, Any]] = []
    checked = 0
    for row in coefficients:
        column = row["column"]
        if column.endswith("__missing") or "=" in column:
            continue
        spec = FEATURES_BY_NAME.get(column)
        if spec is None or spec.expected_direction is Direction.UNCERTAIN:
            continue
        checked += 1
        coef = row["coefficient"]
        expected_positive = spec.expected_direction is Direction.INCREASES_RISK
        if (coef > 0) != expected_positive and abs(coef) > 1e-6:
            # A coefficient whose interval straddles zero is not a real
            # discordance, just an unresolved one.
            ci_low, ci_high = row.get("ci_low"), row.get("ci_high")
            straddles = (
                ci_low is not None and ci_high is not None and ci_low < 0 < ci_high
            )
            discordant.append({
                "feature": column,
                "label": spec.label,
                "domain": str(spec.domain),
                "expected_direction": str(spec.expected_direction),
                "coefficient": coef,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "credible_interval_excludes_zero": (not straddles) if ci_low is not None else None,
            })
    material = [d for d in discordant if d.get("credible_interval_excludes_zero")]
    return {
        "n_checked": checked,
        "n_discordant": len(discordant),
        "n_discordant_credible": len(material),
        "discordant": discordant,
        "interpretation": (
            "A sign opposite to the pre-registered expectation is a prompt for "
            "human review, not evidence of a defect. Only discordances whose "
            "credible interval excludes zero are treated as material."
        ),
    }


def subgroup_metrics(
    y: np.ndarray,
    p: np.ndarray,
    components: np.ndarray,
    subgroups: dict[str, Sequence[Any]],
    *,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    """Metrics within each level of each supplied subgroup variable."""
    out: dict[str, Any] = {}
    y = np.asarray(y)
    p = np.asarray(p)
    for var_name, labels in subgroups.items():
        labels_arr = np.asarray([str(v) if v is not None else "unknown" for v in labels])
        levels: list[dict[str, Any]] = []
        for level in sorted(set(labels_arr.tolist())):
            mask = labels_arr == level
            n = int(mask.sum())
            entry: dict[str, Any] = {"level": level, "n": n, "n_positive": int(y[mask].sum())}
            if n < MIN_SUBGROUP_N or len(np.unique(y[mask])) < 2:
                entry["reported"] = False
                entry["reason"] = (
                    f"Fewer than {MIN_SUBGROUP_N} programs or only one outcome class; "
                    "metrics would not be interpretable."
                )
            else:
                m = evaluate(
                    y[mask], p[mask], components=components[mask],
                    n_bootstrap=n_bootstrap, thresholds=(0.5,),
                )
                entry["reported"] = True
                entry["auroc"] = m["discrimination"]["auroc"]
                entry["brier"] = m["accuracy"]["brier"]
                entry["scaled_brier"] = m["accuracy"]["scaled_brier"]
                entry["calibration_slope"] = m["calibration"]["calibration_slope"]
                entry["calibration_intercept"] = m["calibration"]["calibration_intercept"]
            levels.append(entry)
        out[var_name] = {"levels": levels, "min_n_for_reporting": MIN_SUBGROUP_N}
    return out


def completeness_tertiles(scores: Sequence[float]) -> list[str]:
    """Label rows by evidence-completeness tertile for subgroup reporting."""
    arr = np.asarray(scores, dtype=float)
    if arr.size < 3:
        return ["all"] * arr.size
    q1, q2 = np.quantile(arr, [1 / 3, 2 / 3])
    return [
        "low_completeness" if v <= q1 else ("mid_completeness" if v <= q2 else "high_completeness")
        for v in arr
    ]


def era_labels(dates: Sequence[dt.date], *, n_eras: int = 3) -> list[str]:
    arr = np.asarray([d.toordinal() for d in dates], dtype=float)
    if arr.size < n_eras * MIN_SUBGROUP_N:
        return ["all_years"] * arr.size
    edges = np.quantile(arr, np.linspace(0, 1, n_eras + 1))
    labels = []
    for v in arr:
        for i in range(n_eras):
            if v <= edges[i + 1] or i == n_eras - 1:
                lo = dt.date.fromordinal(int(edges[i]))
                hi = dt.date.fromordinal(int(edges[i + 1]))
                labels.append(f"{lo.year}–{hi.year}")
                break
    return labels


def interval_agreement(
    model_factory: Callable[[], Any],
    transformer: Any,
    calibrator_factory: Callable[[], Any] | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    components_train: np.ndarray,
    X_probe: np.ndarray,
    *,
    n_refits: int = 40,
    level: float = 0.95,
    random_state: int = 0,
) -> dict[str, Any]:
    """Cross-check analytic Laplace intervals against a bootstrap refit.

    The Laplace posterior is a second-order approximation that is exact only
    asymptotically. At a few hundred programs that is an assumption, not a
    fact, so it is tested: the model is refit on ``n_refits`` component-level
    bootstrap resamples, the spread of predictions for each probe row is
    measured, and the resulting interval width is compared with the analytic
    one.

    Agreement within roughly ±25% is treated as acceptable. Systematically
    narrower analytic intervals would mean the published uncertainty is
    optimistic, which the report states plainly rather than filing under
    diagnostics.
    """
    rng = np.random.default_rng(random_state)
    boot_logits: list[np.ndarray] = []
    for _ in range(n_refits):
        idx = component_bootstrap_indices(components_train, rng)
        if idx.size < 20 or len(np.unique(y_train[idx])) < 2:
            continue
        try:
            m = model_factory().fit(X_train[idx], y_train[idx], columns=list(transformer.columns))
            boot_logits.append(np.asarray(m.predict_logit(X_probe)))
        except Exception:  # noqa: BLE001 - a failed refit is a data point, not a crash
            continue

    if len(boot_logits) < 10:
        return {
            "available": False,
            "reason": f"Only {len(boot_logits)} usable bootstrap refits; comparison skipped.",
        }

    boot = np.vstack(boot_logits)
    alpha = (1 - level) / 2
    boot_width = np.quantile(boot, 1 - alpha, axis=0) - np.quantile(boot, alpha, axis=0)

    fitted = model_factory().fit(X_train, y_train, columns=list(transformer.columns))
    analytic = fitted.posterior_logit_samples(X_probe, 2000, np.random.default_rng(1))
    analytic_width = np.quantile(analytic, 1 - alpha, axis=0) - np.quantile(analytic, alpha, axis=0)

    ratio = np.where(boot_width > 1e-9, analytic_width / boot_width, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    median_ratio = float(np.median(ratio)) if ratio.size else float("nan")

    if not np.isfinite(median_ratio):
        verdict = "indeterminate"
    elif 0.75 <= median_ratio <= 1.25:
        verdict = "acceptable"
    elif median_ratio < 0.75:
        verdict = "analytic_intervals_too_narrow"
    else:
        verdict = "analytic_intervals_too_wide"

    return {
        "available": True,
        "n_refits": len(boot_logits),
        "n_probe_rows": int(X_probe.shape[0]),
        "median_width_ratio_analytic_over_bootstrap": median_ratio,
        "p10_ratio": float(np.quantile(ratio, 0.10)) if ratio.size else None,
        "p90_ratio": float(np.quantile(ratio, 0.90)) if ratio.size else None,
        "verdict": verdict,
        "interpretation": (
            "Ratio of the analytic Laplace interval width to a component-level "
            "bootstrap-refit width on the log-odds scale. A ratio near 1 means "
            "the analytic approximation is adequate at this sample size; below "
            "0.75 means published intervals are too narrow."
        ),
    }


def domain_ablation(
    fit_and_score: Callable[[frozenset[str]], dict[str, Any]],
    *,
    domains: Sequence[FeatureDomain] | None = None,
) -> dict[str, Any]:
    """Refit with each feature domain removed, reporting the performance cost.

    ``fit_and_score(excluded_columns) -> dict`` is supplied by the caller so
    this function stays agnostic about how the model is built.
    """
    domains = tuple(domains or list(FeatureDomain))
    baseline = fit_and_score(frozenset())
    results: list[dict[str, Any]] = []
    for domain in domains:
        names = {f.name for f in features_in_domain(domain)}
        scores = fit_and_score(frozenset(names))
        delta_ll = (
            scores["log_loss"] - baseline["log_loss"]
            if scores.get("log_loss") is not None and baseline.get("log_loss") is not None
            else None
        )
        delta_auc = (
            (scores["auroc"] - baseline["auroc"])
            if scores.get("auroc") is not None and baseline.get("auroc") is not None
            else None
        )
        results.append({
            "domain": str(domain),
            "domain_label": DOMAIN_LABELS[domain],
            "n_features_removed": len(names),
            "log_loss": scores.get("log_loss"),
            "auroc": scores.get("auroc"),
            "delta_log_loss_vs_full": delta_ll,
            "delta_auroc_vs_full": delta_auc,
        })
    results.sort(key=lambda r: (r["delta_log_loss_vs_full"] is None, -(r["delta_log_loss_vs_full"] or 0)))
    return {
        "baseline": baseline,
        "by_domain": results,
        "interpretation": (
            "Change in held-out performance when a whole evidence domain is "
            "removed from the model. A domain whose removal costs nothing is "
            "not contributing to the estimate, however prominent it is in the "
            "interface."
        ),
    }


def assemble_report(
    *,
    split_description: str,
    split_sizes: dict[str, int],
    leakage_audit_result: dict[str, Any],
    component_summary: dict[str, Any],
    test_metrics: dict[str, Any],
    calibration_selection: dict[str, Any],
    subgroups: dict[str, Any] | None = None,
    sign_audit: dict[str, Any] | None = None,
    interval_check: dict[str, Any] | None = None,
    ablation: dict[str, Any] | None = None,
    model_comparison: dict[str, Any] | None = None,
    cohort_summary: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Compose the canonical validation-report document."""
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "evaluation_design": {
            "split": split_description,
            "sizes": split_sizes,
            "leakage_audit": leakage_audit_result,
            "grouping": component_summary,
            "statement": (
                "Held-out performance is reported on a temporally forward split "
                "with connected-component grouping. No random train/test split "
                "is reported anywhere in this document, because on this problem "
                "it would be optimistic by an unquantifiable margin."
            ),
        },
        "cohort": cohort_summary or {},
        "primary_metrics": test_metrics,
        "calibration_selection": calibration_selection,
        "subgroup_performance": subgroups or {},
        "coefficient_sign_audit": sign_audit or {},
        "interval_agreement": interval_check or {},
        "domain_ablation": ablation or {},
        "model_comparison": model_comparison or {},
        "limitations": limitations or [],
    }
