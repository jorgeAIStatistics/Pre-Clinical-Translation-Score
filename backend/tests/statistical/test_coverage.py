"""Tests for the interval-coverage diagnostic.

The diagnostic's own first version reported 96.1% coverage against a nominal
95% and called it reassuring — while the median interval was 0.66 wide on a
[0, 1] scale, the mean prediction sat 0.128 above the truth, and every miss fell
on the same side. These tests pin the qualifiers that stop coverage being read
as accuracy, because a metric that flatters its own model is the failure this
codebase exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from trs.validation.coverage import (
    BIAS_MATERIAL,
    WIDTH_UNINFORMATIVE,
    interval_coverage,
)


def test_wide_intervals_do_not_earn_a_reassuring_verdict() -> None:
    """If this breaks, an interval of [0, 1] reports as well calibrated."""
    n = 200
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, n)
    predicted = truth.copy()
    lo = np.clip(predicted - 0.45, 0, 1)
    hi = np.clip(predicted + 0.45, 0, 1)

    res = interval_coverage(predicted, lo, hi, truth)
    assert res.coverage == pytest.approx(1.0)
    assert res.median_interval_width >= WIDTH_UNINFORMATIVE
    # Coverage is ~100%, which is *outside* the nominal band, so this lands on
    # over_covers — the point is that it is never "near_nominal".
    assert res.verdict != "near_nominal"


def test_nominal_coverage_with_wide_intervals_is_flagged_uninformative() -> None:
    """If this breaks, coverage achieved by width reads as coverage earned."""
    rng = np.random.default_rng(1)
    n = 200
    truth = rng.uniform(0.3, 0.7, n)
    predicted = truth + 0.12                      # systematically high
    lo = np.clip(predicted - 0.33, 0, 1)
    hi = np.clip(predicted + 0.33, 0, 1)
    # Push a few truths below the band so misses are one-sided.
    truth[:5] = np.clip(lo[:5] - 0.05, 0, 1)

    res = interval_coverage(predicted, lo, hi, truth)
    assert abs(res.coverage - 0.95) <= 0.10
    assert res.verdict == "nominal_but_uninformative"
    assert "should not be read as reassurance" in res.interpretation
    assert "wide on a scale of 0 to 1" in res.interpretation
    assert res.missed_high == 0 and res.missed_low >= 3


def test_tight_unbiased_intervals_earn_near_nominal() -> None:
    """If this breaks, the diagnostic can never return a clean result.

    A check that always complains is as useless as one that never does.
    """
    rng = np.random.default_rng(2)
    n = 400
    truth = rng.uniform(0.3, 0.7, n)
    noise = rng.normal(0.0, 0.02, n)
    predicted = np.clip(truth + noise, 0, 1)
    lo = np.clip(predicted - 0.045, 0, 1)
    hi = np.clip(predicted + 0.045, 0, 1)

    res = interval_coverage(predicted, lo, hi, truth)
    assert res.median_interval_width < WIDTH_UNINFORMATIVE
    assert abs(res.mean_bias) < BIAS_MATERIAL
    assert res.verdict == "near_nominal", res.interpretation


def test_narrow_intervals_report_under_coverage_and_name_the_missing_term() -> None:
    """If this breaks, an over-confident interval passes silently."""
    rng = np.random.default_rng(3)
    n = 300
    truth = rng.uniform(0.2, 0.8, n)
    predicted = np.clip(truth + rng.normal(0, 0.15, n), 0, 1)
    lo = np.clip(predicted - 0.01, 0, 1)
    hi = np.clip(predicted + 0.01, 0, 1)

    res = interval_coverage(predicted, lo, hi, truth)
    assert res.coverage < 0.5
    assert res.verdict == "under_covers"
    assert "model-misspecification" in res.interpretation


def test_coverage_interval_widens_on_small_samples() -> None:
    """If this breaks, a coverage estimate from 10 rows reads as precise."""
    truth = np.full(10, 0.5)
    predicted = np.full(10, 0.5)
    lo = np.full(10, 0.4)
    hi = np.full(10, 0.6)
    small = interval_coverage(predicted, lo, hi, truth)

    big_truth = np.full(500, 0.5)
    big = interval_coverage(
        np.full(500, 0.5), np.full(500, 0.4), np.full(500, 0.6), big_truth
    )
    assert small.coverage == big.coverage == pytest.approx(1.0)
    small_width = small.coverage_ci_high - small.coverage_ci_low
    big_width = big.coverage_ci_high - big.coverage_ci_low
    assert small_width > big_width * 3, (
        "the Wilson interval did not widen on the smaller sample"
    )


def test_misaligned_inputs_are_rejected() -> None:
    """If this breaks, a silent broadcast produces a meaningless coverage."""
    with pytest.raises(ValueError, match="align"):
        interval_coverage([0.5, 0.6], [0.4], [0.7, 0.8], [0.5, 0.5])
    with pytest.raises(ValueError, match="no rows"):
        interval_coverage([], [], [], [])
