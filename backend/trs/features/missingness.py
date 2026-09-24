"""Completeness metrics and the evidence-confidence grade.

Why a separate grade from the probability
-----------------------------------------
A calibrated 70% risk estimate built on a complete evidence package and a
calibrated 70% built on four data points are the same number and not the same
claim. The probability interval widens for the second (multiple imputation sees
to that), but interval width alone under-communicates: a user reading "68–74%"
versus "55–83%" may not register that the second rests on almost nothing.

So the platform reports two things side by side: the calibrated probability
with its interval, and an explicit **evidence-confidence grade** derived from
observable properties of the evidence base. The grade never modifies the
probability. It qualifies it.

Weighting
---------
Features are weighted by ``FeatureSpec.completeness_weight``, so "80% complete"
means "80% of the evidence that actually moves this estimate is present", not
"80% of table columns are non-null". Without weighting, a program missing its
entire PK/PD bridge but carrying twelve well-documented animal studies would
score as highly complete, which would be actively misleading.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from trs.features.registry import FEATURES, features_in_domain
from trs.features.spec import DOMAIN_LABELS, FeatureDomain, MissingPolicy
from trs.ontology.enums import ConfidenceLevel, MissingnessReason

#: Domains whose absence most damages a translational estimate. A program with
#: no PK/PD bridge and no human-derived evidence cannot be assessed with
#: confidence however complete the rest of the record is.
CRITICAL_DOMAINS: tuple[FeatureDomain, ...] = (
    FeatureDomain.PKPD_TRANSLATION,
    FeatureDomain.TARGET_VALIDATION,
    FeatureDomain.HUMAN_BIOLOGICAL_SUPPORT,
    FeatureDomain.ANIMAL_MODEL_VALIDITY,
)

#: Grade thresholds on the weighted completeness score. Deliberately demanding:
#: "high" requires that essentially the whole schema is populated, because the
#: schema was written to contain only things that matter.
GRADE_THRESHOLDS: tuple[tuple[float, ConfidenceLevel], ...] = (
    (0.85, ConfidenceLevel.HIGH),
    (0.65, ConfidenceLevel.MODERATE),
    (0.40, ConfidenceLevel.LOW),
    (0.00, ConfidenceLevel.VERY_LOW),
)

#: Missingness reasons that should not count against a program. "Not
#: applicable" means the question does not arise; penalising it would mark, for
#: example, every monotherapy as incomplete for lacking combination data.
NON_PENALISED_REASONS: frozenset[str] = frozenset({
    MissingnessReason.NOT_APPLICABLE,
    MissingnessReason.PRESENT,
})


@dataclasses.dataclass(frozen=True, slots=True)
class CompletenessReport:
    overall: float
    by_domain: dict[str, float]
    n_present: int
    n_total: int
    n_not_applicable: int
    #: Domains below the critical threshold, for explicit UI callout.
    critical_gaps: tuple[str, ...]
    #: {reason: count} across all features.
    reason_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_domain": self.by_domain,
            "n_present": self.n_present,
            "n_total": self.n_total,
            "n_not_applicable": self.n_not_applicable,
            "critical_gaps": list(self.critical_gaps),
            "reason_counts": self.reason_counts,
        }


def compute_completeness(
    missingness: dict[str, str],
    *,
    critical_domain_threshold: float = 0.5,
) -> CompletenessReport:
    """Weighted completeness overall and per domain."""
    present_w = 0.0
    total_w = 0.0
    n_present = 0
    n_na = 0
    reason_counts: dict[str, int] = {}
    by_domain: dict[str, float] = {}

    for spec in FEATURES:
        reason = missingness.get(spec.name, MissingnessReason.NOT_EXTRACTED)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if reason == MissingnessReason.PRESENT:
            n_present += 1
        if reason == MissingnessReason.NOT_APPLICABLE:
            n_na += 1

        # ZERO_IS_ABSENCE features are always "present" by construction, so
        # counting them would inflate every score toward a constant floor.
        if spec.missing_policy is MissingPolicy.ZERO_IS_ABSENCE:
            continue
        if reason in NON_PENALISED_REASONS and reason != MissingnessReason.PRESENT:
            continue

        total_w += spec.completeness_weight
        if reason == MissingnessReason.PRESENT:
            present_w += spec.completeness_weight

    for domain in FeatureDomain:
        d_present = 0.0
        d_total = 0.0
        for spec in features_in_domain(domain):
            if spec.missing_policy is MissingPolicy.ZERO_IS_ABSENCE:
                continue
            reason = missingness.get(spec.name, MissingnessReason.NOT_EXTRACTED)
            if reason in NON_PENALISED_REASONS and reason != MissingnessReason.PRESENT:
                continue
            d_total += spec.completeness_weight
            if reason == MissingnessReason.PRESENT:
                d_present += spec.completeness_weight
        by_domain[str(domain)] = (d_present / d_total) if d_total else 1.0

    gaps = tuple(
        str(d) for d in CRITICAL_DOMAINS
        if by_domain.get(str(d), 1.0) < critical_domain_threshold
    )

    return CompletenessReport(
        overall=(present_w / total_w) if total_w else 0.0,
        by_domain=by_domain,
        n_present=n_present,
        n_total=len(FEATURES),
        n_not_applicable=n_na,
        critical_gaps=gaps,
        reason_counts=reason_counts,
    )


def grade_evidence_confidence(
    report: CompletenessReport,
    *,
    n_evidence_items: int,
    interval_width: float | None = None,
    contains_synthetic: bool = False,
) -> tuple[ConfidenceLevel, list[str]]:
    """Assign an evidence-confidence grade and state the reasons for it.

    The grade starts from weighted completeness and is then *only ever
    downgraded* by the additional checks below. Nothing upgrades it: a thin
    evidence base cannot be rescued by a narrow interval, because a narrow
    interval on thin evidence is a property of the model, not the evidence.
    """
    grade = ConfidenceLevel.VERY_LOW
    for threshold, level in GRADE_THRESHOLDS:
        if report.overall >= threshold:
            grade = level
            break

    order = [
        ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW,
        ConfidenceLevel.MODERATE, ConfidenceLevel.HIGH,
    ]
    reasons: list[str] = [
        f"Weighted evidence completeness {report.overall:.0%}."
    ]

    def downgrade(steps: int, why: str) -> None:
        nonlocal grade
        idx = max(0, order.index(grade) - steps)
        if order[idx] != grade:
            reasons.append(why)
            grade = order[idx]
        else:
            reasons.append(why)

    if report.critical_gaps:
        labels = ", ".join(
            DOMAIN_LABELS[FeatureDomain(d)] for d in report.critical_gaps
        )
        downgrade(1, f"Critical evidence domain(s) largely absent: {labels}.")

    if n_evidence_items < 5:
        downgrade(1, f"Only {n_evidence_items} evidence item(s) available at the cutoff.")
    elif n_evidence_items < 12:
        reasons.append(f"Modest evidence base ({n_evidence_items} items).")

    if interval_width is not None and interval_width > 0.35:
        reasons.append(
            f"Wide uncertainty interval ({interval_width:.0%} absolute) reflects "
            "limited information in the inputs."
        )

    if contains_synthetic:
        grade = ConfidenceLevel.VERY_LOW
        reasons.insert(
            0,
            "Estimate depends on synthetic demonstration data and carries no "
            "scientific interpretation.",
        )

    return grade, reasons
