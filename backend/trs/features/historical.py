"""Historical class-performance features, computed strictly as of a cutoff.

Why this module is dangerous and therefore isolated
---------------------------------------------------
"How often has this target failed before?" is one of the most predictive
features available — and the easiest way to destroy a model's validity. Three
distinct leaks are possible:

1. **Future outcomes.** Counting a 2019 failure when scoring a 2014 decision.
2. **Self-inclusion.** Counting the focal program's own outcome in its own
   feature, which makes the label perfectly recoverable from the features.
3. **Fold contamination.** In grouped cross-validation, a test-fold program
   contributing to a training-fold program's historical rate — and vice versa.

Leaks 1 and 2 are handled here, unconditionally: the index only admits outcomes
whose ``outcome_date`` precedes the cutoff, and every query excludes the focal
track and its sibling tracks from the same program.

Leak 3 is real but is deliberately *not* prevented, because preventing it would
be wrong. At the historical decision point the analyst genuinely did know about
other companies' resolved programs, including ones that end up in another CV
fold. Removing them would make the training distribution differ from the
deployment distribution. What matters is that no *future* information enters,
and that the focal program's own fate never does. The consequence — that
grouped folds are not fully independent through this channel — is recorded in
VALIDATION_PLAN.md, and the validation suite reports metrics with these
features ablated so their contribution is visible.

Shrinkage
---------
Raw group rates are catastrophic with small counts: one prior program on a
target gives a rate of 0.0 or 1.0. Rates are therefore beta-binomial shrunk
toward the as-of global base rate with a fixed prior strength, so a group with
one observation barely moves off the base rate and a group with fifty is
trusted.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections import defaultdict
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from trs.db.models.core import Program, ProgramIndication

#: Beta prior strength in pseudo-observations. With m = 10, a group needs ten
#: resolved prior programs to move halfway from the base rate to its own rate.
#: Chosen to be conservative: translational cohorts are small and per-target
#: counts are usually in the single digits.
PRIOR_STRENGTH = 10.0

#: Fallback base rate used only when no resolved outcomes exist before the
#: cutoff at all. 0.5 is the maximum-entropy choice; it deliberately carries no
#: claim about real-world Phase II attrition.
UNINFORMED_BASE_RATE = 0.5


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedOutcome:
    """A prior program outcome usable for historical rates."""

    track_id: str
    program_id: str
    target_id: str | None
    indication_id: str
    mechanism_key: str | None
    outcome_date: dt.date
    failed: bool


def shrunk_rate(n_failed: int, n_total: int, base_rate: float,
                prior_strength: float = PRIOR_STRENGTH) -> float:
    """Beta-binomial posterior mean for a group failure rate.

    ``(failures + m*p0) / (total + m)`` — the posterior mean under a
    ``Beta(m*p0, m*(1-p0))`` prior centred on the as-of base rate.
    """
    alpha = prior_strength * base_rate
    beta = prior_strength * (1.0 - base_rate)
    return (n_failed + alpha) / (n_total + alpha + beta)


class HistoricalRateIndex:
    """In-memory index of resolved outcomes, queryable as of any date.

    Built once per cohort build or scoring batch. Queries are pure functions of
    (cutoff, group key, excluded program) so the same inputs always give the
    same answer — a requirement for reproducing a historical prediction.
    """

    def __init__(
        self,
        outcomes: list[ResolvedOutcome],
        *,
        failure_label_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self._outcomes = sorted(outcomes, key=lambda o: o.outcome_date)
        self._by_target: dict[str, list[ResolvedOutcome]] = defaultdict(list)
        self._by_indication: dict[str, list[ResolvedOutcome]] = defaultdict(list)
        self._by_mechanism: dict[str, list[ResolvedOutcome]] = defaultdict(list)
        for o in self._outcomes:
            if o.target_id:
                self._by_target[o.target_id].append(o)
            self._by_indication[o.indication_id].append(o)
            if o.mechanism_key:
                self._by_mechanism[o.mechanism_key].append(o)

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        failure_labels: frozenset[str],
        success_labels: frozenset[str],
        mechanism_key_fn: Callable[[Program], str | None],
    ) -> HistoricalRateIndex:
        """Build from every track in the database that has a resolved outcome.

        Only outcomes classified as a failure or a success contribute. Tracks
        that are still active, censored, or discontinued for non-scientific
        reasons are excluded entirely — they are not evidence about whether the
        biology works, and counting them as either would bias every rate.
        """
        stmt = (
            select(ProgramIndication, Program)
            .join(Program, Program.id == ProgramIndication.program_id)
            .where(ProgramIndication.outcome_date.is_not(None))
        )
        outcomes: list[ResolvedOutcome] = []
        for track, program in session.execute(stmt):
            label = track.outcome_label
            if label in failure_labels:
                failed = True
            elif label in success_labels:
                failed = False
            else:
                continue
            outcomes.append(
                ResolvedOutcome(
                    track_id=track.id,
                    program_id=track.program_id,
                    target_id=program.primary_target_id,
                    indication_id=track.indication_id,
                    mechanism_key=mechanism_key_fn(program),
                    outcome_date=track.outcome_date,
                    failed=failed,
                )
            )
        return cls(outcomes)

    # ---- queries -----------------------------------------------------------

    def base_rate(self, as_of: dt.date, exclude_program_id: str | None = None) -> tuple[float, int]:
        """As-of global failure rate and the count it rests on."""
        rows = [
            o for o in self._outcomes
            if o.outcome_date <= as_of and o.program_id != exclude_program_id
        ]
        if not rows:
            return UNINFORMED_BASE_RATE, 0
        return sum(o.failed for o in rows) / len(rows), len(rows)

    def _group_stats(
        self,
        bucket: dict[str, list[ResolvedOutcome]],
        key: str | None,
        as_of: dt.date,
        exclude_program_id: str | None,
    ) -> tuple[int, int]:
        if key is None:
            return 0, 0
        rows = [
            o for o in bucket.get(key, [])
            if o.outcome_date <= as_of and o.program_id != exclude_program_id
        ]
        return sum(o.failed for o in rows), len(rows)

    def target_rate(
        self, target_id: str | None, as_of: dt.date, exclude_program_id: str | None
    ) -> tuple[float, int]:
        base, _ = self.base_rate(as_of, exclude_program_id)
        n_failed, n_total = self._group_stats(self._by_target, target_id, as_of, exclude_program_id)
        return shrunk_rate(n_failed, n_total, base), n_total

    def mechanism_rate(
        self, mechanism_key: str | None, as_of: dt.date, exclude_program_id: str | None
    ) -> tuple[float, int]:
        base, _ = self.base_rate(as_of, exclude_program_id)
        n_failed, n_total = self._group_stats(
            self._by_mechanism, mechanism_key, as_of, exclude_program_id
        )
        return shrunk_rate(n_failed, n_total, base), n_total

    def indication_rate(
        self, indication_id: str | None, as_of: dt.date, exclude_program_id: str | None
    ) -> tuple[float, int]:
        base, _ = self.base_rate(as_of, exclude_program_id)
        n_failed, n_total = self._group_stats(
            self._by_indication, indication_id, as_of, exclude_program_id
        )
        return shrunk_rate(n_failed, n_total, base), n_total

    def __len__(self) -> int:
        return len(self._outcomes)


def mechanism_key_for(program: Program) -> str | None:
    """Coarse "same bet" key: target class crossed with modality group.

    Deliberately coarser than the target id, so that a CDK4/6 inhibitor and a
    CDK7 inhibitor inform each other while a CDK4/6 inhibitor and a CDK4/6
    antibody-drug conjugate do not.
    """
    from trs.features.spec import MODALITY_GROUPS, TARGET_CLASS_GROUPS

    target_class = getattr(getattr(program, "target", None), "target_class", None) or "unknown"
    tc = TARGET_CLASS_GROUPS.get(target_class, "other")
    mg = MODALITY_GROUPS.get(program.modality, "other")
    return f"{tc}|{mg}"
