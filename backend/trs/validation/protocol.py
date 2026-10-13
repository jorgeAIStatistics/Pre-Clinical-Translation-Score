"""The declared development/evaluation protocol, and the manifest that fixes it.

Why this module exists
----------------------
The pipeline this replaces split the cohort three ways, scored every candidate
family on the *test* partition, chose the winner using those test labels, and
then reported the winner's performance on the same rows. That makes the test
partition a model-selection partition. Nothing about the arithmetic was wrong;
the number it produced simply did not mean what the documentation said it meant.

A second, quieter problem: the split boundary was drawn on the *assessment*
date alone. Feature availability was enforced to the day by
:class:`trs.features.temporal.AsOfContext`, but a program assessed in 2008
whose outcome resolved in 2019 was still used to fit a model described as
reflecting a historical origin. Features were as-of; labels were not.

So this module declares three things that were previously implicit, and makes
all three checkable:

**One locked evaluation partition.** Chosen before any model is fitted, read
exactly once, by exactly one function. Every tuning, family-selection and
calibration decision happens on development data. A test asserts the property
directly: permute the evaluation labels and the fitted artifact must not change
by one bit.

**Label maturity at the fitting origin.** A development row is admissible only
if its outcome was already knowable at the origin the fit claims to sit at. An
early program-entry date is not sufficient. This is expensive — the median
outcome lag on the demonstration cohort is 5.3 years, so enforcing it roughly
halves the development sample — and the cost is recorded rather than avoided.

**A manifest.** Row identifiers, component identifiers, partition membership,
every date boundary, every eligibility rule, every purged row with its reason,
and a hash over all of it. Two runs that claim the same protocol can be
compared only if their manifest hashes agree; a run that reports a strict
temporal headline cannot quietly have included pre-boundary evaluation rows,
because the manifest would say so.

What this module deliberately does not do
-----------------------------------------
It does not silently fall back. When the data cannot support the strict
protocol, :func:`plan_split` raises, and the caller must ask for the weaker
protocol *by name* — which then travels into the artifact, the API and the
model card as a named limitation rather than as a footnote nobody reads.

Nor does it pretend that an independently seeded synthetic holdout establishes
real-world validity. It establishes that the implementation does what it says.
That is worth having and it is not the same claim.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np

from trs.validation.splitters import GROUP_SCOPE_KEYS, GroupScope


class Protocol(StrEnum):
    """Named evaluation protocols. The name travels with every number."""

    #: Locked evaluation partition; development rows label-mature at the
    #: fitting origin; inner folds forward-chaining with fold-local
    #: preprocessing. The only protocol permitted to describe itself as
    #: temporal deployment performance.
    STRICT_TEMPORAL = "strict_temporal"

    #: Locked evaluation partition and fold-local preprocessing, but label
    #: maturity enforced only at the outer fitting origin, not within each
    #: inner fold. Used when per-fold maturity leaves folds too small to fit.
    #: Carries an explicit limitation.
    TEMPORAL_OUTER_MATURITY = "temporal_outer_maturity"

    #: Locked evaluation partition and fold-local preprocessing, but no label
    #: maturity at all: the boundary is drawn on assessment date only. This is
    #: ordinary retrospective evaluation. It may not be described as deployment
    #: performance.
    RETROSPECTIVE = "retrospective"

    #: What the pipeline did before this module existed: selection and final
    #: evaluation on the same partition. Retained only so historical artifacts
    #: can be read back and labelled honestly. Never produced by new runs.
    LEGACY_EXPLORATORY = "legacy_exploratory"


#: Protocols whose numbers may be presented as temporal deployment estimates.
DEPLOYMENT_CLAIM_PERMITTED: frozenset[str] = frozenset({Protocol.STRICT_TEMPORAL})

#: Whether each protocol enforces label maturity at the outer fitting origin.
PROTOCOL_OUTER_MATURITY: dict[str, bool] = {
    Protocol.STRICT_TEMPORAL: True,
    Protocol.TEMPORAL_OUTER_MATURITY: True,
    Protocol.RETROSPECTIVE: False,
}

#: Whether each protocol additionally enforces label maturity at *each inner
#: fold's own* origin. This is the expensive one. On the demonstration cohort it
#: is infeasible — see :func:`plan_training` — and the measurement that shows
#: why is recorded in the rejection trace rather than hidden.
PROTOCOL_FOLD_MATURITY: dict[str, bool] = {
    Protocol.STRICT_TEMPORAL: True,
    Protocol.TEMPORAL_OUTER_MATURITY: False,
    Protocol.RETROSPECTIVE: False,
}

#: A protocol needs at least this many usable forward-chaining folds before its
#: hyperparameter and family choices mean anything.
MIN_USABLE_FOLDS = 3

#: Below this many development rows, no protocol is worth running: the encoded
#: design has ~110 columns and a fit on fewer rows than this is a curiosity.
MIN_DEVELOPMENT_ROWS = 120

#: Below this many evaluation rows, a held-out metric's confidence interval is
#: wider than any difference it could detect. Reported, not hidden.
MIN_EVALUATION_ROWS = 60

#: Minimum rows and events for one forward-chaining fold to be usable.
MIN_FOLD_TRAIN_ROWS = 40
MIN_FOLD_TEST_ROWS = 10


class ProtocolError(RuntimeError):
    """The requested protocol cannot be satisfied by the data supplied."""


@dataclasses.dataclass(frozen=True, slots=True)
class ProtocolDates:
    """The four dates a retrospective fit needs to keep straight.

    Conflating any two of these is how a leakage-free-looking pipeline leaks.

    ``assessment_end``
        The latest *as-of* date admitted to development. Features for every
        development row were frozen at or before its own as-of date; this is
        the boundary between development and evaluation rows.

    ``fitting_origin``
        The date the fit claims to sit at. Equal to ``assessment_end`` for the
        temporal protocols: the model is described as "what could have been
        fitted on this date".

    ``label_cutoff``
        Outcomes must have been knowable by this date to be used in
        development. Equal to ``fitting_origin`` under
        :attr:`Protocol.STRICT_TEMPORAL`.

    ``data_cutoff``
        Nothing resolving after this date is admissible anywhere, including in
        the evaluation partition. On a real corpus this is the extraction date.
        On the demonstration corpus it matters: the generator assigns outcome
        dates up to 2028, so 24 of 552 cohort rows carry labels for events that
        have not happened.
    """

    assessment_end: dt.date
    fitting_origin: dt.date
    label_cutoff: dt.date
    data_cutoff: dt.date

    def to_dict(self) -> dict[str, str]:
        return {
            "assessment_end": self.assessment_end.isoformat(),
            "fitting_origin": self.fitting_origin.isoformat(),
            "label_cutoff": self.label_cutoff.isoformat(),
            "data_cutoff": self.data_cutoff.isoformat(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PurgedRow:
    """One row removed from consideration, and why."""

    index: int
    row_id: str
    reason: str


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    """The fixed, hashable description of one split.

    Everything a reviewer needs to decide whether two runs are comparable, and
    everything a test needs to assert that development never read an evaluation
    label.
    """

    protocol: str
    grouping_scope: str
    group_keys: tuple[str, ...]
    dates: ProtocolDates
    #: Stable identifiers, in cohort order. Index positions below refer to these.
    row_ids: tuple[str, ...]
    components: tuple[int, ...]
    development: tuple[int, ...]
    evaluation: tuple[int, ...]
    purged: tuple[PurgedRow, ...]
    eligibility_rules: tuple[str, ...]
    label_maturity: str
    limitations: tuple[str, ...]

    # ---- derived ----------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    @property
    def development_index(self) -> np.ndarray:
        return np.asarray(self.development, dtype=int)

    @property
    def evaluation_index(self) -> np.ndarray:
        return np.asarray(self.evaluation, dtype=int)

    @property
    def purge_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.purged:
            counts[p.reason] = counts.get(p.reason, 0) + 1
        return counts

    @property
    def permits_deployment_claim(self) -> bool:
        """Whether numbers from this split may be called deployment performance."""
        return self.protocol in DEPLOYMENT_CLAIM_PERMITTED

    def identity(self) -> dict[str, Any]:
        """The canonical, order-stable content the hash is taken over.

        Row *identifiers* rather than positions, so the hash survives a
        reordering of the cohort query; partition membership as sorted id
        tuples for the same reason.
        """
        dev = sorted(self.row_ids[i] for i in self.development)
        ev = sorted(self.row_ids[i] for i in self.evaluation)
        return {
            "protocol": str(self.protocol),
            "grouping_scope": str(self.grouping_scope),
            "group_keys": list(self.group_keys),
            "dates": self.dates.to_dict(),
            "development_row_ids": dev,
            "evaluation_row_ids": ev,
            "purged": sorted((p.row_id, p.reason) for p in self.purged),
            "eligibility_rules": list(self.eligibility_rules),
            "label_maturity": self.label_maturity,
        }

    @property
    def manifest_hash(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": str(self.protocol),
            "permits_deployment_claim": self.permits_deployment_claim,
            "grouping_scope": str(self.grouping_scope),
            "group_keys": list(self.group_keys),
            "dates": self.dates.to_dict(),
            "n_rows_considered": self.n_rows,
            "n_development": len(self.development),
            "n_evaluation": len(self.evaluation),
            "n_purged": len(self.purged),
            "purged_by_reason": self.purge_counts,
            "eligibility_rules": list(self.eligibility_rules),
            "label_maturity": self.label_maturity,
            "limitations": list(self.limitations),
            "manifest_hash": self.manifest_hash,
        }

    # ---- invariants -------------------------------------------------------

    def verify(
        self,
        as_of_dates: Sequence[dt.date],
        label_available_on: Sequence[dt.date | None],
    ) -> None:
        """Re-check every invariant the manifest asserts. Raises on violation.

        Called after construction and again at the top of evaluation, because a
        manifest that is only checked where it is built is a comment.
        """
        dev = set(self.development)
        ev = set(self.evaluation)
        purged = {p.index for p in self.purged}

        if dev & ev:
            raise ProtocolError(
                f"{len(dev & ev)} rows are in both development and evaluation"
            )
        if dev & purged or ev & purged:
            raise ProtocolError("a purged row was also assigned to a partition")
        if len(dev) + len(ev) + len(purged) != self.n_rows:
            raise ProtocolError(
                f"partitions ({len(dev)} + {len(ev)}) and purges ({len(purged)}) "
                f"do not account for all {self.n_rows} rows"
            )

        comps = np.asarray(self.components)
        dev_comps = {int(c) for c in comps[list(dev)]} if dev else set()
        ev_comps = {int(c) for c in comps[list(ev)]} if ev else set()
        shared = dev_comps & ev_comps
        if shared:
            raise ProtocolError(
                f"{len(shared)} connected components span the development/"
                f"evaluation boundary: {sorted(shared)[:5]}"
            )

        # A strict temporal headline must not contain pre-boundary evaluation
        # rows, and must not contain post-boundary development rows.
        for i in dev:
            if as_of_dates[i] > self.dates.assessment_end:
                raise ProtocolError(
                    f"development row {self.row_ids[i]} has as-of "
                    f"{as_of_dates[i]} after the assessment boundary "
                    f"{self.dates.assessment_end}"
                )
        for i in ev:
            if as_of_dates[i] <= self.dates.assessment_end:
                raise ProtocolError(
                    f"evaluation row {self.row_ids[i]} has as-of "
                    f"{as_of_dates[i]} at or before the assessment boundary "
                    f"{self.dates.assessment_end}; a strict temporal headline "
                    "would be silently including the past"
                )

        if self.label_maturity == "enforced_at_fitting_origin":
            for i in dev:
                la = label_available_on[i]
                if la is None or la > self.dates.label_cutoff:
                    raise ProtocolError(
                        f"development row {self.row_ids[i]} has label available "
                        f"{la}, after the declared label cutoff "
                        f"{self.dates.label_cutoff}"
                    )


# ---------------------------------------------------------------------------
# Planning a split
# ---------------------------------------------------------------------------


def _quantile_date(dates: Sequence[dt.date], q: float) -> dt.date:
    arr = np.sort(np.asarray([d.toordinal() for d in dates]))
    return dt.date.fromordinal(int(np.quantile(arr, q)))


def plan_split(
    row_ids: Sequence[str],
    as_of_dates: Sequence[dt.date],
    label_available_on: Sequence[dt.date | None],
    components: Sequence[int],
    *,
    protocol: Protocol = Protocol.STRICT_TEMPORAL,
    development_fraction: float = 0.75,
    data_cutoff: dt.date | None = None,
    grouping_scope: GroupScope = GroupScope.IDENTITY,
    min_development_rows: int = MIN_DEVELOPMENT_ROWS,
    min_evaluation_rows: int = MIN_EVALUATION_ROWS,
) -> SplitManifest:
    """Fix the development/evaluation split before anything is fitted.

    Order of operations matters and is deliberate:

    1. **Data cutoff.** Rows whose outcome resolves after the corpus cutoff are
       removed first. They are not labelled data; they are the future.
    2. **Assessment boundary.** Drawn at ``development_fraction`` of the as-of
       distribution over the rows that survived step 1.
    3. **Component integrity.** A connected component whose rows straddle the
       boundary goes wholly to *evaluation*, and its earlier rows are purged.
       Later is the conservative direction: it removes rows from development
       rather than letting a future-resolved sibling inform an earlier fit.
    4. **Label maturity.** Under the strict protocol, development rows whose
       outcome was not knowable by the fitting origin are purged. This is a
       row-level purge, not a component-level one: such a row sits wholly
       inside development, so dropping it cannot create a cross-boundary unit.

    Raises :class:`ProtocolError` rather than degrading quietly when the result
    is too small to mean anything. The caller must name a weaker protocol.
    """
    n = len(row_ids)
    if not (n == len(as_of_dates) == len(label_available_on) == len(components)):
        raise ValueError("row_ids, dates, label dates and components must align")
    if n == 0:
        raise ProtocolError("no rows supplied")

    comps = np.asarray(components, dtype=int)
    purged: list[PurgedRow] = []
    rules: list[str] = []
    limitations: list[str] = []

    cutoff = data_cutoff or dt.date.today()
    rules.append(
        f"A row is admissible only if its outcome was resolved by the corpus "
        f"data cutoff {cutoff.isoformat()}."
    )

    # ---- 1. Data cutoff ----------------------------------------------------
    admissible = np.ones(n, dtype=bool)
    for i in range(n):
        la = label_available_on[i]
        if la is None or la > cutoff:
            admissible[i] = False
            purged.append(
                PurgedRow(i, row_ids[i], "label_unresolved_at_data_cutoff")
            )
    if not admissible.any():
        raise ProtocolError(
            f"every row's outcome resolves after the data cutoff {cutoff}"
        )

    # ---- 2. Assessment boundary -------------------------------------------
    live_dates = [as_of_dates[i] for i in np.flatnonzero(admissible)]
    assessment_end = _quantile_date(live_dates, development_fraction)
    rules.append(
        f"Development rows are assessed on or before {assessment_end.isoformat()} "
        f"(the {development_fraction:.0%} quantile of admissible as-of dates); "
        "evaluation rows are assessed strictly after it."
    )

    aso = np.asarray([d.toordinal() for d in as_of_dates])
    side = np.where(aso <= assessment_end.toordinal(), 0, 1)  # 0 = dev, 1 = eval

    # ---- 3. Component integrity -------------------------------------------
    # A component is assigned to the latest side any of its admissible rows
    # falls on. Inadmissible rows do not drag a component forward: they are
    # already gone and are not evidence about anything.
    comp_side: dict[int, int] = {}
    for c in np.unique(comps):
        mask = (comps == c) & admissible
        if not mask.any():
            continue
        comp_side[int(c)] = int(side[mask].max())

    rules.append(
        "Connected components are kept whole. A component straddling the "
        "assessment boundary is assigned entirely to evaluation and its "
        "earlier rows are purged from development."
    )

    development: list[int] = []
    evaluation: list[int] = []
    for i in range(n):
        if not admissible[i]:
            continue
        assigned = comp_side[int(comps[i])]
        if assigned == 1 and side[i] == 0:
            purged.append(
                PurgedRow(i, row_ids[i], "component_straddled_assessment_boundary")
            )
            continue
        (evaluation if assigned == 1 else development).append(i)

    # ---- 4. Label maturity -------------------------------------------------
    fitting_origin = assessment_end
    label_cutoff = assessment_end

    if protocol in (Protocol.STRICT_TEMPORAL, Protocol.TEMPORAL_OUTER_MATURITY):
        label_maturity = "enforced_at_fitting_origin"
        rules.append(
            f"A development row's outcome must have been knowable by the "
            f"fitting origin {fitting_origin.isoformat()}. An early programme "
            "entry date is not sufficient."
        )
        kept: list[int] = []
        for i in development:
            la = label_available_on[i]
            if la is None or la > label_cutoff:
                purged.append(
                    PurgedRow(i, row_ids[i], "label_immature_at_fitting_origin")
                )
            else:
                kept.append(i)
        development = kept
        if protocol is Protocol.TEMPORAL_OUTER_MATURITY:
            limitations.append(
                "Label maturity is enforced at the outer fitting origin but not "
                "within each inner forward-chaining fold, because per-fold "
                "enforcement left folds too small to fit. Inner-fold "
                "hyperparameter and family choices therefore used some outcomes "
                "that would not have been knowable at that fold's own origin. "
                "The locked evaluation partition is unaffected."
            )
    elif protocol is Protocol.RETROSPECTIVE:
        label_maturity = "not_enforced"
        limitations.append(
            "Label maturity is not enforced: the boundary is drawn on "
            "assessment date only, so development rows include programmes whose "
            "outcomes resolved after the fitting origin. These numbers describe "
            "retrospective discrimination and calibration. They are not an "
            "estimate of deployment performance and must not be presented as one."
        )
        rules.append(
            "Label maturity is NOT enforced under this protocol; only the "
            "corpus data cutoff applies."
        )
    else:
        raise ProtocolError(
            f"{protocol} cannot be planned; it exists only to label historical "
            "artifacts produced before this module."
        )

    # ---- 5. Adequacy -------------------------------------------------------
    if len(development) < min_development_rows:
        raise ProtocolError(
            f"protocol {protocol} leaves {len(development)} development rows "
            f"(minimum {min_development_rows}). "
            f"Purges: {_count(purged)}. "
            "Either supply more resolved history or request a weaker protocol "
            "by name — do not lower the threshold to make this pass."
        )
    if len(evaluation) < min_evaluation_rows:
        raise ProtocolError(
            f"protocol {protocol} leaves {len(evaluation)} evaluation rows "
            f"(minimum {min_evaluation_rows}). A held-out metric on fewer rows "
            "than this cannot resolve any difference worth acting on."
        )

    manifest = SplitManifest(
        protocol=str(protocol),
        grouping_scope=str(grouping_scope),
        group_keys=GROUP_SCOPE_KEYS[grouping_scope],
        dates=ProtocolDates(
            assessment_end=assessment_end,
            fitting_origin=fitting_origin,
            label_cutoff=label_cutoff,
            data_cutoff=cutoff,
        ),
        row_ids=tuple(row_ids),
        components=tuple(int(c) for c in comps),
        development=tuple(sorted(development)),
        evaluation=tuple(sorted(evaluation)),
        purged=tuple(purged),
        eligibility_rules=tuple(rules),
        label_maturity=label_maturity,
        limitations=tuple(limitations),
    )
    manifest.verify(as_of_dates, label_available_on)
    return manifest


def _count(purged: Sequence[PurgedRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in purged:
        out[p.reason] = out.get(p.reason, 0) + 1
    return out


@dataclasses.dataclass(frozen=True)
class TrainingPlan:
    """A complete, checkable plan: which rows, which protocol, which folds.

    Produced once, before anything is fitted, and carried into the artifact so
    a reviewer can see what was decided and what was rejected.
    """

    manifest: SplitManifest
    folds: tuple[Fold, ...]
    fold_diagnostics: dict[str, Any]
    #: Every protocol attempted, in order, with the reason each was rejected.
    protocol_trace: tuple[dict[str, str], ...]

    @property
    def protocol(self) -> str:
        return self.manifest.protocol

    @property
    def permits_deployment_claim(self) -> bool:
        return self.manifest.permits_deployment_claim

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_manifest": self.manifest.to_dict(),
            "folds": [f.to_dict() for f in self.folds],
            "fold_diagnostics": self.fold_diagnostics,
            "protocol_trace": [dict(t) for t in self.protocol_trace],
        }


def plan_training(
    row_ids: Sequence[str],
    as_of_dates: Sequence[dt.date],
    label_available_on: Sequence[dt.date | None],
    components: Sequence[int],
    *,
    preference: Sequence[Protocol] = (
        Protocol.STRICT_TEMPORAL,
        Protocol.TEMPORAL_OUTER_MATURITY,
        Protocol.RETROSPECTIVE,
    ),
    min_usable_folds: int = MIN_USABLE_FOLDS,
    n_folds: int = 4,
    **kwargs: Any,
) -> TrainingPlan:
    """Try protocols strongest-first and return the first complete, usable plan.

    A protocol is usable only if it leaves an adequate development partition,
    an adequate evaluation partition *and* enough forward-chaining folds to
    select hyperparameters on. The third condition is why this function exists
    instead of :func:`plan_split` alone: on the demonstration cohort the strict
    protocol passes the row-count checks and then yields **zero** usable folds,
    because its inner folds must also respect label maturity and at a 2009 fold
    origin only a handful of outcomes had resolved. A planner that checked only
    row counts would hand the caller a split it could not tune on.

    This is *not* a silent fallback. Every rejection, with its measured reason,
    is recorded in :attr:`TrainingPlan.protocol_trace`, travels into the model
    artifact, and the accepted protocol is named beside every number derived
    from it. A caller that wants one specific protocol and no substitute should
    call :func:`plan_split` and :func:`forward_chaining_folds` directly.
    """
    trace: list[dict[str, str]] = []
    for proto in preference:
        try:
            manifest = plan_split(
                row_ids, as_of_dates, label_available_on, components,
                protocol=proto, **kwargs,
            )
        except ProtocolError as exc:
            trace.append({"protocol": str(proto), "rejected_because": str(exc)})
            continue

        folds, diag = forward_chaining_folds(
            as_of_dates, label_available_on, components, manifest.development_index,
            n_folds=n_folds,
            enforce_label_maturity=PROTOCOL_FOLD_MATURITY[str(proto)],
        )
        if len(folds) < min_usable_folds:
            skipped = "; ".join(
                f"origin {s['origin']}: {s['reason']}" for s in diag.get("skipped", [])
            )
            trace.append({
                "protocol": str(proto),
                "rejected_because": (
                    f"only {len(folds)} usable forward-chaining fold(s) out of "
                    f"{n_folds} requested (minimum {min_usable_folds}), with "
                    f"per-fold label maturity "
                    f"{'enforced' if PROTOCOL_FOLD_MATURITY[str(proto)] else 'not enforced'}. "
                    f"Development partition had {len(manifest.development)} rows. "
                    f"Skipped folds — {skipped or 'none recorded'}"
                ),
            })
            continue

        trace.append({"protocol": str(proto), "accepted": "yes"})
        return TrainingPlan(
            manifest=manifest,
            folds=tuple(folds),
            fold_diagnostics=diag,
            protocol_trace=tuple(trace),
        )

    raise ProtocolError(
        "no declared protocol can be satisfied by this cohort:\n"
        + "\n".join(
            f"  {t['protocol']}: {t.get('rejected_because', '')}" for t in trace
        )
    )


# ---------------------------------------------------------------------------
# Forward-chaining folds inside development
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Fold:
    """One forward-chaining fold, with the origin it claims to sit at."""

    index: int
    origin: dt.date
    train: np.ndarray
    test: np.ndarray
    #: Rows dropped from this fold's training set, by reason.
    dropped: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "origin": self.origin.isoformat(),
            "n_train": int(self.train.size),
            "n_test": int(self.test.size),
            "dropped": dict(self.dropped),
        }


def forward_chaining_folds(
    as_of_dates: Sequence[dt.date],
    label_available_on: Sequence[dt.date | None],
    components: Sequence[int],
    subset: np.ndarray,
    *,
    n_folds: int = 4,
    min_train_fraction: float = 0.35,
    enforce_label_maturity: bool = True,
    min_train_rows: int = MIN_FOLD_TRAIN_ROWS,
    min_test_rows: int = MIN_FOLD_TEST_ROWS,
) -> tuple[list[Fold], dict[str, Any]]:
    """Expanding-window folds over ``subset``, in the *original* index space.

    Each fold has an explicit origin. Its training rows must be assessed before
    that origin, must belong to no component that appears in the fold's test
    window, and — when ``enforce_label_maturity`` is set — must have had their
    outcome resolved by that origin. That last condition is what makes a fold a
    simulation of fitting at its origin rather than a convenience partition.

    Returns the folds and a diagnostic record. Folds too small to fit are
    skipped and counted; an empty fold list is a reportable outcome, not an
    exception, because the caller has to decide between protocols.
    """
    subset = np.asarray(subset, dtype=int)
    comps = np.asarray(components, dtype=int)
    aso = np.asarray([as_of_dates[i].toordinal() for i in subset])
    order = np.argsort(aso, kind="stable")
    m = subset.size

    diag: dict[str, Any] = {
        "n_subset": int(m),
        "label_maturity_enforced": bool(enforce_label_maturity),
        "skipped": [],
    }
    start = int(m * min_train_fraction)
    if start >= m - 1:
        diag["error"] = "subset too small to form any expanding-window fold"
        return [], diag

    boundaries = np.linspace(start, m, n_folds + 1, dtype=int)[:-1]
    folds: list[Fold] = []

    for k, b in enumerate(boundaries):
        cut = aso[order[min(b, m - 1)]]
        end = boundaries[k + 1] if k + 1 < len(boundaries) else m
        if end >= m:
            test_local = aso >= cut
        else:
            upper = aso[order[min(end, m - 1)]]
            test_local = (aso >= cut) & (aso < upper)

        origin = dt.date.fromordinal(int(cut))
        test_rows = subset[test_local]
        if test_rows.size < min_test_rows:
            diag["skipped"].append(
                {"fold": k, "origin": origin.isoformat(),
                 "reason": f"test window has {int(test_rows.size)} rows"}
            )
            continue

        dropped: dict[str, int] = {}
        train_local = aso < cut

        # Component integrity against this fold's test window.
        test_comps = set(comps[test_rows].tolist())
        straddle = train_local & np.isin(comps[subset], list(test_comps))
        dropped["component_in_test_window"] = int(straddle.sum())
        train_local = train_local & ~straddle

        # Label maturity at this fold's own origin.
        if enforce_label_maturity:
            immature = np.zeros(m, dtype=bool)
            for j, i in enumerate(subset):
                la = label_available_on[i]
                immature[j] = la is None or la.toordinal() > cut
            n_immature = int((train_local & immature).sum())
            dropped["label_immature_at_fold_origin"] = n_immature
            train_local = train_local & ~immature

        train_rows = subset[train_local]
        if train_rows.size < min_train_rows:
            diag["skipped"].append(
                {"fold": k, "origin": origin.isoformat(),
                 "reason": f"training set has {int(train_rows.size)} rows "
                           f"after purges {dropped}"}
            )
            continue

        folds.append(
            Fold(index=k, origin=origin, train=np.sort(train_rows),
                 test=np.sort(test_rows), dropped=dropped)
        )

    diag["n_folds"] = len(folds)
    diag["folds"] = [f.to_dict() for f in folds]
    return folds, diag
