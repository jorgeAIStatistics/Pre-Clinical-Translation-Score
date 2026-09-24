"""Leakage-resistant data splitting.

A random train/test split on this problem is not merely optimistic; it is
meaningless. The reasons are specific and each is handled here.

**One biological bet appears many times.** The same program is developed in
three indications; the same target is attacked by four sponsors; the same
molecule is renamed twice. Random splitting puts near-duplicates of the test
rows into training, and the model learns to recognise the program rather than
to assess the evidence.

**Grouping keys overlap transitively.** Grouping on the program id alone
misses the case of one asset carrying two development codes. The correct object
is the *connected component* of the graph in which two cohort entries are
joined whenever they share a grouping key. :class:`ConnectedComponentGrouper`
builds exactly that, and every splitter here assigns whole components.

**Identity is not similarity, and the distinction decides the split.** Two
sponsors developing different CDK4/6 inhibitors are not the same program; they
are the same *bet*. Treating shared mechanism class as a grouping key makes the
transitive closure swallow the cohort — on the demonstration data it collapses
211 rows into 9 components, which makes any split degenerate. Worse, it is
conceptually wrong: it would remove from training exactly the class-level
information a real analyst possesses.

So two scopes are defined, and both are reported:

``GroupScope.IDENTITY`` — program and compound. This is the *same asset under a
different name*, which is unambiguous leakage. Used for the headline temporal
split, because it mirrors deployment: an analyst at a 2018 decision point
genuinely knew the outcomes of 2010 programs on the same target, and removing
that knowledge would make the training distribution differ from the serving one.

``GroupScope.STRICT`` — program, compound and target. Used for a *secondary*
grouped k-fold, which quantifies how much of the model's performance comes from
memorising target-level outcomes rather than reading evidence. It cannot be
combined with a temporal split: a target's programs span the whole time range,
so every component straddles every boundary. That incompatibility is a property
of the problem, and the report states it rather than silently picking one.

Mechanism class is never a grouping key. It remains available as the grouping
*variable* for the hierarchical model, which is a different purpose entirely.

**The deployment task is prospective.** The product scores a program at a
decision point using only what is then known, to predict something that has not
happened yet. A cross-sectional split trains on 2019 programs to predict 2011
outcomes, which is not the task. :class:`TemporalSplit` is therefore the
headline evaluation, and the grouped splits exist to isolate the contribution
of group leakage specifically.

Each splitter reports how many rows it *dropped* to preserve its invariant.
That number belongs in the validation report: a split that quietly discards a
third of the cohort to stay clean is telling you the cohort is small.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Any

import numpy as np


@dataclasses.dataclass(slots=True)
class SplitAssignment:
    """Indices for one train/calibration/test partition."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    #: Rows excluded to preserve the splitter's invariant, with reasons.
    dropped: dict[str, int]
    description: str

    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "calibration": int(self.calibration.size),
            "test": int(self.test.size),
            "dropped": int(sum(self.dropped.values())),
        }


class GroupScope(StrEnum):
    """Which keys define a leakage-safe unit. See the module docstring."""

    #: Same asset under a different name. Unambiguous leakage.
    IDENTITY = "identity"
    #: Identity plus shared target. Stricter than deployment requires; used for
    #: the target-memorisation sensitivity analysis.
    STRICT = "strict"


GROUP_SCOPE_KEYS: dict[GroupScope, tuple[str, ...]] = {
    GroupScope.IDENTITY: ("program", "compound"),
    GroupScope.STRICT: ("program", "compound", "target"),
}


class ConnectedComponentGrouper:
    """Union–find over the chosen grouping keys, yielding leakage-safe components.

    Two cohort entries land in the same component if they share any key in the
    active scope, or are linked through a chain of such shared keys. Assigning
    whole components to folds guarantees that no unit spans the boundary.

    The cost is granularity: heavily connected cohorts collapse into a few giant
    components, which makes k-fold coarse. :meth:`summary` surfaces that so a
    reader can see when the effective sample size for validation is much smaller
    than the row count.
    """

    def __init__(self, key_names: Sequence[str] = GROUP_SCOPE_KEYS[GroupScope.IDENTITY]) -> None:
        self.key_names = tuple(key_names)
        self._parent: dict[str, str] = {}

    def _find(self, x: str) -> str:
        while self._parent.setdefault(x, x) != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def fit(self, keys: list[dict[str, str | None]]) -> np.ndarray:
        """Return a component label per row."""
        self._parent = {}
        row_tokens: list[list[str]] = []
        for i, row in enumerate(keys):
            tokens = [f"row::{i}"]
            for name in self.key_names:
                value = row.get(name)
                if value:
                    tokens.append(f"{name}::{value}")
            row_tokens.append(tokens)
            for t in tokens[1:]:
                self._union(tokens[0], t)

        labels = np.asarray([self._find(f"row::{i}") for i in range(len(keys))])
        # Re-label to compact integers for readability in reports.
        lookup: dict[str, int] = {}
        out = np.empty(len(keys), dtype=int)
        for i, lab in enumerate(labels):
            out[i] = lookup.setdefault(lab, len(lookup))
        return out

    @staticmethod
    def summary(components: np.ndarray) -> dict[str, Any]:
        _, counts = np.unique(components, return_counts=True)
        return {
            "n_rows": int(components.size),
            "n_components": int(counts.size),
            "largest_component": int(counts.max()) if counts.size else 0,
            "median_component_size": float(np.median(counts)) if counts.size else 0.0,
            "singleton_components": int((counts == 1).sum()),
            "effective_sample_size_note": (
                "Validation folds are formed from connected components, so the "
                "number of independent units is the component count, not the row "
                "count. Confidence intervals are computed by resampling "
                "components for the same reason."
            ),
        }


def temporal_split(
    as_of_dates: Sequence[dt.date],
    components: np.ndarray,
    *,
    train_end: dt.date,
    calibration_end: dt.date,
) -> SplitAssignment:
    """Prospective split: train on the past, test on the future.

    Component integrity is enforced *on top of* the date boundaries. A
    component whose rows straddle a boundary is assigned wholly to the later
    partition, and its earlier rows are dropped. Assigning to the later side is
    the conservative direction: it removes rows from training rather than
    letting future-resolved siblings inform an earlier prediction.
    """
    dates = np.asarray(as_of_dates)
    n = len(dates)
    provisional = np.empty(n, dtype=object)
    provisional[dates <= train_end] = "train"
    provisional[(dates > train_end) & (dates <= calibration_end)] = "calibration"
    provisional[dates > calibration_end] = "test"

    rank = {"train": 0, "calibration": 1, "test": 2}
    comp_partition: dict[int, str] = {}
    for comp in np.unique(components):
        mask = components == comp
        best = max(provisional[mask], key=lambda p: rank[p])
        comp_partition[int(comp)] = best

    final = np.asarray([comp_partition[int(c)] for c in components])
    dropped = {
        "component_straddled_boundary": int(np.sum(final != provisional)),
    }

    return SplitAssignment(
        train=np.flatnonzero(final == "train"),
        calibration=np.flatnonzero(final == "calibration"),
        test=np.flatnonzero(final == "test"),
        dropped=dropped,
        description=(
            f"Temporal split: train as-of ≤ {train_end.isoformat()}, "
            f"calibration ≤ {calibration_end.isoformat()}, test after. "
            "Connected components are kept whole and assigned to their latest "
            "partition."
        ),
    )


def grouped_kfold(
    components: np.ndarray,
    n_splits: int = 5,
    *,
    y: np.ndarray | None = None,
    random_state: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K-fold over connected components, balancing positives across folds.

    Components are sorted by positive count and dealt greedily to the currently
    lightest fold. With few components and an imbalanced outcome, naive random
    assignment routinely produces a fold with no positives at all, which makes
    AUROC undefined and log loss unstable; greedy balancing avoids that without
    peeking at features.
    """
    rng = np.random.default_rng(random_state)
    comps = np.unique(components)
    rng.shuffle(comps)

    if y is None:
        weights = {int(c): int(np.sum(components == c)) for c in comps}
    else:
        y = np.asarray(y)
        weights = {int(c): int(np.sum(y[components == c] == 1)) for c in comps}

    order = sorted(comps, key=lambda c: (-weights[int(c)], int(c)))
    fold_load = [0.0] * n_splits
    fold_of: dict[int, int] = {}
    for c in order:
        f = int(np.argmin(fold_load))
        fold_of[int(c)] = f
        fold_load[f] += weights[int(c)] + 0.25  # tie-break on row count too

    assignments = np.asarray([fold_of[int(c)] for c in components])
    for f in range(n_splits):
        test_idx = np.flatnonzero(assignments == f)
        train_idx = np.flatnonzero(assignments != f)
        if test_idx.size == 0:
            continue
        yield train_idx, test_idx


def nested_temporal_folds(
    as_of_dates: Sequence[dt.date],
    components: np.ndarray,
    *,
    n_folds: int = 4,
    min_train_fraction: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: the honest way to tune hyperparameters here.

    Fold *k* trains on everything before a cut date and tests on the next
    slice. This mirrors deployment — always predicting forward — and, unlike
    k-fold, never lets a later program inform an earlier one. Used for
    hyperparameter selection so that tuning cannot exploit the future either.
    """
    dates = np.asarray(as_of_dates)
    order = np.argsort(dates, kind="stable")
    n = len(dates)
    start = int(n * min_train_fraction)
    if start >= n - 1:
        return []

    boundaries = np.linspace(start, n, n_folds + 1, dtype=int)[:-1]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i, b in enumerate(boundaries):
        cut_date = dates[order[min(b, n - 1)]]
        provisional_train = dates < cut_date
        end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        upper_date = dates[order[min(end, n - 1)]] if end < n else dates.max() + dt.timedelta(days=1)
        provisional_test = (dates >= cut_date) & (dates < upper_date)
        if end >= n:
            provisional_test = dates >= cut_date

        # Enforce component integrity: any component touching the test window
        # is removed from training entirely.
        test_comps = set(components[provisional_test].tolist())
        train_mask = provisional_train & ~np.isin(components, list(test_comps))

        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(provisional_test)
        if train_idx.size >= 20 and test_idx.size >= 5:
            folds.append((train_idx, test_idx))
    return folds


def component_bootstrap_indices(
    components: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One bootstrap replicate drawn at the component level.

    Resampling rows independently would treat the three indications of one
    program as three independent observations and produce confidence intervals
    that are too narrow by roughly the square root of the average component
    size. Every interval in the validation report uses this instead.
    """
    comps = np.unique(components)
    picked = rng.choice(comps, size=comps.size, replace=True)
    lookup = {int(c): np.flatnonzero(components == c) for c in comps}
    return np.concatenate([lookup[int(c)] for c in picked]) if comps.size else np.asarray([], int)


def build_grouping_keys(
    entries: Sequence[Any], compound_ids: Sequence[str | None] | None = None
) -> list[dict[str, str | None]]:
    """Extract grouping keys from :class:`CohortEntry` rows.

    All four keys are always emitted; the active :class:`GroupScope` decides
    which the grouper consumes. ``sponsor`` and ``mechanism`` are carried so the
    leakage audit can *report* overlap on them even when they do not define
    components — knowing that 40% of test sponsors also appear in training is
    useful context even though it is not, by itself, leakage.
    """
    compounds = compound_ids or [None] * len(entries)
    return [
        {
            "program": e.group_program_id,
            "compound": compounds[i],
            "target": e.group_target_id,
            "sponsor": e.group_sponsor_id,
            "mechanism": e.group_mechanism_key,
        }
        for i, e in enumerate(entries)
    ]


def leakage_audit(
    components: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    keys: list[dict[str, str | None]],
    scope: str = GroupScope.IDENTITY,
) -> dict[str, Any]:
    """Assert, rather than assume, that a split is clean.

    Called after every split the validation report uses. Returns the overlap
    counts per key type; the report fails loudly if any is non-zero.
    """
    shared_components = set(components[train_idx].tolist()) & set(components[test_idx].tolist())
    overlaps: dict[str, int] = {"component": len(shared_components)}
    for key in ("program", "compound", "target", "sponsor", "mechanism"):
        tr = {keys[i][key] for i in train_idx if keys[i].get(key)}
        te = {keys[i][key] for i in test_idx if keys[i].get(key)}
        overlaps[key] = len(tr & te)

    scope_keys = GROUP_SCOPE_KEYS[GroupScope(scope)]
    # Only overlap on keys inside the active scope constitutes a failure.
    # Overlap outside it is reported as context, not treated as a defect.
    overlaps["scope"] = str(scope)
    overlaps["enforced_keys"] = list(scope_keys)
    overlaps["clean"] = (
        overlaps["component"] == 0
        and all(overlaps.get(k, 0) == 0 for k in scope_keys)
    )
    overlaps["informational_overlap"] = {
        k: overlaps[k]
        for k in ("target", "sponsor", "mechanism")
        if k not in scope_keys
    }
    return overlaps
