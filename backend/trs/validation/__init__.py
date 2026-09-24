"""Leakage-resistant evaluation: splitting, metrics and reporting."""

from trs.validation.metrics import evaluate, paired_bootstrap_difference
from trs.validation.splitters import (
    GROUP_SCOPE_KEYS,
    ConnectedComponentGrouper,
    GroupScope,
    build_grouping_keys,
    grouped_kfold,
    leakage_audit,
    nested_temporal_folds,
    temporal_split,
)

__all__ = [
    "ConnectedComponentGrouper", "GROUP_SCOPE_KEYS", "GroupScope",
    "build_grouping_keys", "evaluate", "grouped_kfold", "leakage_audit",
    "nested_temporal_folds", "paired_bootstrap_difference", "temporal_split",
]
