"""Feature schema, temporal integrity, aggregation and encoding."""

from trs.features.registry import FEATURES, FEATURES_BY_NAME, feature_names
from trs.features.spec import FEATURE_SCHEMA_VERSION, FeatureDomain, FeatureSpec
from trs.features.temporal import AsOfContext, TemporalIntegrityError

__all__ = [
    "FEATURES", "FEATURES_BY_NAME", "FEATURE_SCHEMA_VERSION",
    "AsOfContext", "FeatureDomain", "FeatureSpec", "TemporalIntegrityError",
    "feature_names",
]
