"""Statistical models, calibration and uncertainty quantification."""

from trs.modeling.base import RiskModel, UncertaintyUnsupported, logit, sigmoid
from trs.modeling.calibration import (
    Calibrator,
    IdentityCalibration,
    IsotonicCalibration,
    LogisticRecalibration,
    select_calibrator,
)
from trs.modeling.gbdt import GradientBoostedRisk
from trs.modeling.hierarchical import HierarchicalLogisticRisk
from trs.modeling.logistic import L2LogisticRisk
from trs.modeling.uncertainty import predict_with_uncertainty, specification_sensitivity

#: Registry of candidate families. Adding one here makes it available to the
#: model-selection gate in trs.pipeline.select.
MODEL_FAMILIES: dict[str, type[RiskModel]] = {
    L2LogisticRisk.family: L2LogisticRisk,
    HierarchicalLogisticRisk.family: HierarchicalLogisticRisk,
    GradientBoostedRisk.family: GradientBoostedRisk,
}

__all__ = [
    "Calibrator", "GradientBoostedRisk", "HierarchicalLogisticRisk",
    "IdentityCalibration", "IsotonicCalibration", "L2LogisticRisk",
    "LogisticRecalibration", "MODEL_FAMILIES", "RiskModel",
    "UncertaintyUnsupported", "logit", "predict_with_uncertainty",
    "select_calibrator", "sigmoid", "specification_sensitivity",
]
