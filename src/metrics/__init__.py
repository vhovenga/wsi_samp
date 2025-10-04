from .classifcation_metrics import CLASSIFICATION_METRICS
from .metric_utils import build_metrics

# Central registry
METRIC_REGISTRY = {}
METRIC_REGISTRY.update(CLASSIFICATION_METRICS)

__all__ = ["METRIC_REGISTRY",
           "build_metrics"]
