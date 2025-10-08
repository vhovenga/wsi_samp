from .feature_extractors import FEATURE_EXTRACTORS
from .aggregators import AGGREGATORS
from .predictors import PREDICTORS
from .model_utils import build_model
from .lit_module import LitMIL

MODEL_REGISTRY = {}
MODEL_REGISTRY.update(FEATURE_EXTRACTORS)
MODEL_REGISTRY.update(AGGREGATORS)
MODEL_REGISTRY.update(PREDICTORS)

__all__ = [
    "FEATURE_EXTRACTORS",
    "AGGREGATORS",
    "PREDICTORS",
    "MODEL_REGISTRY",
    "build_model",
    "LitMIL",
]