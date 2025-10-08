from .base import MILModule
from .feature_extractors import FEATURE_EXTRACTORS
from .aggregators import AGGREGATORS
from .predictors import PREDICTORS


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
        
def build_model(cfg):
    # Feature extractor
    feat_cls = FEATURE_EXTRACTORS[cfg["feature_extractor"]["name"]]
    feature_extractor = feat_cls(**cfg["feature_extractor"].get("params", {}))
    if cfg["feature_extractor"].get("freeze", False):
        freeze_module(feature_extractor)

    # Aggregator
    agg_cls = AGGREGATORS[cfg["aggregator"]["name"]]
    aggregator = agg_cls(**cfg["aggregator"].get("params", {}))

    # Predictor
    predictor_cls = PREDICTORS[cfg["predictor"]["name"]]
    predictor = predictor_cls(**cfg["predictor"].get("params", {}))

    # Micro batch size
    micro_k = cfg.get("micro_batch_size", 64)

    return MILModule(
        feature_extractor=feature_extractor,
        aggregator=aggregator,
        predictor=predictor,
        micro_k=micro_k,
    )
