from .fixed_samplers import FIXED_SAMPLERS
from .feature_dpp_samplers import FEATURE_DPP_SAMPLERS
from .samplers_utils import build_sampler

# Central registry
SAMPLER_REGISTRY = {}
SAMPLER_REGISTRY.update(FIXED_SAMPLERS)
SAMPLER_REGISTRY.update(FEATURE_DPP_SAMPLERS)

__all__ = ["SAMPLER_REGISTRY",
           "build_sampler"]

