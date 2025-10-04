from .fixed_samplers import FIXED_SAMPLERS
from .samplers_utils import build_sampler

# Central registry
SAMPLER_REGISTRY = {}
SAMPLER_REGISTRY.update(FIXED_SAMPLERS)

__all__ = ["SAMPLER_REGISTRY",
           "build_sampler"]

