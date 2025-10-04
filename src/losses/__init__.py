from .classification_losses import CLASSIFICATION_LOSSES
from .loss_utils import build_loss

# Central registry
LOSS_REGISTRY = {}
LOSS_REGISTRY.update(CLASSIFICATION_LOSSES)

__all__ = ["LOSS_REGISTRY", 
           "build_loss"]