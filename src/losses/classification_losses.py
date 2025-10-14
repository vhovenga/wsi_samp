import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Wraps torch.nn.functional.cross_entropy in a Module.
        kwargs are passed to F.cross_entropy, e.g. weight, label_smoothing, reduction.
        """
        super().__init__()
        self.kwargs = kwargs

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, **self.kwargs)
    
class BCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, **self.kwargs)

    
CLASSIFICATION_LOSSES = {
    "CrossEntropy": CrossEntropyLoss,
    "BCE": BCELoss,
}
