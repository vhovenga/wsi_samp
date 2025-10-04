import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, AUROC
from .base import BaseMetric


def _identity(x): return x
def _to_probs_binary(x): return torch.sigmoid(x.squeeze(-1))
def _to_probs_multiclass(x): return F.softmax(x, dim=-1)


class AccuracyMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(name="Accuracy")
        task = kwargs.get("task", "binary")
        self.metric = Accuracy(**kwargs)

        if task == "binary":
            self._transform = _to_probs_binary
        elif task == "multiclass":
            self._transform = _identity  # Accuracy can argmax logits
        else:
            raise ValueError(f"Unsupported task: {task}")

    def update(self, preds, targets):
        self.metric.update(self._transform(preds), targets)

    def compute(self):
        out = self.metric.compute()
        return out.item() if out.numel() == 1 else out

    def reset(self):
        self.metric.reset()


class F1Metric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(name="F1")
        task = kwargs.get("task", "binary")
        self.metric = F1Score(**kwargs)

        if task == "binary":
            self._transform = _to_probs_binary
        elif task == "multiclass":
            self._transform = _identity  # F1 can argmax logits
        else:
            raise ValueError(f"Unsupported task: {task}")

    def update(self, preds, targets):
        self.metric.update(self._transform(preds), targets)

    def compute(self):
        out = self.metric.compute()
        return out.item() if out.numel() == 1 else out

    def reset(self):
        self.metric.reset()


class AUROCMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(name="AUROC")
        task = kwargs.get("task", "binary")
        self.metric = AUROC(**kwargs)

        if task == "binary":
            self._transform = _to_probs_binary
        elif task == "multiclass":
            self._transform = _to_probs_multiclass  # AUROC needs probs
        else:
            raise ValueError(f"Unsupported task: {task}")

    def update(self, preds, targets):
        self.metric.update(self._transform(preds), targets)

    def compute(self):
        out = self.metric.compute()
        return out.item() if out.numel() == 1 else out

    def reset(self):
        self.metric.reset()

CLASSIFICATION_METRICS = {
    "Accuracy": AccuracyMetric,
    "F1": F1Metric,
    "AUROC": AUROCMetric,
}