import abc
from torchmetrics import Metric

class BaseMetric(Metric, metaclass=abc.ABCMeta):
    """Abstract interface for all metrics."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def update(self, preds, targets):
        pass

    @abc.abstractmethod
    def compute(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass