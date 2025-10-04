import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    """
    Classifier head with optional hidden layers and normalization.
    Takes bag-level embeddings of shape (B, D) and returns logits.

    Args:
        in_dim: input feature size D
        num_classes: number of output classes
            - If num_classes == 1 -> (B,), suitable for BCEWithLogitsLoss
            - If num_classes >= 2 -> (B, num_classes), suitable for CrossEntropyLoss
        hidden_dim: size of hidden layer (single layer if provided)
        dropout: dropout rate
        use_bn: whether to use batch normalization
    """
    def __init__(
        self,
        in_dim: int = 2048,
        num_classes: int = 1,
        hidden_dim: int | None = 512,
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        out_dim = num_classes if num_classes >= 2 else 1

        layers = []
        if hidden_dim is not None:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # (B,)
        return logits