import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNetFeatureExtractor(nn.Module):
    """
    Minimal ResNet feature extractor.
    - Accepts inputs of shape:
        * (B, K, 3, H, W)  -> returns (B, K, D)
        * (N, 3, H, W)     -> returns (N, D)
    - No transforms, no resizing, no dataloaders/utilities.
    - Global-average-pooled features (fc replaced by Identity).
    - D is 512 for resnet18/34, 2048 for resnet50/101/152.

    Args:
        model_name: one of {"resnet18","resnet34","resnet50","resnet101","resnet152"}.
        weights: torchvision weights object or None. Example:
                 tvm.ResNet50_Weights.DEFAULT (for ImageNet pretrained).
                 If you're on an older torchvision, pass None and load weights manually if needed.
        train_backbone: if False, backbone is frozen (feature extractor mode).
        return_sequence: if True and input is (B,K,...) return (B,K,D); otherwise flatten to (N,D).
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        weights=None,
        train_backbone: bool = True,
        return_sequence: bool = False,
    ):
        super().__init__()
        model_name = model_name.lower()
        ctor = getattr(tvm, model_name)
        self.backbone = ctor(weights=weights)
        # Replace classification head with Identity; keeps avgpool+flatten inside
        self.backbone.fc = nn.Identity()

        # Infer output dim
        if model_name in ("resnet18", "resnet34"):
            self.out_dim = 512
        else:
            self.out_dim = 2048

        # Freeze if requested
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.return_sequence = return_sequence

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Non-grad convenience path."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,K,3,H,W) or (N,3,H,W)
        returns: (B,K,D) if return_sequence=True and 5D input, else (N,D)
        """
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x = x.view(B * K, C, H, W)
            feats = self.backbone(x)  # (B*K, D)
            if self.return_sequence:
                feats = feats.view(B, K, self.out_dim)  # (B, K, D)
            return feats
        elif x.dim() == 4:
            # (N, 3, H, W)
            return self.backbone(x)  # (N, D)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")