import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNetFeatureExtractor(nn.Module):
    """
    Minimal ResNet feature extractor with automatic pretrained weights.

    - Supports resnet18/34/50/101/152.
    - Automatically loads torchvision pretrained weights.
    - Returns global-average-pooled features (fc replaced by Identity).
    - Output dim: 512 for 18/34, 2048 for 50/101/152.

    Input:
        (B, K, 3, H, W) -> (B, K, D)
        (N, 3, H, W)    -> (N, D)
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        train_backbone: bool = True,
        return_sequence: bool = False,
    ):
        super().__init__()
        model_name = model_name.lower()
        assert model_name in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}, \
            f"Unsupported model: {model_name}"

        # --- Auto-load appropriate weights ---
        if pretrained:
            weights_enum = {
                "resnet18": tvm.ResNet18_Weights.DEFAULT,
                "resnet34": tvm.ResNet34_Weights.DEFAULT,
                "resnet50": tvm.ResNet50_Weights.DEFAULT,
                "resnet101": tvm.ResNet101_Weights.DEFAULT,
                "resnet152": tvm.ResNet152_Weights.DEFAULT,
            }[model_name]
        else:
            weights_enum = None

        # --- Construct backbone ---
        ctor = getattr(tvm, model_name)
        self.backbone = ctor(weights=weights_enum)
        self.backbone.fc = nn.Identity()

        # --- Output dim ---
        self.out_dim = 512 if model_name in ("resnet18", "resnet34") else 2048

        # --- Optionally freeze ---
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
            feats = self.backbone(x)
            if self.return_sequence:
                feats = feats.view(B, K, self.out_dim)
            return feats
        elif x.dim() == 4:
            return self.backbone(x)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {tuple(x.shape)}")