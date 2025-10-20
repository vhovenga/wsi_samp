import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        weights_url: str | None = None,
        train_backbone: bool = True,
        return_sequence: bool = False,
    ):
        super().__init__()
        model_name = model_name.lower()
        assert model_name in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}

        # --- Auto-load appropriate weights or URL ---
        if pretrained:
            if weights_url is not None:
                # Download + load weights from a custom URL
                state_dict = torch.hub.load_state_dict_from_url(weights_url, progress=True)
                weights_enum = None
            else:
                weights_enum = {
                    "resnet18": tvm.ResNet18_Weights.DEFAULT,
                    "resnet34": tvm.ResNet34_Weights.DEFAULT,
                    "resnet50": tvm.ResNet50_Weights.DEFAULT,
                    "resnet101": tvm.ResNet101_Weights.DEFAULT,
                    "resnet152": tvm.ResNet152_Weights.DEFAULT,
                }[model_name]
                state_dict = None
        else:
            weights_enum = None
            state_dict = None

        # --- Construct backbone ---
        ctor = getattr(tvm, model_name)
        self.backbone = ctor(weights=weights_enum)
        if state_dict is not None:
            self.backbone.load_state_dict(state_dict)

        self.backbone.fc = nn.Identity()
        self.out_dim = 512 if model_name in ("resnet18", "resnet34") else 2048

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