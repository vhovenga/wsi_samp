import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple


class MILModule(nn.Module):
    """
    End-to-end MIL forward:
      [B,Kmax,C,H,W] + mask[B,Kmax]
      -> feature_extract()
      -> aggregate()
      -> predict()
    """

    def __init__(
        self,
        feature_extractor: nn.Module,  # [N,C,H,W] -> [N,D]
        aggregator: nn.Module,         # ([B,K,D], mask[B,K]) -> [B,D] or (Z, extras)
        predictor: nn.Module,          # [B,D] -> [B,C] or [B,1]
        micro_k: int = 64,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.aggregator = aggregator
        self.predictor = predictor
        self.micro_k = micro_k

    @property
    def device(self):
        return next(self.parameters()).device

    # ---------------------------------------------------------
    # 1. Feature extraction
    # ---------------------------------------------------------
    def feature_extract(
        self,
        images: torch.Tensor,                # [B,Kmax,C,H,W] (CPU pinned or GPU)
        mask: Optional[torch.Tensor] = None  # [B,Kmax] bool
    ) -> List[torch.Tensor]:
        dev = self.device
        B, Kmax = images.shape[:2]
        feats_per_bag: List[torch.Tensor] = []

        for b in range(B):
            Kb = int(mask[b].sum().item()) if mask is not None else Kmax
            bag_imgs = images[b, :Kb]

            chunks = []
            for s in range(0, Kb, self.micro_k):
                x_mb = bag_imgs[s:s+self.micro_k].to(dev, non_blocking=True)
                f_mb = self.feature_extractor(x_mb)
                chunks.append(f_mb)

            feats_b = torch.cat(chunks, dim=0).unsqueeze(0)  # [1,Kb,D]
            feats_per_bag.append(feats_b)

        return feats_per_bag

    # ---------------------------------------------------------
    # 2. Aggregation
    # ---------------------------------------------------------
    def aggregate(
        self,
        feats_per_bag: List[torch.Tensor],        # [1,Kb,D] per bag
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        B = len(feats_per_bag)
        Z_list: List[torch.Tensor] = []
        extras_list: List[Any] = []

        for b in range(B):
            fb = feats_per_bag[b]
            mb = None
            if mask is not None:
                Kb = fb.shape[1]
                mb = mask[b, :Kb].to(self.device).unsqueeze(0)

            out = self.aggregator(fb, mask=mb)
            if isinstance(out, tuple):
                z_b, extra_b = out
                extras_list.append(extra_b)
            else:
                z_b = out
                extras_list.append(None)
            Z_list.append(z_b)

        Z = torch.cat(Z_list, dim=0)  # [B,D]
        return {"Z": Z, "extras": extras_list}

    # ---------------------------------------------------------
    # 3. Prediction (classification or regression)
    # ---------------------------------------------------------
    def predict(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Apply bag-level prediction head (classifier or regressor).
        """
        return self.predictor(Z)  # [B,C] or [B,1]

