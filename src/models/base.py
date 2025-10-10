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
        feature_extractor: Optional[nn.Module],  # [N,C,H,W] -> [N,D]
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
        
        assert self.feature_extractor is not None, "No feature extractor defined"

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

            feats_b = torch.cat(chunks, dim=0) # [Kb,D]
            feats_per_bag.append(feats_b)

        return feats_per_bag

    # ---------------------------------------------------------
    # 2. Aggregation
    # ---------------------------------------------------------
    def aggregate(
        self,
        feats_per_bag: List[torch.Tensor],        # each [Kb, D]
        mask: Optional[torch.Tensor] = None       # [B, Kmax] optional
    ) -> Dict[str, Any]:
        """
        Aggregate a list of variable-length feature tensors into bag-level embeddings.
        Each entry in feats_per_bag is [Kb, D].
        """
        B = len(feats_per_bag)

        # pad features to [B, Kmax, D]
        Kmax = max(fb.shape[0] for fb in feats_per_bag)
        D = feats_per_bag[0].shape[1]
        feats_pad = torch.zeros(B, Kmax, D, device=self.device, dtype=feats_per_bag[0].dtype)
        mask_pad = torch.zeros(B, Kmax, device=self.device, dtype=torch.bool)

        for b, fb in enumerate(feats_per_bag):
            Kb = fb.shape[0]
            feats_pad[b, :Kb] = fb.to(self.device)
            mask_pad[b, :Kb] = True

        # batched forward through aggregator
        out = self.aggregator(feats_pad, mask=mask_pad)

        if isinstance(out, tuple):
            Z, extras = out
        else:
            Z, extras = out, [None] * B

        # normalize extras output shape
        if not isinstance(extras, list):
            extras = [extras for _ in range(B)]

        return {"Z": Z, "extras": extras}
    # ---------------------------------------------------------
    # 3. Prediction (classification or regression)
    # ---------------------------------------------------------
    def predict(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Apply bag-level prediction head (classifier or regressor).
        """
        return self.predictor(Z)  # [B,C] or [B,1]

