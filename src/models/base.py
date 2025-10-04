import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

class MILModule(nn.Module):
    """
    End-to-end MIL forward:
      CPU batch [B,Kmax,C,H,W] + mask[B,Kmax]  --stream-->  features --aggregate--> Z --classify--> logits
    """
    def __init__(
        self,
        feature_extractor: nn.Module,  # [N,C,H,W] -> [N,D]
        aggregator: nn.Module,         # ([B,K,D], mask[B,K] optional) -> [B,D]  or returns (Z, extras)
        classifier: nn.Module,         # [B,D] -> [B,C] or [B,1]
        micro_k: int = 64,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.aggregator = aggregator
        self.classifier = classifier
        self.micro_k = micro_k

    @property
    def device(self):
        return next(self.parameters()).device

    def _bag_feats_streamK(self, bag_images: torch.Tensor) -> torch.Tensor:
        """
        bag_images: [Kb,C,H,W] (likely CPU, pinned). Streams to self.device in chunks of micro_k.
        returns: [1,Kb,D] on self.device
        """
        dev = self.device
        Kb = bag_images.shape[0]
        chunks: List[torch.Tensor] = []
        for s in range(0, Kb, self.micro_k):
            x_mb = bag_images[s:s+self.micro_k].to(dev, non_blocking=True)  # [mb,C,H,W]
            f_mb = self.feature_extractor(x_mb)                              # [mb,D]
            chunks.append(f_mb)
        return torch.cat(chunks, dim=0).unsqueeze(0)  # [1,Kb,D]

    def forward(
        self,
        images: torch.Tensor,              # [B,Kmax,C,H,W] on CPU (pinned) or GPU
        mask: Optional[torch.Tensor] = None,  # [B,Kmax] bool
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "logits": [B,C] or [B,1],
            "Z":      [B,D],                # bag embeddings
            "extras": Optional[Any],        # e.g., attention weights per bag (list or tensor)
          }
        """
        B, Kmax = images.shape[:2]
        feats_per_bag: List[torch.Tensor] = []
        bag_masks: List[Optional[torch.Tensor]] = []

        # 1) Stream each bag -> features [1,Kb,D]
        for b in range(B):
            Kb = int(mask[b].sum().item()) if mask is not None else Kmax
            feats_b = self._bag_feats_streamK(images[b, :Kb])           # [1,Kb,D] on device
            feats_per_bag.append(feats_b)
            bag_masks.append(mask[b, :Kb].to(self.device) if mask is not None else None)

        # 2) Aggregate each bag independently (keeps code simple for variable K)
        Z_list: List[torch.Tensor] = []
        extras_list: List[Any] = []
        for b in range(B):
            fb = feats_per_bag[b]                # [1,Kb,D]
            mb = bag_masks[b]                    # [Kb] or None
            out = self.aggregator(fb, mask=mb.unsqueeze(0) if mb is not None else None)
            if isinstance(out, tuple):
                z_b, extra_b = out              # [1,D], extras (e.g., attn weights [1,Kb])
                extras_list.append(extra_b)
            else:
                z_b = out                        # [1,D]
                extras_list.append(None)
            Z_list.append(z_b)

        Z = torch.cat(Z_list, dim=0)             # [B,D]
        logits = self.classifier(Z)              # [B,C] or [B,1]

        return {"logits": logits, "Z": Z, "extras": extras_list}