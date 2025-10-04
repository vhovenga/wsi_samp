import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPoolAggregator(nn.Module):
    """
    MIL aggregator with mean pooling over instances.

    Input:  feats [B, K, D]
    Output: bag_emb [B, D]
    """
    def __init__(self, in_dim: int = 2048, 
                 l2_normalize_feats: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.l2_normalize_feats = l2_normalize_feats

    def forward(
        self,
        feats: torch.Tensor,           # [B, K, D]
        lengths: torch.Tensor = None,  # [B] optional
        mask: torch.Tensor = None      # [B, K] optional
    ) -> torch.Tensor:                 # [B, D]
        assert feats.dim() == 3, f"Expected (B,K,D), got {feats.shape}"
        B, K, D = feats.shape
        assert D == self.in_dim, f"in_dim mismatch: {self.in_dim} vs {D}"

        if self.l2_normalize_feats:
            feats = F.normalize(feats, p=2, dim=-1)

        # Build mask if not given
        if mask is None:
            if lengths is not None:
                arange = torch.arange(K, device=feats.device)[None, :]  # [1,K]
                mask = arange < lengths[:, None]                        # [B,K]
            else:
                mask = torch.ones(B, K, dtype=torch.bool, device=feats.device)

        mask = mask.unsqueeze(-1)         # [B,K,1]
        feats_masked = feats * mask       # zero invalid
        denom = mask.sum(dim=1).clamp_min(1)  # [B,1]
        bag_emb = feats_masked.sum(dim=1) / denom  # [B,D]

        return bag_emb