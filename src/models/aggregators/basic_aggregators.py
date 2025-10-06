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

class AttnMeanPoolAggregator(nn.Module):
    """
    MIL aggregator with attention-weighted mean pooling.
    Learns instance weights and computes a softmax-weighted average.

    Input:  feats [B, K, D]
    Output: bag_emb [B, D]
    """
    def __init__(
        self,
        in_dim: int = 2048,
        attn_hidden_dim: int = 64,
        l2_normalize_feats: bool = False,
        softmax_temp: float = 2.0,       # temperature >1 smooths weights
        dropout_p: float = 0.2,          # helps prevent memorization
    ):
        super().__init__()
        self.in_dim = in_dim
        self.l2_normalize_feats = l2_normalize_feats
        self.softmax_temp = softmax_temp

        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.Linear(attn_hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        feats: torch.Tensor,           # [B, K, D]
        lengths: torch.Tensor = None,  # [B] optional
        mask: torch.Tensor = None,     # [B, K] optional
        return_attn: bool = False,     # optional for debugging
    ):
        assert feats.dim() == 3, f"Expected (B,K,D), got {feats.shape}"
        B, K, D = feats.shape
        assert D == self.in_dim, f"in_dim mismatch: {self.in_dim} vs {D}"

        if self.l2_normalize_feats:
            feats = F.normalize(feats, p=2, dim=-1)

        if mask is None:
            if lengths is not None:
                arange = torch.arange(K, device=feats.device)[None, :]
                mask = arange < lengths[:, None]
            else:
                mask = torch.ones(B, K, dtype=torch.bool, device=feats.device)

        # Compute attention logits
        attn_logits = self.attn_net(feats).squeeze(-1)  # [B,K]
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        # Temperature-scaled softmax
        attn_weights = F.softmax(attn_logits / self.softmax_temp, dim=1)  # [B,K]
        attn_weights = attn_weights * mask
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted mean
        bag_emb = torch.sum(feats * attn_weights.unsqueeze(-1), dim=1)  # [B,D]

        if return_attn:
            return bag_emb, attn_weights
        return bag_emb

class MaxPoolAggregator(nn.Module):
    """
    MIL aggregator with max pooling over instances.

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

        # Mask invalid positions by setting them to -inf before max
        mask = mask.unsqueeze(-1)  # [B,K,1]
        feats_masked = feats.masked_fill(~mask, float("-inf"))

        # Max over instances
        bag_emb, _ = feats_masked.max(dim=1)  # [B,D]

        # Replace any -inf (if all masked) with zeros
        bag_emb[bag_emb == float("-inf")] = 0.0

        return bag_emb