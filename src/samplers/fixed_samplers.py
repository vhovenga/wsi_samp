import torch
from typing import List, Dict, Any, Optional, Tuple
from .base_sampler import NonLearnableSampler

class UniformSampler(NonLearnableSampler):
    def __init__(self, sample_size: int, with_replacement: bool = False):
        super().__init__(sample_kwargs = {"sample_size": sample_size, "with_replacement": with_replacement})


    def forward(self, lowres, coords_pad, coord_mask):
        # Return None for scores to indicate uniform
        scores = None
        return scores, {}
 
    def sample(
        self,
        scores: Optional[torch.Tensor],   # always None here
        coord_mask: torch.Tensor,         # [B, Nmax] bool
        sample_size: int,
        with_replacement: bool
        ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Uniformly sample K indices per slide.
        """
        B, Nmax = coord_mask.shape
        device = coord_mask.device
        idx_list: List[torch.Tensor] = []

        for b in range(B):
            valid = torch.nonzero(coord_mask[b], as_tuple=False).squeeze(1)  # [N_b]
            n_valid = valid.numel()
            if n_valid == 0:
                idx_list.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            if with_replacement:
                # sample with replacement
                choice = torch.randint(0, n_valid, (sample_size,), device=device)
                idxs = valid[choice]
            else:
                # sample without replacement
                k_b = min(sample_size, n_valid)
                perm = torch.randperm(n_valid, device=device)[:k_b]
                idxs = valid[perm]

            idx_list.append(idxs)

        return idx_list, {"method": "uniform", "with_replacement": with_replacement}
    
class UseAllSampler(NonLearnableSampler):
    """
    Returns all valid indices for each slide (no learnable params).
    Useful for debugging or when you want the downstream module to handle sub-sampling.
    """
    def __init__(self):
        super().__init__()

    def forward(self, lowres, coords_pad, coord_mask):
        # No scoring needed
        return None, {}

    def sample(
        self,
        scores: Optional[torch.Tensor],    
        coord_mask: torch.Tensor,          # [B, Nmax] bool
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        B, Nmax = coord_mask.shape
        device = coord_mask.device
        idx_list: List[torch.Tensor] = []
        for b in range(B):
            valid = torch.nonzero(coord_mask[b], as_tuple=False).squeeze(1)  # [N_b]
            idx_list.append(valid.to(device=device, dtype=torch.long))
        return idx_list, {"method": "return_all"}
    
FIXED_SAMPLERS = {
    "UniformSampler": UniformSampler,   
    "UseAllSampler": UseAllSampler
}