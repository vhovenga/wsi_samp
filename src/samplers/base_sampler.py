import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

# -----------------------------
# Base class
# -----------------------------
class BaseSampler(nn.Module):
    """
    Batched sampler interface.

    forward(lowres, coords_pad, coord_mask) -> scores or logits per tile: [B, Nmax]
      - Implement this to produce per-tile scores (higher = better).
      - You may return (scores, aux) or just scores.

    sample(scores, coord_mask, K, method, temperature, deterministic) -> List[LongTensor], aux
      - Static method that turns scores+mask into per-slide index lists.
      - You can reuse it from your subclass or override if you need custom behavior.
    """
    def __init__(
            self, 
            sample_kwargs: Dict[str, Any] = {},
            forward_kwargs: Dict[str, Any] = {}
            ):
        super().__init__()
        self.sample_kwargs = sample_kwargs # dynamic knobs (temperature, method, etc.)
        self.forward_kwargs = forward_kwargs

    def forward(
        self,
        batch: Dict[str, Any],
        forward_kwargs: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def sample(
        self,
        scores: Optional[torch.Tensor],   # [B,Nmax] or None (None => uniform over valid)
        coord_mask: Optional[torch.Tensor],         # [B,Nmax] bool
        **kwargs, 
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Returns:
          idx_list: list[LongTensor[K_b]], one per slide
          aux: {"scores": scores, "method": method}
        """
        raise NotImplementedError
    
    def set_kwargs(self, **kwargs):
        self.sample_kwargs.update(kwargs)

    def get_kwargs(self):
        return self.sample_kwargs
    
    @staticmethod
    def _fetch_tiles(views, idx_list, device, aux):
        fetched, lengths = [], []
        for view, idxs_dev in zip(views, idx_list):
            idxs = idxs_dev.detach().cpu().tolist()
            imgs = view.fetch_tiles(idxs)
            fetched.append(imgs)
            lengths.append(imgs.shape[0])

        B = len(fetched)
        Kmax = max(lengths) if lengths else 0
        if Kmax == 0:
            return (torch.empty((B,0,3,1,1), pin_memory=True),
                    torch.zeros((B,0), dtype=torch.bool, device=device),
                    aux)

        C, H, W = fetched[0].shape[1:]
        images_pad = fetched[0].new_zeros((B, Kmax, C, H, W)).pin_memory()
        mask_pad   = torch.zeros((B, Kmax), dtype=torch.bool, device=device)
        for i, imgs in enumerate(fetched):
            k = imgs.shape[0]
            images_pad[i, :k] = imgs
            mask_pad[i, :k] = True
        return images_pad, mask_pad, aux
    

class NonLearnableSampler(BaseSampler):
    @torch.no_grad()
    def select_and_fetch(
        self,
        batch: Dict[str, Any], 
        device: torch.device, 
        ):

        coord_mask = batch["coord_mask"].to(device, non_blocking=True)
        views = batch["views"]

        scores, aux_fwd = self.forward(batch, **self.forward_kwargs)
        idx_list, aux_sel = self.sample(scores, coord_mask, **self.sample_kwargs)
        aux = {**aux_fwd, **aux_sel, "idx_list": idx_list}

        # we don’t care about grads → all ops excluded from autograd
        return self._fetch_tiles(views, idx_list, coord_mask.device, aux)
    
class LearnableSampler(BaseSampler):
    def select_and_fetch(
            self, 
            batch: Dict[str, Any], 
            device: torch.device,
            ):
        coord_mask = batch["coord_mask"].to(device, non_blocking=True)
        views = batch["views"]

        # forward tracked by autograd → grads will flow into sampler params
        scores, aux_fwd = self.forward(batch, **self.forward_kwargs)
        idx_list, aux_sel = self.sample(scores, coord_mask, **self.sample_kwargs)
        aux = {**aux_fwd, **aux_sel, "idx_list": idx_list}

        # explicitly clear big inputs (free memory after forward graph captured)
        torch.cuda.empty_cache()   # optional, forces cache release

        return self._fetch_tiles(views, idx_list, coord_mask.device, aux)