from typing import List, Dict, Any

import numpy as np
import torch
from torch import Tensor
from PIL import Image

def pil_to_tensor(img: Image.Image) -> Tensor:
    arr = np.array(img, copy=False)  # HWC, uint8
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t

def bag_collate_padded(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads variable-length bags to Kmax and returns a mask.
    Use when sampler returns varying K across items.
    """
    B = len(batch)
    Ks = [b["images"].shape[0] for b in batch]
    Kmax = max(Ks)
    C, H, W = batch[0]["images"].shape[1:]
    images_pad = batch[0]["images"].new_zeros((B, Kmax, C, H, W))
    coords_pad = torch.zeros((B, Kmax, 2), dtype=torch.long)
    mask_pad = torch.zeros((B, Kmax), dtype=torch.bool)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    slide_ids: List[str] = []
    tile_ids_batched: List[List[str]] = []

    for i, b in enumerate(batch):
        K = b["images"].shape[0]
        images_pad[i, :K] = b["images"]
        coords_pad[i, :K] = b["coords"]
        mask_pad[i, :K] = True
        slide_ids.append(b["slide_id"])
        tile_ids_batched.append(b["tile_ids"])

    return {
        "images": images_pad,   # [B, Kmax, C, H, W]
        "coords": coords_pad,   # [B, Kmax, 2]
        "mask": mask_pad,       # [B, Kmax]
        "label": labels,        # [B]
        "slide_id": slide_ids,
        "tile_ids": tile_ids_batched,
    }