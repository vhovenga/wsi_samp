from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF

# ---------- Lightweight on-demand tile loader ----------

class SlideView:
    """Loads a subset of tiles for a slide on demand."""
    def __init__(self, root: Path, patch_uris: List[str], image_mode: str = "RGB", patch_transform: Optional[Any] = None):
        self.root = Path(root)
        self.uris = patch_uris
        self.mode = image_mode
        self.tfm = patch_transform

    def fetch(self, idxs: List[int]) -> Tensor:
        imgs: List[Tensor] = []
        for i in idxs:
            with Image.open(self.root / self.uris[i]) as im:
                im = im.convert(self.mode)
                if self.tfm is not None:
                    t = self.tfm(im)    # <- must end with ToTensor
                else:
                    t = TF.to_tensor(im)
            imgs.append(t)
        return torch.stack(imgs, 0) if imgs else torch.zeros(0)

# ---------- Dataset that returns lowres + ragged metadata ----------

@dataclass
class SlideRecord:
    slide_id: str
    label: int
    coords: np.ndarray         # [N,2] int64
    patch_uris: List[str]
    lowres_uri: Optional[str]  # optional; can be None

class SlideDataset(Dataset):
    """
    Slide-level dataset for MIL with a batchable low-res pathway.

    Each item returns:
      {
        "slide_id": str,
        "label": LongTensor [],
        "lowres": FloatTensor [C, H0, W0],      # fixed size for batching
        "coords": np.ndarray [N, 2],            # ragged, padded in collate
        "view": SlideView,                      # to fetch [K,C,H,W] later
      }
    """
    def __init__(
            self,
            root: str | Path,
            split: str,
            tiles_parquet_name: str = "tiles.parquet",
            labels_parquet_rel: str = "labels/slide_labels.parquet",
            splits_parquet_rel: str = "splits/fold1.parquet",
            image_mode: str = "RGB",
            lowres_size: tuple[int, int] = (256, 256),
            lowres_parquet_rel: Optional[str] = None,   # by default None → look under parquet/lowres/thumbnails.parquet
            lowres_transform: Optional[Any] = None,
            patch_transform: Optional[Any] = None,
            cache_lowres_in_ram: bool = True,
        ):
            super().__init__()
            self.root = Path(root).resolve()
            self.split = split
            self.image_mode = image_mode
            self.lowres_size = lowres_size
            self.lowres_transform = lowres_transform
            self.patch_transform = patch_transform
            self.cache_lowres_in_ram = cache_lowres_in_ram

            pq_dir = self.root / "parquet"
            tiles_parquet = pq_dir / tiles_parquet_name
            labels_parquet = pq_dir / labels_parquet_rel
            splits_parquet = pq_dir / splits_parquet_rel

            # if user didn't specify, default to "parquet/lowres/thumbnails.parquet"
            if lowres_parquet_rel is None:
                lowres_parquet = pq_dir / "lowres" / "thumbnails.parquet"
                if not lowres_parquet.exists():
                    warnings.warn(
                        f"[SlideDataset] No lowres parquet found at {lowres_parquet}. "
                        "Falling back to using first tile per slide as lowres."
                    )
                    lowres_parquet = None
            else:
                lowres_parquet = pq_dir / lowres_parquet_rel

            # --- same as before ---
            splits_df = pd.read_parquet(splits_parquet)
            split_slide_ids = set(
                splits_df.loc[splits_df["split"] == split, "slide_id"].drop_duplicates().tolist()
            )
            if not split_slide_ids:
                raise RuntimeError(f"No slides for split='{split}' in {splits_parquet_rel}.")

            labels_df = pd.read_parquet(labels_parquet)
            labels_df = labels_df[labels_df["slide_id"].isin(split_slide_ids)]
            if labels_df.empty:
                raise RuntimeError("Labels missing for selected split slides.")
            labels = {r.slide_id: int(r.label) for r in labels_df.itertuples(index=False)}

            tiles_df = pd.read_parquet(tiles_parquet)
            need = {"slide_id", "tile_id", "x", "y", "patch_uri"}
            if not need.issubset(tiles_df.columns):
                raise RuntimeError(f"tiles.parquet must have columns {need}, got {tiles_df.columns.tolist()}")
            tiles_df = tiles_df[tiles_df["slide_id"].isin(split_slide_ids)].copy()
            if tiles_df.empty:
                raise RuntimeError("No tiles for selected split.")

            # Optional lowres / thumbnail URIs
            lowres_map: Dict[str, Optional[str]] = {}
            if lowres_parquet is not None:
                ldf = pd.read_parquet(lowres_parquet)
                if not {"slide_id", "lowres_uri"}.issubset(ldf.columns):
                    raise RuntimeError("lowres parquet must have columns ['slide_id','lowres_uri']")
                lowres_map = {r.slide_id: r.lowres_uri for r in ldf.itertuples(index=False)}
            else:
                lowres_map = {sid: None for sid in split_slide_ids}

            # build records
            self.records: List[SlideRecord] = []
            for sid, g in tiles_df.groupby("slide_id", sort=True):
                if sid not in labels:
                    continue
                g = g.sort_values("tile_id").reset_index(drop=True)
                coords = g[["x", "y"]].to_numpy(np.int64)
                patch_uris = g["patch_uri"].tolist()
                self.records.append(
                    SlideRecord(
                        slide_id=sid,
                        label=labels[sid],
                        coords=coords,
                        patch_uris=patch_uris,
                        lowres_uri=lowres_map.get(sid, None),
                    )
                )
            if not self.records:
                raise RuntimeError("After filtering, no slides remained with both tiles and labels.")

            self._lowres_cache: Dict[str, Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_lowres(self, rec: SlideRecord) -> Tensor:
        if self.cache_lowres_in_ram and rec.slide_id in self._lowres_cache:
            return self._lowres_cache[rec.slide_id]

        src_uri = rec.lowres_uri if rec.lowres_uri is not None else rec.patch_uris[0]
        with Image.open(self.root / src_uri) as im:
            im = im.convert(self.image_mode)
            im = im.resize(self.lowres_size[::-1], Image.BILINEAR)

            if self.lowres_transform is not None:
                t = self.lowres_transform(im)   # <- must end with ToTensor
            else:
                t = TF.to_tensor(im)            # fallback

        if self.cache_lowres_in_ram:
            self._lowres_cache[rec.slide_id] = t
        return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        lowres = self._load_lowres(rec)                      # [C,H0,W0]
        view = SlideView(self.root, rec.patch_uris, self.image_mode, self.patch_transform)
        return {
            "slide_id": rec.slide_id,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "lowres": lowres,                                # [C,H0,W0] (batchable)
            "coords": rec.coords,                            # ragged, pad in collate
            "view": view,                                    # fetch([idxs]) -> [K,C,H,W]
        }

# ---------- Collate: batch lowres, pad coords, keep views as a list ----------

def slide_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    lows = torch.stack([b["lowres"] for b in batch], 0)  # [B,C,H0,W0]
    labels = torch.stack([b["label"] for b in batch], 0) # [B]

    Ns = [b["coords"].shape[0] for b in batch]
    Nmax = max(Ns)
    coords_pad = torch.zeros(B, Nmax, 2, dtype=torch.float32)
    coord_mask = torch.zeros(B, Nmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["coords"].shape[0]
        coords_pad[i, :n] = torch.as_tensor(b["coords"], dtype=torch.float32)
        coord_mask[i, :n] = True

    return {
        "slide_ids": [b["slide_id"] for b in batch],
        "labels": labels,                # [B]
        "lowres": lows,                  # [B,C,H0,W0]
        "coords_pad": coords_pad,        # [B,Nmax,2]
        "coord_mask": coord_mask,        # [B,Nmax]
        "views": [b["view"] for b in batch],  # list[SlideView]
        "Ns": torch.tensor(Ns, dtype=torch.long),  # [B]
    }

