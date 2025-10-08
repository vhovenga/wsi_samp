from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import pyarrow.dataset as ds

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF


# ---------- Lightweight on-demand tile/feature view ----------
class SlideFeatureView:
    """Loads image tiles, features, or both on demand from disk."""
    def __init__(
        self,
        root: Path,
        patch_uris: List[str],
        feature_uris: Optional[List[str]] = None,
        image_mode: str = "RGB",
        patch_transform: Optional[Any] = None,
        feature_format: str = "pt",
    ):
        self.root = Path(root)
        self.patch_uris = patch_uris
        self.feature_uris = feature_uris
        self.image_mode = image_mode
        self.patch_transform = patch_transform
        self.feature_format = feature_format

    # -----------------------------
    # Fetch image tiles
    # -----------------------------
    def fetch_tiles(self, idxs: Optional[List[int]] = None) -> torch.Tensor:
        """Load image tiles by index (or all if None)."""
        if idxs is None:
            idxs = list(range(len(self.patch_uris)))

        imgs = []
        for i in idxs:
            with Image.open(self.root / self.patch_uris[i]) as im:
                im = im.convert(self.image_mode)
                t = (
                    self.patch_transform(im)
                    if self.patch_transform is not None
                    else TF.to_tensor(im)
                )
                imgs.append(t)

        if len(imgs) == 0:
            return torch.empty((0, 3, 1, 1))

        return torch.stack(imgs, 0)

    # -----------------------------
    # Fetch feature tensors
    # -----------------------------
    def fetch_features(self, idxs: Optional[List[int]] = None) -> torch.Tensor:
        """Load precomputed features by index (or all if None)."""
        if self.feature_uris is None:
            raise ValueError("feature_uris is None; this view has no features.")

        if idxs is None:
            idxs = list(range(len(self.feature_uris)))

        feats = []
        for i in idxs:
            fpath = self.root / self.feature_uris[i]
            if self.feature_format == "pt":
                f = torch.load(fpath, map_location="cpu")
            else:
                f = torch.from_numpy(np.load(fpath)).float()
            feats.append(f)

        if len(feats) == 0:
            return torch.empty((0, feats[0].numel() if feats else 0))

        return torch.stack(feats, 0)
    
    @property
    def has_features(self) -> bool:
        return self.feature_uris is not None


# ---------- Slide-level record ----------
@dataclass
class SlideRecord:
    slide_id: str
    label: int
    coords: np.ndarray
    patch_uris: List[str]
    lowres_uri: Optional[str]
    feature_uris: Optional[List[str]] = None


# ---------- Dataset ----------
class SlideDataset(Dataset):
    """
    Slide-level dataset for MIL with per-slide tiles and feature URIs.
    Uses features stored under:
      slides/<slide_id>/features/feature_set=<name>/tile_*.pt
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        tiles_parquet_name: str = "tiles.parquet",
        labels_parquet_rel: str = "labels/slide_labels.parquet",
        splits_parquet_rel: str = "splits/fold1.parquet",
        features_parquet_rel: str = "features.parquet",
        lowres_parquet_rel: Optional[str] = None,
        image_mode: str = "RGB",
        lowres_size: tuple[int, int] = (256, 256),
        lowres_transform: Optional[Any] = None,
        patch_transform: Optional[Any] = None,
        cache_lowres_in_ram: bool = True,
        feature_format: str = "npy"
    ):
        super().__init__()
        self.root = Path(root).resolve()
        self.split = split
        self.image_mode = image_mode
        self.lowres_size = lowres_size
        self.lowres_transform = lowres_transform
        self.patch_transform = patch_transform
        self.cache_lowres_in_ram = cache_lowres_in_ram
        self.feature_format = feature_format

        pq_dir = self.root / "parquet"
        tiles_parquet = pq_dir / tiles_parquet_name
        labels_parquet = pq_dir / labels_parquet_rel
        splits_parquet = pq_dir / splits_parquet_rel
        features_parquet = pq_dir / features_parquet_rel

        # --- split and labels ---
        splits_df = pd.read_parquet(splits_parquet)
        split_slide_ids = set(
            splits_df.loc[splits_df["split"] == split, "slide_id"]
            .drop_duplicates()
            .tolist()
        )
        if not split_slide_ids:
            raise RuntimeError(f"No slides for split='{split}' in {splits_parquet_rel}.")

        labels_df = pd.read_parquet(labels_parquet)
        labels_df = labels_df[labels_df["slide_id"].isin(split_slide_ids)]
        labels = {r.slide_id: int(r.label) for r in labels_df.itertuples(index=False)}

        # --- tiles parquet ---
        tiles_df = pd.read_parquet(tiles_parquet)
        need = {"slide_id", "tile_id", "x", "y", "patch_uri"}
        if not need.issubset(tiles_df.columns):
            raise RuntimeError(f"tiles.parquet must have {need}, got {tiles_df.columns.tolist()}")
        tiles_df = tiles_df[tiles_df["slide_id"].isin(split_slide_ids)].copy()

        # --- optional features parquet ---
        if features_parquet.exists():
            feats_df = pd.read_parquet(features_parquet)
            feats_df = feats_df[feats_df["slide_id"].isin(split_slide_ids)]
            need_feat = {"slide_id", "tile_id", "feature_uri"}
            if not need_feat.issubset(feats_df.columns):
                raise RuntimeError(f"features.parquet must have {need_feat}, got {feats_df.columns.tolist()}")

            # merge on slide_id + tile_id (keep coords/x/y from tiles)
            tiles_df = tiles_df.merge(feats_df, on=["slide_id", "tile_id", "x", "y"], how="left")

            # diagnostic for missing feature URIs
            missing = tiles_df[tiles_df["feature_uri"].isna()]
            if not missing.empty:
                print(
                    f"[SlideDataset] ⚠️ {len(missing)} tiles missing feature files "
                    f"({missing['slide_id'].nunique()} slides affected)."
                )
                print(missing[["slide_id", "tile_id"]].head(10).to_string(index=False))

            print(f"[SlideDataset] Merged features for {tiles_df['slide_id'].nunique()} slides.")
        else:
            tiles_df["feature_uri"] = None
            print("[SlideDataset] No features.parquet found — proceeding without feature URIs.")

        # --- lowres thumbnails ---
        if lowres_parquet_rel is None:
            lowres_parquet = pq_dir / "lowres" / "thumbnails.parquet"
        else:
            lowres_parquet = pq_dir / lowres_parquet_rel

        if lowres_parquet.exists():
            ldf = pd.read_parquet(lowres_parquet)
            lowres_map = {r.slide_id: r.lowres_uri for r in ldf.itertuples(index=False)}
        else:
            warnings.warn("[SlideDataset] No lowres thumbnails found; using first tile as fallback.")
            lowres_map = {sid: None for sid in split_slide_ids}

        # --- build slide records ---
        self.records: List[SlideRecord] = []
        for sid, g in tiles_df.groupby("slide_id", sort=True):
            if sid not in labels:
                continue

            g = g.sort_values("tile_id").reset_index(drop=True)
            coords = g[["x", "y"]].to_numpy(np.int64)
            patch_uris = g["patch_uri"].tolist()
            feat_uris = g["feature_uri"].tolist() if "feature_uri" in g.columns else None

            self.records.append(
                SlideRecord(
                    slide_id=sid,
                    label=labels[sid],
                    coords=coords,
                    patch_uris=patch_uris,
                    lowres_uri=lowres_map.get(sid, None),
                    feature_uris=feat_uris,
                )
            )

        self._lowres_cache: Dict[str, Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    # ---------- Lowres ----------
    def _load_lowres(self, rec: SlideRecord) -> Tensor:
        if self.cache_lowres_in_ram and rec.slide_id in self._lowres_cache:
            return self._lowres_cache[rec.slide_id]

        src_uri = rec.lowres_uri if rec.lowres_uri else rec.patch_uris[0]
        with Image.open(self.root / src_uri) as im:
            im = im.convert(self.image_mode)
            im = im.resize(self.lowres_size[::-1], Image.BILINEAR)
            t = self.lowres_transform(im) if self.lowres_transform else TF.to_tensor(im)
        if self.cache_lowres_in_ram:
            self._lowres_cache[rec.slide_id] = t
        return t

    # ---------- Get item ----------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        lowres = self._load_lowres(rec)

        view = SlideFeatureView(
            root=self.root,
            patch_uris=rec.patch_uris,
            feature_uris=rec.feature_uris,
            image_mode=self.image_mode,
            patch_transform=self.patch_transform,
            feature_format=self.feature_format,
        )

        return {
            "slide_id": rec.slide_id,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "lowres": lowres,
            "coords": rec.coords,
            "view": view,
        }


# ---------- Collate ----------
def slide_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    lows = torch.stack([b["lowres"] for b in batch], 0)
    labels = torch.stack([b["label"] for b in batch], 0)

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
        "labels": labels,
        "lowres": lows,
        "coords_pad": coords_pad,
        "coord_mask": coord_mask,
        "views": [b["view"] for b in batch],
        "Ns": torch.tensor(Ns, dtype=torch.long),
    }