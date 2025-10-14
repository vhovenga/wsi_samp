import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------- Lightweight on-demand tile/feature view ----------
class SlideFeatureView:
    """Loads image tiles and HDF5-stored features on demand using absolute paths."""

    def __init__(
        self,
        patch_uris: List[str],
        h5_path: Optional[str] = None,
        image_mode: str = "RGB",
        patch_transform: Optional[Any] = None,
    ):
        self.patch_uris = patch_uris
        self.h5_path = h5_path
        self.image_mode = image_mode
        self.patch_transform = patch_transform

    # -----------------------------
    # Fetch image tiles
    # -----------------------------
    def fetch_tiles(self, idxs: Optional[List[int]] = None) -> torch.Tensor:
        """Load image tiles by index (or all if None)."""
        if idxs is None:
            idxs = list(range(len(self.patch_uris)))

        imgs = []
        for i in idxs:
            with Image.open(self.patch_uris[i]) as im:
                im = im.convert(self.image_mode)
                t = self.patch_transform(im) if self.patch_transform else TF.to_tensor(im)
                imgs.append(t)

        if not imgs:
            return torch.empty((0, 3, 1, 1))
        return torch.stack(imgs, 0)

    # -----------------------------
    # Fetch features from h5
    # -----------------------------
    def fetch_features(self, idxs: Optional[List[int]] = None) -> torch.Tensor:
        """Load precomputed features from a per-slide HDF5 file."""
        if self.h5_path is None:
            raise ValueError("No feature HDF5 file provided.")

        with h5py.File(self.h5_path, "r") as f:
            feats = f["features"]
            data = feats[()] if idxs is None else feats[idxs]
        return torch.from_numpy(data).float()

    def fetch_coords(self, idxs: Optional[List[int]] = None) -> np.ndarray:
        """Load patch coordinates (x,y) from HDF5."""
        if self.h5_path is None:
            raise ValueError("No feature HDF5 file provided.")
        with h5py.File(self.h5_path, "r") as f:
            coords = f["coords"]
            data = coords[()] if idxs is None else coords[idxs]
        return data

    def fetch_patch_indices(self, idxs: Optional[List[int]] = None) -> np.ndarray:
        """Load patch_grid_idx values from HDF5."""
        if self.h5_path is None:
            raise ValueError("No feature HDF5 file provided.")
        with h5py.File(self.h5_path, "r") as f:
            pg = f["patch_grid_idx"]
            data = pg[()] if idxs is None else pg[idxs]
        return data

    @property
    def has_features(self) -> bool:
        return self.h5_path is not None


# ---------- Slide-level record ----------
@dataclass
class SlideRecord:
    slide_id: str
    label: int
    coords: np.ndarray
    patch_uris: List[str]
    lowres_uri: Optional[str]
    feature_h5_uri: Optional[str] = None


# ---------- Dataset ----------
class SlideDataset(Dataset):
    """Slide-level dataset with per-slide HDF5 feature files and absolute paths."""

    def __init__(
        self,
        split: str,
        tiles_parquet: str = "tiles.parquet",
        labels_parquet: str = "labels/slide_labels.parquet",
        splits_parquet: str = "splits/fold1.parquet",
        features_parquet: str = "features.parquet",
        lowres_parquet: Optional[str] = None,
        image_mode: str = "RGB",
        lowres_size: tuple[int, int] = (256, 256),
        lowres_transform: Optional[Any] = None,
        patch_transform: Optional[Any] = None,
        cache_lowres_in_ram: bool = True,
        task_type: str = "classification",   # "classification" or "regression"
    ):
        super().__init__()

        assert task_type in {"classification", "regression"}, f"Invalid task_type: {task_type}"
        self.task_type = task_type
        self.image_mode = image_mode
        self.lowres_size = lowres_size
        self.lowres_transform = lowres_transform
        self.patch_transform = patch_transform
        self.cache_lowres_in_ram = cache_lowres_in_ram
        self.split = split

        # --- load parquet tables ---
        tiles_df = pd.read_parquet(tiles_parquet)
        labels_df = pd.read_parquet(labels_parquet)
        splits_df = pd.read_parquet(splits_parquet)
        feats_df = pd.read_parquet(features_parquet)

        # --- get split slide IDs ---
        split_slide_ids = set(splits_df.loc[splits_df["split"] == split, "slide_id"])
        if not split_slide_ids:
            raise RuntimeError(f"No slides for split='{split}' found in {splits_parquet}.")

        # --- handle labels ---
        labels_df = labels_df[labels_df["slide_id"].isin(split_slide_ids)]

        if self.task_type == "classification":
            raw_labels = labels_df.set_index("slide_id")["label"]
            if not np.issubdtype(raw_labels.dtype, np.number):
                cats = sorted(raw_labels.unique().tolist())
                mapping = {cat: i for i, cat in enumerate(cats)}
                labels = {sid: mapping[lbl] for sid, lbl in raw_labels.items()}
                self.label_mapping = mapping
            else:
                labels = {r.slide_id: int(r.label) for r in labels_df.itertuples(index=False)}
                self.label_mapping = None
        else:  # regression
            labels = {r.slide_id: float(r.label) for r in labels_df.itertuples(index=False)}
            self.label_mapping = None

        # --- filter tiles ---
        need = {"slide_id", "patch_grid_idx", "x", "y", "patch_uri"}
        if not need.issubset(tiles_df.columns):
            raise RuntimeError(f"tiles.parquet must have {need}, got {tiles_df.columns.tolist()}")
        tiles_df = tiles_df[tiles_df["slide_id"].isin(split_slide_ids)]

        # --- feature map ---
        feat_map = {r.slide_id: r.feature_uri for r in feats_df.itertuples(index=False)}

        # --- lowres thumbnails ---
        if lowres_parquet:
            lowres_df = pd.read_parquet(lowres_parquet)
            lowres_map = {r.slide_id: r.lowres_uri for r in lowres_df.itertuples(index=False)}
        else:
            warnings.warn("No lowres thumbnails found; using first tile as fallback.")
            lowres_map = {sid: None for sid in split_slide_ids}

        # --- build records ---
        self.records: List[SlideRecord] = []
        for sid, g in tiles_df.groupby("slide_id", sort=True):
            if sid not in labels:
                continue
            g = g.sort_values("patch_grid_idx").reset_index(drop=True)
            coords = g[["x", "y"]].to_numpy(np.int64)
            patch_uris = g["patch_uri"].tolist()
            feature_h5 = feat_map.get(sid, None)
            lowres_uri = lowres_map.get(sid, None)
            self.records.append(
                SlideRecord(
                    slide_id=sid,
                    label=labels[sid],
                    coords=coords,
                    patch_uris=patch_uris,
                    lowres_uri=lowres_uri,
                    feature_h5_uri=feature_h5,
                )
            )

        self._lowres_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    # ---------- Lowres ----------
    def _load_lowres(self, rec: SlideRecord) -> torch.Tensor:
        if self.cache_lowres_in_ram and rec.slide_id in self._lowres_cache:
            return self._lowres_cache[rec.slide_id]

        src_uri = rec.lowres_uri if rec.lowres_uri else rec.patch_uris[0]
        with Image.open(src_uri) as im:
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
        dtype = torch.float if self.task_type == "regression" else torch.long
        view = SlideFeatureView(
            patch_uris=rec.patch_uris,
            h5_path=rec.feature_h5_uri,
            image_mode=self.image_mode,
            patch_transform=self.patch_transform,
        )
        return {
            "slide_id": rec.slide_id,
            "label": torch.tensor(rec.label, dtype=dtype),
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