import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# ---------- Lightweight on-demand tile/feature view ----------
class SlideFeatureView:
    """Loads image tiles and features (HDF5 or .pt) on demand using absolute paths.
    If included_idxs is provided, this view behaves as if only those patches exist.
    """

    def __init__(
        self,
        patch_uris: List[str],
        feature_path: Optional[str] = None,
        image_mode: str = "RGB",
        patch_transform: Optional[Any] = None,
        included_idxs: Optional[np.ndarray] = None,
    ):
        # core data
        self._all_patch_uris = patch_uris
        self.feature_path = feature_path
        self.image_mode = image_mode
        self.patch_transform = patch_transform

        # --- handle inclusion subset ---

        if included_idxs is not None:
            self.included_idxs = np.asarray(included_idxs)  # absolute indices in feature store
            self.patch_uris = patch_uris                    # already the subset; do NOT reindex
        else:
            self.included_idxs = None
            self.patch_uris = patch_uris

        # --- feature file type ---
        if feature_path:
            if feature_path.endswith(".h5"):
                self.feature_type = "h5"
            elif feature_path.endswith(".pt"):
                self.feature_type = "pt"
            else:
                raise ValueError(f"Unsupported feature file: {feature_path}")

    # -----------------------------
    # internal helpers
    # -----------------------------
    def __len__(self) -> int:
        return len(self.patch_uris)

    def _map_idxs(self, idxs: Optional[Sequence[int]]) -> np.ndarray:
        """Map relative indices (subset) → absolute indices (in feature file)."""
        if idxs is None:
            return self.included_idxs if self.included_idxs is not None else np.arange(len(self._all_patch_uris))
        idxs = np.asarray(idxs)
        if self.included_idxs is not None:
            return self.included_idxs[idxs]
        return idxs

    # -----------------------------
    # Fetch image tiles
    # -----------------------------
    def fetch_tiles(self, idxs: Optional[Sequence[int]] = None) -> torch.Tensor:
        imgs = []
        sel = idxs if idxs is not None else range(len(self.patch_uris))
        for i in sel:
            with Image.open(self.patch_uris[i]) as im:
                im = im.convert(self.image_mode)
                t = self.patch_transform(im) if self.patch_transform else TF.to_tensor(im)
                imgs.append(t)
        if not imgs:
            return torch.empty((0, 3, 1, 1))
        return torch.stack(imgs, 0)

    # -----------------------------
    # Unified feature loader
    # -----------------------------
    def fetch_features(self, idxs: Optional[Sequence[int]] = None) -> torch.Tensor:
        if self.feature_path is None:
            raise ValueError("No feature file provided.")
        abs_idxs = self._map_idxs(idxs)

        if self.feature_type == "h5":
            with h5py.File(self.feature_path, "r") as f:
                feats = f["features"][abs_idxs]
            return torch.from_numpy(feats).float()

        elif self.feature_type == "pt":
            data = torch.load(self.feature_path, map_location="cpu")
            feats = data["features"] if isinstance(data, dict) and "features" in data else data
            return feats[abs_idxs].float()

    def fetch_coords(self, idxs: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.feature_path is None:
            raise ValueError("No feature file provided.")
        abs_idxs = self._map_idxs(idxs)

        if self.feature_type == "h5":
            with h5py.File(self.feature_path, "r") as f:
                return f["coords"][abs_idxs]

        elif self.feature_type == "pt":
            data = torch.load(self.feature_path, map_location="cpu")
            coords = data["coords"].numpy() if torch.is_tensor(data["coords"]) else np.asarray(data["coords"])
            return coords[abs_idxs]

    def fetch_patch_indices(self, idxs: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.feature_path is None:
            raise ValueError("No feature file provided.")
        abs_idxs = self._map_idxs(idxs)

        if self.feature_type == "h5":
            with h5py.File(self.feature_path, "r") as f:
                return f["patch_grid_idx"][abs_idxs]

        elif self.feature_type == "pt":
            data = torch.load(self.feature_path, map_location="cpu")
            arr = data["patch_grid_idx"].numpy() if torch.is_tensor(data["patch_grid_idx"]) else np.asarray(data["patch_grid_idx"])
            return arr[abs_idxs]

    @property
    def has_features(self) -> bool:
        return self.feature_path is not None


# ---------- Slide-level record ----------
@dataclass
class SlideRecord:
    slide_id: str
    label: int
    coords: np.ndarray
    patch_uris: List[str]
    lowres_uri: Optional[str]
    feature_uri: Optional[str] = None
    included_idxs: Optional[np.ndarray] = None


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
        need = {"slide_id", "patch_grid_idx", "x", "y", "patch_uri", "include"}
        if not need.issubset(tiles_df.columns):
            raise RuntimeError(f"tiles.parquet must have {need}, got {tiles_df.columns.tolist()}")

        # keep only current split and included patches
        tiles_df = tiles_df[
            (tiles_df["slide_id"].isin(split_slide_ids)) &
            (tiles_df["include"].astype(bool))
        ].copy()

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
        self.records: List[SlideRecord] = []
        for sid, g in tiles_df.groupby("slide_id", sort=True):
            if sid not in labels:
                continue

            g = g.sort_values("patch_grid_idx").reset_index(drop=True)
            coords = g[["x", "y"]].to_numpy(np.int64)
            patch_uris = g["patch_uri"].tolist()
            feature_uri = feat_map.get(sid, None)
            lowres_uri = lowres_map.get(sid, None)
            included_idxs = g["patch_grid_idx"].to_numpy(np.int64)

            self.records.append(
                SlideRecord(
                    slide_id=sid,
                    label=labels[sid],
                    coords=coords,
                    patch_uris=patch_uris,
                    lowres_uri=lowres_uri,
                    feature_uri=feature_uri,
                    included_idxs=included_idxs,
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
            feature_path=rec.feature_uri,
            image_mode=self.image_mode,
            patch_transform=self.patch_transform,
            included_idxs=rec.included_idxs,  
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