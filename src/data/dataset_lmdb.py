import os
import io
import lmdb
import torch
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.io import decode_image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import warnings


# --------------------------------------------------------
# LMDB tile view
# --------------------------------------------------------
class SlideFeatureView:
    _env_cache = {}   # per-process

    def __init__(
        self,
        lmdb_path: str,
        lmdb_keys: List[str],
        feature_path: Optional[str],
        image_mode: str,
        patch_transform: Optional[Any],
        included_idxs: Optional[np.ndarray],
    ):
        self.lmdb_path = lmdb_path
        self.lmdb_keys = lmdb_keys
        self.feature_path = feature_path
        self.image_mode = image_mode
        self.patch_transform = patch_transform
        self.included_idxs = included_idxs

        if feature_path:
            if feature_path.endswith(".h5"):
                self.feature_type = "h5"
            elif feature_path.endswith(".pt"):
                self.feature_type = "pt"
            else:
                raise ValueError(f"Unsupported feature file: {feature_path}")

    @property
    def env(self):
        """Lazy per-worker LMDB environment."""
        if self.lmdb_path not in self._env_cache:
            self._env_cache[self.lmdb_path] = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=512,
            )
        return self._env_cache[self.lmdb_path]
    def __len__(self):
        return len(self.lmdb_keys)

    def _map_idxs(self, idxs):
        if idxs is None:
            return self.included_idxs if self.included_idxs is not None else np.arange(len(self.lmdb_keys))
        idxs = np.asarray(idxs)
        if self.included_idxs is not None:
            return self.included_idxs[idxs]
        return idxs

    # --------------------------------------------------------
    # Tile loader (LMDB only)
    # --------------------------------------------------------
    def fetch_tiles(self, idxs=None):
        idxs = idxs if idxs is not None else range(len(self.lmdb_keys))
        out = []

        with self.env.begin(write=False) as txn:
            for i in idxs:
                raw = txn.get(self.lmdb_keys[i].encode())
                buf = io.BytesIO(raw)
                img = Image.open(buf).convert(self.image_mode)
                img = self.patch_transform(img) if self.patch_transform else TF.to_tensor(img)
                out.append(img)

        return torch.stack(out, 0) if out else torch.empty((0, 3, 1, 1))

    # --------------------------------------------------------
    # Feature loader
    # --------------------------------------------------------
    def fetch_features(self, idxs=None):
        if self.feature_path is None:
            raise ValueError("No feature file")

        abs_idxs = idxs

        if self.feature_type == "h5":
            with h5py.File(self.feature_path, "r") as f:
                feats = f["features"][abs_idxs]
            return torch.from_numpy(feats).float()

        data = torch.load(self.feature_path, map_location="cpu")
        feats = data["features"] if isinstance(data, dict) else data
        return feats[abs_idxs].float()

    def fetch_coords(self, idxs=None):
        if self.feature_path is None:
            raise ValueError("No feature file")
        abs_idxs = self._map_idxs(idxs)

        if self.feature_type == "h5":
            with h5py.File(self.feature_path, "r") as f:
                return f["coords"][abs_idxs]

        data = torch.load(self.feature_path, map_location="cpu")
        arr = data["coords"].numpy() if isinstance(data["coords"], torch.Tensor) else np.asarray(data["coords"])
        return arr[abs_idxs]

    def fetch_patch_indices(self, idxs=None):
        if self.feature_path is None:
            raise ValueError("No feature file")
        abs_idxs = self._map_idxs(idxs)

        if self.feature_type == "h5":
            with h5py.File(self.feature_path, "r") as f:
                return f["patch_grid_idx"][abs_idxs]

        data = torch.load(self.feature_path, map_location="cpu")
        arr = data["patch_grid_idx"].numpy() if isinstance(data["patch_grid_idx"], torch.Tensor) else np.asarray(data["patch_grid_idx"])
        return arr[abs_idxs]


@dataclass
class SlideRecord:
    slide_id: str
    label: int
    coords: np.ndarray
    lowres_uri: Optional[str]
    lmdb_pth: str                     # path to LMDB for this slide
    lmdb_keys: List[str]              # tile keys inside LMDB
    feature_uri: Optional[str]
    included_idxs: Optional[np.ndarray]





# --------------------------------------------------------
# Slide-level dataset, LMDB-only
# --------------------------------------------------------
class SlideDatasetLMDB(Dataset):
    """LMDB version matching the exact API of SlideDataset."""

    def __init__(
        self,
        split: Union[str, Sequence[str]],
        tiles_parquet: str,
        labels_parquet: str,
        splits_parquet: str,
        features_parquet: str,
        lowres_parquet: Optional[str] = None,
        image_mode: str = "RGB",
        lowres_size: tuple[int, int] = (256, 256),
        lowres_transform: Optional[Any] = None,
        patch_transform: Optional[Any] = None,
        cache_lowres_in_ram: bool = True,
        task_type: str = "classification",
    ):
        super().__init__()

        self.task_type = task_type
        self.image_mode = image_mode
        self.lowres_size = lowres_size
        self.lowres_transform = lowres_transform
        self.patch_transform = patch_transform
        self.cache_lowres_in_ram = cache_lowres_in_ram

        tiles_df = pd.read_parquet(tiles_parquet)
        labels_df = pd.read_parquet(labels_parquet)
        splits_df = pd.read_parquet(splits_parquet)
        feats_df = pd.read_parquet(features_parquet)

        # ---- split selection ----
        split_values = [split] if isinstance(split, str) else list(split)
        slide_ids = set(splits_df.loc[splits_df["split"].isin(split_values), "slide_id"])

        # ---- labels ----
        labels_df = labels_df[labels_df["slide_id"].isin(slide_ids)]
        if task_type == "classification":
            raw = labels_df.set_index("slide_id")["label"]
            if not np.issubdtype(raw.dtype, np.number):
                cats = sorted(raw.unique())
                mapping = {c: i for i, c in enumerate(cats)}
                labels = {sid: mapping[lbl] for sid, lbl in raw.items()}
            else:
                labels = {r.slide_id: int(r.label) for r in labels_df.itertuples(index=False)}
        else:
            labels = {r.slide_id: float(r.label) for r in labels_df.itertuples(index=False)}

        # ---- tiles ----
        required = {"slide_id", "patch_grid_idx", "x", "y", "patch_uri", "lmdb_pth", "include"}
        if not required.issubset(tiles_df.columns):
            raise RuntimeError("tiles.parquet missing LMDB columns.")

        tiles_df = tiles_df[
            (tiles_df.slide_id.isin(slide_ids)) &
            (tiles_df.include.astype(bool))
        ].copy()

        # ---- lowres ----
        if lowres_parquet:
            lowres_df = pd.read_parquet(lowres_parquet)
            lowres_map = {r.slide_id: r.lowres_uri for r in lowres_df.itertuples(index=False)}
        else:
            warnings.warn("No lowres thumbnails found; using first tile as fallback.")
            lowres_map = {}

        # ---- feature map ----
        feat_map = {r.slide_id: r.feature_uri for r in feats_df.itertuples(index=False)}

        # ---- records ----
        self.records = []
        for sid, g in tiles_df.groupby("slide_id"):
            if sid not in labels:
                continue

            g = g.sort_values("patch_grid_idx").reset_index(drop=True)
            coords = g[["x", "y"]].to_numpy(np.int64)
            lmdb_pth = g["lmdb_pth"].iloc[0]
            lmdb_keys = g["patch_uri"].tolist()
            included_idxs = g["patch_grid_idx"].to_numpy(np.int64)
            lowres_uri = lowres_map.get(sid)
            feature_uri = feat_map.get(sid)

            self.records.append(
                SlideRecord(
                    slide_id=sid,
                    label=labels[sid],
                    coords=coords,
                    lmdb_pth=lmdb_pth,
                    lmdb_keys=lmdb_keys,
                    lowres_uri=lowres_uri,
                    feature_uri=feature_uri,
                    included_idxs=included_idxs,
                )
            )

        # LMDB env cache
        self._env_cache = {}
        self._lowres_cache = {}

    # --------------------------------------------------------
    # LMDB env caching
    # --------------------------------------------------------
    def _get_env(self, path):
        if path not in self._env_cache:
            self._env_cache[path] = lmdb.open(
                path, readonly=True, lock=False, readahead=False, max_readers=512
            )
        return self._env_cache[path]

    # --------------------------------------------------------
    def __len__(self):
        return len(self.records)

    # --------------------------------------------------------
    def _load_lowres(self, rec):
        if rec.slide_id in self._lowres_cache:
            return self._lowres_cache[rec.slide_id]

        src = rec.lowres_uri
        if src is None:
            # fallback: first tile from LMDB
            env = self._get_env(rec.lmdb_pth)
            with env.begin(write=False) as txn:
                raw = txn.get(rec.lmdb_keys[0].encode())
            img = Image.open(io.BytesIO(raw))
        else:
            img = Image.open(src)

        img = img.convert(self.image_mode)
        img = img.resize(self.lowres_size[::-1], Image.BILINEAR)
        t = self.lowres_transform(img) if self.lowres_transform else TF.to_tensor(img)

        self._lowres_cache[rec.slide_id] = t
        return t

    # --------------------------------------------------------
    def __getitem__(self, idx):
        rec = self.records[idx]
        lowres = self._load_lowres(rec)
        dtype = torch.float if self.task_type == "regression" else torch.long

        view = SlideFeatureView(
            lmdb_path=rec.lmdb_pth,     # <- pass path, not env
            lmdb_keys=rec.lmdb_keys,
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