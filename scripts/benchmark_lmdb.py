import time
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("..")
from src.data.datasets import SlideDataset, slide_collate
from src.data.dataset_lmdb import SlideDatasetLMDB


def time_batch_fetch(loader, n_batches=50, tiles_per_slide=1024):
    """
    For each batch:
        for each view in batch["views"]:
            fetch_tiles(random subset)
    Measures end-to-end batch latency.
    """
    it = iter(loader)
    total = 0.0

    for _ in tqdm(range(n_batches), desc="Benchmarking"):
        batch = next(it)

        views = batch["views"]
        t0 = time.time()

        for view in views:
            N = len(view)
            k = min(tiles_per_slide, N)
            idxs = random.sample(range(N), k)
            tiles = view.fetch_tiles(idxs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        total += (t1 - t0)

    return total / n_batches


if __name__ == "__main__":
    # identical transforms
    patch_tfm = None
    lowres_tfm = None

    # loader config
    dl_cfg = dict(
        batch_size=4,               # MIL batch (your usual)
        num_workers=16,
        pin_memory=True,
        collate_fn=slide_collate,
        shuffle=False,
        persistent_workers=True,
    )

    # ---------- directory dataset ----------
    # ds_dir = SlideDataset(
    #     split="train",
    #     tiles_parquet="../data/tcga_luad_lusc/parquet/tiles.parquet",
    #     labels_parquet="../data/tcga_luad_lusc/parquet/slide_labels.parquet",
    #     splits_parquet="../data/tcga_luad_lusc/parquet/split_1.parquet",
    #     features_parquet="../data/tcga_luad_lusc/parquet/features.parquet",
    #     patch_transform=patch_tfm,
    # )
    # dl_dir = DataLoader(ds_dir, **dl_cfg)

    # ---------- LMDB dataset ----------
    ds_lmdb = SlideDatasetLMDB(
        split="train",
        tiles_parquet="/home/dog/Documents/van/wsi_samp/data/tcga_luad_lusc/lmdb_version/new_tiles.parquet",
        labels_parquet="../data/tcga_luad_lusc/parquet/slide_labels.parquet",
        splits_parquet="../data/tcga_luad_lusc/parquet/split_1.parquet",
        features_parquet="../data/tcga_luad_lusc/parquet/features.parquet",
        patch_transform=patch_tfm,
    )
    print(len(ds_lmdb))
    dl_lmdb = DataLoader(ds_lmdb, **dl_cfg)

    # warm up
    # _ = next(iter(dl_dir))
    _ = next(iter(dl_lmdb))

    # benchmark
    # t_dir = time_batch_fetch(dl_dir, n_batches=50, tiles_per_slide=1024)
    t_lmdb = time_batch_fetch(dl_lmdb, n_batches=50, tiles_per_slide=1024)

    # print(f"\nDirectory Batch Time: {t_dir:.4f} sec")
    print(f"LMDB Batch Time:      {t_lmdb:.4f} sec")
    # print(f"Speedup:              {t_dir / t_lmdb:.2f}×")