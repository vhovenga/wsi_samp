import time
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import cProfile
import pstats

sys.path.append("..")
from src.data.datasets import SlideDataset, slide_collate
from src.data.dataset_lmdb import SlideDatasetLMDB


def time_batch_fetch(loader, n_batches=50, tiles_per_slide=1024):
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.time()
        total += (t1 - t0)

    return total / n_batches


def profile_dir(loader):
    profiler = cProfile.Profile()
    profiler.enable()

    t = time_batch_fetch(loader, n_batches=20, tiles_per_slide=1024)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats("dir_benchmark_1.prof")
    print(f"\nDirectory Mean Batch Time = {t:.4f} sec")
    print("Profile written to dir_benchmark.prof\n")
    print("Run:   snakeviz dir_benchmark.prof")
    print("Or:    python -m pstats dir_benchmark.prof")


if __name__ == "__main__":
    patch_tfm = None
    lowres_tfm = None

    dl_cfg = dict(
        batch_size=4,
        num_workers=16,
        pin_memory=True,
        collate_fn=slide_collate,
        shuffle=False,
        persistent_workers=True,
    )

    # --- regular directory dataset
    ds_dir = SlideDataset(
        split="train",
        tiles_parquet="../data/tcga_luad_lusc/parquet/tiles.parquet",
        labels_parquet="../data/tcga_luad_lusc/parquet/slide_labels.parquet",
        splits_parquet="../data/tcga_luad_lusc/parquet/split_1.parquet",
        features_parquet="../data/tcga_luad_lusc/parquet/features.parquet",
        patch_transform=patch_tfm,
    )

    dl_dir = DataLoader(ds_dir, **dl_cfg)

    # warm-up
    _ = next(iter(dl_dir))

    # --- run profiler ---
    profile_dir(dl_dir)
