import time
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil
import os
import cProfile, pstats
import sys 
sys.path.append("..")
import fetcher_cpp
from src.data.datasets import SlideDataset, slide_collate


def time_batch_fetch(loader, n_batches=30, tiles_per_slide=1024):
    it = iter(loader)

    total_batch = 0.0
    total_read = 0.0
    total_info = 0.0
    total_decode = 0.0

    for _ in tqdm(range(n_batches), desc="Benchmark"):
        batch = next(it)
        views = batch["views"]

        t0 = time.time()

        batch_read = 0.0
        batch_info = 0.0
        batch_decode = 0.0

        for view in views:
            N = len(view)
            k = min(N, tiles_per_slide)
            idxs = random.sample(range(N), k)

            fetcher_cpp.reset_all()
            _ = view.fetch_tiles(idxs)

            batch_read   += fetcher_cpp.get_read_time()
            batch_info   += fetcher_cpp.get_info_time()
            batch_decode += fetcher_cpp.get_decode_time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.time()

        total_batch  += (t1 - t0)
        total_read   += batch_read
        total_info   += batch_info
        total_decode += batch_decode

    return (
        total_batch / n_batches,
        total_read  / n_batches,
        total_info  / n_batches,
        total_decode / n_batches,
    )


def profile_dir(loader):
    profiler = cProfile.Profile()
    profiler.enable()

    proc = psutil.Process(os.getpid())
    io_before = proc.io_counters()

    mean_batch, mean_read, mean_info, mean_decode = time_batch_fetch(
        loader, n_batches=20, tiles_per_slide=1024
    )

    io_after = proc.io_counters()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats("decode_profile.prof")

    read_mb = (io_after.read_bytes - io_before.read_bytes) / 1e6
    write_mb = (io_after.write_bytes - io_before.write_bytes) / 1e6

    print("\n=== RESULTS ===")
    print(f"Batch Time     : {mean_batch:.4f} s")
    print(f"File Read Time : {mean_read:.4f} s")
    print(f"Info Parse     : {mean_info:.4f} s")
    print(f"Decode Time    : {mean_decode:.4f} s")
    print(f"Disk Read      : {read_mb:.2f} MB")
    print(f"Disk Write     : {write_mb:.2f} MB")
    print("Profile saved to decode_profile.prof")


if __name__ == "__main__":
    ds = SlideDataset(
        split="train",
        tiles_parquet="../data/tcga_luad_lusc/parquet/tiles.parquet",
        labels_parquet="../data/tcga_luad_lusc/parquet/slide_labels.parquet",
        splits_parquet="../data/tcga_luad_lusc/parquet/split_1.parquet",
        features_parquet="../data/tcga_luad_lusc/parquet/features.parquet",
        patch_transform=None,
    )

    dl = DataLoader(
        ds,
        batch_size=4,
        num_workers=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=slide_collate,
        persistent_workers=True,
    )

    _ = next(iter(dl))
    profile_dir(dl)
