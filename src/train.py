# src/train.py
import yaml
import torch
import argparse
from torch.utils.data import DataLoader, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms

from data.data_utils import bag_collate_padded
from models.lit_module import LitMIL
from data.datasets import SlideDataset, slide_collate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIL model from config.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file.",
    )

    with open(parser.parse_args().config) as f:
        cfg = yaml.safe_load(f)


    # Example transforms
    lowres_tfm = transforms.Compose([
        transforms.Resize((256, 256)),  # thumbnails to fixed size
        transforms.ToTensor(),
    ])
    patch_tfm = transforms.ToTensor()  # applied when fetching full patches

    # Build a reference dataset just to get the slide count for splitting
    ref_ds = SlideDataset(
        root=cfg["data"]["root"],
        split=cfg["data"].get("split", "train"),
        lowres_transform=lowres_tfm,
        patch_transform=patch_tfm,
    )

    # 80/20 slide-level split (reproducible)
    n_total = len(ref_ds)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Apply the same index split to each base dataset
    train_ds = Subset(ref_ds, train_idx)
    val_ds   = Subset(ref_ds,   val_idx)

    # Loaders: fixed-K train can use default; use padded collate for variable-K val
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=slide_collate
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=slide_collate
    )

    lit_model = LitMIL(cfg)
    logger = TensorBoardLogger("../experiments", name=cfg.get("run_name", "mil_run"))
    trainer = pl.Trainer(logger=logger, **cfg["trainer"])
    trainer.fit(lit_model, train_loader, val_loader)