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
    train_ds = SlideDataset(
        **cfg["dataset"],
        split="train",
        lowres_transform=lowres_tfm,
        patch_transform=patch_tfm,
    )

    val_ds = SlideDataset(
        **cfg["dataset"],
        split="val",
        lowres_transform=lowres_tfm,
        patch_transform=patch_tfm,
    )


    # Loaders: fixed-K train can use default; use padded collate for variable-K val
    train_loader = DataLoader(
        train_ds,
        **cfg["dataloader"],
        shuffle=True,
        pin_memory=True,
        collate_fn=slide_collate
    )

    val_loader = DataLoader(
        val_ds,
        **cfg["dataloader"],
        shuffle=False,
        pin_memory=True,
        collate_fn=slide_collate
    )

    lit_model = LitMIL(cfg)

    trainer_cfg = dict(cfg["trainer"]) 
    use_logger = trainer_cfg.pop("logger", True)  
    if use_logger:
        logger = TensorBoardLogger(
            "../experiments",
            name=cfg.get("run_name", "mil_run"),
        )
    else:
        logger = False

    trainer = pl.Trainer(
        logger=logger,
        **trainer_cfg
    )

    trainer.fit(lit_model, train_loader, val_loader)