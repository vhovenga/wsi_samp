# src/train.py
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.profiler import schedule, tensorboard_trace_handler
from torchvision import transforms

from models.lit_module import LitMIL
from data.datasets import SlideDataset, slide_collate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIL model from config.")
    parser.add_argument("config", type=str, help="Path to config file.")
    with open(parser.parse_args().config) as f:
        cfg = yaml.safe_load(f)

    # Example transforms
    lowres_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    patch_tfm = transforms.ToTensor()

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
    
    run_name = cfg.get("run_name", "mil_run")

    if use_logger:
        logger = TensorBoardLogger(
            "../experiments",
            name=run_name,
        )
    else:
        logger = False

    checkpoint = cfg.get("checkpoint", None)
    if checkpoint is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"../experiments/{run_name}/checkpoints/",
            **checkpoint.get("params", {})    
        )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **trainer_cfg
    )

    trainer.fit(lit_model, train_loader, val_loader)
