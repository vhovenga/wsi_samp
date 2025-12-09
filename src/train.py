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

import torch.nn as nn
import kornia.augmentation as K
import kornia.enhance as KE

from models.lit_module import LitMIL
from data.datasets import SlideDataset, slide_collate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIL model with separate configs.")
    parser.add_argument("module_config", type=str, help="Path to model/training config.")
    parser.add_argument("data_config", type=str, help="Path to dataset config.")
    parser.add_argument("trainer_config", type=str, help="Path to trainer config.")
    args = parser.parse_args()

    # -----------------------------
    # Load configs
    # -----------------------------
    with open(args.module_config) as f:
        module_cfg = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.trainer_config) as f:
        trainer_cfg = yaml.safe_load(f)

    # -----------------------------
    # Transforms
    # -----------------------------

    # -------------------------
    # TorchVision normalization
    # -------------------------
    tfm_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # -------------------------
    # LOW-RES PIPELINE (CPU)
    # -------------------------
    lowres_tfm_train = transforms.Compose([
        transforms.ToTensor(),
        tfm_norm,
    ])

    lowres_tfm_val = transforms.Compose([
        transforms.ToTensor(),
        tfm_norm,
    ])

    # -------------------------
    # PATCH PIPELINE (KORNIA GPU)
    # -------------------------


    patch_tfm_train = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90.0, p=1.0),
        K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
        KE.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
        K.RandomElasticTransform(
            kernel_size=(63, 63),   # needed for sigma=5
            sigma=torch.tensor([5.0, 5.0]),
            p=0.5,
        ),
    )

    # -------------------------
    # VALIDATION PATCH PIPELINE
    # -------------------------
    # Val uses no augmentation; just PIL->tensor then Kornia normalize
    patch_tfm_val = nn.Sequential(
        KE.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    )

    # -----------------------------
    # Datasets / Dataloaders
    # -----------------------------
    train_ds = SlideDataset(
        **data_cfg,
        split="train",
        lowres_transform=lowres_tfm_train,
        patch_transform=patch_tfm_train,
    )
    val_ds = SlideDataset(
        **data_cfg,
        split="val",
        lowres_transform=lowres_tfm_val,
        patch_transform=patch_tfm_val,
    )

    dataloader_cfg = module_cfg.get("dataloader", {})
    train_loader = DataLoader(
        train_ds,
        **dataloader_cfg,
        shuffle=True,
        pin_memory=True,
        collate_fn=slide_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        **dataloader_cfg,
        shuffle=False,
        pin_memory=True,
        collate_fn=slide_collate,
    )

    # -----------------------------
    # Lightning Module
    # -----------------------------
    lit_model = LitMIL(module_cfg)

    ckpt_path = module_cfg.get("checkpoint", None)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)["state_dict"]
        missing, unexpected = lit_model.load_state_dict(checkpoint, strict=False)

        if missing or unexpected:
            print("⚠️ Warning: checkpoint mismatch detected.")
            if missing:
                print("Missing keys:", missing)
            if unexpected:
                print("Unexpected keys:", unexpected)
        else:
            print(f"✅ Successfully loaded weights from checkpoint: {ckpt_path}")

    # -----------------------------
    # Trainer / Logger / Checkpoints
    # -----------------------------
    use_logger = trainer_cfg.pop("logger", True)
    run_name = module_cfg.get("run_name", "mil_run")

    # -----------------------------
    # Logger (Lightning will create version_i)
    # -----------------------------
    logger = TensorBoardLogger("../experiments", name=run_name) if use_logger else None

    # ../experiments/<run_name>/version_i/
    log_dir = logger.log_dir if logger is not None else f"../experiments/{run_name}"

    # -----------------------------
    # Checkpoint callback inside version_i folder
    # -----------------------------
    if trainer_cfg.get("enable_checkpointing", True):
        checkpointer_cfg = module_cfg.get("checkpointer", None)

        if checkpointer_cfg is not None:
            checkpoint_cb = ModelCheckpoint(
                dirpath=f"{log_dir}/checkpoints",
                **checkpointer_cfg.get("params", {}),
            )
        else:
            checkpoint_cb = None
    else:
        checkpoint_cb = None

    callbacks = [checkpoint_cb] if checkpoint_cb is not None else []
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_cfg,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(lit_model, train_loader, val_loader)