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
    tfm_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    lowres_tfm_train = transforms.Compose([
        transforms.ToTensor(),
        tfm_norm
    ])
    patch_tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        tfm_norm,
        transforms.RandomApply([transforms.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.5)
    ])

    lowres_tfm_val = transforms.Compose([transforms.ToTensor(), tfm_norm])
    patch_tfm_val = transforms.Compose([transforms.ToTensor(), tfm_norm])

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

    logger = TensorBoardLogger("../experiments", name=run_name) if use_logger else False

    checkpointer = module_cfg.get("checkpointer", None)
    checkpoint_cb = (
        ModelCheckpoint(
            dirpath=f"../experiments/{run_name}/checkpoints/",
            **checkpointer.get("params", {}),
        )
        if checkpointer is not None
        else None
    )

    callbacks = [checkpoint_cb] if checkpoint_cb else []

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_cfg,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(lit_model, train_loader, val_loader)