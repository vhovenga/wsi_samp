# feature_extract.py
import yaml
import torch
import argparse
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import kornia.enhance as KE
from torch.utils.data import Subset

from models.lit_module import LitMIL
from data.datasets import PatchDataset, nvjpeg_worker_init, make_patch_collate as patch_collate


def predict_with_features(trainer, model, dataloader, feature_out_dir, save_as="pt"):
    # Set attributes that predict_step will read
    model.feature_out_dir = feature_out_dir
    model.save_as = save_as
    trainer.predict(model, dataloaders=dataloader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed feature extraction using trained MIL model.")
    parser.add_argument("module_config", type=str, help="Path to module config file.")
    parser.add_argument("data_config", type=str, help="Path to dataset config file.")
    parser.add_argument("trainer_config", type=str, help="Path to trainer config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.ckpt).")
    parser.add_argument("--feature_out_dir", type=str, required=True, help="Output directory for features.")
    parser.add_argument("--save_as", type=str, default="pt", choices=["pt", "h5"])
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
    # Transforms (normalized ResNet-style)
    # -----------------------------
    patch_tfm = nn.Sequential(
        KE.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    )

    lowres_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # Dataset / Dataloader
    # -----------------------------
    dataset = PatchDataset(
        **data_cfg,
        split=["train", "val"],  # or "test" as needed
        lowres_transform=lowres_tfm,
        patch_transform=patch_tfm,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=5000,
        pin_memory=False,
        collate_fn=patch_collate(patch_tfm)
        )

    # -----------------------------
    # Load model from checkpoint
    # -----------------------------
    model = LitMIL.load_from_checkpoint(args.ckpt, cfg=module_cfg)

    # -----------------------------
    # Trainer setup
    # -----------------------------
    trainer = pl.Trainer(
        accelerator=trainer_cfg.get("accelerator"),
        devices=trainer_cfg.get("devices"),
        strategy=trainer_cfg.get("strategy", None),
        logger=False,
        enable_checkpointing=False,
        use_distributed_sampler=True
        )

    # -----------------------------
    # Run prediction (feature extraction)
    # -----------------------------
    Path(args.feature_out_dir).mkdir(parents=True, exist_ok=True)
    predict_with_features(trainer, model, dataloader, args.feature_out_dir, args.save_as)