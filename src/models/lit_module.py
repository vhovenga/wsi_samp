from typing import Dict, Any
import pytorch_lightning as pl
import torch
from . import build_model
from losses import build_loss
from samplers import build_sampler
from metrics import build_metrics

class LitMIL(pl.LightningModule):
    def __init__(self, exp_cfg):
        super().__init__()
        self.save_hyperparameters(exp_cfg)  
        self.mil_module = build_model(exp_cfg["model"])

        loss_cfg = exp_cfg["loss"]
        self.mil_loss_fn = build_loss(loss_cfg["mil_loss"])
        self.sampler_loss_fn = build_loss(loss_cfg["sampler_loss"]) if "sampler_loss" in loss_cfg else None

        self.opt_cfg = exp_cfg["optimizer"]

        sampler_cfg = exp_cfg["samplers"]
        self.train_sampler = build_sampler(sampler_cfg["train"])
        self.val_sampler = build_sampler(sampler_cfg["val"])
        self.test_sampler = build_sampler(sampler_cfg["test"])

        metrics_cfg = exp_cfg.get("metrics", {})
        self.train_metrics = build_metrics(metrics_cfg.get("train", {}), prefix="train/")
        self.val_metrics = build_metrics(metrics_cfg.get("val", {}), prefix="val/")
        self.test_metrics = build_metrics(metrics_cfg.get("test", {}), prefix="test/")

    # --- step 1: sample patches ---
    def sampler_step(self, batch: Dict[str, Any], stage: str):
        sampler = self.train_sampler if stage == "train" else self.val_sampler
        images_pad, mask_pad, sampler_aux = sampler.select_and_fetch(batch, self.device)
        return images_pad, mask_pad, sampler_aux

    # --- step 2: MIL forward ---
    def mil_step(self, images_pad: torch.Tensor, mask_pad: torch.Tensor):
        out = self.mil_module(images=images_pad, mask=mask_pad)
        return out  # dict with {"logits": [B,C], "Z": [B,D], "extras": ...}

    # --- step 3: loss ---
    def loss_step(self, batch: Dict[str, Any], stage: str, mil_out: Dict[str, Any], sampler_aux: Dict[str, Any]):
        y = batch["labels"].to(self.device, non_blocking=True)
        task_loss = self.mil_loss_fn(mil_out["logits"], y)

        sampler_loss = None
        if self.sampler_loss_fn is not None and stage == "train":
            sampler_loss = self.sampler_loss_fn(sampler_aux, batch, mil_out)

        total_loss = task_loss + (sampler_loss if sampler_loss is not None else 0.0)

        # logging
        self.log(f"{stage}_loss", task_loss,
                 prog_bar=True,
                 on_step=(stage == "train"),
                 on_epoch=True, batch_size=y.size(0))
        if sampler_loss is not None:
            self.log(f"{stage}_sampler_loss", sampler_loss, on_epoch=True)

        return total_loss
    
    # --- lightning hooks ---
    def training_step(self, batch, batch_idx):
        images_pad, mask_pad, sampler_aux = self.sampler_step(batch, "train")
        mil_out = self.mil_step(images_pad, mask_pad)
        loss = self.loss_step(batch, "train", mil_out, sampler_aux)

        # raw outputs always; collection handles transforms internally
        preds_raw = mil_out["logits"]
        y = batch["labels"]
        self.train_metrics.update(preds_raw, y)
        return loss

    def validation_step(self, batch, batch_idx):
        images_pad, mask_pad, sampler_aux = self.sampler_step(batch, "val")
        mil_out = self.mil_step(images_pad, mask_pad)
        loss = self.loss_step(batch, "val", mil_out, sampler_aux)

        preds_raw = mil_out["logits"]
        y = batch["labels"]
        self.val_metrics.update(preds_raw, y)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        # show on bar if you want
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()
    def configure_optimizers(self):
        # collect params from mil + sampler
        params = list(self.mil_module.parameters()) + list(self.train_sampler.parameters())

        # pull optimizer config
        opt_name = self.opt_cfg["name"]
        opt_params = self.opt_cfg.get("params", {})

        # look up the optimizer class in torch.optim
        opt_cls = getattr(torch.optim, opt_name)
        optimizer = opt_cls(params, **opt_params)

        return optimizer