from typing import Dict, Any, List
import pytorch_lightning as pl
import torch
from . import build_model
from losses import build_loss
from samplers import build_sampler
from metrics import build_metrics
from utils.bag_modifiers import build_bag_modifier

class LitMIL(pl.LightningModule):
    def __init__(self, exp_cfg):
        super().__init__()
        self.save_hyperparameters(exp_cfg)

        # --- mode selection ---
        self.mode = exp_cfg.get("mode", "end_to_end")  # or "frozen_feature"

        # --- model config assertions ---
        model_cfg = exp_cfg["model"]

        if self.mode == "end_to_end":
            assert "feature_extractor" in model_cfg, (
                "In end_to_end mode, exp_cfg['model'] must include a 'feature_extractor' entry."
            )
        elif self.mode == "frozen_feature":
            assert "feature_extractor" not in model_cfg, (
                "In frozen_feature mode, exp_cfg['model'] should NOT include a 'feature_extractor'."
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # --- model ---
        self.mil_module = build_model(model_cfg)

        # --- loss ---
        loss_cfg = exp_cfg["loss"]
        self.mil_loss_fn = build_loss(loss_cfg["mil_loss"])
        self.sampler_loss_fn = (
            build_loss(loss_cfg["sampler_loss"]) if "sampler_loss" in loss_cfg else None
        )

        # --- optimizer config ---
        self.opt_cfg = exp_cfg["optimizer"]

        # --- samplers ---
        sampler_cfg = exp_cfg.get("samplers", {})

        assert self.mode in ["end_to_end", "frozen_feature"], f"Unknown mode: {self.mode}"

        if self.mode == "end_to_end":
            assert all(
                k in sampler_cfg for k in ["train", "val", "test"]
            ), "In end_to_end mode, you must specify train, val, and test samplers in exp_cfg['samplers']."
            self.train_sampler = build_sampler(sampler_cfg["train"])
            self.val_sampler = build_sampler(sampler_cfg["val"])
            self.test_sampler = build_sampler(sampler_cfg["test"])

        else:  
            if any(k in sampler_cfg for k in ["train", "val", "test"]):
                import warnings
                warnings.warn(
                    "Samplers were provided in exp_cfg['samplers'], but mode='frozen_feature'. "
                    "These samplers will not be used."
                )
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

        # --- metrics ---
        metrics_cfg = exp_cfg.get("metrics", {})
        self.train_metrics = build_metrics(metrics_cfg.get("train", {}), prefix="train/")
        self.val_metrics = build_metrics(metrics_cfg.get("val", {}), prefix="val/")
        self.test_metrics = build_metrics(metrics_cfg.get("test", {}), prefix="test/")

        # --- optional bag modifier ---
        bag_modifier_cfg = exp_cfg.get("bag_modifier", None)
        self.bag_modifier = (
            build_bag_modifier(bag_modifier_cfg) if bag_modifier_cfg is not None else None
        )

    def on_fit_start(self):
        self.train_metrics = self.train_metrics.to(self.device)
        self.val_metrics = self.val_metrics.to(self.device)
        self.test_metrics = self.test_metrics.to(self.device)

    # --- sampler + mil forward ---
    def end_to_end_forward(self, batch: Dict[str, Any], stage: str):
        images_pad, mask_pad, sampler_aux = self.sampler_step(batch, stage)
        mil_out = self.mil_step(images_pad, mask_pad, bag_ids=batch["slide_ids"])
        return mil_out, sampler_aux

    # --- frozen feature + milforward ---
    def frozen_feature_forward(self, batch: Dict[str, Any], stage: str):
        # expects batch["features"] of shape [B, N, D] and batch["mask"]
        views = batch["views"]
        feats_list = []
        for view in views:
            assert view.has_features, "mode=\"frozen_feature\" requires pre-computed features"
            feats = view.fetch_features()  # [N, D]
            feats_list.append(feats)

        mask = batch["coord_mask"].to(self.device, non_blocking=True)
        agg_out = self.mil_module.aggregate(feats_list, mask=mask)
        Z = agg_out["Z"]
        logits = self.mil_module.predictor(Z)
        mil_out = {"logits": logits, "Z": Z, "extras": agg_out.get("extras", {})}
        sampler_aux = {}  # no sampler in frozen mode
        return mil_out, sampler_aux

    # --- sampler step ---
    def sampler_step(self, batch: Dict[str, Any], stage: str):
        sampler = self.train_sampler if stage == "train" else self.val_sampler
        images_pad, mask_pad, sampler_aux = sampler.select_and_fetch(batch, self.device)
        return images_pad, mask_pad, sampler_aux

    # --- mil forward ---
    def mil_step(self, images_pad: torch.Tensor, mask_pad: torch.Tensor, bag_ids: List[int]):
        bag_feats = self.mil_module.feature_extract(images=images_pad, mask=mask_pad)
        agg_out = self.mil_module.aggregate(bag_feats, mask=mask_pad)
        Z = agg_out["Z"]

        if self.trainer.training and self.bag_modifier is not None:
            Z = self.bag_modifier.compute(bag_ids, Z)

        logits = self.mil_module.predictor(Z)
        return {"logits": logits, "Z": Z, "extras": agg_out["extras"]}

    # --- loss ---
    def loss_step(self, batch: Dict[str, Any], stage: str, mil_out: Dict[str, Any], sampler_aux: Dict[str, Any]):
        logits = mil_out["logits"]
        targets = batch["labels"]
        
        logits = logits.reshape(-1)
        targets = targets.float().reshape(-1)
        
        task_loss = self.mil_loss_fn(mil_out["logits"], targets)
        sampler_loss = None
        if self.sampler_loss_fn is not None and stage == "train" and self.mode != "frozen_feature":
            sampler_loss = self.sampler_loss_fn(sampler_aux, batch, mil_out)
        total_loss = task_loss + (sampler_loss if sampler_loss is not None else 0.0)

        self.log(f"{stage}_loss", task_loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True, batch_size=targets.size(0),
                 sync_dist=True)
        if sampler_loss is not None:
            self.log(f"{stage}_sampler_loss", sampler_loss, on_epoch=True)
        return total_loss

    # --- training / validation steps ---
    def training_step(self, batch, batch_idx):
        if self.mode == "frozen_feature":
            mil_out, sampler_aux = self.frozen_feature_forward(batch, "train")
        else:
            mil_out, sampler_aux = self.end_to_end_forward(batch, "train")

        loss = self.loss_step(batch, "train", mil_out, sampler_aux)
        logits = mil_out["logits"]
        targets = batch["labels"]
        
        logits = logits.reshape(-1)
        targets = targets.float().reshape(-1)
        self.train_metrics.update(logits, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "frozen_feature":
            mil_out, sampler_aux = self.frozen_feature_forward(batch, "val")
        else:
            mil_out, sampler_aux = self.end_to_end_forward(batch, "val")

        loss = self.loss_step(batch, "val", mil_out, sampler_aux)

        logits = mil_out["logits"]
        targets = batch["labels"]
        
        logits = logits.reshape(-1)
        targets = targets.float().reshape(-1)
        self.val_metrics.update(logits, targets)
        return loss

    # --- epoch end hooks ---
    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    # --- optimizer config ---
    def configure_optimizers(self):
        if self.mode == "frozen_feature":
            params = list(self.mil_module.aggregator.parameters()) + list(self.mil_module.predictor.parameters())
        else:
            params = list(self.mil_module.parameters()) + list(self.train_sampler.parameters())

        opt_name = self.opt_cfg["name"]
        opt_params = self.opt_cfg.get("params", {})
        opt_cls = getattr(torch.optim, opt_name)
        optimizer = opt_cls(params, **opt_params)
        return optimizer
