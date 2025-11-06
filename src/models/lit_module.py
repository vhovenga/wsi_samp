from typing import Dict, Any, List
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd 
import h5py 
from tqdm import tqdm 
from pathlib import Path 

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
        self.mode = exp_cfg.get("mode", "end_to_end")  # "end_to_end", "frozen_feature", "end_to_end_hybrid"

        # --- model config assertions ---
        model_cfg = exp_cfg["model"]
        if self.mode in ["end_to_end", "end_to_end_hybrid"]:
            assert "feature_extractor" in model_cfg, (
                f"In {self.mode} mode, exp_cfg['model'] must include a 'feature_extractor' entry."
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
        if self.mode in ["end_to_end", "end_to_end_hybrid"]:
            assert all(k in sampler_cfg for k in ["train", "val", "test"]), (
                f"In {self.mode} mode, you must specify train, val, and test samplers in exp_cfg['samplers']."
            )
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

    # --- sampler step ---
    def sampler_step(self, batch: Dict[str, Any], stage: str):
        sampler = (
            self.train_sampler if stage == "train"
            else self.val_sampler if stage == "val"
            else self.test_sampler
        )
        images_pad, mask_pad, sampler_aux = sampler.select_and_fetch(batch, self.device)
        return images_pad, mask_pad, sampler_aux

    # --- end-to-end ---
    def end_to_end_forward(self, batch: Dict[str, Any], stage: str):
        images_pad, mask_pad, sampler_aux = self.sampler_step(batch, stage)
        mil_out = self.mil_step(images_pad, mask_pad, bag_ids=batch["slide_ids"])
        return mil_out, sampler_aux

    # --- frozen feature ---
    def frozen_feature_forward(self, batch: Dict[str, Any], stage: str):
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
        sampler_aux = {}
        return mil_out, sampler_aux

    # --- end-to-end hybrid (new) ---
    def end_to_end_hybrid_forward(self, batch: Dict[str, Any], stage: str):
        images_pad, mask_pad, sampler_aux = self.sampler_step(batch, stage)
        idx_list = sampler_aux.get("idx_list", None)
        assert idx_list is not None, "Sampler must provide idx_list for hybrid mode"

        # extract features for sampled tiles
        fresh_feats = self.mil_module.feature_extract(images=images_pad, mask=mask_pad)
        combined_feats_list = []

        for view, sampled_idxs, f_new in zip(batch["views"], idx_list, fresh_feats):
            assert view.has_features, "Hybrid mode requires pre-computed features"
            all_feats = view.fetch_features()  # [N, D]
            all_feats = all_feats.to(f_new.device)

            N = all_feats.shape[0]
            mask_unsampled = torch.ones(N, dtype=torch.bool)
            mask_unsampled[sampled_idxs] = False

            unsampled_feats = all_feats[mask_unsampled]
            combined_feats = torch.cat([f_new, unsampled_feats], dim=0)
            combined_feats_list.append(combined_feats)

        mask = batch["coord_mask"].to(self.device, non_blocking=True)
        agg_out = self.mil_module.aggregate(combined_feats_list, mask=mask)
        Z = agg_out["Z"]

        if self.trainer.training and self.bag_modifier is not None:
            Z = self.bag_modifier.compute(batch["slide_ids"], Z)

        logits = self.mil_module.predictor(Z)
        mil_out = {"logits": logits, "Z": Z, "extras": agg_out.get("extras", {})}
        return mil_out, sampler_aux

    # --- mil step ---
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
        if self.sampler_loss_fn is not None and stage == "train" and self.mode not in ["frozen_feature"]:
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
        elif self.mode == "end_to_end_hybrid":
            mil_out, sampler_aux = self.end_to_end_hybrid_forward(batch, "train")
        else:
            mil_out, sampler_aux = self.end_to_end_forward(batch, "train")

        loss = self.loss_step(batch, "train", mil_out, sampler_aux)
        logits = mil_out["logits"]
        targets = batch["labels"]
        self.train_metrics.update(logits.reshape(-1), targets.float().reshape(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "frozen_feature":
            mil_out, sampler_aux = self.frozen_feature_forward(batch, "val")
        elif self.mode == "end_to_end_hybrid":
            mil_out, sampler_aux = self.end_to_end_hybrid_forward(batch, "val")
        else:
            mil_out, sampler_aux = self.end_to_end_forward(batch, "val")

        loss = self.loss_step(batch, "val", mil_out, sampler_aux)
        logits = mil_out["logits"]
        targets = batch["labels"]
        self.val_metrics.update(logits.reshape(-1), targets.float().reshape(-1))
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
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Runs feature extraction for one slide (no sampling).
        Safe for multi-GPU DDP when each rank processes disjoint slides.
        Processes tiles in micro-batches to avoid OOM.
        Also writes a Parquet index: [slide_id, feature_uri].
        """

        import pandas as pd

        feature_out_dir = getattr(self, "feature_out_dir", None)
        save_as = getattr(self, "save_as", "pt")
        micro_k = getattr(self, "micro_k", 64)

        slide_id = batch["slide_ids"][0]
        view = batch["views"][0]
        patch_uris = view.patch_uris
        coords = batch["coords_pad"][0, : len(patch_uris)].cpu().numpy()

        # --- Load all tiles for this slide (CPU tensor) ---
        tiles = view.fetch_tiles()  # [N, C, H, W]
        N = tiles.shape[0]

        # --- Forward in micro-batches ---
        feats_chunks = []
        for s in range(0, N, micro_k):
            x_mb = tiles[s:s+micro_k].to(self.device, non_blocking=True)
            f_mb = self.mil_module.feature_extractor(x_mb)
            if f_mb.ndim == 4:
                f_mb = f_mb.mean(dim=[2, 3])  # global avg pool
            feats_chunks.append(f_mb.detach().cpu())
            del x_mb, f_mb
            torch.cuda.empty_cache()

        feats = torch.cat(feats_chunks, dim=0)  # [N, D]
        patch_grid_idx = np.arange(len(patch_uris))

        # --- Optionally save ---
        if feature_out_dir is not None:
            slide_dir = Path(feature_out_dir) / str(slide_id)
            slide_dir.mkdir(parents=True, exist_ok=True)

            if save_as == "pt":
                feature_path = slide_dir / "features.pt"
                torch.save(
                    {
                        "features": feats,
                        "coords": torch.from_numpy(coords),
                        "patch_grid_idx": torch.from_numpy(patch_grid_idx),
                    },
                    feature_path,
                )
            elif save_as == "h5":
                feature_path = slide_dir / "features.h5"
                with h5py.File(feature_path, "w") as f:
                    f.create_dataset("features", data=feats.numpy(), compression="gzip")
                    f.create_dataset("coords", data=coords)
                    f.create_dataset("patch_grid_idx", data=patch_grid_idx)
            else:
                raise ValueError("save_as must be 'pt' or 'h5'")

            # --- Write / append to Parquet index ---
            df = pd.DataFrame([{"slide_id": slide_id, "feature_uri": str(feature_path)}])
            parquet_path = Path(feature_out_dir) / "features.parquet"

            if parquet_path.exists():
                # append
                existing = pd.read_parquet(parquet_path)
                combined = pd.concat([existing, df], ignore_index=True)
                combined.to_parquet(parquet_path, index=False)
            else:
                df.to_parquet(parquet_path, index=False)

        return {
            "slide_id": slide_id,
            "features": feats,
            "coords": coords,
            "patch_grid_idx": patch_grid_idx,
        }