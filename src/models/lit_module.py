from collections import defaultdict
from typing import Dict, Any, List
import os 
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd 
import h5py 
from tqdm import tqdm 
from pathlib import Path 
import torch.distributed as dist
from tqdm import tqdm

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

        self._predict_index_buffer = []

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
    

    def on_predict_start(self):
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
        else:
            self._rank = 0

        self._base = os.path.join(self.feature_out_dir, f"rank_{self._rank}")
        os.makedirs(self._base, exist_ok=True)

        self._slide_buffers = defaultdict(list)
        self._slide_block_id = defaultdict(int)

        self._flush_every = 1000000

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # feature extraction
        f_mb = self.mil_module.feature_extractor(batch["patches"])
        if f_mb.ndim == 4:
            f_mb = f_mb.mean(dim=[2, 3])  # [B, D]

        slide_ids     = batch["slide_ids"]                      # list of len B
        patch_idxs    = batch["patch_indices"].cpu().tolist()   # [B]
        coords        = batch["coords"].cpu().numpy()           # [B, 2]
        feats_cpu     = f_mb.detach().cpu()                     # [B, D]

        # accumulate into slide-level buffers
        for i in range(len(slide_ids)):
            sid    = slide_ids[i]
            pidx   = patch_idxs[i]
            xy     = coords[i]
            feat   = feats_cpu[i]

            self._slide_buffers[sid].append({
                "patch_idx": pidx,
                "coord": xy,
                "feat": feat,
            })

        # flush if exceeding threshold
        total_buffered = sum(len(v) for v in self._slide_buffers.values())
        if total_buffered >= self._flush_every:
            self._flush_slide_buffers()

    def _flush_slide_buffers(self):
        for sid, records in self._slide_buffers.items():
            slide_dir = os.path.join(self._base, sid)
            os.makedirs(slide_dir, exist_ok=True)

            block_id = self._slide_block_id[sid]
            out_path = os.path.join(slide_dir, f"block_{block_id:05d}.pt")

            # Write block. No read. No merge. No atomic rename needed.
            torch.save(records, out_path)

            self._slide_block_id[sid] += 1

        self._slide_buffers.clear()

    def on_predict_end(self):
        from multiprocessing import Pool, cpu_count

        if self._slide_buffers:
            self._flush_slide_buffers()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if self._rank != 0:
            return

        base = self.feature_out_dir
        rank_dirs = [
            os.path.join(base, d)
            for d in os.listdir(base)
            if d.startswith("rank_")
        ]

        slides = set()
        for rdir in rank_dirs:
            for sid in os.listdir(rdir):
                if os.path.isdir(os.path.join(rdir, sid)):
                    slides.add(sid)
        slides = sorted(slides)

        out_root = base
        tasks = [(sid, rank_dirs, out_root) for sid in slides]

        workers = min(32, cpu_count())

        with Pool(processes=workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(_merge_worker, tasks),
                total=len(tasks),
                ncols=80,
                desc="Merging slides"
            ):
                pass

        for r in rank_dirs:
            os.rmdir(r)

def _merge_worker(args):
    sid, rank_dirs, out_root = args
    merged = []

    for rdir in rank_dirs:
        sdir = os.path.join(rdir, sid)
        if not os.path.isdir(sdir):
            continue

        blocks = sorted(
            b for b in os.listdir(sdir)
            if b.endswith(".pt")
        )

        for blk in blocks:
            path = os.path.join(sdir, blk)
            recs = torch.load(path, map_location="cpu", weights_only=False)
            merged.extend(recs)
            os.remove(path)

        os.rmdir(sdir)

    final_dir = os.path.join(out_root, sid)
    os.makedirs(final_dir, exist_ok=True)
    torch.save(merged, os.path.join(final_dir, "features.pt"))

    return sid
