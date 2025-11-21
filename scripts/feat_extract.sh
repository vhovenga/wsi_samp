#!/bin/bash

SCRIPT_PATH="/home/dog/Documents/van/wsi_samp/src/feature_extract.py"
MODULE_CONFIG="/home/dog/Documents/van/wsi_samp/configs/tcga_lusc-luad/resnet_attn_mean_pool_uniform_25_hybrid.yml"
DATA_CONFIG="/home/dog/Documents/van/wsi_samp/configs/data_configs/lusc-luad_25_data_cfg.yml"
TRAINER_CONFIG="/home/dog/Documents/van/wsi_samp/configs/trainer_configs/full_dog_util.yml"
CHECKPOINT="/home/dog/Documents/van/wsi_samp/experiments/lusc-resnet_attn_mean_pool_uniform_25_hybrid/checkpoints/epoch=99-step=2300.ckpt"
FEATURE_OUT_DIR="/media/ssd1/van/lusc-luad_feats/amp_uniform_25_hybrid_final"
MICRO_K=2048

# ensure output directory exists
mkdir -p "$FEATURE_OUT_DIR"

python "$SCRIPT_PATH" \
  "$MODULE_CONFIG" \
  "$DATA_CONFIG" \
  "$TRAINER_CONFIG" \
  --ckpt "$CHECKPOINT" \
  --feature_out_dir "$FEATURE_OUT_DIR" \
  --micro_k $MICRO_K
