#!/bin/bash

SCRIPT_PATH="/home/alpaca/Documents/van/wsi_sampling/src/train.py"
CONFIG_DIR="/home/alpaca/Documents/van/wsi_sampling/configs/toy_data_prototypes"

for cfg in "$CONFIG_DIR"/*.yml; do
    echo
    echo "============================================================"
    echo "🚀 STARTING TRAINING FOR CONFIG: $(basename "$cfg")"
    echo "============================================================"
    echo

    python "$SCRIPT_PATH" "$cfg"

    echo
    echo "============================================================"
    echo "✅ FINISHED CONFIG: $(basename "$cfg")"
    echo "============================================================"
    echo
done