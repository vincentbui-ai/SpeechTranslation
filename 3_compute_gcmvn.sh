#!/bin/bash
# 3_compute_gcmvn.sh
# Compute Global CMVN statistics (requires audio files)

set -e

REPO_ROOT=$(pwd)
MAX_SAMPLES=${1:-50000}  # Default 50k samples

echo "Computing GCMVN statistics..."
echo "  Max samples: $MAX_SAMPLES"
echo "  Audio files must be accessible at paths in data/vi-en/train.tsv"

python scripts/compute_gcmvn.py \
    --tsv-path data/vi-en/train.tsv \
    --output configs/vi-en/gcmvn.npz \
    --max-samples $MAX_SAMPLES

echo ""
echo "[✓] GCMVN computed: configs/vi-en/gcmvn.npz"
echo "Next step: Start training"
echo "  Run: ./4_train_scratch.sh"
