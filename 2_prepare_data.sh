#!/bin/bash
# 2_prepare_data.sh
# Prepare TSV manifests and SPM models

set -e

REPO_ROOT=$(pwd)
echo "Preparing data in: $REPO_ROOT"

# Step 2.1: JSONL -> TSV
echo "[1/3] Converting JSONL to TSV..."
python scripts/prepare_data.py \
    --datasets-dir datasets \
    --output-dir data

echo ""
echo "[✓] TSV files created:"
ls -lh data/vi-en/*.tsv 2>/dev/null || echo "  (check data/ directory)"
ls -lh data/en-vi/*.tsv 2>/dev/null || echo "  (check data/ directory)"

# Step 2.2: Train SentencePiece
echo ""
echo "[2/3] Training SentencePiece models..."
python scripts/train_spm.py \
    --data-dir data/vi-en \
    --configs-dir configs/vi-en

echo ""
echo "[✓] SPM models created:"
ls -lh configs/vi-en/src_unigram6000/ 2>/dev/null || echo "  (check configs/vi-en/)"
ls -lh configs/vi-en/tgt_unigram6000/ 2>/dev/null || echo "  (check configs/vi-en/)"

# Step 2.3: Update config paths
echo ""
echo "[3/3] Updating config paths..."
# Update config_gcmvn.yaml
sed -i "s|/PATH/TO/SERVER|$REPO_ROOT|g" configs/vi-en/config_gcmvn.yaml
sed -i "s|stats_npz_path:.*|stats_npz_path: $REPO_ROOT/configs/vi-en/gcmvn.npz|g" configs/vi-en/config_gcmvn.yaml

# Update config_mtl_asr_st_ctcst.yaml
sed -i "s|/PATH/TO/SERVER|$REPO_ROOT|g" configs/vi-en/config_mtl_asr_st_ctcst.yaml

echo "[✓] Configs updated with: $REPO_ROOT"

echo ""
echo "Data preparation complete!"
echo "Next step: Compute GCMVN statistics (requires audio access)"
echo "  Run: ./3_compute_gcmvn.sh"
