#!/bin/bash
# preprocess.sh
# Full preprocessing pipeline for NAST-S2X

set -e

# Default values
DATASETS_DIR="datasets"
OUTPUT_DIR="data/nast-vi-en"
VOCAB_SIZE=10000
NUM_WORKERS=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets-dir)
            DATASETS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "NAST-S2X Preprocessing Pipeline"
echo "=========================================="
echo "Input:  $DATASETS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Vocab:  $VOCAB_SIZE"
echo "Workers: $NUM_WORKERS"
echo ""

# Step 1: Convert JSONL to TSV
echo "[Step 1/4] Converting JSONL to TSV..."
python training_guide/scripts/prepare_nast_data.py \
    --datasets-dir "$DATASETS_DIR" \
    --output-dir "$OUTPUT_DIR"

# Step 2: Train SentencePiece
echo ""
echo "[Step 2/4] Training SentencePiece model..."
python training_guide/scripts/train_spm_nast.py \
    --data-dir "$OUTPUT_DIR" \
    --vocab-size "$VOCAB_SIZE" \
    --output-dir "$OUTPUT_DIR/spm"

# Step 3: Extract fbank features
echo ""
echo "[Step 3/4] Extracting fbank features..."
python training_guide/scripts/extract_fbank.py \
    --tsv-path "$OUTPUT_DIR/vi-en/train.tsv" \
    --output-dir "$OUTPUT_DIR/fbank" \
    --num-workers "$NUM_WORKERS"

# Step 4: Compute CMVN
echo ""
echo "[Step 4/4] Computing CMVN statistics..."
python training_guide/scripts/compute_cmvn.py \
    --tsv-path "$OUTPUT_DIR/vi-en/train.tsv" \
    --fbank-dir "$OUTPUT_DIR/fbank" \
    --output "$OUTPUT_DIR/gcmvn.npz" \
    --max-samples 50000

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo "Output location: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Extract acoustic units (see 02_DATA_PREP.md)"
echo "  2. Create config.yaml"
echo "  3. Start training (see 03_TRAINING.md)"
