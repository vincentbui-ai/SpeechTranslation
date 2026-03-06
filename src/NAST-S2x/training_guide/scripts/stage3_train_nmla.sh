#!/bin/bash
# stage3_train_nmla.sh
# Stage 3: NMLA Training (Non-Monotonic Latent Alignment) for NAST-S2X
# Fine-tune with n-gram glancing and alignment

set -e

# Configuration - UPDATE THESE PATHS
CHUNK_SIZE=32  # 32 = 320ms chunks for streaming
DATA_ROOT="/path/to/data/nast-vi-en/fairseq"  # Update this
SAVE_DIR="checkpoints/nast-vi-en/stage3_nmla"
NAST_DIR="src/NAST-S2x/nast"

# IMPORTANT: Update this with the path from Stage 2
CTC_PT="checkpoints/nast-vi-en/stage2_ctc/checkpoint_best.pt"

# GPU/Training configuration
MAX_TOKENS=6400
UPDATE_FREQ=12
NUM_GPUS=4

# Create save directory
mkdir -p ${SAVE_DIR}

echo "=========================================="
echo "Stage 3: NMLA Training"
echo "=========================================="
echo "Data: ${DATA_ROOT}"
echo "Save: ${SAVE_DIR}"
echo "CTC Checkpoint: ${CTC_PT}"
echo "Chunk: ${CHUNK_SIZE} (320ms)"
echo "GPUs: ${NUM_GPUS}"
echo ""

# Verify CTC checkpoint exists
if [ ! -f "${CTC_PT}" ]; then
    echo "ERROR: CTC checkpoint not found: ${CTC_PT}"
    echo "Please complete Stage 2 training first."
    exit 1
fi

# Run training
fairseq-train ${DATA_ROOT} \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --user-dir ${NAST_DIR} \
    --fp16 \
    --finetune-from-model ${CTC_PT} \
    --task nat_speech_to_unit_ctc_modified \
    --arch nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment \
    --src-embedding-copy \
    --target-is-code \
    --target-code-size 1000 \
    --src-upsample-ratio 1 \
    --hidden-upsample-ratio 6 \
    --main-context ${CHUNK_SIZE} \
    --right-context ${CHUNK_SIZE} \
    --unit-size 2 \
    --share-decoder-input-output-embed \
    --rand-pos-encoder 300 \
    --decoder-learned-pos \
    --activation-dropout 0.1 \
    --attention-dropout 0.1 \
    --encoder-max-relative-position 32 \
    --apply-bert-init \
    --noise full_mask \
    --criterion nat_loss_ngram_glat_s2u \
    --glat-p 0.3:0.3@50k \
    --glat-p-unit 0.1:0.1@50k \
    --use-ngram \
    --use-ngram-unit \
    --ngram-size 2 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --clip-norm 10.0 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 0.0003 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' \
    --warmup-updates 4000 \
    --stop-min-lr '1e-09' \
    --max-update 50000 \
    --max-tokens ${MAX_TOKENS} \
    --update-freq ${UPDATE_FREQ} \
    --grouped-shuffling \
    --save-dir ${SAVE_DIR} \
    --ddp-backend=legacy_ddp \
    --no-progress-bar \
    --log-format json \
    --log-interval 100 \
    --save-interval-updates 2000 \
    --keep-interval-updates 10 \
    --save-interval 1000 \
    --keep-last-epochs 10 \
    --fixed-validation-seed 7 \
    --skip-invalid-size-inputs-valid-test \
    --validate-interval 1000 \
    --validate-interval-updates 2000 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu_unit \
    --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 5 \
    --num-workers 8

echo ""
echo "=========================================="
echo "Stage 3 Complete!"
echo "=========================================="
echo "Checkpoint: ${SAVE_DIR}/checkpoint_best.pt"
echo ""
echo "Training Pipeline Complete!"
echo ""
echo "Next steps:"
echo "  1. Test inference: bash test_inference.sh"
echo "  2. Export model for deployment"
echo "  3. Train different chunk sizes (128, 256)"
