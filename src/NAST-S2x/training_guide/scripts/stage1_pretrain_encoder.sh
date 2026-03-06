#!/bin/bash
# stage1_pretrain_encoder.sh
# Stage 1: Encoder Pretraining (ASR) for NAST-S2X
# Train encoder on Speech-to-Text task for better initialization

set -e

# Configuration - UPDATE THESE PATHS
CHUNK_SIZE=32  # 32 = 320ms chunks for streaming
DATA_ROOT="/path/to/data/nast-vi-en/fairseq"  # Update this
SAVE_DIR="checkpoints/nast-vi-en/stage1_encoder"
NAST_DIR="src/NAST-S2x/nast"

# GPU/Training configuration
MAX_TOKENS=40000
UPDATE_FREQ=4
NUM_GPUS=4

# Create save directory
mkdir -p ${SAVE_DIR}

echo "=========================================="
echo "Stage 1: Encoder Pretraining (ASR)"
echo "=========================================="
echo "Data: ${DATA_ROOT}"
echo "Save: ${SAVE_DIR}"
echo "Chunk: ${CHUNK_SIZE} (320ms)"
echo "GPUs: ${NUM_GPUS}"
echo ""

# Run training
fairseq-train ${DATA_ROOT} \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --user-dir ${NAST_DIR} \
    --fp16 \
    --task nat_speech_to_text_ctc_modified \
    --arch nonautoregressive_streaming_speech_transformer_segment_to_segment \
    --src-embedding-copy \
    --max-source-positions 6000 \
    --max-target-positions 1024 \
    --src-upsample-ratio 1 \
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
    --criterion nat_loss_ngram_glat_asr \
    --glat-p 0.5:0.3@50k \
    --label-smoothing 0.01 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --clip-norm 10.0 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 0.001 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' \
    --warmup-updates 10000 \
    --stop-min-lr '1e-09' \
    --max-update 100000 \
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
    --num-workers 8

echo ""
echo "=========================================="
echo "Stage 1 Complete!"
echo "=========================================="
echo "Checkpoint: ${SAVE_DIR}/checkpoint_best.pt"
echo ""
echo "Next: Stage 2 (CTC Pretraining)"
echo "Update stage2_train_ctc.sh with:"
echo "  ENCODER_PT=${SAVE_DIR}/checkpoint_best.pt"
