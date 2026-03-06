#!/bin/bash
# 4_train_scratch.sh
# Train VI→EN from scratch on 6× B200 GPUs

set -e

REPO_ROOT=$(pwd)
STREAMSPEECH=$REPO_ROOT/src/StreamSpeech
FAIRSEQ=$STREAMSPEECH/fairseq

# Configuration
DATA=$REPO_ROOT/data/vi-en
CONFIG=$REPO_ROOT/configs/vi-en
SAVE_DIR=$REPO_ROOT/checkpoints/streamspeech.vi-en.scratch
LOG_FILE=$REPO_ROOT/logs/train.vi-en.$(date +%Y%m%d_%H%M%S).log

# B200 GPU config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
MAX_TOKENS=60000      # Large batch for B200 141GB
UPDATE_FREQ=1         # No gradient accumulation
NUM_WORKERS=16        # High for fast loading

echo "============================================"
echo "Training VI→EN from scratch"
echo "============================================"
echo "GPUs: 6× B200 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Batch: MAX_TOKENS=$MAX_TOKENS, UPDATE_FREQ=$UPDATE_FREQ"
echo "Data: $DATA"
echo "Save: $SAVE_DIR"
echo "Log: $LOG_FILE"
echo "============================================"

mkdir -p $SAVE_DIR
mkdir -p $(dirname $LOG_FILE)

cd $STREAMSPEECH

# Run training with logging
PYTHONPATH=$FAIRSEQ fairseq-train $DATA \
  --user-dir $STREAMSPEECH/researches/ctc_unity \
  \
  --config-yaml $CONFIG/config_gcmvn.yaml \
  --multitask-config-yaml $CONFIG/config_mtl_asr_st_ctcst.yaml \
  \
  --task speech_to_speech_ctc \
  --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --criterion speech_to_unit_2pass_ctc_asr_st \
  --label-smoothing 0.1 --rdrop-alpha 0.0 \
  \
  --arch streamspeech \
  --share-decoder-input-output-embed \
  --encoder-layers 12 \
  --encoder-embed-dim 256 \
  --encoder-ffn-embed-dim 2048 \
  --encoder-attention-heads 4 \
  --translation-decoder-layers 4 \
  --synthesizer-encoder-layers 2 \
  --decoder-layers 2 \
  --decoder-embed-dim 512 \
  --decoder-ffn-embed-dim 2048 \
  --decoder-attention-heads 8 \
  \
  --k1 0 --k2 0 --n1 1 --n2 -1 \
  --chunk-size 999999 \
  \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  \
  --train-subset train \
  --valid-subset dev \
  --ctc-upsample-rate 25 \
  \
  --save-dir $SAVE_DIR \
  --validate-interval-updates 2000 \
  --save-interval-updates 2000 \
  --keep-last-epochs 10 \
  --keep-interval-updates 30 \
  --keep-best-checkpoints 15 \
  \
  --no-progress-bar --log-format json --log-interval 50 \
  \
  --lr 0.001 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-7 \
  --warmup-updates 10000 \
  --optimizer adam \
  --adam-betas "(0.9,0.98)" \
  --clip-norm 1.0 \
  \
  --max-tokens $MAX_TOKENS \
  --max-target-positions 1200 \
  --update-freq $UPDATE_FREQ \
  --attn-type espnet \
  --pos-enc-type rel_pos \
  \
  --seed 42 \
  --fp16 \
  --memory-efficient-fp16 \
  --num-workers $NUM_WORKERS \
  \
  --ddp-backend=legacy_ddp \
  --distributed-world-size 6 \
  \
  2>&1 | tee $LOG_FILE

echo ""
echo "[✓] Training started!"
echo "Monitor with: tail -f $LOG_FILE"
echo ""
echo "Checkpoints will be saved to: $SAVE_DIR"
