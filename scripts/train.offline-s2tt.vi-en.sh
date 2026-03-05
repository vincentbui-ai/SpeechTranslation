#!/usr/bin/env bash
# =============================================================================
# train.offline-s2tt.vi-en.sh
# -----------------------------------------------------------------------------
# Huấn luyện offline Speech-to-Text Translation VI → EN với StreamSpeech
# (multi-task: ASR CTC + ST CTC + MT Transformer)
#
# Dựa trên:
#   src/StreamSpeech/researches/ctc_unity/train_scripts/train.offline-s2st.sh
#
# ⚠️  CHẠY TRÊN SERVER — cần GPU và audio files.
#
# Trước khi chạy:
#   1. Cập nhật SERVER_ROOT bên dưới
#   2. Đảm bảo đã chạy compute_gcmvn.py và cập nhật config_gcmvn.yaml
#   3. Cập nhật đường dẫn trong config_mtl_asr_st_ctcst.yaml (Bước 5 README)
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# CẬP NHẬT THEO SERVER CỦA BẠN
# ---------------------------------------------------------------------------
SERVER_ROOT=/PATH/TO/SERVER/SpeechTranslation

# ---------------------------------------------------------------------------
# Đường dẫn — tự động từ SERVER_ROOT
# ---------------------------------------------------------------------------
STREAMSPEECH_ROOT=$SERVER_ROOT/src/StreamSpeech
FAIRSEQ_ROOT=$STREAMSPEECH_ROOT/fairseq

DATA=$SERVER_ROOT/data/vi-en          # chứa train.tsv, dev.tsv
CONFIG_DIR=$SERVER_ROOT/configs/vi-en

PRETRAIN_CKPT=$SERVER_ROOT/pretrain_models/streamspeech.simultaneous.fr-en.pt
MODEL_NAME=streamspeech.offline-s2tt.vi-en

# ---------------------------------------------------------------------------
# GPU config — điều chỉnh theo VRAM
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 40GB A100  : MAX_TOKENS=22000, UPDATE_FREQ=2
# 24GB 3090  : MAX_TOKENS=14000, UPDATE_FREQ=4
# 16GB V100  : MAX_TOKENS=10000, UPDATE_FREQ=8
MAX_TOKENS=22000
UPDATE_FREQ=2

# ---------------------------------------------------------------------------
# Tải pretrained checkpoint (chỉ cần chạy 1 lần)
# ---------------------------------------------------------------------------
# mkdir -p $SERVER_ROOT/pretrain_models
# pip install huggingface_hub
# python -c "
# from huggingface_hub import hf_hub_download
# hf_hub_download(
#     repo_id='ICTNLP/StreamSpeech_Models',
#     filename='streamspeech.simultaneous.fr-en.pt',
#     local_dir='$SERVER_ROOT/pretrain_models/'
# )"

# ---------------------------------------------------------------------------
# Cài fairseq + SimulEval từ src/StreamSpeech (nếu chưa cài trên server)
# ---------------------------------------------------------------------------
# pip install -e $FAIRSEQ_ROOT/
# pip install -e $STREAMSPEECH_ROOT/SimulEval/

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
mkdir -p $SERVER_ROOT/checkpoints/$MODEL_NAME

cd $STREAMSPEECH_ROOT

PYTHONPATH=$FAIRSEQ_ROOT fairseq-train $DATA \
  --user-dir $STREAMSPEECH_ROOT/researches/ctc_unity \
  \
  --config-yaml           $CONFIG_DIR/config_gcmvn.yaml \
  --multitask-config-yaml $CONFIG_DIR/config_mtl_asr_st_ctcst.yaml \
  \
  --task      speech_to_speech_ctc \
  --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --criterion speech_to_unit_2pass_ctc_asr_st \
  --label-smoothing 0.1 --rdrop-alpha 0.0 \
  \
  --arch streamspeech \
  --share-decoder-input-output-embed \
  --encoder-layers          12 \
  --encoder-embed-dim       256 \
  --encoder-ffn-embed-dim   2048 \
  --encoder-attention-heads 4 \
  --translation-decoder-layers  4 \
  --synthesizer-encoder-layers  2 \
  --decoder-layers          2 \
  --decoder-embed-dim       512 \
  --decoder-ffn-embed-dim   2048 \
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
  --save-dir $SERVER_ROOT/checkpoints/$MODEL_NAME \
  --validate-interval-updates 5000 \
  --save-interval-updates     5000 \
  --keep-last-epochs      5 \
  --keep-interval-updates 20 \
  --keep-best-checkpoints 10 \
  \
  --no-progress-bar --log-format json --log-interval 100 \
  \
  --lr 0.0003 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-7 \
  --warmup-updates 4000 \
  --optimizer adam \
  --adam-betas "(0.9,0.98)" \
  --clip-norm 1.0 \
  \
  --max-tokens           $MAX_TOKENS \
  --max-target-positions 1200 \
  --update-freq          $UPDATE_FREQ \
  --attn-type espnet \
  --pos-enc-type rel_pos \
  \
  --seed 42 --fp16 --num-workers 8 \
  \
  --finetune-from-model $PRETRAIN_CKPT

# ---------------------------------------------------------------------------
# Ghi chú về các flag khác so với script gốc train.offline-s2st.sh:
#
#   --chunk-size 999999   : giữ nguyên — offline mode
#   --finetune-from-model : load weights FR-EN, reset unit-decoder head
#   --lr 0.0003           : thấp hơn gốc (0.001) vì fine-tune
#   --warmup-updates 4000 : thấp hơn gốc (10000) vì fine-tune
#
# Nếu train từ đầu (không có pretrain):
#   bỏ --finetune-from-model
#   đổi --lr 0.001 --warmup-updates 10000
# ---------------------------------------------------------------------------
