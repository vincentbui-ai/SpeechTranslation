#!/bin/bash
# 1_setup_env.sh
# Setup environment on server after cloning/pulling repo

set -e

REPO_ROOT=$(pwd)
echo "Setting up environment in: $REPO_ROOT"

# Install dependencies
echo "[1/4] Installing Python dependencies..."
pip install -e src/StreamSpeech/fairseq/ --no-build-isolation
pip install -e src/StreamSpeech/SimulEval/
pip install sentencepiece torchaudio soundfile sacrebleu huggingface_hub

# Create directories
echo "[2/4] Creating directories..."
mkdir -p pretrain_models/mHuBERT
mkdir -p pretrain_models/vocoder
mkdir -p checkpoints
mkdir -p data
mkdir -p logs

# Download pretrained models (for reference/evaluation)
echo "[3/4] Downloading mHuBERT model..."
wget -q -P pretrain_models/mHuBERT \
    https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt

echo "[4/4] Downloading vocoder (optional, for S2ST inference)..."
wget -q -P pretrain_models/vocoder \
    https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000
wget -q -P pretrain_models/vocoder \
    https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json

echo "[✓] Setup complete!"
echo "  - fairseq installed"
echo "  - SimulEval installed"
echo "  - Models downloaded to pretrain_models/"
