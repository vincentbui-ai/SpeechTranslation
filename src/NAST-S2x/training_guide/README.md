# NAST-S2X Training Guide for Vietnamese-English

Complete guide for training NAST-S2X (Non-Autoregressive Streaming Speech-to-Any Translation) for Vietnamese ↔ English language pair.

---

## Overview

**NAST-S2X** is a fast, end-to-end simultaneous speech-to-speech translation model that:
- Is **28× faster** than autoregressive models
- Supports both **offline** and **streaming** (simultaneous) decoding
- Generates target speech **without** intermediate text decoding
- Uses **CTC-based non-monotonic latent alignment** and **glancing mechanism**

---

## Training Pipeline

| Stage | Description | Duration |
|-------|-------------|----------|
| **Stage 0** | Environment Setup & Installation | 30 min |
| **Stage 1** | Data Preparation (JSONL → fairseq format) | 1-2 hours |
| **Stage 2** | Feature Extraction (fbank + units) | 4-8 hours |
| **Stage 3** | Encoder Pretraining (ASR) | 1-2 days |
| **Stage 4** | CTC Pretraining (Speech-to-Unit) | 2-3 days |
| **Stage 5** | NMLA Training (Non-Monotonic Latent Alignment) | 3-5 days |

---

## Directory Structure

```
src/NAST-S2x/training_guide/
├── README.md                    # This file
├── 01_INSTALL.md               # Installation guide
├── 02_DATA_PREP.md             # Data preparation
├── 03_TRAINING.md              # Training stages
├── 04_INFERENCE.md             # Inference guide
└── scripts/
    ├── prepare_nast_data.py    # JSONL → fairseq TSV
    ├── extract_fbank.py        # Extract fbank features
    ├── extract_units.py        # Extract acoustic units
    └── preprocess.sh           # Full preprocessing pipeline
```

---

## Quick Start

### 1. Installation

```bash
cd src/NAST-S2x

# Install fairseq with NAST modifications
pip install -e fairseq/

# Install SimulEval (for streaming evaluation)
pip install -e SimulEval/

# Install dependencies
pip install sentencepiece torchaudio soundfile
```

### 2. Data Preparation

```bash
# Convert JSONL metadata to fairseq format
python training_guide/scripts/prepare_nast_data.py \
    --datasets-dir ../../../datasets \
    --output-dir ../../../data/nast-vi-en
```

### 3. Training

```bash
# Stage 1: Encoder Pretraining
bash training_guide/scripts/stage1_pretrain_encoder.sh

# Stage 2: CTC Pretraining
bash training_guide/scripts/stage2_train_ctc.sh

# Stage 3: NMLA Training
bash training_guide/scripts/stage3_train_nmla.sh
```

### 4. Inference

```bash
# Offline inference
python training_guide/scripts/inference.py \
    --audio path/to/audio.wav \
    --src-lang vie \
    --tgt-lang eng \
    --checkpoint checkpoints/nast-vi-en/offline.pt
```

---

## Dataset Format

This repository uses **JSONL** format (same as StreamSpeech):

```json
{
  "audio_filepath": "/path/to/audio.flac",
  "duration": 6.839,
  "ori_text": "source text",
  "ori_lang": "Vietnamese",
  "tgt_text": "target translation",
  "tgt_lang": "English"
}
```

**Direction support:**
- VI → EN: Vietnamese speech → English translation
- EN → VI: English speech → Vietnamese translation

---

## Training Stages Details

### Stage 1: Encoder Pretraining (ASR)
Train the encoder on ASR task for better initialization.

**CLI Commands:**
```bash
--arch nonautoregressive_streaming_speech_transformer_segment_to_segment
--task nat_speech_to_text_ctc_modified
--criterion nat_loss_ngram_glat_asr
```

### Stage 2: CTC Pretraining (Speech-to-Unit)
Train on acoustic unit prediction (speech-to-unit).

**CLI Commands:**
```bash
--arch nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment
--task nat_speech_to_unit_ctc_modified
--criterion nat_loss_ngram_glat_s2u
```

### Stage 3: NMLA Training
Fine-tune with Non-Monotonic Latent Alignment.

---

## Configuration

### Chunk Sizes for Streaming

| Chunk Size | Use Case | Latency |
|------------|----------|---------|
| 320ms | Ultra-low latency | < 1s |
| 1280ms | Balanced | ~2-3s |
| 2560ms | High quality | ~4-5s |
| Offline | Best quality | Full audio |

### Model Architecture

```yaml
# Key parameters
src_embedding_copy: true
target_is_code: true
target_code_size: 1000
src_upsample_ratio: 1
hidden_upsample_ratio: 6
main_context: 32      # 320ms chunk
right_context: 32
unit_size: 2
```

---

## Checkpoints

Pretrained models available at [HuggingFace](https://huggingface.co/ICTNLP/NAST-S2X):

| Model | ASR-BLEU | Description |
|-------|----------|-------------|
| chunk_320ms.pt | 19.67 | Ultra-low latency |
| chunk_1280ms.pt | 20.20 | Balanced |
| chunk_2560ms.pt | 24.88 | High quality |
| Offline.pt | 25.82 | Best quality |

For Vietnamese-English, train from scratch or fine-tune from FR-EN checkpoints.

---

## References

- Paper: [A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Any Translation](https://arxiv.org/abs/2406.06937) (ACL 2024)
- Code: https://github.com/ictnlp/NAST-S2x
- Models: https://huggingface.co/ICTNLP/NAST-S2X

---

## Citation

```bibtex
@inproceedings{ma2024nonautoregressive,
  title={A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Any Translation},
  author={Ma, Zhengrui and Fang, Qingkai and Zhang, Shaolei and Guo, Shoutao and Feng, Yang and Zhang, Min},
  booktitle={Proceedings of ACL 2024},
  year={2024},
}
```
