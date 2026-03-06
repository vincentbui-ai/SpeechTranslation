# 02 — Data Preparation

Prepare Vietnamese-English dataset for NAST-S2X training.

---

## Overview

NAST-S2X requires:
1. **Audio features**: 80-dim filterbank (fbank) features
2. **Acoustic units**: Discrete speech units (from HuBERT/k-means)
3. **Manifest files**: TSV format for fairseq
4. **Configuration**: YAML config for data loading

---

## Input Data Format

This repository uses **JSONL** format (from `/datasets/`):

```json
{
  "audio_filepath": "/raid/voice/dataset/audio/.../audio.flac",
  "duration": 6.839,
  "ori_text": "the two stopped just inside the door",
  "ori_lang": "English",
  "tgt_text": "Hai người dừng lại ngay bên trong cửa",
  "tgt_lang": "Vietnamese"
}
```

**Dataset files:**
- `trainset_english.jsonl` (EN → VI)
- `trainset_vietnamese.jsonl` (VI → EN)
- `valset_english.jsonl` (validation)
- `valset_vietnamese.jsonl` (validation)

---

## Step 1: Convert JSONL to TSV

```bash
python training_guide/scripts/prepare_nast_data.py \
    --datasets-dir datasets \
    --output-dir data/nast-vi-en
```

**Output structure:**
```
data/nast-vi-en/
├── train.tsv              # Training manifest
├── dev.tsv                # Validation manifest  
├── test.tsv               # Test manifest
├── vi_train.txt           # Vietnamese text (for SPM)
├── en_train.txt           # English text (for SPM)
└── audio_paths.txt        # List of audio files
```

**TSV format:**
```
audio_path  n_frames  tgt_text  speaker
/path/1.flac  109424  translated text  spk0
/path/2.flac  289280  another text     spk1
```

---

## Step 2: Train SentencePiece Model

```bash
python training_guide/scripts/train_spm_nast.py \
    --data-dir data/nast-vi-en \
    --vocab-size 10000 \
    --output-dir data/nast-vi-en/spm
```

**Output:**
- `spm_unigram10000.model` — SentencePiece model
- `spm_unigram10000.vocab` — Vocabulary
- `spm_unigram10000.txt` — fairseq dictionary

---

## Step 3: Extract Filterbank Features

```bash
python training_guide/scripts/extract_fbank.py \
    --tsv-path data/nast-vi-en/train.tsv \
    --output-dir data/nast-vi-en/fbank \
    --num-workers 8
```

**Output:**
- `train.zip` — Zipped fbank features
- `dev.zip` — Zipped fbank features

**Feature config:**
- Sample rate: 16 kHz
- Window: 25ms
- Shift: 10ms
- Dimensions: 80 (mel-filterbanks)

---

## Step 4: Compute Global CMVN

```bash
python training_guide/scripts/compute_cmvn.py \
    --tsv-path data/nast-vi-en/train.tsv \
    --fbank-dir data/nast-vi-en/fbank \
    --output data/nast-vi-en/gcmvn.npz \
    --max-samples 50000
```

---

## Step 5: Extract Acoustic Units

NAST-S2X requires discrete acoustic units (from HuBERT):

```bash
# Download HuBERT model
python training_guide/scripts/download_hubert.py \
    --output-dir checkpoints/hubert

# Extract units
python training_guide/scripts/extract_units.py \
    --tsv-path data/nast-vi-en/train.tsv \
    --hubert-path checkpoints/hubert/hubert_base_ls960.pt \
    --kmeans-path checkpoints/hubert/kmeans_1000.pt \
    --output-dir data/nast-vi-en/units
```

---

## Step 6: Create fairseq Manifest

```bash
python training_guide/scripts/create_fairseq_manifest.py \
    --tsv-path data/nast-vi-en/train.tsv \
    --units-path data/nast-vi-en/units/train.units \
    --output-dir data/nast-vi-en/fairseq
```

**Output:**
```
data/nast-vi-en/fairseq/
├── train.tsv
├── dev.tsv
├── test.tsv
├── spm_unigram10000.model
├── spm_unigram10000.txt
└── config.yaml
```

---

## Step 7: Create config.yaml

```bash
cat > data/nast-vi-en/config.yaml << 'EOF'
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: data/nast-vi-en/fairseq/spm_unigram10000.model

global_cmvn:
  stats_npz_path: data/nast-vi-en/gcmvn.npz

input_channels: 1
input_feat_per_channel: 80

specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0

transforms:
  '*':
  - global_cmvn
  _train:
  - global_cmvn
  - specaugment

vocab_filename: spm_unigram10000.txt

vocoder:
  checkpoint: checkpoints/vocoder/g_00500000
  config: checkpoints/vocoder/config.json
  type: code_hifigan

vocab_filename_src: spm_unigram10000.txt
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: data/nast-vi-en/fairseq/spm_unigram10000.model
EOF
```

---

## Final Data Structure

```
data/nast-vi-en/
├── fairseq/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── spm_unigram10000.model
│   ├── spm_unigram10000.txt
│   └── config.yaml
├── fbank/
│   ├── train.zip
│   └── dev.zip
├── units/
│   ├── train.units
│   └── dev.units
└── gcmvn.npz
```

---

## Automation Script

Run all preprocessing steps:

```bash
bash training_guide/scripts/preprocess.sh \
    --datasets-dir datasets \
    --output-dir data/nast-vi-en
```

---

## Troubleshooting

### Issue: Out of memory during feature extraction

```bash
# Reduce batch size
python extract_fbank.py --batch-size 32  # default is 64
```

### Issue: Slow unit extraction

```bash
# Use multiple GPUs
python extract_units.py --num-gpus 4
```

---

## Next Steps

→ Proceed to [03_TRAINING.md](03_TRAINING.md) for model training.
