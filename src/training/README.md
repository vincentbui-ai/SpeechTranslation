# Training Pipeline for Speech Translation Models

This directory contains scripts for finetuning SeamlessM4T on custom data.

## Current Status

- `convert_metadata.py` supports **multiple input files** and writes a **single output manifest**.
- `--mode text` is implemented.
- `--mode speech` is **TODO** (not implemented yet).

## Recommended Workflow (STT / SPEECH_TO_TEXT)

Use this workflow for Speech-to-Text training with multitask samples:
- Keep original S2TT sample (`source_audio -> target_text`).
- Duplicate each sample as ASR (`source_audio -> source_text`) at 1:1 by default.

### 1) Prepare Input Metadata

Prepare JSONL files (one JSON object per line), for example:

```json
{"source_audio": "datasets/wavs/a.wav", "source_text": "xin chao", "source_lang": "Vietnamese", "target_text": "hello", "target_lang": "English"}
{"source_audio": "datasets/wavs/b.wav", "source_text": "cam on", "source_lang": "Vietnamese", "target_text": "thank you", "target_lang": "English"}
```

Required fields per record:
- `source_audio`
- `source_text`
- `source_lang`
- `target_text`
- `target_lang`

Language mapping supported:
- `Vietnamese` -> `vie`
- `English` -> `eng`
- `vie` -> `vie`
- `eng` -> `eng`

### 2) Convert Metadata to Manifest

You can pass one or many input files per output manifest:

```bash
# Train manifest from multiple metadata files (default: S2TT + ASR 1:1)
python3 src/training/convert_metadata.py \
  --input_files datasets/train_metadata_vie.json datasets/train_metadata_eng.json \
  --output_file data/manifests/text/train_manifest.json \
  --mode text

# Validation manifest from one file
python3 src/training/convert_metadata.py \
  --input_files datasets/val_metadata.json \
  --output_file data/manifests/text/val_manifest.json \
  --mode text

# Test manifest from one file
python3 src/training/convert_metadata.py \
  --input_files datasets/test_metadata.json \
  --output_file data/manifests/text/test_manifest.json \
  --mode text
```

If you want only translation samples (disable ASR duplication):

```bash
python3 src/training/convert_metadata.py \
  --input_files datasets/train_metadata.json \
  --output_file data/manifests/text/train_manifest.json \
  --mode text \
  --disable_asr
```

Output format is JSONL manifest compatible with finetuning scripts.

### 3) Train Model 1 (Text Tasks)

```bash
bash src/training/train_model1_text.sh \
  --train_dataset data/manifests/text/train_manifest.json \
  --eval_dataset data/manifests/text/val_manifest.json \
  --save_model_to models/model1_text.pt \
  --num_gpus 1
```

`train_model1_text.sh` runs with `--mode SPEECH_TO_TEXT`.

## About Speech Mode

`convert_metadata.py --mode speech` currently raises `NotImplementedError` by design.

When speech mode is implemented, this section should be updated with S2ST/T2ST preprocessing and unit extraction steps.

## Notes

- Input files are expected to be JSONL (`.json` extension is acceptable if each line is one JSON object).
- Multiple input files are concatenated in the order provided via `--input_files`.
- The converter logs task statistics (`s2tt`, `asr`) and language-pair statistics.
- For low-resource finetuning, start with low learning rates (for example `1e-6`).
