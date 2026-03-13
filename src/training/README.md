# Training Pipeline for Speech Translation Models

This directory contains the training infrastructure for finetuning SeamlessM4T models on custom datasets.

## Overview

Two models are trained:

### **Model 1: Text Tasks Only**
**Purpose:** Lightweight model for text output tasks  
**Tasks:**
- **S2TT** - Speech-to-Text Translation (Vietnamese ↔ English)
- **T2TT** - Text-to-Text Translation (Vietnamese ↔ English)  
- **ASR** - Automatic Speech Recognition (Vietnamese, English)

### **Model 2: Unified Model (Recommended)**
**Purpose:** Complete solution supporting ALL tasks  
**Tasks:**
- **S2TT** - Speech-to-Text Translation (Vietnamese ↔ English)
- **T2TT** - Text-to-Text Translation (Vietnamese ↔ English)
- **ASR** - Automatic Speech Recognition (Vietnamese, English)
- **S2ST** - Speech-to-Speech Translation (Vietnamese → English)
- **T2ST** - Text-to-Speech Translation (Vietnamese ↔ English)

**Note:** Model 2 = Model 1 + Speech tasks (S2ST, T2ST)

## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Train both models
bash src/training/run_training_pipeline.sh

# Train only Model 1 (Text tasks only)
bash src/training/run_training_pipeline.sh --model1_only --num_gpus 4

# Train only Model 2 (Unified model with all tasks)
bash src/training/run_training_pipeline.sh --model2_only --num_gpus 4

# Skip data preparation (already prepared)
bash src/training/run_training_pipeline.sh --skip_data_prep --skip_units
```

### Option 2: Step-by-Step

#### Step 1: Prepare Data

```bash
# Convert metadata for Model 1
python src/training/convert_metadata.py \
    --input_files datasets/metadata*.json \
    --output_dir data/manifests/text \
    --mode text \
    --split

# Convert metadata for Model 2 (needs target audio)
python src/training/convert_metadata.py \
    --input_files datasets/metadata*.json \
    --output_dir data/manifests/unified \
    --mode speech \
    --split
```

#### Step 2: Extract Units (for Model 2)

```bash
python src/training/extract_units.py \
    --input_manifests data/manifests/unified/*.json \
    --output_dir data/manifests/unified_with_units \
    --model_name xlsr2_1b_v2
```

#### Step 3: Train Models

```bash
# Train Model 1 (Text tasks only)
bash src/training/train_model1_text.sh \
    --train_dataset data/manifests/text/train_manifest.json \
    --eval_dataset data/manifests/text/val_manifest.json \
    --save_model_to models/model1_text.pt \
    --num_gpus 1

# Train Model 2 (Unified model - ALL 5 tasks)
bash src/training/train_model2_unified.sh \
    --train_dataset data/manifests/unified_with_units/train_manifest_with_units.json \
    --eval_dataset data/manifests/unified_with_units/val_manifest_with_units.json \
    --save_model_to models/model2_unified.pt \
    --num_gpus 1
```

## Directory Structure

```
SpeechTranslation/
├── src/training/
│   ├── __init__.py
│   ├── README.md                            # This file
│   ├── convert_metadata.py                  # Convert metadata format
│   ├── extract_units.py                     # Extract discrete units
│   ├── train_model1_text.sh                 # Train Model 1 (Text only)
│   ├── train_model2_unified.sh              # Train Model 2 (All tasks)
│   └── run_training_pipeline.sh             # Complete pipeline
├── data/
│   └── manifests/
│       ├── text/                           # Model 1 manifests
│       │   ├── train_manifest.json
│       │   ├── val_manifest.json
│       │   └── test_manifest.json
│       ├── unified/                        # Model 2 manifests (pre-units)
│       └── unified_with_units/             # Model 2 manifests (with units)
├── models/                                 # Trained models
│   ├── model1_text.pt                     # Model 1 checkpoint
│   └── model2_unified.pt                  # Model 2 checkpoint
└── logs/                                   # Training logs
```

## Input Data Format

Your metadata files should follow this structure:

```json
{
    "source_audio": "datasets/wavs/audio_001.wav",
    "duration": 2.32,
    "source_text": "văn bản tiếng Việt",
    "source_lang": "Vietnamese",
    "target_text": "English text",
    "target_lang": "English",
    "target_audio": "datasets/wavs/audio_001.wav"
}
```

**Field Descriptions:**
- `source_audio`: Path to source audio file (required for both models)
- `duration`: Audio duration in seconds
- `source_text`: Source language transcript
- `source_lang`: Source language name (Vietnamese/English)
- `target_text`: Target language transcript
- `target_lang`: Target language name (Vietnamese/English)
- `target_audio`: Path to target audio file (required for Model 2)

## Training Modes

### Model 1: Text Tasks Only

| Task | Mode | Description | Data Required |
|------|------|-------------|---------------|
| S2TT | `SPEECH_TO_TEXT` | Speech-to-Text Translation | source_audio + target_text |
| T2TT | `SPEECH_TO_TEXT` | Text-to-Text Translation | source_text + target_text |
| ASR  | `SPEECH_TO_TEXT` | Automatic Speech Recognition | source_audio + source_text (as target) |

All text tasks use `SPEECH_TO_TEXT` mode. For ASR, set `target_text` = `source_text`.

### Model 2: Unified Model (All Tasks)

| Task | Mode | Description | Data Required |
|------|------|-------------|---------------|
| S2TT | `SPEECH_TO_SPEECH` | Speech-to-Text Translation | source_audio + target_text |
| T2TT | `SPEECH_TO_SPEECH` | Text-to-Text Translation | source_text + target_text |
| ASR  | `SPEECH_TO_SPEECH` | Automatic Speech Recognition | source_audio + source_text |
| S2ST | `SPEECH_TO_SPEECH` | Speech-to-Speech Translation | source_audio + target_units |
| T2ST | `SPEECH_TO_SPEECH` | Text-to-Speech Translation | source_text + target_units |

**Note:** Model 2 uses `SPEECH_TO_SPEECH` mode to train both S2T and T2U components simultaneously, enabling all 5 tasks.

## Hyperparameters

### Model 1 (Text Only)

```bash
--mode SPEECH_TO_TEXT
--batch_size 8
--learning_rate 1e-6
--max_epochs 20
--patience 5
```

### Model 2 (Unified)

```bash
--mode SPEECH_TO_SPEECH
--batch_size 4
--learning_rate 5e-7
--max_epochs 20
--patience 5
```

**Why lower learning rate for Model 2?**
- Training both S2T and T2U components
- Requires more careful optimization
- Prevents catastrophic forgetting of pretrained weights

## Multi-GPU Training

```bash
# Model 1 with 4 GPUs
bash src/training/train_model1_text.sh \
    --train_dataset data/manifests/text/train_manifest.json \
    --eval_dataset data/manifests/text/val_manifest.json \
    --save_model_to models/model1_text.pt \
    --num_gpus 4

# Model 2 (Unified) with 4 GPUs
bash src/training/train_model2_unified.sh \
    --train_dataset data/manifests/unified_with_units/train_manifest_with_units.json \
    --eval_dataset data/manifests/unified_with_units/val_manifest_with_units.json \
    --save_model_to models/model2_unified.pt \
    --num_gpus 4
```

## Model Comparison

| Feature | Model 1 (Text) | Model 2 (Unified) |
|---------|----------------|-------------------|
| **Tasks** | 3 (S2TT, T2TT, ASR) | 5 (All tasks) |
| **Output** | Text only | Text + Speech |
| **Speed** | Faster (skips vocoder) | Slower (includes vocoder) |
| **Size** | ~2.3B params | ~2.3B params |
| **Training Time** | Shorter | Longer |
| **Use Case** | Text translation/ASR | Full speech pipeline |

**Recommendation:**
- Use **Model 2** for production (one model handles everything)
- Use **Model 1** if you only need text output (slightly faster inference)
- Both models can coexist and be loaded based on task requirements

## Troubleshooting

### Out of Memory

Reduce batch size or max_src_tokens:
```bash
--batch_size 2
--max_src_tokens 2000
```

### Missing Units for Model 2

Ensure target_audio field exists and units are extracted:
```bash
# Re-extract units
python src/training/extract_units.py \
    --input_manifests data/manifests/unified/*.json \
    --output_dir data/manifests/unified_with_units
```

### Language Code Errors

Supported language mappings:
- Vietnamese → vie
- English → eng

## Advanced Usage

### Custom Data Split

```bash
python src/training/convert_metadata.py \
    --input_files datasets/*.json \
    --output_dir data/manifests \
    --mode text \
    --split \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

### Multiple Metadata Files

```bash
# Wildcard support
bash src/training/train_model1_text.sh \
    --train_dataset "data/manifests/train_*.json" \
    --eval_dataset "data/manifests/val_*.json" \
    --save_model_to models/model1_text.pt

# Multiple specific files
bash src/training/train_model1_text.sh \
    --train_dataset "data/train_vie_eng.json data/train_eng_vie.json" \
    --eval_dataset data/eval.json \
    --save_model_to models/model1_text.pt
```

### Freeze Specific Layers

Edit training scripts and add:
```bash
--freeze_layers "encoder" "decoder.self_attn"
```

## Monitoring Training

Training logs are saved to `logs/` directory. Monitor with:

```bash
# Monitor Model 1
tail -f logs/train_model1.log

# Monitor Model 2
tail -f logs/train_model2.log
```

### Expected Training Progress

**Model 1 (Text):**
```
Epoch 001 / update 00050: train loss=5.2341
Epoch 001 / update 00100: train loss=4.8912
Eval after 100 updates: loss=4.5678 best_loss=4.5678
...
Epoch 010 / update 01000: train loss=2.1234
Eval after 1000 updates: loss=2.3456 best_loss=2.2345 patience_steps_left=5
Saving model
```

**Model 2 (Unified):**
```
Epoch 001 / update 00050: train loss=5.6789 (s2t=3.2345, t2u=2.4444)
Epoch 001 / update 00100: train loss=5.2341 (s2t=2.9876, t2u=2.2465)
Eval after 100 updates: loss=4.8912 best_loss=4.8912
...
```

## Evaluation

After training, evaluate models:

```bash
# Evaluate Model 1
python src/training/evaluate.py \
    --model models/model1_text.pt \
    --test_dataset data/manifests/text/test_manifest.json \
    --tasks s2tt t2tt asr

# Evaluate Model 2 (Unified)
python src/training/evaluate.py \
    --model models/model2_unified.pt \
    --test_dataset data/manifests/unified_with_units/test_manifest_with_units.json \
    --tasks s2tt t2tt asr s2st t2st
```

## Inference Examples

### Model 1 (Text Tasks)

```python
from seamless_communication.inference import Translator

# Load Model 1
translator = Translator("seamlessM4T_v2_large", None, device, dtype)
translator.load_model("models/model1_text.pt")

# S2TT
result = translator.predict(audio_path, task_str="S2TT", tgt_lang="eng")

# T2TT
result = translator.predict(text, task_str="T2TT", src_lang="vie", tgt_lang="eng")

# ASR
result = translator.predict(audio_path, task_str="ASR", tgt_lang="vie")
```

### Model 2 (Unified - All Tasks)

```python
from seamless_communication.inference import Translator

# Load Model 2
translator = Translator("seamlessM4T_v2_large", "vocoder_v2", device, dtype)
translator.load_model("models/model2_unified.pt")

# S2TT (Text output)
text_result = translator.predict(audio_path, task_str="S2TT", tgt_lang="eng")

# S2ST (Speech output)
text_result, speech_result = translator.predict(
    audio_path, task_str="S2ST", tgt_lang="eng"
)

# T2ST (Text-to-Speech)
text_result, speech_result = translator.predict(
    text, task_str="T2ST", src_lang="vie", tgt_lang="eng"
)
```

## Notes

- **Data size:** With only 1.39 hours of data, use very low learning rates (1e-6 to 5e-7)
- **Early stopping:** Patience=5 to prevent overfitting
- **Batch size:** Reduce to 2-4 if OOM
- **Balanced data:** Aim for ~50/50 split between translation directions
- **Model 2 recommendation:** Train Model 2 if you might need speech output in the future
- **Inference speed:** Model 1 is ~30% faster for text-only tasks

## File Structure Summary

```
src/training/
├── convert_metadata.py          # Convert your metadata → seamless format
├── extract_units.py             # Extract units for speech tasks
├── train_model1_text.sh         # Train text-only model
├── train_model2_unified.sh      # Train unified model (all tasks)
└── run_training_pipeline.sh     # Run everything
```

Choose the approach that fits your needs:
- **Quick start:** Use `run_training_pipeline.sh`
- **Custom control:** Run scripts individually
- **Debug/Development:** Run Python scripts directly
