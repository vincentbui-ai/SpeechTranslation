#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

# Training script for Model 2: Unified Speech & Text Model
# Supports ALL 5 tasks:
#   - S2TT (Speech-to-Text Translation)
#   - T2TT (Text-to-Text Translation)
#   - ASR (Automatic Speech Recognition)
#   - S2ST (Speech-to-Speech Translation)
#   - T2ST (Text-to-Speech Translation)
#
# Note: Model 2 = Model 1 + Speech tasks (S2ST, T2ST)

set -e

# Resolve project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Offline/local model settings (can be overridden from shell env)
export FAIRSEQ2_ASSET_DIR="${FAIRSEQ2_ASSET_DIR:-$PROJECT_ROOT/seamless_communication/src/seamless_communication/cards}"
export FAIRSEQ2_CACHE_DIR="${FAIRSEQ2_CACHE_DIR:-$FAIRSEQ2_ASSET_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/checkpoints/hf_home}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/checkpoints/torch_home}"

mkdir -p "$HF_HOME" "$TORCH_HOME"

if [ ! -d "$FAIRSEQ2_ASSET_DIR" ]; then
    echo "Error: FAIRSEQ2_ASSET_DIR not found: $FAIRSEQ2_ASSET_DIR"
    exit 1
fi

echo "FAIRSEQ2_ASSET_DIR=$FAIRSEQ2_ASSET_DIR"
echo "FAIRSEQ2_CACHE_DIR=$FAIRSEQ2_CACHE_DIR"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"

# Default configuration
MODEL_NAME="seamlessM4T_v2_large"
MODE="SPEECH_TO_SPEECH"  # Must be SPEECH_TO_SPEECH to enable all tasks
BATCH_SIZE=4
LEARNING_RATE=5e-7
MAX_EPOCHS=20
PATIENCE=5
WARMUP_STEPS=100
EVAL_STEPS=200
LOG_STEPS=50
MAX_SRC_TOKENS=3000
SEED=42
DEVICE="cuda"
NUM_GPUS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_dataset)
            TRAIN_DATASET="$2"
            shift 2
            ;;
        --eval_dataset)
            EVAL_DATASET="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --save_model_to)
            SAVE_MODEL_TO="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Training Script for Model 2: Unified Speech & Text Model"
            echo ""
            echo "This model supports ALL 5 tasks:"
            echo "  1. S2TT - Speech-to-Text Translation"
            echo "  2. T2TT - Text-to-Text Translation"
            echo "  3. ASR  - Automatic Speech Recognition"
            echo "  4. S2ST - Speech-to-Speech Translation"
            echo "  5. T2ST - Text-to-Speech Translation"
            echo ""
            echo "Note: Model 2 = Model 1 + Speech tasks (S2ST, T2ST)"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --train_dataset PATH     Path to training manifest(s) - supports wildcards"
            echo "  --eval_dataset PATH      Path to evaluation manifest(s) - supports wildcards"
            echo "  --model_name NAME        Model name (default: seamlessM4T_v2_large)"
            echo "  --save_model_to PATH     Path to save trained model"
            echo "  --batch_size N           Batch size (default: 4)"
            echo "  --learning_rate FLOAT    Learning rate (default: 5e-7)"
            echo "  --max_epochs N           Max epochs (default: 20)"
            echo "  --patience N             Early stopping patience (default: 5)"
            echo "  --num_gpus N             Number of GPUs (default: 1)"
            echo "  --device DEVICE          Device (default: cuda)"
            echo ""
            echo "Examples:"
            echo "  # Train unified model"
            echo "  $0 --train_dataset data/train.json --eval_dataset data/eval.json \\"
            echo "     --save_model_to models/model2_unified.pt --num_gpus 4"
            echo ""
            echo "  # Train with multiple datasets"
            echo "  $0 --train_dataset 'data/train_*.json' --eval_dataset data/eval.json \\"
            echo "     --save_model_to models/model2_unified.pt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$TRAIN_DATASET" ] || [ -z "$EVAL_DATASET" ] || [ -z "$SAVE_MODEL_TO" ]; then
    echo "Error: Missing required arguments"
    echo "Use --help for usage information"
    exit 1
fi

# Combine multiple training files if wildcard is used
if [[ "$TRAIN_DATASET" == *"*"* ]]; then
    echo "Combining training files matching: $TRAIN_DATASET"
    COMBINED_TRAIN="data/combined_train_unified_manifest.json"
    python3 -c "
import json
import glob
import sys

files = glob.glob('$TRAIN_DATASET')
if not files:
    print('No files found matching pattern')
    sys.exit(1)

samples = []
for f in sorted(files):
    with open(f, 'r') as fp:
        for line in fp:
            samples.append(json.loads(line))

# Validate samples have required fields
valid_samples = []
for sample in samples:
    has_source_audio = 'source' in sample and 'audio_local_path' in sample['source']
    has_target_text = 'target' in sample and 'text' in sample['target']
    has_target_units = 'target' in sample and 'units' in sample['target'] and sample['target']['units'] is not None
    
    if has_source_audio and has_target_text:
        if has_target_units:
            valid_samples.append(sample)
        else:
            # Keep sample for text tasks only (will skip during speech training)
            valid_samples.append(sample)
    else:
        print(f'Warning: Sample missing required fields, skipping')

# Reassign IDs
for idx, sample in enumerate(valid_samples):
    sample['source']['id'] = idx
    sample['target']['id'] = idx

with open('$COMBINED_TRAIN', 'w') as fp:
    for sample in valid_samples:
        fp.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f'Combined {len(valid_samples)} samples from {len(files)} files')
"
    TRAIN_DATASET="$COMBINED_TRAIN"
fi

# Create output directory
mkdir -p "$(dirname "$SAVE_MODEL_TO")"

echo "=============================================="
echo "Training Model 2: Unified Speech & Text Model"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Mode: $MODE (enables all 5 tasks)"
echo "Train dataset: $TRAIN_DATASET"
echo "Eval dataset: $EVAL_DATASET"
echo "Save to: $SAVE_MODEL_TO"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Max epochs: $MAX_EPOCHS"
echo "Num GPUs: $NUM_GPUS"
echo ""
echo "Supported Tasks:"
echo "  ✓ S2TT - Speech-to-Text Translation"
echo "  ✓ T2TT - Text-to-Text Translation"
echo "  ✓ ASR  - Automatic Speech Recognition"
echo "  ✓ S2ST - Speech-to-Speech Translation"
echo "  ✓ T2ST - Text-to-Speech Translation"
echo "=============================================="

# Build training command
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with torchrun
    CMD="torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=$NUM_GPUS \
        --no-python \
        m4t_finetune"
else
    # Single GPU training
    CMD="python3 -m seamless_communication.cli.m4t.finetune.finetune"
fi

# Run training
$CMD \
    --mode "$MODE" \
    --train_dataset "$TRAIN_DATASET" \
    --eval_dataset "$EVAL_DATASET" \
    --model_name "$MODEL_NAME" \
    --save_model_to "$SAVE_MODEL_TO" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --warmup_steps "$WARMUP_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --log_steps "$LOG_STEPS" \
    --max_src_tokens "$MAX_SRC_TOKENS" \
    --seed "$SEED" \
    --device "$DEVICE"

echo ""
echo "Training complete! Model saved to: $SAVE_MODEL_TO"
echo ""
echo "Model 2 supports all 5 tasks:"
echo "  1. S2TT - Speech-to-Text Translation (vie↔eng)"
echo "  2. T2TT - Text-to-Text Translation (vie↔eng)"
echo "  3. ASR  - Automatic Speech Recognition (vie, eng)"
echo "  4. S2ST - Speech-to-Speech Translation (vie→eng)"
echo "  5. T2ST - Text-to-Speech Translation (vie↔eng)"
echo ""
echo "Next steps:"
echo "  1. Evaluate model: python src/training/evaluate_unified.py --model $SAVE_MODEL_TO"
echo "  2. Test inference: python src/training/inference_unified.py --model $SAVE_MODEL_TO --task s2st"
echo "  3. Export to ONNX: python src/training/export.py --model $SAVE_MODEL_TO"
