#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

# Training script for Model 1: Text-based tasks (S2TT, T2TT, ASR)
# Supports multiple language pairs and multiple training files

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

# Multi-GPU stability settings
export OMP_NUM_THREADS=1                    # Prevent OpenMP thread oversubscription
export MKL_NUM_THREADS=1                    # Prevent MKL thread conflicts
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Better error handling for NCCL
export NCCL_DEBUG=WARN                      # Show NCCL warnings for debugging
export PYTHONFAULTHANDLER=1                 # Better Python crash diagnostics

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
MODE="SPEECH_TO_TEXT"
BATCH_SIZE=8
LEARNING_RATE=1e-6
MAX_EPOCHS=20
PATIENCE=5
WARMUP_STEPS=100
EVAL_STEPS=200
LOG_STEPS=50
MAX_SRC_TOKENS=4000
SEED=42
DEVICE="cuda"
NUM_GPUS=1
CUDA_VISIBLE_DEVICES_VALUE=""

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
        --cuda_visible_devices|--cuda_device_visible)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --train_dataset PATH     Path to training manifest(s) - supports wildcards"
            echo "  --eval_dataset PATH      Path to evaluation manifest(s) - supports wildcards"
            echo "  --model_name NAME        Model name (default: seamlessM4T_v2_large)"
            echo "  --save_model_to PATH     Path to save trained model"
            echo "  --batch_size N           Batch size (default: 8)"
            echo "  --learning_rate FLOAT    Learning rate (default: 1e-6)"
            echo "  --max_epochs N           Max epochs (default: 20)"
            echo "  --patience N             Early stopping patience (default: 5)"
            echo "  --num_gpus N             Number of GPUs (default: 1)"
            echo "  --device DEVICE          Device (default: cuda)"
            echo "  --cuda_visible_devices   CUDA_VISIBLE_DEVICES value (e.g. 6,7)"
            echo ""
            echo "Examples:"
            echo "  $0 --train_dataset data/train_*.json --eval_dataset data/eval.json --save_model_to model.pt"
            echo "  $0 --train_dataset data/train.json --eval_dataset data/eval.json --save_model_to model.pt --cuda_visible_devices 6,7 --num_gpus 2"
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

# Configure CUDA_VISIBLE_DEVICES if requested
if [ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"

    IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES_VALUE"
    SELECTED_GPU_COUNT="${#GPU_LIST[@]}"

    if [ "$NUM_GPUS" -gt "$SELECTED_GPU_COUNT" ]; then
        echo "Error: --num_gpus ($NUM_GPUS) is greater than selected CUDA devices ($SELECTED_GPU_COUNT)"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        exit 1
    fi
fi

# Combine multiple training files if wildcard is used
if [[ "$TRAIN_DATASET" == *"*"* ]]; then
    echo "Combining training files matching: $TRAIN_DATASET"
    COMBINED_TRAIN="data/combined_train_manifest.json"
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

# Reassign IDs
for idx, sample in enumerate(samples):
    sample['source']['id'] = idx
    sample['target']['id'] = idx

with open('$COMBINED_TRAIN', 'w') as fp:
    for sample in samples:
        fp.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f'Combined {len(samples)} samples from {len(files)} files')
"
    TRAIN_DATASET="$COMBINED_TRAIN"
fi

# Create output directory
mkdir -p "$(dirname "$SAVE_MODEL_TO")"

echo "=========================================="
echo "Training Model 1: Text Tasks (S2TT/T2TT/ASR)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Mode: $MODE"
echo "Train dataset: $TRAIN_DATASET"
echo "Eval dataset: $EVAL_DATASET"
echo "Save to: $SAVE_MODEL_TO"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Max epochs: $MAX_EPOCHS"
echo "Num GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "=========================================="

# Build training command (always use torchrun, including 1 GPU)
# Note: If you encounter DataLoader segfaults, try adding --num_workers=0 to finetune args
CMD="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=$NUM_GPUS \
    --master_port=29500 \
    -m seamless_communication.cli.m4t.finetune.finetune"

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
