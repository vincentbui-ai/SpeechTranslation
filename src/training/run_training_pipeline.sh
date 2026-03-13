#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

# Complete training pipeline for both models
# 
# Model 1: Text-based tasks only (S2TT, T2TT, ASR)
# Model 2: Unified model - All 5 tasks (S2TT, T2TT, ASR + S2ST, T2ST)
#
# Usage: ./run_training_pipeline.sh [options]

set -e

# Configuration
DATA_DIR="datasets"
MANIFEST_DIR="data/manifests"
MODEL_DIR="models"
LOG_DIR="logs"

# Parse arguments
SKIP_DATA_PREP=false
SKIP_UNITS=false
TRAIN_MODEL1=true
TRAIN_MODEL2=true
NUM_GPUS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --skip_data_prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip_units)
            SKIP_UNITS=true
            shift
            ;;
        --model1_only)
            TRAIN_MODEL2=false
            shift
            ;;
        --model2_only)
            TRAIN_MODEL1=false
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Complete Training Pipeline for Speech Translation Models"
            echo ""
            echo "Model 1: Text-based tasks (S2TT, T2TT, ASR)"
            echo "Model 2: Unified model - All 5 tasks (S2TT, T2TT, ASR, S2ST, T2ST)"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data_dir DIR           Data directory (default: datasets)"
            echo "  --model_dir DIR          Model output directory (default: models)"
            echo "  --skip_data_prep         Skip data preparation step"
            echo "  --skip_units             Skip unit extraction (Model 2 needs this)"
            echo "  --model1_only            Train only Model 1 (Text tasks)"
            echo "  --model2_only            Train only Model 2 (Unified model)"
            echo "  --num_gpus N             Number of GPUs (default: 1)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run full pipeline (both models)"
            echo "  $0"
            echo ""
            echo "  # Train only Model 1 with 4 GPUs"
            echo "  $0 --model1_only --num_gpus 4"
            echo ""
            echo "  # Train only unified Model 2"
            echo "  $0 --model2_only --num_gpus 2"
            echo ""
            echo "  # Skip data prep (already prepared)"
            echo "  $0 --skip_data_prep --skip_units"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$MANIFEST_DIR" "$MODEL_DIR" "$LOG_DIR"

echo "=============================================="
echo "Speech Translation Training Pipeline"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Num GPUs: $NUM_GPUS"
echo ""
if [ "$TRAIN_MODEL1" = true ]; then
    echo "✓ Train Model 1: Text tasks (S2TT, T2TT, ASR)"
fi
if [ "$TRAIN_MODEL2" = true ]; then
    echo "✓ Train Model 2: Unified model (All 5 tasks)"
fi
echo "=============================================="
echo ""

# ============================================
# STEP 1: Data Preparation
# ============================================
if [ "$SKIP_DATA_PREP" = false ]; then
    echo "=========================================="
    echo "STEP 1: Data Preparation"
    echo "=========================================="
    
    # Find all metadata files
    METADATA_FILES=$(find "$DATA_DIR" -name "metadata*.json" -type f | sort)
    
    if [ -z "$METADATA_FILES" ]; then
        echo "Error: No metadata files found in $DATA_DIR"
        exit 1
    fi
    
    echo "Found metadata files:"
    echo "$METADATA_FILES"
    echo ""
    
    # Convert for Model 1 (Text tasks)
    if [ "$TRAIN_MODEL1" = true ]; then
        echo "Converting metadata for Model 1 (Text tasks)..."
        python3 src/training/convert_metadata.py \
            --input_files $METADATA_FILES \
            --output_dir "$MANIFEST_DIR/text" \
            --mode text \
            --split \
            --train_ratio 0.8 \
            --val_ratio 0.1
        
        echo ""
        echo "Model 1 manifests created in: $MANIFEST_DIR/text/"
        echo "  - train_manifest.json"
        echo "  - val_manifest.json"
        echo "  - test_manifest.json"
        echo ""
    fi
    
    # Convert for Model 2 (Unified model - needs units)
    if [ "$TRAIN_MODEL2" = true ]; then
        echo "Converting metadata for Model 2 (Unified model)..."
        python3 src/training/convert_metadata.py \
            --input_files $METADATA_FILES \
            --output_dir "$MANIFEST_DIR/unified" \
            --mode speech \
            --split \
            --train_ratio 0.8 \
            --val_ratio 0.1
        
        echo ""
        echo "Model 2 manifests created in: $MANIFEST_DIR/unified/"
    fi
else
    echo "Skipping data preparation (using existing manifests)"
fi

# ============================================
# STEP 2: Extract Units (for Model 2)
# ============================================
if [ "$SKIP_UNITS" = false ] && [ "$TRAIN_MODEL2" = true ]; then
    echo ""
    echo "=========================================="
    echo "STEP 2: Extract Units (for Model 2)"
    echo "=========================================="
    
    python3 src/training/extract_units.py \
        --input_manifests "$MANIFEST_DIR/unified"/*.json \
        --output_dir "$MANIFEST_DIR/unified_with_units" \
        --model_name xlsr2_1b_v2
    
    echo ""
    echo "Units extracted to: $MANIFEST_DIR/unified_with_units/"
else
    echo "Skipping unit extraction"
fi

# ============================================
# STEP 3: Train Model 1 (Text Tasks Only)
# ============================================
if [ "$TRAIN_MODEL1" = true ]; then
    echo ""
    echo "=========================================="
    echo "STEP 3: Train Model 1 (Text Tasks Only)"
    echo "=========================================="
    echo "Tasks: S2TT, T2TT, ASR"
    echo ""
    
    bash src/training/train_model1_text.sh \
        --train_dataset "$MANIFEST_DIR/text/train_manifest.json" \
        --eval_dataset "$MANIFEST_DIR/text/val_manifest.json" \
        --save_model_to "$MODEL_DIR/model1_text.pt" \
        --batch_size 8 \
        --learning_rate 1e-6 \
        --max_epochs 20 \
        --patience 5 \
        --num_gpus "$NUM_GPUS" \
        2>&1 | tee "$LOG_DIR/train_model1.log"
    
    echo ""
    echo "Model 1 training complete!"
    echo "Model saved to: $MODEL_DIR/model1_text.pt"
    echo "Log: $LOG_DIR/train_model1.log"
fi

# ============================================
# STEP 4: Train Model 2 (Unified Model)
# ============================================
if [ "$TRAIN_MODEL2" = true ]; then
    echo ""
    echo "=========================================="
    echo "STEP 4: Train Model 2 (Unified Model)"
    echo "=========================================="
    echo "Tasks: S2TT, T2TT, ASR + S2ST, T2ST"
    echo "Note: Model 2 = Model 1 + Speech tasks"
    echo ""
    
    bash src/training/train_model2_unified.sh \
        --train_dataset "$MANIFEST_DIR/unified_with_units/train_manifest_with_units.json" \
        --eval_dataset "$MANIFEST_DIR/unified_with_units/val_manifest_with_units.json" \
        --save_model_to "$MODEL_DIR/model2_unified.pt" \
        --batch_size 4 \
        --learning_rate 5e-7 \
        --max_epochs 20 \
        --patience 5 \
        --num_gpus "$NUM_GPUS" \
        2>&1 | tee "$LOG_DIR/train_model2.log"
    
    echo ""
    echo "Model 2 training complete!"
    echo "Model saved to: $MODEL_DIR/model2_unified.pt"
    echo "Log: $LOG_DIR/train_model2.log"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Training Pipeline Complete!"
echo "=============================================="
echo ""
echo "Models saved to: $MODEL_DIR/"
if [ "$TRAIN_MODEL1" = true ]; then
    echo ""
    echo "Model 1 (Text Tasks):"
    echo "  File: model1_text.pt"
    echo "  Tasks:"
    echo "    - S2TT: Speech-to-Text Translation (vie↔eng)"
    echo "    - T2TT: Text-to-Text Translation (vie↔eng)"
    echo "    - ASR:  Automatic Speech Recognition (vie, eng)"
fi
if [ "$TRAIN_MODEL2" = true ]; then
    echo ""
    echo "Model 2 (Unified Model):"
    echo "  File: model2_unified.pt"
    echo "  Tasks:"
    echo "    - S2TT: Speech-to-Text Translation (vie↔eng)"
    echo "    - T2TT: Text-to-Text Translation (vie↔eng)"
    echo "    - ASR:  Automatic Speech Recognition (vie, eng)"
    echo "    - S2ST: Speech-to-Speech Translation (vie→eng)"
    echo "    - T2ST: Text-to-Speech Translation (vie↔eng)"
    echo ""
    echo "  Note: Model 2 includes all tasks from Model 1 + Speech tasks"
fi
echo ""
echo "Logs saved to: $LOG_DIR/"
echo ""
echo "Next steps:"
echo "  1. Evaluate models: python src/training/evaluate.py --model_dir $MODEL_DIR"
echo "  2. Test inference: python src/training/inference.py --model $MODEL_DIR/model2_unified.pt"
echo "  3. Export models: python src/training/export.py --model_dir $MODEL_DIR"
echo ""
echo "Usage recommendation:"
if [ "$TRAIN_MODEL2" = true ]; then
    echo "  - Use Model 2 for production (supports all tasks)"
    echo "  - Use Model 1 if you only need text output (faster, smaller)"
else
    echo "  - Model 1 trained successfully"
    echo "  - Run with --model2_only to train unified model"
fi
