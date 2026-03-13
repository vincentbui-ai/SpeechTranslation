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
#
# Examples:
#   # Use pre-split files (trainset_*.json, valset_*.json)
#   ./run_training_pipeline.sh \\
#       --train_files "datasets/trainset_*.json" \\
#       --val_files "datasets/valset_*.json" \\
#       --test_files "datasets/testset_*.json"
#
#   # Use explicit files
#   ./run_training_pipeline.sh \\
#       --train_files train_vie.json train_eng.json \\
#       --val_files val_vie.json val_eng.json
#
#   # Legacy mode: split single file
#   ./run_training_pipeline.sh --input_files datasets/metadata.json

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

# File lists (can be multiple)
TRAIN_FILES=""
VAL_FILES=""
TEST_FILES=""
INPUT_FILES=""

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
        --train_files)
            # Collect all train files until next flag
            shift
            TRAIN_FILES=""
            while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
                TRAIN_FILES="$TRAIN_FILES $1"
                shift
            done
            ;;
        --val_files)
            # Collect all val files until next flag
            shift
            VAL_FILES=""
            while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
                VAL_FILES="$VAL_FILES $1"
                shift
            done
            ;;
        --test_files)
            # Collect all test files until next flag
            shift
            TEST_FILES=""
            while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
                TEST_FILES="$TEST_FILES $1"
                shift
            done
            ;;
        --input_files)
            # Legacy mode: collect all input files
            shift
            INPUT_FILES=""
            while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
                INPUT_FILES="$INPUT_FILES $1"
                shift
            done
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
            echo "  --train_files FILES      Training files (supports wildcards, multiple files)"
            echo "  --val_files FILES        Validation files (supports wildcards, multiple files)"
            echo "  --test_files FILES       Test files (supports wildcards, multiple files)"
            echo "  --input_files FILES      Legacy: input files to split (use --train_files instead)"
            echo "  --skip_data_prep         Skip data preparation step"
            echo "  --skip_units             Skip unit extraction (Model 2 needs this)"
            echo "  --model1_only            Train only Model 1 (Text tasks)"
            echo "  --model2_only            Train only Model 2 (Unified model)"
            echo "  --num_gpus N             Number of GPUs (default: 1)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo ""
            echo "  # Use pre-split files with wildcards"
            echo "  $0 --train_files 'datasets/trainset_*.json' \\"
            echo "     --val_files 'datasets/valset_*.json' \\"
            echo "     --test_files 'datasets/testset_*.json'"
            echo ""
            echo "  # Use explicit file lists"
            echo "  $0 --train_files train_vie.json train_eng.json \\"
            echo "     --val_files val_vie.json val_eng.json \\"
            echo "     --test_files test_vie.json test_eng.json"
            echo ""
            echo "  # Legacy: split single file"
            echo "  $0 --input_files datasets/metadata.json"
            echo ""
            echo "  # Train only Model 1 with 4 GPUs"
            echo "  $0 --model1_only --num_gpus 4 \\"
            echo "     --train_files 'trainset_*.json' \\"
            echo "     --val_files 'valset_*.json'"
            echo ""
            echo "  # Train only unified Model 2"
            echo "  $0 --model2_only --num_gpus 2 \\"
            echo "     --train_files 'trainset_*.json' \\"
            echo "     --val_files 'valset_*.json'"
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

# Determine mode
if [ -n "$TRAIN_FILES" ] || [ -n "$VAL_FILES" ] || [ -n "$TEST_FILES" ]; then
    echo "Mode: Pre-split files"
    [ -n "$TRAIN_FILES" ] && echo "  Train files: $TRAIN_FILES"
    [ -n "$VAL_FILES" ] && echo "  Val files: $VAL_FILES"
    [ -n "$TEST_FILES" ] && echo "  Test files: $TEST_FILES"
    USE_SPLIT_FILES=true
elif [ -n "$INPUT_FILES" ]; then
    echo "Mode: Legacy (split input files)"
    echo "  Input files: $INPUT_FILES"
    USE_SPLIT_FILES=false
else
    # Auto-detect files
    echo "Mode: Auto-detect files"
    
    # Look for trainset_* files
    TRAIN_FILES=$(find "$DATA_DIR" -name "trainset_*.json" -type f 2>/dev/null | sort | tr '\n' ' ')
    VAL_FILES=$(find "$DATA_DIR" -name "valset_*.json" -type f 2>/dev/null | sort | tr '\n' ' ')
    TEST_FILES=$(find "$DATA_DIR" -name "testset_*.json" -type f 2>/dev/null | sort | tr '\n' ' ')
    
    if [ -n "$TRAIN_FILES" ] || [ -n "$VAL_FILES" ]; then
        echo "  Auto-detected train files: $TRAIN_FILES"
        echo "  Auto-detected val files: $VAL_FILES"
        echo "  Auto-detected test files: $TEST_FILES"
        USE_SPLIT_FILES=true
    else
        # Fall back to metadata*.json
        INPUT_FILES=$(find "$DATA_DIR" -name "metadata*.json" -type f 2>/dev/null | sort | tr '\n' ' ')
        if [ -n "$INPUT_FILES" ]; then
            echo "  Auto-detected input files: $INPUT_FILES"
            USE_SPLIT_FILES=false
        else
            echo "Error: No data files found in $DATA_DIR"
            echo "Expected: trainset_*.json, valset_*.json, testset_*.json or metadata*.json"
            exit 1
        fi
    fi
fi

if [ "$TRAIN_MODEL1" = true ]; then
    echo ""
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
    
    # Build conversion command
    if [ "$USE_SPLIT_FILES" = true ]; then
        # Use pre-split files
        CONVERT_CMD="python3 src/training/convert_metadata.py"
        
        [ -n "$TRAIN_FILES" ] && CONVERT_CMD="$CONVERT_CMD --train_files $TRAIN_FILES"
        [ -n "$VAL_FILES" ] && CONVERT_CMD="$CONVERT_CMD --val_files $VAL_FILES"
        [ -n "$TEST_FILES" ] && CONVERT_CMD="$CONVERT_CMD --test_files $TEST_FILES"
        
        CONVERT_CMD="$CONVERT_CMD --output_dir $MANIFEST_DIR/text --mode text"
        
        echo "Running: $CONVERT_CMD"
        eval $CONVERT_CMD
        
        echo ""
        echo "Manifests created in: $MANIFEST_DIR/text/"
        echo "  - train_manifest.json"
        echo "  - val_manifest.json"
        if [ -n "$TEST_FILES" ]; then
            echo "  - test_manifest.json"
        fi
        echo ""
        
        # For Model 2 (Unified), also create speech manifests
        if [ "$TRAIN_MODEL2" = true ]; then
            echo "Converting for Model 2 (Unified model - speech mode)..."
            CONVERT_CMD2="python3 src/training/convert_metadata.py"
            
            [ -n "$TRAIN_FILES" ] && CONVERT_CMD2="$CONVERT_CMD2 --train_files $TRAIN_FILES"
            [ -n "$VAL_FILES" ] && CONVERT_CMD2="$CONVERT_CMD2 --val_files $VAL_FILES"
            [ -n "$TEST_FILES" ] && CONVERT_CMD2="$CONVERT_CMD2 --test_files $TEST_FILES"
            
            CONVERT_CMD2="$CONVERT_CMD2 --output_dir $MANIFEST_DIR/unified --mode speech"
            
            echo "Running: $CONVERT_CMD2"
            eval $CONVERT_CMD2
            
            echo ""
            echo "Model 2 manifests created in: $MANIFEST_DIR/unified/"
        fi
    else
        # Legacy mode: split input files
        if [ -z "$INPUT_FILES" ]; then
            echo "Error: No input files specified!"
            exit 1
        fi
        
        echo "Found input files:"
        echo "$INPUT_FILES"
        echo ""
        
        # Convert for Model 1 (Text tasks)
        if [ "$TRAIN_MODEL1" = true ]; then
            echo "Converting metadata for Model 1 (Text tasks)..."
            python3 src/training/convert_metadata.py \
                --input_files $INPUT_FILES \
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
                --input_files $INPUT_FILES \
                --output_dir "$MANIFEST_DIR/unified" \
                --mode speech \
                --split \
                --train_ratio 0.8 \
                --val_ratio 0.1
            
            echo ""
            echo "Model 2 manifests created in: $MANIFEST_DIR/unified/"
        fi
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
