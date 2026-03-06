#!/bin/bash
# 7_eval_simultaneous.sh
# Simultaneous evaluation with latency metrics using SimulEval
# Usage: ./7_eval_simultaneous.sh [checkpoint_path] [chunk_size_ms] [split]

set -e

REPO_ROOT=$(pwd)
STREAMSPEECH=$REPO_ROOT/src/StreamSpeech
FAIRSEQ=$STREAMSPEECH/fairseq

# Configuration
CHECKPOINT_DIR=$REPO_ROOT/checkpoints/streamspeech.vi-en.scratch
CHECKPOINT=${1:-$CHECKPOINT_DIR/checkpoint_best.pt}

# Simultaneous parameters
CHUNK_SIZE=${2:-320}  # ms (320, 480, 640, 960, etc.)
SPLIT=${3:-dev}

echo "============================================"
echo "Simultaneous Evaluation: VI→EN"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Chunk size: ${CHUNK_SIZE}ms"
echo "Split: $SPLIT"
echo "============================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Prepare SimulEval data format if not exists
SIMUL_DIR=$REPO_ROOT/data/vi-en/simuleval/$SPLIT
if [ ! -d "$SIMUL_DIR" ]; then
    echo "[INFO] Creating SimulEval data format..."
    mkdir -p $SIMUL_DIR
    
    # Create wav_list.txt and target.txt
    TSV_FILE=$REPO_ROOT/data/vi-en/${SPLIT}.tsv
    
    # Extract audio paths and references
    tail -n +2 $TSV_FILE | cut -f2 > $SIMUL_DIR/wav_list.txt
    tail -n +2 $TSV_FILE | cut -f4 > $SIMUL_DIR/target.txt
    
    echo "[✓] SimulEval data prepared: $SIMUL_DIR"
fi

# Output directory
RESULTS_DIR=$REPO_ROOT/results/simultaneous_chunk${CHUNK_SIZE}_${SPLIT}
mkdir -p $RESULTS_DIR

echo ""
echo "[1/2] Running SimulEval..."
PYTHONPATH=$FAIRSEQ simuleval \
    --data-bin $REPO_ROOT/configs/vi-en \
    --user-dir $STREAMSPEECH/researches/ctc_unity \
    --agent-dir $STREAMSPEECH/agent \
    --source $SIMUL_DIR/wav_list.txt \
    --target $SIMUL_DIR/target.txt \
    --model-path $CHECKPOINT \
    --config-yaml config_gcmvn.yaml \
    --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $STREAMSPEECH/agent/speech_to_text.s2tt.streamspeech.agent.py \
    --output $RESULTS_DIR \
    --source-segment-size $CHUNK_SIZE \
    --quality-metrics BLEU \
    --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks RTF \
    --device gpu \
    --computation-aware \
    2>&1 | tee $RESULTS_DIR/simuleval.log

echo ""
echo "[2/2] Extracting results..."

# Results are automatically saved by SimulEval
if [ -f "$RESULTS_DIR/scores.tsv" ]; then
    echo ""
    echo "============================================"
    echo "Simultaneous Results (Chunk: ${CHUNK_SIZE}ms)"
    echo "============================================"
    cat $RESULTS_DIR/scores.tsv
    
    echo ""
    echo "Detailed metrics: $RESULTS_DIR/metrics.tsv"
fi

echo ""
echo "[✓] Simultaneous evaluation complete!"
echo "Results saved to: $RESULTS_DIR/"
