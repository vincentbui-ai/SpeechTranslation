#!/bin/bash
# 6_eval_offline.sh
# Offline evaluation for VI→EN Speech-to-Text Translation
# Usage: ./6_eval_offline.sh [checkpoint_path] [split] [beam_size]

set -e

REPO_ROOT=$(pwd)
STREAMSPEECH=$REPO_ROOT/src/StreamSpeech
FAIRSEQ=$STREAMSPEECH/fairseq

# Configuration
DATA=$REPO_ROOT/data/vi-en
CONFIG=$REPO_ROOT/configs/vi-en
CHECKPOINT_DIR=$REPO_ROOT/checkpoints/streamspeech.vi-en.scratch

# Select checkpoint (best or specific)
CHECKPOINT=${1:-$CHECKPOINT_DIR/checkpoint_best.pt}
SPLIT=${2:-dev}  # dev or test
BEAM_SIZE=${3:-5}

echo "============================================"
echo "Offline Evaluation: VI→EN"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "Beam size: $BEAM_SIZE"
echo "============================================"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh $CHECKPOINT_DIR/*.pt 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Output directory
RESULTS_DIR=$REPO_ROOT/results/offline_${SPLIT}_$(basename $CHECKPOINT .pt)
mkdir -p $RESULTS_DIR

echo ""
echo "[1/3] Running inference with fairseq-generate..."
PYTHONPATH=$FAIRSEQ fairseq-generate $DATA \
    --user-dir $STREAMSPEECH/researches/ctc_unity \
    --config-yaml $CONFIG/config_gcmvn.yaml \
    --multitask-config-yaml $CONFIG/config_mtl_asr_st_ctcst.yaml \
    --task speech_to_speech_ctc \
    --target-is-code --target-code-size 1000 --vocoder code_hifigan \
    --path $CHECKPOINT \
    --gen-subset $SPLIT \
    --beam-mt $BEAM_SIZE \
    --beam 1 \
    --max-len-a 1 \
    --max-tokens 10000 \
    --required-batch-size-multiple 1 \
    --results-path $RESULTS_DIR \
    --fp16 \
    2>&1 | tee $RESULTS_DIR/generate.log

echo ""
echo "[2/3] Extracting predictions..."

# Extract ASR (source text) predictions
grep '^A-' $RESULTS_DIR/generate.log | sort -t'-' -k2,2n | cut -f2 > $RESULTS_DIR/asr_predictions.txt
echo "  ASR predictions: $(wc -l < $RESULTS_DIR/asr_predictions.txt) lines"

# Extract ST (target text) predictions  
grep '^D-' $RESULTS_DIR/generate.log | sort -t'-' -k2,2n | cut -f2 > $RESULTS_DIR/st_predictions.txt
echo "  ST predictions: $(wc -l < $RESULTS_DIR/st_predictions.txt) lines"

# Get reference files
REF_SRC=$DATA/${SPLIT}_asr.tsv  # Source text (Vietnamese)
REF_TGT=$DATA/${SPLIT}.tsv      # Target text (English)

# Extract references
tail -n +2 $REF_SRC | cut -f4 > $RESULTS_DIR/asr_references.txt
tail -n +2 $REF_TGT | cut -f4 > $RESULTS_DIR/st_references.txt

echo ""
echo "[3/3] Computing metrics..."

RESULTS_FILE=$RESULTS_DIR/metrics.txt
echo "Evaluation Results" > $RESULTS_FILE
echo "==================" >> $RESULTS_FILE
echo "Checkpoint: $CHECKPOINT" >> $RESULTS_FILE
echo "Split: $SPLIT" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# ASR Metrics (Vietnamese)
echo "### ASR Results (VI Speech → VI Text) ###" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# BLEU for ASR
echo "BLEU (sacrebleu):" >> $RESULTS_FILE
sacrebleu $RESULTS_DIR/asr_references.txt -i $RESULTS_DIR/asr_predictions.txt -w 3 >> $RESULTS_FILE 2>&1 || echo "  (sacrebleu not installed)" >> $RESULTS_FILE

echo "" >> $RESULTS_FILE
echo "### ST Results (VI Speech → EN Text) ###" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# BLEU for ST
echo "BLEU (sacrebleu):" >> $RESULTS_FILE
sacrebleu $RESULTS_DIR/st_references.txt -i $RESULTS_DIR/st_predictions.txt -w 3 >> $RESULTS_FILE 2>&1 || echo "  (sacrebleu not installed)" >> $RESULTS_FILE

echo "" >> $RESULTS_FILE
echo "==================" >> $RESULTS_FILE

# Display results
echo ""
echo "============================================"
echo "Results Summary"
echo "============================================"
cat $RESULTS_FILE

echo ""
echo "[✓] Evaluation complete!"
echo "Results saved to: $RESULTS_DIR/"
echo "  - metrics.txt"
echo "  - asr_predictions.txt"
echo "  - st_predictions.txt"
echo "  - generate.log"
