#!/bin/bash
# 9_eval_all.sh
# Run all evaluations on a checkpoint
# Usage: ./9_eval_all.sh <checkpoint_path> [split]

set -e

CHECKPOINT=${1:?
Usage: $0 <checkpoint_path> [split]
Example: $0 checkpoints/streamspeech.vi-en.scratch/checkpoint_best.pt dev
}
SPLIT=${2:-dev}

echo "============================================"
echo "Complete Evaluation Suite"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "============================================"

# 1. Offline evaluation
echo ""
echo ">>> [1/3] Offline Evaluation"
./6_eval_offline.sh $CHECKPOINT $SPLIT

# 2. Simultaneous evaluation with different chunk sizes
echo ""
echo ">>> [2/3] Simultaneous Evaluation"
for chunk in 320 640 960; do
    echo "  Testing chunk size: ${chunk}ms"
    ./7_eval_simultaneous.sh $CHECKPOINT $chunk $SPLIT
done

echo ""
echo "============================================"
echo "[✓] All evaluations complete!"
echo "Check results/ directory for outputs"
echo "============================================"
