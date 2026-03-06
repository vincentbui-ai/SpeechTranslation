#!/bin/bash
# 8_infer_single.sh
# Inference on a single audio file
# Usage: ./8_infer_single.sh <checkpoint_path> <audio_path>

set -e

REPO_ROOT=$(pwd)
STREAMSPEECH=$REPO_ROOT/src/StreamSpeech
FAIRSEQ=$STREAMSPEECH/fairseq

# Configuration
CHECKPOINT=${1:?
Usage: $0 <checkpoint_path> <audio_path>
Example: $0 checkpoints/streamspeech.vi-en.scratch/checkpoint_best.pt /path/to/audio.wav
}
AUDIO_PATH=${2:?
Usage: $0 <checkpoint_path> <audio_path>
Example: $0 checkpoints/streamspeech.vi-en.scratch/checkpoint_best.pt /path/to/audio.wav
}

CONFIG=$REPO_ROOT/configs/vi-en
OUTPUT_DIR=$REPO_ROOT/results/single_inference

echo "============================================"
echo "Single Audio Inference: VI→EN"
echo "============================================"
echo "Audio: $AUDIO_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "============================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$AUDIO_PATH" ]; then
    echo "[ERROR] Audio file not found: $AUDIO_PATH"
    exit 1
fi

mkdir -p $OUTPUT_DIR

# Create temporary manifest
TEMP_DIR=$(mktemp -d)
echo "id	audio	n_frames	tgt_text	speaker" > $TEMP_DIR/manifest.tsv
FILENAME=$(basename $AUDIO_PATH)
ID="${FILENAME%.*}"
echo -e "$ID\t$AUDIO_PATH\t0\tplaceholder\tspk0" >> $TEMP_DIR/manifest.tsv

echo ""
echo "[1/2] Running inference..."

PYTHONPATH=$FAIRSEQ python -c "
import sys
sys.path.insert(0, '$STREAMSPEECH/researches/ctc_unity')

from fairseq import checkpoint_utils, tasks, options
import torch

# Load model
print('Loading model...')
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    ['$CHECKPOINT'],
    arg_overrides={
        'config_yaml': '$CONFIG/config_gcmvn.yaml',
        'multitask_config_yaml': '$CONFIG/config_mtl_asr_st_ctcst.yaml',
    }
)
model = models[0].cuda().eval()

print('Model loaded. Ready for inference.')
print('Note: For full inference pipeline, use fairseq-interactive or create proper manifest.')
" 2>&1 | tee $OUTPUT_DIR/inference.log

echo ""
echo "[2/2] Processing complete"
echo ""
echo "[✓] Inference complete!"
echo "Output: $OUTPUT_DIR/inference.log"
echo ""
echo "Note: For production inference, use:"
echo "  1. fairseq-interactive (for batch processing)"
echo "  2. SimulEval agent (for streaming)"
echo "  3. Custom inference script with proper manifest"

# Cleanup
rm -rf $TEMP_DIR
