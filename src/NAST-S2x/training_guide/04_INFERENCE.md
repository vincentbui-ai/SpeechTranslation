# 04 — Inference Guide

Run inference with trained NAST-S2X models.

---

## Overview

NAST-S2X supports:
- **Offline inference**: Full audio translation
- **Streaming inference**: Simultaneous translation with chunk sizes

---

## Offline Inference

### 1. Generate Acoustic Units

```bash
CHECKPOINT=checkpoints/nast-vi-en/stage3_nmla/checkpoint_best.pt
DATA_ROOT=data/nast-vi-en/fairseq
OUTPUT_DIR=results/offline/units
NAST_DIR=src/NAST-S2x/nast

fairseq-generate ${DATA_ROOT} \
    --config-yaml config.yaml \
    --gen-subset test \
    --user-dir ${NAST_DIR} \
    --task nat_speech_to_unit_ctc_modified \
    --path ${CHECKPOINT} \
    --results-path ${OUTPUT_DIR} \
    --max-tokens 40000 \
    --beam 1 \
    --iter-decode-max-iter 0 \
    --iter-decode-with-beam 1 \
    --skip-invalid-size-inputs-valid-test
```

### 2. Generate Waveform (Vocoder)

```bash
python src/NAST-S2x/nast/scripts/generate_waveform.py \
    --in-code-file ${OUTPUT_DIR}/generate-test.txt \
    --vocoder-checkpoint checkpoints/vocoder/g_00500000 \
    --vocoder-cfg checkpoints/vocoder/config.json \
    --out-wav-dir results/offline/wav \
    --dur-prediction
```

### 3. Evaluation

```bash
# ASR-BLEU evaluation
python -m fairseq.examples.speech_to_speech.asr_bleu \
    --audio-dirpath results/offline/wav \
    --reference-path data/nast-vi-en/fairseq/test.tsv \
    --lang en
```

---

## Streaming (Simultaneous) Inference

Uses custom SimulEval for latency evaluation.

### 1. Prepare SimulEval Manifest

```bash
python training_guide/scripts/prepare_simuleval.py \
    --tsv-path data/nast-vi-en/fairseq/test.tsv \
    --audio-dir data/nast-vi-en/audio \
    --output-dir results/simuleval
```

### 2. Run Streaming Evaluation

```bash
CHECKPOINT=checkpoints/nast-vi-en/chunk_320ms.pt
CHUNK_SIZE=320

simuleval \
    --agent src/NAST-S2x/nast/agents/nast_s2x_agent.py \
    --source results/simuleval/test.wav_list \
    --target results/simuleval/test.en \
    --output results/simuleval/output \
    --model-path ${CHECKPOINT} \
    --config-yaml data/nast-vi-en/fairseq/config.yaml \
    --chunk-size ${CHUNK_SIZE} \
    --beam 1
```

### 3. View Results

```bash
cat results/simuleval/output/scores.yaml
```

**Metrics:**
- **BLEU**: Translation quality
- **AL (Average Lagging)**: Latency in milliseconds
- **DAL (Differentiable Average Lagging)**: Smoothed latency

---

## Single Audio Inference Script

For translating a single audio file:

```bash
python training_guide/scripts/infer_single.py \
    --audio /path/to/audio.wav \
    --checkpoint checkpoints/nast-vi-en/offline.pt \
    --src-lang vie \
    --tgt-lang eng \
    --output output.wav \
    --chunk-size 320
```

---

## Batch Inference

For large-scale inference:

```bash
python training_guide/scripts/infer_batch.py \
    --input data/nast-vi-en/fairseq/test.tsv \
    --checkpoint checkpoints/nast-vi-en/offline.pt \
    --output-dir results/batch \
    --batch-size 32 \
    --num-gpus 4
```

---

## Inference with Different Chunk Sizes

### 320ms (Ultra-low latency)

```bash
CHUNK_SIZE=320
# ~19 BLEU, latency < 1s
```

### 1280ms (Balanced)

```bash
CHUNK_SIZE=1280
# ~20 BLEU, latency ~2-3s
```

### 2560ms (High quality)

```bash
CHUNK_SIZE=2560
# ~25 BLEU, latency ~4-5s
```

### Offline

```bash
# No chunking, full audio
# Best quality ~26 BLEU
```

---

## Inference Checkpoints

Available checkpoints from HuggingFace:

```bash
# Download all chunk sizes
python << 'EOF'
from huggingface_hub import hf_hub_download

checkpoints = [
    'chunk_320ms.pt',
    'chunk_1280ms.pt', 
    'chunk_2560ms.pt',
    'Offline.pt'
]

for ckpt in checkpoints:
    hf_hub_download(
        repo_id='ICTNLP/NAST-S2X',
        filename=ckpt,
        local_dir='checkpoints/nast-vi-en/'
    )
EOF
```

---

## Evaluation Metrics

### ASR-BLEU

```bash
# Automatic evaluation
python training_guide/scripts/eval_asr_bleu.py \
    --hyp-dir results/offline/wav \
    --ref-file data/nast-vi-en/test.en \
    --output results/metrics.json
```

### Manual Evaluation

```bash
# Compare translations
python training_guide/scripts/compare_translations.py \
    --ref data/nast-vi-en/test.en \
    --hyp results/offline/generate-test.txt \
    --output comparison.html
```

---

## Troubleshooting

### Issue: CUDA out of memory

```bash
# Reduce batch size
fairseq-generate ... --max-tokens 20000
```

### Issue: Audio quality issues

```bash
# Try different vocoder checkpoint
python generate_waveform.py ... --vocoder-checkpoint checkpoints/vocoder/g_00400000
```

### Issue: High latency in streaming

```bash
# Reduce chunk size or use adaptive policy
simuleval ... --chunk-size 160 --adaptive
```

---

## Complete Pipeline Example

```bash
#!/bin/bash
# Full inference pipeline

CHECKPOINT=checkpoints/nast-vi-en/chunk_320ms.pt
DATA_ROOT=data/nast-vi-en/fairseq
OUTPUT=results/inference

# 1. Generate units
echo "Step 1: Generating units..."
fairseq-generate ${DATA_ROOT} \
    --config-yaml config.yaml \
    --gen-subset test \
    --user-dir src/NAST-S2x/nast \
    --path ${CHECKPOINT} \
    --results-path ${OUTPUT}/units \
    --max-tokens 40000

# 2. Generate audio
echo "Step 2: Generating audio..."
python src/NAST-S2x/nast/scripts/generate_waveform.py \
    --in-code-file ${OUTPUT}/units/generate-test.txt \
    --vocoder-checkpoint checkpoints/vocoder/g_00500000 \
    --vocoder-cfg checkpoints/vocoder/config.json \
    --out-wav-dir ${OUTPUT}/wav

# 3. Evaluate
echo "Step 3: Evaluating..."
python training_guide/scripts/eval_asr_bleu.py \
    --hyp-dir ${OUTPUT}/wav \
    --ref-file ${DATA_ROOT}/test.en

echo "Done! Results in ${OUTPUT}/"
```

---

## Next Steps

- For model fine-tuning, see [03_TRAINING.md](03_TRAINING.md)
- For dataset preparation, see [02_DATA_PREP.md](02_DATA_PREP.md)
- For troubleshooting, check the [NAST-S2X GitHub Issues](https://github.com/ictnlp/NAST-S2x/issues)
