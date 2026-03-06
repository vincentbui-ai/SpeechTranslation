# 03 — Training Guide

Train NAST-S2X model for Vietnamese-English speech translation.

---

## Training Pipeline

NAST-S2X training consists of 3 stages:

| Stage | Purpose | Architecture | Duration |
|-------|---------|--------------|----------|
| **Stage 1** | Encoder Pretraining | ASR (Speech-to-Text) | 1-2 days |
| **Stage 2** | CTC Pretraining | Speech-to-Unit | 2-3 days |
| **Stage 3** | NMLA Training | Fine-tuning with alignment | 3-5 days |

---

## Stage 1: Encoder Pretraining (ASR)

Train encoder on Automatic Speech Recognition (ASR) task for better initialization.

### Configuration

```bash
CHUNK_SIZE=32  # 320ms chunks for streaming
DATA_ROOT=data/nast-vi-en/fairseq
SAVE_DIR=checkpoints/nast-vi-en/stage1_encoder
NAST_DIR=src/NAST-S2x/nast
```

### Training Script

```bash
fairseq-train ${DATA_ROOT} \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --user-dir ${NAST_DIR} \
    --fp16 \
    --task nat_speech_to_text_ctc_modified \
    --arch nonautoregressive_streaming_speech_transformer_segment_to_segment \
    --src-embedding-copy \
    --main-context ${CHUNK_SIZE} \
    --right-context ${CHUNK_SIZE} \
    --criterion nat_loss_ngram_glat_asr \
    --glat-p 0.5:0.3@50k \
    --label-smoothing 0.01 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --clip-norm 10.0 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 0.001 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' \
    --warmup-updates 10000 \
    --max-update 100000 \
    --max-tokens 40000 \
    --update-freq 4 \
    --save-dir ${SAVE_DIR} \
    --save-interval-updates 2000 \
    --keep-interval-updates 10 \
    --validate-interval-updates 2000 \
    --best-checkpoint-metric wer \
    --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 5 \
    --num-workers 8
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--arch` | nonautoregressive_streaming_speech_transformer_segment_to_segment | NAST architecture |
| `--task` | nat_speech_to_text_ctc_modified | ASR task |
| `--criterion` | nat_loss_ngram_glat_asr | ASR loss with GLAT |
| `--main-context` | 32 | Left context (320ms) |
| `--right-context` | 32 | Right context (320ms) |

### Expected Results

- WER should decrease to ~15-20% on validation
- Training time: ~24-48 hours on 4x A100 GPUs

---

## Stage 2: CTC Pretraining (Speech-to-Unit)

Train on acoustic unit prediction.

### Configuration

```bash
CHUNK_SIZE=32
DATA_ROOT=data/nast-vi-en/fairseq
SAVE_DIR=checkpoints/nast-vi-en/stage2_ctc
NAST_DIR=src/NAST-S2x/nast
ENCODER_PT=checkpoints/nast-vi-en/stage1_encoder/checkpoint_best.pt
```

### Training Script

```bash
fairseq-train ${DATA_ROOT} \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --user-dir ${NAST_DIR} \
    --fp16 \
    --load-pretrained-encoder-from ${ENCODER_PT} \
    --task nat_speech_to_unit_ctc_modified \
    --arch nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment \
    --src-embedding-copy \
    --target-is-code \
    --target-code-size 1000 \
    --src-upsample-ratio 1 \
    --hidden-upsample-ratio 6 \
    --main-context ${CHUNK_SIZE} \
    --right-context ${CHUNK_SIZE} \
    --unit-size 2 \
    --share-decoder-input-output-embed \
    --rand-pos-encoder 300 \
    --decoder-learned-pos \
    --activation-dropout 0.1 \
    --attention-dropout 0.1 \
    --encoder-max-relative-position 32 \
    --apply-bert-init \
    --noise full_mask \
    --criterion nat_loss_ngram_glat_s2u \
    --glat-p 0.5:0.3@50k \
    --glat-p-unit 0.3:0.1@50k \
    --label-smoothing-unit 0.01 \
    --label-smoothing 0.01 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --clip-norm 10.0 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 0.001 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr '1e-07' \
    --warmup-updates 10000 \
    --stop-min-lr '1e-09' \
    --max-update 150000 \
    --max-tokens 40000 \
    --update-freq 4 \
    --grouped-shuffling \
    --save-dir ${SAVE_DIR} \
    --save-interval-updates 2000 \
    --keep-interval-updates 10 \
    --save-interval 1000 \
    --keep-last-epochs 10 \
    --validate-interval 1000 \
    --validate-interval-updates 2000 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu_unit \
    --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 5 \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 8
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--target-code-size` | 1000 | Number of acoustic units |
| `--unit-size` | 2 | Units per time step |
| `--hidden-upsample-ratio` | 6 | Encoder upsampling ratio |

### Expected Results

- BLEU (unit): ~25-30 on validation
- Training time: ~48-72 hours on 4x A100 GPUs

---

## Stage 3: NMLA Training

Fine-tune with Non-Monotonic Latent Alignment.

### Configuration

```bash
CHUNK_SIZE=32
DATA_ROOT=data/nast-vi-en/fairseq
SAVE_DIR=checkpoints/nast-vi-en/stage3_nmla
NAST_DIR=src/NAST-S2x/nast
CTC_PT=checkpoints/nast-vi-en/stage2_ctc/checkpoint_best.pt
```

### Training Script

Similar to Stage 2, but load from CTC checkpoint:

```bash
fairseq-train ${DATA_ROOT} \
    --config-yaml config.yaml \
    --train-subset train \
    --valid-subset dev \
    --user-dir ${NAST_DIR} \
    --fp16 \
    --load-pretrained-encoder-from ${CTC_PT} \
    --load-pretrained-decoder-from ${CTC_PT} \
    --task nat_speech_to_unit_ctc_modified \
    --arch nonautoregressive_streaming_speech_to_unit_transformer_segment_to_segment \
    ...
    --criterion nat_loss_ngram_glat_s2u \
    --glat-p 0.5:0.5@30k \
    --max-update 100000 \
    ...
```

---

## Training Different Chunk Sizes

Train models for different latency requirements:

### 320ms (Ultra-low latency)

```bash
CHUNK_SIZE=32
# Use scripts above with CHUNK_SIZE=32
```

### 1280ms (Balanced)

```bash
CHUNK_SIZE=128
# Use scripts above with CHUNK_SIZE=128
```

### 2560ms (High quality)

```bash
CHUNK_SIZE=256
# Use scripts above with CHUNK_SIZE=256
```

### Offline (Best quality)

```bash
CHUNK_SIZE=999999
# Or use --main-context -1 for full context
```

---

## GPU Memory Configuration

Adjust based on your GPU:

| GPU VRAM | MAX_TOKENS | UPDATE_FREQ |
|----------|------------|-------------|
| 40 GB (A100) | 40000 | 4 |
| 24 GB (3090/4090) | 24000 | 8 |
| 16 GB (V100) | 16000 | 12 |

---

## Monitoring Training

```bash
# View logs
tail -f checkpoints/nast-vi-en/stage2_ctc/train.log

# Tensorboard
tensorboard --logdir checkpoints/nast-vi-en/

# Check latest checkpoint
ls -lh checkpoints/nast-vi-en/stage2_ctc/checkpoint*.pt
```

---

## Checkpoint Selection

Best checkpoints are saved based on validation metrics:

```
checkpoints/nast-vi-en/stage2_ctc/
├── checkpoint_best.pt      # Best BLEU
├── checkpoint_last.pt      # Latest
├── checkpoint_100000.pt    # Every 1000 steps
└── checkpoint_120000.pt
```

---

## Next Steps

→ Proceed to [04_INFERENCE.md](04_INFERENCE.md) for inference.
