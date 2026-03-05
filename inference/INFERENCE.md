# Inference — Speech-to-Text Translation

Three inference scripts are provided, one for each model library in `src/`.  
All scripts accept a **16 kHz mono WAV or FLAC** audio file and print the translated text.

---

## Prerequisites

Make sure you are inside the project's conda environment before running any script.

```bash
conda activate translation
cd /path/to/SpeechTranslation   # repo root
```

---

## 1. StreamSpeech

**Library:** `src/StreamSpeech` — fairseq CTC-Unity Conformer  
**Script:** `inference/streamspeech_infer.py`

### Install dependencies

```bash
cd src/StreamSpeech
pip install -e fairseq/
pip install -e SimulEval/
pip install sentencepiece torchaudio soundfile
cd ../..
```

### Required files

| Argument | Description |
|---|---|
| `--audio` | Input audio file (WAV / FLAC, 16 kHz) |
| `--checkpoint` | Trained checkpoint `.pt` |
| `--data-bin` | fairseq data-bin directory (e.g. `data/vi-en`) |
| `--config-yaml` | Config YAML inside `data-bin` (optional but recommended) |
| `--spm-model` | Target SentencePiece model (e.g. `configs/vi-en/tgt_unigram6000/spm_unigram_en.model`) |

### Run — VI → EN

```bash
python inference/streamspeech_infer.py \
    --audio       path/to/audio.wav \
    --checkpoint  checkpoints/streamspeech.offline-s2tt.vi-en/checkpoint_best.pt \
    --data-bin    data/vi-en \
    --config-yaml configs/vi-en/config_mtl_asr_st_ctcst.yaml \
    --spm-model   configs/vi-en/tgt_unigram6000/spm_unigram_en.model
```

### Run — EN → VI

```bash
python inference/streamspeech_infer.py \
    --audio       path/to/audio.wav \
    --checkpoint  checkpoints/streamspeech.offline-s2tt.en-vi/checkpoint_best.pt \
    --data-bin    data/en-vi \
    --config-yaml configs/en-vi/config_mtl_asr_st_ctcst.yaml \
    --spm-model   configs/en-vi/tgt_unigram6000/spm_unigram_vi.model
```

### Expected output

```
Translation: The two stopped just inside the door.
```

---

## 2. NAST-S2x

**Library:** `src/NAST-S2x` — Non-Autoregressive Streaming Transformer  
**Script:** `inference/nast_infer.py`

> The agent was designed for SimulEval streaming evaluation.  
> This script wraps it in an offline pass by feeding the full audio as one finished segment.

### Install dependencies

```bash
cd src/StreamSpeech
pip install -e fairseq/
pip install -e SimulEval/
pip install sentencepiece torchaudio
cd ../..
```

### Required files

| Argument | Description |
|---|---|
| `--audio` | Input audio file (WAV / FLAC, 16 kHz) |
| `--model-path` | Trained NAST checkpoint `.pt` |
| `--data-bin` | fairseq data-bin directory |
| `--config-yaml` | Config YAML filename inside `data-bin` (optional) |
| `--spm-model` | Target SentencePiece model (optional) |

### Run — VI → EN

```bash
python inference/nast_infer.py \
    --audio       path/to/audio.wav \
    --model-path  checkpoints/nast.vi-en.pt \
    --data-bin    data/vi-en \
    --config-yaml config_mtl_asr_st_ctcst.yaml \
    --spm-model   configs/vi-en/tgt_unigram6000/spm_unigram_en.model
```

### Run — EN → VI

```bash
python inference/nast_infer.py \
    --audio       path/to/audio.wav \
    --model-path  checkpoints/nast.en-vi.pt \
    --data-bin    data/en-vi \
    --config-yaml config_mtl_asr_st_ctcst.yaml \
    --spm-model   configs/en-vi/tgt_unigram6000/spm_unigram_vi.model
```

### Expected output

```
Translation: I will share a lot of other information as well as other clips.
```

---

## 3. SeamlessM4T

**Library:** `src/seamless_communication` — Meta SeamlessM4T v2  
**Script:** `inference/seamless_infer.py`

> The model weights are downloaded automatically from Hugging Face on first run (~10 GB).  
> Set `HF_HOME` to a directory with enough disk space if needed.

### Install dependencies

```bash
pip install seamless-communication fairseq2 torchaudio
```

### Language codes

Use **ISO 639-3** codes:

| Language | Code |
|---|---|
| Vietnamese | `vie` |
| English | `eng` |

### Required arguments

| Argument | Description |
|---|---|
| `--audio` | Input audio file (WAV / FLAC, 16 kHz) |
| `--src-lang` | Source language code |
| `--tgt-lang` | Target language code |
| `--model-name` | Model variant (default: `seamlessM4T_v2_large`) |

### Run — VI → EN

```bash
python inference/seamless_infer.py \
    --audio    path/to/audio.wav \
    --src-lang vie \
    --tgt-lang eng
```

### Run — EN → VI

```bash
python inference/seamless_infer.py \
    --audio    path/to/audio.wav \
    --src-lang eng \
    --tgt-lang vie
```

### Run with a smaller model (less VRAM)

```bash
python inference/seamless_infer.py \
    --audio      path/to/audio.wav \
    --src-lang   vie \
    --tgt-lang   eng \
    --model-name seamlessM4T_medium
```

### Expected output

```
Translation: The two stopped just inside the door.
```

---

## GPU / CPU notes

All three scripts auto-detect CUDA. To force CPU:

```bash
python inference/<script>.py ... --device cpu
```

Recommended VRAM by model:

| Model | Min VRAM |
|---|---|
| StreamSpeech | 8 GB |
| NAST-S2x | 8 GB |
| SeamlessM4T medium | 8 GB |
| SeamlessM4T v2 large | 16 GB |

---

## Audio preparation

If your audio is not already at 16 kHz mono, convert it with `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```
