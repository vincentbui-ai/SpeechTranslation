# Inference — Speech-to-Text Translation

Inference scripts are provided for model libraries in `src/`.  
All scripts accept a **16 kHz mono WAV or FLAC** audio file and print the translated text.

---

## Prerequisites

Make sure you are inside the project's conda environment before running any script.

```bash
conda activate translation
cd /path/to/SpeechTranslation   # repo root
```

---

## 1. SeamlessM4T

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

All scripts auto-detect CUDA. To force CPU:

```bash
python inference/<script>.py ... --device cpu
```

Recommended VRAM by model:

| Model | Min VRAM |
|---|---|
| SeamlessM4T medium | 8 GB |
| SeamlessM4T v2 large | 16 GB |

---

## Audio preparation

If your audio is not already at 16 kHz mono, convert it with `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```
