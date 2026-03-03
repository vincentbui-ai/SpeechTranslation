# Speech Translation

Repository for training **Speech Translation** models for the **Vietnamese ↔ English** language pair.

---

## Stage 1 — Speech-to-Text Translation (ST)

The goal of Stage 1 is to train an end-to-end speech translation model that takes spoken audio as input and directly outputs translated text in the target language — **without** an intermediate Automatic Speech Recognition (ASR) step.

Two translation directions are trained in parallel:

| Direction | Input | Output |
|---|---|---|
| **VI → EN** | Vietnamese speech | English text |
| **EN → VI** | English speech | Vietnamese text |

---

## Directory Structure

```
SpeechTranslation/
├── README.md
└── datasets/
    ├── trainset_english.jsonl      # Train: EN audio → VI text  (~3.92M samples)
    ├── trainset_vietnamese.jsonl   # Train: VI audio → EN text  (~3.85M samples)
    ├── valset_english.jsonl        # Val:   EN audio → VI text  (~7.9K  samples)
    └── valset_vietnamese.jsonl     # Val:   VI audio → EN text  (~66.6K samples)
```

---

## Data Structure

Each dataset file is in **JSONL** format (JSON Lines) — each line is a self-contained JSON object representing a single training sample.

### Schema

| Field | Type | Description |
|---|---|---|
| `audio_filepath` | `string` | Absolute path to the audio file (`.flac` or `.wav`) |
| `duration` | `float` | Audio duration in seconds |
| `ori_text` | `string` | Original transcript of the audio (source language) |
| `ori_lang` | `string` | Source language: `"English"` or `"Vietnamese"` |
| `tgt_text` | `string` | Translated text (target language) |
| `tgt_lang` | `string` | Target language: `"Vietnamese"` or `"English"` |

### Example — EN → VI

```json
{
  "audio_filepath": "/raid/voice/dataset/audio/.../darkfrigate_06_hawes_64kb_42.flac",
  "duration": 6.839,
  "ori_text": "the two stopped just inside the door you have chalked down the score against us",
  "ori_lang": "English",
  "tgt_text": "Hai người dừng lại ngay bên trong cửa, bạn đã ghi lại điểm số chống lại chúng tôi.",
  "tgt_lang": "Vietnamese"
}
```

### Example — VI → EN

```json
{
  "audio_filepath": "/raid/temp-voice/data/.../2303_youtube_giaothongvantai8_WZ516M73f3-00248822.flac",
  "duration": 5.161,
  "ori_text": "mình sẽ chia sẻ nhiều cái thông tin khác cũng như các cái clip khác về việc hướng dẫn chạy xe an",
  "ori_lang": "Vietnamese",
  "tgt_text": "I will share a lot of other information as well as other clips about how to drive safely.",
  "tgt_lang": "English"
}
```

---

## Dataset Statistics

| File | Direction | Samples |
|---|---|---|
| `trainset_english.jsonl` | EN → VI | ~3,920,336 |
| `trainset_vietnamese.jsonl` | VI → EN | ~3,850,848 |
| `valset_english.jsonl` | EN → VI | ~7,887 |
| `valset_vietnamese.jsonl` | VI → EN | ~66,583 |
| **Total** | | **~7,845,654** |

Audio files are stored in **FLAC** / **WAV** format, sampled at **16 kHz**.
