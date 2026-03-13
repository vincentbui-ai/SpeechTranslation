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

---

## Stage 1 Training Pipeline

Training **SeamlessM4T** model cho cặp ngôn ngữ VI ↔ EN.

> Dữ liệu audio chỉ tồn tại trên server. Các bước **LOCAL** chỉ cần file JSONL (đã có sẵn). Các bước **SERVER** cần truy cập audio files.

### Cấu trúc thư mục sau khi hoàn thành

```
SpeechTranslation/
├── datasets/                    ← JSONL files (đã có)

├── scripts/                     ← Scripts pipeline (đã tạo)
│   ├── prepare_data.py          (LOCAL)
│   ├── train_spm.py             (LOCAL)
│   ├── compute_gcmvn.py         (SERVER)
│   └── train.offline-s2tt.vi-en.sh  (SERVER)
├── configs/vi-en/               ← Configs cho VI-EN (đã tạo)
│   ├── config_gcmvn.yaml
│   ├── config_mtl_asr_st_ctcst.yaml
│   ├── src_unigram6000/         ← SPM VI + tokenized TSVs (Bước 3)
│   │   ├── spm_unigram_vi.model
│   │   ├── spm_unigram_vi.txt   ← fairseq dict
│   │   ├── train.tsv            ← id + tokenized VI text (ASR head)
│   │   └── dev.tsv
│   └── tgt_unigram6000/         ← SPM EN + tokenized TSVs (Bước 3)
│       ├── spm_unigram_en.model
│       ├── spm_unigram_en.txt   ← fairseq dict
│       ├── train.tsv            ← id + tokenized EN text (ST/MT heads)
│       └── dev.tsv
└── data/vi-en/                  ← TSV manifests (Bước 2)
    ├── train.tsv, dev.tsv       ← audio path + n_frames + EN tgt_text
    ├── train_asr.tsv, dev_asr.tsv ← audio path + n_frames + VI ori_text
```



---

### Bước 1 — Chuyển đổi dữ liệu JSONL → TSV `[LOCAL]`

Script đọc JSONL và tạo TSV manifests cho fairseq.
Không cần audio files — `n_frames` tính từ `duration × 16000`.

```bash
# Từ thư mục gốc SpeechTranslation/
python scripts/prepare_data.py \
    --datasets-dir datasets \
    --output-dir   data
```

**Output:**
- `data/vi-en/train.tsv` — VI speech → EN text (main ST task, ~3.85M)
- `data/vi-en/dev.tsv`   — validation (~66.6K)
- `data/vi-en/train_asr.tsv` — VI speech → VI text (cho ASR multitask head)
- `data/vi-en/dev_asr.tsv`

---

### Bước 2 — Train SentencePiece + tạo tokenized TSVs `[LOCAL]`
1. Train SPM unigram 6000-token cho VI và EN
2. Tạo fairseq dict `.txt`
3. Apply SPM tokenization → `configs/vi-en/{src,tgt}_unigram6000/{train,dev}.tsv`
   (chính là `data:` field trong `config_mtl_asr_st_ctcst.yaml`)

```bash
python scripts/train_spm.py \
    --data-dir    data/vi-en \
    --configs-dir configs/vi-en
```

**Output (cần có trước khi training):**
- `configs/vi-en/tgt_unigram6000/spm_unigram_en.{model,txt}` + `{train,dev}.tsv`
- `configs/vi-en/src_unigram6000/spm_unigram_vi.{model,txt}` + `{train,dev}.tsv`

Kiểm tra nhanh:
```bash
head -3 configs/vi-en/tgt_unigram6000/train.tsv
# id    tgt_text
# dark..._42   the two ▁stop ped ▁just ▁inside ...
```

---

### Bước 3 — Copy files lên server `[LOCAL → SERVER]`

```bash
# Thay YOUR_SERVER và /path/on/server theo thực tế
rsync -avz data/vi-en/     YOUR_SERVER:/path/on/server/SpeechTranslation/data/vi-en/
rsync -avz configs/vi-en/  YOUR_SERVER:/path/on/server/SpeechTranslation/configs/vi-en/
rsync -avz scripts/        YOUR_SERVER:/path/on/server/SpeechTranslation/scripts/
```

---

### Bước 4 — Cập nhật đường dẫn server trong config `[SERVER]`

```bash
# Trên server — đổi giá trị SERVER_ROOT theo thực tế
SERVER_ROOT=/raid/SpeechTranslation

sed -i "s|/PATH/TO/SERVER|$SERVER_ROOT|g" \
    $SERVER_ROOT/configs/vi-en/config_gcmvn.yaml \
    $SERVER_ROOT/configs/vi-en/config_mtl_asr_st_ctcst.yaml
```

---

### Bước 5 — Tính GCMVN statistics `[SERVER]`

Chạy 1 lần, dùng 50,000 samples đầu.

```bash
python scripts/compute_gcmvn.py \
    --tsv-path  data/vi-en/train.tsv \
    --output    configs/vi-en/gcmvn.npz \
    --max-samples 50000
```

**Output:** `configs/vi-en/gcmvn.npz` — mean & std của 80-dim fbank features.

---

### Bước 6 — Tải pretrained checkpoint `[SERVER]`

Tải checkpoint SeamlessM4T từ HuggingFace.

---

### Bước 7 — Training `[SERVER]`

Mở `scripts/train.offline-s2tt.vi-en.sh`, cập nhật:

```bash
SERVER_ROOT=/raid/SpeechTranslation   # ← đổi thành đường dẫn thực tế
```

Điều chỉnh theo VRAM GPU:

| GPU VRAM | MAX_TOKENS | UPDATE_FREQ |
|---|---|---|
| 40 GB (A100) | 22000 | 2 |
| 24 GB (3090/4090) | 14000 | 4 |
| 16 GB (V100) | 10000 | 8 |

Chạy:

```bash
bash scripts/train.offline-s2tt.vi-en.sh
```

Checkpoints lưu tại `checkpoints/seamlessm4t.vi-en/`.

---

### Tóm tắt: Local vs Server

| Bước | Chạy ở đâu | Script gốc tham chiếu |
|---|---|---|
| Bước 1 — prepare_data.py | **LOCAL** | — |
| Bước 2 — train_spm.py    | **LOCAL** | — |
| Bước 3 — rsync           | LOCAL → SERVER | — |
| Bước 4 — sed config      | **SERVER** | — |
| Bước 5 — compute_gcmvn.py| **SERVER** | — |
| Bước 6 — tải checkpoint  | **SERVER** | — |
| Bước 7 — training        | **SERVER** | — |
