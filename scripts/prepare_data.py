"""
prepare_data.py
===============
Chuyển đổi JSONL dataset → fairseq TSV manifest format cho Speech-to-Text Translation.

Chạy LOCAL (không cần audio files — dùng duration * 16000 để tính n_frames).

Output structure:
    data/vi-en/
    ├── train.tsv         # VI speech → EN text (main ST task)
    ├── dev.tsv
    ├── train_asr.tsv     # VI speech → VI text (ASR multitask)
    └── dev_asr.tsv

    data/en-vi/
    ├── train.tsv         # EN speech → VI text (main ST task)
    ├── dev.tsv
    ├── train_asr.tsv     # EN speech → EN text (ASR multitask)
    └── dev_asr.tsv

    data/text/
    ├── vi_train.txt      # Toàn bộ văn bản tiếng Việt (để train SPM)
    └── en_train.txt      # Toàn bộ văn bản tiếng Anh (để train SPM)

Usage:
    python scripts/prepare_data.py --datasets-dir datasets --output-dir data
"""

import argparse
import json
import os
from pathlib import Path

SAMPLE_RATE = 16_000  # Hz


def n_frames_from_duration(duration: float) -> int:
    """Tính số frames từ duration (giây) với 16kHz sample rate."""
    return int(duration * SAMPLE_RATE)


def write_tsv(rows: list[dict], output_path: Path, fieldnames: list[str]):
    """Ghi danh sách dict thành file TSV với header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in fieldnames) + "\n")
    print(f"  Wrote {len(rows):,} samples -> {output_path}")


def process_jsonl(jsonl_path: Path) -> list[dict]:
    """Đọc toàn bộ JSONL file, trả về list of dict."""
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i}: {e}")
    return samples


def build_tsv_rows(samples: list[dict], tgt_field: str) -> list[dict]:
    """
    Tạo rows cho TSV từ list of samples.

    Args:
        tgt_field: 'tgt_text' (ST) hoặc 'ori_text' (ASR)
    """
    rows = []
    for sample in samples:
        rows.append({
            "id":       Path(sample["audio_filepath"]).stem,
            "audio":    sample["audio_filepath"],
            "n_frames": n_frames_from_duration(sample["duration"]),
            "tgt_text": sample[tgt_field],
            "speaker":  "spk0",
        })
    return rows


def extract_text(samples: list[dict], field: str) -> list[str]:
    """Trích xuất text từ list of samples theo field name."""
    return [s[field].strip() for s in samples if s.get(field, "").strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Thư mục chứa các file JSONL (mặc định: datasets/)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Thư mục output (mặc định: data/)"
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir   = Path(args.output_dir)

    # Mapping: (jsonl_filename, direction_dir, tgt_field, asr_field)
    splits = [
        # VI → EN: trainset_vietnamese.jsonl
        ("trainset_vietnamese.jsonl", "vi-en", "train"),
        ("valset_vietnamese.jsonl",   "vi-en", "dev"),
        # EN → VI: trainset_english.jsonl
        ("trainset_english.jsonl",    "en-vi", "train"),
        ("valset_english.jsonl",      "en-vi", "dev"),
    ]

    tsv_header = ["id", "audio", "n_frames", "tgt_text", "speaker"]

    vi_texts, en_texts = [], []

    for jsonl_name, direction, split in splits:
        jsonl_path = datasets_dir / jsonl_name
        print(f"\n[{direction}/{split}] Loading {jsonl_path} ...")

        if not jsonl_path.exists():
            print(f"  [ERROR] File not found: {jsonl_path}")
            continue

        samples = process_jsonl(jsonl_path)
        print(f"  Loaded {len(samples):,} samples")

        # --- Main ST TSV (tgt_text = bản dịch) ---
        st_rows = build_tsv_rows(samples, tgt_field="tgt_text")
        write_tsv(st_rows, output_dir / direction / f"{split}.tsv", tsv_header)

        # --- ASR TSV (tgt_text = bản gốc, dùng cho multitask ASR head) ---
        asr_rows = build_tsv_rows(samples, tgt_field="ori_text")
        write_tsv(asr_rows, output_dir / direction / f"{split}_asr.tsv", tsv_header)

        # --- Thu thập text để train SPM ---
        if split == "train":
            ori_lang = samples[0]["ori_lang"] if samples else ""
            tgt_lang = samples[0]["tgt_lang"] if samples else ""

            if direction == "vi-en":
                vi_texts.extend(extract_text(samples, "ori_text"))  # tiếng Việt nguồn
                en_texts.extend(extract_text(samples, "tgt_text"))  # tiếng Anh đích
            else:
                en_texts.extend(extract_text(samples, "ori_text"))  # tiếng Anh nguồn
                vi_texts.extend(extract_text(samples, "tgt_text"))  # tiếng Việt đích

    # --- Ghi text files cho SPM training ---
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    vi_text_path = text_dir / "vi_train.txt"
    en_text_path = text_dir / "en_train.txt"

    with open(vi_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vi_texts))
    print(f"\n[text] Wrote {len(vi_texts):,} Vietnamese sentences -> {vi_text_path}")

    with open(en_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(en_texts))
    print(f"[text] Wrote {len(en_texts):,} English sentences -> {en_text_path}")

    print("\n[DONE] Data preparation complete.")
    print(f"  ST  data:   {output_dir}/vi-en/   &   {output_dir}/en-vi/")
    print(f"  SPM text:   {output_dir}/text/")


if __name__ == "__main__":
    main()
