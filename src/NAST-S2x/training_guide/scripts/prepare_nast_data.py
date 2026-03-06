#!/usr/bin/env python3
"""
prepare_nast_data.py
===================
Convert JSONL metadata to fairseq TSV format for NAST-S2X training.

Usage:
    python prepare_nast_data.py \
        --datasets-dir datasets \
        --output-dir data/nast-vi-en
"""

import argparse
import json
import os
from pathlib import Path

SAMPLE_RATE = 16_000  # Hz


def n_frames_from_duration(duration: float) -> int:
    """Calculate number of frames from duration (seconds) at 16kHz."""
    return int(duration * SAMPLE_RATE)


def load_jsonl(jsonl_path: Path) -> list[dict]:
    """Load JSONL file and return list of samples."""
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


def write_tsv(rows: list[dict], output_path: Path, fieldnames: list[str]):
    """Write list of dicts to TSV file with header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in rows:
            f.write("\t".join(str(row[k]) for k in fieldnames) + "\n")
    print(f"  Wrote {len(rows):,} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to TSV for NAST-S2X"
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/nast-vi-en",
        help="Output directory for TSV files",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)

    # Process both directions
    splits = [
        ("trainset_vietnamese.jsonl", "train", "vi-en"),
        ("valset_vietnamese.jsonl", "dev", "vi-en"),
        ("trainset_english.jsonl", "train", "en-vi"),
        ("valset_english.jsonl", "dev", "en-vi"),
    ]

    fieldnames = ["id", "audio", "n_frames", "tgt_text", "speaker"]
    all_vi_texts = []
    all_en_texts = []

    for jsonl_name, split, direction in splits:
        jsonl_path = datasets_dir / jsonl_name
        print(f"\n[{direction}/{split}] Loading {jsonl_path}...")

        if not jsonl_path.exists():
            print(f"  [ERROR] File not found: {jsonl_path}")
            continue

        samples = load_jsonl(jsonl_path)
        print(f"  Loaded {len(samples):,} samples")

        # Build TSV rows
        rows = []
        for sample in samples:
            rows.append({
                "id": Path(sample["audio_filepath"]).stem,
                "audio": sample["audio_filepath"],
                "n_frames": n_frames_from_duration(sample["duration"]),
                "tgt_text": sample["tgt_text"],
                "speaker": "spk0",
            })

        # Write TSV
        tsv_path = output_dir / direction / f"{split}.tsv"
        write_tsv(rows, tsv_path, fieldnames)

        # Collect text for SentencePiece
        if split == "train":
            if direction == "vi-en":
                all_vi_texts.extend([s["ori_text"].strip() for s in samples])
                all_en_texts.extend([s["tgt_text"].strip() for s in samples])
            else:  # en-vi
                all_en_texts.extend([s["ori_text"].strip() for s in samples])
                all_vi_texts.extend([s["tgt_text"].strip() for s in samples])

    # Write text files for SPM training
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    vi_text_path = text_dir / "vi_train.txt"
    en_text_path = text_dir / "en_train.txt"

    with open(vi_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_vi_texts))
    print(f"\n[text] Wrote {len(all_vi_texts):,} Vietnamese sentences -> {vi_text_path}")

    with open(en_text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_en_texts))
    print(f"[text] Wrote {len(all_en_texts):,} English sentences -> {en_text_path}")

    print("\n[DONE] Data preparation complete.")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
