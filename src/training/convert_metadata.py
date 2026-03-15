# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""Convert multiple metadata JSONL files to seamless training manifest format.

This script is intentionally focused on one output manifest built from one or
more input metadata files.

Text mode behavior:
- Always keep original S2TT sample (source audio/text/lang -> target text/lang).
- Optionally duplicate each sample as ASR (target := source text/lang), default 1:1.

Speech mode behavior:
- TODO: Not implemented yet.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("convert_metadata")


LANG_MAPPING = {
    "Vietnamese": "vie",
    "English": "eng",
    "vie": "vie",
    "eng": "eng",
}

# Global hard limit for filtering source audio duration before conversion.
MAX_DURATION_SEC = 15.0


def _normalize_lang(lang: str) -> str:
    return LANG_MAPPING.get(lang, lang)


def _build_s2tt_sample(data: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
    src_lang = _normalize_lang(data["source_lang"])
    tgt_lang = _normalize_lang(data["target_lang"])

    return {
        "source": {
            "id": sample_id,
            "lang": src_lang,
            "text": data["source_text"],
            "audio_local_path": str(Path(data["source_audio"]).resolve()),
        },
        "target": {
            "id": sample_id,
            "lang": tgt_lang,
            "text": data["target_text"],
        },
    }


def _build_asr_sample(data: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
    src_lang = _normalize_lang(data["source_lang"])

    return {
        "source": {
            "id": sample_id,
            "lang": src_lang,
            "text": data["source_text"],
            "audio_local_path": str(Path(data["source_audio"]).resolve()),
        },
        "target": {
            "id": sample_id,
            "lang": src_lang,
            "text": data["source_text"],
        },
    }


def _count_pair(stats: Dict[str, Any], src_lang: str, tgt_lang: str) -> None:
    pair = f"{src_lang}->{tgt_lang}"
    stats["pairs"][pair] = stats["pairs"].get(pair, 0) + 1


def _load_metadata_records_from_file(input_file: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    raw_lines: list[str] = []

    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(
            f,
            total=total_lines,
            desc=f"Reading {input_file.name}",
            unit="lines",
        ):
            raw_lines.append(line)

    for line_num, line in enumerate(
        tqdm(raw_lines, desc=f"Parsing {input_file.name}", unit="lines"),
        1,
    ):
        raw = line.strip()
        if not raw:
            continue

        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as err:
            logger.error("Line %d: invalid JSON - %s", line_num, err)

    return records


def _load_metadata_records(input_files: list[Path]) -> list[Dict[str, Any]]:
    all_records: list[Dict[str, Any]] = []
    for input_file in input_files:
        all_records.extend(_load_metadata_records_from_file(input_file))

    return all_records


def convert_text_manifest(
    input_files: list[Path],
    output_file: Path,
    enable_asr: bool = True,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total": 0,
        "tasks": {"s2tt": 0, "asr": 0},
        "pairs": {},
        "filtered_by_duration": 0,
    }
    samples = []

    logger.info(
        "Converting text manifest from %d file(s) -> %s (ASR duplicate: %s)",
        len(input_files),
        output_file,
        enable_asr,
    )

    required_keys = {
        "source_audio",
        "source_text",
        "source_lang",
        "target_text",
        "target_lang",
    }

    records = _load_metadata_records(input_files)
    filtered_records: list[Dict[str, Any]] = []

    for idx, data in enumerate(records, 1):
        duration = data.get("duration")
        if duration is None:
            filtered_records.append(data)
            continue

        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            logger.warning(
                "Record %d: invalid duration '%s', keep for conversion",
                idx,
                duration,
            )
            filtered_records.append(data)
            continue

        if duration_value <= MAX_DURATION_SEC:
            filtered_records.append(data)
        else:
            stats["filtered_by_duration"] += 1

    for idx, data in enumerate(
        tqdm(filtered_records, desc="Converting records", unit="records"),
        1,
    ):
        missing = required_keys.difference(data.keys())
        if missing:
            logger.error("Record %d: missing keys %s", idx, sorted(missing))
            continue

        s2tt = _build_s2tt_sample(data, sample_id=len(samples))
        samples.append(s2tt)
        stats["total"] += 1
        stats["tasks"]["s2tt"] += 1
        _count_pair(stats, s2tt["source"]["lang"], s2tt["target"]["lang"])

        if enable_asr:
            asr = _build_asr_sample(data, sample_id=len(samples))
            samples.append(asr)
            stats["total"] += 1
            stats["tasks"]["asr"] += 1
            _count_pair(stats, asr["source"]["lang"], asr["target"]["lang"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"Writing {output_file.name}", unit="samples"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Converted %d samples", stats["total"])
    logger.info(
        "Filtered by duration > %.1f sec: %d record(s)",
        MAX_DURATION_SEC,
        stats["filtered_by_duration"],
    )
    logger.info("Task stats: s2tt=%d, asr=%d", stats["tasks"]["s2tt"], stats["tasks"]["asr"])
    for pair, count in sorted(stats["pairs"].items()):
        logger.info("Pair %s: %d", pair, count)

    return stats


def convert_speech_manifest(input_files: list[Path], output_file: Path) -> None:
    # TODO: implement speech mode conversion for S2ST/T2ST-compatible manifest.
    raise NotImplementedError(
        "mode='speech' is TODO and not implemented yet. "
        "Use --mode text for now."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one or more metadata JSONL files to seamless manifest format"
    )
    parser.add_argument(
        "--input_files",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to metadata JSONL files (e.g., datasets/train_vie.json datasets/train_eng.json)",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to output manifest JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="Conversion mode. speech is TODO.",
    )
    parser.add_argument(
        "--disable_asr",
        action="store_true",
        help="Disable ASR duplication in text mode (default keeps S2TT + ASR 1:1).",
    )
    args = parser.parse_args()

    missing_files = [input_file for input_file in args.input_files if not input_file.exists()]
    if missing_files:
        parser.error(f"Input file(s) not found: {', '.join(str(p) for p in missing_files)}")

    if args.mode == "speech":
        convert_speech_manifest(args.input_files, args.output_file)
        return

    convert_text_manifest(
        input_files=args.input_files,
        output_file=args.output_file,
        enable_asr=not args.disable_asr,
    )


if __name__ == "__main__":
    main()
