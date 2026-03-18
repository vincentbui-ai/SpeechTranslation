"""Data loading utilities."""

import json
from pathlib import Path

TEXT_KEYS = ["ori_text", "text", "transcript", "transcription", "sentence"]


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_file(path: str | Path) -> list[dict]:
    path = Path(path)
    return load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)


def extract_text(sample: dict) -> dict | None:
    for key in TEXT_KEYS:
        if key in sample and sample[key]:
            sample["text"] = sample.pop(key) if key != "text" else sample[key]
            return sample
    return None


def load_samples(paths: list) -> list[dict]:
    samples = []
    for path in paths:
        for sample in load_file(path):
            if (s := extract_text(sample)):
                samples.append(s)
    return samples


def load_translated(path: str | Path) -> list[dict]:
    return load_file(path) if Path(path).exists() else []


def get_untranslated(samples: list[dict], translated: list[dict]) -> list[dict]:
    translated_ids = {s["audio_filepath"] for s in translated}
    return [s for s in samples if s["audio_filepath"] not in translated_ids]


def save_jsonl(samples: list[dict], path: str | Path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
