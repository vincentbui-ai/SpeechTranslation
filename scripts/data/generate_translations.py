#!/usr/bin/env python3
"""
Generate translation data using LLM with concurrent processing.
Converts from metadata.json format to README.md JSONL format.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.data.generators.gemini_client import GeminiLLM


def detect_language(text: str) -> str:
    """Detect if text is Vietnamese or English based on diacritics."""
    vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    return "Vietnamese" if any(c in text.lower() for c in vietnamese_chars) else "English"


def create_translation_prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    """Create translation prompt for Gemini."""
    return f"""Translate the following text from {src_lang} to {tgt_lang}.
Keep the meaning and style intact. Return only the translation, no explanations.

Source ({src_lang}): {text}
Target ({tgt_lang}):"""


def translate_sample(llm: GeminiLLM, sample: dict, sample_idx: int) -> dict | None:
    """Translate a single sample. Returns None if translation fails."""
    ori_text = sample.get("text", "")
    if not ori_text:
        return None

    ori_lang = detect_language(ori_text)
    tgt_lang = "English" if ori_lang == "Vietnamese" else "Vietnamese"

    prompt = create_translation_prompt(ori_text, ori_lang, tgt_lang)

    try:
        response = llm.generate_content(prompt)
        tgt_text = response.text.strip()

        if not tgt_text:
            return None

        return {
            "audio_filepath": sample.get("audio", ""),
            "duration": sample.get("duration", 0.0),
            "ori_text": ori_text,
            "ori_lang": ori_lang,
            "tgt_text": tgt_text,
            "tgt_lang": tgt_lang,
            "_sample_idx": sample_idx,
            "_audio": sample.get("audio", "unknown"),
        }
    except Exception as e:
        return None


def process_metadata(
    metadata_path: str,
    output_path: str,
    api_key: str | None = None,
    max_workers: int = 10,
    batch_size: int = 100
) -> None:
    """Process metadata.json and generate translation JSONL with concurrent processing."""
    # Set API key if provided
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    llm = GeminiLLM(temperature=0.1, max_output_tokens=1024)

    # Read metadata file (handles both single JSON and JSONL)
    samples = []
    with open(metadata_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("["):
            # JSON array format
            samples = json.load(f)
        else:
            # JSONL format (one JSON per line)
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    total_samples = len(samples)
    # Process silently, only progress bar will show

    results_dict = {}
    processed_count = 0
    saved_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(translate_sample, llm, sample, idx): idx
            for idx, sample in enumerate(samples)
        }

        # Process completed tasks with progress bar
        with tqdm(total=total_samples, desc="Translating", unit="sample") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        results_dict[idx] = result
                        processed_count += 1

                        # Save batch periodically
                        if processed_count % batch_size == 0:
                            sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
                            _save_jsonl(sorted_results, output_path)
                            saved_count = len(sorted_results)

                except Exception as e:
                    pass  # Silently skip errors

                pbar.update(1)

    # Final save - sort by index to maintain order
    final_results = [results_dict[i] for i in sorted(results_dict.keys())]
    # Remove internal tracking fields
    for r in final_results:
        r.pop("_sample_idx", None)
        r.pop("_audio", None)

    _save_jsonl(final_results, output_path)


def _save_jsonl(samples: list, output_path: str) -> None:
    """Save samples to JSONL file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate translation data using Gemini LLM with concurrent processing"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input metadata.json or metadata.jsonl file"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--api-key",
        "-k",
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Save progress every N samples (default: 16)"
    )

    args = parser.parse_args()

    process_metadata(
        metadata_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        max_workers=args.workers,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
