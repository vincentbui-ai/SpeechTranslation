# Copyright (c) 2025, Vincent Bui.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
batch_files.py
==============
Massive batch inference for Speech-to-Text Translation using multi-GPU.

This script uses torchrun for distributed inference across multiple GPUs.
Processes JSONL metadata files containing audio paths and metadata.

Usage:
    # Single GPU
    python scripts/inference/batch_files.py \
        --input metadata.jsonl \
        --output results/ \
        --batch-size 8

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/inference/batch_files.py \
        --input metadata.jsonl \
        --output results/ \
        --batch-size 8 \
        --task S2TT

    # Multi-node (2 nodes, 4 GPUs each)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<ADDR> \
        scripts/inference/batch_files.py --input metadata.jsonl --output results/
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch import distributed as dist
from tqdm import tqdm

# Add seamless_communication to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "seamless_communication", "src")
)

from seamless_communication.inference import Translator

TARGET_SR = 16_000
LANG_MAP = {
    "English": "eng",
    "Vietnamese": "vie",
    "eng": "eng",
    "vie": "vie",
}


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSONL file or directory of JSONL files."""
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".jsonl")
        ]
    else:
        files = [path]

    samples = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
    """Load and preprocess audio file to target sample rate."""
    # Handle path mapping if needed
    filepath = filepath.replace("/raid/voice/khanhnd65/ultimate_testset/", "/data/")
    
    wav, sr = torchaudio.load(filepath)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0), TARGET_SR


def setup_distributed():
    """Initialize distributed process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        return rank, world_size
    else:
        # Single process (no torchrun)
        return 0, 1


def run_inference(
    dataset: list[dict],
    rank: int,
    world_size: int,
    batch_size: int,
    model_name: str,
    vocoder_name: str,
    task: str,
) -> list[tuple]:
    """
    Run distributed inference on dataset.

    Args:
        dataset: List of sample dictionaries
        rank: Process rank
        world_size: Total number of processes
        batch_size: Batch size per GPU
        model_name: Model name
        vocoder_name: Vocoder name
        task: Task type (S2TT or S2ST)

    Returns:
        List of result tuples
    """
    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Initialize translator
    translator = Translator(
        model_name,
        vocoder_name,
        torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu"),
        dtype=dtype,
    )

    # Split dataset across processes
    indices = list(range(len(dataset)))
    indices = indices[rank::world_size]

    # Batch indices
    batched_indices = [
        indices[i : i + batch_size]
        for i in range(0, len(indices), batch_size)
    ]

    # Add progress bar for rank 0
    if rank == 0:
        batched_indices = tqdm(
            batched_indices,
            desc=f"[GPU {rank}] Inference",
            unit="batch",
        )

    results = []
    for batch_idx in batched_indices:
        batch_data = [dataset[i] for i in batch_idx]
        
        # Extract metadata
        paths = [d.get("audio_filepath", "") for d in batch_data]
        refs = [d.get("tgt_text", "") for d in batch_data]
        ori_texts = [d.get("ori_text", "") for d in batch_data]
        src_langs = [LANG_MAP.get(d.get("ori_lang", "eng"), "eng") for d in batch_data]
        tgt_langs = [LANG_MAP.get(d.get("tgt_lang", "vie"), "vie") for d in batch_data]

        # Load audio files
        wavs = []
        valid_indices = []
        for i, path in enumerate(paths):
            if not path:
                continue
            try:
                wav, _ = load_audio(path)
                wavs.append(wav)
                valid_indices.append(i)
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to load {path}: {e}")
                continue

        if not wavs:
            continue

        # Process each valid audio
        for i, wav in zip(valid_indices, wavs):
            try:
                wav_input = wav.unsqueeze(0).to(
                    f"cuda:{device}" if torch.cuda.is_available() else "cpu",
                    dtype=dtype,
                )
                text, _ = translator.predict(
                    wav_input,
                    task,
                    tgt_langs[i],
                    src_lang=src_langs[i],
                )
                results.append(
                    (
                        paths[i],
                        ori_texts[i],
                        src_langs[i],
                        tgt_langs[i],
                        refs[i],
                        str(text[0]),
                    )
                )
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to translate {paths[i]}: {e}")
                continue

    return results


def save_results(results: list, output_path: str, rank: int, world_size: int):
    """Save results to CSV file (only rank 0)."""
    if world_size > 1:
        # Gather results from all processes
        outputs = [None for _ in range(world_size)]
        dist.gather_object(results, outputs if rank == 0 else None)
        
        if rank == 0:
            all_results = list(itertools.chain(*outputs))
        else:
            return
    else:
        all_results = results

    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
        output_filepath = os.path.join(output_path, "results.csv")

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("FILEPATH,ORI_TEXT,SRC_LANG,TGT_LANG,REFERENCE,HYPOTHESIS\n")
            for filepath, ori_text, src_lang, tgt_lang, ref, hyp in all_results:
                # Escape commas in text fields
                ori_text = f'"{ori_text}"' if "," in ori_text else ori_text
                ref = f'"{ref}"' if "," in ref else ref
                hyp = f'"{hyp}"' if "," in hyp else hyp
                f.write(f"{filepath},{ori_text},{src_lang},{tgt_lang},{ref},{hyp}\n")

        print(f"\n[done] {len(all_results)} samples saved → {output_filepath}")


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Massive batch inference for Speech Translation (Multi-GPU)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL file or directory containing metadata",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per GPU (default: 8)",
    )
    parser.add_argument(
        "--task",
        default="S2TT",
        choices=["S2TT", "S2ST"],
        help="Task type: S2TT (text) or S2ST (speech)",
    )
    parser.add_argument(
        "--model",
        default="seamlessM4T_v2_large",
        help="Model name (default: seamlessM4T_v2_large)",
    )
    parser.add_argument(
        "--vocoder",
        default="vocoder_v2",
        help="Vocoder name (default: vocoder_v2)",
    )

    args = parser.parse_args()

    # Setup distributed
    rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Loading dataset from: {args.input}")
        print(f"Using {world_size} GPU(s), batch size: {args.batch_size}")

    # Load dataset
    dataset = load_dataset(args.input)
    
    if rank == 0:
        print(f"Total samples: {len(dataset)}")

    # Run inference
    results = run_inference(
        dataset=dataset,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        model_name=args.model,
        vocoder_name=args.vocoder,
        task=args.task,
    )

    # Save results
    save_results(results, args.output, rank, world_size)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
