"""
SeamlessM4T S2TT evaluation script with distributed inference.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/inference/evaluate_s2tt.py --metadata datasets/metadata.json

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/inference/evaluate_s2tt.py \
        --metadata datasets/metadata.json --checkpoint models/translation_v2.pt
"""

import argparse
import json
import os
import string
import sys
from pathlib import Path

import sacrebleu
import torch
import torch.distributed as dist
from jiwer import wer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["FAIRSEQ2_CACHE_DIR"] = str(REPO_ROOT / "checkpoints" / "assets")
sys.path.insert(0, str(REPO_ROOT / "seamless_communication" / "src"))

from seamless_communication.inference import Translator


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/translation_v2.pt")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="s2tt_results.json")
    parser.add_argument("--min-words", type=int, default=5)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load model
    if rank == 0:
        print(f"[1/4] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
    translator = Translator(model_name, vocoder_name, device=device, dtype=dtype)
    translator.model.load_state_dict(checkpoint["model"], strict=False)

    if world_size > 1:
        dist.barrier()

    # Load and filter data
    if rank == 0:
        print(f"[2/4] Loading metadata: {args.metadata}")
    with open(args.metadata, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if len(r.get("source_text", "").split()) >= args.min_words]
    if rank == 0:
        print(f"[2/4] Samples after filtering: {len(rows)}")

    # Run inference
    if rank == 0:
        print("[3/4] Running S2TT inference...")
    lang_map = {"vietnamese": "vie", "english": "eng", "vie": "vie", "eng": "eng"}
    indices = list(range(len(rows)))[rank::world_size]
    iterator = tqdm(indices, desc="Inference") if rank == 0 else indices
    
    results = []
    for idx in iterator:
        row = rows[idx]
        src_lang = lang_map.get(row.get("source_lang", "vie").lower(), "vie")
        tgt_lang = lang_map.get(row.get("target_lang", "eng").lower(), "eng")
        try:
            text_out, _ = translator.predict(input=row["source_audio"], task_str="S2TT", src_lang=src_lang, tgt_lang=tgt_lang)
            pred = str(text_out[0]) if text_out else ""
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] {row['source_audio']}: {e}")
            pred = ""
        results.append((row["source_audio"], row["target_text"], pred, tgt_lang))

    # Gather results
    if world_size > 1:
        dist.barrier()
        gathered = [None] * world_size
        dist.gather_object(results, gathered if rank == 0 else None)
        all_results = [r for g in (gathered if rank == 0 else []) if g for r in g]
    else:
        all_results = results

    # Compute metrics (rank 0 only)
    if rank == 0:
        print(f"[4/4] Computing metrics on {len(all_results)} samples...")
        refs = [" ".join(remove_punctuation(r[1]).lower().split()) for r in all_results]
        preds = [" ".join(remove_punctuation(r[2]).lower().split()) for r in all_results]

        wer_score = wer(refs, preds)
        bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score

        print(f"\nWER: {wer_score * 100:.2f}% | BLEU: {bleu_score:.2f}")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": args.checkpoint, "metadata": args.metadata,
                "num_samples": len(all_results), "wer": wer_score, "bleu": bleu_score,
                "samples": [{"audio": r[0], "reference": r[1], "prediction": r[2]} for r in all_results]
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
