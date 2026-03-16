"""
SeamlessM4T inference with a finetuned checkpoint (offline mode).

Usage:
    python scripts/inference/finetuned_infer.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Set offline environment variables BEFORE importing any libraries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["FAIRSEQ2_CACHE_DIR"] = str(REPO_ROOT / "checkpoints" / "assets")

sys.path.insert(0, str(REPO_ROOT / "seamless_communication" / "src"))

import torch
from seamless_communication.inference import Translator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "models/checkpoint.pt"
AUDIO_PATH = "sample.wav"
TASK = "ASR"  # "S2TT" or "ASR"
SRC_LANG = "vie"  # Required for S2TT
TGT_LANG = "vie"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(audio_path: str, checkpoint_path: str, task: str = "ASR",
          src_lang: str = "vie", tgt_lang: str = "vie") -> str:
    """Run S2TT/ASR inference using a finetuned checkpoint (offline)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

    # Load Translator and finetuned weights
    translator = Translator(model_name, vocoder_name, device=device, dtype=dtype)
    translator.model.text_encoder = None
    translator.model.t2u_model = None
    translator.model.load_state_dict(checkpoint["model"], strict=False)

    # Run inference
    if task == "S2TT":
        text_output, _ = translator.predict(input=audio_path, task_str="S2TT",
                                            tgt_lang=tgt_lang, src_lang=src_lang)
    else:
        text_output, _ = translator.predict(input=audio_path, task_str="ASR",
                                            tgt_lang=tgt_lang)
    return str(text_output[0])


if __name__ == "__main__":
    result = infer(AUDIO_PATH, CHECKPOINT_PATH, TASK, SRC_LANG, TGT_LANG)
    print(f"Output: {result}")
