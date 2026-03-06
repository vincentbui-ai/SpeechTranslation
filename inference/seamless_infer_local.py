# Copyright (c) 2025, Vincent Bui.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SeamlessM4T offline S2TT inference — load model from a local absolute path.

Model:   src/seamless_communication  (Meta SeamlessM4T v2)
Task:    S2TT — Speech to Text Translation

Two ways to point at a local model are shown:
  1. HF hub cache directory  → set SEAMLESS_CACHE_DIR (simplest)
  2. AssetCard override      → pass --model-dir explicitly (recommended)

Usage — quickstart (edit the two paths at the top of __main__):
    python inference/seamless_infer_local.py

Usage — CLI:
    python inference/seamless_infer_local.py \
        --audio      /absolute/path/to/audio.wav \
        --model-dir  /absolute/path/to/seamlessM4T_v2_large \
        --src-lang   vie \
        --tgt-lang   eng
"""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "seamless_communication", "src"))

import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from fairseq2.assets.card import AssetCard

from seamless_communication.inference import (SequenceGeneratorOptions,
                                              Translator)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BASE_MODEL  = "seamlessM4T_v2_large"   # architecture base — always keep this


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: str) -> torch.Tensor:
    """Load audio file, resample to 16 kHz, convert to mono. Returns (1, T)."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
    if wav.shape[0] > 1:                    # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)
    return wav


# ---------------------------------------------------------------------------
# Local model loading
# ---------------------------------------------------------------------------

def build_local_card(model_dir: str) -> AssetCard:
    """
    Create a fairseq2 AssetCard that inherits the SeamlessM4T architecture
    metadata but overrides the checkpoint with a local file URI.

    Expected layout inside model_dir:
        model_dir/
        ├── model.pt          ← model weights
        └── tokenizer.model   ← SentencePiece tokenizer  (optional)

    If model_dir contains a HuggingFace hub snapshot (downloaded with
    huggingface_hub.snapshot_download), the layout is already correct.
    """
    model_dir  = os.path.abspath(model_dir)
    model_file = os.path.join(model_dir, "model.pt")

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Checkpoint not found at {model_file}\n"
            "Make sure model_dir contains 'model.pt'."
        )

    # Register a local metadata provider that overrides the checkpoint URL
    # while inheriting all other arch / tokenizer metadata from the base card.
    local_metadata = {
        "name": f"{BASE_MODEL}_local",
        "@base": BASE_MODEL,                        # inherit arch metadata
        "checkpoint": f"file://{model_file}",       # local file URI
    }
    provider = InProcAssetMetadataProvider([local_metadata])
    asset_store.metadata_providers.insert(0, provider)

    return asset_store.retrieve_card(f"{BASE_MODEL}_local")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def translate(
    audio_path: str,
    src_lang: str,
    tgt_lang: str,
    model_dir: str | None,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    """
    Translate speech audio to text.

    Args:
        audio_path: Absolute path to the input audio file (WAV / FLAC).
        src_lang:   Source language ISO 639-3 code (e.g. "vie", "eng").
        tgt_lang:   Target language ISO 639-3 code (e.g. "eng", "vie").
        model_dir:  Absolute path to the local model directory.
                    Pass None to auto-download from HuggingFace.
        model_name: Base model variant (used when model_dir is None).
        device:     torch.device to run on.
        dtype:      torch.dtype (float16 for GPU, float32 for CPU).

    Returns:
        Translated text string.
    """
    # -- Build model card (local or remote) -----------------------------------
    if model_dir:
        logger.info(f"Loading model from local path: {model_dir}")
        card = build_local_card(model_dir)
    else:
        logger.info(f"Loading model '{model_name}' from HuggingFace hub")
        card = model_name                           # Translator handles the lookup

    # -- Load translator -------------------------------------------------------
    translator = Translator(
        model_name_or_card=card,
        vocoder_name_or_card="vocoder_v2",
        device=device,
        dtype=dtype,
    )

    # -- Load & preprocess audio -----------------------------------------------
    logger.info(f"Loading audio: {audio_path}")
    wav = load_audio(audio_path).to(device)         # (1, T) float32

    # -- Run S2TT inference ----------------------------------------------------
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=5,
        soft_max_seq_len=(1, 200),
    )

    text_output, _ = translator.predict(
        input=wav,
        task_str="S2TT",
        tgt_lang=tgt_lang,
        src_lang=src_lang,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=None,
    )

    return str(text_output[0])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SeamlessM4T S2TT inference with local model support"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Absolute path to input audio (WAV / FLAC, ideally 16 kHz)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Absolute path to local model directory containing 'model.pt'. "
            "If omitted, the model is downloaded from HuggingFace."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=BASE_MODEL,
        choices=["seamlessM4T_medium", "seamlessM4T_large", "seamlessM4T_v2_large"],
        help="Model variant to download (ignored when --model-dir is set).",
    )
    parser.add_argument(
        "--src-lang",
        default="vie",
        help="Source language ISO 639-3 code  [default: vie]",
    )
    parser.add_argument(
        "--tgt-lang",
        default="eng",
        help="Target language ISO 639-3 code  [default: eng]",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    result = translate(
        audio_path=args.audio,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_dir=args.model_dir,
        model_name=args.model_name,
        device=device,
        dtype=dtype,
    )
    print(f"\nTranslation ({args.src_lang} → {args.tgt_lang}): {result}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # QUICKSTART — edit these two paths and run:  python inference/seamless_infer_local.py
    # ------------------------------------------------------------------
    AUDIO_PATH = "/absolute/path/to/sample.wav"
    MODEL_DIR  = "/absolute/path/to/seamlessM4T_v2_large"   # contains model.pt
    SRC_LANG   = "vie"   # Vietnamese input
    TGT_LANG   = "eng"   # English output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    result = translate(
        audio_path=AUDIO_PATH,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        model_dir=MODEL_DIR,
        model_name=BASE_MODEL,
        device=device,
        dtype=dtype,
    )
    print(f"\nTranslation ({SRC_LANG} → {TGT_LANG}): {result}")
