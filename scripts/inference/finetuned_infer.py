"""
SeamlessM4T inference with a finetuned checkpoint (offline mode).

Runs all 5 tasks sequentially:
1. S2ST (Speech to Speech Translation) → Saves WAV file
2. S2TT (Speech to Text Translation) → Prints text
3. T2ST (Text to Speech Translation) → Saves WAV file
4. T2TT (Text to Text Translation) → Prints text
5. ASR (Automatic Speech Recognition) → Prints text

Usage:
    python scripts/inference/finetuned_infer.py
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Set offline environment variables BEFORE importing any libraries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["FAIRSEQ2_CACHE_DIR"] = str(REPO_ROOT / "checkpoints" / "assets")

sys.path.insert(0, str(REPO_ROOT / "seamless_communication" / "src"))

import torch
import torchaudio
from seamless_communication.inference import Translator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "models/translation_v2.pt"
AUDIO_PATH = "/raid/voice/khanhnd65/ultimate_testset/vietnamese_testset/vlsp2022-task1/wavs/2022_1005_00002640_00003214.wav"
INPUT_TEXT = "Hello, how are you today?"
SRC_LANG = "vie"
TGT_LANG = "eng"
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def save_audio(waveform: torch.Tensor, sample_rate: int, output_path: Path) -> None:
    """Save waveform to audio file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, sample_rate)
    print(f"  Audio saved: {output_path}")


def print_result(task: str, text: Optional[str]) -> None:
    """Print task result."""
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")
    if text:
        print(f"Output: {text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all 5 tasks with finetuned checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_name = checkpoint["model_name"]
    vocoder_name = (
        "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
    )

    # Load Translator (full model for all tasks)
    print(f"Loading model: {model_name}")
    translator = Translator(model_name, vocoder_name, device=device, dtype=dtype)
    translator.model.load_state_dict(checkpoint["model"], strict=False)
    print("Model loaded successfully!\n")

    # -------------------------------------------------------------------------
    # Task 1: S2ST (Speech to Speech Translation)
    # -------------------------------------------------------------------------
    print("Running Task 1: S2ST (Speech to Speech Translation)...")
    text_out, speech_out = translator.predict(
        input=AUDIO_PATH,
        task_str="S2ST",
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )
    print_result("S2ST", str(text_out[0]) if text_out else None)
    if speech_out and speech_out.audio_wavs:
        waveform = speech_out.audio_wavs[0][0].to(torch.float32).cpu()
        save_audio(waveform, speech_out.sample_rate, OUTPUT_DIR / "s2st_output.wav")

    # -------------------------------------------------------------------------
    # Task 2: S2TT (Speech to Text Translation)
    # -------------------------------------------------------------------------
    print("\nRunning Task 2: S2TT (Speech to Text Translation)...")
    text_out, _ = translator.predict(
        input=AUDIO_PATH,
        task_str="S2TT",
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )
    print_result("S2TT", str(text_out[0]) if text_out else None)

    # -------------------------------------------------------------------------
    # Task 3: T2ST (Text to Speech Translation)
    # -------------------------------------------------------------------------
    print("\nRunning Task 3: T2ST (Text to Speech Translation)...")
    text_out, speech_out = translator.predict(
        input=INPUT_TEXT,
        task_str="T2ST",
        src_lang="eng",  # Input text is English
        tgt_lang=SRC_LANG,  # Translate to Vietnamese
    )
    print_result("T2ST", str(text_out[0]) if text_out else None)
    if speech_out and speech_out.audio_wavs:
        waveform = speech_out.audio_wavs[0][0].to(torch.float32).cpu()
        save_audio(waveform, speech_out.sample_rate, OUTPUT_DIR / "t2st_output.wav")

    # -------------------------------------------------------------------------
    # Task 4: T2TT (Text to Text Translation)
    # -------------------------------------------------------------------------
    print("\nRunning Task 4: T2TT (Text to Text Translation)...")
    text_out, _ = translator.predict(
        input=INPUT_TEXT,
        task_str="T2TT",
        src_lang="eng",  # Input text is English
        tgt_lang=SRC_LANG,  # Translate to Vietnamese
    )
    print_result("T2TT", str(text_out[0]) if text_out else None)

    # -------------------------------------------------------------------------
    # Task 5: ASR (Automatic Speech Recognition)
    # -------------------------------------------------------------------------
    print("\nRunning Task 5: ASR (Automatic Speech Recognition)...")
    text_out, _ = translator.predict(
        input=AUDIO_PATH,
        task_str="ASR",
        tgt_lang=SRC_LANG,  # Transcribe in source language
    )
    print_result("ASR", str(text_out[0]) if text_out else None)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("All tasks completed!")
    print(f"{'='*60}")
    print(f"Audio outputs saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
