# Copyright (c) 2025, Vincent Bui.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SeamlessM4T offline speech-to-text translation inference.

Model:   seamless_communication  (Meta SeamlessM4T v2)
Task:    S2TT — Speech to Text Translation
         VI → EN : --src-lang vie --tgt-lang eng
         EN → VI : --src-lang eng --tgt-lang vie
Inputs:  16 kHz mono WAV / FLAC audio
Outputs: translated text string

Language codes follow ISO 639-3 (e.g. "vie", "eng").

Usage:
    python scripts/inference/single_file.py \
        --audio    path/to/audio.wav \
        --src-lang vie \
        --tgt-lang eng

    # Optionally specify a local model directory instead of downloading:
    python scripts/inference/single_file.py \
        --audio      path/to/audio.wav \
        --src-lang   vie \
        --tgt-lang   eng \
        --model-name seamlessM4T_v2_large
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "seamless_communication", "src"))

import torch
import torchaudio

from seamless_communication.inference import Translator

SAMPLE_RATE = 16_000


def load_audio(path: str) -> torch.Tensor:
    """Load audio, resample to 16 kHz, convert to mono float32."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav  # (1, T)


def translate(
    audio_path: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    translator = Translator(model_name, "vocoder_v2", device, dtype=dtype)

    wav = load_audio(audio_path).to(device)

    text_output, _ = translator.predict(
        input=wav,
        task_str="S2TT",
        tgt_lang=tgt_lang,
        src_lang=src_lang,
        text_generation_opts=None,
        unit_generation_opts=None,
    )
    return str(text_output[0])


def main():
    parser = argparse.ArgumentParser(description="SeamlessM4T offline S2TT inference")
    parser.add_argument("--audio",      required=True, help="Path to input audio (WAV/FLAC, 16 kHz)")
    parser.add_argument("--src-lang",   required=True, help="Source language ISO 639-3 code (e.g. vie, eng)")
    parser.add_argument("--tgt-lang",   required=True, help="Target language ISO 639-3 code (e.g. eng, vie)")
    parser.add_argument("--model-name", default="seamlessM4T_v2_large",
                        choices=["seamlessM4T_medium", "seamlessM4T_large", "seamlessM4T_v2_large"],
                        help="SeamlessM4T model variant")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    result = translate(
        args.audio,
        args.src_lang,
        args.tgt_lang,
        args.model_name,
        device,
        dtype,
    )
    print(f"Translation: {result}")


if __name__ == "__main__":
    main()
