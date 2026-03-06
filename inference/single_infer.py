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
single_infer.py
===============
Single audio file inference for Speech-to-Text Translation.

Supports SeamlessM4T models for translating a single audio file.
Optionally returns synthesized audio (for S2ST task).

Usage:
    # S2TT (Speech-to-Text Translation)
    python inference/single_infer.py \
        --audio path/to/audio.wav \
        --src-lang vie \
        --tgt-lang eng \
        --task S2TT

    # S2ST (Speech-to-Speech Translation) with audio output
    python inference/single_infer.py \
        --audio path/to/audio.wav \
        --src-lang eng \
        --tgt-lang vie \
        --task S2ST \
        --output-audio path/to/output.wav
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio

# Add seamless_communication to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "seamless_communication", "src")
)

from seamless_communication.inference import Translator

TARGET_SR = 16_000


def load_audio(filepath: str) -> tuple[torch.Tensor, int]:
    """Load and preprocess audio file to target sample rate."""
    wav, sr = torchaudio.load(filepath)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0), TARGET_SR


def translate_audio(
    audio_path: str,
    src_lang: str,
    tgt_lang: str,
    task: str = "S2TT",
    model_name: str = "seamlessM4T_v2_large",
    vocoder_name: str = "vocoder_v2",
    device: Optional[torch.device] = None,
    output_audio_path: Optional[str] = None,
) -> dict:
    """
    Translate a single audio file.

    Args:
        audio_path: Path to input audio file (.wav or .flac)
        src_lang: Source language code (e.g., 'vie', 'eng')
        tgt_lang: Target language code (e.g., 'eng', 'vie')
        task: Translation task - 'S2TT' (text) or 'S2ST' (speech)
        model_name: Model name to use
        vocoder_name: Vocoder name for S2ST
        device: Torch device (defaults to cuda if available)
        output_audio_path: Path to save output audio (required for S2ST)

    Returns:
        Dictionary with 'text' (translated text) and optionally 'audio' tensor
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load model
    translator = Translator(model_name, vocoder_name, device, dtype=dtype)

    # Load and preprocess audio
    wav, _ = load_audio(audio_path)
    wav_input = wav.unsqueeze(0).to(device, dtype=dtype)

    # Run inference
    text_output, speech_output = translator.predict(
        wav_input, task, tgt_lang, src_lang=src_lang
    )
    
    output: dict[str, Any] = {"text": str(text_output[0])}

    # Handle audio output for S2ST
    if task == "S2ST" and speech_output is not None:
        output["audio"] = speech_output.audio_wavs[0][0]
        
        if output_audio_path:
            # Save audio output
            torchaudio.save(
                output_audio_path,
                speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                sample_rate=speech_output.sample_rate,
            )
            print(f"Output audio saved to: {output_audio_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Single audio file inference for Speech Translation"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input audio file (.wav or .flac)",
    )
    parser.add_argument(
        "--src-lang",
        required=True,
        choices=["vie", "eng", "eng_Latn"],
        help="Source language code",
    )
    parser.add_argument(
        "--tgt-lang",
        required=True,
        choices=["vie", "eng", "eng_Latn"],
        help="Target language code",
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
    parser.add_argument(
        "--output-audio",
        help="Path to save output audio (required for S2ST)",
    )
    parser.add_argument(
        "--device",
        help="Device to use (cuda/cpu, defaults to cuda if available)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.task == "S2ST" and not args.output_audio:
        print("Warning: S2ST task specified but no --output-audio path provided")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    print(f"Translating: {args.audio}")
    print(f"{args.src_lang} -> {args.tgt_lang} ({args.task})")

    # Run translation
    result = translate_audio(
        audio_path=args.audio,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        task=args.task,
        model_name=args.model,
        vocoder_name=args.vocoder,
        device=device,
        output_audio_path=args.output_audio,
    )

    print(f"\nTranslated text: {result['text']}")

    return result


if __name__ == "__main__":
    main()
