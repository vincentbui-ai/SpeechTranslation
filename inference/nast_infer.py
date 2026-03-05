# Copyright (c) 2025, Vincent Bui.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
NAST-S2x offline speech-to-text translation inference.

Model:   src/NAST-S2x  (Non-Autoregressive Streaming Transformer)
Task:    VI → EN  (or EN → VI with the appropriate checkpoint)
Inputs:  16 kHz mono WAV / FLAC audio
Outputs: translated text string

The NASTSpeechAgent was designed for SimulEval streaming evaluation.
This script wraps it in a simple offline pass: feed the full audio as
a single finished segment and decode the final WriteAction output.

Usage:
    python inference/nast_infer.py \
        --audio       path/to/audio.wav \
        --model-path  checkpoints/nast.pt \
        --data-bin    data/vi-en \
        --config-yaml config_mtl_asr_st_ctcst.yaml \
        --spm-model   configs/vi-en/tgt_unigram6000/spm_unigram_en.model
"""

import argparse
import sys
import os
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "NAST-S2x"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "StreamSpeech", "fairseq"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "StreamSpeech", "SimulEval"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "StreamSpeech", "researches"))

import torch
import torchaudio

from nast.agents.nast_speech_agent_s2s import NASTSpeechAgent
from simuleval.data.segments import SpeechSegment


SAMPLE_RATE = 16_000


def load_audio_samples(path: str) -> list:
    """Return a flat list of float32 PCM samples at 16 kHz mono."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).tolist()


def build_agent_args(
    model_path: str,
    data_bin: str,
    config_yaml: str | None,
    spm_model: str | None,
    device: str,
) -> types.SimpleNamespace:
    """Build the Namespace that NASTSpeechAgent.__init__ expects."""
    args = types.SimpleNamespace(
        model_path=model_path,
        data_bin=data_bin,
        config_yaml=config_yaml,
        global_stats=None,
        tgt_splitter_type="SentencePiece",
        tgt_splitter_path=spm_model,
        user_dir="examples/simultaneous_translation",
        shift_size=10,
        window_size=25,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        main_context=32,
        right_context=16,
        wait_until=0,
        device="gpu" if device == "cuda" else "cpu",
        global_cmvn=None,
    )
    return args


@torch.inference_mode()
def translate(
    audio_path: str,
    model_path: str,
    data_bin: str,
    config_yaml: str | None,
    spm_model: str | None,
    device: str,
) -> str:
    args = build_agent_args(model_path, data_bin, config_yaml, spm_model, device)
    agent = NASTSpeechAgent(args)
    agent.reset()

    samples = load_audio_samples(audio_path)

    # Feed the full audio as one finished speech segment
    segment = SpeechSegment(index=0, content=samples, sample_rate=SAMPLE_RATE, finished=True)
    agent.states.update_source(segment)

    action = agent.policy()
    return action.content.content if hasattr(action, "content") else str(action)


def main():
    parser = argparse.ArgumentParser(description="NAST-S2x offline S2TT inference")
    parser.add_argument("--audio",       required=True, help="Path to input audio (WAV/FLAC, 16 kHz)")
    parser.add_argument("--model-path",  required=True, help="Path to NAST checkpoint (.pt)")
    parser.add_argument("--data-bin",    required=True, help="Path to fairseq data-bin directory")
    parser.add_argument("--config-yaml", default=None,  help="Config YAML filename inside data-bin")
    parser.add_argument("--spm-model",   default=None,  help="Path to target SentencePiece model")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    result = translate(
        args.audio,
        args.model_path,
        args.data_bin,
        args.config_yaml,
        args.spm_model,
        args.device,
    )
    print(f"Translation: {result}")


if __name__ == "__main__":
    main()
