# Copyright (c) 2025, Vincent Bui.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
StreamSpeech offline speech-to-text translation inference.

Model:   src/StreamSpeech  (fairseq CTC-Unity Conformer)
Task:    VI → EN  (or EN → VI with the appropriate checkpoint)
Inputs:  16 kHz mono WAV / FLAC audio
Outputs: translated text string

Usage:
    python inference/streamspeech_infer.py \
        --audio        path/to/audio.wav \
        --checkpoint   checkpoints/streamspeech.offline-s2tt.vi-en/checkpoint_best.pt \
        --data-bin     data/vi-en \
        --config-yaml  configs/vi-en/config_mtl_asr_st_ctcst.yaml \
        --spm-model    configs/vi-en/tgt_unigram6000/spm_unigram_en.model
"""

import argparse
import sys
import os

# Make sure fairseq and ctc_unity are importable from the repo layout
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "StreamSpeech", "fairseq"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "StreamSpeech", "researches"))

import numpy as np
import torch
import torchaudio
import sentencepiece as spm
import yaml

from fairseq import checkpoint_utils, tasks, utils
from examples.speech_to_text.data_utils import extract_fbank_features


SAMPLE_RATE = 16_000


def load_audio(path: str) -> torch.Tensor:
    """Load audio, resample to 16 kHz, convert to mono float32."""
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)


def extract_features(wav: torch.Tensor, gcmvn: dict | None) -> torch.Tensor:
    """Extract 80-dim log-mel fbank features and apply GCMVN if provided."""
    feats = extract_fbank_features(wav.unsqueeze(0), SAMPLE_RATE)  # (T, 80)
    if gcmvn is not None:
        feats = (feats - gcmvn["mean"]) / gcmvn["std"]
    return torch.from_numpy(feats)


def load_model(checkpoint: str, data_bin: str, config_yaml: str | None, device: torch.device):
    """Load StreamSpeech model from a fairseq checkpoint."""
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    utils.import_user_module(state["cfg"].common)

    task_args = state["cfg"]["task"]
    task_args.data = data_bin
    if config_yaml:
        task_args.config_yaml = config_yaml

    task = tasks.setup_task(task_args)
    model = task.build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"], strict=True)
    model.eval().to(device)
    return model, task


@torch.inference_mode()
def translate(
    audio_path: str,
    checkpoint: str,
    data_bin: str,
    config_yaml: str | None,
    spm_model: str,
    device: torch.device,
) -> str:
    # -- Load GCMVN stats if present in config ---------------------------------
    gcmvn = None
    if config_yaml:
        with open(os.path.join(data_bin, config_yaml)) as f:
            cfg = yaml.safe_load(f)
        if "global_cmvn" in cfg:
            gcmvn = np.load(cfg["global_cmvn"]["stats_npz_path"])

    # -- Audio → features ------------------------------------------------------
    wav = load_audio(audio_path)
    feats = extract_features(wav, gcmvn).to(device)          # (T, 80)
    src_tokens  = feats.unsqueeze(0)                          # (1, T, 80)
    src_lengths = torch.tensor([feats.size(0)], device=device)

    # -- Load model ------------------------------------------------------------
    model, _ = load_model(checkpoint, data_bin, config_yaml, device)

    # -- Forward pass (CTC ST head) -------------------------------------------
    encoder_out = model.encoder(src_tokens, src_lengths)

    # StreamSpeech exposes a CTC ST decoder — use the ctc_st head
    ctc_out = model.mt_task_head(encoder_out["encoder_out"][0])  # (T, 1, V)
    ids = ctc_out.squeeze(1).argmax(dim=-1).tolist()             # greedy

    # CTC blank / repeat collapse (blank_index == 0 in fairseq)
    blank = 0
    deduped = [t for i, t in enumerate(ids) if t != blank and (i == 0 or t != ids[i - 1])]

    # -- Decode with SentencePiece --------------------------------------------
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    # fairseq stores SPM tokens with a leading ▁; join and decode
    text = sp.decode([int(t) for t in deduped])
    return text


def main():
    parser = argparse.ArgumentParser(description="StreamSpeech offline S2TT inference")
    parser.add_argument("--audio",       required=True, help="Path to input audio (WAV/FLAC, 16 kHz)")
    parser.add_argument("--checkpoint",  required=True, help="Path to StreamSpeech checkpoint (.pt)")
    parser.add_argument("--data-bin",    required=True, help="Path to fairseq data-bin directory")
    parser.add_argument("--config-yaml", default=None,  help="Config YAML filename inside data-bin")
    parser.add_argument("--spm-model",   required=True, help="Path to target SentencePiece model")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    result = translate(
        args.audio,
        args.checkpoint,
        args.data_bin,
        args.config_yaml,
        args.spm_model,
        device,
    )
    print(f"Translation: {result}")


if __name__ == "__main__":
    main()
