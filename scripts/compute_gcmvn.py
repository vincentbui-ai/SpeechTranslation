"""
compute_gcmvn.py
================
Tính Global Cepstral Mean and Variance Normalization (GCMVN) statistics
từ tập train để chuẩn hóa filterbank features khi training.

⚠️  CHẠY TRÊN SERVER — cần truy cập audio files.

Input:
    data/vi-en/train.tsv   (hoặc en-vi/train.tsv)
    Audio files thực tế ở server

Output:
    configs/vi-en/gcmvn.npz   (mean & std của 80-dim fbank features)

Usage:
    # Chạy trên server sau khi copy scripts/ và data/ lên server:
    python scripts/compute_gcmvn.py \\
        --tsv-path data/vi-en/train.tsv \\
        --output   configs/vi-en/gcmvn.npz \\
        --max-samples 50000
"""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import torch
    import torchaudio
    from torchaudio.compliance import kaldi
except ImportError:
    raise ImportError("Cài torchaudio: pip install torchaudio")


NUM_MEL_BINS = 80


def compute_fbank(audio_path: str) -> torch.Tensor:
    """Trích xuất 80-dim filterbank features từ audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample về 16kHz nếu cần
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Chuyển về mono nếu stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    features = kaldi.fbank(
        waveform,
        num_mel_bins=NUM_MEL_BINS,
        frame_length=25,   # ms
        frame_shift=10,    # ms
        sample_frequency=16000,
        use_energy=False,
    )
    return features  # (T, 80) - keep as torch.Tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-path",    required=True, help="Path to train.tsv")
    parser.add_argument("--output",      required=True, help="Output .npz path")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Số samples tối đa dùng để tính stats (mặc định: 50000)"
    )
    args = parser.parse_args()

    tsv_path = Path(args.tsv_path)
    output   = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Đọc danh sách audio paths từ TSV
    audio_paths = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_paths.append(row["audio"])

    print(f"Total samples in TSV: {len(audio_paths):,}")
    print(f"Using up to {args.max_samples:,} samples for GCMVN computation")

    # Tính running mean và variance using torch (avoid numpy compatibility issues)
    total_frames = 0
    running_mean = torch.zeros(NUM_MEL_BINS, dtype=torch.float64)
    running_m2   = torch.zeros(NUM_MEL_BINS, dtype=torch.float64)

    errors = 0
    for i, audio_path in enumerate(audio_paths[:args.max_samples]):
        if i % 1000 == 0:
            print(f"  Processing {i:,}/{min(len(audio_paths), args.max_samples):,} ...")

        try:
            feats = compute_fbank(audio_path)  # (T, 80) - torch.Tensor
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  [WARN] {audio_path}: {e}")
            continue

        for frame in feats:
            total_frames += 1
            delta = frame - running_mean
            running_mean += delta / total_frames
            delta2 = frame - running_mean
            running_m2 += delta * delta2

    if total_frames < 2:
        raise RuntimeError("Không đủ dữ liệu để tính GCMVN stats")

    # Convert to numpy only at the end
    variance = (running_m2 / (total_frames - 1)).numpy()
    std      = np.sqrt(variance).astype(np.float32)
    mean     = running_mean.numpy().astype(np.float32)

    np.savez(output, mean=mean, std=std)
    print(f"\n[DONE] GCMVN stats from {total_frames:,} frames ({errors} errors)")
    print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std  range: [{std.min():.4f},  {std.max():.4f}]")
    print(f"  Saved -> {output}")


if __name__ == "__main__":
    main()
