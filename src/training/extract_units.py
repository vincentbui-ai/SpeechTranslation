# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""
Extract discrete units from audio files for speech-to-speech training.
Supports multiple target audio files and parallel processing.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("extract_units")


def load_unit_extractor(
    model_name: str = "xlsr2_1b_v2",
    device: str = "cuda",
):
    """Load unit extractor model."""
    from seamless_communication.models.unit_extractor import UnitExtractor
    
    # Determine kmeans model based on version
    if "v2" in model_name:
        kmeans_uri = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.pt"
        out_layer_idx = 14
    else:
        kmeans_uri = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.pt"
        out_layer_idx = 35
    
    extractor = UnitExtractor(
        model_name_or_card=model_name,
        kmeans_model_uri=kmeans_uri,
        device=torch.device(device),
    )
    
    return extractor, out_layer_idx


def extract_units_from_audio(
    audio_path: str,
    extractor,
    out_layer_idx: int,
) -> Optional[List[int]]:
    """Extract units from a single audio file."""
    try:
        units = extractor.predict(
            audio_path,
            out_layer_idx=out_layer_idx,
        )
        return units.tolist() if isinstance(units, torch.Tensor) else units
    except Exception as e:
        logger.error(f"Failed to extract units from {audio_path}: {e}")
        return None


def process_manifest(
    input_manifest: Path,
    output_manifest: Path,
    model_name: str = "xlsr2_1b_v2",
    device: str = "cuda",
    batch_size: int = 1,
):
    """
    Process manifest file and extract units for all target audio files.
    
    Args:
        input_manifest: Input manifest with target.audio_local_path
        output_manifest: Output manifest with target.units
        model_name: Unit extractor model name
        device: Device to run extraction on
        batch_size: Batch size for extraction (currently only supports 1)
    """
    logger.info(f"Loading unit extractor: {model_name}")
    extractor, out_layer_idx = load_unit_extractor(model_name, device)
    
    # Read input manifest
    samples = []
    with open(input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    logger.info(f"Processing {len(samples)} samples")
    
    # Extract units for each sample
    processed = 0
    failed = 0
    
    for sample in tqdm(samples, desc="Extracting units"):
        if "target" in sample and "audio_local_path" in sample["target"]:
            audio_path = sample["target"]["audio_local_path"]
            units = extract_units_from_audio(audio_path, extractor, out_layer_idx)
            
            if units is not None:
                sample["target"]["units"] = units
                processed += 1
            else:
                failed += 1
                sample["target"]["units"] = None
        else:
            logger.warning(f"Sample missing target audio path: {sample.get('source', {}).get('id')}")
            failed += 1
    
    # Write output manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Processing complete: {processed} success, {failed} failed")
    logger.info(f"Output written to: {output_manifest}")


def process_multiple_manifests(
    input_files: List[Path],
    output_dir: Path,
    model_name: str = "xlsr2_1b_v2",
    device: str = "cuda",
):
    """Process multiple manifest files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}_with_units.json"
        logger.info(f"\nProcessing {input_file}")
        process_manifest(input_file, output_file, model_name, device)


def main():
    parser = argparse.ArgumentParser(
        description="Extract discrete units from target audio files"
    )
    parser.add_argument(
        "--input_manifests",
        nargs="+",
        required=True,
        help="Input manifest files (supports multiple files)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for manifests with units",
    )
    parser.add_argument(
        "--model_name",
        default="xlsr2_1b_v2",
        choices=["xlsr2_1b", "xlsr2_1b_v2"],
        help="Unit extractor model name",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run extraction on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for extraction",
    )
    
    args = parser.parse_args()
    
    # Resolve input files
    input_files = []
    for pattern in args.input_manifests:
        path = Path(pattern)
        if path.exists():
            input_files.append(path)
        else:
            import glob
            input_files.extend([Path(p) for p in glob.glob(pattern)])
    
    if not input_files:
        logger.error("No input files found!")
        return
    
    input_files = sorted(set(input_files))
    logger.info(f"Found {len(input_files)} input files")
    
    process_multiple_manifests(
        input_files=input_files,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
