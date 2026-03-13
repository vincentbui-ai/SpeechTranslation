# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""
Convert metadata from project format to seamless_communication format.
Supports multiple files, multiple language pairs, and both text/speech modes.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("convert_metadata")


# Mapping from full language names to M4T language codes
LANG_MAPPING = {
    "Vietnamese": "vie",
    "English": "eng",
    "vie": "vie",
    "eng": "eng",
}


def convert_to_seamless_format(
    input_file: Path,
    output_file: Path,
    mode: str = "text",  # "text" or "speech"
    extract_units: bool = False,
) -> Dict[str, int]:
    """
    Convert metadata.json to seamless_communication manifest format.
    
    Args:
        input_file: Path to input metadata.json
        output_file: Path to output manifest.json
        mode: "text" or "speech" - determines whether to include units
        extract_units: Whether to extract units from target audio (for speech mode)
    
    Returns:
        Statistics dict with counts per language pair
    """
    stats = {"total": 0, "pairs": {}}
    samples = []
    
    logger.info(f"Converting {input_file} to {output_file} (mode: {mode})")
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Map language names to codes
                src_lang = LANG_MAPPING.get(data["source_lang"], data["source_lang"])
                tgt_lang = LANG_MAPPING.get(data["target_lang"], data["target_lang"])
                
                # Create seamless format sample
                sample = {
                    "source": {
                        "id": line_num - 1,
                        "lang": src_lang,
                        "text": data["source_text"],
                        "audio_local_path": str(Path(data["source_audio"]).resolve()),
                    },
                    "target": {
                        "id": line_num - 1,
                        "lang": tgt_lang,
                        "text": data["target_text"],
                    }
                }
                
                # Add audio and units for speech mode
                if mode == "speech":
                    if "target_audio" in data and data["target_audio"]:
                        sample["target"]["audio_local_path"] = str(
                            Path(data["target_audio"]).resolve()
                        )
                        
                        # Extract units if requested
                        if extract_units:
                            # Units will be extracted later in preprocessing
                            sample["target"]["units"] = None
                    else:
                        logger.warning(
                            f"Line {line_num}: Missing target_audio for speech mode"
                        )
                        continue
                
                samples.append(sample)
                
                # Update stats
                pair_key = f"{src_lang}->{tgt_lang}"
                stats["pairs"][pair_key] = stats["pairs"].get(pair_key, 0) + 1
                stats["total"] += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Line {line_num}: Error processing - {e}")
                continue
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Converted {stats['total']} samples")
    for pair, count in stats["pairs"].items():
        logger.info(f"  {pair}: {count} samples")
    
    return stats


def process_multiple_files(
    input_files: List[Path],
    output_dir: Path,
    mode: str = "text",
    extract_units: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Process multiple metadata files and combine them.
    
    Args:
        input_files: List of input metadata.json files
        output_dir: Output directory for converted manifests
        mode: "text" or "speech"
        extract_units: Whether to extract units
    
    Returns:
        Dictionary mapping input files to their statistics
    """
    all_stats = {}
    combined_samples = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for input_file in input_files:
        logger.info(f"\nProcessing {input_file}")
        
        # Create individual output file
        output_file = output_dir / f"{input_file.stem}_manifest.json"
        stats = convert_to_seamless_format(
            input_file, output_file, mode, extract_units
        )
        all_stats[str(input_file)] = stats
        
        # Load samples for combining
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                combined_samples.append(json.loads(line))
    
    # Reassign IDs for combined file
    for idx, sample in enumerate(combined_samples):
        sample["source"]["id"] = idx
        sample["target"]["id"] = idx
    
    # Write combined file
    combined_file = output_dir / "combined_manifest.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        for sample in combined_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Combined manifest: {combined_file}")
    logger.info(f"Total samples: {len(combined_samples)}")
    logger.info(f"{'='*60}")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata to seamless_communication format"
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Input metadata.json files (supports multiple files and wildcards)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for converted manifests",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="Conversion mode: text (S2TT/T2TT/ASR) or speech (S2ST/T2ST)",
    )
    parser.add_argument(
        "--extract_units",
        action="store_true",
        help="Extract units from target audio (for speech mode)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test sets",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    
    args = parser.parse_args()
    
    # Resolve input files
    input_files = []
    for pattern in args.input_files:
        path = Path(pattern)
        if path.exists():
            input_files.append(path)
        else:
            # Try glob pattern
            import glob
            input_files.extend([Path(p) for p in glob.glob(pattern)])
    
    if not input_files:
        logger.error("No input files found!")
        return
    
    input_files = sorted(set(input_files))
    logger.info(f"Found {len(input_files)} input files")
    
    # Process files
    all_stats = process_multiple_files(
        input_files=input_files,
        output_dir=args.output_dir,
        mode=args.mode,
        extract_units=args.extract_units,
    )
    
    # Split if requested
    if args.split:
        from sklearn.model_selection import train_test_split
        
        combined_file = args.output_dir / "combined_manifest.json"
        with open(combined_file, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        
        # Split by language pair to maintain balance
        samples_by_pair = {}
        for sample in samples:
            pair = f"{sample['source']['lang']}->{sample['target']['lang']}"
            if pair not in samples_by_pair:
                samples_by_pair[pair] = []
            samples_by_pair[pair].append(sample)
        
        train_samples, val_samples, test_samples = [], [], []
        
        for pair, pair_samples in samples_by_pair.items():
            train_val, test = train_test_split(
                pair_samples, test_size=1-args.train_ratio-args.val_ratio, random_state=42
            )
            val_size = args.val_ratio / (args.train_ratio + args.val_ratio)
            train, val = train_test_split(train_val, test_size=val_size, random_state=42)
            
            train_samples.extend(train)
            val_samples.extend(val)
            test_samples.extend(test)
            
            logger.info(f"{pair}: {len(train)} train, {len(val)} val, {len(test)} test")
        
        # Write split files
        for split_name, split_samples in [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ]:
            output_file = args.output_dir / f"{split_name}_manifest.json"
            with open(output_file, "w", encoding="utf-8") as f:
                for idx, sample in enumerate(split_samples):
                    sample["source"]["id"] = idx
                    sample["target"]["id"] = idx
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(split_samples)} samples to {output_file}")


if __name__ == "__main__":
    main()
