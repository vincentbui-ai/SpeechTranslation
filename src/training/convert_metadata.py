# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""
Convert metadata from project format to seamless_communication format.
Supports multiple files, multiple language pairs, and both text/speech modes.

Examples:
    # Split single file into train/val/test
    python convert_metadata.py --input_files datasets/metadata.json --output_dir data/manifests --split

    # Use separate files for train/val/test
    python convert_metadata.py \\
        --train_files trainset_vietnamese.json trainset_english.json \\
        --val_files valset_vietnamese.json valset_english.json \\
        --test_files testset_vietnamese.json testset_english.json \\
        --output_dir data/manifests

    # Use wildcards
    python convert_metadata.py \\
        --train_files "trainset_*.json" \\
        --val_files "valset_*.json" \\
        --output_dir data/manifests
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

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
    start_id: int = 0,
) -> Dict[str, any]:
    """
    Convert metadata.json to seamless_communication manifest format.
    
    Args:
        input_file: Path to input metadata.json
        output_file: Path to output manifest.json
        mode: "text" or "speech" - determines whether to include units
        extract_units: Whether to extract units from target audio (for speech mode)
        start_id: Starting ID for samples (for combining multiple files)
    
    Returns:
        Statistics dict with counts per language pair and list of samples
    """
    stats = {"total": 0, "pairs": {}}
    samples = []
    
    logger.info(f"Converting {input_file} to {output_file} (mode: {mode})")
    
    # Pre-count lines for progress bar
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    
    samples = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"Reading {input_file.name}"), 1):
            try:
                data = json.loads(line.strip())
                
                # Map language names to codes
                src_lang = LANG_MAPPING.get(data["source_lang"], data["source_lang"])
                tgt_lang = LANG_MAPPING.get(data["target_lang"], data["target_lang"])
                
                # Create seamless format sample
                sample = {
                    "source": {
                        "id": start_id + line_num - 1,
                        "lang": src_lang,
                        "text": data["source_text"],
                        "audio_local_path": str(Path(data["source_audio"]).resolve()),
                    },
                    "target": {
                        "id": start_id + line_num - 1,
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
    
    # Write output with buffered I/O for better performance
    output_file.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 1000
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"Writing {output_file.name}", unit="samples"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Converted {stats['total']} samples")
    for pair, count in stats["pairs"].items():
        logger.info(f"  {pair}: {count} samples")
    
    return {"stats": stats, "samples": samples}


def resolve_files(patterns: Optional[List[str]]) -> List[Path]:
    """Resolve file patterns to actual files."""
    if not patterns:
        return []
    
    import glob
    files = []
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            files.append(path)
        else:
            # Try glob pattern
            matched = [Path(p) for p in glob.glob(pattern)]
            if matched:
                files.extend(matched)
            else:
                logger.warning(f"No files found for pattern: {pattern}")
    
    return sorted(set(files))


def _convert_single_file(args: tuple) -> Dict:
    """Helper for parallel processing."""
    input_file, temp_output, mode, extract_units, start_id = args
    return convert_to_seamless_format(input_file, temp_output, mode, extract_units, start_id)


def combine_and_write_manifest(
    input_files: List[Path],
    output_file: Path,
    mode: str = "text",
    extract_units: bool = False,
    parallel: bool = True,
    max_workers: int = 4,
) -> Dict[str, int]:
    """
    Combine multiple input files into a single manifest.
    
    Args:
        input_files: List of input metadata.json files
        output_file: Output manifest file
        mode: "text" or "speech"
        extract_units: Whether to extract units
        parallel: Use multiprocessing for faster conversion
        max_workers: Number of parallel workers
    
    Returns:
        Combined statistics
    """
    all_samples = []
    all_stats = {"total": 0, "pairs": {}}
    current_id = 0
    
    if parallel and len(input_files) > 1:
        logger.info(f"Processing {len(input_files)} files in parallel with {max_workers} workers")
        
        # Prepare args for parallel processing
        task_args = [
            (input_file, output_file.parent / f"{input_file.stem}_temp.json", mode, extract_units, current_id + i * 100000)
            for i, input_file in enumerate(input_files)
        ]
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(_convert_single_file, task_args),
                total=len(input_files),
                desc="Processing files",
                unit="file"
            ))
        
        # Merge results
        for result in results:
            all_samples.extend(result["samples"])
            
            all_stats["total"] += result["stats"]["total"]
            for pair, count in result["stats"]["pairs"].items():
                all_stats["pairs"][pair] = all_stats["pairs"].get(pair, 0) + count
        
        # Clean up temp files
        for input_file in input_files:
            temp_output = output_file.parent / f"{input_file.stem}_temp.json"
            temp_output.unlink(missing_ok=True)
    else:
        # Sequential processing with tqdm
        for input_file in tqdm(input_files, desc="Processing files", unit="file"):
            logger.info(f"\nProcessing {input_file}")
            
            temp_output = output_file.parent / f"{input_file.stem}_temp.json"
            result = convert_to_seamless_format(
                input_file, temp_output, mode, extract_units, start_id=current_id
            )
            
            all_samples.extend(result["samples"])
            current_id += result["stats"]["total"]
            
            all_stats["total"] += result["stats"]["total"]
            for pair, count in result["stats"]["pairs"].items():
                all_stats["pairs"][pair] = all_stats["pairs"].get(pair, 0) + count
            
            temp_output.unlink(missing_ok=True)
    
    # Reassign IDs for combined file
    for idx, sample in enumerate(tqdm(all_samples, desc="Reassigning IDs", unit="samples")):
        sample["source"]["id"] = idx
        sample["target"]["id"] = idx
    
    # Write combined file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(all_samples, desc=f"Writing {output_file.name}", unit="samples"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Combined manifest: {output_file}")
    logger.info(f"Total samples: {len(all_samples)}")
    for pair, count in all_stats["pairs"].items():
        logger.info(f"  {pair}: {count} samples")
    logger.info(f"{'='*60}")
    
    return all_stats


def split_combined_manifest(
    input_files: List[Path],
    output_dir: Path,
    mode: str = "text",
    extract_units: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, any]:
    """
    Combine all input files and split into train/val/test sets.
    Maintains balance across language pairs.
    """
    from sklearn.model_selection import train_test_split
    
    # Combine all samples
    all_samples = []
    for input_file in tqdm(input_files, desc="Reading input files", unit="file"):
        logger.info(f"\nProcessing {input_file}")
        
        temp_output = output_dir / f"{input_file.stem}_manifest.json"
        result = convert_to_seamless_format(
            input_file, temp_output, mode, extract_units, start_id=len(all_samples)
        )
        all_samples.extend(result["samples"])
    
    # Split by language pair to maintain balance
    samples_by_pair = {}
    for sample in all_samples:
        pair = f"{sample['source']['lang']}->{sample['target']['lang']}"
        if pair not in samples_by_pair:
            samples_by_pair[pair] = []
        samples_by_pair[pair].append(sample)
    
    train_samples, val_samples, test_samples = [], [], []
    
    for pair, pair_samples in samples_by_pair.items():
        if len(pair_samples) < 3:
            logger.warning(f"Pair {pair} has only {len(pair_samples)} samples, adding all to train")
            train_samples.extend(pair_samples)
            continue
            
        test_size = 1 - train_ratio - val_ratio
        if test_size <= 0:
            test_size = 0.1
            
        train_val, test = train_test_split(
            pair_samples, test_size=test_size, random_state=42
        )
        
        if len(train_val) > 0 and val_ratio > 0:
            val_size = val_ratio / (train_ratio + val_ratio)
            val_size = max(val_size, 1.0 / len(train_val))  # Ensure at least 1 sample
            train, val = train_test_split(train_val, test_size=val_size, random_state=42)
        else:
            train = train_val
            val = []
        
        train_samples.extend(train)
        val_samples.extend(val)
        test_samples.extend(test)
        
        logger.info(f"{pair}: {len(train)} train, {len(val)} val, {len(test)} test")
    
    # Reassign IDs and write split files
    stats = {}
    for split_name, split_samples in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        if split_samples:
            for idx, sample in enumerate(split_samples):
                sample["source"]["id"] = idx
                sample["target"]["id"] = idx
            
            output_file = output_dir / f"{split_name}_manifest.json"
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in tqdm(split_samples, desc=f"Writing {split_name}", unit="samples"):
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(split_samples)} samples to {output_file}")
            stats[split_name] = len(split_samples)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata to seamless_communication format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Split single file into train/val/test
    python convert_metadata.py --input_files datasets/metadata.json --output_dir data/manifests --split

    # Use separate files for train/val/test
    python convert_metadata.py \\
        --train_files trainset_vietnamese.json trainset_english.json \\
        --val_files valset_vietnamese.json valset_english.json \\
        --test_files testset_*.json \\
        --output_dir data/manifests
        """
    )
    
    # Input options - either old style or new style
    parser.add_argument(
        "--input_files",
        nargs="*",
        help="Input metadata.json files (legacy, use --train_files/--val_files instead)",
    )
    parser.add_argument(
        "--train_files",
        nargs="*",
        help="Training metadata.json files (supports wildcards)",
    )
    parser.add_argument(
        "--val_files",
        nargs="*",
        help="Validation metadata.json files (supports wildcards)",
    )
    parser.add_argument(
        "--test_files",
        nargs="*",
        help="Test metadata.json files (supports wildcards)",
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
        help="Split combined files into train/val/test sets (legacy mode)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (for --split mode)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (for --split mode)",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if using new style (separate train/val/test files)
    if args.train_files or args.val_files or args.test_files:
        # New style: separate files for each split
        train_files = resolve_files(args.train_files)
        val_files = resolve_files(args.val_files)
        test_files = resolve_files(args.test_files)
        
        logger.info(f"Found {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")
        
        # Process each split
        if train_files:
            combine_and_write_manifest(
                train_files,
                output_dir / "train_manifest.json",
                args.mode,
                args.extract_units,
            )
        
        if val_files:
            combine_and_write_manifest(
                val_files,
                output_dir / "val_manifest.json",
                args.mode,
                args.extract_units,
            )
        
        if test_files:
            combine_and_write_manifest(
                test_files,
                output_dir / "test_manifest.json",
                args.mode,
                args.extract_units,
            )
        
        # Also create combined manifest for reference
        all_files = train_files + val_files + test_files
        if all_files:
            combine_and_write_manifest(
                all_files,
                output_dir / "combined_manifest.json",
                args.mode,
                args.extract_units,
            )
    
    elif args.input_files:
        # Legacy mode: either split or just convert
        input_files = resolve_files(args.input_files)
        
        if not input_files:
            logger.error("No input files found!")
            return
        
        logger.info(f"Found {len(input_files)} input files")
        
        if args.split:
            # Split mode
            split_combined_manifest(
                input_files=input_files,
                output_dir=output_dir,
                mode=args.mode,
                extract_units=args.extract_units,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
        else:
            # Just combine without splitting
            combine_and_write_manifest(
                input_files,
                output_dir / "combined_manifest.json",
                args.mode,
                args.extract_units,
            )
    else:
        parser.error("Must specify either --input_files or --train_files/--val_files/--test_files")


if __name__ == "__main__":
    main()
