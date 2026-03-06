#!/usr/bin/env python3
"""
Compress enriched NPZ files by sparsifying soft policy targets.

Soft policy targets (quality band) have many small non-zero values that compress
poorly. This script zeros out negligible mass (below threshold), renormalizes,
and re-saves. The resulting sparser array compresses much better.

Usage:
    python compress_enriched_npz.py [--threshold 50] [--output-dir ...] [--in-place]

Examples:
    python compress_enriched_npz.py
        # Outputs to enriched_npz_compressed/
    python compress_enriched_npz.py --in-place
        # Overwrites originals (backup recommended)
    python compress_enriched_npz.py --threshold 100
        # Stricter sparsification (zeros out moves with <1% mass)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "Training_Dataset" / "all_human_pros" / "enriched_npz"
DEFAULT_OUTPUT = REPO_ROOT / "Training_Dataset" / "all_human_pros" / "enriched_npz_compressed"
POLICY_SCALE = 10000
# ────────────────────────────────────────────────────────────────────


def compress_one_file(
    npz_path: Path, output_path: Path, threshold: int, policy_scale: int
) -> tuple[int, int]:
    """
    Sparsify policy targets in one NPZ file. Returns (original_bytes, compressed_bytes).
    """
    with np.load(npz_path) as npz:
        arrays = {k: npz[k].copy() for k in npz.files}

    policy = arrays["policyTargetsNCMove"]
    player_policy = policy[:, 0, :].astype(np.int32)  # work in int32 to avoid overflow

    # Zero out values below threshold
    mask = player_policy >= threshold
    player_policy_sparse = np.where(mask, player_policy, 0)

    # Renormalize each row to sum to policy_scale
    row_sums = player_policy_sparse.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # avoid div by zero (should not happen)
    player_policy_sparse = np.round(
        player_policy_sparse.astype(np.float64) * policy_scale / row_sums
    ).astype(np.int32)

    # Fix any rounding drift: ensure sum is exactly policy_scale
    row_sums_final = player_policy_sparse.sum(axis=1)
    drift = policy_scale - row_sums_final
    for i in range(len(drift)):
        if drift[i] != 0 and mask[i].any():
            # Add drift to the largest non-zero entry
            idx = np.flatnonzero(mask[i])
            if len(idx) > 0:
                argmax = idx[np.argmax(player_policy_sparse[i, idx])]
                player_policy_sparse[i, argmax] += drift[i]

    policy[:, 0, :] = np.clip(player_policy_sparse, 0, 32767).astype(np.int16)
    arrays["policyTargetsNCMove"] = policy

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Measure size: save to temp to get byte count (or compare file sizes)
    orig_size = npz_path.stat().st_size
    np.savez_compressed(output_path, **arrays)
    new_size = output_path.stat().st_size

    return orig_size, new_size


def main():
    parser = argparse.ArgumentParser(
        description="Compress enriched NPZ by sparsifying soft policy targets"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: enriched_npz_compressed, or input-dir if --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite files in place (use with caution; backup first)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Zero out policy values below this (scaled units, POLICY_SCALE=10000; 50=0.5%%)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        output_dir = args.input_dir
        if not args.dry_run:
            print("WARNING: --in-place will overwrite originals. Backup first.")
    else:
        output_dir = args.output_dir or args.input_dir.parent / "enriched_npz_compressed"

    npz_files = sorted(args.input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files in {args.input_dir}")
        sys.exit(0)

    print(f"Input:  {args.input_dir}")
    print(f"Output: {output_dir}")
    print(
        f"Threshold: {args.threshold} (zeros out moves with <{100*args.threshold/POLICY_SCALE:.2f}% mass)"
    )
    print(f"Files: {len(npz_files)}")
    if args.dry_run:
        print("DRY RUN - no files will be written")
        return

    total_orig = 0
    total_new = 0
    for i, npz_path in enumerate(npz_files):
        out_path = output_dir / npz_path.name
        orig, new = compress_one_file(npz_path, out_path, args.threshold, POLICY_SCALE)
        total_orig += orig
        total_new += new
        pct = 100 * (1 - new / orig) if orig > 0 else 0
        print(
            f"  [{i+1}/{len(npz_files)}] {npz_path.name}: {orig/1e9:.2f}GB -> {new/1e9:.2f}GB ({pct:.1f}% smaller)"
        )

    print(
        f"\nTotal: {total_orig/1e9:.2f}GB -> {total_new/1e9:.2f}GB saved {100*(1-total_new/total_orig):.1f}%"
    )


if __name__ == "__main__":
    main()
