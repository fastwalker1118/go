#!/usr/bin/env python3
"""
Split a single raw NPZ file into train/val at game boundaries.

Detects game boundaries by stone-count drops (new game = fewer stones than
previous position). Assigns last N games to val, rest to train.

Usage:
    python split_npz_by_game.py \
        --input data/raw_npz/data0.npz \
        --train-dir data/raw_train \
        --val-dir data/raw_val \
        --val-fraction 0.1
"""

import argparse
import os
from pathlib import Path

import numpy as np


def detect_game_boundaries(bin_packed):
    """Find position indices where a new game starts (stone count drops)."""
    n = bin_packed.shape[0]
    stones_per_pos = np.zeros(n, dtype=np.int32)

    for i in range(n):
        unpacked = np.unpackbits(bin_packed[i], axis=1)[:, : 19 * 19]
        stones_per_pos[i] = unpacked[1].sum() + unpacked[2].sum()

    boundaries = [0]
    for i in range(1, n):
        if stones_per_pos[i] < stones_per_pos[i - 1] - 10:
            boundaries.append(i)
    return boundaries


def main():
    parser = argparse.ArgumentParser(description="Split NPZ by game boundaries")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input NPZ file")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--val-games",
        type=int,
        default=None,
        help="Fixed number of val games (overrides --val-fraction)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    data = dict(np.load(args.input))
    n = data["binaryInputNCHWPacked"].shape[0]
    print(f"Total positions: {n}")

    print("Detecting game boundaries...")
    boundaries = detect_game_boundaries(data["binaryInputNCHWPacked"])
    n_games = len(boundaries)
    print(f"Detected {n_games} games (avg {n / n_games:.1f} positions/game)")

    if args.val_games is not None:
        val_count = args.val_games
    else:
        val_count = max(int(n_games * args.val_fraction), 1)
    train_count = n_games - val_count

    if train_count <= 0 or val_count <= 0:
        raise ValueError(f"Bad split: {train_count} train, {val_count} val games")

    # Split at game boundary
    split_idx = boundaries[train_count]
    print(f"Train: {train_count} games, {split_idx} positions")
    print(f"Val:   {val_count} games, {n - split_idx} positions")

    # Save
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train_data = {k: v[:split_idx] for k, v in data.items()}
    val_data = {k: v[split_idx:] for k, v in data.items()}

    train_path = args.train_dir / "data0.npz"
    val_path = args.val_dir / "data0.npz"

    np.savez(train_path, **train_data)
    np.savez(val_path, **val_data)

    # Verify no overlap
    t_bin = train_data["binaryInputNCHWPacked"]
    v_bin = val_data["binaryInputNCHWPacked"]
    matches = 0
    check_n = min(50, v_bin.shape[0])
    for i in range(check_n):
        if np.any(np.all(t_bin == v_bin[i], axis=(1, 2))):
            matches += 1
    print(f"\nOverlap check: {matches}/{check_n} val positions found in train")
    if matches == 0:
        print("No overlap — split is clean.")
    else:
        print("WARNING: Some overlap detected (possible duplicate games in source SGFs)")

    print(f"\nSaved: {train_path} ({train_data['binaryInputNCHWPacked'].shape[0]} pos)")
    print(f"Saved: {val_path} ({val_data['binaryInputNCHWPacked'].shape[0]} pos)")


if __name__ == "__main__":
    main()
