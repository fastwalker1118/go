#!/usr/bin/env python3
"""
Evaluate KataGo model checkpoints on a held-out test set.

Computes top-1 and top-3 move-prediction accuracy against human ground truth.

Usage (from NPZ files already converted):
    python evaluate_models.py \
        --test-dir /path/to/npz_files \
        --checkpoints model_a.ckpt model_b.ckpt \
        --labels "Base b18c384" "Soft-target finetuned"

Usage (from SGF folder — auto-converts first):
    python evaluate_models.py \
        --sgf-dir /path/to/kejie_testset \
        --checkpoints model_a.ckpt model_b.ckpt \
        --labels "Base b18c384" "Soft-target finetuned"
"""

import sys
import os
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add KataGo python directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from katago.train.load_model import load_model
from katago.train import modelconfigs
from katago.train.data_processing_pytorch import build_history_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLICY_LEN = 362  # 19*19 + 1 (pass)


def convert_sgfs_to_npz(
    sgf_dir: Path, output_dir: Path, config_file: Path = None, only_player: str = None
):
    """Convert a folder of SGF files to a single NPZ using KataGo binary."""
    katago_bin = REPO_ROOT / "cpp" / "build" / "katago"
    if not katago_bin.exists():
        katago_bin = REPO_ROOT / "cpp" / "katago"
    if not katago_bin.exists():
        raise FileNotFoundError(
            f"KataGo binary not found at {REPO_ROOT / 'cpp' / 'build' / 'katago'} "
            f"or {REPO_ROOT / 'cpp' / 'katago'}. Please build KataGo first."
        )

    if config_file is None:
        config_file = REPO_ROOT / "simple_training_config.cfg"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = output_dir / "test_data.npz"

    cmd = [
        str(katago_bin),
        "writetrainingdata_simple",
        "-config",
        str(config_file),
        "-sgfdir",
        str(sgf_dir),
        "-output",
        str(out_npz),
        "-verbosity",
        "1",
    ]
    if only_player:
        cmd += ["-only-player", only_player]
    logger.info(f"Converting SGFs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not out_npz.exists():
        raise RuntimeError("Conversion produced no output NPZ file.")
    logger.info(f"Converted SGFs -> {out_npz}")
    return output_dir


def unpack_binary_input(packed: np.ndarray, pos_len: int) -> np.ndarray:
    """Unpack binaryInputNCHWPacked to float32 spatial features."""
    unpacked = np.unpackbits(packed, axis=2)
    unpacked = unpacked[:, :, : pos_len * pos_len]
    n, c, _ = unpacked.shape
    return unpacked.reshape(n, c, pos_len, pos_len).astype(np.float32)


def load_all_test_data(npz_dir: Path, pos_len: int):
    """Load and concatenate all NPZ files in a directory. Returns arrays."""
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

    all_binary = []
    all_global_input = []
    all_policy_targets = []
    all_global_targets = []

    for f in npz_files:
        with np.load(f) as npz:
            all_binary.append(npz["binaryInputNCHWPacked"])
            all_global_input.append(npz["globalInputNC"])
            all_policy_targets.append(npz["policyTargetsNCMove"])
            all_global_targets.append(npz["globalTargetsNC"])

    binary = np.concatenate(all_binary, axis=0)
    global_input = np.concatenate(all_global_input, axis=0)
    policy_targets = np.concatenate(all_policy_targets, axis=0).astype(np.float32)
    global_targets = np.concatenate(all_global_targets, axis=0)

    # Ground truth move index for each position
    gt_moves = np.argmax(policy_targets[:, 0, :], axis=1)

    logger.info(f"Loaded {binary.shape[0]} positions from {len(npz_files)} file(s)")
    return binary, global_input, policy_targets, global_targets, gt_moves


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str,
    binary_packed: np.ndarray,
    global_input: np.ndarray,
    global_targets: np.ndarray,
    gt_moves: np.ndarray,
    pos_len: int,
    batch_size: int,
    device: torch.device,
    use_swa: bool = False,
):
    """Run inference with one checkpoint and return top-k accuracy stats."""
    logger.info(f"Loading checkpoint: {ckpt_path}")
    model, swa_model, _ = load_model(
        ckpt_path,
        use_swa=use_swa,
        device=device,
        pos_len=pos_len,
        verbose=False,
    )
    if use_swa and swa_model is not None:
        model = swa_model
    model.eval()

    model_config = model.config
    h_base, h_builder = build_history_matrices(model_config, device)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)

    n_samples = binary_packed.shape[0]
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    # Per-move-number accuracy tracking
    move_num_top1 = defaultdict(lambda: [0, 0])  # [correct, total]
    move_num_top3 = defaultdict(lambda: [0, 0])

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        b_bin_np = unpack_binary_input(binary_packed[start:end], pos_len)
        b_bin = torch.from_numpy(b_bin_np).to(device)
        b_global = torch.from_numpy(global_input[start:end].copy()).to(device)
        b_gtargets = torch.from_numpy(global_targets[start:end].copy()).to(device)

        # Apply history matrices — include all history for deterministic evaluation.
        # Training uses stochastic dropout (rand >= 0.98) but that adds noise to eval.
        # Use the actual history inclusion flags from the data instead.
        include_history = b_gtargets[:, 36:41]
        h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)
        b_bin = torch.einsum("bijk,bil->bljk", b_bin, h_matrix)
        b_global = b_global * F.pad(
            include_history, (0, num_global_features - include_history.shape[1]), value=1.0
        )

        outputs = model(b_bin, b_global)
        policy_logits = outputs[0][0][:, 0, :]  # (batch, 362) — main policy head

        # Mask occupied positions and ko — channels 1,2 are own/opp stones
        stones = b_bin_np[:, 1, :, :] + b_bin_np[:, 2, :, :]
        legal = (1.0 - stones).reshape(b_bin_np.shape[0], -1)
        policy_mask = np.concatenate(
            [legal, np.ones((legal.shape[0], 1), dtype=np.float32)], axis=1
        )
        policy_mask_t = torch.from_numpy(policy_mask).to(device)
        policy_logits = policy_logits + (1.0 - policy_mask_t) * (-1e9)

        # Top-k predictions
        _, topk_indices = torch.topk(policy_logits, k=5, dim=1)
        topk_np = topk_indices.cpu().numpy()

        batch_gt = gt_moves[start:end]
        for i in range(end - start):
            gt = batch_gt[i]
            preds = topk_np[i]
            if preds[0] == gt:
                top1_correct += 1
            if gt in preds[:3]:
                top3_correct += 1
            if gt in preds[:5]:
                top5_correct += 1

        if (start // batch_size) % 50 == 0:
            done = min(end, n_samples)
            logger.info(f"  Progress: {done}/{n_samples} positions")

    results = {
        "n_samples": n_samples,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / n_samples,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / n_samples,
        "top5_correct": top5_correct,
        "top5_accuracy": top5_correct / n_samples,
    }
    return results


def print_results_table(all_results: list):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)

    # Header
    header = f"{'Model':<35} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10} {'Positions':>10}"
    print(header)
    print("-" * 80)

    for r in all_results:
        label = r["label"]
        res = r["results"]
        row = (
            f"{label:<35} "
            f"{res['top1_accuracy']:>9.2%} "
            f"{res['top3_accuracy']:>9.2%} "
            f"{res['top5_accuracy']:>9.2%} "
            f"{res['n_samples']:>10,}"
        )
        print(row)

    print("-" * 80)

    # Improvement row (if exactly 2 models)
    if len(all_results) == 2:
        base = all_results[0]["results"]
        fine = all_results[1]["results"]
        d1 = fine["top1_accuracy"] - base["top1_accuracy"]
        d3 = fine["top3_accuracy"] - base["top3_accuracy"]
        d5 = fine["top5_accuracy"] - base["top5_accuracy"]
        sign1 = "+" if d1 >= 0 else ""
        sign3 = "+" if d3 >= 0 else ""
        sign5 = "+" if d5 >= 0 else ""
        print(
            f"{'Improvement':<35} "
            f"{sign1}{d1:>8.2%} "
            f"{sign3}{d3:>8.2%} "
            f"{sign5}{d5:>8.2%}"
        )

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate KataGo checkpoints on a test set")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test-dir", type=Path, help="Directory of pre-converted NPZ test files")
    group.add_argument(
        "--sgf-dir", type=Path, help="Directory of SGF files (will be auto-converted)"
    )

    parser.add_argument(
        "--checkpoints",
        "-m",
        nargs="+",
        required=True,
        help="One or more checkpoint paths to evaluate",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        default=None,
        help="Labels for each checkpoint (default: filenames)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pos-len", type=int, default=19)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-swa", action="store_true", help="Use SWA model if available")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="KataGo config for SGF conversion (default: simple_training_config.cfg)",
    )
    parser.add_argument(
        "--only-player",
        type=str,
        default=None,
        help="Only evaluate positions where this player moved (matched against PB/PW in SGF)",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Resolve labels
    labels = args.labels
    if labels is None:
        labels = [Path(c).stem for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        parser.error("Number of --labels must match number of --checkpoints")

    # Get test data directory (convert SGFs if needed)
    if args.sgf_dir is not None:
        if not args.sgf_dir.is_dir():
            logger.error(f"SGF directory not found: {args.sgf_dir}")
            sys.exit(1)
        # Include player name in cache dir so different filters don't collide
        suffix = f"_{args.only_player}" if args.only_player else ""
        converted_dir = args.sgf_dir / f"npz{suffix}"
        if converted_dir.exists() and list(converted_dir.glob("*.npz")):
            logger.info(f"Using existing converted data in {converted_dir}")
        else:
            convert_sgfs_to_npz(args.sgf_dir, converted_dir, args.config, args.only_player)
        test_dir = converted_dir
    else:
        test_dir = args.test_dir
        if not test_dir.is_dir():
            logger.error(f"Test directory not found: {test_dir}")
            sys.exit(1)

    # Load test data once
    binary, global_input, policy_targets, global_targets, gt_moves = load_all_test_data(
        test_dir, args.pos_len
    )

    # Evaluate each checkpoint
    all_results = []
    for ckpt, label in zip(args.checkpoints, labels):
        if not Path(ckpt).exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)

        results = evaluate_checkpoint(
            ckpt_path=ckpt,
            binary_packed=binary,
            global_input=global_input,
            global_targets=global_targets,
            gt_moves=gt_moves,
            pos_len=args.pos_len,
            batch_size=args.batch_size,
            device=device,
            use_swa=args.use_swa,
        )
        all_results.append({"label": label, "results": results})

        logger.info(
            f"{label}: top-1={results['top1_accuracy']:.4f}, "
            f"top-3={results['top3_accuracy']:.4f}, "
            f"top-5={results['top5_accuracy']:.4f}"
        )

    # Print comparison table
    print_results_table(all_results)


if __name__ == "__main__":
    main()
