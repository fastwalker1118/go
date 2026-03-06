#!/usr/bin/env python3
"""
Ablation study: compare training's pacc1 logic vs eval accuracy.

Starts from the EXACT training validation logic, then removes factors one by
one to find what causes the accuracy gap.

Conditions tested:
  A) "train_val_exact"   – replicate training val exactly:
       stochastic history, random symmetry, weighted, soft-target argmax
  B) "no_weighting"      – same as A but global_weight=1
  C) "no_symmetry"       – same as A but no random symmetry
  D) "actual_history"    – same as A but use actual history flags (like eval)
  E) "no_weight_no_sym"  – remove both weighting and symmetry
  F) "eval_replica"      – eval script logic: actual history, no symmetry,
       unweighted, illegal-move masking, one-hot argmax from target

Usage:
    python ablation_accuracy.py \
        --test-dir KataGo/Training_Dataset/human_pros_softtarget/shuffled/val \
        --checkpoint KataGo/models/b18c384-allhuman-run1/checkpoint.ckpt
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from katago.train.load_model import load_model
from katago.train import modelconfigs
from katago.train.data_processing_pytorch import (
    build_history_matrices,
    apply_symmetry,
    apply_symmetry_policy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def unpack_binary_input(packed: np.ndarray, pos_len: int) -> np.ndarray:
    unpacked = np.unpackbits(packed, axis=2)
    unpacked = unpacked[:, :, : pos_len * pos_len]
    n, c, _ = unpacked.shape
    return unpacked.reshape(n, c, pos_len, pos_len).astype(np.float32)


def load_all_test_data(npz_dir: Path, pos_len: int):
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

    all_binary, all_global_input, all_policy_targets, all_global_targets = [], [], [], []

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

    logger.info(f"Loaded {binary.shape[0]} positions from {len(npz_files)} file(s)")
    return binary, global_input, policy_targets, global_targets


def apply_history_stochastic(
    batch_bin, batch_global, batch_gtargets, h_base, h_builder, num_global_features
):
    """Training's stochastic history dropout (98% keep probability)."""
    should_stop = torch.rand_like(batch_gtargets[:, 36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop, axis=1, dtype=torch.float32) <= 0.1).to(
        torch.float32
    )

    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)
    batch_bin = torch.einsum("bijk,bil->bljk", batch_bin, h_matrix)
    batch_global = batch_global * F.pad(
        include_history, (0, num_global_features - include_history.shape[1]), value=1.0
    )
    return batch_bin, batch_global


def apply_history_actual(
    batch_bin, batch_global, batch_gtargets, h_base, h_builder, num_global_features
):
    """Eval's actual-flag history (deterministic)."""
    include_history = batch_gtargets[:, 36:41]

    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)
    batch_bin = torch.einsum("bijk,bil->bljk", batch_bin, h_matrix)
    batch_global = batch_global * F.pad(
        include_history, (0, num_global_features - include_history.shape[1]), value=1.0
    )
    return batch_bin, batch_global


@torch.no_grad()
def run_ablation(
    ckpt_path: str,
    binary_packed: np.ndarray,
    global_input: np.ndarray,
    policy_targets: np.ndarray,
    global_targets: np.ndarray,
    pos_len: int,
    batch_size: int,
    device: torch.device,
    use_swa: bool = False,
    n_symmetry_samples: int = 1,
):
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

    # Conditions to track
    conditions = {
        "A_train_val_exact": dict(
            history="stochastic", symmetry=True, weighted=True, mask_illegal=False
        ),
        "B_no_weighting": dict(
            history="stochastic", symmetry=True, weighted=False, mask_illegal=False
        ),
        "C_no_symmetry": dict(
            history="stochastic", symmetry=False, weighted=True, mask_illegal=False
        ),
        "D_actual_history": dict(
            history="actual", symmetry=True, weighted=True, mask_illegal=False
        ),
        "E_no_wt_no_sym": dict(
            history="stochastic", symmetry=False, weighted=False, mask_illegal=False
        ),
        "F_eval_replica": dict(
            history="actual", symmetry=False, weighted=False, mask_illegal=True
        ),
    }

    # Accumulators: {condition: {weighted_correct, weight_sum, correct, total, raw_weighted_sum}}
    accum = {
        c: {
            "weighted_correct": 0.0,
            "weight_sum": 0.0,
            "correct": 0,
            "total": 0,
            "raw_weighted_sum": 0.0,
        }
        for c in conditions
    }

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bs = end - start

        b_bin_np = unpack_binary_input(binary_packed[start:end], pos_len)
        b_global_np = global_input[start:end].copy()
        b_gtargets_np = global_targets[start:end].copy()
        b_policy_np = policy_targets[start:end].copy()

        # Ground truth: argmax of target policy (channel 0)
        target_player = b_policy_np[:, 0, :]
        gt_moves = np.argmax(target_player, axis=1)

        # Weights from globalTargets
        global_weight = b_gtargets_np[:, 25]  # may be >1 for enriched data
        policy_weight = b_gtargets_np[:, 26]  # always 1.0

        for cond_name, cfg in conditions.items():
            # --- History ---
            b_bin_t = torch.from_numpy(b_bin_np.copy()).to(device)
            b_global_t = torch.from_numpy(b_global_np.copy()).to(device)
            b_gtargets_t = torch.from_numpy(b_gtargets_np.copy()).to(device)

            if cfg["history"] == "stochastic":
                b_bin_t, b_global_t = apply_history_stochastic(
                    b_bin_t, b_global_t, b_gtargets_t, h_base, h_builder, num_global_features
                )
            else:
                b_bin_t, b_global_t = apply_history_actual(
                    b_bin_t, b_global_t, b_gtargets_t, h_base, h_builder, num_global_features
                )

            # --- Symmetry ---
            # Training applies the SAME random symmetry to both input and target,
            # then compares argmax(pred) == argmax(rotated_target). Since rotation
            # is a permutation, this is equivalent to comparing in the original
            # frame. So symmetry should NOT affect accuracy — we just run model
            # on the rotated input and compare rotated pred with rotated target.
            if cfg["symmetry"]:
                symm = int(torch.randint(0, 8, (1,)).item())
                b_bin_sym = apply_symmetry(b_bin_t, symm).contiguous()
                # Also rotate the target to match
                b_policy_t = torch.from_numpy(b_policy_np.copy()).to(device)
                b_policy_sym = apply_symmetry_policy(b_policy_t, symm, pos_len)
                gt_moves_this = torch.argmax(b_policy_sym[:, 0, :], dim=1).cpu().numpy()
                outputs = model(b_bin_sym, b_global_t)
                policy_logits = outputs[0][0][:, 0, :]
            else:
                outputs = model(b_bin_t, b_global_t)
                policy_logits = outputs[0][0][:, 0, :]
                gt_moves_this = gt_moves

            # --- Illegal move masking (eval does this, training doesn't) ---
            if cfg["mask_illegal"]:
                # Use the unpacked binary features after history transform
                b_bin_eval = b_bin_t.cpu().numpy()
                stones = b_bin_eval[:, 1, :, :] + b_bin_eval[:, 2, :, :]
                legal = (1.0 - stones).reshape(bs, -1)
                policy_mask = np.concatenate([legal, np.ones((bs, 1), dtype=np.float32)], axis=1)
                policy_mask_t = torch.from_numpy(policy_mask).to(device)
                policy_logits = policy_logits + (1.0 - policy_mask_t) * (-1e9)

            # --- Compare predictions ---
            preds = torch.argmax(policy_logits, dim=1).cpu().numpy()
            correct = preds == gt_moves_this

            if cfg["weighted"]:
                w = global_weight * policy_weight
                accum[cond_name]["weighted_correct"] += float(np.sum(w * correct))
                accum[cond_name]["weight_sum"] += float(np.sum(w))
                # Training-style: sum(w * correct) / N (not / sum(w))
                accum[cond_name]["raw_weighted_sum"] += float(np.sum(w * correct))
            accum[cond_name]["correct"] += int(np.sum(correct))
            accum[cond_name]["total"] += bs

        if (start // batch_size) % 50 == 0:
            logger.info(f"  Progress: {min(end, n_samples)}/{n_samples}")

    return conditions, accum


def print_ablation_table(conditions, accum):
    print("\n" + "=" * 90)
    print("ABLATION STUDY: Training pacc1 vs Eval accuracy")
    print("=" * 90)
    print(
        f"{'Condition':<25} {'History':<12} {'Symm':<6} {'Wt':<6} {'Mask':<6} {'Wt/SumW':>10} {'Wt/N (train)':>13} {'Unweighted':>12}"
    )
    print("-" * 100)

    for cond_name, cfg in conditions.items():
        a = accum[cond_name]
        unweighted_acc = a["correct"] / a["total"] if a["total"] > 0 else 0
        if cfg["weighted"] and a["weight_sum"] > 0:
            weighted_proper = a["weighted_correct"] / a["weight_sum"]
            weighted_train_style = a["raw_weighted_sum"] / a["total"]
            wt_str = f"{weighted_proper:>9.2%}"
            wt_train_str = f"{weighted_train_style:>12.2%}"
        else:
            wt_str = "    N/A   "
            wt_train_str = "       N/A   "
        print(
            f"{cond_name:<25} "
            f"{cfg['history']:<12} "
            f"{'Y' if cfg['symmetry'] else 'N':<6} "
            f"{'Y' if cfg['weighted'] else 'N':<6} "
            f"{'Y' if cfg['mask_illegal'] else 'N':<6} "
            f"{wt_str} "
            f"{wt_train_str} "
            f"{unweighted_acc:>11.2%}"
        )

    print("-" * 90)

    # Explain the key comparisons
    a_train = accum["A_train_val_exact"]
    f_eval = accum["F_eval_replica"]
    if a_train["weight_sum"] > 0 and a_train["total"] > 0:
        train_pacc1_proper = a_train["weighted_correct"] / a_train["weight_sum"]
        train_pacc1_train_style = a_train["raw_weighted_sum"] / a_train["total"]
        eval_acc = f_eval["correct"] / f_eval["total"]
        mean_weight = a_train["weight_sum"] / a_train["total"]
        print(f"\n  Mean global_weight:                  {mean_weight:.4f}")
        print(f"  Training pacc1 (sum(w*c)/sum(w)):    {train_pacc1_proper:.4%}")
        print(f"  Training pacc1 (sum(w*c)/N — WandB): {train_pacc1_train_style:.4%}")
        print(f"  Eval accuracy (unweighted):          {eval_acc:.4%}")
        print(f"  Gap (WandB style vs eval):           {train_pacc1_train_style - eval_acc:+.4%}")

    # Show per-factor contribution
    print("\n  Factor contributions (compared to A_train_val_exact unweighted):")
    base = accum["A_train_val_exact"]["correct"] / accum["A_train_val_exact"]["total"]
    for cond_name in [
        "B_no_weighting",
        "C_no_symmetry",
        "D_actual_history",
        "E_no_wt_no_sym",
        "F_eval_replica",
    ]:
        acc = accum[cond_name]["correct"] / accum[cond_name]["total"]
        print(f"    {cond_name:<25}: {acc:.4%} (delta: {acc - base:+.4%})")

    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ablation: training pacc1 vs eval accuracy")
    parser.add_argument("--test-dir", type=Path, required=True, help="Directory of NPZ test files")
    parser.add_argument("--checkpoint", "-m", required=True, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pos-len", type=int, default=19)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-swa", action="store_true")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples for faster testing"
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    binary, global_input, policy_targets, global_targets = load_all_test_data(
        args.test_dir, args.pos_len
    )

    if args.max_samples is not None and args.max_samples < binary.shape[0]:
        logger.info(f"Limiting to {args.max_samples} samples")
        binary = binary[: args.max_samples]
        global_input = global_input[: args.max_samples]
        policy_targets = policy_targets[: args.max_samples]
        global_targets = global_targets[: args.max_samples]

    conditions, accum = run_ablation(
        ckpt_path=args.checkpoint,
        binary_packed=binary,
        global_input=global_input,
        policy_targets=policy_targets,
        global_targets=global_targets,
        pos_len=args.pos_len,
        batch_size=args.batch_size,
        device=device,
        use_swa=args.use_swa,
    )

    print_ablation_table(conditions, accum)


if __name__ == "__main__":
    main()
