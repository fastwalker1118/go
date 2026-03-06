#!/usr/bin/python3
import sys
import os

# Add paths for imports (now in training/ directory)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "python"))
sys.path.insert(0, os.path.join(root_dir, "training"))

import argparse
import traceback
import random
import math
import time
import logging
import contextlib
import json
import datetime
from datetime import timezone
import gc
import shutil
import glob
import numpy as np
import itertools
import copy
import atexit
from collections import defaultdict
from typing import Dict, List

# Add Training_Code to path for lora import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler, autocast

from katago.train import modelconfigs
from katago.train.model_pytorch import Model, ExtraOutputs, MetadataEncoder
from katago.train.metrics_pytorch import Metrics
from katago.utils.push_back_generator import PushBackGenerator
from katago.train import load_model
from katago.train import data_processing_pytorch
from katago.train.metrics_logging import accumulate_metrics, log_metrics, clear_metric_nonfinite

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Global loss tracking
loss_history = {
    "steps": [],
    "train_policy_loss": [],
    "train_total_loss": [],
    "val_policy_loss": [],
    "val_total_loss": [],
    "learning_rates": [],
}


def log_readable_loss(step, metrics, phase="train"):
    """Log losses in a readable format and store for plotting"""
    global loss_history

    # Extract key losses
    policy_loss = metrics.get("policy_loss", 0.0)
    total_loss = metrics.get("total_loss", 0.0)

    # Store in history (avoid duplicates)
    if step not in loss_history["steps"]:
        loss_history["steps"].append(step)
        loss_history["train_policy_loss"].append(0)
        loss_history["train_total_loss"].append(0)
        loss_history["val_policy_loss"].append(0)
        loss_history["val_total_loss"].append(0)
        loss_history["learning_rates"].append(0)

    idx = loss_history["steps"].index(step)
    if phase == "train":
        loss_history["train_policy_loss"][idx] = policy_loss
        loss_history["train_total_loss"][idx] = total_loss
    else:
        loss_history["val_policy_loss"][idx] = policy_loss
        loss_history["val_total_loss"][idx] = total_loss

    # Print readable format
    print(f"\n{'='*60}")
    print(f"STEP {step:,} - {phase.upper()} LOSSES")
    print(f"{'='*60}")
    print(f"Policy Loss:    {policy_loss:.6f}")
    print(f"Total Loss:     {total_loss:.6f}")
    if "value_loss" in metrics:
        print(f"Value Loss:     {metrics['value_loss']:.6f}")
    if "accuracy" in metrics:
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"{'='*60}\n")


def create_loss_plot(save_path="loss_convergence.png"):
    """Create a comprehensive loss convergence plot"""
    global loss_history

    if not loss_history["steps"]:
        print("No loss data to plot")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Policy-Only Fine-tuning Loss Convergence", fontsize=16, fontweight="bold")

        steps = loss_history["steps"]

        # Policy Loss Plot
        if any(loss_history["train_policy_loss"]):
            ax1.plot(
                steps,
                loss_history["train_policy_loss"],
                "b-",
                label="Train Policy Loss",
                linewidth=2,
            )
        if any(loss_history["val_policy_loss"]):
            ax1.plot(
                steps, loss_history["val_policy_loss"], "r-", label="Val Policy Loss", linewidth=2
            )
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Policy Loss")
        ax1.set_title("Policy Loss Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Total Loss Plot
        if any(loss_history["train_total_loss"]):
            ax2.plot(
                steps,
                loss_history["train_total_loss"],
                "b-",
                label="Train Total Loss",
                linewidth=2,
            )
        if any(loss_history["val_total_loss"]):
            ax2.plot(
                steps, loss_history["val_total_loss"], "r-", label="Val Total Loss", linewidth=2
            )
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("Total Loss")
        ax2.set_title("Total Loss Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Loss Difference Plot
        if len(steps) > 1 and any(loss_history["train_policy_loss"]):
            policy_diff = np.diff([x for x in loss_history["train_policy_loss"] if x > 0])
            if len(policy_diff) > 0:
                ax3.plot(
                    steps[1 : len(policy_diff) + 1],
                    policy_diff,
                    "g-",
                    label="Policy Loss Change",
                    linewidth=2,
                )
                ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Training Steps")
        ax3.set_ylabel("Loss Change")
        ax3.set_title("Loss Change (Convergence Rate)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Learning Rate Plot
        if any(loss_history["learning_rates"]):
            ax4.plot(
                steps, loss_history["learning_rates"], "orange", label="Learning Rate", linewidth=2
            )
            ax4.set_yscale("log")
        else:
            ax4.text(
                0.5,
                0.5,
                "Learning Rate\nData Not Available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
        ax4.set_xlabel("Training Steps")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("Learning Rate Schedule")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss convergence plot saved to: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create loss plot: {e}")


# POLICY-ONLY FINE-TUNING MODIFICATION
def freeze_except_policy(raw_model):
    """Freeze all parameters except policy head for fine-tuning"""
    print("=== POLICY-ONLY FINE-TUNING MODE ===")
    print("Freezing all parameters except policy head...")

    # Freeze everything first
    for param in raw_model.parameters():
        param.requires_grad = False

    # Unfreeze only policy head
    for param in raw_model.policy_head.parameters():
        param.requires_grad = True

    # Also unfreeze intermediate policy head if it exists
    if (
        hasattr(raw_model, "intermediate_policy_head")
        and raw_model.intermediate_policy_head is not None
    ):
        for param in raw_model.intermediate_policy_head.parameters():
            param.requires_grad = True
        print("Also unfroze intermediate policy head")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)"
    )
    print("=== This will train MUCH faster! ===")

    return raw_model


# POLICY + VALUE FINE-TUNING MODIFICATION
def freeze_except_policy_and_value(raw_model):
    """Freeze all parameters except policy head and core value head components"""
    print("=== POLICY + VALUE FINE-TUNING MODE ===")
    print("Freezing all parameters except policy head and core value components...")

    # Freeze everything first
    for param in raw_model.parameters():
        param.requires_grad = False

    # Unfreeze policy head
    for param in raw_model.policy_head.parameters():
        param.requires_grad = True

    # Unfreeze intermediate policy head if it exists
    if (
        hasattr(raw_model, "intermediate_policy_head")
        and raw_model.intermediate_policy_head is not None
    ):
        for param in raw_model.intermediate_policy_head.parameters():
            param.requires_grad = True
        print("Also unfroze intermediate policy head")

    # Unfreeze core value head components (Phase 1)
    value_head = raw_model.value_head

    # Main value prediction components
    for param in value_head.linear_valuehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_miscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_moremiscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear2.parameters():
        param.requires_grad = True

    print("Unfroze core value head components:")
    print("  - linear_valuehead (main win/loss/draw prediction)")
    print("  - linear_miscvaluehead (auxiliary value predictions)")
    print("  - linear_moremiscvaluehead (more auxiliary predictions)")
    print("  - linear2 (dense processing layer)")

    # Also unfreeze intermediate value head if it exists
    if (
        hasattr(raw_model, "intermediate_value_head")
        and raw_model.intermediate_value_head is not None
    ):
        ivalue_head = raw_model.intermediate_value_head
        for param in ivalue_head.linear_valuehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_miscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_moremiscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear2.parameters():
            param.requires_grad = True
        print("Also unfroze intermediate value head core components")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    )
    print("=== Policy + Value training will provide MCTS alignment! ===")

    return raw_model


# MINIMAL LAYERS + HEADS FINE-TUNING MODIFICATION
def freeze_except_final_layers_and_heads(raw_model):
    """Freeze all parameters except final normalization, policy head, and core value head components (~200K params)"""
    print("=== MINIMAL LAYERS + HEADS FINE-TUNING MODE ===")
    print("Freezing all parameters except final norm and heads (targeting ~200K params)...")

    # Freeze everything first
    for param in raw_model.parameters():
        param.requires_grad = False

    # Unfreeze policy head
    for param in raw_model.policy_head.parameters():
        param.requires_grad = True

    # Unfreeze intermediate policy head if it exists
    if (
        hasattr(raw_model, "intermediate_policy_head")
        and raw_model.intermediate_policy_head is not None
    ):
        for param in raw_model.intermediate_policy_head.parameters():
            param.requires_grad = True
        print("Also unfroze intermediate policy head")

    # Unfreeze core value head components
    value_head = raw_model.value_head
    for param in value_head.linear_valuehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_miscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_moremiscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear2.parameters():
        param.requires_grad = True

    # Unfreeze intermediate value head if it exists
    if (
        hasattr(raw_model, "intermediate_value_head")
        and raw_model.intermediate_value_head is not None
    ):
        ivalue_head = raw_model.intermediate_value_head
        for param in ivalue_head.linear_valuehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_miscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_moremiscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear2.parameters():
            param.requires_grad = True
        print("Also unfroze intermediate value head core components")

    # CAREFULLY unfreeze ONLY final normalization + heads (targeting ~200K parameters)
    # Skip CNN blocks entirely to keep parameter count low

    # Unfreeze trunk final normalization (crucial for feature adaptation)
    if hasattr(raw_model, "norm_trunkfinal"):
        for param in raw_model.norm_trunkfinal.parameters():
            param.requires_grad = True
        print("  - Unfroze norm_trunkfinal (final normalization layer)")

    print("Unfroze components:")
    print("  - Policy heads (move prediction)")
    print("  - Core value head components")
    print("  - Final trunk normalization (feature scaling)")
    print("  - NO CNN blocks unfrozen (keeping parameter count minimal)")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    )

    # Estimate parameter counts for safety check
    if trainable_params > 500000:  # More than 500K parameters
        print("⚠️  WARNING: Training >500K parameters - may require lower learning rate!")
        print("⚠️  Consider using policy+value mode instead for fewer parameters")
    elif trainable_params < 150000:  # Less than 150K parameters
        print("✅ Minimal parameter count - safe for aggressive fine-tuning")
    else:
        print("✅ Parameter count looks good for fine-tuning")

    print("=== Minimal layers + heads training for targeted adaptation! ===")

    return raw_model


def freeze_except_final_cnn_block_and_heads(raw_model):
    """Freeze all parameters except final CNN block, normalization, and heads (~3M params)"""
    print("=== FINAL CNN BLOCK + HEADS FINE-TUNING MODE ===")
    print("Freezing all parameters except final CNN block and heads (targeting ~3M params)...")

    # Freeze everything first
    for param in raw_model.parameters():
        param.requires_grad = False

    # Unfreeze final CNN block (last block, works for b18/b28/etc)
    last_block_idx = len(raw_model.blocks) - 1
    for name, param in raw_model.named_parameters():
        if name.startswith(f"blocks.{last_block_idx}."):
            param.requires_grad = True

    # Unfreeze final normalization
    if hasattr(raw_model, "norm_trunkfinal"):
        for param in raw_model.norm_trunkfinal.parameters():
            param.requires_grad = True

    # Unfreeze policy head
    for param in raw_model.policy_head.parameters():
        param.requires_grad = True

    # Unfreeze intermediate policy head if it exists
    if (
        hasattr(raw_model, "intermediate_policy_head")
        and raw_model.intermediate_policy_head is not None
    ):
        for param in raw_model.intermediate_policy_head.parameters():
            param.requires_grad = True

    # Unfreeze core value head components
    value_head = raw_model.value_head
    for param in value_head.linear_valuehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_miscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_moremiscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear2.parameters():
        param.requires_grad = True

    # Unfreeze intermediate value head if it exists
    if (
        hasattr(raw_model, "intermediate_value_head")
        and raw_model.intermediate_value_head is not None
    ):
        ivalue_head = raw_model.intermediate_value_head
        for param in ivalue_head.linear_valuehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_miscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_moremiscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear2.parameters():
            param.requires_grad = True

    print("Unfroze components:")
    print(f"  - Final CNN block (blocks.{last_block_idx}.*) - full feature learning")
    print("  - Policy heads (move prediction)")
    print("  - Core value head components")
    print("  - Final trunk normalization")
    print("  - Earlier blocks remain frozen (preserving base features)")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    )

    # Parameter count warnings
    if trainable_params > 2000000:  # More than 2M parameters
        print("⚠️  WARNING: Training >2M parameters - use lower learning rate (0.1x or less)!")
        print("⚠️  This mode allows the final CNN block to learn new features")
    else:
        print("✅ Parameter count manageable for careful fine-tuning")

    print("=== Final CNN block training - enables new feature learning! ===")

    return raw_model


def freeze_value_head_only(raw_model):
    """Freeze ALL value head parameters (main + intermediate if present). Train everything else."""
    print("=== TRAIN TRUNK + POLICY (NO VALUE) MODE ===")
    print("Training entire ResNet trunk and Policy head.")
    print("Freezing ONLY the Value head...")

    # Unfreeze everything first (default state)
    for param in raw_model.parameters():
        param.requires_grad = True

    # Freeze main value head entirely
    if hasattr(raw_model, "value_head") and raw_model.value_head is not None:
        for param in raw_model.value_head.parameters():
            param.requires_grad = False

    # Freeze intermediate value head if it exists
    if (
        hasattr(raw_model, "intermediate_value_head")
        and raw_model.intermediate_value_head is not None
    ):
        for param in raw_model.intermediate_value_head.parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
    )

    return raw_model


def compute_policy_only_loss(metrics, raw_model, soft_policy_weight_scale: float):
    """Compute a policy-only loss from the metrics dict (avoids value/aux heads)."""
    version = raw_model.config["version"]
    if version <= 11:
        policy_opt_loss_scale = 1.0
        long_policy_opt_loss_scale = 0.0
        short_policy_opt_loss_scale = 0.0
    else:
        policy_opt_loss_scale = 0.93
        long_policy_opt_loss_scale = 0.10
        short_policy_opt_loss_scale = 0.20

    return (
        metrics["p0loss_sum"] * policy_opt_loss_scale
        + metrics["p1loss_sum"]
        + metrics["p0softloss_sum"] * soft_policy_weight_scale
        + metrics["p1softloss_sum"] * soft_policy_weight_scale
        + metrics["p0lopt_sum"] * long_policy_opt_loss_scale
        + metrics["p0sopt_sum"] * short_policy_opt_loss_scale
    )


# HANDLE COMMAND AND ARGS -------------------------------------------------------------------

if __name__ == "__main__":

    description = """
    Train neural net on Go positions from npz files of batches from selfplay.
    MODIFIED for policy-only fine-tuning on human games.
    """

    parser = argparse.ArgumentParser(description=description, add_help=False)
    required_args = parser.add_argument_group("required arguments")
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )

    required_args.add_argument(
        "-traindir", help="Dir to write to for recording training results", required=True
    )
    required_args.add_argument(
        "-datadir",
        help="Directory with a train and val subdir of npz data, output by shuffle.py",
        required=False,
    )
    required_args.add_argument(
        "-latestdatadir",
        help="Use the latest subdirectory within this dir as the datadir, periodically checking for most recent",
        required=False,
    )
    optional_args.add_argument(
        "-exportdir", help="Directory to export models periodically", required=False
    )
    optional_args.add_argument(
        "-exportprefix", help="Prefix to append to names of models", required=False
    )
    optional_args.add_argument(
        "-initial-checkpoint",
        help="If no training checkpoint exists, initialize from this checkpoint",
        required=False,
    )

    required_args.add_argument(
        "-pos-len",
        help="Spatial edge length of expected training data, e.g. 19 for 19x19 Go",
        type=int,
        required=True,
    )
    required_args.add_argument(
        "-batch-size", help="Per-GPU batch size to use for training", type=int, required=True
    )
    optional_args.add_argument(
        "-samples-per-epoch",
        help="Number of data samples to consider as one epoch",
        type=int,
        required=False,
    )
    optional_args.add_argument(
        "-model-kind", help="String name for what model config to use", required=False
    )
    optional_args.add_argument(
        "-lr-scale", help="LR multiplier on the hardcoded schedule", type=float, required=False
    )
    optional_args.add_argument(
        "-lr-scale-auto", help="LR auto scaling", required=False, action="store_true"
    )
    optional_args.add_argument(
        "-gnorm-clip-scale",
        help="Multiplier on gradient clipping threshold",
        type=float,
        required=False,
    )
    optional_args.add_argument(
        "-sub-epochs",
        help="Reload training data up to this many times per epoch",
        type=int,
        default=1,
        required=False,
    )
    optional_args.add_argument(
        "-swa-period-samples",
        help="How frequently to average an SWA sample, in samples",
        type=float,
        required=False,
    )
    optional_args.add_argument(
        "-swa-scale",
        help="Number of samples to average in expectation together for SWA",
        type=float,
        required=False,
    )
    optional_args.add_argument(
        "-lookahead-k", help="Use lookahead optimizer", type=int, default=6, required=False
    )
    optional_args.add_argument(
        "-lookahead-alpha", help="Use lookahead optimizer", type=float, default=0.5, required=False
    )
    optional_args.add_argument(
        "-lookahead-print",
        help="Only print on lookahead syncs",
        required=False,
        action="store_true",
    )

    optional_args.add_argument(
        "-multi-gpus", help="Use multiple gpus, comma-separated device ids", required=False
    )
    optional_args.add_argument(
        "-use-fp16", help="Use fp16 training", required=False, action="store_true"
    )

    optional_args.add_argument(
        "-epochs-per-export",
        help="Export model once every this many epochs",
        type=int,
        required=False,
    )
    optional_args.add_argument(
        "-export-prob", help="Export model with this probablity", type=float, required=False
    )
    optional_args.add_argument(
        "-max-epochs-this-instance",
        help="Terminate training after this many more epochs",
        type=int,
        required=False,
    )
    optional_args.add_argument(
        "-max-training-samples",
        help="Terminate training after about this many training steps in samples",
        type=int,
        required=False,
    )
    optional_args.add_argument(
        "-sleep-seconds-per-epoch", help="Sleep this long between epochs", type=int, required=False
    )
    optional_args.add_argument(
        "-max-train-bucket-per-new-data",
        help="When data added, add this many train rows per data row to bucket",
        type=float,
        required=False,
    )
    optional_args.add_argument(
        "-max-train-bucket-size",
        help="Approx total number of train rows allowed if data stops",
        type=float,
        required=False,
    )
    optional_args.add_argument(
        "-max-train-steps-since-last-reload",
        help="Approx total of training allowed if shuffling stops",
        type=float,
        required=False,
    )
    optional_args.add_argument(
        "-stop-when-train-bucket-limited",
        help="Terminate due to train bucket rather than waiting for more",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-max-val-samples",
        help="Approx max of validation samples per epoch",
        type=int,
        required=False,
    )
    optional_args.add_argument(
        "-randomize-val",
        help="Randomize order of validation files",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-no-export", help="Do not export models", required=False, action="store_true"
    )
    optional_args.add_argument(
        "-no-repeat-files",
        help="Track what shuffled data was used and do not repeat, even when killed and resumed",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-quit-if-no-data",
        help="If no data, quit instead of waiting for data",
        required=False,
        action="store_true",
    )

    optional_args.add_argument("-gnorm-stats-debug", required=False, action="store_true")

    optional_args.add_argument(
        "-brenorm-avg-momentum",
        type=float,
        help="Set brenorm running avg rate to this value",
        required=False,
    )
    optional_args.add_argument(
        "-brenorm-target-rmax",
        type=float,
        help="Gradually adjust brenorm rmax to this value",
        required=False,
    )
    optional_args.add_argument(
        "-brenorm-target-dmax",
        type=float,
        help="Gradually adjust brenorm dmax to this value",
        required=False,
    )
    optional_args.add_argument(
        "-brenorm-adjustment-scale",
        type=float,
        help="How many samples to adjust brenorm params all but 1/e of the way to target",
        required=False,
    )

    optional_args.add_argument(
        "-soft-policy-weight-scale",
        type=float,
        default=8.0,
        help="Soft policy loss coeff",
        required=False,
    )
    optional_args.add_argument(
        "-disable-optimistic-policy",
        help="Disable optimistic policy",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-meta-kata-only-soft-policy",
        help="Mask soft policy on non-kata rows using sgfmeta",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-value-loss-scale",
        type=float,
        default=0.6,
        help="Additional value loss coeff",
        required=False,
    )
    optional_args.add_argument(
        "-td-value-loss-scales",
        type=str,
        default="0.6,0.6,0.6",
        help="Additional td value loss coeffs, 3 comma separated values",
        required=False,
    )
    optional_args.add_argument(
        "-seki-loss-scale",
        type=float,
        default=1.0,
        help="Additional seki loss coeff",
        required=False,
    )
    optional_args.add_argument(
        "-variance-time-loss-scale",
        type=float,
        default=1.0,
        help="Additional variance time loss coeff",
        required=False,
    )

    optional_args.add_argument(
        "-main-loss-scale", type=float, help="Loss factor scale for main head", required=False
    )
    optional_args.add_argument(
        "-intermediate-loss-scale",
        type=float,
        help="Loss factor scale for intermediate head",
        required=False,
    )

    # Wandb
    optional_args.add_argument(
        "-wandb-project", help="Wandb project name (enables wandb logging)", required=False
    )
    optional_args.add_argument("-wandb-run-name", help="Wandb run name", required=False)

    # Fine-tuning options
    optional_args.add_argument(
        "-freeze-except-policy-and-value",
        help="Freeze all parameters except policy head and core value head components",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-freeze-except-final-layers-and-heads",
        help="Freeze all parameters except final CNN layers, policy head, and core value head components",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-freeze-except-final-cnn-block-and-heads",
        help="Freeze all parameters except final CNN block (27), normalization, and heads (~3M params)",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-freeze-value-head",
        help="Freeze ONLY the value head, train trunk and policy head",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-policy-loss-only",
        help="Use policy-only loss (ignore all value/aux losses)",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-reset-train-state",
        help="When initializing from a checkpoint, ignore its train_state/optimizer/running_metrics so LR schedule restarts cleanly",
        required=False,
        action="store_true",
    )

    # Finetuning / small-dataset controls
    optional_args.add_argument(
        "-base-per-sample-lr",
        type=float,
        default=0.00003,
        help="Base learning rate per sample before warmup/lr-scale",
        required=False,
    )
    optional_args.add_argument(
        "-disable-warmup",
        help="Disable LR warmup (useful for small-dataset finetuning)",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-disable-swa",
        help="Disable SWA model averaging (useful for small-dataset finetuning)",
        required=False,
        action="store_true",
    )

    # KL regularization against a base model (prevents overfitting for small datasets)
    optional_args.add_argument(
        "-kl-base-checkpoint",
        help="Path to base model checkpoint for KL regularization. When set, adds KL(finetuned || base) penalty to loss.",
        required=False,
    )
    optional_args.add_argument(
        "-kl-beta",
        type=float,
        default=1.0,
        help="KL regularization strength (default 1.0). Higher = stay closer to base model.",
        required=False,
    )

    # LoRA (Low-Rank Adaptation) for conv layers
    optional_args.add_argument(
        "-use-lora",
        help="Use LoRA on conv layers instead of full fine-tuning",
        required=False,
        action="store_true",
    )
    optional_args.add_argument(
        "-lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default 8). Lower = fewer params.",
        required=False,
    )
    optional_args.add_argument(
        "-lora-alpha",
        type=float,
        default=1.0,
        help="LoRA scaling factor (default 1.0)",
        required=False,
    )
    optional_args.add_argument(
        "-lora-blocks",
        type=str,
        default=None,
        help="Comma-separated block indices for LoRA (default: last 2 blocks)",
        required=False,
    )

    # Move history modules (sequential move reasoning)
    optional_args.add_argument(
        "-use-move-history",
        type=str,
        default=None,
        help="Move history module type: embedding, lstm, or combined",
        required=False,
    )
    optional_args.add_argument(
        "-history-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for move history module (default 128)",
        required=False,
    )

    # Label smoothing for policy targets
    optional_args.add_argument(
        "-label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0 = off, 0.1 = mix 10%% uniform)",
        required=False,
    )

    args = vars(parser.parse_args())


def get_longterm_checkpoints_dir(traindir):
    return os.path.join(traindir, "longterm_checkpoints")


def make_dirs(args):
    traindir = args["traindir"]
    exportdir = args["exportdir"]

    if not os.path.exists(traindir):
        os.makedirs(traindir)
    if exportdir is not None and not os.path.exists(exportdir):
        os.makedirs(exportdir)

    longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)
    if not os.path.exists(longterm_checkpoints_dir):
        os.makedirs(longterm_checkpoints_dir)


def multiprocessing_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23456"
    logging.info("Running torch.distributed.init_process_group")
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(
        f"Returned from torch.distributed.init_process_group, my rank = {rank}, world_size={world_size}"
    )


def multiprocessing_cleanup():
    torch.distributed.destroy_process_group()


def main(rank: int, world_size: int, args, multi_gpu_device_ids, readpipes, writepipes, barrier):
    traindir = args["traindir"]
    datadir = args["datadir"]
    latestdatadir = args["latestdatadir"]
    exportdir = args["exportdir"]
    exportprefix = args["exportprefix"]
    initial_checkpoint = args["initial_checkpoint"]

    pos_len = args["pos_len"]
    batch_size = args["batch_size"]
    samples_per_epoch = args["samples_per_epoch"]
    model_kind = args["model_kind"]
    lr_scale = args["lr_scale"]
    lr_scale_auto = args["lr_scale_auto"]
    base_per_sample_lr = args["base_per_sample_lr"]
    disable_warmup = args["disable_warmup"]
    disable_swa = args["disable_swa"]
    gnorm_clip_scale = args["gnorm_clip_scale"]
    sub_epochs = args["sub_epochs"]
    swa_period_samples = args["swa_period_samples"]
    swa_scale = args["swa_scale"]
    lookahead_k = args["lookahead_k"]
    lookahead_alpha = args["lookahead_alpha"]
    lookahead_print = args["lookahead_print"]
    freeze_policy_and_value = args["freeze_except_policy_and_value"]
    freeze_final_layers_and_heads = args["freeze_except_final_layers_and_heads"]
    freeze_final_cnn_block_and_heads = args["freeze_except_final_cnn_block_and_heads"]
    freeze_value_head = args["freeze_value_head"]
    policy_loss_only = args["policy_loss_only"]
    reset_train_state = args["reset_train_state"]

    # Validate freeze options - only one can be used at a time
    freeze_options_count = sum(
        [
            freeze_policy_and_value,
            freeze_final_layers_and_heads,
            freeze_final_cnn_block_and_heads,
            freeze_value_head,
        ]
    )
    if freeze_options_count > 1:
        raise ValueError("Error: Only one freeze option can be specified at a time")

    use_fp16 = args["use_fp16"]

    epochs_per_export = args["epochs_per_export"]
    export_prob = args["export_prob"]
    max_epochs_this_instance = args["max_epochs_this_instance"]
    max_training_samples = args["max_training_samples"]
    sleep_seconds_per_epoch = args["sleep_seconds_per_epoch"]
    max_train_bucket_per_new_data = args["max_train_bucket_per_new_data"]
    max_train_bucket_size = args["max_train_bucket_size"]
    max_train_steps_since_last_reload = args["max_train_steps_since_last_reload"]
    stop_when_train_bucket_limited = args["stop_when_train_bucket_limited"]
    max_val_samples = args["max_val_samples"]
    randomize_val = args["randomize_val"]
    no_export = args["no_export"]
    no_repeat_files = args["no_repeat_files"]
    quit_if_no_data = args["quit_if_no_data"]

    gnorm_stats_debug = args["gnorm_stats_debug"]

    brenorm_target_rmax = args["brenorm_target_rmax"]
    brenorm_target_dmax = args["brenorm_target_dmax"]
    brenorm_avg_momentum = args["brenorm_avg_momentum"]
    brenorm_adjustment_scale = args["brenorm_adjustment_scale"]

    soft_policy_weight_scale = args["soft_policy_weight_scale"]
    disable_optimistic_policy = args["disable_optimistic_policy"]
    meta_kata_only_soft_policy = args["meta_kata_only_soft_policy"]
    value_loss_scale = args["value_loss_scale"]
    td_value_loss_scales = [float(x) for x in args["td_value_loss_scales"].split(",")]
    seki_loss_scale = args["seki_loss_scale"]
    variance_time_loss_scale = args["variance_time_loss_scale"]

    main_loss_scale = args["main_loss_scale"]
    intermediate_loss_scale = args["intermediate_loss_scale"]

    kl_base_checkpoint = args["kl_base_checkpoint"]
    kl_beta = args["kl_beta"]

    use_lora = args["use_lora"]
    lora_rank = args["lora_rank"]
    lora_alpha = args["lora_alpha"]
    lora_blocks_str = args["lora_blocks"]
    lora_blocks = None
    if lora_blocks_str is not None:
        lora_blocks = [int(x) for x in lora_blocks_str.split(",")]

    use_move_history = args["use_move_history"]
    history_hidden_dim = args["history_hidden_dim"]
    label_smoothing = args["label_smoothing"]

    if lr_scale is None:
        lr_scale = 1.0
    if lr_scale_auto:
        assert lr_scale == 1.0, "Cannot specify both lr_scale and lr_scale_auto"

    assert not (
        not datadir and not latestdatadir
    ), "Must specify one of -datadir and -latestdatadir"
    assert not (datadir and latestdatadir), "Must specify only one of -datadir and -latestdatadir"

    if samples_per_epoch is None:
        samples_per_epoch = 1000000
    if max_train_bucket_size is None:
        max_train_bucket_size = 1.0e30
    if epochs_per_export is None:
        epochs_per_export = 1
    if disable_swa:
        # SWA is either fully enabled (both fields set) or fully disabled (both None).
        swa_period_samples = None
        swa_scale = None
    else:
        if swa_period_samples is None:
            swa_period_samples = max(1, samples_per_epoch // 2)
        if swa_scale is None:
            swa_scale = 8

    assert lookahead_alpha > 0.0 and lookahead_alpha <= 1.0
    if lookahead_alpha >= 1.0:  # 1.0 means to disable lookahead optimizer
        lookahead_alpha = None
        lookahead_k = None

    longterm_checkpoints_dir = get_longterm_checkpoints_dir(traindir)

    assert (swa_period_samples is None) == (swa_scale is None)
    assert (lookahead_k is None) == (lookahead_alpha is None)

    # SET UP LOGGING -------------------------------------------------------------

    logging.root.handlers = []
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(traindir, f"train{rank}.log"), mode="a"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(traindir, f"train{rank}.log"), mode="a"),
            ],
        )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # FIGURE OUT MULTIGPU ------------------------------------------------------------
    if world_size > 1:
        multiprocessing_setup(rank, world_size)
        atexit.register(multiprocessing_cleanup)
        assert torch.cuda.is_available()

    if torch.cuda.is_available():
        my_gpu_id = multi_gpu_device_ids[rank]
        torch.cuda.set_device(my_gpu_id)
        logging.info("Using GPU device: " + torch.cuda.get_device_name())
        device = torch.device("cuda", my_gpu_id)
    elif torch.backends.mps.is_available():  # Check for Apple Metal Performance Shaders
        my_gpu_id = multi_gpu_device_ids[rank]
        logging.info("Using MPS device")
        device = torch.device("mps", my_gpu_id)
    else:
        logging.warning("WARNING: No GPU, using CPU")
        device = torch.device("cpu")

    seed = int.from_bytes(os.urandom(7), sys.byteorder)
    logging.info(f"Seeding torch with {seed}")
    torch.manual_seed(seed)

    # LOAD MODEL ---------------------------------------------------------------------

    def lr_scale_auto_factor(train_state):
        if not lr_scale_auto:
            return 1.0

        if train_state["global_step_samples"] < 200_000_000:
            return 8.00
        if train_state["global_step_samples"] < 400_000_000:
            return 4.00
        if train_state["global_step_samples"] < 500_000_000:
            return 2.00
        if train_state["global_step_samples"] < 550_000_000:
            return 1.00
        if train_state["global_step_samples"] < 600_000_000:
            return 0.50
        if train_state["global_step_samples"] < 650_000_000:
            return 0.25
        return 0.25

    def get_checkpoint_path():
        return os.path.join(traindir, "checkpoint.ckpt")

    def get_checkpoint_prev_path(i):
        return os.path.join(traindir, f"checkpoint_prev{i}.ckpt")

    NUM_SHORTTERM_CHECKPOINTS_TO_KEEP = 4

    def save(
        ddp_model,
        swa_model,
        optimizer,
        metrics_obj,
        running_metrics,
        train_state,
        last_val_metrics,
        path=None,
    ):
        if gnorm_stats_debug:
            logging.warning("Skipping save since debugging gnorm stats")
            return
        if rank == 0:
            state_dict = {}
            state_dict["model"] = ddp_model.state_dict()
            state_dict["optimizer"] = optimizer.state_dict()
            state_dict["metrics"] = metrics_obj.state_dict()
            state_dict["running_metrics"] = running_metrics
            state_dict["train_state"] = train_state
            state_dict["last_val_metrics"] = last_val_metrics
            state_dict["config"] = model_config

            if swa_model is not None:
                state_dict["swa_model"] = swa_model.state_dict()

            if path is not None:
                logging.info("Saving checkpoint: " + path)
                torch.save(state_dict, path + ".tmp")
                time.sleep(1)
                os.replace(path + ".tmp", path)
            else:
                logging.info("Saving checkpoint: " + get_checkpoint_path())
                for i in reversed(range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP - 1)):
                    if os.path.exists(get_checkpoint_prev_path(i)):
                        os.replace(get_checkpoint_prev_path(i), get_checkpoint_prev_path(i + 1))
                if os.path.exists(get_checkpoint_path()):
                    shutil.copy(get_checkpoint_path(), get_checkpoint_prev_path(0))
                torch.save(state_dict, get_checkpoint_path() + ".tmp")
                os.replace(get_checkpoint_path() + ".tmp", get_checkpoint_path())

    def get_weight_decay(
        raw_model, lr_scale, warmup_scale, train_state, running_metrics, group_name
    ):
        lr_scale_with_auto = lr_scale * lr_scale_auto_factor(train_state)
        if raw_model.get_norm_kind() == "fixup" or raw_model.get_norm_kind() == "fixscale":
            if (
                group_name == "input"
                or group_name == "normal"
                or group_name == "normal_gamma"
                or group_name == "output"
            ):
                return 0.000001 * world_size * batch_size / 256.0
            elif group_name == "input_noreg" or group_name == "noreg":
                return 0.00000001 * world_size * batch_size / 256.0
            elif group_name == "output_noreg":
                return 0.00000001 * world_size * batch_size / 256.0
            else:
                assert False
        elif (
            raw_model.get_norm_kind() == "bnorm"
            or raw_model.get_norm_kind() == "brenorm"
            or raw_model.get_norm_kind() == "fixbrenorm"
            or raw_model.get_norm_kind() == "fixscaleonenorm"
        ):
            if group_name == "input" or group_name == "normal" or group_name == "normal_gamma":
                adaptive_scale = 1.0
                if "sums" in running_metrics and "norm_normal_batch" in running_metrics["sums"]:
                    norm_normal_batch = (
                        running_metrics["sums"]["norm_normal_batch"]
                        / running_metrics["weights"]["norm_normal_batch"]
                    )
                    baseline = train_state["modelnorm_normal_baseline"]
                    ratio = norm_normal_batch / (baseline + 1e-30)
                    # Adaptive weight decay keeping model norm around the baseline level so that batchnorm effective lr is held constant
                    # throughout training, covering a range of 16x from bottom to top.
                    adaptive_scale = math.pow(2.0, 2.0 * math.tanh(math.log(ratio + 1e-30) * 1.5))

                # Batch norm gammas can be regularized a bit less, doing them just as much empirically seemed to be a bit more unstable
                gamma_scale = 0.125 if group_name == "normal_gamma" else 1.0

                # The theoretical scaling for keeping us confined to a surface of equal model norm should go proportionally with lr_scale.
                # because the strength of drift away from that surface goes as lr^2 and weight decay itself is scaled by lr, so we need
                # one more factor of lr to make weight decay strength equal drift strength.
                # However, at low lr it tends to be the case that gradient norm increases slightly
                # while at high lr it tends to be the case that gradient norm decreases, which means drift strength scales a bit slower
                # than expected.
                # So we scale sublinearly with lr_scale so as to slightly preadjust to this effect.
                # Adaptive scale should then help keep us there thereafter.
                return (
                    0.00125
                    * world_size
                    * batch_size
                    / 256.0
                    * math.pow(lr_scale_with_auto * warmup_scale, 0.75)
                    * adaptive_scale
                    * gamma_scale
                )
            elif group_name == "output":
                return 0.000001 * world_size * batch_size / 256.0
            elif group_name == "input_noreg" or group_name == "noreg":
                return (
                    0.000001
                    * world_size
                    * batch_size
                    / 256.0
                    * math.pow(lr_scale_with_auto * warmup_scale, 0.75)
                )
            elif group_name == "output_noreg":
                return 0.00000001 * world_size * batch_size / 256.0
            else:
                assert False
        else:
            assert False

    def get_param_groups(raw_model, train_state, running_metrics):
        reg_dict: Dict[str, List] = {}
        raw_model.add_reg_dict(reg_dict)
        param_groups = []
        num_reg_dict_params = 0
        for group_name in [
            "input",
            "input_noreg",
            "normal",
            "normal_gamma",
            "noreg",
            "output",
            "output_noreg",
        ]:
            if len(reg_dict[group_name]) > 0:
                param_groups.append(
                    {
                        "params": reg_dict[group_name],
                        "weight_decay": get_weight_decay(
                            raw_model,
                            lr_scale,
                            warmup_scale=1.0,
                            train_state=train_state,
                            running_metrics=running_metrics,
                            group_name=group_name,
                        ),
                        "group_name": group_name,
                    }
                )
            num_reg_dict_params += len(reg_dict[group_name])
        num_params = len(list(raw_model.parameters()))
        assert (
            num_params == num_reg_dict_params
        ), "Reg dict does not have entries for all params in model"
        return param_groups

    def load():
        if not os.path.exists(get_checkpoint_path()):
            logging.info("No preexisting checkpoint found at: " + get_checkpoint_path())
            for i in range(NUM_SHORTTERM_CHECKPOINTS_TO_KEEP):
                if os.path.exists(get_checkpoint_prev_path(i)):
                    raise Exception(
                        f"No preexisting checkpoint found, but {get_checkpoint_prev_path(i)} exists, something is wrong with the training dir"
                    )

            if initial_checkpoint is not None:
                if os.path.exists(initial_checkpoint):
                    logging.info(f"Using initial checkpoint: {initial_checkpoint}")
                    path_to_load_from = initial_checkpoint
                else:
                    raise Exception(
                        "No preexisting checkpoint found, initial checkpoint provided is invalid: {initial_checkpoint}"
                    )
            else:
                path_to_load_from = None
        else:
            path_to_load_from = get_checkpoint_path()

        if path_to_load_from is None:
            logging.info("Initializing new model!")
            assert (
                model_kind is not None
            ), "Model kind is none or unspecified but the model is being created fresh"
            model_config = modelconfigs.config_of_name[model_kind]
            logging.info(str(model_config))
            raw_model = Model(model_config, pos_len)
            raw_model.initialize()

            raw_model.to(device)
            # Apply fine-tuning
            if use_lora:
                from lora import apply_lora_to_model

                raw_model, _ = apply_lora_to_model(
                    raw_model, target_blocks=lora_blocks, rank=lora_rank, alpha=lora_alpha
                )
            elif freeze_final_cnn_block_and_heads:
                raw_model = freeze_except_final_cnn_block_and_heads(raw_model)
            elif freeze_final_layers_and_heads:
                raw_model = freeze_except_final_layers_and_heads(raw_model)
            elif freeze_policy_and_value:
                raw_model = freeze_except_policy_and_value(raw_model)
            elif freeze_value_head:
                raw_model = freeze_value_head_only(raw_model)
            else:
                raw_model = freeze_except_policy(raw_model)
            if use_move_history is not None:
                from move_history import apply_move_history_to_model

                raw_model, _ = apply_move_history_to_model(
                    raw_model,
                    module_type=use_move_history,
                    c_trunk=raw_model.c_trunk,
                    hidden_dim=history_hidden_dim,
                    pos_len=pos_len,
                )
            if world_size > 1:
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    raw_model, device_ids=[device]
                )
            else:
                ddp_model = raw_model

            swa_model = None
            if rank == 0 and swa_scale is not None:
                new_factor = 1.0 / swa_scale
                ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (
                    cur_param - avg_param
                )
                swa_model = AveragedModel(raw_model, avg_fn=ema_avg)

            metrics_obj = Metrics(batch_size, world_size, raw_model)
            running_metrics = {}
            train_state = {}
            last_val_metrics = {}

            train_state["global_step_samples"] = 0

            with torch.no_grad():
                norms = Metrics.get_model_norms(raw_model)
                modelnorm_normal_baseline = norms["normal"]
                train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline
                logging.info(f"Model norm normal baseline computed: {modelnorm_normal_baseline}")

            # Use SGD optimizer for policy-only fine-tuning
            optimizer = torch.optim.SGD(
                get_param_groups(raw_model, train_state, running_metrics),
                lr=1.0,
                momentum=0.9,
                weight_decay=0.0001,
            )

            return (
                model_config,
                ddp_model,
                raw_model,
                swa_model,
                optimizer,
                metrics_obj,
                running_metrics,
                train_state,
                last_val_metrics,
            )
        else:
            state_dict = torch.load(path_to_load_from, map_location=device, weights_only=False)
            model_config = (
                state_dict["config"]
                if "config" in state_dict
                else modelconfigs.config_of_name[model_kind]
            )
            logging.info(str(model_config))
            raw_model = Model(model_config, pos_len)
            raw_model.initialize()

            train_state = {}
            if "train_state" in state_dict:
                train_state = state_dict["train_state"]
            else:
                logging.info(
                    "WARNING: Train state not found in state dict, using fresh train state"
                )

            # Optionally discard accumulated state so fine-tuning restarts LR schedule and momentum cleanly.
            if reset_train_state:
                logging.info(
                    "Resetting train_state/optimizer/running_metrics from checkpoint as requested (-reset-train-state)"
                )
                train_state = {}

            # Do this before loading the state dict, while the model is initialized to fresh values, to get a good baseline
            if "modelnorm_normal_baseline" not in train_state:
                logging.info("Computing modelnorm_normal_baseline since not in train state")
                with torch.no_grad():
                    norms = Metrics.get_model_norms(raw_model)
                    modelnorm_normal_baseline = norms["normal"]
                    train_state["modelnorm_normal_baseline"] = modelnorm_normal_baseline
                    logging.info(
                        f"Model norm normal baseline computed: {modelnorm_normal_baseline}"
                    )

            # Strip off any "module." from when the model was saved with DDP or other things
            model_state_dict = load_model.load_model_state_dict(state_dict)
            raw_model.load_state_dict(model_state_dict)

            raw_model.to(device)
            # Apply fine-tuning AFTER loading pre-trained weights
            if use_lora:
                from lora import apply_lora_to_model

                raw_model, _ = apply_lora_to_model(
                    raw_model, target_blocks=lora_blocks, rank=lora_rank, alpha=lora_alpha
                )
            elif freeze_final_cnn_block_and_heads:
                raw_model = freeze_except_final_cnn_block_and_heads(raw_model)
            elif freeze_final_layers_and_heads:
                raw_model = freeze_except_final_layers_and_heads(raw_model)
            elif freeze_policy_and_value:
                raw_model = freeze_except_policy_and_value(raw_model)
            elif freeze_value_head:
                raw_model = freeze_value_head_only(raw_model)
            else:
                raw_model = freeze_except_policy(raw_model)
            if use_move_history is not None:
                from move_history import apply_move_history_to_model

                raw_model, _ = apply_move_history_to_model(
                    raw_model,
                    module_type=use_move_history,
                    c_trunk=raw_model.c_trunk,
                    hidden_dim=history_hidden_dim,
                    pos_len=pos_len,
                )
            if world_size > 1:
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    raw_model, device_ids=[device]
                )
            else:
                ddp_model = raw_model

            swa_model = None
            if rank == 0 and swa_scale is not None:
                new_factor = 1.0 / swa_scale
                ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (
                    cur_param - avg_param
                )
                swa_model = AveragedModel(raw_model, avg_fn=ema_avg)
                swa_model_state_dict = load_model.load_swa_model_state_dict(state_dict)
                if swa_model_state_dict is not None:
                    swa_model.load_state_dict(swa_model_state_dict)

            metrics_obj = Metrics(batch_size, world_size, raw_model)
            if not reset_train_state and "metrics" in state_dict:
                metrics_obj.load_state_dict(state_dict["metrics"])
            else:
                logging.info("WARNING: Metrics not found in state dict, using fresh metrics")

            running_metrics = {}
            if not reset_train_state and "running_metrics" in state_dict:
                running_metrics = state_dict["running_metrics"]
            else:
                logging.info(
                    "WARNING: Running metrics not found in state dict, using fresh running metrics"
                )

            last_val_metrics = {}
            if not reset_train_state and "last_val_metrics" in state_dict:
                last_val_metrics = state_dict["last_val_metrics"]
            else:
                logging.info(
                    "WARNING: Running metrics not found in state dict, using fresh last val metrics"
                )

            # Use SGD optimizer for policy-only fine-tuning
            optimizer = torch.optim.SGD(
                get_param_groups(raw_model, train_state, running_metrics),
                lr=1.0,
                momentum=0.9,
                weight_decay=0.0001,
            )
            if not reset_train_state and "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
            else:
                logging.info("WARNING: Optimizer not found in state dict, using fresh optimizer")

            return (
                model_config,
                ddp_model,
                raw_model,
                swa_model,
                optimizer,
                metrics_obj,
                running_metrics,
                train_state,
                last_val_metrics,
            )

    (
        model_config,
        ddp_model,
        raw_model,
        swa_model,
        optimizer,
        metrics_obj,
        running_metrics,
        train_state,
        last_val_metrics,
    ) = load()

    # Load frozen base model for KL regularization
    kl_base_model = None
    if kl_base_checkpoint is not None:
        logging.info(f"Loading base model for KL regularization from: {kl_base_checkpoint}")
        logging.info(f"KL beta = {kl_beta}")
        base_state_dict = torch.load(kl_base_checkpoint, map_location=device, weights_only=False)
        base_model_config = (
            base_state_dict["config"] if "config" in base_state_dict else model_config
        )
        kl_base_model = Model(base_model_config, pos_len)
        kl_base_model.initialize()
        base_model_state_dict = load_model.load_model_state_dict(base_state_dict)
        kl_base_model.load_state_dict(base_model_state_dict)
        kl_base_model.to(device)
        kl_base_model.eval()
        for param in kl_base_model.parameters():
            param.requires_grad = False
        logging.info("Base model loaded and frozen for KL regularization")
        del base_state_dict, base_model_state_dict

    if "global_step_samples" not in train_state:
        train_state["global_step_samples"] = 0
    if max_train_bucket_per_new_data is not None and "train_bucket_level" not in train_state:
        train_state["train_bucket_level"] = samples_per_epoch
    if "train_steps_since_last_reload" not in train_state:
        train_state["train_steps_since_last_reload"] = 0
    if "export_cycle_counter" not in train_state:
        train_state["export_cycle_counter"] = 0
    if "window_start_data_row_idx" not in train_state:
        train_state["window_start_data_row_idx"] = 0
    if "total_num_data_rows" not in train_state:
        train_state["total_num_data_rows"] = 0
    if "old_train_data_dirs" not in train_state:
        train_state["old_train_data_dirs"] = []
    if "data_files_used" not in train_state:
        train_state["data_files_used"] = set()
    if "swa_sample_accum" not in train_state:
        train_state["swa_sample_accum"] = 0.0

    if intermediate_loss_scale is not None:
        assert (
            raw_model.get_has_intermediate_head()
        ), "Model must have intermediate head to use intermediate loss"

    # If the user specified an intermediate head but no loss scale, pick something reasonable by default
    if raw_model.get_has_intermediate_head():
        if intermediate_loss_scale is None and main_loss_scale is None:
            if model_config["trunk_normless"]:
                # fson-bnh default
                assert model_config["intermediate_head_blocks"] == len(
                    model_config["block_kind"]
                ), "If these are unequal, don't know what you intend, please specify intermediate_loss_scale"
                intermediate_loss_scale = 0.8
                main_loss_scale = 0.2
            else:
                # Intermediate head in the middle of the trunk
                intermediate_loss_scale = 0.5
                main_loss_scale = 0.5
        elif intermediate_loss_scale is None:
            assert (
                False
            ), "Please specify both of main_loss_scale and intermediate_loss_scale or neither when using an architecture with an intermediate head."

    logging.info(f"swa_period_samples {swa_period_samples}")
    logging.info(f"swa_scale {swa_scale}")
    logging.info(f"lookahead_alpha {lookahead_alpha}")
    logging.info(f"lookahead_k {lookahead_k}")
    logging.info(f"base_per_sample_lr {base_per_sample_lr}")
    logging.info(f"disable_warmup {disable_warmup}")
    logging.info(f"disable_swa {disable_swa}")
    logging.info(f"soft_policy_weight_scale {soft_policy_weight_scale}")
    logging.info(f"disable_optimistic_policy {disable_optimistic_policy}")
    logging.info(f"meta_kata_only_soft_policy {meta_kata_only_soft_policy}")
    logging.info(f"value_loss_scale {value_loss_scale}")
    logging.info(f"td_value_loss_scales {td_value_loss_scales}")
    logging.info(f"seki_loss_scale {seki_loss_scale}")
    logging.info(f"variance_time_loss_scale {variance_time_loss_scale}")
    logging.info(f"main_loss_scale {main_loss_scale}")
    logging.info(f"intermediate_loss_scale {intermediate_loss_scale}")

    # Print all model parameters just to get a summary
    total_num_params = 0
    total_trainable_params = 0
    logging.info("Parameters in model:")
    for name, param in raw_model.named_parameters():
        product = 1
        for dim in param.shape:
            product *= int(dim)
        if param.requires_grad:
            total_trainable_params += product
        total_num_params += product
        logging.info(f"{name}, {list(param.shape)}, {product} params")
    logging.info(f"Total num params: {total_num_params}")
    logging.info(f"Total trainable params: {total_trainable_params}")

    # Initialize wandb
    wandb_project = args.get("wandb_project")
    wandb_run_name = args.get("wandb_run_name")
    use_wandb = HAS_WANDB and wandb_project is not None and rank == 0
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_kind": model_kind,
                "batch_size": batch_size,
                "pos_len": pos_len,
                "lr_scale": lr_scale,
                "base_per_sample_lr": base_per_sample_lr,
                "soft_policy_weight_scale": soft_policy_weight_scale,
                "use_fp16": use_fp16,
                "policy_loss_only": policy_loss_only,
                "freeze_value_head": freeze_value_head,
                "total_trainable_params": total_trainable_params,
                "total_num_params": total_num_params,
            },
        )
        logging.info(f"Wandb initialized: project={wandb_project}, run={wandb_run_name}")
    elif wandb_project is not None and not HAS_WANDB:
        logging.warning(
            "wandb not installed, skipping wandb logging. Install with: pip install wandb"
        )

    lookahead_cache = {}
    if lookahead_k is not None:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                lookahead_cache[param] = torch.zeros_like(param.data)
                lookahead_cache[param] = lookahead_cache[param].copy_(param.data)
        logging.info(f"Using lookahead optimizer {lookahead_alpha} {lookahead_k}")

    # EPOCHS AND LR ---------------------------------------------------------------------

    def update_and_return_lr_and_wd():
        per_sample_lr = base_per_sample_lr * lr_scale * lr_scale_auto_factor(train_state)

        # Warmup for initial training (skip entirely when fine-tuning)
        warmup_scale = 1.0
        if not disable_warmup:
            if (
                model_config["norm_kind"] == "fixup"
                or model_config["norm_kind"] == "fixscale"
                or model_config["norm_kind"] == "fixscaleonenorm"
            ):
                if train_state["global_step_samples"] < 1000000:
                    warmup_scale = 1.0 / 5.0
                elif train_state["global_step_samples"] < 2000000:
                    warmup_scale = 1.0 / 3.0
                elif train_state["global_step_samples"] < 4000000:
                    warmup_scale = 1.0 / 2.0
                elif train_state["global_step_samples"] < 6000000:
                    warmup_scale = 1.0 / 1.4
            elif (
                model_config["norm_kind"] == "bnorm"
                or model_config["norm_kind"] == "brenorm"
                or model_config["norm_kind"] == "fixbrenorm"
            ):
                if train_state["global_step_samples"] < 250000:
                    warmup_scale = 1.0 / 20.0
                elif train_state["global_step_samples"] < 500000:
                    warmup_scale = 1.0 / 14.0
                elif train_state["global_step_samples"] < 750000:
                    warmup_scale = 1.0 / 10.0
                elif train_state["global_step_samples"] < 1000000:
                    warmup_scale = 1.0 / 7.0
                elif train_state["global_step_samples"] < 1250000:
                    warmup_scale = 1.0 / 5.0
                elif train_state["global_step_samples"] < 1500000:
                    warmup_scale = 1.0 / 3.0
                elif train_state["global_step_samples"] < 1750000:
                    warmup_scale = 1.0 / 2.0
                elif train_state["global_step_samples"] < 2000000:
                    warmup_scale = 1.0 / 1.4
                else:
                    warmup_scale = 1.0 / 1.0
            else:
                assert False

        normal_weight_decay = 0.0

        for param_group in optimizer.param_groups:
            group_name = param_group["group_name"]
            if group_name == "input":
                group_scale = 1.0
            elif group_name == "input_noreg":
                group_scale = 1.0
            elif group_name == "normal":
                group_scale = 1.0
            elif group_name == "normal_gamma":
                group_scale = 1.0
            elif group_name == "output":
                group_scale = 0.5
            elif group_name == "noreg":
                group_scale = 1.0
            elif group_name == "output_noreg":
                group_scale = 0.5
            else:
                assert False

            changed = False

            # For lookahead optimizer, use weight decay appropriate for lr scale,
            # but tell optimizer to take larger steps so as to maintain the same
            # effective learning rate after lookahead averaging.
            if lookahead_alpha is not None:
                new_lr_this_group = per_sample_lr * warmup_scale * group_scale / lookahead_alpha
            else:
                new_lr_this_group = per_sample_lr * warmup_scale * group_scale

            if param_group["lr"] != new_lr_this_group:
                param_group["lr"] = new_lr_this_group
                changed = True

            new_weight_decay_this_group = get_weight_decay(
                raw_model,
                lr_scale,
                warmup_scale=warmup_scale,
                train_state=train_state,
                running_metrics=running_metrics,
                group_name=group_name,
            )
            if param_group["weight_decay"] != new_weight_decay_this_group:
                param_group["weight_decay"] = new_weight_decay_this_group
                changed = True

            if group_name == "normal":
                normal_weight_decay = param_group["weight_decay"]

            if changed:
                logging.info(
                    f"Param group {param_group['group_name']} lr {param_group['lr']} weight_decay {param_group['weight_decay']}"
                )

        return per_sample_lr * warmup_scale, normal_weight_decay

    last_brenorm_update_samples_this_instance = train_state["global_step_samples"]

    def maybe_update_brenorm_params():
        nonlocal last_brenorm_update_samples_this_instance

        if model_config["norm_kind"] == "brenorm" or model_config["norm_kind"] == "fixbrenorm":
            if "brenorm_rmax" not in train_state:
                train_state["brenorm_rmax"] = 1.0
            if "brenorm_dmax" not in train_state:
                train_state["brenorm_dmax"] = 0.0

            num_samples_elapsed = (
                train_state["global_step_samples"] - last_brenorm_update_samples_this_instance
            )
            factor = math.exp(-num_samples_elapsed / brenorm_adjustment_scale)
            train_state["brenorm_rmax"] = train_state["brenorm_rmax"] + (1.0 - factor) * (
                brenorm_target_rmax - train_state["brenorm_rmax"]
            )
            train_state["brenorm_dmax"] = train_state["brenorm_dmax"] + (1.0 - factor) * (
                brenorm_target_dmax - train_state["brenorm_dmax"]
            )

            raw_model.set_brenorm_params(
                brenorm_avg_momentum, train_state["brenorm_rmax"], train_state["brenorm_dmax"]
            )
            last_brenorm_update_samples_this_instance = train_state["global_step_samples"]

    # DATA RELOADING GENERATOR ------------------------------------------------------------

    # Some globals
    last_curdatadir = None
    trainfilegenerator = None
    vdatadir = None

    def maybe_reload_training_data():
        nonlocal last_curdatadir
        nonlocal trainfilegenerator
        nonlocal vdatadir

        assert (
            rank == 0
        ), "Helper ddp training processes should not call maybe_reload_training_data"

        while True:
            if datadir:
                curdatadir = os.path.realpath(datadir)
            elif latestdatadir:
                curdatadir = max(
                    (
                        os.path.realpath(os.path.join(latestdatadir, item))
                        for item in os.listdir(latestdatadir)
                        if os.path.isdir(os.path.join(latestdatadir, item))
                        and not item.endswith(".tmp")
                    ),
                    key=os.path.getmtime,
                    default=os.path.join(os.path.realpath(latestdatadir), "*"),
                )

            # Different directory - new shuffle
            if curdatadir != last_curdatadir:
                if not os.path.exists(curdatadir):
                    if quit_if_no_data:
                        logging.info(
                            "Shuffled data path does not exist, there seems to be no data or not enough data yet, quitting: %s"
                            % curdatadir
                        )
                        sys.exit(0)
                    logging.info(
                        "Shuffled data path does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s"
                        % curdatadir
                    )
                    time.sleep(30)
                    continue

                trainjsonpath = os.path.join(curdatadir, "train.json")
                if not os.path.exists(trainjsonpath):
                    if quit_if_no_data:
                        logging.info(
                            "Shuffled data train.json file does not exist, there seems to be no data or not enough data yet, quitting: %s"
                            % trainjsonpath
                        )
                        sys.exit(0)
                    logging.info(
                        "Shuffled data train.json file does not exist, there seems to be no shuffled data yet, waiting and trying again later: %s"
                        % trainjsonpath
                    )
                    time.sleep(30)
                    continue

                logging.info("Updated training data: " + curdatadir)
                last_curdatadir = curdatadir

                with open(trainjsonpath) as f:
                    datainfo = json.load(f)
                    train_state["window_start_data_row_idx"] = datainfo["range"][0]
                    train_state["total_num_data_rows"] = datainfo["range"][1]

                # Fill the buckets
                if max_train_bucket_per_new_data is not None:
                    if "train_bucket_level_at_row" not in train_state:
                        train_state["train_bucket_level_at_row"] = train_state[
                            "total_num_data_rows"
                        ]
                    if (
                        train_state["total_num_data_rows"]
                        > train_state["train_bucket_level_at_row"]
                    ):
                        new_row_count = (
                            train_state["total_num_data_rows"]
                            - train_state["train_bucket_level_at_row"]
                        )
                        logging.info(
                            "Advancing trainbucket row %.0f to %.0f, %.0f new rows"
                            % (
                                train_state["train_bucket_level_at_row"],
                                train_state["total_num_data_rows"],
                                new_row_count,
                            )
                        )
                        train_state["train_bucket_level_at_row"] = train_state[
                            "total_num_data_rows"
                        ]
                        logging.info(
                            "Fill per data %.3f, Max bucket size %.0f"
                            % (max_train_bucket_per_new_data, max_train_bucket_size)
                        )
                        logging.info(
                            "Old rows in bucket: %.0f" % train_state["train_bucket_level"]
                        )
                        train_state["train_bucket_level"] += (
                            new_row_count * max_train_bucket_per_new_data
                        )
                        cap = max(max_train_bucket_size, samples_per_epoch)
                        if train_state["train_bucket_level"] > cap:
                            train_state["train_bucket_level"] = cap
                        logging.info(
                            "New rows in bucket: %.0f" % train_state["train_bucket_level"]
                        )
                    if (
                        train_state["total_num_data_rows"]
                        < train_state["train_bucket_level_at_row"]
                    ):
                        # Bucket went backward! This must be a network imported from a different run, reset the train bucket level
                        logging.warning(
                            "Train bucket last filled at %d rows but now there are only %d rows!"
                            % (
                                train_state["train_bucket_level_at_row"],
                                train_state["total_num_data_rows"],
                            )
                        )
                        logging.warning(
                            "Data was deleted or this network was transplanted into a new run, resetting the train bucket fill rows"
                        )
                        train_state["train_bucket_level_at_row"] = train_state[
                            "total_num_data_rows"
                        ]

                logging.info(
                    "Train steps since last reload: %.0f -> 0"
                    % train_state["train_steps_since_last_reload"]
                )
                train_state["train_steps_since_last_reload"] = 0

                # Load training data files
                tdatadir = os.path.join(curdatadir, "train")
                train_files = [
                    os.path.join(tdatadir, fname)
                    for fname in os.listdir(tdatadir)
                    if fname.endswith(".npz")
                ]
                epoch0_train_files = [
                    path for path in train_files if path not in train_state["data_files_used"]
                ]
                if no_repeat_files:
                    logging.info(
                        f"Dropping {len(train_files)-len(epoch0_train_files)}/{len(train_files)} files in: {tdatadir} as already used"
                    )
                else:
                    logging.info(
                        f"Skipping {len(train_files)-len(epoch0_train_files)}/{len(train_files)} files in: {tdatadir} as already used first pass"
                    )

                if len(train_files) <= 0 or (no_repeat_files and len(epoch0_train_files) <= 0):
                    if quit_if_no_data:
                        logging.info(f"No new training files found in: {tdatadir}, quitting")
                        sys.exit(0)
                    logging.info(
                        f"No new training files found in: {tdatadir}, waiting 30s and trying again"
                    )
                    time.sleep(30)
                    continue

                # Update history of what training data we used
                if tdatadir not in train_state["old_train_data_dirs"]:
                    train_state["old_train_data_dirs"].append(tdatadir)
                # Clear out tracking of sufficiently old files
                while len(train_state["old_train_data_dirs"]) > 20:
                    old_dir = train_state["old_train_data_dirs"][0]
                    train_state["old_train_data_dirs"] = train_state["old_train_data_dirs"][1:]
                    for filename in list(train_state["data_files_used"]):
                        if filename.startswith(old_dir):
                            train_state["data_files_used"].remove(filename)

                def train_files_gen():
                    train_files_shuffled = epoch0_train_files.copy()
                    while True:
                        random.shuffle(train_files_shuffled)
                        for filename in train_files_shuffled:
                            logging.info("Yielding training file for dataset: " + filename)
                            train_state["data_files_used"].add(filename)
                            yield filename
                        if no_repeat_files:
                            break
                        else:
                            train_files_shuffled = train_files.copy()
                            train_state["data_files_used"] = set()

                trainfilegenerator = PushBackGenerator(train_files_gen())
                vdatadir = os.path.join(curdatadir, "val")

            # Same directory as before, no new shuffle
            else:
                if max_train_steps_since_last_reload is not None:
                    if (
                        train_state["train_steps_since_last_reload"]
                        + 0.99 * samples_per_epoch / sub_epochs
                        > max_train_steps_since_last_reload
                    ):
                        logging.info(
                            "Too many train steps since last reload, waiting 5m and retrying (current %f)"
                            % train_state["train_steps_since_last_reload"]
                        )
                        time.sleep(300)
                        continue

            break

    # Load all the files we should train on during a subepoch
    def get_files_for_subepoch():
        nonlocal trainfilegenerator

        assert rank == 0, "Helper ddp training processes should not call get_files_for_subepoch"

        num_batches_per_epoch = int(round(samples_per_epoch / batch_size))
        num_batches_per_subepoch = num_batches_per_epoch / sub_epochs

        # Pick enough files to get the number of batches we want
        train_files_to_use = []
        batches_to_use_so_far = 0
        found_enough = False
        for filename in trainfilegenerator:
            jsonfilename = os.path.splitext(filename)[0] + ".json"
            with open(jsonfilename) as f:
                trainfileinfo = json.load(f)

            num_batches_this_file = trainfileinfo["num_rows"] // batch_size
            if num_batches_this_file <= 0:
                continue

            if batches_to_use_so_far + num_batches_this_file > num_batches_per_subepoch:
                # If we're going over the desired amount, randomly skip the file with probability equal to the
                # proportion of batches over - this makes it so that in expectation, we have the desired number of batches
                if (
                    batches_to_use_so_far > 0
                    and random.random()
                    <= (batches_to_use_so_far + num_batches_this_file - num_batches_per_subepoch)
                    / num_batches_this_file
                ):
                    trainfilegenerator.push_back(filename)
                    found_enough = True
                    break

            train_files_to_use.append(filename)
            batches_to_use_so_far += num_batches_this_file

            # Sanity check - load a max of 100000 files.
            if (
                batches_to_use_so_far >= num_batches_per_subepoch
                or len(train_files_to_use) > 100000
            ):
                found_enough = True
                break

        if found_enough:
            return train_files_to_use
        return None

    # METRICS -----------------------------------------------------------------------------------
    def detensorify_metrics(metrics):
        ret = {}
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                ret[key] = metrics[key].detach().cpu().item()
            else:
                ret[key] = metrics[key]
        return ret

    if rank == 0:
        train_metrics_out = open(os.path.join(traindir, "metrics_train.json"), "a")
        val_metrics_out = open(os.path.join(traindir, "metrics_val.json"), "a")
    else:
        train_metrics_out = open(os.path.join(traindir, f"metrics_train_rank{rank}.json"), "a")
        val_metrics_out = open(os.path.join(traindir, f"metrics_val_rank{rank}.json"), "a")

    # TRAIN! -----------------------------------------------------------------------------------

    last_longterm_checkpoint_save_time = datetime.datetime.now()
    num_epochs_this_instance = 0
    print_train_loss_every_batches = 100 if not gnorm_stats_debug else 1000

    if "sums" not in running_metrics:
        running_metrics["sums"] = defaultdict(float)
    else:
        running_metrics["sums"] = defaultdict(float, running_metrics["sums"])
    if "weights" not in running_metrics:
        running_metrics["weights"] = defaultdict(float)
    else:
        running_metrics["weights"] = defaultdict(float, running_metrics["weights"])

    torch.backends.cudnn.benchmark = True

    if use_fp16:
        logging.info("Training in FP16! Creating scaler")
        scaler = GradScaler()
    else:
        logging.info("Training in FP32.")

    # All ddp threads should be lined up at this point before continuing
    if barrier is not None:
        barrier.wait()

    while True:
        if (
            max_epochs_this_instance is not None
            and max_epochs_this_instance >= 0
            and num_epochs_this_instance >= max_epochs_this_instance
        ):
            logging.info("Hit max epochs this instance, done")
            break
        if (
            max_training_samples is not None
            and train_state["global_step_samples"] >= max_training_samples
        ):
            logging.info("Hit max training samples, done")
            break

        if rank == 0:
            maybe_reload_training_data()

            if max_train_bucket_per_new_data is not None:
                if train_state["train_bucket_level"] > 0.99 * samples_per_epoch:
                    logging.info(
                        "Consuming %.0f rows from train bucket (%.0f -> %.0f)"
                        % (
                            samples_per_epoch,
                            train_state["train_bucket_level"],
                            train_state["train_bucket_level"] - samples_per_epoch,
                        )
                    )
                    train_state["train_bucket_level"] -= samples_per_epoch
                else:
                    if stop_when_train_bucket_limited:
                        logging.info(
                            "Exceeding train bucket, not enough new data rows, terminating (current level %f)"
                            % train_state["train_bucket_level"]
                        )
                        break
                    else:
                        logging.info(
                            "Exceeding train bucket, not enough new data rows, waiting 5m and retrying (current level %f)"
                            % train_state["train_bucket_level"]
                        )
                        time.sleep(300)
                        continue

        # DDP need to wait on the main process after reloading data and/or training bucket waiting
        if barrier is not None:
            barrier.wait()

        logging.info("GC collect")
        gc.collect()

        clear_metric_nonfinite(running_metrics["sums"], running_metrics["weights"])

        logging.info("=========================================================================")
        logging.info("BEGINNING NEXT EPOCH " + str(num_epochs_this_instance))
        logging.info("=========================================================================")
        logging.info("Current time: " + str(datetime.datetime.now()))
        logging.info("Global step: %d samples" % (train_state["global_step_samples"]))
        logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
        logging.info(f"Training dir: {traindir}")
        logging.info(f"Export dir: {exportdir}")
        if use_fp16:
            logging.info(f"Current grad scale: {scaler.get_scale()}")

        lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()
        maybe_update_brenorm_params()

        # SUB EPOCH LOOP -----------
        batch_count_this_epoch = 0
        last_train_stats_time = time.perf_counter()
        for i in range(sub_epochs):

            if rank == 0:
                if i != 0:
                    maybe_reload_training_data()
                train_files_to_use = get_files_for_subepoch()
                while train_files_to_use is None or len(train_files_to_use) <= 0:
                    if quit_if_no_data:
                        logging.info("Not enough data files to fill a subepoch! Quitting.")
                        sys.exit(0)
                    logging.info(
                        "Not enough data files to fill a subepoch! Waiting 5m before retrying."
                    )
                    time.sleep(300)
                    maybe_reload_training_data()
                    train_files_to_use = get_files_for_subepoch()

                if barrier is not None:
                    barrier.wait()
                for wpipe in writepipes:
                    wpipe.send(train_files_to_use)
                # Wait briefly just in case to reduce chance of races with filesystem or anything else
                time.sleep(5)
            else:
                if barrier is not None:
                    barrier.wait()
                train_files_to_use = readpipes[rank - 1].recv()

            # DDP need to wait on the main process after reloading data and sending files to train with
            if barrier is not None:
                barrier.wait()

            logging.info("Beginning training subepoch!")
            logging.info("This subepoch, using files: " + str(train_files_to_use))
            logging.info("Currently up to data row " + str(train_state["total_num_data_rows"]))
            lookahead_counter = 0
            for batch in data_processing_pytorch.read_npz_training_data(
                train_files_to_use,
                batch_size,
                world_size,
                rank,
                pos_len=pos_len,
                device=device,
                randomize_symmetries=True,  # Always use symmetries for data augmentation
                include_meta=raw_model.get_has_metadata_encoder(),
                model_config=model_config,
            ):
                optimizer.zero_grad(set_to_none=True)
                extra_outputs = None
                # if raw_model.get_has_metadata_encoder():
                #     extra_outputs = ExtraOutputs([MetadataEncoder.OUTMEAN_KEY,MetadataEncoder.OUTLOGVAR_KEY])

                if use_fp16:
                    with autocast():
                        model_outputs = ddp_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(
                                batch["metadataInputNC"]
                                if raw_model.get_has_metadata_encoder()
                                else None
                            ),
                            extra_outputs=extra_outputs,
                        )
                    model_outputs = raw_model.float32ify_output(model_outputs)
                else:
                    model_outputs = ddp_model(
                        batch["binaryInputNCHW"],
                        batch["globalInputNC"],
                        input_meta=(
                            batch["metadataInputNC"]
                            if raw_model.get_has_metadata_encoder()
                            else None
                        ),
                        extra_outputs=extra_outputs,
                    )

                postprocessed = raw_model.postprocess_output(model_outputs)

                # Apply label smoothing to policy targets if enabled
                if label_smoothing > 0.0:
                    policy_targets = batch["policyTargetsNCMove"]
                    mask = batch["binaryInputNCHW"][:, 0:1, :, :]
                    n, _, h, w = mask.shape
                    # Create uniform distribution over legal moves + pass
                    legal_mask = torch.cat(
                        (mask.reshape(n, 1, h * w), mask.new_ones((n, 1, 1))), dim=2
                    )  # N x 1 x (HW+1)
                    uniform = legal_mask / legal_mask.sum(dim=2, keepdim=True)
                    # Smooth: (1-eps)*target + eps*uniform
                    batch["policyTargetsNCMove"] = (
                        1.0 - label_smoothing
                    ) * policy_targets + label_smoothing * uniform

                metrics = metrics_obj.metrics_dict_batchwise(
                    raw_model,
                    postprocessed,
                    extra_outputs,
                    batch,
                    is_training=True,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    disable_optimistic_policy=disable_optimistic_policy,
                    meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=seki_loss_scale,
                    variance_time_loss_scale=variance_time_loss_scale,
                    main_loss_scale=main_loss_scale,
                    intermediate_loss_scale=intermediate_loss_scale,
                )

                # DDP averages loss across instances, so to preserve LR as per-sample lr, we scale by world size.
                if policy_loss_only:
                    loss = (
                        compute_policy_only_loss(metrics, raw_model, soft_policy_weight_scale)
                        * world_size
                    )
                    metrics["loss_sum"] = loss / world_size
                else:
                    loss = metrics["loss_sum"] * world_size

                # KL regularization: penalize divergence from base model's policy
                if kl_base_model is not None:
                    with torch.no_grad():
                        base_outputs = kl_base_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(
                                batch["metadataInputNC"]
                                if kl_base_model.get_has_metadata_encoder()
                                else None
                            ),
                        )
                        base_postprocessed = kl_base_model.postprocess_output(base_outputs)
                        # policy_logits shape: (N, num_policy_outputs, moves)
                        # index 0 = main player policy
                        # postprocessed[0] = main head tuple, [0] = policy_logits (N, num_outputs, moves)
                        base_policy_logits = base_postprocessed[0][0][:, 0, :]

                    finetuned_policy_logits = postprocessed[0][0][:, 0, :]
                    kl_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(finetuned_policy_logits, dim=-1),
                        torch.nn.functional.softmax(base_policy_logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss = loss + kl_beta * kl_loss * batch_size
                    metrics["kl_loss_batch"] = kl_loss.detach().item()

                # Reduce gradients across DDP
                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if (
                    model_config["norm_kind"] == "fixup"
                    or model_config["norm_kind"] == "fixscale"
                    or model_config["norm_kind"] == "fixscaleonenorm"
                ):
                    gnorm_cap = 2500.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
                elif (
                    model_config["norm_kind"] == "bnorm"
                    or model_config["norm_kind"] == "brenorm"
                    or model_config["norm_kind"] == "fixbrenorm"
                ):
                    gnorm_cap = 5500.0 * (1.0 if gnorm_clip_scale is None else gnorm_clip_scale)
                else:
                    assert False

                if gnorm_stats_debug:
                    stats = metrics_obj.get_specific_norms_and_gradient_stats(raw_model)
                    for stat, value in stats.items():
                        metrics[stat] = value

                if (
                    "use_repvgg_learning_rate" in model_config
                    and model_config["use_repvgg_learning_rate"]
                ):
                    gradscale_constant = torch.tensor(
                        [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
                        dtype=torch.float32,
                        device=device,
                        requires_grad=False,
                    ).view(1, 1, 3, 3)
                    for name, param in ddp_model.named_parameters():
                        if (
                            "normactconv" in name
                            and ".conv.weight" in name
                            and len(param.shape) == 4
                            and param.shape[2] == 3
                            and param.shape[3] == 3
                        ):
                            if param.grad is not None:
                                param.grad *= gradscale_constant

                # Loosen gradient clipping as we shift to smaller learning rates
                gnorm_cap = gnorm_cap / math.sqrt(
                    max(0.0000001, lr_scale * lr_scale_auto_factor(train_state))
                )

                gnorm = (
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gnorm_cap)
                    .detach()
                    .cpu()
                    .item()
                )

                if math.isfinite(gnorm) and abs(gnorm < 1e30):
                    metrics["gnorm_batch"] = gnorm
                    exgnorm = max(0.0, gnorm - gnorm_cap)
                    metrics["exgnorm_sum"] = exgnorm * batch_size

                metrics["pslr_batch"] = lr_right_now
                metrics["wdnormal_batch"] = normal_weight_decay_right_now
                metrics["gnorm_cap_batch"] = gnorm_cap
                metrics["window_start_batch"] = train_state["window_start_data_row_idx"]
                metrics["window_end_batch"] = train_state["total_num_data_rows"]

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                batch_count_this_epoch += 1
                train_state["train_steps_since_last_reload"] += batch_size * world_size
                train_state["global_step_samples"] += batch_size * world_size

                metrics = detensorify_metrics(metrics)

                if lookahead_k is not None and lookahead_print:
                    # Only accumulate metrics when lookahead is synced if lookahead_print is True
                    if lookahead_counter == 0:
                        accumulate_metrics(
                            running_metrics["sums"],
                            running_metrics["weights"],
                            metrics,
                            batch_size,
                            decay=math.exp(-0.001 * lookahead_k),
                            new_weight=1.0,
                        )
                    else:
                        accumulate_metrics(
                            running_metrics["sums"],
                            running_metrics["weights"],
                            metrics,
                            batch_size,
                            decay=1.0,
                            new_weight=0.0,
                        )
                else:
                    accumulate_metrics(
                        running_metrics["sums"],
                        running_metrics["weights"],
                        metrics,
                        batch_size,
                        decay=0.999,
                        new_weight=1.0,
                    )

                if batch_count_this_epoch % print_train_loss_every_batches == 0:

                    if (
                        model_config["norm_kind"] == "brenorm"
                        or model_config["norm_kind"] == "fixbrenorm"
                    ):
                        metrics["brn_rmax"] = train_state["brenorm_rmax"]
                        metrics["brn_dmax"] = train_state["brenorm_dmax"]
                        metrics["brn_mmnt"] = brenorm_avg_momentum
                        upper_rclippage = []
                        lower_rclippage = []
                        dclippage = []
                        raw_model.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
                        metrics["brn_ruclip"] = sum(upper_rclippage) / max(
                            len(upper_rclippage), 1.0
                        )
                        metrics["brn_rlclip"] = sum(lower_rclippage) / max(
                            len(lower_rclippage), 1.0
                        )
                        metrics["brn_dclip"] = sum(dclippage) / max(len(dclippage), 1.0)

                    t1 = time.perf_counter()
                    timediff = t1 - last_train_stats_time
                    last_train_stats_time = t1
                    metrics["time_since_last_print"] = timediff
                    log_metrics(
                        running_metrics["sums"],
                        running_metrics["weights"],
                        metrics,
                        train_metrics_out,
                    )

                    if use_wandb:
                        train_log = {"train/step": train_state["global_step_samples"]}
                        for m in running_metrics["sums"]:
                            w = running_metrics["weights"].get(m, 1.0)
                            if w > 0:
                                if m.endswith("_sum"):
                                    train_log[f"train/{m[:-4]}"] = float(
                                        running_metrics["sums"][m] / w
                                    )
                                elif m.endswith("_batch"):
                                    train_log[f"train/{m}"] = float(running_metrics["sums"][m] / w)
                        train_log["train/lr"] = lr_right_now
                        wandb.log(train_log, step=train_state["global_step_samples"])

                # Update LR more frequently at the start for smoother warmup ramp and wd adjustment
                if (
                    train_state["global_step_samples"] <= 50000000
                    and batch_count_this_epoch % 50 == 0
                ):
                    lr_right_now, normal_weight_decay_right_now = update_and_return_lr_and_wd()

                # Update batch renorm parameters
                if batch_count_this_epoch % 500 == 0:
                    maybe_update_brenorm_params()

                # Perform lookahead
                in_between_lookaheads = False
                if lookahead_k is not None:
                    lookahead_counter += 1
                    if lookahead_counter >= lookahead_k:
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                slow_param_data = lookahead_cache[param]
                                slow_param_data.add_(
                                    param.data.detach() - slow_param_data, alpha=lookahead_alpha
                                )
                                param.data.copy_(slow_param_data)
                        lookahead_counter = 0
                        in_between_lookaheads = False
                    else:
                        in_between_lookaheads = True

                # Perform SWA
                if swa_model is not None and swa_scale is not None:
                    train_state["swa_sample_accum"] += batch_size * world_size
                    # Only snap SWA when lookahead slow params are in sync.
                    if (
                        train_state["swa_sample_accum"] >= swa_period_samples
                        and not in_between_lookaheads
                    ):
                        train_state["swa_sample_accum"] = 0
                        logging.info("Accumulating SWA")
                        swa_model.update_parameters(raw_model)

            logging.info("Finished training subepoch!")

        # END SUB EPOCH LOOP ------------

        # Discard the gradient updates from the leftover batches in the sub epoch from lookahead.
        # This wastes a very tiny bit, but makes it so that we can be in sync and deterministic on ends of subepochs/epochs.
        if lookahead_k is not None:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    slow_param_data = lookahead_cache[param]
                    param.data.copy_(slow_param_data)

        if rank == 0:
            train_state["export_cycle_counter"] += 1

        save(
            ddp_model,
            swa_model,
            optimizer,
            metrics_obj,
            running_metrics,
            train_state,
            last_val_metrics,
        )

        num_epochs_this_instance += 1

        # Validate
        if rank == 0:
            logging.info("Beginning validation after epoch!")
            val_files = []
            if os.path.exists(vdatadir):
                val_files = [
                    os.path.join(vdatadir, fname)
                    for fname in os.listdir(vdatadir)
                    if fname.endswith(".npz")
                ]
            if randomize_val:
                random.shuffle(val_files)
            else:
                # Sort to ensure deterministic order to validation files in case we use only a subset
                val_files = sorted(val_files)
            if len(val_files) == 0:
                logging.info("No validation files, skipping validation step")
            else:
                with torch.no_grad():
                    ddp_model.eval()
                    val_metric_sums = defaultdict(float)
                    val_metric_weights = defaultdict(float)
                    val_samples = 0
                    t0 = time.perf_counter()
                    for batch in data_processing_pytorch.read_npz_training_data(
                        val_files,
                        batch_size,
                        world_size=1,  # Only the main process validates
                        rank=0,  # Only the main process validates
                        pos_len=pos_len,
                        device=device,
                        randomize_symmetries=True,
                        include_meta=raw_model.get_has_metadata_encoder(),
                        model_config=model_config,
                    ):
                        model_outputs = ddp_model(
                            batch["binaryInputNCHW"],
                            batch["globalInputNC"],
                            input_meta=(
                                batch["metadataInputNC"]
                                if raw_model.get_has_metadata_encoder()
                                else None
                            ),
                        )
                        postprocessed = raw_model.postprocess_output(model_outputs)
                        extra_outputs = None
                        metrics = metrics_obj.metrics_dict_batchwise(
                            raw_model,
                            postprocessed,
                            extra_outputs,
                            batch,
                            is_training=False,
                            soft_policy_weight_scale=soft_policy_weight_scale,
                            disable_optimistic_policy=disable_optimistic_policy,
                            meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                            value_loss_scale=value_loss_scale,
                            td_value_loss_scales=td_value_loss_scales,
                            seki_loss_scale=seki_loss_scale,
                            variance_time_loss_scale=variance_time_loss_scale,
                            main_loss_scale=main_loss_scale,
                            intermediate_loss_scale=intermediate_loss_scale,
                        )
                        metrics = detensorify_metrics(metrics)
                        accumulate_metrics(
                            val_metric_sums,
                            val_metric_weights,
                            metrics,
                            batch_size,
                            decay=1.0,
                            new_weight=1.0,
                        )
                        val_samples += batch_size
                        if max_val_samples is not None and val_samples > max_val_samples:
                            break
                        val_metric_sums["nsamp_train"] = running_metrics["sums"]["nsamp"]
                        val_metric_weights["nsamp_train"] = running_metrics["weights"]["nsamp"]
                        val_metric_sums["wsum_train"] = running_metrics["sums"]["wsum"]
                        val_metric_weights["wsum_train"] = running_metrics["weights"]["wsum"]
                    last_val_metrics["sums"] = val_metric_sums
                    last_val_metrics["weights"] = val_metric_weights
                    log_metrics(val_metric_sums, val_metric_weights, metrics, val_metrics_out)

                    if use_wandb:
                        val_log = {}
                        for m in val_metric_sums:
                            w = val_metric_weights.get(m, 1.0)
                            if w > 0:
                                if m.endswith("_sum"):
                                    val_log[f"val/{m[:-4]}"] = float(val_metric_sums[m] / w)
                                elif m.endswith("_batch"):
                                    val_log[f"val/{m}"] = float(val_metric_sums[m] / w)
                        wandb.log(val_log, step=train_state["global_step_samples"])

                    t1 = time.perf_counter()
                    logging.info(f"Validation took {t1-t0} seconds")
                    ddp_model.train()

        if rank == 0:
            logging.info("Export cycle counter = " + str(train_state["export_cycle_counter"]))

            is_time_to_export = False
            if train_state["export_cycle_counter"] >= epochs_per_export:
                if no_export:
                    train_state["export_cycle_counter"] = epochs_per_export
                else:
                    train_state["export_cycle_counter"] = 0
                    is_time_to_export = True

            skip_export_this_time = False
            if export_prob is not None:
                if random.random() > export_prob:
                    skip_export_this_time = True
                    logging.info("Skipping export model this time")

            if (
                not no_export
                and is_time_to_export
                and not skip_export_this_time
                and exportdir is not None
                and not gnorm_stats_debug
            ):
                # Export a model for testing, unless somehow it already exists
                modelname = "%s-s%d-d%d" % (
                    exportprefix,
                    train_state["global_step_samples"],
                    train_state["total_num_data_rows"],
                )
                savepath = os.path.join(exportdir, modelname)
                savepathtmp = os.path.join(exportdir, modelname + ".tmp")
                if os.path.exists(savepath):
                    logging.info("NOT saving model, already exists at: " + savepath)
                else:
                    os.mkdir(savepathtmp)
                    logging.info("SAVING MODEL FOR EXPORT TO: " + savepath)
                    save(
                        ddp_model,
                        swa_model,
                        optimizer,
                        metrics_obj,
                        running_metrics,
                        train_state,
                        last_val_metrics,
                        path=os.path.join(savepathtmp, "model.ckpt"),
                    )
                    time.sleep(2)
                    os.rename(savepathtmp, savepath)

        if sleep_seconds_per_epoch is None:
            time.sleep(1)
        else:
            time.sleep(sleep_seconds_per_epoch)

        if rank == 0:
            now = datetime.datetime.now()
            if now - last_longterm_checkpoint_save_time >= datetime.timedelta(hours=12):
                last_longterm_checkpoint_save_time = now
                dated_name = datetime.datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                save(
                    ddp_model,
                    swa_model,
                    optimizer,
                    metrics_obj,
                    running_metrics,
                    train_state,
                    last_val_metrics,
                    path=os.path.join(longterm_checkpoints_dir, f"{dated_name}.ckpt"),
                )

    train_metrics_out.close()
    val_metrics_out.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    multi_gpus = args["multi_gpus"]
    num_gpus_used = 1
    multi_gpu_device_ids = []
    if multi_gpus is not None:
        for piece in multi_gpus.split(","):
            piece = piece.strip()
            multi_gpu_device_ids.append(int(piece))
        num_gpus_used = len(multi_gpu_device_ids)
    else:
        multi_gpu_device_ids = [0]

    make_dirs(args)

    readpipes = []
    writepipes = []

    if num_gpus_used > 1:
        torch.multiprocessing.set_start_method("spawn")

        world_size = num_gpus_used
        barrier = torch.multiprocessing.Barrier(num_gpus_used)

        for i in range(world_size - 1):
            rpipe, wpipe = torch.multiprocessing.Pipe()
            readpipes.append(rpipe)
            writepipes.append(wpipe)

        torch.multiprocessing.spawn(
            main,
            nprocs=num_gpus_used,
            args=(world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier),
        )
    else:
        rank = 0
        world_size = 1
        barrier = None
        main(rank, world_size, args, multi_gpu_device_ids, readpipes, writepipes, barrier)
