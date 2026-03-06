#!/usr/bin/env python3
"""
Freeze Modes:

| Mode | What Gets TRAINED | Trainable Params | Use Case |
|------|------------------|------------------|----------|
| `policy` | Policy head only | ~200K (0.2%) | Fastest fine-tuning |
| `policy-value` | Policy + value | ~500K-1M (0.5-1%) | MCTS-aligned |
| `minimal-layers` | Final norm + heads | ~150-200K (0.15%) | Safest |
| `final-block-only` | Block 27 + policy only | ~1.5M (2%) | Learn new play style |
| `no-value` | Trunk + Policy (NO Value) | ~99% | Adapt style + reading |


Examples:
    # Train final block + policy only (for imitating play style)
    python training/train_with_freezing.py \
        -traindir runs/finetune \
        -datadir data/sgf_shuf \
        -initial-checkpoint model_train.ckpt \
        -pos-len 19 -batch-size 512 \
        -freeze-mode final-block-only
"""

import sys
import os
import argparse

# Add directories to path for imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "python"))

import torch


# ============================================================================
# FREEZING FUNCTIONS - Core fine-tuning logic
# ============================================================================


def train_policy_only(model):
    """Train ONLY policy head. Freezes all other parameters (~200K trainable params)."""
    print("=== TRAIN POLICY ONLY ===")
    print("Training ONLY policy head...")
    print("Freezing all other parameters...")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.policy_head.parameters():
        param.requires_grad = True

    if hasattr(model, "intermediate_policy_head") and model.intermediate_policy_head is not None:
        for param in model.intermediate_policy_head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print("=" * 50)
    return model


def train_policy_and_value(model):
    """Train policy head + core value head. Freezes trunk (~500K-1M trainable params)."""
    print("=== TRAIN POLICY + VALUE ===")
    print("Training policy head + core value head...")
    print("Freezing trunk and CNN blocks...")

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze policy head
    for param in model.policy_head.parameters():
        param.requires_grad = True

    if hasattr(model, "intermediate_policy_head") and model.intermediate_policy_head is not None:
        for param in model.intermediate_policy_head.parameters():
            param.requires_grad = True

    # Unfreeze core value head components
    value_head = model.value_head
    for param in value_head.linear_valuehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_miscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_moremiscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear2.parameters():
        param.requires_grad = True

    # Unfreeze intermediate value head if exists
    if hasattr(model, "intermediate_value_head") and model.intermediate_value_head is not None:
        ivalue_head = model.intermediate_value_head
        for param in ivalue_head.linear_valuehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_miscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_moremiscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear2.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("=" * 50)
    return model


def train_minimal_layers(model):
    """Train final norm + all heads. Freezes CNN blocks (~150-200K trainable params)."""
    print("=== TRAIN MINIMAL LAYERS ===")
    print("Training final normalization + all heads...")
    print("Freezing all CNN blocks (targeting ~200K params)...")

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze policy head
    for param in model.policy_head.parameters():
        param.requires_grad = True

    if hasattr(model, "intermediate_policy_head") and model.intermediate_policy_head is not None:
        for param in model.intermediate_policy_head.parameters():
            param.requires_grad = True

    # Unfreeze core value head
    value_head = model.value_head
    for param in value_head.linear_valuehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_miscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear_moremiscvaluehead.parameters():
        param.requires_grad = True
    for param in value_head.linear2.parameters():
        param.requires_grad = True

    if hasattr(model, "intermediate_value_head") and model.intermediate_value_head is not None:
        ivalue_head = model.intermediate_value_head
        for param in ivalue_head.linear_valuehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_miscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear_moremiscvaluehead.parameters():
            param.requires_grad = True
        for param in ivalue_head.linear2.parameters():
            param.requires_grad = True

    # Unfreeze final trunk normalization
    if hasattr(model, "norm_trunkfinal"):
        for param in model.norm_trunkfinal.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if trainable > 500000:
        print("⚠️  WARNING: >500K parameters - use lower learning rate!")
    elif trainable < 150000:
        print("✅ Minimal parameter count - safe for aggressive fine-tuning")

    print("=" * 50)
    return model


def train_final_block_only(model):
    """Train ONLY final CNN block 27 + policy head. Freezes blocks 0-26 and value head (~1.5M trainable params)."""
    print("=== TRAIN FINAL BLOCK + POLICY ONLY ===")
    print("Training ONLY final CNN block 27 + policy head (targeting ~1.5M params)...")
    print("Freezing CNN blocks 0-26 and value head...")

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final CNN block (last block, works for b18/b28/etc)
    last_block_idx = len(model.blocks) - 1
    for name, param in model.named_parameters():
        if name.startswith(f"blocks.{last_block_idx}."):
            param.requires_grad = True

    # Unfreeze final normalization
    if hasattr(model, "norm_trunkfinal"):
        for param in model.norm_trunkfinal.parameters():
            param.requires_grad = True

    # Unfreeze policy head
    for param in model.policy_head.parameters():
        param.requires_grad = True

    if hasattr(model, "intermediate_policy_head") and model.intermediate_policy_head is not None:
        for param in model.intermediate_policy_head.parameters():
            param.requires_grad = True

    # NOTE: Value head is FROZEN - we keep the pretrained value network

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"✅ Value head frozen - using pretrained position evaluation")

    if trainable > 2000000:
        print("⚠️  WARNING: >2M parameters - use lower learning rate (0.1x or less)!")

    print("=" * 50)
    return model


def train_no_value(model):
    """Train trunk + policy head. Freezes ONLY value head components."""
    print("=== TRAIN TRUNK + POLICY (NO VALUE) ===")
    print("Training entire ResNet trunk and Policy head.")
    print("Freezing ONLY the Value head...")

    # Unfreeze everything first (default state)
    for param in model.parameters():
        param.requires_grad = True

    # Freeze main value head entirely
    if hasattr(model, "value_head") and model.value_head is not None:
        for param in model.value_head.parameters():
            param.requires_grad = False

    # Freeze intermediate value head if it exists
    if hasattr(model, "intermediate_value_head") and model.intermediate_value_head is not None:
        for param in model.intermediate_value_head.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print("=" * 50)
    return model


FREEZE_MODES = {
    "policy": train_policy_only,
    "policy-value": train_policy_and_value,
    "minimal-layers": train_minimal_layers,
    "final-block-only": train_final_block_only,
    "no-value": train_no_value,
}


# ============================================================================
# MONKEY PATCH - Elegant hook into train.py
# ============================================================================


def apply_freeze_hook():
    """Hook into Model.to() to apply freezing after model is moved to device."""
    from katago.train.model_pytorch import Model

    original_to = Model.to

    def patched_to(self, *args, **kwargs):
        """Apply freezing after moving model to device."""
        result = original_to(self, *args, **kwargs)

        # Apply freezing once after model is on device
        if hasattr(self, "_freeze_mode") and not hasattr(self, "_freeze_applied"):
            freeze_func = FREEZE_MODES[self._freeze_mode]
            freeze_func(self)
            self._freeze_applied = True

        return result

    Model.to = patched_to


def patch_model_init():
    """Patch Model.__init__ to store freeze mode."""
    from katago.train.model_pytorch import Model

    original_init = Model.__init__

    def patched_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)

        # Check if freeze mode is set globally
        import sys

        if hasattr(sys.modules[__name__], "_global_freeze_mode"):
            self._freeze_mode = sys.modules[__name__]._global_freeze_mode

        return result

    Model.__init__ = patched_init


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def parse_args():
    """Parse command line arguments, extracting freeze-mode."""

    # Extract freeze-mode if present
    freeze_mode = None
    filtered_argv = [sys.argv[0]]

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-freeze-mode":
            i += 1
            freeze_mode = sys.argv[i]
        else:
            filtered_argv.append(sys.argv[i])
        i += 1

    # Validate freeze mode
    if freeze_mode and freeze_mode not in FREEZE_MODES:
        print(f"❌ Invalid freeze mode: {freeze_mode}")
        print(f"   Valid modes: {', '.join(FREEZE_MODES.keys())}")
        sys.exit(1)

    # Update sys.argv for train.py
    sys.argv = filtered_argv

    return freeze_mode


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔒 KataGo Training with Parameter Freezing")
    print("=" * 70)

    # Parse arguments (modifies sys.argv for train.py)
    freeze_mode = parse_args()

    if freeze_mode:
        print(f"\n✅ Freeze mode: {freeze_mode}")
        print(f"📝 Will freeze all parameters except: {freeze_mode} components\n")

        # Store freeze mode globally for the patches to access
        sys.modules[__name__]._global_freeze_mode = freeze_mode

        # Apply patches
        apply_freeze_hook()
        patch_model_init()
    else:
        print("\n⚠️  No freeze mode specified - training all parameters")
        print("   Use -freeze-mode [MODE] to enable freezing")
        print(f"   Available modes: {', '.join(FREEZE_MODES.keys())}\n")

    # Use subprocess approach (proven to work by minimal_layers_training.py)
    # Call train_policy_only.py with appropriate freeze flag
    print("🚀 Launching training...\n")

    import subprocess

    # Map our freeze modes to train_policy_only.py flags
    freeze_flag_map = {
        "policy": None,  # Default mode in train_policy_only.py
        "policy-value": ["-freeze-except-policy-and-value"],
        "minimal-layers": ["-freeze-except-final-layers-and-heads"],
        "final-block-only": ["-freeze-except-final-cnn-block-and-heads"],
        # Train trunk+policy, freeze value weights, and remove value/aux losses from the training signal.
        "no-value": ["-freeze-value-head", "-policy-loss-only"],
    }

    # Build command (now using training/train_policy_only.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_policy_only.py")
    cmd = ["python", train_script] + sys.argv[1:]  # Pass through all args

    # Add freeze flag(s) if needed (policy mode is default, so no flag needed)
    if freeze_mode and freeze_mode != "policy":
        freeze_flags = freeze_flag_map.get(freeze_mode)
        if freeze_flags:
            for flag in freeze_flags:
                cmd.append(flag)

    # Run training
    result = subprocess.call(cmd)
    sys.exit(result)
