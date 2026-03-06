"""
LoRA (Low-Rank Adaptation) for Conv2d layers.

Wraps existing Conv2d layers with trainable low-rank delta:
    W_new = W_frozen + (A @ B).reshape(W.shape)

Usage:
    apply_lora_to_model(model, target_blocks=[16, 17], rank=8)
    # Freezes everything, adds LoRA to specified blocks + policy head
"""

import torch
import torch.nn as nn
import math


class LoRAConv2d(nn.Module):
    """Wraps a frozen Conv2d with a trainable low-rank update."""

    def __init__(self, original_conv: nn.Conv2d, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha

        # Freeze original weights
        for param in self.original_conv.parameters():
            param.requires_grad = False

        out_channels = original_conv.out_channels
        in_channels = original_conv.in_channels
        k_h, k_w = original_conv.kernel_size

        # Reshape weight as (out_channels, in_channels * k_h * k_w)
        fan_in = in_channels * k_h * k_w

        # Low-rank matrices: delta_W = (A @ B).reshape(out, in, k, k)
        # A: (out_channels, rank), B: (rank, fan_in)
        # Create on same device as original conv
        device = original_conv.weight.device
        self.lora_A = nn.Parameter(torch.zeros(out_channels, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, fan_in, device=device))

        # Initialize A with Kaiming, B with zeros (so delta starts at 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / rank

    @property
    def weight(self):
        """Expose original weight for compatibility with reg_dict and other code that accesses .weight"""
        return self.original_conv.weight

    @property
    def bias(self):
        """Expose original bias for compatibility."""
        return self.original_conv.bias

    @property
    def in_channels(self):
        return self.original_conv.in_channels

    @property
    def out_channels(self):
        return self.original_conv.out_channels

    @property
    def kernel_size(self):
        return self.original_conv.kernel_size

    @property
    def stride(self):
        return self.original_conv.stride

    @property
    def padding(self):
        return self.original_conv.padding

    @property
    def dilation(self):
        return self.original_conv.dilation

    @property
    def groups(self):
        return self.original_conv.groups

    def forward(self, x):
        # Original conv output
        out = self.original_conv(x)

        # LoRA delta: compute low-rank weight update and apply as conv
        delta_w = (self.lora_A @ self.lora_B).reshape(self.original_conv.weight.shape)
        delta_w = delta_w * self.scaling

        # Apply delta as convolution (same padding/stride/groups as original)
        out = out + nn.functional.conv2d(
            x,
            delta_w,
            bias=None,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups,
        )
        return out

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


def _replace_conv_with_lora(parent_module, attr_name, rank, alpha):
    """Replace a Conv2d attribute on parent_module with a LoRAConv2d wrapper."""
    original = getattr(parent_module, attr_name)
    if not isinstance(original, nn.Conv2d):
        return 0
    lora_conv = LoRAConv2d(original, rank=rank, alpha=alpha)
    setattr(parent_module, attr_name, lora_conv)
    lora_params = (
        rank * original.out_channels
        + rank * original.in_channels * original.kernel_size[0] * original.kernel_size[1]
    )
    return lora_params


def register_lora_params_in_reg_dict(model):
    """Patch model.add_reg_dict to include LoRA parameters in the 'normal' group."""
    original_add_reg_dict = model.add_reg_dict

    def patched_add_reg_dict(reg_dict):
        original_add_reg_dict(reg_dict)
        # Add LoRA params to 'normal' group for proper weight decay
        for name, module in model.named_modules():
            if isinstance(module, LoRAConv2d):
                reg_dict["normal"].append(module.lora_A)
                reg_dict["normal"].append(module.lora_B)

    model.add_reg_dict = patched_add_reg_dict


def apply_lora_to_model(model, target_blocks=None, include_policy_head=True, rank=8, alpha=1.0):
    """
    Apply LoRA to specified blocks and optionally the policy head.

    Args:
        model: KataGo Model instance
        target_blocks: List of block indices to apply LoRA to (e.g., [16, 17])
                       If None, applies to last 2 blocks.
        include_policy_head: Whether to also apply LoRA to policy head convs
        rank: LoRA rank (lower = fewer params, higher = more expressive)
        alpha: LoRA scaling factor

    Returns:
        model with LoRA applied, total LoRA trainable params
    """
    if target_blocks is None:
        # Default: last 2 blocks
        num_blocks = len(model.blocks)
        target_blocks = [num_blocks - 2, num_blocks - 1]

    print(f"=== APPLYING LoRA (rank={rank}, alpha={alpha}) ===")
    print(f"Target blocks: {target_blocks}")
    print(f"Include policy head: {include_policy_head}")

    # First, freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False

    total_lora_params = 0

    # Apply LoRA to target blocks
    for block_idx in target_blocks:
        block = model.blocks[block_idx]
        count = 0
        for name, module in block.named_modules():
            # Find Conv2d layers (but not inside LoRAConv2d wrappers)
            if isinstance(module, nn.Conv2d):
                # Navigate to parent and replace
                parts = name.split(".")
                parent = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]
                params_added = _replace_conv_with_lora(parent, attr, rank, alpha)
                if params_added > 0:
                    total_lora_params += params_added
                    count += 1
        print(f"  Block {block_idx}: {count} conv layers wrapped with LoRA")

    # Apply LoRA to policy head
    if include_policy_head:
        policy_count = 0
        for head_name in ["policy_head", "intermediate_policy_head"]:
            head = getattr(model, head_name, None)
            if head is None:
                continue
            for name, module in head.named_modules():
                if isinstance(module, nn.Conv2d):
                    parts = name.split(".")
                    parent = head
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    attr = parts[-1]
                    params_added = _replace_conv_with_lora(parent, attr, rank, alpha)
                    if params_added > 0:
                        total_lora_params += params_added
                        policy_count += 1
            print(f"  {head_name}: {policy_count} conv layers wrapped with LoRA")

    # Also unfreeze the policy head's non-conv parameters (linear layers, biases)
    if include_policy_head:
        for head_name in ["policy_head", "intermediate_policy_head"]:
            head = getattr(model, head_name, None)
            if head is None:
                continue
            for name, param in head.named_parameters():
                if "lora_" not in name and "original_conv" not in name:
                    param.requires_grad = True
                    total_lora_params += param.numel()

    # Patch add_reg_dict so LoRA params are included in optimizer param groups
    register_lora_params_in_reg_dict(model)

    # Count actual trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA params estimate: {total_lora_params:,}")
    print(f"  Actual trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"=== LoRA APPLIED ===")

    return model, trainable
