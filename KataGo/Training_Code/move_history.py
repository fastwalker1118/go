"""
Move History Modules for KataGo player-style fine-tuning.

Augments the frozen CNN trunk with sequential move reasoning:
  - MoveSequenceEmbedding: learns dense embeddings of past move coordinates
  - TrunkFeatureLSTM: extracts trunk features at past move locations, runs LSTM
  - Combined: both embedding + LSTM

Usage:
    apply_move_history_to_model(model, module_type='lstm', pos_len=19)
    # Injects history features between trunk and policy head via forward hook
"""

import torch
import torch.nn as nn


def extract_move_coords(input_spatial, history_channels=(9, 10, 11, 12, 13)):
    """Extract (row, col) from binary move history planes. Vectorized.

    Args:
        input_spatial: N x C x H x W tensor
        history_channels: which channels contain move history

    Returns:
        rows: N x num_history (long), -1 for pass/missing
        cols: N x num_history (long), -1 for pass/missing
        has_move: N x num_history (bool)
    """
    N, C, H, W = input_spatial.shape
    num_history = len(history_channels)
    ch_idx = torch.tensor(history_channels, device=input_spatial.device)

    # N x num_history x H x W
    planes = input_spatial[:, ch_idx, :, :]

    # Flatten spatial: N x num_history x (H*W)
    flat = planes.reshape(N, num_history, -1)

    # Move exists if any cell is nonzero
    has_move = flat.sum(dim=-1) > 0.5  # N x num_history

    # argmax gives flat index of the 1
    indices = flat.argmax(dim=-1)  # N x num_history

    rows = indices // W
    cols = indices % W

    # Mark missing moves as -1
    rows = torch.where(has_move, rows, torch.full_like(rows, -1))
    cols = torch.where(has_move, cols, torch.full_like(cols, -1))

    return rows, cols, has_move


class MoveSequenceEmbedding(nn.Module):
    """Learns dense embeddings of past move (row, col) coordinates.

    Each of the 5 past moves is embedded as (row_embed, col_embed),
    concatenated, and processed through an MLP to produce a c_trunk-dim
    vector that's broadcast-added to trunk features before policy head.
    """

    def __init__(self, pos_len=19, c_trunk=384, hidden_dim=128, num_history=5):
        super().__init__()
        self.pos_len = pos_len
        self.num_history = num_history
        embed_per = hidden_dim // 2

        # +1 for padding index (pass/missing moves)
        self.row_embed = nn.Embedding(pos_len + 1, embed_per, padding_idx=pos_len)
        self.col_embed = nn.Embedding(pos_len + 1, embed_per, padding_idx=pos_len)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_history, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_trunk),
        )

        # Zero-init output so history module starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, input_spatial, trunk_out):
        rows, cols, has_move = extract_move_coords(input_spatial)
        N = rows.shape[0]

        # Replace -1 with padding index
        rows_pad = torch.where(has_move, rows, torch.full_like(rows, self.pos_len))
        cols_pad = torch.where(has_move, cols, torch.full_like(cols, self.pos_len))

        # Embed: N x 5 x embed_per
        row_emb = self.row_embed(rows_pad)
        col_emb = self.col_embed(cols_pad)

        # Concat row+col: N x 5 x hidden_dim
        move_emb = torch.cat([row_emb, col_emb], dim=-1)

        # Flatten and MLP: N x (5*hidden_dim) → N x c_trunk
        out = self.mlp(move_emb.reshape(N, -1))

        # Broadcast to spatial: N x c_trunk x 1 x 1
        return out.unsqueeze(-1).unsqueeze(-1)


class TrunkFeatureLSTM(nn.Module):
    """Extracts trunk CNN features at past move locations, runs LSTM.

    After the trunk computes a (c_trunk x H x W) feature map, this module:
    1. Looks up the feature vector at each of the 5 past move locations
    2. Feeds the sequence through an LSTM (oldest→newest)
    3. Projects the final hidden state to c_trunk dimensions
    4. Broadcast-adds to trunk features before policy head
    """

    def __init__(self, c_trunk=384, hidden_dim=128, num_history=5):
        super().__init__()
        self.c_trunk = c_trunk
        self.num_history = num_history

        self.lstm = nn.LSTM(c_trunk, hidden_dim, batch_first=True)
        self.project = nn.Linear(hidden_dim, c_trunk)

        # Zero-init projection so history starts as identity
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, input_spatial, trunk_out):
        rows, cols, has_move = extract_move_coords(input_spatial)
        N = trunk_out.shape[0]
        device = trunk_out.device

        batch_idx = torch.arange(N, device=device)

        # Extract features at each move location: list of N x C tensors
        features = []
        for i in range(self.num_history):
            r = rows[:, i].clamp(min=0)
            c = cols[:, i].clamp(min=0)
            feat = trunk_out[batch_idx, :, r, c]  # N x C
            # Zero out features for missing/pass moves
            feat = feat * has_move[:, i : i + 1].float()
            features.append(feat)

        # Reverse so sequence goes oldest→newest (channel 13→9)
        features = features[::-1]

        # Stack: N x 5 x C
        seq = torch.stack(features, dim=1)

        # LSTM: N x 5 x hidden_dim
        lstm_out, (h_n, _) = self.lstm(seq)

        # Final hidden: N x hidden_dim
        h_final = h_n.squeeze(0)

        # Project: N x c_trunk
        out = self.project(h_final)

        return out.unsqueeze(-1).unsqueeze(-1)


class CombinedHistory(nn.Module):
    """Combines MoveSequenceEmbedding and TrunkFeatureLSTM."""

    def __init__(self, pos_len=19, c_trunk=384, hidden_dim=128, num_history=5):
        super().__init__()
        self.embedding = MoveSequenceEmbedding(pos_len, c_trunk, hidden_dim, num_history)
        self.lstm = TrunkFeatureLSTM(c_trunk, hidden_dim, num_history)

        # Combine both outputs with a learned gate
        self.gate = nn.Linear(c_trunk * 2, c_trunk)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, input_spatial, trunk_out):
        emb_out = self.embedding(input_spatial, trunk_out).squeeze(-1).squeeze(-1)  # N x C
        lstm_out = self.lstm(input_spatial, trunk_out).squeeze(-1).squeeze(-1)  # N x C
        combined = torch.cat([emb_out, lstm_out], dim=-1)  # N x 2C
        out = self.gate(combined)  # N x C
        return out.unsqueeze(-1).unsqueeze(-1)


def _register_history_params_in_reg_dict(model):
    """Patch model.add_reg_dict to include history module params."""
    original_add_reg_dict = model.add_reg_dict

    def patched_add_reg_dict(reg_dict):
        original_add_reg_dict(reg_dict)
        for name, param in model.history_module.named_parameters():
            reg_dict["normal"].append(param)

    model.add_reg_dict = patched_add_reg_dict


def apply_move_history_to_model(
    model, module_type="lstm", c_trunk=384, hidden_dim=128, pos_len=19
):
    """Apply a move history module to the model.

    Injects history features between trunk and policy head using a forward
    pre-hook on the policy head. The history module's output is broadcast-added
    to the trunk features before they enter the policy head.

    Args:
        model: KataGo Model instance
        module_type: 'embedding', 'lstm', or 'combined'
        c_trunk: trunk channel count (384 for b18c384)
        hidden_dim: internal dimension of history module
        pos_len: board size (19)

    Returns:
        model, num_trainable_params
    """
    if module_type == "embedding":
        module = MoveSequenceEmbedding(pos_len=pos_len, c_trunk=c_trunk, hidden_dim=hidden_dim)
    elif module_type == "lstm":
        module = TrunkFeatureLSTM(c_trunk=c_trunk, hidden_dim=hidden_dim)
    elif module_type == "combined":
        module = CombinedHistory(pos_len=pos_len, c_trunk=c_trunk, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown module_type: {module_type}")

    # Move to same device as model
    device = next(model.parameters()).device
    module = module.to(device)

    # Register as submodule so it's in state_dict
    model.history_module = module

    # Patch forward to cache input_spatial for the hook
    original_forward = model.forward

    def new_forward(input_spatial, input_global, input_meta=None, extra_outputs=None):
        model._cached_input_spatial = input_spatial.detach()
        return original_forward(input_spatial, input_global, input_meta, extra_outputs)

    model.forward = new_forward

    # Pre-hook on policy_head: inject history features into trunk output
    def policy_pre_hook(module_head, args):
        x = args[0]  # trunk features: N x c_trunk x H x W
        input_spatial = model._cached_input_spatial
        history_delta = model.history_module(input_spatial, x)
        new_x = x + history_delta
        return (new_x,)

    model.policy_head.register_forward_pre_hook(policy_pre_hook)

    # Also hook intermediate policy head if it exists
    if hasattr(model, "intermediate_policy_head") and model.intermediate_policy_head is not None:
        model.intermediate_policy_head.register_forward_pre_hook(policy_pre_hook)

    # Register params in reg_dict for optimizer
    _register_history_params_in_reg_dict(model)

    # Count params
    history_params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"=== MOVE HISTORY MODULE ({module_type}) ===")
    print(f"  History module params: {history_params:,}")
    print(f"  Total trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"=== APPLIED ===")

    return model, history_params
