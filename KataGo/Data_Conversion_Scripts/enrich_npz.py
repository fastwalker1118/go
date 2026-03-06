#!/usr/bin/env python3
"""
Enrich NPZ training data with soft policy targets from KataGo's raw neural net.

Quality-band approach: For each position, builds a
distribution over moves at similar quality level (in logit space) to the human's
actual move. Uses KataGo's value head (winrate) for game-closeness-weighted
blunder detection. Output is sparsified for smaller files. Saves logitGapN and
gameClosenessN metadata for optional auxiliary head training.

Usage:
    python enrich_npz.py \
        --input-dir data/my_dataset/raw_npz \
        --output-dir data/my_dataset/enriched_npz \
        --checkpoint /path/to/katago_checkpoint.ckpt \
        --alpha 0.5 --beta 0.5

The output NPZ files have the same format and can be used directly with
convert_dataset.sh's shuffle step or with training scripts.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add KataGo python directory to path so we can import the model
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from katago.train.load_model import load_model
from katago.train import modelconfigs
from katago.train.data_processing_pytorch import build_history_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLICY_LEN = 362  # 19*19 + 1 pass
POLICY_SCALE = 10000  # Scale factor for storing soft policy as int16


def unpack_binary_input(packed: np.ndarray, pos_len: int) -> np.ndarray:
    """Unpack binaryInputNCHWPacked to float32 spatial features."""
    unpacked = np.unpackbits(packed, axis=2)
    unpacked = unpacked[:, :, : pos_len * pos_len]
    n, c, _ = unpacked.shape
    return unpacked.reshape(n, c, pos_len, pos_len).astype(np.float32)


def enrich_single_npz(
    npz_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    model_config: dict,
    h_base: torch.Tensor,
    h_builder: torch.Tensor,
    num_global_features: int,
    device: torch.device,
    alpha: float,
    beta: float,
    temperature: float,
    batch_size: int,
    pos_len: int,
    sparsify_threshold: int,
    compress_output: bool,
):
    """Process one NPZ file: add quality-band soft policy targets and blunder weights."""
    logger.info(f"Processing {npz_path.name}...")

    with np.load(npz_path) as npz:
        binary_packed = npz["binaryInputNCHWPacked"]
        global_input = npz["globalInputNC"]
        policy_targets = npz["policyTargetsNCMove"]
        global_targets = npz["globalTargetsNC"].copy()
        score_distr = npz["scoreDistrN"]
        value_targets = npz["valueTargetsNCHW"]
        metadata_input = npz["metadataInputNC"] if "metadataInputNC" in npz else None
        q_value_targets = npz["qValueTargetsNCMove"] if "qValueTargetsNCMove" in npz else None

    n_samples = binary_packed.shape[0]
    human_moves = np.argmax(policy_targets[:, 0, :], axis=1)

    # Run inference per-batch (unpack only the current batch to avoid OOM)
    all_katago_policy = np.zeros((n_samples, POLICY_LEN), dtype=np.float32)
    all_winrate = np.zeros(n_samples, dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        b_bin_np = unpack_binary_input(binary_packed[start:end], pos_len)
        b_bin = torch.from_numpy(b_bin_np).to(device)
        b_global = torch.from_numpy(global_input[start:end].copy()).to(device)
        b_gtargets = torch.from_numpy(global_targets[start:end]).to(device)

        include_history = b_gtargets[:, 36:41]
        h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)
        b_bin = torch.einsum("bijk,bil->bljk", b_bin, h_matrix)
        b_global = b_global * torch.nn.functional.pad(
            include_history, (0, num_global_features - include_history.shape[1]), value=1.0
        )

        with torch.no_grad():
            outputs = model(b_bin, b_global)

            # Policy head -> softmax
            out_policy = outputs[0][0]
            policy_logits = out_policy[:, 0, :]
            mask = b_bin[:, 0, :, :].reshape(b_bin.shape[0], -1)
            policy_mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            policy_logits = policy_logits + (1.0 - policy_mask) * (-1e9)
            policy_probs = F.softmax(policy_logits, dim=1)

            # Value head -> winrate (P(win for current player))
            out_value = outputs[0][1]
            value_probs = F.softmax(out_value, dim=1)
            winrate = value_probs[:, 0]

        all_katago_policy[start:end] = policy_probs.cpu().numpy()
        all_winrate[start:end] = winrate.cpu().numpy()

    eps = 1e-10
    human_prob = all_katago_policy[np.arange(n_samples), human_moves]

    # Quality-band soft policy: Gaussian kernel in logit space centered on human's move quality
    log_policy = np.log(all_katago_policy + eps)
    log_human = np.log(human_prob + eps)[:, None]
    log_dist_sq = (log_policy - log_human) ** 2
    quality_band = np.exp(-log_dist_sq / (2.0 * temperature**2))
    quality_band = quality_band * (all_katago_policy > 1e-8).astype(np.float32)
    quality_band = quality_band / np.maximum(quality_band.sum(axis=1, keepdims=True), eps)

    one_hot = np.zeros((n_samples, POLICY_LEN), dtype=np.float32)
    one_hot[np.arange(n_samples), human_moves] = 1.0
    soft_policy = alpha * one_hot + (1.0 - alpha) * quality_band

    # Blunder-based weights: game_closeness * logit_gap
    best_prob = all_katago_policy.max(axis=1)
    logit_gap = np.log(np.maximum(best_prob, eps)) - np.log(np.maximum(human_prob, eps))
    logit_gap = np.maximum(logit_gap, 0.0)
    game_closeness = 4.0 * all_winrate * (1.0 - all_winrate)
    blunder = game_closeness * logit_gap
    weight_multiplier = 1.0 + beta * blunder

    # Convert to int16 and sparsify
    scaled = np.clip(np.round(soft_policy * POLICY_SCALE), 0, 32767).astype(np.int16)
    scaled_f = scaled.astype(np.int32)
    mask = scaled_f >= sparsify_threshold
    scaled_f = np.where(mask, scaled_f, 0)
    row_sums = scaled_f.sum(axis=1, keepdims=True)
    all_zero = row_sums.ravel() == 0
    if all_zero.any():
        scaled_f[all_zero, human_moves[all_zero]] = POLICY_SCALE
        row_sums = np.maximum(scaled_f.sum(axis=1, keepdims=True), 1)
    else:
        row_sums = np.maximum(row_sums, 1)
    scaled_f = np.round(scaled_f.astype(np.float64) * POLICY_SCALE / row_sums).astype(np.int32)
    scaled = np.clip(scaled_f, 0, 32767).astype(np.int16)

    new_policy_targets = policy_targets.copy()
    new_policy_targets[:, 0, :] = scaled
    global_targets[:, 25] = global_targets[:, 25] * weight_multiplier

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "binaryInputNCHWPacked": binary_packed,
        "globalInputNC": global_input,
        "policyTargetsNCMove": new_policy_targets,
        "globalTargetsNC": global_targets,
        "scoreDistrN": score_distr,
        "valueTargetsNCHW": value_targets,
        "logitGapN": logit_gap.astype(np.float32),
        "gameClosenessN": game_closeness.astype(np.float32),
    }
    if metadata_input is not None:
        arrays["metadataInputNC"] = metadata_input
    if q_value_targets is not None:
        arrays["qValueTargetsNCMove"] = q_value_targets

    if compress_output:
        np.savez_compressed(output_path, **arrays)
    else:
        np.savez(output_path, **arrays)

    band_size = np.mean(np.sum(quality_band > 0.01, axis=1))
    quality_ratio = human_prob / np.maximum(best_prob, eps)
    top1 = np.mean(quality_ratio >= 0.99) * 100
    logger.info(
        f"  {npz_path.name}: {n_samples} samples, top-1: {top1:.1f}%, "
        f"mean band: {band_size:.1f}, mean weight: {np.mean(weight_multiplier):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Enrich NPZ with quality-band soft policy targets"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing NPZ files from writetrainingdata_simple",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for enriched NPZ files",
    )
    parser.add_argument(
        "--checkpoint",
        "-m",
        type=Path,
        required=True,
        help="Path to KataGo model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight on human's exact move vs quality-band (default: 0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Blunder upweighting scale (default: 0.5). weight = 1 + beta * (game_closeness * logit_gap)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Gaussian sigma in logit space for quality band (default: 1.0, lower=tighter)",
    )
    parser.add_argument(
        "--sparsify-threshold",
        type=int,
        default=50,
        help="Zero out policy values below this (0.5%% of scale=10000) for smaller files (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for model inference (default: 256)",
    )
    parser.add_argument(
        "--pos-len",
        type=int,
        default=19,
        help="Board size (default: 19)",
    )
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Use SWA model if available in checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--compress-output",
        action="store_true",
        help="Use np.savez_compressed per file (smaller output, recommended for pipelines)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not args.checkpoint.is_file():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from {args.checkpoint}...")
    model, swa_model, _ = load_model(
        str(args.checkpoint),
        use_swa=args.use_swa,
        device=device,
        pos_len=args.pos_len,
        verbose=True,
    )
    if args.use_swa and swa_model is not None:
        model = swa_model
        logger.info("Using SWA model")
    model.eval()

    model_config = model.config
    h_base, h_builder = build_history_matrices(model_config, device)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)

    npz_files = sorted(args.input_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No NPZ files in {args.input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(npz_files)} NPZ files")
    logger.info(
        f"Settings: alpha={args.alpha}, beta={args.beta}, temp={args.temperature}, "
        f"sparsify_threshold={args.sparsify_threshold}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for npz_path in npz_files:
        output_path = args.output_dir / npz_path.name
        enrich_single_npz(
            npz_path=npz_path,
            output_path=output_path,
            model=model,
            model_config=model_config,
            h_base=h_base,
            h_builder=h_builder,
            num_global_features=num_global_features,
            device=device,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.temperature,
            batch_size=args.batch_size,
            pos_len=args.pos_len,
            sparsify_threshold=args.sparsify_threshold,
            compress_output=args.compress_output,
        )

    logger.info("Enrichment complete.")


if __name__ == "__main__":
    main()
