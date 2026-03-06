# Data Conversion Scripts

End-to-end pipeline for converting raw SGF game records into shuffled NPZ training data for KataGo fine-tuning.

## Overview

```
SGF files (human game records)
    |
    v
[convert_dataset.sh]  <-- main entry point, orchestrates everything
    |
    |-- Step 1: SGF -> raw NPZ        (via KataGo C++ binary: writetrainingdata_simple)
    |-- Step 1.5: Enrich with soft     (optional, via enrich_npz.py)
    |             policy targets
    |-- Step 2: Split train/val
    |-- Step 3: Shuffle                (via KataGo's python/shuffle.py)
    |
    v
shuffled/{train,val}/  <-- ready for training
```

## Scripts

### convert_dataset.sh -- Main Pipeline

Runs the full conversion from a folder of SGF files to shuffled train/val NPZ batches.

```bash
# Basic usage
./convert_dataset.sh -i /path/to/sgfs -o /path/to/output

# Filter to one player's moves only
./convert_dataset.sh \
    -i Training_Dataset/Raw_files/Game_Record_Ran \
    -o Training_Dataset/ran_dataset \
    --only-player "然然燃燃" \
    --val-games 20

# With soft-target enrichment (requires a model checkpoint)
./convert_dataset.sh \
    -i Training_Dataset/Raw_files/All_human_pros \
    -o Training_Dataset/human_pros_softtarget \
    --enrich-checkpoint models/b28_best.ckpt \
    --enrich-alpha 0.5 --enrich-beta 0.5
```

**Key arguments:**
| Arg | Default | Description |
|-----|---------|-------------|
| `-i, --input` | (required) | Directory containing `.sgf` files |
| `-o, --output` | (required) | Output directory for processed data |
| `-c, --config` | `simple_training_config.cfg` | KataGo board config (in repo root) |
| `--train-ratio` | 0.9 | Fraction of data for training |
| `--only-player` | (none) | Only extract moves by this player name |
| `--val-games` | (none) | Fixed validation game count (overrides ratio) |
| `--enrich-checkpoint` | (none) | Model checkpoint for soft-target enrichment |
| `--enrich-alpha` | 0.5 | Blend weight: human move vs quality-band |
| `--enrich-beta` | 0.5 | Blunder upweighting scale |
| `--enrich-batch-size` | 256 | Inference batch size for enrichment |

**Output structure:**
```
output_dir/
  raw_npz/              # Intermediate NPZ from C++ converter
  enriched_npz_compressed/  # (if enrichment enabled) Enriched + compressed
  raw_train/            # NPZ files linked for training split
  raw_val/              # NPZ files linked for validation split
  shuffled/
    train/              # FINAL: shuffled training batches (.npz + .json)
    val/                # FINAL: shuffled validation batches (.npz + .json)
```

**Prerequisites:**
- KataGo binary built at `cpp/build/katago` or `cpp/katago`
- Python 3 with numpy, torch (for enrichment only)

---

### enrich_npz.py -- Soft Policy Target Enrichment

Adds quality-band soft policy targets to raw NPZ files using a KataGo model checkpoint.

**What it does:** For each board position, runs KataGo inference and builds a probability distribution over moves at a similar quality level (in logit space) to the human's actual move. This teaches the model *which alternative moves are reasonable at the human's skill level*, not just the single move played.

**How quality-band works:**
1. Get KataGo's policy logits for all legal moves
2. Find the logit of the human's actual move
3. Apply a Gaussian kernel in logit space centered on that value -- moves at similar quality get high weight
4. Blend: `soft_target = alpha * one_hot_human + (1-alpha) * quality_band`

**Additional features:**
- Blunder-weighted samples: `weight = 1 + beta * (game_closeness * logit_gap)` -- mistakes in close games get upweighted
- Sparsification: small probability values zeroed out and renormalized for smaller files
- `--compress-output`: uses `np.savez_compressed` for further size reduction

```bash
# Standalone usage (usually called by convert_dataset.sh)
python enrich_npz.py \
    --input-dir data/raw_npz \
    --output-dir data/enriched_npz \
    --checkpoint models/b28_best.ckpt \
    --alpha 0.5 --beta 0.5 --compress-output
```

---

### compress_enriched_npz.py -- Post-hoc Compression

Re-sparsifies and compresses already-enriched NPZ files. Useful if enrichment was run without `--compress-output`.

```bash
python compress_enriched_npz.py --input-dir data/enriched_npz --threshold 50
python compress_enriched_npz.py --in-place  # overwrites originals
```

---

### download_waltheri_sgfs.py -- Download from Waltheri

Scrapes professional game records from ps.waltheri.net using Playwright (headless browser).

```bash
pip install playwright tqdm && playwright install chromium
python download_waltheri_sgfs.py                          # all players
python download_waltheri_sgfs.py --only-player "Lee Sedol" --workers 2
```

### download_19x19.py -- Download from Chinese Sites

Downloads SGFs from Chinese Go record libraries (requires manual login). Opens a headed browser for authentication.

```bash
python download_19x19.py --url "https://..." --output ./sgf_output
```

---

## NPZ Format

The NPZ files contain these arrays (per board position):

| Array | Shape | Description |
|-------|-------|-------------|
| `binaryInputNCHWPacked` | `[N, C, packed]` | Bit-packed board features (stone positions, liberties, etc.) |
| `globalInputNC` | `[N, G]` | Global features (komi, turn number, etc.) |
| `policyTargetsNCMove` | `[N, 1, 362]` | Move labels (362 = 19*19 + pass). int16, sums to 10000. One-hot for raw, soft distribution after enrichment |
| `globalTargetsNC` | `[N, T]` | Global targets (win/loss, score, weights). Column 25 = sample weight |
| `scoreDistrN` | `[N, S]` | Score distribution target |
| `valueTargetsNCHW` | `[N, C, H, W]` | Spatial value targets (ownership) |

After enrichment, these are added:
| `logitGapN` | `[N]` | How much worse the human's move was vs best (logit space) |
| `gameClosenessN` | `[N]` | 4*winrate*(1-winrate), peaks at 0.5 winrate |

## Typical Workflow

```bash
# 1. Get SGF files (download or use existing)
python download_waltheri_sgfs.py -o Training_Dataset/Raw_files/All_human_pros

# 2. Convert + enrich + shuffle in one command
./convert_dataset.sh \
    -i Training_Dataset/Raw_files/All_human_pros \
    -o Training_Dataset/all_human_pros_enriched \
    --enrich-checkpoint models/b28_best.ckpt

# 3. Train (from Training_Code/)
python Training_Code/train_policy_only.py \
    -datadir Training_Dataset/all_human_pros_enriched/shuffled \
    -initial-checkpoint models/b18c384.ckpt \
    -traindir runs/my_run \
    -pos-len 19 -batch-size 512
```
