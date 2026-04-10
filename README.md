# Embodied Robotic Agent for Stylistic Go Imitation

An embodied robotic agent that imitates the playing style of a specific Go professional. Three components: player-style AI, board perception, and robotic stone placement.

## Results

| Component | Metric | Value |
|-----------|--------|-------|
| **Player AI** | Top-1 accuracy (Nie Weiping) | 59.4% (+7.7% over baseline) |
| **Placement** | Median 3D error | 4.2 mm |
| **Placement** | Success rate | 100% |

## Repository Structure

```
KataGo/                          # Player-style fine-tuning
  Training_Code/                 # Fine-tuning scripts
    train_policy_only.py         # Main training (5 freeze modes)
    evaluate_models.py           # Top-k move prediction eval
  Data_Conversion_Scripts/       # SGF → NPZ data pipeline
  python/katago/                 # KataGo runtime (model, data loading)

Controller-IK/                   # Robotic stone placement
  controller_ik/                 # MuJoCo IK pick-and-place controller
  franka_rl/                     # PPO policy (Isaac Lab)
    config.py                    # Environment and training config
    env.py                       # Parallel GPU environment
    teacher.py                   # Online IK teacher
    rewards.py                   # ManiSkill3-style staircase reward
    train.py                     # Training entry point
    eval.py                      # Placement precision evaluation
  models/                        # MuJoCo scene and Franka meshes
  scripts/                       # IK controller scripts

vision/                          # Board perception
  go_stone_detector.py           # Grounded-SAM-2 stone detection

paper/                           # Progress report (LaTeX)
```

## Setup

### KataGo Fine-Tuning

```bash
pip install torch numpy wandb

# Download base model from KataGo releases:
# https://github.com/lightvector/KataGo → b18c384nbt-humanv0

# Prepare data (requires compiled KataGo binary for SGF → NPZ conversion)
bash KataGo/Data_Conversion_Scripts/convert_dataset.sh

# Train
PYTHONPATH=KataGo/python python KataGo/Training_Code/train_policy_only.py \
  -traindir runs/my-run \
  -datadir <path-to-shuffled-npz> \
  -initial-checkpoint <path-to-base-model> \
  -pos-len 19 -batch-size 128 \
  -policy-loss-only -reset-train-state -disable-swa -disable-warmup \
  -base-per-sample-lr 0.00001 \
  -freeze-value-head

# Evaluate
PYTHONPATH=KataGo/python python KataGo/Training_Code/evaluate_models.py \
  --sgf-dir <path-to-sgfs> \
  --only-player "Nie Weiping" \
  --checkpoints <checkpoint-path>
```

### IK Controller (MuJoCo)

```bash
pip install mujoco numpy

cd Controller-IK
python scripts/run.py --headless          # Run pick-and-place
python scripts/validate.py                # Validate reachability
python scripts/eval_ik_precision.py       # Measure placement accuracy
```

### RL Policy (Isaac Lab)

Requires [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) with IsaacSim 5.x.

```bash
cd Controller-IK
python franka_rl/train.py --headless --num_envs 2048   # Train
python franka_rl/eval.py --headless --checkpoint <path> # Evaluate
```

### Board Perception

Requires [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2).

```bash
python vision/go_stone_detector.py --image <path-to-board-photo>
```

## Key Design Decisions

- **Delta-from-current actions**: `q_target = q_current + a * 0.2 rad`. Avoids compounding error from absolute joint targets.
- **Online IK teacher + PPO from scratch**: Combined task + imitation reward, annealed over training. No behavioral cloning pretraining needed.
- **Adaptive suction curriculum**: Threshold tightens from 50 mm → 1 mm based on measured accuracy.
- **Era-aware fine-tuning**: Go styles differ by decade; fine-tune on the target player's era first, then specialize.

## Citation

```
@article{qi2026embodied,
  title={Embodied Robotic Agent for Stylistic Go Imitation},
  author={Qi, Ran},
  year={2026}
}
```

## License

MIT
