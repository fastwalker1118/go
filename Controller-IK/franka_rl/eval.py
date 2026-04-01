"""Evaluate trained Franka pick-and-place policy — measure placement precision."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=5000)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from franka_rl.config import FrankaPickPlaceEnvCfg, FrankaPickPlaceAgentCfg
from franka_rl.env import FrankaPickPlaceEnv

env_cfg = FrankaPickPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs
agent_cfg = FrankaPickPlaceAgentCfg()
env = FrankaPickPlaceEnv(env_cfg, headless=True)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/tmp/eval", device="cuda:0")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cuda:0")

# Monkey-patch reset_idx to capture pre-reset placement data
_original_reset_idx = env.reset_idx
_placement_errors_3d = []
_placement_errors_xy = []


def _patched_reset_idx(env_ids):
    # Capture placement before reset clears flags
    for idx in env_ids:
        i = idx.item()
        if env.has_been_attached[i]:
            stone = env.stone_pos_w[i]
            target = env.target_pos_w[i]
            err_3d = torch.norm(stone - target).item()
            err_xy = torch.norm(stone[:2] - target[:2]).item()
            _placement_errors_3d.append(err_3d)
            _placement_errors_xy.append(err_xy)
    _original_reset_idx(env_ids)


env.reset_idx = _patched_reset_idx

obs, _ = env.reset()
for step in range(args.num_steps):
    with torch.no_grad():
        obs_norm = runner.obs_normalizer(obs)
        actions = policy(obs_norm)
    obs, rew, done, info = env.step(actions)

    if (step + 1) % 500 == 0:
        print(
            f"  Step {step + 1}/{args.num_steps}: "
            f"{len(_placement_errors_3d)} placement episodes collected"
        )

errors_3d = np.array(_placement_errors_3d) if _placement_errors_3d else np.array([0.0])
errors_xy = np.array(_placement_errors_xy) if _placement_errors_xy else np.array([0.0])

print(f"\n{'='*60}")
print(f"EVALUATION RESULTS ({len(errors_3d)} episodes)")
print(f"{'='*60}")
if len(errors_3d) > 1:
    print(
        f"3D Error: mean={errors_3d.mean() * 1000:.1f}mm, std={errors_3d.std() * 1000:.1f}mm, "
        f"median={np.median(errors_3d) * 1000:.1f}mm"
    )
    print(f"XY Error: mean={errors_xy.mean() * 1000:.1f}mm, std={errors_xy.std() * 1000:.1f}mm")
    print(
        f"Percentiles (3D): "
        f"p25={np.percentile(errors_3d, 25) * 1000:.1f}mm, "
        f"p50={np.percentile(errors_3d, 50) * 1000:.1f}mm, "
        f"p75={np.percentile(errors_3d, 75) * 1000:.1f}mm, "
        f"p90={np.percentile(errors_3d, 90) * 1000:.1f}mm, "
        f"p95={np.percentile(errors_3d, 95) * 1000:.1f}mm"
    )
    print(f"Within 1mm: {(errors_3d < 0.001).mean()*100:.1f}%")
    print(f"Within 2mm: {(errors_3d < 0.002).mean()*100:.1f}%")
    print(f"Within 5mm: {(errors_3d < 0.005).mean()*100:.1f}%")
    print(f"Within 10mm: {(errors_3d < 0.01).mean()*100:.1f}%")
    print(f"Within 20mm: {(errors_3d < 0.02).mean()*100:.1f}%")
    print(f"Within 50mm: {(errors_3d < 0.05).mean()*100:.1f}%")
    print(
        f"Pick rate: {len(errors_3d) / (args.num_envs * args.num_steps / 150) * 100:.1f}% "
        f"({len(errors_3d)} picks in {args.num_steps} steps)"
    )
else:
    print("No placements recorded!")

app.close()
