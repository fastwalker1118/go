#!/usr/bin/env python3
"""Evaluate IK controller placement precision with tight suction threshold.

Runs many episodes with randomized stone/target positions and measures
final placement error. Uses 1mm suction threshold (near-zero tolerance).

Usage:
    python scripts/eval_ik_precision.py --num_episodes 200 --suction_dist 0.001
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import mujoco
from controller_ik import (
    PickPlaceConfig,
    IKSolver,
    SuctionController,
    PickPlaceStateMachine,
    State,
)

logger = logging.getLogger("eval_ik")


def run_one_episode(model, data, cfg):
    """Run one pick-and-place episode, return (success, placement_error_3d, placement_error_xy)."""
    ik = IKSolver(model, cfg)
    suction = SuctionController(model, cfg)
    sm = PickPlaceStateMachine(cfg, ik, suction)

    nj = cfg.n_joints
    M_full = np.zeros((model.nv, model.nv))
    ctc_kp, ctc_kd = 900.0, 60.0

    def apply_ctc(q_target):
        q_err = q_target - data.qpos[:nj]
        qdot = data.qvel[:nj]
        a_des = ctc_kp * q_err - ctc_kd * qdot
        mujoco.mj_fullM(model, M_full, data.qM)
        tau = M_full[:nj, :nj] @ a_des + data.qfrc_bias[:nj]
        data.ctrl[:nj] = tau

    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Set stone position
    stone_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "stone_free")
    if stone_jnt_id >= 0:
        qadr = model.jnt_qposadr[stone_jnt_id]
        data.qpos[qadr : qadr + 3] = cfg.stone_init_pos
        data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]

    # Update target site
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
    if target_site_id >= 0:
        model.site_pos[target_site_id] = [
            cfg.target_pos[0],
            cfg.target_pos[1],
            cfg.table_height + 0.001,
        ]

    mujoco.mj_forward(model, data)

    # Settle stone
    for _ in range(500):
        apply_ctc(cfg.home_q)
        mujoco.mj_step(model, data)

    # Run task
    step = 0
    while data.time < cfg.max_sim_time and sm.state != State.DONE:
        if step % cfg.ik_recompute_steps == 0:
            sm.update(data, data.time)
        apply_ctc(sm.q_target)
        mujoco.mj_step(model, data)
        step += 1

    if sm.state == State.DONE:
        stone_pos = suction.stone_position(data)
        err_3d = np.linalg.norm(stone_pos - cfg.target_pos)
        err_xy = np.linalg.norm(stone_pos[:2] - cfg.target_pos[:2])
        return True, err_3d, err_xy
    else:
        return False, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument(
        "--suction_dist", type=float, default=0.001, help="Suction attach distance (m)"
    )
    parser.add_argument(
        "--randomize", action="store_true", help="Randomize stone/target positions"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cfg = PickPlaceConfig()
    cfg.model_path = str(_PROJECT_ROOT / cfg.model_path)
    cfg.suction_attach_dist = args.suction_dist
    cfg.max_sim_time = 20.0
    cfg.realtime_factor = 0.0

    model = mujoco.MjModel.from_xml_path(cfg.model_path)
    data = mujoco.MjData(model)

    successes = 0
    errors_3d = []
    errors_xy = []

    # Define random ranges for stone/target
    stone_x_range = (0.15, 0.35)
    stone_y_range = (-0.25, -0.10)
    target_x_range = (0.35, 0.55)
    target_y_range = (0.0, 0.25)

    print(
        f"Evaluating IK controller: {args.num_episodes} episodes, "
        f"suction_dist={args.suction_dist * 1000:.1f}mm"
    )
    print(f"Randomize positions: {args.randomize}")
    print()

    for ep in range(args.num_episodes):
        if args.randomize:
            sx = np.random.uniform(*stone_x_range)
            sy = np.random.uniform(*stone_y_range)
            tx = np.random.uniform(*target_x_range)
            ty = np.random.uniform(*target_y_range)
            cfg.stone_init_pos = np.array([sx, sy, cfg.table_height + cfg.stone_half_height])
            cfg.target_pos = np.array([tx, ty, cfg.table_height + cfg.stone_half_height])

        success, err_3d, err_xy = run_one_episode(model, data, cfg)
        if success:
            successes += 1
            errors_3d.append(err_3d)
            errors_xy.append(err_xy)

        if (ep + 1) % 20 == 0:
            sr = successes / (ep + 1) * 100
            print(f"  Episode {ep+1}/{args.num_episodes}: success_rate={sr:.1f}%")

    errors_3d = np.array(errors_3d) if errors_3d else np.array([0.0])
    errors_xy = np.array(errors_xy) if errors_xy else np.array([0.0])

    print(f"\n{'='*60}")
    print(f"IK CONTROLLER EVALUATION ({args.num_episodes} episodes)")
    print(f"  Suction distance: {args.suction_dist * 1000:.1f}mm")
    print(f"  Randomized: {args.randomize}")
    print(f"{'='*60}")
    print(
        f"Success rate: {successes}/{args.num_episodes} ({successes/args.num_episodes*100:.1f}%)"
    )
    if len(errors_3d) > 1:
        print(
            f"3D Error: mean={errors_3d.mean() * 1000:.2f}mm, std={errors_3d.std() * 1000:.2f}mm, "
            f"median={np.median(errors_3d) * 1000:.2f}mm"
        )
        print(
            f"XY Error: mean={errors_xy.mean() * 1000:.2f}mm, std={errors_xy.std() * 1000:.2f}mm"
        )
        print(
            f"Percentiles (3D): "
            f"p25={np.percentile(errors_3d, 25) * 1000:.2f}mm, "
            f"p50={np.percentile(errors_3d, 50) * 1000:.2f}mm, "
            f"p75={np.percentile(errors_3d, 75) * 1000:.2f}mm, "
            f"p90={np.percentile(errors_3d, 90) * 1000:.2f}mm, "
            f"p95={np.percentile(errors_3d, 95) * 1000:.2f}mm"
        )
        print(f"Within 0.5mm: {(errors_3d < 0.0005).mean()*100:.1f}%")
        print(f"Within 1mm: {(errors_3d < 0.001).mean()*100:.1f}%")
        print(f"Within 2mm: {(errors_3d < 0.002).mean()*100:.1f}%")
        print(f"Within 5mm: {(errors_3d < 0.005).mean()*100:.1f}%")
        print(f"Within 10mm: {(errors_3d < 0.01).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
