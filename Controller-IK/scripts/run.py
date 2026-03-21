#!/usr/bin/env python3
"""FR3 IK Pick-and-Place Controller — main entry point.

Runs a MuJoCo simulation of a Franka FR3 robot picking up a go stone
with a suction cup and placing it at a target location, using inverse
kinematics for motion control.

Usage:
    python scripts/run.py                     # interactive viewer
    python scripts/run.py --headless          # no GUI
    python scripts/run.py --stone 0.35 -0.1   # custom stone xy
    python scripts/run.py --target 0.55 0.2   # custom target xy
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path so `controller_ik` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import mujoco  # noqa: E402

try:
    import mujoco.viewer as _mj_viewer  # noqa: E402
except ImportError:
    _mj_viewer = None
from controller_ik import (  # noqa: E402
    PickPlaceConfig,
    IKSolver,
    SuctionController,
    PickPlaceStateMachine,
    State,
    Diagnostics,
)

logger = logging.getLogger("pick_place")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FR3 IK Pick-and-Place Controller")
    p.add_argument("--headless", action="store_true", help="Run without viewer")
    p.add_argument(
        "--stone",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Stone initial XY position (m)",
    )
    p.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Target placement XY position (m)",
    )
    p.add_argument(
        "--max-time",
        type=float,
        default=30.0,
        help="Maximum simulation time in seconds (default: 30)",
    )
    p.add_argument(
        "--realtime",
        type=float,
        default=1.0,
        help="Realtime factor (0 = max speed, default: 1.0)",
    )
    p.add_argument(
        "--no-ori",
        action="store_true",
        help="Disable orientation control in IK (position-only)",
    )
    p.add_argument(
        "--skip-diag",
        action="store_true",
        help="Skip pre-flight diagnostics",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PickPlaceConfig:
    """Create config, resolving the model path relative to project root."""
    cfg = PickPlaceConfig()
    cfg.model_path = str(_PROJECT_ROOT / cfg.model_path)
    cfg.max_sim_time = args.max_time
    cfg.realtime_factor = args.realtime

    if args.stone:
        cfg.stone_init_pos = np.array(
            [args.stone[0], args.stone[1], cfg.table_height + cfg.stone_half_height]
        )
    if args.target:
        cfg.target_pos = np.array(
            [args.target[0], args.target[1], cfg.table_height + cfg.stone_half_height]
        )
    if args.no_ori:
        cfg.ik_use_orientation = False

    return cfg


def set_stone_position(model, data, cfg):
    """Override stone initial position from config (in case it differs from XML)."""
    stone_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "stone_free")
    if stone_jnt_id < 0:
        return
    qadr = model.jnt_qposadr[stone_jnt_id]
    data.qpos[qadr : qadr + 3] = cfg.stone_init_pos
    data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]  # identity quat

    # Also update the target site visual
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
    if target_site_id >= 0:
        model.site_pos[target_site_id] = [
            cfg.target_pos[0],
            cfg.target_pos[1],
            cfg.table_height + 0.001,
        ]


def run_simulation(model, data, sm, ik, suction, diag, cfg, headless=False):
    """Main simulation loop."""

    # Computed-torque PD gains (critically damped, ~0.15s settling)
    ctc_kp = 900.0  # position gain (rad/s^2 per rad)
    ctc_kd = 60.0  # velocity gain (rad/s^2 per rad/s)

    nj = cfg.n_joints
    M_full = np.zeros((model.nv, model.nv))

    def apply_ctc(q_target):
        """Apply computed torque control for the current target."""
        q_err = q_target - data.qpos[:nj]
        qdot = data.qvel[:nj]
        a_des = ctc_kp * q_err - ctc_kd * qdot
        mujoco.mj_fullM(model, M_full, data.qM)
        tau = M_full[:nj, :nj] @ a_des + data.qfrc_bias[:nj]
        data.ctrl[:nj] = tau

    # Reset to home keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    set_stone_position(model, data, cfg)
    mujoco.mj_forward(model, data)

    # Let stone settle while holding home pose
    logger.info("Settling stone on table...")
    for _ in range(500):
        apply_ctc(cfg.home_q)
        mujoco.mj_step(model, data)

    step = 0
    wall_start = time.monotonic()

    def loop_body():
        nonlocal step
        sim_time = data.time

        # Recompute IK periodically (not every step — CTC interpolates smoothly)
        if step % cfg.ik_recompute_steps == 0:
            sm.update(data, sim_time)

        # Computed torque control
        apply_ctc(sm.q_target)

        # Periodic diagnostics
        if step % cfg.diag_interval == 0:
            diag.print_state(sm.state_name, sim_time)

        # Step physics
        mujoco.mj_step(model, data)
        step += 1

        # Realtime pacing
        if cfg.realtime_factor > 0 and not headless:
            wall_elapsed = time.monotonic() - wall_start
            sim_elapsed = data.time
            target_wall = sim_elapsed / cfg.realtime_factor
            if wall_elapsed < target_wall:
                time.sleep(target_wall - wall_elapsed)

    if headless:
        logger.info("Running headless simulation (max %.1fs)...", cfg.max_sim_time)
        while data.time < cfg.max_sim_time and sm.state != State.DONE:
            loop_body()
    else:
        if _mj_viewer is None:
            logger.error("mujoco.viewer not available. Install with: pip install mujoco")
            logger.info("Falling back to headless mode.")
            while data.time < cfg.max_sim_time and sm.state != State.DONE:
                loop_body()
            return

        logger.info("Launching viewer (close window or Ctrl+C to stop)...")
        with _mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < cfg.max_sim_time:
                loop_body()
                viewer.sync()
                if sm.state == State.DONE:
                    logger.info("Task complete! Viewer remains open.")
                    # Keep viewer alive — freeze physics, just render
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break

    # Final report
    _print_summary(sm, suction, diag, data)


def _print_summary(sm, suction, diag, data):
    """Print task completion summary."""
    logger.info("=" * 60)
    if sm.state == State.DONE:
        stone_pos = suction.stone_position(data)
        target = sm.cfg.target_pos
        placement_err = np.linalg.norm(stone_pos[:2] - target[:2])
        logger.info("TASK COMPLETED SUCCESSFULLY")
        logger.info("  Final stone pos : [%.4f, %.4f, %.4f]", *stone_pos)
        logger.info("  Target pos      : [%.4f, %.4f, %.4f]", *target)
        logger.info("  Placement error : %.4f m (XY)", placement_err)
    else:
        logger.warning("TASK DID NOT COMPLETE (final state: %s)", sm.state_name)
        diag.print_state(sm.state_name, data.time)
    logger.info("  Sim time: %.2f s", data.time)
    logger.info("=" * 60)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = build_config(args)
    logger.info("Loading model: %s", cfg.model_path)

    model = mujoco.MjModel.from_xml_path(cfg.model_path)
    data = mujoco.MjData(model)

    # Build components
    ik = IKSolver(model, cfg)
    suction = SuctionController(model, cfg)
    sm = PickPlaceStateMachine(cfg, ik, suction)
    diag = Diagnostics(model, data, cfg, ik, suction)

    # Pre-flight checks
    if not args.skip_diag:
        if not diag.run_all_checks():
            logger.error("Pre-flight checks failed. Use --skip-diag to bypass.")
            sys.exit(1)

    run_simulation(model, data, sm, ik, suction, diag, cfg, headless=args.headless)


if __name__ == "__main__":
    main()
