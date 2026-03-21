#!/usr/bin/env python3
"""Pre-flight validation for the FR3 IK pick-and-place system.

Loads the MuJoCo model, verifies all components are present, tests IK
solutions for critical waypoints, and checks the stone resting position.

Usage:
    python scripts/validate.py
    python scripts/validate.py --stone 0.35 -0.1 --target 0.55 0.2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import mujoco  # noqa: E402
from controller_ik import (  # noqa: E402
    PickPlaceConfig,
    IKSolver,
    SuctionController,
    Diagnostics,
)

logger = logging.getLogger("validate")


def parse_args():
    p = argparse.ArgumentParser(description="Validate FR3 IK pick-place setup")
    p.add_argument("--stone", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PickPlaceConfig()
    cfg.model_path = str(_PROJECT_ROOT / cfg.model_path)
    if args.stone:
        cfg.stone_init_pos = np.array(
            [args.stone[0], args.stone[1], cfg.table_height + cfg.stone_half_height]
        )
    if args.target:
        cfg.target_pos = np.array(
            [args.target[0], args.target[1], cfg.table_height + cfg.stone_half_height]
        )

    # --- Load model ---
    logger.info("Loading model: %s", cfg.model_path)
    try:
        model = mujoco.MjModel.from_xml_path(cfg.model_path)
    except Exception as e:
        logger.error("Failed to compile model: %s", e)
        sys.exit(1)
    data = mujoco.MjData(model)
    logger.info("Model compiled OK (nq=%d, nv=%d, nu=%d)", model.nq, model.nv, model.nu)

    # --- Build components ---
    ik = IKSolver(model, cfg)
    suction = SuctionController(model, cfg)
    diag = Diagnostics(model, data, cfg, ik, suction)

    # --- Run all checks ---
    passed = diag.run_all_checks()

    # --- Extra: workspace sweep ---
    logger.info("-" * 60)
    logger.info("Workspace sweep (grid of table positions):")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    reachable = 0
    total = 0
    for x in np.linspace(0.2, 0.7, 6):
        for y in np.linspace(-0.25, 0.25, 6):
            target = np.array([x, y, cfg.stone_grasp_height])
            data.qpos[: cfg.n_joints] = cfg.home_q
            mujoco.mj_forward(model, data)
            _, ok, err = ik.solve(data, target)
            total += 1
            if ok:
                reachable += 1
            mark = "+" if ok else "x"
            logger.debug("  [%s] (%.2f, %.2f) err=%.4f", mark, x, y, err)
    logger.info("  Reachable: %d/%d (%.0f%%)", reachable, total, 100 * reachable / total)

    # --- Summary ---
    logger.info("=" * 60)
    if passed:
        logger.info("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        logger.error("SOME CHECKS FAILED — review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
