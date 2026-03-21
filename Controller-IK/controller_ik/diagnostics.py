"""Diagnostics and validation utilities for the pick-and-place system."""

import logging
import numpy as np
import mujoco

from .config import PickPlaceConfig
from .ik_solver import IKSolver
from .suction import SuctionController

logger = logging.getLogger(__name__)


class Diagnostics:
    """Pre-flight checks, runtime monitoring, and state inspection."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: PickPlaceConfig,
        ik_solver: IKSolver,
        suction: SuctionController,
    ):
        self.model = model
        self.data = data
        self.cfg = config
        self.ik = ik_solver
        self.suction = suction

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------

    def validate_model(self) -> bool:
        """Check that the compiled model has all expected bodies, sites, etc."""
        ok = True
        required_bodies = ["link0", "tcp", "stone", "table"]
        for name in required_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                logger.error("Missing body: %s", name)
                ok = False

        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.tcp_site_name)
        if sid < 0:
            logger.error("Missing site: %s", self.cfg.tcp_site_name)
            ok = False

        eid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, self.cfg.suction_weld_name
        )
        if eid < 0:
            logger.error("Missing equality constraint: %s", self.cfg.suction_weld_name)
            ok = False

        if self.model.nu != self.cfg.n_joints:
            logger.error(
                "Actuator count mismatch: model has %d, config expects %d",
                self.model.nu,
                self.cfg.n_joints,
            )
            ok = False

        if ok:
            logger.info("Model validation PASSED")
        return ok

    def check_home_configuration(self) -> dict:
        """Reset to home keyframe and report TCP pose."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        tcp_pos, tcp_mat = self.ik.get_tcp_pose(self.data)
        tcp_z = tcp_mat[:, 2]

        info = {
            "tcp_pos": tcp_pos,
            "tcp_z_axis": tcp_z,
            "joints": self.data.qpos[: self.cfg.n_joints].copy(),
        }
        logger.info("Home TCP position : [%.4f, %.4f, %.4f]", *tcp_pos)
        logger.info("Home TCP z-axis   : [%.4f, %.4f, %.4f]", *tcp_z)
        logger.info(
            "Home joints (deg) : %s",
            np.rad2deg(info["joints"]).round(1),
        )
        return info

    def check_reachability(self) -> dict:
        """Test IK solutions for the stone position and target position."""
        results = {}

        # Reset to home
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        test_points = {
            "stone_pre_grasp": np.array(
                [
                    self.cfg.stone_init_pos[0],
                    self.cfg.stone_init_pos[1],
                    self.cfg.approach_z,
                ]
            ),
            "stone_grasp": np.array(
                [
                    self.cfg.stone_init_pos[0],
                    self.cfg.stone_init_pos[1],
                    self.cfg.stone_grasp_height,
                ]
            ),
            "target_transport": np.array(
                [
                    self.cfg.target_pos[0],
                    self.cfg.target_pos[1],
                    self.cfg.approach_z,
                ]
            ),
            "target_place": np.array(
                [
                    self.cfg.target_pos[0],
                    self.cfg.target_pos[1],
                    self.cfg.place_height,
                ]
            ),
        }

        all_ok = True
        for label, pos in test_points.items():
            # Reset to home before each test
            self.data.qpos[: self.cfg.n_joints] = self.cfg.home_q
            mujoco.mj_forward(self.model, self.data)

            q, converged, err = self.ik.solve(self.data, pos)
            results[label] = {"q": q, "converged": converged, "error": err}

            status = "OK" if converged else "FAILED"
            logger.info(
                "  IK %-20s : %s  (err=%.5f m, pos=[%.3f, %.3f, %.3f])",
                label,
                status,
                err,
                *pos,
            )
            if not converged:
                all_ok = False

        if all_ok:
            logger.info("Reachability check PASSED — all waypoints reachable")
        else:
            logger.warning("Reachability check FAILED — some waypoints unreachable")

        return results

    def check_stone_on_table(self) -> bool:
        """Verify the stone is resting on the table surface."""
        stone_pos = self.suction.stone_position(self.data)
        stone_z = stone_pos[2]

        table_z = self.cfg.table_height
        expected_z = table_z + self.cfg.stone_half_height
        dz = abs(stone_z - expected_z)

        on_table = dz < 0.005  # 5mm tolerance
        if on_table:
            logger.info(
                "Stone position OK: [%.4f, %.4f, %.4f] (expected z≈%.4f)",
                *stone_pos,
                expected_z,
            )
        else:
            logger.warning(
                "Stone NOT on table: z=%.4f (expected ≈%.4f, delta=%.4f)",
                stone_z,
                expected_z,
                dz,
            )
        return on_table

    # ------------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------------

    def print_state(self, state_name: str, sim_time: float) -> None:
        """Print a compact state summary to the log."""
        tcp_pos = self.data.site_xpos[self.ik.tcp_site_id]
        stone_pos = self.suction.stone_position(self.data)
        dist = self.suction.tcp_stone_distance(self.data)
        _ = self.data.qpos[: self.cfg.n_joints]

        logger.info(
            "[t=%6.2fs] %-11s | TCP=[%+.3f %+.3f %+.3f] | "
            "Stone=[%+.3f %+.3f %+.3f] | Dist=%.4f | Suction=%s",
            sim_time,
            state_name,
            *tcp_pos,
            *stone_pos,
            dist,
            "ON" if self.suction.is_active else "off",
        )

    def run_all_checks(self) -> bool:
        """Run all pre-flight checks. Returns True if all pass."""
        logger.info("=" * 60)
        logger.info("Running pre-flight diagnostics")
        logger.info("=" * 60)

        ok = self.validate_model()
        if not ok:
            return False

        self.check_home_configuration()

        # Let the stone settle on the table (using CTC to hold home pose)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        nj = self.cfg.n_joints
        M = np.zeros((self.model.nv, self.model.nv))
        for _ in range(500):
            q_err = self.cfg.home_q - self.data.qpos[:nj]
            a_des = 900 * q_err - 60 * self.data.qvel[:nj]
            mujoco.mj_fullM(self.model, M, self.data.qM)
            self.data.ctrl[:nj] = M[:nj, :nj] @ a_des + self.data.qfrc_bias[:nj]
            mujoco.mj_step(self.model, self.data)
        self.check_stone_on_table()

        # Reset again for reachability tests
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        reach = self.check_reachability()

        all_reached = all(r["converged"] for r in reach.values())
        logger.info("=" * 60)
        logger.info("Diagnostics %s", "PASSED" if (ok and all_reached) else "FAILED")
        logger.info("=" * 60)
        return ok and all_reached
