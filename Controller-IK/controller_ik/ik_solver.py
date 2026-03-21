"""Damped least-squares IK solver for MuJoCo models."""

import logging
import numpy as np
import mujoco

from .config import PickPlaceConfig

logger = logging.getLogger(__name__)


class IKSolver:
    """Damped least-squares inverse kinematics with null-space optimization.

    Supports position-only (3-DOF) and position+orientation (6-DOF) modes.
    Uses null-space projection to bias the solution toward the home configuration.
    """

    def __init__(self, model: mujoco.MjModel, config: PickPlaceConfig):
        self.model = model
        self.cfg = config
        self.nj = config.n_joints

        # Look up site/body IDs
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.tcp_site_name)
        if self.tcp_site_id < 0:
            raise ValueError(f"Site '{config.tcp_site_name}' not found in model")

        # Separate MjData for IK computation (avoids disturbing the simulation)
        self.ik_data = mujoco.MjData(model)

        # Pre-allocate Jacobian buffers (3 x nv)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

        # Compute and cache the approach orientation from home config
        self._approach_quat = None
        self._compute_approach_orientation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        sim_data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve IK for the given Cartesian target.

        Args:
            sim_data: Current simulation data (reads qpos from it).
            target_pos: (3,) desired TCP position in world frame.
            target_quat: (4,) desired TCP orientation quaternion (w,x,y,z).
                         If None and ik_use_orientation is True, uses the
                         pre-computed approach orientation.

        Returns:
            q: (n_joints,) joint angles (radians).
            converged: True if error is below tolerance.
            error: final error norm.
        """
        if target_quat is None and self.cfg.ik_use_orientation:
            target_quat = self._approach_quat

        # Copy current robot state into IK workspace
        self.ik_data.qpos[:] = sim_data.qpos[:]
        self.ik_data.qvel[:] = 0

        use_ori = target_quat is not None
        if use_ori:
            target_mat = np.zeros(9)
            mujoco.mju_quat2Mat(target_mat, target_quat)
            target_mat = target_mat.reshape(3, 3)

        for _ in range(self.cfg.ik_max_iter):
            mujoco.mj_fwdPosition(self.model, self.ik_data)

            # Position error
            site_pos = self.ik_data.site_xpos[self.tcp_site_id].copy()
            pos_err = target_pos - site_pos

            # Orientation error (axis-angle)
            if use_ori:
                site_mat = self.ik_data.site_xmat[self.tcp_site_id].reshape(3, 3)
                ori_err = self._orientation_error(site_mat, target_mat)
                err = np.concatenate([pos_err, self.cfg.ik_ori_weight * ori_err])
            else:
                err = pos_err

            err_norm = np.linalg.norm(pos_err)
            ori_norm = np.linalg.norm(ori_err) if use_ori else 0.0
            if err_norm < self.cfg.ik_pos_tol and ori_norm < self.cfg.ik_ori_tol:
                return self.ik_data.qpos[: self.nj].copy(), True, err_norm

            # Compute Jacobian at TCP site
            self.jacp[:] = 0
            self.jacr[:] = 0
            mujoco.mj_jacSite(self.model, self.ik_data, self.jacp, self.jacr, self.tcp_site_id)

            # Extract robot-joint columns only
            Jp = self.jacp[:, : self.nj]
            if use_ori:
                Jr = self.jacr[:, : self.nj]
                J = np.vstack([Jp, self.cfg.ik_ori_weight * Jr])
            else:
                J = Jp

            # Damped least-squares: dq = J^T (J J^T + lambda^2 I)^{-1} err
            lam = self.cfg.ik_damping
            JJT = J @ J.T + lam**2 * np.eye(J.shape[0])

            # Solve for both err and J in one call (shared factorization)
            if self.cfg.ik_null_space_gain > 0:
                rhs = np.column_stack([err.reshape(-1, 1), J])
                sol = np.linalg.solve(JJT, rhs)
                dq = J.T @ sol[:, 0]
                JpJ = J.T @ sol[:, 1:]
            else:
                dq = J.T @ np.linalg.solve(JJT, err)

            # Null-space bias toward home configuration
            if self.cfg.ik_null_space_gain > 0:
                null_proj = np.eye(self.nj) - JpJ
                q_current = self.ik_data.qpos[: self.nj]
                dq_null = null_proj @ (self.cfg.home_q - q_current)
                dq += self.cfg.ik_null_space_gain * dq_null

            # Clamp step size
            dq_norm = np.linalg.norm(dq)
            if dq_norm > self.cfg.ik_max_dq:
                dq *= self.cfg.ik_max_dq / dq_norm

            # Apply
            self.ik_data.qpos[: self.nj] += self.cfg.ik_step_size * dq

            # Enforce joint limits
            for i in range(self.nj):
                lo, hi = self.model.jnt_range[i]
                qadr = self.model.jnt_qposadr[i]
                self.ik_data.qpos[qadr] = np.clip(self.ik_data.qpos[qadr], lo, hi)

        # Did not converge — return best effort
        return self.ik_data.qpos[: self.nj].copy(), False, err_norm

    @property
    def approach_quat(self) -> np.ndarray:
        """Pre-computed quaternion for top-down approach orientation."""
        return self._approach_quat.copy()

    def get_tcp_pose(self, sim_data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return current TCP (position, rotation_matrix) from simulation data."""
        pos = sim_data.site_xpos[self.tcp_site_id].copy()
        mat = sim_data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        return pos, mat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_approach_orientation(self):
        """Compute the 'suction pointing straight down' quaternion.

        We start from the home configuration, compute the TCP orientation,
        then adjust it so the TCP z-axis points exactly downward.
        """
        self.ik_data.qpos[: self.nj] = self.cfg.home_q
        self.ik_data.qvel[:] = 0
        mujoco.mj_fwdPosition(self.model, self.ik_data)

        home_mat = self.ik_data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
        tcp_z = home_mat[:, 2]

        # Build a rotation matrix with z-axis = [0, 0, -1] (pointing down).
        # Keep the x-axis projection onto the horizontal plane from home pose.
        z_desired = np.array([0.0, 0.0, -1.0])
        x_home = home_mat[:, 0]
        # Project x_home onto horizontal plane
        x_proj = x_home - np.dot(x_home, z_desired) * z_desired
        x_norm = np.linalg.norm(x_proj)
        if x_norm < 1e-6:
            x_proj = np.array([1.0, 0.0, 0.0])
        else:
            x_proj /= x_norm

        y_desired = np.cross(z_desired, x_proj)
        y_desired /= np.linalg.norm(y_desired)
        x_desired = np.cross(y_desired, z_desired)

        approach_mat = np.column_stack([x_desired, y_desired, z_desired])

        self._approach_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self._approach_quat, approach_mat.flatten())

        logger.info(
            "Approach orientation computed. TCP z-axis in home config: %s",
            tcp_z.round(3),
        )

    @staticmethod
    def _orientation_error(current_mat: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
        """Compute orientation error as a 3D rotation vector.

        Returns the axis-angle vector that rotates current_mat into target_mat.
        """
        error_mat = target_mat @ current_mat.T
        trace = np.clip((np.trace(error_mat) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(trace)
        if angle < 1e-6:
            return np.zeros(3)
        # Extract rotation axis
        axis = np.array(
            [
                error_mat[2, 1] - error_mat[1, 2],
                error_mat[0, 2] - error_mat[2, 0],
                error_mat[1, 0] - error_mat[0, 1],
            ]
        )
        sin_angle = np.sin(angle)
        if abs(sin_angle) < 1e-6:
            # Near pi: extract axis from diagonal (R = 2*n*n^T - I)
            diag = np.diag(error_mat)
            axis = np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, 1.0))
            if error_mat[0, 1] + error_mat[1, 0] < 0:
                axis[1] = -axis[1]
            if error_mat[0, 2] + error_mat[2, 0] < 0:
                axis[2] = -axis[2]
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis /= norm
        else:
            axis /= 2.0 * sin_angle
        return angle * axis
