"""Suction gripper controller using MuJoCo weld equality constraint."""

import logging
import numpy as np
import mujoco

from .config import PickPlaceConfig

logger = logging.getLogger(__name__)


class SuctionController:
    """Manages suction cup attach/detach via a MuJoCo weld constraint.

    When activated, enables a weld constraint that rigidly attaches the stone
    to the TCP body.  The relative pose is captured at activation time so the
    stone stays wherever it was when suction engaged.
    """

    NEQDATA = 11  # MuJoCo weld eq_data size

    def __init__(self, model: mujoco.MjModel, config: PickPlaceConfig):
        self.model = model
        self.cfg = config

        # Look up IDs
        self.weld_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, config.suction_weld_name
        )
        if self.weld_id < 0:
            raise ValueError(f"Equality constraint '{config.suction_weld_name}' not found")

        self.tcp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.tcp_body_name)
        self.stone_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, config.stone_body_name
        )
        if self.tcp_body_id < 0 or self.stone_body_id < 0:
            raise ValueError("TCP or stone body not found in model")

        # Geom IDs for contact-based suction check
        self.cup_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "suction_cup")
        self.stone_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stone_geom")
        if self.cup_geom_id < 0 or self.stone_geom_id < 0:
            logger.warning("Suction cup or stone geom not found — falling back to distance check")

        self._active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, data: mujoco.MjData) -> bool:
        """Engage suction if the cup is in physical contact with the stone.

        Checks MuJoCo's contact list for an active contact between the
        suction_cup geom and the stone_geom. Falls back to distance check
        if geom IDs are unavailable.

        Returns True if suction was successfully activated.
        """
        if self._active:
            return True

        if not self._check_contact(data):
            return False

        # Capture current relative pose and store in weld constraint
        self._update_weld_relpose(data)
        self._set_constraint_active(data, True)
        self._active = True
        logger.info("Suction ENGAGED (contact detected)")
        return True

    def deactivate(self, data: mujoco.MjData):
        """Release suction — stone becomes a free body again."""
        if not self._active:
            return
        self._set_constraint_active(data, False)
        self._active = False
        logger.info("Suction RELEASED")

    def tcp_stone_distance(self, data: mujoco.MjData) -> float:
        """Public accessor for the distance between TCP and stone center."""
        return self._tcp_stone_distance(data)

    def stone_position(self, data: mujoco.MjData) -> np.ndarray:
        """Current stone position in world frame."""
        return data.xpos[self.stone_body_id].copy()

    def tcp_position(self, data: mujoco.MjData) -> np.ndarray:
        """Current TCP body position in world frame."""
        return data.xpos[self.tcp_body_id].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_contact(self, data: mujoco.MjData) -> bool:
        """Check for physical contact between suction cup and stone geoms."""
        if self.cup_geom_id < 0 or self.stone_geom_id < 0:
            # Fallback: distance check
            return self._tcp_stone_distance(data) <= self.cfg.suction_attach_dist

        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 == self.cup_geom_id and g2 == self.stone_geom_id) or (
                g1 == self.stone_geom_id and g2 == self.cup_geom_id
            ):
                return True
        return False

    def _tcp_stone_distance(self, data: mujoco.MjData) -> float:
        tcp_pos = data.xpos[self.tcp_body_id]
        stone_pos = data.xpos[self.stone_body_id]
        return float(np.linalg.norm(tcp_pos - stone_pos))

    def _set_constraint_active(self, data: mujoco.MjData | None, active: bool):
        """Enable or disable the weld constraint, handling API variations."""
        val = 1 if active else 0
        activated = False
        # MuJoCo >= 3.1: runtime activation lives in data.eq_active
        if data is not None:
            try:
                data.eq_active[self.weld_id] = val
                activated = True
            except (AttributeError, IndexError):
                pass
        # Also set model.eq_active0 (initial state, survives resets)
        try:
            self.model.eq_active0[self.weld_id] = val
            activated = True
        except (AttributeError, IndexError):
            pass
        if not activated:
            try:
                self.model.eq_active[self.weld_id] = val
            except AttributeError:
                logger.error("Cannot set eq_active — MuJoCo version unsupported")

    def _update_weld_relpose(self, data: mujoco.MjData):
        """Set the weld constraint's relpose to the current relative pose."""
        tcp_pos = data.xpos[self.tcp_body_id]
        tcp_mat = data.xmat[self.tcp_body_id].reshape(3, 3)
        stone_pos = data.xpos[self.stone_body_id]
        stone_mat = data.xmat[self.stone_body_id].reshape(3, 3)

        # Relative position: stone in TCP frame
        rel_pos = tcp_mat.T @ (stone_pos - tcp_pos)

        # Relative rotation: stone frame in TCP frame
        rel_mat = tcp_mat.T @ stone_mat
        rel_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())

        # Write into eq_data: [anchor(3), pos(3), quat(4), torquescale(1)]
        # (pos comes before quat in the compiled model!)
        eq = self.model.eq_data
        if eq.ndim == 2:
            eq[self.weld_id, 0:3] = 0.0  # anchor at body1 origin
            eq[self.weld_id, 3:6] = rel_pos  # relative position
            eq[self.weld_id, 6:10] = rel_quat  # relative orientation
            eq[self.weld_id, 10] = 1.0  # torquescale
        else:
            off = self.weld_id * self.NEQDATA
            eq[off : off + 3] = 0.0
            eq[off + 3 : off + 6] = rel_pos
            eq[off + 6 : off + 10] = rel_quat
            eq[off + 10] = 1.0
