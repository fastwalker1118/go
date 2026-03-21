"""Pick-and-place state machine for IK controller."""

import logging
from enum import IntEnum, auto

import numpy as np
import mujoco

from .config import PickPlaceConfig
from .suction import SuctionController
from .ik_solver import IKSolver

logger = logging.getLogger(__name__)


class State(IntEnum):
    HOME = auto()
    PRE_GRASP = auto()
    APPROACH = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    LOWER = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


_STATE_NAMES = {s: s.name for s in State}


class PickPlaceStateMachine:
    """Orchestrates the pick-and-place sequence.

    Each state defines a Cartesian target for the TCP and a transition
    condition.  The controller recomputes IK toward the current target and
    advances states when conditions are met.
    """

    def __init__(
        self,
        config: PickPlaceConfig,
        ik_solver: IKSolver,
        suction: SuctionController,
    ):
        self.cfg = config
        self.ik = ik_solver
        self.suction = suction

        self._state = State.HOME
        self._state_start_time: float = 0.0
        self._q_target = config.home_q.copy()

        # Live stone position (updated each step from sim)
        self._stone_pos = config.stone_init_pos.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def state_name(self) -> str:
        return _STATE_NAMES[self._state]

    @property
    def q_target(self) -> np.ndarray:
        """Current joint-space target for the actuators."""
        return self._q_target

    def update(self, data: mujoco.MjData, sim_time: float) -> None:
        """Advance the state machine by one tick.

        Reads the current simulation state, checks transition conditions,
        computes new IK targets as needed.
        """
        # Keep stone position up-to-date (only meaningful before grasping)
        if not self.suction.is_active:
            self._stone_pos = self.suction.stone_position(data)

        prev = self._state
        self._check_transition(data, sim_time)

        if self._state != prev:
            self._state_start_time = sim_time
            logger.info("[t=%.2fs] State: %s -> %s", sim_time, _STATE_NAMES[prev], self.state_name)

        # Recompute IK target for the current state
        self._compute_target(data, sim_time)

    # ------------------------------------------------------------------
    # State transition logic
    # ------------------------------------------------------------------

    def _check_transition(self, data: mujoco.MjData, t: float) -> None:
        dt_in_state = t - self._state_start_time
        tcp_pos, _ = self.ik.get_tcp_pose(data)
        s = self._state

        if s == State.HOME:
            joint_err = np.linalg.norm(data.qpos[: self.cfg.n_joints] - self.cfg.home_q)
            if joint_err < self.cfg.home_threshold:
                self._state = State.PRE_GRASP

        elif s == State.PRE_GRASP:
            target = self._pre_grasp_pos()
            if np.linalg.norm(tcp_pos - target) < self.cfg.pos_threshold:
                self._state = State.APPROACH

        elif s == State.APPROACH:
            target = self._grasp_pos()
            if np.linalg.norm(tcp_pos - target) < self.cfg.pos_threshold:
                self._state = State.GRASP

        elif s == State.GRASP:
            if self.suction.is_active and dt_in_state > self.cfg.suction_settle_time:
                self._state = State.LIFT

        elif s == State.LIFT:
            target = self._lift_pos()
            if np.linalg.norm(tcp_pos - target) < self.cfg.pos_threshold:
                self._state = State.TRANSPORT

        elif s == State.TRANSPORT:
            target = self._transport_pos()
            if np.linalg.norm(tcp_pos - target) < self.cfg.pos_threshold:
                self._state = State.LOWER

        elif s == State.LOWER:
            target = self._place_pos()
            if np.linalg.norm(tcp_pos - target) < self.cfg.pos_threshold:
                self._state = State.RELEASE

        elif s == State.RELEASE:
            if not self.suction.is_active and dt_in_state > self.cfg.suction_settle_time:
                self._state = State.RETREAT

        elif s == State.RETREAT:
            target = self._retreat_pos()
            if np.linalg.norm(tcp_pos - target) < self.cfg.pos_threshold * 2:
                self._state = State.DONE

    # ------------------------------------------------------------------
    # IK target computation
    # ------------------------------------------------------------------

    def _compute_target(self, data: mujoco.MjData, t: float) -> None:
        s = self._state

        if s == State.HOME:
            self._q_target = self.cfg.home_q.copy()
            return

        if s == State.DONE:
            self._q_target = self.cfg.home_q.copy()
            return

        # All other states: compute a Cartesian target and solve IK
        target_pos = self._get_cartesian_target()

        # In GRASP state, try to activate suction each tick
        if s == State.GRASP:
            self.suction.activate(data)

        # In RELEASE state, deactivate suction
        if s == State.RELEASE:
            self.suction.deactivate(data)

        q, converged, err = self.ik.solve(data, target_pos)
        self._q_target = q

        if not converged and s not in (State.GRASP, State.RELEASE):
            logger.debug("IK did not converge for %s (err=%.4f)", self.state_name, err)

    def _get_cartesian_target(self) -> np.ndarray:
        s = self._state
        if s == State.PRE_GRASP:
            return self._pre_grasp_pos()
        if s == State.APPROACH:
            return self._grasp_pos()
        if s == State.GRASP:
            return self._grasp_pos()
        if s == State.LIFT:
            return self._lift_pos()
        if s == State.TRANSPORT:
            return self._transport_pos()
        if s == State.LOWER:
            return self._place_pos()
        if s == State.RELEASE:
            return self._place_pos()
        if s == State.RETREAT:
            return self._retreat_pos()
        return self._pre_grasp_pos()

    # ------------------------------------------------------------------
    # Waypoint helpers
    # ------------------------------------------------------------------

    def _pre_grasp_pos(self) -> np.ndarray:
        """Above the stone at approach height."""
        return np.array([self._stone_pos[0], self._stone_pos[1], self.cfg.approach_z])

    def _grasp_pos(self) -> np.ndarray:
        """Just above the stone surface — close enough for suction."""
        return np.array(
            [
                self._stone_pos[0],
                self._stone_pos[1],
                self.cfg.stone_grasp_height,
            ]
        )

    def _lift_pos(self) -> np.ndarray:
        """Above stone pickup location at lift height."""
        return np.array(
            [
                self._stone_pos[0],
                self._stone_pos[1],
                self.cfg.table_height + self.cfg.lift_height,
            ]
        )

    def _transport_pos(self) -> np.ndarray:
        """Above the target at approach height."""
        return np.array(
            [
                self.cfg.target_pos[0],
                self.cfg.target_pos[1],
                self.cfg.approach_z,
            ]
        )

    def _place_pos(self) -> np.ndarray:
        """At the target placement height."""
        return np.array(
            [
                self.cfg.target_pos[0],
                self.cfg.target_pos[1],
                self.cfg.place_height,
            ]
        )

    def _retreat_pos(self) -> np.ndarray:
        """Above the target after release."""
        return np.array(
            [
                self.cfg.target_pos[0],
                self.cfg.target_pos[1],
                self.cfg.table_height + self.cfg.lift_height,
            ]
        )
