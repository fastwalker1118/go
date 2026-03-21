"""Configuration for the FR3 IK pick-and-place controller."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PickPlaceConfig:
    # --- Paths ---
    model_path: str = "models/scene.xml"

    # --- Robot ---
    n_joints: int = 7
    home_q: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
    )
    joint_names: list = field(default_factory=lambda: ["j1", "j2", "j3", "j4", "j5", "j6", "j7"])
    tcp_site_name: str = "tcp_site"
    tcp_body_name: str = "tcp_tip"
    stone_body_name: str = "stone"

    # --- IK solver ---
    ik_max_iter: int = 100
    ik_damping: float = 0.01
    ik_pos_tol: float = 0.001  # 1 mm
    ik_ori_tol: float = 0.02  # ~1.1 deg
    ik_step_size: float = 0.5
    ik_max_dq: float = 0.2  # max joint change per iteration (rad)
    ik_null_space_gain: float = 0.5  # null-space preference toward home
    ik_use_orientation: bool = True  # use 6-DOF IK (pos + ori)
    ik_ori_weight: float = 0.3  # relative weight of orientation vs position
    ik_recompute_steps: int = 5  # recompute IK every N sim steps

    # --- Suction ---
    suction_weld_name: str = "suction_weld"
    suction_attach_dist: float = 0.02  # max distance to engage suction (m)
    suction_settle_time: float = 0.3  # seconds to wait after grasp/release

    # --- Task geometry ---
    stone_init_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.2, -0.25, 0.0094]  # stone bowl area (outside the board)
        )
    )
    target_pos: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.15, 0.0094]))
    table_height: float = 0.005  # top surface of table geom
    stone_half_height: float = 0.0044
    approach_height: float = 0.12  # TCP height above table for approach
    grasp_offset: float = -0.002  # TCP slightly into stone top to ensure contact
    lift_height: float = 0.12  # how high to lift stone above table

    # --- State machine ---
    pos_threshold: float = 0.005  # position error threshold for transition (m)
    home_threshold: float = 0.05  # joint error threshold for home (rad)

    # --- Simulation ---
    sim_dt: float = 0.002
    max_sim_time: float = 30.0  # max sim seconds before timeout
    diag_interval: int = 500  # print diagnostics every N steps
    realtime_factor: float = 1.0  # 1.0 = real-time, 0 = as fast as possible

    @property
    def stone_grasp_height(self) -> float:
        """TCP z when grasping stone (above table surface)."""
        return self.table_height + self.stone_half_height * 2 + self.grasp_offset

    @property
    def approach_z(self) -> float:
        """TCP z for approach waypoints."""
        return self.table_height + self.approach_height

    @property
    def place_height(self) -> float:
        """TCP z when placing stone at target."""
        return self.table_height + self.stone_half_height * 2 + self.grasp_offset
