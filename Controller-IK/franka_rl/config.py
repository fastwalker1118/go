"""Configuration for Franka Panda pick-and-place RL training.

Scene: Franka Panda (7 DOF) with virtual suction cup on a tabletop.
Task: Pick up a go stone and place it at a target location.
Adapted from the G1 RL_controller but simplified for a fixed-base 7-DOF arm.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# ── Go stone geometry (biconvex, matching Controller-IK) ──────────────────
STONE_DIAMETER = 0.0219  # m
STONE_HEIGHT = 0.0088  # m (total)
STONE_HALF_H = 0.0044  # m
STONE_MASS = 0.0068  # kg

# ── Table geometry ──────────────────────────────────────────────────────────
# Matching Controller-IK MuJoCo scene: table surface at Z≈0.005, robot base at Z=0
TABLE_SURFACE_Z = 0.005  # thin platform at ground level
TABLE_HALF_THICKNESS = 0.005
TABLE_X = 0.45  # center of table in front of robot
TABLE_SIZE_XY = 0.60  # 60cm square table

# ── Suction ─────────────────────────────────────────────────────────────────
# Offset from panda_hand body frame to suction cup tip.
# Franka hand local +Z points from wrist toward fingertips (downward in home config).
SUCTION_TIP_OFFSET = 0.1034  # m — matches panda hand to fingertip distance
SUCTION_LOCAL_AXIS = (0.0, 0.0, 1.0)  # +Z of panda_hand frame

# ── Go board grid (same as G1 setup) ───────────────────────────────────────
GRID_SPACING = 0.022  # 22mm between lines
GRID_SIZE = GRID_SPACING * 18  # 396mm
GRID_X0 = TABLE_X - GRID_SIZE / 2
GRID_Y0 = -GRID_SIZE / 2

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "RL_controller",
    "assets",
)

# ── Franka Panda with high PD gains and fixed base ────────────────────────
FRANKA_PICK_PLACE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)


@configclass
class FrankaPickPlaceSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PICK_PLACE_CFG

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_SIZE_XY, TABLE_SIZE_XY, TABLE_HALF_THICKNESS * 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.13)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_X, 0.0, TABLE_HALF_THICKNESS),
        ),
    )

    go_stone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoStone",
        spawn=sim_utils.CylinderCfg(
            radius=STONE_DIAMETER / 2,
            height=STONE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=5.0,
                max_linear_velocity=5.0,
                max_angular_velocity=50.0,
                linear_damping=5.0,
                angular_damping=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=STONE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_X, 0.0, TABLE_SURFACE_Z + STONE_HALF_H + 0.001),
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class FrankaPickPlaceEnvCfg:
    device: str = "cuda:0"

    scene: FrankaPickPlaceSceneCfg = FrankaPickPlaceSceneCfg(num_envs=4096, env_spacing=2.0)

    # Simulation
    sim_dt: float = 0.005  # 200 Hz physics
    decimation: int = 4  # 50 Hz control
    max_episode_length_s: float = 12.0  # 12s — enough for full pick-place cycle

    only_positive_rewards: bool = False

    # Actions — delta-from-current (not absolute offset from default)
    use_delta_actions: bool = True  # target = current_q + action * delta_scale
    delta_scale: float = 0.2  # max 0.2 rad/step/joint — tuned for 7-DOF Franka
    action_scale: float = 0.25  # fallback for absolute mode
    clip_actions: float = 1.0  # clip to [-1, 1] for delta mode
    action_repeat: int = 4

    # Observations
    clip_observations: float = 100.0
    add_noise: bool = True
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.15

    # Table
    table_surface_z: float = TABLE_SURFACE_Z

    # Stone spawn — in front of robot, on table
    stone_spawn_x: tuple[float, float] = (0.35, 0.35)
    stone_spawn_y: tuple[float, float] = (-0.15, -0.15)
    stone_spawn_z: float = TABLE_SURFACE_Z + STONE_HALF_H + 0.002

    # Target — fixed position on table (matching Controller-IK)
    single_target: bool = True
    single_target_xy: tuple[float, float] = (0.5, 0.15)
    target_z: float = TABLE_SURFACE_Z + STONE_HALF_H + 0.002

    # Grid-based targets (when single_target=False)
    use_grid_targets: bool = True
    target_distance_curriculum: bool = True
    target_distance_start: float = 0.10
    target_distance_end: float = 0.40
    target_distance_curriculum_steps: int = 120_000
    target_jitter: float = 0.005

    # Suction — adaptive curriculum, caps at 2mm
    suction_threshold_start: float = 0.05  # 5cm start (easy exploration)
    suction_threshold_end: float = 0.002  # 2mm final
    suction_curriculum_steps: int = 1  # adaptive (handled in env)
    suction_tip_offset: float = SUCTION_TIP_OFFSET
    suction_local_axis: tuple = SUCTION_LOCAL_AXIS

    # EE body name (Franka Panda)
    ee_body_name: str = "panda_hand"

    # Arm joint names (7 DOF)
    arm_joint_names: list = None  # set in __post_init__

    # Termination
    stone_fallen_z: float = -1.0

    # Reward
    reward_scale: float = 1.0
    termination_weight: float = -200.0
    grasp_urgency_weight: float = -0.01

    # Online teacher
    use_online_teacher: bool = True

    # Imitation annealing — PPO+imitation from scratch (no BC)
    imitation_anneal_steps: int = 300_000  # ~6K iters — first 50% of training

    # Noise annealing
    noise_anneal: bool = True
    noise_anneal_steps: int = 100_000

    # Early stopping
    early_stop_success_rate: float = 0.95
    early_stop_window: int = 20

    # NO BC pretraining — train from scratch with imitation reward
    bc_num_epochs: int = 0
    bc_steps_per_epoch: int = 0
    bc_lr: float = 1e-3

    # No DAgger blending — pure student stepping
    dagger_anneal_steps: int = 0

    # Standalone mode — NO residual, student outputs full delta actions
    residual_mode: bool = False
    residual_scale: float = 0.1
    residual_penalty_weight: float = -0.1
    action_rate_weight: float = -0.5

    def __post_init__(self):
        self.arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]


@configclass
class FrankaPickPlaceAgentCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 48
    max_iterations: int = 30000
    empirical_normalization: bool = True

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,  # high initial exploration (no BC to protect)
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # small entropy for exploration (training from scratch)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",  # adaptive LR decay based on KL divergence
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    save_interval: int = 200
    experiment_name: str = "franka_pick_place"
    run_name: str = ""
    logger: str = "wandb"
    wandb_project: str = "franka_pick_place"
    resume: bool = False
    load_run: str = "2026-*"
    load_checkpoint: str = "model_.*.pt"
