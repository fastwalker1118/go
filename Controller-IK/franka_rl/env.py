"""Franka Panda pick-and-place environment with virtual suction cup.

Actions: 7 joint position targets + 1 suction command (>0 = on) = 8 dims.
All 7 arm joints are RL-controlled. Finger joints locked at home.

Observations (actor):
    joint_pos - default (7), joint_vel (7), last_action (8),
    ee_pos (3), stone_pos (3), target_pos (3),
    stone_to_tip (3), target_to_stone (3),
    suction_active (1), is_attached (1)
    Total: 39

Observations (critic): actor + stone_lin_vel (3) = 42
"""

from __future__ import annotations

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.assets import RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply
from isaaclab.sim import PhysxCfg, SimulationContext
from rsl_rl.env import VecEnv

from franka_rl.config import (
    FrankaPickPlaceEnvCfg,
    STONE_HALF_H,
    GRID_SPACING,
    GRID_X0,
    GRID_Y0,
)
import franka_rl.rewards as rew_fns


class _NoopCommandGen:
    def set_debug_vis(self, val):
        pass

    def compute(self, dt):
        pass

    def reset(self, env_ids):
        pass


class FrankaPickPlaceEnv(VecEnv):
    """Franka Panda pick-and-place with virtual suction — 7 DOF + suction."""

    def __init__(self, cfg: FrankaPickPlaceEnvCfg, headless: bool = False):
        self.cfg = cfg
        self.headless = headless
        self.device = cfg.device
        self.physics_dt = cfg.sim_dt
        # step_dt = time per step() call = decimation * sim_dt * action_repeat
        # But episode_length_buf increments once per step(), so max_episode_length
        # should be in step() units
        self.step_dt = cfg.decimation * cfg.sim_dt * cfg.action_repeat
        self.num_envs = cfg.scene.num_envs

        _seed = getattr(cfg.scene, "seed", 42)
        if hasattr(cfg.scene, "seed"):
            try:
                delattr(cfg.scene, "seed")
            except (AttributeError, TypeError):
                pass
        self.seed(_seed)

        self.command_generator = _NoopCommandGen()

        # ---- Simulation context ----
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim_dt,
            render_interval=cfg.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # ---- Scene ----
        self.scene = InteractiveScene(cfg.scene)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.go_stone: RigidObject = self.scene["go_stone"]

        # ---- Resolve body / joint indices ----
        ee_ids, _ = self.robot.find_bodies(cfg.ee_body_name)
        self.ee_body_idx = ee_ids[0]

        arm_ids, _ = self.robot.find_joints(cfg.arm_joint_names)
        self.arm_joint_ids = arm_ids

        # Finger joints (locked at home — not part of action space)
        finger_ids, _ = self.robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])
        self.finger_joint_ids = finger_ids

        # ---- Per-joint action scales ----
        num_j = self.robot.data.default_joint_pos.shape[1]
        self._joint_scales = torch.zeros(num_j, device=self.device)
        self._joint_scales[arm_ids] = cfg.action_scale

        # ---- Buffers ----
        self._init_buffers()
        self._init_online_teacher()
        self._init_grid_positions()

        # Print EE info for debugging
        ee_pos = self.robot.data.body_pos_w[0, self.ee_body_idx]
        print(f"[Franka] EE body '{cfg.ee_body_name}' at index {self.ee_body_idx}")
        print(f"[Franka] Home EE pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"[Franka] Arm joint IDs: {arm_ids}")
        print(
            f"[Franka] Num joints: {num_j}, Arm DOF: {len(arm_ids)}, Actions: {self.num_actions}"
        )

        # ---- Initial reset ----
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)

    def _init_online_teacher(self):
        self._online_teacher = None
        self._teacher_actions = None
        if self.cfg.use_online_teacher:
            from franka_rl.teacher import OnlineTeacher

            self._online_teacher = OnlineTeacher(self, action_repeat=self.cfg.action_repeat)

    def _init_grid_positions(self):
        stone_cx = sum(self.cfg.stone_spawn_x) / 2.0
        stone_cy = sum(self.cfg.stone_spawn_y) / 2.0

        positions = []
        for row in range(19):
            for col in range(19):
                x = GRID_X0 + row * GRID_SPACING
                y = GRID_Y0 + col * GRID_SPACING
                positions.append((x, y))

        grid_pos = torch.tensor(positions, device=self.device, dtype=torch.float32)
        dists = torch.norm(
            grid_pos - torch.tensor([[stone_cx, stone_cy]], device=self.device), dim=-1
        )
        sorted_indices = dists.argsort()
        self._sorted_grid_pos = grid_pos[sorted_indices]
        self._sorted_grid_dists = dists[sorted_indices]
        print(
            f"[Grid] {len(positions)} positions, dist range "
            f"[{dists.min():.3f}, {dists.max():.3f}]m"
        )

    # ===================================================================== #
    #  Buffer initialisation
    # ===================================================================== #

    def _init_buffers(self):
        N = self.num_envs
        dev = self.device

        self.num_arm_joints = len(self.arm_joint_ids)
        self.num_actions = self.num_arm_joints + 1  # arm joints + suction cmd
        self.max_episode_length = int(np.ceil(self.cfg.max_episode_length_s / self.step_dt))

        self.actions = torch.zeros(N, self.num_actions, device=dev)
        self.last_actions = torch.zeros(N, self.num_actions, device=dev)

        self.episode_length_buf = torch.zeros(N, device=dev, dtype=torch.long)
        self.reset_buf = torch.zeros(N, device=dev, dtype=torch.bool)
        self.time_out_buf = torch.zeros(N, device=dev, dtype=torch.bool)

        # Task state
        self.target_pos_w = torch.zeros(N, 3, device=dev)
        self.is_attached = torch.zeros(N, device=dev, dtype=torch.bool)
        self.has_been_attached = torch.zeros(N, device=dev, dtype=torch.bool)
        self.suction_active = torch.zeros(N, device=dev, dtype=torch.bool)
        self.is_released = torch.zeros(N, device=dev, dtype=torch.bool)
        self._released_stone_pos = torch.zeros(N, 3, device=dev)

        # Cached quantities
        self.ee_pos_w = torch.zeros(N, 3, device=dev)
        self.stone_pos_w = torch.zeros(N, 3, device=dev)
        self.suction_tip_pos_w = torch.zeros(N, 3, device=dev)

        axis = torch.tensor(self.cfg.suction_local_axis, device=dev, dtype=torch.float32)
        axis = axis / axis.norm().clamp(min=1e-6)
        self._suction_local_offset = (axis * self.cfg.suction_tip_offset).unsqueeze(0)

        # Obs: joint_pos(7) + joint_vel(7) + actions(8) + ee/stone/target/vecs(18) + flags(2) = 39
        J = self.num_arm_joints
        A = self.num_actions
        self.num_obs = J + J + A + 3 + 3 + 3 + 3 + 3 + 1 + 1  # 39
        self.num_privileged_obs = self.num_obs + 3  # + stone_lin_vel

        self.obs_buf = torch.zeros(N, self.num_obs, device=dev)
        self.privileged_obs_buf = torch.zeros(N, self.num_privileged_obs, device=dev)
        self.rew_buf = torch.zeros(N, device=dev)
        self.extras = {}

        self._step_count = 0
        self._current_suction_threshold = self.cfg.suction_threshold_start
        self._tighten_cooldown = 0

        # Episode success tracking
        self._episode_success = torch.zeros(N, device=dev, dtype=torch.bool)
        self._recent_successes = 0
        self._recent_episodes = 0

        # Noise
        if self.cfg.add_noise:
            self._build_noise_vec()

    def _build_noise_vec(self):
        vec = torch.zeros(self.num_obs, device=self.device)
        J = self.num_arm_joints
        idx = 0
        vec[idx : idx + J] = self.cfg.noise_joint_pos
        idx += J
        vec[idx : idx + J] = self.cfg.noise_joint_vel
        idx += J
        self.noise_scale_vec = vec

    # ===================================================================== #
    #  VecEnv interface
    # ===================================================================== #

    def get_observations(self):
        self._compute_observations()
        self.extras["observations"] = {"critic": self.privileged_obs_buf}
        return self.obs_buf, self.extras

    def _inner_step(self, actions: torch.Tensor):
        """Execute one control step: apply actions, simulate, update state."""
        joint_actions = actions[:, : self.num_arm_joints]
        self.suction_active[:] = actions[:, self.num_arm_joints] > 0

        clipped = torch.clip(joint_actions, -self.cfg.clip_actions, self.cfg.clip_actions)

        if self.cfg.use_delta_actions:
            # Delta-from-current: target = current_q + action * delta_scale
            current_q = self.robot.data.joint_pos[:, self.arm_joint_ids]
            targets_arm = current_q + clipped * self.cfg.delta_scale
            # Clamp to joint limits
            lo = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 0]
            hi = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 1]
            targets_arm = torch.clamp(targets_arm, lo, hi)
        else:
            # Absolute: target = default_q + action * scale
            targets_arm = (
                clipped * self._joint_scales[self.arm_joint_ids]
                + self.robot.data.default_joint_pos[:, self.arm_joint_ids]
            )

        # Build full joint targets (arm + locked fingers)
        full_targets = self.robot.data.default_joint_pos.clone()
        full_targets[:, self.arm_joint_ids] = targets_arm

        for _ in range(self.cfg.decimation):
            self.robot.set_joint_position_target(full_targets)
            # Direct position write for arm joints — precise tracking
            ik_vel = torch.zeros_like(targets_arm)
            self.robot.write_joint_state_to_sim(targets_arm, ik_vel, joint_ids=self.arm_joint_ids)

            # Move attached stones with EE
            if self.is_attached.any():
                attached_ids = self.is_attached.nonzero(as_tuple=False).flatten()
                ee_pos = self.robot.data.body_pos_w[attached_ids, self.ee_body_idx]
                ee_quat = self.robot.data.body_quat_w[attached_ids, self.ee_body_idx]
                local_off = self._suction_local_offset.expand(len(attached_ids), -1)
                world_off = quat_apply(ee_quat, local_off)
                stone_pos = ee_pos + world_off
                stone_pos[:, 2] -= STONE_HALF_H
                stone_quat = self.go_stone.data.root_quat_w[attached_ids]
                self.go_stone.write_root_pose_to_sim(
                    torch.cat([stone_pos, stone_quat], dim=-1), attached_ids
                )
                self.go_stone.write_root_velocity_to_sim(
                    torch.zeros(len(attached_ids), 6, device=self.device), attached_ids
                )

            # Pin released stones
            if self.is_released.any():
                pin_ids = self.is_released.nonzero(as_tuple=False).flatten()
                current_z = self.go_stone.data.root_pos_w[pin_ids, 2]
                landed = current_z < self.cfg.table_surface_z + STONE_HALF_H + 0.01
                if landed.any():
                    land_ids = pin_ids[landed]
                    pin_quat = self.go_stone.data.root_quat_w[land_ids]
                    self.go_stone.write_root_pose_to_sim(
                        torch.cat([self._released_stone_pos[land_ids], pin_quat], dim=-1), land_ids
                    )
                    self.go_stone.write_root_velocity_to_sim(
                        torch.zeros(len(land_ids), 6, device=self.device), land_ids
                    )

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self._update_state()
        self._update_suction()

    def step(self, actions: torch.Tensor):
        self._step_count += 1
        self.episode_length_buf += 1  # once per policy decision, not per inner step
        self.last_actions[:] = self.actions
        self.actions[:] = actions
        self._maybe_tighten_suction()

        # Online teacher
        if self._online_teacher is not None:
            self._teacher_actions = self._online_teacher.compute_actions()

        # Residual mode: teacher + RL corrections
        if self.cfg.residual_mode and self._teacher_actions is not None:
            step_actions = self._teacher_actions.clone()
            joint_residual = (
                actions[:, : self.num_arm_joints].clamp(-1.0, 1.0) * self.cfg.residual_scale
            )
            step_actions[:, : self.num_arm_joints] += joint_residual
        # DAgger blend: early in training, blend student+teacher for stepping
        elif self._teacher_actions is not None and not self.cfg.residual_mode:
            dagger_steps = getattr(self.cfg, "dagger_anneal_steps", 0)
            if dagger_steps > 0 and self._step_count < dagger_steps:
                alpha = 1.0 - self._step_count / dagger_steps  # 1→0
                # Clip student actions to teacher-scale range before blending
                student_clipped = actions.clamp(-10.0, 10.0)
                step_actions = alpha * self._teacher_actions + (1.0 - alpha) * student_clipped
                # Suction: always follow teacher (student has ON-bias from BC)
                step_actions[:, self.num_arm_joints] = self._teacher_actions[
                    :, self.num_arm_joints
                ]
            else:
                step_actions = actions
        else:
            step_actions = actions

        # Action repeat with reward accumulation
        total_reward = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.cfg.action_repeat):
            self._inner_step(step_actions)
            self.reset_buf[:], self.time_out_buf[:] = self._check_termination()
            self.rew_buf[:] = self._compute_rewards()
            total_reward += self.rew_buf

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)
                break

        if not self.headless:
            self.sim.render()

        self.rew_buf[:] = total_reward
        self._compute_observations()
        self.extras["observations"] = {"critic": self.privileged_obs_buf}
        self.extras["time_outs"] = self.time_out_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        return self.get_observations()

    # ===================================================================== #
    #  Reset
    # ===================================================================== #

    def reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        self._recent_successes += self._episode_success[env_ids].sum().item()
        self._recent_episodes += len(env_ids)
        self._episode_success[env_ids] = False

        self.scene.reset(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()

        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        # Stone position
        stone_pos = torch.zeros(n, 3, device=self.device)
        stone_pos[:, 0] = origins[:, 0] + torch.empty(n, device=self.device).uniform_(
            *self.cfg.stone_spawn_x
        )
        stone_pos[:, 1] = origins[:, 1] + torch.empty(n, device=self.device).uniform_(
            *self.cfg.stone_spawn_y
        )
        stone_pos[:, 2] = self.cfg.stone_spawn_z
        stone_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, -1)
        self.go_stone.write_root_pose_to_sim(torch.cat([stone_pos, stone_quat], dim=-1), env_ids)
        self.go_stone.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids)

        # Target position
        if self.cfg.single_target:
            tx, ty = self.cfg.single_target_xy
            self.target_pos_w[env_ids, 0] = origins[:, 0] + tx
            self.target_pos_w[env_ids, 1] = origins[:, 1] + ty
        elif self.cfg.use_grid_targets:
            if self.cfg.target_distance_curriculum:
                progress = min(
                    self._step_count / max(self.cfg.target_distance_curriculum_steps, 1), 1.0
                )
                max_dist = self.cfg.target_distance_start + progress * (
                    self.cfg.target_distance_end - self.cfg.target_distance_start
                )
                num_valid = max(int((self._sorted_grid_dists <= max_dist).sum().item()), 1)
                indices = torch.randint(0, num_valid, (n,), device=self.device)
                local_xy = self._sorted_grid_pos[indices]
            else:
                indices = torch.randint(0, len(self._sorted_grid_pos), (n,), device=self.device)
                local_xy = self._sorted_grid_pos[indices]

            self.target_pos_w[env_ids, 0] = origins[:, 0] + local_xy[:, 0]
            self.target_pos_w[env_ids, 1] = origins[:, 1] + local_xy[:, 1]

            jitter = self.cfg.target_jitter
            if jitter > 0:
                self.target_pos_w[env_ids, 0] += torch.empty(n, device=self.device).uniform_(
                    -jitter, jitter
                )
                self.target_pos_w[env_ids, 1] += torch.empty(n, device=self.device).uniform_(
                    -jitter, jitter
                )
        self.target_pos_w[env_ids, 2] = self.cfg.target_z

        # Reset teacher
        if self._online_teacher is not None:
            self._online_teacher.reset(env_ids)

        # Reset state
        self.is_attached[env_ids] = False
        self.has_been_attached[env_ids] = False
        self.suction_active[env_ids] = False
        self.is_released[env_ids] = False
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

        self.stone_pos_w[env_ids] = stone_pos[:, :3]
        self.ee_pos_w[env_ids] = self.robot.data.body_pos_w[env_ids, self.ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[env_ids, self.ee_body_idx]
        local_offset = self._suction_local_offset.expand(len(env_ids), -1)
        world_offset = quat_apply(ee_quat, local_offset)
        self.suction_tip_pos_w[env_ids] = self.ee_pos_w[env_ids] + world_offset

    # ===================================================================== #
    #  State helpers
    # ===================================================================== #

    def _update_state(self):
        self.ee_pos_w[:] = self.robot.data.body_pos_w[:, self.ee_body_idx]
        self.stone_pos_w[:] = self.go_stone.data.root_pos_w[:, :3]
        ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx]
        local_offset = self._suction_local_offset.expand(self.num_envs, -1)
        world_offset = quat_apply(ee_quat, local_offset)
        self.suction_tip_pos_w[:] = self.ee_pos_w + world_offset

    @property
    def suction_threshold(self) -> float:
        return self._current_suction_threshold

    def _maybe_tighten_suction(self):
        """Adaptive curriculum: tighten suction when fine accuracy > 90%."""
        if not hasattr(self, "_current_suction_threshold"):
            self._current_suction_threshold = self.cfg.suction_threshold_start
            self._tighten_cooldown = 0
        if self._tighten_cooldown > 0:
            self._tighten_cooldown -= 1
            return
        self.get_success_rate(reset=False)
        # Check phase_success from recent rewards
        stone_to_target = torch.norm(self.stone_pos_w - self.target_pos_w, dim=-1)
        fine_rate = (
            ((stone_to_target < 0.01) & (~self.suction_active) & self.has_been_attached)
            .float()
            .mean()
            .item()
        )
        if fine_rate > 0.15 and self._current_suction_threshold > self.cfg.suction_threshold_end:
            old = self._current_suction_threshold
            self._current_suction_threshold = max(old * 0.7, self.cfg.suction_threshold_end)
            self._tighten_cooldown = 200  # wait 200 steps before next tighten
            print(
                f"[Suction] Tightened: {old * 1000:.1f}mm -> "
                f"{self._current_suction_threshold * 1000:.1f}mm"
            )

    def _update_suction(self):
        tip_to_stone = torch.norm(self.suction_tip_pos_w - self.stone_pos_w, dim=-1)
        close_enough = tip_to_stone < self.suction_threshold

        was_attached = self.is_attached.clone()
        self.is_attached = (
            self.is_attached | (self.suction_active & close_enough)
        ) & self.suction_active
        self.has_been_attached |= self.is_attached
        self.is_released &= ~self.is_attached

        # Detect release
        just_released = was_attached & ~self.is_attached
        if just_released.any():
            release_ids = just_released.nonzero(as_tuple=False).flatten()
            self.go_stone.write_root_velocity_to_sim(
                torch.zeros(len(release_ids), 6, device=self.device), release_ids
            )
            self.is_released[release_ids] = True
            self._released_stone_pos[release_ids] = self.stone_pos_w[release_ids].clone()
            self._released_stone_pos[release_ids, 2] = self.cfg.table_surface_z + STONE_HALF_H

        # Pin released stones
        if self.is_released.any():
            pin_ids = self.is_released.nonzero(as_tuple=False).flatten()
            current_z = self.go_stone.data.root_pos_w[pin_ids, 2]
            landed = current_z < self.cfg.table_surface_z + STONE_HALF_H + 0.01
            if landed.any():
                land_ids = pin_ids[landed]
                pin_quat = self.go_stone.data.root_quat_w[land_ids]
                self.go_stone.write_root_pose_to_sim(
                    torch.cat([self._released_stone_pos[land_ids], pin_quat], dim=-1), land_ids
                )
                self.go_stone.write_root_velocity_to_sim(
                    torch.zeros(len(land_ids), 6, device=self.device), land_ids
                )
                self.stone_pos_w[land_ids] = self._released_stone_pos[land_ids]

        if self.is_attached.any():
            mask = self.is_attached
            ee_quat = self.robot.data.body_quat_w[mask, self.ee_body_idx]
            local_off = self._suction_local_offset.expand(mask.sum(), -1)
            world_off = quat_apply(ee_quat, local_off)
            self.stone_pos_w[mask] = self.ee_pos_w[mask] + world_off
            self.stone_pos_w[mask, 2] -= STONE_HALF_H

    # ===================================================================== #
    #  Observations
    # ===================================================================== #

    def _compute_observations(self):
        robot = self.robot

        # All positions relative to robot base (fixed at origin, so world == base frame)
        actor_obs = torch.cat(
            [
                robot.data.joint_pos[:, self.arm_joint_ids]
                - robot.data.default_joint_pos[:, self.arm_joint_ids],
                robot.data.joint_vel[:, self.arm_joint_ids],
                self.actions,
                self.ee_pos_w,
                self.stone_pos_w,
                self.target_pos_w,
                self.stone_pos_w - self.suction_tip_pos_w,
                self.target_pos_w - self.stone_pos_w,
                self.suction_active.float().unsqueeze(-1),
                self.is_attached.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        if self.cfg.add_noise:
            if self.cfg.noise_anneal:
                noise_progress = min(self._step_count / max(self.cfg.noise_anneal_steps, 1), 1.0)
            else:
                noise_progress = 1.0
            noise = torch.randn_like(actor_obs) * self.noise_scale_vec * noise_progress
            actor_obs = actor_obs + noise

        actor_obs = torch.clip(actor_obs, -self.cfg.clip_observations, self.cfg.clip_observations)

        stone_lin_vel = self.go_stone.data.root_lin_vel_w[:, :3]
        critic_obs = torch.cat([actor_obs, stone_lin_vel], dim=-1)
        critic_obs = torch.clip(
            critic_obs, -self.cfg.clip_observations, self.cfg.clip_observations
        )

        self.obs_buf[:] = actor_obs
        self.privileged_obs_buf[:] = critic_obs

    # ===================================================================== #
    #  Rewards
    # ===================================================================== #

    def _compute_rewards(self) -> torch.Tensor:
        task_reward, task_breakdown = rew_fns.compute_dense_reward(self)
        imitate_reward = rew_fns.compute_imitation_reward(self, self.actions)
        grasp_urg = rew_fns.grasp_urgency_penalty(self)

        # Curriculum annealing
        if self.cfg.residual_mode:
            w_task = 1.0
            w_imitate = 0.0
        else:
            anneal_steps = self.cfg.imitation_anneal_steps
            progress = min(self._step_count / anneal_steps, 1.0)
            w_task = 0.5 + 0.5 * progress  # 0.5 → 1.0 (always strong task signal)
            w_imitate = 0.5 - 0.5 * progress  # 0.5 → 0.0

        task_reward_norm = task_reward / 10.0

        termination = rew_fns.termination_penalty(self)

        total = (
            w_task * 10.0 * task_reward_norm
            + w_imitate * 10.0 * imitate_reward
            + self.cfg.grasp_urgency_weight * grasp_urg
            + self.cfg.termination_weight * termination
        )

        if self.cfg.residual_mode:
            clamped_actions = self.actions[:, : self.num_arm_joints].clamp(-1.0, 1.0)
            residual_norm = (clamped_actions**2).mean(dim=-1)
            total = total + self.cfg.residual_penalty_weight * residual_norm
            action_rate = (
                (
                    self.actions[:, : self.num_arm_joints].clamp(-1.0, 1.0)
                    - self.last_actions[:, : self.num_arm_joints].clamp(-1.0, 1.0)
                )
                ** 2
            ).mean(dim=-1)
            total = total + self.cfg.action_rate_weight * action_rate

        # Track success
        stone_to_target = torch.norm(self.stone_pos_w - self.target_pos_w, dim=-1)
        success = (stone_to_target < 0.05) & (~self.suction_active) & self.has_been_attached
        self._episode_success |= success

        ee_to_stone = torch.norm(self.suction_tip_pos_w - self.stone_pos_w, dim=-1)

        log = {
            "rew/task": task_reward.mean().item(),
            "rew/imitate": imitate_reward.mean().item(),
            "rew/w_task": w_task,
            "rew/w_imitate": w_imitate,
            "rew/total": total.mean().item(),
            "task/ee_to_stone": ee_to_stone.mean().item(),
            "task/attach_rate": self.is_attached.float().mean().item(),
            "task/success_rate": self._recent_successes / max(self._recent_episodes, 1),
            "task/suction_rate": self.suction_active.float().mean().item(),
        }
        if self._online_teacher is not None:
            from franka_rl.teacher import PHASE_NAMES

            phase = self._online_teacher.phase
            for i, name in enumerate(PHASE_NAMES):
                log[f"teacher/{name.lower()}"] = (phase == i).float().mean().item()
        log.update(task_breakdown)
        self.extras["log"] = log
        return total

    # ===================================================================== #
    #  Termination
    # ===================================================================== #

    def _check_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        stone_fallen = self.stone_pos_w[:, 2] < self.cfg.stone_fallen_z
        time_out = self.episode_length_buf >= self.max_episode_length
        reset = stone_fallen | time_out
        return reset, time_out

    # ===================================================================== #
    #  Seed
    # ===================================================================== #

    @staticmethod
    def seed(seed: int = -1) -> int:
        import isaacsim.core.utils.torch as torch_utils

        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def get_success_rate(self, reset: bool = True) -> float:
        rate = self._recent_successes / max(self._recent_episodes, 1)
        if reset:
            self._recent_successes = 0
            self._recent_episodes = 0
        return rate
