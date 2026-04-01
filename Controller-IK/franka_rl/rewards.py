"""Reward functions for Franka pick-and-place — ManiSkill3 staircase.

Two components:
  1. Task reward (staircase, 0 to ~20): guides toward pick-transport-place
  2. Imitation reward: matches online IK teacher actions

Combined: r = w_task * r_task + w_imitate * r_imitate
Weights annealed over training (imitation-heavy early, task-heavy late).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from franka_rl.env import FrankaPickPlaceEnv


def compute_dense_reward(env: "FrankaPickPlaceEnv") -> tuple[torch.Tensor, dict]:
    """ManiSkill3-style staircase reward for suction pick-and-place."""
    # Phase 1: Reach — suction tip approaching stone
    tip_to_stone = torch.norm(env.suction_tip_pos_w - env.stone_pos_w, dim=-1)
    reaching_reward = 1.0 - torch.tanh(5.0 * tip_to_stone)
    reward = reaching_reward  # [0, 1]

    # Phase 2: Grasp — stone attached via suction
    grasp_reward = env.is_attached.float()
    reward = reward + grasp_reward  # [0, 2]

    # Phase 2.5: Lift — reward for lifting stone above table
    table_z = env.cfg.table_surface_z
    lift_height = (env.stone_pos_w[:, 2] - table_z).clamp(min=0.0)
    lift_reward = torch.tanh(10.0 * lift_height) * env.is_attached.float()
    reward = reward + lift_reward  # [0, 3]

    # Phase 3: Transport — XY distance reduction while holding
    stone_to_target_xy = torch.norm(env.stone_pos_w[:, :2] - env.target_pos_w[:, :2], dim=-1)
    stone_to_target_3d = torch.norm(env.stone_pos_w - env.target_pos_w, dim=-1)
    transport_reward = (1.0 - torch.tanh(5.0 * stone_to_target_xy)) * env.is_attached.float()
    reward = reward + transport_reward  # [0, 4]

    # Phase 4: Near target — stone within 5cm XY AND was grasped
    near_target = (stone_to_target_xy < 0.05) & env.has_been_attached
    suction_release_reward = (~env.suction_active).float()
    place_reward_3d = 1.0 - torch.tanh(5.0 * stone_to_target_3d)
    precision_bonus = torch.exp(-5000.0 * stone_to_target_3d**2)
    reward[near_target] = (
        5.0
        + place_reward_3d[near_target]
        + 3.0 * precision_bonus[near_target]
        + suction_release_reward[near_target]
    )

    # Phase 5: Success — within 10mm, released, was grasped
    at_target = stone_to_target_3d < 0.01
    released = ~env.suction_active
    success = at_target & released & env.has_been_attached
    accuracy_bonus = torch.exp(-50000.0 * stone_to_target_3d**2)
    reward[success] = 9.0 + 10.0 * accuracy_bonus[success] + suction_release_reward[success]

    dropped = env.has_been_attached & ~env.is_attached & ~near_target

    breakdown = {
        "task/phase_reach": reaching_reward.mean().item(),
        "task/phase_grasp": grasp_reward.mean().item(),
        "task/phase_lift": lift_reward.mean().item(),
        "task/phase_transport": transport_reward.mean().item(),
        "task/phase_near": near_target.float().mean().item(),
        "task/phase_success": success.float().mean().item(),
        "task/phase_dropped": dropped.float().mean().item(),
        "task/stone_to_target_xy": stone_to_target_xy.mean().item(),
        "task/stone_to_target_3d": stone_to_target_3d.mean().item(),
        "task/lift_height": lift_height.mean().item(),
    }

    return reward, breakdown


def compute_imitation_reward(
    env: "FrankaPickPlaceEnv", student_actions: torch.Tensor
) -> torch.Tensor:
    """Reward for matching online teacher actions. Returns per-env reward in [0, 1]."""
    if env._teacher_actions is None:
        return torch.zeros(env.num_envs, device=env.device)

    expert_actions = env._teacher_actions
    # Compare all action dims (7 arm joints + 1 suction = 8)
    per_joint_sq = (student_actions - expert_actions) ** 2
    reward = torch.exp(-5.0 * per_joint_sq).mean(dim=-1)
    return reward


def grasp_urgency_penalty(env: "FrankaPickPlaceEnv") -> torch.Tensor:
    """Per-step penalty when stone hasn't been grasped yet."""
    return (~env.has_been_attached).float()


def termination_penalty(env: "FrankaPickPlaceEnv") -> torch.Tensor:
    """Penalize non-timeout terminations."""
    return env.reset_buf.float() * (~env.time_out_buf).float()
