"""Online IK teacher for Franka Panda pick-and-place.

7-DOF DLS IK with phase-based state machine.
State machine: REACH -> LOWER -> GRASP -> LIFT -> TRANSPORT -> PLACE -> RELEASE -> RETRACT -> DONE
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from franka_rl.env import FrankaPickPlaceEnv

from franka_rl.config import TABLE_SURFACE_Z, STONE_HEIGHT

# Phase constants
REACH = 0
LOWER = 1
GRASP = 2
LIFT = 3
TRANSPORT = 4
PLACE = 5
RELEASE = 6
RETRACT = 7
DONE = 8

PHASE_NAMES = [
    "REACH",
    "LOWER",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETRACT",
    "DONE",
]

# Heights — matching Controller-IK geometry
STONE_SURFACE = TABLE_SURFACE_Z + STONE_HEIGHT / 2 + 0.002  # ~0.011
APPROACH_HEIGHT = TABLE_SURFACE_Z + 0.12  # 0.125
GRASP_HEIGHT = STONE_SURFACE  # ~0.011
LIFT_HEIGHT = TABLE_SURFACE_Z + 0.06  # 0.065
PLACE_HEIGHT = STONE_SURFACE  # ~0.011

# Thresholds
POS_THRESHOLD_COARSE = 0.02
POS_THRESHOLD_FINE = 0.02
LOWER_WAIT = 50
GRASP_WAIT = 30
PLACE_WAIT = 50
RELEASE_WAIT = 10


class OnlineTeacher:
    """IK teacher that runs online alongside the RL student.

    At each call to compute_actions():
    1. Reads current state (tip pos, stone pos, etc.)
    2. Determines phase of pick-place
    3. Computes 7-DOF IK action (joint targets + suction)
    4. Advances state machine
    """

    def __init__(self, env: "FrankaPickPlaceEnv", action_repeat: int = 1):
        self.env = env
        self.device = env.device
        self.action_repeat = action_repeat
        N = env.num_envs

        self.arm_joint_ids = list(env.arm_joint_ids)
        self.damping = 0.05
        self.ik_gain = 0.7
        self.dq_limit = 0.2

        # State machine buffers
        self.phase = torch.zeros(N, device=self.device, dtype=torch.long)
        self.wait_counter = torch.zeros(N, device=self.device, dtype=torch.long)
        self.grasp_stone_pos = torch.zeros(N, 3, device=self.device)

        # Suction tip local offset
        axis = torch.tensor(env.cfg.suction_local_axis, device=self.device, dtype=torch.float32)
        axis = axis / axis.norm().clamp(min=1e-6)
        self._tip_local_offset = (axis * env.cfg.suction_tip_offset).unsqueeze(0)

        self.ee_body_idx = env.ee_body_idx

        # Jacobian offsets
        jac = env.robot.root_physx_view.get_jacobians()
        num_jac_bodies = jac.shape[1]
        num_jac_dofs = jac.shape[3]
        num_body_pos = env.robot.data.body_pos_w.shape[1]
        num_joints = env.robot.data.joint_pos.shape[1]
        self.jac_body_offset = num_body_pos - num_jac_bodies
        self.base_dof_offset = num_jac_dofs - num_joints
        self.arm_dof_ids = [i + self.base_dof_offset for i in self.arm_joint_ids]

        print(
            f"[OnlineTeacher] {N} envs, {len(self.arm_joint_ids)}-DOF IK, "
            f"action_repeat={action_repeat}"
        )

    def reset(self, env_ids: torch.Tensor):
        self.phase[env_ids] = REACH
        self.wait_counter[env_ids] = 0

    def _get_tip_pos(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        robot = self.env.robot
        ee_pos = robot.data.body_pos_w[:, self.ee_body_idx]
        ee_quat = robot.data.body_quat_w[:, self.ee_body_idx]
        world_offset = quat_apply(ee_quat, self._tip_local_offset.expand(ee_pos.shape[0], -1))
        return ee_pos + world_offset

    def _compute_ik(self, target_pos_w: torch.Tensor) -> torch.Tensor:
        """7-DOF DLS IK: position + cup-down orientation."""
        from isaaclab.utils.math import quat_apply

        robot = self.env.robot
        N = target_pos_w.shape[0]

        tip_pos_w = self._get_tip_pos()
        pos_error = target_pos_w - tip_pos_w

        # Cup-down orientation: we want the suction axis to point down (-Z world)
        ee_quat = robot.data.body_quat_w[:, self.ee_body_idx]
        # Local +Z of hand should align with world -Z (cup pointing down)
        current_z = quat_apply(
            ee_quat, torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(N, -1)
        )
        desired_z = torch.tensor([[0.0, 0.0, -1.0]], device=self.device).expand(N, -1)
        ori_error = torch.cross(current_z, desired_z, dim=-1)

        # Get Jacobian for arm joints
        jacobian_full = robot.root_physx_view.get_jacobians()
        jac_row = self.ee_body_idx - self.jac_body_offset
        ee_jac = jacobian_full[:, jac_row, :, :]
        J_pos_body = ee_jac[:, :3, :][:, :, self.arm_dof_ids]
        J_rot = ee_jac[:, 3:, :][:, :, self.arm_dof_ids]

        # Tip Jacobian: J_tip = J_pos - skew(r) @ J_rot
        r = quat_apply(ee_quat, self._tip_local_offset.expand(N, -1))
        skew = torch.zeros(N, 3, 3, device=self.device)
        skew[:, 0, 1] = -r[:, 2]
        skew[:, 0, 2] = r[:, 1]
        skew[:, 1, 0] = r[:, 2]
        skew[:, 1, 2] = -r[:, 0]
        skew[:, 2, 0] = -r[:, 1]
        skew[:, 2, 1] = r[:, 0]
        J_tip_pos = J_pos_body - torch.bmm(skew, J_rot)

        # Orientation weighting
        ori_weight = torch.full((N, 1), 0.03, device=self.device)
        transport_mask = self.phase == TRANSPORT
        ori_weight[transport_mask] = 0.01

        dx_pos = pos_error.unsqueeze(-1)
        dx_ori = (ori_weight * ori_error).unsqueeze(-1)
        dx = torch.cat([dx_pos, dx_ori], dim=1)

        J_ori_weighted = ori_weight.unsqueeze(-1) * J_rot
        J = torch.cat([J_tip_pos, J_ori_weighted], dim=1)

        JJT = torch.bmm(J, J.transpose(1, 2))
        damping_eye = self.damping**2 * torch.eye(6, device=self.device).unsqueeze(0)
        JJT_inv = torch.linalg.solve(
            JJT + damping_eye,
            torch.eye(6, device=self.device).unsqueeze(0).expand(N, -1, -1),
        )
        J_pinv = torch.bmm(J.transpose(1, 2), JJT_inv)
        dq_task = torch.bmm(J_pinv, dx).squeeze(-1)

        # Null-space bias toward home (disabled during lower/grasp/transport/place)
        arm_pos = robot.data.joint_pos[:, self.arm_joint_ids]
        home = robot.data.default_joint_pos[:, self.arm_joint_ids]
        n_dof = len(self.arm_joint_ids)
        null_proj = torch.eye(n_dof, device=self.device).unsqueeze(0) - torch.bmm(J_pinv, J)
        null_bias = torch.bmm(null_proj, (home - arm_pos).unsqueeze(-1)).squeeze(-1)
        null_gain = torch.full((N, 1), 0.1, device=self.device)
        low_phases = (
            (self.phase == LOWER)
            | (self.phase == GRASP)
            | (self.phase == TRANSPORT)
            | (self.phase == PLACE)
        )
        null_gain[low_phases] = 0.0
        dq = self.ik_gain * dq_task + null_gain * null_bias

        # Joint-limit saturation
        lo = robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 0]
        hi = robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 1]
        at_lo = (arm_pos - lo) < 0.01
        at_hi = (hi - arm_pos) < 0.01
        dq = dq.clone()
        dq[at_lo & (dq < 0)] = 0.0
        dq[at_hi & (dq > 0)] = 0.0

        dq = dq.clamp(-self.dq_limit, self.dq_limit)
        desired = arm_pos + dq
        return torch.clamp(desired, lo, hi)

    def _compute_target_pos(self) -> torch.Tensor:
        env = self.env
        N = env.num_envs
        target = torch.zeros(N, 3, device=self.device)

        stone_pos = env.stone_pos_w
        goal_pos = env.target_pos_w
        grasp_pos = self.grasp_stone_pos

        for phase_id, (xy_src, height) in [
            (REACH, (stone_pos, APPROACH_HEIGHT)),
            (LOWER, (stone_pos, GRASP_HEIGHT)),
            (GRASP, (grasp_pos, GRASP_HEIGHT)),
            (LIFT, (grasp_pos, LIFT_HEIGHT)),
            (TRANSPORT, (goal_pos, LIFT_HEIGHT)),
            (PLACE, (goal_pos, PLACE_HEIGHT)),
            (RELEASE, (goal_pos, PLACE_HEIGHT)),
            (RETRACT, (goal_pos, APPROACH_HEIGHT)),
            (DONE, (goal_pos, APPROACH_HEIGHT)),
        ]:
            mask = self.phase == phase_id
            if mask.any():
                target[mask, 0] = xy_src[mask, 0]
                target[mask, 1] = xy_src[mask, 1]
                target[mask, 2] = height

        target[:, 2] = torch.clamp(target[:, 2], min=TABLE_SURFACE_Z + 0.003)
        return target

    def compute_actions(self) -> torch.Tensor:
        """Compute teacher actions for all envs from current state."""
        env = self.env
        N = env.num_envs

        target_pos_w = self._compute_target_pos()
        desired_arm_pos = self._compute_ik(target_pos_w)

        actions = torch.zeros(N, env.num_actions, device=self.device)

        if env.cfg.use_delta_actions:
            # Delta action: (desired - current) / delta_scale
            current_pos = env.robot.data.joint_pos[:, self.arm_joint_ids]
            delta = desired_arm_pos - current_pos
            arm_actions = delta / env.cfg.delta_scale
        else:
            # Absolute action: (desired - default) / scale
            default_pos = env.robot.data.default_joint_pos
            arm_scales = env._joint_scales[self.arm_joint_ids]
            arm_actions = (
                desired_arm_pos - default_pos[:, self.arm_joint_ids]
            ) / arm_scales.clamp(min=1e-6)

        actions[:, self.arm_joint_ids] = arm_actions.clamp(
            -env.cfg.clip_actions, env.cfg.clip_actions
        )

        # Suction ON from LOWER through PLACE
        suction_on = (self.phase >= LOWER) & (self.phase <= PLACE)
        actions[:, env.num_arm_joints] = suction_on.float() * 2.0

        self._advance_state(target_pos_w)
        return actions

    def _advance_state(self, target_pos_w: torch.Tensor):
        env = self.env
        tip_pos_w = self._get_tip_pos()
        pos_error = torch.norm(tip_pos_w - target_pos_w, dim=-1)
        coarse = pos_error < POS_THRESHOLD_COARSE
        fine = pos_error < POS_THRESHOLD_FINE

        done = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)

        mask = (self.phase == REACH) & coarse & ~done
        self.phase[mask] = LOWER
        self.wait_counter[mask] = 0
        done |= mask

        lower_mask = (self.phase == LOWER) & ~done
        self.wait_counter[lower_mask] += self.action_repeat
        mask = lower_mask & ((self.wait_counter >= LOWER_WAIT) | coarse)
        self.phase[mask] = GRASP
        self.wait_counter[mask] = 0
        done |= mask
        if mask.any():
            self.grasp_stone_pos[mask] = env.stone_pos_w[mask].clone()

        grasp_mask = (self.phase == GRASP) & ~done
        self.wait_counter[grasp_mask] += self.action_repeat
        mask = grasp_mask & (self.wait_counter >= GRASP_WAIT) & env.is_attached
        self.phase[mask] = LIFT
        done |= mask

        mask = (self.phase == LIFT) & coarse & env.is_attached & ~done
        self.phase[mask] = TRANSPORT
        done |= mask

        mask = (self.phase == TRANSPORT) & fine & env.is_attached & ~done
        self.phase[mask] = PLACE
        self.wait_counter[mask] = 0
        done |= mask

        place_mask = (self.phase == PLACE) & ~done
        self.wait_counter[place_mask] += self.action_repeat
        mask = place_mask & ((self.wait_counter >= PLACE_WAIT) | fine)
        self.phase[mask] = RELEASE
        self.wait_counter[mask] = 0
        done |= mask

        release_mask = (self.phase == RELEASE) & ~done
        self.wait_counter[release_mask] += self.action_repeat
        mask = release_mask & (self.wait_counter >= RELEASE_WAIT)
        self.phase[mask] = RETRACT
        done |= mask

        mask = (self.phase == RETRACT) & coarse & ~done
        self.phase[mask] = DONE

        # Dropped stone during lift/transport -> reset to REACH
        dropped = (
            env.has_been_attached
            & ~env.is_attached
            & (self.phase >= LIFT)
            & (self.phase <= TRANSPORT)
        )
        stone_xy_dist = torch.norm(env.stone_pos_w[:, :2] - env.target_pos_w[:, :2], dim=-1)
        dropped = dropped & (stone_xy_dist > 0.05)
        if dropped.any():
            self.phase[dropped] = REACH
