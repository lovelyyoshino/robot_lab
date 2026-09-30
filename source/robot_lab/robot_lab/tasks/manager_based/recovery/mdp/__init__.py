# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.0
# @last_modified 2026-08-25
# @changelog
#   - v0.1.0 (2026-08-25): 建立恢复任务 MDP 包的扁平导出。

"""恢复任务 MDP 包。

目录结构（method 分离，便于 method1/method2 直接对照消融）：

    mdp/
      _utils.py              — 共享基础设施（ED/CW、步计数器、关节分组、dt）
      observations.py        — asymmetric actor/critic 观测（含 priv_*）
      method1_deng/
        events.py            — 加权跌倒姿态复位 + 自由落体零力矩
        rewards.py           — 13 项 ED/CW 整形奖励
      method2_getup/         — （待写）mujoco_playground 风格对照基线

公开名统一在此扁平导出，env cfg 只需
``from robot_lab.tasks.manager_based.recovery import mdp as recovery_mdp``，
无需知道某个函数具体在哪个子包。
"""

from ._utils import FREEFALL_STEPS, RECOVERY_STEPS_PER_ITER

from .observations import (
    joint_pos_legs,
    joint_vel_legs,
    prev_joint_pos_legs,
    prev_joint_vel_legs,
    prev_wheel_vel,
    priv_base_ang_vel,
    priv_base_height,
    priv_base_lin_vel,
    priv_body_contact_force,
    priv_wheel_contact,
    reset_prev_obs_cache,
    wheel_vel,
)

# ── method1 — Deng et al. (arXiv:2506.05516) ──
from .method1_deng.events import (
    DEFAULT_FALL_POSE_WEIGHTS,
    reset_with_fall_pose,
    zero_action_freefall,
)
from .method1_deng.rewards import (
    MIN_BASE_HEIGHT,
    NOMINAL_BASE_HEIGHT,
    UPRIGHT_GZ_THRESHOLD,
    check_recovery_success,
    recovery_action_rate,
    recovery_arm_wheel_coord,
    recovery_base_height,
    recovery_base_orientation,
    recovery_body_collision,
    recovery_joint_acceleration,
    recovery_joint_deviation,
    recovery_joint_velocity,
    recovery_stand_joint_pos,
    recovery_step_counter,
    recovery_success_rate,
    recovery_support_state,
    recovery_torques,
    recovery_wheel_velocity,
)
from .grounded import (
    recovery_height_progress,
    recovery_launch_roll_rate,
    recovery_launch_vertical_speed,
    recovery_orientation_progress,
)

# ── method2 — mujoco_playground getup 风格（消融对照基线）──
from .method2_getup.events import reset_getup
from .method2_getup.rewards import (
    getup_action_rate,
    getup_dof_acc,
    getup_dof_pos_limits,
    getup_dof_vel,
    getup_orientation,
    getup_posture,
    getup_stand_still,
    getup_torques,
    getup_torso_height,
)

__all__ = [
    # 常量
    "FREEFALL_STEPS",
    "RECOVERY_STEPS_PER_ITER",
    "MIN_BASE_HEIGHT",
    "NOMINAL_BASE_HEIGHT",
    "UPRIGHT_GZ_THRESHOLD",
    "DEFAULT_FALL_POSE_WEIGHTS",
    # 观测 — actor
    "joint_pos_legs",
    "joint_vel_legs",
    "wheel_vel",
    "prev_joint_pos_legs",
    "prev_joint_vel_legs",
    "prev_wheel_vel",
    "reset_prev_obs_cache",
    # 观测 — critic 特权
    "priv_base_height",
    "priv_base_lin_vel",
    "priv_base_ang_vel",
    "priv_wheel_contact",
    "priv_body_contact_force",
    # method1 — 事件
    "reset_with_fall_pose",
    "zero_action_freefall",
    # method1 — 奖励
    "recovery_step_counter",
    "recovery_stand_joint_pos",
    "recovery_base_height",
    "recovery_base_orientation",
    "recovery_support_state",
    "recovery_arm_wheel_coord",
    "recovery_body_collision",
    "recovery_action_rate",
    "recovery_joint_deviation",
    "recovery_joint_velocity",
    "recovery_joint_acceleration",
    "recovery_torques",
    "recovery_wheel_velocity",
    "check_recovery_success",
    "recovery_success_rate",
    "recovery_orientation_progress",
    "recovery_height_progress",
    "recovery_launch_roll_rate",
    "recovery_launch_vertical_speed",
    "recovery_leg_velocity",
    "recovery_phase_wheel_velocity",
    # method2 — 事件
    "reset_getup",
    # method2 — 奖励（9 项固定权重 + gate，无 ED/CW）
    "getup_orientation",
    "getup_torso_height",
    "getup_posture",
    "getup_stand_still",
    "getup_action_rate",
    "getup_torques",
    "getup_dof_pos_limits",
    "getup_dof_acc",
    "getup_dof_vel",
]
