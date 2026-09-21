# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 导出四轮起立阶段奖励和宽限期终止。
#   - v0.1.0 (2026-08-25): 建立直立任务 MDP 包的扁平导出。

"""直立（双轮平衡）任务 MDP 包。

    mdp/
      commands.py       — pitch 指令项（采样区间限于 probe2 实测可保持区）
      observations.py   — asymmetric obs（actor **含** base_lin_vel）
      rewards.py        — 姿态跟踪 / roll 惩罚 / 穿越 / 平衡保持
      events.py         — 直立姿态复位（双峰：四轮起点 + 可保持区）
"""

from .commands import PitchCommand, PitchCommandCfg
from .events import reset_upright_pose
from .observations import (
    joint_pos_legs,
    joint_vel_legs,
    priv_base_height,
    priv_wheel_contact,
    wheel_vel,
)
from .terminations import BACKWARD_GZ_LIMIT, excessive_roll, fell_over, fell_over_after_grace
from .rewards import (
    HOLDABLE_PITCH_MAX,
    HOLDABLE_PITCH_MIN,
    OPTIMAL_PITCH,
    UNHOLDABLE_PITCH_MAX,
    UPRIGHT_BASE_HEIGHT,
    balance_hold,
    getup_success,
    pitch_tracking,
    rise_progress,
    roll_penalty,
    transit_speed,
    upright_height,
    wheel_effort_penalty,
)

__all__ = [
    # 指令
    "PitchCommand",
    "PitchCommandCfg",
    # 常量（probe2 实测）
    "HOLDABLE_PITCH_MIN",
    "HOLDABLE_PITCH_MAX",
    "OPTIMAL_PITCH",
    "UNHOLDABLE_PITCH_MAX",
    "UPRIGHT_BASE_HEIGHT",
    # 事件
    "reset_upright_pose",
    # 终止（不可用内置 bad_orientation，见 terminations.py）
    "BACKWARD_GZ_LIMIT",
    "excessive_roll",
    "fell_over",
    "fell_over_after_grace",
    # 观测
    "joint_pos_legs",
    "joint_vel_legs",
    "wheel_vel",
    "priv_base_height",
    "priv_wheel_contact",
    # 奖励
    "pitch_tracking",
    "rise_progress",
    "getup_success",
    "roll_penalty",
    "upright_height",
    "transit_speed",
    "balance_hold",
    "wheel_effort_penalty",
]
