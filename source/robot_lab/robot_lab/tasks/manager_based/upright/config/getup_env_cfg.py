# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 四轮与双轮起点都按碰撞几何复位，避免穿透冲量。
#   - v0.1.0 (2026-09-16): 新增四轮到双轮倒立的第二阶段训练配置。

"""四轮姿态到双轮倒立的第二阶段训练配置。

该任务继承纯平衡环境并保持 42 维观测和 12 维动作合同不变，因此可以从
``RobotLab-Isaac-Upright-Parallelogram-Rover-v0`` 的平衡 checkpoint 继续训练。
四轮起点不是 recovery：目标是沿已知俯仰方向主动抬升并进入可保持区。
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_core
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.upright import mdp as upright_mdp
from robot_lab.tasks.manager_based.upright.config.upright_env_cfg import (
    LEG_JOINT_NAMES,
    ParallelogramRoverUprightEnvCfg,
    UprightEventsCfg,
    UprightRewardsCfg,
    UprightTerminationsCfg,
)


@configclass
class GetupEventsCfg(UprightEventsCfg):
    """四轮起点占 35%，其余样本维持平衡能力，降低灾难性遗忘。"""

    reset_pose = EventTerm(
        func=upright_mdp.reset_upright_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "quad_prob": 0.35,
            "collision_clearance": 0.002,
            "pitch_range": (1.45, 1.70),
            "leg_joint_noise": 0.1,
            "base_height": upright_mdp.UPRIGHT_BASE_HEIGHT,
        },
    )


@configclass
class GetupRewardsCfg(UprightRewardsCfg):
    """在原平衡奖励上增加四轮起立的探索梯度和完成信号。"""

    rise_progress = RewTerm(
        func=upright_mdp.rise_progress,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    getup_success = RewTerm(
        func=upright_mdp.getup_success,
        weight=4.0,
        params={"command_name": "pitch_command", "asset_cfg": SceneEntityCfg("robot")},
    )
    upright_height = RewTerm(
        func=upright_mdp.upright_height,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": upright_mdp.UPRIGHT_BASE_HEIGHT,
            "min_pitch": upright_mdp.HOLDABLE_PITCH_MIN,
        },
    )

    # 起立需要一次强动作，但仍保留较弱的平滑与加速度约束。
    action_rate = RewTerm(func=mdp_core.action_rate_l2, weight=-5e-3)
    joint_acc = RewTerm(
        func=mdp_core.joint_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )


@configclass
class GetupTerminationsCfg(UprightTerminationsCfg):
    """允许 3 秒动态起立，超时仍未离开四轮区则结束 episode。"""

    fell_over = DoneTerm(
        func=upright_mdp.fell_over_after_grace,
        params={
            "max_gz_forward": 0.85,
            "max_gz_backward": upright_mdp.BACKWARD_GZ_LIMIT,
            "forward_grace_steps": 150,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ParallelogramRoverUprightGetupEnvCfg(ParallelogramRoverUprightEnvCfg):
    """起立阶段使用 6 秒预算，为抬升和进入稳定区各留约 3 秒。"""

    events: GetupEventsCfg = GetupEventsCfg()
    rewards: GetupRewardsCfg = GetupRewardsCfg()
    terminations: GetupTerminationsCfg = GetupTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0
