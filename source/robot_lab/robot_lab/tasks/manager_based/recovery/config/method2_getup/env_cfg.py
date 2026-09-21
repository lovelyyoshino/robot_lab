# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.0
# @last_modified 2026-08-25
# @changelog
#   - v0.1.0 (2026-08-25): 建立 method2 对照基线环境（继承 method1，仅换 events/rewards）。

"""恢复任务对照基线环境（method2 = mujoco_playground getup 风格）。

**消融设计**：继承 `ParallelogramRoverRecoveryEnvCfg`，**只替换 events 与 rewards**。
scene / actions / observations / terminations / 时序常量全部沿用 method1，
确保 A/B 对比的唯一自变量是"有无 ED/CW 时间整形"。

要回答的问题：**ED/CW 对本构型是否必需？**
- 若 method2 成功率接近 method1 → ED/CW 不必需，可用更简单的方案
- 若 method2 明显更差 → ED/CW 是关键机制，值得其复杂度

参考：`Kitjesen/recovery` 的 `config/method2_getup/env_cfg.py`
（其 method2 同样只覆写 events/rewards，注释明确"两方法可在 tensorboard 上直接并排对比"）。
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_core
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.recovery import mdp as recovery_mdp
from robot_lab.tasks.manager_based.recovery.config.method1_deng.env_cfg import (
    BASE_LINK_NAME,
    ParallelogramRoverRecoveryEnvCfg,
)


# ── 事件（与 method1 的核心差异：60/40 drop、无零力矩窗口）──

@configclass
class GetupEventsCfg:
    """启动期 DR 沿用 method1 量级；复位改为 60/40 伯努利分支。

    **无 freefall 事件** —— 这是消融的自变量之一：method2 不做零力矩塌落，
    策略从 t=0 就带正常增益接管，探索完全依赖复位分布的随机性。
    """

    randomize_friction = EventTerm(
        func=mdp_core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 与 method1 同值（0.3-1.0 真实轮胎-地量级），保证对比公平
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # 60% 从 0.5 m 随机姿态落下 / 40% 默认位姿；全部加根速度扰动
    reset_getup = EventTerm(
        func=recovery_mdp.reset_getup,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "drop_prob": 0.6,
            "drop_height": 0.5,
            "root_vel_range": 0.5,
        },
    )

    # prev_* 观测缓存复位（与 method1 同——这是观测正确性要求，非消融变量）
    reset_obs_cache = EventTerm(
        func=recovery_mdp.reset_prev_obs_cache,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ── 奖励（9 项普通权重 + gate，无 ED/CW）──

@configclass
class GetupRewardsCfg:
    """mujoco_playground getup 风格：固定权重 + 条件门控，无时间整形。

    权重取自 mujoco_playground Go1 getup 的量级，并按本机器人 10 N·m 执行器下调
    惩罚项（与 method1 同样的下调逻辑，保证对比只差在整形机制）。

    `posture` / `stand_still` 带 gate（直立 ∧ 达高才发放）——这是 method2
    代替 ED 防作弊的机制。
    """

    # ── 任务奖励（固定权重）──
    orientation = RewTerm(
        func=recovery_mdp.getup_orientation,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torso_height = RewTerm(
        func=recovery_mdp.getup_torso_height,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": recovery_mdp.NOMINAL_BASE_HEIGHT},
    )
    posture = RewTerm(
        func=recovery_mdp.getup_posture,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": recovery_mdp.MIN_BASE_HEIGHT},
    )
    stand_still = RewTerm(
        func=recovery_mdp.getup_stand_still,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": recovery_mdp.MIN_BASE_HEIGHT},
    )

    # ── 惩罚（固定权重，无 CW 衰减）──
    action_rate = RewTerm(func=recovery_mdp.getup_action_rate, weight=-1e-3)
    torques = RewTerm(
        func=recovery_mdp.getup_torques,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_pos_limits = RewTerm(
        func=recovery_mdp.getup_dof_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_acc = RewTerm(
        func=recovery_mdp.getup_dof_acc,
        weight=-2.5e-8,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_vel = RewTerm(
        func=recovery_mdp.getup_dof_vel,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── 日志项（与 method1 同指标，便于并排对比）──
    step_counter = RewTerm(func=recovery_mdp.recovery_step_counter, weight=1e-6)
    success_rate = RewTerm(
        func=recovery_mdp.recovery_success_rate,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ── 终止（与 method1 同：仅 timeout）──

@configclass
class GetupTerminationsCfg:
    """仅 timeout。mujoco_playground 的能量终止默认阈值为 inf（等效关闭），故不设。"""

    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)


# ── 主配置 ──

@configclass
class ParallelogramRoverRecoveryGetupEnvCfg(ParallelogramRoverRecoveryEnvCfg):
    """method2 对照基线：继承 method1 的 scene/actions/observations，换 events/rewards。"""

    events: GetupEventsCfg = GetupEventsCfg()
    rewards: GetupRewardsCfg = GetupRewardsCfg()
    terminations: GetupTerminationsCfg = GetupTerminationsCfg()
