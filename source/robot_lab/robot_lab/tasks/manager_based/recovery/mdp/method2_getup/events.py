# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.0
# @last_modified 2026-08-25
# @changelog
#   - v0.1.0 (2026-08-25): 移植 mujoco_playground 风格 60/40 drop 复位，作 method1 的对照基线。

"""恢复任务的对照基线事件（method2 = mujoco_playground getup 风格）。

参考：`google-deepmind/mujoco_playground` 的 `locomotion/go1/getup.py`，
经 `Kitjesen/recovery` 的 `mdp/method2_getup/events.py` 适配到 Isaac Lab。

**与 method1 的差异（这是消融的自变量）**：

| | method1 (EDRS) | method2 (getup) |
|---|---|---|
| 复位 | 加权姿态池 + 1.1 m 下落 | **60/40 伯努利分支**：60% 从 0.5 m 随机姿态落下、40% 默认位姿 |
| 零力矩窗口 | 有（2 s，塌落成多样姿态）| **无** —— 策略从 t=0 就带正常增益作用 |
| 课程来源 | ED/CW 时间整形 | **纯靠复位分布** |

即 method2 把"探索"完全交给复位随机性，不做任何时间维度的奖励整形。
若 method2 也能达到相近成功率，说明 ED/CW 对本构型不是必需的——这正是要回答的问题。

**共享部分**：scene / actions / observations / terminations 全部继承 method1，
确保 A/B 对比的唯一变量是 events + rewards。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .._utils import _ensure_step_counter, _get_joint_groups
from ..method1_deng.events import _derive_followers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_getup(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_prob: float = 0.6,
    drop_height: float = 0.5,
    root_vel_range: float = 0.5,
):
    """60/40 伯努利复位（mujoco_playground getup 风格）。

    每个环境独立抽样：
      - 概率 `drop_prob`（默认 0.6）：抬到 `drop_height`（0.5 m）+ SO(3) 均匀随机朝向
        + 关节角在 soft limit 内均匀随机
      - 否则（0.4）：默认位姿（正立），仅加根速度扰动

    所有环境都加 `Uniform(-root_vel_range, +root_vel_range)` 的 6-DoF 根速度扰动。

    **无零力矩窗口** —— 与 method1 的核心差异。mujoco_playground 依赖一段
    纯物理 settle 再启动策略；Isaac Lab 的复位管线没有对应钩子，故直接让策略
    从首步接管（参考实现同样如此处理，其注释说明策略需自行应对首几步的未稳定动力学）。

    **平行四边形适配**：关节随机化只作用于 12 个主动关节；8 个 follower 由
    master 派生（与 method1 一致，否则 mimic 约束初值不自洽会产生冲击力矩）。
    """
    if len(env_ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    drop_mask = torch.rand(n, device=device) < drop_prob

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    # 所有环境都加根速度扰动（mujoco_playground 行为）
    root_state[:, 7:13] = (torch.rand((n, 6), device=device) * 2.0 - 1.0) * root_vel_range

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    drop_idx = torch.where(drop_mask)[0]
    if drop_idx.numel() > 0:
        k = drop_idx.numel()
        root_state[drop_idx, 2] = drop_height
        # SO(3) 均匀随机四元数（Shoemake 1992）
        u1 = torch.rand(k, device=device)
        u2 = torch.rand(k, device=device) * 2.0 * math.pi
        u3 = torch.rand(k, device=device) * 2.0 * math.pi
        root_state[drop_idx, 3:7] = torch.stack(
            [
                torch.sqrt(1.0 - u1) * torch.sin(u2),
                torch.sqrt(1.0 - u1) * torch.cos(u2),
                torch.sqrt(u1) * torch.sin(u3),
                torch.sqrt(u1) * torch.cos(u3),
            ],
            dim=1,
        )
        # 腿关节在 soft limit 内均匀随机。
        #
        # ⚠️ **只能取 active_leg(8)，绝不能用 active_all(12)**：
        # 轮关节是 continuous 关节，其 `soft_joint_pos_limits = [-inf, +inf]`（实测），
        # 于是 `lo + rand*(hi-lo)` = `-inf + rand*inf` = **NaN**。
        # 把 NaN 写进 joint_pos 会让 PhysX 首步就发散，之后所有奖励/观测全变 NaN，
        # 训练在首次 alg.update() 即崩 `normal expects all elements of std >= 0.0`。
        # （method1 的 reset 只扰动 active_leg，故未踩到此坑。）
        # 轮子的初始角度本身无物理意义（连续关节），保持默认即可。
        groups = _get_joint_groups(env, asset)
        leg_ids = groups["active_leg"]
        soft = asset.data.soft_joint_pos_limits[env_ids][drop_idx][:, leg_ids]
        rand = torch.rand(k, len(leg_ids), device=device)
        joint_pos[drop_idx.unsqueeze(1), leg_ids.unsqueeze(0)] = (
            soft[..., 0] + rand * (soft[..., 1] - soft[..., 0])
        )

    # follower 必须从 master 派生（平行四边形特有）
    _derive_followers(asset, joint_pos, joint_vel)

    # 写入前守卫：非有限值写进 PhysX 会导致静默发散（表现为训练首次更新即
    # `normal expects std >= 0`，极难从该报错回溯到复位函数）。在此显式拦截。
    if not torch.isfinite(joint_pos).all() or not torch.isfinite(root_state).all():
        bad_j = [asset.joint_names[i] for i in range(joint_pos.shape[1])
                 if not torch.isfinite(joint_pos[:, i]).all()]
        raise RuntimeError(
            f"reset_getup 生成了非有限状态，拒绝写入 PhysX。\n"
            f"  非有限关节: {bad_j}\n"
            f"  root_state finite: {bool(torch.isfinite(root_state).all())}\n"
            f"  常见原因：对 continuous 关节（轮）按 soft limit [-inf,inf] 采样。"
        )

    asset.write_root_state_to_sim(root_state, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 复位步计数器（method2 不用 ED，但 success_rate 的末秒门仍依赖它）
    _ensure_step_counter(env)
    env._recovery_step_count[env_ids] = -1

    # 清 prev_action，避免 episode 边界的假 action_rate 尖峰
    if hasattr(env, "action_manager"):
        prev = getattr(env.action_manager, "prev_action", None)
        if torch.is_tensor(prev):
            prev[env_ids] = 0.0
    if hasattr(env, "_getup_prev_prev_action"):
        env._getup_prev_prev_action[env_ids] = 0.0
