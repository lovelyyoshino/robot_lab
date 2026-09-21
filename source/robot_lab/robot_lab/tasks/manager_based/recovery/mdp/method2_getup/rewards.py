# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.0
# @last_modified 2026-08-25
# @changelog
#   - v0.1.0 (2026-08-25): 移植 mujoco_playground getup 风格 9 项普通权重奖励，作 ED/CW 消融对照。

"""恢复任务的对照基线奖励（method2 = mujoco_playground getup 风格）。

参考：`google-deepmind/mujoco_playground` 的 `locomotion/go1/getup.py`，
经 `Kitjesen/recovery` 的 `mdp/method2_getup/rewards.py` 适配。

**与 method1 的核心差异（消融自变量）**：**无任何时间维度整形**。
method1 用 `ED(t)=(t/T)³` 让任务奖励随 episode 推进而增强、用 `CW(i)=β·decay^i`
让行为惩罚随训练迭代衰减；method2 全部用**固定权重**，探索完全依赖复位随机性。

**gate 机制**（本方案的替代设计）：`posture` 与 `stand_still` 两项带门控——
仅当机器人**已经直立且达到目标高度**时才发放。这是 mujoco_playground 的做法：
用"条件发放"代替"时间整形"来避免策略在错误姿态上刷分。

9 项：orientation / torso_height / posture* / stand_still* / action_rate /
torques / dof_pos_limits / dof_acc / dof_vel   （* = 带 gate）

**平行四边形适配**：所有关节类项只统计 12 个主动关节（排除 8 个被动 follower）；
`torques`/`dof_acc`/`dof_vel` 进一步只算 8 个腿关节——轮子是速度控制的连续关节，
其加速度/速度在起旋时可比腿关节大若干数量级，会淹没梯度。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .._utils import _get_joint_groups
from ..method1_deng.rewards import MIN_BASE_HEIGHT, NOMINAL_BASE_HEIGHT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── gate 辅助 ──

def _is_upright(asset: Articulation, ori_tol: float) -> torch.Tensor:
    """机体是否直立：projected_gravity 与 [0,0,-1] 的距离 < ori_tol。"""
    ideal = torch.tensor([0.0, 0.0, -1.0], device=asset.data.projected_gravity_b.device)
    return torch.norm(asset.data.projected_gravity_b - ideal, dim=1) < ori_tol


def _at_height(asset: Articulation, min_height: float) -> torch.Tensor:
    """机体高度是否达标（同 method1 的 belly-flat 门思路）。"""
    return asset.data.root_pos_w[:, 2] > min_height


def _gate(asset: Articulation, ori_tol: float, min_height: float) -> torch.Tensor:
    """门控 = 直立 ∧ 达到高度。返回 float 掩码。"""
    return (_is_upright(asset, ori_tol) & _at_height(asset, min_height)).float()


# ── 9 项奖励 ──

def getup_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(−2·‖[0,0,−1] − g_body‖²)：直立时 1，平滑衰减。

    与 method1 的 `recovery_base_orientation` 形式接近，但**不乘 ED**。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    err = torch.norm(asset.data.projected_gravity_b - ideal, dim=1)
    return torch.exp(-2.0 * torch.square(err))


def getup_torso_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = NOMINAL_BASE_HEIGHT,
) -> torch.Tensor:
    """exp(min(h, z_des)) − 1：单边奖励，超过目标高度无额外收益。

    单边设计的意义：避免策略用"跳起来"骗取高度奖励（那不是站立）。
    mujoco_playground 原式即为此形。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return torch.exp(torch.clamp(h, max=target_height)) - 1.0


def getup_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ori_tol: float = 0.3,
    min_height: float = MIN_BASE_HEIGHT,
) -> torch.Tensor:
    """gate · exp(−0.5·‖q_leg − q_default‖²)：**仅在已直立且达高时**奖励标准位姿。

    gate 是本方案代替 ED 的机制：不直立就拿不到位姿奖励，
    从而避免"趴着摆出标准关节角"这类作弊。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    err = asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]
    reward = torch.exp(-0.5 * torch.sum(torch.square(err), dim=1))
    return _gate(asset, ori_tol, min_height) * reward


def getup_stand_still(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ori_tol: float = 0.3,
    min_height: float = MIN_BASE_HEIGHT,
) -> torch.Tensor:
    """gate · exp(−0.5·‖a‖²)：站稳后奖励小动作（鼓励静止而非持续挣扎）。

    同样带 gate —— 否则策略会学会"趴着不动"来拿这份奖励。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    reward = torch.exp(-0.5 * torch.sum(torch.square(action), dim=1))
    return _gate(asset, ori_tol, min_height) * reward


def getup_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """‖aₜ−aₜ₋₁‖² + ‖aₜ−2aₜ₋₁+aₜ₋₂‖²：一阶 + 二阶动作差分。

    mujoco_playground 同时惩罚一阶与二阶差分（后者抑制振荡）。
    需要缓存 t-2 的动作，故在 env 上挂 `_getup_prev_prev_action`。
    """
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if not (torch.is_tensor(action) and torch.is_tensor(prev)):
        return torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "_getup_prev_prev_action"):
        env._getup_prev_prev_action = prev.clone()
    prev2 = env._getup_prev_prev_action

    first = torch.sum(torch.square(action - prev), dim=1)
    second = torch.sum(torch.square(action - 2.0 * prev + prev2), dim=1)

    env._getup_prev_prev_action = prev.clone()
    return first + second


def getup_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """√(Στ²) + Σ|τ| （仅腿关节）。

    mujoco_playground 的 Go1 只有腿；本机器人的轮是速度控制的低力矩高速执行器，
    其力矩量纲与腿关节不同，混算会失衡，故只统计 8 个腿关节。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    tau = asset.data.applied_torque[:, leg_ids]
    return torch.sqrt(torch.sum(torch.square(tau), dim=1)) + torch.sum(torch.abs(tau), dim=1)


def getup_dof_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ soft limit 越界量（仅腿关节；轮是连续关节无位置限位）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    q = asset.data.joint_pos[:, leg_ids]
    soft = asset.data.soft_joint_pos_limits[:, leg_ids]
    below = (soft[..., 0] - q).clamp(min=0.0)
    above = (q - soft[..., 1]).clamp(min=0.0)
    return torch.sum(below + above, dim=1)


def getup_dof_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ q̈²（仅腿关节）。

    排除轮子的理由：轮起旋时角加速度可比腿关节大若干数量级，
    会完全主导该项梯度（参考实现原注释同此）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    return torch.sum(torch.square(asset.data.joint_acc[:, leg_ids]), dim=1)


def getup_dof_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_velocity: float = 6.2832,
) -> torch.Tensor:
    """Σ clamp(|q̇_leg| − max_velocity, 0)²：仅惩罚超过阈值的腿关节速度。

    软阈值形式（非直接惩罚 q̇²）：恢复过程需要腿快速摆动，
    只在超出 2π rad/s 时才罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    excess = (asset.data.joint_vel[:, leg_ids].abs() - max_velocity).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)
