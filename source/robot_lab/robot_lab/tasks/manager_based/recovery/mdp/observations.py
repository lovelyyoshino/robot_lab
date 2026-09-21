# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.1
# @last_modified 2026-09-16
# @changelog
#   - v0.1.1 (2026-09-16): 历史仅按控制步推进；统一 actor/critic，并按环境延迟复位历史。
#   - v0.1.0 (2026-08-25): asymmetric actor/critic 观测；actor 排除 base_lin_vel 与 follower。

"""恢复任务的观测项（asymmetric actor-critic）。

参考：`Kitjesen/recovery` 的 `thunder_recovery/mdp/observations.py`。

分两组：

- **actor（部署可得）**：本体感受。**不含 base_lin_vel** —— 参考实现原注释：
  倒地后 IMU 积分漂移过于严重，线速度不可用。改为仅用角速度 + 重力投影 + 关节量。
- **critic（仅仿真可得）**：actor 全量（干净版）+ `priv_*` 特权信号。
  critic 用真值加速价值估计，actor 只看部署可得量 —— 这是 sim2real 的标准做法，
  也是现有 velocity 环境已在用的模式（`velocity_env_cfg.py` critic 含 height_scan、policy 不含）。

**平行四边形特有**：所有关节类观测**只取 12 个主动关节**，排除 8 个被动 follower。
理由与奖励项一致：follower 是 master 的镜像，喂给网络等于输入冗余维度，
既浪费容量又可能让网络学到虚假相关。参考实现（Thunder 无 mimic）不需要这一步。

维度（本机器人 12 主动 = 8 腿 + 4 轮）：

    actor:
      projected_gravity          3
      base_ang_vel               3
      joint_pos_legs             8   (arm×4 + qudong×4)
      joint_vel_legs             8
      wheel_vel                  4
      prev_joint_pos_legs        8
      prev_joint_vel_legs        8
      prev_wheel_vel             4
      last_action               12
                              ────
                                58

    critic: 58 (干净) + priv_base_lin_vel 3 + priv_base_ang_vel 3
            + priv_base_height 1 + priv_wheel_contact 4 + priv_body_contact_force N
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ._utils import _get_joint_groups

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── actor：本体感受（部署可得）──

def joint_pos_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """8 个主动腿关节的相对位置（arm×4 + qudong×4），形状 (N, 8)。

    相对默认位姿，与奖励项的偏差定义一致。排除轮关节（连续关节绝对角无意义）
    与 follower（被动镜像）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _get_joint_groups(env, asset)["active_leg"]
    return asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]


def joint_vel_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """8 个主动腿关节速度，形状 (N, 8)。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _get_joint_groups(env, asset)["active_leg"]
    return asset.data.joint_vel[:, ids]


def wheel_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """4 个轮关节速度，形状 (N, 4)。

    与腿关节分开成独立 term（而非合并成 12 维）：轮速量纲与腿角速度差异大，
    分开便于各自设 scale/noise。参考实现同样分开。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _get_joint_groups(env, asset)["wheel"]
    return asset.data.joint_vel[:, ids]


@dataclass
class _ObservationHistory:
    """每个观测项独立保存前后两帧；复位掩码只标记需要重新初始化的环境。"""

    current: torch.Tensor
    previous: torch.Tensor
    step: int
    reset_pending: torch.Tensor | None = None


def _cache_prev(env: ManagerBasedRLEnv, key: str, current: torch.Tensor) -> torch.Tensor:
    """每个控制步只推进一次历史，同一步的所有读取返回相同的 t-1 值。

    actor、critic、记录器及额外的 get_observations 都可能调用同一个 term；
    因此不能用调用次数代替物理时间。首次读取或指定环境复位后的首帧，用当前值
    填充历史，避免跨 episode 的虚假差分。复位标记由 reset_prev_obs_cache 设置。
    """
    if not hasattr(env, "_recovery_obs_history"):
        env._recovery_obs_history = {}
    history = env._recovery_obs_history.get(key)
    step = env.common_step_counter
    if history is None:
        history = _ObservationHistory(current.clone(), current.clone(), step)
        env._recovery_obs_history[key] = history
    elif history.step != step:
        history.previous.copy_(history.current)
        history.current.copy_(current)
        history.step = step

    if history.reset_pending is not None:
        # 等所有 reset 事件完成后才取新状态，兼容同一步先记录终态再自动复位。
        mask = history.reset_pending.reshape(-1, *([1] * (current.ndim - 1)))
        history.previous.copy_(torch.where(mask, current, history.previous))
        history.current.copy_(torch.where(mask, current, history.current))
        history.reset_pending = None

    # 观测管理器可能就地缩放/加噪；返回副本，不能让消费者污染缓存。
    return history.previous.clone()


def prev_joint_pos_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """t-1 时刻的腿关节位置，形状 (N, 8)。

    提供一阶时间信息，使策略能推断接触状态变化（参考实现原注释：
    时序信息让机器人更好评估自身接触状态）。
    """
    return _cache_prev(env, "joint_pos_legs", joint_pos_legs(env, asset_cfg))


def prev_joint_vel_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """t-1 时刻的腿关节速度，形状 (N, 8)。"""
    return _cache_prev(env, "joint_vel_legs", joint_vel_legs(env, asset_cfg))


def prev_wheel_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """t-1 时刻的轮速，形状 (N, 4)。"""
    return _cache_prev(env, "wheel_vel", wheel_vel(env, asset_cfg))


def reset_prev_obs_cache(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """复位事件：仅使指定环境的历史失效，下一次读取时从新 episode 初始化。

    在 cfg 中以 ``mode="reset"`` 绑定；None 表示全部环境。保留 asset_cfg 参数
    以兼容已有事件配置。这里不读取关节状态，避免依赖姿态复位事件的执行顺序。
    """
    if env_ids is not None and len(env_ids) == 0:
        return
    for history in getattr(env, "_recovery_obs_history", {}).values():
        if history.reset_pending is None:
            history.reset_pending = torch.zeros(
                history.current.shape[0], dtype=torch.bool, device=history.current.device
            )
        history.reset_pending[slice(None) if env_ids is None else env_ids] = True


# ── critic：特权观测（仅仿真可得）──

def priv_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体绝对高度 root_pos_w[:, 2]，形状 (N, 1)。

    actor 无法从关节编码器恢复绝对高度；critic 直接用它评估恢复进度——
    这正是恢复任务的主要成功维度（见 `check_recovery_success` 的高度判据）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def priv_base_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体系线速度真值，形状 (N, 3)。

    actor **完全没有**线速度输入（倒地时 IMU 积分漂移过大）；critic 用干净真值。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def priv_base_ang_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体系角速度真值，形状 (N, 3)。actor 侧的同名量会加噪，此处为干净值。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def priv_wheel_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """每轮二值接触状态，形状 (N, 4)。

    支撑状态的最强信号。真机没有干净的每轮接触传感器，critic 可利用仿真真值。
    """
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "priv_wheel_contact 需要由 body_names 正则解析出的 sensor_cfg.body_ids；"
            "没有安全的索引回退（错误索引会静默喂入错误接触体）。"
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
    return (magnitude > threshold).float()


def priv_body_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """非轮部件（base/arm/qudong）的每体接触力幅值，形状 (N, num_bodies)。

    告知 critic 机器人何时在用机体/腿拖地或撞地 —— actor 无法获取该信号。
    """
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "priv_body_contact_force 需要由 body_names 正则解析出的 sensor_cfg.body_ids；"
            "没有安全的索引回退。"
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    return torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
