# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.1
# @last_modified 2026-09-17
# @changelog
#   - v0.1.1 (2026-09-17): 更正轮接触观测的传感器索引说明，不改变观测合同。
#   - v0.1.0 (2026-08-25): 直立任务观测；actor **含** base_lin_vel（与 recovery 相反）。

"""直立任务的观测项。

**与 recovery 观测的关键差异：actor 必须含 base_lin_vel。**

recovery 的 actor 刻意不含线速度（参考实现原注释：倒地后 IMU 积分漂移过大）。
但直立是**倒立摆平衡**任务，其时间常数仅 √(L/g)≈0.24 s——没有线速度反馈
无法判断"正在往哪倒"，只靠角速度会滞后半个周期。轮速也不能替代：
打滑或接触切换时轮速与机体速度解耦。

（Codex 审查亦指出这一点：纯轮速在打滑/接触切换时不可靠。）

关节量同样只取 12 个主动关节，排除 8 个被动 follower ——理由与 recovery 一致：
follower 是 master 的镜像，喂给网络是冗余维度。

维度：
    actor:
      pitch_command              1
      projected_gravity          3
      base_lin_vel               3   ← recovery 没有，直立必需
      base_ang_vel               3
      joint_pos_legs             8
      joint_vel_legs             8
      wheel_vel                  4
      last_action               12
                              ────
                                42
    critic: 42 + priv_base_height 1 + priv_wheel_contact 4 = 47
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _joint_groups(env: ManagerBasedRLEnv, asset: Articulation) -> dict[str, torch.Tensor]:
    """解析并缓存主动腿/轮关节索引（排除 follower）。

    独立于 recovery 的同名函数——遵守 PLAN §3.0-A：任务族之间不跨族继承，
    各自声明。这里只需两组（腿/轮），不需要 recovery 的四组。
    """
    cache = "_upright_joint_groups"
    cached = getattr(env, cache, None)
    if cached is not None:
        return cached

    leg_names = (*PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES, *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES)
    leg_ids, _ = asset.find_joints(list(leg_names), preserve_order=True)
    wheel_ids, _ = asset.find_joints(list(PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES), preserve_order=True)
    device = asset.data.joint_pos.device
    groups = {
        "leg": torch.tensor(leg_ids, dtype=torch.long, device=device),
        "wheel": torch.tensor(wheel_ids, dtype=torch.long, device=device),
    }
    setattr(env, cache, groups)
    return groups


# ── actor 观测 ──

def joint_pos_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """8 个主动腿关节的相对位置 (N, 8)。排除轮（连续关节）与 follower（被动）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _joint_groups(env, asset)["leg"]
    return asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]


def joint_vel_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """8 个主动腿关节速度 (N, 8)。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _joint_groups(env, asset)["leg"]
    return asset.data.joint_vel[:, ids]


def wheel_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """4 个轮速 (N, 4)。倒立摆的主要控制量，量纲与腿关节不同故独立成项。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _joint_groups(env, asset)["wheel"]
    return asset.data.joint_vel[:, ids]


# ── critic 特权观测 ──

def priv_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体绝对高度 (N, 1)。actor 无法从编码器恢复，critic 用真值评估直立质量。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def priv_wheel_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """各轮二值接触状态 (N, num_wheels)。

    此项仅输入 critic；body_ids 必须由接触传感器自身的 body_names 解析。
    旧探针混用了 articulation 的体顺序，导致轮接触被误标为臂接触；不存在
    由该证据支持的“只有一个轮报力”结论。严格支撑奖励另检查全部非轮接触，
    因而不能仅凭这四个轮接触值判断是否由底盘共同承重。
    """
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "priv_wheel_contact 需要由 body_names 正则解析出的 sensor_cfg.body_ids；"
            "没有安全的索引回退。"
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids]
    return (magnitude > threshold).float()
