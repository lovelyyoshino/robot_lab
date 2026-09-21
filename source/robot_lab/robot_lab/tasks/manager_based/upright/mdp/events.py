# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.4.0
# @last_modified 2026-09-17
# @changelog
#   - v0.4.0 (2026-09-17): 首次reset用已初始化执行器/PhysX限速，避免读取尚未填充的零软限速缓存。
#   - v0.3.0 (2026-09-17): 可选四轮前后初速与 URDF 同步轮速，仅修改复位子集，静止默认不变。
#   - v0.2.0 (2026-09-16): 用当前 URDF 碰撞几何计算无穿透复位高度，保留显式旧高度探针。
#   - v0.1.0 (2026-08-25): 直立复位事件；双峰分布（四轮起点 + 可保持区）。

"""直立任务的复位事件。

**双峰复位分布**（PLAN §4 P1）：
  - 一部分环境从**四轮姿态**（pitch≈0）起步 → 学"起立"
  - 其余从**可保持区**（75°-105°）起步 → 学"平衡"

为何要双峰而非单一分布：
  - 只从四轮起步：策略必须先偶然学会起立才能拿到任何平衡奖励，
    而起立要穿越 20°-60° 不可保持区，探索概率极低 → 大概率收敛到"不起立"
  - 只从可保持区起步：学会了平衡但永远学不会从四轮起立
  → 双峰同时提供两种梯度；且随课程推进可调整比例

**平行四边形特有**：8 个 follower 必须从 4 个 arm master 符号派生
（`zhuanxiang = -arm`, `lower_arm = +arm`），否则 mimic 约束初值不自洽，
PhysX 首个物理步会产生冲击力矩。

**写入前守卫**：检测非有限值。recovery 曾因对 continuous 关节（轮，
soft limit = [-inf,inf]）采样而产生 NaN，症状是训练首次更新即
`normal expects std >= 0`，离根因隔三层。这里预先拦截。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS,
    PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
)

# 复位高度默认值取自 rewards 的实测常数，避免两处各自硬编码后静默漂移
from .rewards import UPRIGHT_BASE_HEIGHT
from .geometry import RoverCollisionGeometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mimic_relations(asset: Articulation) -> tuple[tuple[int, int, float, float], ...]:
    """解析并缓存 (follower_id, master_id, multiplier, offset)。

    本函数是现有 locomotion `mdp.py::_mimic_relation_ids` 的**族内副本**
    （PLAN §3.0-A：现有文件只读，需变体则复制）。
    """
    cache = "_upright_mimic_relations"
    cached = getattr(asset, cache, None)
    if cached is not None:
        return cached
    relations = []
    for f_name, m_name, mult, off in PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS:
        ids, _ = asset.find_joints([f_name, m_name], preserve_order=True)
        relations.append((int(ids[0]), int(ids[1]), float(mult), float(off)))
    result = tuple(relations)
    setattr(asset, cache, result)
    return result


def _derive_followers(asset: Articulation, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
    """就地把 8 个 follower 从 master 派生。"""
    for f_id, m_id, mult, off in _mimic_relations(asset):
        joint_pos[:, f_id] = mult * joint_pos[:, m_id] + off
        joint_vel[:, f_id] = mult * joint_vel[:, m_id]


def _leg_ids(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    """8 个主动腿关节索引（缓存）。"""
    cache = "_upright_reset_leg_ids"
    cached = getattr(env, cache, None)
    if cached is not None:
        return cached
    names = (*PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES, *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES)
    ids, _ = asset.find_joints(list(names), preserve_order=True)
    t = torch.tensor(ids, dtype=torch.long, device=asset.data.joint_pos.device)
    setattr(env, cache, t)
    return t


def reset_upright_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    quad_prob: float = 0.3,
    pitch_range: tuple[float, float] = (1.45, 1.70),
    leg_joint_noise: float = 0.1,
    base_height: float = UPRIGHT_BASE_HEIGHT,
    collision_clearance: float | None = None,
    quad_forward_velocity_range: tuple[float, float] = (0.0, 0.0),
):
    """双峰复位：`quad_prob` 概率从四轮姿态、其余从可保持区姿态。

    Args:
        quad_prob: 从四轮姿态（pitch≈0）起步的比例。默认 0.3——
            早期可设 0（先学平衡），课程后期提高（加入起立训练）。
        pitch_range: 直立起点的 pitch 采样区间（rad）。应落在
            probe2 实测可保持区 [1.31, 1.83] 内。
        leg_joint_noise: 腿关节扰动 Uniform(±noise) rad。**只施加于 8 个
            主动腿关节**——轮是 continuous 关节，其 soft limit 为 ±inf，
            按限位采样会产生 NaN（recovery 踩过此坑）。
        base_height: 未启用几何复位时的固定高度，仅用于对照。
            旧0.35m虽然底盘未穿地，但后轮已穿地，不再作为训练默认值。
        collision_clearance: 非 None 时按实际 URDF 和随机关节角计算所有碰撞体的最低点，
            覆盖 base_height。训练配置启用此项；None 仅用于复现旧固定高度实验。
        quad_forward_velocity_range: 仅四轮起点的合成纵向初速范围（m/s，正向为 +x）。
            非零样本转向置直、轮速按 URDF 同步；不绑定下一回合运动命令，也不代表策略接管。
            默认 (0,0) 保留旧分布和 RNG；不向直立起点注入速度。
    """
    if (len(quad_forward_velocity_range) != 2
            or not all(math.isfinite(v) for v in quad_forward_velocity_range)
            or not -0.5 <= quad_forward_velocity_range[0] <= quad_forward_velocity_range[1] <= 0.5):
        raise ValueError("四轮直行初速范围须有限、有序且在 [-0.5,0.5] m/s 内")
    moving_reset = any(v != 0.0 for v in quad_forward_velocity_range)
    if moving_reset and collision_clearance is None:
        raise ValueError("带速起立必须启用碰撞几何复位，不能从旧悬空高度开始")
    if len(env_ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 7:] = 0.0

    # 双峰：quad_mask 为 True 的从四轮起步（pitch=0，保持默认高度）
    quad_mask = torch.rand(n, device=device) < quad_prob
    pitch = torch.empty(n, device=device).uniform_(*pitch_range)
    pitch = torch.where(quad_mask, torch.zeros_like(pitch), pitch)

    # 绕 y 轴 pitch 的四元数 [w,x,y,z]
    #
    # ⚠️ **y 分量取 -sin(θ/2)，不是 +sin(θ/2)**（probe14 实测钉死）：
    # 只有该符号才与 `rewards._pitch_from_gravity` 的 `atan2(-g_x, -g_z)` 自洽。
    # 实测对照（6 个角度全程一致）：
    #     θ=+90° 时  y=+sin → 反解 -90°（错）；y=-sin → 反解 +90°（对）
    # 且 y=-sin 在 θ=90° 给出 g=[-1,0,0]，与 probe2 实测的真实直立姿态一致。
    # 用错符号的后果：pitch_tracking 的误差恒为 -180°，奖励恒 0，
    # 且机器人被摆成前倾而非后仰（非轮部件大面积触地，undesired_contacts 爆表）。
    half = pitch * 0.5
    root_state[:, 3] = torch.cos(half)
    root_state[:, 4] = 0.0
    root_state[:, 5] = -torch.sin(half)
    root_state[:, 6] = 0.0

    # 直立起点抬高机体；四轮起点用默认高度
    default_z = asset.data.default_root_state[env_ids][:, 2]
    root_state[:, 2] = torch.where(
        quad_mask, default_z, torch.full_like(default_z, base_height)
    ) + env.scene.env_origins[env_ids][:, 2]

    # 关节：只扰动 8 个主动腿关节，follower 派生，轮保持默认
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    leg = _leg_ids(env, asset)
    joint_pos[:, leg] += torch.empty(n, len(leg), device=device).uniform_(
        -leg_joint_noise, leg_joint_noise
    )
    soft = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(soft[..., 0], soft[..., 1])
    _derive_followers(asset, joint_pos, joint_vel)

    if collision_clearance is not None:
        geometry = getattr(asset, "_upright_collision_geometry", None)
        if geometry is None:
            geometry = RoverCollisionGeometry(asset.cfg.spawn.asset_path, asset.joint_names)
            asset._upright_collision_geometry = geometry
        # 旧0.35m只检查底盘间隙，漏掉后轮约10cm穿透；必须对整车所有碰撞体取最小值。
        root_state[:, 2] = geometry.root_height(joint_pos, root_state[:, 3:7], collision_clearance)
        root_state[:, 2] += env.scene.env_origins[env_ids, 2]

    if moving_reset and quad_mask.any():
        assert geometry is not None
        speeds = torch.empty(int(quad_mask.sum()), device=device).uniform_(*quad_forward_velocity_range)
        root_state[quad_mask, 7] = speeds
        moving = quad_mask & (root_state[:, 7] != 0.0)
        if moving.any():
            # 转向噪声与严格无侧滑直行不相容；仅对带速四轮样本置直，不改平衡样本。
            steering, _ = asset.find_joints(list(PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES), preserve_order=True)
            rows = moving.nonzero(as_tuple=True)[0]
            joint_pos[rows[:, None], steering] = 0.0
            wheels, _ = asset.find_joints(list(PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES), preserve_order=True)
            wheel_vel = geometry.straight_wheel_velocities(
                joint_pos[moving], root_state[moving, 3:7], root_state[moving, 7:10],
                PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
            )
            # soft_joint_vel_limits 在首次 actuator.compute 才填充，首reset仍全零。
            # 使用初始化时已写入的PhysX硬限速和DCMotor无载限速，并按真实DOF顺序对齐。
            motor = asset.actuators["wheels"]
            motor_ids = list(range(len(asset.joint_names)))[motor.joint_indices] if isinstance(
                motor.joint_indices, slice) else list(motor.joint_indices)
            motor_order = [motor_ids.index(wheel) for wheel in wheels]
            limits = torch.minimum(
                asset.data.joint_vel_limits[env_ids[moving]][:, wheels],
                motor.velocity_limit[env_ids[moving]][:, motor_order],
            )
            if not torch.isfinite(limits).all() or torch.any(limits <= 0) or torch.any(wheel_vel.abs() > limits):
                raise ValueError("轮速度限位无效或合成初态超过执行器/PhysX速度限位，拒绝静默裁剪")
            joint_vel[rows[:, None], wheels] = wheel_vel
            # 转向置直后重新计算高度，避免保留之前带转向噪声的碰撞几何。
            root_state[moving, 2] = geometry.root_height(joint_pos[moving], root_state[moving, 3:7], collision_clearance)
            root_state[moving, 2] += env.scene.env_origins[env_ids[moving], 2]

    # 写入前守卫（见模块 docstring）
    if (not torch.isfinite(joint_pos).all() or not torch.isfinite(joint_vel).all()
            or not torch.isfinite(root_state).all()):
        bad = [asset.joint_names[i] for i in range(joint_pos.shape[1])
               if not torch.isfinite(joint_pos[:, i]).all()]
        raise RuntimeError(
            f"reset_upright_pose 生成了非有限状态，拒绝写入 PhysX。\n"
            f"  非有限关节: {bad}\n"
            f"  root_state finite: {bool(torch.isfinite(root_state).all())}"
        )

    asset.write_root_state_to_sim(root_state, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 清 prev_action，避免 episode 边界的假 action_rate 尖峰
    if hasattr(env, "action_manager"):
        prev = getattr(env.action_manager, "prev_action", None)
        if prev is not None and torch.is_tensor(prev):
            prev[env_ids] = 0.0
