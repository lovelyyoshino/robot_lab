# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.4.0
# @last_modified 2026-09-17
# @changelog
#   - v0.4.0 (2026-09-17): 局部复位同步清理全部进展奖励缓存，阻断跨episode差分。
#   - v0.3.0 (2026-09-17): 显式可选准备期腿电机制动，零刚度/零目标速度，接管恢复原增益。
#   - v0.2.0 (2026-09-17): 保存接管前各稳定门失败计数和末次物理量，区分翻正与持续运动。
#   - v0.1.0 (2026-09-16): 正常并行物理步中完成被动沉降、接触门、零延时接管及独立控制时钟。

"""真实地面沉降后立即接管；不在任何局部 reset 回调内推进整个仿真。

reset：抬升并采样倒姿，关闭驱动 → termination：采集实际触地和稳定时长 →
interval：对满足条件的环境恢复增益、清动作历史 → 下一物理步：control_steps=1。
准备时间不消耗控制预算；超时/非有限状态明确记为 initialization_failed。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ._utils import _env_dt, _get_joint_groups
from .grounded import reset_grounded_progress
from .method1_deng.events import _sample_root_velocity, zero_action_freefall
from .observations import reset_prev_obs_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initialize_settled_recovery(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    preparation_leg_damping: float = 0.0,
) -> None:
    """只初始化所选环境；可选阻尼为电机制动，不是已标定的被动物理摩擦。

    默认零阻尼保留零力矩协议。非零模式只对8个主动腿关节施加 -d*qdot，
    无姿态目标、无轮驱动力；不能与零力矩模式混报成功率。
    """
    if not math.isfinite(preparation_leg_damping) or preparation_leg_damping < 0.0:
        raise ValueError("preparation_leg_damping 必须为有限非负数")
    if hasattr(env, "_recovery_preparation_leg_damping") and env._recovery_preparation_leg_damping != preparation_leg_damping:
        raise ValueError("preparation_leg_damping 不可在局部 reset 时改变；需重新构建环境")
    env._recovery_preparation_leg_damping = preparation_leg_damping
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_recovery_control_active"):
        for name in ("control_active", "takeover_pending", "initialization_failed"):
            setattr(env, f"_recovery_{name}", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        for name in ("control_steps", "settle_steps", "settle_streak", "initialization_last_step",
                     "initialization_failure_code"):
            setattr(env, f"_recovery_{name}", torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        env._recovery_takeover_root_state_w = torch.zeros_like(asset.data.root_state_w)
        env._recovery_takeover_joint_pos = torch.zeros_like(asset.data.joint_pos)
        env._recovery_takeover_joint_vel = torch.zeros_like(asset.data.joint_vel)
        env._recovery_takeover_gravity_b = torch.zeros_like(asset.data.projected_gravity_b)
        env._recovery_preparation_gate_names = (
            "contact", "linear_speed", "angular_speed", "leg_speed", "fallen", "finite",
        )
        env._recovery_preparation_gate_failures = torch.zeros(
            env.num_envs, 6, dtype=torch.long, device=env.device,
        )
        env._recovery_preparation_metrics = {}

    ids = slice(None) if env_ids is None else env_ids
    for name in ("control_active", "takeover_pending", "initialization_failed", "control_steps",
                 "settle_steps", "settle_streak", "initialization_failure_code"):
        getattr(env, f"_recovery_{name}")[ids] = 0
    for name in ("root_state_w", "joint_pos", "joint_vel", "gravity_b"):
        getattr(env, f"_recovery_takeover_{name}")[ids] = 0
    env._recovery_preparation_gate_failures[ids] = 0
    for value in env._recovery_preparation_metrics.values():
        value[ids] = 0
    env._recovery_initialization_last_step[ids] = env.common_step_counter
    reset_grounded_progress(env, env_ids)
    zero_action_freefall(env, env_ids, asset_cfg)


def update_settled_initialization(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    settle_time_s: float = 0.3,
    max_settle_time_s: float = 6.0,
    max_lin_speed: float = 0.10,
    max_ang_speed: float = 0.20,
    max_leg_speed: float = 0.30,
    min_contact_force: float = 1.0,
    require_fallen: bool = True,
    fallen_gz_min: float = -0.7,
) -> torch.Tensor:
    """首个 termination term：每物理控制步采样一次并返回准备失败，而非技能失败。

    不用猜测高度判断触地；汇总传感器自身全部 body 接触，不拿 articulation 的
    body 索引访问传感器（两者排序可不同），也不假定接触力会归属某个轮子。
    轮子可能悬空自转，因此稳定门只约束根速度及主动腿速度，不假定轮速为零。
    返回 True 的环境从未接管；失败码 1=准备超时，2=非有限物理状态。
    """
    positive = (settle_time_s, max_settle_time_s, max_lin_speed, max_ang_speed, max_leg_speed, min_contact_force)
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("沉降时长、速度上限和接触阈值必须为有限正数")
    if settle_time_s > max_settle_time_s:
        raise ValueError("settle_time_s 不能超过 max_settle_time_s")
    if not math.isfinite(fallen_gz_min) or not -1.0 <= fallen_gz_min <= 1.0:
        raise ValueError("fallen_gz_min 必须在 [-1,1] 内")
    if not hasattr(env, "_recovery_control_active"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    fresh = env._recovery_initialization_last_step != env.common_step_counter
    active = env._recovery_control_active
    preparing = fresh & ~active & ~env._recovery_initialization_failed
    env._recovery_control_steps += (fresh & active).long()
    env._recovery_settle_steps += preparing.long()
    env._recovery_initialization_last_step[fresh] = env.common_step_counter

    asset = env.scene[asset_cfg.name]
    contacts = env.scene.sensors[sensor_cfg.name].data.net_forces_w
    finite = (
        torch.isfinite(asset.data.root_state_w).all(dim=1)
        & torch.isfinite(asset.data.joint_pos).all(dim=1)
        & torch.isfinite(asset.data.joint_vel).all(dim=1)
        & torch.isfinite(contacts).all(dim=(1, 2))
        & torch.isfinite(asset.data.projected_gravity_b).all(dim=1)
    )
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    contact_force = torch.linalg.vector_norm(contacts, dim=-1).sum(dim=1)
    linear_speed = asset.data.root_lin_vel_w.norm(dim=1)
    angular_speed = asset.data.root_ang_vel_w.norm(dim=1)
    leg_speed = asset.data.joint_vel[:, leg_ids].abs().amax(dim=1)
    gravity_z = asset.data.projected_gravity_b[:, 2]
    contact_ok = contact_force >= min_contact_force
    linear_ok = linear_speed <= max_lin_speed
    angular_ok = angular_speed <= max_ang_speed
    leg_ok = leg_speed <= max_leg_speed
    fallen_ok = gravity_z >= fallen_gz_min if require_fallen else torch.ones_like(finite)
    gates = torch.stack((contact_ok, linear_ok, angular_ok, leg_ok, fallen_ok, finite), dim=1)
    stable = gates.all(dim=1)
    # 只记录准备期，接管后的策略运动不能污染“为什么未能接管”的诊断。
    env._recovery_preparation_gate_failures += (preparing[:, None] & ~gates).long()
    metrics = {
        "contact_force_n": contact_force, "linear_speed_m_s": linear_speed,
        "angular_speed_rad_s": angular_speed, "max_leg_speed_rad_s": leg_speed, "gravity_z": gravity_z,
        "contact_ok": contact_ok, "linear_speed_ok": linear_ok, "angular_speed_ok": angular_ok,
        "leg_speed_ok": leg_ok, "fallen_ok": fallen_ok, "finite_ok": finite,
    }
    for name, value in metrics.items():
        if name not in env._recovery_preparation_metrics:
            env._recovery_preparation_metrics[name] = torch.zeros_like(value)
        env._recovery_preparation_metrics[name][preparing] = value[preparing]
    streak = torch.where(stable, env._recovery_settle_streak + 1, 0)
    env._recovery_settle_streak[preparing] = streak[preparing]
    hold_steps = math.ceil(settle_time_s / _env_dt(env) - 1e-9)
    timeout_steps = math.ceil(max_settle_time_s / _env_dt(env) - 1e-9)
    ready = preparing & (env._recovery_settle_streak >= hold_steps)
    invalid = preparing & ~finite
    timed_out = preparing & ~ready & (env._recovery_settle_steps >= timeout_steps)
    env._recovery_takeover_pending[preparing] = ready[preparing]
    env._recovery_initialization_failed |= invalid | timed_out
    env._recovery_initialization_failure_code[timed_out] = 1
    env._recovery_initialization_failure_code[invalid] = 2
    return env._recovery_initialization_failed.clone()


def activate_settled_recovery(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    takeover_velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
    """控制步末接管，下一物理步立即受控；空速度范围保留真实沉降后的残余速度。

    非空 takeover_velocity_range 在真实地面状态上设置世界系初速度，用于动量扰动
    接管测试；不改位置/姿态/关节状态，不能称为静止倒姿实验。接管前的随机动作
    从未执行，因此动作历史清零，t-1 关节观测以接管实态初始化。
    """
    if not hasattr(env, "_recovery_control_active"):
        raise RuntimeError("settled_ground 必须先执行 initialize_settled_recovery")
    selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    selected[slice(None) if env_ids is None else env_ids] = True
    ready_ids = torch.where(selected & env._recovery_takeover_pending & ~env._recovery_initialization_failed)[0]
    asset = env.scene[asset_cfg.name]
    if len(ready_ids) > 0:
        if takeover_velocity_range:
            velocity = _sample_root_velocity(len(ready_ids), env.device, takeover_velocity_range)
            asset.write_root_velocity_to_sim(velocity, env_ids=ready_ids)
        env._recovery_takeover_root_state_w[ready_ids] = asset.data.root_state_w[ready_ids]
        env._recovery_takeover_joint_pos[ready_ids] = asset.data.joint_pos[ready_ids]
        env._recovery_takeover_joint_vel[ready_ids] = asset.data.joint_vel[ready_ids]
        env._recovery_takeover_gravity_b[ready_ids] = asset.data.projected_gravity_b[ready_ids]
        env._recovery_control_active[ready_ids] = True
        env._recovery_takeover_pending[ready_ids] = False
        env._recovery_control_steps[ready_ids] = 0
        env._recovery_step_count[ready_ids] = -1
        env.episode_length_buf[ready_ids] = 0
        env.action_manager.reset(ready_ids)
        reset_prev_obs_cache(env, ready_ids, asset_cfg)
    zero_action_freefall(env, env_ids, asset_cfg)


def settled_control_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """仅消耗接管后的控制预算；初始化超时由 preceding termination 单独记录。"""
    if not hasattr(env, "_recovery_control_active"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._recovery_control_active & (env._recovery_control_steps >= env.max_episode_length)
