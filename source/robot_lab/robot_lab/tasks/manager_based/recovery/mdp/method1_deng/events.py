# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.5.0
# @last_modified 2026-09-17
# @changelog
#   - v0.5.0 (2026-09-17): 显式横滚角段预训练入口；拒绝非法权重/角度并保留旧有效路径RNG。
#   - v0.4.0 (2026-09-17): 支持显式 settled 准备腿阻尼制动；维持零刚度和零速度目标。
#   - v0.3.0 (2026-09-17): 可选 URDF 全碰撞近地倒姿复位，保留旧高处跌落默认行为。
#   - v0.2.0 (2026-09-16): 支持世界系初始根速度范围，复用阶段感知的零驱动力掩码。
#   - v0.1.2 (2026-09-16): 自由落体同步 PhysX 隐式驱动增益，仅切换指定环境；拒绝传送回退。
#   - v0.1.1 (2026-09-16): 跌落高度相对环境原点，避免非零地形高度导致穿地或跌落距离漂移。
#   - v0.1.0 (2026-08-25): 移植 EDRS 自由落体复位；新增平行四边形 follower 派生与加权跌倒姿态池。

"""恢复任务的事件项（EDRS method1）。

两个核心事件：

- ``reset_with_fall_pose``：按**加权姿态池**采样初始跌倒姿态（PLAN §11.2）。
  用侧躺、仰躺及倾斜姿态提高目标场景的采样密度；这些权重是人工设定，
  尚未由真实失败轨迹估计。uniform SO(3)可覆盖附近姿态，但不会集中采样指定场景。

- ``zero_action_freefall``：前 2 s 强制零力矩（零化 per-env stiffness/damping +
  PD 目标钉在当前位置），让机器人真实塌落成多样跌倒姿态。

**平行四边形特有**：8 个被动 follower（zhuanxiang/lower_arm）必须在复位时
从 4 个 arm master **符号派生**（`zhuanxiang = -arm`, `lower_arm = +arm`），
否则 PhysX mimic 约束在首个物理步会因初值不自洽而产生冲击力矩。
参考实现（Thunder 无 mimic 机构）没有这一步。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_lab.assets.parallelogram_rover import PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS

from .._utils import _ensure_step_counter, _get_joint_groups, _is_freefall

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── 平行四边形 mimic 关系解析 ──

def _mimic_relations(asset: Articulation) -> tuple[tuple[int, int, float, float], ...]:
    """解析并缓存 (follower_id, master_id, multiplier, offset) 四元组。

    本函数是现有 locomotion `mdp.py::_mimic_relation_ids` 的**族内副本**
    （遵守 PLAN §3.0-A：现有文件只读，需变体则复制）。差异在于本副本用于
    跌倒姿态复位，master 角度来自姿态池而非默认位姿。
    """
    cache = "_recovery_mimic_relations"
    cached = getattr(asset, cache, None)
    if cached is not None:
        return cached

    relations = []
    for follower_name, master_name, multiplier, offset in PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS:
        ids, _ = asset.find_joints([follower_name, master_name], preserve_order=True)
        relations.append((int(ids[0]), int(ids[1]), float(multiplier), float(offset)))
    result = tuple(relations)
    setattr(asset, cache, result)
    return result


def _derive_followers(asset: Articulation, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> None:
    """就地把 8 个 follower 的位置/速度从其 master 派生。

    follower 永远是 master 的镜像，不参与随机化、不作为独立自由度。
    位置用 `multiplier * master + offset`，速度用 `multiplier * master_vel`（offset 是常量，微分为 0）。
    """
    for follower_id, master_id, multiplier, offset in _mimic_relations(asset):
        joint_pos[:, follower_id] = multiplier * joint_pos[:, master_id] + offset
        joint_vel[:, follower_id] = multiplier * joint_vel[:, master_id]


# ── 姿态采样 ──

def _random_quat_so3(n: int, device: str) -> torch.Tensor:
    """SO(3) 上的均匀随机单位四元数（Shoemake 1992），返回 (n,4) 的 [w,x,y,z]。"""
    u1 = torch.rand(n, device=device)
    u2 = torch.rand(n, device=device) * 2.0 * math.pi
    u3 = torch.rand(n, device=device) * 2.0 * math.pi
    return torch.stack(
        [
            torch.sqrt(1.0 - u1) * torch.sin(u2),
            torch.sqrt(1.0 - u1) * torch.cos(u2),
            torch.sqrt(u1) * torch.sin(u3),
            torch.sqrt(u1) * torch.cos(u3),
        ],
        dim=1,
    )


def _quat_about_axis(angle: torch.Tensor, axis: str) -> torch.Tensor:
    """绕单一轴旋转的四元数，返回 (n,4) 的 [w,x,y,z]。"""
    half = angle * 0.5
    c, s = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(s)
    if axis == "x":
        return torch.stack([c, s, zero, zero], dim=1)
    if axis == "y":
        return torch.stack([c, zero, s, zero], dim=1)
    return torch.stack([c, zero, zero, s], dim=1)


def _validate_fall_pose_parameters(pose_weights, roll_tilt_range_deg=None):
    """在任何采样或物理写入前校验；未知类别不能退化为全姿态随机训练。"""
    if roll_tilt_range_deg is not None:
        if pose_weights is not None:
            raise ValueError("roll_tilt_range_deg 与 pose_weights 不能同时指定")
        if not isinstance(roll_tilt_range_deg, (list, tuple)) or len(roll_tilt_range_deg) != 2:
            raise ValueError("roll_tilt_range_deg 必须为两个有限角度 (min,max)")
        try:
            low, high = (float(value) for value in roll_tilt_range_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError("roll_tilt_range_deg 必须为两个有限角度") from exc
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high <= 180.0):
            raise ValueError("roll_tilt_range_deg 必须满足 0 < min <= max <= 180 度")
        return low, high
    weights = DEFAULT_FALL_POSE_WEIGHTS if pose_weights is None else pose_weights
    labels = {"upright_fail", "side_lying", "prone", "uniform_so3"}
    if not isinstance(weights, dict) or not weights or set(weights) - labels:
        raise ValueError("pose_weights 必须仅包含已知姿态类别且非空")
    try:
        values = [float(value) for value in weights.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError("pose_weights 必须为有限非负权重") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in values) or not any(value > 0 for value in values):
        raise ValueError("pose_weights 必须为有限非负权重，且至少一项为正")
    # 采样使用默认torch浮点精度；极端有限Python数也不能溢出/下溢后触发随机回退。
    total = torch.tensor(values).sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("pose_weights 在采样浮点精度下的总和必须为有限正数")
    return None


def _sample_fall_quat(
    n: int, device: str, pose_weights: dict[str, float] | None, roll_tilt_range_deg=None,
) -> torch.Tensor:
    """按人工加权姿态池采样跌倒朝向；不把它称为实测失败分布。

    姿态类别与依据：
      - ``upright_fail``：绕 y 轴 ±20°–60°的倾斜初态；仍需实际沉降确认倒地。
      - ``side_lying``：绕 x 轴 ±90°的两侧躺姿。
      - ``prone``：历史名称，实际绕 x 轴180°，车轮朝上，为仰躺。
      - ``uniform_so3``：均匀 SO(3)，作泛化尾部

    ``pose_weights`` 未归一化也可（内部按和归一）。未列出的类别权重视为 0。
    显式 ``roll_tilt_range_deg`` 只用于独立课程，绕X轴随机取双侧角段，不再采样姿态池。
    """
    bounds = _validate_fall_pose_parameters(pose_weights, roll_tilt_range_deg)
    if bounds is not None:
        angles = torch.empty(n, device=device).uniform_(math.radians(bounds[0]), math.radians(bounds[1]))
        signs = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
        return _quat_about_axis(angles * signs, "x")
    pose_weights = DEFAULT_FALL_POSE_WEIGHTS if pose_weights is None else pose_weights
    # 旧有效输入保留原类别顺序、tensor精度与RNG调用顺序。
    labels = ["upright_fail", "side_lying", "prone", "uniform_so3"]
    w = torch.tensor([max(0.0, float(pose_weights.get(k, 0.0))) for k in labels], device=device)
    cat = torch.multinomial(w / w.sum(), n, replacement=True)

    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1.0

    # 前倾/后仰初态；区间是场景设置，不是物理不可保持边界。
    m = cat == 0
    if bool(m.any()):
        k = int(m.sum())
        ang = torch.empty(k, device=device).uniform_(math.radians(20.0), math.radians(60.0))
        sign = torch.where(torch.rand(k, device=device) < 0.5, -1.0, 1.0)
        quat[m] = _quat_about_axis(ang * sign, "y")

    # side_lying: 绕 x 轴 ±90°
    m = cat == 1
    if bool(m.any()):
        k = int(m.sum())
        sign = torch.where(torch.rand(k, device=device) < 0.5, -1.0, 1.0)
        ang = torch.full((k,), math.radians(90.0), device=device) * sign
        quat[m] = _quat_about_axis(ang, "x")

    # 历史prone标签实际为仰躺，车轮朝上。
    m = cat == 2
    if bool(m.any()):
        k = int(m.sum())
        quat[m] = _quat_about_axis(torch.full((k,), math.pi, device=device), "x")

    # uniform_so3: 泛化尾部
    m = cat == 3
    if bool(m.any()):
        quat[m] = _random_quat_so3(int(m.sum()), device)

    return quat


# ── 事件：复位 ──

DEFAULT_FALL_POSE_WEIGHTS: dict[str, float] = {
    "upright_fail": 0.40,
    "side_lying": 0.20,
    "prone": 0.20,
    "uniform_so3": 0.20,
}
"""默认姿态池权重（PLAN §11.2）。

注：PLAN 表中 `stair_tumble` 占 0.10，但"台阶翻滚"由**地形** proportion 实现
（跌倒发生在台阶上），不是独立朝向类别；故其权重合并进 uniform_so3。
"""


def _sample_root_velocity(
    count: int, device: str, velocity_range: dict[str, tuple[float, float]] | None,
) -> torch.Tensor:
    """采样世界系线速度(m/s)和角速度(rad/s)，先完整校验再改变物理状态。"""
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    ranges = {} if velocity_range is None else velocity_range
    if not ranges:
        # 默认零速度不消耗 RNG，保留原 drop 协议的种子采样序列。
        return torch.zeros(count, 6, device=device)
    unknown = set(ranges) - set(axes)
    if unknown:
        raise ValueError(f"未知根速度轴：{sorted(unknown)}")
    bounds = []
    for axis in axes:
        pair = ranges.get(axis, (0.0, 0.0))
        if len(pair) != 2 or not all(math.isfinite(v) for v in pair) or pair[0] > pair[1]:
            raise ValueError(f"根速度 {axis} 必须为有限的 (min,max)，实际为 {pair}")
        bounds.append(pair)
    limits = torch.tensor(bounds, device=device)
    return limits[:, 0] + torch.rand(count, 6, device=device) * (limits[:, 1] - limits[:, 0])


def reset_with_fall_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    drop_height: float = 1.1,
    leg_joint_noise: float = 0.3,
    pose_weights: dict[str, float] | None = None,
    root_velocity_range: dict[str, tuple[float, float]] | None = None,
    collision_clearance: float | None = None,
    roll_tilt_range_deg: tuple[float, float] | None = None,
):
    """按加权姿态池复位：随机跌倒朝向 + 抬升下落 + 主动关节扰动 + follower 派生。

    对齐 EDRS 论文 §III-A（随机朝向、1.1 m 自由落体、随机关节角、零力矩），
    但朝向采样改为加权姿态池（见 `_sample_fall_quat`）。

    Args:
        drop_height: 相对环境原点的抬升高度（米）。论文用 1.1，配合 2 s 零力矩窗口确保完全塌落。
        leg_joint_noise: 主动腿关节角扰动幅度 Uniform(±noise) rad。
            **仅施加于 8 个主动腿关节**；轮关节保持默认（连续关节角无意义），
            follower 由 master 派生。扰动后 clamp 到 soft limit，避免 PhysX
            在首个物理步 clamp 而产生冲击力矩。
        pose_weights: 姿态类别权重，None 时用 `DEFAULT_FALL_POSE_WEIGHTS`。
        root_velocity_range: 世界系 x/y/z 线速度及 roll/pitch/yaw 角速度范围；未指定轴为零。
        collision_clearance: 平地近地复位时按真实碰撞几何覆盖 drop_height；None 保留高处跌落协议。
        roll_tilt_range_deg: 显式双侧横滚角段（度）；与姿态池互斥，仅供课程预训练。
    """
    if len(env_ids) == 0:
        return
    _validate_fall_pose_parameters(pose_weights, roll_tilt_range_deg)
    asset: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    root_velocity = _sample_root_velocity(n, device, root_velocity_range)
    # 初速度只影响指定环境；地面倒姿不能靠手填 z 保证，应使用 settled_ground 协议。
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 2] = env.scene.env_origins[env_ids, 2] + drop_height
    root_state[:, 3:7] = _sample_fall_quat(n, device, pose_weights, roll_tilt_range_deg)
    root_state[:, 7:] = root_velocity

    # 关节状态：仅扰动主动腿关节，follower 派生，轮子保持默认
    groups = _get_joint_groups(env, asset)
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    leg_ids = groups["active_leg"]
    noise = torch.empty(n, len(leg_ids), device=device).uniform_(-leg_joint_noise, leg_joint_noise)
    joint_pos[:, leg_ids] += noise

    # clamp 到 soft limit（防 PhysX 首步 clamp 冲击）
    soft = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(soft[..., 0], soft[..., 1])

    # follower 必须在 clamp 之后派生——否则 clamp 会破坏 mimic 一致性
    _derive_followers(asset, joint_pos, joint_vel)

    if collision_clearance is not None:
        geometry = getattr(asset, "_recovery_collision_geometry", None)
        if geometry is None:
            from robot_lab.tasks.manager_based.upright.mdp.geometry import RoverCollisionGeometry

            geometry = RoverCollisionGeometry(asset.cfg.spawn.asset_path, asset.joint_names)
            asset._recovery_collision_geometry = geometry
        # 先派生 mimic 再求整车最低点；近地倒姿仍须通过真实物理沉降门，不能直接标记已接管。
        root_state[:, 2] = geometry.root_height(joint_pos, root_state[:, 3:7], collision_clearance)
        root_state[:, 2] += env.scene.env_origins[env_ids, 2]

    # 写入前守卫：非有限值写进 PhysX 会静默发散（表现为训练首次更新即
    # `normal expects std >= 0`，难以回溯到复位函数）。method2 曾因对轮关节
    # （soft limit = [-inf,inf]）采样而踩到，故两处 reset 都加此拦截。
    if not torch.isfinite(joint_pos).all() or not torch.isfinite(root_state).all():
        bad_j = [asset.joint_names[i] for i in range(joint_pos.shape[1])
                 if not torch.isfinite(joint_pos[:, i]).all()]
        raise RuntimeError(
            f"reset_with_fall_pose 生成了非有限状态，拒绝写入 PhysX。\n"
            f"  非有限关节: {bad_j}\n"
            f"  常见原因：对 continuous 关节（轮）按 soft limit [-inf,inf] 采样。"
        )

    asset.write_root_state_to_sim(root_state, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 复位步计数器为 -1：首次自增回到 0（见 _utils._advance_step_counter）
    _ensure_step_counter(env)
    env._recovery_step_count[env_ids] = -1

    # 清 action manager 的 prev_action，避免 episode 边界产生假的 action_rate 尖峰
    if hasattr(env, "action_manager"):
        prev = getattr(env.action_manager, "prev_action", None)
        if torch.is_tensor(prev):
            prev[env_ids] = 0.0


# ── 事件：自由落体零力矩 ──

def _cache_actuator_gains(env: ManagerBasedRLEnv, asset: Articulation) -> None:
    """在首次清零前保存增益；不支持真实释放驱动时直接拒绝运行。"""
    if hasattr(env, "_recovery_gain_cache"):
        return

    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, act in asset.actuators.items():
        s, d = getattr(act, "stiffness", None), getattr(act, "damping", None)
        if not (
            torch.is_tensor(s) and torch.is_tensor(d) and s.ndim == 2
            and s.shape == d.shape and s.shape[0] == env.num_envs
            and hasattr(act, "joint_indices") and hasattr(act, "is_implicit_model")
        ):
            raise RuntimeError(
                f"recovery freefall 不支持执行器 {name}：必须提供逐环境 PD 增益和关节索引，"
                "不能通过传送关节伪装自由落体。"
            )
        if act.is_implicit_model and not all(
            callable(getattr(asset, method, None))
            for method in ("write_joint_stiffness_to_sim", "write_joint_damping_to_sim")
        ):
            raise RuntimeError("recovery freefall 需要 Articulation 的 PhysX 刚度/阻尼写入接口。")
        cache[name] = (s.clone(), d.clone())
    env._recovery_gain_cache = cache
    env._recovery_freefall_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def zero_action_freefall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """准备窗口按协议释放驱动；默认零力矩，窗口结束后恢复原增益。

    settled 可显式设置 preparation_leg_damping，仅对主动腿关节施加零速度
    阻尼制动。它改变准备控制模式，必须随实验记录，不能当作纯自由落体。

    ImplicitActuator 的张量只用于扭矩估算，真实 PD 在 PhysX 内，必须同步写入物理
    刚度/阻尼。DCMotor 在 Python 侧计算力矩，只改模型增益，不能把其阻尼写入 PhysX
    再算一遍。两者仅在进入/退出窗口时切换增益，避免每个控制步 CPU 往返。
    自由落体期间也清除已有前馈力矩；本任务的 action 必须继续使用位置/速度控制，
    不能另挂每个物理子步覆写前馈力矩的 effort action。

    需在 cfg 中绑定**两次**：``mode="reset"``（使首个物理步就已零力矩）
    与 ``mode="interval"``（后续每步维持，直到退出窗口）。

    不支持可变增益或缺少物理写入接口时直接报错，不使用刚性传送替代自由落体。

    ⚠️ **签名约束（勿加默认值）**：`env_ids` 必须是**无默认值**的位置参数。
    Isaac Lab 的 EventManager 用 `min_argc=2` 校验签名，并会把参数重排为
    "无默认值优先"（`manager_base.py:363-369`）。若写成 `env_ids=None`，
    重排后校验集合不匹配，构建环境时直接抛
    "expects mandatory parameters: [] ... but received: ['asset_cfg']"。
    interval 模式下 EventManager 会传 `env_ids=None`，函数内部已处理该情况。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _ensure_step_counter(env)
    _cache_actuator_gains(env, asset)
    brake = getattr(env, "_recovery_preparation_leg_damping", 0.0)
    leg_ids = _get_joint_groups(env, asset)["active_leg"] if brake > 0.0 else None

    subset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if env_ids is not None:
        subset[:] = False
        subset[env_ids] = True
    in_freefall = _is_freefall(env)
    ff_idx = torch.where(in_freefall & subset)[0]
    changed_ids = torch.where(subset & (in_freefall != env._recovery_freefall_active))[0]
    if len(changed_ids) > 0:
        active = in_freefall[changed_ids, None]
        for name, act in asset.actuators.items():
            s_ref, d_ref = env._recovery_gain_cache[name]
            stiffness = torch.where(active, 0.0, s_ref[changed_ids])
            if leg_ids is not None:
                actuator_ids = torch.arange(asset.num_joints, device=env.device)[act.joint_indices] if isinstance(
                    act.joint_indices, slice
                ) else torch.as_tensor(act.joint_indices, device=env.device)
                preparation_damping = torch.isin(actuator_ids, leg_ids).to(d_ref.dtype) * brake
            else:
                preparation_damping = 0.0
            damping = torch.where(active, preparation_damping, d_ref[changed_ids])
            act.stiffness[changed_ids] = stiffness
            act.damping[changed_ids] = damping
            if act.is_implicit_model:
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=act.joint_indices, env_ids=changed_ids)
                asset.write_joint_damping_to_sim(damping, joint_ids=act.joint_indices, env_ids=changed_ids)
        env._recovery_freefall_active[changed_ids] = in_freefall[changed_ids]
    if len(ff_idx) > 0:
        asset.set_joint_position_target(asset.data.joint_pos[ff_idx], env_ids=ff_idx)
        asset.set_joint_effort_target(torch.zeros_like(asset.data.joint_pos[ff_idx]), env_ids=ff_idx)
        if leg_ids is not None:
            # 防止残留速度指令把阻尼变成主动加速；本任务腿部 action 只设置位置目标。
            asset.set_joint_velocity_target(
                torch.zeros((len(ff_idx), len(leg_ids)), device=env.device), joint_ids=leg_ids, env_ids=ff_idx,
            )
