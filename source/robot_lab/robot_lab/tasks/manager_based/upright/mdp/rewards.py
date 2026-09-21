# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.3.2
# @last_modified 2026-09-17
# @changelog
#   - v0.3.2 (2026-09-17): 撤销旧穿地探针对力矩/可保持范围的过强物理结论，数值配置不变。
#   - v0.3.1 (2026-09-16): 过渡奖励使用真实 pitch 正向导数，排除非轮承重、侧躺退化和非有限状态。
#   - v0.3.0 (2026-09-16): 接触门排除借底盘支撑的奖励；高度改相对地面并纠正旧靶值。
#   - v0.2.0 (2026-09-16): 新增四轮起立进度/成功奖励，并为高度奖励增加姿态门控。
#   - v0.1.0 (2026-08-25): 直立任务奖励；工作区 75°-105°，含穿越不可保持区的 transit 项。

"""直立（双轮平衡）任务的奖励项。

当前采样区间约75°–105°、过渡奖励窗口20°–60°，均是训练配置，
不是已验证的机械可行域。旧探针使用错误碰撞初态和接触索引，其力矩余量、
“只能在此区间保持”“横滚不可恢复”等结论不再作为依据。
真正的可达和可保持范围需在无穿透复位、真实轮支撑和独立批量验收下确认。

**与 WTW 参考实现的关键差异（符号！）**：
`corl_rewards.py:148 _reward_orientation_control` 返回的是
`sum(square(g_xy - desired_g_xy))` —— 那是**误差**，WTW 用负权重。
本实现改为 `exp(-err/σ²)` 正奖励形式（与本项目其他 exp 奖励一致），
**权重为正**。直接照抄原式并配正权重会奖励误差，是隐蔽的符号错误。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .support import current_support

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── 当前训练目标（不是机械能力保证）──

HOLDABLE_PITCH_MIN = 1.31
"""当前直立指令区间下界（rad，≈75°）；历史名称保留以兼容配置。"""

HOLDABLE_PITCH_MAX = 1.83
"""当前直立指令区间上界（rad，≈105°），须独立验证保持能力。"""

OPTIMAL_PITCH = 1.57
"""名义直立目标（rad，约90°）；尚不主张它是最省力姿态。"""

UNHOLDABLE_PITCH_MAX = 1.05
"""过渡奖励窗口上界（rad，≈60°）；历史名字不代表物理不可保持定理。"""

UPRIGHT_BASE_HEIGHT = 0.46
"""名义直立奖励高度约0.46m；正式复位由完整碰撞几何计算。

2026-09-16 修正：以下旧 probe20 解释已作废，不能当作可行高度范围。
旧0.35m只检查底盘，漏掉后轮约0.10m穿透；求解器首步弹出不代表复位合法。
保留下面原始记录用于追溯，不再用于设置训练或验收阈值。

初版取 0.30（"保守估计，待 L2 smoke 校准"），但**偏低导致复位即贴地**。
probe20 在 pitch=90° 下扫复位高度、只推进 1 物理步（不给时间倒下）实测：

| base_h | base_link 接触力 | 四轮合力 | 判定 |
|---|---|---|---|
| 0.30 | **41.95 N** | 5.42 N | 机体压地，轮子几乎不承重 |
| **0.32** | **0.00 N** | 30.40 N | 干净分界 |
| 0.36 | 0.00 N | 85.29 N | 正常 |
| 0.40 | 0.00 N | 40.52 N | 正常 |
| 0.45 | 0.00 N | **0.00 N** | 悬空，首步自由落体 |

有效窗口 **[0.32, 0.40]**，取中值 0.35 兼顾两端余量。

用 0.30 的后果（probe19 实测）：`pitch mean=90.4°`（姿态完全正确）时
`base_link` 仍有 12.23% 接触率 —— 复位瞬间机体就压在地上，
之后策略无论怎么调姿态都甩不掉这份惩罚。属复位缺陷，非策略问题。

⚠️ 该常数同时用作复位高度与 `upright_height` 奖励靶值，改动需同步核对两处。
"""


# ── 姿态跟踪 ──

_PITCH_DEGENERATE_EPS = 1e-3
"""判定 pitch 退化的阈值：当 sqrt(g_x²+g_z²) 小于此值时 pitch 无定义。"""


def _pitch_from_gravity(asset: Articulation) -> torch.Tensor:
    """从 projected_gravity 反解机体 pitch（rad）。

    绕 y 轴 pitch θ 时，机体系重力 g = [-sin θ, 0, -cos θ]。
    故 θ = atan2(-g_x, -g_z)，值域 (-π, π]，正立时 θ=0、后仰 90° 时 θ=π/2。

    用 atan2 而非 asin：asin 在 |g_x|→1 时数值不稳，且无法区分
    pitch=90° 与 pitch=270°（后者 g_z 符号相反）。

    ⚠️ **退化情形处理**：完全侧躺时 g=[0,±1,0]，g_x 与 g_z 同时为 0，
    此时 pitch **在数学上无定义**，而 `atan2` 的返回值取决于零的符号——
    实测 `atan2(0,0) = -π` 但 `atan2(-0,-0) = 0`，且 g_x,g_z 都趋于 0 时
    返回值可以是任意角（如 1e-9/-1e-9 → -45°）。这种不稳定值会给梯度注入噪声。

    故显式检测退化并返回 π/2（可保持区中心 = 最优工作点）：
    该值使 `pitch_tracking` 给出中性偏高的分数，但侧躺姿态会被
    `roll_penalty`（g_y²=1，最大惩罚）强力压制，净效果仍是强负激励。
    这比让 pitch 取随机值更可控。
    """
    g = asset.data.projected_gravity_b
    gx, gz = g[:, 0], g[:, 2]
    pitch = torch.atan2(-gx, -gz)
    # 退化时（x-z 平面分量趋零，即完全侧躺）pitch 无定义 → 用最优点占位
    degenerate = torch.sqrt(gx * gx + gz * gz) < _PITCH_DEGENERATE_EPS
    return torch.where(degenerate, torch.full_like(pitch, OPTIMAL_PITCH), pitch)


def pitch_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "pitch_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.25,
) -> torch.Tensor:
    """exp(−(θ − θ_cmd)² / σ²)：跟踪指令俯仰角。**正奖励**。

    ⚠️ 与 WTW `_reward_orientation_control` 的形式不同：那个返回平方误差
    （配负权重使用）。这里转为 exp 正奖励，权重为正。σ=0.25 rad（≈14°）
    给出足够宽的梯度带——可保持区总宽 0.52 rad，σ 太小会使区间边缘无梯度。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    err = _pitch_from_gravity(asset) - cmd
    no_body_support, _ = current_support(env)
    return torch.exp(-torch.square(err) / (sigma**2)) * no_body_support


def roll_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体系重力的 y 分量平方 —— 横滚惩罚。

    当前权重 -10.0，旨在抑制两后轮支撑时的侧倾。实际横滚可恢复范围与
    转向、关节动作和接触条件有关，尚未由受控实验确定。
    用 g_y² 而非 roll 角：前者在小角度下即给出梯度，且无需反三角函数。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def upright_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = UPRIGHT_BASE_HEIGHT,
    sigma: float = 0.1,
    min_pitch: float | None = None,
) -> torch.Tensor:
    """exp(−(h − h_target)² / σ²)：维持直立时的机体高度。

    防止策略用"贴地翘头"骗取 pitch 奖励（那不是站立）。起立阶段必须传入
    ``min_pitch``，否则四轮初始高度也可能获得非零奖励，形成不抬头的局部最优。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - target_height
    reward = torch.exp(-torch.square(err) / (sigma**2))
    if min_pitch is not None:
        reward *= (_pitch_from_gravity(asset) >= min_pitch).float()
    no_body_support, _ = current_support(env)
    return reward * no_body_support


def rise_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    start_pitch: float = 0.0,
    target_pitch: float = OPTIMAL_PITCH,
    roll_sigma: float = 0.35,
) -> torch.Tensor:
    """四轮到目标姿态的归一化稠密奖励，并抑制侧翻取巧。

    原 ``pitch_tracking`` 在四轮姿态与 90 度目标相差约 1.57 rad，指数奖励几乎为零，
    PPO 很难从随机动作中发现完整起立轨迹。本项在整个抬升区间提供线性梯度；达到
    目标后饱和为 1，最终稳定性仍由 ``pitch_tracking`` 和 ``balance_hold`` 负责。
    """
    if target_pitch <= start_pitch:
        raise ValueError("target_pitch 必须大于 start_pitch")
    asset: Articulation = env.scene[asset_cfg.name]
    pitch = _pitch_from_gravity(asset)
    progress = ((pitch - start_pitch) / (target_pitch - start_pitch)).clamp(0.0, 1.0)
    roll_gate = torch.exp(-torch.square(asset.data.projected_gravity_b[:, 1]) / (roll_sigma**2))
    no_body_support, _ = current_support(env)
    return progress * roll_gate * no_body_support


def getup_success(
    env: ManagerBasedRLEnv,
    command_name: str = "pitch_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_tol: float = 0.20,
    min_height: float = 0.30,
    ang_vel_tol: float = 0.75,
    max_abs_gy: float = 0.25,
) -> torch.Tensor:
    """姿态、高度、角速度和横滚同时达标时给出起立完成奖励。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 0]
    pitch = _pitch_from_gravity(asset)
    local_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    stable = asset.data.root_ang_vel_b.norm(dim=1) < ang_vel_tol
    level_roll = asset.data.projected_gravity_b[:, 1].abs() < max_abs_gy
    _, wheel_support = current_support(env)
    return (
        ((pitch - command).abs() < pitch_tol)
        & (local_height >= min_height)
        & stable
        & level_roll
        & wheel_support
    ).float()


# ── 穿越不可保持区 ──

def transit_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lower: float = 0.35,
    upper: float = UNHOLDABLE_PITCH_MAX,
) -> torch.Tensor:
    """仅奖励不可保持区内朝正 pitch 目标的进展，不能奖励倒落或借底盘振动。

    当前任务的目标在正 pitch 的 75°-105°，过渡区为 20°-60°，因此只奖励
    ``d atan2(-g_x, -g_z) / dt > 0``。重力在机体系满足 ``gdot = -omega × g``；
    倾斜时 roll/yaw 均可能影响该角度，不能将导数简化成 ``-omega_y``。

    接触门仅排除非轮承重，不要求整个起立过程始终由双后轮支撑。
    与既有 pitch 反解共用退化阈值；侧躺退化、非有限值或运算溢出返回零，
    不用 ``0 * NaN`` 掩码污染整批奖励。区间外不发放此项，不改变其他奖励。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pitch = _pitch_from_gravity(asset)
    in_transit = (pitch > lower) & (pitch < upper)
    gravity = asset.data.projected_gravity_b
    omega = asset.data.root_ang_vel_b
    radius_squared = gravity[:, 0].square() + gravity[:, 2].square()
    regular = (
        torch.isfinite(gravity).all(dim=-1)
        & torch.isfinite(omega).all(dim=-1)
        & torch.isfinite(radius_squared)
        & (radius_squared >= _PITCH_DEGENERATE_EPS**2)
    )
    safe_gravity = torch.where(regular[:, None], gravity, 0.0)
    safe_omega = torch.where(regular[:, None], omega, 0.0)
    denominator = torch.where(regular, radius_squared, 1.0)
    gravity_rate = -torch.linalg.cross(safe_omega, safe_gravity, dim=-1)
    # atan2 的商法则；纯世界 yaw 的 omega 与 g 平行，叉积为零，不会误发起立奖励。
    pitch_rate = (
        safe_gravity[:, 2] * gravity_rate[:, 0] - safe_gravity[:, 0] * gravity_rate[:, 2]
    ) / denominator
    no_body_support, _ = current_support(env)
    eligible = in_transit & regular & no_body_support & torch.isfinite(pitch_rate)
    return torch.where(eligible, pitch_rate.clamp_min(0.0), 0.0)


# ── 平衡质量 ──

def balance_hold(
    env: ManagerBasedRLEnv,
    command_name: str = "pitch_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_tol: float = 0.2,
    ang_vel_tol: float = 0.5,
) -> torch.Tensor:
    """在可保持区内**姿态达标且角速度小**时给固定奖励——鼓励"稳住"。

    与 `pitch_tracking` 的区别：后者是连续梯度（引导靠近），本项是
    离散的"已稳住"信号（区分"正在经过目标角"与"停在目标角"）。
    倒立摆的时间常数仅 0.24 s，若不显式奖励低角速度，策略容易学成
    高频振荡穿越目标角而非真正平衡。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    pitch = _pitch_from_gravity(asset)
    near = (pitch - cmd).abs() < pitch_tol
    calm = asset.data.root_ang_vel_b.norm(dim=1) < ang_vel_tol
    _, wheel_support = current_support(env)
    return (near & calm & wheel_support).float()


def wheel_effort_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """轮力矩平方和 —— 抑制过度用力。

    probe2 实测 90° 仅需 1.55 N·m（极限的 15%），故正常平衡时该项应很小；
    若它变大说明策略在用蛮力而非利用重心，值得惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
