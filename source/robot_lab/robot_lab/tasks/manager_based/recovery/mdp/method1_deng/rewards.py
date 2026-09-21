# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.1
# @last_modified 2026-09-17
# @changelog
#   - v0.2.1 (2026-09-17): 撤销接触合并与机械可行性的旧探针推断，保留EDRS数值基线。
#   - v0.2.0 (2026-09-16): settled_ground 准备期逐环境屏蔽奖励，含非有限初始化失败样本。
#   - v0.1.1 (2026-09-16): 朝向奖励区分正立/仰躺；高度奖励与成功门使用环境原点相对高度。
#   - v0.1.0 (2026-08-25): 移植 EDRS 13 项 ED/CW 奖励；权重按 10 N·m 执行器下调，成功判据排除 follower。

"""恢复任务的奖励项（EDRS method1）。

参考：`Kitjesen/recovery` 的 `thunder_recovery/mdp/method1_deng/rewards.py`
（Deng et al., arXiv:2506.05516）。

Episode 时间线（T=5 s, 50 Hz, 250 步）：

    ┌──────────────────┬─────────────────────┬──────────────────────┐
    │ 自由落体          │ 探索                 │ 收敛                  │
    │ t ∈ [0, 2 s]     │ t ∈ [2, ~3.5 s]     │ t ∈ [~3.5, 5 s]      │
    │ ED 0 → 0.064     │ ED 0.064 → 0.34     │ ED 0.34 → 1.0        │
    │ 增益=0，关节自由   │ 策略生效，任务奖励弱   │ 任务奖励主导 → 收敛    │
    └──────────────────┴─────────────────────┴──────────────────────┘

与参考实现的三处关键差异（均有实测/文档依据）：

1. **行为惩罚权重再降一个量级**：参考实现已把论文权重从 -2e-2 降到 -3e-3
   （其 README：论文权重在 KYON 上调的，对 Thunder 过度惩罚）。本机器人腿部
   执行器仅 **10 N·m**（Thunder 为 120 N·m，差 12 倍），故进一步降到 -1e-3 量级。

2. **成功判据排除 follower 关节**：8 个被动关节是 master 的镜像，计入 norm
   等于重复计数并使阈值失去物理意义。

3. **`min_base_height` 门按本机几何重标**：参考用 0.30 m（其 spawn z=0.55）；
   本机 spawn z=0.37，四轮落地时 base 原点高 ≈ WHEEL_R − off_center_z
   ≈ 0.1085 + 0.175 ≈ 0.28 m，故门设 **0.20 m**（明显高于趴地、低于标称站高）。
   ⚠️ 该值需在 L2 smoke 用实测 base_height 分布校准。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .._utils import (
    _advance_step_counter,
    _control_phase_reward,
    _env_dt,
    _ensure_step_counter,
    _get_cw,
    _get_ed,
    _get_joint_groups,
    _is_freefall,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NOMINAL_BASE_HEIGHT = 0.332
"""标称站立高度（米），**实测值**（probe4/probe7：正立稳定后 mean=0.332，8 env 一致）。

注：URDF FK 推算为 0.2835（轮半径 0.1085 + 轮轴偏移 0.175），实测高 0.049。
差异来自默认关节位姿下腿部构型（arm=0 时轮组并非正下方）。**以实测为准**。
"""

MIN_BASE_HEIGHT = 0.26
"""support/success 的机体高度门（米），**由实测分布校准**。

校准依据（probe4/7/8 实测）：
    零动作塌落沉降 = 0.229   ← 门必须高于此值，否则趴着即可持续刷分
    正常站立       = 0.332   ← 门必须低于此值，否则真站起来也拿不到分
    侧躺           = 0.318   ← 高度与站立仅差 0.014，**故必须配合姿态判据**
→ 取 0.26（区间 (0.229, 0.332] 内偏下，留策略未完全站直的余量）

⚠️ 初版设 0.20 是按 FK 推算反推的，实测证明**低于塌落高度 0.229**，
门形同失效（smoke 实测 support 累计达 4.0 = 趴着刷满）。
"""

UPRIGHT_GZ_THRESHOLD = -0.7
"""判定"正立"的重力投影阈值：`projected_gravity_b[:, 2] < -0.7`。

实测三姿态的 gz（probe8）：站立 -1.0 / 侧躺 0.0 / 仰躺 +1.0。
取 -0.7 可干净分离站立，且对未完全站直（gz≈-0.8）仍宽容。
"""


# ── 步计数器驱动项（副作用项）──

def recovery_step_counter(env: ManagerBasedRLEnv) -> torch.Tensor:
    """推进 ED 时钟。**必须每步执行**。

    返回恒为 0；真实作用是副作用式自增。在 cfg 中权重设 1e-6（微小但非零）——
    某些 Isaac Lab 版本会把权重恰为 0 或 ≤1e-9 的项从 reward manager 中剪除，
    那样 ED 时钟就不会推进，整个 ED/CW 机制静默失效。
    """
    _advance_step_counter(env)
    return torch.zeros(env.num_envs, device=env.device)


# ── 任务奖励（× ED）──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """ED · exp(−‖q_active − q_default‖² / σ²)：趋近站立关节位姿。

    仅统计 **8 个主动腿关节**（arm + qudong）：轮关节是连续关节、其绝对角无意义；
    follower 是被动镜像。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    err = asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]
    return _control_phase_reward(env, _get_ed(env) * torch.exp(-torch.sum(torch.square(err), dim=1) / (sigma**2)))


def recovery_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = NOMINAL_BASE_HEIGHT,
    sigma: float = 0.15,
) -> torch.Tensor:
    """ED · exp(−(h − h_target)² / σ²)：抬升机体到标称站高。

    高度相对各环境原点，避免地形整体抬升后同样的站姿得到不同奖励。
    这是恢复任务的主驱动项（参考实现权重最高，120）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    err = height - target_height
    return _control_phase_reward(env, _get_ed(env) * torch.exp(-torch.square(err) / (sigma**2)))


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.3,
) -> torch.Tensor:
    """ED · exp(−‖g − [0,0,−1]‖² / σ²)：恢复到正立，排除仰躺。

    只惩罚 g_xy 会让正立和仰躺同时满分。加入 (g_z+1)² 后与成功判据同源，
    且小倾角 θ 附近误差仍为 θ² + O(θ⁴)，保留原 sigma 的局部尺度。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    orientation_error = torch.sum(torch.square(gravity[:, :2]), dim=1) + torch.square(gravity[:, 2] + 1.0)
    return _control_phase_reward(env, _get_ed(env) * torch.exp(-orientation_error / (sigma**2)))


def recovery_support_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_base_height: float = MIN_BASE_HEIGHT,
    max_gz: float = UPRIGHT_GZ_THRESHOLD,
) -> torch.Tensor:
    """历史EDRS基线的姿态/高度奖励，不等于真实四轮支撑或技能成功。

    旧探针误用 articulation 索引读取接触传感器，曾将错位接触解释成碰撞体
    合并；该解释已撤销。按 sensor.body_names 可正确识别四轮承重。
    此函数保留旧基线数值便于比较，新Grounded任务使用 grounded.py 的真实
    四轮接触、无非轮承重和速度门；不能拿这里的奖励值报告恢复成功率。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 姿态：重力在机体 z 轴投影接近 -1 表示正立（侧躺≈0、仰躺≈+1）
    upright = asset.data.projected_gravity_b[:, 2] < max_gz
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    base_up = height > min_base_height
    return _control_phase_reward(env, (upright & base_up & ~_is_freefall(env)).float())


def recovery_arm_wheel_coord(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(1 − ED) · |arm 角速度| · |轮角速度| 的耦合项：鼓励 **arm 摆动与轮转联动**。

    **这是本实现相对参考的重要调整**。probe2 实测（126 组开环扫描）结论：

      - 侧躺翻正最佳组合 = `arm=-0.4` + 单轮转 → Δg=1.414
      - 俯卧翻正最佳组合 = `arm=-0.4` + 一对轮 8 rad/s → Δg=2.000
      - 而**单一恒速轮转**（不动 arm）200 步仅使姿态变化 6.6° —— 几乎无效

    即：翻正的主力是 **arm 摆重心**，轮转是辅助。参考实现的 `wheel_leg_coord`
    以轮为主，对本构型不适用，故改为对称耦合形式，并用 (1−ED) 门控——
    仅在探索期（早期）激励联动，收敛期让位于精确站姿。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    groups = _get_joint_groups(env, asset)
    arm_ids, wheel_ids = groups["active_leg"][:4], groups["wheel"]  # 前 4 个是 arm

    arm_rate = asset.data.joint_vel[:, arm_ids].abs().mean(dim=1)
    wheel_rate = asset.data.joint_vel[:, wheel_ids].abs().mean(dim=1)
    coord = arm_rate * wheel_rate

    return _control_phase_reward(env, (1.0 - _get_ed(env)) * coord * (~_is_freefall(env)).float())


# ── 行为惩罚（× CW，随训练迭代衰减）──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    force_clip: float = 50.0,
) -> torch.Tensor:
    """CW · Σ clip(‖λ_b‖, 0, force_clip)²：非轮部件（base/arm/qudong）触地惩罚。

    对每体接触力做 clip：复位瞬间的冲击接触会产生极大力值，不 clip 会在训练早期
    产生梯度尖峰。参考实现同样做了 clip（论文原式无 clip）。
    """
    if _is_freefall(env).all():
        return torch.zeros(env.num_envs, device=env.device)
    if sensor_cfg.body_ids is None or sensor_cfg.body_ids == slice(None):
        raise RuntimeError(
            "recovery_body_collision 需要由 body_names 正则解析出的 sensor_cfg.body_ids；"
            "没有安全的索引回退方案。"
        )
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, :, :]
    magnitude = torch.norm(forces, dim=-1)[:, sensor_cfg.body_ids].clamp(max=force_clip)
    reward = _get_cw(env) * torch.sum(torch.square(magnitude), dim=1)
    return _control_phase_reward(env, reward)


def recovery_action_rate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """CW · Σ (aₜ − aₜ₋₁)²：抑制动作抖动。

    **与参考实现的索引差异（重要）**：参考实现用关节 ID 索引 action tensor，
    这在本机器人上会取错维度——因为 20 个关节里 8 个是 follower，而策略动作
    只有 12 维。本实现直接对**整个 12 维动作向量**求差分（顺序为
    `[fl,rl,fr,rr]_arm | [..]_qudong | [..]_wheel`，即 12-D 合同顺序），
    不做关节 ID 映射，从根本上避免错位。
    """
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    if not (torch.is_tensor(action) and torch.is_tensor(prev)):
        return torch.zeros(env.num_envs, device=env.device)
    reward = _get_cw(env) * torch.sum(torch.square(action - prev), dim=1)
    return _control_phase_reward(env, reward)


# ── 常驻惩罚（不过 ED/CW 门，自由落体后恒生效）──

def recovery_joint_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ (q_active − q_default)²：常驻的"朝站姿走"梯度。

    **这一项是参考实现作者额外加的，不在论文 Table I 中**，作用是解决
    "ED 早期≈0 导致任务奖励无梯度、探索无方向"的问题：它不过 ED 门，
    自由落体结束后即持续提供朝向默认位姿的梯度。

    仅统计 8 个主动腿关节（理由同 `recovery_stand_joint_pos`）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]
    err = asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]
    return _control_phase_reward(env, torch.sum(torch.square(err), dim=1) * (~_is_freefall(env)).float())


def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ q̇_active²：抑制主动关节高速运动。

    ⚠️ 权重需谨慎：参考实现作者实测论文权重(-2e-2)在 iter 300 时是单项最大惩罚
    (-1.0/episode)，把策略逼进"低运动"局部最优；其降到 -3e-3。本机器人执行器更弱
    (10 vs 120 N·m)，cfg 中进一步降到 -1e-3。

    **排除 follower**：其速度是 master 的镜像，计入等于重复惩罚同一物理量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    active_ids = _get_joint_groups(env, asset)["active_all"]
    value = torch.sum(torch.square(asset.data.joint_vel[:, active_ids]), dim=1) * (~_is_freefall(env)).float()
    return _control_phase_reward(env, value)


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ q̈_active²：抑制关节加速度（平滑性）。排除 follower。"""
    asset: Articulation = env.scene[asset_cfg.name]
    active_ids = _get_joint_groups(env, asset)["active_all"]
    value = torch.sum(torch.square(asset.data.joint_acc[:, active_ids]), dim=1) * (~_is_freefall(env)).float()
    return _control_phase_reward(env, value)


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ τ_active²：能耗/力矩惩罚。排除 follower（其力矩由约束产生，非策略可控）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    active_ids = _get_joint_groups(env, asset)["active_all"]
    value = torch.sum(torch.square(asset.data.applied_torque[:, active_ids]), dim=1) * (~_is_freefall(env)).float()
    return _control_phase_reward(env, value)


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ ω_wheel²：抑制轮速。

    轮转可参与翻转，惩罚过重可能限制探索；不能由旧开环探针推断其必要速度
    或唯一可行性。本实现权重为待实验验证的 -1e-3。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_ids = _get_joint_groups(env, asset)["wheel"]
    value = torch.sum(torch.square(asset.data.joint_vel[:, wheel_ids]), dim=1) * (~_is_freefall(env)).float()
    return _control_phase_reward(env, value)


# ── 成功判据与日志 ──

def check_recovery_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = MIN_BASE_HEIGHT,
    joint_threshold: float = 0.5,
    vel_threshold: float = 0.5,
    ori_threshold: float = 0.3,
) -> torch.Tensor:
    """论文四判据（环境原点相对高度 / 关节偏差 / 关节速度 / 姿态）的逐环境布尔值。

    **与参考实现的差异**：关节相关的两个判据（偏差、速度）**只统计 8 个主动腿关节**。
    参考实现用 `asset.data.joint_pos` 全量——在本机器人上会把 8 个 follower 计入
    norm，使阈值 0.5 rad 的物理含义失效（follower 是 master 的镜像，其偏差与
    master 强相关，等于把同一误差计两次）。

    轮关节也排除：连续关节的绝对角无意义，其速度由 `vel_threshold` 单独看会
    过度严格（恢复结束瞬间轮子可能仍在滑转）。

    **阈值校准（勿随意收紧）**：
      - `ori_threshold=0.3`（≈17° 倾斜）。参考实现用 0.1，但那只容许 **5.7°**，
        倾斜 10° 即判失败（‖g−ideal‖=0.174）—— 对刚完成翻正的机器人过于严苛，
        且与 `support_state` 的 45° 判据严重不一致。0.3 仍能干净排除
        侧躺（1.414）与仰躺（2.0）。
      - `vel_threshold=0.5` rad/s。参考用 0.1，但恢复结束瞬间腿关节仍有残余速度；
        0.1 会把"已站稳但仍在微调"判为失败，低估真实成功率。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    leg_ids = _get_joint_groups(env, asset)["active_leg"]

    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    h_ok = height > height_threshold
    dev = asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]
    j_ok = torch.norm(dev, dim=1) < joint_threshold
    v_ok = asset.data.joint_vel[:, leg_ids].abs().max(dim=1).values < vel_threshold

    ideal = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    o_ok = torch.norm(asset.data.projected_gravity_b - ideal, dim=1) < ori_threshold

    return h_ok & j_ok & v_ok & o_ok


def recovery_success_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """仅在 episode 最后 1 s 生效的 0/1 成功指示（**纯日志项，权重 1e-6**）。

    只看末 1 s 是为了让 tensorboard 反映"最终是否站起来"，而不是中途的部分得分。
    这是 PLAN §6 的主验收指标（目标 ≥0.85，<0.3 为停训闸门）。
    """
    success = check_recovery_success(env, asset_cfg).float()
    _ensure_step_counter(env)
    last_second = (
        env._recovery_step_count >= (env.max_episode_length - int(1.0 / _env_dt(env)))
    ).float()
    return _control_phase_reward(env, success * last_second)
