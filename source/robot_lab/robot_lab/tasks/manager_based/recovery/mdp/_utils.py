# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): settled_ground 的 ED/释放门使用接管时钟，保持原 drop 协议不变。
#   - v0.1.0 (2026-08-25): 移植 EDRS 的 ED/CW 双重时间 shaping 与关节分组，适配平行四边形 12 主动 + 8 从动布局。

"""恢复任务的共享基础设施：ED/CW 整形系数、步计数器、关节分组、控制步长解析。

参考实现：`Kitjesen/recovery`（Deng et al., arXiv:2506.05516）的 `thunder_recovery/mdp/_utils.py`。

与参考实现的关键差异（平行四边形特有）：
  1. Thunder 是 12 腿关节 + 4 轮，本机器人是 **8 主动腿关节(arm/qudong) + 4 轮 + 8 被动 follower**；
     因此关节分组必须区分三类，而非参考实现的两类。
  2. 轮关节正则 `.*foot.*` → `.*wheel_joint$`（命名差异）。
  3. 被动 follower（zhuanxiang/lower_arm）由 mimic 约束驱动，**不得计入任何主动关节统计**
     （否则奖励/成功判据会被被动量污染）。

模块私有（前缀下划线）；对外导出集中在 `mdp/__init__.py`。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_MIMIC_FOLLOWER_JOINT_NAMES,
    PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── 常量 ──

FREEFALL_STEPS = 100
"""零力矩自由落体窗口时长（控制步）。50 Hz 下为 2 s，与 EDRS 论文 §III-A 一致。"""

RECOVERY_STEPS_PER_ITER = 48
"""每个 PPO rollout 的 env.step() 次数。

**必须**与 `agents/rsl_rl_ppo_cfg.py` 的 `num_steps_per_env` 保持一致——CW 衰减的时间基准
依赖该比值。Isaac Lab 不向 reward manager 暴露 rollout 长度，故在此镜像该常量。
不一致的后果：CW 衰减速度与预期不符，行为惩罚过早或过晚失效。
"""


# ── 每环境状态 ──

def _ensure_step_counter(env: ManagerBasedRLEnv) -> None:
    """惰性创建每环境步计数器（int64 精确计数，避免 float 累加误差）。"""
    if not hasattr(env, "_recovery_step_count"):
        env._recovery_step_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)


def _advance_step_counter(env: ManagerBasedRLEnv) -> None:
    """每次 env.step() 精确自增一次。

    由零权重的 `recovery_step_counter` 奖励项驱动（reward manager 保证每步调用一次）。
    `reset_with_freefall` 会把计数器置 -1，使复位后首次自增回到 0 —— 于是奖励项读到的
    计数范围恒为 {0, 1, ..., T-1}，`_is_freefall`（< FREEFALL_STEPS）正好覆盖 100 步，
    且 ED 在最后一步恰好达到 1.0。

    去重依据 `common_step_counter`：同一物理步内多次调用只自增一次。
    """
    _ensure_step_counter(env)
    if hasattr(env, "_recovery_control_steps"):
        # settled_ground 在 termination 阶段先记录已完成的真实受控步，不能再二次自增。
        env._recovery_step_count.copy_(env._recovery_control_steps - 1)
        return
    if not hasattr(env, "_recovery_ed_last_step"):
        env._recovery_ed_last_step = -1

    current_step = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    if current_step != env._recovery_ed_last_step:
        env._recovery_step_count += 1
        env._recovery_step_count.clamp_(max=int(env.max_episode_length) - 1)
        env._recovery_ed_last_step = current_step


def _is_freefall(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回仍处于自由落体阶段的环境掩码（step_count < FREEFALL_STEPS）。"""
    _ensure_step_counter(env)
    if hasattr(env, "_recovery_control_active"):
        return ~env._recovery_control_active
    return env._recovery_step_count < FREEFALL_STEPS


def _control_phase_reward(env: ManagerBasedRLEnv, value: torch.Tensor) -> torch.Tensor:
    """settled 准备样本没有被策略控制；用 where 归零，避免失败样本出现 0×NaN。"""
    if hasattr(env, "_recovery_control_active"):
        return torch.where(env._recovery_control_active, value, 0.0)
    return value


# ── 关节分组（平行四边形三分法）──

def _get_joint_groups(env: ManagerBasedRLEnv, asset: Articulation) -> dict[str, torch.Tensor]:
    """解析并缓存四组关节索引。

    与参考实现的两分法（leg/wheel）不同，平行四边形必须三分：

      - ``active_leg``：8 个主动腿关节（arm 抬升 + qudong 转向）
      - ``wheel``：4 个轮关节
      - ``active_all``：12 个主动关节（= active_leg + wheel，顺序为 12-D 动作合同顺序）
      - ``follower``：8 个被动 mimic 关节（zhuanxiang/lower_arm）

    **为何必须排除 follower**：它们由 PhysX mimic 硬约束驱动（`zhuanxiang = -arm`,
    `lower_arm = +arm`），其位置/速度是 master 的镜像。若计入关节速度惩罚或成功判据，
    等价于把同一物理量重复计数，并使阈值失去物理意义。

    索引按名称查询（`find_joints(preserve_order=True)`），不依赖 URDF 内部顺序。
    """
    cache_name = "_recovery_joint_groups"
    cached = getattr(env, cache_name, None)
    if cached is not None:
        return cached

    def _resolve(names: tuple[str, ...], label: str) -> torch.Tensor:
        ids, resolved = asset.find_joints(list(names), preserve_order=True)
        if len(ids) != len(names):
            raise RuntimeError(
                f"recovery: 关节分组 '{label}' 解析失败，期望 {len(names)} 个、实得 {len(ids)} 个。"
                f"\n  期望: {list(names)}\n  实得: {resolved}"
                f"\n  资产关节: {asset.joint_names}"
            )
        return torch.tensor(ids, dtype=torch.long, device=asset.data.joint_pos.device)

    active_leg_names = (*PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES, *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES)
    groups = {
        "active_leg": _resolve(active_leg_names, "active_leg(arm+qudong)"),
        "wheel": _resolve(PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES, "wheel"),
        "active_all": _resolve(PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES, "active_all(12-D 合同)"),
        "follower": _resolve(PARALLELOGRAM_ROVER_MIMIC_FOLLOWER_JOINT_NAMES, "follower(被动)"),
    }
    setattr(env, cache_name, groups)
    return groups


# ── 时间解析 ──

def _env_dt(env: ManagerBasedRLEnv) -> float:
    """控制步时长（秒）。优先用 env.step_dt，回退到 sim.dt × decimation。"""
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    if hasattr(env, "cfg") and hasattr(env.cfg, "sim") and hasattr(env.cfg, "decimation"):
        return float(env.cfg.sim.dt) * float(env.cfg.decimation)
    return 1.0 / 50.0


# ── ED / CW 整形系数（EDRS 论文 Eq.1 / Eq.3）──

def _get_ed(env: ManagerBasedRLEnv, k: int = 3) -> torch.Tensor:
    """Episode-based Dynamic 系数（论文 Eq.1，归一化形式）。

    ``ED(t) = (t / T)^k``，取值 [0, 1]，默认 k=3。

    作用：episode 早期任务奖励几乎为 0（放开姿态探索，允许大幅翻滚试错），
    后期趋近 1（收敛到精确站姿）。这是 EDRS 解决"稀疏终态奖励导致探索不足"的核心机制。

    归一化到 [0,1] 使论文 Table I 的权重可直接套用，且 `ED` / `1-ED` 两个门自解释。
    """
    _ensure_step_counter(env)
    dt = _env_dt(env)
    step_count = getattr(env, "_recovery_control_steps", env._recovery_step_count)
    t_sec = step_count.float() * dt
    total_sec = float(env.max_episode_length) * dt
    return (t_sec / total_sec).clamp_(0.0, 1.0) ** k


def _get_cw(env: ManagerBasedRLEnv, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight（论文 Eq.3）：``CW(i) = beta * decay^i``。

    i 为训练迭代数，用 ``common_step_counter / RECOVERY_STEPS_PER_ITER`` 近似。
    beta=0.3, decay=0.968 时：CW 在第 ~35 迭代降到 0.1，第 ~100 迭代降到 ~0.01。

    作用：行为惩罚（碰撞、动作抖动）在训练早期约束姿态，数十迭代后逐步让位于
    任务奖励与 ED 整形，避免早期惩罚把策略锁在"不敢动"的局部最优。
    """
    if hasattr(env, "common_step_counter"):
        iteration = env.common_step_counter / RECOVERY_STEPS_PER_ITER
    else:
        iteration = 0
    return beta * (decay**iteration)
