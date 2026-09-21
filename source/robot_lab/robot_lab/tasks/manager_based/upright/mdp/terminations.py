# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 新增带起立宽限期的前塌终止，允许从四轮姿态开始训练。
#   - v0.1.0 (2026-08-25): 横滚终止项；不可用内置 bad_orientation（它测总倾角会误杀直立）。

"""直立任务的终止项。

**为何不能用内置 `mdp.bad_orientation`**（重要）：

其实现（`isaaclab/envs/mdp/terminations.py:59`）是

    torch.acos(-projected_gravity_b[:, 2]).abs() > limit_angle

即测**机体 z 轴与世界竖直方向的总夹角**。对四足行驶任务这没问题（机体应始终近水平），
但直立任务的目标姿态本身就是 pitch 90°，此时 `g_z ≈ 0` → `acos(0) = π/2 ≈ 1.57`。
任何 `limit_angle < 1.57` 都会**在机器人刚站起来的瞬间终止 episode**，
使直立永远无法被训练。

本任务需要的是**只测横滚、不管俯仰**的判据 —— 因为：
  - 俯仰是任务目标（75°-105° 可保持区），必须允许
  - 横滚是不可恢复失败（两后轮间距 0.56 m，arm 无侧摆自由度）

故用机体系重力的 y 分量：`|g_y|` 直接反映横滚，与 pitch 解耦。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .rewards import HOLDABLE_PITCH_MAX

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


BACKWARD_GZ_LIMIT = -math.cos(HOLDABLE_PITCH_MAX)
"""后翻终止阈值，**由 `HOLDABLE_PITCH_MAX` 推导，不硬编码**。

= `-cos(1.83)` = 0.25631，对应 pitch 104.85°（即可保持区上界本身）。

⚠️ 初版硬编码 0.26，对应 105.07° —— 比常量宽 **0.22°**。
虽小，但同一物理边界被写成两套参数化（rad 角度 / g_z 分量）后，
改一处不改另一处就会静默产生缝隙；而 §4 课程会把 `pitch_range`
逐步扩到 `hard_limits`(=1.83)，届时该缝隙正好落在指令上界处被激活。
本任务已两次被"阈值留缝"坑到（见本文件 `fell_over` docstring），故此处取严。
"""


def excessive_roll(
    env: ManagerBasedRLEnv,
    max_gy: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """横滚超限即终止。**与 pitch 解耦**。

    判据 `|g_y| > max_gy`：机体系重力的 y 分量。
      - 完全直立或任意 pitch 且无横滚 → g_y = 0
      - 完全侧躺 → |g_y| = 1
      - `max_gy = 0.7` ≈ 横滚 44°（arcsin(0.7)），已超出 0.56 m 支撑宽度可挽回范围

    为何用 g_y 而非 roll 角：前者无需反三角函数、在小角度即有梯度，
    且与奖励项 `roll_penalty`（用 g_y²）同源，判据一致。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 1].abs() > max_gy


def fell_over(
    env: ManagerBasedRLEnv,
    max_gz_forward: float = 0.85,
    max_gz_backward: float = BACKWARD_GZ_LIMIT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体前塌或后翻即终止。**判据前后不对称**。

    ⚠️ **初版用对称判据 `|g_z| > 0.85`，被策略钻了空子**（probe16 实测）：

    `|g_z| > 0.85` 对应 pitch < 32°（前塌）**或 pitch > 148°（后翻）**。
    可保持区上界是 105°，于是 **105°→148° 的 43° 区间成了不终止的死区**。
    实测零动作 rollout：机器人后仰到 122°-133°（最大 147.4°，正好卡在边界），
    base_link 贴地、`undesired_contacts` 触发率 **60.3%**，但 episode 从不终止
    —— 它躺在背上把 200 步跑满。这与"早终止让样本预算集中在可挽回状态"的
    设计意图正好相反：躺背状态没有 recovery 策略是回不来的。

    修正为两个独立阈值（g_z 从 -1 单调增到 +1 对应 pitch 0°→180°）：

    | 姿态 | pitch | g_z | 判定 |
    |---|---|---|---|
    | 四轮平放 | 0° | -1.00 | 终止（前塌）|
    | 前塌边界 | 32° | -0.85 | 终止（前塌）|
    | 可保持下界 | 75° | -0.26 | 保留 |
    | 最优 | 90° | 0.00 | 保留 |
    | **可保持上界=后翻边界** | **105°** | **+0.26** | 边界（略超即终止）|
    | 上界外 | 115° | +0.42 | 终止（后翻）|
    | 躺背 | 133° | +0.68 | 终止（后翻）|

    **`max_gz_backward` 的定值经过两轮实测修正**：

    - 初版 0.85（对称）→ 105°-148° 死区，策略躺背跑满 episode（probe16）
    - 二版 0.50（=120°，给上界留 15° 余量）→ **死区只是变小没消失**：
      probe19 实测接触样本 `pitch mean=107.7°`、**70% 落在 105°-120° 余量内**、
      最大值 120.0 恰好顶在阈值上，且「接触 ∩ 终止」= **0.00%**
      —— 即它精确地待在"贴地但不终止"的缝里。
    - 三版 0.26（=105°）：**余量归零**，与 probe2 实测的可保持上界严格一致。

    为何不该留余量：`HOLDABLE_PITCH_MAX=105°` 是 probe2 测的**姿态可保持上界**，
    它只说明"该角度下轮力矩够用"，**没有**说明该角度下机体离地。
    我按"平衡需要修正余量"的直觉加的 15°，在物理上没有依据，
    结果只是把作弊区间从 43° 缩到 15°。策略两次都精确找到了缝的边界。

    ⚠️ 若课程启用 `quad_prob > 0`（从四轮起步训起立），前塌判据须放宽或关闭，
    否则四轮起点会在第一步就被判终止。后翻判据不受影响。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2]
    collapsed_forward = g_z < -max_gz_forward
    tipped_backward = g_z > max_gz_backward
    return collapsed_forward | tipped_backward


def fell_over_after_grace(
    env: ManagerBasedRLEnv,
    max_gz_forward: float = 0.85,
    max_gz_backward: float = BACKWARD_GZ_LIMIT,
    forward_grace_steps: int = 150,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """起立宽限期内忽略前塌，但始终拦截后翻。

    四轮起点本来就满足 ``g_z < -0.85``，直接使用 ``fell_over`` 会在首步终止。
    宽限期只豁免前塌；后翻和横滚仍由安全终止立即处理，避免无意义 rollout。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g_z = asset.data.projected_gravity_b[:, 2]
    grace_finished = env.episode_length_buf >= forward_grace_steps
    collapsed_forward = (g_z < -max_gz_forward) & grace_finished
    tipped_backward = g_z > max_gz_backward
    return collapsed_forward | tipped_backward
