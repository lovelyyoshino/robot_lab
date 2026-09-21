# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.1.0
# @last_modified 2026-08-25
# @changelog
#   - v0.1.0 (2026-08-25): pitch 指令项；采样区间限制在 probe2 实测的可保持区。

"""直立任务的俯仰角指令项。

Isaac Lab 内置命令项只有 velocity / pose / pose_2d / null，没有通用标量命令，
故为 `pitch_cmd` 写一个最小实现（4 个抽象方法）。

**采样区间由 probe2 实测决定，不可随意放宽**：

    可保持区   75°-105°  = 1.31-1.83 rad   ← 只在此区间采样
    不可保持区 20°-60°   = 0.35-1.05 rad   ← 机体必塌回四轮，不作为目标
    最优点     90°       = 1.57 rad        ← 所需轮力矩最小(1.55 N·m)

若把指令采到不可保持区，策略会收到"去那里"的信号，但物理上到了就会塌——
等于强迫它学一个不可能满足的目标，训练必然不收敛。这是本项目
"参数必须由实测而非直觉决定"的又一例。

同时提供**课程接口**：`set_pitch_range()` 允许 curriculum term 在训练中
逐步扩展采样范围（从最优点邻域扩到可保持区全宽）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PitchCommand(CommandTerm):
    """标量俯仰角指令，按 [lo, hi] 均匀采样。

    `command` 形状 (num_envs, 1)，单位 rad。
    """

    cfg: PitchCommandCfg

    def __init__(self, cfg: PitchCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._pitch = torch.zeros(self.num_envs, 1, device=self.device)
        # 当前采样区间（课程可修改；初值取 cfg）
        self._lo = float(cfg.pitch_range[0])
        self._hi = float(cfg.pitch_range[1])
        # 指标：跟踪误差（供 curriculum 判断是否该扩范围）
        self.metrics["pitch_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"PitchCommand:\n"
            f"\tnum_envs: {self.num_envs}\n"
            f"\tcurrent range: [{self._lo:.3f}, {self._hi:.3f}] rad "
            f"([{self._lo * 57.2958:.1f}, {self._hi * 57.2958:.1f}] deg)\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """当前 pitch 指令，形状 (num_envs, 1)，单位 rad。"""
        return self._pitch

    @property
    def pitch_range(self) -> tuple[float, float]:
        """当前采样区间（课程可读）。"""
        return (self._lo, self._hi)

    def set_pitch_range(self, lo: float, hi: float) -> None:
        """课程接口：修改采样区间。

        会被 clamp 到 cfg 的 `hard_limits` 内 —— 防止课程把范围扩到
        物理上不可保持的区间（那会让训练目标变成不可满足）。
        """
        hard_lo, hard_hi = self.cfg.hard_limits
        self._lo = max(float(lo), float(hard_lo))
        self._hi = min(float(hi), float(hard_hi))
        if self._lo > self._hi:  # 防御：区间被压反
            self._lo = self._hi

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        r = torch.rand(n, 1, device=self.device)
        self._pitch[env_ids] = self._lo + r * (self._hi - self._lo)

    def _update_command(self):
        """本项无需逐步更新（指令在重采样间保持恒定）。"""
        pass

    def _update_metrics(self):
        """记录当前俯仰角与指令的绝对误差，供 curriculum 使用。"""
        asset = self._env.scene[self.cfg.asset_name]
        g = asset.data.projected_gravity_b
        gx, gz = g[:, 0], g[:, 2]
        pitch = torch.atan2(-gx, -gz)
        # 与 rewards._pitch_from_gravity 一致的退化处理
        degenerate = torch.sqrt(gx * gx + gz * gz) < 1e-3
        pitch = torch.where(degenerate, torch.full_like(pitch, 1.57), pitch)
        self.metrics["pitch_error"] = (pitch - self._pitch[:, 0]).abs()


@configclass
class PitchCommandCfg(CommandTermCfg):
    """`PitchCommand` 的配置。"""

    class_type: type = PitchCommand

    asset_name: str = "robot"
    """读取姿态用的资产名。"""

    pitch_range: tuple[float, float] = (1.45, 1.70)
    """初始采样区间（rad）。默认 1.45-1.70（≈83°-97°）—— 最优点 90° 的邻域。

    课程会逐步扩到 `hard_limits`。从窄区间起步的理由：先在最省力的
    工作点（90°，1.55 N·m）学会平衡，再向区间边缘扩展。
    """

    hard_limits: tuple[float, float] = (1.31, 1.83)
    """采样区间的硬上下界（rad）= probe2 实测的可保持区 75°-105°。

    **课程不得越过此界**。越界即把目标设在物理上不可保持的姿态，
    策略无论如何都无法满足。
    """
