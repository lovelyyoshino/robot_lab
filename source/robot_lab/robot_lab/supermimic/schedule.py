#!/usr/bin/env python3
"""
schedule.py — 越障续训用的纯函数：线性爬坡、地形课程等级的保存/恢复。

@author pony
@date 2026-10-29
@version v1.1.0
@last_modified 2026-10-30
@changelog
    - v1.1.0 (2026-10-30): next_terrain_levels（课程到顶随机回落）、saturation_reached（课程饱和判据）
    - v1.0.0 (2026-10-29): 初始创建（d3 第 8000 轮横向退化后：课程挂钩横向惩罚、续训保留课程进度）
"""

from __future__ import annotations

import torch


def linear_ramp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """x≤x0 取 y0，x≥x1 取 y1，中间线性插值；x1≤x0 时直接跳到 y1。"""
    if x1 <= x0:
        return y1 if x >= x0 else y0
    t = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
    return y0 + t * (y1 - y0)


def restore_terrain_levels(saved: torch.Tensor, num_envs: int, max_level: int,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """把存档里的 terrain_levels 映射到当前环境数。

    环境数相同则逐个还原（地形类型列按 env id 固定，等级一一对应）；不同则从存档分布里有放回抽样，
    保持课程进度的分布而不是从第 0 级重新爬。结果裁到 [0, max_level-1]，防止换了地形行数后越界。"""
    saved = saved.flatten().long().cpu()
    if saved.numel() == num_envs:
        levels = saved.clone()
    else:
        idx = torch.randint(0, saved.numel(), (num_envs,), generator=generator)
        levels = saved[idx]
    return levels.clamp(0, max_level - 1)


def next_terrain_levels(levels: torch.Tensor, passed: torch.Tensor, failed: torch.Tensor, max_level: int,
                        top_resample: bool = True) -> torch.Tensor:
    """课程升降一级；top_resample 时通关最高级的环境随机落到 [0, max_level)，与 Isaac Lab terrain_levels 默认一致。

    scratch_40k 实测：夹在最高级不动时，8000 轮后 89% 环境都在 16 cm，矮台阶不再练、回放全是同一种片段，
    约 10500 轮崩成原地不动。随机回落让各级一直有数据。"""
    raw = levels + passed.long() - failed.long()
    if top_resample:
        return torch.where(raw >= max_level, torch.randint_like(raw, max_level), raw.clamp(min=0))
    return raw.clamp(0, max_level - 1)


def saturation_reached(ever_top: torch.Tensor, levels: torch.Tensor, max_level: int,
                       threshold: float) -> tuple[torch.Tensor, bool]:
    """累计"曾到过最高级"的环境；占比 ≥ threshold 视为课程饱和。

    顶层随机回落后瞬时的最高级占比稳定在 1/行数 附近，不能再当进度信号，所以用累计值。"""
    ever = ever_top | (levels >= max_level - 1)
    return ever, bool(ever.float().mean().item() >= threshold)
