"""
diagnostics.py — 世界模型开环误差与常速度基线，评测脚本和校验脚本共用，保证口径一致。

@author pony
@date 2026-10-26
@version v0.3.0
@last_modified 2026-10-26
@changelog
    - v0.3.0 (2026-10-26): 支持 start_mask（按起点筛选）与 extra_metrics（如轮心高度、俯仰误差）。
    - v0.2.0 (2026-10-26): step_fn 增加 context 参数，支持带 ξ 的世界模型。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

from typing import Callable

import torch

from .quat import projected_gravity
from .state import JVEL, LIN, NUM_LEG_JOINTS, POS, QUAT, JPOS


def _tilt(state):
    return torch.acos(torch.clamp(-projected_gravity(state[..., QUAT])[..., 2], -1.0, 1.0))


@torch.no_grad()
def open_loop_errors(states, actions, dones, step_fn: Callable, horizons=(1, 8, 32), chunk: int = 8192,
                     contexts=None, start_mask=None, extra_metrics: dict | None = None):
    """states/actions/dones: [T, N, ...]；step_fn(state, action, context) → next_state。

    contexts 可选 [T, N, C]；为 None 时传 None。起点 t 只在 t..t+k-1 都没有 done 时计入，
    start_mask [T, N] 可进一步限定起点（如"台阶边附近"）。extra_metrics: {名称: f(pred, target) → [M]}。
    返回物理单位的平均误差。
    """
    T = states.shape[0]
    out = {}
    for k in horizons:
        if k >= T:
            continue
        alive = torch.ones(T - k, states.shape[1], dtype=torch.bool, device=states.device)
        for j in range(k):
            alive &= ~dones[j:T - k + j]
        if start_mask is not None:
            alive &= start_mask[: T - k]
        idx = alive.nonzero(as_tuple=False)  # [M, 2] = (t, env)
        if idx.numel() == 0:
            continue
        sums = {"pos_m": 0.0, "lin_vel_mps": 0.0, "tilt_rad": 0.0, "leg_pos_rad": 0.0, "wheel_vel_radps": 0.0}
        sums.update({name: 0.0 for name in (extra_metrics or {})})
        for part in idx.split(chunk):
            t, e = part[:, 0], part[:, 1]
            pred = states[t, e]
            for j in range(k):
                ctx = contexts[t + j, e] if contexts is not None else None
                pred = step_fn(pred, actions[t + j, e], ctx)
            target = states[t + k, e]
            sums["pos_m"] += (pred[:, POS] - target[:, POS]).norm(dim=-1).sum().item()
            sums["lin_vel_mps"] += (pred[:, LIN] - target[:, LIN]).norm(dim=-1).sum().item()
            sums["tilt_rad"] += (_tilt(pred) - _tilt(target)).abs().sum().item()
            sums["leg_pos_rad"] += (pred[:, JPOS][:, :NUM_LEG_JOINTS] - target[:, JPOS][:, :NUM_LEG_JOINTS]).abs().mean(-1).sum().item()
            sums["wheel_vel_radps"] += (pred[:, JVEL][:, NUM_LEG_JOINTS:] - target[:, JVEL][:, NUM_LEG_JOINTS:]).abs().mean(-1).sum().item()
            for name, fn in (extra_metrics or {}).items():
                sums[name] += fn(pred, target).sum().item()
        m = idx.shape[0]
        out[f"k{k}"] = {"samples": m, **{name: v / m for name, v in sums.items()}}
    return out
