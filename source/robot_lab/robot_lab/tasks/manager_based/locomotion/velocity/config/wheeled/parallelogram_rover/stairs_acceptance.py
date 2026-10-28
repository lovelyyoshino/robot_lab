"""
stairs_acceptance.py — 楼梯验收的途中倾角门（8/12/16 cm 三档），评测、诊断、play 共用。

独立成文件而不放进 vision_preview.stairs_profile：vision_contract 对 vision_preview.py 做 sha256，
改动那里会让所有已训练的 Preview checkpoint 合同失配。倾角门只属于验收，不影响训练目标。

@author pony
@date 2026-10-27
@version v1.0.0
@last_modified 2026-10-27
@changelog
    - v1.0.0 (2026-10-27): 初始创建；16 cm 途中倾角门放宽到 45°。
"""

from __future__ import annotations

import math

# 31.8° 即 −g_z ≥ 0.85，是全部地形共用的直立门。
# 16 cm 放宽到 45°（用户 2026-10-27 决定）：轴距 0.694 m 跨 1–2 级时静态俯仰已有 13–28°，
# 再叠加起抬瞬态，31.8° 在 16 cm 上过严；45° 仍远低于训练翻车终止 68.75°。
# 只放宽途中跌倒判据；终点稳定仍要求 31.8° 内（顶端平台是平地）。
CLIMB_TILT_DEG = (31.8, 31.8, 45.0)


def climb_tilt_deg(profile: dict, override: float | None = None) -> float:
    """profile 为 vision_preview.stairs_profile 的返回值（stage 为 1–3）。"""
    value = CLIMB_TILT_DEG[profile["stage"] - 1] if override is None else override
    if not 0 < value < 90:
        raise ValueError(f"途中倾角门必须位于 (0, 90)°：{value}")
    return float(value)


def min_climb_cos(profile: dict, override: float | None = None) -> float:
    return math.cos(math.radians(climb_tilt_deg(profile, override)))
