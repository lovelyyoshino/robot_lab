"""
stairs_acceptance.py — 楼梯验收几何与倾角门，评测、诊断、play、accept.py 共用。

独立成文件而不放进 vision_preview.stairs_profile：vision_contract 对 vision_preview.py 做 sha256，
改动那里会让所有已训练的 Preview checkpoint 合同失配。倾角门只属于验收，不影响训练目标。
本文件只用标准库：accept.py 在不启动 Isaac 的进程里按文件路径加载它。

@author pony
@date 2026-10-27
@version v1.1.0
@last_modified 2026-11-05
@changelog
    - v1.1.0 (2026-11-05): 越障验收楼梯档位（8/12/16/17.5 cm × 踏面 0.40/0.28 m）、坡度相对俯仰门、按踢面取翻车门。
    - v1.0.0 (2026-10-27): 初始创建；16 cm 途中倾角门放宽到 45°。
"""

from __future__ import annotations

import math

# 31.8° 即 −g_z ≥ 0.85，是全部地形共用的直立门。
# 16 cm 放宽到 45°（用户 2026-10-27 决定）：轴距 0.694 m 跨 1–2 级时静态俯仰已有 13–28°，
# 再叠加起抬瞬态，31.8° 在 16 cm 上过严；45° 仍远低于训练翻车终止 68.75°。
# 只放宽途中跌倒判据；终点稳定仍要求 31.8° 内（顶端平台是平地）。
CLIMB_TILT_DEG = (31.8, 31.8, 45.0)
# 踢面达到该高度即按 16 cm 档放宽途中翻车门（12 cm 与 16 cm 之间取中）。
RELAXED_CLIMB_RISER_M = 0.14

# 楼梯验收档位：踢面 × 踏面；级数沿用三档课程（8→2、12→3、16→4），17.5 cm 取 4 级。
ACCEPTANCE_STAIR_RISERS_M = (0.08, 0.12, 0.16, 0.175)
ACCEPTANCE_STAIR_TREADS_M = (0.40, 0.28)
ACCEPTANCE_STAIR_STEP_COUNTS = {0.08: 2, 0.12: 3, 0.16: 4, 0.175: 4}
# 俯仰须跟随坡度：P95(|pitch|) ≤ 坡度 − 5°，单回合最大 |pitch| ≤ 坡度 + 3°。
STAIR_PITCH_P95_MARGIN_DEG = 5.0
STAIR_PITCH_MAX_MARGIN_DEG = 3.0


def climb_tilt_deg(profile: dict, override: float | None = None) -> float:
    """profile 为 vision_preview.stairs_profile 的返回值（stage 为 1–3）。"""
    value = CLIMB_TILT_DEG[profile["stage"] - 1] if override is None else override
    if not 0 < value < 90:
        raise ValueError(f"途中倾角门必须位于 (0, 90)°：{value}")
    return float(value)


def min_climb_cos(profile: dict, override: float | None = None) -> float:
    return math.cos(math.radians(climb_tilt_deg(profile, override)))


def climb_tilt_for_riser(riser_m: float) -> float:
    """按踢面高度取途中翻车门，与三档课程一致（8/12 cm 31.8°，≥14 cm 45°）。"""
    if not math.isfinite(riser_m) or riser_m <= 0:
        raise ValueError(f"踢面高度必须为有限正数：{riser_m}")
    return CLIMB_TILT_DEG[2] if riser_m >= RELAXED_CLIMB_RISER_M else CLIMB_TILT_DEG[0]


def stair_slope_deg(riser_m: float, tread_m: float) -> float:
    """楼梯坡度 atan(踢面/踏面)。"""
    if not (math.isfinite(riser_m) and math.isfinite(tread_m) and riser_m > 0 and tread_m > 0):
        raise ValueError(f"踢面和踏面必须为有限正数：{riser_m}, {tread_m}")
    return math.degrees(math.atan2(riser_m, tread_m))


def stairs_pitch_gates(riser_m: float, tread_m: float) -> dict:
    """坡度相对俯仰门；坡度不超过 P95 余量时门为负、无法满足，直接拒绝这种档位。"""
    slope = stair_slope_deg(riser_m, tread_m)
    if slope <= STAIR_PITCH_P95_MARGIN_DEG:
        raise ValueError(f"坡度 {slope:.1f}° 不大于 {STAIR_PITCH_P95_MARGIN_DEG}°，坡度相对俯仰门无意义")
    return {"slope_deg": slope,
            "pitch_abs_p95_deg": slope - STAIR_PITCH_P95_MARGIN_DEG,
            "pitch_abs_max_deg": slope + STAIR_PITCH_MAX_MARGIN_DEG}
