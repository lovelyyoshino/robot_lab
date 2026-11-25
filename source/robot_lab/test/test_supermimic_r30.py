"""
test_supermimic_r30.py — 第三十轮：去掉短踏面列 + 翻车代价 −10000 / 软存活温度 0.02（保留第二十九轮卡住加码）。

第二十九轮卡住加码让贴立面每步多付到 −20，而翻车只记一次 −20×dt、之后价值为 0，策略学成在第一级立面前主动后翻。
这里把"翻车必须比一直停着更差"写成对 launcher 配方的账：在想象地平线（32 步）末尾翻车的折扣代价要大于
停住（停滞 + 线性停滞 + 卡住加码，全开）的无穷折扣代价；同时软存活项在正常爬楼倾角下几乎不计。

@author pony
@date 2026-11-23
@version v1.0.0
@last_modified 2026-11-23

@changelog
    - v1.0.0 (2026-11-23): 初始创建
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

import test_supermimic_obstacle as T  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUN_TRAIN = REPO / "scripts" / "supermimic" / "run_train.sh"
DT, GAMMA, HORIZON = 0.02, 0.99, 32  # CONTROL_DT、--gamma、想象地平线（步）


def _args() -> str:
    """launcher 的 TRAIN_ARGS（去掉注释行）。"""
    text = RUN_TRAIN.read_text()
    body = text[text.index("TRAIN_ARGS=("):]
    body = body[:body.index("\n)\n")]
    return "\n".join(l for l in body.split("\n") if not l.strip().startswith("#"))


def _flag(args: str, name: str, n: int = 1) -> list[float]:
    m = re.search(rf"--{name}((?:\s+-?[\d.]+){{{n}}})(?=\s)", args)
    assert m, name
    return [float(v) for v in m.group(1).split()]


def test_run_train_r30_recipe():
    args = _args()
    assert "--terminated -10000.0 --tilt_limit 0.8 --tilt_temperature 0.02\n" in args
    assert "--stairs_short_frac 0.0\n" in args and "--stairs_short_frac 0.05" not in args
    assert "--terminated -20.0" not in args
    # 第二十九轮卡住加码与第二十八轮 A–D 保留
    for flag in ("--stall_hold -20.0 --stall_hold_grace_s 1.0 --stall_hold_ramp_s 2.0 --stall_hold_gain_m 0.01\n",
                 "--stall -5.0 --stall_speed_ratio 0.25\n", "--stall_linear -3.0 --stall_linear_cap 1.0\n",
                 "--curriculum_goal_lane_termination 0\n", "--course -20.0\n", "--stairs_tread_rule 0.12 0.15 0.36\n"):
        assert flag in args, flag
    assert "# @version v2.21.0" in RUN_TRAIN.read_text()


def _stall_forever(args: str) -> float:
    """停住且卡住加码全开时每步真实代价的无穷折扣和（正数）。"""
    per_step = (abs(_flag(args, "stall")[0]) + abs(_flag(args, "stall_linear")[0]) * _flag(args, "stall_linear_cap")[0]
                + abs(_flag(args, "stall_hold")[0])) * DT
    return per_step / (1.0 - GAMMA)


def _flip_at_horizon(terminated: float) -> float:
    return abs(terminated) * DT * GAMMA ** HORIZON


def test_flip_costs_more_than_holding_still_forever():
    args = _args()
    stall = _stall_forever(args)
    assert stall == pytest.approx(56.0)
    assert _flip_at_horizon(_flag(args, "terminated")[0]) > 2.0 * stall
    # 第二十九轮的 −20：翻车约 0.29，比停住便宜两个量级 → 学成主动后翻
    assert _flip_at_horizon(-20.0) < 0.01 * stall


@pytest.mark.parametrize("value", [-7.0, -16.0, -35.0])
def test_imagined_return_prefers_survival_near_risers(value):
    """想象回报 ≈ terminated·dt·(1 − s) + γ·s·V：对存活 s 的导数 = |terminated|·dt + γ·V，须为正。

    V 取第二十九轮探针里立面前的价值（翻车回合第一级前 −7、回合最低 −16，到终点回合最低 −34）。"""
    terminated = _flag(_args(), "terminated")[0]
    assert abs(terminated) * DT + GAMMA * value > 0
    assert abs(-20.0) * DT + GAMMA * value < 0  # 第二十九轮：越接近翻倒越划算


@pytest.mark.parametrize("profile", [T.SINGLE, T.STAIRS])
def test_r30_soft_termination_only_bites_near_the_limit(profile):
    """只改翻车两项（其余用默认奖励），正常爬楼倾角每步几乎不计，接近 46° 限位时每步代价远大于停住。"""
    args = _args()
    task = T._task(profile, terminated=_flag(args, "terminated")[0], tilt_temperature=_flag(args, "tilt_temperature")[0])

    def soft(deg):
        return T._terms(task, T._state(0.3, pitch=math.radians(deg)))["terminated"].item()

    for deg in (0.0, 15.0, 28.0):
        assert soft(deg) > -0.1, deg
    assert soft(40.0) < -40.0
    assert soft(55.0) < -9900.0


def test_short_stairs_column_is_off_by_default_and_gated():
    train = (REPO / "scripts" / "supermimic" / "train_terrain.py").read_text()
    assert "if args_cli.stairs_short_frac > 0:" in train
    ot = T.ot
    assert not any(ot.column_explore(list(ot.MIX), list(ot.MIX.values()), ot.NUM_COLS))
