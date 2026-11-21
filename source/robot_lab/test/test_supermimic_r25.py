"""
test_supermimic_r25.py — 第二十五轮：翻车代价 −10000 + 软存活温度 0.02（只在接近倾倒限位时计入，翻车比驻车更差）。

@author pony
@date 2026-11-19
@version v1.0.0
@last_modified 2026-11-19

@changelog
    - v1.0.0 (2026-11-19): 初始创建
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

import test_supermimic_obstacle as T  # noqa: E402

R25 = dict(stall=0.0, stall_linear=-20.0, stall_linear_cap=5.0, progress=12.0, prelift_front=10.0,
           terminated=-10000.0, tilt_temperature=0.02)
LIMIT = 0.8  # rad，与 env 倾角终止一致


def _soft_term(task, pitch_deg):
    return T._terms(task, T._state(0.3, pitch=math.radians(pitch_deg)))["terminated"].item()


def test_tilt_temperature_must_be_positive():
    with pytest.raises(ValueError, match="tilt_temperature"):
        T._task(tilt_temperature=0.0)


def test_default_soft_termination_is_unchanged():
    old = T._task()
    assert old.cfg.terminated == -20.0 and old.cfg.tilt_temperature == 0.05
    s = T._state(0.3, pitch=math.radians(30.0))
    expected = -20.0 * (1.0 - torch.sigmoid(torch.tensor(
        (math.cos(math.radians(30.0)) - math.cos(LIMIT)) / (0.05 * math.sin(LIMIT))))).item()
    assert T._terms(old, s)["terminated"].item() == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("profile", [T.SINGLE, T.STAIRS])
def test_r25_soft_termination_spares_climbing_tilts_and_bites_near_the_limit(profile):
    task = T._task(profile, **R25)
    # 成功爬楼最大倾角 28°：每步记入的翻车代价可忽略（< 0.1，相对驻车每步约 −20）
    for deg in (0.0, 15.0, 28.0):
        assert _soft_term(task, deg) > -0.1, deg
    # 接近 46° 限位：每步代价比停住每步（约 −20）重一个量级以上；越过限位按一次翻车计
    assert _soft_term(task, 40.0) < -40.0
    assert _soft_term(task, math.degrees(LIMIT)) == pytest.approx(-5000.0, rel=1e-3)
    assert _soft_term(task, 55.0) < -9900.0
    # 旧温度 0.05 下同样 −10000 会在 28° 就每步记约 −20（等于把正常爬楼当成驻车来罚）
    loose = T._task(profile, **dict(R25, tilt_temperature=0.05))
    assert _soft_term(loose, 28.0) < -15.0


def test_r25_tipping_gradient_points_back_upright():
    """想象里软存活对倾角的梯度：翻车代价够大时，在 40° 附近把车往回扶（实测约为第二十四轮 −20 / 温度 0.05 时的 89 倍）。"""
    def grad(task):
        pitch = torch.tensor(math.radians(40.0), requires_grad=True)
        rotation = T.quat.axis_angle_to_quat(torch.stack([torch.zeros(()), -pitch, torch.zeros(())])[None])
        s = T.state.make_state(torch.tensor([[0.3, 0.0, T.GROUND_Z]]), rotation, torch.zeros(1, 3), torch.zeros(1, 3),
                               torch.zeros(1, 12), torch.zeros(1, 12))
        T._terms(task, s)["terminated"].sum().backward()
        return pitch.grad.item()
    g25, g24 = grad(T._task(**R25)), grad(T._task(**dict(R25, terminated=-20.0, tilt_temperature=0.05)))
    assert g25 < 0 and g24 < 0  # 奖励随倾角增大而下降
    assert abs(g25) > 50 * abs(g24)
