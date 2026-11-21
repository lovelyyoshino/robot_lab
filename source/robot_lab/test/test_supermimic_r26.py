"""
test_supermimic_r26.py — 第二十六轮：第二十二轮配置 + 登顶后回正窗口（post_climb_relief）。

@author pony
@date 2026-11-19
@version v1.0.0
@last_modified 2026-11-19

@changelog
    - v1.0.0 (2026-11-19): 初始创建
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

import test_supermimic_obstacle as T  # noqa: E402

ROLLING = ("toe", "flat_steer", "wheel_sync", "wheel_slip")
EDGE, HEIGHT = 1.5, 0.12
UP = lambda x: HEIGHT if x >= EDGE else 0.0  # noqa: E731
DOWN = lambda x: HEIGHT if x < EDGE else 0.0  # noqa: E731
FLAT = lambda x: 0.0  # noqa: E731


def _task(profile, **cfg):
    limits = torch.tensor([[-0.58, 0.58]] * 4 + [[-1.3, 1.3]] * 4)
    pattern = torch.tensor([[x, y, 20.0] for x in (0.0, 0.3) for y in (-0.2, 0.2)])
    return T.PreviewTask(T._map(profile, x_max=6.0), T.KIN, pattern, limits, T.PreviewRewardCfg(**cfg),
                         lane_center_y=T.LANES)


def _rolling_state(rear_past_edge, ground):
    """后轴轮心在立面后 rear_past_edge 米处、四轮着地；打角与四轮轮速各不相同，平地滚动四项都非零。"""
    x = EDGE - T.REAR_X + rear_past_edge
    s = T._state(x, z=T.GROUND_Z + ground, vx=0.3)
    s[0, 13 + 4:13 + 8] = torch.tensor([0.10, -0.05, 0.08, 0.02])
    s[0, 25 + 8:25 + 12] = torch.tensor([2.0, 3.5, 2.6, 3.0])
    return s


def _rolling(task, s):
    terms = T._terms(task, s)
    return torch.stack([terms[name] for name in ROLLING]).squeeze(-1)


def test_post_climb_relief_validation():
    with pytest.raises(ValueError, match="post_climb_relief"):
        _task(FLAT, post_climb_relief=1.5)
    with pytest.raises(ValueError, match="post_climb_relief"):
        _task(FLAT, post_climb_relief=-0.1)
    with pytest.raises(ValueError, match="post_climb_window_m"):
        _task(FLAT, post_climb_relief=1.0, post_climb_window_m=0.0)


def test_default_is_unchanged_and_never_flags_post_climb():
    task = _task(UP)
    s = _rolling_state(0.3, HEIGHT)
    assert not bool(task.terrain_context(s)["post_climb"].any())
    off = _rolling(task, s)
    assert (off != 0).all()
    cfg = task.cfg
    rolling = task._flat_rolling(s, task.terrain_context(s))
    for i, name in enumerate(ROLLING):
        expected = getattr(cfg, name) * task.terrain_context(s)["flat"] * rolling[name]
        assert torch.equal(off[i], expected.squeeze(-1))


def test_relief_applies_only_just_after_climbing_onto_a_platform():
    off, on, half = _task(UP), _task(UP, post_climb_relief=1.0), _task(UP, post_climb_relief=0.5)
    just_up = _rolling_state(0.3, HEIGHT)
    assert bool(on.terrain_context(just_up)["post_climb"].all())
    assert float(on.terrain_context(just_up)["flat"]) > 0.95
    assert torch.allclose(_rolling(on, just_up), torch.zeros(4))
    assert torch.allclose(_rolling(half, just_up), 0.5 * _rolling(off, just_up))
    # 窗口外（后轴已离立面 1.3 m）不减免
    far = _rolling_state(1.3, HEIGHT)
    assert not bool(on.terrain_context(far)["post_climb"].any())
    assert torch.equal(_rolling(on, far), _rolling(off, far))


@pytest.mark.parametrize("profile, ground", [(FLAT, 0.0), (DOWN, 0.0)])
def test_relief_does_not_touch_flat_driving_or_after_stepping_down(profile, ground):
    off, on = _task(profile), _task(profile, post_climb_relief=1.0)
    s = _rolling_state(0.3, ground)
    assert not bool(on.terrain_context(s)["post_climb"].any())
    assert torch.equal(_rolling(on, s), _rolling(off, s))
    assert (_rolling(off, s) != 0).all()
