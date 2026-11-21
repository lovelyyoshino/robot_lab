"""
test_supermimic_r24.py — 第二十四轮：线性停滞放开到倒车区间（来回抖不比停住便宜）、actor 逐维动作上界（摆臂不超关节限位）。

@author pony
@date 2026-11-18
@version v1.0.0
@last_modified 2026-11-18

@changelog
    - v1.0.0 (2026-11-18): 初始创建
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
from supermimic import symmetry, terrain_map  # noqa: E402
from supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from supermimic.networks import Actor  # noqa: E402
from supermimic.state import NUM_ACTIONS  # noqa: E402

R24 = dict(stall=0.0, stall_linear=-20.0, stall_linear_cap=5.0, progress=12.0, prelift_front=10.0)
TWO_PI = 6.2831853
BOUND = (2.72,) * 4 + (TWO_PI,) * 8


def _stall(task, vx):
    return T._terms(task, T._state(0.3, vx=vx))["stall_linear"].item()


def test_stall_linear_cap_default_is_unchanged_and_cap_opens_reverse():
    old, new = T._task(stall_linear=-20.0), T._task(stall_linear=-20.0, stall_linear_cap=5.0)
    for vx in (0.3, 0.075, 0.04, 0.0):  # 门槛 0.25 × 0.3 = 0.075 m/s 以上不计，停住 = 权重
        assert _stall(old, vx) == pytest.approx(_stall(new, vx), abs=1e-5)
    assert _stall(new, 0.0) == pytest.approx(-20.0, rel=1e-4) and _stall(new, 0.3) == 0.0
    assert _stall(old, -0.3) == pytest.approx(-20.0, rel=1e-4)  # 原来倒车封顶在停住的代价
    assert _stall(new, -0.15) == pytest.approx(-60.0, rel=1e-3)
    assert _stall(new, -0.3) == pytest.approx(-100.0, rel=1e-3) and _stall(new, -1.0) == pytest.approx(-100.0, rel=1e-3)
    with pytest.raises(ValueError, match="stall_linear_cap"):
        T._task(stall_linear_cap=0.5)


@pytest.mark.parametrize("profile", [T.SINGLE, T.STAIRS])
def test_jitter_with_zero_net_speed_is_not_cheaper_than_standing_still(profile):
    """第二十三轮驻车：72% 时间前蹭 0.11 m/s、28% 时间倒车（净速度 0）。旧权重下比停住便宜，第二十四轮权重下不便宜。"""
    fwd, share = 0.11, 0.72
    back = -fwd * share / (1.0 - share)

    def gain(task, x):
        total = lambda vx: sum(v.item() for v in T._terms(task, T._state(x, vx=vx)).values())  # noqa: E731
        return share * total(fwd) + (1.0 - share) * total(back) - total(0.0)

    old = T._task(profile, stall=-5.0, stall_linear=-3.0, progress=1.0)
    new = T._task(profile, **R24)
    for x in (0.3, 0.95):  # 平地；前轮在立面前约 0.2 m（减速带内）
        assert gain(new, x) <= 1e-3, x
    assert gain(old, 0.95) > 3.0  # 旧权重：抖动每步比停住多得 3 以上


def test_actor_scalar_bound_is_bit_identical_and_keeps_state_dict():
    torch.manual_seed(0)
    actor = Actor(20, (16,), TWO_PI)
    obs = torch.randn(7, 20) * 5
    raw = actor.actor(obs)
    assert torch.equal(actor(obs), TWO_PI * torch.tanh(raw / TWO_PI))
    assert not any("action_bound" in k for k in actor.state_dict())


def test_actor_per_dim_bound_limits_arms_only_and_scripts(tmp_path):
    torch.manual_seed(0)
    actor = Actor(20, (16,), BOUND).eval()
    with torch.no_grad():
        actor.actor[-1].weight.mul_(50.0)  # 把原始输出推到很大
    out = actor(torch.randn(256, 20) * 5)
    assert out[:, :4].abs().max() <= 2.72 + 1e-5 and out[:, :4].abs().max() > 2.6
    assert out[:, 4:].abs().max() > 5.0 and out[:, 4:].abs().max() <= TWO_PI + 1e-5
    # 上界以内与标量版同式；state_dict 带上界，重建后输出一致
    small = Actor(20, (16,), BOUND)
    clone = Actor(20, (16,), BOUND)
    clone.load_state_dict(small.state_dict())
    obs = torch.randn(5, 20)
    assert torch.equal(small(obs), clone(obs)) and "action_bound" in small.state_dict()
    torch.jit.script(actor).save(str(tmp_path / "p.pt"))
    assert torch.allclose(torch.jit.load(str(tmp_path / "p.pt"))(obs), actor(obs), atol=1e-6)
    for bad in ((1.0,) * 11, (1.0,) * 11 + (0.0,)):
        with pytest.raises(ValueError, match="action_bound"):
            Actor(20, (16,), bad)


def test_learner_with_per_dim_bound_trains_saves_and_exports(tmp_path):
    torch.manual_seed(0)
    task = T._task(T.STAIRS, **R24)
    limits = tuple((-2.72, 2.72) for _ in range(4)) + tuple((-6.37, 6.37) for _ in range(4)) + (None,) * 4
    cfg = SuperMimicCfg(num_envs=8, buffer_len=40, wm_hidden=(32,), actor_hidden=(32,), critic_hidden=(32,),
                        wm_window=4, wm_batch=16, wm_updates=2, horizon=4, imag_batch=8, warmup_iterations=0,
                        context_dim=symmetry.CONTEXT_DIM, aux_dim=0, action_bound=BOUND, action_limits=limits,
                        action_saturation_coef=0.02)
    hmap = task.hmap
    learner = SuperMimicLearner(cfg, "cpu", extra_fn=lambda s: terrain_map.relief_features(hmap, T.KIN, s),
                                extra_dim=terrain_map.RELIEF_DIM, task=task)
    assert torch.isfinite(learner._raw_hi[:8]).all() and bool(learner._raw_limited[:8].all())
    # 摆臂的饱和界按自己的上界换算（2.72·atanh(0.999)），不是按 2π
    assert learner._raw_hi[0].item() == pytest.approx(2.72 * math.atanh(0.999), rel=1e-4)
    prev = torch.zeros(8, 12)
    cmd = torch.tensor([[0.3, 0.0, 0.0]]).expand(8, 3)
    for i in range(12):
        s = torch.cat([T._state(0.4 + 0.05 * i + 0.01 * e, vx=0.2) for e in range(8)])
        action = learner.act(task.policy_obs(s, cmd, prev))
        assert action[:, :4].abs().max() <= 2.72 + 0.5  # 探索噪声之外不超界
        learner.observe(s, action, prev, cmd, torch.zeros(8, dtype=torch.bool), torch.zeros(8, symmetry.CONTEXT_DIM))
        prev = action
    logs = learner.update()
    assert math.isfinite(logs["actor_loss"])
    learner.export_actor(tmp_path / "policy.pt")
    obs = task.policy_obs(torch.cat([T._state(0.5 + 0.1 * e) for e in range(3)]), cmd[:3], torch.zeros(3, 12))
    source = learner.actor_ema if learner.actor_ema is not None else learner.actor
    out = torch.jit.load(str(tmp_path / "policy.pt"))(obs)
    assert torch.allclose(out, source(obs), atol=1e-6) and out.shape == (3, NUM_ACTIONS)
    torch.save(learner.state_dict(), tmp_path / "ck.pt")  # train_terrain 的存档方式
    data = torch.load(tmp_path / "ck.pt", weights_only=False)
    rebuilt = Actor(task.obs_dim, tuple(data["cfg"]["actor_hidden"]), data["cfg"]["action_bound"],
                    squash=data["cfg"]["action_squash"])  # export_ckpt.py / scan_student.py 的重建方式
    rebuilt.load_state_dict(data.get("actor_ema", data["actor"]))
    assert torch.allclose(rebuilt(obs), source(obs), atol=1e-6)
