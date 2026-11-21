"""
test_supermimic_alt_lift.py — 楼梯左右轮替奖励（preview_task v0.20.0）：默认关等于原行为、左右镜像不变、等高受罚、领先轮加权。

@author pony
@date 2026-11-16
@version v1.0.0
@last_modified 2026-11-16

@changelog
    - v1.0.0 (2026-11-16): 初始创建
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

from supermimic import kinematics, state, terrain_map  # noqa: E402
from supermimic.preview_task import PreviewRewardCfg, PreviewTask  # noqa: E402

KIN = kinematics.RoverKinematics()
LIMITS = torch.tensor([[-0.58, 0.58]] * 4 + [[-1.3, 1.3]] * 4)
ON = dict(stairs_climb_pair_sum=1.0, stairs_climb_trail_weight=0.25, stairs_alt_lift=-3.0, stairs_contact_climb_m=0.05)


def _pattern():
    return torch.tensor([[x, y, 20.0] for x in (0.0, 0.3) for y in (-0.2, 0.0, 0.2)])


def _stairs_task(**overrides) -> PreviewTask:
    # 关于 y=0 对称的 12 cm × 30 cm 楼梯（第一级立面 x=1.2）
    xs = torch.arange(0.0, 4.0 + 1e-9, 0.02)
    heights = torch.tensor([[0.12 * min(max(int((float(x) - 1.2) // 0.3) + 1, 0), 6)] * 101 for x in xs])
    hmap = terrain_map.HeightMap(heights, (0.0, -1.0), 0.02)
    cfg = replace(PreviewRewardCfg(), **overrides)
    return PreviewTask(hmap, KIN, _pattern(), LIMITS, cfg, lane_center_y=torch.tensor([0.0]))


def _ctx(clear_left, clear_right, lead=0.0, rise=0.12, stairs=True):
    """后轴两轮（rl=1, rr=3）贴立面的手工几何上下文；前轴远离立面。"""
    n = len(clear_left)
    clearance = torch.zeros(n, 4)
    clearance[:, 1], clearance[:, 3] = torch.tensor(clear_left), torch.tensor(clear_right)
    lead_t = torch.full((n, 4), 1.0)
    lead_t[:, 1] = lead_t[:, 3] = lead
    return {"lead": lead_t, "clearance": clearance, "support_clearance": clearance.clone(),
            "riser_rise": torch.full((n, 4), rise), "single": torch.zeros(n, 4, dtype=torch.bool),
            "support_margin": torch.full((n, 4), 0.05), "ground": torch.zeros(n, 4),
            "stairs_zone": torch.full((n,), stairs)}


def _states(n, x0=0.7):
    torch.manual_seed(3)
    joints = torch.zeros(n, 12)
    joints[:, :4] = (torch.rand(n, 4) - 0.5) * 0.8
    pos = torch.stack((torch.rand(n) * 0.6 + x0, (torch.rand(n) - 0.5) * 0.1, torch.full((n,), 0.37)), -1)
    quat = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]) + 0.03 * torch.randn(n, 4), dim=-1)
    return state.make_state(pos, quat, torch.randn(n, 3) * 0.1, torch.randn(n, 3) * 0.1, joints,
                            torch.randn(n, 12) * 0.1)


def test_defaults_off_match_original_terms():
    base, off = _stairs_task(stairs_climb_pair_sum=1.0), _stairs_task(stairs_climb_pair_sum=1.0,
                                                                      stairs_climb_trail_weight=1.0)
    s, prev, cmd = _states(16), torch.zeros(16, 12), torch.tensor([[0.3, 0.0, 0.0]]).expand(16, 3)
    a, b = base.reward_terms(s, s, cmd, prev, prev), off.reward_terms(s, s, cmd, prev, prev)
    assert "stairs_alt_lift" not in a and a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_alt_lift_penalises_equal_heights_only_on_stairs_contact():
    task = _stairs_task(**ON)
    cost = task._stairs_alt_lift(_ctx([0.0, 0.03, 0.08, 0.0], [0.0, 0.03, 0.0, 0.08]))
    assert torch.allclose(cost[:2], torch.ones(2), atol=1e-6)  # 等高：满额
    assert torch.allclose(cost[2:], torch.zeros(2), atol=1e-6)  # 差过半个立面高：无代价，左右先上均可
    mid = task._stairs_alt_lift(_ctx([0.03], [0.0]))
    assert 0.0 < float(mid) < 1.0
    assert float(task._stairs_alt_lift(_ctx([0.0], [0.0], stairs=False))) == 0.0  # 非楼梯区
    assert float(task._stairs_alt_lift(_ctx([0.0], [0.0], lead=0.015))) == 1.0  # 贴着立面 lead 略大于 0 仍算
    assert float(task._stairs_alt_lift(_ctx([0.0], [0.0], lead=0.06))) == 0.0  # 离立面 6 cm 不算
    narrow = _stairs_task(**{**ON, "stairs_contact_climb_m": 0.0})
    assert float(narrow._stairs_alt_lift(_ctx([0.0], [0.0], lead=0.015))) == 0.0  # 原窗口 lead ≤ 0
    # 等高处的梯度为 0（鞍点），只要有一点差就把两轮往相反方向推
    ctx = _ctx([0.004], [0.0])
    ctx["clearance"].requires_grad_(True)
    task._stairs_alt_lift(ctx).sum().backward()
    g = ctx["clearance"].grad[0]
    assert g[1] < 0 < g[3]


def test_leader_weighting_is_symmetric_at_tie_and_favours_higher_wheel():
    pair_sum = _stairs_task(stairs_climb_pair_sum=1.0)
    leader = _stairs_task(**ON)
    # 立面 12 cm、lead = 0 时翻越目标净空 3 cm（下限）；都低于目标
    tie = _ctx([0.01], [0.01])
    assert torch.allclose(leader._prelift_tracking(tie), 0.625 * pair_sum._prelift_tracking(tie), atol=1e-6)
    ctx = _ctx([0.02], [0.0])
    ctx["clearance"].requires_grad_(True)
    leader._prelift_tracking(ctx).sum().backward()
    g = ctx["clearance"].grad[0]
    assert g[1] > 0 and g[3] > 0 and g[1] > 2.0 * g[3]  # 两轮都该往上，领先轮的梯度大得多
    swapped = _ctx([0.0], [0.02])
    assert torch.allclose(leader._prelift_tracking(swapped), leader._prelift_tracking(_ctx([0.02], [0.0])))


def test_new_terms_invariant_under_left_right_swap():
    # 真实 URDF 左右有约 4 mm 不对称（symmetry 测试用理想对称运动学），这里直接在几何上下文上交换左右轮
    torch.manual_seed(5)
    task = _stairs_task(**ON)
    n = 256
    ctx = {"lead": torch.rand(n, 4) * 0.15 - 0.1, "clearance": torch.rand(n, 4) * 0.1 - 0.01,
           "riser_rise": torch.rand(n, 4) * 0.1 + 0.06, "single": torch.zeros(n, 4, dtype=torch.bool),
           "support_margin": torch.rand(n, 4) * 0.05, "ground": torch.zeros(n, 4),
           "stairs_zone": torch.rand(n) > 0.3}
    ctx["support_clearance"] = ctx["clearance"] + torch.rand(n, 4) * 0.01
    sw = [2, 3, 0, 1]
    swapped = {k: (v[:, sw] if v.dim() == 2 else v) for k, v in ctx.items()}
    a, b = task._stairs_alt_lift(ctx), task._stairs_alt_lift(swapped)
    assert torch.allclose(a, b, atol=1e-6) and (a > 0).any()
    assert torch.allclose(task._prelift_tracking(ctx), task._prelift_tracking(swapped), atol=1e-6)


def test_real_states_reward_terms_finite_and_differentiable():
    task = _stairs_task(**ON)
    s = _states(64, x0=0.9).requires_grad_(True)
    prev, cmd = torch.zeros(64, 12), torch.tensor([[0.3, 0.0, 0.0]]).expand(64, 3)
    terms = task.reward_terms(s, s, cmd, prev, prev)
    assert (terms["stairs_alt_lift"] < 0).any()
    total = sum(v.sum() for v in terms.values())
    total.backward()
    assert torch.isfinite(s.grad).all()


def test_contact_climb_window_exempts_wheel_against_stairs_riser_from_support_cost():
    ctx = _ctx([0.02], [0.0], lead=0.02)
    ctx["lifted"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    ctx["dropping"] = torch.zeros(1, 4, dtype=torch.bool)
    ctx["support_margin"] = torch.tensor([[0.03, -0.034, 0.03, -0.026]])
    _, airborne_old, _ = _stairs_task(**{**ON, "stairs_contact_climb_m": 0.0})._support_weights(ctx)
    _, airborne_new, _ = _stairs_task(**ON)._support_weights(ctx)
    assert float(airborne_old[0, 1]) == 1.0 and float(airborne_new[0, 1]) == 0.0
    ctx["stairs_zone"] = torch.tensor([False])  # 楼梯区外不放宽
    _, airborne_out, _ = _stairs_task(**ON)._support_weights(ctx)
    assert float(airborne_out[0, 1]) == 1.0
