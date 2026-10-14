"""前瞻抬轮的几何与反作弊回归，不把奖励可计算当作已学会越障。

@author pony
@date 2026-09-30
@version v0.2.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 回归8/12/16cm课程、早期楼梯覆盖和单侧主动完成。
    - (2026-10-13): 回归落地冲击基线、起抬失败信用、接近姿态余量及逐轮课程。
    - (2026-10-12): 复现离地前净空学习断点，锁定动作质量晋级与通行加分，不放宽独立验收。
    - (2026-10-12): 验证路线命令在Teacher/Student观测中可辨别偏航，删除诊断势能冒充奖励的旧测试。
    - (2026-10-12): 覆盖奖励裁剪、未知继承项隔离及零偏航负约束，保留路线与完整动作回归。
    - (2026-10-12): 锁定路线晋级、阶段地形覆盖及四车混合回放的楼梯展示。
    - (2026-10-11): 覆盖离地前减载引导、速度奖励零基线与预瞄前颠簸，替代已撤回的停滞计时方案。
    - (2026-10-11): 复现侧倾早离地补领奖，验证真实起抬、渐进抬升及全宽漏抬约束。
    - (2026-10-10): 验证10cm时序、远处早抬惩罚和时机奖励延后结算，防止一抬就放套利。
    - (2026-10-10): 全宽双侧先后越缘落轮对照独立验收，不要求两轮同时悬空。
    - (2026-10-10): 左右窄障碍完整轨迹对照奖励与独立验收，覆盖提前落轮和局部地面切换。
    - (2026-10-10): 覆盖最后8cm保持、提前落轮锁死、顺序越缘与连续承重落轮结算。
    - (2026-10-09): 覆盖左右悬架收缩仍接地的低位基线，拒绝悬轮和无承重冒充。
    - (2026-10-09): 锁定窄障碍目标轮完成后必须领取一次完成奖励的回归。
    - (2026-10-09): 覆盖窄障碍持续减载、支撑拓扑、缺失信号、碰撞惩罚与分类统计。
    - v0.2.0 (2026-10-08): 覆盖常悬轮、先后抬轮、早抬、后退重领与reset隔离。
    - v0.1.0 (2026-09-30): 覆盖前方高度差、主动关节抬轮、移动门和未知地形拒绝。
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def preview(monkeypatch):
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview.py"
    )
    spec = importlib.util.spec_from_file_location("vision_preview_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def scene(*, height=0.07, edge=0.75, lift=0.0, speed=0.4):
    x, y = torch.meshgrid(torch.linspace(-0.8, 1.2, 21), torch.linspace(-0.5, 0.5, 11), indexing="ij")
    hits = torch.stack((x.flatten(), y.flatten(), (x >= edge).float().flatten() * height), -1)[None]
    neutral = torch.tensor([[0.35, 0.3, -0.24], [-0.35, 0.3, -0.24],
                            [0.35, -0.3, -0.24], [-0.35, -0.3, -0.24]])
    root = torch.tensor([[0.0, 0.0, 0.34]])
    wheels = neutral[None] + root[:, None]
    wheels[:, [0, 2], 2] += lift
    return dict(
        ray_hits_w=hits, root_pos_w=root, root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        wheel_centers_w=wheels, neutral_centers_b=neutral, wheel_radii=torch.full((4,), 0.1),
        command_x=torch.tensor([0.4]), velocity_x=torch.tensor([speed]),
        supported_wheels=torch.tensor([2]), nonwheel_contact=torch.tensor([False]),
    )


def test_near_step_asks_front_wheels_to_lift_before_contact(preview):
    out = preview.preview_measurements(**scene(lift=0.07))
    assert out["preview"].tolist() == [[True, False, True, False]]
    assert out["prelift"].tolist() == [[True, False, True, False]]
    assert out["potential"].item() > 0.2
    assert torch.all(out["advance_m"][out["preview"]] >= 0.08)


def test_stairs_have_three_height_stages_and_clear_top_targets(preview):
    profiles = [preview.stairs_profile(d) for d in (0.0, 0.5, 1.0)]
    assert [p["step_height_m"] for p in profiles] == [0.08, 0.12, 0.16]
    assert [p["num_steps"] for p in profiles] == [2, 3, 4]
    assert [p["target_distance_m"] for p in profiles] == pytest.approx([2.5, 2.6, 3.0])
    assert all(p["first_edge_m"] - 0.34724 - 0.1085 > 0.65 for p in profiles)


def test_all_stages_expose_stairs_and_no_dual_active_gate(preview):
    assert all(weights.get("pyramid_stairs_inv", 0) > 0 for weights in preview.PREVIEW_STAGE_WEIGHTS)
    columns = ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "flat")
    levels = torch.tensor([0, 1, 2, 2, 5, 5, 8])
    kinds = torch.tensor([0, 2, 0, 3, 1, 3, 3])
    assert preview.preview_route_promotion_mask(levels, kinds, columns).tolist() == [
        True, True, False, True, False, True, True,
    ]


def test_direct_ground_rays_do_not_mistake_adjacent_step_for_ground(preview):
    data = scene(edge=0.4)
    # 邻近的稀疏高度点在8cm顶面，但轮心仍在前缘之前的平地。
    data["ground_hits_w"] = data["wheel_centers_w"].clone()
    data["ground_hits_w"][..., 2] = 0.
    sample = preview.preview_measurements(**data)
    assert sample["clearance_m"][0, 0].item() == pytest.approx(0., abs=1e-6)
    assert sample["ground_valid"].all()


def test_stair_rearm_requires_new_edge_grounding_and_keeps_credit_deduplication(preview):
    state = preview.PreliftRewardState(1, "cpu")

    def sample(edge, lead, lift=0., ground=0.):
        value = event_sample(preview, lift=(lift, 0.))
        value["terrain_kind"] = torch.tensor([3])
        value["obstacle_xy"][..., 0] = edge
        value["wheel_centers_w"][..., 0] = edge - 0.1 - lead
        value["wheel_centers_w"][..., 2] += ground
        value["clearance_m"][..., :] = 0.
        value["clearance_m"][0, 0] = lift
        value["wheel_support_force_n"].fill_(20.)
        if lift >= 0.005:
            value["wheel_support_force_n"][0, 0] = 0.
        value["preview"][..., :] = False
        value["preview"][:, [0, 2]] = True
        return value

    state.update(sample(1.2, .4))
    for _ in range(3):
        state.update(sample(1.2, .10, .04))
    state.update(sample(1.2, -.01, .04))
    for _ in range(3):
        result = state.update(sample(1.2, -.2, ground=.08))
    assert result["front_completion"].item()
    assert state.stair_completions.item() == 1
    # 同阶再次抬落和后退，不重新武装或再次支付完整动作奖励。
    for _ in range(6):
        assert not state.update(sample(1.2, .10, .04))["front_completion"].item()
    for _ in range(2):
        state.update(sample(1.6, .10, ground=.08))
    assert state.edge_xy[0, 0, 0].item() == pytest.approx(1.2)
    state.update(sample(1.6, .10, ground=.08))
    assert state.edge_xy[0, 0, 0].item() == pytest.approx(1.6)
    assert not state.takeoff_seen.any()
    assert state.stair_opportunities[0, [0, 2]].tolist() == [2, 2]
    # 上一级右轮被动上阶也能面对下一级，不以主动动作作为重新武装条件。
    for _ in range(3):
        state.update(sample(1.6, .10, .04, ground=.08))
    state.update(sample(1.6, -.01, .04, ground=.08))
    for _ in range(3):
        result = state.update(sample(1.6, -.2, ground=.16))
    assert result["front_completion"].item()
    assert state.stair_completions.item() == 2
    assert not state.completed[0, 2]
    assert state.elapsed_s.item() > .30
    state.reset()
    assert not state.stair_opportunities.any() and not state.stair_completions.any()
    assert torch.isneginf(state.completion_edge_x).all()


def test_route_stability_cannot_be_stitched_across_unstable_steps(preview):
    state = preview.PreliftRewardState(1, "cpu")
    sample = event_sample(preview)
    sample["route_passed"] = torch.tensor([True])
    for _ in range(14):
        assert not state.update(sample)["route_completion"].item()
    sample["route_passed"].fill_(False)
    state.update(sample)
    sample["route_passed"].fill_(True)
    assert not state.update(sample)["route_completion"].item()


def test_diagnostic_distance_is_retained_before_prelift_window(preview):
    data = scene(lift=0.0)
    data["root_pos_w"][:, 0] = -0.5
    data["wheel_centers_w"][:, :, 0] -= 0.5
    out = preview.preview_measurements(**data)
    assert not out["preview"].any()
    assert torch.all(out["advance_m"] == 0)
    assert torch.all(out["unbounded_advance_m"] > 0.65)


def event_sample(preview, *, lift=(0.0, 0.0), root_x=0.15, eligible=True):
    data = scene()
    data["root_pos_w"][:, 0] += root_x
    data["wheel_centers_w"][:, :, 0] += root_x
    data["wheel_centers_w"][:, 0, 2] += lift[0]
    data["wheel_centers_w"][:, 2, 2] += lift[1]
    out = preview.preview_measurements(**data)
    out["eligible"].fill_(eligible)
    out["wheel_support_force_n"] = torch.where(out["clearance_m"] >= 0.005, 0., 20.)
    return out


def prime_baseline(preview, state, *, kind=0):
    """模拟稳定接近而非出生后三帧；不能直接给状态写入合格基线。"""
    for _ in range(40):
        state.update(narrow_sample(preview, kind=kind, root_x=-0.15))


def test_spawn_impact_is_not_unloading_credit(preview):
    state = preview.PreliftRewardState(1, "cpu")
    # 实跑的冲击先峰值后衰减，旧实现把约149N冻结成参考。
    for force in [212., 36., 39.] + [20.] * 37:
        sample = narrow_sample(preview, kind=0, root_x=-0.15)
        sample["wheel_support_force_n"].fill_(force)
        state.update(sample)
    assert state.baseline_force[0, 0].item() == pytest.approx(20.)
    sample = narrow_sample(preview, kind=0)
    assert state.update(sample)["credit"].item() == 0.
    assert not state.completed.any()


def test_baseline_cannot_be_learned_inside_preparation_or_after_retreat(preview):
    state = preview.PreliftRewardState(1, "cpu")
    for _ in range(40):
        state.update(narrow_sample(preview))
    assert not state.baseline_samples.any()
    prime_baseline(preview, state, kind=1)
    assert not state.baseline_samples.any()
    state.reset()
    prime_baseline(preview, state, kind=1)
    assert state.baseline_samples[0, 0] >= 6


def test_baseline_uses_contact_median_without_rejecting_exploration_noise(preview):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state, kind=1)
    for force in (10., 30., 20., 200., 21., 19.):
        sample = narrow_sample(preview, root_x=-0.15, force=force)
        state.update(sample)
    assert state.baseline_samples[0, 0].item() == 6
    assert state.baseline_force[0, 0].item() == pytest.approx(20.5)
    # 离地/断触会清掉接近段候选，不把不连续承重拼接成合格参考。
    state.update(narrow_sample(preview, root_x=-0.15, force=0.))
    assert state.baseline_samples[0, 0].item() == 0
    for _ in range(5):
        state.update(narrow_sample(preview, root_x=-0.15))
    assert state.baseline_samples[0, 0].item() == 5


def test_valid_attempt_keeps_small_credit_but_repetition_does_not_pay(preview):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state)
    total = 0.
    for height in (0.006, 0.021, 0.):
        total += state.update(event_sample(preview, lift=(height, 0.)))["credit"].item()
    assert total > 0.
    assert state.hold_broken[0, 0] and not state.front_paid.any()
    for _ in range(10):
        assert state.update(event_sample(preview, lift=(0.04, 0.)))["credit"].item() == 0.
        state.update(event_sample(preview))


def test_stance_reserve_cost_only_applies_to_grounded_far_approach(preview):
    state = preview.PreliftRewardState(1, "cpu")
    far = event_sample(preview, root_x=-0.2)
    far["articulation_lift_m"].fill_(0.09)
    assert state.update(far)["stance_penalty"].item() > 0.
    near = event_sample(preview)
    near["articulation_lift_m"].fill_(0.09)
    assert state.update(near)["stance_penalty"].item() == 0.
    far["articulation_lift_m"].fill_(0.02)
    assert state.update(far)["stance_penalty"].item() == 0.


def test_early_curriculum_accepts_either_narrow_side_but_requires_stairs_at_boundary(preview):
    columns = ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv")
    levels = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    kinds = torch.tensor([1, 2, 1, 2, 0, 0, 1])
    assert preview.preview_route_promotion_mask(levels, kinds, columns).tolist() == [
        True, True, True, True, False, False, True,
    ]


@pytest.mark.parametrize("target", [0, 2])
def test_grounded_unloading_gets_guidance_before_takeoff(preview, target):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state)
    sample = event_sample(preview, root_x=0.13)
    sample["wheel_support_force_n"][0, target] = 10.
    sample["clearance_m"][0, target] = 0.001
    result = state.update(sample)
    assert result["credit"].item() > 0
    assert not result["events"].any() and not state.takeoff_seen.any()
    assert state.update(sample)["credit"].item() == 0
    state.update(event_sample(preview, root_x=0.13))
    assert state.update(sample)["credit"].item() == 0


def test_body_squat_without_unloading_cannot_earn_preparation(preview):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state)
    sample = event_sample(preview)
    sample["articulation_lift_m"][:, [0, 2]] = 0.10
    assert state.update(sample)["credit"].item() == 0


@pytest.mark.parametrize("kind,target", [(0, 0), (0, 2), (1, 0), (2, 2)])
def test_clearance_progress_is_rewarded_before_takeoff_and_capped(preview, kind, target):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state, kind=kind)
    sample = narrow_sample(preview, kind=kind)
    sample["wheel_support_force_n"][0, target] = 10.
    state.update(sample)
    # 固定减载比例，只改变尚未达到5mm的净空；旧奖励在这里完全没有增量。
    for clearance in (0.001, 0.002, 0.004):
        sample["wheel_centers_w"][0, target, 2] += clearance - sample["clearance_m"][0, target]
        sample["clearance_m"][0, target] = clearance
        result = state.update(sample)
        assert result["credit"].item() > 0
        assert not result["events"].any() and not state.takeoff_seen.any()
        assert state.update(sample)["credit"].item() == 0
    peak = state.best.clone()
    sample["clearance_m"][0, target] = 0.
    state.update(sample)
    sample["clearance_m"][0, target] = 0.004
    assert state.update(sample)["credit"].item() == 0
    assert torch.equal(state.best, peak)
    assert peak.max() <= state.cfg.preparation_fraction


@pytest.mark.parametrize("fault", ["outside_window", "loaded", "wrong_side", "unsupported", "contacted"])
def test_subtakeoff_clearance_cannot_bypass_timing_or_contact(preview, fault):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state, kind=1)
    sample = narrow_sample(preview, root_x=0.05 if fault == "outside_window" else 0.15)
    target = 2 if fault == "wrong_side" else 0
    sample["wheel_support_force_n"][0, target] = 20. if fault == "loaded" else 10.
    if fault == "unsupported":
        sample["narrow_supported"][:] = False
    if fault == "contacted":
        state.contacted[:] = True
    state.update(sample)
    sample["clearance_m"][0, target] = 0.004
    assert state.update(sample)["credit"].item() == 0
    assert not state.takeoff_seen.any()


@pytest.mark.parametrize("command", [0.0, 0.01, 0.2, 0.4, 0.6])
def test_velocity_guidance_is_zero_at_rest_and_bounded(preview, command):
    target = torch.tensor([[command, 0.]])
    rest = preview.centered_velocity_tracking(target, torch.zeros_like(target), 0.5)
    perfect = preview.centered_velocity_tracking(target, target, 0.5)
    reverse = preview.centered_velocity_tracking(target, -target, 0.5)
    assert torch.isfinite(rest).all() and torch.isfinite(reverse).all()
    assert rest.item() == pytest.approx(math.exp(-command**2 / 0.5**2) if command < 0.1 else 0.)
    assert perfect.item() == pytest.approx(1.)
    assert -1 <= reverse.item() <= 1
    if command >= 0.1:
        assert reverse.item() <= 0


def test_velocity_guidance_rejects_lateral_drift_and_invalid_std(preview):
    target = torch.tensor([[0.4, 0.]])
    assert preview.centered_velocity_tracking(target, torch.tensor([[0., 0.4]]), 0.5).item() < 0
    for std in (0., -1., float("nan")):
        with pytest.raises(ValueError):
            preview.centered_velocity_tracking(target, target, std)


def test_flat_approach_bounce_does_not_consume_obstacle_attempt(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview, root_x=-0.5))
    state.update(event_sample(preview, root_x=-0.5, lift=(0.01, 0.)))
    assert not state.takeoff_seen.any()
    state.update(event_sample(preview))
    assert state.update(event_sample(preview, lift=(0.07, 0.)))["events"][0, 0]


@pytest.mark.parametrize("fault", ["early", "unsupported", "no_baseline", "wrong_wheel", "nonfinite"])
def test_preparation_does_not_bypass_timing_support_or_target(preview, fault):
    state = preview.PreliftRewardState(1, "cpu")
    if fault != "no_baseline":
        prime_baseline(preview, state, kind=1)
    sample = narrow_sample(preview, root_x=-0.1 if fault == "early" else 0.15)
    sample["wheel_support_force_n"][0, 2 if fault == "wrong_wheel" else 0] = 10.
    if fault == "unsupported":
        sample["narrow_supported"][:] = False
    if fault == "nonfinite":
        sample["wheel_support_force_n"][0, 0] = float("nan")
    result = state.update(sample)
    assert result["credit"].item() == 0
    assert not result["events"].any()


@pytest.mark.parametrize("target", [0, 2])
@pytest.mark.parametrize("land_before_window", [False, True])
def test_body_roll_early_takeoff_cannot_collect_late_joint_credit(preview, target, land_before_window):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview, root_x=-0.15))
    early = event_sample(preview, root_x=-0.05)
    early["articulation_lift_m"][0, target] = -0.04
    early["clearance_m"][0, target] = 0.03
    early["wheel_centers_w"][0, target, 2] += 0.03
    early["wheel_support_force_n"][0, target] = 0.
    for _ in range(15):
        assert state.update(early)["credit"].item() == 0.
    if land_before_window:
        state.update(event_sample(preview, root_x=0.0))
    late = event_sample(preview, lift=(0.07, 0.) if target == 0 else (0., 0.07))
    result = state.update(late)
    assert not result["events"].any()
    assert result["credit"].item() == 0.
    assert state.early_lift_seen[0, target]


@pytest.mark.parametrize("target", [0, 2])
def test_actual_takeoff_window_is_separate_from_height_threshold(preview, target):
    state = preview.PreliftRewardState(1, "cpu")
    grounded = event_sample(preview, root_x=-0.05)
    grounded["articulation_lift_m"][0, target] = 0.04
    assert state.update(grounded)["idle_penalty"].item() == 0.
    onset = event_sample(preview, root_x=0.13)
    onset["clearance_m"][0, target] = 0.006
    onset["wheel_centers_w"][0, target, 2] += 0.006
    onset["wheel_support_force_n"][0, target] = 0.
    onset["articulation_lift_m"][0, target] = -0.01
    result = state.update(onset)
    assert not result["events"].any() and not result["hold_broken"].any()
    lifted = event_sample(preview, root_x=0.19, lift=(0.07, 0.) if target == 0 else (0., 0.07))
    result = state.update(lifted)
    assert result["events"][0, target]
    assert result["takeoff_lead_m"][0, target].item() == pytest.approx(0.12)
    assert result["height_lead_m"][0, target].item() == pytest.approx(0.06, abs=1e-6)


def test_full_width_unprepared_collision_is_recorded_without_avoidance_penalty(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    sample = event_sample(preview, root_x=0.16)
    sample["wheel_resisting_force_n"] = torch.tensor([[100., 0., 100., 0.]])
    first = state.update(sample)
    assert first["unprepared_collision"][0, [0, 2]].all()
    assert first["credit"].item() == 0.
    assert state.update(sample)["credit"].item() == 0.


def test_top_landing_is_not_a_front_wall_collision(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    for _ in range(3):
        state.update(event_sample(preview, lift=(0.08, 0.08)))
    top = event_sample(preview, lift=(0.08, 0.08), root_x=0.26)
    top["clearance_m"][:, [0, 2]] = 0.
    top["wheel_support_force_n"][:] = 20.
    top["wheel_resisting_force_n"] = torch.tensor([[20., 0., 20., 0.]])
    result = state.update(top)
    assert result["held_to_edge"][0, [0, 2]].all()
    assert not result["hold_broken"].any()


def test_support_lost_during_takeoff_ramp_cannot_be_requalified(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    state.update(event_sample(preview, lift=(0.006, 0.)))
    unsupported = event_sample(preview, lift=(0.008, 0.))
    unsupported["eligible_gates"]["supported"].fill_(False)
    unsupported["eligible"].fill_(False)
    state.update(unsupported)
    assert not state.update(event_sample(preview, lift=(0.07, 0.)))["events"].any()


def test_full_width_missing_other_wheel_is_recorded_without_stopping_incentive(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    first = state.update(event_sample(preview, root_x=0.26))
    assert first["missed_wheels"][0, [0, 2]].all()
    assert first["credit"].item() == 0.
    assert state.update(event_sample(preview, root_x=0.30))["credit"].item() == 0.


def test_aligned_route_completion_pays_once_and_reset_rearms(preview):
    state = preview.PreliftRewardState(1, "cpu")
    sample = event_sample(preview, root_x=0.0)
    sample["route_passed"] = torch.tensor([True])
    for _ in range(14):
        assert not state.update(sample)["route_completion"].item()
    first = state.update(sample)
    assert first["route_completion"].item()
    assert first["credit"].item() == pytest.approx(state.cfg.route_completion_credit)
    assert state.update(sample)["credit"].item() == 0.
    state.reset()
    for _ in range(14):
        state.update(sample)
    assert state.update(sample)["credit"].item() == pytest.approx(state.cfg.route_completion_credit)


def test_step_route_completion_accepts_passive_crossing_only_after_continuous_stability(preview):
    state = preview.PreliftRewardState(1, "cpu")
    sample = event_sample(preview)
    sample["preview"][:] = False
    sample["terrain_kind"] = torch.tensor([0])
    sample["route_passed"] = torch.tensor([True])
    assert not state.update(sample)["route_completion"].item()
    sample["preview"][:, [0, 2]] = True
    assert not state.update(sample)["route_completion"].item()
    assert not state.route_paid.item()
    for _ in range(13):
        result = state.update(sample)
    assert result["route_completion"].item()
    assert not state.completed.any()


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_step_route_bonus_follows_real_sequence_and_terminal_progress(preview, kind):
    state = preview.PreliftRewardState(1, "cpu")
    def sample(*, raised=False, root_x=0.15, passed=False):
        if kind:
            value = narrow_sample(preview, kind=kind, raised=raised, root_x=root_x)
        else:
            value = event_sample(preview, lift=(0.07, 0.07) if raised else (0., 0.), root_x=root_x)
            value["terrain_kind"] = torch.tensor([kind])
        value["route_passed"] = torch.tensor([passed])
        return value
    for _ in range(40):
        state.update(sample(root_x=-0.15))
    for _ in range(4):
        state.update(sample(raised=True))
    state.update(sample(raised=True, root_x=0.26))
    for _ in range(3):
        result = state.update(sample(root_x=0.30))
    assert result["front_completion"].item()
    assert not result["route_completion"].item()
    for _ in range(14):
        assert not state.update(sample(root_x=0.30, passed=True))["route_completion"].item()
    result = state.update(sample(root_x=0.30, passed=True))
    assert result["route_completion"].item()
    assert result["credit"].item() == pytest.approx(state.cfg.route_completion_credit)
    assert state.update(sample(root_x=0.30, passed=True))["credit"].item() == 0.


def test_near_field_scan_resolves_eight_ten_fourteen_centimeters_without_more_inputs(preview):
    cfg = SimpleNamespace(x_positions=preview.PREVIEW_SCAN_X, y_positions=preview.PREVIEW_SCAN_Y)
    starts, directions = preview.near_field_height_pattern(cfg, "cpu")
    assert starts.shape == directions.shape == (187, 3)
    assert torch.all(directions[:, 2] == -1)
    world_x = starts[:, 0] + 0.30
    # 使用已审核URDF中立轮前缘，模拟40cm宽台阶；旧10cm均匀扫描三者完全一样。
    maps = []
    for lead in (0.08, 0.10, 0.14):
        front = 0.455741035 + lead
        maps.append((world_x >= front) & (world_x <= front + 0.40))
    assert all(not torch.equal(maps[i], maps[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert torch.allclose(starts[:, 1].view(11, 17), -starts[:, 1].view(11, 17).flip(0))


def test_takeoff_missing_force_and_birth_airborne_do_not_arm(preview):
    state = preview.PreliftRewardState(1, "cpu")
    birth = event_sample(preview, root_x=-1., lift=(0.03, 0.03))
    state.update(birth)
    assert not state.takeoff_seen.any()
    state.update(event_sample(preview))
    high = event_sample(preview, lift=(0.07, 0.07))
    high.pop("wheel_support_force_n")
    assert not state.update(high)["events"].any()


def test_far_target_lift_is_penalized_inside_perception_range(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview, root_x=0.0))
    result = state.update(event_sample(preview, root_x=0.0, lift=(0.07, 0.)))
    assert result["lead_m"][0, 0].item() == pytest.approx(0.25)
    assert not result["events"].any()
    assert result["credit"].item() == 0.
    assert result["idle_penalty"].item() > 0.


def test_transient_lift_keeps_attempt_credit_but_cannot_claim_completion(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview, root_x=0.15))
    rise = state.update(event_sample(preview, root_x=0.15, lift=(0.07, 0.)))
    drop = state.update(event_sample(preview, root_x=0.15))
    assert rise["events"][0, 0]
    assert drop["hold_broken"][0, 0]
    assert 0 < (rise["credit"] + drop["credit"]).item() < state.cfg.front_completion_credit
    assert not state.front_paid.any()
    repeat = state.update(event_sample(preview, root_x=0.15, lift=(0.07, 0.)))
    assert repeat["credit"].item() == 0.
    assert repeat["idle_penalty"].item() > 0.


@pytest.mark.parametrize("root_x,expected", [(0., 0.30), (0.20, 0.10)])
def test_horizontal_edge_refinement_removes_scan_grid_offset(preview, root_x, expected):
    sample = event_sample(preview, root_x=root_x)
    sample["terrain_kind"] = torch.tensor([0])
    hits = torch.full_like(sample["wheel_centers_w"], float("inf"))
    hits[0, [0, 2]] = torch.tensor([[0.75, 0.3, 0.0175], [0.75, -0.3, 0.0175]])
    sample = preview.refine_step_edges(sample, hits)
    result = preview.PreliftRewardState(1, "cpu").update(sample)
    assert result["lead_m"][0, [0, 2]].tolist() == pytest.approx([expected, expected], abs=1e-6)
    assert sample["edge_distance_valid"].tolist() == [[True, False, True, False]]
    assert result["action_window"][0, 0].item() is (root_x > 0)


def test_missing_step_edge_cannot_fall_back_to_false_precise_distance(preview):
    sample = event_sample(preview)
    sample["terrain_kind"] = torch.tensor([0])
    sample = preview.refine_step_edges(sample, torch.full_like(sample["wheel_centers_w"], float("inf")))
    assert not sample["preview"].any()
    assert not sample["edge_distance_valid"].any()
    assert torch.isnan(sample["edge_distance_m"]).all()
    result = preview.PreliftRewardState(1, "cpu").update(sample)
    assert not result["events"].any() and torch.isfinite(result["credit"]).all()


def test_nonstep_terrain_keeps_scan_based_preview(preview):
    sample = event_sample(preview)
    sample["terrain_kind"] = torch.tensor([-1])
    original = sample["preview"].clone()
    sample = preview.refine_step_edges(sample, torch.full_like(sample["wheel_centers_w"], float("inf")))
    assert torch.equal(sample["preview"], original)


def test_near_wheel_does_not_exempt_far_target_but_allows_rear_cooperation(preview):
    for far_front in (True, False):
        state = preview.PreliftRewardState(1, "cpu")
        sample = event_sample(preview)
        sample["wheel_centers_w"][0, 2, 0] -= 0.20
        state.update(sample)
        wheel = 2 if far_front else 1
        sample["articulation_lift_m"][0, wheel] = 0.07
        sample["clearance_m"][0, wheel] = 0.07
        result = state.update(sample)
        assert (result["idle_penalty"].item() > 0) is far_front


def test_early_hover_inside_preview_cannot_collect_event_on_window_entry(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview, root_x=0.0))
    state.update(event_sample(preview, root_x=0.0, lift=(0.07, 0.07)))
    result = state.update(event_sample(preview, lift=(0.07, 0.07)))
    assert not result["events"].any()
    assert result["credit"].item() == 0.


def test_training_and_acceptance_share_the_ten_centimeter_window(preview, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    import vision_prelift_evaluation as acceptance

    cfg = preview.PreviewCfg()
    assert cfg.min_advance_m == acceptance.PRELIFT_ACTION_MIN_ADVANCE_M
    assert cfg.action_max_advance_m == acceptance.PRELIFT_ACTION_MAX_ADVANCE_M
    assert cfg.takeoff_clearance_m == acceptance.PRELIFT_TAKEOFF_CLEARANCE_M
    assert cfg.lookahead_m == acceptance.PRELIFT_PERCEPTION_MAX_ADVANCE_M


@pytest.mark.parametrize("scenario", ["single_step", "single_step_left", "single_step_right",
                                    "pyramid_stairs_inv", "boxes"])
def test_fixed_play_terrain_includes_existing_low_stairs_and_boxes(scenario, preview, monkeypatch):
    # 只提取纯配置入口，避免导入play.py时启动Kit；原生回放另外执行冒烟验证。
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/play.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_is_preview_task", "_configure_preview_play_terrain"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {
        "_PREVIEW_TASK_SUFFIXES": ("Parallelogram-Rover-Vision-Teacher-Preview-v0",),
        "_PREVIEW_SCENARIOS": {"single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes"},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    monkeypatch.setitem(sys.modules, "robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview", preview)
    selected = SimpleNamespace(proportion=0.05, step_height_range=(0.08, 0.12), grid_height_range=(0.06, 0.12))
    selected.height_levels, selected.step_counts = (.08, .12, .16), (2, 3, 4)
    selected.first_edge_m, selected.step_width = 1.2, .4
    selected.copy = lambda: copy.copy(selected)
    terrain = SimpleNamespace(sub_terrains={scenario: selected})
    cfg = SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=terrain)),
                          curriculum=SimpleNamespace(terrain_levels=object()))
    scope["_configure_preview_play_terrain"](
        cfg, "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0", scenario, 0.5, 4, 142,
    )
    assert terrain.difficulty_range == (0.5, 0.5)
    assert terrain.num_rows == 1 and terrain.num_cols == 4
    assert list(terrain.sub_terrains) == [scenario]
    assert terrain.sub_terrains[scenario].proportion == 1.
    assert selected.proportion == 0.05
    assert cfg.curriculum.terrain_levels is None


def test_preview_route_progress_rejects_sideways_and_turning(preview):
    origins = torch.zeros((4, 3))
    positions = torch.tensor([[0.0, 3.0, 0.0], [2.6, 0.1, 0.0],
                              [2.6, 0.1, 0.0], [2.6, 0.8, 0.0]])
    upright = torch.tensor([True, True, False, True])
    alignment = torch.tensor([1.0, 1.0, 1.0, 0.0])
    passed, failed = preview.preview_route_outcome(positions, origins, upright, alignment)
    assert passed.tolist() == [False, True, False, False]
    assert failed.tolist() == [True, False, True, True]
    off_lane, _ = preview.preview_route_outcome(
        torch.tensor([[2.6, 0.25, 0.0]]), torch.zeros((1, 3)), torch.tensor([True]), torch.tensor([1.0]),
    )
    assert not off_lane.item()


def test_preview_reward_uses_world_route_not_body_forward(preview):
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview_reward.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "preview_velocity_tracking"]
    scope = {"torch": torch, "centered_velocity_tracking": preview.centered_velocity_tracking}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    data = SimpleNamespace(root_lin_vel_w=torch.tensor([[0.4, 0.0], [0.0, 0.4]]),
                           root_quat_w=torch.tensor([[1., 0., 0., 0.], [0.7071068, 0., 0., 0.7071068]]),
                           projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]))
    robot = SimpleNamespace(data=data)
    command = SimpleNamespace(get_command=lambda name: torch.tensor([[0.4, 0., 0.], [0.4, 0., 0.]]))
    env = SimpleNamespace(scene={"robot": robot}, command_manager=command)
    reward = scope["preview_velocity_tracking"](env, 0.5, "base_velocity")
    assert reward[0].item() == pytest.approx(1.)
    assert reward[1].item() < 0.


@pytest.mark.parametrize("yaw", [0., -math.pi / 2, math.pi / 2, math.pi, 0.1])
def test_route_command_observation_reveals_heading_without_adding_dimensions(yaw):
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview_reward.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "preview_route_command"]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    quat = torch.tensor([[math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]])
    command = torch.tensor([[0.4, 0., 0.]])
    data = SimpleNamespace(root_quat_w=quat)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                          command_manager=SimpleNamespace(get_command=lambda _: command))
    observed = scope["preview_route_command"](env, "base_velocity")
    assert observed.shape == command.shape
    assert observed.tolist()[0] == pytest.approx([0.4 * math.cos(yaw), -0.4 * math.sin(yaw), 0.], abs=1e-6)
    # 观测变换不能原位修改CommandManager，否则奖励的世界路线也会随车转动。
    assert command.tolist()[0] == pytest.approx([0.4, 0., 0.])


@pytest.mark.parametrize("student", [False, True])
def test_route_observation_is_shared_by_actor_critic_and_distillation_teacher(student):
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview_env_cfg.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "configure_preview_observations"]
    function = object()
    scope = {"preview_route_command": function}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    names = ("policy", "critic", "teacher") if student else ("policy", "critic")
    observations = SimpleNamespace(**{
        name: SimpleNamespace(velocity_commands=SimpleNamespace(func=object(),
                                                               params={"command_name": "base_velocity"}))
        for name in names
    })
    scope["configure_preview_observations"](observations)
    for name in names:
        assert getattr(observations, name).velocity_commands.func is function
        assert getattr(observations, name).velocity_commands.params == {"command_name": "base_velocity"}


def test_yaw_regularization_has_no_stationary_bonus_and_is_symmetric():
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview_reward.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "preview_yaw_rate_l2"]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    data = SimpleNamespace(root_ang_vel_w=torch.tensor([[1., 2., 0.], [0., 0., 0.4], [0., 0., -0.4]]))
    values = scope["preview_yaw_rate_l2"](SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}))
    assert values.tolist() == pytest.approx([0., 0.16, 0.16])


def test_preview_reward_selection_preserves_safety_params_and_removes_conflicts():
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview_env_cfg.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "configure_preview_rewards"]
    scope = {"RewardTermCfg": SimpleNamespace, "mdp_core": SimpleNamespace(flat_orientation_l2=object()),
             "preview_velocity_tracking": object(), "preview_yaw_rate_l2": object(),
             "AnticipatoryWheelLift": object()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    # 不只是对照权重表：未知继承项必须关闭，保留项不能丢掉joint/sensor筛选参数。
    keep = ("track_lin_vel_xy_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "joint_torques_l2",
            "joint_pos_limits", "action_rate_l2", "joint_deviation_steering_l1", "undesired_contacts",
            "contact_forces")
    remove = ("joint_pos_penalty", "track_ang_vel_z_exp", "upward", "stand_still",
              "feet_contact_without_cmd", "joint_acc_l2", "joint_vel_steering_l2", "joint_power",
              "future_inherited_reward")
    rewards = SimpleNamespace(**{name: SimpleNamespace(weight=1., func=object(), params={"selector": name})
                                 for name in keep + remove}, disabled=None)
    before = {name: getattr(rewards, name) for name in keep}
    scope["configure_preview_rewards"](rewards)
    assert all(getattr(rewards, name) is None for name in remove)
    assert all(getattr(rewards, name) is before[name] for name in keep)
    assert all(getattr(rewards, name).params == {"selector": name} for name in keep)
    active = {name: term for name, term in vars(rewards).items() if term is not None}
    assert len(active) == 12
    assert {name for name, term in active.items() if term.weight > 0} == {
        "track_lin_vel_xy_exp", "anticipatory_wheel_lift",
    }
    assert all(math.isfinite(term.weight) and term.weight != 0 for term in active.values())
    assert rewards.track_lin_vel_xy_exp.func is scope["preview_velocity_tracking"]
    # 重复配置仍产生同一目标，而不是把已添加的三项误删后丢失。
    scope["configure_preview_rewards"](rewards)
    assert {name for name, term in vars(rewards).items() if term is not None} == set(active)


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("ids", [torch.arange(3), slice(None)])
def test_preview_curriculum_changes_level_after_two_stable_routes_without_action_gate(preview, completed, ids):
    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview_reward.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "preview_route_curriculum"]
    scope = {"torch": torch, "preview_route_outcome": preview.preview_route_outcome,
             "preview_route_targets": preview.preview_route_targets,
             "preview_route_promotion_mask": preview.preview_route_promotion_mask,
             "terrain_column_names": preview.terrain_column_names,
             "sample_preview_terrain_columns": preview.sample_preview_terrain_columns}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    names = ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes", "flat")
    proportions = (8, 4, 4, 3, 3, 2)
    config = SimpleNamespace(num_cols=24, curriculum=True,
                             sub_terrains={name: SimpleNamespace(proportion=p) for name, p in zip(names, proportions)})
    terrain = SimpleNamespace(terrain_levels=torch.tensor([1, 1, 1]), terrain_types=torch.zeros(3, dtype=torch.long),
                              env_origins=torch.zeros((3, 3)), terrain_origins=torch.zeros((9, 24, 3)),
                              cfg=SimpleNamespace(terrain_generator=config))
    robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[0., 3., 0.], [2.6, 0., 0.], [2.6, 0., 0.]]),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]] * 3),
        projected_gravity_b=torch.tensor([[0., 0., -1.]] * 3),
    ))
    # SimpleNamespace不代理特殊方法，测试中使用显式场景类型模拟Isaac的scene访问。
    class Scene:
        def __init__(self):
            self.terrain = terrain

        def __getitem__(self, name):
            return robot

    state = preview.PreliftRewardState(3, "cpu")
    state.front_paid[:] = completed
    state.completed[:] = completed
    state.route_hold_s[:] = 0.40
    env = SimpleNamespace(num_envs=3, scene=Scene(), episode_length_buf=torch.tensor([100, 100, 0]),
                          reward_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(
                              func=SimpleNamespace(state=state))))
    scope["preview_route_curriculum"](env, ids)
    assert terrain.terrain_levels.tolist() == [0, 1, 1]
    terrain.terrain_types.zero_()
    scope["preview_route_curriculum"](env, ids)
    assert terrain.terrain_levels.tolist() == [0, 2, 1]
    # 课程在奖励reset之前只读取完成证据，不能提前清空后续日志所需状态。
    assert state.front_paid.tolist() == [completed] * 3


def test_preview_curriculum_requires_stage_obstacles(preview):
    columns = ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes", "flat")
    levels = torch.tensor([0, 0, 3, 3, 6, 6])
    terrain_types = torch.tensor([5, 0, 0, 2, 5, 3])
    assert preview.preview_route_promotion_mask(levels, terrain_types, columns).tolist() == [
        False, True, True, True, False, True,
    ]


def test_curriculum_does_not_require_both_narrow_sides_or_both_active_wheels(preview):
    columns = ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes", "flat")
    levels = torch.tensor([2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6])
    types = torch.tensor([0, 0, 1, 2, 1, 2, 1, 2, 0, 1, 3, 4])
    assert preview.preview_route_promotion_mask(levels, types, columns).tolist() == [
        False, False, True, True, True, True, False, False, True, True, True, True,
    ]


def test_preview_curriculum_samples_stage_terrain_families(preview):
    names = ("single_step", "single_step_left", "single_step_right",
             "pyramid_stairs_inv", "boxes", "flat")
    proportions = (8, 4, 4, 3, 3, 2)
    columns = preview.terrain_column_names(names, proportions, 24, curriculum=True)
    assert set(columns) == set(names)
    levels = torch.tensor([0] * 200 + [4] * 200 + [8] * 200)
    torch.manual_seed(142)
    selected = preview.sample_preview_terrain_columns(levels, columns)
    sampled = [columns[index] for index in selected.tolist()]
    assert set(sampled[:200]) == set(names)
    assert {"single_step_left", "single_step_right"} <= set(sampled[200:400])
    assert {"pyramid_stairs_inv", "boxes"} <= set(sampled[400:])


def test_four_car_mixed_play_includes_stairs():
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/play.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_is_preview_task", "_configure_preview_play_terrain"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"_PREVIEW_TASK_SUFFIXES": ("Parallelogram-Rover-Vision-Teacher-Preview-v0",),
             "_PREVIEW_SCENARIOS": {"single_step", "single_step_left", "single_step_right",
                                    "pyramid_stairs_inv", "boxes"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    terrains = {}
    for name in ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes", "flat"):
        value = SimpleNamespace(proportion=1., step_height_range=(0.08, 0.12), grid_height_range=(0.06, 0.12))
        value.copy = lambda value=value: copy.copy(value)
        terrains[name] = value
    terrain = SimpleNamespace(sub_terrains=terrains)
    cfg = SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=terrain,
                                                                          max_init_terrain_level=None)),
                          curriculum=SimpleNamespace(terrain_levels=object()))
    scope["_configure_preview_play_terrain"](
        cfg, "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0", "mixed", 0., 4, 142,
    )
    assert list(terrain.sub_terrains)[:4] == ["single_step", "single_step_left",
                                           "single_step_right", "pyramid_stairs_inv"]
    assert terrain.num_cols == 5
    assert cfg.curriculum.terrain_levels is None


def test_prelift_hold_continues_inside_last_eight_centimeters(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    for _ in range(3):
        state.update(event_sample(preview, lift=(0.07, 0.07)))
    sample = event_sample(preview, lift=(0.07, 0.07), root_x=0.21)
    result = state.update(sample)
    assert 0 < result["lead_m"][0, 0] < state.cfg.min_advance_m
    assert not result["hold_broken"].any()


def test_prelift_dropped_before_edge_never_gets_completion_after_relift(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    for _ in range(3):
        state.update(event_sample(preview, lift=(0.07, 0.07)))
    dropped = state.update(event_sample(preview))
    assert dropped["hold_broken"][0, [0, 2]].all()
    for _ in range(4):
        assert not state.update(event_sample(preview, lift=(0.07, 0.07)))["events"].any()
    state.update(event_sample(preview, lift=(0.07, 0.07), root_x=0.26))
    for _ in range(4):
        assert not state.update(event_sample(preview, root_x=0.30))["front_completion"].any()


def test_grounded_suspension_preparation_is_not_a_takeoff_penalty(preview):
    sample = preview.preview_measurements(**scene(height=0.0))
    state = preview.PreliftRewardState(1, "cpu")
    sample["articulation_lift_m"][:, [0, 2]] = 0.005
    assert state.update(sample)["idle_penalty"].item() == 0
    sample["articulation_lift_m"][:, [0, 2]] = 0.07
    sample["clearance_m"].zero_()
    assert state.update(sample)["idle_penalty"].item() == 0


def test_drop_on_crossing_frame_cannot_escape_hold_failure(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    for _ in range(3):
        state.update(event_sample(preview, lift=(0.07, 0.07)))
    result = state.update(event_sample(preview, root_x=0.26))
    assert result["hold_broken"][0, [0, 2]].all()
    assert not state.passed_front_edge.any()
    assert result["credit"].item() == pytest.approx(-2 * state.cfg.hold_violation_penalty)
    assert state.update(event_sample(preview, root_x=0.26))["credit"].item() == 0.


def test_holding_survives_deceleration_and_local_ground_switch(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    for _ in range(3):
        state.update(event_sample(preview, lift=(0.07, 0.07)))
    sample = event_sample(preview, lift=(0.07, 0.07), root_x=0.26, eligible=False)
    sample["eligible_gates"]["forward_motion"].fill_(False)
    sample["clearance_m"][:, [0, 2]] = -0.01
    result = state.update(sample)
    assert not result["hold_broken"].any()
    assert result["held_to_edge"][0, [0, 2]].all()


@pytest.mark.parametrize("dt,frames,expected", [(0.01, 2, False), (0.01, 6, True), (0.02, 3, True)])
def test_hold_duration_uses_control_time_not_fixed_steps(preview, dt, frames, expected):
    state = preview.PreliftRewardState(1, "cpu")
    grounded = event_sample(preview)
    grounded["control_dt_s"] = dt
    state.update(grounded)
    for frame in range(frames):
        sample = event_sample(preview, lift=(0.07, 0.07), root_x=0.26 if frame == frames - 1 else 0.15)
        sample["control_dt_s"] = dt
        result = state.update(sample)
    assert result["held_to_edge"][0, 0].item() is expected
    assert result["hold_broken"][0, 0].item() is not expected


@pytest.mark.parametrize("fault", [None, "missing_force", "unsupported", "high", "bounce"])
def test_release_needs_real_continuous_contact(preview, fault):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    for _ in range(3):
        state.update(event_sample(preview, lift=(0.07, 0.07)))
    state.update(event_sample(preview, lift=(0.07, 0.07), root_x=0.26))
    for frame in range(3):
        sample = event_sample(preview, lift=(0.07, 0.07) if fault == "high" else (0., 0.), root_x=0.30)
        if fault != "missing_force":
            sample["wheel_support_force_n"] = torch.full((1, 4), 20.)
        else:
            sample.pop("wheel_support_force_n")
        if fault == "unsupported":
            sample["eligible_gates"]["supported"].fill_(False)
        if fault == "bounce" and frame == 1:
            sample["wheel_support_force_n"].zero_()
        result = state.update(sample)
        assert result["front_completion"].item() is (fault is None and frame == 2)
    assert state.release_paid[0, 0].item() is (fault is None)


def test_event_reward_rejects_constant_hover_and_caps_repeated_lifts(preview):
    state = preview.PreliftRewardState(1, "cpu")
    high = event_sample(preview, lift=(0.07, 0.0))
    assert sum(state.update(high)["credit"].item() for _ in range(80)) == 0
    state.reset()
    state.update(event_sample(preview))
    first = state.update(high)["credit"].item()
    assert first > 0
    assert sum(state.update(high)["credit"].item() for _ in range(80)) == 0
    for _ in range(10):
        state.update(event_sample(preview))
        assert state.update(high)["credit"].item() == 0


def test_front_completion_remembers_sequential_not_simultaneous_lifts(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    left = state.update(event_sample(preview, lift=(0.07, 0.0)))
    for _ in range(3):
        right = state.update(event_sample(preview, lift=(0.07, 0.07)))
    assert not left["front_completion"].item()
    assert not right["front_completion"].item()
    assert state.completed[0, [0, 2]].tolist() == [True, True]
    crossed = state.update(event_sample(preview, lift=(0.07, 0.07), root_x=0.26))
    assert crossed["held_to_edge"][0, [0, 2]].all()
    for frame in range(3):
        landed = event_sample(preview, root_x=0.30)
        landed["wheel_support_force_n"] = torch.full((1, 4), 20.)
        result = state.update(landed)
        assert result["front_completion"].item() is (frame == 2)
    assert result["credit"].item() == pytest.approx(state.cfg.front_completion_credit, abs=2e-6)
    assert not state.update(landed)["front_completion"].item()


@pytest.mark.parametrize("first", [0, 1])
def test_full_width_sequential_release_matches_independent_acceptance(preview, monkeypatch, first):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    reward = preview.PreliftRewardState(1, "cpu")
    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    evaluation = PreliftEvaluation(1, 0.02, require_hold_to_edge=True, required_active_count=1)
    payments = []

    def advance(leads, lifts, clearances, forces, *, done=False):
        order = (first, 1 - first)
        values = [tuple(pair[order.index(wheel)] for wheel in range(2))
                  for pair in (leads, lifts, clearances, forces)]
        leads, lifts, clearances, forces = values
        sample = event_sample(preview, lift=lifts)
        # 状态级轨迹模拟同一前缘的两轮前后错位，落在台阶上仍有进场高度但局部净空为零。
        sample["wheel_centers_w"][0, [0, 2], 0] = (
            sample["obstacle_xy"][0, [0, 2], 0] - sample["wheel_radii"][[0, 2]] - torch.tensor(leads)
        )
        sample["clearance_m"][0, [0, 2]] = torch.tensor(clearances)
        sample["wheel_support_force_n"] = torch.full((1, 4), 20.)
        sample["wheel_support_force_n"][0, [0, 2]] = torch.tensor(forces)
        result = reward.update(sample)
        payments.append(int(result["front_completion"].item()))
        traversal.update([True], [False], [done], [True])
        evaluation.update([{
            "lead_m": result["lead_m"][0, [0, 2]].tolist(), "lift_m": lifts,
            "clearance_m": lifts, "reward_clearance_m": clearances,
            "wheel_support_force_n": forces, "eligible": True, "hold_eligible": True,
        }], traversal)
        state = evaluation.records[0] if done else evaluation.pending[0]
        for buffer, pending_key, record_key in (
            (reward.passed_front_edge, "held_to_edge", "front_wheels_held_to_edge"),
            (reward.release_paid, "released", "front_wheels_released"),
            (reward.hold_broken, "hold_broken", "front_wheel_hold_failures"),
        ):
            assert buffer[0, [0, 2]].tolist() == state[record_key if done else pending_key]
        assert sum(force == 0. for force in forces) <= 1

    advance((0.35, 0.55), (0., 0.), (0., 0.), (20., 20.))
    for _ in range(3):
        advance((0.10, 0.30), (0.07, 0.), (0.07, 0.), (0., 20.))
    advance((0.04, 0.24), (0.07, 0.), (0.07, 0.), (0., 20.))
    advance((-0.01, 0.19), (0.07, 0.), (0.07, 0.), (0., 20.))
    for _ in range(3):
        advance((-0.05, 0.15), (0.07, 0.), (0., 0.), (20., 20.))
    assert reward.release_paid[0, [0, 2]].tolist() == [wheel == first for wheel in range(2)]
    assert evaluation.pending[0]["lifted"][1 - first] is None
    assert reward.front_paid.item()
    for _ in range(3):
        advance((-0.08, 0.12), (0.07, 0.07), (0., 0.07), (20., 0.))
    advance((-0.16, 0.04), (0.07, 0.07), (0., 0.07), (20., 0.))
    advance((-0.21, -0.01), (0.07, 0.07), (0., 0.07), (20., 0.))
    for frame in range(3):
        advance((-0.23, -0.03), (0.07, 0.07), (0., 0.), (20., 20.), done=frame == 2)
    assert sum(payments) == 1
    assert reward.front_paid.item()
    assert evaluation.summary(traversal.summary(0.70))["passed"]


def test_rear_wheel_lift_does_not_replace_front_prelift_credit(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    data = scene()
    data["wheel_centers_w"][:, 1, 2] += 0.07
    result = state.update(preview.preview_measurements(**data))
    assert not result["events"][0, 1].item()
    assert result["credit"].item() == 0


def test_front_completion_credit_exceeds_single_event_credit(preview):
    cfg = preview.PreviewCfg()
    assert cfg.front_completion_credit > cfg.wheel_event_credit
    # 全宽障碍的双前轮总额度应显著高于单侧满额，避免单侧悬轮成为局部最优。
    single = cfg.progress_credit + cfg.wheel_event_credit
    assert 2 * single + cfg.front_completion_credit >= 3 * single


def test_early_hover_then_entering_window_is_not_a_new_event(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview, root_x=-0.8))
    state.update(event_sample(preview, root_x=-0.8, lift=(0.07, 0.07)))
    assert state.update(event_sample(preview, lift=(0.07, 0.07)))["credit"].item() == 0


def test_crossing_then_retreat_does_not_rearm_event(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    state.update(event_sample(preview, root_x=0.5))
    assert state.update(event_sample(preview, lift=(0.07, 0.07)))["credit"].item() == 0


def test_reward_reset_isolates_episodes(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    state.update(event_sample(preview, lift=(0.07, 0.07)))
    state.reset([0])
    assert not state.completed.any()
    assert state.update(event_sample(preview, lift=(0.07, 0.07)))["credit"].item() == 0


def test_ineligible_transition_consumed_without_delayed_payment(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    assert state.update(event_sample(preview, lift=(0.07, 0.07), eligible=False))["credit"].item() == 0
    assert state.update(event_sample(preview, lift=(0.07, 0.07)))["credit"].item() == 0


@pytest.mark.parametrize("corner", [0, 2])
def test_narrow_lane_pays_only_affected_side(preview, corner):
    state = preview.PreliftRewardState(1, "cpu")
    for lift in (0., 0.07):
        data = scene(lift=lift)
        data["root_pos_w"][:, 0] += 0.15
        data["wheel_centers_w"][:, :, 0] += 0.15
        y = data["ray_hits_w"][..., 1]
        data["ray_hits_w"][..., 2] *= (y > 0) if corner == 0 else (y < 0)
        sample = preview.preview_measurements(**data)
        sample["wheel_support_force_n"] = torch.where(sample["clearance_m"] >= 0.005, 0., 20.)
        result = state.update(sample)
    assert result["events"][0, corner]
    assert result["events"].sum().item() == 1
    assert not result["front_completion"].item()


def test_invalid_scan_after_acquisition_cannot_trigger_event(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(event_sample(preview))
    data = scene(lift=0.07)
    data["ray_hits_w"].fill_(float("nan"))
    result = state.update(preview.preview_measurements(**data))
    assert result["credit"].item() == 0
    assert torch.isfinite(result["idle_penalty"]).all()


def test_partial_reset_does_not_clear_other_environment(preview):
    state = preview.PreliftRewardState(2, "cpu")
    def batch(sample):
        return {key: ({name: value.repeat(2) for name, value in item.items()} if isinstance(item, dict)
                      else item if key == "wheel_radii" else item.repeat(2, *([1] * (item.ndim - 1))))
                for key, item in sample.items()}
    state.update(batch(event_sample(preview)))
    state.update(batch(event_sample(preview, lift=(0.07, 0.07))))
    state.reset([0])
    result = state.update(batch(event_sample(preview, lift=(0.07, 0.07))))
    assert result["credit"].tolist() == [0., 0.]
    assert state.completed[:, [0, 2]].tolist() == [[False, False], [True, True]]


def test_flat_hover_penalized_but_grounded_wheels_not(preview):
    state = preview.PreliftRewardState(1, "cpu")
    low = state.update(preview.preview_measurements(**scene(height=0.0)))
    high = state.update(preview.preview_measurements(**scene(height=0.0, lift=0.07)))
    assert low["idle_penalty"].item() == pytest.approx(0., abs=1e-5)
    assert high["idle_penalty"].item() > 0
    assert high["credit"].item() == 0


def narrow_sample(preview, *, kind=1, raised=False, supported=True, force=None, root_x=0.15):
    sample = event_sample(preview, root_x=root_x)
    sample["terrain_kind"] = torch.tensor([kind])
    sample["wheel_support_force_n"] = torch.tensor([[20., 20., 20., 20.]])
    sample["wheel_resisting_force_n"] = torch.zeros(1, 4)
    sample["narrow_supported"] = torch.tensor([supported])
    sample["control_dt_s"] = 0.02
    target = 0 if kind == 1 else 2
    if raised:
        sample["wheel_support_force_n"][0, target] = 10.
        sample["clearance_m"][0, target] = 0.006
        sample["wheel_centers_w"][0, target, 2] += 0.006
    if force is not None:
        sample["wheel_support_force_n"][0, target] = force
    return sample


@pytest.mark.parametrize("kind,target", [(1, 0), (2, 2)])
def test_narrow_unloading_requires_stable_baseline_and_three_clearance_samples(preview, kind, target):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state, kind=kind)
    raised = narrow_sample(preview, kind=kind, raised=True)
    assert not state.update(raised)["events"].any()
    assert not state.update(raised)["events"].any()
    result = state.update(raised)
    assert result["events"].sum() == 1 and result["events"][0, target]
    assert not result["front_completion"].item()
    assert result["credit"].item() == pytest.approx(state.cfg.wheel_event_credit)
    assert state.update(raised)["credit"].item() == 0.
    crossed = state.update(narrow_sample(preview, kind=kind, raised=True, root_x=0.26))
    assert crossed["held_to_edge"][0, target]
    for frame in range(3):
        result = state.update(narrow_sample(preview, kind=kind, root_x=0.30, force=2.))
        assert result["front_completion"].item() is (frame == 2)
    assert result["credit"].item() == pytest.approx(state.cfg.front_completion_credit, abs=2e-6)
    state.reset()
    assert not state.baseline_samples.any() and not state.unprepared_collision_seen.any()
    for _ in range(4):
        assert not state.update(raised)["events"].any()


@pytest.mark.parametrize("kind,target", [(1, 0), (2, 2)])
@pytest.mark.parametrize("grounded_lift", [0.03, 0.089])
def test_narrow_grounded_articulation_lift_still_arms_unloading(preview, kind, target, grounded_lift):
    state = preview.PreliftRewardState(1, "cpu")
    for _ in range(40):
        sample = narrow_sample(preview, kind=kind, root_x=-0.15)
        sample["articulation_lift_m"][0, target] = grounded_lift
        sample["clearance_m"][0, target] = 0.002
        state.update(sample)
    assert state.baseline_samples[0, target].item() == 6
    assert state.baseline_force[0, target].item() == pytest.approx(20.)
    assert state.low_seen[0, target]

    for frame in range(3):
        sample = narrow_sample(preview, kind=kind, raised=True)
        sample["articulation_lift_m"][0, target] = grounded_lift
        result = state.update(sample)
        assert result["events"].sum().item() == int(frame == 2)
    assert result["events"][0, target]
    assert not result["front_completion"].item()
    assert result["credit"].item() == pytest.approx(state.cfg.wheel_event_credit)
    assert state.update(sample)["credit"].item() == 0.


@pytest.mark.parametrize("kind,target", [(1, 0), (2, 2)])
@pytest.mark.parametrize("fault", ["hover", "no_load", "blocked", "invalid_clearance", "invalid_lift"])
def test_narrow_grounded_baseline_rejects_invalid_low_states(preview, kind, target, fault):
    state = preview.PreliftRewardState(1, "cpu")
    for _ in range(40):
        sample = narrow_sample(preview, kind=kind, root_x=0.32 if fault == "blocked" else -0.15)
        sample["articulation_lift_m"][0, target] = 0.03
        sample["clearance_m"][0, target] = 0.006 if fault == "hover" else 0.002
        if fault == "no_load":
            sample["wheel_support_force_n"][0, target] = 0.
        elif fault == "blocked":
            sample["wheel_resisting_force_n"][0, target] = 20.
        elif fault == "invalid_clearance":
            sample["clearance_m"][0, target] = float("nan")
        elif fault == "invalid_lift":
            sample["articulation_lift_m"][0, target] = float("nan")
        result = state.update(sample)
        assert not result["events"].any()
    assert state.baseline_samples[0, target].item() == 0
    assert not state.low_seen[0, target]


@pytest.mark.parametrize("kind,target", [(1, 0), (2, 2)])
@pytest.mark.parametrize("grounded_lift", [0.03, 0.089])
@pytest.mark.parametrize("fault", [None, "no_clearance", "unsupported", "blocked", "late"])
def test_narrow_reward_and_evaluation_match_grounded_unloading(
    preview, monkeypatch, kind, target, grounded_lift, fault,
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    reward = preview.PreliftRewardState(1, "cpu")
    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    front = 0 if target == 0 else 1
    evaluation = PreliftEvaluation(1, 0.02, required_front_wheels=(front,), allow_load_drop=True)
    for frame in range(43):
        raised = frame >= 40
        root_x = (0.32 if raised and fault == "blocked" else 0.5 if raised and fault == "late"
                  else 0.15 if raised else -0.15)
        sample = narrow_sample(
            preview, kind=kind, raised=raised, supported=not (raised and fault == "unsupported"), root_x=root_x,
        )
        sample["articulation_lift_m"][0, target] = grounded_lift
        if not raised or fault == "no_clearance":
            sample["clearance_m"][0, target] = 0.002
        if raised and fault == "blocked":
            sample["wheel_resisting_force_n"][0, target] = 20.
        result = reward.update(sample)
        previous = [event is not None for event in evaluation.pending[0]["lifted"]]
        traversal.update([True], [False], [False], [True])
        evaluation.update([{
            "lead_m": result["lead_m"][0, [0, 2]].tolist(),
            "lift_m": sample["articulation_lift_m"][0, [0, 2]].tolist(),
            "clearance_m": sample["clearance_m"][0, [0, 2]].tolist(),
            "wheel_support_force_n": sample["wheel_support_force_n"][0, [0, 2]].tolist(),
            "blocking_collision": [raised and fault == "blocked" and wheel == front for wheel in range(2)],
            "reward_eligible": bool(sample["eligible"][0]),
            "eligible": bool(sample["eligible"][0] & sample["narrow_supported"][0]),
        }], traversal)
        events = [event is not None and not previous[wheel]
                  for wheel, event in enumerate(evaluation.pending[0]["lifted"])]
        assert result["events"][0, [0, 2]].tolist() == events
        assert reward.baseline_samples[0, [0, 2]].tolist() == evaluation.pending[0]["baseline_samples"]
    assert bool(reward.completed[0, target]) is (fault is None)
    assert not reward.front_paid[0]  # 提前事件不等于已保持到前缘并落轮。


@pytest.mark.parametrize("kind,target", [(1, 0), (2, 2)])
@pytest.mark.parametrize("fault", [None, "drop", "low_crossing", "no_release", "bounce", "slow_ground_switch"])
def test_narrow_complete_sequence_matches_independent_acceptance(preview, monkeypatch, kind, target, fault):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    reward = preview.PreliftRewardState(1, "cpu")
    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    front = 0 if target == 0 else 1
    evaluation = PreliftEvaluation(
        1, 0.02, required_front_wheels=(front,), allow_load_drop=True, require_hold_to_edge=True,
    )
    completion_payments = []

    def advance(*, raised=False, root_x=0.15, force=None, done=False, switched_ground=False):
        sample = narrow_sample(preview, kind=kind, raised=raised, root_x=root_x, force=force)
        sample["articulation_lift_m"][0, target] = 0.03
        approach_clearance = sample["clearance_m"][0, [0, 2]].tolist()
        if switched_ground:
            # 局部扫描已命中障碍顶面，保持仍相对进场地面检查；减速不得中断保持。
            sample["clearance_m"][0, target] = -0.01
            sample["eligible"].fill_(False)
            sample["eligible_gates"]["forward_motion"].fill_(False)
        result = reward.update(sample)
        completion_payments.append(int(result["front_completion"].item()))
        traversal.update([True], [False], [done], [True])
        evaluation.update([{
            "lead_m": result["lead_m"][0, [0, 2]].tolist(),
            "lift_m": sample["articulation_lift_m"][0, [0, 2]].tolist(),
            "clearance_m": approach_clearance,
            "reward_clearance_m": sample["clearance_m"][0, [0, 2]].tolist(),
            "wheel_support_force_n": sample["wheel_support_force_n"][0, [0, 2]].tolist(),
            "eligible": bool(sample["eligible"][0]), "hold_eligible": True,
        }], traversal)
        state = evaluation.records[0] if done else evaluation.pending[0]
        for buffer, key in ((reward.passed_front_edge, "front_wheels_held_to_edge" if done else "held_to_edge"),
                            (reward.release_paid, "front_wheels_released" if done else "released"),
                            (reward.hold_broken, "front_wheel_hold_failures" if done else "hold_broken")):
            assert buffer[0, target].item() is state[key][front]
        return result

    for _ in range(40):
        advance(root_x=-0.15)
    for _ in range(3):
        advance(raised=True)
    if fault == "drop":
        advance(root_x=0.10)
        advance(raised=True, root_x=0.12)
    advance(raised=True, root_x=0.21)
    advance(raised=fault != "low_crossing", root_x=0.26, switched_ground=fault == "slow_ground_switch")
    for frame in range(3):
        advance(raised=fault == "no_release", root_x=0.30,
                force=0. if fault == "bounce" and frame == 1 else 2., done=frame == 2)
    expected = fault in (None, "slow_ground_switch")
    assert reward.front_paid[0].item() is expected
    assert sum(completion_payments) == int(expected)
    assert evaluation.records[0]["success"] is expected


@pytest.mark.parametrize("fault", ["bounce", "no_clearance", "nan", "missing", "negative", "unsupported", "wrong_side"])
def test_narrow_rejects_false_unloading_events(preview, fault):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state, kind=1)
    for frame in range(4):
        sample = narrow_sample(preview, raised=True, supported=fault != "unsupported")
        if fault == "bounce" and frame % 2:
            sample = narrow_sample(preview)
        if fault == "no_clearance":
            sample["clearance_m"].zero_()
        if fault in {"nan", "negative"}:
            sample["wheel_support_force_n"][0, 0] = float("nan") if fault == "nan" else -1.
        if fault == "missing":
            sample.pop("wheel_support_force_n")
        if fault == "wrong_side":
            sample["wheel_support_force_n"] = sample["wheel_support_force_n"][:, [2, 1, 0, 3]]
            sample["clearance_m"] = sample["clearance_m"][:, [2, 1, 0, 3]]
        result = state.update(sample)
        assert not result["events"].any()
        assert torch.isfinite(result["credit"]).all()
        if fault in {"nan", "missing", "negative", "unsupported"}:
            assert result["credit"].item() == 0.


def test_blocking_force_is_recorded_once_and_not_position_only(preview):
    state = preview.PreliftRewardState(1, "cpu")
    state.update(narrow_sample(preview))
    near = narrow_sample(preview, root_x=0.32)
    assert not state.update(near)["unprepared_collision"].any()
    near["wheel_resisting_force_n"][0, 0] = 20.
    result = state.update(near)
    assert result["unprepared_collision"][0, 0]
    assert result["credit"].item() == 0.
    assert state.update(near)["credit"].item() == 0.
    for _ in range(3):
        assert not state.update(narrow_sample(preview, raised=True))["events"].any()


def test_prepared_wheel_is_not_logged_as_unprepared_collision(preview):
    state = preview.PreliftRewardState(1, "cpu")
    prime_baseline(preview, state, kind=1)
    for _ in range(3):
        state.update(narrow_sample(preview, raised=True))
    near = narrow_sample(preview, root_x=0.32)
    near["wheel_resisting_force_n"][0, 0] = 20.
    assert not state.update(near)["unprepared_collision"].any()


@pytest.mark.parametrize("forces,expected", [([0, 40, 40, 0], True), ([40, 0, 0, 40], True),
    ([40, 40, 0, 0], False), ([0, 0, 0, 0], False), ([0, 2, 2, 0], False),
    ([20, 20, 20, 0], True), ([float("nan"), 40, 40, 0], False)])
def test_narrow_support_requires_diagonal_or_three_wheels_and_half_weight(preview, forces, expected):
    assert preview.narrow_support_mask(torch.tensor([forces]), torch.tensor([100.])).item() is expected


def test_column_types_match_isaac_curriculum_and_reject_random_mixed_layout(preview):
    names = ["single_step", "single_step_left", "single_step_right", "stairs", "boxes", "flat"]
    proportions = [0.60, 0.15, 0.15, 0.05, 0.03, 0.02]
    assert preview.terrain_column_kinds(names, proportions, 20, curriculum=True).tolist() == (
        [0] * 12 + [1] * 3 + [2] * 3 + [-1] * 2
    )
    assert preview.terrain_column_kinds(["single_step_right"], [1.], 4, curriculum=False).tolist() == [2] * 4
    with pytest.raises(ValueError, match="curriculum"):
        preview.terrain_column_kinds(names, proportions, 20, curriculum=False)


def test_reset_logs_separate_target_event_and_collision_rates(preview, monkeypatch):
    from types import ModuleType, SimpleNamespace

    package = ModuleType("preview_reward_test")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, package.__name__ + ".vision_preview", preview)
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = object
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    path = Path(preview.__file__).with_name("vision_preview_reward.py")
    spec = importlib.util.spec_from_file_location(package.__name__ + ".vision_preview_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward = module.AnticipatoryWheelLift.__new__(module.AnticipatoryWheelLift)
    reward._env = SimpleNamespace(extras={}, step_dt=0.02)
    reward.preview_cfg = preview.PreviewCfg()
    reward.state = preview.PreliftRewardState(4, "cpu")
    reward.state.terrain_kind[:] = torch.tensor([0, 1, 2, 1])
    reward.state.seen[:, [0, 2]] = True
    reward.state.completed[0, [0, 2]] = True
    reward.state.completed[1, 0] = True
    reward.state.completed[3, 2] = True  # 错侧事件不能算左窄成功。
    reward.state.front_paid[0] = True
    reward.state.route_paid[0] = True
    reward.state.unprepared_collision_seen[2, 2] = True
    # 未看见目标就终止的回合不得进入已见障碍的早抬率分子。
    reward.state.early_lift_seen[:, 0] = True
    reward.state.seen[2, 0] = False
    reward.reset(torch.tensor([0, 1, 2]))
    metrics = reward._env.extras["log"]
    for name, event, collision in [("single_step", 1., 0.), ("single_step_left", 1., 0.),
                                    ("single_step_right", 0., 1.)]:
        assert metrics[f"Metrics/prelift/{name}/event_rate"].item() == event
        assert metrics[f"Metrics/prelift/{name}/unprepared_collision_rate"].item() == collision
        assert metrics[f"Metrics/prelift/{name}/sequence_complete_rate"].item() == (name == "single_step")
        assert metrics[f"Metrics/prelift/{name}/route_reward_rate"].item() == (name == "single_step")
    assert reward.state.terrain_kind.tolist() == [-1, -1, -1, 1]
    assert reward.state.completed[3, 2]
    assert metrics["Metrics/prelift/fl_early_lift_rate"].item() == 1.


def test_step_position_is_reproducible_and_outside_initial_preview(preview):
    front_range = (1.20, 1.70)
    positions = [preview.step_front_x_for_difficulty(value, front_range, 42)
                 for value in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert positions == [preview.step_front_x_for_difficulty(value, front_range, 42)
                         for value in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(1.20 <= x <= 1.70 for x in positions)
    assert len(set(positions)) == len(positions)
    assert all(x - 0.35 - 0.10 > preview.PreviewCfg().lookahead_m for x in positions)
    assert positions != [preview.step_front_x_for_difficulty(value, front_range, 142)
                         for value in (0.1, 0.3, 0.5, 0.7, 0.9)]


def test_lateral_step_bounds_select_only_target_wheel_track(preview):
    assert preview.step_lateral_bounds(0.30, 0.36) == pytest.approx((0.12, 0.48))
    assert preview.step_lateral_bounds(-0.30, 0.36) == pytest.approx((-0.48, -0.12))
    assert preview.step_lateral_bounds(0.0, 3.0) == pytest.approx((-1.5, 1.5))


@pytest.mark.parametrize("center,width", [(float("nan"), 0.36), (0.0, 0.0), (0.0, -0.1)])
def test_invalid_lateral_step_bounds_fail(preview, center, width):
    with pytest.raises(ValueError):
        preview.step_lateral_bounds(center, width)


@pytest.mark.parametrize("difficulty,front_range", [(-0.1, (1.2, 1.7)), (1.1, (1.2, 1.7)),
                                                   (0.5, (1.7, 1.2))])
def test_invalid_step_position_inputs_fail(preview, difficulty, front_range):
    with pytest.raises(ValueError):
        preview.step_front_x_for_difficulty(difficulty, front_range, 42)


@pytest.mark.parametrize("height,edge", [(0.0, 0.75), (0.07, -0.6), (0.30, 0.75)])
def test_flat_behind_and_unreachable_obstacles_have_zero_geometric_potential(preview, height, edge):
    out = preview.preview_measurements(**scene(height=height, edge=edge, lift=0.07))
    assert not out["preview"].any()
    assert out["potential"].item() == 0.0


def test_torso_jump_is_not_active_suspension_lift(preview):
    data = scene()
    data["root_pos_w"][:, 2] += 0.07
    data["wheel_centers_w"][:, :, 2] += 0.07
    out = preview.preview_measurements(**data)
    assert out["preview"].any()
    assert not out["prelift"].any()
    assert out["potential"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("speed", [0.0, -0.2])
def test_stationary_or_reverse_is_ineligible_for_geometric_prelift(preview, speed):
    out = preview.preview_measurements(**scene(lift=0.07, speed=speed))
    assert out["potential"].item() == 0.0
    assert not out["prelift"].any()


@pytest.mark.parametrize("fault", ["airborne", "body_support", "invalid_rays", "zero_command"])
def test_invalid_or_unsupported_states_have_zero_geometric_potential(preview, fault):
    data = scene(lift=0.07)
    if fault == "airborne":
        data["supported_wheels"].zero_()
    elif fault == "body_support":
        data["nonwheel_contact"].fill_(True)
    elif fault == "invalid_rays":
        data["ray_hits_w"].fill_(float("nan"))
    else:
        data["command_x"].zero_()
    out = preview.preview_measurements(**data)
    assert torch.isfinite(out["potential"]).all()
    assert out["potential"].item() == 0.0
    assert not out["prelift"].any()


@pytest.mark.parametrize("field", ["root_pos_w", "root_quat_w", "wheel_centers_w"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_geometry_never_poison_geometric_potential(preview, field, value):
    data = scene(lift=0.07)
    data[field].fill_(value)
    out = preview.preview_measurements(**data)
    assert torch.isfinite(out["potential"]).all()
    assert out["potential"].item() == 0.0
    assert not out["prelift"].any()


def test_obstacle_outside_wheel_lane_is_ignored(preview):
    data = scene(lift=0.07)
    hits = data["ray_hits_w"]
    hits[..., 2] *= (hits[..., 1].abs() < 0.05)
    assert not preview.preview_measurements(**data)["preview"].any()


def test_heading_and_world_translation_do_not_change_preview(preview):
    data = scene(lift=0.07)
    before = preview.preview_measurements(**data)
    transform = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = torch.tensor([5.0, -3.0, 2.0])
    for name in ("root_pos_w", "wheel_centers_w", "ray_hits_w"):
        data[name] = data[name] @ transform.T + shift
    data["root_quat_w"] = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    after = preview.preview_measurements(**data)
    torch.testing.assert_close(after["potential"], before["potential"], atol=1e-5, rtol=1e-5)
    assert after["prelift"].tolist() == before["prelift"].tolist()


def test_body_pitch_without_joint_motion_is_not_lift(preview):
    data = scene()
    angle = 0.15
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    data["wheel_centers_w"] = data["neutral_centers_b"][None] @ rotation.T + data["root_pos_w"][:, None]
    data["root_quat_w"] = torch.tensor([[math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]])
    out = preview.preview_measurements(**data)
    assert not out["prelift"].any()
    assert out["potential"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("settings", [{"lookahead_m": -1.0}, {"min_advance_m": 0.8},
                                      {"min_rise_m": float("nan")}, {"min_support_wheels": 5}])
def test_bad_preview_parameters_fail_early(preview, settings):
    with pytest.raises(ValueError):
        preview.PreviewCfg(**settings)
