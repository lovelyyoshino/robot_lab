"""提前抬轮必须与同一episode的最终通行绑定。

@author pony
@date 2026-09-30
@version v0.2.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 全宽允许任一前轮完整主动，拒绝左右轮拼接及非目标侧冒充窄障碍动作。
    - (2026-10-13): 独立基线测试覆盖落地冲击与稳定接近，窄障碍正例补齐真实采样过程。
    - (2026-10-11): 对照预瞄前已落地的颠簸与持续悬轮，后者仍须判为早抬。
    - (2026-10-11): 使用真实接地/离地样本，覆盖侧倾早离地与高度达标延后的独立验收。
    - (2026-10-10): 有效动作窗口收窄为8–14cm，覆盖远处抬起后悬着进入窗口不得补领。
    - (2026-10-10): 验收持续保持到前缘与稳定落轮，保留事件率用于分段诊断。
    - (2026-10-09): 覆盖左右轮相对车身抬高仍承重的基线和窄障碍低位判据。
    - (2026-10-09): 验证70%配额门、持续减载事件及独立严格2cm指标。
    - v0.2.0 (2026-10-08): 对齐势能字段，明确顺序抬轮可通过事件验收。
    - v0.1.0 (2026-09-30): 拒绝碰后抬轮、单轮冒充、跨reset污染与不完整评测。
"""

from pathlib import Path

import pytest


@pytest.fixture
def trackers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation
    return SkillEvaluation(1, 2, 0.1, 0.1, 2), PreliftEvaluation(1, 0.1)


def step(trackers, *, lead=0.1, lift=(0.03, 0.03), reward_clearance=(0.025, 0.025),
         eligible=True, done=False, fallen=False):
    traversal, prelift = trackers
    traversal.update([True], [fallen], [done], [True])
    prelift.update([{"lead_m": [lead, lead], "lift_m": lift, "clearance_m": list(lift),
                     "wheel_support_force_n": [20. if height < 0.005 else 0. for height in lift],
                     "reward_clearance_m": reward_clearance,
                     "eligible": eligible, "preview": [True, True], "reward_eligible": eligible,
                     "preview_potential_raw": 0.1}], traversal)


@pytest.mark.parametrize("lands_before_approach", [True, False])
def test_far_bounce_only_consumes_attempt_if_hover_continues(trackers, lands_before_approach):
    step(trackers, lead=0.9, lift=(0., 0.))
    step(trackers, lead=0.8, lift=(0.01, 0.01))
    step(trackers, lead=0.3, lift=(0., 0.) if lands_before_approach else (0.01, 0.01))
    step(trackers, lead=0.1, done=True)
    result = trackers[1].records[0]
    assert result["required_front_wheels_prelift"] is lands_before_approach
    assert result["front_wheel_takeoffs"][0]["lead_distance_m"] == pytest.approx(
        0.1 if lands_before_approach else 0.3,
    )


@pytest.mark.parametrize("fault", [None, "drop", "low_crossing", "no_release", "release_bounce"])
def test_temporal_acceptance_requires_hold_until_edge_and_stable_release(trackers, fault):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    evaluator = PreliftEvaluation(1, 0.02, require_hold_to_edge=True)

    def advance(lead, high=True, *, contact=False, done=False):
        traversal.update([True], [False], [done], [True])
        evaluator.update([{
            "lead_m": [lead, lead], "lift_m": [0.03 if high else 0.] * 2,
            "clearance_m": [0.03 if high else 0.] * 2,
            "reward_clearance_m": [0. if contact else 0.03] * 2,
            "wheel_support_force_n": [20. if contact or not high else 0.] * 2,
            "eligible": True, "hold_eligible": True,
        }], traversal)

    advance(0.3, high=False)
    for _ in range(3):
        advance(0.1)
    if fault == "drop":
        advance(0.09, high=False)
        advance(0.08)
    advance(0.04)
    advance(-0.01, high=fault != "low_crossing")
    for frame in range(3):
        advance(-0.15, contact=fault != "no_release" and not (fault == "release_bounce" and frame == 1),
                done=frame == 2)
    record = evaluator.records[0]
    assert record["required_front_wheels_prelift"]
    assert record["success"] is (fault is None)
    report = evaluator.summary(traversal.summary(0.70))
    assert report["front_wheel_event_rates"] == {"fl": 1., "fr": 1.}
    assert report["front_wheel_release_rates"] == {"fl": float(fault is None), "fr": float(fault is None)}


def test_full_quota_success(trackers):
    step(trackers, lift=(0.0, 0.0))
    step(trackers)
    step(trackers, lead=-1, done=True)
    step(trackers, lift=(0.0, 0.0))
    step(trackers)
    step(trackers, lead=-1, done=True)
    assert trackers[1].summary(trackers[0].summary(0.85))["passed"]
    assert trackers[1].summary(trackers[0].summary(0.85))["gates"]["low_ground_clearance_threshold_m"] == 0.005
    assert trackers[1].records[0]["front_wheels"][0]["lead_distance_m"] == 0.1


@pytest.mark.parametrize("kwargs", [{"lead": 0.01}, {"eligible": False}, {"lift": (0.03, 0.0)},
                                  {"lead": 0.15}, {"lead": 0.25}, {"lead": 0.65},
                                  {"lead": 1.0}, {"lead": float("nan")}])
def test_late_stationary_single_wheel_or_invalid_fails(trackers, kwargs):
    step(trackers, lift=(0.0, 0.0))
    step(trackers, **kwargs)
    step(trackers, lead=-1, done=True)
    assert not trackers[1].records[0]["success"]


def test_retreat_after_contact_cannot_count_as_anticipation(trackers):
    step(trackers, lift=(0.0, 0.0))
    step(trackers, lead=-0.01)
    step(trackers, lead=0.3)
    step(trackers, lead=-1, done=True)
    assert not trackers[1].records[0]["success"]


def test_lift_without_terminal_traversal_fails(trackers):
    step(trackers, lift=(0.0, 0.0))
    step(trackers)
    step(trackers, lead=-1, done=True, fallen=True)
    assert not trackers[1].records[0]["success"]


def test_reset_isolation_and_incomplete_denominator(trackers):
    step(trackers, lift=(0.0, 0.0))
    step(trackers)
    step(trackers, lead=-1, done=True)
    assert not trackers[1].summary(trackers[0].summary(0.85))["passed"]
    step(trackers, lead=-1, done=True)
    assert [r["success"] for r in trackers[1].records] == [True, False]
    step(trackers, done=True)
    assert len(trackers[1].records) == 2


def test_permanently_raised_wheels_do_not_count(trackers):
    step(trackers)
    step(trackers, done=True)
    assert not trackers[1].records[0]["success"]


def test_early_raise_is_reported_and_cannot_be_paid_on_window_entry(trackers):
    step(trackers, lead=0.4, lift=(0., 0.))
    step(trackers, lead=0.3)
    step(trackers, lead=0.1)
    step(trackers, lead=-1, done=True)
    record = trackers[1].records[0]
    assert not record["success"]
    assert record["diagnostics"]["early_lift_steps"] == [1, 1]
    assert record["diagnostics"]["first_early_lift_lead_m"] == [0.3, 0.3]
    assert trackers[1].summary(trackers[0].summary(0.70))["front_wheel_early_lift_rates"] == {"fl": 1., "fr": 1.}


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("target", [0, 1])
def test_takeoff_timing_is_not_joint_threshold_timing(trackers, early, target):
    traversal, evaluator = trackers
    for frame, (lead, lift, clearance, force) in enumerate((
        (0.4, 0.04, 0.002, 20.),
        (0.30 if early else 0.12, -0.04, 0.006, 0.),
        (0.06, 0.03, 0.08, 0.),
        (-0.03, 0.03, 0.08, 20.),
    )):
        lifts, clearances, forces = [0., 0.], [0., 0.], [20., 20.]
        lifts[target], clearances[target], forces[target] = lift, clearance, force
        traversal.update([True], [False], [frame == 3], [True])
        evaluator.update([{
            "lead_m": [lead] * 2, "lift_m": lifts, "clearance_m": clearances,
            "wheel_support_force_n": forces, "eligible": True,
        }], traversal)
    record = evaluator.records[0]
    assert record["front_wheel_takeoffs"][target]["lead_distance_m"] == (0.3 if early else 0.12)
    if early:
        assert record["front_wheels"][target] is None
        assert record["diagnostics"]["first_early_lift_lead_m"][target] == 0.3
    else:
        assert record["front_wheels"][target]["lead_distance_m"] == 0.12
        assert record["front_wheels"][target]["height_threshold_lead_m"] == 0.06


def test_initial_drop_before_first_ground_contact_is_not_early_takeoff(trackers):
    traversal, evaluator = trackers
    traversal.update([True], [False], [False], [True])
    evaluator.update([{
        "lead_m": [1.2, 1.2], "lift_m": [0., 0.], "clearance_m": [0.015, 0.015],
        "wheel_support_force_n": [0., 0.], "eligible": False, "preview": [False, False],
    }], traversal)
    assert evaluator.pending[0]["takeoff"] == [None, None]
    assert evaluator.pending[0]["diagnostics"]["early_lift_steps"] == [0, 0]


def test_episode_diagnostics_explain_missing_prelift_event(trackers):
    step(trackers, lift=(0.0, 0.0))
    step(trackers, lift=(0.015, 0.0), eligible=False)
    step(trackers, lead=-0.01, lift=(0.03, 0.0), done=True)
    diagnostics = trackers[1].records[0]["diagnostics"]
    assert diagnostics["preview_steps"] == [3, 3]
    assert diagnostics["reward_eligible_preview_steps"] == [2, 2]
    assert diagnostics["eligible_lead_window_steps"] == [1, 1]
    assert diagnostics["joint_threshold_steps_in_window"] == [0, 0]
    assert diagnostics["max_reward_clearance_m_in_window"] == [0.025, 0.025]
    assert diagnostics["low_seen_before_lift"] == [True, True]
    assert diagnostics["geometric_potential_sum"] == pytest.approx(0.3)


def test_sequential_front_lifts_pass_without_simultaneous_hover(trackers):
    step(trackers, lift=(0., 0.))
    step(trackers, lift=(0.03, 0.))
    step(trackers, lift=(0., 0.03))
    step(trackers, lead=-1, done=True)
    assert trackers[1].records[0]["success"]
    assert trackers[1].summary(trackers[0].summary(0.85))["front_wheel_event_rates"] == {"fl": 1., "fr": 1.}


def test_threshold_without_eligibility_reports_failed_gate(trackers):
    traversal, prelift = trackers
    step(trackers, lift=(0.0, 0.0))
    traversal.update([True], [False], [False], [True])
    prelift.update([{
        "lead_m": [0.1, 0.1], "lift_m": [0.03, 0.03],
        "clearance_m": [0.03, 0.03], "eligible": False,
        "eligibility_gates": {"supported": False, "forward_motion": True},
    }], traversal)
    step(trackers, lead=-1, done=True)
    diagnostics = prelift.records[0]["diagnostics"]
    assert diagnostics["ineligible_threshold_reasons"]["supported"] == [1, 1]
    assert diagnostics["ineligible_threshold_reasons"]["forward_motion"] == [0, 0]


@pytest.mark.parametrize("required,lift,expected", [((0,), (0.03, 0.0), True),
                                                     ((1,), (0.03, 0.0), False),
                                                     ((1,), (0.0, 0.03), True)])
def test_narrow_obstacle_requires_its_own_front_wheel(trackers, required, lift, expected):
    traversal, _ = trackers
    from vision_prelift_evaluation import PreliftEvaluation

    prelift = PreliftEvaluation(1, 0.1, required_front_wheels=required)
    for current_lift, lead, done in [((0.0, 0.0), 0.2, False), (lift, 0.1, False), (lift, -1, True)]:
        traversal.update([True], [False], [done], [True])
        prelift.update([{"lead_m": [lead, lead], "lift_m": current_lift,
                         "clearance_m": list(current_lift), "eligible": True,
                         "wheel_support_force_n": [20. if h < 0.005 else 0. for h in current_lift]}], traversal)
    assert prelift.records[0]["success"] is expected
    assert prelift.summary(traversal.summary(0.85))["gates"]["required_front_wheels"] == list(required)


@pytest.mark.parametrize("target", [0, 1])
def test_full_width_accepts_one_complete_active_side_and_passive_other_side(trackers, target):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, .02, .02, 2)
    prelift = PreliftEvaluation(1, .02, required_active_count=1, require_hold_to_edge=True)
    for frame in range(8):
        lift = [0., 0.]
        clearance, force = [0., 0.], [20., 20.]
        if 1 <= frame <= 4:
            lift[target] = clearance[target] = .03
            force[target] = 0.
        lead = .30 if frame == 0 else .10 if frame < 4 else -.01
        done = frame == 7
        traversal.update([True], [False], [done], [True])
        prelift.update([{"lead_m": [lead, lead], "lift_m": lift, "clearance_m": clearance,
                         "reward_clearance_m": clearance, "eligible": True,
                         "wheel_support_force_n": force}], traversal)
    assert prelift.records[0]["success"]
    assert not prelift.records[0]["both_front_wheels_prelift"]
    assert prelift.summary(traversal.summary(.70))["gates"]["required_active_count"] == 1


def test_narrow_obstacle_accepts_early_target_wheel_unloading_without_large_clearance(trackers):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.1, 0.1, 2)
    prelift = PreliftEvaluation(1, 0.1, required_front_wheels=(0,), allow_load_drop=True)
    for lead, force, clearance, done in ([(0.4, [20.0, 30.0], 0.0, False)] * 8
                                         + [(0.10, [10.0, 30.0], 0.005, False),
                                            (-1.0, [8.0, 30.0], 0.005, True)]):
        traversal.update([True], [False], [done], [True])
        prelift.update([{
            "lead_m": [lead, lead], "lift_m": [0.005, 0.0], "clearance_m": [clearance, 0.0],
            "wheel_support_force_n": force, "eligible": True,
        }], traversal)
    record = prelift.records[0]
    assert record["success"]
    assert record["front_wheels"][0]["lead_distance_m"] == 0.10
    assert prelift.summary(traversal.summary(0.70))["gates"]["min_load_drop_fraction"] == 0.35


def test_independent_baseline_rejects_spawn_impact_and_freezes_before_unloading(trackers):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 3)
    prelift = PreliftEvaluation(1, 0.02, allow_load_drop=True)
    for lead, force in [(0.4, f) for f in [212., 36., 39.] + [20.] * 37] + [(0.2, 10.)] * 10:
        traversal.update([True], [False], [False], [True])
        prelift.update([{"lead_m": [lead, lead], "lift_m": [0., 0.], "clearance_m": [0., 0.],
                         "wheel_support_force_n": [force, force], "eligible": True}], traversal)
    state = prelift.pending[0]
    assert state["baseline_force_n"] == pytest.approx([20., 20.])
    assert state["baseline_samples"] == [6, 6]
    assert state["baseline_locked"] == [True, True]
    assert state["lifted"] == [None, None]


def test_narrow_obstacle_load_drop_after_contact_does_not_count(trackers):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.1, 0.1, 2)
    prelift = PreliftEvaluation(1, 0.1, required_front_wheels=(0,), allow_load_drop=True)
    for lead, force, done in ((0.2, [20.0, 30.0], False),
                              (-0.01, [10.0, 30.0], False),
                              (-1.0, [8.0, 30.0], True)):
        traversal.update([True], [False], [done], [True])
        prelift.update([{
            "lead_m": [lead, lead], "lift_m": [0.0, 0.0], "clearance_m": [0.0, 0.03],
            "wheel_support_force_n": force, "eligible": True,
        }], traversal)
    assert not prelift.records[0]["success"]
    assert prelift.records[0]["front_wheels"][0] is None


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("grounded_lift", [0.03, 0.089])
def test_narrow_grounded_articulation_lift_still_arms_unloading(trackers, target, grounded_lift):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    prelift = PreliftEvaluation(1, 0.02, required_front_wheels=(target,), allow_load_drop=True)
    for frame in range(44):
        unloaded = 40 <= frame < 43
        done = frame == 43
        forces, lift, clearance = [20., 20.], [0., 0.], [0., 0.]
        forces[target] = 10. if unloaded else 20.
        lift[target] = grounded_lift
        clearance[target] = 0.006 if unloaded else 0.002
        traversal.update([True], [False], [done], [True])
        prelift.update([{
            "lead_m": [-1., -1.] if done else [0.1, 0.1] if unloaded else [0.4, 0.4],
            "lift_m": lift, "clearance_m": clearance,
            "wheel_support_force_n": forces, "eligible": True,
        }], traversal)
        if frame == 39:
            assert prelift.pending[0]["baseline_samples"][target] == 6
            assert prelift.pending[0]["low_seen"][target]
    record = prelift.records[0]
    assert record["success"]
    assert record["front_wheels"][target]["event_type"] == "unloaded_clearance"
    assert record["front_wheels"][1 - target] is None
    assert record["strict_front_wheel_lifts"][target] is None
    assert record["diagnostics"]["baseline_samples"][target] == 6
    assert record["diagnostics"]["baseline_force_n"][target] == pytest.approx(20.)
    gates = prelift.summary(traversal.summary(0.70))["gates"]
    assert gates["low_lift_threshold_m"] is None
    assert gates["low_ground_clearance_threshold_m"] == 0.005
    assert gates["low_support_force_threshold_n"] == 1.0
    assert gates["baseline_min_support_force_n"] == 4.0
    assert gates["baseline_required_samples"] == 6


@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("fault", ["hover", "no_load", "blocked", "invalid_clearance", "invalid_lift"])
def test_narrow_grounded_baseline_rejects_invalid_low_states(trackers, target, fault):
    from vision_prelift_evaluation import PreliftEvaluation

    traversal, _ = trackers
    prelift = PreliftEvaluation(1, 0.02, required_front_wheels=(target,), allow_load_drop=True)
    for _ in range(40):
        forces, lift, clearance, blocked = [20., 20.], [0., 0.], [0., 0.], [False, False]
        lift[target] = 0.03
        clearance[target] = 0.006 if fault == "hover" else 0.002
        if fault == "no_load":
            forces[target] = 0.
        elif fault == "blocked":
            blocked[target] = True
        elif fault == "invalid_clearance":
            clearance[target] = float("nan")
        elif fault == "invalid_lift":
            lift[target] = float("nan")
        traversal.update([True], [False], [False], [True])
        prelift.update([{
            "lead_m": [0.4, 0.4], "lift_m": lift, "clearance_m": clearance,
            "wheel_support_force_n": forces, "blocking_collision": blocked, "eligible": True,
        }], traversal)
    assert prelift.pending[0]["baseline_samples"][target] == 0
    assert not prelift.pending[0]["low_seen"][target]


@pytest.mark.parametrize("successes,passed", [(22, False), (23, True), (25, True)])
def test_seventy_percent_gate_needs_23_of_32(trackers, successes, passed):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    pair = SkillEvaluation(1, 32, 0.1, 0.1, 2), PreliftEvaluation(1, 0.1)
    for episode in range(32):
        step(pair, lift=(0., 0.))
        step(pair, eligible=episode < successes)
        step(pair, lead=-1, done=True)
    assert pair[1].summary(pair[0].summary(0.70))["passed"] is passed


@pytest.mark.parametrize("fault", [None, "bounce", "no_clearance", "missing_force", "nan_force",
                                  "wrong_side", "blocked", "no_support"])
def test_narrow_requires_sustained_clearance_before_blocking_contact(trackers, fault):
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    prelift = PreliftEvaluation(1, 0.02, required_front_wheels=(0,), allow_load_drop=True)
    def advance(force=20., clearance=0., *, eligible=True, blocked=False, done=False, lift=0.005, lead=0.1):
        traversal.update([True], [False], [done], [True])
        prelift.update([{
            "lead_m": [lead, lead], "lift_m": [lift, 0.0], "clearance_m": [clearance, 0.0],
            "wheel_support_force_n": [force, 20.], "eligible": eligible,
            "blocking_collision": [blocked, False],
        }], traversal)
    for _ in range(40):
        advance(lead=0.4)
    for frame in range(3):
        force = {"missing_force": None, "nan_force": float("nan"), "wrong_side": 20.}.get(fault, 10.)
        clearance = 0. if fault == "no_clearance" or (fault == "bounce" and frame != 0) else 0.006
        advance(force, clearance, eligible=fault != "no_support", blocked=fault == "blocked")
    if fault is None:
        # 宽松事件发生之后仍能独立记录严格2cm抬轮。
        advance(0., 0.03, lift=0.03)
    advance(done=True)
    assert prelift.records[0]["success"] is (fault is None)
    if fault is None:
        assert prelift.records[0]["strict_front_wheel_lifts"][0] is not None


def test_depth_probe_counts_obstacle_pixels_in_each_front_wheel_lane(trackers):
    import torch
    from vision_prelift_evaluation import wheel_lane_pixel_counts

    points = torch.tensor([
        [1.0, 0.30, 0.04], [1.0, -0.30, 0.04], [1.0, 0.22, 0.04], [1.0, 0.50, 0.04],
    ])
    candidates = torch.tensor([True, True, True, False])
    assert wheel_lane_pixel_counts(points, candidates, [0.30, -0.30], 0.09) == [2, 1]


def test_depth_probe_rejects_invalid_wheel_lanes(trackers):
    import torch
    from vision_prelift_evaluation import wheel_lane_pixel_counts

    points = torch.zeros((1, 3))
    candidates = torch.ones((1,), dtype=torch.bool)
    with pytest.raises(ValueError):
        wheel_lane_pixel_counts(points, candidates, [float("nan"), 0.3], 0.09)
    with pytest.raises(ValueError):
        wheel_lane_pixel_counts(points, candidates, [0.3, -0.3], 0.0)


def test_depth_probe_runs_preview_diagnostics_for_single_step(monkeypatch, trackers):
    import importlib
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    repo_root = Path(__file__).parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "scripts/reinforcement_learning/rsl_rl"))
    evaluation = importlib.import_module("evaluate_parallelogram_vision")

    preview_name = (
        "robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled."
        "parallelogram_rover.vision_preview"
    )
    parts = preview_name.split(".")
    for length in range(1, len(parts)):
        name = ".".join(parts[:length])
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    preview_module = ModuleType(preview_name)
    preview_module.collect_preview = lambda env: {
        "wheel_centers_w": torch.tensor([[[0.0, 0.3, 0.2], [0.0, 0.3, 0.2],
                                           [0.0, -0.3, 0.2], [0.0, -0.3, 0.2]]]),
        "wheel_radii": torch.full((4,), 0.1),
    }
    preview_module.step_front_x_for_difficulty = lambda difficulty, bounds, seed: 0.5
    preview_module.step_lateral_bounds = lambda center, width: (center - width / 2, center + width / 2)
    monkeypatch.setitem(sys.modules, preview_name, preview_module)

    for name in ("isaaclab", "isaaclab.utils"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    math_module = ModuleType("isaaclab.utils.math")
    math_module.convert_camera_frame_orientation_convention = lambda quat, **kwargs: quat
    math_module.quat_apply = lambda quat, vectors: vectors
    math_module.quat_mul = lambda left, right: left
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)

    class Scene:
        env_origins = torch.zeros((1, 3))

        def __init__(self):
            self.sensors = {
                "depth_camera": SimpleNamespace(data=SimpleNamespace(
                    output={"distance_to_image_plane": torch.ones((1, 2, 2, 1))},
                    intrinsic_matrices=torch.eye(3).unsqueeze(0),
                )),
                "height_scanner": SimpleNamespace(data=SimpleNamespace(
                    ray_hits_w=torch.zeros((1, 1, 3)),
                )),
            }
            self.robot = SimpleNamespace(
                data=SimpleNamespace(body_pos_w=torch.zeros((1, 1, 3)),
                                     body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])),
                find_bodies=lambda name: ([0], ["base_link"]),
            )

        def __getitem__(self, name):
            return getattr(self, name)

    terrain = SimpleNamespace(
        step_front_x_range=(0.4, 0.6), step_height_range=(0.04, 0.1), step_width=0.1,
        lane_center_y=0.0, lane_width=3.0,
    )
    env = SimpleNamespace(
        scene=Scene(),
        cfg=SimpleNamespace(
            scene=SimpleNamespace(
                depth_camera=SimpleNamespace(offset=SimpleNamespace(
                    convention="world", pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                )),
                terrain=SimpleNamespace(terrain_generator=SimpleNamespace(
                    sub_terrains={"single_step": terrain},
                )),
            ),
            observations=SimpleNamespace(policy=SimpleNamespace(depth_image=SimpleNamespace(
                params={"crop_top": 0, "crop_bottom": 0, "min_depth": 0.05, "max_depth": 3.0},
            ))),
            vision_preview={"lane_half_width_m": 0.09},
        ),
    )
    args = SimpleNamespace(scenario="single_step", difficulty=0.0, seed=142, depth_snapshot=None)

    result = evaluation.capture_depth_probe(env, args, save_snapshot=False)

    assert result["scope"] == "render_env0_not_visual_policy_acceptance"
    assert result["visible_step_pixels_by_front_wheel"] == {"fl": 0, "fr": 0}
    assert result["target_wheel_visible_step_pixels"] == [0, 0]
