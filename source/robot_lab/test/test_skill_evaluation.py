"""技能验收回归：瞬时成功、异步失败、终态和非有限状态不能制造通过结果。

@author pony
@date 2026-09-16
@version v0.2.0
@last_modified 2026-09-16
@changelog
    - v0.2.0 (2026-09-16): 接管时钟排除准备时间，但准备失败保留总体分母。
    - v0.1.0 (2026-09-16): 覆盖实际发现的成功率统计缺陷与连续保持边界。
"""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/skill_evaluation.py"
spec = importlib.util.spec_from_file_location("skill_evaluation", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SkillEvaluation = module.SkillEvaluation


def rollout(stable, *, hold=0.5, deadline=4.0, fail_at_end=False, invalid_step=None):
    tracker = SkillEvaluation(1, 1, 0.02, hold, deadline)
    for index, value in enumerate(stable):
        end = index == len(stable) - 1
        tracker.update([value], [end and fail_at_end], [end], [index != invalid_step])
    return tracker.summary(0.85)


def test_early_half_second_then_unstable_episode_does_not_pass_balance():
    result = rollout([True] * 25 + [False] * 1474 + [True], hold=30.0, deadline=1.0)
    assert result["complete"]
    assert not result["passed"]


def test_terminal_sample_failure_is_not_hidden_by_previous_stability():
    assert not rollout([True] * 249 + [False])["passed"]


def test_episode_must_remain_stable_after_acquisition_deadline():
    assert not rollout([True] * 150 + [False] * 60 + [True] * 40)["passed"]


def test_balanced_thirty_seconds_passes_after_settling():
    result = rollout([False] * 50 + [True] * 1500, hold=30.0, deadline=1.02)
    assert result["passed"]
    assert result["mean_time_to_stable_s"] == pytest.approx(1.02)
    assert result["episode_results"][0]["terminal_stable_time_s"] == 30.0


def test_timeout_with_physical_failure_is_failure():
    assert not rollout([True] * 250, fail_at_end=True)["passed"]


def test_nonfinite_state_cannot_be_forgotten_after_recovery():
    assert not rollout([True] * 250, invalid_step=5)["passed"]


def test_fast_failing_environment_cannot_fill_other_environments_quota():
    tracker = SkillEvaluation(2, 2, 0.02, 0.04, 1.0)
    for i in range(8):
        tracker.update([False, True], [True, False], [False, i in (3, 7)], [True, True], [0.0, 0.1])
    result = tracker.summary(0.5)
    assert result["complete"]
    assert result["episodes_by_env"] == [2, 2]
    assert result["episodes"] == 4
    assert result["success_rate"] == 0.5
    assert tracker.pitch_samples == 10


def test_incomplete_evaluation_never_passes():
    tracker = SkillEvaluation(2, 1, 0.02, 0.02, 1.0)
    tracker.update([True, True], [False, False], [True, False], [True, True])
    result = tracker.summary(0.85)
    assert result["success_rate"] == 1.0
    assert not result["complete"]
    assert not result["passed"]


@pytest.mark.parametrize("hold", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_hold_time_is_rejected(hold):
    with pytest.raises(ValueError):
        SkillEvaluation(1, 1, 0.02, hold, 1.0)


def test_preparation_does_not_consume_acquisition_deadline():
    tracker = SkillEvaluation(1, 1, .02, .04, .02)
    for _ in range(100):
        tracker.update([True], [False], [False], [True], active=[False])
    tracker.update([True], [False], [False], [True], active=[True])
    tracker.update([True], [False], [True], [True], active=[True])
    result = tracker.summary(.85)
    assert result["passed"]
    assert result["episode_results"][0]["preparation_time_s"] == 2.0
    assert result["episode_results"][0]["duration_s"] == .04


def test_initialization_failure_is_not_dropped_from_overall_denominator():
    tracker = SkillEvaluation(2, 1, .02, .02, .02)
    tracker.update([True, True], [True, False], [False, True], [True, True],
                   active=[False, True], initialization_failed=[True, False])
    result = tracker.summary(.85)
    assert result["complete"] and not result["passed"]
    assert result["success_rate"] == .5
    assert result["initialization_failure_count"] == 1
    assert result["conditional_takeover_success_rate"] == 1.
