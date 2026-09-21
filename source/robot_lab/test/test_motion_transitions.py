"""运动后停止的两段验收回归；真实度量函数在 CPU 上执行。

@author pony
@date 2026-09-16
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 为两阶段评测补充物理步前后根位置并覆盖停止漂移门。
    - v0.1.0 (2026-09-17): 覆盖两段通过、前段失败保留、提前终止和逐环境时钟隔离。
    - v0.1.0 (2026-09-16): 先锁定未移动但末态停住的假成功，再覆盖时钟和复位隔离。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def evaluation(monkeypatch):
    scripts = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("motion_transition_test", scripts / "evaluate_parallelogram_motion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stationary_robot_cannot_pass_stop_without_first_moving(evaluation):
    protocol = evaluation.StopProtocol(
        dt=.1, move_deadline_s=.2, move_hold_time_s=.3,
        stop_deadline_s=.2, stop_hold_time_s=.3,
    )
    tracker = evaluation.StopEvaluation(1, 1, protocol)
    for step in range(1, protocol.episode_steps + 1):
        tracker.update_motion(
            [True], [False], [step == protocol.episode_steps], [True],
            [[0., 0., 0.]], [protocol.command_at(step - 1)], [True],
            episode_steps=[step], world_positions=[[0., 0.]], previous_world_positions=[[0., 0.]],
        )
    record = tracker.records[0]
    assert not record["success"]
    assert not record["move_phase"]["passed"]
    assert record["move_phase"]["ground_forward_distance_m"] == 0
    assert record["stop_phase"]["passed"]
    assert not tracker.summary(.85)["passed"]


def _protocol(evaluation):
    return evaluation.StopProtocol(.1, .2, .3, .2, .3)


def test_success_requires_both_phases_and_records_relative_stop_time(evaluation):
    protocol = _protocol(evaluation)
    tracker = evaluation.StopEvaluation(1, 1, protocol)
    for step in range(1, protocol.episode_steps + 1):
        command = protocol.command_at(step - 1)
        tracker.update_motion([True], [False], [step == protocol.episode_steps], [True],
                              [[command[0], 0., 0.]], [command], [True], episode_steps=[step],
                              world_positions=[[step * .02, 0.]],
                              previous_world_positions=[[(step - 1) * .02, 0.]])
    record = tracker.records[0]
    assert record["success"]
    assert record["move_phase"]["passed"] and record["stop_phase"]["passed"]
    assert record["command_switch_time_s"] == pytest.approx(.5)
    assert record["time_to_stable_s"] == pytest.approx(.1)
    assert tracker.summary(.95)["passed"]


@pytest.mark.parametrize("failure", ["no_support", "nan", "early_done", "late_movement"])
def test_movement_failure_survives_successful_stop(evaluation, failure):
    protocol = _protocol(evaluation)
    tracker = evaluation.StopEvaluation(1, 1, protocol)
    for step in range(1, protocol.episode_steps + 1):
        command = protocol.command_at(step - 1)
        before_stop = step <= protocol.move_steps
        velocity = command[0]
        supported = not (failure == "no_support" and before_stop)
        finite = not (failure == "nan" and step == 1)
        stable = not (failure == "late_movement" and step <= 3)
        terminated = failure == "early_done" and step == 3
        tracker.update_motion([stable], [terminated], [step == protocol.episode_steps], [finite],
                              [[velocity, 0., 0.]], [command], [supported], episode_steps=[step],
                              world_positions=[[step * .02, 0.]],
                              previous_world_positions=[[(step - 1) * .02, 0.]])
        if terminated:
            break
    assert not tracker.records[0]["success"]


def test_local_reset_does_not_switch_other_environment_command_or_reuse_history(evaluation):
    protocol = _protocol(evaluation)
    tracker = evaluation.StopEvaluation(2, 2, protocol)
    steps = [0, 0]
    while not tracker.complete:
        steps = [step + 1 if tracker.counts[i] < 2 else step for i, step in enumerate(steps)]
        commands = [protocol.command_at(max(0, step - 1)) for step in steps]
        terminated = [tracker.counts[0] == 0 and steps[0] == 2, False]
        truncated = [step == protocol.episode_steps for step in steps]
        tracker.update_motion([True] * 2, terminated, truncated, [True] * 2,
                              [[command[0], 0., 0.] for command in commands], commands, [True] * 2,
                              episode_steps=steps, world_positions=[[0., 0.], [0., 0.]],
                              previous_world_positions=[[0., 0.], [0., 0.]])
        for i in range(2):
            if terminated[i] or truncated[i]:
                steps[i] = 0
    assert sum(record["success"] for record in tracker.records) == 3
    assert tracker.counts == [2, 2]
    assert [r["success"] for r in tracker.records if r["env_id"] == 0] == [False, True]


def test_missed_clock_and_wrong_command_are_rejected(evaluation):
    protocol = _protocol(evaluation)
    for step, command in ((2, [.25, 0.]), (1, [0., 0.])):
        tracker = evaluation.StopEvaluation(1, 1, protocol)
        with pytest.raises(ValueError):
            tracker.update_motion([True], [False], [False], [True], [[.25, 0., 0.]],
                                  [command], [True], episode_steps=[step], world_positions=[[0., 0.]],
                                  previous_world_positions=[[0., 0.]])


def test_stop_phase_rejects_world_drift_even_when_reported_velocity_is_zero(evaluation):
    protocol = _protocol(evaluation)
    tracker = evaluation.StopEvaluation(1, 1, protocol)
    position = 0.0
    for step in range(1, protocol.episode_steps + 1):
        command = protocol.command_at(step - 1)
        previous = position
        position += .20 if step > protocol.move_steps else .02
        tracker.update_motion(
            [True], [False], [step == protocol.episode_steps], [True],
            [[command[0], 0., 0.]], [command], [True], episode_steps=[step],
            world_positions=[[position, 0.]], previous_world_positions=[[previous, 0.]],
        )
    record = tracker.records[0]
    assert not record["success"]
    assert not record["stop_phase"]["stationary_drift_gate_passed"]
    assert record["stop_phase"]["reference_xy_m"] == pytest.approx([.1, 0.])
    assert record["stop_phase"]["maximum_horizontal_displacement_m"] == pytest.approx(1.0)


@pytest.mark.parametrize("field", ["dt", "move_deadline_s", "move_hold_time_s", "stop_deadline_s", "stop_hold_time_s"])
def test_protocol_rejects_nonfinite_time(evaluation, field):
    kwargs = {"dt": .1, field: float("nan")}
    with pytest.raises(ValueError):
        evaluation.StopProtocol(**kwargs)
