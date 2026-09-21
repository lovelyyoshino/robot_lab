"""保护 Grounded 接管探针的启动、逐环境时钟与采样合同。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 覆盖真实首次active=1、响应时钟、局部reset和漏采样失败。
"""

import ast
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts/tools/probe_recovery_takeover.py"


def _parse_args_function():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")


def test_launcher_registration_precedes_required_flags():
    body = _parse_args_function().body
    launcher_index = next(
        index for index, node in enumerate(body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add_app_launcher_args"
    )
    required_index = next(
        index for index, node in enumerate(body)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "argument"
    )
    assert launcher_index < required_index


def test_probe_uses_pre_reset_termination_snapshot():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    text = ast.unparse(main)
    assert "cfg.terminations.takeover_probe_snapshot" in text
    assert "TerminationTermCfg(func=snapshot)" in text
    assert "get_dof_stiffnesses" in text
    assert "get_dof_dampings" in text
    assert "get_dof_actuation_forces" in text
    assert "physx_actuation_forces" in text


def test_probe_reports_failures_and_uses_named_active_leg_ids():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    text = ast.unparse(main)
    assert "PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES" in text
    assert "PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES" in text
    assert "find_joints(list(active_leg_names), preserve_order=True)" in text
    assert "'active_leg_joint_ids'" in text
    assert "'TAKEOVER_PROBE_STAGE reset_begin'" in text
    assert "joint_pos[index][:8]" not in text
    assert "applied_torque[index][:8]" not in text
    assert "except BaseException:" in source
    assert "traceback.print_exc()" in source


def _classify_records(records, num_envs=2, response_steps=8):
    """独立执行纯分类函数，让时钟回归不依赖启动 Isaac 应用。"""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "classify_takeover_records"
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SCRIPT), "exec"), namespace)
    return namespace["classify_takeover_records"](records, num_envs, response_steps)


def _clock_record(step, *, active=(False, False), clock=(0, 0), pending=(False, False), failed=(False, False)):
    return {
        "common_step": step,
        "control_active": list(active),
        "control_steps": list(clock),
        "takeover_pending": list(pending),
        "initialization_failed": list(failed),
    }


def test_first_terminal_and_response_are_frozen_across_local_reset():
    records = [_clock_record(19, pending=(True, False), failed=(False, True))]
    for step in range(20, 33):
        records.append(_clock_record(
            step,
            active=(True, step >= 28),
            clock=(step - 19, max(0, step - 27)),
            pending=(False, step == 27),
        ))

    result = _classify_records(records)

    assert result[0]["outcome"] == "response_captured"
    assert result[0]["pre_activation"]["common_step"] == 19
    assert result[0]["first_active"]["common_step"] == 20
    assert result[0]["response"]["common_step"] == 27
    assert result[0]["activation_common_step"] == 20
    assert result[0]["activation_control_steps"] == 1
    assert result[0]["response_common_step"] == 27
    assert result[0]["response_control_steps"] == 8
    assert result[0]["response_elapsed_control_steps"] == 7
    assert "may_start_at_1" in result[0]["clock_convention"]
    assert result[1]["outcome"] == "initialization_failed"
    assert result[1]["pre_activation"]["common_step"] == 19
    assert result[1]["first_active"] is None
    assert result[1]["response"] is None


def test_staggered_environments_use_their_own_response_clock():
    records = []
    for step in range(19, 25):
        records.append(_clock_record(
            step,
            active=(step >= 20, step >= 22),
            clock=(max(0, step - 19), max(0, step - 21)),
            pending=(step == 19, step == 21),
        ))

    result = _classify_records(records, response_steps=3)

    assert [item["activation_common_step"] for item in result] == [20, 22]
    assert [item["response_common_step"] for item in result] == [22, 24]
    assert [item["response_control_steps"] for item in result] == [3, 3]
    assert [item["response_elapsed_control_steps"] for item in result] == [2, 2]


def test_first_active_snapshot_can_also_be_response():
    records = [
        _clock_record(19, pending=(True, False)),
        _clock_record(20, active=(True, False), clock=(1, 0)),
    ]

    result = _classify_records(records, response_steps=1)

    assert result[0]["outcome"] == "response_captured"
    assert result[0]["response"] is result[0]["first_active"]
    assert result[0]["response_elapsed_control_steps"] == 0


def test_reset_before_response_never_uses_next_episode():
    records = [
        _clock_record(19, pending=(True, True)),
        _clock_record(20, active=(True, True), clock=(1, 1)),
        _clock_record(21, active=(False, True), clock=(0, 2)),
        _clock_record(22, active=(False, True), clock=(0, 3), pending=(True, False)),
        _clock_record(23, active=(True, True), clock=(1, 4)),
        _clock_record(24, active=(True, True), clock=(2, 5)),
        _clock_record(25, active=(True, True), clock=(3, 6)),
    ]

    result = _classify_records(records, response_steps=3)

    assert result[0]["outcome"] == "control_ended_before_response"
    assert result[0]["activation_common_step"] == 20
    assert result[0]["response"] is None
    assert result[1]["response_common_step"] == 22


def test_reset_before_activation_never_uses_next_episode():
    records = [
        _clock_record(19, pending=(True, False)),
        _clock_record(20),
        _clock_record(21, pending=(True, False)),
        _clock_record(22, active=(True, False), clock=(1, 0)),
    ]

    result = _classify_records(records, response_steps=1)

    assert result[0]["outcome"] == "reset_before_activation"
    assert result[0]["first_active"] is None
    assert result[0]["response"] is None


@pytest.mark.parametrize("next_step", [19, 18, 21])
def test_rejects_duplicate_backward_or_missing_common_steps(next_step):
    with pytest.raises(ValueError, match="common_step 必须连续递增"):
        _classify_records([_clock_record(19), _clock_record(next_step)])


def test_rejects_control_clock_regression_while_still_active():
    records = [
        _clock_record(19, pending=(True, False)),
        _clock_record(20, active=(True, False), clock=(1, 0)),
        _clock_record(21, active=(True, False), clock=(2, 0)),
        _clock_record(22, active=(True, False), clock=(1, 0)),
    ]
    with pytest.raises(ValueError, match="env 0 的 control_steps 倒退"):
        _classify_records(records)


def test_rejects_regression_immediately_after_activation():
    records = [
        _clock_record(19, pending=(True, False)),
        _clock_record(20, active=(True, False), clock=(1, 0)),
        _clock_record(21, active=(True, False), clock=(0, 0)),
    ]
    with pytest.raises(ValueError, match="env 0 的 control_steps 倒退"):
        _classify_records(records)


def test_rejects_skipped_activation_snapshot():
    records = [
        _clock_record(19, pending=(True, False)),
        _clock_record(20, active=(True, False), clock=(2, 0)),
    ]
    with pytest.raises(ValueError, match="env 0 跳过 activation 边界"):
        _classify_records(records)


def test_rejects_active_records_without_preceding_pending():
    with pytest.raises(ValueError, match="env 0 跳过 pending/activation 边界"):
        _classify_records([_clock_record(20, active=(True, False), clock=(1, 0))])


def test_empty_or_incomplete_records_fail_explicitly():
    with pytest.raises(ValueError, match="records 不能为空"):
        _classify_records([])
    record = _clock_record(19)
    del record["control_steps"]
    with pytest.raises(ValueError, match="缺少字段: control_steps"):
        _classify_records([record])
