"""恢复脚本动作的限位、独立随机数、接管时钟与诊断身份回归。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 实际关节响应统计须排除准备、非有限值和已结束样本。
    - v0.1.0 (2026-09-17): 覆盖分段动作生成与正式技能验收隔离。
"""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "scripts/reinforcement_learning/rsl_rl"
spec = importlib.util.spec_from_file_location("recovery_action_program", SCRIPTS / "recovery_action_program.py")
program_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program_module)


def make_program(**overrides):
    kwargs = dict(
        default_targets=torch.zeros(3, 12), scales=torch.tensor([.25] * 8 + [.8] * 4),
        leg_limits=torch.tensor([[[-.6, .6]] * 4 + [[-1.2, 1.2]] * 4] * 3),
        wheel_velocity_limits=torch.full((3, 4), 20.), command_clip=torch.tensor([[-100., 100.]] * 12),
        raw_clip=100., dt=.02, segment_s=.5, active_s=3., seed=731,
        leg_fraction=1., wheel_speed=5.,
    )
    kwargs.update(overrides)
    return program_module.RecoveryActionProgram(**kwargs)


def test_sampling_preserves_global_rng_and_paired_program_identity():
    before = torch.random.get_rng_state().clone()
    small = make_program(leg_fraction=.25)
    large = make_program(leg_fraction=1.)
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
    torch.testing.assert_close(small.targets[:, :, :8] * 4, large.targets[:, :, :8])
    torch.testing.assert_close(small.targets[:, :, 8:], large.targets[:, :, 8:], atol=0, rtol=0)
    torch.testing.assert_close(make_program().targets, large.targets, atol=0, rtol=0)


def test_physical_targets_respect_asymmetric_limits_and_inverse_action_mapping():
    defaults = torch.zeros(3, 12)
    defaults[:, 0] = .2
    limits = torch.tensor([[[-.6, .6]] * 4 + [[-1.2, 1.2]] * 4] * 3)
    limits[:, 0] = torch.tensor([.1, .6])
    program = make_program(default_targets=defaults, leg_limits=limits)
    assert (program.targets[:, :, :8] >= limits[:, None, :, 0]).all()
    assert (program.targets[:, :, :8] <= limits[:, None, :, 1]).all()
    assert program.targets[:, :, 8:].abs().max() <= 5.
    active = torch.tensor([True, True, False])
    actions = program.actions(torch.tensor([0, 25, 0]), active)
    actual = actions * program.scales + defaults
    torch.testing.assert_close(actual[0], program.targets[0, 0])
    torch.testing.assert_close(actual[1], program.targets[1, 1])
    torch.testing.assert_close(actual[2], defaults[2])


def test_preparation_does_not_consume_program_and_hold_returns_default_pd():
    program = make_program()
    active = torch.tensor([True, True, False])
    actions = program.actions(torch.tensor([149, 150, 70]), active)
    torch.testing.assert_close(actions[0] * program.scales[0], program.targets[0, 5])
    assert not actions[1:].any()
    # 下一次局部接管从0开始，不受其它环境已进入保持段影响。
    actions = program.actions(torch.tensor([150, 150, 0]), torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(actions[2] * program.scales[2], program.targets[2, 0])


@pytest.mark.parametrize("overrides", [
    {"leg_fraction": -1}, {"leg_fraction": 1.1}, {"wheel_speed": float("nan")},
    {"wheel_speed": 21}, {"segment_s": .015}, {"active_s": 3.1}, {"active_s": 4.5},
    {"scales": torch.zeros(12)}, {"raw_clip": .01},
    {"command_clip": torch.tensor([[-.01, .01]] * 12)},
    {"leg_limits": torch.full((3, 8, 2), float("inf"))},
    {"default_targets": torch.ones(3, 12) * 2},
])
def test_invalid_or_saturating_contracts_fail_before_sampling(overrides):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        make_program(**overrides)
    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)


def test_diagnostic_report_cannot_claim_checkpoint_policy_success():
    program = make_program()
    result = {"passed": True, "complete": True, "success_rate": 1.}
    program.mark_diagnostic(result)
    assert result["passed"] is False and result["diagnostic_passed"] is True
    assert result["checkpoint_policy_executed"] is False
    assert result["acceptance_eligible"] is False
    assert result["action_source"] == "segmented_joint_targets_with_active_pd"
    assert result["action_program"]["targets_shape"] == [3, 6, 12]
    assert result["action_program"]["segment_steps"] == 25
    assert result["action_program"]["active_steps"] == 150


def test_transient_support_is_separate_from_terminal_hold_and_finished_rows_freeze():
    program = make_program()
    active = finite = torch.ones(3, dtype=torch.bool)
    stable = torch.tensor([True, False, False])
    support = torch.tensor([True, True, False])
    for _ in range(3):
        program.observe(torch.tensor([-1., -.8, .2]), stable, support, active, finite)
    program.finished[0] = True
    for _ in range(3):
        program.observe(torch.tensor([-1., -.8, .2]), stable, support, active, finite)
    assert program.episode_metrics(0)["maximum_stable_time_s"] == pytest.approx(.06)
    assert program.episode_metrics(1)["maximum_stable_time_s"] == 0
    assert program.episode_metrics(1)["maximum_four_wheel_support_time_s"] == pytest.approx(.12)
    assert not program.actions(torch.zeros(3, dtype=torch.long), active)[0].any()


def test_zero_amplitude_program_is_default_pd_not_zero_torque():
    program = make_program(leg_fraction=0., wheel_speed=0.)
    assert not program.raw_actions.any()
    result = {"passed": False}
    program.mark_diagnostic(result)
    assert result["action_program"]["hold_mode"] == "zero_raw_action_default_pose_active_pd"


def test_joint_response_statistics_exclude_preparation_finished_and_nonfinite():
    program = make_program()
    active = torch.tensor([True, False, True])
    positions, torques = torch.full((3, 12), .3), torch.full((3, 12), 4.)
    positions[2, 0] = float("nan")
    program.observe_joints(positions, torques, active)
    assert program.episode_metrics(0)["active_joint_range_rad"][0] == pytest.approx([.3, .3])
    assert program.episode_metrics(1)["active_joint_range_rad"] is None
    assert program.episode_metrics(2)["active_joint_range_rad"] is None
    program.finished[0] = True
    positions[:] = .8
    program.observe_joints(positions, torques * 2, active)
    assert program.episode_metrics(0)["maximum_actuator_estimated_torque_nm"] == [4.] * 12
    assert program.episode_metrics(2)["maximum_actuator_estimated_torque_nm"] == [8.] * 12


def load_validator():
    path = SCRIPTS / "evaluate_parallelogram_skills.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "_validate_action_program_args"]
    assert len(nodes) == 1
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_validate_action_program_args"]


@pytest.mark.parametrize("override", [
    {"skill": "upright-balance"}, {"recovery_protocol": "controlled_tilt"},
    {"episodes_per_env": 2}, {"diagnostic_zero_action": True},
    {"task": "RobotLab-Isaac-Recovery-Settled-Parallelogram-Rover-v0"}, {"hold_time_s": .1},
])
def test_program_only_uses_one_episode_strict_grounded_recovery(override):
    args = dict(diagnostic_action_program=True, skill="recovery", recovery_protocol="settled_ground",
                task="RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0", episodes_per_env=1,
                diagnostic_zero_action=False, hold_time_s=None)
    args.update(override)
    with pytest.raises(ValueError):
        load_validator()(SimpleNamespace(**args))


def test_program_cli_does_not_silently_ignore_parameters_or_overwrite_evidence(tmp_path):
    with pytest.raises(ValueError, match="显式启用"):
        load_validator()(SimpleNamespace(diagnostic_action_program=False, program_leg_fraction=.5))
    output = tmp_path / "existing.json"
    output.write_text('{"evidence":"preserve"}')
    args = SimpleNamespace(diagnostic_action_program=True, skill="recovery", recovery_protocol="settled_ground",
                           task="RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0", episodes_per_env=1,
                           diagnostic_zero_action=False, hold_time_s=None, output=output)
    with pytest.raises(ValueError, match="不会覆盖"):
        load_validator()(args)
    assert output.read_text() == '{"evidence":"preserve"}'
