"""连续倾角课程的采样、旧RNG合同、局部接管与诊断隔离回归。

@author pony
@date 2026-09-17
@version v0.2.1
@last_modified 2026-09-17
@changelog
    - v0.2.1 (2026-09-17): CLI真实解析回归同时加载新增的分段动作诊断校验器。
    - v0.2.0 (2026-09-17): 零动作对照不得调用策略、修改模板或混入正式验收。
    - v0.1.0 (2026-09-17): 仅CPU检查新课程，禁止把课程诊断标记为正式验收。
"""

from __future__ import annotations

import ast
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import test_recovery_initialization as initialization

# 复用相同CPU替身，显式暴露fixture供pytest按参数名注入。
env = initialization.env
recovery = initialization.recovery


ROOT = Path(__file__).parents[3]
RECOVERY = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/recovery"
EVALUATOR = ROOT / "scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py"


def _legacy_sample(n, weights, events):
    """逐字保留旧有效采样算法，同时比较输出与调用后的RNG状态。"""
    labels = ["upright_fail", "side_lying", "prone", "uniform_so3"]
    w = torch.tensor([max(0.0, float(weights.get(k, 0.0))) for k in labels])
    cat = torch.multinomial(w / w.sum(), n, replacement=True)
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.0
    m = cat == 0
    if bool(m.any()):
        k = int(m.sum())
        ang = torch.empty(k).uniform_(math.radians(20.0), math.radians(60.0))
        sign = torch.where(torch.rand(k) < 0.5, -1.0, 1.0)
        quat[m] = events._quat_about_axis(ang * sign, "y")
    m = cat == 1
    if bool(m.any()):
        k = int(m.sum())
        sign = torch.where(torch.rand(k) < 0.5, -1.0, 1.0)
        quat[m] = events._quat_about_axis(torch.full((k,), math.pi / 2) * sign, "x")
    m = cat == 2
    if bool(m.any()):
        quat[m] = events._quat_about_axis(torch.full((int(m.sum()),), math.pi), "x")
    m = cat == 3
    if bool(m.any()):
        quat[m] = events._random_quat_so3(int(m.sum()), "cpu")
    return quat


@pytest.mark.parametrize("weights", [
    {"upright_fail": .4, "side_lying": .2, "prone": .2, "uniform_so3": .2},
    {"side_lying": 1.0}, {"prone": 1.0}, {"uniform_so3": 1.0},
])
def test_legacy_valid_pose_sampler_preserves_values_and_rng(recovery, weights):
    torch.manual_seed(543)
    expected = _legacy_sample(64, weights, recovery.events)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(543)
    actual = recovery.events._sample_fall_quat(64, "cpu", weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize("weights", [
    {"roll_tilt": 1.0}, {"side_lying": 1.0, "typo": 0.0}, {},
    {"side_lying": 0.0}, {"side_lying": -1.0},
    {"side_lying": float("nan")}, {"prone": float("inf")},
    {"side_lying": 1e300}, {"side_lying": 1e-300},
])
def test_invalid_weights_fail_without_rng_consumption(recovery, weights):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        recovery.events._sample_fall_quat(8, "cpu", weights)
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_explicit_roll_interval_has_both_signs_and_unit_quaternions(recovery):
    torch.manual_seed(543)
    quat = recovery.events._sample_fall_quat(128, "cpu", None, roll_tilt_range_deg=(35., 55.))
    angle = 2 * torch.atan2(quat[:, 1], quat[:, 0]) * 180 / math.pi
    assert torch.all((angle.abs() >= 35.) & (angle.abs() <= 55.))
    assert (angle < 0).any() and (angle > 0).any()
    assert not quat[:, 2:].any()
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones(128))


@pytest.mark.parametrize("bounds", [(-1, 55), (0, 55), (55, 35), (35, 181),
                                        (35,), (35, 45, 55), (float("nan"), 55), (35, float("inf"))])
def test_invalid_roll_interval_rejected_before_reset_write_or_rng(recovery, env, bounds):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        recovery.events.reset_with_fall_pose(env, torch.tensor([1]), roll_tilt_range_deg=bounds)
    assert env.scene["robot"].writes == 0
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_explicit_tilt_and_pose_weights_cannot_silently_override_each_other(recovery, env):
    with pytest.raises(ValueError, match="pose_weights"):
        recovery.events.reset_with_fall_pose(
            env, torch.tensor([1]), pose_weights={"side_lying": 1.}, roll_tilt_range_deg=(35., 55.),
        )
    assert env.scene["robot"].writes == 0


def test_tilt_reset_keeps_geometry_mimic_and_unselected_environments(recovery, env):
    asset = env.scene["robot"]
    env.scene.env_origins[1, 2] = 2.
    before = asset.data.root_state_w.clone()

    def root_height(joint_pos, root_quat, clearance):
        for follower, master, multiplier, offset in asset._recovery_mimic_relations:
            torch.testing.assert_close(joint_pos[:, follower], joint_pos[:, master] * multiplier + offset)
        assert clearance == .002
        assert 35 <= abs(float(2 * torch.atan2(root_quat[0, 1], root_quat[0, 0]) * 180 / math.pi)) <= 55
        return torch.tensor([.46])

    asset._recovery_collision_geometry = SimpleNamespace(root_height=root_height)
    recovery.events.reset_with_fall_pose(
        env, torch.tensor([1]), roll_tilt_range_deg=(35., 55.), collision_clearance=.002,
    )
    assert asset.data.root_pos_w[1, 2].item() == pytest.approx(2.46)
    torch.testing.assert_close(asset.data.root_state_w[[0, 2]], before[[0, 2]])


def test_immediate_takeover_keeps_peers_and_resets_selected_history(recovery, env):
    tree = ast.parse((RECOVERY / "mdp/failure_replay.py").read_text())
    initialize = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "initialize_failure_takeover")
    namespace = {
        "initialize_settled_recovery": recovery.settled.initialize_settled_recovery,
        "zero_action_freefall": recovery.events.zero_action_freefall,
    }
    exec(compile(ast.Module(body=[initialize], type_ignores=[]), "immediate_takeover", "exec"), namespace)
    recovery.settled.initialize_settled_recovery(env, None)
    env._recovery_takeover_pending[:] = True
    recovery.settled.activate_settled_recovery(env, None)
    env._recovery_control_steps[:] = 7
    env.scene["robot"].data.joint_pos[:] = .1
    recovery.observations.prev_joint_pos_legs(env)
    env.scene["robot"].data.joint_pos[1] = .2
    env.action_manager.reset(torch.tensor([1]))
    recovery.observations.reset_prev_obs_cache(env, torch.tensor([1]))
    namespace["initialize_failure_takeover"](env, torch.tensor([1]))
    assert env._recovery_control_active.tolist() == [True, True, True]
    assert env._recovery_control_steps.tolist() == [7, 0, 7]
    assert env._recovery_settle_steps[1] == 0
    assert env.scene["robot"].writes == 0
    torch.testing.assert_close(env.scene["robot"].sim_stiffness[:, :8], torch.full((3, 8), 20.))
    torch.testing.assert_close(env.scene["robot"].sim_damping[:, :8], torch.ones(3, 8))
    history = recovery.observations.prev_joint_pos_legs(env)
    torch.testing.assert_close(history, torch.tensor([.1, .2, .1])[:, None].expand(3, 8))
    env.common_step_counter += 1
    recovery.settled.update_settled_initialization(env)
    assert env._recovery_control_steps.tolist() == [8, 1, 8]


def _load_eval_helpers():
    tree = ast.parse(EVALUATOR.read_text())
    wanted = {"_validate_controlled_tilt_args", "_mark_controlled_tilt_diagnostic", "_evaluation_actions",
              "_validate_action_program_args"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), "exec"), namespace)
    return namespace


def _args(**overrides):
    args = dict(
        skill="recovery", recovery_protocol="controlled_tilt", scenario=None,
        roll_tilt_range_deg=None, settled_initialization="near_ground", preparation_leg_damping=None,
        takeover_velocity_x=0., takeover_pitch_rate=0., failure_states=None, allow_training_pool_smoke=False,
        diagnostic_zero_action=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("overrides", [
    {"scenario": "side_lying"}, {"settled_initialization": "drop"},
    {"preparation_leg_damping": .2}, {"takeover_velocity_x": .1},
    {"takeover_pitch_rate": .1}, {"roll_tilt_range_deg": (55., 35.)},
    {"skill": "upright-balance"}, {"failure_states": Path("fake.npz")},
    {"allow_training_pool_smoke": True},
])
def test_tilt_diagnostic_rejects_mixed_protocols(overrides):
    with pytest.raises(ValueError):
        _load_eval_helpers()["_validate_controlled_tilt_args"](_args(**overrides))


def test_formal_protocol_rejects_curriculum_range():
    with pytest.raises(ValueError):
        _load_eval_helpers()["_validate_controlled_tilt_args"](
            _args(recovery_protocol="settled_ground", scenario="side_lying", roll_tilt_range_deg=(35., 55.))
        )


def test_successful_tilt_diagnostic_cannot_be_formal_acceptance():
    result = {"passed": True, "complete": True, "success_rate": 1., "episode_results": []}
    _load_eval_helpers()["_mark_controlled_tilt_diagnostic"](result, (35., 55.))
    assert result["acceptance_eligible"] is False and result["passed"] is False
    assert result["diagnostic_passed"] is True
    assert result["roll_tilt_range_deg"] == [35., 55.]
    assert result["action_source"] == "checkpoint_policy" and result["checkpoint_policy_executed"]


def test_formal_protocol_rejects_zero_action_diagnostic():
    with pytest.raises(ValueError, match="diagnostic_zero_action"):
        _load_eval_helpers()["_validate_controlled_tilt_args"](
            _args(recovery_protocol="settled_ground", diagnostic_zero_action=True)
        )


def test_zero_action_diagnostic_records_active_pd_not_policy_success():
    result = {"passed": True, "complete": True, "success_rate": 1., "episode_results": []}
    _load_eval_helpers()["_mark_controlled_tilt_diagnostic"](result, (35., 55.), True)
    assert result["action_source"] == "zero_policy_action_with_active_pd"
    assert not result["checkpoint_policy_executed"] and not result["passed"]
    assert not result["acceptance_eligible"] and result["diagnostic_passed"]


def test_zero_actions_skip_policy_and_preserve_template():
    def forbidden_policy(obs):
        raise AssertionError("零动作对照不得调用checkpoint策略")

    template = torch.ones(4, 12, dtype=torch.float64)
    result = _load_eval_helpers()["_evaluation_actions"](forbidden_policy, None, template, True)
    assert result.shape == template.shape and result.dtype == template.dtype and not result.any()
    assert template.eq(1).all()


def test_normal_actions_preserve_policy_input_and_output():
    obs, actions = object(), torch.ones(4, 12)

    def policy(actual_obs):
        assert actual_obs is obs
        return actions

    assert _load_eval_helpers()["_evaluation_actions"](policy, obs, torch.zeros(4, 12), False) is actions


def test_tilt_diagnostic_records_actual_signed_roll():
    angle = math.radians(-40.)
    result = {"passed": False, "complete": True, "episode_results": [
        {"takeover_root_state_w": [0., 0., .4, math.cos(angle / 2), math.sin(angle / 2), 0., 0.]},
        {"takeover_root_state_w": None},
    ]}
    _load_eval_helpers()["_mark_controlled_tilt_diagnostic"](result, (35., 55.))
    assert result["episode_results"][0]["takeover_roll_deg"] == pytest.approx(-40.)
    assert result["episode_results"][1]["takeover_roll_deg"] is None
    assert not result["diagnostic_passed"] and not result["passed"]


@pytest.mark.parametrize("protocol,extra,scenario", [
    ("drop_then_recover", [], "upright_fail"),
    ("settled_ground", ["--scenario", "side_lying"], "side_lying"),
    ("controlled_tilt", ["--roll_tilt_range_deg", "35", "55"], None),
    ("controlled_tilt", ["--diagnostic_zero_action"], None),
])
def test_cli_defaults_preserve_legacy_scenario_and_tilt_is_explicit(monkeypatch, tmp_path, protocol, extra, scenario):
    tree = ast.parse(EVALUATOR.read_text())
    parser_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = _load_eval_helpers()
    namespace.update({"argparse": argparse, "Path": Path,
                      "RECOVERY_SCENARIOS": ("upright_fail", "side_lying", "prone", "uniform_so3"),
                      "TASKS": {"recovery": "unused"},
                      "AppLauncher": SimpleNamespace(add_app_launcher_args=lambda parser: None)})
    exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(EVALUATOR), "exec"), namespace)
    checkpoint = tmp_path / "not_loaded.pt"
    checkpoint.touch()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--task", "unused", "--skill", "recovery",
                                     "--checkpoint", str(checkpoint), "--recovery_protocol", protocol, *extra])
    args = namespace["parse_args"]()
    assert args.scenario == scenario
    assert args.diagnostic_zero_action == ("--diagnostic_zero_action" in extra)


def test_tilt_task_config_keeps_baseline_contract_and_reset_order():
    source = (RECOVERY / "config/settled_ground/tilt_env_cfg.py").read_text()
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes[0].bases[0].id == "SettledRecoveryEventsCfg"
    assert classes[1].bases[0].id == "ParallelogramRoverGroundedRecoveryEnvCfg"
    assignments = {node.targets[0].id: node.value for node in classes[0].body if isinstance(node, ast.Assign)}
    assert set(assignments) == {"reset_fall_pose", "freefall_init", "freefall_hold"}
    assert ast.literal_eval(assignments["freefall_hold"]) is None
    assert "initialize_failure_takeover" in ast.unparse(assignments["freefall_init"])
    assert not any(word in source for word in ("rewards:", "observations:", "actions:"))
    inherited_history_event = object()
    baseline_rewards, baseline_observations, baseline_actions = object(), object(), object()
    baseline = type("Baseline", (), {
        "__post_init__": lambda self: None,
        "rewards": baseline_rewards, "observations": baseline_observations, "actions": baseline_actions,
    })
    namespace = {
        "configclass": lambda cls: cls,
        "EventTerm": lambda **kwargs: SimpleNamespace(**kwargs),
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "initialize_failure_takeover": object(), "reset_with_fall_pose": object(),
        "SettledRecoveryEventsCfg": type("Events", (), {"reset_obs_cache": inherited_history_event}),
        "ParallelogramRoverGroundedRecoveryEnvCfg": baseline,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), "tilt_env_cfg", "exec"), namespace)
    cfg = namespace["ParallelogramRoverControlledTiltRecoveryEnvCfg"]()
    cfg.__post_init__()
    assert cfg.rewards is baseline_rewards and cfg.observations is baseline_observations and cfg.actions is baseline_actions
    assert cfg.events.reset_obs_cache is inherited_history_event
    assert cfg.events.reset_fall_pose.params["roll_tilt_range_deg"] == (35., 55.)
    assert cfg.events.reset_fall_pose.params["pose_weights"] is None
    assert cfg.events.reset_fall_pose.params["collision_clearance"] == .002
    assert cfg.episode_length_s == 5 and not cfg.acceptance_eligible


def test_tilt_registration_and_runner_are_independent():
    source = (RECOVERY / "__init__.py").read_text()
    registrations = [node for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register"]
    tilt = [node for node in registrations if any(
        keyword.arg == "id" and ast.literal_eval(keyword.value) == "RobotLab-Isaac-Recovery-ControlledTilt-Parallelogram-Rover-v0"
        for keyword in node.keywords)]
    assert len(tilt) == 1
    assert "tilt_env_cfg:ParallelogramRoverControlledTiltRecoveryEnvCfg" in ast.unparse(tilt[0])
    source = (RECOVERY / "config/agents/rsl_rl_ppo_cfg.py").read_text()
    runner = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                  and node.name == "ParallelogramRoverControlledTiltRecoveryPPORunnerCfg")
    assert runner.bases[0].id == "ParallelogramRoverRecoveryPPORunnerCfg"
    assert "parallelogram_rover_recovery_controlled_tilt" in ast.unparse(runner)
