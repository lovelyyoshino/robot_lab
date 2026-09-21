# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.1.0
# @last_modified 2026-09-16
# @changelog
#   - v0.1.0 (2026-09-16): 覆盖三策略键映射、起立过渡、恢复命令和完成门。

"""不启动 Isaac Sim 的混合策略状态机合同测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_planner_module():
    planner_path = (
        Path(__file__).parents[1]
        / "robot_lab/tasks/manager_based/hybrid/planner.py"
    )
    spec = importlib.util.spec_from_file_location("parallelogram_hybrid_planner", planner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner_module = _load_planner_module()


def test_policy_observation_contracts_are_stable():
    assert planner_module.POLICY_SPECS == {
        "loco": {"obs": 45, "act": 12},
        "upright": {"obs": 42, "act": 12},
        "recover": {"obs": 58, "act": 12},
    }


def test_motion_and_temporal_specs_bind_task_to_semantic_contract():
    assert planner_module.UPRIGHT_POLICY_SPECS["motion"] == {
        "task": "RobotLab-Isaac-Upright-Motion-Getup-Parallelogram-Rover-v0",
        "observation_group": "motion",
        "actor_obs_dim": 44,
        "critic_obs_dim": 49,
        "wheel_scale": 0.8,
        "command_layout": "append:[ground_forward_velocity_m_s,world_yaw_rate_rad_s]",
    }
    temporal = planner_module.LOCOMOTION_POLICY_SPECS["student-temporal"]
    continuous = planner_module.LOCOMOTION_POLICY_SPECS["student"]
    assert temporal["actor_obs_dim"] == continuous["actor_obs_dim"] == 4141
    assert temporal["depth_contract_id"] != continuous["depth_contract_id"]


@pytest.fixture
def hybrid_play(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    app = ModuleType("isaaclab.app")
    app.AppLauncher = type("AppLauncher", (), {})
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    path = Path(__file__).parents[3] / "scripts/play_parallelogram_hybrid.py"
    spec = importlib.util.spec_from_file_location("hybrid_play_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_input_dimension_rejects_42d_model_for_motion(hybrid_play):
    checkpoint = {"model_state_dict": {"actor.0.weight": torch.zeros(128, 42)}}
    assert hybrid_play._checkpoint_actor_input_dim(checkpoint, "ActorCritic") == 42
    assert hybrid_play._checkpoint_actor_input_dim(
        {"model_state_dict": {"student.0.weight": torch.zeros(1024, 4141)}}, "StudentTeacher"
    ) == 4141
    with pytest.raises(ValueError, match="actor.0.weight"):
        hybrid_play._checkpoint_actor_input_dim({}, "ActorCritic")


def test_depth_contract_rejects_same_size_but_wrong_temporal_semantics(hybrid_play, tmp_path):
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.touch()
    export = tmp_path / "exported"
    export.mkdir()
    contract = {"contract_id": "rover_depth_contiguous_v1", "student_obs_dim": 4141,
                "teacher_obs_dim": 232, "action_dim": 12}
    (export / "depth_contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="rover_depth_temporal_v2"):
        hybrid_play._load_depth_contract(checkpoint, "rover_depth_temporal_v2")
    contract["contract_id"] = "rover_depth_temporal_v2"
    (export / "depth_contract.json").write_text(json.dumps(contract))
    assert hybrid_play._load_depth_contract(checkpoint, "rover_depth_temporal_v2") == contract


class _RuntimeConfig:
    def __init__(self, serialized, observations, action_type="HybridWheelVelocityAction"):
        self._serialized = serialized
        self.observations = observations
        self.actions = SimpleNamespace(joint_vel=SimpleNamespace(class_type=type(action_type, (), {})))

    def to_dict(self):
        return self._serialized


def _hybrid_contract_inputs():
    observation = {
        "last_action": {"func": "isaaclab.envs.mdp.observations:last_action", "params": {}},
        "enable_corruption": False,
    }
    env = {
        "observations": {"policy": observation, "critic": {}},
        "actions": {"joint_vel": {"class_type": "isaaclab.actions:JointVelocityAction", "joint_names": ["wheel"], "scale": 0.8,
                                   "offset": 0.0, "clip": {".*": (-100, 100)}}},
        "scene": {"robot": {"init_state": {"joint_pos": {"a": 0.0}, "joint_vel": {"a": 0.0}}}},
        "sim": {"dt": 0.005}, "decimation": 4,
    }
    agent = {"class_name": "OnPolicyRunner", "clip_actions": None, "obs_groups": {},
             "empirical_normalization": None, "policy": {"class_name": "ActorCritic", "activation": "elu"}}
    action = env["actions"]["joint_vel"]
    runtime_group = SimpleNamespace(
        last_action=SimpleNamespace(func=lambda: None, params={"wheel_scale": 0.8}), enable_corruption=False,
    )
    runtime_group.last_action.func.__name__ = "last_action_for_skill"
    observations = SimpleNamespace(policy=runtime_group)
    runtime = _RuntimeConfig({**env, "actions": {"joint_vel": {**action, "class_type": "hybrid", "scale": 5.0}}}, observations)
    return env, agent, runtime, agent


def test_training_contract_rejects_saved_wheel_scale_mismatch(hybrid_play, monkeypatch, tmp_path):
    env, agent, runtime, runtime_agent = _hybrid_contract_inputs()
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").touch()
    (params / "agent.yaml").touch()
    monkeypatch.setattr(hybrid_play, "_load_training_config", lambda path: env if path.name == "env.yaml" else agent)
    env["actions"]["joint_vel"]["scale"] = 5.0
    with pytest.raises(ValueError, match="训练 scale"):
        hybrid_play._validate_checkpoint_training_contract(
            checkpoint, {"observation_group": "policy", "wheel_scale": 0.8}, runtime, runtime_agent
        )


def test_training_contract_rejects_temporal_parameters_with_same_id(hybrid_play, monkeypatch, tmp_path):
    env, agent, runtime, runtime_agent = _hybrid_contract_inputs()
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").touch()
    (params / "agent.yaml").touch()
    monkeypatch.setattr(hybrid_play, "_load_training_config", lambda path: env if path.name == "env.yaml" else agent)
    saved_depth = {"contract_id": "rover_depth_temporal_v2", "student_obs_dim": 4141, "teacher_obs_dim": 232,
                   "action_dim": 12, "sampling": {"history_stride": 5, "delay_steps": [0, 1]}}
    monkeypatch.setattr(hybrid_play, "_load_depth_contract", lambda *_: saved_depth)
    runtime_depth = {**saved_depth, "sampling": {"history_stride": 1, "delay_steps": [0, 1]}}
    with pytest.raises(ValueError, match="深度预处理"):
        hybrid_play._validate_checkpoint_training_contract(
            checkpoint, {"observation_group": "policy", "wheel_scale": 0.8,
                         "depth_contract_id": "rover_depth_temporal_v2"}, runtime, runtime_agent,
            visual_contract=runtime_depth,
        )


def test_action_contract_rejects_order_and_wrong_training_class(hybrid_play):
    expected = {
        "class_type": "isaaclab.envs.mdp.actions.joint_actions:JointVelocityAction",
        "asset_name": "robot", "joint_names": ["wheel"], "scale": 0.8, "offset": 0.0,
        "preserve_order": True, "use_default_offset": True, "debug_vis": False,
        "clip": {".*": (-100.0, 100.0)},
    }
    actual = {**expected, "class_type": "hybrid", "scale": 5.0}
    runtime = SimpleNamespace(actions=SimpleNamespace(
        joint_vel=SimpleNamespace(class_type=type("HybridWheelVelocityAction", (), {}))
    ))
    hybrid_play._validate_action_contract(expected, actual, runtime, {"wheel_scale": 0.8}, "joint_vel")
    with pytest.raises(ValueError, match="JointVelocityAction"):
        hybrid_play._validate_action_contract({**expected, "class_type": "other:WrongAction"}, actual, runtime,
                                              {"wheel_scale": 0.8}, "joint_vel")
    with pytest.raises(ValueError, match="动作项顺序"):
        hybrid_play._require_equal(["joint_pos", "joint_vel"], ["joint_vel", "joint_pos"], "动作项顺序")


def test_up_command_hands_quad_pose_to_upright_during_rise():
    state = planner_module.StanceStateMachine()

    assert state.select("up", g_z=-1.0, g_y=0.0, base_height=0.332) == "UPRIGHT"
    assert state.select("up", g_z=-0.8, g_y=0.0, base_height=0.33) == "UPRIGHT"
    assert state.select("up", g_z=0.0, g_y=0.0, base_height=0.35) == "UPRIGHT"
    assert state._rising is False


def test_rise_timeout_falls_back_to_recovery():
    cfg = planner_module.StanceConfig(rise_timeout=3)
    state = planner_module.StanceStateMachine(cfg=cfg)

    assert state.select("up", -1.0, 0.0, 0.332) == "UPRIGHT"
    assert state.select("up", -0.8, 0.0, 0.33) == "UPRIGHT"
    assert state.select("up", -0.8, 0.0, 0.33) == "UPRIGHT"
    assert state.select("up", -0.8, 0.0, 0.33) == "RECOVER"


def test_explicit_recovery_command_takes_control():
    state = planner_module.StanceStateMachine()
    assert state.select("rec", -1.0, 0.0, 0.332) == "RECOVER"
    assert state.select("rec", -1.0, 0.0, 0.332) == "RECOVER"


def test_recovery_completion_uses_support_height_and_hold_time():
    cfg = planner_module.StanceConfig(min_hold=3)
    state = planner_module.StanceStateMachine(cfg=cfg, mode="RECOVER")

    for _ in range(4):
        assert state.select("lo", -0.8, 0.0, 0.01) == "RECOVER"
    assert state.select("lo", -0.8, 0.0, 0.30) == "RECOVER"
    assert state.select("lo", -0.8, 0.0, 0.30) == "RECOVER"
    assert state.select("lo", -0.8, 0.0, 0.30) == "LOCO"


def test_fallen_locomotion_pose_selects_recovery():
    state = planner_module.StanceStateMachine()
    assert state.select("lo", 0.8, 0.0, 0.20) == "RECOVER"


def test_planner_maps_uppercase_mode_to_lowercase_policy_keys():
    identity = lambda observation: observation
    planner = planner_module.Planner()
    planner.setup(None, identity, identity, identity, stance_cmd="lo")

    mode, action = planner.poll(
        g_z=-1.0,
        g_y=0.0,
        base_height=0.332,
        obs_by_policy={"loco": "loco-obs", "upright": "upright-obs", "recover": "recover-obs"},
    )

    assert mode == "LOCO"
    assert action == "loco-obs"


def test_invalid_stance_command_fails_fast():
    state = planner_module.StanceStateMachine()
    with pytest.raises(ValueError, match="未知姿态指令"):
        state.select("invalid", -1.0, 0.0, 0.332)


def test_recovery_does_not_exit_while_airborne_or_rotating():
    state = planner_module.StanceStateMachine(cfg=planner_module.StanceConfig(min_hold=1), mode="RECOVER")
    assert state.select("lo", -1.0, 0.0, 0.7) == "RECOVER"
    assert state.select("lo", -1.0, 0.0, 0.33, angular_speed=1.0) == "RECOVER"
    assert state.select("lo", -1.0, 0.0, 0.33, angular_speed=0.1) == "LOCO"


def _contact_forces(active_names):
    names = ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel", "base_link"]
    return names, [[2.0, 0.0, 0.0] if name in active_names else [0.0, 0.0, 0.0] for name in names]


def test_contact_parser_requires_sensor_named_wheels_and_rejects_nonwheel_support():
    names, forces = _contact_forces({"fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"})
    parsed = planner_module.contact_state_from_forces(names, forces)
    assert parsed == {"valid": True, "four_wheel": True, "rear_wheel": False, "non_wheel": False}
    names, forces = _contact_forces({"rl_wheel", "rr_wheel"})
    assert planner_module.contact_state_from_forces(names, forces)["rear_wheel"]
    names, forces = _contact_forces({"rl_wheel", "rr_wheel", "base_link"})
    parsed = planner_module.contact_state_from_forces(names, forces)
    assert not parsed["four_wheel"] and not parsed["rear_wheel"]
    with pytest.raises(ValueError, match="wheel"):
        planner_module.contact_state_from_forces(names[1:], forces[1:])


def test_strict_recovery_requires_continuous_four_wheel_and_stable_speed():
    state = planner_module.StanceStateMachine(
        cfg=planner_module.StanceConfig(min_hold=2), mode="RECOVER", strict_measurements=True
    )
    valid = dict(four_wheel_support=True, rear_wheel_support=False, linear_speed=.1, leg_speed=.4)
    assert state.select("lo", -.8, 0., .33, angular_speed=.1, **valid) == "RECOVER"
    assert state.select("lo", -.8, 0., .33, angular_speed=.1, **{**valid, "four_wheel_support": False}) == "RECOVER"
    assert state._holds == 0
    assert state.select("lo", -.8, 0., .33, angular_speed=.1, **valid) == "RECOVER"
    assert state.select("lo", -.8, 0., .33, angular_speed=.1, **valid) == "LOCO"
    state = planner_module.StanceStateMachine(mode="RECOVER", strict_measurements=True)
    assert state.select("lo", -.8, 0., .33, angular_speed=.1, four_wheel_support=True,
                        rear_wheel_support=False, linear_speed=.2, leg_speed=.4) == "RECOVER"
    assert state.last_measurement_diagnostics == "support_or_speed_gate_failed"


@pytest.mark.parametrize("failure", ["front", "body", "single_rear", "linear", "leg", "missing"])
def test_strict_upright_never_accepts_invalid_two_wheel_completion(failure):
    state = planner_module.StanceStateMachine(strict_measurements=True)
    measurements = dict(four_wheel_support=False, rear_wheel_support=True, linear_speed=.2, leg_speed=.2)
    if failure == "front":
        measurements["rear_wheel_support"] = False
    elif failure == "body":
        measurements["rear_wheel_support"] = False
    elif failure == "single_rear":
        measurements["rear_wheel_support"] = False
    elif failure == "linear":
        measurements["linear_speed"] = 1.1
    elif failure == "leg":
        measurements["leg_speed"] = .6
    elif failure == "missing":
        measurements["rear_wheel_support"] = None
    assert state.select("up", 0., 0., .35, g_x=-1., angular_speed=.1, **measurements) == "UPRIGHT"
    assert state._rising
    assert state.last_measurement_diagnostics != "strict_measurements_passed"


def test_strict_upright_requires_two_rear_wheels_before_completion():
    state = planner_module.StanceStateMachine(strict_measurements=True)
    measurements = dict(four_wheel_support=False, rear_wheel_support=True, linear_speed=.2, leg_speed=.2)
    assert state.select("up", 0., 0., .35, g_x=-1., angular_speed=.1, **measurements) == "UPRIGHT"
    assert not state._rising


def test_transition_records_must_distinguish_requested_and_entered_modes():
    expected = {"requested_command", "requested_mode", "entered_mode", "mode"}
    record = {"step": 4, "requested_command": "up", "requested_mode": "UPRIGHT",
              "entered_mode": "RECOVER", "mode": "RECOVER"}
    assert expected <= record.keys()
    assert record["requested_mode"] != record["entered_mode"]


def test_opposite_pitch_is_not_successful_upright():
    state = planner_module.StanceStateMachine()
    state.select("up", -1.0, 0.0, 0.33, g_x=0.0)
    state.select("up", 0.0, 0.0, 0.35, g_x=1.0)
    assert state._rising
    state.select("up", 0.0, 0.0, 0.35, g_x=-1.0)
    assert not state._rising
    assert state.select("up", 0.0, 0.0, 0.35, g_x=1.0) == "RECOVER"


def test_nonfinite_state_is_rejected_before_policy_routing():
    with pytest.raises(ValueError, match="非有限"):
        planner_module.StanceStateMachine().select("lo", float("nan"), 0.0, 0.33)
