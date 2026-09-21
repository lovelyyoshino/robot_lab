# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 覆盖真实 YAML 标签、安全拒绝和观测/动作语义迁移门。
#   - v0.1.0 (2026-09-16): 用真实 Torch MLP 验证输入语义扩维、完整参数复制和拒绝错误 checkpoint。

"""不启动仿真验证两轮运动 warm start；模型输出等价不等于动作技能达标。"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn


path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/upright_warm_start.py"
spec = importlib.util.spec_from_file_location("upright_warm_start", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ActorCriticMLP(nn.Module):
    """与当前 RSL state_dict 同构，真实执行 Linear/ELU，而非 mock 网络输出。"""

    is_recurrent = False
    state_dependent_std = False
    actor_obs_normalization = False
    critic_obs_normalization = False

    def __init__(self, actor_dim, critic_dim, noise_key="std"):
        super().__init__()
        # 复用激活对象，覆盖 RSL MLP.named_children 会去重的实际结构。
        activation = nn.ELU()
        self.actor = nn.Sequential(nn.Linear(actor_dim, 16), activation, nn.Linear(16, 8), activation, nn.Linear(8, 12))
        self.critic = nn.Sequential(nn.Linear(critic_dim, 12), activation, nn.Linear(12, 8), activation, nn.Linear(8, 1))
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.register_parameter(noise_key, nn.Parameter(torch.ones(12)))


def _snapshot(network):
    return {key: value.clone() for key, value in network.state_dict().items()}


def _assert_unchanged(actual, expected):
    assert set(actual) == set(expected)
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("noise_key", ["std", "log_std"])
def test_arbitrary_new_commands_leave_actor_and_critic_outputs_exactly_unchanged(noise_key):
    torch_rng = torch.Generator().manual_seed(42)
    old, new = ActorCriticMLP(42, 47, noise_key), ActorCriticMLP(44, 49, noise_key)
    with torch.no_grad():
        for name, parameter in old.named_parameters():
            if name not in ("std", "log_std"):
                parameter.copy_(torch.randint(-16, 17, parameter.shape, generator=torch_rng).float() / 256)
    source = _snapshot(old)
    source_before = copy.deepcopy(source)
    new_parameter_ids = {key: id(value) for key, value in new.named_parameters()}
    metadata = module.warm_start_upright_motion(source, new)
    assert {key: id(value) for key, value in new.named_parameters()} == new_parameter_ids

    # 用二进制可精确表示的输入/第一层系数排除 GEMM 累加舍入差异，检查真正的函数保持。
    actor_obs = torch.randint(-16, 17, (13, 42), generator=torch_rng).float() / 16
    critic_obs = torch.randint(-16, 17, (13, 47), generator=torch_rng).float() / 16
    for scale in (0.0, 0.1, 1.0, 1.0e6):
        extra = torch.randn(13, 2, generator=torch_rng) * scale
        torch.testing.assert_close(old.actor(actor_obs), new.actor(torch.cat((actor_obs, extra), -1)), rtol=0, atol=0)
        torch.testing.assert_close(old.critic(critic_obs), new.critic(torch.cat((critic_obs, extra), -1)), rtol=0, atol=0)
    new_state = new.state_dict()
    for key, value in source.items():
        if key in ("actor.0.weight", "critic.0.weight"):
            torch.testing.assert_close(new_state[key][:, :value.shape[1]], value, rtol=0, atol=0)
            assert new_state[key][:, -2:].count_nonzero() == 0
        else:
            torch.testing.assert_close(new_state[key], value, rtol=0, atol=0)
    _assert_unchanged(source, source_before)
    _assert_unchanged(old.state_dict(), source_before)
    assert metadata["source_actor_obs_dim"] == 42 and metadata["target_actor_obs_dim"] == 44
    assert metadata["source_critic_obs_dim"] == 47 and metadata["target_critic_obs_dim"] == 49
    assert not metadata["optimizer_transferred"] and not metadata["iteration_transferred"]
    assert not metadata["normalizer_transferred"] and not metadata["skill_success_verified"]
    json.dumps(metadata)


def test_random_float_network_preserves_outputs_to_float32_numerical_tolerance():
    old, new = ActorCriticMLP(42, 47), ActorCriticMLP(44, 49)
    module.warm_start_upright_motion(old.state_dict(), new)
    for role, width in (("actor", 42), ("critic", 47)):
        obs = torch.randn(32, width)
        augmented = torch.cat((obs, torch.randn(32, 2) * 3), -1)
        torch.testing.assert_close(getattr(old, role)(obs), getattr(new, role)(augmented), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "nested_checkpoint", "wrong_actor_input", "wrong_critic_input",
    "wrong_deep_layer", "wrong_noise_shape", "nan", "infinite", "non_tensor", "dtype",
    "normalizer", "non_string_key", "nonpositive_std", "different_noise_schema",
])
def test_bad_source_schema_is_rejected_without_partial_target_mutation(mutation):
    old, new = ActorCriticMLP(42, 47), ActorCriticMLP(44, 49)
    source = _snapshot(old)
    if mutation == "missing":
        del source["critic.4.bias"]
    elif mutation == "extra":
        source["optimizer_state_dict"] = torch.ones(1)
    elif mutation == "nested_checkpoint":
        source = {"model_state_dict": source, "iter": 500}
    elif mutation == "wrong_actor_input":
        source["actor.0.weight"] = torch.zeros(16, 45)
    elif mutation == "wrong_critic_input":
        source["critic.0.weight"] = torch.zeros(12, 48)
    elif mutation == "wrong_deep_layer":
        source["critic.2.weight"] = torch.zeros(9, 12)
    elif mutation == "wrong_noise_shape":
        source["std"] = torch.ones(24)
    elif mutation == "nan":
        source["critic.4.bias"].fill_(float("nan"))
    elif mutation == "infinite":
        source["actor.2.bias"].fill_(float("inf"))
    elif mutation == "non_tensor":
        source["std"] = [1.0] * 12
    elif mutation == "dtype":
        source["critic.4.weight"] = source["critic.4.weight"].double()
    elif mutation == "normalizer":
        source["actor_obs_normalizer._mean"] = torch.zeros(42)
    elif mutation == "non_string_key":
        source[123] = torch.ones(1)
    elif mutation == "nonpositive_std":
        source["std"][0] = 0
    elif mutation == "different_noise_schema":
        source["log_std"] = source.pop("std")
    before = _snapshot(new)
    with pytest.raises((ValueError, TypeError)):
        module.warm_start_upright_motion(source, new)
    _assert_unchanged(new.state_dict(), before)


@pytest.mark.parametrize("mutation", [
    "actor_normalization", "critic_normalization", "non_identity_normalizer", "wrong_activation",
    "wrong_input_dim", "wrong_action_dim", "hidden_connection", "extra_parameter",
    "recurrent", "dependent_std", "no_std",
])
def test_unknown_target_contract_is_rejected(mutation):
    old, new = ActorCriticMLP(42, 47), ActorCriticMLP(44, 49)
    if mutation == "actor_normalization":
        new.actor_obs_normalization = True
    elif mutation == "critic_normalization":
        new.critic_obs_normalization = True
    elif mutation == "non_identity_normalizer":
        new.actor_obs_normalizer = nn.BatchNorm1d(44)
    elif mutation == "wrong_activation":
        new.actor[1] = nn.Tanh()
    elif mutation == "wrong_input_dim":
        new.actor[0] = nn.Linear(42, 16)
    elif mutation == "wrong_action_dim":
        new.actor[-1] = nn.Linear(8, 24)
    elif mutation == "hidden_connection":
        new.critic[2] = nn.Linear(13, 8)
    elif mutation == "extra_parameter":
        new.register_buffer("unexpected", torch.ones(1))
    elif mutation == "recurrent":
        new.is_recurrent = True
    elif mutation == "dependent_std":
        new.state_dependent_std = True
    elif mutation == "no_std":
        del new.std
    before = _snapshot(new)
    with pytest.raises(ValueError):
        module.warm_start_upright_motion(old.state_dict(), new)
    _assert_unchanged(new.state_dict(), before)


@pytest.mark.parametrize("kwargs", [
    {"source_actor_obs_normalization": True}, {"source_critic_obs_normalization": True},
    {"source_activation": "relu"},
])
def test_declared_source_normalization_or_activation_mismatch_is_rejected(kwargs):
    old, new = ActorCriticMLP(42, 47), ActorCriticMLP(44, 49)
    with pytest.raises(ValueError):
        module.warm_start_upright_motion(old.state_dict(), new, **kwargs)


def test_log_std_overflow_is_rejected():
    old, new = ActorCriticMLP(42, 47, "log_std"), ActorCriticMLP(44, 49, "log_std")
    source = _snapshot(old)
    source["log_std"].fill_(1000)
    with pytest.raises(ValueError, match="标准差"):
        module.warm_start_upright_motion(source, new)


def test_training_yaml_accepts_only_safe_isaac_tuple_and_slice_tags(tmp_path):
    path = tmp_path / "env.yaml"
    path.write_text("viewer:\n  eye: !!python/tuple [3.0, 3.0, 2.0]\nindices: !!python/object/apply:builtins.slice [null, null, null]\n")
    result = module.load_training_config(path)
    assert result == {"viewer": {"eye": (3.0, 3.0, 2.0)}, "indices": slice(None)}


@pytest.mark.parametrize("payload", [
    "value: !!python/object/apply:os.system ['false']",
    "value: !!python/name:builtins.eval ''",
    "value: !!python/object/apply:builtins.slice [0, 1, 0]",
    "[not, a, mapping]",
])
def test_training_yaml_rejects_executable_unknown_or_invalid_tags(tmp_path, payload):
    import yaml

    path = tmp_path / "env.yaml"
    path.write_text(payload)
    with pytest.raises((ValueError, yaml.YAMLError)):
        module.load_training_config(path)


def _config_contract():
    prefixes = {
        "policy": ["pitch_command", "projected_gravity", "base_lin_vel", "base_ang_vel", "joint_pos_legs", "joint_vel_legs", "wheel_vel", "last_action"],
    }
    prefixes["critic"] = prefixes["policy"] + ["priv_base_height", "priv_wheel_contact"]
    groups = {}
    for group, names in prefixes.items():
        groups[group] = {"concatenate_terms": True, "concatenate_dim": -1, "history_length": None,
                         "flatten_history_dim": True, "enable_corruption": group == "policy"}
        for name in names:
            groups[group][name] = {"func": "module:" + name, "params": {"indices": slice(None)},
                                   "scale": None, "clip": None, "modifiers": None, "noise": None,
                                   "history_length": 0, "flatten_history_dim": True}
    env = {"observations": groups, "actions": {"legs": {"joint_names": ["a", "b"], "scale": .25}},
           "sim": {"dt": .005}, "decimation": 4,
           "scene": {"robot": {"init_state": {"joint_pos": {"a": -.61}, "joint_vel": {".*": 0.}}}}}
    target = copy.deepcopy(env)
    for group in prefixes:
        target["observations"][group]["motion_command"] = {"func": "isaaclab.envs.mdp.observations:generated_commands",
                                                           "params": {"command_name": "motion_command"},
                                                           "scale": None, "clip": None, "modifiers": None, "noise": None,
                                                           "history_length": 0, "flatten_history_dim": True}
    agent = {"clip_actions": None, "obs_groups": {}, "empirical_normalization": None,
             "policy": {"class_name": "ActorCritic", "activation": "elu", "noise_std_type": "scalar",
                        "actor_obs_normalization": False, "critic_obs_normalization": False,
                        "actor_hidden_dims": [128] * 3, "critic_hidden_dims": [128] * 3}}
    active = {group: names + ["motion_command"] for group, names in prefixes.items()}
    return env, target, agent, copy.deepcopy(agent), active


def test_complete_training_contract_accepts_only_the_appended_motion_command():
    result = module.validate_upright_motion_contract(*_config_contract())
    assert result["configuration_contract_checked"] is True


@pytest.mark.parametrize("mutation", ["function_module", "params", "clip", "modifiers", "noise", "history",
                                      "action_scale", "action_order", "default_pose", "dt", "wrapper_clip", "active_order"])
def test_training_contract_rejects_semantic_drift_even_with_same_dimensions(mutation):
    source, target, source_agent, target_agent, active = _config_contract()
    term = target["observations"]["policy"]["base_lin_vel"]
    if mutation == "function_module":
        term["func"] = "different_module:base_lin_vel"
    elif mutation in ("params", "clip", "modifiers", "noise"):
        term[mutation] = {"changed": True}
    elif mutation == "history":
        target["observations"]["policy"]["history_length"] = 2
    elif mutation == "action_scale":
        target["actions"]["legs"]["scale"] = 5.
    elif mutation == "action_order":
        target["actions"]["legs"]["joint_names"].reverse()
    elif mutation == "default_pose":
        target["scene"]["robot"]["init_state"]["joint_pos"]["a"] = .61
    elif mutation == "dt":
        target["sim"]["dt"] = .01
    elif mutation == "wrapper_clip":
        target_agent["clip_actions"] = 1.
    elif mutation == "active_order":
        active["policy"][0], active["policy"][1] = active["policy"][1], active["policy"][0]
    with pytest.raises(ValueError):
        module.validate_upright_motion_contract(source, target, source_agent, target_agent, active)
