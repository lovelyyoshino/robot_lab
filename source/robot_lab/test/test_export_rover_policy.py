"""离线 rover 策略导出合同测试；不启动 Isaac/GPU。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 直接读取真实训练 YAML，覆盖函数/尺度/关节/周期及 normalizer 严格导出。
    - v0.1.0 (2026-09-17): 验证 motion 合同拒绝路径与导出工具的前置检查。
"""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch


@pytest.fixture
def exporter_module():
    path = Path(__file__).parents[3] / "scripts/tools/export_rover_policy.py"
    spec = importlib.util.spec_from_file_location("export_rover_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def real_config(exporter_module):
    """Return a self-contained motion contract instead of a deleted run artifact.

    The exporter tests exercise the semantic validator, not a particular training
    run.  Keeping the fixture in the test makes it reproducible after old logs are
    cleaned and prevents a missing experiment directory from masking regressions.
    """
    def term(name, *, params=None):
        if name in ("pitch_command", "motion_command"):
            func = "isaaclab.envs.mdp.observations:generated_commands"
            params = {"command_name": name}
        elif name == "last_action":
            func = "isaaclab.envs.mdp.observations:last_action"
            params = {}
        elif name == "priv_wheel_contact":
            func = "robot_lab.tasks.manager_based.upright.mdp.observations:priv_wheel_contact"
            params = {"sensor_cfg": {"name": "contact_forces", "body_names": "^(fl|rl|fr|rr)_wheel$"}}
        else:
            func = exporter_module.ASSET_FUNCTIONS[name]
            params = {"asset_cfg": {"name": "robot", "preserve_order": False}}
        return {
            "func": func, "params": params, "modifiers": None, "noise": None,
            "clip": None, "scale": None, "history_length": 0, "flatten_history_dim": True,
        }

    policy_terms = {name: term(name) for name in exporter_module.POLICY_TERMS}
    critic_terms = {name: term(name) for name in exporter_module.CRITIC_TERMS}
    agent = {
        "class_name": "OnPolicyRunner",
        "empirical_normalization": None,
        "obs_groups": {},
        "policy": {
            "class_name": "ActorCritic", "init_noise_std": 1.0, "noise_std_type": "scalar",
            "actor_obs_normalization": False, "critic_obs_normalization": False,
            "actor_hidden_dims": [128, 128, 128], "critic_hidden_dims": [128, 128, 128],
            "activation": "elu",
        },
    }
    actions = {
        "joint_pos": {"joint_names": list(exporter_module.JOINT_POS), "scale": 0.25, "offset": 0.0},
        "joint_vel": {"joint_names": list(exporter_module.JOINT_VEL), "scale": 0.8, "offset": 0.0},
    }
    default = {"joint_pos": dict(exporter_module.DEFAULT_JOINT_STATE),
               "joint_vel": dict(exporter_module.DEFAULT_JOINT_STATE)}
    env = {
        "observations": {"policy": policy_terms, "critic": critic_terms},
        "actions": actions,
        "scene": {"robot": {"init_state": default}},
        "sim": {"dt": 0.005},
        "decimation": 4,
    }
    return agent, env


def test_contract_accepts_real_motion_yaml(exporter_module, real_config):
    agent, env = real_config
    assert exporter_module.validate_motion_contract(agent, env) == {
        "actor_observation_dim": 44, "critic_observation_dim": 49, "action_dim": 12, "control_dt_s": .02,
        "actor_normalization": False, "critic_normalization": False,
    }


@pytest.mark.parametrize("target,key,value", [("policy", "class_name", "StudentTeacher"), ("agent", "obs_groups", {"policy": ["other"]})])
def test_contract_rejects_distillation_or_unknown_groups(exporter_module, real_config, target, key, value):
    agent, env = deepcopy(real_config)
    (agent["policy"] if target == "policy" else agent)[key] = value
    with pytest.raises(ValueError):
        exporter_module.validate_motion_contract(agent, env)


@pytest.mark.parametrize("mutator", [
    lambda env: env["observations"]["policy"]["base_lin_vel"].update(func="wrong:function"),
    lambda env: env["observations"]["policy"]["base_lin_vel"].update(scale=.5),
    lambda env: env["actions"]["joint_pos"].update(joint_names=list(reversed(env["actions"]["joint_pos"]["joint_names"]))),
    lambda env: env["sim"].update(dt=.01),
])
def test_contract_rejects_same_dimension_semantic_changes(exporter_module, real_config, mutator):
    agent, env = deepcopy(real_config)
    mutator(env)
    with pytest.raises(ValueError):
        exporter_module.validate_motion_contract(agent, env)


def test_contract_rejects_observation_order_change(exporter_module, real_config):
    agent, env = deepcopy(real_config)
    env["observations"]["policy"] = dict(reversed(tuple(env["observations"]["policy"].items())))
    with pytest.raises(ValueError, match="观测项顺序"):
        exporter_module.validate_motion_contract(agent, env)


def test_contract_rejects_legacy_normalization_and_unknown_std_flag(exporter_module, real_config):
    agent, env = deepcopy(real_config)
    agent["empirical_normalization"] = True
    with pytest.raises(ValueError, match="empirical_normalization"):
        exporter_module.validate_motion_contract(agent, env)
    agent, env = deepcopy(real_config)
    agent["policy"]["state_dependent_std"] = True
    with pytest.raises(ValueError, match="state_dependent_std"):
        exporter_module.validate_motion_contract(agent, env)


def test_normalizer_strict_load_and_official_jit_are_equivalent(exporter_module, real_config, tmp_path):
    agent, env = deepcopy(real_config)
    agent["policy"]["actor_obs_normalization"] = True
    contract = exporter_module.validate_motion_contract(agent, env)
    policy = exporter_module.build_policy(agent, contract).cpu().eval()
    with torch.no_grad():
        policy.actor_obs_normalizer.mean.fill_(.25)
        policy.actor_obs_normalizer.std.fill_(1.5)
    checkpoint = tmp_path / "normalizer.pt"
    torch.save({"model_state_dict": policy.state_dict(), "iter": 3}, checkpoint)
    restored = exporter_module.build_policy(agent, contract).cpu().eval()
    assert exporter_module.load_checkpoint(restored, checkpoint) == 3
    exporter_module._module(exporter_module.OFFICIAL_EXPORTER, "export_test_official").export_policy_as_jit(
        restored, restored.actor_obs_normalizer, str(tmp_path), "normalizer_policy.pt",
    )
    inputs = torch.randn(8, 44)
    scripted = torch.jit.load(str(tmp_path / "normalizer_policy.pt"))
    torch.testing.assert_close(restored.actor(restored.actor_obs_normalizer(inputs)), scripted(inputs))


def test_checkpoint_extra_key_is_rejected_after_strict_load(exporter_module, real_config, tmp_path):
    agent, env = real_config
    policy = exporter_module.build_policy(agent, exporter_module.validate_motion_contract(agent, env))
    state = policy.state_dict()
    state["unexpected"] = torch.zeros(1)
    checkpoint = tmp_path / "extra.pt"
    torch.save({"model_state_dict": state, "iter": 1}, checkpoint)
    with pytest.raises(ValueError, match="合同不一致"):
        exporter_module.load_checkpoint(policy, checkpoint)
