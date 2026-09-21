# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-17
# @version v0.1.0
# @last_modified 2026-09-17
# @changelog
#   - v0.1.0 (2026-09-17): 覆盖语义合同、回合泄漏、actor-only 拟合与 RSL 安全重载。

"""CPU 上验证行为克隆初始化合同；不启动仿真，也不把离线拟合当作技能验收。"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn


@pytest.fixture
def pretraining(monkeypatch):
    directory = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("recovery_pretraining_test_module",
                                                directory / "pretrain_recovery_demonstrations.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class TinyPolicy(nn.Module):
    """保持真实参数命名、normalizer 和前向合同，仅缩小隐藏层以加快 CPU 回归。"""

    is_recurrent = False
    state_dependent_std = False
    actor_obs_normalization = False
    critic_obs_normalization = False

    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.ones(12))
        self.actor = nn.Sequential(nn.Linear(58, 16), nn.ReLU(), nn.Linear(16, 12))
        self.critic = nn.Sequential(nn.Linear(70, 16), nn.ReLU(), nn.Linear(16, 1))
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()


@pytest.fixture
def configuration(pretraining):
    terms = {
        name: {"func": f"recovery.observations:{name}", "params": {"joint_ids": slice(None)},
               "noise": {"n_min": -0.01, "n_max": 0.01}, "scale": None,
               "clip": None, "history_length": 0, "flatten_history_dim": True, "modifiers": None}
        for name in pretraining.OBSERVATION_ORDER
    }
    legs, wheels = [f"leg_{index}" for index in range(8)], [f"wheel_{index}" for index in range(4)]
    actions = {
        "joint_pos": {"joint_names": legs, "scale": 0.25, "offset": 0.0, "use_default_offset": True,
                      "preserve_order": True, "class_type": "actions:JointPositionAction"},
        "joint_vel": {"joint_names": wheels, "scale": 0.8, "offset": 0.0, "use_default_offset": True,
                      "preserve_order": True, "class_type": "actions:JointVelocityAction"},
    }
    default = {"joint_pos": {".*": 0.0}, "joint_vel": {".*": 0.0}}
    env = {"observations": {"policy": {"enable_corruption": True, "concatenate_terms": True, **terms}},
           "actions": actions, "scene": {"robot": {"init_state": default}}, "sim": {"dt": 0.005}, "decimation": 4}
    agent = {"policy": copy.deepcopy(pretraining.SUPPORTED_POLICY), "clip_actions": None, "obs_groups": {},
             "empirical_normalization": None, "algorithm": {"learning_rate": 0.001}}
    group = copy.deepcopy(env["observations"]["policy"])
    group["enable_corruption"] = False
    contract = pretraining.canonical_config({
        "actor_observation_dim": 58, "critic_observation_dim": 70, "action_dim": 12,
        "policy_config": agent["policy"], "wrapper_raw_clip": None,
        "observation_order": pretraining.OBSERVATION_ORDER,
        "observation_term_dims": [[dim] for dim in pretraining.OBSERVATION_DIMS],
        "observation_group": group, "action_config": actions, "default_joint_state": default,
        "action_joint_names": legs + wheels, "action_scales": [0.25] * 8 + [0.8] * 4,
        "action_offsets": [0.0] * 12, "step_dt": 0.02,
    })
    metadata = {
        "schema": "rover_recovery_demonstrations_v1", "contract": contract,
        "source_evaluation_path": "/fixture/train.json", "source_evaluation_sha256": "a" * 64,
        "source": {"task": "grounded_fixture", "seed": 742, "checkpoint_sha256": "b" * 64,
                   "action_source": "segmented_joint_targets_with_active_pd",
                   "evaluation_scope": "bounded_action_program_search_not_policy_acceptance",
                   "recovery_support_protocol": "four_wheel_stable_v1"},
        "episodes": [{"trajectory_index": 0, "env_id": 31, "episode_index": 0, "program_id": 31},
                     {"trajectory_index": 1, "env_id": 109, "episode_index": 0, "program_id": 109}],
    }
    return env, agent, metadata


@pytest.fixture
def arrays():
    generator = np.random.default_rng(22)
    observations = generator.normal(size=(500, 58)).astype(np.float32)
    scale = np.array([2.0] * 4 + [5.0] * 4 + [37.5] * 4, dtype=np.float32)
    actions = ((0.2 + 0.03 * observations[:, :12]) * scale).astype(np.float32)
    return {"observations": observations, "actions": actions,
            "episode_id": np.repeat(np.arange(2, dtype=np.int64), 250),
            "control_step": np.tile(np.arange(250, dtype=np.int64), 2)}


def test_source_contract_accepts_only_evaluation_corruption_difference(pretraining, configuration):
    env, agent, metadata = configuration
    before = copy.deepcopy(metadata)
    pretraining.validate_source_contract(metadata, env, agent, "b" * 64)
    assert metadata == before
    assert env["observations"]["policy"]["enable_corruption"] is True


@pytest.mark.parametrize("mutation", [
    lambda data: data["contract"].update(step_dt=0.04),
    lambda data: data["contract"].update(wrapper_raw_clip=1.0),
    lambda data: data["contract"]["policy_config"].update(activation="elu"),
    lambda data: data["contract"]["observation_group"]["projected_gravity"].update(func="wrong:gravity"),
    lambda data: data["contract"]["observation_group"]["projected_gravity"].update(scale=2.0),
    lambda data: data["contract"]["observation_group"]["projected_gravity"]["noise"].update(n_max=0.02),
    lambda data: data["contract"]["action_config"]["joint_pos"].update(scale=0.5),
    lambda data: data["contract"]["default_joint_state"]["joint_pos"].update({".*": 0.2}),
    lambda data: data["contract"]["action_scales"].__setitem__(8, 1.0),
    lambda data: data["contract"]["action_offsets"].__setitem__(0, 0.1),
    lambda data: data["source"].update(checkpoint_sha256="c" * 64),
])
def test_semantic_mismatch_rejected(pretraining, configuration, mutation):
    env, agent, metadata = configuration
    mutation(metadata)
    with pytest.raises(ValueError, match="不匹配"):
        pretraining.validate_source_contract(metadata, env, agent, "b" * 64)


@pytest.mark.parametrize("field", ["observations", "actions"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_examples_rejected_before_actor_changes(pretraining, arrays, configuration, field, value):
    policy = TinyPolicy()
    before = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    arrays[field][0, 0] = value
    with pytest.raises(ValueError, match="非有限"):
        pretraining.fit_actor(policy, arrays, configuration[2], epochs=1)
    for key, tensor in policy.state_dict().items():
        assert torch.equal(tensor, before[key])


def test_incomplete_or_frame_split_episode_rejected(pretraining, arrays, configuration):
    arrays["control_step"][12] = 11
    with pytest.raises(ValueError, match="完整连续"):
        pretraining.validate_arrays(arrays, configuration[2])


def test_source_leakage_rejected_even_if_json_hash_is_rewritten(pretraining, configuration):
    metadata = configuration[2]
    copied = copy.deepcopy(metadata)
    with pytest.raises(ValueError, match="泄漏"):
        pretraining.validate_dataset_separation(metadata, copied, "same", "same")
    with pytest.raises(ValueError, match="泄漏"):
        pretraining.validate_dataset_separation(metadata, copied, "train", "validation")
    copied["source_evaluation_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="泄漏"):
        pretraining.validate_dataset_separation(metadata, copied, "train", "validation")


def test_same_program_different_seed_is_explicitly_not_independent(pretraining, configuration):
    metadata = configuration[2]
    validation = copy.deepcopy(metadata)
    validation["source"]["seed"] = 842
    validation["source_evaluation_sha256"] = "c" * 64
    result = pretraining.validate_dataset_separation(metadata, validation, "train", "validation")
    assert result["program_overlap"] == [31, 109]
    assert result["source_disjoint"] is True
    assert result["statistically_independent"] is False
    assert result["random_frame_split"] is False


def test_fitting_changes_only_actor_and_is_reproducible(pretraining, arrays, configuration):
    torch.manual_seed(19)
    policy = TinyPolicy()
    copy_policy = copy.deepcopy(policy)
    before = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    metadata = configuration[2]
    result = pretraining.fit_actor(policy, arrays, metadata, epochs=12, batch_size=250, learning_rate=0.01, seed=4)
    repeated = pretraining.fit_actor(copy_policy, arrays, metadata, epochs=12, batch_size=250, learning_rate=0.01, seed=4)
    assert result == repeated
    assert result["train_final"]["normalized_mse"] < result["train_initial"]["normalized_mse"]
    assert result["skill_success_verified"] is False
    assert result["selection"] == "final_epoch_no_validation_selection"
    assert any(not torch.equal(tensor, before[key]) for key, tensor in policy.state_dict().items()
               if key.startswith("actor."))
    for key, tensor in policy.state_dict().items():
        if not key.startswith("actor."):
            assert torch.equal(tensor, before[key])
        assert torch.equal(tensor, copy_policy.state_dict()[key])
    assert all(parameter.requires_grad for parameter in policy.parameters())
    assert policy.training is True


def test_validation_does_not_change_loss_scale_or_selected_weights(pretraining, arrays, configuration):
    policy = TinyPolicy()
    other = copy.deepcopy(policy)
    validation = copy.deepcopy(arrays)
    validation["actions"] *= 100.0
    base = pretraining.fit_actor(policy, arrays, configuration[2], epochs=2, seed=19)
    result = pretraining.fit_actor(other, arrays, configuration[2], validation=validation,
                                  validation_metadata=configuration[2], epochs=2, seed=19)
    assert result["action_loss_scales"] == base["action_loss_scales"]
    assert result["validation_final"]["normalized_mse"] > result["train_final"]["normalized_mse"]
    for key, tensor in policy.state_dict().items():
        assert torch.equal(tensor, other.state_dict()[key])


def test_checkpoint_contains_fresh_all_parameter_adam_and_zero_ppo_iteration(pretraining, tmp_path):
    policy = TinyPolicy()
    checkpoint = pretraining.make_initialization_checkpoint(policy, {"epochs": 400}, 0.001)
    assert checkpoint["iter"] == 0
    assert checkpoint["optimizer_state_dict"]["state"] == {}
    assert len(checkpoint["optimizer_state_dict"]["param_groups"][0]["params"]) == len(list(policy.parameters()))
    assert checkpoint["infos"]["ppo_updates"] == 0
    assert checkpoint["infos"]["skill_success_verified"] is False
    path = tmp_path / pretraining.CHECKPOINT_NAME
    torch.save(checkpoint, path)
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    restored = TinyPolicy()
    pretraining.load_source_policy(restored, loaded)
    torch.optim.Adam(restored.parameters()).load_state_dict(loaded["optimizer_state_dict"])
    observations = torch.randn(3, 58)
    assert torch.equal(policy.actor(observations), restored.actor(observations))


def test_real_rsl_tensordict_policy_and_ppo_optimizer_load(pretraining, configuration, tmp_path):
    pytest.importorskip("rsl_rl")
    from rsl_rl.algorithms import PPO
    from tensordict import TensorDict

    policy = pretraining.build_policy(configuration[1], "cpu")
    checkpoint = pretraining.make_initialization_checkpoint(policy, {}, 0.001)
    path = tmp_path / pretraining.CHECKPOINT_NAME
    torch.save(checkpoint, path)
    reloaded = torch.load(path, weights_only=True, map_location="cpu")
    target = pretraining.build_policy(configuration[1], "cpu")
    pretraining.load_source_policy(target, reloaded)
    ppo = PPO(target, device="cpu")
    ppo.optimizer.load_state_dict(reloaded["optimizer_state_dict"])
    observations = TensorDict({"policy": torch.randn(3, 58), "critic": torch.randn(3, 70)}, batch_size=[3])
    assert torch.equal(policy.act_inference(observations), target.act_inference(observations))
    assert torch.equal(policy.evaluate(observations), target.evaluate(observations))
    assert ppo.optimizer.state_dict()["state"] == {}


def test_run_copies_original_configs_records_lineage_and_never_overwrites(
    pretraining, configuration, arrays, tmp_path, monkeypatch,
):
    env, agent, metadata = configuration
    source_dir = tmp_path / "source"
    params = source_dir / "params"
    params.mkdir(parents=True)
    for name, value in (("env.yaml", env), ("agent.yaml", agent)):
        (params / name).write_text(yaml.dump(value, sort_keys=False), encoding="utf-8")
    source_policy = TinyPolicy()
    source_path = source_dir / "model_399.pt"
    torch.save({"model_state_dict": source_policy.state_dict(), "iter": 399,
                "optimizer_state_dict": {"old_state_must_not_be_loaded": True}}, source_path)
    metadata["source"]["checkpoint_sha256"] = pretraining.sha256_file(source_path)
    dataset = tmp_path / "train.npz"
    dataset.write_bytes(b"injected_loader_fixture")
    monkeypatch.setattr(pretraining, "load_demonstrations", lambda path: (arrays, metadata))
    monkeypatch.setattr(pretraining, "build_policy", lambda config, device: TinyPolicy().to(device))
    args = argparse.Namespace(source_checkpoint=source_path, train_data=dataset, validation_data=None,
                              output_dir=tmp_path / "result", epochs=2, batch_size=250,
                              learning_rate=0.001, seed=942, device="cpu")
    checkpoint_path = pretraining.run_pretraining(args)
    assert checkpoint_path.name == "behavior_cloning_initialization.pt"
    assert not list(args.output_dir.glob("model_*.pt"))
    for name in ("env.yaml", "agent.yaml"):
        assert (args.output_dir / "params" / name).read_bytes() == (params / name).read_bytes()
    lineage = json.loads((args.output_dir / "params/bc_lineage.json").read_text())
    assert lineage["source_iteration"] == 399
    assert lineage["iteration_reset_to"] == 0
    assert lineage["old_critic_retained"] is True
    assert lineage["std_trained"] is False
    assert lineage["optimizer_state_empty"] is True
    assert lineage["train_data"]["sha256"] == pretraining.sha256_file(dataset)
    assert lineage["output_checkpoint_sha256"] == pretraining.sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    assert checkpoint["optimizer_state_dict"]["state"] == {}
    for key, tensor in checkpoint["model_state_dict"].items():
        if not key.startswith("actor."):
            assert torch.equal(tensor, source_policy.state_dict()[key])
    with pytest.raises(FileExistsError, match="独占新建"):
        pretraining.run_pretraining(args)


def test_nonfinite_source_checkpoint_is_rejected_before_mutation(pretraining):
    policy = TinyPolicy()
    checkpoint = {"model_state_dict": copy.deepcopy(policy.state_dict())}
    checkpoint["model_state_dict"]["actor.0.weight"][0, 0] = float("nan")
    before = copy.deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match="非有限"):
        pretraining.load_source_policy(policy, checkpoint)
    for key, tensor in policy.state_dict().items():
        assert torch.equal(tensor, before[key])
