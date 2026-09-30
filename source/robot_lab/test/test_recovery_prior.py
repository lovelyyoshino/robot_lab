# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-30
# @version v0.4.0
# @last_modified 2026-09-30
# @changelog
#   - v0.4.0 (2026-09-30): 覆盖3秒后闭环、局部reset和续训动作合同漂移。
#   - v0.3.0 (2026-09-30): 覆盖接近正立时释放先验的消融 profile。
#   - v0.2.0 (2026-09-30): 覆盖分阶段 trust-region 与 profile 选择。
#   - v0.2.0 (2026-09-30): 覆盖来源任务/动作语义和混合program身份选择。
#   - v0.1.0 (2026-09-30): 覆盖分段时钟、左右镜像、有界 residual 和零均值初始化。

"""program31 residual 先验的纯 Python/Torch 回归测试。"""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/reinforcement_learning/rsl_rl/recovery_prior.py"
spec = importlib.util.spec_from_file_location("recovery_prior_test_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_prior(tmp_path, profile="constant"):
    targets = np.zeros((32, 6, 12), dtype=np.float32)
    targets[31, 0] = np.arange(12, dtype=np.float32)
    targets[31, 1] = np.arange(12, dtype=np.float32) + 10
    # row1 模拟同一 program 的 canonical、未镜像评测环境；row31 仅用于
    # program_id 身份选择，真实 NPZ 也会为每个 vectorized env 保存一行。
    targets[1] = targets[31]
    metadata = {
        "source": {"task": module.EXPECTED_SOURCE_TASK, "action_program": {
            "targets": targets.tolist(), "default_targets": np.zeros((32, 12)).tolist(),
            "action_scales": np.ones((32, 12)).tolist(), "segment_steps": 25,
            "active_steps": 150, "step_dt": .02, "source_program_ids": [31] * 32,
            "mirrored": [True] + [False] * 31,
            "targets_shape": [32, 6, 12], "joint_names": list(module.EXPECTED_JOINT_NAMES),
            "target_units": module.EXPECTED_TARGET_UNITS, "hold_mode": module.EXPECTED_HOLD_MODE,
        }},
    }
    path = tmp_path / "prior.npz"
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    return module.RecoveryPrior.from_npz(path, program_id=31, num_envs=2,
                                         residual_profile=profile)


def env(active=True, side=1.0, step=0):
    return SimpleNamespace(
        device=torch.device("cpu"), num_envs=2,
        _recovery_control_active=torch.tensor([active, active]),
        _recovery_control_steps=torch.tensor([step, step]),
        _recovery_takeover_gravity_b=torch.tensor([[0., side, 0.], [0., -side, 0.]]),
    )


def test_prior_preserves_segment_clock_and_mirrors_physical_order(tmp_path):
    prior = make_prior(tmp_path)
    raw = prior.compose(env(step=25), torch.zeros(2, 12))
    # gravity_y>0 的 env0 使用镜像；env1 使用 canonical program。
    expected = (np.arange(12)[[2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]] + 10).astype(np.float32)
    expected[4:8] *= -1
    np.testing.assert_allclose(raw[0].numpy(), expected)
    np.testing.assert_allclose(raw[1].numpy(), np.arange(12, dtype=np.float32) + 10)


def test_prior_residual_is_bounded_and_inactive_is_zero(tmp_path):
    prior = make_prior(tmp_path)
    residual = torch.full((2, 12), 100.)
    composed = prior.compose(env(side=-1.0, step=0), residual)
    # first env: target + configured scale; no unbounded action can bypass tanh.
    assert torch.all(composed[0] <= torch.arange(12, dtype=torch.float32) + module.DEFAULT_RESIDUAL_SCALE + 1e-6)
    inactive = prior.compose(env(active=False, step=0), residual)
    assert torch.count_nonzero(inactive) == 0
    hold = prior.compose(env(step=150), residual)
    assert torch.count_nonzero(hold) == 0


def test_launch_protected_profile_changes_only_the_selected_segment(tmp_path):
    prior = make_prior(tmp_path, profile="launch_protected")
    residual = torch.full((2, 12), 100.)
    first = prior.compose(env(side=-1.0, step=0), residual)
    third = prior.compose(env(side=-1.0, step=50), residual)
    assert torch.allclose(
        first[0], torch.arange(12, dtype=torch.float32) +
        torch.as_tensor(module.LAUNCH_PROTECTED_RESIDUAL_SCALE[0]), atol=1e-6,
    )
    assert torch.allclose(
        third[0], torch.as_tensor(module.LAUNCH_PROTECTED_RESIDUAL_SCALE[2]), atol=1e-6,
    )
    assert not torch.equal(first[0], third[0])


def test_residual_profile_scale_is_not_mutable_global():
    profile = module.residual_profile_scale("launch_protected")
    profile[0, 0] = 99.0
    assert module.LAUNCH_PROTECTED_RESIDUAL_SCALE[0, 0] != 99.0


def test_upright_release_disables_prior_but_keeps_residual_path(tmp_path):
    prior = make_prior(tmp_path, profile="upright_release")
    upright = env(side=-1.0, step=0)
    upright.scene = {"robot": type("Robot", (), {
        "data": type("Data", (), {
            "projected_gravity_b": torch.tensor([[0., 0., -0.9], [0., 0., -0.9]])
        })()
    })()}
    released = prior.compose(upright, torch.zeros(2, 12))
    assert torch.count_nonzero(released) == 0

    side = env(side=-1.0, step=0)
    side.scene = {"robot": type("Robot", (), {
        "data": type("Data", (), {
            "projected_gravity_b": torch.tensor([[0., -1., 0.], [0., -1., 0.]])
        })()
    })()}
    retained = prior.compose(side, torch.zeros(2, 12))
    np.testing.assert_allclose(retained[0].numpy(), np.arange(12, dtype=np.float32))


def test_zero_mean_initialization_clears_last_actor_layer():
    class Layer:
        def __init__(self):
            self.weight = torch.ones(3, 4)
            self.bias = torch.ones(3)

    class Policy:
        actor = [Layer()]
        std = torch.ones(3)

    policy = Policy()
    result = module.initialize_residual_policy(policy, std=.1)
    assert result["actor_mean_zero"] is True
    assert not policy.actor[-1].weight.any() and not policy.actor[-1].bias.any()
    assert torch.all(policy.std == .1)


def test_mixed_program_prior_selects_requested_canonical_row(tmp_path):
    targets = np.zeros((4, 6, 12), dtype=np.float32)
    targets[0] = 1.0
    targets[1] = 7.0
    targets[2] = 2.0
    targets[3] = 8.0
    metadata = {
        "source": {"task": module.EXPECTED_SOURCE_TASK, "action_program": {
            "targets": targets.tolist(), "default_targets": np.zeros((4, 12)).tolist(),
            "action_scales": np.ones((4, 12)).tolist(), "segment_steps": 25,
            "active_steps": 150, "step_dt": .02, "source_program_ids": [7, 31, 7, 31],
            "mirrored": [False, False, True, True], "targets_shape": [4, 6, 12],
            "joint_names": list(module.EXPECTED_JOINT_NAMES),
            "target_units": module.EXPECTED_TARGET_UNITS, "hold_mode": module.EXPECTED_HOLD_MODE,
        }},
    }
    path = tmp_path / "mixed.npz"
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    prior = module.RecoveryPrior.from_npz(path, program_id=31, num_envs=1)
    one_env = SimpleNamespace(
        device=torch.device("cpu"), num_envs=1,
        _recovery_control_active=torch.ones(1, dtype=torch.bool),
        _recovery_control_steps=torch.zeros(1, dtype=torch.long),
        _recovery_takeover_gravity_b=torch.tensor([[0., -1., 0.]]),
    )
    composed = prior.compose(one_env, torch.zeros(1, 12))
    np.testing.assert_allclose(composed.numpy(), targets[1, 0][None])


@pytest.mark.parametrize("mutation", [
    lambda meta: meta["source"].update(task="RobotLab-Isaac-Recovery-Upside-Parallelogram-Rover-v0"),
    lambda meta: meta["source"]["action_program"].update(target_units="normalized"),
    lambda meta: meta["source"]["action_program"].update(joint_names=list(reversed(module.EXPECTED_JOINT_NAMES))),
])
def test_prior_rejects_semantic_contract_mismatch(tmp_path, mutation):
    path = tmp_path / "bad.npz"
    metadata = {"source": {"task": module.EXPECTED_SOURCE_TASK, "action_program": {
        "targets": np.zeros((1, 6, 12)).tolist(), "default_targets": np.zeros((1, 12)).tolist(),
        "action_scales": np.ones((1, 12)).tolist(), "segment_steps": 25, "active_steps": 150,
        "step_dt": .02, "source_program_ids": [31], "mirrored": [False],
        "targets_shape": [1, 6, 12], "joint_names": list(module.EXPECTED_JOINT_NAMES),
        "target_units": module.EXPECTED_TARGET_UNITS, "hold_mode": module.EXPECTED_HOLD_MODE,
    }}}
    mutation(metadata)
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError):
        module.RecoveryPrior.from_npz(path, program_id=31, num_envs=1)


def test_runtime_contract_rejects_replaced_prior_bytes(tmp_path):
    path = tmp_path / "prior.npz"
    path.write_bytes(b"prior-v1")
    runtime = module.prior_runtime_contract(
        path, program_id=31, residual_profile="constant", task="residual-task",
    )
    module.validate_prior_runtime_contract(runtime, path, program_id=31, task="residual-task")
    path.write_bytes(b"prior-v2")
    with pytest.raises(ValueError, match="哈希"):
        module.validate_prior_runtime_contract(runtime, path, program_id=31, task="residual-task")


def test_settle_feedback_keeps_bounded_residual_after_prior_window(tmp_path):
    prior = make_prior(tmp_path, profile="settle_feedback")
    legacy = make_prior(tmp_path, profile="constant")
    residual = torch.full((2, 12), 100.)
    torch.testing.assert_close(prior.compose(env(step=149), residual), legacy.compose(env(step=149), residual))
    for step in (150, 200, 249):
        actual = prior.compose(env(step=step), residual)
        torch.testing.assert_close(actual, torch.as_tensor(module.SETTLE_RESIDUAL_SCALE).expand(2, -1))
        assert torch.count_nonzero(prior.compose(env(step=step), torch.zeros(2, 12))) == 0
    mixed = env(step=200)
    mixed._recovery_control_active[0] = False
    mixed._recovery_control_steps[0] = 0
    actual = prior.compose(mixed, residual)
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[1]) == 12


def test_settle_feedback_gradient_exists_only_for_controlled_environments(tmp_path):
    prior = make_prior(tmp_path, profile="settle_feedback")
    state = env(step=175)
    state._recovery_control_active[1] = False
    actions = torch.zeros(2, 12, requires_grad=True)
    prior.compose(state, actions).sum().backward()
    torch.testing.assert_close(actions.grad[0], torch.as_tensor(module.SETTLE_RESIDUAL_SCALE))
    assert torch.count_nonzero(actions.grad[1]) == 0


@pytest.mark.parametrize("entrypoint", ["train.py", "play.py", "evaluate_parallelogram_skills.py"])
def test_all_entrypoints_resolve_contract_before_creating_environment(entrypoint):
    tree = ast.parse((SCRIPT.parent / entrypoint).read_text())
    checks = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "resolve_checkpoint_prior"]
    creates = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "make"
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "gym"]
    assert len(checks) == len(creates) == 1
    assert checks[0] < creates[0]


def test_residual_contract_resolves_auto_but_rejects_silent_migration(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
    side_task = "RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0"
    replay_task = "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0"
    path = tmp_path / "prior.npz"
    path.write_bytes(b"prior")
    checkpoint = tmp_path / "run" / "model_500.pt"
    params = checkpoint.parent / "params"
    params.mkdir(parents=True)
    runtime = module.prior_runtime_contract(path, program_id=31, residual_profile="constant", task=side_task)
    (params / "recovery_residual_runtime.json").write_text(json.dumps(runtime))
    assert module.resolve_checkpoint_prior(checkpoint, path, program_id=31, task=side_task) == "constant"
    with pytest.raises(ValueError, match="profile"):
        module.resolve_checkpoint_prior(
            checkpoint, path, program_id=31, task=side_task, residual_profile="settle_feedback",
        )
    with pytest.raises(ValueError, match="program_id"):
        module.resolve_checkpoint_prior(checkpoint, path, program_id=7, task=side_task)
    with pytest.raises(ValueError, match="任务"):
        module.resolve_checkpoint_prior(checkpoint, path, program_id=31, task=replay_task)
    # 只有训练迁移入口允许Side和失败重放之间切换，回放/评测仍严格匹配任务。
    assert module.resolve_checkpoint_prior(
        checkpoint, path, program_id=31, task=replay_task, allow_task_transfer=True,
    ) == "constant"
    assert module.resolve_checkpoint_prior(
        checkpoint, path, program_id=31, task=side_task, residual_profile="settle_feedback",
        allow_profile_transfer=True,
    ) == "settle_feedback"
    runtime["action_semantics"]["settle_residual_scale"][0] = 9.0
    (params / "recovery_residual_runtime.json").write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="动作语义"):
        module.resolve_checkpoint_prior(checkpoint, path, program_id=31, task=side_task)


def test_weight_only_transfer_keeps_new_optimizer_and_iteration(tmp_path):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(3, 2)
            self.std = torch.nn.Parameter(torch.ones(2))

    source, target = Policy(), Policy()
    checkpoint = tmp_path / "model.pt"
    (tmp_path / "params").mkdir()
    (tmp_path / "params/recovery_residual_runtime.json").write_text('{"residual_profile":"constant"}')
    torch.save({"model_state_dict": source.state_dict(), "optimizer_state_dict": {"fake": 1}, "iter": 500}, checkpoint)
    optimizer = torch.optim.Adam(target.parameters())
    metadata = module.warm_start_residual_policy(target, checkpoint, std=.15)
    torch.testing.assert_close(source.actor.weight, target.actor.weight)
    torch.testing.assert_close(target.std, torch.full((2,), .15))
    assert not optimizer.state
    assert not metadata["optimizer_restored"] and not metadata["iteration_restored"]


def test_missing_runtime_never_guesses_residual_contract(tmp_path):
    path = tmp_path / "prior.npz"
    path.write_bytes(b"prior")
    with pytest.raises(FileNotFoundError, match="recovery_residual_runtime"):
        module.resolve_checkpoint_prior(tmp_path / "model.pt", path, program_id=31, task="side")


@pytest.mark.parametrize("schema", ["recovery_residual_runtime_v1", "recovery_residual_runtime_v2"])
def test_legacy_runtime_keeps_original_profile_and_requires_recorded_identity(tmp_path, schema):
    path = tmp_path / "prior.npz"
    path.write_bytes(b"prior")
    runtime = {"schema": schema, "prior_program_id": 31, "residual_profile": "constant", "task": "side",
               "prior_data": str(path)}
    if schema.endswith("v2"):
        runtime["prior_sha256"] = module.prior_file_sha256(path)
    module.validate_prior_runtime_contract(runtime, path, program_id=31, task="side")
    runtime.pop("prior_program_id")
    with pytest.raises(ValueError, match="program_id"):
        module.validate_prior_runtime_contract(runtime, path, program_id=31, task="side")
