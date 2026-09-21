"""训练源 checkpoint 谱系的哈希、迭代与输入校验回归。

@author pony
@date 2026-09-17
@version v1.2.0
@last_modified 2026-09-17
@changelog
    - v1.2.0 (2026-09-17): 覆盖 checkpoint 学习率恢复、显式覆盖及多 param group 拒绝。
    - v1.1.0 (2026-09-17): 覆盖 scalar/log 探索噪声缩放与 Adam 动量重置。
    - v1.0.0 (2026-09-17): 覆盖 resume 谱系生成、落盘和损坏输入拒绝。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def lineage():
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/experiment_lineage.py"
    spec = importlib.util.spec_from_file_location("experiment_lineage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_source_metadata_records_immutable_identity(lineage, tmp_path):
    checkpoint = tmp_path / "model_17.pt"
    torch.save({"iter": 17, "model_state_dict": {"weight": torch.ones(2)}}, checkpoint)
    metadata = lineage.checkpoint_source_metadata(
        checkpoint, load_mode="resume", task="RobotLab-Test-v0", seed=542,
    )
    assert metadata["source_checkpoint"] == str(checkpoint.resolve())
    assert metadata["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["source_checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert metadata["source_iteration"] == 17
    output = lineage.write_checkpoint_source_metadata(tmp_path / "params", metadata)
    assert json.loads(output.read_text()) == metadata


@pytest.mark.parametrize("noise_type", ["scalar", "log"])
def test_scale_policy_exploration_noise_preserves_ratios_and_resets_moments(lineage, noise_type):
    class Policy(torch.nn.Module):
        state_dependent_std = False
        noise_std_type = noise_type

        def __init__(self):
            super().__init__()
            initial = torch.tensor([.2, .4, .8])
            if noise_type == "scalar":
                self.std = torch.nn.Parameter(initial)
            else:
                self.log_std = torch.nn.Parameter(initial.log())

    policy = Policy()
    parameter = next(policy.parameters())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    parameter.sum().backward()
    optimizer.step()
    before = policy.std.detach().clone() if noise_type == "scalar" else policy.log_std.detach().exp().clone()
    metadata = lineage.scale_policy_exploration_noise(policy, optimizer, .25)
    actual = policy.std.detach() if noise_type == "scalar" else policy.log_std.detach().exp()
    torch.testing.assert_close(actual, before * .25, atol=1e-6, rtol=1e-6)
    assert metadata["before_mean"] == pytest.approx(before.mean().item())
    assert metadata["after_mean"] == pytest.approx(actual.mean().item())
    assert metadata["optimizer_moments_reset"] == ["exp_avg", "exp_avg_sq"]
    assert not optimizer.state[parameter]["exp_avg"].any()
    assert not optimizer.state[parameter]["exp_avg_sq"].any()


@pytest.mark.parametrize(("override", "expected"), [(None, 2.5e-5), (1.0e-4, 1.0e-4)])
def test_restore_resumed_learning_rate_synchronizes_algorithm_and_optimizer(lineage, override, expected):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=2.5e-5)

    class Algorithm:
        learning_rate = 1.0e-3

    algorithm = Algorithm()
    algorithm.optimizer = optimizer
    metadata = lineage.restore_resumed_learning_rate(algorithm, override)
    assert algorithm.learning_rate == pytest.approx(expected)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    assert metadata["configured_learning_rate_before_restore"] == pytest.approx(1.0e-3)
    assert metadata["checkpoint_optimizer_learning_rates"] == pytest.approx([2.5e-5])
    assert metadata["override_requested"] is (override is not None)
    assert metadata["effective_learning_rate"] == pytest.approx(expected)


def test_restore_resumed_learning_rate_rejects_ambiguous_param_groups(lineage):
    first = torch.nn.Parameter(torch.ones(1))
    second = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([{"params": [first], "lr": 1.0e-5}, {"params": [second], "lr": 2.0e-5}])
    algorithm = type("Algorithm", (), {"learning_rate": 1.0e-3, "optimizer": optimizer})()
    with pytest.raises(ValueError, match="不一致"):
        lineage.restore_resumed_learning_rate(algorithm)


def test_restore_resumed_learning_rate_keeps_loaded_adam_state(lineage, tmp_path):
    parameter = torch.nn.Parameter(torch.ones(1))
    source = torch.optim.Adam([parameter], lr=1.0e-5)
    parameter.sum().backward()
    source.step()
    checkpoint = tmp_path / "optimizer.pt"
    torch.save(source.state_dict(), checkpoint)
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = torch.optim.Adam([resumed_parameter], lr=1.0e-3)
    resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    algorithm = SimpleNamespace(learning_rate=1.0e-3, optimizer=resumed)
    moments = {key: value.clone() for key, value in resumed.state[resumed_parameter].items()}
    assert resumed.param_groups[0]["lr"] != algorithm.learning_rate
    lineage.restore_resumed_learning_rate(algorithm)
    assert algorithm.learning_rate == resumed.param_groups[0]["lr"] == 1.0e-5
    for key, value in moments.items():
        torch.testing.assert_close(resumed.state[resumed_parameter][key], value)


@pytest.mark.parametrize("rate", [0., -1., float("nan"), float("inf"), True, None])
def test_restore_resumed_learning_rate_rejects_invalid_checkpoint_rate(lineage, rate):
    algorithm = SimpleNamespace(learning_rate=1.0e-3, optimizer=SimpleNamespace(param_groups=[{"lr": rate}]))
    with pytest.raises(ValueError, match="checkpoint optimizer"):
        lineage.restore_resumed_learning_rate(algorithm)
    assert algorithm.learning_rate == 1.0e-3


@pytest.mark.parametrize("rate", [0., -1., float("nan"), float("inf"), True])
def test_restore_resumed_learning_rate_rejects_invalid_override(lineage, rate):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-5)
    algorithm = type("Algorithm", (), {"learning_rate": 1.0e-3, "optimizer": optimizer})()
    with pytest.raises(ValueError, match="覆盖值"):
        lineage.restore_resumed_learning_rate(algorithm, rate)


@pytest.mark.parametrize("scale", [0., -1., 1.01, float("nan"), float("inf")])
def test_scale_policy_exploration_noise_rejects_invalid_scale(lineage, scale):
    policy = torch.nn.Module()
    policy.state_dependent_std = False
    policy.noise_std_type = "scalar"
    policy.std = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam(policy.parameters())
    with pytest.raises(ValueError, match="比例"):
        lineage.scale_policy_exploration_noise(policy, optimizer, scale)


@pytest.mark.parametrize(
    ("payload", "load_mode", "task", "seed", "message"),
    [
        ({"model_state_dict": {}}, "resume", "task", 1, "iter"),
        ({"iter": True}, "resume", "task", 1, "iter"),
        ({"iter": 0}, "unknown", "task", 1, "加载模式"),
        ({"iter": 0}, "resume", "", 1, "task"),
        ({"iter": 0}, "resume", "task", True, "seed"),
    ],
)
def test_checkpoint_source_metadata_rejects_ambiguous_lineage(
    lineage, tmp_path, payload, load_mode, task, seed, message,
):
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        lineage.checkpoint_source_metadata(checkpoint, load_mode=load_mode, task=task, seed=seed)
