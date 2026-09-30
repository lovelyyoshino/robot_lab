"""真实CPU RSL-PPO上验证预热、锚点、固定std及保存续训，不能代替恢复验收。

@author pony
@date 2026-09-29
@version v0.1.0
@last_modified 2026-09-29
@changelog
    - v0.1.0 (2026-09-29): 覆盖真实更新边界、来源拒绝和可重载保护状态。
"""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    import recovery_protected as module

    data = tmp_path / "demo.npz"
    data.write_bytes(b"unit-test-fixture")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    info = {"initialization_type": "offline_actor_behavior_cloning_not_ppo", "ppo_updates": 0,
            "bc_lineage": {"phase_conditioned": True, "train_data": {"sha256": digest}}}
    source = tmp_path / "source.pt"
    torch.save({"infos": info}, source)
    arrays = {"observations": np.zeros((250, 60), dtype=np.float32),
              "actions": np.full((250, 12), .3, dtype=np.float32), "control_step": np.arange(250)}
    metadata = {"source": {}, "contract": {"action_scales": [.6109] * 4 + [1.57] * 4 + [10.] * 4}}
    monkeypatch.setattr(module, "_load_anchor", lambda *args: (arrays, metadata))
    torch.manual_seed(15)

    def make_runner():
        obs = TensorDict({"policy": torch.randn(4, 60), "critic": torch.randn(4, 72)}, batch_size=[4])
        policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
                             actor_hidden_dims=[16], critic_hidden_dims=[16], activation="relu",
                             actor_obs_normalization=False, critic_obs_normalization=False)
        alg = PPO(policy, num_learning_epochs=1, num_mini_batches=1,
                  learning_rate=3e-5, schedule="fixed", entropy_coef=0, device="cpu")
        alg.init_storage("rl", 4, 2, obs, [12])
        runner = SimpleNamespace(device="cpu", alg=alg, log_dir=None, current_learning_iteration=0)

        def save(path, infos=None):
            torch.save({"infos": infos, "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": alg.optimizer.state_dict()}, path)

        runner.save = save
        return runner, obs

    def attach(runner, path=source, **kwargs):
        return module.attach_recovery_protection(
            runner, task=module.PHASE_TASK, demo_path=data, env={}, agent={}, source_checkpoint=path,
            warmup_iterations=1, anchor_batch_size=32, seed=42, **kwargs,
        )

    return module, make_runner, attach, source, data


def _update(runner, obs):
    with torch.inference_mode():
        for _ in range(2):
            runner.alg.act(obs)
            runner.alg.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        runner.alg.compute_returns(obs)
    return runner.alg.update()


def test_real_ppo_freezes_actor_only_then_applies_anchor(setup):
    _, make_runner, attach, _, _ = setup
    runner, obs = make_runner()
    state = attach(runner)
    before = copy.deepcopy(runner.alg.policy.state_dict())
    assert _update(runner, obs)["recovery_critic_only"] == 1.
    after = runner.alg.policy.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items() if k.startswith("actor.") or k == "std")
    assert any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith("critic."))
    result = _update(runner, obs)
    assert result["recovery_critic_only"] == 0. and np.isfinite(result["recovery_anchor_loss"])
    assert state["update_count"] == 2
    assert torch.equal(before["std"], runner.alg.policy.std)
    torch.testing.assert_close(before["std"] * torch.tensor([.6109] * 4 + [1.57] * 4 + [10.] * 4),
                               torch.tensor([.02] * 4 + [.05] * 4 + [.5] * 4))


def test_save_resume_restores_sampler_optimizer_and_next_iteration(setup, tmp_path):
    _, make_runner, attach, _, _ = setup
    runner, obs = make_runner()
    attach(runner)
    _update(runner, obs)
    _update(runner, obs)
    path = tmp_path / "guarded.pt"
    runner.save(path)
    saved = torch.load(path, weights_only=True)
    other, _ = make_runner()
    other.alg.policy.load_state_dict(saved["model_state_dict"])
    other.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
    state = attach(other, path=path)
    assert other.current_learning_iteration == state["update_count"] == 2
    again = tmp_path / "again.pt"
    other.save(again)
    restored = torch.load(again, weights_only=True)["infos"]["recovery_protection"]
    original = saved["infos"]["recovery_protection"]
    assert torch.equal(restored["sampler_rng_state"], original["sampler_rng_state"])
    assert restored["anchor_optimizer_state"]["state"]
    with pytest.raises(ValueError, match="anchor_coef"):
        attach(other, path=path, anchor_coef=.04)


def test_validation_dataset_and_unprotected_ppo_are_rejected(setup):
    module, make_runner, attach, source, data = setup
    runner, _ = make_runner()
    data.write_bytes(b"different-data")
    with pytest.raises(ValueError, match="训练集"):
        attach(runner)
    torch.save({"infos": {}}, source)
    with pytest.raises(ValueError, match="完整保护状态"):
        module._state(source)
