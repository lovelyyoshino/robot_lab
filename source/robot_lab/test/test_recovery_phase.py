"""阶段观测与离线转换的一致性，覆盖接管、左右锁定和首层函数保留。

@author pony
@date 2026-09-29
@version v0.1.0
@last_modified 2026-09-29
@changelog
    - v0.1.0 (2026-09-29): 使用真实示范结构验证60/72维转换，不启动仿真。
"""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts/reinforcement_learning/rsl_rl"))
    import recovery_phase_training

    path = root / "source/robot_lab/robot_lab/tasks/manager_based/recovery/mdp/phase.py"
    spec = importlib.util.spec_from_file_location("phase_observation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return recovery_phase_training, module


def test_online_phase_starts_at_takeover_and_initial_side_does_not_flip(modules):
    _, online = modules
    env = SimpleNamespace(_recovery_control_steps=torch.tensor([50, 0, 25, 200]), step_dt=.02,
                          _recovery_control_active=torch.tensor([False, True, True, True]),
                          _recovery_takeover_gravity_b=torch.tensor([[0., 1., 0.], [0., -1., 0.],
                                                                   [0., 1., 0.], [0., -1., 0.]]))
    torch.testing.assert_close(online.recovery_phase(env).flatten(), torch.tensor([0., 0., 1/6, 1.]))
    assert online.recovery_initial_side(env).flatten().tolist() == [0., -1., 1., -1.]
    env._recovery_control_steps[3] = 0
    assert online.recovery_phase(env)[3].item() == 0.


def test_observation_shape_probe_before_first_reset(modules):
    _, online = modules
    env = SimpleNamespace(num_envs=3, device="cpu")
    assert online.recovery_phase(env).shape == (3, 1)
    assert online.recovery_initial_side(env).eq(0).all()


def test_offline_conversion_matches_online_clock_and_preserves_source(modules):
    helper, online = modules
    x = np.zeros((500, 58), dtype=np.float32)
    x[0, 1], x[250, 1] = -1., 1.
    x[1:250, 1] = 1.  # 翻滚后方向改变，仍必须保留初始镜像。
    arrays = {"observations": x, "control_step": np.tile(np.arange(250), 2),
              "episode_id": np.repeat(np.arange(2), 250)}
    metadata = {"source": {"task": helper.SIDE_TASK, "action_program": {"active_steps": 150}},
                "contract": {"actor_observation_dim": 58, "critic_observation_dim": 70, "step_dt": .02,
                             "observation_order": ["old"], "observation_term_dims": [[58]],
                             "observation_group": {}}}
    before = copy.deepcopy(metadata)
    converted, info = helper.phase_demonstrations(arrays, metadata)
    assert metadata == before and arrays["observations"].shape == (500, 58)
    assert converted["observations"].shape == (500, 60)
    assert np.all(converted["observations"][:250, -1] == -1)
    env = SimpleNamespace(_recovery_control_steps=torch.from_numpy(arrays["control_step"]), step_dt=.02,
                          _recovery_control_active=torch.ones(500, dtype=torch.bool))
    np.testing.assert_allclose(converted["observations"][:, -2], online.recovery_phase(env).numpy().flatten())
    assert info["contract"]["critic_observation_dim"] == 72
    metadata["source"]["action_program"]["active_steps"] = 125
    with pytest.raises(ValueError, match="三秒"):
        helper.phase_demonstrations(arrays, metadata)


def test_zero_column_expansion_preserves_linear_functions(modules):
    helper, _ = modules
    state = {"actor.0.weight": torch.randn(4, 58), "critic.0.weight": torch.randn(4, 70),
             "std": torch.ones(12)}
    new = helper.phase_policy_state(state)
    for role, width in (("actor", 58), ("critic", 70)):
        x = torch.randn(7, width)
        expanded = torch.cat((x, torch.randn(7, 2)), dim=1)
        torch.testing.assert_close(x @ state[role + ".0.weight"].T, expanded @ new[role + ".0.weight"].T)
    assert state["actor.0.weight"].shape[1] == 58
