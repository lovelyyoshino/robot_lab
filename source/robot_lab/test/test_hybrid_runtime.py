# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.2.0
# @last_modified 2026-09-17
# @changelog
#   - v0.2.0 (2026-09-17): 覆盖裁剪后轮速历史、reset 首步和恢复历史联动。
#   - v0.1.0 (2026-09-16): 覆盖策略切换时动作单位转换、物理尺度和上限。

"""不用仿真验证动作适配；动力学和策略加载由 hybrid 仿真入口检查。"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    module = ModuleType("isaaclab.envs.mdp.actions")

    class JointVelocityAction:
        def process_actions(self, actions):
            self.processed = (actions * self._scale).clamp(self._clip[..., 0], self._clip[..., 1])

    module.JointVelocityAction = JointVelocityAction
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp.actions", module)
    path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/hybrid/runtime.py"
    spec = importlib.util.spec_from_file_location("hybrid_runtime_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("old,new", [(5.0, 0.8), (0.8, 5.0)])
def test_previous_action_preserves_physical_target_across_switch(runtime, old, new):
    action = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    wheel_target = action[:, 8:] * old
    env = SimpleNamespace(action_manager=SimpleNamespace(action=action), _hybrid_applied_wheel_scale=old,
                          _hybrid_last_wheel_target=wheel_target)
    result = runtime.last_action_for_skill(env, new)
    torch.testing.assert_close(result[:, :8], action[:, :8])
    torch.testing.assert_close(result[:, 8:] * new, wheel_target)
    result.zero_()
    assert action[0, -1] == 11


def test_same_scale_preserves_raw_action_even_if_wheel_target_was_clipped(runtime):
    raw = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=raw), _hybrid_applied_wheel_scale=5.0,
        _hybrid_last_wheel_target=torch.full((1, 4), 31.4159265359),
    )
    observed = runtime.last_action_for_skill(env, 5.0)
    torch.testing.assert_close(observed, raw)


def test_cross_scale_uses_clipped_target_not_saturated_raw_action(runtime):
    raw = torch.zeros(1, 12)
    raw[:, 8:] = 100.0
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=raw), _hybrid_applied_wheel_scale=5.0,
        _hybrid_last_wheel_target=torch.full((1, 4), 31.4159265359),
    )
    observed = runtime.last_action_for_skill(env, .8)
    torch.testing.assert_close(observed[:, :8], raw[:, :8])
    torch.testing.assert_close(observed[:, 8:], torch.full((1, 4), 31.4159265359 / .8))


def test_previous_action_uses_clipped_physical_target_and_reset_is_zero(runtime):
    action = torch.zeros(1, 12)
    action[:, 8:] = 100.0
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=action), _hybrid_applied_wheel_scale=5.0,
        _hybrid_last_wheel_target=torch.full((1, 4), 31.4159265359),
    )
    converted = runtime.last_action_for_skill(env, .8)
    torch.testing.assert_close(converted[:, 8:], torch.full((1, 4), 31.4159265359 / .8))
    del env._hybrid_last_wheel_target
    env._hybrid_applied_wheel_scale = None
    assert runtime.last_action_for_skill(env, 5.0)[:, 8:].count_nonzero() == 0


def test_cross_scale_rejects_nonfinite_or_wrong_shape_history(runtime):
    action = torch.zeros(2, 12)
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=action), _hybrid_applied_wheel_scale=5.0,
        _hybrid_last_wheel_target=torch.full((2, 4), float("nan")),
    )
    with pytest.raises(RuntimeError, match="无效"):
        runtime.last_action_for_skill(env, .8)
    env._hybrid_last_wheel_target = torch.zeros(1, 4)
    with pytest.raises(RuntimeError, match="无效"):
        runtime.last_action_for_skill(env, .8)


def test_actual_wheel_target_changes_with_mode_and_rejects_unknown_scale(runtime):
    term = runtime.HybridWheelVelocityAction()
    term._env = SimpleNamespace(_hybrid_requested_wheel_scale=5.0)
    term._clip = torch.zeros(1, 4, 2)
    term.process_actions(torch.tensor([[2.0, -2.0, 1000.0, -1000.0]]))
    torch.testing.assert_close(term.processed, torch.tensor([[10.0, -10.0, 31.4159265359, -31.4159265359]]))
    term._env._hybrid_requested_wheel_scale = 0.8
    term.process_actions(torch.tensor([[2.0, -2.0, 1000.0, -1000.0]]))
    torch.testing.assert_close(term.processed, torch.tensor([[1.6, -1.6, 100.0, -100.0]]))
    assert term._env._hybrid_applied_wheel_scale == 0.8
    torch.testing.assert_close(term._env._hybrid_last_wheel_target, term.processed)
    term._env._hybrid_requested_wheel_scale = 2.0
    with pytest.raises(ValueError, match="尺度"):
        term.process_actions(torch.zeros(1, 4))


def test_reset_hybrid_history_resets_only_selected_envs(runtime, monkeypatch):
    calls = []
    recovery = ModuleType("robot_lab.tasks.manager_based.recovery.mdp.observations")
    recovery.reset_prev_obs_cache = lambda env, ids, asset_cfg=None: calls.append((env, ids, asset_cfg))
    monkeypatch.setitem(sys.modules, "robot_lab.tasks.manager_based.recovery.mdp.observations", recovery)
    history = torch.ones(3, 4)
    env = SimpleNamespace(_hybrid_last_wheel_target=history, _hybrid_applied_wheel_scale=.8)
    runtime.reset_hybrid_history(env, torch.tensor([1]))
    torch.testing.assert_close(history, torch.tensor([[1.] * 4, [0.] * 4, [1.] * 4]))
    assert env._hybrid_applied_wheel_scale is None
    assert calls[0][1].tolist() == [1]
