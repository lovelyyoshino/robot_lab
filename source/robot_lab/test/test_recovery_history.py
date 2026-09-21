# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.1.0
# @last_modified 2026-09-16
# @changelog
#   - v0.1.0 (2026-09-16): 验证共享历史、重复读取、逐环境复位与 58 维观测边界。

"""直接加载真实观测函数，用 CPU 张量验证 recovery 历史，不启动 Isaac Sim。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def observations(monkeypatch):
    # 仅替换依赖框架的类型和关节索引查询；所有缓存、复位和观测计算均执行真实代码。
    modules = {
        name: ModuleType(name)
        for name in (
            "isaaclab",
            "isaaclab.assets",
            "isaaclab.managers",
            "isaaclab.sensors",
            "recovery_history_test_package",
            "recovery_history_test_package._utils",
        )
    }

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    modules["isaaclab.assets"].Articulation = object
    modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
    modules["isaaclab.sensors"].ContactSensor = object
    modules["recovery_history_test_package"].__path__ = []
    modules["recovery_history_test_package._utils"]._get_joint_groups = (
        lambda env, asset: env.joint_groups
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/recovery/mdp/observations.py"
    spec = importlib.util.spec_from_file_location("recovery_history_test_package.observations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env():
    defaults = torch.arange(20, dtype=torch.float32).repeat(3, 1) / 10
    data = SimpleNamespace(
        default_joint_pos=defaults,
        joint_pos=defaults + 1,
        joint_vel=torch.arange(60, dtype=torch.float32).reshape(3, 20),
    )
    return SimpleNamespace(
        common_step_counter=0,
        num_envs=3,
        device="cpu",
        scene={"robot": SimpleNamespace(data=data)},
        joint_groups={
            "active_leg": torch.tensor([0, 3, 6, 9, 1, 4, 7, 10]),
            "wheel": torch.tensor([2, 5, 8, 11]),
        },
    )


@pytest.mark.parametrize("term", ["joint_pos_legs", "joint_vel_legs", "wheel_vel"])
def test_actor_critic_and_repeated_reads_share_previous_step(observations, env, term):
    current = getattr(observations, term)
    previous = getattr(observations, "prev_" + term)
    initial = current(env).clone()
    torch.testing.assert_close(previous(env), initial)

    for step in (1, 2):
        expected = current(env).clone()
        env.common_step_counter = step
        env.scene["robot"].data.joint_pos += 10
        env.scene["robot"].data.joint_vel += 20
        for _ in range(3):
            torch.testing.assert_close(previous(env), expected)


def test_consumers_cannot_mutate_cached_state(observations, env):
    current = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    initial = current.clone()
    first = observations._cache_prev(env, "sensor", current)
    first.fill_(-100)
    current.add_(10)
    torch.testing.assert_close(observations._cache_prev(env, "sensor", current), initial)

    env.common_step_counter += 1
    torch.testing.assert_close(observations._cache_prev(env, "sensor", current), initial)
    next_expected = current.clone()
    current.add_(10)
    observations._cache_prev(env, "sensor", current)
    env.common_step_counter += 1
    torch.testing.assert_close(observations._cache_prev(env, "sensor", current), next_expected)


@pytest.mark.parametrize("read_before_reset", [False, True])
def test_partial_reset_preserves_other_environments_history(observations, env, read_before_reset):
    previous = observations.prev_joint_vel_legs
    initial = observations.joint_vel_legs(env).clone()
    previous(env)
    env.common_step_counter = 1
    env.scene["robot"].data.joint_vel += 10
    before_reset = observations.joint_vel_legs(env).clone()
    if read_before_reset:
        # 记录器可在自动复位之前读取终态，随后仍要在同一控制步读取新 episode。
        previous(env)

    observations.reset_prev_obs_cache(env, torch.tensor([1]))
    # 故意在缓存复位事件之后改物理状态，验证不依赖 reset 事件排列顺序。
    env.scene["robot"].data.joint_vel[1] = 500
    fresh = observations.joint_vel_legs(env).clone()
    expected = initial.clone()
    expected[1] = fresh[1]
    torch.testing.assert_close(previous(env), expected)
    torch.testing.assert_close(previous(env), expected)

    env.common_step_counter = 2
    env.scene["robot"].data.joint_vel += 20
    next_expected = before_reset.clone()
    next_expected[1] = fresh[1]
    torch.testing.assert_close(previous(env), next_expected)


def test_multiple_resets_before_read_accumulate_environment_ids(observations, env):
    previous = observations.prev_wheel_vel
    initial = previous(env)
    observations.reset_prev_obs_cache(env, torch.tensor([0]))
    env.scene["robot"].data.joint_vel[0] = 100
    observations.reset_prev_obs_cache(env, torch.tensor([2]))
    env.scene["robot"].data.joint_vel[2] = 200
    expected = initial.clone()
    expected[[0, 2]] = observations.wheel_vel(env)[[0, 2]]
    torch.testing.assert_close(previous(env), expected)


def test_empty_reset_is_noop_and_none_resets_all(observations, env):
    previous = observations.prev_joint_pos_legs
    initial = previous(env)
    env.scene["robot"].data.joint_pos += 50
    observations.reset_prev_obs_cache(env, torch.tensor([], dtype=torch.long))
    torch.testing.assert_close(previous(env), initial)
    observations.reset_prev_obs_cache(env, None)
    torch.testing.assert_close(previous(env), observations.joint_pos_legs(env))


def test_reset_before_cache_creation_uses_first_new_episode_state(observations, env):
    observations.reset_prev_obs_cache(env, torch.tensor([0, 1, 2]))
    env.scene["robot"].data.joint_pos += 30
    torch.testing.assert_close(observations.prev_joint_pos_legs(env), observations.joint_pos_legs(env))


def test_public_observations_keep_58d_contract_and_ignore_followers(observations, env):
    data = env.scene["robot"].data
    data.joint_pos[:, 12:] = 1e6
    data.joint_vel[:, 12:] = 1e6
    terms = [
        torch.zeros(3, 3),
        torch.zeros(3, 3),
        observations.joint_pos_legs(env),
        observations.joint_vel_legs(env),
        observations.wheel_vel(env),
        observations.prev_joint_pos_legs(env),
        observations.prev_joint_vel_legs(env),
        observations.prev_wheel_vel(env),
        torch.zeros(3, 12),
    ]
    assert [term.shape[-1] for term in terms] == [3, 3, 8, 8, 4, 8, 8, 4, 12]
    actor = torch.cat(terms, dim=-1)
    assert actor.shape == (3, 58)
    assert actor.abs().max() < 1e6
    torch.testing.assert_close(
        actor[:, 6:14],
        data.joint_pos[:, env.joint_groups["active_leg"]]
        - data.default_joint_pos[:, env.joint_groups["active_leg"]],
    )
