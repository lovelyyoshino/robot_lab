# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.1.1
# @last_modified 2026-09-16
# @changelog
#   - v0.1.1 (2026-09-16): 覆盖自由落体模型/PhysX 增益同步、逐环境边界和不支持执行器拒绝。
#   - v0.1.0 (2026-09-16): 验证正立奖励方向、近正立尺度、环境高度平移与局部跌落复位。

"""在 CPU 上执行真实 recovery 奖励和复位函数，隔离 Isaac Sim 运行时。"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def recovery(monkeypatch):
    names = (
        "isaaclab",
        "isaaclab.assets",
        "isaaclab.managers",
        "isaaclab.sensors",
        "robot_lab",
        "robot_lab.assets",
        "robot_lab.assets.parallelogram_rover",
        "recovery_rewards_test_package",
        "recovery_rewards_test_package.method1_deng",
    )
    modules = {name: ModuleType(name) for name in names}

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    modules["isaaclab.assets"].Articulation = object
    modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
    modules["isaaclab.sensors"].ContactSensor = object
    for suffix in (
        "MIMIC_FOLLOWER_JOINT_NAMES",
        "POLICY_ACTION_JOINT_NAMES",
        "STEERING_JOINT_NAMES",
        "SUSPENSION_JOINT_NAMES",
        "WHEEL_JOINT_NAMES",
        "MIMIC_JOINT_RELATIONS",
    ):
        setattr(modules["robot_lab.assets.parallelogram_rover"], "PARALLELOGRAM_ROVER_" + suffix, ())
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, relative_path):
        path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/recovery/mdp" / relative_path
        spec = importlib.util.spec_from_file_location("recovery_rewards_test_package." + name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    # 时间整形和函数主体都加载真实源码；只预置关节解析结果，避免加载 USD 资产。
    return SimpleNamespace(
        utils=load("_utils", "_utils.py"),
        rewards=load("method1_deng.rewards", "method1_deng/rewards.py"),
        events=load("method1_deng.events", "method1_deng/events.py"),
    )


def make_env(heights, origins=None):
    n = len(heights)
    if origins is None:
        origins = torch.zeros(n, 3)
    root_pos = origins.clone()
    root_pos[:, 2] += torch.tensor(heights)
    default_root = torch.zeros(n, 13)
    default_root[:, :3] = torch.tensor([0.1, -0.2, 0.37])
    default_root[:, 3] = 1
    data = SimpleNamespace(
        root_pos_w=root_pos,
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
        default_root_state=default_root,
        joint_pos=torch.zeros(n, 20),
        default_joint_pos=torch.zeros(n, 20),
        joint_vel=torch.zeros(n, 20),
        soft_joint_pos_limits=torch.tensor([-math.pi, math.pi]).repeat(n, 20, 1),
    )
    asset = SimpleNamespace(data=data, joint_names=[str(i) for i in range(20)])

    def write_root_state_to_sim(state, env_ids):
        asset.written_root = (state.clone(), env_ids.clone())
        data.root_pos_w[env_ids] = state[:, :3]

    def write_joint_state_to_sim(position, velocity, env_ids):
        data.joint_pos[env_ids] = position
        data.joint_vel[env_ids] = velocity

    asset.write_root_state_to_sim = write_root_state_to_sim
    asset.write_joint_state_to_sim = write_joint_state_to_sim
    asset._recovery_mimic_relations = tuple(
        (12 + 2 * i + j, i, (-1.0, 1.0)[j], 0.0) for i in range(4) for j in range(2)
    )

    class Scene(dict):
        pass

    scene = Scene(robot=asset)
    scene.env_origins = origins
    return SimpleNamespace(
        scene=scene,
        device="cpu",
        num_envs=n,
        step_dt=0.02,
        max_episode_length=250,
        common_step_counter=249,
        _recovery_step_count=torch.full((n,), 249, dtype=torch.long),
        _recovery_joint_groups={
            "active_leg": torch.arange(8),
            "wheel": torch.arange(8, 12),
            "active_all": torch.arange(12),
            "follower": torch.arange(12, 20),
        },
        action_manager=SimpleNamespace(prev_action=torch.ones(n, 12)),
    )


def test_orientation_unique_optimum_is_upright_and_roll_pitch_symmetric(recovery):
    angles = torch.linspace(0.0, math.pi, 13)
    env = make_env([0.332] * (2 * len(angles)))
    gravity = env.scene["robot"].data.projected_gravity_b
    gravity[:13, 0] = torch.sin(angles)
    gravity[:13, 2] = -torch.cos(angles)
    gravity[13:, 1] = torch.sin(angles)
    gravity[13:, 2] = -torch.cos(angles)
    reward = recovery.rewards.recovery_base_orientation(env)
    torch.testing.assert_close(reward[:13], reward[13:])
    assert torch.all(reward[1:13] < reward[:12])
    torch.testing.assert_close(reward[0], recovery.utils._get_ed(env)[0])
    assert reward[12] < 1e-15


def test_small_tilt_preserves_existing_sigma_scale(recovery):
    angles = torch.tensor([0.0, 0.01, 0.05, 0.1], dtype=torch.float64)
    env = make_env([0.332] * len(angles))
    env.scene["robot"].data.projected_gravity_b = torch.stack(
        [torch.sin(angles), torch.zeros_like(angles), -torch.cos(angles)], dim=-1
    )
    reward = recovery.rewards.recovery_base_orientation(env) / recovery.utils._get_ed(env)
    old_near_upright = torch.exp(-torch.sin(angles).square() / (0.3**2))
    torch.testing.assert_close(reward, old_near_upright, rtol=3e-4, atol=0)


def test_orientation_reward_keeps_episode_dynamic_scaling(recovery):
    env = make_env([0.332] * 3)
    env._recovery_step_count[:] = torch.tensor([0, 125, 249])
    expected_ed = torch.tensor([0.0, 0.5**3, (249 / 250)**3])
    torch.testing.assert_close(recovery.rewards.recovery_base_orientation(env), expected_ed)


def test_height_rewards_and_success_are_invariant_to_origin_translation(recovery):
    heights = [0.332, 0.20, 0.40, 0.332]
    baseline = make_env(heights)
    shifted = make_env(
        heights,
        torch.tensor([[4.0, 2.0, 8.0], [-1.0, 3.0, 5.0], [0.0, 0.0, -3.0], [1.0, 2.0, 2.0]]),
    )
    for env in (baseline, shifted):
        env.scene["robot"].data.projected_gravity_b[3] = torch.tensor([0.0, 0.0, 1.0])
    for function in (
        recovery.rewards.recovery_base_height,
        recovery.rewards.recovery_support_state,
        recovery.rewards.check_recovery_success,
    ):
        torch.testing.assert_close(function(baseline), function(shifted))
    assert recovery.rewards.check_recovery_success(shifted).tolist() == [True, False, True, False]
    assert recovery.rewards.recovery_support_state(shifted).tolist() == [1, 0, 1, 0]
    height_reward = recovery.rewards.recovery_base_height(shifted)
    assert height_reward[0] > height_reward[1]
    assert height_reward[0] > height_reward[2]


def test_support_remains_disabled_during_freefall(recovery):
    env = make_env([0.332, 0.332])
    env._recovery_step_count[:] = torch.tensor([99, 100])
    assert recovery.rewards.recovery_support_state(env).tolist() == [0, 1]


def test_partial_fall_reset_uses_relative_height_and_preserves_other_envs(recovery):
    origins = torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 0.0], [5.0, 6.0, -2.0]])
    env = make_env([0.332] * 3, origins)
    asset = env.scene["robot"]
    untouched_position = asset.data.root_pos_w[1].clone()
    default_root_before = asset.data.default_root_state.clone()
    reset_ids = torch.tensor([2, 0])
    recovery.events.reset_with_fall_pose(
        env, reset_ids, drop_height=1.1, leg_joint_noise=0.0, pose_weights={"prone": 1.0}
    )
    written, ids = asset.written_root
    torch.testing.assert_close(ids, reset_ids)
    torch.testing.assert_close(written[:, 2] - origins[reset_ids, 2], torch.full((2,), 1.1))
    torch.testing.assert_close(written[:, :2], origins[reset_ids, :2] + default_root_before[reset_ids, :2])
    torch.testing.assert_close(asset.data.root_pos_w[1], untouched_position)
    torch.testing.assert_close(asset.data.default_root_state, default_root_before)
    torch.testing.assert_close(torch.norm(written[:, 3:7], dim=1), torch.ones(2))
    assert torch.count_nonzero(written[:, 7:]) == 0
    assert env._recovery_step_count.tolist() == [-1, 249, -1]
    assert env.action_manager.prev_action.sum(dim=1).tolist() == [0, 12, 0]


@pytest.fixture
def driven_env():
    env = make_env([0.332] * 3)
    asset = env.scene["robot"]
    asset.actuators = {
        "legs": SimpleNamespace(
            stiffness=torch.tensor([[20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]),
            damping=torch.tensor([[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]]),
            joint_indices=[1, 4],
            is_implicit_model=True,
        ),
        "wheels": SimpleNamespace(
            stiffness=torch.zeros(3, 2),
            damping=torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
            joint_indices=[2, 5],
            is_implicit_model=False,
        ),
    }
    # 模型张量和物理增益刻意分开，只有 write_* API 可以改变模拟的物理状态。
    asset.sim_stiffness = torch.zeros(3, 20)
    asset.sim_damping = torch.zeros(3, 20)
    asset.sim_stiffness[:, [1, 4]] = asset.actuators["legs"].stiffness
    asset.sim_damping[:, [1, 4]] = asset.actuators["legs"].damping
    asset.gain_writes = []
    asset.effort_target = torch.ones(3, 20)
    asset.position_target = torch.full((3, 20), 5.0)

    def write_gain(kind, value, joint_ids, env_ids):
        getattr(asset, "sim_" + kind)[env_ids[:, None], joint_ids] = value
        asset.gain_writes.append((kind, tuple(joint_ids), env_ids.clone()))

    asset.write_joint_stiffness_to_sim = lambda value, joint_ids, env_ids: write_gain(
        "stiffness", value, joint_ids, env_ids
    )
    asset.write_joint_damping_to_sim = lambda value, joint_ids, env_ids: write_gain(
        "damping", value, joint_ids, env_ids
    )
    asset.set_joint_position_target = lambda value, env_ids: asset.position_target.__setitem__(env_ids, value)
    asset.set_joint_effort_target = lambda value, env_ids: asset.effort_target.__setitem__(env_ids, value)
    return env


def test_freefall_synchronizes_implicit_physics_without_adding_explicit_pd(recovery, driven_env):
    env = driven_env
    asset = env.scene["robot"]
    env._recovery_step_count[:] = torch.tensor([0, 99, 100])
    original_stiffness = asset.sim_stiffness.clone()
    original_damping = asset.sim_damping.clone()
    recovery.events.zero_action_freefall(env, None)

    for act in asset.actuators.values():
        assert torch.count_nonzero(act.stiffness[:2]) == 0
        assert torch.count_nonzero(act.damping[:2]) == 0
    assert torch.count_nonzero(asset.sim_stiffness[:2]) == 0
    assert torch.count_nonzero(asset.sim_damping[:2]) == 0
    torch.testing.assert_close(asset.sim_stiffness[2], original_stiffness[2])
    torch.testing.assert_close(asset.sim_damping[2], original_damping[2])
    assert asset.effort_target.sum(dim=1).tolist() == [0, 0, 20]
    assert all(write[1] == (1, 4) for write in asset.gain_writes)
    assert torch.count_nonzero(asset.sim_damping[:, [2, 5]]) == 0


def test_freefall_restores_each_environment_gain_at_boundary_only(recovery, driven_env):
    env = driven_env
    asset = env.scene["robot"]
    originals = {name: (act.stiffness.clone(), act.damping.clone()) for name, act in asset.actuators.items()}
    env._recovery_step_count[:] = 0
    recovery.events.zero_action_freefall(env, None)
    first_writes = len(asset.gain_writes)
    env._recovery_step_count[:] = 99
    recovery.events.zero_action_freefall(env, None)
    assert len(asset.gain_writes) == first_writes

    env._recovery_step_count[:] = 100
    recovery.events.zero_action_freefall(env, torch.tensor([1]))
    for name, act in asset.actuators.items():
        torch.testing.assert_close(act.stiffness[1], originals[name][0][1])
        torch.testing.assert_close(act.damping[1], originals[name][1][1])
        assert torch.count_nonzero(act.damping[[0, 2]]) == 0
    torch.testing.assert_close(asset.sim_damping[1, [1, 4]], originals["legs"][1][1])
    assert torch.count_nonzero(asset.sim_damping[[0, 2]]) == 0
    assert env._recovery_freefall_active.tolist() == [True, False, True]
    recovery.events.zero_action_freefall(env, None)
    for name, act in asset.actuators.items():
        torch.testing.assert_close(act.stiffness, originals[name][0])
        torch.testing.assert_close(act.damping, originals[name][1])
    restored_writes = len(asset.gain_writes)
    recovery.events.zero_action_freefall(env, None)
    assert len(asset.gain_writes) == restored_writes
    assert torch.count_nonzero(asset.sim_damping[:, [2, 5]]) == 0


def test_freefall_partial_reset_preserves_running_environments_and_joint_state(recovery, driven_env):
    env = driven_env
    asset = env.scene["robot"]
    sim_before = asset.sim_stiffness.clone()
    state_before = asset.data.joint_pos.clone()
    env._recovery_step_count[:] = torch.tensor([-1, 249, -1])
    recovery.events.zero_action_freefall(env, slice(0, 1))
    assert torch.count_nonzero(asset.sim_stiffness[0]) == 0
    torch.testing.assert_close(asset.sim_stiffness[1:], sim_before[1:])
    torch.testing.assert_close(asset.data.joint_pos, state_before)
    assert asset.effort_target.sum(dim=1).tolist() == [0, 20, 20]
    recovery.events.zero_action_freefall(env, torch.tensor([], dtype=torch.long))
    torch.testing.assert_close(asset.sim_stiffness[1:], sim_before[1:])


@pytest.mark.parametrize("failure", ["gain_shape", "missing_physics_api"])
def test_freefall_rejects_unsupported_actuator_instead_of_teleport(recovery, driven_env, failure):
    env = driven_env
    asset = env.scene["robot"]
    if failure == "gain_shape":
        asset.actuators["wheels"].damping = torch.ones(2)
    else:
        asset.write_joint_damping_to_sim = None
    env._recovery_step_count[:] = 0
    physical_before = asset.sim_stiffness.clone()
    with pytest.raises(RuntimeError, match="freefall"):
        recovery.events.zero_action_freefall(env, None)
    torch.testing.assert_close(asset.sim_stiffness, physical_before)
    assert asset.gain_writes == []
    assert not hasattr(env, "_recovery_gain_cache")
