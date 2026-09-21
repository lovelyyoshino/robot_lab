# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.4.0
# @last_modified 2026-09-17
# @changelog
#   - v0.4.0 (2026-09-17): 加载真实 Grounded 依赖并验证局部初始化清理进展历史。
#   - v0.3.0 (2026-09-17): 验证显式准备期阻尼制动、零目标速度和局部接管恢复。
#   - v0.2.0 (2026-09-17): 准备失败记录独立门限计数与真实终态诊断，局部复位清理。
#   - v0.1.0 (2026-09-16): 覆盖真实接触准入、控制时钟、局部复位、速度注入与准备奖励/历史门。

"""只隔离 Isaac 类型；真实 recovery 状态机、观测、奖励和复位函数在 CPU 上执行。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def recovery(monkeypatch):
    names = (
        "isaaclab", "isaaclab.assets", "isaaclab.managers", "isaaclab.sensors",
        "robot_lab", "robot_lab.assets", "robot_lab.assets.parallelogram_rover",
        "recovery_initialization_test_package", "recovery_initialization_test_package.method1_deng",
    )
    modules = {name: ModuleType(name) for name in names}

    class SceneEntityCfg:
        def __init__(self, name, body_ids=None):
            self.name, self.body_ids = name, body_ids

    modules["isaaclab.assets"].Articulation = object
    modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
    modules["isaaclab.sensors"].ContactSensor = object
    for suffix in ("MIMIC_FOLLOWER_JOINT_NAMES", "POLICY_ACTION_JOINT_NAMES", "STEERING_JOINT_NAMES",
                   "SUSPENSION_JOINT_NAMES", "WHEEL_JOINT_NAMES", "MIMIC_JOINT_RELATIONS"):
        setattr(modules["robot_lab.assets.parallelogram_rover"], "PARALLELOGRAM_ROVER_" + suffix, ())
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, relative_path):
        path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/recovery/mdp" / relative_path
        spec = importlib.util.spec_from_file_location("recovery_initialization_test_package." + name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    return SimpleNamespace(
        utils=load("_utils", "_utils.py"),
        observations=load("observations", "observations.py"),
        events=load("method1_deng.events", "method1_deng/events.py"),
        rewards=load("method1_deng.rewards", "method1_deng/rewards.py"),
        grounded=load("grounded", "grounded.py"),
        settled=load("settled_ground", "settled_ground.py"),
        entity_cfg=SceneEntityCfg,
    )


@pytest.fixture
def env():
    root = torch.zeros(3, 13)
    root[:, 2] = 0.32
    root[:, 3] = 1.0
    data = SimpleNamespace(
        root_state_w=root,
        root_pos_w=root[:, :3],
        root_lin_vel_w=root[:, 7:10],
        root_ang_vel_w=root[:, 10:13],
        default_root_state=root.clone(),
        projected_gravity_b=torch.tensor([[0.0, 1.0, 0.0]]).repeat(3, 1),
        joint_pos=torch.zeros(3, 20),
        joint_vel=torch.zeros(3, 20),
        default_joint_pos=torch.zeros(3, 20),
        joint_acc=torch.ones(3, 20),
        applied_torque=torch.ones(3, 20),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(3, 20, 1),
    )
    data.root_lin_vel_b = data.root_lin_vel_w
    data.root_ang_vel_b = data.root_ang_vel_w
    asset = SimpleNamespace(data=data, joint_names=[str(i) for i in range(20)], writes=0)
    asset.actuators = {
        "legs": SimpleNamespace(stiffness=torch.full((3, 8), 20.0), damping=torch.ones(3, 8),
                                joint_indices=list(range(8)), is_implicit_model=True),
        "wheels": SimpleNamespace(stiffness=torch.zeros(3, 4), damping=torch.full((3, 4), 0.5),
                                  joint_indices=list(range(8, 12)), is_implicit_model=False),
    }
    asset._recovery_mimic_relations = tuple((12 + i, i // 2, (-1.0, 1.0)[i % 2], 0.0) for i in range(8))
    asset.sim_stiffness = torch.zeros(3, 20)
    asset.sim_damping = torch.zeros(3, 20)
    asset.sim_stiffness[:, :8] = 20.0
    asset.sim_damping[:, :8] = 1.0
    asset.write_joint_stiffness_to_sim = lambda value, joint_ids, env_ids: asset.sim_stiffness.__setitem__(
        (env_ids[:, None], joint_ids), value
    )
    asset.write_joint_damping_to_sim = lambda value, joint_ids, env_ids: asset.sim_damping.__setitem__(
        (env_ids[:, None], joint_ids), value
    )
    asset.set_joint_position_target = lambda value, env_ids: None
    asset.set_joint_effort_target = lambda value, env_ids: None
    asset.velocity_target = torch.full((3, 20), 9.0)
    asset.set_joint_velocity_target = lambda value, joint_ids, env_ids: asset.velocity_target.__setitem__(
        (env_ids[:, None], joint_ids), value
    )

    def write_root(state, env_ids):
        asset.writes += 1
        root[env_ids] = state

    asset.write_root_state_to_sim = write_root
    asset.write_root_velocity_to_sim = lambda value, env_ids: root.__setitem__((env_ids, slice(7, 13)), value)
    asset.write_joint_state_to_sim = lambda position, velocity, env_ids: (
        data.joint_pos.__setitem__(env_ids, position), data.joint_vel.__setitem__(env_ids, velocity)
    )
    contacts = torch.zeros(3, 20, 3)
    contacts[:, 7, 2] = 100.0

    class Scene(dict):
        pass

    scene = Scene(robot=asset)
    scene.env_origins = torch.zeros(3, 3)
    scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(
        net_forces_w=contacts, net_forces_w_history=contacts[:, None]
    ))}
    actions = SimpleNamespace(action=torch.ones(3, 12), prev_action=torch.full((3, 12), 2.0))
    actions.reset = lambda ids: (actions.action.__setitem__(ids, 0.0), actions.prev_action.__setitem__(ids, 0.0))

    def forbidden_step():
        raise AssertionError("局部初始化不允许嵌套仿真 step")

    return SimpleNamespace(
        scene=scene, device="cpu", num_envs=3, step_dt=0.02, max_episode_length=250,
        common_step_counter=0, episode_length_buf=torch.zeros(3, dtype=torch.long),
        action_manager=actions, sim=SimpleNamespace(step=forbidden_step),
        _recovery_joint_groups={"active_leg": torch.arange(8), "wheel": torch.arange(8, 12),
                                "active_all": torch.arange(12), "follower": torch.arange(12, 20)},
    )


def sample(recovery, env, **kwargs):
    env.common_step_counter += 1
    env.episode_length_buf += 1
    return recovery.settled.update_settled_initialization(env, settle_time_s=0.06, **kwargs)


def test_contact_and_continuous_stability_required_not_guessed_height(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    contacts = env.scene.sensors["contact_forces"].data.net_forces_w
    contacts[0] = 0.0
    env.scene["robot"].data.root_lin_vel_w[1, 0] = 0.11
    for _ in range(3):
        sample(recovery, env)
    assert env._recovery_takeover_pending.tolist() == [False, False, True]
    assert not env._recovery_control_active.any()
    assert torch.count_nonzero(env.scene["robot"].sim_stiffness) == 0
    # 同一时刻的重复观测/termination调用不能凑足稳定时长。
    old_steps = env._recovery_settle_steps.clone()
    recovery.settled.update_settled_initialization(env, settle_time_s=0.06)
    torch.testing.assert_close(env._recovery_settle_steps, old_steps)
    contacts[0, 7, 2] = 100.0
    sample(recovery, env)
    contacts[0] = 0.0
    sample(recovery, env)
    assert env._recovery_settle_streak[0] == 0


def test_explicit_preparation_brake_is_dissipative_and_restores_on_takeover(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None, preparation_leg_damping=0.2)
    asset = env.scene["robot"]
    assert torch.count_nonzero(asset.sim_stiffness) == 0
    torch.testing.assert_close(asset.sim_damping[:, :8], torch.full((3, 8), 0.2))
    assert torch.count_nonzero(asset.sim_damping[:, 8:]) == 0
    assert torch.count_nonzero(asset.actuators["wheels"].damping) == 0
    assert torch.count_nonzero(asset.velocity_target[:, :8]) == 0
    velocity = torch.linspace(-2.0, 2.0, 24).reshape(3, 8)
    torque = -asset.sim_damping[:, :8] * velocity
    assert torch.all(torque * velocity <= 0)
    for _ in range(3):
        sample(recovery, env)
    recovery.settled.activate_settled_recovery(env, torch.tensor([0]))
    torch.testing.assert_close(asset.sim_damping[0, :8], torch.ones(8))
    torch.testing.assert_close(asset.sim_damping[1:, :8], torch.full((2, 8), 0.2))
    assert torch.all(asset.sim_stiffness[0, :8] == 20)
    recovery.settled.initialize_settled_recovery(env, torch.tensor([0]), preparation_leg_damping=0.2)
    torch.testing.assert_close(asset.sim_damping[:, :8], torch.full((3, 8), 0.2))


@pytest.mark.parametrize("damping", [-0.1, float("nan"), float("inf")])
def test_invalid_preparation_brake_rejected_before_state_change(recovery, env, damping):
    with pytest.raises(ValueError, match="damping"):
        recovery.settled.initialize_settled_recovery(env, None, preparation_leg_damping=damping)
    assert not hasattr(env, "_recovery_control_active")


def test_preparation_brake_contract_cannot_change_at_partial_reset(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None, preparation_leg_damping=0.2)
    before = env.scene["robot"].sim_damping.clone()
    with pytest.raises(ValueError, match="damping"):
        recovery.settled.initialize_settled_recovery(env, torch.tensor([1]), preparation_leg_damping=0.3)
    torch.testing.assert_close(env.scene["robot"].sim_damping, before)


def test_handoff_has_zero_clock_clears_history_and_next_tick_is_controlled(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    recovery.observations.prev_joint_pos_legs(env)
    for _ in range(3):
        sample(recovery, env)
    env.scene["robot"].data.joint_pos[:] = 0.2
    root_before = env.scene["robot"].data.root_state_w.clone()
    recovery.settled.activate_settled_recovery(env, torch.tensor([0]))
    assert env._recovery_control_active.tolist() == [True, False, False]
    assert env._recovery_control_steps.tolist() == [0, 0, 0]
    assert env.episode_length_buf.tolist() == [0, 3, 3]
    assert env.action_manager.action.sum(dim=1).tolist() == [0, 12, 12]
    assert env.action_manager.prev_action.sum(dim=1).tolist() == [0, 24, 24]
    torch.testing.assert_close(recovery.observations.prev_joint_pos_legs(env)[0], torch.full((8,), 0.2))
    torch.testing.assert_close(env._recovery_takeover_root_state_w[0], root_before[0])
    torch.testing.assert_close(env._recovery_takeover_joint_pos[0], env.scene["robot"].data.joint_pos[0])
    assert torch.all(env.scene["robot"].sim_stiffness[0, :8] == 20)
    assert torch.count_nonzero(env.scene["robot"].sim_stiffness[1:]) == 0
    sample(recovery, env)
    assert env._recovery_control_steps.tolist() == [1, 0, 0]
    recovery.utils._advance_step_counter(env)
    recovery.utils._advance_step_counter(env)
    assert env._recovery_step_count.tolist() == [0, -1, -1]
    torch.testing.assert_close(recovery.utils._get_ed(env), torch.tensor([(1 / 250)**3, 0.0, 0.0]))


def test_partial_reset_only_releases_selected_environment(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    for _ in range(3):
        sample(recovery, env)
    recovery.settled.activate_settled_recovery(env, None)
    sample(recovery, env)
    recovery.settled.initialize_settled_recovery(env, torch.tensor([1]))
    assert env._recovery_control_active.tolist() == [True, False, True]
    assert env._recovery_control_steps.tolist() == [1, 0, 1]
    assert env._recovery_settle_steps.tolist() == [3, 0, 3]
    assert env.scene["robot"].sim_stiffness[:, :8].sum(dim=1).tolist() == [160, 0, 160]
    assert env.scene["robot"].writes == 0


def test_partial_initialize_clears_both_progress_terms_without_erasing_peers(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    env._recovery_control_active.fill_(True)
    funcs = (recovery.grounded.recovery_orientation_progress, recovery.grounded.recovery_height_progress)
    for func in funcs:
        func(env)
    env.common_step_counter += 1
    env.scene["robot"].data.projected_gravity_b[:, 2] = -1.0
    before = [func(env).clone() for func in funcs]
    assert all(torch.all(value > 0) for value in before)
    recovery.settled.initialize_settled_recovery(env, torch.tensor([1]))
    for func, old in zip(funcs, before):
        actual = func(env)
        assert actual[1] == 0
        torch.testing.assert_close(actual[[0, 2]], old[[0, 2]])


def test_preparation_failure_cannot_be_silently_activated(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    env.scene.sensors["contact_forces"].data.net_forces_w[0] = 0
    env.scene["robot"].data.projected_gravity_b[1] = torch.tensor([0.0, 0.0, -1.0])
    env.scene["robot"].data.root_state_w[2, 2] = float("nan")
    for _ in range(3):
        failed = sample(recovery, env, max_settle_time_s=0.06)
    assert failed.tolist() == [True, True, True]
    assert env._recovery_initialization_failure_code.tolist() == [1, 1, 2]
    recovery.settled.activate_settled_recovery(env, None)
    assert not env._recovery_control_active.any()
    assert not recovery.settled.settled_control_time_out(env).any()


def test_timeout_only_counts_real_control_ticks(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    env._recovery_control_steps[:] = 1000
    assert not recovery.settled.settled_control_time_out(env).any()
    env._recovery_control_active[0] = True
    assert recovery.settled.settled_control_time_out(env).tolist() == [True, False, False]


def test_takeover_velocity_injection_preserves_ground_configuration(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    for _ in range(3):
        sample(recovery, env)
    pose_before = env.scene["robot"].data.root_state_w[:, :7].clone()
    joints_before = env.scene["robot"].data.joint_pos.clone()
    recovery.settled.activate_settled_recovery(
        env, torch.tensor([1]), takeover_velocity_range={"x": (0.3, 0.3), "pitch": (0.5, 0.5)}
    )
    torch.testing.assert_close(env.scene["robot"].data.root_state_w[:, :7], pose_before)
    torch.testing.assert_close(env.scene["robot"].data.joint_pos, joints_before)
    torch.testing.assert_close(env._recovery_takeover_root_state_w[1, 7:], torch.tensor([0.3, 0, 0, 0, 0.5, 0]))
    assert torch.count_nonzero(env.scene["robot"].data.root_state_w[[0, 2], 7:]) == 0


@pytest.mark.parametrize("invalid_preparation", [False, True])
def test_all_rewards_are_zero_during_preparation_in_mixed_batch(recovery, env, invalid_preparation):
    recovery.settled.initialize_settled_recovery(env, None)
    env._recovery_control_active[1] = True
    env._recovery_control_steps[1] = 200
    env.scene["robot"].data.joint_vel[:] = 0.2
    env.scene["robot"].data.joint_pos[:] = 0.3
    if invalid_preparation:
        data = env.scene["robot"].data
        for field in ("joint_vel", "joint_pos", "root_state_w", "projected_gravity_b", "joint_acc", "applied_torque"):
            getattr(data, field)[[0, 2]] = float("nan")
    reward_names = (
        "recovery_step_counter", "recovery_base_height", "recovery_base_orientation", "recovery_stand_joint_pos",
        "recovery_support_state", "recovery_arm_wheel_coord", "recovery_action_rate", "recovery_joint_deviation",
        "recovery_joint_velocity", "recovery_joint_acceleration", "recovery_torques", "recovery_wheel_velocity",
        "recovery_success_rate",
    )
    for name in reward_names:
        value = getattr(recovery.rewards, name)(env)
        assert value[[0, 2]].tolist() == [0, 0], name
    value = recovery.rewards.recovery_body_collision(env, recovery.entity_cfg("contact_forces", body_ids=[7]))
    assert value[[0, 2]].tolist() == [0, 0]
    assert value[1] > 0


def test_reset_velocity_uses_world_axes_and_keeps_other_environments(recovery, env):
    before = env.scene["robot"].data.root_state_w.clone()
    recovery.events.reset_with_fall_pose(
        env, torch.tensor([1]), root_velocity_range={"x": (1.0, 1.0), "roll": (-0.2, -0.2)}
    )
    torch.testing.assert_close(env.scene["robot"].data.root_state_w[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(env.scene["robot"].data.root_state_w[1, 7:], torch.tensor([1.0, 0, 0, -0.2, 0, 0]))


@pytest.mark.parametrize("ranges", [{"unknown": (0, 1)}, {"x": (1, -1)}, {"pitch": (0, float("inf"))}])
def test_invalid_velocity_rejected_before_physical_writes(recovery, env, ranges):
    with pytest.raises(ValueError):
        recovery.events.reset_with_fall_pose(env, torch.tensor([0]), root_velocity_range=ranges)
    assert env.scene["robot"].writes == 0


def test_default_velocity_sampling_keeps_zero_velocity_and_rng_state(recovery):
    before = torch.random.get_rng_state().clone()
    assert torch.count_nonzero(recovery.events._sample_root_velocity(3, "cpu", None)) == 0
    torch.testing.assert_close(torch.random.get_rng_state(), before)


def test_preparation_failure_diagnostics_identify_gate_without_relaxing_it(recovery, env):
    recovery.settled.initialize_settled_recovery(env, None)
    env.scene["robot"].data.projected_gravity_b[0, 2] = -1.
    env.scene["robot"].data.root_lin_vel_w[1, 0] = .2
    env.scene["robot"].data.joint_vel[2, 0] = .4
    for _ in range(3):
        sample(recovery, env, max_settle_time_s=.06)
    assert env._recovery_initialization_failed.all()
    diag = env._recovery_preparation_metrics
    assert diag["fallen_ok"].tolist() == [False, True, True]
    assert diag["linear_speed_ok"].tolist() == [True, False, True]
    assert diag["leg_speed_ok"].tolist() == [True, True, False]
    assert env._recovery_preparation_gate_failures[:, [1, 3, 4]].tolist() == [[0, 0, 3], [3, 0, 0], [0, 3, 0]]
    recovery.settled.initialize_settled_recovery(env, torch.tensor([1]))
    assert env._recovery_preparation_gate_failures[1].sum() == 0
    assert env._recovery_preparation_gate_failures[[0, 2]].sum() == 6


def test_near_ground_reset_uses_derived_joint_geometry_and_local_origin(recovery, env):
    asset = env.scene["robot"]
    env.scene.env_origins[1, 2] = 2.0
    before = asset.data.root_state_w.clone()

    def root_height(joint_pos, root_quat, clearance):
        assert joint_pos.shape == (1, 20) and root_quat.shape == (1, 4)
        assert clearance == .002
        for follower, master, multiplier, offset in asset._recovery_mimic_relations:
            torch.testing.assert_close(joint_pos[:, follower], joint_pos[:, master] * multiplier + offset)
        return torch.tensor([.42])

    asset._recovery_collision_geometry = SimpleNamespace(root_height=root_height)
    recovery.events.reset_with_fall_pose(env, torch.tensor([1]), collision_clearance=.002)
    assert asset.data.root_pos_w[1, 2].item() == pytest.approx(2.42)
    torch.testing.assert_close(asset.data.root_state_w[[0, 2]], before[[0, 2]])
