# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.1.0
# @last_modified 2026-09-16
# @changelog
#   - v0.1.0 (2026-09-16): 锁定起立过渡奖励的方向、接触、倾斜角速度耦合及非有限边界。

"""用实际奖励函数和刚体旋转有限差分验证过渡信号，不启动仿真或训练。"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def rewards(monkeypatch):
    # 只替换框架类型，角度反解、接触门和奖励计算均执行生产实现。
    for name in ("isaaclab", "isaaclab.assets", "isaaclab.managers", "upright_rewards_test_package"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["isaaclab.assets"].Articulation = object
    sys.modules["isaaclab.managers"].SceneEntityCfg = lambda name: SimpleNamespace(name=name)
    directory = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/upright/mdp"
    for name in ("support", "rewards"):
        spec = importlib.util.spec_from_file_location(f"upright_rewards_test_package.{name}", directory / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


def _gravity(pitch=0.7, gy=0.0):
    radius = math.sqrt(1.0 - gy * gy)
    return torch.tensor([-math.sin(pitch) * radius, gy, -math.cos(pitch) * radius], dtype=torch.float64)


def _environment(gravity, omega, *, contact_names=("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel")):
    class Scene(dict):
        pass

    gravity = torch.as_tensor(gravity, dtype=torch.float64).reshape(-1, 3).clone()
    omega = torch.as_tensor(omega, dtype=gravity.dtype).reshape(-1, 3).clone()
    names = ["rr_wheel", "rl_qudong", "fl_wheel", "base_link", "rl_wheel", "fr_wheel"]
    forces = gravity.new_zeros((len(gravity), len(names), 3))
    for name in contact_names:
        forces[:, names.index(name), 2] = 50.0
    data = SimpleNamespace(projected_gravity_b=gravity, root_ang_vel_b=omega)
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.sensors = {"contact_forces": SimpleNamespace(body_names=names, data=SimpleNamespace(net_forces_w=forces))}
    return SimpleNamespace(scene=scene)


def _finite_difference_pitch_rate(rewards, gravity, omega, h=1e-6):
    """独立参考：先用矩阵指数积分刚体旋转，再反解两个时刻的实际 pitch。"""
    x, y, z = omega
    skew = omega.new_tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    plus = torch.linalg.matrix_exp(-h * skew) @ gravity
    minus = torch.linalg.matrix_exp(h * skew) @ gravity
    plus_asset = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=plus[None]))
    minus_asset = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=minus[None]))
    return (rewards._pitch_from_gravity(plus_asset) - rewards._pitch_from_gravity(minus_asset)) / (2 * h)


def test_forward_rise_gets_reward_but_falling_toward_four_wheels_does_not(rewards):
    gravity = _gravity().repeat(3, 1)
    env = _environment(gravity, [[0.0, -4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(rewards.transit_speed(env), torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("contact_names", [
    ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel", "base_link"),
    ("rl_wheel", "rr_wheel", "rl_qudong"),
    ("base_link",),
])
def test_nonwheel_support_cannot_collect_transit_reward(rewards, contact_names):
    env = _environment(_gravity(), [0.0, -4.0, 0.0], contact_names=contact_names)
    assert rewards.transit_speed(env).item() == 0.0


@pytest.mark.parametrize("contact_names", [
    ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel"),
    ("rl_wheel", "rr_wheel"),
    ("rl_wheel",),
])
def test_transition_does_not_require_two_rear_wheel_support(rewards, contact_names):
    env = _environment(_gravity(), [0.0, -2.0, 0.0], contact_names=contact_names)
    assert rewards.transit_speed(env).item() == pytest.approx(2.0)


@pytest.mark.parametrize("yaw_rate", [-3.0, 3.0])
def test_pure_world_yaw_of_tilted_robot_is_not_pitch_progress(rewards, yaw_rate):
    gravity = _gravity(gy=0.6)
    # 世界竖直轴投影到机体系是 -g；纯世界 yaw 不改变机体系重力，更不改变 pitch。
    omega = -gravity * yaw_rate
    reference = _finite_difference_pitch_rate(rewards, gravity, omega)
    torch.testing.assert_close(reference, torch.zeros_like(reference), rtol=0, atol=1e-9)
    assert abs(omega[1]) > 1.0  # 旧 abs(omega_y) 和简单 -omega_y 都会误判其中一个方向。
    actual = rewards.transit_speed(_environment(gravity, omega))
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=1e-12)


@pytest.mark.parametrize("omega", [
    (-3.0, 0.1, -2.0),  # 实际 pitch 增长，但 -omega_y < 0。
    (3.0, -0.1, 2.0),   # 实际 pitch 减少，但 -omega_y > 0。
    (0.5, -0.7, 0.3),
    (2.0, 1.0, -4.0),
])
def test_tilted_roll_yaw_coupling_matches_actual_pitch_derivative(rewards, omega):
    gravity = _gravity(gy=0.6)
    omega = torch.tensor(omega, dtype=torch.float64)
    expected = _finite_difference_pitch_rate(rewards, gravity, omega).clamp_min(0)
    actual = rewards.transit_speed(_environment(gravity, omega))
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)


def test_random_transit_poses_match_rigid_body_rotation_finite_difference(rewards):
    generator = torch.Generator().manual_seed(782)
    for _ in range(25):
        pitch = 0.45 + torch.rand((), generator=generator).item() * 0.45
        gy = (torch.rand((), generator=generator).item() - 0.5) * 1.2
        gravity = _gravity(pitch, gy)
        omega = torch.randn(3, dtype=torch.float64, generator=generator) * 2.0
        expected = _finite_difference_pitch_rate(rewards, gravity, omega).clamp_min(0)
        actual = rewards.transit_speed(_environment(gravity, omega))
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("pitch", [-0.7, 0.2, 1.2, 1.57, 2.0])
def test_outside_transition_window_does_not_receive_progress_reward(rewards, pitch):
    env = _environment(_gravity(pitch), [0.0, -4.0, 0.0])
    assert rewards.transit_speed(env).item() == 0.0


@pytest.mark.parametrize("component,bad", [
    ("gravity", float("nan")), ("gravity", float("inf")),
    ("omega", float("nan")), ("omega", float("inf")),
    ("contacts", float("nan")), ("contacts", float("inf")),
])
def test_invalid_sensor_rows_return_zero_without_poisoning_other_environments(rewards, component, bad):
    env = _environment(_gravity().repeat(2, 1), [[0.0, -2.0, 0.0]] * 2)
    if component == "gravity":
        env.scene["robot"].data.projected_gravity_b[0, 0] = bad
    elif component == "omega":
        env.scene["robot"].data.root_ang_vel_b[0, 1] = bad
    else:
        env.scene.sensors["contact_forces"].data.net_forces_w[0, 3, 0] = bad
    result = rewards.transit_speed(env)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0], dtype=torch.float64))


def test_degenerate_pitch_and_overflow_do_not_generate_nonfinite_reward(rewards):
    gravity = torch.stack((
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        _gravity() * 1e-6,
        _gravity() * 1e200,
    ))
    env = _environment(gravity, [[float("inf"), -1.0, 0.0], [0.0, -1.0, 0.0],
                                 [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    result = rewards.transit_speed(env)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_reward_does_not_modify_gravity_omega_or_contact_tensors(rewards):
    env = _environment(_gravity(gy=0.3), [0.2, -1.0, 0.4])
    data = env.scene["robot"].data
    tensors = [data.projected_gravity_b, data.root_ang_vel_b,
               env.scene.sensors["contact_forces"].data.net_forces_w]
    before = [value.clone() for value in tensors]
    rewards.transit_speed(env)
    for actual, expected in zip(tensors, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
