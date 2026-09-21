"""将底盘支撑漏洞锁定在真实奖励调用处，不以存在接触惩罚代替验证。

@author pony
@date 2026-09-16
@version v0.1.0
@last_modified 2026-09-16
@changelog
    - v0.1.0 (2026-09-16): 覆盖底盘/驱动杆/单轮/四轮/悬空与不同body顺序。
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    for name in ("isaaclab", "isaaclab.assets", "isaaclab.managers", "upright_support_tests"):
        mod = ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["isaaclab.assets"].Articulation = object
    sys.modules["isaaclab.managers"].SceneEntityCfg = lambda name: SimpleNamespace(name=name)
    loaded = {}
    for name in ("support", "rewards"):
        path = Path(__file__).parents[1] / f"robot_lab/tasks/manager_based/upright/mdp/{name}.py"
        spec = importlib.util.spec_from_file_location("upright_support_tests." + name, path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        loaded[name] = mod
    return SimpleNamespace(**loaded)


def environment():
    class Scene(dict):
        pass
    names = ["rr_wheel", "base_link", "fl_wheel", "rl_qudong", "fr_wheel", "rl_wheel"]
    forces = torch.zeros(6, len(names), 3)
    contacts = ((0, 5), (0, 5, 1), (0, 5, 3), (0,), (0, 5, 2, 4), ())
    for env_id, bodies in enumerate(contacts):
        forces[env_id, list(bodies), 2] = 50.
    data = SimpleNamespace(
        projected_gravity_b=torch.tensor([[-1., 0., 0.]]).repeat(6, 1),
        root_ang_vel_b=torch.zeros(6, 3),
        root_pos_w=torch.tensor([[0., 0., .46]]).repeat(6, 1),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces), body_names=names)}
    scene.env_origins = torch.zeros(6, 3)
    return SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: torch.full((6, 1), torch.pi/2)))


def test_balance_and_getup_reward_require_actual_two_rear_wheels(modules):
    env = environment()
    expected = torch.tensor([1., 0., 0., 0., 0., 0.])
    torch.testing.assert_close(modules.rewards.balance_hold(env), expected)
    torch.testing.assert_close(modules.rewards.getup_success(env), expected)


def test_base_supported_pose_cannot_collect_dense_task_rewards(modules):
    env = environment()
    for name in ("pitch_tracking", "upright_height", "rise_progress"):
        reward = getattr(modules.rewards, name)(env)
        assert reward[0] > .9
        assert reward[1] == 0
        assert reward[2] == 0


def test_height_reward_is_invariant_to_environment_origin(modules):
    env = environment()
    before = modules.rewards.upright_height(env)
    env.scene.env_origins[:, 2] += 2.0
    env.scene["robot"].data.root_pos_w[:, 2] += 2.0
    torch.testing.assert_close(before, modules.rewards.upright_height(env))


def test_contact_order_and_nonfinite_cannot_fake_support(modules):
    env = environment()
    sensor = env.scene.sensors["contact_forces"]
    force, names = sensor.data.net_forces_w, sensor.body_names
    expected = modules.support.support_masks(force, names)
    order = [5, 3, 1, 4, 0, 2]
    actual = modules.support.support_masks(force[:, order], [names[i] for i in order])
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b)
    force[0, 0, 0] = float("nan")
    assert not modules.support.support_masks(force, names)[1][0]
