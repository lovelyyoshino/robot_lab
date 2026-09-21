"""真实 URDF 的带速起立复位回归；只隔离 Isaac 写入接口，不模拟技能成功。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 模拟真实首reset的零软限速缓存，检查执行器与PhysX已初始化限制。
    - v0.1.0 (2026-09-17): 覆盖正反轮速、局部复位、混合初态、零速兼容及非法参数拒绝。
"""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[1]
URDF = ROOT / "data/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"
JOINTS = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]
CORNERS = ("fl", "fr", "rl", "rr")


@pytest.fixture
def reset_module(monkeypatch):
    names = ("isaaclab", "isaaclab.assets", "isaaclab.managers", "robot_lab", "robot_lab.assets",
             "robot_lab.assets.parallelogram_rover", "upright_reset_test", "upright_reset_test.rewards")
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["isaaclab.assets"].Articulation = object
    sys.modules["isaaclab.managers"].SceneEntityCfg = lambda name: SimpleNamespace(name=name)
    sys.modules["upright_reset_test.rewards"].UPRIGHT_BASE_HEIGHT = .35
    constants = sys.modules["robot_lab.assets.parallelogram_rover"]
    for group, suffix in (("SUSPENSION", "arm"), ("STEERING", "qudong"), ("WHEEL", "wheel")):
        setattr(constants, f"PARALLELOGRAM_ROVER_{group}_JOINT_NAMES", tuple(f"{c}_{suffix}_joint" for c in CORNERS))
    constants.PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS = tuple(
        (f"{c}_{suffix}_joint", f"{c}_arm_joint", sign, 0.)
        for c in CORNERS for suffix, sign in (("zhuanxiang", -1.), ("lower_arm", 1.)))
    for name in ("geometry", "events"):
        spec = importlib.util.spec_from_file_location(
            f"upright_reset_test.{name}", ROOT / f"robot_lab/tasks/manager_based/upright/mdp/{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return sys.modules["upright_reset_test.events"]


@pytest.fixture
def env():
    class Scene(dict):
        pass

    root = torch.zeros(3, 13)
    root[:, 2:4] = torch.tensor([.37, 1.])
    data = SimpleNamespace(default_root_state=root.clone(), root_state_w=root.clone(),
                           default_joint_pos=torch.zeros(3, len(JOINTS)),
                           joint_pos=torch.zeros(3, len(JOINTS)), joint_vel=torch.zeros(3, len(JOINTS)),
                           soft_joint_pos_limits=torch.tensor([-.6, .6]).repeat(3, len(JOINTS), 1),
                           soft_joint_vel_limits=torch.zeros(3, len(JOINTS)),
                           joint_vel_limits=torch.full((3, len(JOINTS)), 31.4))
    wheels = [JOINTS.index(f"{c}_wheel_joint") for c in CORNERS]
    data.soft_joint_pos_limits[:, wheels] = torch.tensor([-float("inf"), float("inf")])
    asset = SimpleNamespace(data=data, joint_names=JOINTS, writes=0,
                            cfg=SimpleNamespace(spawn=SimpleNamespace(asset_path=str(URDF))))
    # Isaac soft_joint_vel_limits到首次actuator.compute才填充；reset时必须仍可校验真实限速。
    asset.actuators = {"wheels": SimpleNamespace(joint_indices=wheels[::-1], velocity_limit=torch.full((3, 4), 31.4))}
    asset.find_joints = lambda names, preserve_order: ([JOINTS.index(name) for name in names], list(names))

    def write_root(state, env_ids):
        asset.writes += 1
        data.root_state_w[env_ids] = state

    asset.write_root_state_to_sim = write_root
    asset.write_joint_state_to_sim = lambda pos, vel, env_ids: (
        data.joint_pos.__setitem__(env_ids, pos), data.joint_vel.__setitem__(env_ids, vel))
    scene = Scene(robot=asset)
    scene.env_origins = torch.tensor([[2., 3., 0.], [10., 10., 1.], [-4., 1., 2.]])
    return SimpleNamespace(scene=scene, device="cpu", num_envs=3,
                           action_manager=SimpleNamespace(prev_action=torch.ones(3, 12)))


@pytest.mark.parametrize("speed", [-.3, .25])
def test_moving_quad_syncs_all_wheel_signs_and_preserves_other_envs(reset_module, env, speed):
    asset = env.scene["robot"]
    before = asset.data.root_state_w.clone()
    reset_module.reset_upright_pose(env, torch.tensor([2, 0]), quad_prob=1., collision_clearance=.002,
                                    quad_forward_velocity_range=(speed, speed))
    for index in (2, 0):
        assert asset.data.root_state_w[index, 7] == pytest.approx(speed)
        for c in CORNERS:
            assert asset.data.joint_vel[index, JOINTS.index(f"{c}_wheel_joint")] == pytest.approx(speed / .1085)
            assert asset.data.joint_pos[index, JOINTS.index(f"{c}_qudong_joint")] == 0
            arm = asset.data.joint_pos[index, JOINTS.index(f"{c}_arm_joint")]
            assert asset.data.joint_pos[index, JOINTS.index(f"{c}_zhuanxiang_joint")] == -arm
            assert asset.data.joint_pos[index, JOINTS.index(f"{c}_lower_arm_joint")] == arm
        lows = asset._upright_collision_geometry.lowest_points(
            asset.data.joint_pos[index:index + 1], asset.data.root_state_w[index:index + 1, 3:7])
        minimum_z = min(float(v[0]) for v in lows.values()) + float(asset.data.root_state_w[index, 2])
        assert minimum_z == pytest.approx(float(env.scene.env_origins[index, 2]) + .002, abs=1e-6)
        assert not env.action_manager.prev_action[index].any()
    torch.testing.assert_close(asset.data.root_state_w[1], before[1])
    assert not asset.data.joint_vel[1].any()
    assert env.action_manager.prev_action[1].eq(1.).all()


def test_mixed_reset_does_not_inject_velocity_into_upright_samples(reset_module, env, monkeypatch):
    monkeypatch.setattr(reset_module.torch, "rand", lambda n, device: torch.tensor([.1, .9, .2]))
    reset_module.reset_upright_pose(env, torch.arange(3), quad_prob=.35, collision_clearance=.002,
                                    quad_forward_velocity_range=(.1, .3))
    data = env.scene["robot"].data
    assert .1 <= data.root_state_w[0, 7] <= .3
    assert .1 <= data.root_state_w[2, 7] <= .3
    assert not data.root_state_w[1, 7:].any()
    assert not data.joint_vel[1].any()
    assert data.root_state_w[1, 5] < -.6


def test_default_and_explicit_zero_preserve_rng_and_existing_reset(reset_module, env):
    torch.manual_seed(145)
    reset_module.reset_upright_pose(env, torch.arange(3), collision_clearance=.002)
    data = env.scene["robot"].data
    original = (data.root_state_w.clone(), data.joint_pos.clone(), torch.random.get_rng_state())
    torch.manual_seed(145)
    reset_module.reset_upright_pose(env, torch.arange(3), collision_clearance=.002,
                                    quad_forward_velocity_range=(0., 0.))
    for actual, expected in zip((data.root_state_w, data.joint_pos, torch.random.get_rng_state()), original):
        torch.testing.assert_close(actual, expected)
    assert not data.joint_vel.any()


@pytest.mark.parametrize("speed_range", [(.2, -.2), (float("nan"), .1), (0., float("inf")), (-.51, .1), (0., .51), (0.,)])
def test_invalid_speed_range_fails_before_any_simulator_write(reset_module, env, speed_range):
    with pytest.raises(ValueError):
        reset_module.reset_upright_pose(env, torch.arange(3), collision_clearance=.002,
                                        quad_forward_velocity_range=speed_range)
    assert env.scene["robot"].writes == 0


def test_moving_requires_ground_geometry_and_valid_wheel_limit(reset_module, env):
    with pytest.raises(ValueError, match="碰撞几何"):
        reset_module.reset_upright_pose(env, torch.arange(3), quad_forward_velocity_range=(.2, .2))
    env.scene["robot"].data.joint_vel_limits[:] = .01
    with pytest.raises(ValueError, match="速度限位"):
        reset_module.reset_upright_pose(env, torch.arange(3), quad_prob=1., collision_clearance=.002,
                                        quad_forward_velocity_range=(.2, .2))
    assert env.scene["robot"].writes == 0


def test_empty_subset_does_not_write(reset_module, env):
    reset_module.reset_upright_pose(env, torch.tensor([], dtype=torch.long), collision_clearance=.002,
                                    quad_forward_velocity_range=(.2, .2))
    assert env.scene["robot"].writes == 0


@pytest.mark.parametrize("bad", [.01, 0., float("nan")])
def test_motor_limit_checked_in_actual_joint_order_before_write(reset_module, env, bad):
    env.scene["robot"].actuators["wheels"].velocity_limit[2, 0] = bad
    with pytest.raises(ValueError, match="速度限位"):
        reset_module.reset_upright_pose(env, torch.tensor([2, 0]), quad_prob=1., collision_clearance=.002,
                                        quad_forward_velocity_range=(.2, .2))
    assert env.scene["robot"].writes == 0
