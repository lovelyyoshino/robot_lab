"""用实际 URDF 验证初始碰撞间隙，防止旧 0.35m 直立复位将后轮埋地。

@author pony
@date 2026-09-16
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 用真实轮半径/轴验证前后直行及航向旋转；拒绝伪无侧滑初态。
    - v0.1.0 (2026-09-16): 覆盖四轮、直立、任意关节/姿态及旋转不变性。
"""

import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("rover_geometry", ROOT / "robot_lab/tasks/manager_based/upright/mdp/geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
URDF = ROOT / "data/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"
JOINTS = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]


def test_zero_joint_quad_and_upright_heights_use_wheels_not_base():
    model = geometry.RoverCollisionGeometry(str(URDF), JOINTS)
    q = torch.zeros(2, len(JOINTS), dtype=torch.float64)
    quat = torch.tensor([[1., 0, 0, 0], [math.sqrt(.5), 0, -math.sqrt(.5), 0]], dtype=q.dtype)
    lows = model.lowest_points(q, quat)
    height = model.root_height(q, quat)
    # 旧直立高度使后轮穿透约10cm，不能以首步弹出后的高度反推初始化正确。
    assert lows["rl_wheel"][1] + .35 < -.09
    assert height[1] > .45
    assert height[0] > .30
    for low in lows.values():
        assert torch.all(low + height >= .002 - 1e-10)
    assert torch.allclose(height + torch.stack(list(lows.values())).amin(0), torch.full((2,), .002, dtype=q.dtype))


def test_random_pose_joint_batch_has_no_collision_penetration():
    torch.manual_seed(17)
    model = geometry.RoverCollisionGeometry(str(URDF), JOINTS)
    q = torch.rand(16, len(JOINTS)) - .5
    quat = torch.randn(16, 4)
    height = model.root_height(q, quat)
    for low in model.lowest_points(q, quat).values():
        assert torch.all(low + height >= .002 - 1e-6)


def test_quaternion_sign_does_not_change_collision_height():
    model = geometry.RoverCollisionGeometry(str(URDF), JOINTS)
    q = torch.zeros(3, len(JOINTS))
    quat = torch.randn(3, 4)
    torch.testing.assert_close(model.root_height(q, quat), model.root_height(q, -quat))


@pytest.mark.parametrize("clearance", [-1., float("nan"), float("inf")])
def test_invalid_clearance_rejected(clearance):
    model = geometry.RoverCollisionGeometry(str(URDF), JOINTS)
    with pytest.raises(ValueError):
        model.root_height(torch.zeros(1, len(JOINTS)), torch.tensor([[1., 0, 0, 0]]), clearance)


@pytest.mark.parametrize("yaw", [0., math.pi / 2, -math.pi / 2, math.pi])
@pytest.mark.parametrize("speed", [-.3, 0., .25])
def test_rolling_wheel_velocities_use_actual_urdf_and_heading(yaw, speed):
    model = geometry.RoverCollisionGeometry(str(URDF), JOINTS)
    q = torch.zeros(1, len(JOINTS), dtype=torch.float64)
    quat = q.new_tensor([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]])
    velocity = q.new_tensor([[speed * math.cos(yaw), speed * math.sin(yaw), 0.]])
    wheels = [f"{c}_wheel_joint" for c in ("fl", "fr", "rl", "rr")]
    result = model.straight_wheel_velocities(q, quat, velocity, wheels)
    torch.testing.assert_close(result, q.new_full((1, 4), speed / .1085))


@pytest.mark.parametrize("fault", ["steering", "lateral", "vertical", "nan", "tilted_axis"])
def test_straight_rolling_rejects_side_slip_or_invalid_states(fault):
    model = geometry.RoverCollisionGeometry(str(URDF), JOINTS)
    q = torch.zeros(1, len(JOINTS))
    quat = torch.tensor([[1., 0., 0., 0.]])
    velocity = torch.tensor([[.2, 0., 0.]])
    if fault == "steering":
        q[:, JOINTS.index("fl_qudong_joint")] = .2
    elif fault == "lateral":
        velocity[:, 1] = .1
    elif fault == "vertical":
        velocity[:, 2] = .1
    elif fault == "nan":
        velocity[:, 0] = float("nan")
    else:
        quat[:] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
    with pytest.raises(ValueError):
        model.straight_wheel_velocities(q, quat, velocity, ["fl_wheel_joint"])
