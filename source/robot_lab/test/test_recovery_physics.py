"""恢复物理合同工具的纯 CPU 回归测试。

@author pony
@date 2026-09-28
@version v0.1.0
@last_modified 2026-09-28
@changelog
    - v0.1.0 (2026-09-28): 覆盖标定拒绝、URDF 审计与 DCMotor 参数映射。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "scripts/reinforcement_learning/rsl_rl/recovery_physics.py"


@pytest.fixture
def physics():
    spec = importlib.util.spec_from_file_location("recovery_physics_test_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(physics):
    groups = {
        name: {
            "stiffness": 0.0 if name == "wheels" else 20.0,
            "damping": 0.0 if name == "wheels" else 1.0,
            "effort_limit_nm": 10.0,
            "saturation_effort_nm": 10.0,
            "no_load_speed_rad_s": 40.0,
            "velocity_limit_rad_s": 5.0,
        }
        for name in physics.GROUPS
    }
    return {
        "schema": physics.SCHEMA,
        "status": "measured",
        "parameter_frame": "joint_output",
        "source_evidence": "bench-report-2026-09-28",
        "measurement_date": "2026-09-28",
        "units": physics.UNITS.copy(),
        "groups": groups,
    }


def test_template_is_unverified_and_rejected(physics):
    path = ROOT / "source/robot_lab/data/Robots/custom/parallelogram_rover/calibration/actuator_profile.template.json"
    with pytest.raises(ValueError, match="status=measured"):
        physics.load_motor_profile(path)


def test_load_accepts_zero_pd_gains_and_digest(physics, tmp_path):
    profile = _profile(physics)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = physics.load_motor_profile(path)
    assert loaded["groups"]["wheels"]["stiffness"] == 0.0
    assert len(physics.profile_digest(loaded)) == 64


def test_audit_urdf_reports_missing_collision_and_never_hardware_verified(physics):
    path = ROOT / "source/robot_lab/data/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"
    result = physics.audit_urdf(path)
    assert result["total_mass_kg"] > 0
    assert "fl_arm" in result["missing_collision_visual_links"]
    assert result["unsupported_collision_geometries"] == []
    assert result["mimic_joints"][0]["joint"] != result["mimic_joints"][0]["parent_joint"]
    assert result["hardware_feasibility_verified"] is False
    assert len(result["sha256"]) == 64


def test_apply_maps_no_load_to_model_and_limit_to_physx(monkeypatch, physics):
    class FakeDCMotorCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_actuators = types.ModuleType("isaaclab.actuators")
    fake_actuators.DCMotorCfg = FakeDCMotorCfg
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", fake_actuators)

    class Actuator:
        joint_names_expr = ["joint"]
        friction = 0.25
        armature = 0.03

    robot = types.SimpleNamespace(actuators={name: Actuator() for name in physics.GROUPS})
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(robot=robot))
    old = copy.deepcopy(cfg.scene.robot)
    physics.apply_motor_profile(cfg, _profile(physics))
    for name in physics.GROUPS:
        actuator = cfg.scene.robot.actuators[name]
        assert actuator.velocity_limit == 40.0
        assert actuator.velocity_limit_sim == 5.0
        assert actuator.effort_limit_sim == 10.0
        assert actuator.joint_names_expr == old.actuators[name].joint_names_expr
        assert actuator.armature == old.actuators[name].armature
        assert actuator.friction == old.actuators[name].friction
    assert isinstance(old.actuators["wheels"], Actuator)


def test_apply_dict_does_not_bypass_status(monkeypatch, physics):
    class FakeDCMotorCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_actuators = types.ModuleType("isaaclab.actuators")
    fake_actuators.DCMotorCfg = FakeDCMotorCfg
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", fake_actuators)
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(robot=types.SimpleNamespace(actuators={})))
    invalid = _profile(physics)
    invalid["status"] = "unverified"
    with pytest.raises(ValueError, match="status=measured"):
        physics.apply_motor_profile(cfg, invalid)
