"""旧 45 维 MuJoCo 播放器的物理子步控制回归；不启动 MuJoCo 或 ONNX。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 锁定腿位置 PD 使用当前子步状态，且动作目标在 decimation 内不漂移。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def player(monkeypatch):
    mujoco = ModuleType("mujoco")
    mujoco.MjModel = object
    mujoco.MjData = object
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "onnxruntime", ModuleType("onnxruntime"))
    path = Path(__file__).parents[3] / "scripts/mujoco/play_parallelogram_rover.py"
    module = ModuleType("player_test")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def test_policy_targets_preserve_old_12d_action_scales_and_joint_limits(player):
    legs, wheels = player._policy_targets(np.full(12, 100., dtype=np.float32))
    np.testing.assert_allclose(legs[:4], .6109)
    np.testing.assert_allclose(legs[4:], 1.57)
    np.testing.assert_allclose(wheels, 500.)


def test_leg_pd_reads_current_substep_state(player, monkeypatch):
    calls = {"position": 0, "velocity": 0}
    def joint_position(*_):
        calls["position"] += 1
        return 0. if calls["position"] <= 8 else .2
    def joint_velocity(*_):
        calls["velocity"] += 1
        return 0. if calls["velocity"] <= 8 else .5
    monkeypatch.setattr(player, "_joint_position", joint_position)
    monkeypatch.setattr(player, "_joint_velocity", joint_velocity)
    data = SimpleNamespace(ctrl=np.zeros(12))
    targets = np.full(8, .25)
    player._set_leg_torques(None, data, targets)
    first = data.ctrl[:8].copy()
    player._set_leg_torques(None, data, targets)
    second = data.ctrl[:8].copy()
    np.testing.assert_allclose(first, 5.)
    np.testing.assert_allclose(second, .5)


def test_control_loop_recomputes_leg_and_wheel_pd_each_physics_substep():
    path = Path(__file__).parents[3] / "scripts/mujoco/play_parallelogram_rover.py"
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loops = [node for node in ast.walk(main) if isinstance(node, ast.For)]
    substep_loop = next(
        node for node in loops
        if isinstance(node.target, ast.Name) and node.target.id == "_"
        and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range" and isinstance(node.iter.args[0], ast.Attribute)
        and isinstance(node.iter.args[0].value, ast.Name) and node.iter.args[0].value.id == "args"
        and node.iter.args[0].attr == "decimation"
    )
    calls = [node.func.id for node in ast.walk(substep_loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "_set_leg_torques" in calls
    assert "_set_wheel_torques" in calls
