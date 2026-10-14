"""深度角落面板的开关、相机隔离、像素映射和生命周期回归。

@author pony
@date 2026-09-30
@version v0.1.0
@last_modified 2026-09-30
@changelog
    - v0.1.0 (2026-09-30): 覆盖 1–4 台、5 台关闭、无界面禁用、实时只读和释放。
"""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

PREFIX = "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-"


@pytest.fixture
def overlay():
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/vision_depth_overlay.py"
    spec = importlib.util.spec_from_file_location("depth_overlay_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = dict(
        task=PREFIX + "Teacher-Preview-v0",
        num_envs=4,
        headless=False,
        keyboard=False,
        disable_depth_overlay=False,
        livestream=-1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "variant", ["Teacher-v0", "Teacher-Preview-v0", "Student-v0", "Student-Temporal-v0", "Student-Temporal-Preview-v0"]
)
@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 64, 0, -1, None])
def test_auto_gate(overlay, variant, count):
    expected = count is not None and 1 <= count <= 4
    assert overlay.depth_overlay_requested(args(task=PREFIX + variant, num_envs=count), {}) is expected


@pytest.mark.parametrize(
    "settings,environ",
    [
        ({"headless": True}, {}),
        ({}, {"HEADLESS": "1"}),
        ({"disable_depth_overlay": True}, {}),
        ({"livestream": 2}, {}),
        ({}, {"LIVESTREAM": "1"}),
        ({"task": "RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0"}, {}),
        ({"task": None}, {}),
    ],
)
def test_disabled_modes(overlay, settings, environ):
    assert not overlay.depth_overlay_requested(args(**settings), environ)


def test_keyboard_actual_count_and_explicit_livestream_off(overlay):
    assert overlay.depth_overlay_requested(args(keyboard=True, num_envs=20), {})
    assert overlay.depth_overlay_requested(args(livestream=0), {"LIVESTREAM": "2"})
    assert not overlay.depth_overlay_requested(args(keyboard=True, headless=True), {})
    assert overlay.depth_overlay_requested(args(task="robot_lab.tasks:" + PREFIX + "Teacher-v0"), {})


def cfg(camera=None):
    return SimpleNamespace(
        scene=SimpleNamespace(depth_camera=camera),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        num_rerenders_on_reset=0,
        observations={"policy": {"dim": 232}},
    )


def test_teacher_camera_is_play_only_and_does_not_change_observations(overlay, monkeypatch):
    expected = SimpleNamespace(data_types=[overlay.DATA_TYPE], update_period=1.0)
    monkeypatch.setattr(overlay, "_teacher_display_camera", lambda task: deepcopy(expected))
    env_cfg = cfg()
    original_observations = deepcopy(env_cfg.observations)
    camera = overlay.configure_display_camera(env_cfg, PREFIX + "Teacher-Preview-v0")
    assert env_cfg.scene.depth_camera is camera
    assert camera.update_period == pytest.approx(0.02)
    assert env_cfg.num_rerenders_on_reset == 2
    assert env_cfg.observations == original_observations
    assert expected.update_period == 1.0


def test_student_reuses_exact_sensor_and_never_reconfigures_it(overlay, monkeypatch):
    def forbidden(*unused):
        raise AssertionError("student must not create a replacement camera")

    monkeypatch.setattr(overlay, "_teacher_display_camera", forbidden)
    camera = SimpleNamespace(data_types=[overlay.DATA_TYPE], update_period=0.02, offset=(1, 2, 3))
    env_cfg = cfg(camera)
    before = deepcopy(env_cfg)
    assert overlay.configure_display_camera(env_cfg, PREFIX + "Student-Temporal-Preview-v0") is camera
    assert env_cfg == before


def test_missing_student_camera_is_not_silently_repaired(overlay):
    with pytest.raises(ValueError, match="student 缺少"):
        overlay.configure_display_camera(cfg(), PREFIX + "Student-v0")


def test_depth_pixels_have_fixed_units_and_do_not_modify_sensor(overlay):
    depth = np.array([[0.1, 1.3, 2.5, 10.0], [np.nan, np.inf, -np.inf, 0.0]], dtype=np.float32)
    before = depth.copy()
    rgba = overlay.depth_to_rgba(depth, 0.1, 2.5)
    assert rgba.dtype == np.uint8 and rgba.shape == (2, 4, 4)
    assert (rgba[..., 3] == 255).all()
    assert rgba[0, :, 0].tolist() == [255, 128, 0, 0]
    assert (rgba[1, :, :3] == [210, 60, 210]).all()
    np.testing.assert_equal(depth, before)
    # 不随场景最近点/最远点重标定，同一距离在不同帧始终同色。
    assert overlay.depth_to_rgba(np.array([[1.3]]), 0.1, 2.5)[0, 0].tolist() == rgba[0, 1].tolist()


@pytest.mark.parametrize("limits", [(1, 1), (-1, 2), (0, np.inf), (np.nan, 2), (2, 1)])
def test_invalid_depth_scale_fails(overlay, limits):
    with pytest.raises(ValueError):
        overlay.depth_to_rgba(np.ones((2, 3)), *limits)


@pytest.mark.parametrize("shape", [(0, 3), (3,), (2, 3, 1)])
def test_invalid_single_frame_shape_fails(overlay, shape):
    with pytest.raises(ValueError):
        overlay.depth_to_rgba(np.empty(shape), 0.1, 2.5)


@pytest.mark.parametrize("enabled,count,gui", [(False, 4, True), (True, 5, True), (True, 4, False)])
def test_no_ui_access_when_disabled(overlay, enabled, count, gui):
    # 没有 scene 属性，若禁用分支仍读相机/加载 UI，测试会直接失败。
    env = SimpleNamespace(num_envs=count, sim=SimpleNamespace(has_gui=lambda: gui))
    assert overlay.create_depth_overlay(env, PREFIX + "Teacher-v0", enabled=enabled) is None


@pytest.fixture
def fake_ui(monkeypatch):
    widgets = []

    class Widget:
        def __init__(self, *values, **options):
            self.text = values[0] if values else ""
            self.options = options
            self.visible = True
            self.clears = 0
            widgets.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def clear(self):
            self.clears += 1

    class Provider:
        def set_bytes_data(self, data, size):
            self.rgba = np.frombuffer(data, dtype=np.uint8).copy().reshape(size[1], size[0], 4)
            self.size = size

    ui = ModuleType("omni.ui")
    for name in ("VStack", "HStack", "ZStack", "Label", "Rectangle", "Spacer", "ImageWithProvider"):
        setattr(ui, name, Widget)
    ui.Percent = float
    ui.ByteImageProvider = Provider
    ui.IwpFillPolicy = SimpleNamespace(IWP_PRESERVE_ASPECT_FIT=1)
    omni = ModuleType("omni")
    omni.ui = ui
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.ui", ui)
    return Widget(), widgets


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("role", ["teacher", "student"])
def test_panels_update_each_car_and_release_readonly_sensor(overlay, fake_ui, count, role):
    frame, widgets = fake_ui
    viewport = SimpleNamespace(get_frame=lambda name: frame)
    depth = torch.linspace(0.1, 2.5, count).view(count, 1, 1, 1).expand(count, 2, 3, 1).clone()
    camera = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(clipping_range=(0.1, 2.5))),
        data=SimpleNamespace(output={overlay.DATA_TYPE: depth}),
        frame=torch.arange(count),
    )
    panel = overlay.DepthOverlay(viewport, camera, count, role=role)
    assert len(panel._providers) == count
    titles = [w.text for w in widgets if isinstance(w.text, str) and w.text.startswith("Rover")]
    assert titles == [f"Rover {i + 1} | env_{i}" for i in range(count)]
    assert any(w.text == ("Teacher: display only" if role == "teacher" else "Student: live sensor") for w in widgets)
    before = depth.clone()
    panel.update()
    torch.testing.assert_close(depth, before)
    for index, provider in enumerate(panel._providers):
        assert provider.size == [3, 2]
        expected = overlay.depth_to_rgba(depth[index, ..., 0].numpy(), 0.1, 2.5)
        np.testing.assert_equal(provider.rgba, expected)
        assert panel._frame_labels[index].text == f"RAW 3x2 | frame {index}"
    depth.fill_(2.5)
    camera.frame += 1
    panel.update()
    assert not panel._providers[0].rgba[..., :3].any()
    assert panel._frame_labels[0].text.endswith("frame 1")
    panel.close()
    panel.close()
    panel.update()
    assert panel._camera is None and not panel._providers and not frame.visible
    assert frame.clears == 2
