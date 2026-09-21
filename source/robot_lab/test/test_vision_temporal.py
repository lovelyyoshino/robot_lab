# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.1.1
# @last_modified 2026-09-17
# @changelog
#   - v0.1.1 (2026-09-17): 覆盖相机周期与 policy 控制周期的时序合同拒绝路径。
#   - v0.1.0 (2026-09-16): CPU 验证深度采样、扰动、幂等、复位隔离和旧 4141 维合同。

"""只替换 Isaac 类型，不替换待测预处理和缓存实现；本测试不证明视觉策略通行率。"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def vision(monkeypatch):
    class Node(SimpleNamespace):
        pass

    class SceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    # 框架替身只负责类型构造；运行的是实际模块、实际环形缓存和实际配置变体方法。
    modules = {name: ModuleType(name) for name in (
        "isaaclab", "isaaclab.managers", "isaaclab.envs", "isaaclab.envs.mdp",
        "isaaclab.sim", "isaaclab.sensors", "isaaclab.terrains", "isaaclab.utils",
        "isaaclab.utils.noise", "vision_temporal_test_package", "vision_temporal_test_package.mdp",
        "vision_temporal_test_package.rough_env_cfg",
        "robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg",
    )}
    modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
    modules["isaaclab.managers"].ManagerTermBase = ManagerTermBase
    modules["isaaclab.managers"].ObservationTermCfg = Node
    modules["isaaclab.utils"].configclass = lambda cls: cls
    modules["isaaclab.sim"].PinholeCameraCfg = Node
    modules["isaaclab.sensors"].TiledCameraCfg = type("TiledCameraCfg", (Node,), {"OffsetCfg": Node})
    modules["isaaclab.terrains"].MeshGapTerrainCfg = Node
    modules["isaaclab.utils.noise"].AdditiveUniformNoiseCfg = Node
    modules["vision_temporal_test_package"].__path__ = []
    modules["vision_temporal_test_package.rough_env_cfg"].ParallelogramRoverRoughEnvCfg = Node
    base = modules["robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg"]
    base.MySceneCfg = Node
    base.ObservationsCfg = type("ObservationsCfg", (Node,), {"PolicyCfg": Node})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[1] / (
        "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover"
    )
    loaded = {}
    for name in ("vision_temporal", "vision_mdp", "vision_env_cfg"):
        spec = importlib.util.spec_from_file_location(f"vision_temporal_test_package.{name}", path / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


def _buffer(vision, **overrides):
    params = dict(history_length=3, history_stride=1, delay_steps=(0, 0), noise_half_range=0.0,
                  drop_probability=0.0, freeze_probability=0.0)
    params.update(overrides)
    return vision.vision_temporal.DepthTemporalBuffer(vision.vision_temporal.DepthTemporalCfg(**params))


def _frame(value, count=3, pixels=2):
    return torch.full((count, pixels), float(value))


def _env(depth):
    return SimpleNamespace(
        common_step_counter=0,
        scene=SimpleNamespace(sensors={
            "depth_camera": SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": depth})),
        }),
        cfg=SimpleNamespace(observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=False))),
    )


def test_depth_normalization_is_bounded_and_does_not_modify_sensor(vision):
    depth = torch.linspace(0, 3, 2 * 20 * 32).reshape(2, 20, 32, 1)
    depth[0, 2, :3, 0] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    original = depth.clone()
    normalized = vision.vision_mdp.normalized_depth_image(_env(depth))
    assert normalized.shape == (2, 512)
    assert torch.isfinite(normalized).all()
    assert normalized.min() == -1 and normalized.max() == 1
    torch.testing.assert_close(normalized[0, :3], torch.tensor([1.0, 1.0, -1.0]))
    torch.testing.assert_close(depth, original, equal_nan=True)


@pytest.mark.parametrize("params", [{"crop_top": -1}, {"crop_bottom": -1}, {"crop_top": 20},
                                     {"min_depth": 3.0, "max_depth": 2.0}, {"max_depth": float("nan")},
                                     {"min_depth": -float("inf")}, {"crop_top": 1.5}])
def test_invalid_depth_preprocessing_is_rejected(vision, params):
    with pytest.raises(ValueError):
        vision.vision_mdp.normalized_depth_image(_env(torch.zeros(1, 20, 32, 1)), **params)


def test_history_stride_and_initial_fill_are_oldest_to_newest(vision):
    buffer = _buffer(vision, history_stride=2)
    for step in range(7):
        out = buffer.update(_frame(step), step).reshape(3, 3, 2)
        expected = torch.tensor([max(step - 4, 0), max(step - 2, 0), step], dtype=torch.float32)
        torch.testing.assert_close(out[0, :, 0], expected)


def test_raw_capture_delay_precedes_delivered_history(vision):
    buffer = _buffer(vision, delay_steps=(2, 2))
    for step in range(6):
        out = buffer.update(_frame(step), step).reshape(3, 3, 2)
        expected = torch.tensor([max(step - 4, 0), max(step - 3, 0), max(step - 2, 0)], dtype=torch.float32)
        torch.testing.assert_close(out[0, :, 0], expected)


def test_drop_holds_exact_last_frame_and_reset_ignores_old_data(vision):
    buffer = _buffer(vision, drop_probability=1.0)
    initial = buffer.update(_frame(0.1), 0)
    for step in range(1, 5):
        torch.testing.assert_close(buffer.update(_frame(step), step), initial)
    buffer.reset(torch.tensor([1]))
    expected = initial.clone()
    expected[1] = 0.9
    torch.testing.assert_close(buffer.update(_frame(0.9), 4), expected)


def test_freeze_duration_does_not_change_on_repeated_reads(vision, monkeypatch):
    buffer = _buffer(vision, freeze_probability=0.5, freeze_steps=(3, 3))
    initial = buffer.update(_frame(0), 0)
    events = iter([True, False, False, False])

    def sample_event(probability, count, device):
        fired = next(events) if probability == 0.5 else False
        return torch.full((count,), fired, dtype=torch.bool, device=device)

    monkeypatch.setattr(buffer, "_events", sample_event)
    for step in (1, 2, 3):
        torch.testing.assert_close(buffer.update(_frame(step), step), initial)
        torch.testing.assert_close(buffer.update(_frame(99), step), initial)
    out = buffer.update(_frame(4), 4).reshape(3, 3, 2)
    torch.testing.assert_close(out[0, :, 0], torch.tensor([0.0, 0.0, 4.0]))


@pytest.mark.parametrize("read_before_reset", [False, True])
def test_reset_at_same_step_isolated_from_other_environments(vision, read_before_reset):
    buffer = _buffer(vision, delay_steps=(1, 1), history_stride=2)
    buffer.update(_frame(0), 0)
    buffer.update(_frame(1), 1)
    if read_before_reset:
        buffer.update(_frame(2), 2)
    buffer.reset(torch.tensor([1]))
    new = _frame(2)
    new[1] = 20
    out = buffer.update(new, 2).reshape(3, 3, 2)
    torch.testing.assert_close(out[0, :, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(out[1], torch.full((3, 2), 20.0))
    torch.testing.assert_close(out[2], out[0])
    torch.testing.assert_close(buffer.update(_frame(99), 2), out.flatten(1))


def test_resets_accumulate_empty_reset_is_noop_and_full_reset_allows_new_counter(vision):
    buffer = _buffer(vision)
    initial = buffer.update(_frame(0), 10)
    buffer.reset(torch.tensor([], dtype=torch.long))
    torch.testing.assert_close(buffer.update(_frame(9), 10), initial)
    buffer.reset([0])
    buffer.reset([2])
    expected = initial.clone()
    expected[[0, 2]] = 3
    torch.testing.assert_close(buffer.update(_frame(3), 10), expected)
    buffer.reset()
    torch.testing.assert_close(buffer.update(_frame(7), 0), torch.full_like(expected, 7))


def test_noise_is_sampled_once_per_capture_and_callers_cannot_mutate_cache(vision):
    buffer = _buffer(vision, noise_half_range=0.1)
    buffer.generator = torch.Generator().manual_seed(42)
    frame = _frame(0.5)
    first = buffer.update(frame, 0)
    saved = first.clone()
    assert (saved - 0.5).abs().max() <= 0.1
    first.zero_()
    frame.fill_(0.9)
    torch.testing.assert_close(buffer.update(frame, 0), saved)
    next_out = buffer.update(frame, 1).reshape(3, 3, 2)
    torch.testing.assert_close(next_out[:, :2], saved.reshape(3, 3, 2)[:, 1:])


def test_corruption_is_seed_reproducible_and_new_noise_does_not_change_frozen_frame(vision):
    params = dict(history_stride=2, delay_steps=(0, 2), drop_probability=0.4,
                  freeze_probability=0.3, freeze_steps=(2, 4), noise_half_range=0.1)
    left, right = _buffer(vision, **params), _buffer(vision, **params)
    left.generator = torch.Generator().manual_seed(4)
    right.generator = torch.Generator().manual_seed(4)
    for step in range(20):
        data = _frame(step / 30)
        torch.testing.assert_close(left.update(data, step), right.update(data, step))
    held = _buffer(vision, drop_probability=1.0, noise_half_range=0.1)
    first = held.update(_frame(0.5), 0)
    torch.testing.assert_close(held.update(_frame(0.9), 1), first)


def test_disable_corruption_removes_delay_drop_freeze_and_noise_but_keeps_stride(vision):
    buffer = _buffer(vision, history_stride=2, delay_steps=(2, 2), drop_probability=1.0,
                     freeze_probability=1.0, noise_half_range=0.2)
    for step in range(6):
        out = buffer.update(_frame(step), step, enable_corruption=False).reshape(3, 3, 2)
        torch.testing.assert_close(out[0, :, 0], torch.tensor([max(step - 4, 0), max(step - 2, 0), step]).float())


@pytest.mark.parametrize("params", [
    {"history_length": 0}, {"history_stride": 0}, {"history_stride": 1.5},
    {"delay_steps": (-1, 2)}, {"delay_steps": (2, 1)}, {"delay_steps": (0.1, 2)},
    {"freeze_steps": (0, 2)}, {"freeze_steps": (3, 2)}, {"drop_probability": 1.1},
    {"freeze_probability": -0.1}, {"drop_probability": float("nan")}, {"noise_half_range": -1},
])
def test_invalid_temporal_parameters_fail_early(vision, params):
    with pytest.raises(ValueError):
        _buffer(vision, **params)


def test_missing_steps_shape_and_dtype_changes_are_rejected(vision):
    buffer = _buffer(vision)
    buffer.update(_frame(0), 5)
    for step in (4, 7):
        with pytest.raises(ValueError, match="控制步"):
            buffer.update(_frame(0), step)
    for frame in (_frame(0, count=1), _frame(0).double(), torch.zeros(3, 2, dtype=torch.long)):
        with pytest.raises(ValueError):
            buffer.update(frame, 6)


def test_manager_term_maintains_4141_shape_and_resets_after_dimension_probe(vision):
    cfg = SimpleNamespace(history_length=0, noise=None, params={})
    env = _env(torch.full((3, 20, 32, 1), 0.1))
    term = vision.vision_mdp.TemporalDepthObservation(cfg, env)
    assert term(env).shape == (3, 4096)
    term.reset()
    env.scene.sensors["depth_camera"].data.output["distance_to_image_plane"].fill_(2.5)
    obs = term(env)
    torch.testing.assert_close(obs, torch.ones(3, 4096))
    assert torch.cat((torch.zeros(3, 45), obs), dim=-1).shape == (3, 4141)
    env.common_step_counter = 1
    env.scene.sensors["depth_camera"].data.output["distance_to_image_plane"].fill_(0.1)
    out = term(env)
    assert out[:, -512:].eq(-1).all()
    assert out[:, :-512].eq(1).all()


@pytest.mark.parametrize("history,noise", [(8, None), (0, object())])
def test_manager_history_and_noise_cannot_be_stacked_twice(vision, history, noise):
    with pytest.raises(ValueError, match="不能叠加"):
        vision.vision_mdp.TemporalDepthObservation(
            SimpleNamespace(history_length=history, noise=noise, params={}),
            _env(torch.zeros(1, 20, 32, 1)),
        )


def test_manager_refuses_silent_temporal_parameter_change(vision):
    env = _env(torch.zeros(1, 20, 32, 1))
    term = vision.vision_mdp.TemporalDepthObservation(SimpleNamespace(history_length=0, noise=None, params={}), env)
    with pytest.raises(ValueError, match="重建环境"):
        term(env, history_stride=1)


def test_temporal_variant_keeps_legacy_untouched_and_contract_is_live(vision, monkeypatch):
    module = vision.vision_env_cfg
    legacy = module.VisionStudentPolicyCfg.depth_image
    assert legacy.func is vision.vision_mdp.normalized_depth_image
    assert legacy.history_length == 8
    assert legacy.noise.n_min == -0.02 and legacy.noise.n_max == 0.02
    cfg = module.ParallelogramRoverVisionStudentTemporalEnvCfg()
    cfg.observations = SimpleNamespace(policy=SimpleNamespace(
        depth_image=copy.deepcopy(legacy), enable_corruption=True,
    ))
    cfg.scene = SimpleNamespace(depth_camera=module.ParallelogramRoverVisionSceneCfg.depth_camera)
    cfg.sim = SimpleNamespace(dt=0.005)
    cfg.decimation = 4
    monkeypatch.setattr(module.ParallelogramRoverVisionStudentEnvCfg, "__post_init__", lambda self: None)
    cfg.__post_init__()
    contract = module.vision_depth_contract(cfg)
    assert contract["student_obs_dim"] == 4141
    assert contract["teacher_obs_dim"] == 232 and contract["action_dim"] == 12
    assert contract["frame_age_steps_oldest_to_newest"] == [35, 30, 25, 20, 15, 10, 5, 0]
    assert contract["history_span_s"] == pytest.approx(0.7)
    assert contract["contract_id"] == "rover_depth_temporal_v2"
    json.dumps(contract)
    cfg.observations.policy.depth_image.params["history_stride"] = 3
    assert module.vision_depth_contract(cfg)["history_span_s"] == pytest.approx(0.42)
    cfg.observations.policy.depth_image = legacy
    legacy_contract = module.vision_depth_contract(cfg)
    assert legacy_contract["contract_id"] == "rover_depth_contiguous_v1"
    assert legacy_contract["student_obs_dim"] == 4141
    assert legacy_contract["history_span_s"] == pytest.approx(0.14)
    assert legacy.history_length == 8 and legacy.noise is not None


@pytest.mark.parametrize("camera_period", [0.01, 0.04, 0.0, float("nan")])
def test_depth_camera_period_must_match_policy_control_step(vision, camera_period):
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        scene=SimpleNamespace(depth_camera=SimpleNamespace(update_period=camera_period)),
    )
    with pytest.raises(ValueError, match="update_period"):
        vision.vision_env_cfg._validate_depth_camera_timing(cfg)
