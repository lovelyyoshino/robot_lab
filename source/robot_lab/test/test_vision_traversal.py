"""验证真实评测回调的地形高度与自动reset前采样；不代表策略通行成功。

@author pony
@date 2026-09-17
@version v0.4.0
@last_modified 2026-09-17
@changelog
    - v0.4.0 (2026-09-17): 覆盖 v4 地形过滤归属、真实质量、轮承重、非轮/无支撑失败关闭。
    - v0.3.0 (2026-09-17): 区分窄沟合法全正无穷未命中与损坏扫描，禁止把跨沟直接判失败。
    - v0.2.0 (2026-09-17): 增加单射线失效、派生状态失效、平台偏置与通行边界回归。
    - v0.1.0 (2026-09-17): 覆盖楼梯下降、坡面、掉沟和无效地面测量。
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def evaluator(monkeypatch):
    directory = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("vision_traversal_test", directory / "evaluate_parallelogram_vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(*, ground_z=0.0, origin_z=1.0, root_z=0.33):
    class Scene(dict):
        pass

    state = torch.zeros(1, 13)
    state[:, :3] = torch.tensor([2.7, 0.0, root_z])
    state[:, 3] = 1.0
    data = SimpleNamespace(
        root_state_w=state, root_pos_w=state[:, :3],
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        root_lin_vel_b=torch.tensor([[0.4, 0.0, 0.0]]),
        joint_pos=torch.zeros(1, 20), joint_vel=torch.zeros(1, 20),
    )
    names = ["base_link", "fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel"]
    masses = torch.full((1, len(names)), 2.0)
    scene = Scene(robot=SimpleNamespace(data=data, body_names=names,
                                       root_physx_view=SimpleNamespace(get_masses=lambda: masses)))
    scene.env_origins = torch.tensor([[0.0, 0.0, origin_z]])
    hits = torch.zeros(1, 9, 3)
    hits[:, :, 2] = ground_z
    scene.sensors = {"height_scanner_base": SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))}
    for name in names:
        force = torch.zeros(1, 1, 1, 3)
        force[..., 2] = 0.0 if name == "base_link" else 30.0
        scene.sensors["vision_ground_" + name] = SimpleNamespace(
            body_names=[name], cfg=SimpleNamespace(filter_prim_paths_expr=["/World/ground/terrain/mesh"]),
            contact_physx_view=SimpleNamespace(filter_count=1), data=SimpleNamespace(force_matrix_w=force),
        )
    cfg = SimpleNamespace(sim=SimpleNamespace(gravity=(0.0, 0.0, -9.81)),
                          scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type="generator", prim_path="/World/ground")))
    return SimpleNamespace(scene=scene, cfg=cfg, num_envs=1)


@pytest.mark.parametrize("scenario", ["pyramid_stairs", "pyramid_stairs_inv", "hf_pyramid_slope"])
def test_descending_terrain_below_center_is_not_a_fall(evaluator, scenario):
    env = make_env()
    assert not evaluator.capture_traversal(env, scenario=scenario).item()
    assert env._vision_eval_sample[0] == [True]


def test_gap_floor_cannot_be_used_as_successful_route(evaluator):
    env = make_env(ground_z=-1.0, origin_z=0.0, root_z=-0.67)
    assert evaluator.capture_traversal(env, scenario="gaps").item()
    assert env._vision_eval_sample[0] == [False]


def test_reaching_progress_with_rollover_is_failure(evaluator):
    env = make_env(origin_z=0.0)
    env.scene["robot"].data.projected_gravity_b[:, 2] = 0.0
    assert evaluator.capture_traversal(env, scenario="boxes").item()


def test_nonfinite_terrain_measurement_cannot_pass(evaluator):
    env = make_env(origin_z=0.0)
    env.scene.sensors["height_scanner_base"].data.ray_hits_w.fill_(float("nan"))
    assert evaluator.capture_traversal(env, scenario="boxes").item()
    assert env._vision_eval_sample[1] == [False]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_single_invalid_ray_rejects_the_sample(evaluator, value):
    env = make_env(origin_z=0.0)
    env.scene.sensors["height_scanner_base"].data.ray_hits_w[:, 4, 2] = value
    assert evaluator.capture_traversal(env, scenario="pyramid_stairs").item()
    assert env._vision_eval_sample[:2] == ([False], [False])


@pytest.mark.parametrize("field", ["projected_gravity_b", "root_lin_vel_b", "joint_pos", "joint_vel"])
def test_invalid_robot_state_rejects_the_sample(evaluator, field):
    env = make_env(origin_z=0.0)
    getattr(env.scene["robot"].data, field)[:, 0] = float("nan")
    assert evaluator.capture_traversal(env, scenario="boxes").item()
    assert env._vision_eval_sample[:2] == ([False], [False])


@pytest.mark.parametrize("platform_z", [-3.0, 0.0, 4.0])
def test_gap_gate_uses_platform_height_instead_of_world_zero(evaluator, platform_z):
    env = make_env(ground_z=platform_z - 1.0, origin_z=platform_z, root_z=platform_z + 0.33)
    assert not evaluator.capture_traversal(env, scenario="gaps").item()
    env.scene["robot"].data.root_pos_w[:, 2] = platform_z - 0.67
    assert evaluator.capture_traversal(env, scenario="gaps").item()
    assert env._vision_eval_sample[0] == [False]


@pytest.mark.parametrize("x,y,fallen,stable", [(2.4, 0.0, False, False), (2.7, 0.0, False, True),
                                             (2.7, 1.1, True, False), (3.8, 0.0, True, False)])
def test_progress_and_lane_gates_still_apply_on_descending_terrain(evaluator, x, y, fallen, stable):
    env = make_env()
    env.scene["robot"].data.root_pos_w[:, :2] = torch.tensor([x, y])
    assert evaluator.capture_traversal(env, scenario="hf_pyramid_slope").item() == fallen
    assert env._vision_eval_sample[0] == [stable]


def test_unknown_terrain_cannot_silently_disable_the_height_gate(evaluator):
    with pytest.raises(ValueError, match="未知视觉评测地形"):
        evaluator.capture_traversal(make_env(), scenario="gap")


def test_gap_allows_geometric_no_return_without_allowing_fall_or_corrupt_scan(evaluator):
    env = make_env(origin_z=0.0)
    hits = env.scene.sensors["height_scanner_base"].data.ray_hits_w
    # Isaac raycast_mesh 未命中时 xyz 均为 +inf，跨过无底窄沟属于正常几何结果。
    hits[:, 4, :] = float("inf")
    assert not evaluator.capture_traversal(env, scenario="gaps").item()
    assert env._vision_eval_sample[:2] == ([False], [True])
    hits.fill_(float("inf"))
    assert not evaluator.capture_traversal(env, scenario="gaps").item()
    assert env._vision_eval_sample[:2] == ([False], [True])
    hits.zero_()
    assert not evaluator.capture_traversal(env, scenario="gaps").item()
    assert env._vision_eval_sample[:2] == ([True], [True])
    env.scene["robot"].data.root_pos_w[:, 2] = -.3
    assert evaluator.capture_traversal(env, scenario="gaps").item()


@pytest.mark.parametrize("bad_hit", [[0., 0., float("inf")], [float("nan")] * 3, [-float("inf")] * 3])
def test_gap_rejects_corrupted_hit_triplets(evaluator, bad_hit):
    env = make_env(origin_z=0.0)
    env.scene.sensors["height_scanner_base"].data.ray_hits_w[:, 4] = torch.tensor(bad_hit)
    assert evaluator.capture_traversal(env, scenario="gaps").item()
    assert env._vision_eval_sample[1] == [False]


def set_force(env, name, force):
    env.scene.sensors["vision_ground_" + name].data.force_matrix_w[:] = torch.tensor(force)


def test_airborne_crossing_is_allowed_but_terminal_scan_alone_is_not_support(evaluator):
    env = make_env(origin_z=0.0)
    for name in env.scene["robot"].body_names:
        set_force(env, name, [0., 0., 0.])
    for progress in (1.6, 2.7):
        env.scene["robot"].data.root_pos_w[:, 0] = progress
        assert not evaluator.capture_traversal(env, scenario="gaps").item()
        assert env._vision_eval_sample[:2] == ([False], [True])
        assert env._vision_eval_support[0]["supporting_wheels"] == []


def test_three_wheels_and_half_actual_weight_are_both_required(evaluator):
    env = make_env(origin_z=0.0)
    set_force(env, "rr_wheel", [0., 0., 0.])
    evaluator.capture_traversal(env, scenario="boxes")
    assert env._vision_eval_sample[0] == [True]
    env.scene["robot"].root_physx_view.get_masses().fill_(10.)
    evaluator.capture_traversal(env, scenario="boxes")
    assert env._vision_eval_sample[0] == [False]
    assert env._vision_eval_support[0]["robot_weight_n"] == pytest.approx(490.5)
    env.scene["robot"].root_physx_view.get_masses().fill_(2.)
    set_force(env, "fl_wheel", [0., 0., 0.])
    evaluator.capture_traversal(env, scenario="boxes")
    assert env._vision_eval_sample[0] == [False]


@pytest.mark.parametrize("force", [[80., 0., 0.], [0., 0., -80.], [0., 0., 1.0]])
def test_sidewall_downward_and_exact_threshold_are_not_wheel_support(evaluator, force):
    env = make_env(origin_z=0.0)
    for name in ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel"):
        set_force(env, name, force)
    evaluator.capture_traversal(env, scenario="boxes")
    assert env._vision_eval_sample[:2] == ([False], [True])


def test_nonwheel_terrain_contact_breaks_hold_not_intermediate_episode(evaluator):
    env = make_env(origin_z=0.0)
    set_force(env, "base_link", [0., 1.01, 0.])
    assert not evaluator.capture_traversal(env, scenario="boxes").item()
    assert env._vision_eval_sample[0] == [False]
    set_force(env, "base_link", [0., 1., 0.])
    evaluator.capture_traversal(env, scenario="boxes")
    assert env._vision_eval_sample[0] == [True]


def test_sensor_order_is_independent_and_unfiltered_self_contact_is_ignored(evaluator):
    env = make_env(origin_z=0.0)
    env.scene["robot"].body_names.reverse()
    for key, sensor in env.scene.sensors.items():
        if key.startswith("vision_ground_"):
            sensor.data.net_forces_w = torch.full((1, 1, 3), 1e6)
    evaluator.capture_traversal(env, scenario="boxes")
    assert env._vision_eval_sample[0] == [True]
    assert env._vision_eval_support[0]["nonwheel_max_normal_force_n"] == 0.


@pytest.mark.parametrize("fault", ["missing", "wrong_body", "wrong_filter", "none", "bad_shape", "empty_filter"])
def test_missing_or_misattributed_ground_sensor_fails_closed(evaluator, fault):
    env = make_env(origin_z=0.0)
    sensor = env.scene.sensors["vision_ground_fl_wheel"]
    if fault == "missing":
        del env.scene.sensors["vision_ground_fl_wheel"]
    elif fault == "wrong_body":
        sensor.body_names = ["base_link"]
    elif fault == "wrong_filter":
        sensor.cfg.filter_prim_paths_expr = ["/World/envs/env_.*/Robot/.*"]
    elif fault == "none":
        sensor.data.force_matrix_w = None
    elif fault == "bad_shape":
        sensor.data.force_matrix_w = torch.zeros(1, 2, 1, 3)
    else:
        sensor.contact_physx_view.filter_count = 0
    with pytest.raises(RuntimeError):
        evaluator.capture_traversal(env, scenario="boxes")


@pytest.mark.parametrize("fault", ["nan_force", "inf_force", "nan_mass", "negative_mass"])
def test_nonfinite_or_invalid_contact_mass_terminates_without_json_nan(evaluator, fault):
    import json

    env = make_env(origin_z=0.0)
    if fault.endswith("force"):
        set_force(env, "fl_wheel", [0., 0., float("nan") if fault.startswith("nan") else float("inf")])
    else:
        env.scene["robot"].root_physx_view.get_masses()[0, 0] = float("nan") if fault.startswith("nan") else -1.
    assert evaluator.capture_traversal(env, scenario="boxes").item()
    assert env._vision_eval_sample[:2] == ([False], [False])
    json.dumps(env._vision_eval_support, allow_nan=False)


def test_loss_of_support_at_final_sample_cannot_pass(evaluator):
    env = make_env(origin_z=0.0)
    tracker = evaluator.SkillEvaluation(1, 1, .02, .5, 7.5)
    for tick in range(400):
        if tick == 399:
            for name in env.scene["robot"].body_names:
                set_force(env, name, [0., 0., 0.])
        fallen = evaluator.capture_traversal(env, scenario="boxes").tolist()
        stable, finite, _, _ = env._vision_eval_sample
        tracker.update(stable, fallen, [tick == 399], finite)
    assert tracker.summary(.85)["passed"] is False
    assert tracker.records[0]["terminal_stable_time_s"] == 0.


def test_contact_aggregation_overflow_cannot_pass(evaluator):
    env = make_env(origin_z=0.)
    for name in ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel"):
        set_force(env, name, [0., 0., torch.finfo(torch.float32).max / 2.])
    assert evaluator.capture_traversal(env, scenario="boxes").item()
    assert env._vision_eval_sample[:2] == ([False], [False])


def test_support_evidence_contains_every_actual_body_mass(evaluator):
    env = make_env(origin_z=0.)
    masses = env.scene["robot"].root_physx_view.get_masses()
    masses[0, 0] = 7.5
    evaluator.capture_traversal(env, scenario="boxes")
    evidence = env._vision_eval_support[0]
    assert evidence["body_names"] == env.scene["robot"].body_names
    assert evidence["body_mass_kg"] == masses[0].tolist()
    assert evidence["robot_weight_n"] == pytest.approx(sum(evidence["body_mass_kg"]) * 9.81)


@pytest.mark.parametrize("merged_fixed", [False, True])
def test_sensor_configuration_is_one_body_to_exact_terrain_not_self(evaluator, monkeypatch, tmp_path, merged_fixed):
    import sys
    from types import ModuleType

    fake_sensors = ModuleType("isaaclab.sensors")
    fake_sensors.ContactSensorCfg = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", fake_sensors)
    asset = tmp_path / "rover.urdf"
    names = ["base_link", "fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel", "fixed_payload"]
    asset.write_text('<robot>' + ''.join(f'<link name="{name}"/>' for name in names)
                     + '<joint name="mount" type="fixed"><parent link="base_link"/><child link="fixed_payload"/></joint>'
                     + '</robot>')
    cfg = SimpleNamespace(scene=SimpleNamespace(
        robot=SimpleNamespace(spawn=SimpleNamespace(asset_path=str(asset), activate_contact_sensors=True,
                                                    merge_fixed_joints=merged_fixed)),
        terrain=SimpleNamespace(terrain_type="generator", prim_path="/World/ground"),
    ))
    metadata = evaluator.configure_support_sensors(cfg)
    expected = names[:-1] if merged_fixed else names
    assert metadata["body_names"] == expected
    assert metadata["sensor_count_per_env"] == len(expected)
    for name in expected:
        sensor = getattr(cfg.scene, "vision_ground_" + name)
        assert sensor.prim_path == f"{{ENV_REGEX_NS}}/Robot/{name}"
        assert sensor.filter_prim_paths_expr == ["/World/ground/terrain/mesh"]
        assert sensor.history_length == 0
        assert sensor.update_period == 0.


def test_all_body_mass_shape_is_required_not_only_wheel_mass(evaluator):
    env = make_env(origin_z=0.)
    env.scene["robot"].root_physx_view.get_masses = lambda: torch.ones(1, 4)
    with pytest.raises(ValueError, match="全部机器人刚体"):
        evaluator.capture_traversal(env, scenario="boxes")
