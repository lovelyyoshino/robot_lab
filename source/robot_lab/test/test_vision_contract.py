"""同维度不同语义、错误蒸馏角色必须在启动仿真前拒绝。

@author pony
@date 2026-09-30
@version v0.2.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 复杂楼梯几何和成功截断变更必须拒绝旧合同，观测维度相同不能绕过。
    - (2026-10-12): 执行play实际配置控制流，训练合同须在回放地形缩减前保存。
    - (2026-10-12): 速度参数回归改检查实际使用的reward_terms，不再测试重复合同字段。
    - (2026-10-12): 合同实际构建过滤关闭奖励，全部有效项的增删、权重和参数变更均需拒绝。
    - (2026-10-12): 路线课程行列数或阶段函数变化时禁止加载旧模型。
    - (2026-10-11): 速度奖励零基线及std变更必须随训练目标合同校验。
    - (2026-10-11): 同样187点但近场坐标改变必须拒绝旧Teacher和蒸馏来源。
    - v0.2.0 (2026-10-08): 统一主动抬轮合同并测试同名目标下实现变更拒绝。
    - v0.1.0 (2026-09-30): 覆盖Preview模型角色、扫描/相机错配、缺合同和扰动差异。
"""

import copy
from pathlib import Path

import pytest


@pytest.fixture
def contract(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    import vision_contract
    return vision_contract


def sample(role):
    return {"version": "rover_vision_prelift", "role": role,
            "shared": {"scan": [0.3, 0, 20], "training_objective": {
                "version": "active_wheel_prelift", "step_terrains": {"single_step": {
                    "step_front_x_range": [1.2, 1.7], "step_height_range": [0.04, 0.1]}}}},
            "depth": {"angle": 25} if role == "student" else None}


def test_same_role_and_teacher_to_student(contract):
    contract.validate_contract(sample("teacher"), sample("teacher"))
    contract.validate_contract(sample("student"), sample("student"))
    contract.validate_contract(sample("teacher"), sample("student"), teacher_source=True)


@pytest.mark.parametrize("source,target,distillation", [
    ("student", "student", True), ("teacher", "student", False), ("student", "teacher", False),
    ("teacher", "teacher", True),
])
def test_roles_must_match_load_intent(contract, source, target, distillation):
    with pytest.raises(ValueError, match="角色"):
        contract.validate_contract(sample(source), sample(target), teacher_source=distillation)


@pytest.mark.parametrize("key,value", [("shared", {"scan": [0, 0, 20]}), ("depth", {"angle": 15})])
def test_dimension_equality_cannot_hide_changed_semantics(contract, key, value):
    changed = copy.deepcopy(sample("student"))
    changed[key] = value
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(changed, sample("student"))


@pytest.mark.parametrize("student", [False, True])
def test_same_scan_count_does_not_make_near_field_coordinates_compatible(contract, student):
    target = sample("student" if student else "teacher")
    target["shared"]["height_scanner"] = {"pattern_cfg": {"x_positions": [0.20, 0.22, 0.24]}}
    old = copy.deepcopy(target)
    old["role"] = "teacher"
    old["shared"]["height_scanner"]["pattern_cfg"]["x_positions"] = [0.10, 0.20, 0.30]
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(old, target, teacher_source=student)


def test_old_training_objective_cannot_resume_under_new_reward(contract):
    old_target = copy.deepcopy(sample("teacher"))
    old_target["shared"].pop("training_objective")
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(old_target, sample("teacher"))


def test_implementation_change_rejected_without_creating_new_task_version(contract):
    target = sample("teacher")
    target["shared"]["training_objective"]["implementation_sha256"] = {"reward": "new"}
    changed = copy.deepcopy(target)
    changed["shared"]["training_objective"]["implementation_sha256"] = {"reward": "old"}
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(changed, target)


def test_preview_velocity_objective_change_cannot_silently_resume(contract):
    target = sample("teacher")
    target["shared"]["training_objective"]["reward_terms"] = {"track_lin_vel_xy_exp": {
        "func": "preview_velocity_tracking", "weight": 5., "params": {"std": 0.5},
    }}
    changed = copy.deepcopy(target)
    changed["shared"]["training_objective"]["reward_terms"]["track_lin_vel_xy_exp"]["params"]["std"] = 0.3
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(changed, target)


@pytest.fixture
def reward_contract(contract, tmp_path):
    asset = tmp_path / "robot.urdf"
    asset.write_text("<robot name='reward_contract_test'/>", encoding="utf-8")
    env = {
        "scene": {"robot": {"spawn": {"asset_path": str(asset)}, "init_state": {}, "actuators": {}},
                  "height_scanner": dict.fromkeys(("offset", "pattern_cfg", "ray_alignment",
                                                   "update_period", "mesh_prim_paths"), None),
                  "terrain": {"terrain_generator": {"sub_terrains": {"flat": {"proportion": 1.}},
                                                    "num_rows": 10, "num_cols": 24,
                                                    "difficulty_range": [0, 1], "curriculum": True}}},
        "observations": {"policy": {}}, "actions": {}, "sim": {"dt": 0.005}, "decimation": 4,
        "vision_preview": {}, "commands": {"base_velocity": {"ranges": {}}},
        "curriculum": {"terrain_levels": {}},
        "rewards": {"track_lin_vel_xy_exp": {"func": "preview_velocity_tracking", "weight": 5.,
                                              "params": {"std": 0.5}},
                    "safety": {"func": "contact_forces", "weight": -0.1, "params": {"threshold": 10.}},
                    "removed": None, "disabled": {"func": "upward", "weight": 0}, "metadata": "ignored"},
    }
    agent = {"policy": {"actor_hidden_dims": [512, 256, 128], "actor_obs_normalization": False,
                        "activation": "elu"}, "clip_actions": None, "obs_groups": {}}
    build = lambda: contract.build_contract(next(task for task, role in contract.PREVIEW_TASKS.items()
                                                if role == "teacher"), env, agent)
    return env, build


def test_contract_build_records_only_active_reward_terms(reward_contract):
    env, build = reward_contract
    result = build()
    assert result["shared"]["training_objective"]["reward_terms"] == {
        name: env["rewards"][name] for name in ("track_lin_vel_xy_exp", "safety")
    }


@pytest.mark.parametrize("field,value", [("height_levels", [.08, .12, .18]), ("step_counts", [2, 4, 5])])
def test_stair_geometry_change_cannot_silently_resume(contract, reward_contract, field, value):
    env, build = reward_contract
    stairs = {"function": "ascending_stairs_terrain", "proportion": .25,
              "height_levels": [.08, .12, .16], "step_counts": [2, 3, 4], "first_edge_m": 1.2}
    env["scene"]["terrain"]["terrain_generator"]["sub_terrains"]["pyramid_stairs_inv"] = stairs
    original = build()
    stairs[field] = value
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(original, build())


@pytest.mark.parametrize("change", ["remove", "add", "weight", "params", "func"])
def test_any_active_reward_change_cannot_silently_resume(contract, reward_contract, change):
    env, build = reward_contract
    original = build()
    if change == "remove":
        env["rewards"]["safety"] = None
    elif change == "add":
        env["rewards"]["new"] = {"func": "new_reward", "weight": -0.1}
    elif change == "weight":
        env["rewards"]["safety"]["weight"] = -0.2
    elif change == "params":
        env["rewards"]["safety"]["params"]["threshold"] = 20.
    else:
        env["rewards"]["safety"]["func"] = "another_reward"
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(original, build())


def test_preview_curriculum_layout_change_cannot_silently_resume(contract):
    target = sample("teacher")
    target["shared"]["training_objective"]["terrain_layout"] = {"num_rows": 10, "num_cols": 24}
    changed = copy.deepcopy(target)
    changed["shared"]["training_objective"]["terrain_layout"]["num_cols"] = 20
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(changed, target)


def test_play_captures_training_contract_before_overriding_terrain(contract, reward_contract, monkeypatch):
    import ast
    from types import SimpleNamespace

    env, build = reward_contract
    original = build()
    terrain = SimpleNamespace(num_rows=10, num_cols=24, difficulty_range=[0, 1], curriculum=True)
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=5, terrain=SimpleNamespace(
            terrain_generator=terrain, max_init_terrain_level=0,
        )),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
        sim=SimpleNamespace(device="cpu"), events=SimpleNamespace(),
        curriculum=SimpleNamespace(terrain_levels={}),
    )
    captured = []

    def capture(task, cfg, agent):
        layout = env["scene"]["terrain"]["terrain_generator"]
        for name in ("num_rows", "num_cols", "difficulty_range", "curriculum"):
            layout[name] = getattr(cfg.scene.terrain.terrain_generator, name)
        env["curriculum"]["terrain_levels"] = cfg.curriculum.terrain_levels
        target = build()
        contract.validate_contract(original, target)
        captured.append(target)
        return target

    monkeypatch.setattr(contract, "contract_for_cfg", capture)
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/play.py"
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    snapshot = next(index for index, node in enumerate(main.body)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "preview_contract"
                            for target in node.targets))
    # 执行真实入口直到合同快照；只替换CLI和仿真对象，才能暴露“先改场景再校验”的顺序错误。
    main.body = main.body[:snapshot + 1]
    main.decorator_list = []
    for arg in main.args.args:
        arg.annotation = None
    namespace = {
        "args_cli": SimpleNamespace(task=next(iter(contract.PREVIEW_TASKS)), num_envs=5, device=None),
        "cli_args": SimpleNamespace(update_rsl_rl_cfg=lambda agent, args: agent,
                                    handle_deprecated_rsl_rl_cfg=lambda agent, version: agent),
        "DEFAULT_PLAY_ENVS": 5, "installed_version": "3.0.1",
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
    namespace["main"](env_cfg, SimpleNamespace(seed=42))
    assert captured == [original]


def test_changed_obstacle_position_cannot_reuse_checkpoint(contract):
    changed = copy.deepcopy(sample("teacher"))
    changed["shared"]["training_objective"]["step_terrains"]["single_step"]["step_front_x_range"] = [0.9, 0.9]
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(changed, sample("teacher"))


def test_changed_side_obstacle_geometry_cannot_reuse_checkpoint(contract):
    target = sample("teacher")
    target["shared"]["training_objective"]["step_terrains"]["single_step_left"] = {
        "lane_width": 0.36, "lane_center_y": 0.30,
    }
    changed = copy.deepcopy(target)
    changed["shared"]["training_objective"]["step_terrains"]["single_step_left"]["lane_center_y"] = -0.30
    with pytest.raises(ValueError, match="合同不匹配"):
        contract.validate_contract(changed, target)


def test_missing_contract_does_not_create_fake_metadata(contract, tmp_path):
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="缺少"):
        contract.validate_checkpoint(checkpoint, sample("teacher"))
    assert not (tmp_path / "params").exists()


def test_noise_ignored_but_term_order_preserved(contract):
    a = {"x": {"func": "x", "scale": 1, "noise": {"min": -0.1}},
         "y": {"func": "y"}, "enable_corruption": True}
    b = copy.deepcopy(a)
    b["x"]["noise"] = None
    b["enable_corruption"] = False
    assert contract.observation_contract(a) == contract.observation_contract(b)
    assert contract.observation_contract(a) != contract.observation_contract({"y": b["y"], "x": b["x"]})


def test_legacy_task_stays_untouched(contract):
    assert contract.build_contract("legacy", {}, {}) is None
    contract.validate_checkpoint("does_not_exist.pt", None)


def test_copied_json_cannot_disguise_teacher_as_student(contract, tmp_path):
    import torch

    checkpoint = tmp_path / "model_1.pt"
    torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(2, 3)}}, checkpoint)
    contract.save_contract(tmp_path / "params", sample("student"))
    with pytest.raises(ValueError, match="实际权重"):
        contract.validate_checkpoint(checkpoint, sample("student"))
