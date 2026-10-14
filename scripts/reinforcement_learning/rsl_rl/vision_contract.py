"""前瞻视觉模型的输入/动作语义合同；同维数不表示可兼容。

@author pony
@date 2026-09-30
@version v0.2.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 保存直楼梯及方块完整几何，防止高度/级数变化后静默续训。
    - (2026-10-12): 速度奖励由reward_terms统一记录，删除重复的velocity_tracking合同字段。
    - (2026-10-12): 保存全部有效奖励配置，裁剪或调整基础奖励也会拒绝不兼容续训。
    - (2026-10-12): 锁定三阶段地形布局、路线命令与晋级函数，拒绝旧Preview模型误续训。
    - (2026-10-11): 保存Preview速度奖励函数、权重和参数，防止仅形状相同却改变静止收益的误续训。
    - v0.2.0 (2026-10-08): 单一主动抬轮入口用配置和实现指纹防止误续训，不增加任务版本。
    - v0.1.0 (2026-09-30): 在新任务训练、续训与回放前校验 teacher/student 来源。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREFIX = "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-"
PREVIEW_TASKS = {PREFIX + "Teacher-Preview-v0": "teacher",
                 PREFIX + "Student-Temporal-Preview-v0": "student"}


def canonical(value):
    """只接受配置数据，不使用 repr 或执行任意序列化对象。"""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if callable(value):
        return value.__module__ + ":" + value.__qualname__
    raise TypeError(f"不支持的视觉合同字段类型：{type(value).__name__}")


def observation_contract(group):
    # 噪声开关属于训练/评测扰动，不属于网络输入变换；保留实际 term 顺序。
    terms = [[name, {k: v for k, v in term.items() if k != "noise"}]
             for name, term in group.items() if isinstance(term, dict) and "func" in term]
    options = {k: v for k, v in group.items() if k != "enable_corruption" and not isinstance(v, dict)}
    return {"terms_in_order": terms, "options": options}


def build_contract(task, env, agent, *, depth=None):
    """使用 gym.make 前最终配置，避免 manager 解析后的索引污染静态合同。"""
    if task not in PREVIEW_TASKS:
        return None
    role = PREVIEW_TASKS[task]
    is_student = role == "student"
    group = env["observations"]["teacher" if is_student else "policy"]
    robot = env["scene"]["robot"]
    asset = Path(robot["spawn"]["asset_path"])
    if not asset.is_file():
        raise FileNotFoundError(f"视觉合同需要本地机器人资产：{asset}")
    scan = env["scene"]["height_scanner"]
    terrain_cfg = env["scene"]["terrain"]["terrain_generator"]
    rewards = env["rewards"]
    policy = agent["policy"]
    reward_dir = Path(__file__).resolve().parents[3] / (
        "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover"
    )
    result = {
        "version": "rover_vision_prelift", "role": role,
        "shared": {
            "teacher_observation": observation_contract(group),
            "height_scanner": {k: scan[k] for k in (
                "offset", "pattern_cfg", "ray_alignment", "update_period", "mesh_prim_paths")},
            "actions": env["actions"], "wrapper_clip": agent["clip_actions"],
            "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "default_state": robot["init_state"], "actuators": robot["actuators"],
            "control_dt": env["sim"]["dt"] * env["decimation"],
            "preview": env["vision_preview"],
            "training_objective": {
                "version": env.get("vision_preview_objective_version", "unversioned"),
                "implementation_sha256": {
                    name: hashlib.sha256((reward_dir / name).read_bytes()).hexdigest()
                    for name in ("vision_preview.py", "vision_preview_reward.py")
                },
                "terrain_proportions": {
                    name: terrain["proportion"]
                    for name, terrain in terrain_cfg["sub_terrains"].items()
                },
                "terrain_layout": {name: terrain_cfg[name] for name in (
                    "num_rows", "num_cols", "difficulty_range", "curriculum"
                )},
                "route_command": env["commands"]["base_velocity"]["ranges"],
                "terrain_curriculum": env["curriculum"]["terrain_levels"],
                "episode_length_s": env.get("episode_length_s"),
                "terminations": env.get("terminations", {}),
                "complex_terrains": {
                    name: terrain for name, terrain in terrain_cfg["sub_terrains"].items()
                    if not name.startswith("single_step") and name != "flat"
                },
                "step_terrains": {
                    terrain_name: {
                        name: terrain[name] for name in (
                            "step_height_range", "step_front_x_range", "step_width", "lane_width", "lane_center_y"
                        )
                    }
                    for terrain_name, terrain in terrain_cfg["sub_terrains"].items()
                    if terrain_name.startswith("single_step")
                },
                "reward_terms": {
                    name: term for name, term in rewards.items()
                    if isinstance(term, dict) and "func" in term and term.get("weight", 0) != 0
                },
            },
            "teacher_network": {
                "hidden_dims": policy["teacher_hidden_dims" if is_student else "actor_hidden_dims"],
                "normalization": policy["teacher_obs_normalization" if is_student else "actor_obs_normalization"],
                "activation": policy["activation"],
            },
        },
        "own_policy": policy,
        "obs_groups": agent["obs_groups"],
    }
    if is_student:
        if depth is None:
            raise ValueError("student必须保存实际深度合同")
        result["student_observation"] = observation_contract(env["observations"]["policy"])
        result["depth"] = {k: v for k, v in depth.items() if k != "corruption_enabled"}
    # init_noise_std 仅决定初始化探索，不影响训练后 checkpoint 的接口语义。
    result["own_policy"] = {k: v for k, v in policy.items() if k != "init_noise_std"}
    return canonical(result)


def contract_for_cfg(task, env_cfg, agent_cfg):
    if task not in PREVIEW_TASKS:
        return None
    depth = None
    if PREVIEW_TASKS[task] == "student":
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover import (
            vision_env_cfg,
        )

        depth = vision_env_cfg.vision_depth_contract(env_cfg)
    return build_contract(task, env_cfg.to_dict(), agent_cfg.to_dict(), depth=depth)


def validate_contract(source, target, *, teacher_source=False):
    """蒸馏只比较共享 teacher 接口；同角色续训/回放同时验证深度及网络接口。"""
    if source.get("version") != target["version"]:
        raise ValueError("视觉模型合同版本不匹配；不能将旧模型当作Preview模型加载")
    expected = "teacher" if teacher_source else target["role"]
    if source.get("role") != expected or (teacher_source and target["role"] != "student"):
        raise ValueError("视觉 checkpoint 角色错误：新student需teacher，resume/play需同角色模型")
    keys = ("shared",) if teacher_source else tuple(target)
    for key in keys:
        if source.get(key) != target[key]:
            raise ValueError(f"视觉合同不匹配：{key}；禁止按相同观测维数静默加载")


def validate_checkpoint(checkpoint, target, *, teacher_source=False):
    if target is None:
        return
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    contract_path = path.parent / "params" / "vision_contract.json"
    if not contract_path.is_file():
        raise ValueError(f"缺少Preview训练合同：{contract_path}；请使用新任务训练，不要手工补假合同")
    validate_contract(json.loads(contract_path.read_text(encoding="utf-8")), target, teacher_source=teacher_source)
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", {}) if isinstance(payload, dict) else {}
    role = "teacher" if teacher_source else target["role"]
    expected_key = "actor.0.weight" if role == "teacher" else "student.0.weight"
    if expected_key not in state:
        raise ValueError(f"checkpoint实际权重不是{role}模型；不能只复制另一run的JSON合同")


def save_contract(directory, contract):
    if contract is not None:
        path = Path(directory) / "vision_contract.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
