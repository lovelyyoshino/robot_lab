#!/usr/bin/env python3
"""同一台 Isaac 仿真车执行行走、起立/倒立、倒地恢复策略。

@author pony
@date 2026-09-16
@version v0.4.0
@last_modified 2026-09-17
@changelog
    - v0.4.0 (2026-09-17): 先验证训练合同再导出，补齐动作项顺序和原始 action class 校验。
    - v0.3.0 (2026-09-17): 增加 44 维 motion 与显式 Temporal 视觉合同校验，拒绝仅按输入维数切换。
    - v0.2.0 (2026-09-16): 替换零动作占位，加载三个 checkpoint、共享物理状态并切换动作尺度。

运行本脚本仅验证仿真策略集成；各技能需先通过独立验收。
upright-checkpoint 应使用经过起立阶段训练的模型，纯平衡模型不能代表四轮起立能力。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


TASKS = {
    "blind": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0",
    "student": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-v0",
    "upright": "RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0",
    "recover": "RobotLab-Isaac-Recovery-Parallelogram-Rover-v0",
}

_ACTIVE_LEG_JOINT = re.compile(r"^(?:fl|rl|fr|rr)_(?:arm|qudong)_joint$")


def _strict_stance_measurements(robot, contact_sensor, contact_state_from_forces):
    """从唯一的单车传感器快照构造状态机严格门；传感器/关节名异常立即失败。"""
    contact = contact_state_from_forces(contact_sensor.body_names, contact_sensor.data.net_forces_w[0])
    joint_names = list(robot.joint_names)
    active_leg_ids = [index for index, name in enumerate(joint_names) if _ACTIVE_LEG_JOINT.fullmatch(name)]
    if len(active_leg_ids) != 8:
        raise RuntimeError(f"主动腿关节名不完整，期望8个 arm/qudong，实际 {len(active_leg_ids)}")
    data = robot.data
    linear_speed = float(data.root_lin_vel_w[0].norm())
    leg_speed = float(data.joint_vel[0, active_leg_ids].abs().amax())
    if not all(math.isfinite(value) for value in (linear_speed, leg_speed)):
        raise RuntimeError("严格状态机的线速度或主动腿速度非有限")
    return contact, linear_speed, leg_speed


def _checkpoint_actor_input_dim(checkpoint: dict, network_type: str) -> int:
    """从权重读取 actor 输入，先于网络加载暴露错误的策略/环境组合。"""
    key = "student.0.weight" if network_type == "StudentTeacher" else "actor.0.weight"
    weight = checkpoint.get("model_state_dict", {}).get(key)
    if weight is None or weight.ndim != 2:
        raise ValueError(f"checkpoint 缺少可验证的 {key}，拒绝猜测观测合同")
    return int(weight.shape[1])


def _load_depth_contract(path: Path, expected_id: str) -> dict:
    """读取随 Temporal/连续帧 checkpoint 导出的合同，不能因同为 4141 维而省略。"""
    candidates = (path.parent / "exported" / "depth_contract.json", path.parent / "params" / "depth_contract.json")
    for candidate in candidates:
        if candidate.is_file():
            contract = json.loads(candidate.read_text())
            break
    else:
        raise ValueError(f"视觉 checkpoint 缺少 depth_contract.json：{path.parent}")
    if contract.get("contract_id") != expected_id:
        raise ValueError(
            f"视觉 checkpoint 合同为 {contract.get('contract_id')!r}，但所选策略要求 {expected_id!r}"
        )
    required = {"student_obs_dim": 4141, "teacher_obs_dim": 232, "action_dim": 12}
    if any(contract.get(name) != value for name, value in required.items()):
        raise ValueError("视觉 checkpoint 的观测/动作维度合同不完整或不匹配")
    return contract


def _load_training_config(path: Path) -> dict:
    """复用 warm-start 的安全 YAML loader，不接受缺失训练合同的 checkpoint。"""
    loader_path = Path(__file__).parent / "reinforcement_learning/rsl_rl/upright_warm_start.py"
    spec = importlib.util.spec_from_file_location("hybrid_upright_warm_start", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载训练合同安全读取器")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_training_config(path)


def _canonical(value):
    """将 YAML tuple、配置对象和路径转为可比较的纯值，避免显示格式影响合同。"""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _canonical(item) for key, item in vars(value).items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _require_equal(expected, actual, label: str) -> None:
    if _canonical(expected) != _canonical(actual):
        raise ValueError(f"{label} 与 checkpoint 保存的训练合同不一致")


def _without(mapping: dict, *keys: str) -> dict:
    return {key: value for key, value in mapping.items() if key not in keys}


def _uniform_action_value(value, joint_names: list[str], label: str):
    """将 Isaac 的 per-regex action 配置归一为逐关节值，比较执行语义而非 YAML 写法。"""
    if not isinstance(value, dict):
        return value
    resolved = []
    for joint_name in joint_names:
        matches = [item for pattern, item in value.items() if __import__("re").fullmatch(pattern, joint_name)]
        if len(matches) != 1:
            raise ValueError(f"{label} 无法唯一解析关节 {joint_name!r} 的动作值")
        resolved.append(matches[0])
    if not resolved or any(item != resolved[0] for item in resolved[1:]):
        raise ValueError(f"{label} 必须对所有动作关节使用同一值")
    return resolved[0]


def _validate_action_contract(expected_action: dict, actual_action: dict, runtime_cfg, policy_spec: dict, name: str) -> None:
    """验证动作关节/offset/scale/clip；只允许 hybrid 动态接管 wheel scale 与 clip。"""
    for field in ("asset_name", "joint_names", "offset", "preserve_order", "use_default_offset", "debug_vis"):
        _require_equal(expected_action.get(field), actual_action.get(field), f"actions.{name}.{field}")
    joint_names = expected_action.get("joint_names")
    if not isinstance(joint_names, list) or not joint_names:
        raise ValueError(f"actions.{name}.joint_names 必须为非空训练保存顺序")
    expected_scale = _uniform_action_value(expected_action.get("scale"), joint_names, f"actions.{name}.scale")
    actual_scale = _uniform_action_value(actual_action.get("scale"), joint_names, f"actions.{name}.scale")
    expected_class = expected_action.get("class_type")
    required_class = "JointVelocityAction" if name == "joint_vel" else "JointPositionAction"
    if not isinstance(expected_class, str) or not expected_class.endswith(f":{required_class}"):
        raise ValueError(f"actions.{name}.class_type 不是训练所需的 {required_class}")
    if name == "joint_vel":
        if expected_scale != policy_spec["wheel_scale"]:
            raise ValueError(f"actions.joint_vel 的训练 scale 不符合策略规格：{expected_scale}")
        if getattr(runtime_cfg.actions.joint_vel.class_type, "__name__", None) != "HybridWheelVelocityAction":
            raise ValueError("共享环境未启用 HybridWheelVelocityAction，不能保证轮速动作语义")
        # 静态 loco 选择器可以是另一模式的值；执行前由 runtime 按当前策略 scale 改写。
        if actual_scale not in (0.8, 5.0):
            raise ValueError(f"共享环境的 wheel scale 非法：{actual_scale}")
        expected_clip = expected_action.get("clip")
        if expected_clip != {".*": (-100.0, 100.0)} and expected_clip != {".*": (-31.4159265359, 31.4159265359)}:
            raise ValueError("actions.joint_vel 的训练 clip 不是已验证的 rover 轮速上限")
    else:
        _require_equal(expected_scale, actual_scale, f"actions.{name}.scale")
        _require_equal(expected_action.get("clip"), actual_action.get("clip"), f"actions.{name}.clip")


def _validate_previous_action_term(expected: dict, runtime_term, wheel_scale: float, label: str) -> None:
    """仅允许 hybrid 的单位换算包装，其他上一动作观测语义不得漂移。"""
    expected_func = expected.get("func", "")
    if not isinstance(expected_func, str) or not expected_func.endswith(":last_action"):
        raise ValueError(f"{label} 原训练项不是标准 last_action，拒绝替换为 hybrid 包装")
    if runtime_term.func.__name__ != "last_action_for_skill" or runtime_term.params != {"wheel_scale": wheel_scale}:
        raise ValueError(f"{label} 未使用受控的上一动作单位换算")


def _validate_observation_group(expected: dict, runtime_group, *, wheel_scale: float, label: str) -> None:
    """严格比较观测项次序与每项处理；只豁免受控的上一动作单位包装。"""
    actual = _canonical(runtime_group)
    expected_terms = [name for name, value in expected.items() if isinstance(value, dict) and "func" in value]
    actual_terms = [name for name, value in actual.items() if isinstance(value, dict) and "func" in value]
    if expected_terms != actual_terms:
        raise ValueError(f"{label} 观测项顺序或字段不匹配")
    for name in expected_terms:
        expected_term = expected[name]
        if name in {"actions", "last_action"}:
            _validate_previous_action_term(expected_term, getattr(runtime_group, name), wheel_scale, f"{label}.{name}")
            _require_equal(_without(expected_term, "func", "params"), _without(actual[name], "func", "params"), f"{label}.{name} 处理")
        else:
            _require_equal(expected_term, actual[name], f"{label}.{name}")
    expected_options = {name: value for name, value in expected.items() if name not in expected_terms}
    actual_options = {name: value for name, value in actual.items() if name not in actual_terms}
    # 集成烟测仅允许从训练时开启的扰动收紧为关闭，不能反向开启或改变其它处理。
    expected_corruption = expected_options.pop("enable_corruption", None)
    actual_corruption = actual_options.pop("enable_corruption", None)
    if actual_corruption != expected_corruption and not (expected_corruption is True and actual_corruption is False):
        raise ValueError(f"{label}.enable_corruption 不属于允许的确定性烟测收紧")
    _require_equal(expected_options, actual_options, f"{label} 组处理")


def _validate_checkpoint_training_contract(path: Path, policy_spec: dict, runtime_cfg, runtime_agent_cfg, *,
                                           training_group: str = "policy", visual_contract: dict | None = None) -> dict:
    """将 checkpoint 的真实训练合同与当前 hybrid 配置逐项比较，任一缺失均 fail closed。

    `runtime_cfg` 可因共享物理环境增加其它观测组。actor 是唯一在线输入，故逐项核验；
    critic 只用于训练，在线不读取，仍由保存的 runner 网络定义和 strict state-dict 加载校验。
    """
    params = path.parent / "params"
    env_path, agent_path = params / "env.yaml", params / "agent.yaml"
    if not env_path.is_file() or not agent_path.is_file():
        raise ValueError(f"checkpoint 缺少 params/env.yaml 或 params/agent.yaml：{path.parent}")
    saved_env = _load_training_config(env_path)
    saved_agent = _load_training_config(agent_path)
    current_env = _canonical(runtime_cfg)
    current_agent = _canonical(runtime_agent_cfg)
    try:
        saved_group = saved_env["observations"][training_group]
        runtime_group = getattr(runtime_cfg.observations, policy_spec["observation_group"])
        _validate_observation_group(saved_group, runtime_group, wheel_scale=policy_spec["wheel_scale"],
                                    label=f"observations.{training_group}")
        _require_equal(list(saved_env["actions"]), list(current_env["actions"]), "动作项顺序")
        for name, expected_action in saved_env["actions"].items():
            _validate_action_contract(expected_action, current_env["actions"][name], runtime_cfg, policy_spec, name)
        for name in ("joint_pos", "joint_vel"):
            _require_equal(saved_env["scene"]["robot"]["init_state"][name],
                           current_env["scene"]["robot"]["init_state"][name], f"默认{name}")
        _require_equal(saved_env["sim"]["dt"], current_env["sim"]["dt"], "sim.dt")
        _require_equal(saved_env["decimation"], current_env["decimation"], "decimation")
        for name in ("class_name", "clip_actions", "obs_groups", "empirical_normalization"):
            _require_equal(saved_agent.get(name), current_agent.get(name), f"runner.{name}")
        _require_equal(saved_agent["policy"], current_agent["policy"], "runner.policy")
    except (KeyError, AttributeError) as error:
        raise ValueError(f"训练合同缺少必须字段：{error}") from error
    if visual_contract is not None:
        saved_depth = _load_depth_contract(path, policy_spec["depth_contract_id"])
        _require_equal(_without(saved_depth, "corruption_enabled"), _without(visual_contract, "corruption_enabled"),
                       "Temporal/连续帧深度预处理")
    return {
        "env_yaml": str(env_path), "agent_yaml": str(agent_path), "checked": True,
        "actor_observation_checked": True, "critic_runtime_input_checked": False,
        "critic_runtime_input_reason": "critic_is_not_read_during_inference",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    required = [parser.add_argument(f"--{name}-checkpoint", type=Path) for name in ("loco", "upright", "recover")]
    parser.add_argument("--loco-policy", choices=("blind", "student", "student-temporal"), default="blind")
    parser.add_argument("--upright-policy", choices=("upright", "motion"), default="upright")
    parser.add_argument("--hybrid-steps", type=int, default=1000)
    parser.add_argument("--schedule", default="0:lo,200:up,600:lo", help="控制步:指令，指令为 lo/up/rec。")
    parser.add_argument("--velocity", type=float, default=0.2)
    parser.add_argument("--upright-velocity", type=float, default=0.0, help="motion 两轮态的地面前进指令 m/s。")
    parser.add_argument("--upright-yaw-rate", type=float, default=0.0, help="motion 两轮态的世界 yaw 指令 rad/s。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    for arg in required:
        arg.required = True
    args = parser.parse_args()
    if args.hybrid_steps < 1:
        parser.error("hybrid-steps 必须为正")
    schedule = {}
    try:
        for entry in args.schedule.split(","):
            step, command = entry.split(":")
            step = int(step)
            if step < 0 or command not in ("lo", "up", "rec") or step in schedule:
                raise ValueError
            schedule[step] = command
        if 0 not in schedule:
            raise ValueError
    except ValueError:
        parser.error("schedule 必须从 0 开始，格式如 0:lo,200:up,600:lo，步数不可重复")
    args.schedule = schedule
    for name in ("loco", "upright", "recover"):
        path = getattr(args, name + "_checkpoint").expanduser().resolve()
        if not path.is_file():
            parser.error(f"checkpoint 不存在：{path}")
        setattr(args, name + "_checkpoint", path)
    if args.loco_policy.startswith("student"):
        args.enable_cameras = True
    return args


def load_policy(task, path, env, policy_spec, device):
    """按任务网络配置严格重载；额外技能只创建网络，不创建第二个仿真环境。"""
    import torch
    from rsl_rl.modules import ActorCritic, StudentTeacher
    from tensordict import TensorDict

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    network_cfg = cfg.policy.to_dict()
    network_type = network_cfg.pop("class_name")
    group = policy_spec["observation_group"]
    obs = env.observation_manager.compute(update_history=False)
    if int(obs[group].shape[-1]) != policy_spec["actor_obs_dim"]:
        raise ValueError(
            f"{task} 的 {group} 观测为 {obs[group].shape[-1]} 维，"
            f"但策略合同要求 {policy_spec['actor_obs_dim']} 维"
        )
    selected = TensorDict({"policy": obs[group], "critic": obs[group]}, batch_size=[1])
    if network_type == "StudentTeacher":
        selected["teacher"] = obs["teacher"]
        network = StudentTeacher(selected, {"policy": ["policy"], "teacher": ["teacher"]}, 12, **network_cfg)
    elif network_type == "ActorCritic":
        # critic 不参与推理，但严格加载仍使用 checkpoint 对应的训练维度。
        critic_dim = policy_spec.get("critic_obs_dim", obs["critic"].shape[-1])
        selected["critic"] = torch.zeros(1, critic_dim, device=device)
        network = ActorCritic(selected, {"policy": ["policy"], "critic": ["critic"]}, 12, **network_cfg)
    else:
        raise ValueError(f"当前集成入口不支持网络：{network_type}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_dim = _checkpoint_actor_input_dim(checkpoint, network_type)
    if checkpoint_dim != policy_spec["actor_obs_dim"]:
        raise ValueError(
            f"{path.name} 的 actor 输入为 {checkpoint_dim} 维，不能加载到要求 "
            f"{policy_spec['actor_obs_dim']} 维的 {task} 合同"
        )
    network.to(device).load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.eval()
    return network, cfg.clip_actions, cfg


def export_validated_policy(network, policy_cfg, path: Path) -> None:
    """仅在所有 checkpoint 训练合同通过后导出，避免失败模型覆盖已有部署产物。"""
    from isaaclab_rl.rsl_rl import export_policy_as_jit

    policy_type = policy_cfg.policy.class_name
    normalizer = network.student_obs_normalizer if policy_type == "StudentTeacher" else network.actor_obs_normalizer
    export_policy_as_jit(network, normalizer, str(path.parent / "exported"), filename="policy.pt")


def main(args, app):
    import gymnasium as gym
    import torch
    from tensordict import TensorDict

    from isaaclab.managers import EventTermCfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.manager_based.hybrid.planner import (
        LOCOMOTION_POLICY_SPECS,
        UPRIGHT_POLICY_SPECS,
        StanceStateMachine,
        contact_state_from_forces,
    )
    from robot_lab.tasks.manager_based.hybrid.runtime import (
        HybridWheelVelocityAction,
        last_action_for_skill,
        reset_hybrid_history,
    )
    from robot_lab.tasks.manager_based.recovery.config.method1_deng.env_cfg import RecoveryObservationsCfg
    from robot_lab.tasks.manager_based.upright.config.motion_env_cfg import MotionCommandsCfg, MotionObservationsCfg
    from robot_lab.tasks.manager_based.upright.config.upright_env_cfg import UprightCommandsCfg, UprightObservationsCfg

    loco_spec = LOCOMOTION_POLICY_SPECS[args.loco_policy]
    upright_spec = UPRIGHT_POLICY_SPECS[args.upright_policy]
    task = loco_spec["task"]
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = (args.hybrid_steps + 1) * cfg.sim.dt * cfg.decimation
    # 先在平地验证同车切换，障碍性能由视觉专用入口评测。
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum = None
    for name in ("randomize_push_robot", "push_robot", "randomize_apply_external_force_torque"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {}
    cfg.events.randomize_reset_base.params["velocity_range"] = {}
    cfg.events.hybrid_reset_history = EventTermCfg(func=reset_hybrid_history, mode="reset")
    # 接管恢复动作时不能由行走的倒地终止项提前 reset。
    for name in list(vars(cfg.terminations)):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    cfg.commands.pitch_command = UprightCommandsCfg().pitch_command
    cfg.commands.pitch_command.pitch_range = (1.57, 1.57)
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.velocity, args.velocity)
    cmd.ranges.lin_vel_y = cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.heading_command = False
    cmd.rel_standing_envs = cmd.rel_heading_envs = 0.0
    cmd.debug_vis = False
    if args.upright_policy == "motion":
        motion_commands = MotionCommandsCfg().motion_command
        motion_commands.fixed_stage = 2
        motion_commands.fixed_command = (args.upright_velocity, args.upright_yaw_rate)
        cfg.commands.motion_command = motion_commands
        cfg.observations.motion = MotionObservationsCfg.PolicyCfg()
    else:
        cfg.observations.upright = UprightObservationsCfg.PolicyCfg()
    cfg.observations.recover = RecoveryObservationsCfg.PolicyCfg()
    for group_name, scale in (
        ("policy", loco_spec["wheel_scale"]),
        (upright_spec["observation_group"], upright_spec["wheel_scale"]),
        ("recover", 0.8),
    ):
        group = getattr(cfg.observations, group_name)
        group.enable_corruption = False
        action_term = "actions" if group_name == "policy" else "last_action"
        previous = getattr(group, action_term)
        previous.func = last_action_for_skill
        previous.params = {"wheel_scale": scale}
    cfg.actions.joint_vel.class_type = HybridWheelVelocityAction
    depth_contract = None
    runtime_depth_contract = None
    if "depth_contract_id" in loco_spec:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import vision_depth_contract

        depth_contract = _load_depth_contract(args.loco_checkpoint, loco_spec["depth_contract_id"])
        runtime_contract = vision_depth_contract(cfg)
        if runtime_contract["contract_id"] != depth_contract["contract_id"]:
            raise ValueError("当前视觉环境预处理与 checkpoint 导出的深度合同不一致")
        for key in ("student_obs_dim", "teacher_obs_dim", "action_dim", "history_span_s"):
            if runtime_contract[key] != depth_contract[key]:
                raise ValueError(f"当前视觉环境的 {key} 与 checkpoint 合同不一致")
        # 短集成测试关闭随机噪声/时延/丢帧以获得可复现执行；历史拓扑、相机和
        # 展平顺序仍与 checkpoint 合同严格一致。鲁棒性必须由专用视觉评测验收。
        runtime_depth_contract = runtime_contract
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        models = {}
        models["loco"] = load_policy(task, args.loco_checkpoint, env, loco_spec, env.device)
        models["upright"] = load_policy(upright_spec["task"], args.upright_checkpoint, env, upright_spec, env.device)
        models["recover"] = load_policy(TASKS["recover"], args.recover_checkpoint, env, {
            "observation_group": "recover", "actor_obs_dim": 58, "critic_obs_dim": 70, "wheel_scale": 0.8,
        }, env.device)
        contract_checks = {
            "loco": _validate_checkpoint_training_contract(
                args.loco_checkpoint, loco_spec, cfg, models["loco"][2], visual_contract=runtime_depth_contract
            ),
            "upright": _validate_checkpoint_training_contract(
                args.upright_checkpoint, upright_spec, cfg, models["upright"][2]
            ),
            "recover": _validate_checkpoint_training_contract(
                args.recover_checkpoint,
                {"observation_group": "recover", "actor_obs_dim": 58, "critic_obs_dim": 70, "wheel_scale": 0.8},
                cfg, models["recover"][2],
            ),
        }
        for name, (network, _, policy_cfg) in models.items():
            export_validated_policy(network, policy_cfg, getattr(args, name + "_checkpoint"))
        machine = StanceStateMachine(strict_measurements=True)
        command = "lo"
        transitions = []
        strict_gate_events = []
        counts = {"LOCO": 0, "UPRIGHT": 0, "RECOVER": 0}
        action_peak = 0.0
        for step in range(args.hybrid_steps):
            if not app.is_running():
                break
            command = args.schedule.get(step, command)
            robot = env.scene["robot"]
            contact_sensor = env.scene.sensors.get("contact_forces")
            if contact_sensor is None:
                raise RuntimeError("hybrid 生产路径缺少 contact_forces，拒绝无接触状态切换")
            gravity = robot.data.projected_gravity_b[0]
            height = float(robot.data.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
            if (
                not torch.isfinite(robot.data.root_state_w).all()
                or not torch.isfinite(robot.data.joint_pos).all()
                or not torch.isfinite(robot.data.joint_vel).all()
            ):
                raise RuntimeError("机器人状态非有限，停止集成测试")
            contact, linear_speed, leg_speed = _strict_stance_measurements(
                robot, contact_sensor, contact_state_from_forces
            )
            mode = machine.select(
                command, float(gravity[2]), float(gravity[1]), height,
                g_x=float(gravity[0]), angular_speed=float(robot.data.root_ang_vel_b[0].norm()),
                four_wheel_support=contact["four_wheel"], rear_wheel_support=contact["rear_wheel"],
                linear_speed=linear_speed, leg_speed=leg_speed,
            )
            requested_mode = {"lo": "LOCO", "up": "UPRIGHT", "rec": "RECOVER"}[command]
            strict_gate_events.append({
                "step": step,
                "requested_mode": requested_mode,
                "entered_mode": mode,
                "measurement_gate": machine.last_measurement_diagnostics,
                "four_wheel_support": contact["four_wheel"],
                "rear_wheel_support": contact["rear_wheel"],
                "linear_speed_m_s": linear_speed,
                "active_leg_speed_rad_s": leg_speed,
            })
            if not transitions or transitions[-1]["mode"] != mode:
                transitions.append({
                    "step": step, "requested_command": command, "requested_mode": requested_mode,
                    "entered_mode": mode, "mode": mode,
                })
            name = mode.lower()
            network, clip, _ = models[name]
            group = loco_spec["observation_group"] if name == "loco" else (
                upright_spec["observation_group"] if name == "upright" else "recover"
            )
            inputs = TensorDict({"policy": obs[group]}, batch_size=[1])
            with torch.inference_mode():
                action = network.act_inference(inputs)
                if action.shape != (1, 12) or not torch.isfinite(action).all():
                    raise RuntimeError(f"{name} 输出动作无效：{action}")
                if clip is not None:
                    action = action.clamp(-clip, clip)
                env._hybrid_requested_wheel_scale = (
                    loco_spec["wheel_scale"] if name == "loco" else upright_spec["wheel_scale"] if name == "upright" else 0.8
                )
                obs, _, terminated, truncated, _ = env.step(action)
            if terminated.any() or truncated.any():
                raise RuntimeError("集成测试发生意外自动复位")
            counts[mode] += 1
            action_peak = max(action_peak, float(action.abs().max()))
        result = {
            "steps": sum(counts.values()), "requested_steps": args.hybrid_steps,
            "mode_steps": counts, "transitions": transitions, "strict_gate_events": strict_gate_events,
            "action_abs_max": action_peak,
            "seed": args.seed, "loco_policy": args.loco_policy, "upright_policy": args.upright_policy,
            "upright_motion_command": [args.upright_velocity, args.upright_yaw_rate] if args.upright_policy == "motion" else None,
            "depth_contract": depth_contract,
            "runtime_depth_contract": runtime_depth_contract,
            "depth_corruption_mode": "disabled_for_deterministic_integration" if depth_contract else None,
            "checkpoint_training_contracts": contract_checks,
            "scope": "single_robot_policy_execution; not skill_success_or_hardware_acceptance",
            "checkpoints": {},
        }
        for name in models:
            path = getattr(args, name + "_checkpoint")
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            result["checkpoints"][name] = {"path": str(path), "sha256": checksum}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result["steps"] == args.hybrid_steps
    finally:
        env.close()


if __name__ == "__main__":
    args = parse_args()
    app = AppLauncher(args).app
    status = 2
    try:
        status = 0 if main(args, app) else 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(status)
        app.close()
    raise SystemExit(status)
