"""显式将 blind45 actor 初始化为 teacher232；不恢复优化器、迭代或训练成绩。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 独立保存首次PPO更新前模型，隔离迁移与优化退化。
    - v0.1.0 (2026-09-17): 严格迁移兼容 MLP，新增地形输入零初始化，保留 235 维 critic。
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
from pathlib import Path
import re

import torch
from torch import nn

from upright_warm_start import load_training_config


TEACHER_TASK = "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-v0"
_NETWORK_KEY = re.compile(r"(?:actor|critic)\.(?:0|[1-9][0-9]*)\.(?:weight|bias)\Z")
_PREFIX = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
_CRITIC = ["base_lin_vel"] + _PREFIX + ["height_scan"]


def save_teacher_initialization(runner, log_dir) -> dict:
    """保存可单独评测的初始化快照，不调用会触发远程logger的runner.save。

    名称不匹配默认model_*.pt，防止自动续训误选零更新权重；独占创建保留旧证据。
    与当前RSL ActorCritic的checkpoint字段一致，但不声明具有视觉能力。
    """
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise ValueError("teacher初始化快照必须在首次PPO更新之前保存")
    scope = "initialization_before_ppo_not_skill_acceptance"
    payload = {"model_state_dict": runner.alg.policy.state_dict(),
               "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 0,
               "infos": {"scope": scope}}
    path = Path(log_dir).resolve() / "teacher_initialization.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"checkpoint": str(path), "sha256": digest, "optimizer_updates": 0, "scope": scope}


def validate_vision_warm_start_request(args, agent) -> bool:
    """只在显式新 teacher PPO 时启用；CLI 与 Hydra 的续训入口均不得混入。"""
    if args.vision_teacher_warm_start is None:
        return False
    if (args.task != TEACHER_TASK or agent["resume"] or agent["class_name"] != "OnPolicyRunner"
            or agent["algorithm"]["class_name"] != "PPO"):
        raise ValueError("vision_teacher_warm_start 只允许新 teacher PPO，不能用于 resume 或 student")
    if any(getattr(args, name, None) is not None for name in (
        "upright_warm_start", "checkpoint", "load_run", "resume_noise_scale", "resume_learning_rate",
    )) or agent["load_checkpoint"] != "model_.*.pt" or agent["load_run"] != ".*":
        raise ValueError("vision_teacher_warm_start 不能与其他 warm start、checkpoint、load_run 或 resume 覆盖混用")
    return True


def initialize_teacher_from_blind(checkpoint_path, target_policy, *, target_env, target_agent,
                                  active_terms, term_dims, task: str, seed: int) -> dict:
    """安全读取本地旧配置/纯权重，记录模型及 YAML 字节哈希，不调用 runner.load。"""
    if task != TEACHER_TASK or type(seed) is not int:
        raise ValueError("视觉 warm start 需要明确 teacher task 和整数 seed")
    if target_agent.get("resume") is not False:
        raise ValueError("视觉 warm start 不得恢复旧训练状态")
    path = Path(checkpoint_path).expanduser().resolve()
    source_params = path.parent / "params"
    env_path, agent_path = source_params / "env.yaml", source_params / "agent.yaml"
    source_env = load_training_config(env_path)
    source_agent = load_training_config(agent_path)
    metadata = validate_blind_teacher_contract(
        source_env, target_env, source_agent, target_agent, active_terms, term_dims,
    )
    # 白名单 YAML loader 只接受 tuple/slice 数据；checkpoint 只用 weights_only，绝不回退 pickle 执行。
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    iteration = checkpoint.get("iter") if isinstance(checkpoint, dict) else None
    if type(iteration) is not int or iteration < 0 or "model_state_dict" not in checkpoint:
        raise ValueError("源 checkpoint 缺少有效 model_state_dict/非负整数 iter")
    metadata.update({
        "load_mode": "vision_teacher_warm_start", "task": task, "seed": seed,
        "source_checkpoint": str(path), "source_checkpoint_sha256": digest,
        "source_checkpoint_size_bytes": path.stat().st_size, "source_iteration": iteration,
        "source_env_config": str(env_path), "source_env_sha256": hashlib.sha256(env_path.read_bytes()).hexdigest(),
        "source_agent_config": str(agent_path), "source_agent_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
        "source_experiment_name": source_agent.get("experiment_name"), "target_initial_iteration": 0,
    })
    metadata.update(warm_start_vision_teacher(
        checkpoint["model_state_dict"], target_policy, source_policy_cfg=source_agent["policy"],
    ))
    return metadata


def validate_blind_teacher_contract(source_env, target_env, source_agent, target_agent, active_terms, term_dims) -> dict:
    """旧 actor 是严格 45 维前缀，critic235 整组保持；实际 manager 顺序/维度也必须匹配。"""
    def same(left, right, name):
        if left != right:
            raise ValueError(f"{name} 合同不匹配，拒绝视觉迁移")

    def terms(group):
        return [name for name, value in group.items() if isinstance(value, dict) and "func" in value]

    try:
        for name, source_order, target_order in (("policy", _PREFIX, _PREFIX + ["height_scan"]),
                                                  ("critic", _CRITIC, _CRITIC)):
            source, target = source_env["observations"][name], target_env["observations"][name]
            same(terms(source), source_order, f"{name} 源顺序")
            same(terms(target), target_order, f"{name} 目标顺序")
            same(list(active_terms[name]), target_order, f"{name} 实际启用顺序")
            for term in source_order:
                same(source[term], target[term], f"observations.{name}.{term}")
            # policy 唯一新增项为原来 None 的 height_scan；不能连带改变组处理。
            excluded = set(target_order)
            same({key: value for key, value in source.items() if key not in excluded},
                 {key: value for key, value in target.items() if key not in excluded}, f"{name} 组预处理")
            expected_dims = ([3] if name == "critic" else []) + [3, 3, 3, 12, 12, 12, 187]
            same([tuple(value) for value in term_dims[name]], [(width,) for width in expected_dims], f"{name} 实际维度")
        height = copy.deepcopy(source_env["observations"]["critic"]["height_scan"])
        height["noise"] = {"func": "isaaclab.utils.noise.noise_model:uniform_noise", "operation": "add",
                           "n_min": -0.1, "n_max": 0.1}
        same(target_env["observations"]["policy"]["height_scan"], height, "teacher 新增 height_scan")
        same(source_env["scene"]["height_scanner"], target_env["scene"]["height_scanner"], "扫描坐标/顺序/尺度")
        same(list(source_env["actions"]), list(target_env["actions"]), "动作项顺序")
        same(source_env["actions"], target_env["actions"], "动作类型/顺序/尺度/偏置/裁剪")
        for name in ("joint_pos", "joint_vel"):
            same(source_env["scene"]["robot"]["init_state"][name],
                 target_env["scene"]["robot"]["init_state"][name], f"默认关节 {name}")
        same(source_env["sim"]["dt"], target_env["sim"]["dt"], "物理周期")
        same(source_env["decimation"], target_env["decimation"], "控制降采样")
        for agent, label in ((source_agent, "source"), (target_agent, "target")):
            same(agent["class_name"], "OnPolicyRunner", f"{label} runner")
            same(agent["algorithm"]["class_name"], "PPO", f"{label} algorithm")
            if agent["empirical_normalization"] not in (None, False):
                raise ValueError("empirical_normalization 必须明确为 None 或 False")
            if agent["obs_groups"] not in ({}, {"policy": ["policy"], "critic": ["critic"]}):
                raise ValueError("只支持明确的 policy/critic 路由或 RSL 默认路由")
            for role in ("actor", "critic"):
                same(agent["policy"][f"{role}_obs_normalization"], False, f"{label}.{role} 归一化")
        for name in ("clip_actions", "empirical_normalization"):
            same(source_agent[name], target_agent[name], f"runner.{name}")
        for name in ("class_name", "activation", "noise_std_type", "actor_obs_normalization",
                     "critic_obs_normalization", "actor_hidden_dims", "critic_hidden_dims"):
            same(source_agent["policy"][name], target_agent["policy"][name], f"policy.{name}")
    except (KeyError, TypeError) as error:
        raise ValueError(f"缺少或损坏视觉迁移配置字段: {error}") from error
    return {
        "configuration_contract_checked": True,
        "checked_contracts": ["full_actor_prefix45", "full_critic235", "actual_term_order_and_dimensions",
                              "group_preprocessing", "height_scanner", "actions", "default_joint_state",
                              "control_timing", "runner_routing_and_normalization", "network_config"],
        "source_code_behavior_equivalence_verified": False,
        "environment_distribution_equivalence_verified": False,
        "environment_change_notice": "terrain/friction/rewards may differ; warm start is initialization, not resume or skill acceptance",
    }


def _validate_network(target, source_policy_cfg):
    """配置限定激活/归一化，运行时结构防止同维循环网络或隐式 normalizer 混入。"""
    if not isinstance(target, nn.Module):
        raise TypeError("target_policy 必须是 torch.nn.Module")
    if source_policy_cfg.get("class_name") != "ActorCritic" or source_policy_cfg.get("activation") != "elu":
        raise ValueError("只支持已核实的 ActorCritic ELU 源策略")
    if source_policy_cfg.get("state_dependent_std", False) or getattr(target, "state_dependent_std", False):
        raise ValueError("不支持 state-dependent std")
    if getattr(target, "is_recurrent", False):
        raise ValueError("不支持循环网络")
    if getattr(target, "obs_groups", None) != {"policy": ["policy"], "critic": ["critic"]}:
        raise ValueError("目标网络必须实际读取 policy 和 critic 两组")
    for role, input_dim, output_dim in (("actor", 232, 12), ("critic", 235, 1)):
        if (source_policy_cfg.get(f"{role}_obs_normalization") is not False
                or getattr(target, f"{role}_obs_normalization", None) is not False
                or type(getattr(target, f"{role}_obs_normalizer", None)) is not nn.Identity):
            raise ValueError("源与目标的观测归一化必须显式关闭且 normalizer 为 Identity")
        network = getattr(target, role, None)
        if not isinstance(network, nn.Sequential) or len(network) < 3 or len(network) % 2 != 1:
            raise ValueError(f"{role} 必须为 Linear/ELU 交替且以 Linear 结束的 MLP")
        width, hidden = input_dim, []
        for index, layer in enumerate(network):
            if index % 2 == 0:
                if type(layer) is not nn.Linear or layer.bias is None or layer.in_features != width:
                    raise ValueError(f"{role}.{index} Linear 输入或 bias 不匹配")
                width = layer.out_features
                hidden.append(width)
            elif type(layer) is not nn.ELU or layer.alpha != 1.0 or layer.inplace:
                raise ValueError("ELU 必须为 alpha=1 且非原地")
        if width != output_dim or hidden[:-1] != source_policy_cfg.get(f"{role}_hidden_dims"):
            raise ValueError(f"{role} 隐藏层或输出维度不匹配")


def warm_start_vision_teacher(source_state_dict, target_policy, *, source_policy_cfg) -> dict:
    """预检后原位复制，actor 新增 187 列置零，critic235 完整保留。

    只在相同有限输入下保持原网络数学函数；GEMM 扩维可产生浮点舍入差异。
    不声明更换地形/摩擦/奖励后仍保留成功率，也不向 student 迁移。
    """
    _validate_network(target_policy, source_policy_cfg)
    if not isinstance(source_state_dict, Mapping) or not source_state_dict:
        raise TypeError("source_state_dict 必须是非空参数映射")
    if any(not isinstance(key, str) for key in source_state_dict):
        raise TypeError("参数键必须为字符串")
    target_state = target_policy.state_dict()
    if set(source_state_dict) != set(target_state):
        raise ValueError("源/目标参数键必须完全一致，不能丢弃 normalizer 或未知状态")
    if any(key not in ("std", "log_std") and not _NETWORK_KEY.fullmatch(key) for key in target_state):
        raise ValueError("策略含未知参数，拒绝部分加载")
    noise_keys = [key for key in ("std", "log_std") if key in target_state]
    expected_noise = {"scalar": "std", "log": "log_std"}.get(source_policy_cfg.get("noise_std_type"))
    if noise_keys != [expected_noise]:
        raise ValueError("源/目标必须具有相同且唯一的状态无关 std/log_std")
    prepared = {}
    for name, target in target_state.items():
        source = source_state_dict[name]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"{name} 必须是 Tensor")
        if source.layout != torch.strided or source.is_meta or target.is_meta:
            raise ValueError(f"{name} 必须是已分配的稠密张量")
        if not source.is_floating_point() or not torch.isfinite(source).all() or source.dtype != target.dtype:
            raise ValueError(f"{name} 必须是等 dtype 的有限浮点参数")
        if name == "actor.0.weight":
            if source.shape != (target.shape[0], 45) or target.shape[1] != 232:
                raise ValueError("actor 输入必须严格为 45 → 232")
            value = torch.zeros_like(target)
            value[:, :45].copy_(source.detach().to(target.device))
        else:
            if source.shape != target.shape:
                raise ValueError(f"{name} 形状不匹配")
            value = source.detach().to(target.device).clone()
        if name in noise_keys:
            std = source if name == "std" else source.exp()
            if source.shape != (12,) or not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("动作标准差必须为 12 维有限正值")
        prepared[name] = value
    # 完整验证和分配后才修改，保留 Parameter 对象供全新 optimizer 使用。
    with torch.no_grad():
        for name, value in prepared.items():
            target_state[name].copy_(value)
    return {
        "schema_version": 1, "migration": "blind45_to_vision_teacher232",
        "source_actor_obs_dim": 45, "target_actor_obs_dim": 232,
        "source_critic_obs_dim": 235, "target_critic_obs_dim": 235, "action_dim": 12,
        "initialization": "copy_actor_prefix45_zero_height187_copy_critic235",
        "actor_new_columns_half_open": [45, 232], "critic_initialization": "copy_all_compatible_parameters",
        "copied_parameter_tensors": len(prepared), "action_std_parameter": noise_keys[0],
        "normalizer_transferred": False, "optimizer_transferred": False, "iteration_transferred": False,
        "skill_success_verified": False,
    }
