#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-17
# @version v0.3.0
# @last_modified 2026-09-29
# @changelog
#   - v0.4.0 (2026-09-29): 阶段示范支持显式时钟策略，避免状态漂移使分段动作失配。
#   - v0.3.0 (2026-09-29): 增加阶段BC、切换加权和分段物理误差，保留原示范文件。
#   - v0.2.0 (2026-09-28): 支持分组动作尺度、镜像程序来源及新恢复控制合同传递。
#   - v0.1.0 (2026-09-17): 严格校验示范来源与控制语义，仅拟合 actor 并输出零迭代 BC 初始化。

"""成功脚本回合的受限离线行为克隆；拟合误差不是闭环恢复技能的证据。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from recovery_demonstrations import canonical_config, load_demonstrations
from upright_warm_start import load_training_config


ACTOR_DIM, CRITIC_DIM, ACTION_DIM = 58, 70, 12
CHECKPOINT_NAME = "behavior_cloning_initialization.pt"
OBSERVATION_DIMS = [3, 3, 8, 8, 4, 8, 8, 4, 12]
OBSERVATION_ORDER = [
    "projected_gravity", "base_ang_vel", "joint_pos_legs", "joint_vel_legs", "wheel_vel",
    "prev_joint_pos_legs", "prev_joint_vel_legs", "prev_wheel_vel", "last_action",
]
SUPPORTED_POLICY = {
    "class_name": "ActorCritic", "init_noise_std": 1.0, "noise_std_type": "scalar",
    "actor_obs_normalization": False, "critic_obs_normalization": False,
    "actor_hidden_dims": [128, 128, 128], "critic_hidden_dims": [128, 128, 128], "activation": "relu",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(actual, expected, name: str) -> None:
    if canonical_config(actual) != canonical_config(expected):
        raise ValueError(f"{name} 合同不匹配")


def _default_value(patterns: dict, joint: str) -> float:
    matches = [float(value) for pattern, value in patterns.items() if re.fullmatch(pattern, joint)]
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise ValueError(f"默认关节状态不能唯一解析: {joint}")
    return matches[0]


def _validate_action_vectors(contract: dict, source_env: dict) -> None:
    names: list[str] = []
    scales: list[float] = []
    offsets: list[float] = []
    for term_name, term in source_env["actions"].items():
        if term_name not in ("joint_pos", "joint_vel") or term.get("preserve_order") is not True:
            raise ValueError("仅支持有序 joint_pos/joint_vel 动作")
        joints = term["joint_names"]
        names.extend(joints)
        if isinstance(term["scale"], dict):
            scales.extend(_default_value(term["scale"], joint) for joint in joints)
        elif isinstance(term["scale"], (int, float)) and not isinstance(term["scale"], bool):
            scales.extend([term["scale"]] * len(joints))
        else:
            raise ValueError("动作项scale必须是数值或唯一匹配关节的映射")
        if term.get("use_default_offset") is True:
            defaults = source_env["scene"]["robot"]["init_state"][term_name]
            offsets.extend(_default_value(defaults, joint) for joint in joints)
        else:
            offsets.extend([term["offset"]] * len(joints))
    _same(contract["action_joint_names"], names, "动作关节顺序")
    for key, expected in (("action_scales", scales), ("action_offsets", offsets)):
        value = np.asarray(contract[key], dtype=np.float64)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"{key} 必须为 12 维有限值")
        if not np.allclose(value, expected, rtol=1e-6, atol=1e-7):
            raise ValueError(f"{key} 与源动作物理映射不匹配")


def validate_source_contract(metadata: dict, source_env: dict, source_agent: dict, checkpoint_sha: str) -> None:
    """维数相同不足以迁移；仅允许显式关闭观测 corruption 这一评测差异。"""
    if metadata.get("schema") != "rover_recovery_demonstrations_v1":
        raise ValueError("不支持的示范 schema")
    try:
        contract = metadata["contract"]
        _same(contract["actor_observation_dim"], ACTOR_DIM, "actor 维数")
        _same(contract["critic_observation_dim"], CRITIC_DIM, "critic 维数")
        _same(contract["action_dim"], ACTION_DIM, "动作维数")
        _same(source_agent["policy"], SUPPORTED_POLICY, "受支持源策略")
        _same(contract["policy_config"], source_agent["policy"], "策略配置")
        if source_agent.get("empirical_normalization") not in (None, False):
            raise ValueError("不支持 runner 观测归一化")
        if source_agent.get("obs_groups") not in ({}, {"policy": ["policy"], "critic": ["critic"]}):
            raise ValueError("不支持其他 runner 观测分组")
        _same(contract["wrapper_raw_clip"], source_agent["clip_actions"], "wrapper 原始动作裁剪")
        _same(contract["observation_order"], OBSERVATION_ORDER, "观测项顺序")
        _same(contract["observation_term_dims"], [[dim] for dim in OBSERVATION_DIMS], "逐项观测维数")
        observation_group = copy.deepcopy(source_env["observations"]["policy"])
        observation_group["enable_corruption"] = False
        _same(contract["observation_group"], observation_group, "观测函数/参数/尺度/历史")
        actual_terms = [name for name, value in observation_group.items()
                        if isinstance(value, dict) and "func" in value]
        _same(actual_terms, OBSERVATION_ORDER, "源配置观测顺序")
        _same(contract["action_config"], source_env["actions"], "动作配置")
        default = source_env["scene"]["robot"]["init_state"]
        _same(contract["default_joint_state"], {key: default[key] for key in ("joint_pos", "joint_vel")},
              "默认关节状态")
        expected_dt = source_env["sim"]["dt"] * source_env["decimation"]
        if not math.isclose(float(contract["step_dt"]), expected_dt, rel_tol=0, abs_tol=1e-10):
            raise ValueError("step_dt 与源控制周期不匹配")
        _validate_action_vectors(contract, source_env)
        source = metadata["source"]
        _same(source["checkpoint_sha256"], checkpoint_sha, "示范源 checkpoint")
        _same(source["action_source"], "segmented_joint_targets_with_active_pd", "示范动作来源")
        _same(source["evaluation_scope"], "bounded_action_program_search_not_policy_acceptance", "示范评测口径")
        _same(source["recovery_support_protocol"], "four_wheel_stable_v1", "原 Grounded 门")
        if "control_contract" in contract:
            from recovery_contract import control_contract, validate_control_contract

            validate_control_contract(control_contract(source_env, source_agent), contract["control_contract"])
            _same(source.get("control_contract"), contract["control_contract"], "评测/示范物理控制合同")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"示范/源配置缺少必要合同: {exc}") from exc


def validate_arrays(arrays: Mapping[str, np.ndarray], metadata: dict) -> None:
    """在训练边界再次拒绝坏数据；完整成功门由共享 NPZ loader 检查。"""
    actor_width = metadata["contract"]["actor_observation_dim"]
    if actor_width not in (58, 60):
        raise ValueError("仅支持原58维或显式转换的60维示范")
    for key, width in (("observations", actor_width), ("actions", ACTION_DIM)):
        value = arrays.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2:
            raise ValueError(f"{key} 必须为二维 float32 数组")
        if value.shape[1] != width or not np.isfinite(value).all():
            raise ValueError(f"{key} 维数不符或包含非有限值")
    count = len(arrays["observations"])
    if count == 0 or len(arrays["actions"]) != count:
        raise ValueError("示范不能为空且 observation/action 行数必须相同")
    for key in ("episode_id", "control_step"):
        value = arrays.get(key)
        if not isinstance(value, np.ndarray) or value.dtype != np.int64 or value.shape != (count,):
            raise ValueError(f"{key} 必须为 N 维 int64 数组")
    episodes = metadata.get("episodes", [])
    if not episodes or not np.array_equal(np.unique(arrays["episode_id"]), np.arange(len(episodes))):
        raise ValueError("episode_id 必须完整对应 episodes，禁止帧级随机划分")
    for index, episode in enumerate(episodes):
        if episode.get("trajectory_index") != index:
            raise ValueError("trajectory_index 必须与 episode_id 一致")
        clock = arrays["control_step"][arrays["episode_id"] == index]
        if not np.array_equal(clock, np.arange(250)):
            raise ValueError("示范必须包含完整连续的 250 步回合")


def _episode_keys(metadata: dict) -> tuple[set, set]:
    source = metadata["source"]
    source_keys, rollout_keys = set(), set()
    for episode in metadata["episodes"]:
        index = (episode["env_id"], episode["episode_index"])
        source_keys.add((metadata["source_evaluation_sha256"], *index))
        # 改写 JSON 或复制 NPZ 不应把同一 seed 的同一回合包装成新验证样本。
        rollout_keys.add((source["task"], source["seed"], source["checkpoint_sha256"], *index))
    if len(source_keys) != len(metadata["episodes"]):
        raise ValueError("数据集内部包含重复回合来源")
    return source_keys, rollout_keys


def validate_dataset_separation(train: dict, validation: dict | None, train_sha: str, validation_sha: str | None) -> dict:
    train_keys, train_rollouts = _episode_keys(train)
    def program_ids(metadata):
        identities = metadata["source"].get("action_program", {}).get("source_program_ids")
        return {identities[episode["env_id"]] if identities is not None else episode["program_id"]
                for episode in metadata["episodes"]}

    train_programs = program_ids(train)
    result = {"split_unit": "complete_source_episode", "random_frame_split": False,
              "validation_present": validation is not None, "statistically_independent": False,
              "train_program_ids": sorted(train_programs), "validation_program_ids": [], "program_overlap": [],
              "interpretation": "offline_fit_only_not_closed_loop_skill_success"}
    if validation is None:
        return result
    validation_keys, validation_rollouts = _episode_keys(validation)
    if train_sha == validation_sha or train_keys & validation_keys or train_rollouts & validation_rollouts:
        raise ValueError("训练/验证发生 dataset 或 episode 来源泄漏")
    validation_programs = program_ids(validation)
    result.update({"validation_program_ids": sorted(validation_programs),
                   "program_overlap": sorted(train_programs & validation_programs),
                   "source_disjoint": True, "train_seed": train["source"]["seed"],
                   "validation_seed": validation["source"]["seed"]})
    return result


def build_policy(source_agent: dict, device: str | torch.device, *, phase_conditioned: bool = False) -> nn.Module:
    """使用本机 RSL TensorDict 接口；不依赖 Isaac 仿真器。"""
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict

    config = copy.deepcopy(source_agent["policy"])
    _same(config, SUPPORTED_POLICY, "受支持源策略")
    config.pop("class_name")
    extra = 2 if phase_conditioned else 0
    observations = TensorDict({"policy": torch.zeros(1, ACTOR_DIM + extra), "critic": torch.zeros(1, CRITIC_DIM + extra)},
                              batch_size=[1])
    return ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, ACTION_DIM, **config).to(device)


def _validate_policy(policy: nn.Module) -> None:
    if getattr(policy, "is_recurrent", False) or getattr(policy, "state_dependent_std", False):
        raise ValueError("仅支持非循环且状态无关 std 的策略")
    actor = getattr(policy, "actor", None)
    if not isinstance(actor, nn.Sequential) or not len(actor) or not isinstance(actor[0], nn.Linear):
        raise ValueError("actor必须为受支持的MLP")
    actor_width = actor[0].in_features
    if actor_width not in (58, 60):
        raise ValueError("仅支持58/70或60/72观测")
    for role, width, output_width in (("actor", actor_width, ACTION_DIM), ("critic", actor_width + 12, 1)):
        if getattr(policy, f"{role}_obs_normalization", None) is not False:
            raise ValueError("不允许训练或丢弃 normalizer")
        if type(getattr(policy, f"{role}_obs_normalizer", None)) is not nn.Identity:
            raise ValueError("normalizer 必须为 Identity")
        network = getattr(policy, role, None)
        if not isinstance(network, nn.Sequential) or len(network) < 3 or len(network) % 2 != 1:
            raise ValueError(f"{role} 必须为 ReLU MLP")
        for index, layer in enumerate(network):
            if index % 2 == 0:
                if type(layer) is not nn.Linear or layer.bias is None or layer.in_features != width:
                    raise ValueError(f"{role} Linear 合同不匹配")
                width = layer.out_features
            elif type(layer) is not nn.ReLU or layer.inplace:
                raise ValueError(f"{role} 激活函数必须为非原地 ReLU")
        if width != output_width:
            raise ValueError(f"{role} 输出维数不匹配")
    state = policy.state_dict()
    if any(key != "std" and not re.fullmatch(r"(?:actor|critic)\.\d+\.(?:weight|bias)", key) for key in state):
        raise ValueError("策略包含不支持的参数或归一化状态")
    if "std" not in state or state["std"].shape != (ACTION_DIM,) or not (state["std"] > 0).all():
        raise ValueError("std 必须为 12 维正值")
    if any(not tensor.is_floating_point() or not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("模型包含非有限值或非浮点参数")


def load_source_policy(policy: nn.Module, checkpoint: dict) -> None:
    state = checkpoint.get("model_state_dict")
    target = policy.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(target):
        raise ValueError("源模型参数键与目标不一致")
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != target[key].shape
                or tensor.dtype != target[key].dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"源参数 {key} 形状/dtype 不符或包含非有限值")
    policy.load_state_dict(state, strict=True)
    _validate_policy(policy)


def transition_weights(arrays, metadata, *, weight=8., window=1):
    """按回合内时钟加权，包含启动、换段和归零前后，不跨reset连接相邻样本。"""
    if not math.isfinite(weight) or weight < 1 or type(window) is not int or not 0 <= window <= 5:
        raise ValueError("transition_weight必须有限且>=1，transition_window须为0到5的整数")
    program = metadata["source"].get("action_program", {})
    segment, active = program.get("segment_steps", 25), program.get("active_steps", 150)
    if type(segment) is not int or type(active) is not int or segment <= 0 or active <= 0 or active % segment:
        raise ValueError("动作程序分段时钟无效")
    steps = arrays["control_step"]
    startup = steps <= window
    transition = np.zeros(len(steps), dtype=bool)
    for boundary in range(segment, active + 1, segment):
        transition |= np.abs(steps - boundary) <= window
    transition &= ~startup
    groups = {"startup": startup, "transition_window": transition,
              "motion_interior": (steps < active) & ~startup & ~transition,
              "hold": (steps >= active) & ~startup & ~transition}
    return np.where(startup | transition, weight, 1.).astype(np.float32), groups


def _metrics(policy: nn.Module, arrays: dict, scale: torch.Tensor, batch_size: int,
             metadata: dict, transition_weight: float, transition_window: int) -> dict:
    total, per_action = 0.0, np.zeros(ACTION_DIM)
    per_episode: dict[int, list[float]] = {}
    errors, losses = [], []
    device = next(policy.parameters()).device
    with torch.no_grad():
        for start in range(0, len(arrays["observations"]), batch_size):
            end = start + batch_size
            observations = torch.as_tensor(arrays["observations"][start:end], device=device)
            target = torch.as_tensor(arrays["actions"][start:end], device=device)
            error = policy.actor(observations).double() - target.double()
            loss = ((error / scale.double()) ** 2).mean(dim=1)
            if not torch.isfinite(loss).all():
                raise ValueError("拟合指标包含非有限值")
            values = loss.cpu().numpy()
            errors.append(error.cpu().numpy())
            losses.append(values)
            total += float(values.sum())
            per_action += error.square().sum(dim=0).cpu().numpy()
            for index, value in zip(arrays["episode_id"][start:end], values, strict=True):
                per_episode.setdefault(int(index), []).append(float(value))
    count = len(arrays["observations"])
    errors, losses = np.concatenate(errors), np.concatenate(losses)
    weights, groups = transition_weights(arrays, metadata, weight=transition_weight, window=transition_window)
    physical = errors * np.asarray(metadata["contract"]["action_scales"])
    phase_metrics = {
        name: {"samples": int(mask.sum()), "normalized_mse": float(losses[mask].mean()),
               "physical_rmse_per_action": np.sqrt(np.square(physical[mask]).mean(axis=0)).tolist()}
        for name, mask in groups.items() if mask.any()
    }
    return {"normalized_mse": total / count, "raw_mse_per_action": (per_action / count).tolist(),
            "weighted_normalized_mse": float(np.average(losses, weights=weights)),
            "phase_metrics": phase_metrics, "physical_error_units": "pre_clip_rad_then_rad_s",
            "episode_normalized_mse": {str(key): float(np.mean(values)) for key, values in per_episode.items()}}


def fit_actor(policy: nn.Module, train: dict, train_metadata: dict, *, validation: dict | None = None,
              validation_metadata: dict | None = None, epochs: int = 400, learning_rate: float = 0.001,
              batch_size: int = 128, seed: int = 942,
              transition_weight: float = 8., transition_window: int = 1,
              phase_open_loop: bool = False) -> dict:
    """冻结 critic/std，训练结束保留最终 epoch；验证数据不参与尺度估计或模型选择。"""
    if epochs <= 0 or batch_size <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs/batch_size/learning_rate 必须为正有限值")
    validate_arrays(train, train_metadata)
    if validation is not None:
        if validation_metadata is None:
            raise ValueError("validation 必须附带完整元数据")
        validate_arrays(validation, validation_metadata)
    _validate_policy(policy)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(policy.parameters()).device
    observations = torch.as_tensor(train["observations"], device=device)
    if phase_open_loop:
        if observations.shape[1] != 60:
            raise ValueError("phase_open_loop 只允许60维阶段示范")
        # 脚本动作由阶段和镜像决定；训练时屏蔽58维瞬时状态，防止闭环微小漂移
        # 把策略带离分段动作。最后两列仍保留为可部署的阶段/初始倒向输入。
        observations = observations.clone()
        observations[:, :-2] = 0.0
    actions = torch.as_tensor(train["actions"], device=device)
    sample_weights, _ = transition_weights(train, train_metadata, weight=transition_weight, window=transition_window)
    # 使用全数据平均权重归一，避免每个minibatch随机分母改变加权目标。
    sample_weights = torch.as_tensor(sample_weights / sample_weights.mean(), device=device)
    scale = actions.double().square().mean(dim=0).sqrt().clamp_min(1.0).float()
    before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    flags = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
    was_training = policy.training
    policy.eval()
    history = []
    try:
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name.startswith("actor."))
        optimizer = torch.optim.Adam(policy.actor.parameters(), lr=learning_rate)
        def metrics(data, info):
            metric_data = data
            if phase_open_loop:
                metric_data = dict(data)
                metric_data["observations"] = data["observations"].copy()
                metric_data["observations"][:, :-2] = 0.0
            return _metrics(policy, metric_data, scale, batch_size, info, transition_weight, transition_window)

        initial_train = metrics(train, train_metadata)
        initial_validation = metrics(validation, validation_metadata) if validation is not None else None
        for epoch in range(epochs):
            permutation = torch.randperm(len(observations), generator=generator)
            for indices in permutation.split(batch_size):
                indices = indices.to(device)
                prediction = policy.actor(observations[indices])
                loss = (((prediction - actions[indices]) / scale).square().mean(dim=1) * sample_weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("BC loss 非有限，拒绝导出")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.actor.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
            score = metrics(train, train_metadata)["weighted_normalized_mse"]
            history.append(score)
            if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
                print(f"BC epoch {epoch + 1}/{epochs}: train weighted normalized MSE={score:.8g}", flush=True)
        final_train = metrics(train, train_metadata)
        final_validation = metrics(validation, validation_metadata) if validation is not None else None
        _validate_policy(policy)
        for key, tensor in policy.state_dict().items():
            if not key.startswith("actor.") and not torch.equal(tensor, before[key]):
                raise RuntimeError(f"非 actor 参数被修改: {key}")
        if phase_open_loop:
            first = policy.actor[0]
            if not isinstance(first, nn.Linear) or first.in_features != 60:
                raise ValueError("phase_open_loop 需要60维阶段Actor")
            # 保证上线时即使瞬时状态非零，也严格沿用只由阶段决定的函数。
            with torch.no_grad():
                first.weight[:, :-2] = 0.0
    finally:
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(flags[name])
        policy.train(was_training)
    return {"epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size, "seed": seed,
            "loss": "mean(normalized_sample_weight * mean(((prediction-target)/train_action_rms_clamped_min_1)^2))",
            "transition_weight": transition_weight, "transition_window": transition_window,
            "phase_open_loop": phase_open_loop,
            "action_loss_scales": scale.detach().cpu().tolist(), "scale_fit_source": "train_only",
            "selection": "final_epoch_no_validation_selection", "max_grad_norm": 10.0,
            "train_initial": initial_train, "train_final": final_train,
            "validation_initial": initial_validation, "validation_final": final_validation,
            "train_loss_by_epoch": history, "critic_unchanged": True, "std_unchanged": True,
            "normalizers_unchanged": True, "skill_success_verified": False}


def make_initialization_checkpoint(policy: nn.Module, lineage: dict, ppo_learning_rate: float) -> dict:
    """PPO 与 BC Adam 绝不混用；全参数顺序必须与 PPO(policy.parameters()) 一致。"""
    _validate_policy(policy)
    if not math.isfinite(ppo_learning_rate) or ppo_learning_rate <= 0:
        raise ValueError("PPO learning_rate 必须为正有限值")
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_learning_rate)
    return {"model_state_dict": {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "iter": 0,
            "infos": {"initialization_type": "offline_actor_behavior_cloning_not_ppo",
                      "ppo_updates": 0, "skill_success_verified": False, "bc_lineage": lineage}}


def _dataset_lineage(path: Path, digest: str, metadata: dict, count: int) -> dict:
    return {"path": str(path.resolve()), "sha256": digest, "samples": count,
            "source_evaluation_path": metadata["source_evaluation_path"],
            "source_evaluation_sha256": metadata["source_evaluation_sha256"],
            "seed": metadata["source"]["seed"], "episodes": metadata["episodes"]}


def run_pretraining(args: argparse.Namespace) -> Path:
    source_path, train_path, output = Path(args.source_checkpoint), Path(args.train_data), Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output_dir 必须独占新建，不能覆盖: {output}")
    params = source_path.parent / "params"
    source_agent, source_env = load_training_config(params / "agent.yaml"), load_training_config(params / "env.yaml")
    source_sha = sha256_file(source_path)
    train_sha = sha256_file(train_path)
    train, train_metadata = load_demonstrations(train_path)
    validate_source_contract(train_metadata, source_env, source_agent, source_sha)
    validate_arrays(train, train_metadata)
    validation = validation_metadata = validation_sha = validation_path = None
    if args.validation_data:
        validation_path = Path(args.validation_data)
        validation_sha = sha256_file(validation_path)
        validation, validation_metadata = load_demonstrations(validation_path)
        validate_source_contract(validation_metadata, source_env, source_agent, source_sha)
        validate_arrays(validation, validation_metadata)
        _same(validation_metadata["contract"], train_metadata["contract"], "训练/验证合同")
    separation = validate_dataset_separation(train_metadata, validation_metadata, train_sha, validation_sha)
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    torch.manual_seed(args.seed)
    policy = build_policy(source_agent, args.device)
    load_source_policy(policy, source)
    phase_conditioned = getattr(args, "phase_conditioned", False)
    output_env, output_agent = source_env, source_agent
    if phase_conditioned:
        from recovery_phase_training import phase_configs, phase_demonstrations, phase_policy_state

        output_env, output_agent = phase_configs(source_env, source_agent)
        train, train_metadata = phase_demonstrations(train, train_metadata)
        if validation is not None:
            validation, validation_metadata = phase_demonstrations(validation, validation_metadata)
        policy = build_policy(output_agent, args.device, phase_conditioned=True)
        load_source_policy(policy, {"model_state_dict": phase_policy_state(source["model_state_dict"])})
    transition_weight = getattr(args, "transition_weight", 8.)
    transition_window = getattr(args, "transition_window", 1)
    phase_open_loop = bool(getattr(args, "phase_open_loop", False))
    if phase_open_loop and not phase_conditioned:
        raise ValueError("phase_open_loop 必须与 --phase_conditioned 一起使用")
    transition_weights(train, train_metadata, weight=transition_weight, window=transition_window)
    # mkdir(exist_ok=False) 是独占声明；此前失败不创建目录，此后失败留下可诊断但不可覆盖的目录。
    output.mkdir(parents=True, exist_ok=False)
    saved_params = output / "params"
    saved_params.mkdir()
    for name, config in (("env.yaml", output_env), ("agent.yaml", output_agent)):
        if phase_conditioned:
            with (saved_params / name).open("x", encoding="utf-8") as stream:
                yaml.dump(config, stream, sort_keys=False)
        else:
            shutil.copyfile(params / name, saved_params / name)
    for name in ("recovery_control_contract.json", "recovery_motor_profile.json", "recovery_physics_audit.json"):
        if phase_conditioned and name == "recovery_control_contract.json":
            continue
        if (params / name).is_file():
            shutil.copyfile(params / name, saved_params / name)
    if phase_conditioned:
        from recovery_contract import control_contract, save_control_contract, validate_control_contract

        target_contract = control_contract(output_env, output_agent)
        validate_control_contract(train_metadata["contract"]["control_contract"], target_contract)
        save_control_contract(saved_params, target_contract)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        metrics = fit_actor(policy, train, train_metadata, validation=validation,
                            validation_metadata=validation_metadata, epochs=args.epochs,
                            learning_rate=args.learning_rate, batch_size=args.batch_size, seed=args.seed,
                            transition_weight=transition_weight, transition_window=transition_window,
                            phase_open_loop=phase_open_loop)
    finally:
        torch.use_deterministic_algorithms(deterministic_before, warn_only=warn_only_before)
    validation_lineage = None
    if validation is not None:
        assert validation_path is not None and validation_sha is not None and validation_metadata is not None
        validation_lineage = _dataset_lineage(validation_path, validation_sha, validation_metadata,
                                             len(validation["actions"]))
    lineage = {
        "schema": "rover_recovery_bc_initialization_v1", "initialization_type": "offline_actor_behavior_cloning",
        "source_checkpoint": str(source_path.resolve()), "source_checkpoint_sha256": source_sha,
        "phase_conditioned": phase_conditioned,
        "phase_open_loop": phase_open_loop,
        "derived_features": train_metadata.get("derived_features"),
        "source_iteration": source.get("iter"), "iteration_reset_to": 0, "ppo_updates": 0,
        "old_critic_retained": True, "std_trained": False, "normalizers_trained": False,
        "source_optimizer_retained": False, "bc_optimizer_retained": False, "optimizer": "fresh_all_policy_Adam",
        "optimizer_state_empty": True, "skill_success_verified": False,
        "source_env_sha256": sha256_file(params / "env.yaml"),
        "source_agent_sha256": sha256_file(params / "agent.yaml"),
        "contract": train_metadata["contract"],
        "allowed_source_difference": "explicit_phase_append" if phase_conditioned else "policy.enable_corruption=False",
        "train_data": _dataset_lineage(train_path, train_sha, train_metadata, len(train["actions"])),
        "validation_data": validation_lineage,
        "separation": separation, "fit": metrics, "device": str(args.device),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "deterministic_algorithms": True,
        "limitations": ["离线误差仅证明脚本动作拟合，不代表闭环恢复技能成功。",
                        "同一程序在不同 seed 的成功轨迹相关；来源分离不等于统计独立泛化。",
                        "critic 与 std 保留源参数，尚未针对 BC actor 执行 PPO 更新。"],
    }
    checkpoint = make_initialization_checkpoint(policy, lineage, output_agent["algorithm"]["learning_rate"])
    checkpoint_path = output / CHECKPOINT_NAME
    torch.save(checkpoint, checkpoint_path)
    lineage["output_checkpoint_sha256"] = sha256_file(checkpoint_path)
    with (saved_params / "bc_lineage.json").open("x", encoding="utf-8") as stream:
        json.dump(canonical_config(lineage), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--validation_data")
    parser.add_argument("--output_dir", required=True, help="必须是尚不存在的新目录")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase_conditioned", action="store_true",
                        help="从原Side示范显式转换到60/72维Side-Phase任务，不修改源NPZ。")
    parser.add_argument("--transition_weight", type=float, default=8., help="启动/切换附近样本的损失权重，1为等权。")
    parser.add_argument("--transition_window", type=int, default=1, help="切换前后控制步数，0到5。")
    parser.add_argument("--phase_open_loop", action="store_true",
                        help="阶段示范只用阶段/初始倒向生成脚本动作，屏蔽瞬时状态漂移；仅用于程序化示范初始化。")
    args = parser.parse_args()
    print(f"Saved BC initialization (not PPO or skill acceptance): {run_pretraining(args)}")


if __name__ == "__main__":
    main()
