# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 安全读取 Isaac YAML，迁移前校验完整观测处理与动作合同。
#   - v0.1.0 (2026-09-16): 严格迁移 42/47 维起立网络到 44/49 维两轮运动网络，新增指令列零初始化。

"""两轮运动的显式 warm start；不读取 checkpoint 文件，也不恢复 PPO 训练状态。

输入仅为已加载的 ``model_state_dict``，目标为新建的 RSL-RL ActorCritic。
源任务与目标任务必须保持前 42/47 项观测含义、尺度和关节/动作顺序不变；
state_dict 不能证明这些语义或源激活函数，调用方需核对源训练配置。
迁移只保留初始化时的函数，不证明源策略具备合格的起立、平衡或运动能力。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from torch import nn


_INPUT_CONTRACT = {"actor.0.weight": (42, 44), "critic.0.weight": (47, 49)}
_NETWORK_KEY = re.compile(r"(?:actor|critic)\.(?:0|[1-9][0-9]*)\.(?:weight|bias)\Z")
_ACTOR_TERMS = [
    "pitch_command", "projected_gravity", "base_lin_vel", "base_ang_vel",
    "joint_pos_legs", "joint_vel_legs", "wheel_vel", "last_action",
]
_PREFIXES = {"policy": _ACTOR_TERMS, "critic": _ACTOR_TERMS + ["priv_base_height", "priv_wheel_contact"]}


class _TrainingConfigLoader(yaml.SafeLoader):
    """仅接纳 Isaac 配置中的 tuple/slice 数据，绝不导入或执行 YAML 指定对象。"""


def _construct_slice(loader, node):
    values = loader.construct_sequence(node, deep=True)
    if len(values) != 3 or any(value is not None and type(value) is not int for value in values):
        raise ValueError("配置 slice 必须为三个整数或 None")
    if values[2] == 0:
        raise ValueError("配置 slice 步长不能为零")
    return slice(*values)


_TrainingConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True))
)
_TrainingConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_slice)


def load_training_config(path: str | Path) -> dict:
    """读取训练保存的配置；普通 SafeLoader 不识别其标签，unsafe_load 则会执行任意对象。"""
    with Path(path).open(encoding="utf-8") as stream:
        result = yaml.load(stream, Loader=_TrainingConfigLoader)
    if not isinstance(result, dict):
        raise ValueError("训练配置根节点必须为字典")
    return result


def validate_upright_motion_contract(source_env, target_env, source_agent, target_agent, target_active_terms) -> dict:
    """比较保存配置与实际目标；只放行末尾新增两列运动指令的迁移。

    函数完整限定名、params、clip/modifiers/noise、历史和动作语义均需一致。
    本检查证明配置一致，不证明不同源码版本的同名函数行为或物理技能一致。
    """
    def same(source, target, name):
        if source != target:
            raise ValueError(f"{name} 合同不匹配，拒绝迁移")

    try:
        for group_name, prefix in _PREFIXES.items():
            source = source_env["observations"][group_name]
            target = target_env["observations"][group_name]
            term_names = lambda group: [name for name, value in group.items() if isinstance(value, dict) and "func" in value]
            same(term_names(source), prefix, f"{group_name} 源观测顺序")
            same(term_names(target), prefix + ["motion_command"], f"{group_name} 目标观测顺序")
            same(list(target_active_terms[group_name]), prefix + ["motion_command"], f"{group_name} 实际启用顺序")
            for name in prefix:
                same(source[name], target[name], f"observations.{group_name}.{name}")
            source_options = {name: value for name, value in source.items() if name not in prefix}
            target_options = {name: value for name, value in target.items() if name not in prefix + ["motion_command"]}
            same(source_options, target_options, f"observations.{group_name} 组处理")
            same(target["motion_command"], {
                "func": "isaaclab.envs.mdp.observations:generated_commands",
                "params": {"command_name": "motion_command"}, "modifiers": None, "noise": None,
                "clip": None, "scale": None, "history_length": 0, "flatten_history_dim": True,
            }, f"observations.{group_name}.motion_command")

        same(list(source_env["actions"]), list(target_env["actions"]), "动作项顺序")
        same(source_env["actions"], target_env["actions"], "动作类型/关节顺序/尺度/偏置/裁剪")
        for name in ("joint_pos", "joint_vel"):
            same(source_env["scene"]["robot"]["init_state"][name],
                 target_env["scene"]["robot"]["init_state"][name], f"默认关节状态 {name}")
        same(source_env["sim"]["dt"], target_env["sim"]["dt"], "物理周期")
        same(source_env["decimation"], target_env["decimation"], "控制降采样")
        for name in ("clip_actions", "obs_groups", "empirical_normalization"):
            same(source_agent[name], target_agent[name], f"runner.{name}")
        for name in ("class_name", "activation", "noise_std_type", "actor_obs_normalization",
                     "critic_obs_normalization", "actor_hidden_dims", "critic_hidden_dims"):
            same(source_agent["policy"][name], target_agent["policy"][name], f"policy.{name}")
    except KeyError as error:
        raise ValueError(f"缺少迁移所需的训练配置字段: {error}") from error
    return {
        "configuration_contract_checked": True,
        "checked_contracts": ["full_observation_terms_and_order", "group_preprocessing", "actions",
                              "default_joint_state", "control_timing", "runner_preprocessing", "network_config"],
        "source_code_behavior_equivalence_verified": False,
    }


def _validate_target_network(policy: nn.Module) -> None:
    """只接受当前无归一化、非循环、状态无关 std 的 ELU MLP 合同。"""
    if not isinstance(policy, nn.Module):
        raise TypeError("target_policy 必须是 torch.nn.Module")
    if getattr(policy, "is_recurrent", False) or getattr(policy, "state_dependent_std", False):
        raise ValueError("warm start 不支持循环网络或 state-dependent std")
    for role, input_dim, output_dim in (("actor", 44, 12), ("critic", 49, 1)):
        if getattr(policy, f"{role}_obs_normalization", None) is not False:
            raise ValueError(f"{role} 观测归一化必须显式关闭")
        if type(getattr(policy, f"{role}_obs_normalizer", None)) is not nn.Identity:
            raise ValueError(f"{role} normalizer 必须是 Identity，不能丢弃已有归一化状态")
        network = getattr(policy, role, None)
        if not isinstance(network, nn.Sequential) or len(network) < 3 or len(network) % 2 != 1:
            raise ValueError(f"{role} 必须为 Linear/ELU 交替、以 Linear 结束的非空 MLP")
        # RSL MLP 复用同一个 ELU 对象，named_children 会去重；按 Sequential 的位置迭代。
        previous_width = input_dim
        for index, layer in enumerate(network):
            if index % 2 == 0:
                if type(layer) is not nn.Linear or layer.bias is None:
                    raise ValueError(f"{role}.{index} 必须是带 bias 的 Linear")
                if layer.in_features != previous_width:
                    raise ValueError(f"{role}.{index} 输入宽度与上一层输出不匹配")
                previous_width = layer.out_features
            elif type(layer) is not nn.ELU or layer.alpha != 1.0 or layer.inplace:
                raise ValueError(f"{role}.{index} 必须是 alpha=1、非原地的 ELU，与源任务一致")
        if network[0].in_features != input_dim or network[-1].out_features != output_dim:
            raise ValueError(f"{role} 目标网络必须是 {input_dim} → {output_dim}")


def warm_start_upright_motion(
    source_state_dict: Mapping[str, torch.Tensor],
    target_policy: nn.Module,
    *,
    source_actor_obs_normalization: bool = False,
    source_critic_obs_normalization: bool = False,
    source_activation: str = "elu",
) -> dict:
    """预检全部权重后迁移；源和目标均不允许未知键、错维或隐式 dtype 转换。

    actor 输入前 42 列、critic 输入前 47 列原样复制，末尾各两个速度指令列置零。
    所有其他参数（包括 std/log_std）必须完整、等形状、等 dtype 地复制。
    验证失败时不修改目标，也不修改源张量；设备可不同，但不会改动 optimizer/iteration。
    仅保证有限指令下数学函数相同，跨尺寸 GEMM 的浮点舍入不属于逐位一致承诺。
    """
    if not isinstance(source_state_dict, Mapping) or not source_state_dict:
        raise TypeError("source_state_dict 必须是非空模型参数映射，不是完整 runner checkpoint")
    if source_actor_obs_normalization is not False or source_critic_obs_normalization is not False:
        raise ValueError("源观测归一化必须关闭；当前迁移不能丢弃或重建 normalizer")
    if source_activation != "elu":
        raise ValueError("当前只支持已核实采用 ELU 的 upright 源任务")
    if any(not isinstance(key, str) for key in source_state_dict):
        raise TypeError("source_state_dict 的键必须全部为字符串")
    if any("normalizer" in key or "normalization" in key for key in source_state_dict):
        raise ValueError("源 checkpoint 含 normalizer 状态，不能按无归一化模型迁移")
    _validate_target_network(target_policy)
    target_state = target_policy.state_dict()
    unknown = sorted(key for key in target_state if key not in ("std", "log_std") and not _NETWORK_KEY.fullmatch(key))
    if unknown:
        raise ValueError(f"目标策略含不支持的参数键: {unknown}")
    missing = sorted(set(target_state) - set(source_state_dict))
    extra = sorted(set(source_state_dict) - set(target_state))
    if missing or extra:
        raise ValueError(f"源/目标参数键必须完全一致；missing={missing}, extra={extra}")
    noise_keys = [key for key in ("std", "log_std") if key in target_state]
    if len(noise_keys) != 1:
        raise ValueError("策略必须且只能包含一个状态无关的 std 或 log_std")
    for key in _INPUT_CONTRACT:
        if key not in target_state:
            raise ValueError(f"缺少当前 RSL ActorCritic 的输入层参数 {key}")

    prepared = {}
    for key, target in target_state.items():
        source = source_state_dict[key]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"{key} 必须是 Tensor；请仅传 model_state_dict")
        if source.layout != torch.strided or source.is_meta or target.is_meta:
            raise ValueError(f"{key} 只能使用已分配的稠密 Tensor")
        if not source.is_floating_point() or not torch.isfinite(source).all():
            raise ValueError(f"{key} 必须是有限浮点参数")
        if source.dtype != target.dtype:
            raise ValueError(f"{key} dtype 不匹配: {source.dtype} → {target.dtype}，拒绝隐式转换")
        if key in _INPUT_CONTRACT:
            old_dim, new_dim = _INPUT_CONTRACT[key]
            if (
                source.ndim != 2 or target.ndim != 2
                or source.shape[1] != old_dim or target.shape[1] != new_dim
                or source.shape[0] != target.shape[0]
            ):
                raise ValueError(f"{key} 必须保持输出宽度且只将输入 {old_dim} → {new_dim}")
            value = torch.zeros_like(target)
            value[:, :old_dim].copy_(source.detach().to(device=target.device))
        else:
            if source.shape != target.shape:
                raise ValueError(f"{key} 形状不匹配: {tuple(source.shape)} → {tuple(target.shape)}")
            value = source.detach().to(device=target.device).clone()
        if key in noise_keys:
            if source.shape != (12,):
                raise ValueError("动作 std/log_std 必须恰好对应 12 个动作")
            std = source if key == "std" else source.exp()
            if not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("动作标准差必须有限且严格为正")
        prepared[key] = value

    # 先验证、分配全部参数，再一次性进入修改阶段；不用 strict=False 吞掉任何缺失键。
    # state_dict 中 Tensor 与参数共享存储，copy_ 保留新 optimizer 指向的 Parameter 对象。
    with torch.no_grad():
        for key, value in prepared.items():
            target_state[key].copy_(value)

    return {
        "schema_version": 1,
        "migration": "upright_balance_getup_to_motion",
        "source_actor_obs_dim": 42,
        "source_critic_obs_dim": 47,
        "target_actor_obs_dim": 44,
        "target_critic_obs_dim": 49,
        "action_dim": 12,
        "initialization": "copy_existing_columns_append_two_zero_command_columns",
        "actor_new_columns": [42, 43],
        "critic_new_columns": [47, 48],
        "source_activation": source_activation,
        "source_activation_evidence": "caller_verified_training_config_not_encoded_in_state_dict",
        "required_observation_semantics": "unchanged_order_and_scale_of_existing_42_actor_and_47_critic_features",
        "copied_parameter_tensors": len(prepared),
        "action_std_parameter": noise_keys[0],
        "normalization_enabled": False,
        "normalizer_transferred": False,
        "optimizer_transferred": False,
        "iteration_transferred": False,
        "skill_success_verified": False,
    }
