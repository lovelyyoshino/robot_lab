"""恢复阶段的安全 PPO 接管：critic 预热并持续保留脚本动作锚点。

该模块只包装已经构建的 RSL-RL runner，不修改安装目录中的 rsl-rl-lib。
阶段示范仍从原58维 NPZ显式扩展为60维；PPO闭环成功率必须另行评测。

@author pony
@date 2026-09-29
@version v0.2.0
@last_modified 2026-09-29
@changelog
    - v0.2.0 (2026-09-29): 提高换段锚点保护强度，并记录其独立训练合同。
    - v0.1.0 (2026-09-29): 增加critic预热、动作锚点和训练来源合同校验。
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from recovery_demonstrations import load_demonstrations
from recovery_phase_training import PHASE_TASK, phase_demonstrations


def _state(path: Path) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("infos"), dict):
        raise ValueError("受保护恢复训练需要带infos的BC初始化checkpoint")
    kind = loaded["infos"].get("initialization_type", "")
    if not kind.startswith("offline_actor_behavior_cloning") and "recovery_protection" not in loaded["infos"]:
        raise ValueError("受保护恢复训练只接受阶段BC或带完整保护状态的PPO模型")
    return loaded


def _load_anchor(data_path: Path, env: dict, agent: dict):
    """校验动作合同并只返回阶段观测/动作；不把验证集混入在线锚点。"""
    arrays, metadata = load_demonstrations(data_path)
    arrays, metadata = phase_demonstrations(arrays, metadata)
    from recovery_contract import control_contract, validate_control_contract

    target = control_contract(env, agent)
    validate_control_contract(metadata["contract"]["control_contract"], target)
    if arrays["observations"].shape[1] != 60 or arrays["actions"].shape[1] != 12:
        raise ValueError("阶段锚点必须为60维观测/12维动作")
    return arrays, metadata


def attach_recovery_protection(
    runner,
    *,
    task: str,
    demo_path: str | Path,
    env: dict,
    agent: dict,
    source_checkpoint: str | Path,
    warmup_iterations: int = 10,
    anchor_coef: float = 0.2,
    anchor_batch_size: int = 2048,
    anchor_transition_weight: float = 20.,
    anchor_transition_window: int = 2,
    seed: int = 942,
):
    """将保护逻辑挂到runner.learn使用的update/save边界上。"""
    if task != PHASE_TASK:
        raise ValueError("受保护恢复训练只允许60/72维Side-Phase任务")
    if type(warmup_iterations) is not int or warmup_iterations < 0:
        raise ValueError("recovery_warmup_iterations必须为非负整数")
    if not np.isfinite(anchor_coef) or anchor_coef < 0:
        raise ValueError("recovery_anchor_coef必须为有限非负数")
    if type(anchor_batch_size) is not int or anchor_batch_size < 1:
        raise ValueError("recovery_anchor_batch_size必须为正整数")
    if not np.isfinite(anchor_transition_weight) or anchor_transition_weight < 1:
        raise ValueError("recovery_anchor_transition_weight必须有限且>=1")
    if type(anchor_transition_window) is not int or not 0 <= anchor_transition_window <= 5:
        raise ValueError("recovery_anchor_transition_window必须为0到5的整数")
    checkpoint = _state(Path(source_checkpoint).expanduser().resolve())
    previous = checkpoint["infos"].get("recovery_protection")
    if previous is None and (checkpoint["infos"].get("ppo_updates") != 0
                             or not checkpoint["infos"].get("bc_lineage", {}).get("phase_conditioned")):
        raise ValueError("初始化必须是显式扩维且零PPO更新的阶段BC")
    if runner.alg.policy.is_recurrent or runner.alg.is_multi_gpu or runner.alg.rnd or runner.alg.symmetry:
        raise ValueError("恢复保护只支持单设备、无RND/对称增强的前馈PPO")
    if runner.alg.schedule != "fixed" or runner.alg.entropy_coef != 0:
        raise ValueError("恢复保护需要固定学习率且entropy_coef=0")
    arrays, metadata = _load_anchor(Path(demo_path).expanduser().resolve(), env, agent)
    observations = torch.as_tensor(arrays["observations"], device=runner.device)
    actions = torch.as_tensor(arrays["actions"], device=runner.device)
    scale = actions.square().mean(dim=0).sqrt().clamp_min(1.0)
    from pretrain_recovery_demonstrations import transition_weights

    weights, _ = transition_weights(
        arrays, metadata, weight=anchor_transition_weight, window=anchor_transition_window,
    )
    weights = torch.as_tensor(weights / weights.mean(), device=runner.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # 独立Adam避免锚点梯度污染PPO动量；coef缩放学习率而非对Adam无效的统一loss缩放。
    anchor_optimizer = torch.optim.Adam(runner.alg.policy.actor.parameters(), lr=runner.alg.learning_rate * anchor_coef)
    original_update = runner.alg.update
    original_save = runner.save
    state = {
        "schema": "recovery_protection_v1", "warmup_iterations": warmup_iterations,
        "anchor_coef": float(anchor_coef), "anchor_batch_size": anchor_batch_size,
        "update_count": 0, "actor_unfrozen": warmup_iterations == 0,
        "demo_path": str(Path(demo_path).expanduser().resolve()),
        "demo_sha256": hashlib.sha256(Path(demo_path).read_bytes()).hexdigest(),
        "demo_samples": int(len(actions)), "skill_success_verified": False,
        "anchor_learning_rate": runner.alg.learning_rate * anchor_coef,
        "transition_weight": float(anchor_transition_weight), "transition_window": anchor_transition_window,
        "noise_std_physical": [0.02] * 4 + [0.05] * 4 + [0.5] * 4,
        "noise_mode": "fixed_gaussian_std_not_bounded_samples",
    }
    if previous is None:
        train_hash = checkpoint["infos"]["bc_lineage"]["train_data"]["sha256"]
        if state["demo_sha256"] != train_hash:
            raise ValueError("锚点数据必须是BC的训练集，不能换成验证集")
        physical_scale = torch.as_tensor(metadata["contract"]["action_scales"], device=runner.device)
        with torch.no_grad():
            runner.alg.policy.std.copy_(torch.tensor(state["noise_std_physical"], device=runner.device) / physical_scale.abs())
        runner.alg.optimizer.state.pop(runner.alg.policy.std, None)
    else:
        for key in ("schema", "warmup_iterations", "anchor_coef", "anchor_batch_size", "demo_sha256",
                    "anchor_learning_rate", "transition_weight", "transition_window", "noise_std_physical", "noise_mode"):
            if previous.get(key) != state[key]:
                raise ValueError(f"续训保护参数/来源不匹配: {key}")
        if type(previous.get("update_count")) is not int or previous["update_count"] < 0:
            raise ValueError("保护更新计数损坏")
        state["update_count"] = previous["update_count"]
        generator.set_state(previous["sampler_rng_state"].cpu())
        anchor_optimizer.load_state_dict(previous["anchor_optimizer_state"])
        # RSL默认iter指向最后已完成更新；恢复下一轮编号，避免重复model_N标签。
        runner.current_learning_iteration = state["update_count"]

    def set_actor_trainable(enabled: bool):
        for parameter in runner.alg.policy.actor.parameters():
            parameter.requires_grad_(enabled)
        # 固定小噪声不等于动作有硬界；仍由原action term执行目标裁剪。
        runner.alg.policy.std.requires_grad_(False)
        runner.alg.policy.std.grad = None

    state["actor_unfrozen"] = state["update_count"] >= warmup_iterations
    set_actor_trainable(state["actor_unfrozen"])

    @functools.wraps(original_update)
    def protected_update():
        if state["update_count"] == warmup_iterations:
            set_actor_trainable(True)
            state["actor_unfrozen"] = True
        result = original_update()
        if state["actor_unfrozen"] and anchor_coef > 0:
            count = min(anchor_batch_size, len(observations))
            indices = torch.randperm(len(observations), generator=generator, device="cpu")[:count].to(runner.device)
            prediction = runner.alg.policy.actor(observations[indices])
            loss = ((((prediction - actions[indices]) / scale).square().mean(dim=1)) * weights[indices]).mean()
            if not torch.isfinite(loss):
                raise ValueError("恢复示范锚点损失非有限")
            anchor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.alg.policy.actor.parameters(), 10.0, error_if_nonfinite=True)
            anchor_optimizer.step()
            result["recovery_anchor_loss"] = float(loss.detach().cpu())
        state["update_count"] += 1
        result["recovery_critic_only"] = float(not state["actor_unfrozen"])
        return result

    @functools.wraps(original_save)
    def protected_save(path, infos=None):
        merged = dict(infos or {})
        merged["recovery_protection"] = {
            **state, "sampler_rng_state": generator.get_state(),
            "anchor_optimizer_state": anchor_optimizer.state_dict(),
        }
        result = original_save(path, merged)
        # JSON仅保存可读摘要，完整优化器/RNG随checkpoint保存；路径缺失时不伪造日志目录。
        if runner.log_dir is not None:
            summary = Path(runner.log_dir) / "params/recovery_protection.json"
            summary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    runner.alg.update = protected_update
    runner.save = protected_save
    runner._recovery_protection = state
    return state
