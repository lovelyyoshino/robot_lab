"""训练源 checkpoint 的不可变谱系记录。

@author pony
@date 2026-09-17
@version v1.2.0
@last_modified 2026-09-17
@changelog
    - v1.2.0 (2026-09-17): 恢复或显式覆盖 PPO 续训学习率，避免内部标量覆盖 checkpoint optimizer。
    - v1.1.0 (2026-09-17): 支持续训时按比例重设探索噪声并清理对应优化器动量。
    - v1.0.0 (2026-09-17): 记录续训/蒸馏源模型哈希、迭代、任务与 seed。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


LOAD_MODES = {"resume", "distillation_teacher"}


def restore_resumed_learning_rate(algorithm, override: float | None = None) -> dict:
    """同步 PPO 内部学习率与已加载 optimizer，显式覆盖时两者一起更新。"""
    optimizer = getattr(algorithm, "optimizer", None)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise ValueError("续训算法缺少 optimizer param group")
    checkpoint_rates = []
    for group in groups:
        rate = group.get("lr")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError("checkpoint optimizer 学习率必须是数值")
        rate = float(rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("checkpoint optimizer 学习率必须是有限正数")
        checkpoint_rates.append(rate)
    configured = getattr(algorithm, "learning_rate", None)
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        raise ValueError("PPO 内部学习率必须是数值")
    configured = float(configured)
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("PPO 内部学习率必须是有限正数")
    if override is None:
        effective = checkpoint_rates[0]
        if any(not math.isclose(rate, effective, rel_tol=1e-12) for rate in checkpoint_rates[1:]):
            raise ValueError("多个 optimizer param group 学习率不一致，无法恢复为 PPO 单一学习率")
    else:
        if isinstance(override, bool) or not isinstance(override, (int, float)):
            raise ValueError("续训学习率覆盖值必须是数值")
        effective = float(override)
        if not math.isfinite(effective) or effective <= 0:
            raise ValueError("续训学习率覆盖值必须是有限正数")
    algorithm.learning_rate = effective
    for group in groups:
        group["lr"] = effective
    return {
        "configured_learning_rate_before_restore": configured,
        "checkpoint_optimizer_learning_rates": checkpoint_rates,
        "override_requested": override is not None,
        "effective_learning_rate": effective,
    }


def scale_policy_exploration_noise(policy, optimizer, scale: float) -> dict:
    """缩放已加载的非状态相关探索噪声，并清零该参数的 Adam 动量。"""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("续训探索噪声比例必须在 (0,1] 内")
    if getattr(policy, "state_dependent_std", False):
        raise ValueError("暂不支持缩放状态相关探索噪声")
    noise_type = getattr(policy, "noise_std_type", None)
    if noise_type == "scalar" and hasattr(policy, "std"):
        parameter = policy.std
        before = parameter.detach().clone()
        with torch.no_grad():
            parameter.mul_(scale)
        after = parameter.detach().clone()
    elif noise_type == "log" and hasattr(policy, "log_std"):
        parameter = policy.log_std
        before = parameter.detach().exp().clone()
        with torch.no_grad():
            parameter.add_(math.log(scale))
        after = parameter.detach().exp().clone()
    else:
        raise ValueError(f"不支持的探索噪声参数化: {noise_type}")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all() or (after <= 0).any():
        raise ValueError("探索噪声包含非有限值或非正值")
    reset_moments = []
    for name, value in optimizer.state.get(parameter, {}).items():
        if name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} and torch.is_tensor(value):
            value.zero_()
            reset_moments.append(name)
    return {
        "scale": scale,
        "noise_std_type": noise_type,
        "before_mean": before.mean().item(),
        "before_min": before.min().item(),
        "before_max": before.max().item(),
        "after_mean": after.mean().item(),
        "after_min": after.min().item(),
        "after_max": after.max().item(),
        "optimizer_moments_reset": sorted(reset_moments),
    }


def checkpoint_source_metadata(checkpoint_path, *, load_mode: str, task: str, seed: int) -> dict:
    """构造源模型谱系；拒绝缺迭代号的非训练 checkpoint。"""
    if load_mode not in LOAD_MODES:
        raise ValueError(f"不支持的 checkpoint 加载模式: {load_mode}")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"源 checkpoint 不存在: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    iteration = checkpoint.get("iter") if isinstance(checkpoint, dict) else None
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("源 checkpoint 缺少有效的非负整数 iter")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("实验 seed 必须为整数")
    if not isinstance(task, str) or not task:
        raise ValueError("实验 task 不能为空")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("源 checkpoint 必须为非空文件")
    return {
        "schema_version": 1,
        "load_mode": load_mode,
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": digest,
        "source_checkpoint_size_bytes": size,
        "source_iteration": iteration,
        "task": task,
        "seed": seed,
    }


def write_checkpoint_source_metadata(params_dir, metadata: dict) -> Path:
    """将谱系写入新 run 的 params；训练配置与该文件共同构成来源合同。"""
    output = Path(params_dir) / "resume_lineage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output
