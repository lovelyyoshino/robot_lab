#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-30
# @version v0.5.0
# @last_modified 2026-09-30
# @changelog
#   - v0.5.0 (2026-09-30): 独立settle_feedback保持期闭环；统一续训/回放动作合同校验。
#   - v0.3.0 (2026-09-30): 增加 upright_release profile；接近正立时释放固定先验，
#     用于验证 program31 越过正立后的过冲是否是失败根因。
#   - v0.2.0 (2026-09-30): 增加 launch_protected 分阶段 residual trust-region；
#     早期弹射只允许小幅纠偏，保持旧 constant 配置可复现。
#   - v0.1.0 (2026-09-30): 新增 program31 prior + tanh residual，支持左右镜像、
#     接管时钟和物理动作尺度校验。
#   - v0.4.0 (2026-09-30): 严格校验来源任务、动作语义和 program_id；混合程序数据按
#     目标身份选择 canonical 行，防止固定先验静默执行错误动作。

"""用于恢复任务的分段动作先验与有界 residual 动作包装器。

该模块把已经通过物理筛选的 program31 作为低层动作先验：前三秒按固定分段
目标执行，Actor 只输出有界 residual。这样 PPO 从第一轮就拥有“能翻”的动作，
又可以利用当前姿态、速度和历史动作纠正模型误差。它不修改 Isaac Lab 的动作
合同，包装器在送入原环境前完成 raw action 合成。

"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

ACTION_DIM = 12
SEGMENT_COUNT = 6
EXPECTED_SOURCE_TASK = "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0"
EXPECTED_JOINT_NAMES = (
    "fl_arm_joint", "rl_arm_joint", "fr_arm_joint", "rr_arm_joint",
    "fl_qudong_joint", "rl_qudong_joint", "fr_qudong_joint", "rr_qudong_joint",
    "fl_wheel_joint", "rl_wheel_joint", "fr_wheel_joint", "rr_wheel_joint",
)
EXPECTED_TARGET_UNITS = "first_8_joint_position_rad_then_4_wheel_velocity_rad_s"
EXPECTED_HOLD_MODE = "zero_raw_action_default_pose_active_pd"
DEFAULT_RESIDUAL_SCALE = np.asarray([0.20] * 4 + [0.20] * 4 + [0.25] * 4, dtype=np.float32)
"""raw action residual 上限；对应 arm≈0.12rad、qudong≈0.31rad、wheel≈2.5rad/s。"""

# program31 的前两个 0.5 s 是蓄力/弹射窗口。PPO 在这个窗口学到的较大
# correction 会直接改变接触顺序，表现为停在侧面或完全倒置；后四段再逐步
# 放开 residual，让策略处理落地和稳定。这里限制的是 raw action 修正，不改
# program31 目标、不改执行器力矩，也不把动作裁剪放宽。
LAUNCH_PROTECTED_RESIDUAL_SCALE = np.asarray(
    [
        [0.10] * 4 + [0.12] * 4 + [0.12] * 4,
        [0.14] * 4 + [0.16] * 4 + [0.18] * 4,
        [0.18] * 4 + [0.19] * 4 + [0.22] * 4,
        [0.20] * 4 + [0.20] * 4 + [0.25] * 4,
        [0.20] * 4 + [0.20] * 4 + [0.25] * 4,
        [0.20] * 4 + [0.20] * 4 + [0.25] * 4,
    ],
    dtype=np.float32,
)
RESIDUAL_PROFILES = {
    "constant": np.broadcast_to(DEFAULT_RESIDUAL_SCALE, (SEGMENT_COUNT, ACTION_DIM)).copy(),
    "launch_protected": LAUNCH_PROTECTED_RESIDUAL_SCALE,
    "upright_release": np.broadcast_to(DEFAULT_RESIDUAL_SCALE, (SEGMENT_COUNT, ACTION_DIM)).copy(),
    "settle_feedback": np.broadcast_to(DEFAULT_RESIDUAL_SCALE, (SEGMENT_COUNT, ACTION_DIM)).copy(),
}
"""可复现的 residual trust-region 配置；profile 名称会写入训练谱系。"""

UPRIGHT_RELEASE_PROFILES = frozenset(("upright_release",))
UPRIGHT_GRAVITY_THRESHOLD = -0.75
SETTLE_RESIDUAL_SCALE = np.asarray([0.10] * 8 + [0.10] * 4, dtype=np.float32)
"""3秒后围绕默认姿态纠偏：摆臂约0.061rad、转向约0.157rad、轮速1rad/s。"""

_MIRROR_INDEX = np.asarray([2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9], dtype=np.int64)


def prior_file_sha256(path: str | Path) -> str:
    """返回先验 NPZ 的字节哈希；路径可搬迁，但内容不能静默替换。"""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"恢复动作先验数据不存在: {source}")
    with source.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prior_runtime_contract(path: str | Path, *, program_id: int, residual_profile: str,
                           task: str) -> dict[str, Any]:
    """构造随 checkpoint 保存的先验谱系；不把可变路径当作唯一身份。"""
    return {
        "schema": "recovery_residual_runtime_v3",
        "prior_data": str(Path(path).expanduser().resolve()),
        "prior_sha256": prior_file_sha256(path),
        "prior_program_id": int(program_id),
        "residual_profile": residual_profile,
        "task": task,
        "action_semantics": profile_action_semantics(residual_profile),
    }


def profile_action_semantics(profile: str) -> dict[str, Any]:
    """保存数值与门控语义，防止profile同名但代码改变后仍被当成兼容。"""
    return {
        "composition": "prior_plus_tanh_residual_v1",
        "active_residual_scale": residual_profile_scale(profile).tolist(),
        "settle_residual_scale": (SETTLE_RESIDUAL_SCALE if profile == "settle_feedback"
                                  else np.zeros(ACTION_DIM, dtype=np.float32)).tolist(),
        "prior_after_active_steps": "zero_raw_default_pose_pd",
        "upright_release_gravity_z": UPRIGHT_GRAVITY_THRESHOLD if profile in UPRIGHT_RELEASE_PROFILES else None,
        "inactive_action": "zero",
    }


def validate_prior_runtime_contract(runtime: dict[str, Any], path: str | Path, *,
                                    program_id: int, task: str) -> None:
    """验证回放/评测传入的 NPZ 与训练时内容一致；兼容旧版无哈希谱系。"""
    if not isinstance(runtime, dict):
        raise ValueError("recovery_residual_runtime.json 必须是对象")
    schema = runtime.get("schema")
    if schema not in ("recovery_residual_runtime_v1", "recovery_residual_runtime_v2", "recovery_residual_runtime_v3"):
        raise ValueError("未知 recovery_residual_runtime schema")
    recorded_task = runtime.get("task")
    if not isinstance(recorded_task, str) or recorded_task != task:
        raise ValueError(f"checkpoint先验任务为{recorded_task!r}，当前任务为{task!r}")
    recorded_id = runtime.get("prior_program_id")
    if type(recorded_id) is not int or recorded_id != program_id:
        raise ValueError(f"checkpoint先验program_id为{recorded_id}，当前请求为{program_id}")
    recorded_hash = runtime.get("prior_sha256")
    if schema != "recovery_residual_runtime_v1" and not recorded_hash:
        raise ValueError("先验谱系缺少NPZ哈希")
    if recorded_hash is not None and recorded_hash != prior_file_sha256(path):
        raise ValueError("checkpoint记录的先验NPZ哈希与当前文件不一致，拒绝静默替换动作先验")
    if schema == "recovery_residual_runtime_v1" and recorded_hash is None:
        recorded_path = runtime.get("prior_data")
        if (not isinstance(recorded_path, str)
                or Path(recorded_path).expanduser().resolve() != Path(path).expanduser().resolve()):
            raise ValueError("旧v1先验无历史哈希，不能更换prior_data路径")
        warnings.warn("旧v1先验只有路径、没有历史哈希；本次兼容加载不能证明历史文件未变。", UserWarning, stacklevel=2)
    profile = runtime.get("residual_profile")
    if profile not in RESIDUAL_PROFILES:
        raise ValueError(f"checkpoint residual profile无效: {profile!r}")
    if schema != "recovery_residual_runtime_v3" and profile == "settle_feedback":
        raise ValueError("settle_feedback要求v3动作语义，不能冒用旧版合同")
    if (schema == "recovery_residual_runtime_v3"
            and runtime.get("action_semantics") != profile_action_semantics(profile)):
        raise ValueError("checkpoint动作语义与当前profile实现不一致")


def resolve_checkpoint_prior(checkpoint: str | Path, path: str | Path, *, program_id: int,
                             task: str, residual_profile: str = "auto", allow_task_transfer: bool = False,
                             allow_profile_transfer: bool = False) -> str:
    """统一resume/play/eval的入口；迁移必须由显式新实验入口授权，不能猜合同。"""
    runtime_path = Path(checkpoint).expanduser().resolve().parent / "params/recovery_residual_runtime.json"
    with runtime_path.open(encoding="utf-8") as stream:
        runtime = json.load(stream)
    if not isinstance(runtime, dict):
        raise ValueError("recovery_residual_runtime.json 必须是对象")
    recorded_task = runtime.get("task")
    if not isinstance(recorded_task, str) or not recorded_task:
        raise ValueError("checkpoint缺少来源任务")
    if allow_task_transfer and recorded_task != task:
        # 迁移权限仅覆盖相同60/72维控制族，不能把任意任务当成同合同。
        from recovery_contract import is_residual_recovery_task

        if not is_residual_recovery_task(recorded_task) or not is_residual_recovery_task(task):
            raise ValueError("只允许Side-Residual与FailureReplay-Residual任务间迁移")
    validate_prior_runtime_contract(runtime, path, program_id=program_id,
                                    task=recorded_task if allow_task_transfer else task)
    old_profile = runtime["residual_profile"]
    selected = old_profile if residual_profile == "auto" else residual_profile
    if selected not in RESIDUAL_PROFILES:
        raise ValueError(f"未知 residual profile: {selected!r}")
    if selected != old_profile and not allow_profile_transfer:
        raise ValueError(
            f"checkpoint residual profile为{old_profile!r}，请求为{selected!r}；"
            "resume/play/evaluate不能更换动作语义。训练迁移请显式使用--recovery_residual_warm_start。"
        )
    return selected


def warm_start_residual_policy(policy: Any, checkpoint: str | Path, *, std: float = 0.15) -> dict[str, Any]:
    """只迁移兼容的网络权重；调用前校验控制/先验合同，不携带旧优化器或迭代。"""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("residual policy std 必须为有限正数")
    source = Path(checkpoint).expanduser().resolve()
    loaded = torch.load(source, map_location="cpu", weights_only=True)
    state = loaded.get("model_state_dict") if isinstance(loaded, dict) else None
    if not isinstance(state, dict) or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("residual warm start缺少有限model_state_dict")
    policy.load_state_dict(state, strict=True)
    with torch.no_grad():
        policy.std.fill_(std)
    with (source.parent / "params/recovery_residual_runtime.json").open(encoding="utf-8") as stream:
        source_contract = json.load(stream)
    return {"schema": "recovery_residual_warm_start_v1", "source_checkpoint": str(source),
            "source_checkpoint_sha256": prior_file_sha256(source), "std": std,
            "source_prior_contract": source_contract,
            "optimizer_restored": False, "iteration_restored": False, "skill_success_verified": False}


def residual_profile_scale(profile: str) -> np.ndarray:
    """返回独立副本，防止调用方原地修改全局 trust-region。"""
    if profile not in RESIDUAL_PROFILES:
        choices = ", ".join(sorted(RESIDUAL_PROFILES))
        raise ValueError(f"未知 residual profile: {profile!r}，可选 {choices}")
    return np.asarray(RESIDUAL_PROFILES[profile], dtype=np.float32).copy()


def _metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"恢复动作先验数据不存在: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("恢复动作先验 NPZ 缺少 metadata")
        value = archive["metadata"]
        if value.shape != () or value.dtype.kind not in "OUS":
            raise ValueError("恢复动作先验 metadata 必须是标量 JSON 字符串")
        try:
            metadata = json.loads(str(value.item()))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("恢复动作先验 metadata 不是有效 JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("恢复动作先验 metadata 必须是对象")
    return metadata


def _mirror_targets(targets: np.ndarray) -> np.ndarray:
    reflected = targets[..., _MIRROR_INDEX].copy()
    reflected[..., 4:8] *= -1.0
    return reflected


def _term_vector(term: Any, name: str, num_envs: int) -> np.ndarray:
    value = getattr(term, name, None)
    if value is None:
        raise ValueError(f"恢复动作先验无法读取 action term 的 {name}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((num_envs, len(term._joint_names)), float(array), dtype=np.float32)
    elif array.ndim == 1:
        array = np.broadcast_to(array[None], (num_envs, array.shape[0])).copy()
    if array.shape != (num_envs, len(term._joint_names)):
        raise ValueError(f"恢复动作先验 {name} 维数与 action term 不匹配: {array.shape}")
    return array


class RecoveryPrior:
    """从示范 NPZ 构造可在多个 vectorized env 中复用的 raw action 先验。"""

    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        program_id: int,
        num_envs: int,
        residual_scale: np.ndarray | list[float] | None = None,
        residual_profile: str = "constant",
    ) -> None:
        if residual_profile not in RESIDUAL_PROFILES:
            choices = ", ".join(sorted(RESIDUAL_PROFILES))
            raise ValueError(f"未知 residual profile: {residual_profile!r}，可选 {choices}")
        source = metadata.get("source", {})
        source_task = source.get("task")
        if source_task != EXPECTED_SOURCE_TASK:
            raise ValueError(
                "恢复动作先验必须来自固定Side脚本任务 "
                f"{EXPECTED_SOURCE_TASK!r}，实际为 {source_task!r}"
            )
        program = source.get("action_program", {})
        targets = np.asarray(program.get("targets"), dtype=np.float32)
        defaults = np.asarray(program.get("default_targets"), dtype=np.float32)
        scales = np.asarray(program.get("action_scales"), dtype=np.float32)
        if targets.ndim != 3 or targets.shape[1:] != (SEGMENT_COUNT, ACTION_DIM):
            raise ValueError("恢复动作先验 targets 必须为 [program,6,12]")
        if program.get("targets_shape") not in (None, list(targets.shape)):
            raise ValueError("恢复动作先验 targets_shape 与 targets 实际维数不一致")
        if tuple(program.get("joint_names", ())) != EXPECTED_JOINT_NAMES:
            raise ValueError("恢复动作先验 joint_names 与12维动作合同不一致")
        if program.get("target_units") != EXPECTED_TARGET_UNITS:
            raise ValueError("恢复动作先验 target_units 不是位置/轮速物理单位合同")
        if program.get("hold_mode") != EXPECTED_HOLD_MODE:
            raise ValueError("恢复动作先验 hold_mode 不是默认姿态主动PD保持合同")
        if defaults.shape != (targets.shape[0], ACTION_DIM) or scales.shape != defaults.shape:
            raise ValueError("恢复动作先验 default_targets/action_scales 维数不匹配")
        if type(program_id) is not int or program_id < 0:
            raise ValueError("program_id 必须是非负整数")
        if not np.isfinite(targets).all() or not np.isfinite(defaults).all() or not np.isfinite(scales).all():
            raise ValueError("恢复动作先验包含非有限值")
        segment_steps = program.get("segment_steps")
        active_steps = program.get("active_steps")
        step_dt = program.get("step_dt")
        if (type(segment_steps) is not int or type(active_steps) is not int or segment_steps < 1
                or active_steps != segment_steps * SEGMENT_COUNT or not math.isfinite(float(step_dt))):
            raise ValueError("恢复动作先验的分段时钟无效")

        # NPZ 中每行是一个评测环境，可能已经按该环境初始倒向做过镜像。
        # 先还原任意一个未镜像行作为 canonical program，不能假设 row==program_id
        # 永远是未镜像；否则换一批采集 seed 会把目标再镜像一次。
        mirrored_flags = np.asarray(program.get("mirrored", []), dtype=bool)
        if mirrored_flags.shape not in ((), (targets.shape[0],)):
            raise ValueError("恢复动作先验 mirrored 必须逐行对应 targets")
        if mirrored_flags.shape == ():
            mirrored_flags = np.zeros(targets.shape[0], dtype=bool)

        source_program_ids = np.asarray(program.get("source_program_ids", []))
        if source_program_ids.size:
            if source_program_ids.shape != (targets.shape[0],):
                raise ValueError("恢复动作先验 source_program_ids 必须逐行对应 targets")
            if not np.issubdtype(source_program_ids.dtype, np.integer):
                raise ValueError("恢复动作先验 source_program_ids 必须是整数")
            matching = np.flatnonzero(source_program_ids == program_id)
            if len(matching) == 0:
                raise ValueError("动作先验中不存在请求的 program_id")
        else:
            # 兼容极早期单程序 NPZ：没有来源列表时只能把 program_id 当作行号。
            if program_id >= targets.shape[0]:
                raise ValueError(f"program_id 必须在 [0,{targets.shape[0]}) 内")
            matching = np.asarray([program_id], dtype=np.int64)
        canonical_indices = matching[~mirrored_flags[matching]]
        base_index = int(canonical_indices[0] if len(canonical_indices) else matching[0])
        target = targets[base_index]
        if mirrored_flags.shape == (targets.shape[0],) and mirrored_flags[base_index]:
            target = _mirror_targets(target)
        mirrored_target = _mirror_targets(target)
        default = defaults[base_index]
        scale = scales[base_index]
        if (np.abs(scale) < 1e-8).any():
            raise ValueError("恢复动作先验 action scale 不能为零")
        # 同一 program 的不同环境行只能表达左右镜像，不能偷偷携带另一套动作合同。
        if len(matching) > 1:
            if not np.allclose(defaults[matching], default, rtol=1e-6, atol=1e-7):
                raise ValueError("同一 program 的 default_targets 不一致")
            if not np.allclose(scales[matching], scale, rtol=1e-6, atol=1e-7):
                raise ValueError("同一 program 的 action_scales 不一致")
        self.raw_unmirrored = ((target - default[None]) / scale[None]).astype(np.float32)
        self.raw_mirrored = ((mirrored_target - default[None]) / scale[None]).astype(np.float32)
        if not np.isfinite(self.raw_unmirrored).all() or not np.isfinite(self.raw_mirrored).all():
            raise ValueError("恢复动作先验 raw action 非有限")
        self.segment_steps = segment_steps
        self.active_steps = active_steps
        self.step_dt = float(step_dt)
        self.program_id = program_id
        self.base_index = base_index
        self.expected_scales = scale.copy()
        self.expected_offsets = default.copy()
        self.num_envs = num_envs
        if residual_scale is None:
            matrix = residual_profile_scale(residual_profile)
        else:
            matrix = np.asarray(residual_scale, dtype=np.float32)
            if matrix.shape == (ACTION_DIM,):
                matrix = np.broadcast_to(matrix, (SEGMENT_COUNT, ACTION_DIM)).copy()
            if residual_profile != "constant":
                raise ValueError("显式 residual_scale 不能同时指定非constant profile")
        if (matrix.shape != (SEGMENT_COUNT, ACTION_DIM) or not np.isfinite(matrix).all()
                or (matrix < 0).any()):
            raise ValueError("residual_scale 必须是12维向量或[6,12]有限非负矩阵")
        self.residual_scale = torch.as_tensor(matrix)
        self.residual_profile = residual_profile
        self.release_upright = residual_profile in UPRIGHT_RELEASE_PROFILES
        self.source_program_ids = source_program_ids

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        program_id: int = 31,
        num_envs: int,
        residual_scale: np.ndarray | list[float] | None = None,
        residual_profile: str = "constant",
    ) -> RecoveryPrior:
        return cls(_metadata(path), program_id=program_id, num_envs=num_envs,
                   residual_scale=residual_scale, residual_profile=residual_profile)

    def _raw_tensor(self, env: Any) -> torch.Tensor:
        device = getattr(env, "device", None) or getattr(env.scene["robot"].data, "device", "cpu")
        steps = getattr(env, "_recovery_control_steps", None)
        active = getattr(env, "_recovery_control_active", None)
        if steps is None:
            steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        if steps.shape != (self.num_envs,) or active.shape != (self.num_envs,):
            raise ValueError("恢复动作先验与 vectorized env 数量不匹配")
        steps = steps.to(device=device, dtype=torch.long)
        active = active.to(device=device, dtype=torch.bool)
        segment = (steps // self.segment_steps).clamp(max=SEGMENT_COUNT - 1)
        unmirrored = torch.as_tensor(self.raw_unmirrored, device=device)
        mirrored = torch.as_tensor(self.raw_mirrored, device=device)
        side_state = getattr(env, "_recovery_takeover_gravity_b", None)
        if side_state is None:
            mirror = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        else:
            side = side_state[:, 1]
            mirror = torch.isfinite(side) & (side > 0)
        raw = unmirrored[segment].clone()
        raw[mirror] = mirrored[segment[mirror]]
        enabled = active & (steps < self.active_steps)
        if self.release_upright:
            # 只读取当前姿态，不写状态；缺少该字段时保持旧行为，避免测试/非Isaac
            # wrapper 静默改变合同。阈值对应接近正立而非最终稳定门。
            scene = getattr(env, "scene", None)
            gravity = None
            if scene is not None:
                try:
                    gravity = getattr(scene["robot"].data, "projected_gravity_b", None)
                except (KeyError, TypeError, AttributeError):
                    gravity = None
            if isinstance(gravity, torch.Tensor) and gravity.shape == (self.num_envs, 3):
                near_upright = torch.isfinite(gravity).all(dim=-1) & (
                    gravity[:, 2] <= UPRIGHT_GRAVITY_THRESHOLD
                )
                enabled = enabled & ~near_upright.to(device=enabled.device)
        raw = torch.where(enabled[:, None], raw, torch.zeros_like(raw))
        return raw

    def compose(self, env: Any, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 2 or residual.shape != (self.num_envs, ACTION_DIM):
            raise ValueError("恢复 residual 必须是 [num_envs,12]")
        if not torch.isfinite(residual).all():
            raise ValueError("恢复 residual 包含非有限值")
        prior = self._raw_tensor(env).to(device=residual.device, dtype=residual.dtype)
        active = getattr(env, "_recovery_control_active", None)
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=residual.device)
        else:
            active = active.to(device=residual.device, dtype=torch.bool)
        steps = getattr(env, "_recovery_control_steps", None)
        if steps is None:
            steps = torch.zeros(self.num_envs, dtype=torch.long, device=residual.device)
        else:
            steps = steps.to(device=residual.device, dtype=torch.long)
        if self.residual_profile != "settle_feedback":
            active = active & (steps < self.active_steps)
        segment = (steps // self.segment_steps).clamp(max=SEGMENT_COUNT - 1)
        scale = self.residual_scale.to(device=residual.device, dtype=residual.dtype)[segment]
        if self.residual_profile == "settle_feedback":
            # 固定动作在3秒归零，但保持期不能切断策略梯度；只放开小幅闭环。
            settle_scale = torch.as_tensor(SETTLE_RESIDUAL_SCALE, device=residual.device, dtype=residual.dtype)
            scale = torch.where((steps >= self.active_steps)[:, None], settle_scale, scale)
        correction = torch.tanh(residual) * scale
        correction = torch.where(active[:, None], correction, torch.zeros_like(correction))
        return prior + correction


class RecoveryPriorActionWrapper(gym.Wrapper):
    """最小 Gym-compatible wrapper；避免让 IsaacLab action contract 发生迁移变化。"""

    def __init__(self, env: Any, prior: RecoveryPrior):
        super().__init__(env)
        self.env = env
        self.prior = prior

    def step(self, action):
        composed = self.prior.compose(self.unwrapped, action)
        return self.env.step(composed)


def attach_recovery_prior(env: Any, path: str | Path, *, program_id: int = 31,
                          residual_scale: np.ndarray | list[float] | None = None,
                          residual_profile: str = "constant") -> RecoveryPriorActionWrapper:
    """从原始 Gym env 构造 wrapper；必须在 RslRlVecEnvWrapper 之前调用。"""
    base = getattr(env, "unwrapped", env)
    num_envs = int(getattr(base, "num_envs", 0))
    if num_envs <= 0:
        raise ValueError("恢复动作先验要求正的 vectorized env 数量")
    prior = RecoveryPrior.from_npz(path, program_id=program_id, num_envs=num_envs,
                                   residual_scale=residual_scale, residual_profile=residual_profile)
    manager = getattr(base, "action_manager", None)
    if manager is None or list(getattr(manager, "active_terms", [])) != ["joint_pos", "joint_vel"]:
        raise ValueError("恢复动作先验只支持 joint_pos + joint_vel 动作合同")
    terms = [manager.get_term(name) for name in manager.active_terms]
    scales = []
    offsets = []
    for term in terms:
        scales.append(_term_vector(term, "_scale", num_envs))
        offsets.append(_term_vector(term, "_offset", num_envs))
    actual_scales = np.concatenate(scales, axis=1)[0]
    actual_offsets = np.concatenate(offsets, axis=1)[0]
    expected_scales = prior.expected_scales
    expected_offsets = prior.expected_offsets
    if not np.allclose(actual_scales, expected_scales, rtol=1e-5, atol=1e-6):
        raise ValueError("恢复动作先验 action scale 与当前环境不一致")
    if not np.allclose(actual_offsets, expected_offsets, rtol=1e-5, atol=1e-6):
        raise ValueError("恢复动作先验 action offset 与当前环境不一致")
    current_dt = getattr(base, "step_dt", None)
    if current_dt is not None and not math.isclose(float(current_dt), prior.step_dt, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("恢复动作先验 step_dt 与当前环境不一致")
    return RecoveryPriorActionWrapper(env, prior)


def initialize_residual_policy(policy: Any, *, std: float = 0.15) -> dict[str, Any]:
    """把 residual 均值初始化为零，避免首轮 PPO 破坏可行脚本动作。"""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("residual policy std 必须为有限正数")
    actor = getattr(policy, "actor", None)
    if actor is None or not len(actor) or not hasattr(actor[-1], "weight"):
        raise ValueError("无法识别 residual Actor 的末层")
    layer = actor[-1]
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()
        if hasattr(policy, "std"):
            policy.std.fill_(std)
    return {"schema": "recovery_residual_initialization_v1", "actor_mean_zero": True,
            "std": float(std), "bounded_by": "tanh_residual_scale", "skill_success_verified": False}
