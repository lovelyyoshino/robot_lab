# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.1.0
# @last_modified 2026-09-16
# @changelog
#   - v0.1.0 (2026-09-16): 新增独立深度时序缓存，支持显式采样间距、链路扰动和逐环境复位。

"""与仿真器解耦的深度时间合同；输入是已经裁剪、归一化的单帧。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DepthTemporalCfg:
    """所有时长都以 policy 控制步计数，stride=5 表示相邻输出帧相隔 5 步。"""

    history_length: int = 8
    history_stride: int = 5
    delay_steps: tuple[int, int] = (0, 1)
    drop_probability: float = 0.05
    freeze_probability: float = 0.01
    freeze_steps: tuple[int, int] = (2, 5)
    noise_half_range: float = 0.02

    def __post_init__(self):
        for name in ("history_length", "history_stride"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name, minimum in (("delay_steps", 0), ("freeze_steps", 1)):
            value = getattr(self, name)
            if (
                not isinstance(value, (tuple, list)) or len(value) != 2
                or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
                or value[0] < minimum or value[1] < value[0]
            ):
                raise ValueError(f"{name} 必须是从 {minimum} 开始的有效闭区间整数对")
        for name in ("drop_probability", "freeze_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        if not math.isfinite(self.noise_half_range) or self.noise_half_range < 0:
            raise ValueError("noise_half_range 必须是非负有限数")

    def contract(self, control_dt: float) -> dict:
        """供 checkpoint 导出器记录真实时间语义；维度相同不等于可共用预处理。"""
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt 必须是正的有限秒数")
        return {
            "contract_id": "rover_depth_temporal_v2",
            "history_owner": "TemporalDepthObservation",
            "history_order": "oldest_to_newest",
            "history_fill": "repeat_first_received_frame_after_reset",
            "control_dt_s": control_dt,
            "history_span_s": (self.history_length - 1) * self.history_stride * control_dt,
            "frame_age_steps_oldest_to_newest": list(
                range((self.history_length - 1) * self.history_stride, -1, -self.history_stride)
            ),
            "frame_age_semantics": "delivered_history_slot_age_not_raw_capture_age",
            "delay_semantics": "per_control_step_raw_frame_selection_before_delivery",
            "drop_and_freeze_semantics": "hold_last_received_frame_including_its_noise",
            "sampling": asdict(self),
        }


class DepthTemporalBuffer:
    """缓存深度采集与到达历史，输出固定长度、由旧到新的时序向量。

    缓存每个控制步更新一次。跳帧发生在输出采样，不是减少相机采集次数。
    延迟先作用于原始采集流，丢帧/冻结再作用于到达流；旧帧的噪声不会每次读时重采样。
    ``reset`` 只标记对应环境，首次新帧到达时再填充，避免读到 reset 之前的渲染内容。
    """

    def __init__(self, cfg: DepthTemporalCfg, *, generator: torch.Generator | None = None):
        self.cfg = cfg
        self.generator = generator
        self._last_step: int | None = None
        self._raw: torch.Tensor | None = None
        # _raw 是惰性分配的唯一哨兵；其余缓冲区由 _allocate 原子地创建。
        self._delivered: torch.Tensor
        self._pending_reset: torch.Tensor
        self._freeze_remaining: torch.Tensor
        self._last_received: torch.Tensor
        self._output: torch.Tensor
        self._raw_pointer = 0
        self._history_pointer = 0

    def reset(self, env_ids=None) -> None:
        """允许空集、逐环境和全量复位；同一步 reset 后仍能读到新 episode。"""
        if self._raw is None:
            return
        if env_ids is None:
            self._pending_reset.fill_(True)
            self._freeze_remaining.zero_()
            self._last_step = None
        else:
            self._pending_reset[env_ids] = True
            self._freeze_remaining[env_ids] = 0

    def _allocate(self, frame: torch.Tensor) -> None:
        count, width = frame.shape
        self._raw = frame.new_empty((self.cfg.delay_steps[1] + 1, count, width))
        capacity = (self.cfg.history_length - 1) * self.cfg.history_stride + 1
        self._delivered = frame.new_empty((capacity, count, width))
        self._pending_reset = torch.ones(count, dtype=torch.bool, device=frame.device)
        self._freeze_remaining = torch.zeros(count, dtype=torch.long, device=frame.device)
        self._last_received = frame.new_empty((count, width))
        self._output = frame.new_empty((count, self.cfg.history_length * width))
        self._ages = torch.arange(
            capacity - 1, -1, -self.cfg.history_stride, dtype=torch.long, device=frame.device
        )

    def _integers(self, bounds, count, device):
        if bounds[0] == bounds[1]:
            return torch.full((count,), bounds[0], dtype=torch.long, device=device)
        return torch.randint(bounds[0], bounds[1] + 1, (count,), device=device, generator=self.generator)

    def _events(self, probability, count, device):
        if probability in (0.0, 1.0):
            return torch.full((count,), probability == 1.0, dtype=torch.bool, device=device)
        return torch.rand(count, device=device, generator=self.generator) < probability

    def update(
        self, frame: torch.Tensor, control_step: int, *, enable_corruption: bool = True
    ) -> torch.Tensor:
        """传入每步一帧，重复读取返回副本；跳过控制步必须显式 reset，不能伪造缺失采样。"""
        if not isinstance(control_step, int) or isinstance(control_step, bool) or control_step < 0:
            raise ValueError("control_step 必须是非负整数")
        if frame.ndim != 2 or min(frame.shape) < 1 or not frame.is_floating_point():
            raise ValueError("深度时序输入必须是非空浮点 (N, pixels) 张量")
        if self._raw is None:
            self._allocate(frame)
        elif (
            frame.shape != self._last_received.shape
            or frame.device != self._raw.device
            or frame.dtype != self._raw.dtype
        ):
            raise ValueError("深度输入的形状、设备和 dtype 必须与缓存一致")
        assert self._raw is not None
        if self._last_step is not None and control_step not in (self._last_step, self._last_step + 1):
            raise ValueError("深度缓存必须每个 policy 控制步调用；计数回退或漏步后请显式 reset()")

        new_step = self._last_step != control_step
        if new_step:
            if self._last_step is not None:
                self._raw_pointer = (self._raw_pointer + 1) % self._raw.shape[0]
                self._history_pointer = (self._history_pointer + 1) % self._delivered.shape[0]
            ids = torch.arange(frame.shape[0], device=frame.device)
        else:
            ids = self._pending_reset.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return self._output.clone()

        fresh = frame[ids].clone()
        if enable_corruption and self.cfg.noise_half_range > 0:
            noise = torch.rand(fresh.shape, dtype=fresh.dtype, device=fresh.device, generator=self.generator)
            fresh = (fresh + (2.0 * noise - 1.0) * self.cfg.noise_half_range).clamp(-1.0, 1.0)
        reset_mask = self._pending_reset[ids]
        reset_ids = ids[reset_mask]
        self._raw[self._raw_pointer, ids] = fresh
        self._raw[:, reset_ids] = fresh[reset_mask]

        # 首帧不允许延迟/掉包引用上一 episode；扰动从第二个有效控制步开始。
        count = ids.numel()
        if enable_corruption:
            delay = self._integers(self.cfg.delay_steps, count, frame.device)
        else:
            delay = torch.zeros_like(ids)
        delayed = self._raw[(self._raw_pointer - delay) % self._raw.shape[0], ids]
        if enable_corruption:
            remaining = self._freeze_remaining[ids]
            starts = (
                (remaining == 0)
                & self._events(self.cfg.freeze_probability, count, frame.device)
                & ~reset_mask
            )
            duration = self._integers(self.cfg.freeze_steps, count, frame.device)
            remaining = torch.where(starts, duration, remaining)
            hold = ((remaining > 0) | self._events(self.cfg.drop_probability, count, frame.device)) & ~reset_mask
            received = torch.where(hold[:, None], self._last_received[ids], delayed)
            self._freeze_remaining[ids] = (remaining - 1).clamp_min(0)
        else:
            received = delayed
            self._freeze_remaining[ids] = 0
        self._last_received[ids] = received
        self._delivered[self._history_pointer, ids] = received
        self._delivered[:, reset_ids] = received[reset_mask]
        offsets = (self._history_pointer - self._ages) % self._delivered.shape[0]
        selected = self._delivered[offsets[:, None], ids[None, :]].transpose(0, 1)
        self._output[ids] = selected.reshape(count, -1)
        self._pending_reset[ids] = False
        self._last_step = control_step
        return self._output.clone()
