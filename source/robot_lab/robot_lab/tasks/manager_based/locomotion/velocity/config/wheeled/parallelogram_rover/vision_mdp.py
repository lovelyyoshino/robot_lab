# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 增加显式时序观测类与复位回调，保留原连续 8 帧预处理合同。
#   - v0.1.0 (2026-09-16): 新增深度裁剪、归一化和展平观测。

"""平行四边形轮足车视觉行走的观测函数。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .vision_temporal import DepthTemporalBuffer, DepthTemporalCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def normalized_depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    data_type: str = "distance_to_image_plane",
    min_depth: float = 0.1,
    max_depth: float = 2.5,
    crop_top: int = 2,
    crop_bottom: int = 2,
) -> torch.Tensor:
    """将深度图裁剪并归一化到 ``[-1, 1]``，返回部署友好的平面向量。

    ``+inf``/``nan`` 视为最远距离，``-inf`` 视为最近距离，保持既有 checkpoint 预处理。
    函数先 clone，不能原地修改传感器共享缓冲区，否则调试视图和其他观测组会被污染。
    """
    if not math.isfinite(min_depth) or not math.isfinite(max_depth) or max_depth <= min_depth:
        raise ValueError("深度上下界必须有限，且 max_depth 必须大于 min_depth")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (crop_top, crop_bottom)):
        raise ValueError("深度裁剪行数必须为整数")
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type].clone()
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise RuntimeError(f"期望深度图形状为 (N,H,W,1)，实际为 {tuple(depth.shape)}")
    if crop_top < 0 or crop_bottom < 0 or crop_top + crop_bottom >= depth.shape[1]:
        raise ValueError("上下裁剪行数不能覆盖整幅深度图")

    depth = depth[:, crop_top : depth.shape[1] - crop_bottom, :, 0]
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth = depth.clamp(min_depth, max_depth)
    depth = 2.0 * (depth - min_depth) / (max_depth - min_depth) - 1.0
    return depth.flatten(start_dim=1)


class TemporalDepthObservation(ManagerTermBase):
    """仅用于显式 Temporal student；自身管理历史，ObsTerm.history_length 必须为 0。

    manager 在解析维度时也会调用本函数，随后自动调用 reset；不能依赖第一次调用就是训练第一帧。
    ``policy.enable_corruption=False`` 会关闭深度噪声和链路扰动，但保留确定性历史采样间距。
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.history_length != 0 or cfg.noise is not None:
            raise ValueError("TemporalDepthObservation 自身管理历史和噪声，不能叠加 manager history/noise")
        defaults = DepthTemporalCfg()
        self._temporal_cfg = DepthTemporalCfg(**{
            name: cfg.params.get(name, getattr(defaults, name))
            for name in DepthTemporalCfg.__dataclass_fields__
        })
        self._history = DepthTemporalBuffer(self._temporal_cfg)

    def reset(self, env_ids=None) -> None:
        self._history.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        min_depth: float = 0.1,
        max_depth: float = 2.5,
        crop_top: int = 2,
        crop_bottom: int = 2,
        history_length: int = 8,
        history_stride: int = 5,
        delay_steps: tuple[int, int] = (0, 1),
        drop_probability: float = 0.05,
        freeze_probability: float = 0.01,
        freeze_steps: tuple[int, int] = (2, 5),
        noise_half_range: float = 0.02,
    ) -> torch.Tensor:
        requested = DepthTemporalCfg(
            history_length, history_stride, delay_steps, drop_probability, freeze_probability,
            freeze_steps, noise_half_range,
        )
        if self.cfg.history_length != 0 or self.cfg.noise is not None:
            raise ValueError("TemporalDepthObservation 不允许叠加 manager history/noise")
        if requested != self._temporal_cfg:
            raise ValueError("观测管理器创建后不能原地修改时序参数；请重建环境以保持缓存和部署合同一致")
        frame = normalized_depth_image(env, sensor_cfg, data_type, min_depth, max_depth, crop_top, crop_bottom)
        return self._history.update(
            frame, env.common_step_counter,
            enable_corruption=env.cfg.observations.policy.enable_corruption,
        )
