"""
quat.py — 世界模型积分与观测重建所需的四元数工具（wxyz，与 Isaac Lab 一致）。

自带实现而不引用 isaaclab.utils.math，是为了让核心模块和单测不依赖仿真器；
约定与 Isaac Lab 相同，`isaac_adapter` 读出的状态可以直接使用。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """把机体系向量 v 旋到世界系。"""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """把世界系向量 v 旋到机体系。"""
    return quat_apply(quat_conj(q), v)


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    g = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    g[..., 2] = -1.0
    return quat_apply_inverse(q, g)


def axis_angle_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    # 小角度时 sin(θ/2)/θ → 1/2；用泰勒展开避免 0/0，并保证该处梯度有限。
    angle = rotvec.norm(dim=-1, keepdim=True)
    small = angle < 1e-6
    safe = torch.where(small, torch.ones_like(angle), angle)
    scale = torch.where(small, 0.5 - angle.square() / 48.0, torch.sin(0.5 * safe) / safe)
    return torch.cat((torch.cos(0.5 * angle), rotvec * scale), dim=-1)


def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    # q 与 -q 表示同一旋转；取 w>=0 的半球，保证结果是最短旋转。
    q = torch.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])
    small = sin_half < 1e-6
    scale = torch.where(small, 2.0 * torch.ones_like(sin_half), angle / sin_half.clamp_min(1e-9))
    return xyz * scale


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=-1)
