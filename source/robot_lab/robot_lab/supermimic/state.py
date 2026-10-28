"""
state.py — 平行四边形车的扁平状态向量、航向不变特征与增量积分。

状态只保留 12 个主动关节（顺序 = 动作合同 arm4, steer4, wheel4）；8 个 mimic
从动关节由 arm 推出，不进世界模型。轮子是连续关节（限位 ±inf），位置无意义，
因此世界模型只预测 8 个腿关节的位置增量，轮子只预测速度。

增量全部在当前机体系表达，所以模型对世界 XY 平移和航向不变，同一组样本可以
覆盖任意朝向。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import torch

from .quat import axis_angle_to_quat, projected_gravity, quat_apply, quat_apply_inverse, quat_conj, quat_mul
from .quat import quat_normalize, quat_to_axis_angle

NUM_JOINTS = 12
NUM_LEG_JOINTS = 8  # 4 arm + 4 steer，位置有界；后 4 个是轮子
NUM_ACTIONS = 12

# 扁平状态布局：pos_w(3) quat_w(4) lin_vel_b(3) ang_vel_b(3) joint_pos_rel(12) joint_vel(12)
POS = slice(0, 3)
QUAT = slice(3, 7)
LIN = slice(7, 10)
ANG = slice(10, 13)
JPOS = slice(13, 25)
JVEL = slice(25, 37)
STATE_DIM = 37

# 增量布局：dlin_b(3) dang_b(3) dpos_b(3) drot_b(3) djpos_leg(8) djvel(12)
DELTA_DIM = 32
FEATURE_DIM = 3 + 3 + 3 + NUM_LEG_JOINTS + NUM_JOINTS  # 29


def make_state(pos_w, quat_w, lin_vel_b, ang_vel_b, joint_pos_rel, joint_vel) -> torch.Tensor:
    joint_pos_rel = joint_pos_rel.clone()
    # 轮子位置无界且不影响动力学，统一置零，避免数值漂移污染归一化统计。
    joint_pos_rel[..., NUM_LEG_JOINTS:] = 0.0
    return torch.cat((pos_w, quat_normalize(quat_w), lin_vel_b, ang_vel_b, joint_pos_rel, joint_vel), dim=-1)


def features(state: torch.Tensor) -> torch.Tensor:
    """航向不变的世界模型输入特征（不含世界位置与航向）。"""
    return torch.cat(
        (
            projected_gravity(state[..., QUAT]),
            state[..., LIN],
            state[..., ANG],
            state[..., JPOS][..., :NUM_LEG_JOINTS],
            state[..., JVEL],
        ),
        dim=-1,
    )


def compute_delta(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    q = state[..., QUAT]
    dpos_b = quat_apply_inverse(q, next_state[..., POS] - state[..., POS])
    drot_b = quat_to_axis_angle(quat_mul(quat_conj(q), next_state[..., QUAT]))
    return torch.cat(
        (
            next_state[..., LIN] - state[..., LIN],
            next_state[..., ANG] - state[..., ANG],
            dpos_b,
            drot_b,
            (next_state[..., JPOS] - state[..., JPOS])[..., :NUM_LEG_JOINTS],
            next_state[..., JVEL] - state[..., JVEL],
        ),
        dim=-1,
    )


def integrate(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """compute_delta 的逆：在机体系增量上推进一步，保持可导。"""
    q = state[..., QUAT]
    pos = state[..., POS] + quat_apply(q, delta[..., 6:9])
    quat = quat_normalize(quat_mul(q, axis_angle_to_quat(delta[..., 9:12])))
    lin = state[..., LIN] + delta[..., 0:3]
    ang = state[..., ANG] + delta[..., 3:6]
    leg = state[..., JPOS][..., :NUM_LEG_JOINTS] + delta[..., 12:20]
    wheel = torch.zeros_like(state[..., JPOS][..., NUM_LEG_JOINTS:])
    jvel = state[..., JVEL] + delta[..., 20:32]
    return torch.cat((pos, quat, lin, ang, leg, wheel, jvel), dim=-1)


def constant_velocity_step(state: torch.Tensor, dt: float) -> torch.Tensor:
    """基线：速度保持不变、位姿/关节按当前速度外推一步。

    世界模型开环误差必须明显优于它才有意义；否则说明模型只学到了"什么都不变"。
    """
    delta = torch.zeros(state.shape[:-1] + (DELTA_DIM,), dtype=state.dtype, device=state.device)
    delta[..., 6:9] = state[..., LIN] * dt
    delta[..., 9:12] = state[..., ANG] * dt
    delta[..., 12:20] = state[..., JVEL][..., :NUM_LEG_JOINTS] * dt
    return integrate(state, delta)


def state_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """预测状态与真值的差，表达在增量空间（与 delta 同布局），供多步损失归一化。"""
    q = target[..., QUAT]
    return torch.cat(
        (
            pred[..., LIN] - target[..., LIN],
            pred[..., ANG] - target[..., ANG],
            quat_apply_inverse(q, pred[..., POS] - target[..., POS]),
            quat_to_axis_angle(quat_mul(quat_conj(q), pred[..., QUAT])),
            (pred[..., JPOS] - target[..., JPOS])[..., :NUM_LEG_JOINTS],
            pred[..., JVEL] - target[..., JVEL],
        ),
        dim=-1,
    )
