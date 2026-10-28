"""
scan_student.py — 深度历史 → 预测高度扫描 → 冻结 Teacher actor 的 latent 对齐学生。

把 187 点高度扫描当作显式 latent：学生只学"从深度看出地形"，控制完全复用已训好的
Teacher-Preview actor（232 = 45 本体 + 187 扫描），不再像动作级 DAgger 那样从头学控制。
导出的模块输入仍是 4141 维（45 本体 + 8×16×32 深度，旧→新、逐帧行优先），
与现有 Student 的深度合同 `rover_depth_temporal_v2` 保持同一接口。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .networks import mlp

PROPRIO_DIM = 45
SCAN_DIM = 187
DEPTH_FRAMES, DEPTH_H, DEPTH_W = 8, 16, 32
STUDENT_OBS_DIM = PROPRIO_DIM + DEPTH_FRAMES * DEPTH_H * DEPTH_W  # 4141
TEACHER_OBS_DIM = PROPRIO_DIM + SCAN_DIM  # 232


class DepthScanEncoder(nn.Module):
    """逐帧 CNN 提特征，GRU 沿 8 帧时间轴汇聚（帧间隔由深度合同决定，这里不感知）。"""

    def __init__(self, latent_dim: int = 128, hidden=(256, 256)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),  # 16×32 → 8×16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),  # → 4×8
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 8, latent_dim), nn.ELU(),
        )
        self.gru = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.head = mlp(latent_dim + PROPRIO_DIM, hidden, SCAN_DIM)
        # TorchScript 不能读取模块级全局常量，导出前必须挂成实例属性。
        self.proprio_dim, self.frames, self.height, self.width = PROPRIO_DIM, DEPTH_FRAMES, DEPTH_H, DEPTH_W

    def forward(self, student_obs: torch.Tensor) -> torch.Tensor:
        proprio = student_obs[:, : self.proprio_dim]
        depth = student_obs[:, self.proprio_dim :]
        batch = depth.shape[0]
        frame_feat = self.cnn(depth.reshape(batch * self.frames, 1, self.height, self.width))
        _, h = self.gru(frame_feat.reshape(batch, self.frames, -1))
        # Teacher 观测里的扫描被 clip 到 [-1, 1]；tanh 保持同一取值域。
        return torch.tanh(self.head(torch.cat((h[-1], proprio), dim=-1)))


def teacher_actor_from_rsl_rl(model_state_dict: dict, hidden=(512, 256, 128)) -> nn.Sequential:
    """从 rsl_rl ActorCritic 权重重建 Teacher actor（无观测归一化，ELU）。"""
    actor = mlp(TEACHER_OBS_DIM, hidden, 12)
    actor.load_state_dict({k[len("actor."):]: v for k, v in model_state_dict.items() if k.startswith("actor.")})
    return actor.requires_grad_(False).eval()


class ScanStudentPolicy(nn.Module):
    """部署形态：4141 → 预测扫描 → 冻结 Teacher actor → 12 维动作。"""

    def __init__(self, encoder: DepthScanEncoder, teacher_actor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.teacher_actor = teacher_actor
        self.proprio_dim = PROPRIO_DIM

    def forward(self, student_obs: torch.Tensor) -> torch.Tensor:
        scan = self.encoder(student_obs)
        return self.teacher_actor(torch.cat((student_obs[:, : self.proprio_dim], scan), dim=-1))


@dataclass
class ScanStudentCfg:
    lr: float = 3e-4
    action_loss_coef: float = 1.0
    epochs: int = 4
    minibatch: int = 4096
    max_grad_norm: float = 1.0
    teacher_drive_start: float = 0.75
    teacher_drive_end: float = 0.10
    teacher_drive_decay_updates: int = 2000


def student_losses(policy: ScanStudentPolicy, student_obs, teacher_obs, cfg: ScanStudentCfg):
    """扫描重建 + 动作对齐。

    动作标签用干净 teacher 观测；预测动作用学生自己的（可能带噪的）本体观测，
    与部署时输入一致。只靠扫描 MSE 时，Teacher 敏感的少数扫描点误差会被平均掉，
    动作项把这部分误差按 Teacher 的敏感度加权回来。
    """
    scan_pred = policy.encoder(student_obs)
    scan_loss = (scan_pred - teacher_obs[:, PROPRIO_DIM:]).square().mean()
    with torch.no_grad():
        action_target = policy.teacher_actor(teacher_obs)
    action_pred = policy.teacher_actor(torch.cat((student_obs[:, :PROPRIO_DIM], scan_pred), dim=-1))
    action_loss = (action_pred - action_target).square().mean()
    return scan_loss + cfg.action_loss_coef * action_loss, scan_loss, action_loss
