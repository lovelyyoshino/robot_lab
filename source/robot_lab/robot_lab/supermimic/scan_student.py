"""
scan_student.py — 深度历史 → 预测高度扫描 → 冻结 Teacher actor 的 latent 对齐学生。

@author pony
@date 2026-10-26
@version v0.3.0
@last_modified 2026-11-03
@changelog
    - v0.3.0 (2026-11-03): 增加台阶/边缘与摆臂加权、独立传感扰动合同和有限梯度更新保护。
    - v0.2.0 (2026-11-02): 原生SuperMimic教师与独立车道输入，保留tanh/clamp及深度消融边界。
    - v0.1.0 (2026-10-26): 初始创建

把 187 点高度扫描当作显式 latent：学生只学"从深度看出地形"，控制完全复用已训好的
Teacher-Preview actor（232 = 45 本体 + 187 扫描），不再像动作级 DAgger 那样从头学控制。
导出的模块输入仍是 4141 维（45 本体 + 8×16×32 深度，旧→新、逐帧行优先），
与现有 Student 的深度合同 `rover_depth_temporal_v2` 保持同一接口。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .networks import Actor, mlp

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
        depth = student_obs[:, self.proprio_dim : self.proprio_dim + self.frames * self.height * self.width]
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


def teacher_actor_from_checkpoint(checkpoint: dict) -> tuple[nn.Module, bool]:
    """原样保留 SuperMimic 的 tanh/clamp 和车道输入，避免转换成 PPO 时改变动作。"""
    if "model_state_dict" in checkpoint:
        return teacher_actor_from_rsl_rl(checkpoint["model_state_dict"]), False
    cfg, weights = checkpoint["cfg"], checkpoint["actor"]
    obs_dim = weights["actor.0.weight"].shape[1]
    if obs_dim not in (TEACHER_OBS_DIM, TEACHER_OBS_DIM + 1):
        raise ValueError(f"深度扫描学生仅支持232/233维地形教师，实际为{obs_dim}")
    actor = Actor(obs_dim, tuple(cfg["actor_hidden"]), cfg["action_bound"], cfg["action_squash"])
    actor.load_state_dict(weights)
    return actor.requires_grad_(False).eval(), obs_dim == TEACHER_OBS_DIM + 1


def student_depth_contract(contract: dict, lane_obs: bool, observation_noise: dict | None = None) -> dict:
    """深度历史保持原顺序；车道定位作为最后一列，不参与深度编码。"""
    out = dict(contract)
    if observation_noise is not None:
        out["observation_noise"] = dict(observation_noise)
    if lane_obs:
        from .lane_offset import LANE_OBS_CLIP, LANE_OBS_SCALE

        out.update(student_obs_dim=STUDENT_OBS_DIM + 1, teacher_obs_dim=TEACHER_OBS_DIM + 1,
                   extra_observations=[{"term": "lane_offset", "index": STUDENT_OBS_DIM,
                                        "scale": LANE_OBS_SCALE, "clip": LANE_OBS_CLIP}])
    return out


def configure_student_observation_noise(env_cfg, *, clean_proprio: bool = False, camera_corruption: bool = True):
    """独立关闭本体噪声，避免关闭整组 corruption 时连带丢掉相机延迟、丢帧和噪声。"""
    policy = env_cfg.observations.policy
    if clean_proprio:
        for name, term in vars(policy).items():
            if name != "depth_image" and hasattr(term, "noise"):
                term.noise = None
    if not camera_corruption:
        depth = policy.depth_image
        depth.noise = None
        for name, value in (("noise_half_range", 0.0), ("drop_probability", 0.0),
                            ("freeze_probability", 0.0), ("delay_steps", (0, 0))):
            if name in depth.params:
                depth.params[name] = value


def configure_student_lane(env_cfg, lane_obs: bool) -> None:
    if lane_obs:
        from isaaclab.managers import ObservationTermCfg

        from .lane_offset import lane_offset_obs

        for name in ("policy", "teacher"):
            getattr(env_cfg.observations, name).lane_offset = ObservationTermCfg(func=lane_offset_obs)


def configure_saved_student(env_cfg, policy_path) -> dict:
    """回放/验收从保存的输入合同配置车道列，返回用于严格比较的当前合同。"""
    import json

    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import (
        vision_depth_contract,
    )

    saved = json.loads((Path(policy_path).parent.parent / "depth_contract.json").read_text())
    lane_obs = bool(saved.get("extra_observations"))
    configure_student_lane(env_cfg, lane_obs)
    noise = saved.get("observation_noise")
    if noise is not None:
        configure_student_observation_noise(env_cfg, **noise)
    return student_depth_contract(vision_depth_contract(env_cfg), lane_obs, noise)


class ScanStudentPolicy(nn.Module):
    """部署形态：4141 → 预测扫描 → 冻结 Teacher actor → 12 维动作。"""

    def __init__(self, encoder: DepthScanEncoder, teacher_actor: nn.Module, lane_obs: bool = False):
        super().__init__()
        self.encoder = encoder
        self.teacher_actor = teacher_actor.requires_grad_(False).eval()
        self.proprio_dim = PROPRIO_DIM
        self.lane_obs = lane_obs
        self.obs_dim = STUDENT_OBS_DIM + int(lane_obs)

    def actor_observation(self, student_obs: torch.Tensor, scan: torch.Tensor) -> torch.Tensor:
        observation = torch.cat((student_obs[:, : self.proprio_dim], scan), dim=-1)
        if self.lane_obs:
            observation = torch.cat((observation, student_obs[:, -1:]), dim=-1)
        return observation

    def forward(self, student_obs: torch.Tensor) -> torch.Tensor:
        if student_obs.shape[-1] != self.obs_dim:
            raise ValueError("深度扫描学生输入维度不符，请检查深度历史与车道定位合同")
        scan = self.encoder(student_obs)
        return self.teacher_actor(self.actor_observation(student_obs, scan))


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
    scan_step_weight: float = 1.0
    scan_edge_weight: float = 1.0
    scan_edge_threshold: float = 0.03
    arm_action_weight: float = 1.0
    scan_grid_shape: tuple[int, int] = (11, 17)

    def __post_init__(self):
        for name in ("lr", "max_grad_norm", "scan_edge_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        for name in ("action_loss_coef", "scan_step_weight", "scan_edge_weight", "arm_action_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < (0.0 if name == "action_loss_coef" else 1.0):
                raise ValueError(f"{name} 必须有限且不小于 {0 if name == 'action_loss_coef' else 1}")
        if (len(self.scan_grid_shape) != 2 or math.prod(self.scan_grid_shape) != SCAN_DIM
                or any(not isinstance(size, int) or size <= 0 for size in self.scan_grid_shape)):
            raise ValueError("扫描网格必须覆盖187个点")
        if self.epochs <= 0 or self.minibatch <= 0:
            raise ValueError("epochs 和 minibatch 必须为正数")
        if not 0 <= self.teacher_drive_end <= self.teacher_drive_start <= 1:
            raise ValueError("教师驱动概率必须满足 0 <= end <= start <= 1")


def resolve_student_config(saved: dict | None = None, **overrides) -> ScanStudentCfg:
    """续训先恢复完整损失配置，仅显式命令行值覆盖；旧存档缺失的新字段保持兼容默认。"""
    values = asdict(ScanStudentCfg())
    if saved is not None:
        values.update(saved)
    values.update({name: value for name, value in overrides.items() if value is not None})
    values["scan_grid_shape"] = tuple(values["scan_grid_shape"])
    return ScanStudentCfg(**values)


def create_student_run_directory(root: str | Path, run_name: str) -> Path:
    """续训也创建新目录，避免覆盖已有导出、输入合同和实验来源记录。"""
    if Path(run_name).name != run_name or run_name in ("", ".", ".."):
        raise ValueError("run_name 必须是单个目录名")
    path = Path(root) / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def restore_student_optimizer(optimizer, saved: dict, cfg: ScanStudentCfg):
    """Adam恢复会覆盖构造时的lr；恢复后显式对齐实际学习率与已记录配置。"""
    optimizer.load_state_dict(saved)
    for group in optimizer.param_groups:
        group["lr"] = cfg.lr


def clone_student_observations(observations: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """仿真器可能在step中原地复用观测buffer，标签必须在推进仿真之前独立持有。"""
    return observations["policy"].detach().clone(), observations["teacher"].detach().clone()


@torch.no_grad()
def scan_edge_mask(teacher_scan: torch.Tensor, cfg: ScanStudentCfg) -> torch.Tensor:
    """按xy展平网格标出相邻高差两侧，避免只监督台阶高面而漏掉起抬时序的前缘。"""
    grid = teacher_scan.reshape(-1, *cfg.scan_grid_shape)
    edges = torch.zeros_like(grid, dtype=torch.bool)
    dx = (grid[..., 1:] - grid[..., :-1]).abs() >= cfg.scan_edge_threshold
    dy = (grid[:, 1:] - grid[:, :-1]).abs() >= cfg.scan_edge_threshold
    edges[..., 1:] |= dx
    edges[..., :-1] |= dx
    edges[:, 1:] |= dy
    edges[:, :-1] |= dy
    return edges.reshape_as(teacher_scan)


@torch.no_grad()
def scan_loss_weights(teacher_scan: torch.Tensor, cfg: ScanStudentCfg) -> torch.Tensor:
    """扫描值越小地面越高；台阶区域/边缘权重取较大者，避免相乘隐式改变损失尺度。"""
    weights = torch.ones_like(teacher_scan)
    if cfg.scan_step_weight == cfg.scan_edge_weight == 1.0:
        return weights
    raised = teacher_scan.amax(-1, keepdim=True) - teacher_scan >= cfg.scan_edge_threshold
    weights = torch.where(raised, cfg.scan_step_weight, weights)
    return torch.where(scan_edge_mask(teacher_scan, cfg), weights.clamp_min(cfg.scan_edge_weight), weights)


def student_losses(policy: ScanStudentPolicy, student_obs, teacher_obs, cfg: ScanStudentCfg):
    """扫描重建 + 动作对齐。

    动作标签用干净 teacher 观测；预测动作用学生自己的（可能带噪的）本体观测，
    与部署时输入一致。只靠扫描 MSE 时，Teacher 敏感的少数扫描点误差会被平均掉，
    动作项把这部分误差按 Teacher 的敏感度加权回来。
    """
    scan_pred = policy.encoder(student_obs)
    scan_target = teacher_obs[:, PROPRIO_DIM:TEACHER_OBS_DIM]
    scan_weights = scan_loss_weights(scan_target, cfg)
    scan_loss = ((scan_pred - scan_target).square() * scan_weights).sum() / scan_weights.sum()
    with torch.no_grad():
        action_target = policy.teacher_actor(teacher_obs)
    action_pred = policy.teacher_actor(policy.actor_observation(student_obs, scan_pred))
    action_weights = torch.ones_like(action_target)
    action_weights[:, :4] = cfg.arm_action_weight
    action_loss = ((action_pred - action_target).square() * action_weights).sum() / action_weights.sum()
    return scan_loss + cfg.action_loss_coef * action_loss, scan_loss, action_loss


def train_student_batch(policy, optimizer, student_obs, teacher_obs, cfg: ScanStudentCfg):
    """非有限损失或梯度立即终止，不能把污染后的编码器继续送进仿真器。"""
    optimizer.zero_grad(set_to_none=True)
    losses = student_losses(policy, student_obs, teacher_obs, cfg)
    if not all(torch.isfinite(value).all() for value in losses):
        raise FloatingPointError("深度学生loss非有限；已拒绝更新")
    losses[0].backward()
    torch.nn.utils.clip_grad_norm_(policy.encoder.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return tuple(value.detach() for value in losses)
