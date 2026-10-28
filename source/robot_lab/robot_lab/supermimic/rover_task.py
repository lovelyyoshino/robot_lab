"""
rover_task.py — 平地速度跟踪任务的解析观测、奖励与软存活，供想象 rollout 求导。

观测必须与 `ParallelogramRoverRoughEnvCfg` 的 45 维 policy 组逐列一致
（ang_vel*0.25, gravity, cmd, joint_pos_rel 轮位置置零, joint_vel*0.05, last_action），
否则想象中训练的 actor 部署到仿真/RL-SAR 时输入语义不同；`isaac_adapter` 冒烟时会
逐列核对。奖励只用状态可算的项，不含接触力（接触力不报在 *_wheel 上，且世界模型不预测它）。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-10-26
@changelog
    - v0.2.0 (2026-10-26): 关节加速度三项改由 acc_sq（世界模型 aux 头 / env 真值）提供；观测按 ±100 截断。
    - v0.1.1 (2026-10-26): lin_vel_z / ang_vel_xy 补乘正立门控，与 robot_lab 实现逐项一致（verify_pipeline 核对）。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .quat import projected_gravity
from .state import ANG, JPOS, JVEL, LIN, NUM_LEG_JOINTS, QUAT

OBS_DIM = 45
COMMAND_DIM = 3
CONTROL_DT = 0.02  # sim.dt 0.005 × decimation 4
ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
SUSPENSION = slice(0, 4)
STEERING = slice(4, 8)
WHEELS = slice(8, 12)
OBS_CLIP = 100.0  # 与 velocity_env_cfg 各 policy ObsTerm 的 clip=(-100, 100) 一致
# 世界模型 aux 头的监督量 = 这三个 env 奖励项的未加权值（Σ joint_acc²），顺序固定。
ACC_AUX_NAMES = ("joint_acc_l2", "joint_acc_steering_l2", "joint_acc_wheel_l2")
AUX_DIM = len(ACC_AUX_NAMES)


def policy_observation(state: torch.Tensor, command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    joint_pos = state[..., JPOS].clone()
    joint_pos[..., NUM_LEG_JOINTS:] = 0.0
    return torch.cat(
        (
            state[..., ANG] * ANG_VEL_SCALE,
            projected_gravity(state[..., QUAT]),
            command,
            joint_pos,
            state[..., JVEL] * JOINT_VEL_SCALE,
            last_action,
        ),
        dim=-1,
    ).clamp(-OBS_CLIP, OBS_CLIP)


@dataclass
class RoverRewardCfg:
    """权重沿用 rough 基线中对平地有意义的项；是近似复刻，不保证与 env reward 数值相等。"""

    track_lin_vel: float = 5.0
    track_lin_vel_std: float = 0.5
    track_ang_vel: float = 1.5
    track_ang_vel_std: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    action_rate: float = -0.005
    suspension_pos: float = -0.25
    steering_deviation: float = -0.1
    steering_vel: float = -2.0e-3
    # joint_acc：ArticulationData.update 每个 5 ms 物理子步都读一次 joint_acc，所以 env 惩罚的是
    # 最后一个子步的瞬时加速度（含接触冲击），控制步 Δv/0.02 只有它的约 1/10。世界模型因此
    # 额外带一个 aux 头，直接回归 env 算出的 Σacc²（ACC_AUX_NAMES），想象时用它代替差分代理。
    suspension_acc: float = -2.5e-7
    steering_acc: float = -5.0e-7
    wheel_acc: float = -2.5e-9
    upward: float = 0.10
    # 软存活：倾角越过 tilt_limit 时存活概率按 sigmoid 下降；与仿真硬终止阈值一致。
    tilt_limit: float = 1.0
    tilt_temperature: float = 0.05


def _upright_gate(gravity: torch.Tensor) -> torch.Tensor:
    # 与 robot_lab track_*_exp 相同：机体翻倒时不给跟踪奖励，防止躺着刷分。
    return torch.clamp(-gravity[..., 2], 0.0, 0.7) / 0.7


def reward_terms(state, next_state, command, action, last_action, cfg: RoverRewardCfg,
                 acc_sq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """在 next_state 上评估（与 Isaac Lab 在 step 之后算奖励的时序相同）。

    acc_sq: [..., 3] = 悬挂/转向/轮子的 Σ(joint_acc²)，来自世界模型 aux 头或 env 真值；
    缺省时退回控制步差分代理（只用于没有 aux 头的场合，量级偏小）。
    """
    gravity = projected_gravity(next_state[..., QUAT])
    gate = _upright_gate(gravity)
    lin = next_state[..., LIN]
    ang = next_state[..., ANG]
    lin_err = (command[..., :2] - lin[..., :2]).square().sum(-1)
    ang_err = (command[..., 2] - ang[..., 2]).square()
    joint_pos = next_state[..., JPOS]
    if acc_sq is None:
        acc_sq = proxy_acc_sq(state, next_state)
    return {
        "track_lin_vel": cfg.track_lin_vel * torch.exp(-lin_err / cfg.track_lin_vel_std**2) * gate,
        "track_ang_vel": cfg.track_ang_vel * torch.exp(-ang_err / cfg.track_ang_vel_std**2) * gate,
        "lin_vel_z": cfg.lin_vel_z * lin[..., 2].square() * gate,
        "ang_vel_xy": cfg.ang_vel_xy * ang[..., :2].square().sum(-1) * gate,
        "action_rate": cfg.action_rate * (action - last_action).square().sum(-1),
        "suspension_pos": cfg.suspension_pos * joint_pos[..., SUSPENSION].norm(dim=-1) * gate,
        "steering_deviation": cfg.steering_deviation * joint_pos[..., STEERING].abs().sum(-1),
        "steering_vel": cfg.steering_vel * next_state[..., JVEL][..., STEERING].square().sum(-1),
        "suspension_acc": cfg.suspension_acc * acc_sq[..., 0],
        "steering_acc": cfg.steering_acc * acc_sq[..., 1],
        "wheel_acc": cfg.wheel_acc * acc_sq[..., 2],
        "upward": cfg.upward * (1.0 - gravity[..., 2]).square(),
    }


def proxy_acc_sq(state, next_state) -> torch.Tensor:
    acc = (next_state[..., JVEL] - state[..., JVEL]) / CONTROL_DT
    return torch.stack(
        (acc[..., SUSPENSION].square().sum(-1), acc[..., STEERING].square().sum(-1), acc[..., WHEELS].square().sum(-1)),
        dim=-1,
    )


def reward(state, next_state, command, action, last_action, cfg: RoverRewardCfg,
           acc_sq: torch.Tensor | None = None) -> torch.Tensor:
    terms = reward_terms(state, next_state, command, action, last_action, cfg, acc_sq)
    # Isaac Lab RewardManager 把加权和乘以 step_dt；保持同一量纲便于和 PPO 曲线对照。
    return torch.stack(list(terms.values()), dim=-1).sum(-1) * CONTROL_DT


def tilt(state: torch.Tensor) -> torch.Tensor:
    """仅用于判定和日志；acos 在正立（cos=1）处梯度无穷，不能出现在可导路径上。"""
    return torch.acos(torch.clamp(-projected_gravity(state[..., QUAT])[..., 2], -1.0, 1.0))


def survival(state: torch.Tensor, cfg: RoverRewardCfg) -> torch.Tensor:
    """单步存活概率，可导；替代 bad_orientation 硬终止。

    在 cos 空间比较（-g_z 与 cos(tilt_limit)），与按角度判定等价，但处处梯度有限。
    温度换算到 cos 空间：d cos/dθ = -sin θ，因此除以 sin(tilt_limit)。
    """
    cos_tilt = -projected_gravity(state[..., QUAT])[..., 2]
    cos_limit = math.cos(cfg.tilt_limit)
    return torch.sigmoid((cos_tilt - cos_limit) / (cfg.tilt_temperature * math.sin(cfg.tilt_limit)))


def hard_terminated(state: torch.Tensor, cfg: RoverRewardCfg) -> torch.Tensor:
    return tilt(state) > cfg.tilt_limit


class FlatTask:
    """learner 的任务接口（平地速度跟踪）：policy_obs / reward / survival / critic_extra。"""

    obs_dim = OBS_DIM
    critic_extra_dim = 0

    def __init__(self, cfg: RoverRewardCfg):
        self.cfg = cfg

    def policy_obs(self, state, command, last_action):
        return policy_observation(state, command, last_action)

    def reward_terms(self, state, next_state, command, action, last_action, acc_sq=None):
        return reward_terms(state, next_state, command, action, last_action, self.cfg, acc_sq)

    def reward(self, state, next_state, command, action, last_action, acc_sq=None):
        return reward(state, next_state, command, action, last_action, self.cfg, acc_sq)

    def survival(self, state):
        return survival(state, self.cfg)

    def critic_extra(self, state):
        return state[..., :0]


def sample_commands(num: int, device, lin_x=(0.4, 1.0), lin_y=(-0.15, 0.15), ang_z=(-0.5, 0.5)) -> torch.Tensor:
    lo = torch.tensor((lin_x[0], lin_y[0], ang_z[0]), device=device)
    hi = torch.tensor((lin_x[1], lin_y[1], ang_z[1]), device=device)
    return lo + (hi - lo) * torch.rand(num, 3, device=device)
