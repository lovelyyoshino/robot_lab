"""
networks.py — 世界模型、actor、critic 与在线归一化。

actor 的 MLP 层命名与 rsl_rl ActorCritic 的 `actor.N` 一致（45→512/256/128→12, ELU），
因此导出的权重可以直接放进现有 45 维 blind 部署链路；区别只在输出端多了一个
有界 tanh（见 Actor.forward），由 `export_actor` 一并写进 TorchScript。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-10-26
@changelog
    - v0.3.0 (2026-10-26): actor 支持 squash="clamp"，用于从 rsl_rl PPO 权重无损热启动。
    - v0.2.0 (2026-10-26): 世界模型接入动力学上下文 ξ 与关节加速度 aux 头；critic 读 ξ；actor 界改 2π。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import torch
from torch import nn

from .state import DELTA_DIM, FEATURE_DIM, NUM_ACTIONS, features, integrate


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), act()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """按批合并的均值/方差；只在 world model 更新前用真实数据刷新，不参与求导。"""

    def __init__(self, dim: int, eps: float = 1e-4, min_std: float = 1e-3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.min_std = min_std

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.count = total

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt().clamp_min(self.min_std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class WorldModel(nn.Module):
    """f(features, action, ξ) → 机体系增量，外加 aux 头预测本步 env 的 Σjoint_acc²。

    ξ（context）是每回合 reset 时随机化的执行器增益/质量等；策略看不到，但世界模型必须看到，
    否则它只能学到对随机化取平均的动力学，actor 会利用"平均模型"在个别回合上的系统偏差。
    """

    def __init__(self, hidden=(1024, 1024, 1024), context_dim: int = 0, aux_dim: int = 0, aux_hidden=(256, 256)):
        super().__init__()
        self.context_dim, self.aux_dim = context_dim, aux_dim
        self.in_dim = FEATURE_DIM + NUM_ACTIONS + context_dim
        self.input_norm = RunningNorm(self.in_dim)
        self.delta_norm = RunningNorm(DELTA_DIM)
        self.net = mlp(self.in_dim, hidden, DELTA_DIM, act=nn.SiLU)
        # aux 头独立于动力学主干：Σacc² 是重尾的冲击量，与增量共享输出层会挤占动力学精度。
        self.aux_norm = RunningNorm(max(aux_dim, 1))
        self.aux_net = mlp(self.in_dim, aux_hidden, aux_dim, act=nn.SiLU) if aux_dim else None

    def _inputs(self, state, action, context=None):
        parts = [features(state), action]
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError(f"世界模型需要 {self.context_dim} 维 context")
            parts.append(context)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def update_normalizers(self, state, action, delta, context=None, aux=None) -> None:
        self.input_norm.update(self._inputs(state, action, context))
        self.delta_norm.update(delta)
        if self.aux_dim and aux is not None:
            self.aux_norm.update(aux)

    def forward(self, state, action, context=None) -> torch.Tensor:
        """返回下一步状态；梯度可穿过 state 和 action。"""
        x = self.input_norm.normalize(self._inputs(state, action, context))
        delta = self.delta_norm.denormalize(self.net(x))
        return integrate(state, delta)

    def aux_normalized(self, state, action, context=None) -> torch.Tensor:
        return self.aux_net(self.input_norm.normalize(self._inputs(state, action, context)))

    def aux(self, state, action, context=None) -> torch.Tensor:
        """本步 Σacc²（物理单位，非负）；MSE 训练给出条件均值，正好是期望惩罚所需的量。"""
        return self.aux_norm.denormalize(self.aux_normalized(state, action, context)).clamp_min(0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 45, hidden=(512, 256, 128), action_bound: float = 6.2831853,
                 squash: str = "tanh"):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, NUM_ACTIONS)
        # 有界输出让想象 rollout 不会把动作推到世界模型从未见过的区域。
        # 界取 2π：轮速动作 ×5 后正好是 env 的轮速目标 clip ±31.4 rad/s。轮子是 DCMotor
        # （stiffness 0、damping 0.5、effort 10 N·m），扭矩 ≈ 0.5×(目标−实际)，实测 1 m/s 需 ~8.8 rad/s，
        # 要出满扭矩目标须 ≥ ~29 rad/s；原来的界 4（20 rad/s）只能出约一半扭矩。
        self.action_bound = action_bound
        # "clamp"：界内与 rsl_rl 线性输出完全相同，从 PPO 权重热启动时动作不变（tanh 会把 3 压成 2.3）；
        # 界外梯度为 0，与 env 自身的动作 clip 一致。
        if squash not in ("tanh", "clamp"):
            raise ValueError(f"未知 squash: {squash}")
        self.use_tanh = squash == "tanh"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor(obs)
        if self.use_tanh:
            return self.action_bound * torch.tanh(raw / self.action_bound)
        return raw.clamp(-self.action_bound, self.action_bound)


class Critic(nn.Module):
    """输入航向不变特征 + 命令 + 上一动作；价值不需要部署，可用特权量。"""

    def __init__(self, hidden=(512, 256, 128), context_dim: int = 0):
        super().__init__()
        self.in_dim = FEATURE_DIM + 3 + NUM_ACTIONS + context_dim
        self.norm = RunningNorm(self.in_dim)
        self.net = mlp(self.in_dim, hidden, 1)

    def inputs(self, state, command, last_action, context):
        return torch.cat((features(state), command, last_action, context), dim=-1)

    def forward(self, state, command, last_action, context) -> torch.Tensor:
        return self.net(self.norm.normalize(self.inputs(state, command, last_action, context))).squeeze(-1)
