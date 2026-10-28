"""
replay.py — 按 [时间, 环境] 存储的环形回放，供世界模型窗口训练和想象起点采样。

第 t 行存"动作前"的状态、该步动作、上一动作（即观测里的 last_action）、命令、
动力学上下文 ξ（该回合的执行器增益/质量等），以及 done[t]=该步之后回合是否结束、
aux[t]=该步 env 实际算出的辅助监督量（关节加速度平方和，见 rover_task）。
done 之后的下一行是 reset 后的新状态，不是物理后继，所以任何跨过 done 的转移都必须掩掉；
终止和超时截断一视同仁。

@author pony
@date 2026-10-26
@version v0.3.0
@last_modified 2026-10-26
@changelog
    - v0.3.0 (2026-10-26): 存真实奖励与真终止标记，供 critic 拟合真实回报。
    - v0.2.0 (2026-10-26): 新增 context（ξ）与 aux（env 关节加速度项）两列。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import torch

from .state import NUM_ACTIONS, STATE_DIM


class ReplayBuffer:
    def __init__(self, capacity: int, num_envs: int, device, command_dim: int = 3, context_dim: int = 0,
                 aux_dim: int = 0):
        self.capacity, self.num_envs, self.device = capacity, num_envs, torch.device(device)
        shape = (capacity, num_envs)
        self.state = torch.zeros(*shape, STATE_DIM, device=self.device)
        self.action = torch.zeros(*shape, NUM_ACTIONS, device=self.device)
        self.prev_action = torch.zeros(*shape, NUM_ACTIONS, device=self.device)
        self.command = torch.zeros(*shape, command_dim, device=self.device)
        self.context = torch.zeros(*shape, context_dim, device=self.device)
        self.aux = torch.zeros(*shape, aux_dim, device=self.device)
        self.done = torch.ones(*shape, dtype=torch.bool, device=self.device)
        # 真实 env 奖励与"真终止"（区别于超时/成功截断），供 critic 在真实回报上拟合。
        self.reward = torch.zeros(*shape, device=self.device)
        self.terminated = torch.zeros(*shape, dtype=torch.bool, device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, prev_action, command, done, context=None, aux=None, reward=None,
            terminated=None) -> None:
        i = self.ptr
        self.state[i] = state
        self.action[i] = action
        self.prev_action[i] = prev_action
        self.command[i] = command
        if context is not None:
            self.context[i] = context
        if aux is not None:
            self.aux[i] = aux
        self.done[i] = done.bool()
        self.reward[i] = 0.0 if reward is None else reward
        self.terminated[i] = done.bool() if terminated is None else terminated.bool()
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _physical(self, logical: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + logical) % self.capacity

    def sample_windows(self, batch: int, length: int) -> dict[str, torch.Tensor]:
        """返回 k=length 步窗口；valid[:, j] 表示第 j 个转移及其之前都未跨回合。

        最后一行的后继状态尚未写入，所以起点最多到 size-1-length。
        """
        if self.size < length + 1:
            raise RuntimeError(f"回放仅 {self.size} 行，不足以采样 {length} 步窗口")
        start = torch.randint(0, self.size - length, (batch,), device=self.device)
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        offsets = torch.arange(length + 1, device=self.device)
        rows = self._physical(start[:, None] + offsets[None, :])  # [B, k+1]
        cols = env[:, None].expand_as(rows)
        r, c = rows[:, :-1], cols[:, :-1]
        valid = torch.cumprod((~self.done[r, c]).float(), dim=1)
        return {
            "state": self.state[rows, cols],  # [B, k+1, S]
            "action": self.action[r, c],
            "prev_action": self.prev_action[r, c],
            "command": self.command[r, c],
            "context": self.context[r, c],
            "aux": self.aux[r, c],
            "reward": self.reward[r, c],
            "done": self.done[r, c],
            "terminated": self.terminated[r, c],
            "valid": valid,  # [B, k]
        }

    def sample_states(self, batch: int) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("回放为空")
        rows = self._physical(torch.randint(0, self.size, (batch,), device=self.device))
        cols = torch.randint(0, self.num_envs, (batch,), device=self.device)
        return {
            "state": self.state[rows, cols],
            "prev_action": self.prev_action[rows, cols],
            "command": self.command[rows, cols],
            "context": self.context[rows, cols],
        }
