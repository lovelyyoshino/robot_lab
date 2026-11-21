"""
replay.py — 按 [时间, 环境] 存储的环形回放，供世界模型窗口训练和想象起点采样。

@author pony
@date 2026-10-26
@version v0.4.0
@last_modified 2026-11-04
@changelog
    - (2026-11-04): 窗口事件优先级只计有效前缀；状态采样仍允许终止动作前的真实状态。
    - v0.4.0 (2026-10-28): 每行可存优先级（台阶边/翻车等关键事件），窗口与起点按 priority_fraction 混合采样。
    - v0.3.0 (2026-10-26): 存真实奖励与真终止标记，供 critic 拟合真实回报。
    - v0.2.0 (2026-10-26): 新增 context（ξ）与 aux（env 关节加速度项）两列。
    - v0.1.0 (2026-10-26): 初始创建

第 t 行存"动作前"的状态、该步动作、上一动作（即观测里的 last_action）、命令、
动力学上下文 ξ（该回合的执行器增益/质量等），以及 done[t]=该步之后回合是否结束、
aux[t]=该步 env 实际算出的辅助监督量（关节加速度平方和，见 rover_task）。
done 之后的下一行是 reset 后的新状态，不是物理后继，所以任何跨过 done 的转移都必须掩掉；
终止和超时截断一视同仁。
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
        # 关键事件标记（0/1）：从零学越障时，台阶边和翻车片段只占回放很小一部分，均匀采样时世界模型在那里最不准
        self.priority = torch.zeros(*shape, device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, prev_action, command, done, context=None, aux=None, reward=None,
            terminated=None, priority=None) -> None:
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
        self.priority[i] = 0.0 if priority is None else priority.float()
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _physical(self, logical: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + logical) % self.capacity

    def _pick(self, batch: int, high: int, length: int, priority_fraction: float, valid_prefix: bool = True):
        """(逻辑起点, env)；priority_fraction 的样本从 4 倍候选里挑窗口内优先级最高的（go2w 事件优先回放的做法）。"""
        start = torch.randint(0, high, (batch,), device=self.device)
        env = torch.randint(0, self.num_envs, (batch,), device=self.device)
        n = int(batch * priority_fraction)
        if n <= 0:
            return start, env
        cs = torch.randint(0, high, (4 * n,), device=self.device)
        ce = torch.randint(0, self.num_envs, (4 * n,), device=self.device)
        rows = self._physical(cs[:, None] + torch.arange(max(length, 1), device=self.device)[None, :])
        cols = ce[:, None].expand_as(rows)
        priority = self.priority[rows, cols]
        if valid_prefix:
            # reset之后的事件不会参与该窗口监督，不能用它把无效窗口排到采样前列。
            valid = (~self.done[rows, cols]).float().cumprod(-1)
            priority = priority * valid
        score = priority.amax(-1) + 1e-3 * torch.rand(4 * n, device=self.device)
        top = score.topk(n).indices
        start[:n], env[:n] = cs[top], ce[top]
        return start, env

    def sample_windows(self, batch: int, length: int, priority_fraction: float = 0.0) -> dict[str, torch.Tensor]:
        """返回 k=length 步窗口；valid[:, j] 表示第 j 个转移及其之前都未跨回合。

        最后一行的后继状态尚未写入，所以起点最多到 size-1-length。
        """
        if self.size < length + 1:
            raise RuntimeError(f"回放仅 {self.size} 行，不足以采样 {length} 步窗口")
        start, env = self._pick(batch, self.size - length, length, priority_fraction)
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

    def sample_states(self, batch: int, priority_fraction: float = 0.0) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("回放为空")
        start, cols = self._pick(batch, self.size, 1, priority_fraction, valid_prefix=False)
        rows = self._physical(start)
        return {
            "state": self.state[rows, cols],
            "prev_action": self.prev_action[rows, cols],
            "command": self.command[rows, cols],
            "context": self.context[rows, cols],
        }
