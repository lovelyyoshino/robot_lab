"""dagger_distillation.py — Preview student 的 DAgger 混合蒸馏与学习率衰减。

原生 Distillation 只在 student 自己能到达的状态上收集 teacher 标签；student 不会大幅抬轮时，
teacher 抬轮后才出现的越障状态几乎采不到（实测这些状态误差是自驱状态的3.5倍）。
这里按环境、按回合以衰减概率改由 teacher 动作驱动，监督目标仍是同一状态下的 teacher 动作，
损失、网络和 checkpoint 结构与原生实现一致，play/评测只走 act_inference，不受混合影响。

@author pony
@date 2026-10-09
@version v1.0.0
@last_modified 2026-10-09
@changelog
    - v1.0.0 (2026-10-09): 初始创建；回合级 teacher 驱动混合、线性学习率衰减、每轮更新数可控。
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation


def linear_schedule(start: float, end: float, progress: int, horizon: int) -> float:
    """progress 超过 horizon 后保持终值；horizon<=0 表示不衰减。"""
    if horizon <= 0:
        return start
    fraction = min(max(progress, 0) / horizon, 1.0)
    return start + (end - start) * fraction


class DaggerDistillation(Distillation):
    """回合级混合：同一回合要么全由 teacher 驱动、要么全由 student 驱动。

    逐步切换会把两个策略的动作拼成第三种不连贯行为，产生两者都不会访问的状态，
    因此只在环境 reset 时重抽驱动者。
    """

    def __init__(self, policy, *, teacher_drive_start: float = 0.5, teacher_drive_end: float = 0.0,
                 teacher_drive_decay_updates: int = 4000, learning_rate_end: float | None = None,
                 learning_rate_decay_updates: int = 0, **kwargs):
        super().__init__(policy, **kwargs)
        if not 0.0 <= teacher_drive_end <= teacher_drive_start <= 1.0:
            raise ValueError("teacher 驱动比例必须满足 0 <= end <= start <= 1")
        if learning_rate_end is not None and not 0.0 < learning_rate_end <= self.learning_rate:
            raise ValueError("learning_rate_end 必须为正且不高于初始学习率")
        self.teacher_drive_start = teacher_drive_start
        self.teacher_drive_end = teacher_drive_end
        self.teacher_drive_decay_updates = teacher_drive_decay_updates
        self.learning_rate_start = self.learning_rate
        self.learning_rate_end = self.learning_rate if learning_rate_end is None else learning_rate_end
        self.learning_rate_decay_updates = learning_rate_decay_updates
        self.teacher_driven: torch.Tensor | None = None
        self._teacher_samples: torch.Tensor | None = None
        self._total_samples = 0

    @property
    def teacher_drive_probability(self) -> float:
        return linear_schedule(self.teacher_drive_start, self.teacher_drive_end,
                               self.num_updates, self.teacher_drive_decay_updates)

    def _resample(self, mask: torch.Tensor) -> None:
        if self.teacher_driven is None:
            raise RuntimeError("必须先初始化每个环境的驱动者状态")
        draws = torch.rand(int(mask.sum()), device=mask.device) < self.teacher_drive_probability
        self.teacher_driven[mask] = draws

    def act(self, obs):
        student_actions = self.policy.act(obs).detach()
        teacher_actions = self.policy.evaluate(obs).detach()
        if self.teacher_driven is None:
            self.teacher_driven = torch.zeros(student_actions.shape[0], dtype=torch.bool, device=student_actions.device)
            self._resample(torch.ones_like(self.teacher_driven))
        # 监督标签始终是 teacher 动作；只改变谁驱动仿真，即改变采到哪些状态。
        self.transition.actions = torch.where(self.teacher_driven[:, None], teacher_actions, student_actions)
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        driven_count = self.teacher_driven.sum()
        self._teacher_samples = driven_count if self._teacher_samples is None else self._teacher_samples + driven_count
        self._total_samples += student_actions.shape[0]
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        super().process_env_step(obs, rewards, dones, extras)
        done = dones.view(-1).bool()
        if self.teacher_driven is not None and done.any():
            self._resample(done)

    def update(self):
        # 先按本轮进度设学习率；resume 时 train.py 会把 num_updates 对齐到 checkpoint 迭代号。
        learning_rate = linear_schedule(self.learning_rate_start, self.learning_rate_end,
                                        self.num_updates, self.learning_rate_decay_updates)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.learning_rate = learning_rate
        probability = self.teacher_drive_probability
        result = super().update()
        result["teacher_drive_probability"] = probability
        result["teacher_driven_fraction"] = (
            self._teacher_samples.item() / self._total_samples if self._total_samples else 0.0
        )
        self._teacher_samples = None
        self._total_samples = 0
        return result


def register() -> None:
    """DistillationRunner 用 eval(class_name) 在其模块命名空间解析算法类，需显式注册。"""
    from rsl_rl.runners import distillation_runner

    distillation_runner.DaggerDistillation = DaggerDistillation
