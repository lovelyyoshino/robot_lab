"""不依赖仿真器的逐环境技能验收统计。

@author pony
@date 2026-09-16
@version v0.2.0
@last_modified 2026-09-16
@changelog
    - v0.2.0 (2026-09-16): 准备时钟与受控时钟分离，初始化失败不从尝试分母消失。
    - v0.1.0 (2026-09-16): 固定每环境配额，按终态连续保持时长验收，拒绝不完整评测。
"""

from __future__ import annotations

import math
from statistics import mean


class SkillEvaluation:
    """每次消费物理步结束、自动 reset 之前的状态；不接受 reset 后的初始姿态。"""

    def __init__(
        self, num_envs: int, episodes_per_env: int, dt: float,
        hold_time_s: float, acquisition_deadline_s: float,
    ):
        if num_envs < 1 or episodes_per_env < 1:
            raise ValueError("环境数与每环境 episode 数必须为正")
        if not all(math.isfinite(x) and x > 0 for x in (dt, hold_time_s, acquisition_deadline_s)):
            raise ValueError("控制周期、保持时间和进入稳定区时限必须为有限正数")
        self.num_envs = num_envs
        self.quota = episodes_per_env
        self.dt = dt
        self.hold_steps = math.ceil(hold_time_s / dt - 1e-9)
        self.deadline_steps = math.floor(acquisition_deadline_s / dt + 1e-9)
        self.counts = [0] * num_envs
        self.steps = [0] * num_envs
        self.streaks = [0] * num_envs
        self.invalid = [False] * num_envs
        self.preparation_steps = [0] * num_envs
        self.initialization_failed = [False] * num_envs
        self.records: list[dict] = []
        self.pitch_squared_error = 0.0
        self.pitch_samples = 0

    @property
    def complete(self) -> bool:
        return all(count == self.quota for count in self.counts)

    def update(self, stable, terminated, truncated, finite, pitch_errors=None, *, active=None, initialization_failed=None) -> None:
        """失败与 timeout 同时发生时仍判失败；配额已满的环境不再影响指标。"""
        batches = (stable, terminated, truncated, finite)
        if any(len(batch) != self.num_envs for batch in batches):
            raise ValueError("每组状态必须包含全部环境")
        if pitch_errors is not None and len(pitch_errors) != self.num_envs:
            raise ValueError("pitch 误差必须包含全部环境")
        active = [True] * self.num_envs if active is None else active
        initialization_failed = [False] * self.num_envs if initialization_failed is None else initialization_failed
        if len(active) != self.num_envs or len(initialization_failed) != self.num_envs:
            raise ValueError("准备/接管状态必须包含全部环境")
        for i in range(self.num_envs):
            if self.counts[i] >= self.quota:
                continue
            self.steps[i] += int(active[i])
            self.preparation_steps[i] += int(not active[i])
            self.initialization_failed[i] |= bool(initialization_failed[i])
            valid = bool(finite[i])
            self.invalid[i] |= not valid
            self.streaks[i] = self.streaks[i] + 1 if active[i] and valid and stable[i] else 0
            if active[i] and pitch_errors is not None and math.isfinite(pitch_errors[i]):
                self.pitch_squared_error += pitch_errors[i] ** 2
                self.pitch_samples += 1
            if not (terminated[i] or truncated[i]):
                continue
            # 首个稳定采样发生在一次完整物理步之后；只认可延续到终态的连续区间。
            start_step = self.steps[i] - self.streaks[i] + 1
            success = (
                bool(truncated[i]) and not bool(terminated[i]) and not self.invalid[i]
                and not self.initialization_failed[i] and self.steps[i] > 0
                and self.streaks[i] >= self.hold_steps and start_step <= self.deadline_steps
            )
            self.records.append({
                "env_id": i,
                "episode_index": self.counts[i],
                "success": success,
                "duration_s": self.steps[i] * self.dt,
                "preparation_time_s": self.preparation_steps[i] * self.dt,
                "initialization_failed": self.initialization_failed[i],
                "terminal_stable_time_s": self.streaks[i] * self.dt,
                "time_to_stable_s": start_step * self.dt if success else None,
                "time_to_confirmed_success_s": (
                    (start_step + self.hold_steps - 1) * self.dt if success else None
                ),
                "terminated": bool(terminated[i]),
                "truncated": bool(truncated[i]),
                "nonfinite_seen": self.invalid[i],
            })
            self.counts[i] += 1
            self.steps[i] = self.streaks[i] = 0
            self.invalid[i] = False
            self.preparation_steps[i] = 0
            self.initialization_failed[i] = False

    def summary(self, minimum_success_rate: float) -> dict:
        """中断/样本不足时 passed 必须为 False，不能只对已完成样本宣告通过。"""
        successes = [record for record in self.records if record["success"]]
        initialization_failures = sum(record["initialization_failed"] for record in self.records)
        initialized = len(self.records) - initialization_failures
        rate = len(successes) / len(self.records) if self.records else 0.0
        return {
            "complete": self.complete,
            "expected_episodes": self.num_envs * self.quota,
            "episodes": len(self.records),
            "episodes_by_env": self.counts.copy(),
            "success_rate": rate,
            "initialization_failure_count": initialization_failures,
            "conditional_takeover_success_rate": len(successes) / initialized if initialized else None,
            "mean_time_to_stable_s": mean(r["time_to_stable_s"] for r in successes) if successes else None,
            "mean_time_to_confirmed_success_s": (
                mean(r["time_to_confirmed_success_s"] for r in successes) if successes else None
            ),
            "pitch_rmse_deg": (
                math.degrees(math.sqrt(self.pitch_squared_error / self.pitch_samples))
                if self.pitch_samples else None
            ),
            "gates": {
                "minimum_success_rate": minimum_success_rate,
                "minimum_terminal_stable_time_s": self.hold_steps * self.dt,
                "maximum_time_to_stable_s": self.deadline_steps * self.dt,
            },
            "passed": self.complete and rate >= minimum_success_rate,
            "episode_results": self.records,
        }
