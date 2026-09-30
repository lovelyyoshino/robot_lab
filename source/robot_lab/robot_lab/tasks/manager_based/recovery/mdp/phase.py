"""侧躺恢复的接管时钟和锁定倒向；不使用全局训练时钟或当前翻滚方向。

@author pony
@date 2026-09-29
@version v0.1.0
@last_modified 2026-09-29
@changelog
    - v0.1.0 (2026-09-29): 增加可部署的阶段/初始倒向观测，准备期输出零。
"""

import math

import torch


def recovery_phase(env, duration_s: float = 3.0):
    """前三秒从0递增到1，随后保持1；reset/接管使用逐环境控制步数。"""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s 必须为有限正数")
    # ObservationManager在首次reset之前推断形状，此时接管状态尚未建立。
    if not hasattr(env, "_recovery_control_steps"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    phase = (env._recovery_control_steps.float() * env.step_dt / duration_s).clamp(0., 1.)
    return torch.where(env._recovery_control_active, phase, 0.).unsqueeze(-1)


def recovery_initial_side(env):
    """接管时IMU重力y分量的符号；翻正过程中不能重新选择左右镜像。"""
    if not hasattr(env, "_recovery_takeover_gravity_b"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    side = env._recovery_takeover_gravity_b[:, 1].sign()
    return torch.where(env._recovery_control_active, side, 0.).unsqueeze(-1)
