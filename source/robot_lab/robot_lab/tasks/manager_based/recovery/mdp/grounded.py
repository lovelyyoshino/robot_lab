"""接地恢复奖励：连续翻正信号与真实四轮承重，保留58维输入合同。

@author pony
@date 2026-09-17
@version v0.4.0
@last_modified 2026-09-29
@changelog
    - v0.5.0 (2026-09-30): 增加弹射窗口正确方向滚转率和向上速度奖励。
    - v0.4.0 (2026-09-29): 阶段任务支持按秒速率计算进展，分离腿/轮速度成本。
    - v0.3.0 (2026-09-17): 进展 shaping 改为有界有符号势能差，加入局部 reset 与重复读取缓存。
    - v0.2.0 (2026-09-17): 增加逐控制步去重的姿态/高度正向进展 shaping。
    - v0.1.0 (2026-09-17): 避免侧躺靠高度/默认关节获高分，准备期逐环境归零。
"""

import math

import torch


def _active_value(env, value):
    active = env._recovery_control_active
    finite = torch.isfinite(value)
    return torch.where(active & finite, value, 0.0)


def _leg_ids(env):
    # 复位已按真实关节名建立分组；不按20自由度内部顺序猜主动关节。
    return env._recovery_joint_groups["active_leg"]


def recovery_orientation_score(env):
    """从仰躺到正立连续增分；侧躺不会像窄高斯方向奖励一样几乎没有信号。"""
    gravity = env.scene["robot"].data.projected_gravity_b
    score = (1.0 - gravity[:, 2]).clamp(0, 2) * 0.5
    return _active_value(env, torch.where(torch.isfinite(gravity).all(-1), score, 0.0))


def reset_grounded_progress(env, env_ids=None) -> None:
    """同时清理所有进展项的指定行；同一控制步 reset 后也不能返回旧奖励。"""
    ids = slice(None) if env_ids is None else env_ids
    for entry in getattr(env, "_grounded_progress_state", {}).values():
        entry["initialized"][ids] = False
        entry["previous"][ids] = 0.0
        entry["reward"][ids] = 0.0


def _progress_physical_finite(env):
    data = env.scene["robot"].data
    forces = env.scene.sensors["contact_forces"].data.net_forces_w
    return (
        torch.isfinite(data.root_state_w).all(-1)
        & torch.isfinite(data.projected_gravity_b).all(-1)
        & torch.isfinite(data.joint_pos).all(-1)
        & torch.isfinite(data.joint_vel).all(-1)
        & torch.isfinite(forces).all(dim=(1, 2))
        & torch.isfinite(env.scene.env_origins).all(-1)
    )


def _progress_reward(env, key: str, potential: torch.Tensor) -> torch.Tensor:
    """连续控制步间的有符号势能差；调用方先约束势能，差分本身不截断。

    同一步重复读取返回缓存副本。准备、非有限状态、漏采样或时间倒退会
    中断历史；局部 reset 由 initialize_settled_recovery 显式清理全部项。
    终止步计入最后一次真实变化后清历史，不把下一 episode 当连续动作。
    这是工程 shaping：完整连续轨迹的未折扣往返和为零，但没有引入 gamma
    和吸收终态，不能声称折扣回报下策略不变或任意截断轨迹不可刷分。
    """
    if potential.ndim != 1 or potential.shape[0] != env.num_envs:
        raise ValueError("进展奖励输入必须是逐环境一维张量")
    state = getattr(env, "_grounded_progress_state", None)
    if state is None:
        state = {}
        env._grounded_progress_state = state
    entry = state.get(key)
    if entry is None:
        entry = {
            "previous": torch.zeros_like(potential),
            "initialized": torch.zeros(env.num_envs, dtype=torch.bool, device=potential.device),
            "reward": torch.zeros_like(potential),
            "last_step": -1,
        }
        state[key] = entry

    finite = torch.isfinite(potential) & _progress_physical_finite(env)
    active = env._recovery_control_active & finite
    step = int(env.common_step_counter)
    if entry["last_step"] == step:
        return entry["reward"].clone()

    if step != entry["last_step"] + 1:
        entry["initialized"].zero_()
    value = torch.where(finite, potential.detach(), 0.0)
    reward = torch.where(active & entry["initialized"], value - entry["previous"], 0.0)
    entry["previous"].copy_(value)
    terminal = getattr(env, "reset_buf", torch.zeros_like(active))
    entry["initialized"].copy_(active & ~terminal.bool())
    entry["reward"].copy_(reward)
    entry["last_step"] = step
    return reward.clone()


def recovery_orientation_progress(env, gain: float = 8.0, as_rate: bool = False) -> torch.Tensor:
    """势能为 gain*(1-g_z)，限定在 [0,2*gain]；倒退计负分。"""
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("orientation progress gain 必须为有限正数")
    gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    potential = (1.0 - gravity_z).clamp(0.0, 2.0) * gain
    return _progress_units(env, _progress_reward(env, "orientation_gz", potential), as_rate)


def recovery_height_progress(
    env, scale: float = 0.05, target_height: float = 0.332, as_rate: bool = False,
) -> torch.Tensor:
    """正立方向门乘以有界高度势能；侧躺抬高和超过目标高度不继续加分。"""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("height progress scale 必须为有限正数")
    if not math.isfinite(target_height) or target_height <= 0.0:
        raise ValueError("height progress target_height 必须为有限正数")
    data = env.scene["robot"].data
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    potential = _orientation_gate(env) * height.clamp(0.0, target_height) / scale
    return _progress_units(env, _progress_reward(env, "height", potential), as_rate)


def _progress_units(env, value, as_rate):
    # Isaac RewardManager统一乘step_dt；速率模式抵消该缩放，旧任务默认行为不变。
    if not as_rate:
        return value
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
        raise ValueError("进展速率要求有限正step_dt")
    return value / env.step_dt


def _phase_velocity_cost(env, ids, standing_multiplier):
    if not math.isfinite(standing_multiplier) or standing_multiplier < 1:
        raise ValueError("standing_multiplier 必须有限且不小于1")
    speed_squared = env.scene["robot"].data.joint_vel[:, ids].square().sum(-1)
    multiplier = 1. + (standing_multiplier - 1.) * _orientation_gate(env)
    return _active_value(env, speed_squared * multiplier)


def recovery_leg_velocity(env, standing_multiplier: float = 5.0):
    """只统计8个主动腿；恢复早期允许运动，翻正后连续加强平稳成本。"""
    return _phase_velocity_cost(env, _leg_ids(env), standing_multiplier)


def recovery_phase_wheel_velocity(env, standing_multiplier: float = 5.0):
    """四轮只在本项计费，避免与腿速度项重复；硬速度/力矩限制保持不变。"""
    return _phase_velocity_cost(env, env._recovery_joint_groups["wheel"], standing_multiplier)


def _launch_window(env, duration_s: float) -> torch.Tensor:
    """返回接管后的早期发力窗口；准备期和窗口外都严格为零。"""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("launch window 必须是有限正数")
    if not hasattr(env, "_recovery_control_steps"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    phase = env._recovery_control_steps.float() * env.step_dt
    return env._recovery_control_active & (phase < duration_s)


def recovery_launch_roll_rate(env, duration_s: float = 1.2, max_rate: float = 4.0) -> torch.Tensor:
    """弹射窗口内的正确方向滚转角速度奖励。

    直接奖励 ``ω×g`` 导致的正立势能增加，而不是奖励无方向的高速旋转：
    ``d(1-g_z)/dt ≈ ω_x g_y - ω_y g_x``。这与初始左右倒向无关，能避免
    策略通过反向空转刷分，同时给“机械臂摆动 + 轮子发力”一个短暂动力学梯度。
    """
    if not math.isfinite(max_rate) or max_rate <= 0.0:
        raise ValueError("max_rate 必须是有限正数")
    data = env.scene["robot"].data
    gravity = data.projected_gravity_b
    omega = data.root_ang_vel_b
    rate = omega[:, 0] * gravity[:, 1] - omega[:, 1] * gravity[:, 0]
    value = torch.relu(rate).clamp(max=max_rate) / max_rate
    finite = torch.isfinite(rate) & torch.isfinite(value)
    return _active_value(env, torch.where(_launch_window(env, duration_s) & finite, value, 0.0))


def recovery_launch_vertical_speed(env, duration_s: float = 1.2, max_speed: float = 1.0) -> torch.Tensor:
    """弹射窗口内有限的向上速度奖励；只奖励正速度并截断，避免鼓励持续跳跃。"""
    if not math.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError("max_speed 必须是有限正数")
    data = env.scene["robot"].data
    speed = torch.relu(data.root_lin_vel_w[:, 2]).clamp(max=max_speed) / max_speed
    finite = torch.isfinite(speed)
    return _active_value(env, torch.where(_launch_window(env, duration_s) & finite, speed, 0.0))


def _orientation_gate(env):
    return (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0, 1).square()


def recovery_oriented_height(env, target_height=0.332, sigma=0.15):
    """只有开始朝正立倾斜后才奖励站高；侧躺高度接近目标也不能刷高度分。"""
    height = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return _active_value(env, _orientation_gate(env) * torch.exp(-((height - target_height) / sigma).square()))


def recovery_oriented_joints(env, sigma=0.5):
    """默认腿姿只在正立方向奖励；恢复早期允许改变腿姿产生翻转力矩。"""
    data = env.scene["robot"].data
    error = data.joint_pos[:, _leg_ids(env)] - data.default_joint_pos[:, _leg_ids(env)]
    return _active_value(env, _orientation_gate(env) * torch.exp(-error.square().sum(-1) / sigma**2))


def recovery_four_wheel_support(env, contact_threshold=1.0):
    """用接触传感器本身的名字解析四轮，要求无非轮承重及正常姿态/高度。"""
    sensor = env.scene.sensors["contact_forces"]
    names = sensor.body_names
    wheels = ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel")
    if len(set(names)) != len(names) or not all(name in names for name in wheels):
        raise ValueError("恢复接触传感器必须包含唯一的四轮名称")
    force = sensor.data.net_forces_w
    if force.shape[1:] != (len(names), 3):
        raise ValueError("恢复接触力与传感器名称数量不一致")
    contacts = force.norm(dim=-1) > contact_threshold
    wheel_ids = [names.index(name) for name in wheels]
    other_ids = [i for i, name in enumerate(names) if name not in wheels]
    data = env.scene["robot"].data
    gravity = data.projected_gravity_b
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    finite = torch.isfinite(force).all(dim=(1, 2)) & torch.isfinite(gravity).all(-1) & torch.isfinite(height)
    supported = contacts[:, wheel_ids].all(-1) & ~contacts[:, other_ids].any(-1)
    supported &= finite & (gravity[:, 2] < -0.7) & (height >= 0.26) & (height <= 0.45)
    return _active_value(env, supported.float())


def recovery_stable_hold(env):
    """四轮承重后再要求根速度和主动腿速度足够低；轮子绝对角不参加判据。"""
    data = env.scene["robot"].data
    stable = recovery_four_wheel_support(env).bool()
    stable &= data.root_lin_vel_w.norm(dim=-1) < 0.15
    stable &= data.root_ang_vel_w.norm(dim=-1) < 0.75
    stable &= data.joint_vel[:, _leg_ids(env)].abs().amax(-1) < 0.5
    stable &= torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.joint_vel).all(-1)
    return _active_value(env, stable.float())
