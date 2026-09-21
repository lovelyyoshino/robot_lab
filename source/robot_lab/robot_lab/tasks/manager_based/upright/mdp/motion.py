"""双后轮姿态的地面运动指令、真实支撑奖励与成功课程。

@author pony
@date 2026-09-16
@version v0.7.0
@last_modified 2026-09-17
@changelog
    - v0.7.0 (2026-09-17): 将零前进负误差改为正向质量分数，避免提前跌倒规避惩罚。
    - v0.6.0 (2026-09-17): 增加零前进速度稠密误差项，并显式扩大漂移惩罚有效区间。
    - v0.5.0 (2026-09-17): 将静止世界位移纳入课程成功门，命令切换清零连续成功计时。
    - v0.4.0 (2026-09-17): 增加按环境 reset/停车切换管理的世界 XY 漂移锚定项。
    - v0.3.0 (2026-09-17): 静止课程不原地污染姿态门，构造时校验静止采样比例。
    - v0.2.0 (2026-09-17): 在运动阶段显式采样静止指令，并对静止跟踪及课程使用更严格的漂移门。
    - v0.1.0 (2026-09-16): 追加速度/yaw 指令，避免 90 度时机体前向轴退化。
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .support import current_support


MOTION_RANGES = (
    ((0.0, 0.0), (0.0, 0.0)),
    ((0.0, 0.3), (-0.5, 0.5)),
    ((-0.3, 0.5), (-1.0, 1.0)),
)


def ground_heading(quaternion: torch.Tensor, eps: float = 1e-5):
    """返回地面 forward/lateral 与有效门；完全侧翻时输出零，不能伪造航向。

    wxyz 旋转的 body-Y 先投影到水平面，再以 lateral × world-up 定义前进。
    body-X 在 90 度直立时竖直，不可用于定义平面前进速度。
    """
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("四元数必须是 [N,4] 的 wxyz")
    finite = torch.isfinite(quaternion).all(-1)
    safe = torch.where(finite[:, None], quaternion, torch.zeros_like(quaternion))
    norm = safe.norm(dim=-1, keepdim=True)
    w, x, y, z = (safe / norm.clamp_min(eps)).unbind(-1)
    lateral = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), torch.zeros_like(w)), -1)
    horizontal_norm = lateral.norm(dim=-1, keepdim=True)
    valid = finite & (norm[:, 0] > eps) & (horizontal_norm[:, 0] > eps)
    lateral = torch.where(valid[:, None], lateral / horizontal_norm.clamp_min(eps), 0.0)
    forward = torch.stack((lateral[:, 1], -lateral[:, 0], torch.zeros_like(w)), -1)
    return forward, lateral, valid


def ground_motion(quaternion: torch.Tensor, linear_world: torch.Tensor, angular_world: torch.Tensor):
    """返回 [地面 vx, 地面 vy, 世界 yaw]、pitch/roll 角速率及有效门。

    稳定性只限制世界水平角速度，允许策略绕世界 Z 转向；退化/非有限输入不通过门。
    """
    forward, lateral, valid = ground_heading(quaternion)
    if linear_world.shape != forward.shape or angular_world.shape != forward.shape:
        raise ValueError("根速度必须与四元数批量对齐，形状 [N,3]")
    valid &= torch.isfinite(linear_world).all(-1) & torch.isfinite(angular_world).all(-1)
    velocity = torch.stack(((linear_world * forward).sum(-1), (linear_world * lateral).sum(-1), angular_world[:, 2]), -1)
    tilt_rate = angular_world[:, :2].norm(dim=-1)
    return torch.where(valid[:, None], velocity, 0.0), torch.where(valid, tilt_rate, 0.0), valid


def ground_velocity_observation(env) -> torch.Tensor:
    """供部署/诊断使用；不替换旧 42 维前缀中的机体系 base_lin_vel。"""
    data = env.scene["robot"].data
    return ground_motion(data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w)[0]


def stationary_command_mask(command: torch.Tensor) -> torch.Tensor:
    """仅精确零指令进入静止合同，避免极小的运动命令被误判为停车。"""
    return torch.isfinite(command).all(-1) & (command == 0).all(-1)


def sample_motion_commands(num_envs: int, stage: int, device="cpu", stationary_fraction=0.0) -> torch.Tensor:
    if stage not in range(len(MOTION_RANGES)):
        raise ValueError("motion stage 必须为 0、1 或 2")
    if not math.isfinite(stationary_fraction) or not 0 <= stationary_fraction <= 1:
        raise ValueError("静止指令比例必须为 [0,1] 内的有限数")
    command = torch.empty(num_envs, 2, device=device)
    for column, (lower, upper) in enumerate(MOTION_RANGES[stage]):
        command[:, column].uniform_(lower, upper)
    # stage 0 的范围本来就是零；仅在扩展运动范围后混入原地保持样本。
    if stage > 0 and stationary_fraction > 0:
        command[torch.rand(num_envs, device=device) < stationary_fraction] = 0.0
    return command


def motion_state(env, command_name="motion_command", pitch_tol=0.20, velocity_tol=0.10,
                 yaw_tol=0.20, lateral_tol=0.15, tilt_rate_tol=0.75):
    """训练课程与评测共享真实支撑、坐标及跟踪门，避免同名指标各自解释。"""
    data = env.scene["robot"].data
    velocity, tilt_rate, finite = ground_motion(data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w)
    gravity = data.projected_gravity_b
    height = data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    command = env.command_manager.get_command(command_name)
    pitch_command = env.command_manager.get_command("pitch_command")[:, 0]
    pitch = torch.atan2(-gravity[:, 0], -gravity[:, 2])
    pitch_error = pitch - pitch_command
    finite &= torch.isfinite(gravity).all(-1) & torch.isfinite(height) & torch.isfinite(command).all(-1)
    finite &= torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.joint_vel).all(-1)
    finite &= torch.isfinite(pitch_command)
    _, supported = current_support(env)
    posture = finite & supported & (pitch_error.abs() < pitch_tol) & (gravity[:, 1].abs() < 0.25)
    posture &= (height >= 0.30) & (height <= 0.65) & (tilt_rate < tilt_rate_tol)
    error = velocity[:, [0, 2]] - command
    stable = posture & (error[:, 0].abs() < velocity_tol) & (error[:, 1].abs() < yaw_tol)
    stable &= velocity[:, 1].abs() < lateral_tol
    return {"velocity": velocity, "command": command, "error": error, "pitch_error": pitch_error,
            "finite": finite, "supported": supported, "posture": posture, "stable": stable}


def motion_balance_hold(env) -> torch.Tensor:
    """替换旧的全角速度模长门，否则非零 yaw 指令会与平衡奖励冲突。"""
    return motion_curriculum_state(env)["stable"].float()


def motion_curriculum_state(env, command_name="motion_command"):
    """课程只对精确静止命令收紧漂移门，旧通用评测门保持不变。

    静止必须有 reset/命令管理器建立的世界 XY 锚点；缺失时不能视为成功。
    使用整个停车阶段的最大位移，越界后返回原点也不能抹去失败历史。
    """
    state = motion_state(env, command_name=command_name)
    stationary = stationary_command_mask(state["command"])
    strict_stable = state["posture"].clone()
    strict_stable &= state["error"][:, 0].abs() < 0.03
    strict_stable &= state["error"][:, 1].abs() < 0.08
    strict_stable &= state["velocity"][:, 1].abs() < 0.03
    drift_limit = float(getattr(env, "_motion_stationary_drift_limit_m", 0.30))
    drift = _motion_xy_squared_displacement(env)
    anchor_state = getattr(env, "_motion_xy_anchor_state", None)
    if anchor_state is None:
        strict_stable[:] = False
    else:
        anchor_state["maximum_squared"].copy_(torch.maximum(anchor_state["maximum_squared"], drift))
        strict_stable &= anchor_state["maximum_squared"] <= drift_limit * drift_limit
    state["stable"] = torch.where(stationary, strict_stable, state["stable"])
    return state


def motion_velocity_tracking(
    env, velocity_sigma=0.25, yaw_sigma=0.5, stationary_velocity_sigma=0.04, stationary_yaw_sigma=0.10,
) -> torch.Tensor:
    """静止样本惩罚慢漂移；非零运动命令维持原有跟踪奖励尺度。"""
    state = motion_state(env)
    sigmas = torch.tensor(
        (velocity_sigma, yaw_sigma, stationary_velocity_sigma, stationary_yaw_sigma),
        device=state["error"].device,
        dtype=state["error"].dtype,
    )
    if not torch.isfinite(sigmas).all() or (sigmas <= 0).any():
        return torch.zeros_like(state["error"][:, 0])
    stationary = stationary_command_mask(state["command"])
    velocity_scale = torch.where(stationary, sigmas[2], sigmas[0])
    yaw_scale = torch.where(stationary, sigmas[3], sigmas[1])
    score = torch.exp(-state["error"][:, 0].square() / velocity_scale.square())
    score += torch.exp(-state["error"][:, 1].square() / yaw_scale.square())
    valid = state["posture"] & state["finite"] & torch.isfinite(score)
    return torch.where(valid, score, 0.0)


def motion_lateral_slip(env) -> torch.Tensor:
    state = motion_state(env)
    return torch.where(state["posture"], state["velocity"][:, 1].square(), 0.0)


def motion_zero_forward_velocity_quality(
    env,
    forward_scale=0.10,
    lateral_scale=0.10,
    yaw_scale=0.20,
) -> torch.Tensor:
    """零前进及原地转向的有界质量分数，补足窄指数奖励的远端梯度。

    非零前进仍由通用 tracking 奖励学习，避免该项改变运动阶段的速度权衡。
    分数始终非负，失稳或无效时为零，不能通过提前终止来少受累计负奖励。
    """
    parameters = (forward_scale, lateral_scale, yaw_scale)
    if not all(math.isfinite(value) and value > 0 for value in parameters):
        raise ValueError("零前进速度质量尺度必须为有限正数")
    state = motion_state(env)
    command_finite = torch.isfinite(state["command"]).all(-1)
    zero_forward = command_finite & (state["command"][:, 0] == 0)
    normalized = (
        state["error"][:, 0].square() / (forward_scale * forward_scale)
        + state["velocity"][:, 1].square() / (lateral_scale * lateral_scale)
        + state["error"][:, 1].square() / (yaw_scale * yaw_scale)
    )
    valid = state["posture"] & state["finite"] & command_finite & torch.isfinite(normalized)
    quality = torch.where(valid, 1.0 / (1.0 + normalized.clamp_min(0.0)), 0.0)
    return torch.where(zero_forward, quality, 0.0)


def reset_motion_xy_anchor(env, env_ids=None) -> None:
    """仅清理指定环境；command.reset 的重采样随后记录 reset event 写入的新位置。"""
    state = getattr(env, "_motion_xy_anchor_state", None)
    if state is None:
        return
    ids = slice(None) if env_ids is None else env_ids
    state["valid"][ids] = False
    state["holding"][ids] = False
    state["maximum_squared"][ids] = 0.0


def _sync_motion_xy_anchor(env, command, env_ids=None) -> None:
    """在下一物理步前锁存停车原点；连续零命令重采样不能擦掉累计漂移。

    Isaac 的 reset event 先写根状态，随后 command.reset 重采样；正常 step
    在 reward 之后更新 command。这里利用该顺序，使 reset 后第一步与停车
    减速第一步都计入位移。奖励读取不改变锚点，也不通过事后重基准抹去首步。
    """
    position = env.scene["robot"].data.root_pos_w[:, :2]
    state = getattr(env, "_motion_xy_anchor_state", None)
    if state is None:
        state = {
            "anchor": torch.zeros_like(position),
            "valid": torch.zeros(env.num_envs, dtype=torch.bool, device=position.device),
            "holding": torch.zeros(env.num_envs, dtype=torch.bool, device=position.device),
            "maximum_squared": torch.zeros(env.num_envs, device=position.device),
        }
        env._motion_xy_anchor_state = state
    ids = slice(None) if env_ids is None else env_ids
    holding = torch.isfinite(command[ids]).all(-1) & (command[ids, 0] == 0)
    entering = holding & ~state["holding"][ids]
    state["anchor"][ids] = torch.where(entering[:, None], position[ids].detach(), state["anchor"][ids])
    state["valid"][ids] = torch.where(entering, torch.isfinite(position[ids]).all(-1), state["valid"][ids])
    state["maximum_squared"][ids] = torch.where(entering, 0.0, state["maximum_squared"][ids])
    state["holding"][ids] = holding


def _motion_xy_squared_displacement(env) -> torch.Tensor:
    """返回未截断的锚点距离；缺失/非有限状态以 inf 拒绝成功。"""
    position = env.scene["robot"].data.root_pos_w[:, :2]
    state = getattr(env, "_motion_xy_anchor_state", None)
    if state is None:
        return torch.full_like(position[:, 0], float("inf"))
    valid = state["valid"] & state["holding"] & torch.isfinite(position).all(-1)
    squared = (position - state["anchor"]).square().sum(-1)
    return torch.where(valid, squared, float("inf"))


def motion_xy_drift(env, maximum_displacement_m=3.0) -> torch.Tensor:
    """零前进命令（含原地转向）的位移平方；仅为训练 shaping。

    不改变 44/49 维网络合同，也没有向 actor 提供绝对位置反馈，因此它能
    抑制持续漂移，但不能保证外力推动后回到原点。最终仍须独立长时验收。
    """
    if not math.isfinite(maximum_displacement_m) or maximum_displacement_m <= 0:
        raise ValueError("XY 漂移惩罚距离必须为有限正数")
    command = env.command_manager.get_command("motion_command")
    state = getattr(env, "_motion_xy_anchor_state", None)
    if state is None:
        raise RuntimeError("XY 锚点尚未初始化；必须先调用 MotionCommand.reset")
    holding = torch.isfinite(command).all(-1) & (command[:, 0] == 0)
    squared = _motion_xy_squared_displacement(env)
    # 8 秒训练内的持续漂移不能在 1 米后失去梯度；无效状态使用同一显式上限。
    maximum_squared = maximum_displacement_m * maximum_displacement_m
    return torch.where(holding, squared.clamp(0.0, maximum_squared), 0.0)


class MotionCurriculum:
    """仅以连续真实成功控制步推进；reset、失稳或漏采样均中断连续性。

    门要求一定比例环境在当前时刻都已有连续成功历史，不累计旧成功片段。
    checkpoint 不保存仿真课程状态，续训/评测须显式 fixed_stage 或 initial_stage。
    """

    def __init__(self, num_envs, dt, success_time_s=1.0, success_fraction=0.6,
                 initial_stage=0, fixed_stage=None, device="cpu"):
        if initial_stage not in range(3) or (fixed_stage is not None and fixed_stage not in range(3)):
            raise ValueError("motion stage 必须为 0、1 或 2")
        if num_envs < 1 or not all(math.isfinite(x) and x > 0 for x in (dt, success_time_s, success_fraction)):
            raise ValueError("课程大小、周期和成功阈值必须为有限正数")
        if success_fraction > 1:
            raise ValueError("成功环境比例不得大于 1")
        self.stage = initial_stage if fixed_stage is None else fixed_stage
        self.fixed_stage = fixed_stage
        self.required_steps = math.ceil(success_time_s / dt - 1e-9)
        self.required_envs = math.ceil(success_fraction * num_envs)
        self.streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_step = -1

    def reset(self, env_ids):
        self.streak[env_ids] = 0

    def update(self, stable: torch.Tensor, step: int) -> bool:
        if stable.shape != self.streak.shape or stable.dtype != torch.bool:
            raise ValueError("课程输入须为逐环境 bool 成功门")
        if step == self.last_step:
            return False
        if step < self.last_step:
            raise ValueError("课程控制步不可倒退")
        if step != self.last_step + 1:
            self.streak.zero_()
        self.last_step = step
        self.streak = torch.where(stable, self.streak + 1, 0)
        if self.fixed_stage is not None or self.stage == 2:
            return False
        if int((self.streak >= self.required_steps).sum()) < self.required_envs:
            return False
        self.stage += 1
        self.streak.zero_()
        return True


class MotionCommand(CommandTerm):
    """2 维 [前进速度 m/s, 世界 Z 转速 rad/s]；未输出的 lin_y 始终为零。"""

    def __init__(self, cfg, env):
        if not math.isfinite(cfg.stationary_fraction) or not 0 <= cfg.stationary_fraction <= 1:
            raise ValueError("静止指令比例必须为 [0,1] 内的有限数")
        drift_limit = float(getattr(cfg, "stationary_drift_limit_m", 0.30))
        if not math.isfinite(drift_limit) or drift_limit <= 0:
            raise ValueError("静止世界位移门必须为有限正数")
        super().__init__(cfg, env)
        self.curriculum = MotionCurriculum(
            self.num_envs, env.step_dt, cfg.success_time_s, cfg.success_fraction,
            cfg.initial_stage, cfg.fixed_stage, self.device,
        )
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._observed_command = self._command.clone()
        self._observed_pitch = None
        self._env._motion_stationary_drift_limit_m = drift_limit
        self._last_metric_step = -1
        for name in ("stage", "forward_error", "yaw_error", "continuous_success_s"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._validate_fixed_command()

    def _validate_fixed_command(self):
        if self.cfg.fixed_command is None:
            return
        if self.cfg.fixed_stage is None or len(self.cfg.fixed_command) != 2:
            raise ValueError("固定命令必须同时固定课程 stage，格式 (vx,yaw)")
        for value, (lower, upper) in zip(self.cfg.fixed_command, MOTION_RANGES[self.cfg.fixed_stage]):
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError("固定命令超出该 stage 的速度范围")

    @property
    def command(self):
        return self._command

    @property
    def planar_command(self):
        """诊断接口 [vx,0,yaw]；actor 只追加 command 的两列。"""
        return torch.stack((self._command[:, 0], torch.zeros_like(self._command[:, 0]), self._command[:, 1]), -1)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.curriculum.reset(env_ids)
        reset_motion_xy_anchor(self._env, env_ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        if self.cfg.fixed_command is not None:
            self._command[env_ids] = self._command.new_tensor(self.cfg.fixed_command)
        else:
            self._command[env_ids] = sample_motion_commands(
                len(env_ids), self.curriculum.stage, self.device, self.cfg.stationary_fraction,
            )
        self._sync_command_history()
        _sync_motion_xy_anchor(self._env, self._command, env_ids)

    def _sync_command_history(self):
        """速度或姿态目标改变即中断成功计时；完全相同目标保持连续。"""
        pitch = self._env.command_manager.get_command("pitch_command")
        changed = torch.any(self._command != self._observed_command, dim=-1)
        if self._observed_pitch is not None:
            changed |= torch.any(pitch != self._observed_pitch, dim=-1)
        self.curriculum.streak.masked_fill_(changed, 0)
        self._observed_command.copy_(self._command)
        self._observed_pitch = pitch.clone()
        return changed

    def _update_command(self):
        # 同时支持受控实验直接改命令；此钩子仍在下一物理步前执行。
        self._sync_command_history()
        _sync_motion_xy_anchor(self._env, self._command)

    def _update_metrics(self):
        step = int(self._env.common_step_counter)
        if step == self._last_metric_step:
            return
        self._last_metric_step = step
        changed_command = self._sync_command_history()
        state = motion_curriculum_state(self._env)
        # command.compute 在自动 reset 后调用，刚复位的样本不可当作本步真实成功。
        stable = state["stable"] & (self._env.episode_length_buf > 0) & ~changed_command
        changed = self.curriculum.update(stable, step)
        self.metrics["stage"].fill_(self.curriculum.stage)
        self.metrics["forward_error"].copy_(torch.nan_to_num(state["error"][:, 0].abs()))
        self.metrics["yaw_error"].copy_(torch.nan_to_num(state["error"][:, 1].abs()))
        self.metrics["continuous_success_s"].copy_(self.curriculum.streak * self._env.step_dt)
        if changed:
            self._resample_command(torch.arange(self.num_envs, device=self.device))


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    resampling_time_range: tuple[float, float] = (3.0, 5.0)
    debug_vis: bool = False
    initial_stage: int = 0
    fixed_stage: int | None = None
    fixed_command: tuple[float, float] | None = None
    stationary_fraction: float = 0.0
    stationary_drift_limit_m: float = 0.30
    success_time_s: float = 1.0
    success_fraction: float = 0.6
