"""受限分段关节目标：区分恢复探索不足与翻正后保持失败，仅用于仿真诊断。

@author pony
@date 2026-09-17
@version v0.3.0
@last_modified 2026-09-28
@changelog
    - v0.3.0 (2026-09-28): 固定程序按倒向镜像并重新校验目标，记录速度及估算功率。
    - v0.2.0 (2026-09-17): 记录真实主动关节行程和执行器估算力矩，区分动作输入与实际响应。
    - v0.1.0 (2026-09-17): 独立RNG采样、物理限位/动作逆变换与严格诊断身份。
"""

from __future__ import annotations

import math

import torch


class RecoveryActionProgram:
    """每个环境一个可复放程序；接管前不执行，前三秒探索，余下时间回默认PD。

    不改变增益、力矩或关节限位，也不直接写物理状态。动作经原action manager执行。
    独立CPU生成器使幅度对照共享随机方向且不改变仿真初始化的全局随机序列。
    随机搜索找到的轨迹只说明该样本可恢复，既不是策略，也不是泛化成功率。
    """

    def __init__(self, *, default_targets, scales, leg_limits, wheel_velocity_limits,
                 command_clip, raw_clip, dt, segment_s, active_s, seed, leg_fraction, wheel_speed):
        if default_targets.ndim != 2 or default_targets.shape[1] != 12 or not len(default_targets):
            raise ValueError("default_targets 必须是非空 [环境数,12]")
        if not all(math.isfinite(v) and v > 0 for v in (dt, segment_s, active_s)):
            raise ValueError("时间参数必须为有限正数")
        if not 0 <= leg_fraction <= 1 or not math.isfinite(wheel_speed) or wheel_speed < 0:
            raise ValueError("leg_fraction 必须在[0,1]，wheel_speed必须有限非负")
        if active_s > 3.5:
            raise ValueError("active_s最多3.5秒，保留原4秒进入稳定区与5秒终态的保持预算")
        ratios = (segment_s / dt, active_s / segment_s)
        if any(abs(v - round(v)) > 1e-7 or round(v) < 1 for v in ratios):
            raise ValueError("segment_s必须是控制周期整数倍，active_s必须是完整分段的整数倍")
        if type(seed) is not int or seed < 0:
            raise ValueError("程序seed必须为非负整数")
        if raw_clip is not None and (not math.isfinite(raw_clip) or raw_clip <= 0):
            raise ValueError("raw_clip必须为有限正数或None")
        self.defaults = default_targets.detach().clone()
        self.num_envs = len(self.defaults)
        self.scales = torch.broadcast_to(torch.as_tensor(scales, device=self.defaults.device), self.defaults.shape).clone()
        if leg_limits.shape != (self.num_envs, 8, 2) or wheel_velocity_limits.shape != (self.num_envs, 4):
            raise ValueError("物理限位必须逐环境匹配8腿/4轮")
        self.leg_limits = leg_limits.detach().clone()
        self.wheel_limits = wheel_velocity_limits.detach().clone()
        self.command_clip = torch.broadcast_to(
            torch.as_tensor(command_clip, device=self.defaults.device), (self.num_envs, 12, 2),
        ).clone()
        if not all(torch.isfinite(t).all() for t in
                   (self.defaults, self.scales, self.leg_limits, self.wheel_limits, self.command_clip)):
            raise ValueError("动作合同含非有限数值")
        if (self.scales == 0).any() or (self.leg_limits[..., 0] >= self.leg_limits[..., 1]).any():
            raise ValueError("动作scale不能为零，关节下限必须小于上限")
        if (self.command_clip[..., 0] > self.command_clip[..., 1]).any():
            raise ValueError("动作目标clip上下界倒置")
        if (self.wheel_limits <= 0).any() or (wheel_speed > self.wheel_limits).any():
            raise ValueError("轮速度目标超过物理速度限位")
        if ((self.defaults[:, :8] < self.leg_limits[..., 0])
                | (self.defaults[:, :8] > self.leg_limits[..., 1])).any():
            raise ValueError("默认腿目标不在限位内")
        if self.defaults[:, 8:].any():
            raise ValueError("诊断保持段要求默认轮速为零")
        self.dt = dt
        self.segment_steps = round(ratios[0])
        self.num_segments = round(ratios[1])
        self.active_steps = self.segment_steps * self.num_segments
        self.seed, self.leg_fraction, self.wheel_speed = seed, leg_fraction, wheel_speed
        self.raw_clip = raw_clip

        generator = torch.Generator(device="cpu").manual_seed(seed)
        direction = (torch.rand(self.num_envs, self.num_segments, 12, generator=generator) * 2 - 1).to(self.defaults)
        default_legs = self.defaults[:, None, :8]
        direction_legs = direction[:, :, :8]
        span = torch.where(direction_legs >= 0, self.leg_limits[:, None, :, 1] - default_legs,
                           default_legs - self.leg_limits[:, None, :, 0])
        self.targets = torch.cat((default_legs + direction_legs * span * leg_fraction,
                                  direction[:, :, 8:] * wheel_speed), dim=-1)
        self.raw_actions = (self.targets - self.defaults[:, None, :]) / self.scales[:, None, :]
        # clip会改变实验变量；拒绝饱和而不是让名义目标与真正输入悄悄不同。
        if raw_clip is not None and (self.raw_actions.abs() > raw_clip).any():
            raise ValueError("程序原始动作超过wrapper clip")
        for targets in (self.targets, self.defaults[:, None, :]):
            if ((targets < self.command_clip[:, None, :, 0])
                    | (targets > self.command_clip[:, None, :, 1])).any():
                raise ValueError("程序物理目标超过action term clip")
        device = self.defaults.device
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.min_gravity_z = torch.full((self.num_envs,), float("inf"), device=device)
        self.stable_streak = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.support_streak = torch.zeros_like(self.stable_streak)
        self.max_stable_steps = torch.zeros_like(self.stable_streak)
        self.max_support_steps = torch.zeros_like(self.stable_streak)
        self.joint_min = torch.full_like(self.defaults, float("inf"))
        self.joint_max = torch.full_like(self.defaults, -float("inf"))
        self.max_estimated_torque = torch.zeros_like(self.defaults)
        self.max_joint_velocity = torch.zeros_like(self.defaults)
        self.max_estimated_power = torch.zeros_like(self.defaults)
        # 保留未镜像候选；后续诊断可以对每个候选独立按初始倒向镜像，
        # 不能在已经镜像的 targets 上二次变换。
        self._candidate_targets = self.targets.detach().clone()
        self.source_program_ids = list(range(self.num_envs))
        self.mirrored = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    def _commit_targets(self, targets, source_program_ids, mirrored):
        """校验并提交物理目标；任何越界都必须显式失败，不能静默clip。"""
        if targets.shape != (self.num_envs, self.num_segments, 12):
            raise ValueError("动作目标必须是[环境数,分段数,12]")
        if len(source_program_ids) != self.num_envs:
            raise ValueError("source_program_ids必须逐环境记录")
        if mirrored.shape != (self.num_envs,) or mirrored.dtype != torch.bool:
            raise ValueError("镜像选择必须是逐环境bool")
        raw = (targets - self.defaults[:, None]) / self.scales[:, None]
        if ((targets[:, :, :8] < self.leg_limits[:, None, :, 0])
                | (targets[:, :, :8] > self.leg_limits[:, None, :, 1])).any():
            raise ValueError("固定或镜像目标超过接收环境的腿限位")
        if (targets[:, :, 8:].abs() > self.wheel_limits[:, None]).any():
            raise ValueError("固定或镜像目标超过接收环境的轮速限位")
        if ((targets < self.command_clip[:, None, :, 0])
                | (targets > self.command_clip[:, None, :, 1])).any():
            raise ValueError("固定或镜像目标超过action term clip")
        if self.raw_clip is not None and (raw.abs() > self.raw_clip).any():
            raise ValueError("固定或镜像原始动作超过wrapper clip")
        self.targets, self.raw_actions = targets, raw
        self.source_program_ids = list(source_program_ids)
        self.mirrored = mirrored.clone()

    def mirror_candidates(self, mirror_mask):
        """逐环境镜像各自随机候选，供动作搜索使用。"""
        if mirror_mask.shape != (self.num_envs,) or mirror_mask.dtype != torch.bool:
            raise ValueError("镜像选择必须是逐环境bool")
        reflected = self._candidate_targets[:, :, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]].clone()
        reflected[:, :, 4:8] *= -1
        targets = torch.where(mirror_mask[:, None, None], reflected, self._candidate_targets)
        self._commit_targets(targets, list(range(self.num_envs)), mirror_mask)

    def select_program(self, index, mirror_mask=None):
        """选择一个候选并按真实物理目标镜像：交换左右，只有转向反号。"""
        if type(index) is not int or not 0 <= index < self.num_envs:
            raise ValueError("program_id必须落在本次采样环境范围")
        targets = self._candidate_targets[index:index + 1].expand(self.num_envs, -1, -1).clone()
        mirrored = torch.zeros_like(self.mirrored) if mirror_mask is None else mirror_mask
        reflected = targets[:, :, [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]].clone()
        reflected[:, :, 4:8] *= -1
        targets = torch.where(mirrored[:, None, None], reflected, targets)
        self._commit_targets(targets, [index] * self.num_envs, mirrored)

    def observe_joints(self, positions, estimated_torque, active, velocities=None):
        """记录实际关节状态；implicit的applied_torque是估算值，不冒充PhysX驱动力测量。"""
        if positions.shape != self.defaults.shape or estimated_torque.shape != self.defaults.shape:
            raise ValueError("诊断关节状态必须为按动作合同排序的[环境数,12]")
        enabled = active & ~self.finished
        valid = enabled & torch.isfinite(positions).all(-1) & torch.isfinite(estimated_torque).all(-1)
        self.joint_min = torch.minimum(self.joint_min, torch.where(valid[:, None], positions, float("inf")))
        self.joint_max = torch.maximum(self.joint_max, torch.where(valid[:, None], positions, -float("inf")))
        self.max_estimated_torque = torch.maximum(
            self.max_estimated_torque, torch.where(valid[:, None], estimated_torque.abs(), 0.),
        )
        if velocities is not None:
            if velocities.shape != self.defaults.shape:
                raise ValueError("诊断速度必须为按动作合同排序的[环境数,12]")
            valid &= torch.isfinite(velocities).all(-1)
            self.max_joint_velocity = torch.maximum(
                self.max_joint_velocity, torch.where(valid[:, None], velocities.abs(), 0.),
            )
            self.max_estimated_power = torch.maximum(
                self.max_estimated_power, torch.where(valid[:, None], (velocities * estimated_torque).abs(), 0.),
            )

    def actions(self, control_steps, active):
        """control_steps是下一步之前的已执行步数；局部接管0从第一段开始。"""
        if control_steps.shape != (self.num_envs,) or active.shape != (self.num_envs,):
            raise ValueError("控制时钟和active必须逐环境匹配")
        if control_steps.dtype not in (torch.int32, torch.int64) or (control_steps < 0).any():
            raise ValueError("控制时钟必须是非负整数")
        if active.dtype != torch.bool:
            raise ValueError("active必须为bool张量")
        segment = (control_steps // self.segment_steps).clamp(max=self.num_segments - 1)
        action = self.raw_actions[torch.arange(self.num_envs, device=segment.device), segment]
        enabled = active & ~self.finished & (control_steps < self.active_steps)
        return torch.where(enabled[:, None], action, 0.)

    def observe(self, gravity_z, stable, support, active, finite):
        """只能在termination快照中消费reset前状态；瞬时翻正不能代替终态验收。"""
        enabled = active & ~self.finished
        valid = enabled & finite & torch.isfinite(gravity_z)
        self.min_gravity_z = torch.minimum(self.min_gravity_z, torch.where(valid, gravity_z, float("inf")))
        self.stable_streak = torch.where(valid & stable, self.stable_streak + 1, 0)
        self.support_streak = torch.where(valid & support, self.support_streak + 1, 0)
        self.max_stable_steps = torch.maximum(self.max_stable_steps, self.stable_streak)
        self.max_support_steps = torch.maximum(self.max_support_steps, self.support_streak)

    def episode_metrics(self, index):
        value = self.min_gravity_z[index].item()
        return {
            "program_id": index,
            "minimum_gravity_z": value if math.isfinite(value) else None,
            "maximum_stable_time_s": self.max_stable_steps[index].item() * self.dt,
            "maximum_four_wheel_support_time_s": self.max_support_steps[index].item() * self.dt,
            "active_joint_range_rad": (
                torch.stack((self.joint_min[index], self.joint_max[index]), dim=-1).cpu().tolist()
                if torch.isfinite(self.joint_min[index]).all() else None
            ),
            "maximum_actuator_estimated_torque_nm": self.max_estimated_torque[index].cpu().tolist(),
            "maximum_joint_velocity_rad_s": self.max_joint_velocity[index].cpu().tolist(),
            "maximum_actuator_estimated_power_w": self.max_estimated_power[index].cpu().tolist(),
            "response_sampling": "control_step_not_physics_substep_peak",
        }

    def mark_diagnostic(self, result):
        """保留同一成功门的数值，禁止被加载但未执行的checkpoint获得技能成绩。"""
        result.update({
            "evaluation_scope": "bounded_action_program_search_not_policy_acceptance",
            "acceptance_eligible": False, "diagnostic_passed": bool(result["passed"]), "passed": False,
            "action_source": "segmented_joint_targets_with_active_pd", "checkpoint_policy_executed": False,
            "action_program": {
                "seed": self.seed, "leg_fraction": self.leg_fraction, "wheel_speed_rad_s": self.wheel_speed,
                "segment_steps": self.segment_steps, "active_steps": self.active_steps, "step_dt": self.dt,
                "targets_shape": list(self.targets.shape), "targets": self.targets.cpu().tolist(),
                "source_program_ids": self.source_program_ids, "mirrored": self.mirrored.cpu().tolist(),
                "default_targets": self.defaults.cpu().tolist(), "action_scales": self.scales.cpu().tolist(),
                "leg_soft_limits_rad": self.leg_limits.cpu().tolist(),
                "wheel_velocity_limits_rad_s": self.wheel_limits.cpu().tolist(), "wrapper_raw_clip": self.raw_clip,
                "target_units": "first_8_joint_position_rad_then_4_wheel_velocity_rad_s",
                "hold_mode": "zero_raw_action_default_pose_active_pd",
            },
        })


def program_from_environment(env, args, raw_clip):
    """读取真实action term的顺序、scale和offset，不按20关节内部索引猜测。"""
    from robot_lab.assets.parallelogram_rover import PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES

    manager = env.action_manager
    if manager.active_terms != ["joint_pos", "joint_vel"]:
        raise ValueError("程序只支持恢复joint_pos + joint_vel动作合同")
    terms = [manager.get_term(name) for name in manager.active_terms]
    if [type(term).__name__ for term in terms] != ["JointPositionAction", "JointVelocityAction"]:
        raise ValueError("程序不支持相对目标或其它动作类型")
    names = tuple(name for term in terms for name in term._joint_names)
    if names != PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES:
        raise ValueError("动作关节语义或顺序不匹配")
    robot = env.scene["robot"]
    joint_ids, resolved = robot.find_joints(list(names), preserve_order=True)
    if tuple(resolved) != names:
        raise ValueError("机器人关节解析与动作合同不一致")
    scales, offsets, clips = [], [], []
    for term in terms:
        shape = (env.num_envs, term.action_dim)
        scales.append(torch.broadcast_to(torch.as_tensor(term._scale, device=env.device), shape))
        offsets.append(torch.broadcast_to(torch.as_tensor(term._offset, device=env.device), shape))
        clips.append(term._clip)
    program = RecoveryActionProgram(
        default_targets=torch.cat(offsets, dim=-1), scales=torch.cat(scales, dim=-1),
        leg_limits=robot.data.soft_joint_pos_limits[:, joint_ids[:8]],
        wheel_velocity_limits=robot.data.joint_vel_limits[:, joint_ids[8:]],
        command_clip=torch.cat(clips, dim=1), raw_clip=raw_clip, dt=env.step_dt,
        segment_s=args.program_segment_s, active_s=args.program_active_s, seed=args.program_seed,
        leg_fraction=args.program_leg_fraction, wheel_speed=args.program_wheel_speed,
    )
    program.joint_ids = joint_ids
    if getattr(args, "program_mirror", False):
        mask = robot.data.projected_gravity_b[:, 1] > 0
        if getattr(args, "program_id", None) is not None:
            program.select_program(args.program_id, mask)
        else:
            program.mirror_candidates(mask)
    elif getattr(args, "program_id", None) is not None:
        program.select_program(args.program_id)
    return program, list(names)
