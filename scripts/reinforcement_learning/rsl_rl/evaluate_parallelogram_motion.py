#!/usr/bin/env python3
"""44 维两轮行走的固定命令验收；不把末态停住当作完成行走。

@author pony
@date 2026-09-16
@version v0.13.0
@last_modified 2026-09-17
@changelog
    - v0.13.0 (2026-09-17): wrapper复位异常时先释放底层环境，避免关闭回调遮蔽真实错误。
    - v0.12.0 (2026-09-17): 区分静止与合成带速起立验收，逐回合保存真实初态及初始命令。
    - v0.11.0 (2026-09-17): 允许隔离速度求解迭代数，全部覆盖继续强制标为非验收诊断。
    - v0.10.0 (2026-09-17): 采集实际COM姿态及COM处link轴惯量，避免只对齐质量就宣称模型一致。
    - v0.9.0 (2026-09-17): 独立腿速度约束对照；覆盖值写入谱系并强制降级为诊断。
    - v0.8.0 (2026-09-17): 可选逐物理子步驱动/接触trace，不改训练与验收协议。
    - v0.7.0 (2026-09-17): 可选首回合逐步trace，区分link/COM，原始观测/动作与reset前末态。
    - v0.6.0 (2026-09-17): 保存逐环境质量/材料及真实初态以诊断慢漂移和跨引擎差异，不改变成功门。
    - v0.5.0 (2026-09-17): 停止命令同步父类的 XY 锚点和连续成功计时。
    - v0.4.0 (2026-09-17): 世界XY漂移门及阶段起点隔离；缩短保持时间只输出smoke结果。
    - v0.3.0 (2026-09-17): 可选记录失败episode末段真实状态，供独立恢复重放，禁止用reset后状态。
    - v0.2.0 (2026-09-17): 增加前进后停止两段协议，逐环境时钟及独立成功门。
    - v0.1.0 (2026-09-16): 独立 balance/getup 协议、真实后轮支撑与逐 episode 运动指标。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from skill_evaluation import SkillEvaluation


TASKS = {
    "balance": "RobotLab-Isaac-Upright-Motion-Parallelogram-Rover-v0",
    "getup": "RobotLab-Isaac-Upright-Motion-Getup-Parallelogram-Rover-v0",
}
SCENARIOS = {
    "stationary": (0.0, 0.0, 0),
    "forward": (0.25, 0.0, 1),
    "backward": (-0.20, 0.0, 2),
    "turn-left": (0.10, 0.50, 2),
    "turn-right": (0.10, -0.50, 2),
    "stop": (0.25, 0.0, 1),
}


def capture_dynamics_parameters(env) -> dict:
    """记录随机化后的 PhysX 值；不能把配置范围或 default_mass 当作实际物理参数。

    材料按碰撞 shape 排列，一个 body 可有多个 shape，不以 body_names 错配材料。
    当前 motion 的质量/材料随机化仅在 startup 执行，本快照在 reset 后、控制前读取。
    这是诊断证据，不改变评测初态、随机化范围或技能成功判据。
    """
    import torch

    robot = env.scene["robot"]
    names = list(robot.body_names)
    masses = robot.root_physx_view.get_masses().detach().cpu()
    materials = robot.root_physx_view.get_material_properties().detach().cpu()
    if not names or len(names) != len(set(names)) or masses.shape != (env.num_envs, len(names)):
        raise ValueError("实际质量必须与全部环境和唯一 body_names 对齐")
    if materials.ndim != 3 or materials.shape[0] != env.num_envs or materials.shape[1] < 1 or materials.shape[2] != 3:
        raise ValueError("实际材料必须按 PhysX shape 顺序提供 [N,S,3]")
    total_mass = masses.sum(dim=1)
    if (not torch.isfinite(masses).all() or (masses < 0).any()
            or not torch.isfinite(total_mass).all() or (total_mass <= 0).any()):
        raise ValueError("实际质量必须有限非负且逐环境总质量为有限正数")
    if not torch.isfinite(materials).all() or (materials < 0).any() or (materials[..., 2] > 1).any():
        raise ValueError("实际摩擦系数须有限非负，恢复系数须为 [0,1]")
    # 实测get_inertias与URDF完整link轴张量一致，非主惯量；不能按COM姿态再次旋转。
    coms = robot.root_physx_view.get_coms().detach().cpu().clone()
    inertias = robot.root_physx_view.get_inertias().detach().cpu().clone()
    if coms.shape != (env.num_envs, len(names), 7) or not torch.isfinite(coms).all():
        raise ValueError("COM姿态必须按body_names提供有限[N,B,7]数据")
    if not torch.allclose(torch.linalg.vector_norm(coms[..., 3:], dim=-1), torch.ones_like(masses), atol=1e-5):
        raise ValueError("COM姿态四元数必须归一化")
    if inertias.shape != (env.num_envs, len(names), 9) or not torch.isfinite(inertias).all():
        raise ValueError("COM处link轴惯量必须按body_names提供有限[N,B,9]数据")
    tensors = inertias.reshape(env.num_envs, len(names), 3, 3)
    if (not torch.allclose(tensors, tensors.transpose(-1, -2), atol=1e-8, rtol=1e-5)
            or (torch.linalg.eigvalsh(tensors) <= 0).any()):
        raise ValueError("动态刚体惯量必须对称正定")
    joints = list(robot.joint_names)
    if not joints or len(joints) != len(set(joints)):
        raise ValueError("初态快照必须包含全部唯一关节名，含被动 follower")
    initial_state = {}
    for name, value, columns in (
        ("root_state_w", robot.data.root_state_w, 13),
        ("joint_pos", robot.data.joint_pos, len(joints)),
        ("joint_vel", robot.data.joint_vel, len(joints)),
        ("env_origins_w", env.scene.env_origins, 3),
    ):
        if value.shape != (env.num_envs, columns) or not torch.isfinite(value).all():
            raise ValueError(f"初态快照 {name} 形状错误或包含非有限状态")
        initial_state[name] = value.detach().cpu().tolist()
    initial_state.update({
        "joint_names": joints,
        "root_state_layout": "world_position_xyz_quaternion_wxyz_world_linear_xyz_world_angular_xyz",
        # 同 seed 不是同初态；即使复制运动学状态，仍不包含求解器缓存/接触 warm start。
        "contains_contact_solver_state": False,
        "sampling": "first_explicit_reset_before_first_policy_action",
    })
    return {
        "source": "physx_runtime_after_startup_randomization_and_reset",
        "environment_ids": list(range(env.num_envs)),
        "body_names": names,
        "body_mass_kg": masses.tolist(),
        "body_inertial_contract": "physx_link_axes_com_inertia_v2",
        "body_com_pose_xyzw": coms.tolist(),
        "body_inertia_link_kg_m2": inertias.tolist(),
        "total_mass_kg": total_mass.tolist(),
        "material_shape_order": "physx_articulation_shape_index_not_body_index",
        "material_columns": ["static_friction", "dynamic_friction", "restitution"],
        "shape_material_properties": materials.tolist(),
        "initial_kinematic_state": initial_state,
    }


class MotionEvaluation(SkillEvaluation):
    """共享严格终态保持统计，并记录全过程误差和地面速度积分。

    距离为随车地面航向上的速度积分，不冒充世界坐标净位移。运动命令还要求
    积分进度至少达到指令积分的一半，防止只在末态偶然满足姿态门。
    """

    def __init__(self, *args, stationary_drift_limit_m=0.30, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(stationary_drift_limit_m) or stationary_drift_limit_m <= 0:
            raise ValueError("静止漂移门必须为有限正数")
        self.stationary_drift_limit_m = stationary_drift_limit_m
        self.motion_totals = [self._empty() for _ in range(self.num_envs)]

    @staticmethod
    def _empty():
        return {"samples": 0, "vx_error2": 0.0, "yaw_error2": 0.0,
                "forward": 0.0, "path": 0.0, "yaw": 0.0, "target_forward": 0.0,
                "target_yaw": 0.0, "support_samples": 0, "command": None,
                "reference_xy": None, "position_samples": 0, "max_displacement": 0.0,
                "position_invalid": False}

    def update_motion(self, stable, terminated, truncated, finite, velocities, commands, supported,
                      pitch_errors=None, *, world_positions, previous_world_positions):
        batches = (stable, terminated, truncated, finite, velocities, commands, supported,
                   world_positions, previous_world_positions)
        if any(len(batch) != self.num_envs for batch in batches):
            raise ValueError("运动评测每组状态必须包含全部环境")
        finite = list(finite)
        for index in range(self.num_envs):
            if self.counts[index] >= self.quota:
                continue
            velocity, command = velocities[index], commands[index]
            if len(velocity) != 3 or len(command) != 2:
                raise ValueError("地面速度须为 [vx,vy,yaw]，命令须为 [vx,yaw]")
            total = self.motion_totals[index]
            xy, previous_xy = world_positions[index], previous_world_positions[index]
            if len(xy) != 2 or len(previous_xy) != 2:
                raise ValueError("根位置必须是物理步前后真实世界坐标 [x,y]")
            position_finite = all(math.isfinite(v) for v in (*xy, *previous_xy))
            total["position_invalid"] |= not position_finite
            if position_finite:
                # 用阶段首次施加动作前的位置作参考，不能漏掉首步或跨reset继承上一位置。
                if total["reference_xy"] is None:
                    total["reference_xy"] = list(previous_xy)
                total["position_samples"] += 1
                total["max_displacement"] = max(total["max_displacement"], math.dist(xy, total["reference_xy"]))
            finite[index] &= position_finite
            finite[index] &= all(math.isfinite(value) for value in (*velocity, *command))
            if not finite[index]:
                continue
            if total["command"] is not None and list(command) != total["command"]:
                raise ValueError("固定命令验收不允许 episode 中途改变目标")
            total["command"] = list(command)
            total["samples"] += 1
            total["support_samples"] += int(supported[index])
            total["vx_error2"] += (velocity[0] - command[0])**2
            total["yaw_error2"] += (velocity[2] - command[1])**2
            total["forward"] += velocity[0] * self.dt
            total["path"] += math.hypot(velocity[0], velocity[1]) * self.dt
            total["yaw"] += velocity[2] * self.dt
            total["target_forward"] += command[0] * self.dt
            total["target_yaw"] += command[1] * self.dt
        before = len(self.records)
        super().update(stable, terminated, truncated, finite, pitch_errors)
        for record in self.records[before:]:
            index = record["env_id"]
            total = self.motion_totals[index]
            count = total["samples"]
            forward_ratio = total["forward"] / total["target_forward"] if abs(total["target_forward"]) > 1e-9 else None
            yaw_ratio = total["yaw"] / total["target_yaw"] if abs(total["target_yaw"]) > 1e-9 else None
            progress_ok = count > 0 and all(ratio is None or ratio >= 0.5 for ratio in (forward_ratio, yaw_ratio))
            drift_limit = self.stationary_drift_limit_m if total["command"] == [0.0, 0.0] else None
            drift_ok = (not total["position_invalid"] and total["position_samples"] > 0
                        and (drift_limit is None or total["max_displacement"] <= drift_limit))
            record.update({
                "command_forward_yaw": total["command"],
                "velocity_rmse_m_s": math.sqrt(total["vx_error2"] / count) if count else None,
                "yaw_rate_rmse_rad_s": math.sqrt(total["yaw_error2"] / count) if count else None,
                "ground_forward_distance_m": total["forward"],
                "horizontal_path_length_m": total["path"],
                "yaw_integral_rad": total["yaw"],
                "command_forward_distance_m": total["target_forward"],
                "command_yaw_integral_rad": total["target_yaw"],
                "forward_progress_ratio": forward_ratio,
                "yaw_progress_ratio": yaw_ratio,
                "two_rear_wheel_support_fraction": total["support_samples"] / count if count else None,
                "progress_gate_passed": progress_ok,
                "reference_xy_m": total["reference_xy"],
                "maximum_horizontal_displacement_m": total["max_displacement"] if total["position_samples"] else None,
                "stationary_drift_limit_m": drift_limit,
                "stationary_drift_gate_passed": drift_ok,
            })
            record["success"] &= progress_ok and drift_ok
            self.motion_totals[index] = self._empty()


class StopProtocol:
    """先前进保持、再发零速指令；所有边界对齐控制步，指令作用于下一物理步。"""

    def __init__(self, dt, move_deadline_s=1.0, move_hold_time_s=5.0,
                 stop_deadline_s=2.0, stop_hold_time_s=5.0, forward_speed=.25):
        values = (dt, move_deadline_s, move_hold_time_s, stop_deadline_s, stop_hold_time_s, forward_speed)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("停止协议的时长、周期及前进速度必须为有限正数")
        self.dt = dt
        self.forward_speed = forward_speed
        self.move_deadline_s = move_deadline_s
        self.move_hold_time_s = move_hold_time_s
        self.stop_deadline_s = stop_deadline_s
        self.stop_hold_time_s = stop_hold_time_s
        self.move_steps = math.ceil((move_deadline_s + move_hold_time_s) / dt - 1e-9)
        self.stop_steps = math.ceil((stop_deadline_s + stop_hold_time_s) / dt - 1e-9)
        self.episode_steps = self.move_steps + self.stop_steps

    def command_at(self, completed_steps):
        if completed_steps < 0:
            raise ValueError("episode 时钟不能为负")
        return [self.forward_speed if completed_steps < self.move_steps else 0.0, 0.0]

    def metadata(self):
        return {
            "switch_time_s": self.move_steps * self.dt,
            "move_deadline_s": self.move_deadline_s, "move_hold_time_s": self.move_hold_time_s,
            "stop_deadline_s": self.stop_deadline_s, "stop_hold_time_s": self.stop_hold_time_s,
            "forward_command_m_s": self.forward_speed, "episode_steps": self.episode_steps,
            "phase_clock": "per_environment_episode_steps_before_auto_reset",
        }


class StopEvaluation(SkillEvaluation):
    """组合两段实际指标；前进段失败永久保留，不能被末态静止掩盖。"""

    def __init__(self, num_envs, episodes_per_env, protocol):
        super().__init__(num_envs, episodes_per_env, protocol.dt,
                         protocol.stop_hold_time_s, protocol.stop_deadline_s)
        self.protocol = protocol
        self.phase_trackers = [self._new_phases() for _ in range(num_envs)]

    def _new_phases(self):
        p = self.protocol
        return (MotionEvaluation(1, 1, p.dt, p.move_hold_time_s, p.move_deadline_s),
                MotionEvaluation(1, 1, p.dt, p.stop_hold_time_s, p.stop_deadline_s,
                                 stationary_drift_limit_m=0.50))

    def update_motion(self, stable, terminated, truncated, finite, velocities, commands,
                      supported, pitch_errors=None, *, episode_steps, world_positions, previous_world_positions):
        batches = (stable, terminated, truncated, finite, velocities, commands, supported, episode_steps,
                   world_positions, previous_world_positions)
        if any(len(batch) != self.num_envs for batch in batches):
            raise ValueError("停止协议状态必须覆盖全部环境")
        if pitch_errors is not None and len(pitch_errors) != self.num_envs:
            raise ValueError("pitch误差必须覆盖全部环境")
        p = self.protocol
        for i in range(self.num_envs):
            if self.counts[i] >= self.quota:
                continue
            step = episode_steps[i]
            if step != self.steps[i] + 1 or step > p.episode_steps:
                raise ValueError("必须逐控制步采样 reset 前的 episode 时钟，不能漏步或复用旧 episode")
            if list(commands[i]) != p.command_at(step - 1):
                raise ValueError("动作所用指令与停止协议切换时刻不一致")
            self.steps[i] = step
            move, stop = self.phase_trackers[i]
            in_move = step <= p.move_steps
            phase = move if in_move else stop
            done = bool(terminated[i] or truncated[i])
            end_phase = step == p.move_steps if in_move else step == p.episode_steps
            if not in_move and end_phase and not done:
                raise ValueError("停止协议终点必须结束 episode")
            phase.update_motion(
                [bool(stable[i]) and bool(supported[i])], [bool(terminated[i])],
                [bool(truncated[i]) or end_phase], [bool(finite[i])], [velocities[i]],
                [commands[i]], [supported[i]], None if pitch_errors is None else [pitch_errors[i]],
                world_positions=[world_positions[i]], previous_world_positions=[previous_world_positions[i]],
            )
            if not done:
                continue
            move_record = dict(move.records[0]) if move.records else {"success": False}
            stop_record = dict(stop.records[0]) if stop.records else {"success": False}
            move_record["passed"] = move_record["success"]
            stop_record["passed"] = stop_record["success"]
            success = bool(move_record["passed"] and stop_record["passed"] and not terminated[i]
                           and truncated[i] and step == p.episode_steps)
            self.records.append({
                "env_id": i, "episode_index": self.counts[i], "success": success,
                "duration_s": step * p.dt, "preparation_time_s": 0.0, "initialization_failed": False,
                "terminal_stable_time_s": stop_record.get("terminal_stable_time_s", 0.0),
                "time_to_stable_s": stop_record.get("time_to_stable_s") if success else None,
                "time_to_confirmed_success_s": stop_record.get("time_to_confirmed_success_s") if success else None,
                "terminated": bool(terminated[i]), "truncated": bool(truncated[i]),
                "nonfinite_seen": move_record.get("nonfinite_seen", False) or stop_record.get("nonfinite_seen", False),
                "command_switch_time_s": p.move_steps * p.dt if step > p.move_steps else None,
                "stop_time_origin": "zero_velocity_command", "move_phase": move_record, "stop_phase": stop_record,
            })
            self.counts[i] += 1
            self.steps[i] = 0
            self.phase_trackers[i] = self._new_phases()

    def summary(self, minimum_success_rate):
        result = super().summary(minimum_success_rate)
        result["stop_protocol"] = self.protocol.metadata()
        result["gates"]["both_motion_and_stop_phases_required"] = True
        result["gates"]["time_origin"] = "zero_velocity_command_for_stop_metrics"
        return result


def apply_diagnostic_drive_override(env_cfg, leg_velocity_limit, *, solver_velocity_iterations=None):
    """隔离PhysX硬限速/求解迭代，保留默认训练与轮/mimic参数；不能当作正式验收。"""
    if leg_velocity_limit is not None and (not math.isfinite(leg_velocity_limit) or leg_velocity_limit <= 0):
        raise ValueError("诊断腿速度限制须为有限正数")
    if solver_velocity_iterations is not None and (
            type(solver_velocity_iterations) is not int or not 0 <= solver_velocity_iterations <= 255):
        raise ValueError("诊断速度迭代数须为0–255整数")
    result = {}
    if leg_velocity_limit is not None:
        actuators = env_cfg.scene.robot.actuators
        original = {name: actuators[name].velocity_limit_sim for name in ("suspension", "steering")}
        for name in original:
            actuators[name].velocity_limit_sim = leg_velocity_limit
        result.update(leg_velocity_limit_sim_rad_s=leg_velocity_limit, original_leg_velocity_limit_sim_rad_s=original)
    if solver_velocity_iterations is not None:
        props = env_cfg.scene.robot.spawn.articulation_props
        result.update(solver_velocity_iterations=solver_velocity_iterations,
                      original_solver_velocity_iterations=props.solver_velocity_iteration_count,
                      solver_iteration_source="articulation_configuration_before_spawn")
        props.solver_velocity_iteration_count = solver_velocity_iterations
    return result


def validate_getup_initial_speed(phase, speed_range):
    """带速初态只属于四轮起立；默认零速不得改变旧评测身份。"""
    if (len(speed_range) != 2 or not all(math.isfinite(v) for v in speed_range)
            or not -0.5 <= speed_range[0] <= speed_range[1] <= 0.5):
        raise ValueError("起立初速范围须有限、有序且在 [-0.5,0.5] m/s 内")
    moving = any(v != 0.0 for v in speed_range)
    if moving and phase != "getup":
        raise ValueError("带速初态仅适用于 --phase getup，不能改变 balance 初态")
    return moving


def capture_episode_initial_states(env, env_ids=None):
    """在显式/自动 reset 后采样实际状态，不用请求的速度区间冒充逐回合初速度。"""
    import torch

    ids = list(range(env.num_envs)) if env_ids is None else list(env_ids)
    robot = env.scene["robot"]
    values = {
        "root_state_w": robot.data.root_state_w[ids],
        "joint_pos": robot.data.joint_pos[ids],
        "joint_vel": robot.data.joint_vel[ids],
        "command_forward_yaw": env.command_manager.get_command("motion_command")[ids],
    }
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise ValueError("起立回合初态包含非有限值")
    snapshots = {key: value.detach().cpu().tolist() for key, value in values.items()}
    return {env_id: {
        **{key: value[row] for key, value in snapshots.items()},
        "joint_names": list(robot.joint_names),
        "sampling": "after_reset_before_first_policy_action",
        "contains_contact_solver_state": False,
    } for row, env_id in enumerate(ids)}


def evaluation_scope(phase, scenario, hold_time_s, *, diagnostic_drive=False, moving_getup=False):
    """允许短保持诊断，但不能把其成功率升级为标准验收。"""
    if diagnostic_drive:
        return "diagnostic_dynamics_override"
    standard = 5.0 if scenario == "stop" else (30.0 if phase == "balance" else 10.0)
    if hold_time_s is not None and hold_time_s < standard:
        return "custom_duration_smoke"
    return "synthetic_moving_getup_evaluation" if moving_getup else "standard_evaluation"


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_arg = parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phase", choices=tuple(TASKS), default="balance")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="stationary")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes_per_env", type=int, default=2)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--hold_time_s", type=float)
    parser.add_argument("--getup_initial_speed_range", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("MIN", "MAX"), help="合成四轮纵向初速 m/s，轮速同步；非真实策略接管。仅 getup。")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure_states", type=Path, help="另存失败末段NPZ；目标文件必须不存在。")
    parser.add_argument("--trace_output", type=Path, help="另存首回合前缀的跨引擎诊断JSON，禁止覆盖。")
    parser.add_argument("--trace_steps", type=int, default=12)
    parser.add_argument("--trace_env_id", type=int, default=0)
    parser.add_argument("--trace_physics", action="store_true", help="附加逐物理子步驱动和接触诊断；须配合trace_output。")
    parser.add_argument("--diagnostic_leg_velocity_limit", type=float,
                        help="仅覆盖隐式主动腿的PhysX速度约束；结果不可用于技能验收。")
    parser.add_argument("--diagnostic_solver_velocity_iterations", type=int, choices=range(256),
                        help="仅诊断覆盖articulation速度迭代数，不改变位置迭代数或默认训练。")
    AppLauncher.add_app_launcher_args(parser)
    checkpoint_arg.required = True
    args = parser.parse_args()
    try:
        validate_getup_initial_speed(args.phase, args.getup_initial_speed_range)
    except ValueError as exc:
        parser.error(str(exc))
    if args.num_envs < 1 or args.episodes_per_env < 1:
        parser.error("环境数及 episode 配额必须为正")
    if args.hold_time_s is not None and (not math.isfinite(args.hold_time_s) or args.hold_time_s <= 0):
        parser.error("保持时长必须是有限正数")
    if args.diagnostic_leg_velocity_limit is not None and (
            not math.isfinite(args.diagnostic_leg_velocity_limit) or args.diagnostic_leg_velocity_limit <= 0):
        parser.error("诊断腿速度限制须为有限正数")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error("checkpoint 不存在")
    if args.failure_states is not None and args.failure_states.exists():
        parser.error("failure_states 已存在，必须选择新文件保留原始证据")
    if args.output is not None and args.output.exists():
        parser.error("output 已存在，必须选择新文件保留原始证据")
    if not 1 <= args.trace_steps <= 1000 or not 0 <= args.trace_env_id < args.num_envs:
        parser.error("trace步数须为1–1000，环境索引须有效")
    if args.trace_output is not None:
        if args.trace_output.exists() or (args.output and args.trace_output.resolve() == args.output.resolve()):
            parser.error("trace_output须为新的独立文件，不得覆盖评测报告")
    elif args.trace_physics:
        parser.error("trace_physics需要trace_output")
    return args


def create_motion_environment(task, env_cfg, clip_actions):
    """wrapper 构造会立即 reset；失败也必须清除 raw env 的仿真停止回调。"""
    import gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    raw_env = gym.make(task, cfg=env_cfg)
    try:
        return RslRlVecEnvWrapper(raw_env, clip_actions=clip_actions)
    except BaseException:
        raw_env.close()
        raise


def main(args, simulation_app):
    from contextlib import ExitStack

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.managers import TerminationTermCfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.manager_based.upright.mdp.motion import MotionCommand, motion_state
    from robot_lab.tasks.manager_based.recovery.mdp.failure_states import FailureStateRecorder, file_sha256
    from motion_trace import MotionTraceRecorder

    task = TASKS[args.phase]
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    diagnostic_drive = apply_diagnostic_drive_override(
        env_cfg, args.diagnostic_leg_velocity_limit, solver_velocity_iterations=args.diagnostic_solver_velocity_iterations)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.events.reset_pose.params["quad_prob"] = float(args.phase == "getup")
    moving_getup = validate_getup_initial_speed(args.phase, args.getup_initial_speed_range)
    env_cfg.events.reset_pose.params["quad_forward_velocity_range"] = tuple(args.getup_initial_speed_range)
    initial_state_protocol = {
        "name": "synthetic_straight_rolling_quad" if moving_getup else "stationary_reset",
        "quad_forward_velocity_range_m_s": list(args.getup_initial_speed_range),
        "wheel_initialization": "urdf_radius_and_axis_no_longitudinal_slip" if moving_getup else "zero",
        "steering_initialization": "zero_for_nonzero_speed_quad_samples" if moving_getup else "configured_noise",
        "velocity_frame": "world_x_equals_initial_body_forward",
        "motion_command_relation": "independent_fixed_scenario_command_not_previous_episode_command",
        "is_locomotion_policy_handoff": False,
    }
    env_cfg.commands.pitch_command.pitch_range = (math.pi / 2, math.pi / 2)
    forward, yaw, stage = SCENARIOS[args.scenario]
    command_cfg = env_cfg.commands.motion_command
    command_cfg.fixed_stage = stage
    command_cfg.fixed_command = (forward, yaw)
    command_cfg.resampling_time_range = (1e9, 1e9)
    deadline, default_hold = (1.0, 30.0) if args.phase == "balance" else (3.0, 10.0)
    hold = default_hold if args.hold_time_s is None else args.hold_time_s
    env_cfg.episode_length_s = deadline + hold
    tracker = MotionEvaluation(args.num_envs, args.episodes_per_env, env_cfg.sim.dt * env_cfg.decimation, hold, deadline)
    if args.scenario == "stop":
        protocol = StopProtocol(env_cfg.sim.dt * env_cfg.decimation, move_deadline_s=deadline,
                                stop_hold_time_s=5.0 if args.hold_time_s is None else args.hold_time_s)

        class StopCommand(MotionCommand):
            """在 command manager 更新点切换，下一次观测和下一次动作使用同一命令。"""

            def _update_command(self):
                self.command[:, 0] = torch.where(self._env.episode_length_buf < protocol.move_steps,
                                                  protocol.forward_speed, 0.0)
                self.command[:, 1] = 0.0
                # 父类负责清空命令切换前的连续成功计时，并在切换停车时
                # 锁存当前世界 XY；否则减速位移会被错误地重基准。
                super()._update_command()

        command_cfg.class_type = StopCommand
        env_cfg.episode_length_s = protocol.episode_steps * protocol.dt
        tracker = StopEvaluation(args.num_envs, args.episodes_per_env, protocol)

    recorder = None
    trace = MotionTraceRecorder(args.trace_env_id, args.trace_steps) if args.trace_output is not None else None

    def capture_terminal_state(env):
        # termination 最后一个 term 在 reset 之前采样；不能拿 reset 后姿态冒充末态。
        if trace is not None:
            trace.capture_post(env)
        state = motion_state(env)
        env._motion_evaluation_sample = {
            key: state[key].cpu().tolist()
            for key in ("stable", "finite", "velocity", "command", "supported", "pitch_error")
        }
        env._motion_evaluation_sample["episode_steps"] = env.episode_length_buf.cpu().tolist()
        env._motion_evaluation_sample["world_xy"] = env.scene["robot"].data.root_pos_w[:, :2].cpu().tolist()
        if recorder is not None:
            robot = env.scene["robot"]
            roots = robot.data.root_state_w.clone()
            roots[:, :3] -= env.scene.env_origins
            recorder.append(roots.cpu().numpy(), robot.data.joint_pos.cpu().numpy(),
                            robot.data.joint_vel.cpu().numpy(), env.action_manager.action.cpu().numpy(),
                            env.episode_length_buf.cpu().tolist())
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    env_cfg.terminations.motion_snapshot = TerminationTermCfg(func=capture_terminal_state)
    env = create_motion_environment(task, env_cfg, agent_cfg.clip_actions)
    trace_hooks = ExitStack()
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.reset()
        dynamics_parameters = capture_dynamics_parameters(env.unwrapped)
        initial_states = capture_episode_initial_states(env.unwrapped)
        if args.trace_physics:
            trace_hooks.enter_context(trace.physics_hooks(env.unwrapped))
        if args.failure_states is not None:
            recorder = FailureStateRecorder(args.num_envs, env.unwrapped.scene["robot"].joint_names, {
                "task": task, "scenario": args.scenario, "seed": args.seed,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
                "asset_sha256": file_sha256(env_cfg.scene.robot.spawn.asset_path),
                "control_dt_s": env.unwrapped.step_dt, "terrain": "plane",
                "initial_state_protocol": initial_state_protocol,
            })
        if obs["policy"].shape[-1] != 44 or obs["critic"].shape[-1] != 49:
            raise RuntimeError("motion 评测必须使用 44/49 维任务，禁止混用旧 42/47 维 checkpoint")
        while not tracker.complete and simulation_app.is_running():
            with torch.inference_mode():
                previous_world_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].cpu().tolist()
                actions = policy(obs)
                if actions.shape != (args.num_envs, 12) or not torch.isfinite(actions).all():
                    raise RuntimeError("motion 策略必须输出有限的 12 维动作")
                if trace is not None:
                    trace.begin(env.unwrapped, obs["policy"], actions)
                obs, _, dones, _ = env.step(actions)
                if trace is not None:
                    trace.finish(env.unwrapped.reset_terminated[args.trace_env_id].item(),
                                 env.unwrapped.reset_time_outs[args.trace_env_id].item())
            sample = env.unwrapped._motion_evaluation_sample
            before = len(tracker.records)
            tracker.update_motion(
                sample["stable"], env.unwrapped.reset_terminated.cpu().tolist(),
                env.unwrapped.reset_time_outs.cpu().tolist(), sample["finite"],
                sample["velocity"], sample["command"], sample["supported"], sample["pitch_error"],
                world_positions=sample["world_xy"], previous_world_positions=previous_world_xy,
                **({"episode_steps": sample["episode_steps"]} if args.scenario == "stop" else {}),
            )
            for record in tracker.records[before:]:
                record["initial_state"] = initial_states[record["env_id"]]
            reset_ids = dones.nonzero(as_tuple=True)[0].cpu().tolist()
            if reset_ids:
                # 先归档旧回合，再采自动 reset 后的新回合；不可用新状态覆盖旧初态。
                initial_states.update(capture_episode_initial_states(env.unwrapped, reset_ids))
            if recorder is not None:
                records = {record["env_id"]: record for record in tracker.records[before:]}
                for index, done in enumerate(dones.cpu().tolist()):
                    if done:
                        record = records.get(index)
                        recorder.finish(index, failed=record is not None and not record["success"],
                                        label={} if record is None else {
                                            "env_id": index, "episode_index": record["episode_index"],
                                            "terminated": record["terminated"], "truncated": record["truncated"],
                                        })
            runner.alg.policy.reset(dones)
    finally:
        trace_hooks.close()
        env.close()
    result = tracker.summary(0.95 if args.phase == "balance" else 0.85)
    if recorder is not None:
        result["failure_states"] = recorder.save(args.failure_states)
    if trace is not None:
        result["motion_trace"] = trace.save(args.trace_output, {
            "task": task, "scenario": args.scenario, "seed": args.seed,
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
            "asset_sha256": file_sha256(env_cfg.scene.robot.spawn.asset_path),
            "physics_dt_s": env_cfg.sim.dt, "decimation": env_cfg.decimation,
            "control_dt_s": env_cfg.sim.dt * env_cfg.decimation,
            "dynamics_parameters": dynamics_parameters,
            "diagnostic_drive_override": diagnostic_drive,
            "initial_state_protocol": initial_state_protocol,
        })
    with args.checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    result.update({
        "task": task, "phase": args.phase, "scenario": args.scenario, "seed": args.seed,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
        "protocol": "move_then_stop_true_rear_wheel_support_v2" if args.scenario == "stop" else "fixed_command_true_rear_wheel_support_v2",
        "motion_evaluation_protocol": "motion_v2_world_drift",
        "evaluation_scope": evaluation_scope(args.phase, args.scenario, args.hold_time_s,
                                             diagnostic_drive=bool(diagnostic_drive), moving_getup=moving_getup),
        "initial_state_protocol": initial_state_protocol,
        "diagnostic_drive_override": diagnostic_drive,
        "fixed_stage": stage, "command_forward_yaw": [forward, yaw], "lin_y_command": 0.0,
        "scope": "flat_ground_move_then_stop" if args.scenario == "stop" else "flat_ground_fixed_command_motion",
        "observation_dimensions": {"actor": 44, "critic": 49}, "action_dimension": 12,
        "wheel_action_scale": 0.8, "episode_length_s": env_cfg.episode_length_s,
        "tracking_tolerances": {"forward_m_s": 0.10, "yaw_rad_s": 0.20, "lateral_m_s": 0.15},
        "minimum_integrated_progress_ratio": 0.5,
        "stationary_drift_limit_m": 0.50 if args.scenario == "stop" else 0.30,
        "position_reference": "world_xy_before_first_physics_step_of_phase",
        "dynamics_parameters": dynamics_parameters,
    })
    result["passed"] &= result["evaluation_scope"] in ("standard_evaluation", "synthetic_moving_getup_evaluation")
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    return result


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    args = parse_args()
    app = AppLauncher(args).app
    exit_code = 2
    try:
        result = main(args, app)
        exit_code = 0 if result["passed"] else 1
    except Exception:
        # 先报告原始异常，避免仿真器关闭异常或渲染等待吞掉真正的失败位置。
        import traceback
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(exit_code)
        app.close()
    raise SystemExit(exit_code)
