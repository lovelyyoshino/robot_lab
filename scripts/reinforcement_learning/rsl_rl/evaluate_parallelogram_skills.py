#!/usr/bin/env python3
"""平行四边形轮足车恢复与倒立策略的批量验收入口。

@author pony
@date 2026-09-16
@version v0.20.0
@last_modified 2026-09-30
@changelog
    - v0.20.0 (2026-09-30): 支持保持期闭环profile，拒绝缺谱系时猜测动作合同。
    - v0.19.0 (2026-09-30): 评测在创建Isaac环境前校验Residual先验NPZ哈希。
    - v0.18.0 (2026-09-30): 支持FailureReplay-Residual使用同一先验合同进行失败状态微调评测。
    - v0.17.0 (2026-09-30): 支持 upright_release 先验释放消融并记录 profile。
    - v0.16.0 (2026-09-30): 评估自动读取训练时 residual profile，阻止 wrapper 配置错配。
    - v0.15.0 (2026-09-30): 正式评估支持Side-Residual先验与同源动作合同校验。
    - v0.14.0 (2026-09-28): 侧躺/翻覆独立验收、镜像脚本示范、物理控制合同与响应统计。
    - v0.13.0 (2026-09-17): 可选保存严格成功脚本回合的动作前观测与reset前执行确认示范。
    - v0.12.0 (2026-09-17): 增加受限分段动作诊断，复用Grounded接管与reset前严格终态验收。
    - v0.11.0 (2026-09-17): 倾角课程增加零策略动作/主动PD对照，显式区分策略执行与加载。
    - v0.10.0 (2026-09-17): 独立controlled_tilt诊断；保留课程指标但强制不可作为正式验收。
    - v0.9.0 (2026-09-17): Grounded-Progress显式使用四轮稳定验收协议。
    - v0.8.0 (2026-09-17): 核查训练/评测池分离，显式同池烟测不能通过技能验收。
    - v0.7.0 (2026-09-17): 独立真实失败状态即时接管评测，修复upright读取不存在freefall配置。
    - v0.6.0 (2026-09-17): 明确准备期电机制动模式和关节名称，避免静默改变零力矩协议。
    - v0.5.0 (2026-09-17): 保存准备门失败计数、真实末态与接管有效标志，不把零缓存当接管。
    - v0.4.0 (2026-09-16): 接入地面即时恢复协议，计时分离；双轮验收要求真实后轮支撑。
    - v0.3.0 (2026-09-16): 稳定判据排除机体触地，记录接触比例与初始姿态，防止借地支撑。
    - v0.2.0 (2026-09-16): 采集 reset 前终态、固定逐环境配额、严格连续保持和完整性门。
    - v0.1.0 (2026-09-16): 新增分姿态恢复、倒立平衡和四轮起立的 JSON 指标评测。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from skill_evaluation import SkillEvaluation


RECOVERY_SCENARIOS = ("upright_fail", "side_lying", "prone", "uniform_so3")
TASKS = {
    "recovery": "RobotLab-Isaac-Recovery-Parallelogram-Rover-v0",
    "upright-balance": "RobotLab-Isaac-Upright-Parallelogram-Rover-v0",
    "upright-getup": "RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0",
}


def _validate_action_program_args(args):
    """诊断只复用严格Grounded单回合，不允许修改保持门或混用课程。"""
    program_defaults = {"program_seed": 731, "program_segment_s": .5, "program_active_s": 3.,
                        "program_leg_fraction": 1., "program_wheel_speed": 5.,
                        "program_id": None, "program_mirror": False}
    if not getattr(args, "diagnostic_action_program", False):
        if getattr(args, "demonstration_output", None) is not None:
            raise ValueError("demonstration_output只用于显式脚本诊断")
        if any(getattr(args, name, value) != value for name, value in program_defaults.items()):
            raise ValueError("program参数必须同时显式启用diagnostic_action_program")
        return
    if (args.skill != "recovery" or args.recovery_protocol != "settled_ground"
            or args.task not in (
                "RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0",
                "RobotLab-Isaac-Recovery-Grounded-Progress-Parallelogram-Rover-v0",
                "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0",
                "RobotLab-Isaac-Recovery-Upside-Parallelogram-Rover-v0",
            )
            or args.episodes_per_env != 1 or args.diagnostic_zero_action or args.hold_time_s is not None):
        raise ValueError("action_program只接受Grounded recovery/settled_ground单回合及默认保持门")
    if getattr(args, "program_id", None) is not None and not 0 <= args.program_id < args.num_envs:
        raise ValueError("program_id必须落在本次num_envs范围")
    if getattr(args, "program_mirror", False) and getattr(args, "scenario", None) != "side_lying":
        raise ValueError("program_mirror只用于明确side_lying场景")
    output = getattr(args, "output", None)
    if output is not None and output.exists():
        raise ValueError("action_program不会覆盖已有输出，请指定新证据路径")
    demonstration_output = getattr(args, "demonstration_output", None)
    if demonstration_output is not None:
        if output is None or demonstration_output.suffix != ".npz":
            raise ValueError("示范采集必须同时指定评测JSON与.npz输出")
        if demonstration_output.exists() or demonstration_output.resolve() == output.resolve():
            raise ValueError("示范输出已存在或与源JSON重叠，拒绝覆盖")


def _validate_controlled_tilt_args(args):
    """启动仿真前阻断课程和正式倒姿协议混用，不静默覆盖用户指定的场景。"""
    controlled = args.recovery_protocol == "controlled_tilt"
    if not controlled:
        if args.diagnostic_zero_action:
            raise ValueError("diagnostic_zero_action 只用于 controlled_tilt 诊断")
        if args.roll_tilt_range_deg is not None:
            raise ValueError("roll_tilt_range_deg 只用于 controlled_tilt 诊断")
        return
    if args.skill != "recovery" or args.scenario is not None:
        raise ValueError("controlled_tilt 只接受 recovery，不能指定正式 --scenario")
    if (args.settled_initialization != "near_ground" or args.preparation_leg_damping is not None
            or args.takeover_velocity_x or args.takeover_pitch_rate
            or args.failure_states is not None or getattr(args, "failure_states_output", None) is not None
            or args.allow_training_pool_smoke):
        raise ValueError("controlled_tilt 不接受沉降、准备制动、速度注入或失败池参数")
    if args.roll_tilt_range_deg is not None:
        bounds = args.roll_tilt_range_deg
        if (len(bounds) != 2 or not all(math.isfinite(value) for value in bounds)
                or not 0 < bounds[0] <= bounds[1] <= 180):
            raise ValueError("roll_tilt_range_deg 必须满足 0 < min <= max <= 180 度")


def _evaluation_actions(policy, obs, action_template, zero_action):
    """零策略动作仍经过默认关节目标及PD，不是零力矩或关闭电机的被动跌落。"""
    import torch

    return torch.zeros_like(action_template) if zero_action else policy(obs)


def _classify_recovery_failure(record):
    """把终态按姿态分桶，避免只报告一个 success_rate 而看不出失败机制。"""
    if record.get("success"):
        return "success"
    if record.get("initialization_failed"):
        return "initialization_failure"
    gravity = record.get("terminal_gravity_b")
    if not isinstance(gravity, list) or len(gravity) != 3 or not all(math.isfinite(float(v)) for v in gravity):
        return "nonfinite_or_missing_terminal_state"
    if gravity[2] > 0.75:
        return "upside_down"
    if abs(gravity[1]) > 0.70 and abs(gravity[2]) < 0.70:
        return "side_lying"
    if gravity[2] > -0.70:
        return "not_upright"
    return "unstable_upright"


def _mark_controlled_tilt_diagnostic(result, roll_tilt_range_deg, zero_action=False):
    """课程即使100%成功也不能通过正式验收；保留实际初始横滚用于按难度分桶。"""
    result["evaluation_scope"] = "controlled_tilt_pretraining_diagnostic_not_recovery_acceptance"
    result["acceptance_eligible"] = False
    result["diagnostic_passed"] = bool(result["passed"])
    result["passed"] = False
    result["action_source"] = "zero_policy_action_with_active_pd" if zero_action else "checkpoint_policy"
    result["checkpoint_policy_executed"] = not zero_action
    result["roll_tilt_range_deg"] = list(roll_tilt_range_deg)
    for episode in result["episode_results"]:
        root = episode.get("takeover_root_state_w")
        if root is None:
            episode["takeover_roll_deg"] = None
            continue
        w, x, y, z = root[3:7]
        episode["takeover_roll_deg"] = math.degrees(math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)))


def parse_args():
    """先注册 AppLauncher，再标记必填参数，兼容其内部提前 parse_known_args。"""
    parser = argparse.ArgumentParser(description="评测平行四边形轮足车 recovery/upright checkpoint。")
    required = [parser.add_argument("--task"), parser.add_argument("--checkpoint")]
    required.append(parser.add_argument("--skill", choices=tuple(TASKS)))
    parser.add_argument("--scenario", default=None, choices=RECOVERY_SCENARIOS)
    parser.add_argument("--recovery_protocol", choices=("drop_then_recover", "settled_ground", "failure_replay", "controlled_tilt"), default="drop_then_recover")
    parser.add_argument("--roll_tilt_range_deg", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="仅controlled_tilt诊断：双侧横滚角段，默认35–55度，不属于正式恢复验收。")
    parser.add_argument("--diagnostic_zero_action", action="store_true",
                        help="仅controlled_tilt：加载同一checkpoint保留初始化路径，但不执行策略；零动作仍有PD控制。")
    parser.add_argument("--diagnostic_action_program", action="store_true",
                        help="仅Grounded单回合：分段关节目标搜索，不执行加载的策略，结果不作技能验收。")
    parser.add_argument("--program_seed", type=int, default=731)
    parser.add_argument("--program_id", type=int, help="复用该随机程序；仍在接收环境逐项校验物理目标。")
    parser.add_argument("--program_mirror", action="store_true", help="side_lying按初始倒向镜像固定程序。")
    parser.add_argument("--recovery_motor_profile", type=Path, help="实测电机JSON；默认读取checkpoint标定快照。")
    parser.add_argument("--recovery_prior_data", type=Path,
                        help="Side-Residual策略的program NPZ；须与训练时先验相同。")
    parser.add_argument("--recovery_prior_program_id", type=int, default=31)
    parser.add_argument("--recovery_residual_profile",
                        choices=("auto", "constant", "launch_protected", "upright_release", "settle_feedback"),
                        default="auto",
                        help="Side-Residual residual profile；auto从checkpoint谱系读取。")
    parser.add_argument("--program_segment_s", type=float, default=0.5)
    parser.add_argument("--program_active_s", type=float, default=3.0)
    parser.add_argument("--program_leg_fraction", type=float, default=1.0,
                        help="从默认位置到逐关节软限位的比例[0,1]，不改变动作倍率或物理限位。")
    parser.add_argument("--program_wheel_speed", type=float, default=5.0,
                        help="分段目标轮速的最大幅值rad/s，仍受原物理速度与力矩上限约束。")
    parser.add_argument("--demonstration_output", type=Path,
                        help="仅脚本诊断：保存完整严格成功回合的动作前58维观测/实际12维动作，非策略成绩。")
    parser.add_argument("--failure_states", type=Path, help="failure_replay必须使用独立评测池，不能把训练池成绩当泛化。")
    parser.add_argument(
        "--failure_states_output", type=Path,
        help="普通恢复评测另存失败末态NPZ，供后续FailureReplay-Residual微调；目标必须不存在。",
    )
    parser.add_argument("--allow_training_pool_smoke", action="store_true",
                        help="允许同池或未知来源的接口诊断，结果强制不作为验收通过。")
    parser.add_argument("--settled_initialization", choices=("near_ground", "drop"), default="near_ground",
                        help="settled_ground 近地倒姿或高处跌落对照；两者使用相同沉降稳定门。")
    parser.add_argument("--takeover_velocity_x", type=float, default=0.0)
    parser.add_argument("--takeover_pitch_rate", type=float, default=0.0)
    parser.add_argument("--preparation_leg_damping", type=float, default=None,
                        help="覆盖任务的准备腿阻尼(Nm*s/rad)；settled默认0，grounded默认0.2。")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--episodes_per_env", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold_time_s", type=float, default=None, help="默认恢复 0.5s，平衡 30s，起立 10s。")
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    for argument in required:
        argument.required = True
    args = parser.parse_args()
    try:
        _validate_controlled_tilt_args(args)
        _validate_action_program_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.scenario is None and args.recovery_protocol != "controlled_tilt":
        args.scenario = ("side_lying" if "Recovery-Side-" in args.task else
                         "prone" if "Recovery-Upside-" in args.task else "upright_fail")
    if args.num_envs < 1 or args.episodes_per_env < 1:
        parser.error("num_envs 和 episodes_per_env 必须为正")
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint 不存在：{args.checkpoint}")
    if args.failure_states_output is not None:
        args.failure_states_output = args.failure_states_output.expanduser().resolve()
        if args.failure_states_output.exists():
            parser.error("failure_states_output 已存在，必须选择新文件保留原始失败证据")
        if args.output is not None and args.failure_states_output == args.output.expanduser().resolve():
            parser.error("failure_states_output 不能与 output JSON 相同")
    if args.recovery_prior_data is not None:
        args.recovery_prior_data = args.recovery_prior_data.expanduser().resolve()
        if not args.recovery_prior_data.is_file():
            parser.error(f"recovery_prior_data 不存在：{args.recovery_prior_data}")
    return args


def main(args, simulation_app) -> dict:
    """终态通过最后一个 termination term 采集，避免自动 reset 抹掉失败姿态。"""
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.managers import TerminationTermCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401
    from recovery_contract import (
        GROUNDED_TASKS, configure_physics, control_contract,
        is_residual_recovery_task, validate_checkpoint_contract,
    )
    from robot_lab.tasks.manager_based.upright.mdp.support import support_masks

    expected_task = TASKS[args.skill]
    _validate_controlled_tilt_args(args)
    _validate_action_program_args(args)
    action_program = None
    action_program_joint_names = None
    demonstration_collector = None
    failure_state_recorder = None
    replay = args.skill == "recovery" and args.recovery_protocol == "failure_replay"
    controlled_tilt = args.skill == "recovery" and args.recovery_protocol == "controlled_tilt"
    if args.failure_states_output is not None and (
            args.skill != "recovery" or args.recovery_protocol != "settled_ground"
            or replay or controlled_tilt or args.diagnostic_action_program):
        raise ValueError("failure_states_output只用于settled_ground普通策略评测，不能混入重放或脚本诊断")
    # 此分组共用独立控制时钟；controlled_tilt/replay并未经过被动沉降。
    settled = args.skill == "recovery" and args.recovery_protocol in ("settled_ground", "failure_replay", "controlled_tilt")
    grounded = replay or controlled_tilt or (settled and args.task in GROUNDED_TASKS)
    if settled:
        replay_task = (
            "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0"
            if args.task.endswith("FailureReplay-Residual-Parallelogram-Rover-v0") else
            "RobotLab-Isaac-Recovery-FailureReplay-Parallelogram-Rover-v0"
        )
        expected_task = ("RobotLab-Isaac-Recovery-ControlledTilt-Parallelogram-Rover-v0" if controlled_tilt else
                         replay_task if replay else
                         args.task if grounded else "RobotLab-Isaac-Recovery-Settled-Parallelogram-Rover-v0")
    if args.task != expected_task:
        raise ValueError(f"当前协议需要任务 {expected_task}，实际为 {args.task}")
    if not settled and (args.takeover_velocity_x or args.takeover_pitch_rate):
        raise ValueError("接管速度只用于 settled_ground 协议")
    if replay and (args.takeover_velocity_x or args.takeover_pitch_rate or args.preparation_leg_damping):
        raise ValueError("failure_replay保留记录动量并立即控制，不能注入准备阻尼或额外速度")
    if replay and (args.failure_states is None or not args.failure_states.is_file()):
        raise ValueError("failure_replay需要存在的 --failure_states NPZ")
    if args.failure_states is not None and not replay:
        raise ValueError("failure_states只用于failure_replay协议")
    if args.allow_training_pool_smoke and not replay:
        raise ValueError("allow_training_pool_smoke只用于failure_replay协议")
    pool_separation = None
    if replay:
        from robot_lab.tasks.manager_based.recovery.mdp.failure_states import evaluation_pool_status

        pool_separation = evaluation_pool_status(
            args.checkpoint, args.failure_states, allow_training_pool_smoke=args.allow_training_pool_smoke,
        )
    if args.preparation_leg_damping is not None and (
        not math.isfinite(args.preparation_leg_damping) or args.preparation_leg_damping < 0
    ):
        raise ValueError("preparation_leg_damping 必须为有限非负数")
    if not settled and args.preparation_leg_damping:
        raise ValueError("准备阻尼只用于 settled_ground 协议")
    defaults = {"recovery": (0.5, 4.0), "upright-balance": (30.0, 1.0), "upright-getup": (10.0, 3.0)}
    hold_time, deadline = defaults[args.skill]
    if args.hold_time_s is not None:
        hold_time = args.hold_time_s
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    configure_physics(env_cfg, args.task, getattr(args, "recovery_motor_profile", None), args.checkpoint)
    recovery_control = control_contract(env_cfg.to_dict(), agent_cfg.to_dict()) if args.skill == "recovery" else None
    if recovery_control is not None and not args.diagnostic_action_program:
        validate_checkpoint_contract(args.checkpoint, recovery_control)
    if args.preparation_leg_damping is None:
        freefall = getattr(env_cfg.events, "freefall_init", None)
        args.preparation_leg_damping = freefall.params.get("preparation_leg_damping", 0.0) if freefall else 0.0
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args.skill == "recovery":
        if replay:
            env_cfg.events.reset_fall_pose.params["failure_state_path"] = str(args.failure_states.resolve())
        elif controlled_tilt:
            if args.roll_tilt_range_deg is not None:
                env_cfg.events.reset_fall_pose.params["roll_tilt_range_deg"] = tuple(args.roll_tilt_range_deg)
        else:
            env_cfg.events.reset_fall_pose.params["pose_weights"] = {
                name: float(name == args.scenario) for name in RECOVERY_SCENARIOS
            }
        env_cfg.episode_length_s = max(5.0, deadline + hold_time)
        if settled and not (replay or controlled_tilt):
            env_cfg.events.freefall_init.params["preparation_leg_damping"] = args.preparation_leg_damping
            env_cfg.events.reset_fall_pose.params["collision_clearance"] = (
                0.002 if args.settled_initialization == "near_ground" else None
            )
        if settled and not (replay or controlled_tilt) and (args.takeover_velocity_x or args.takeover_pitch_rate):
            env_cfg.events.freefall_hold.params["takeover_velocity_range"] = {
                "x": (args.takeover_velocity_x,) * 2,
                "pitch": (args.takeover_pitch_rate,) * 2,
            }
    else:
        env_cfg.events.reset_pose.params["quad_prob"] = float(args.skill == "upright-getup")
        # 指令固定在 90°，隔离随机指令跳变对连续稳定性评测的影响。
        env_cfg.commands.pitch_command.pitch_range = (1.57, 1.57)
        env_cfg.episode_length_s = deadline + hold_time

    tracker = SkillEvaluation(
        args.num_envs, args.episodes_per_env, env_cfg.sim.dt * env_cfg.decimation, hold_time, deadline,
    )
    contact_samples = 0
    total_samples = 0

    def capture_terminal_state(env):
        """此 term 永不触发终止，只保留物理步末状态；后续读取不再访问 reset 后机器人。"""
        robot = env.scene["robot"]
        gravity = robot.data.projected_gravity_b
        height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        omega = robot.data.root_ang_vel_b.norm(dim=1)
        finite = (
            torch.isfinite(robot.data.root_state_w).all(dim=1)
            & torch.isfinite(robot.data.joint_pos).all(dim=1)
            & torch.isfinite(robot.data.joint_vel).all(dim=1)
        )
        if failure_state_recorder is not None:
            roots = robot.data.root_state_w.clone()
            roots[:, :3] -= env.scene.env_origins
            failure_state_recorder.append(
                roots.cpu().numpy(), robot.data.joint_pos.cpu().numpy(),
                robot.data.joint_vel.cpu().numpy(), env.action_manager.action.cpu().numpy(),
                env.episode_length_buf.cpu().tolist(),
            )
        # 传感器与 articulation 的 body 顺序不同，所有接触查询以传感器命名为准。
        sensor = env.scene.sensors["contact_forces"]
        base_ids, _ = sensor.find_bodies("base_link")
        body_contact = sensor.data.net_forces_w[:, base_ids].norm(dim=-1).amax(dim=1) > 1.0
        no_body_support, two_wheel_support = support_masks(sensor.data.net_forces_w, sensor.body_names)
        pitch_error = None
        if args.skill == "recovery":
            # 排除空中姿态经过正立的瞬间，并要求角速度足够低。
            stable = (gravity[:, 2] < -0.7) & (height >= 0.26) & (height <= 0.45) & (omega < 0.75)
            if not settled:
                stable &= env.episode_length_buf > 100
            if grounded:
                from robot_lab.tasks.manager_based.recovery.mdp.grounded import recovery_stable_hold

                stable &= recovery_stable_hold(env).bool()
        else:
            pitch = torch.atan2(-gravity[:, 0], -gravity[:, 2])
            pitch_error = pitch - env.command_manager.get_command("pitch_command")[:, 0]
            stable = (pitch_error.abs() < 0.20) & (height >= 0.30) & (height <= 0.65)
            stable &= (omega < 0.75) & (gravity[:, 1].abs() < 0.25)
            stable &= two_wheel_support
        stable &= no_body_support
        if action_program is not None:
            from robot_lab.tasks.manager_based.recovery.mdp.grounded import recovery_four_wheel_support

            action_program.observe(
                gravity[:, 2], stable, recovery_four_wheel_support(env).bool(),
                env._recovery_control_active & (env._recovery_control_steps > 0), finite,
            )
            action_program.observe_joints(
                robot.data.joint_pos[:, action_program.joint_ids],
                robot.data.applied_torque[:, action_program.joint_ids],
                env._recovery_control_active & (env._recovery_control_steps > 0),
                velocities=robot.data.joint_vel[:, action_program.joint_ids],
            )
        if settled:
            active = env._recovery_control_active & (env._recovery_control_steps > 0)
            initialization_failed = env._recovery_initialization_failed
            phase = {
                "active": active.cpu().tolist(),
                "failed": initialization_failed.cpu().tolist(),
                "failure_codes": env._recovery_initialization_failure_code.cpu().tolist(),
                "takeover_root_state_w": env._recovery_takeover_root_state_w.cpu().tolist(),
                "takeover_joint_pos": env._recovery_takeover_joint_pos.cpu().tolist(),
                "takeover_joint_vel": env._recovery_takeover_joint_vel.cpu().tolist(),
                "takeover_valid": env._recovery_control_active.cpu().tolist(),
                "preparation_gate_failures": env._recovery_preparation_gate_failures.cpu().tolist(),
                "preparation_metrics": {name: value.cpu().tolist() for name, value in env._recovery_preparation_metrics.items()},
                "terminal_root_state_w": robot.data.root_state_w.cpu().tolist(),
                "terminal_joint_vel": robot.data.joint_vel.cpu().tolist(),
                "terminal_gravity_b": gravity.cpu().tolist(),
            }
        else:
            phase = {"active": [True] * env.num_envs, "failed": [False] * env.num_envs}
        if demonstration_collector is not None:
            demonstration_collector.confirm_step(
                env._recovery_control_steps, env._recovery_control_active, env.action_manager.action,
                root_state_w=robot.data.root_state_w, joint_pos=robot.data.joint_pos, joint_vel=robot.data.joint_vel,
            )
        env._skill_eval_sample = (
            stable.cpu().tolist(), finite.cpu().tolist(),
            pitch_error.cpu().tolist() if pitch_error is not None else None,
            body_contact.cpu().tolist(), phase,
        )
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    env_cfg.terminations.evaluation_snapshot = TerminationTermCfg(func=capture_terminal_state)
    # 构建manager前保存声明式合同，避免运行时解析SceneEntityCfg和RSL pop字段改变记录。
    demonstration_contract = None
    if args.demonstration_output is not None:
        from recovery_demonstrations import canonical_config

        demonstration_contract = canonical_config({
            "actor_observation_dim": 58, "critic_observation_dim": 70, "action_dim": 12,
            "step_dt": tracker.dt, "policy_config": agent_cfg.to_dict()["policy"],
            "wrapper_raw_clip": agent_cfg.clip_actions,
            "control_contract": recovery_control,
            "observation_group": env_cfg.observations.policy.to_dict(),
            "action_config": env_cfg.actions.to_dict(),
            "default_joint_state": {"joint_pos": env_cfg.scene.robot.init_state.joint_pos,
                                    "joint_vel": env_cfg.scene.robot.init_state.joint_vel},
        })
    residual_profile = None
    if is_residual_recovery_task(args.task):
        if args.recovery_prior_data is None:
            raise ValueError("Residual恢复评测必须提供 --recovery_prior_data")
        from recovery_prior import resolve_checkpoint_prior

        residual_profile = resolve_checkpoint_prior(
            args.checkpoint, args.recovery_prior_data,
            program_id=args.recovery_prior_program_id, task=args.task,
            residual_profile=args.recovery_residual_profile,
        )
    elif args.recovery_prior_data is not None:
        raise ValueError("--recovery_prior_data 只允许用于Residual恢复评测")
    raw_env = gym.make(args.task, cfg=env_cfg)
    if is_residual_recovery_task(args.task):
        from recovery_prior import attach_recovery_prior

        raw_env = attach_recovery_prior(
            raw_env, args.recovery_prior_data, program_id=args.recovery_prior_program_id,
            residual_profile=residual_profile,
        )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        joint_names = list(env.unwrapped.scene["robot"].joint_names)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.reset()
        if args.failure_states_output is not None:
            from robot_lab.tasks.manager_based.recovery.mdp.failure_states import FailureStateRecorder, file_sha256

            failure_state_recorder = FailureStateRecorder(
                args.num_envs, env.unwrapped.scene["robot"].joint_names,
                {
                    "task": args.task,
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": file_sha256(args.checkpoint),
                    "asset_sha256": file_sha256(env_cfg.scene.robot.spawn.asset_path),
                    "control_dt_s": env.unwrapped.step_dt,
                    "terrain": "plane",
                    "initialization": args.recovery_protocol,
                },
            )
        if args.diagnostic_action_program:
            from recovery_action_program import program_from_environment

            action_program, action_program_joint_names = program_from_environment(
                env.unwrapped, args, agent_cfg.clip_actions,
            )
            if demonstration_contract is not None:
                from recovery_demonstrations import RecoveryDemonstrationCollector

                manager = env.unwrapped.observation_manager
                if manager.group_obs_dim["policy"] != (58,) or manager.group_obs_dim["critic"] != (70,):
                    raise ValueError("示范采集只支持58/70恢复合同；阶段BC请显式转换")
                demonstration_contract.update({
                    "observation_order": list(manager.active_terms["policy"]),
                    "observation_term_dims": [list(shape) for shape in manager.group_obs_term_dim["policy"]],
                    "action_joint_names": action_program_joint_names,
                    "action_scales": action_program.scales[0].cpu().tolist(),
                    "action_offsets": action_program.defaults[0].cpu().tolist(),
                })
                if not ((action_program.scales == action_program.scales[0]).all()
                        and (action_program.defaults == action_program.defaults[0]).all()):
                    raise ValueError("暂不支持逐环境不同的动作合同")
                demonstration_collector = RecoveryDemonstrationCollector(
                    args.num_envs, env.unwrapped.max_episode_length, demonstration_contract,
                )
        failure_pool = getattr(env.unwrapped, "_failure_state_pool", None)
        failure_pool_metadata = None if failure_pool is None else {
            "path": str(Path(failure_pool["path"]).resolve()), "sha256": failure_pool["sha256"],
            "states": len(failure_pool["root_state"]), "scope": "kinematic_replay_not_solver_state_reproduction",
        }
        while not tracker.complete and simulation_app.is_running():
            with torch.inference_mode():
                if action_program is None:
                    actions = _evaluation_actions(
                        policy, obs, env.unwrapped.action_manager.action, args.diagnostic_zero_action,
                    )
                else:
                    actions = action_program.actions(
                        env.unwrapped._recovery_control_steps, env.unwrapped._recovery_control_active,
                    )
                if not torch.isfinite(actions).all():
                    raise RuntimeError("策略输出非有限动作，终止评测")
                if demonstration_collector is not None:
                    robot = env.unwrapped.scene["robot"]
                    demonstration_collector.before_step(
                        obs["policy"], actions, env.unwrapped._recovery_control_steps,
                        env.unwrapped._recovery_control_active, root_state_w=robot.data.root_state_w,
                        joint_pos=robot.data.joint_pos, joint_vel=robot.data.joint_vel,
                    )
                obs, _, dones, _ = env.step(actions)
            stable, finite, pitch_errors, body_contacts, phase = env.unwrapped._skill_eval_sample
            for index, contact in enumerate(body_contacts):
                if tracker.counts[index] < tracker.quota and phase["active"][index]:
                    contact_samples += int(contact)
                    total_samples += 1
            before = len(tracker.records)
            tracker.update(
                stable, env.unwrapped.reset_terminated.cpu().tolist(),
                env.unwrapped.reset_time_outs.cpu().tolist(), finite, pitch_errors,
                active=phase["active"], initialization_failed=phase["failed"],
            )
            if settled:
                for record in tracker.records[before:]:
                    index = record["env_id"]
                    for key in ("failure_codes", "takeover_root_state_w", "takeover_joint_pos", "takeover_joint_vel"):
                        record[key] = phase[key][index]
                    record["takeover_valid"] = phase["takeover_valid"][index]
                    if not record["takeover_valid"]:
                        for key in ("takeover_root_state_w", "takeover_joint_pos", "takeover_joint_vel"):
                            record[key] = None
                    record["preparation_gate_failure_steps"] = dict(zip(
                        env.unwrapped._recovery_preparation_gate_names, phase["preparation_gate_failures"][index],
                    ))
                    record["preparation_terminal_metrics"] = {
                        name: values[index] if math.isfinite(values[index]) else None
                        for name, values in phase["preparation_metrics"].items()
                    }
                    for key in ("terminal_root_state_w", "terminal_joint_vel", "terminal_gravity_b"):
                        record[key] = [value if math.isfinite(value) else None for value in phase[key][index]]
                    if args.skill == "recovery":
                        record["failure_state"] = _classify_recovery_failure(record)
                    if action_program is not None:
                        record["action_program_diagnostic"] = action_program.episode_metrics(index)
                        action_program.finished[index] = True
                    if demonstration_collector is not None:
                        demonstration_collector.finish_episode(record)
                    if failure_state_recorder is not None:
                        failure_state_recorder.finish(
                            index, failed=not bool(record["success"]),
                            label={
                                "env_id": record["env_id"],
                                "episode_index": record["episode_index"],
                                "terminated": record["terminated"],
                                "truncated": record["truncated"],
                                "failure_state": record.get("failure_state"),
                            },
                        )
            runner.alg.policy.reset(dones)
    finally:
        env.close()

    required_rate = 0.95 if args.skill == "upright-balance" else 0.85
    if args.skill == "recovery" and not (replay or controlled_tilt) and args.scenario == "uniform_so3":
        required_rate = 0.60
    result = tracker.summary(required_rate)
    with args.checkpoint.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result.update({
        "skill": args.skill, "task": args.task, "seed": args.seed,
        "scenario": ("continuous_roll_tilt" if controlled_tilt else "recorded_failure_states" if replay
                     else args.scenario if args.skill == "recovery" else None),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
        "step_dt": tracker.dt, "episode_length_s": env_cfg.episode_length_s,
        "protocol": args.recovery_protocol if args.skill == "recovery" else "fixed_90_degree_rear_wheel_support_v2",
        "recovery_support_protocol": "four_wheel_stable_v1" if grounded else "legacy_pose_height_v1" if args.skill == "recovery" else None,
        "settled_initialization": args.settled_initialization if settled and not (replay or controlled_tilt) else None,
        "controlled_tilt_initialization": "urdf_collision_clearance_2mm_immediate_control" if controlled_tilt else None,
        "failure_states": failure_pool_metadata,
        "failure_pool_separation": pool_separation,
        "preparation_leg_damping_nm_s_rad": args.preparation_leg_damping if settled else None,
        "preparation_control_mode": "immediate_takeover" if replay or controlled_tilt else (
            "leg_damping_brake" if args.preparation_leg_damping else "zero_torque") if settled else None,
        "recovery_residual_profile": residual_profile,
        "joint_names": joint_names,
        "takeover_velocity_x": args.takeover_velocity_x, "takeover_pitch_rate": args.takeover_pitch_rate,
        "upright_initialization": "urdf_collision_clearance_2mm" if args.skill != "recovery" else None,
        "base_contact_force_threshold_n": 1.0,
        "base_contact_fraction": contact_samples / total_samples if total_samples else None,
        "control_contract": recovery_control,
        "hardware_feasibility_verified": False,
    })
    if replay:
        result["evaluation_scope"] = pool_separation["scope"]
        result["gates"]["failure_pool_data_separation_verified"] = pool_separation["data_separation_verified"]
        result["passed"] &= pool_separation["data_separation_verified"]
    if controlled_tilt:
        _mark_controlled_tilt_diagnostic(
            result, env_cfg.events.reset_fall_pose.params["roll_tilt_range_deg"], args.diagnostic_zero_action,
        )
    if action_program is not None:
        action_program.mark_diagnostic(result)
        result["action_program"]["joint_names"] = action_program_joint_names
    if demonstration_collector is not None:
        # JSON先独占落盘，再把其真实哈希写入NPZ；避免两文件相互哈希引用。
        result["demonstration_dataset"] = {
            "path": str(args.demonstration_output.resolve()), "requested": True,
            "successful_episodes": sum(record["success"] for record in result["episode_results"]),
        }
    if failure_state_recorder is not None:
        result["failure_states_output"] = {
            "path": str(args.failure_states_output.resolve()),
            "requested": True,
            "failed_episodes": sum(not record["success"] for record in result["episode_results"]),
        }
        result["failure_states_output"].update(failure_state_recorder.save(args.failure_states_output))
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 新诊断不得覆盖已有证据；原策略评测的既有文件行为保持不变。
        with args.output.open("x" if action_program is not None else "w", encoding="utf-8") as output:
            output.write(payload + "\n")
    if demonstration_collector is not None:
        manifest = demonstration_collector.save(args.demonstration_output, args.output, result)
        print("RECOVERY_DEMONSTRATIONS " + json.dumps(manifest, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    args = parse_args()
    app = AppLauncher(args).app
    exit_code = 2
    try:
        result = main(args, app)
        # 课程退出码只表示诊断完整；JSON的passed始终为false，不冒充技能验收。
        diagnostic_complete = (args.recovery_protocol == "controlled_tilt" or args.diagnostic_action_program) and result["complete"]
        exit_code = 0 if result["passed"] or diagnostic_complete else 1
    finally:
        # 显式通知 Kit；禁用 fast shutdown 会触发当前驱动的卸载崩溃。
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(exit_code)
        app.close()
    raise SystemExit(exit_code)
