# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.16.0
# @last_modified 2026-10-28
# @changelog
#   - v0.16.0 (2026-10-28): --supermimic_terrain 自动识别方案 B 策略并追加车道偏移观测。
#   - v0.15.0 (2026-10-27): 新增 --supermimic_terrain 回放 supermimic 越障策略（与 --scan_student 同一路径）。
#   - v0.14.0 (2026-10-26): 新增 --scan_student 回放 supermimic 深度→扫描学生；主循环抽为 _play_loop 共用。
#   - (2026-10-12): 先冻结训练合同再调整回放地形，删除与场景配置重复的课程开关。
#   - (2026-10-12): Preview混合回放明确排布全宽、左右窄障碍、楼梯和方块；四车含楼梯。
#   - (2026-10-11): 只读输出首次真实离地/高度达标距离和承重，避免把瞬时几何当起抬时机。
#   - (2026-10-10): 显示真实前缘距离与10cm动作窗口，增加已有低楼梯/方块固定回放。
#   - (2026-10-10): Preview回放支持固定场景/难度与低频抬轮诊断，区分预瞄动作和窗口外跺脚。
#   - (2026-10-09): Preview回放保留可追溯地形列分类，关闭回放中的地形晋级。
#   - v0.12.0 (2026-09-30): 1–4 台视觉车回放自动显示角落深度，teacher 相机仅供显示。
#   - v0.11.0 (2026-09-30): Preview回放/导出校验相机、扫描与模型角色，拒绝同维度误加载。
#   - v0.10.0 (2026-09-30): 支持settle_feedback，统一校验profile数值与残差门控合同。
#   - v0.8.0 (2026-09-30): 回放支持FailureReplay-Residual并在创建环境前注入失败池。
#   - v0.9.0 (2026-09-30): 回放校验Residual先验NPZ哈希，阻止动作表静默漂移。
#   - v0.7.0 (2026-09-30): 支持 upright_release 先验释放消融并校验 profile。
#   - v0.6.0 (2026-09-30): 回放自动读取训练时 residual profile，阻止 wrapper 配置错配。
#   - v0.5.0 (2026-09-30): 回放/导出支持Side-Residual先验动作合成。
#   - v0.4.0 (2026-09-28): 恢复回放自动加载标定快照并拒绝动作/观测/物理合同错配。
#   - v0.3.0 (2026-09-16): 深度策略导出携带对应预处理/时间合同，避免4141维变体误用。
#   - v0.2.0 (2026-09-16): 增加有限步回放、视觉相机自动开启，并关闭正确的推扰事件。
#   - v0.1.0 (2026-09-16): 兼容无 locomotion 随机化/curriculum 的 recovery 与 upright 任务。

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from vision_depth_overlay import (  # isort: skip
    DEFAULT_PLAY_ENVS, configure_display_camera, create_depth_overlay, depth_overlay_requested,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=None, help="可选的回放步数，适合无界面导出检查。")
parser.add_argument(
    "--preview_scenario",
    choices=("mixed", "single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes"),
    default="mixed",
    help="Preview回放地形；可固定全宽/左右窄障碍、低楼梯或方块，mixed保留混合课程。",
)
parser.add_argument(
    "--preview_difficulty",
    type=float,
    default=0.0,
    help="固定Preview难度[0,1]；单障碍8–12cm，低楼梯/方块6–12cm。mixed忽略此参数。",
)
parser.add_argument(
    "--preview_debug",
    action="store_true",
    help="每5步打印最多4车的前缘距离、10cm动作窗口、抬升量和净空；只读诊断。",
)
parser.add_argument("--disable_depth_overlay", action="store_true", help="关闭 1–4 台视觉车的自动深度角落面板。")
parser.add_argument("--scan_student", action="store_true",
                    help="回放 scripts/supermimic/train_scan_student.py 的深度→扫描学生；--checkpoint 指向其 "
                         "exported/policy.pt，省略时取 logs/supermimic_scan 下最新 run。仅用于 Student-Temporal-Preview。")
parser.add_argument("--supermimic_terrain", action="store_true",
                    help="回放 scripts/supermimic/train_terrain.py 的越障策略；--checkpoint 指向其 exported/policy.pt，"
                         "省略时取 --load_run（或最新 run）。仅用于 Vision-Teacher-Preview。")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--recovery_motor_profile", type=str, default=None,
                    help="恢复电机实测配置；省略时自动读取checkpoint对应的标定快照。")
parser.add_argument("--recovery_prior_data", type=str, default=None,
                    help="Residual恢复回放使用的成功 program NPZ；须与训练时相同。")
parser.add_argument("--failure_states", type=Path, default=None,
                    help="FailureReplay-Residual 回放使用的失败状态池 NPZ；须与微调时相同。")
parser.add_argument("--recovery_prior_program_id", type=int, default=31,
                    help="Side-Residual 回放使用的固定动作程序编号，默认31。")
parser.add_argument("--recovery_residual_profile",
                    choices=("auto", "constant", "launch_protected", "upright_release", "settle_feedback"),
                    default="auto",
                    help="Side-Residual residual profile；auto从checkpoint的params/recovery_residual_runtime.json读取。")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# teacher 仅在小规模 GUI 回放补显示相机；student 无论是否显示都需要相机进行策略推理。
show_depth_overlay = depth_overlay_requested(args_cli)
if args_cli.video or (args_cli.task and "Vision-Student" in args_cli.task) or show_depth_overlay:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import json
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


_PREVIEW_TASK_SUFFIXES = (
    "Parallelogram-Rover-Vision-Teacher-Preview-v0",
    "Parallelogram-Rover-Vision-Student-Temporal-Preview-v0",
)
_PREVIEW_SCENARIOS = {"single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes"}


def _is_preview_task(task: str | None) -> bool:
    """识别前瞻任务，避免把回放专用地形参数误套到旧视觉任务。"""
    name = (task or "").split(":")[-1]
    return any(name.endswith(suffix) for suffix in _PREVIEW_TASK_SUFFIXES)


def _configure_preview_play_terrain(env_cfg, task: str | None, scenario: str, difficulty: float, num_envs: int, seed):
    """配置Preview回放的可复现场景；不参与checkpoint合同或训练配置。"""
    if not _is_preview_task(task):
        if scenario != "mixed" or difficulty != 0.0:
            raise ValueError("--preview_scenario/--preview_difficulty 只适用于 Vision-*Preview-v0 任务")
        return
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("--preview_difficulty 必须位于[0,1]")
    terrain = env_cfg.scene.terrain.terrain_generator
    if scenario == "mixed":
        # 4车时前4列覆盖全宽、左右窄障碍及楼梯；第5车显示方块。
        names = ("single_step", "single_step_left", "single_step_right", "pyramid_stairs_inv", "boxes")
        terrain.sub_terrains = {name: terrain.sub_terrains[name].copy() for name in names}
        for item in terrain.sub_terrains.values():
            item.proportion = 0.2
        terrain.num_rows = 3
        terrain.num_cols = 5
        terrain.difficulty_range = (0.25, 0.75)
        terrain.curriculum = True
        terrain.seed = seed
        env_cfg.scene.terrain.max_init_terrain_level = 1
        if env_cfg.curriculum is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
        print("[INFO] Preview play terrain: mixed=full/left/right/stairs/boxes, difficulty=0.25..0.75")
        return
    if scenario not in _PREVIEW_SCENARIOS or scenario not in terrain.sub_terrains:
        raise ValueError(f"Preview任务不包含可选地形：{scenario}")
    selected = terrain.sub_terrains[scenario].copy()
    selected.proportion = 1.0
    terrain.sub_terrains = {scenario: selected}
    terrain.num_rows = 1
    terrain.num_cols = max(int(num_envs), 1)
    terrain.difficulty_range = (difficulty, difficulty)
    terrain.curriculum = False
    terrain.seed = seed
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if env_cfg.curriculum is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    if scenario == "pyramid_stairs_inv":
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview import stairs_profile
        profile = stairs_profile(difficulty, selected.height_levels, selected.step_counts,
                                 selected.first_edge_m, selected.step_width)
        print(f"[INFO] Preview play terrain: stairs={profile['step_height_m']:.2f}m × "
              f"{profile['num_steps']}, target={profile['target_distance_m']:.2f}m, num_envs={num_envs}")
        return
    heights = selected.grid_height_range if scenario == "boxes" else selected.step_height_range
    print(
        f"[INFO] Preview play terrain: scenario={scenario}, difficulty={difficulty:.3f}, "
        f"height={heights[0] + difficulty * (heights[1] - heights[0]):.3f}m, "
        f"num_envs={num_envs}"
    )


def _emit_preview_debug(env, task: str | None, step: int):
    """读取当前步的几何预瞄状态；不调用奖励状态机，避免诊断改变训练语义。"""
    if not _is_preview_task(task):
        return
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview import (
        collect_preview,
    )

    sample = collect_preview(env)
    labels = {0: "single_step", 1: "single_step_left", 2: "single_step_right", 3: "stairs", -1: "unknown"}
    kinds = sample["terrain_kind"].detach().cpu().tolist()
    preview = sample["preview"].detach().cpu().tolist()
    eligible = sample["eligible"].detach().cpu().tolist()
    advance = sample["unbounded_advance_m"].detach().cpu().tolist()
    lift = sample["articulation_lift_m"].detach().cpu().tolist()
    clearance = sample["clearance_m"].detach().cpu().tolist()
    prelift = sample["prelift"].detach().cpu().tolist()
    edge = sample["edge_distance_m"].detach().cpu().tolist()
    edge_valid = sample["edge_distance_valid"].detach().cpu().tolist()
    force = sample["wheel_support_force_n"].detach().cpu().tolist()
    state = env.reward_manager.get_term_cfg("anticipatory_wheel_lift").func.state
    takeoff = state.takeoff_lead.detach().cpu().tolist()
    height = state.height_lead.detach().cpu().tolist()
    cfg = env.cfg.vision_preview
    print(f"[PreviewDebug] step={step} (fl,fr; target=0.10m, action_window=0.08–0.14m; scan_advance仅为栅格估计)")
    for env_id in range(min(env.num_envs, 4)):
        phases = []
        for wheel in (0, 2):
            distance = edge[env_id][wheel]
            if not edge_valid[env_id][wheel]:
                phases.append("unknown")
            elif distance > cfg["action_max_advance_m"]:
                phases.append("too_early")
            elif distance >= cfg["min_advance_m"]:
                phases.append("lift_window")
            else:
                phases.append("hold_or_pass")
        print(
            f"  env_{env_id}: terrain={labels.get(int(kinds[env_id]), 'unknown')} "
            f"eligible={bool(eligible[env_id])} "
            f"edge_lead_m=({edge[env_id][0]:.3f},{edge[env_id][2]:.3f}) phase={tuple(phases)} "
            f"scan_advance_m=({advance[env_id][0]:.3f},{advance[env_id][2]:.3f}) "
            f"preview=({bool(preview[env_id][0])},{bool(preview[env_id][2])}) "
            f"prelift=({bool(prelift[env_id][0])},{bool(prelift[env_id][2])}) "
            f"lift_m=({lift[env_id][0]:.3f},{lift[env_id][2]:.3f}) "
            f"clearance_m=({clearance[env_id][0]:.3f},{clearance[env_id][2]:.3f}) "
            f"support_N=({force[env_id][0]:.1f},{force[env_id][2]:.1f}) "
            f"takeoff_lead_m=({takeoff[env_id][0]:.3f},{takeoff[env_id][2]:.3f}) "
            f"height_lead_m=({height[env_id][0]:.3f},{height[env_id][2]:.3f})"
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else DEFAULT_PLAY_ENVS

    # handle deprecated configurations
    agent_cfg = cli_args.handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    from vision_contract import contract_for_cfg, save_contract, validate_checkpoint

    # 训练布局属于模型来源；须先冻结合同，回放的行列数/难度与课程开关不能污染校验。
    preview_contract = contract_for_cfg(args_cli.task, env_cfg, agent_cfg)

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # standalone recovery/upright 不声明 locomotion 专用随机化和 curriculum。
    if env_cfg.events is not None:
        for event_name in ("randomize_apply_external_force_torque", "randomize_push_robot", "push_robot"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
    if env_cfg.curriculum is not None:
        for curriculum_name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            if hasattr(env_cfg.curriculum, curriculum_name):
                setattr(env_cfg.curriculum, curriculum_name, None)

    if preview_contract is not None:
        # 混合/固定场景统一负责地形排列和关闭晋级，不重复改课程开关。
        _configure_preview_play_terrain(
            env_cfg,
            args_cli.task,
            args_cli.preview_scenario,
            args_cli.preview_difficulty,
            env_cfg.scene.num_envs,
            args_cli.seed if args_cli.seed is not None else env_cfg.seed,
        )
    elif args_cli.preview_scenario != "mixed" or args_cli.preview_difficulty != 0.0:
        raise ValueError("--preview_scenario/--preview_difficulty 只适用于 Vision-*Preview-v0 任务")

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.scan_student or args_cli.supermimic_terrain:
        resume_path = None  # 下方在 logs/supermimic_scan / logs/supermimic_terrain 里找导出
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.scan_student:
        # 扫描学生是 TorchScript，没有 rsl_rl 的 params/；改核对它训练时写下的深度合同。
        from vision_contract import validate_scan_student

        if not args_cli.task.endswith("Vision-Student-Temporal-Preview-v0"):
            raise ValueError("--scan_student 只适用于 Vision-Student-Temporal-Preview-v0")
        if not args_cli.checkpoint:
            runs = sorted(p for p in Path("logs/supermimic_scan").glob("*/exported/policy.pt"))
            if not runs:
                raise FileNotFoundError("logs/supermimic_scan 下没有已导出的扫描学生")
            resume_path = str(max(runs, key=lambda p: p.stat().st_mtime).resolve())
        from robot_lab.supermimic.scan_student import configure_saved_student

        validate_scan_student(resume_path, configure_saved_student(env_cfg, resume_path))
        log_dir = os.path.dirname(os.path.dirname(resume_path))
    elif args_cli.supermimic_terrain:
        # 越障策略是 TorchScript（232→12），观测/动作合同与热启动 Teacher 相同，按 run 目录的 vision_contract.json 校验。
        from vision_contract import validate_supermimic_terrain

        if args_cli.scan_student:
            raise ValueError("--supermimic_terrain 与 --scan_student 不能同时使用")
        if not args_cli.checkpoint:
            root = Path("logs/supermimic_terrain")
            runs = [root / agent_cfg.load_run / "exported/policy.pt"] if agent_cfg.load_run not in (None, ".*") \
                else list(root.glob("*/exported/policy.pt"))
            runs = [p for p in runs if p.is_file()]
            if not runs:
                raise FileNotFoundError(f"{root} 下没有已导出的越障策略（--load_run={agent_cfg.load_run}）")
            resume_path = str(max(runs, key=lambda p: p.stat().st_mtime).resolve())
        from vision_contract import restore_height_scan_period

        source_path = Path(resume_path).parent.parent / "params/vision_contract.json"
        source_contract = json.loads(source_path.read_text())
        restore_height_scan_period(env_cfg, source_contract)
        preview_contract["shared"]["height_scanner"]["update_period"] = env_cfg.scene.height_scanner.update_period
        validate_supermimic_terrain(resume_path, preview_contract)
        from robot_lab.supermimic.lane_offset import add_lane_offset_term, policy_uses_lane_offset

        if policy_uses_lane_offset(resume_path):
            add_lane_offset_term(env_cfg)  # 方案 B 策略：追加第 233 维车道偏移
        log_dir = os.path.dirname(os.path.dirname(resume_path))
    else:
        log_dir = os.path.dirname(resume_path)
        validate_checkpoint(resume_path, preview_contract)

    from recovery_contract import configure_failure_replay_env, is_residual_recovery_task

    residual_task = is_residual_recovery_task(args_cli.task)
    configure_failure_replay_env(env_cfg, args_cli.task, args_cli.failure_states)
    residual_profile = args_cli.recovery_residual_profile
    if residual_task:
        if args_cli.recovery_prior_data is None:
            raise ValueError("Residual恢复回放必须提供 --recovery_prior_data")
        from recovery_prior import resolve_checkpoint_prior

        residual_profile = resolve_checkpoint_prior(
            resume_path, args_cli.recovery_prior_data,
            program_id=args_cli.recovery_prior_program_id, task=args_cli.task,
            residual_profile=residual_profile,
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    from recovery_contract import configure_physics, control_contract, is_rover_recovery, validate_checkpoint_contract

    configure_physics(env_cfg, args_cli.task, args_cli.recovery_motor_profile, resume_path)
    recovery_control = control_contract(env_cfg.to_dict(), agent_cfg.to_dict()) if is_rover_recovery(args_cli.task) else None
    if recovery_control is not None:
        validate_checkpoint_contract(resume_path, recovery_control)

    if show_depth_overlay:
        configure_display_camera(env_cfg, args_cli.task)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if residual_task:
        from recovery_prior import attach_recovery_prior

        env = attach_recovery_prior(
            env, args_cli.recovery_prior_data, program_id=args_cli.recovery_prior_program_id,
            residual_profile=residual_profile,
        )
    elif args_cli.recovery_prior_data is not None:
        raise ValueError("--recovery_prior_data 只允许用于Residual恢复任务")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if args_cli.scan_student or args_cli.supermimic_terrain:
        # 两者都是无状态 TorchScript，直接吃 policy 观测，不走 rsl_rl runner
        _play_loop(env, _scan_student_policy(resume_path, env.unwrapped.device), lambda dones: None)
        return
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    save_contract(export_model_dir, preview_contract)
    if recovery_control is not None:
        os.makedirs(export_model_dir, exist_ok=True)
        with open(os.path.join(export_model_dir, "recovery_control_contract.json"), "w", encoding="utf-8") as stream:
            json.dump({"contract": recovery_control, "deployment_verified": False,
                       "warning": "新动作尺度不可配旧rl_sar YAML；需独立runtime和sim2sim验收。"},
                      stream, ensure_ascii=False, indent=2, allow_nan=False)

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    if args_cli.task and "Parallelogram-Rover-Vision-Student" in args_cli.task:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import vision_depth_contract

        with open(os.path.join(export_model_dir, "depth_contract.json"), "w", encoding="utf-8") as stream:
            json.dump(vision_depth_contract(env_cfg), stream, ensure_ascii=False, indent=2, allow_nan=False)

    reset = policy.reset if version.parse(installed_version) >= version.parse("4.0.0") else policy_nn.reset
    _play_loop(env, policy, reset)


def _scan_student_policy(path, device):
    """无状态 TorchScript：扫描学生 4141→12（GRU 只在 8 帧窗口内展开）或越障策略 232→12，都不需要 reset。"""
    scripted = torch.jit.load(path, map_location=device).eval()

    def policy(obs):
        return scripted(obs["policy"])

    return policy


def _play_loop(env, policy, reset):
    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    depth_overlay = None
    try:
        depth_overlay = create_depth_overlay(env.unwrapped, args_cli.task, enabled=show_depth_overlay)
        while simulation_app.is_running():
            start_time = time.time()
            # 在自动reset前采样，避免已终止环境丢失碰撞前的时序状态。
            if args_cli.preview_debug and (timestep + 1) % 5 == 0:
                _emit_preview_debug(env.unwrapped, args_cli.task, timestep + 1)
            # run everything in inference mode
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if depth_overlay is not None:
                    depth_overlay.update()
                reset(dones)
            timestep += 1
            if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                break
            if args_cli.video and timestep == args_cli.video_length:
                break
            if args_cli.keyboard:
                camera_follow(env)
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if depth_overlay is not None:
            depth_overlay.close()
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
