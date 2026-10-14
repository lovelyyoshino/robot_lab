# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.24.0
# @last_modified 2026-10-09
# @changelog
#   - v0.24.0 (2026-10-09): 识别DaggerDistillation为蒸馏算法，续训时对齐其衰减进度。
#   - v0.23.0 (2026-09-30): Preview视觉任务在创建环境前验证teacher/student语义并保存合同。
#   - v0.22.0 (2026-09-30): 续训校验先验语义；显式权重迁移支持settle_feedback新实验。
#   - v0.20.0 (2026-09-30): 允许FailureReplay-Residual使用同一先验和60/72维Actor微调。
#   - v0.21.0 (2026-09-30): Residual先验谱系记录NPZ哈希，阻止回放时静默替换动作表。
#   - v0.19.0 (2026-09-30): 增加 upright_release 先验释放消融，constant 仍为默认基线。
#   - v0.18.0 (2026-09-30): Side-Residual增加分阶段 residual profile，默认保护前1秒弹射。
#   - v0.17.0 (2026-09-30): 新增Side-Residual动作先验、零均值Actor初始化及显式先验合同入口。
#   - v0.16.0 (2026-09-29): 默认保护切换锚点改为20倍/2步，降低PPO破坏脚本时序的风险。
#   - v0.15.0 (2026-09-29): 阶段恢复增加critic预热、示范锚点、物理噪声和可续训快照。
#   - v0.14.0 (2026-09-28): 恢复独立电机标定/控制合同快照，创建仿真前拒绝错误尺度续训。
#   - v0.13.0 (2026-09-17): teacher迁移后保存独立零更新快照，避免将PPO退化误判为扩维错误。
#   - v0.12.0 (2026-09-17): 显式 blind45→teacher232 warm start，严格合同及谱系记录，不继承优化器或迭代。
#   - v0.11.0 (2026-09-17): 恢复 checkpoint 的有效 PPO 学习率，并支持显式可追溯覆盖。
#   - v0.10.0 (2026-09-17): 支持可追溯的 resume 探索噪声缩放，避免稳定策略微调被旧噪声破坏。
#   - v0.9.0 (2026-09-17): 为全部 resume/蒸馏 run 保存源 checkpoint 哈希与迭代谱系。
#   - v0.8.0 (2026-09-17): motion 从真实 episode 起点计时；提供固定阶段入口，避免 Hydra 的 None→int 类型拒绝。
#   - v0.7.0 (2026-09-17): 恢复训练继承全部历史失败池快照，缺旧来源谱系时明确标记未知。
#   - v0.6.0 (2026-09-17): 暴露 FailureReplay 失败池参数，兼容 Hydra 并拒绝两种入口指向不同数据。
#   - v0.5.0 (2026-09-17): 将真实失败池字节/哈希随实验保存，防止训练来源仅有可变路径。
#   - v0.4.0 (2026-09-16): 修复旧 YAML 标签读取并严格校验观测预处理、动作及控制周期。
#   - v0.3.0 (2026-09-16): 增加显式42/47到44/49 warm start，重建optimizer并校验源输入前缀。
#   - v0.3.0 (2026-09-16): student自动启动相机，并从最终有效配置保存深度时间合同。
#   - v0.2.0 (2026-09-16): 恢复/主动起立从真实 episode 起点采样，避免随机时钟跳过动作阶段。
#   - v0.1.0 (2026-09-16): 支持从绝对 checkpoint 路径跨实验继续训练。

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--upright_warm_start", type=str, default=None, help="以旧42/47维upright权重初始化44/49维motion；不是resume。")
parser.add_argument("--vision_teacher_warm_start", type=Path, default=None,
                    help="以兼容 blind45 权重初始化 teacher232；新增 height187 列置零，不是 resume 或 student 蒸馏。")
parser.add_argument("--motion_fixed_stage", type=int, choices=(0, 1, 2), default=None,
                    help="固定 motion 速度课程阶段；省略时按成功门晋级。")
parser.add_argument("--resume_noise_scale", type=float, default=None,
                    help="续训时缩放 checkpoint 探索标准差并清零其优化器动量；范围 (0,1]。")
parser.add_argument("--resume_learning_rate", type=float, default=None,
                    help="显式覆盖 PPO 续训学习率；省略时继承 checkpoint optimizer 的有效值。")
parser.add_argument("--recovery_protected", action="store_true",
                    help="Side-Phase专用：BC后先critic预热，再以脚本动作锚点保护Actor。")
parser.add_argument("--recovery_demo_data", type=Path, default=None,
                    help="受保护恢复训练使用的原58维成功示范NPZ；运行时显式转换为60维。")
parser.add_argument("--recovery_warmup_iterations", type=int, default=15,
                    help="受保护恢复训练冻结Actor/std的PPO更新次数。")
parser.add_argument("--recovery_anchor_coef", type=float, default=0.2,
                    help="示范锚点Adam学习率与PPO学习率之比；不是PPO目标中的loss系数。")
parser.add_argument("--recovery_anchor_batch_size", type=int, default=2048,
                    help="每次锚点更新抽取的成功示范帧数。")
parser.add_argument("--recovery_anchor_transition_weight", type=float, default=20.,
                    help="锚点损失在启动/换段附近的权重。")
parser.add_argument("--recovery_anchor_transition_window", type=int, default=2,
                    help="锚点换段窗口，控制步数。")
parser.add_argument("--recovery_motor_profile", type=Path,
                    help="恢复专用实测电机JSON；省略保留仿真假设，resume自动读取源run快照。")
parser.add_argument("--recovery_prior_data", type=Path,
                    help="Residual恢复任务使用的成功 program NPZ；必须包含 action_program.targets。")
parser.add_argument("--recovery_prior_program_id", type=int, default=31,
                    help="Side-Residual 使用的固定动作程序编号，默认31。")
parser.add_argument("--recovery_residual_std", type=float, default=0.15,
                    help="Side-Residual 新策略的初始归一化探索std；只作用于新建run。")
parser.add_argument("--recovery_residual_profile",
                    choices=("auto", "constant", "launch_protected", "upright_release", "settle_feedback"),
                    default="auto",
                    help="新训练auto=constant；续训auto恢复源合同；settle_feedback允许3秒后小幅纠偏。")
parser.add_argument("--recovery_residual_warm_start", type=Path,
                    help="显式迁移Residual权重到新profile/课程，重建optimizer和迭代；不能与resume混用。")
parser.add_argument(
    "--failure_states",
    type=Path,
    default=None,
    help="FailureReplay 使用的已采集失败状态 NPZ；只接受显式路径，不覆盖原文件。",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video or (args_cli.task and "Vision-Student" in args_cli.task):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import json
import hashlib
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip
from experiment_lineage import (
    checkpoint_source_metadata,
    restore_resumed_learning_rate,
    scale_policy_exploration_noise,
    write_checkpoint_source_metadata,
)

# import logger
logger = logging.getLogger(__name__)

# DaggerDistillation 只改变采样驱动者，仍需加载teacher并走蒸馏谱系/合同校验。
DISTILLATION_ALGORITHMS = {"Distillation", "DaggerDistillation"}

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = cli_args.handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    if args_cli.recovery_protected:
        from recovery_phase_training import PHASE_TASK

        if (args_cli.task != PHASE_TASK or not agent_cfg.resume or args_cli.recovery_demo_data is None
                or agent_cfg.algorithm.class_name != "PPO" or agent_cfg.class_name != "OnPolicyRunner"
                or args_cli.distributed or args_cli.upright_warm_start or args_cli.vision_teacher_warm_start):
            raise ValueError("recovery_protected只支持Side-Phase单设备PPO resume，须提供训练示范")
        if args_cli.resume_noise_scale is not None:
            raise ValueError("recovery_protected按物理单位固定探索std，不能叠加resume_noise_scale")
    elif args_cli.recovery_demo_data is not None:
        raise ValueError("recovery_demo_data必须与recovery_protected一起使用")

    # 任务分类集中在 recovery_contract，训练/评估/回放不能各自维护字符串白名单。
    from recovery_contract import (
        configure_failure_replay_env, is_residual_recovery_task,
    )

    residual_task = is_residual_recovery_task(args_cli.task)
    if residual_task and args_cli.recovery_prior_data is None:
        raise ValueError("Residual恢复训练必须提供 --recovery_prior_data")
    if not residual_task and args_cli.recovery_prior_data is not None:
        raise ValueError("--recovery_prior_data 只允许用于Residual恢复任务")
    if args_cli.recovery_residual_warm_start is not None:
        if (not residual_task or agent_cfg.resume or agent_cfg.algorithm.class_name != "PPO"
                or args_cli.distributed or args_cli.recovery_protected
                or args_cli.upright_warm_start or args_cli.vision_teacher_warm_start):
            raise ValueError("recovery_residual_warm_start只允许单设备新Residual PPO，不能混用其他初始化/resume")

    vision_warm_start_enabled = False
    if args_cli.vision_teacher_warm_start is not None:
        from vision_warm_start import validate_vision_warm_start_request

        # 在创建环境前拒绝 CLI/Hydra 加载冲突，未启用时完全保留既有训练路径。
        vision_warm_start_enabled = validate_vision_warm_start_request(args_cli, agent_cfg.to_dict())

    if args_cli.motion_fixed_stage is not None:
        motion_command = getattr(getattr(env_cfg, "commands", None), "motion_command", None)
        if motion_command is None:
            raise ValueError("--motion_fixed_stage 仅适用于 motion / motion-getup 任务")
        motion_command.fixed_stage = args_cli.motion_fixed_stage

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    configure_failure_replay_env(env_cfg, args_cli.task, args_cli.failure_states)

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name in DISTILLATION_ALGORITHMS:
        requested_checkpoint = os.path.expanduser(str(agent_cfg.load_checkpoint))
        if os.path.isfile(requested_checkpoint):
            # 起立第二阶段与视觉蒸馏需要跨 experiment 加载教师/前一阶段权重。
            resume_path = retrieve_file_path(requested_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    from recovery_contract import (
        configure_physics, control_contract, is_rover_recovery, save_control_contract,
        save_physics_evidence, validate_checkpoint_contract,
    )

    recovery_source = resume_path if agent_cfg.resume else args_cli.recovery_residual_warm_start
    if residual_task:
        from recovery_prior import resolve_checkpoint_prior

        if recovery_source is not None:
            args_cli.recovery_residual_profile = resolve_checkpoint_prior(
                recovery_source, args_cli.recovery_prior_data,
                program_id=args_cli.recovery_prior_program_id, task=args_cli.task,
                residual_profile=args_cli.recovery_residual_profile,
                allow_task_transfer=True,
                allow_profile_transfer=args_cli.recovery_residual_warm_start is not None,
            )
        elif args_cli.recovery_residual_profile == "auto":
            args_cli.recovery_residual_profile = "constant"

    recovery_profile = configure_physics(
        env_cfg, args_cli.task, args_cli.recovery_motor_profile,
        recovery_source,
    )
    recovery_control = control_contract(env_cfg.to_dict(), agent_cfg.to_dict()) if is_rover_recovery(args_cli.task) else None
    if recovery_control is not None and recovery_source is not None:
        validate_checkpoint_contract(recovery_source, recovery_control)

    from vision_contract import contract_for_cfg, save_contract, validate_checkpoint

    preview_contract = contract_for_cfg(args_cli.task, env_cfg, agent_cfg)
    if preview_contract is not None and (agent_cfg.resume or agent_cfg.algorithm.class_name in DISTILLATION_ALGORITHMS):
        validate_checkpoint(resume_path, preview_contract, teacher_source=not agent_cfg.resume)

    # 在加载/尺度/物理合同确认后才创建环境；与旧行驶/视觉任务完全隔离。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if residual_task:
        from recovery_prior import attach_recovery_prior

        env = attach_recovery_prior(
            env, args_cli.recovery_prior_data, program_id=args_cli.recovery_prior_program_id,
            residual_profile=args_cli.recovery_residual_profile,
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    # Runner 会原位解析 policy 配置，保存显式归一化与路由合同用于迁移校验。
    vision_target_agent_cfg = agent_cfg.to_dict() if vision_warm_start_enabled else None
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    warm_start_metadata = None
    recovery_residual_metadata = None
    if residual_task and not agent_cfg.resume:
        from recovery_prior import initialize_residual_policy, warm_start_residual_policy

        if args_cli.recovery_residual_warm_start is not None:
            recovery_residual_metadata = warm_start_residual_policy(
                runner.alg.policy, args_cli.recovery_residual_warm_start, std=args_cli.recovery_residual_std,
            )
        else:
            recovery_residual_metadata = initialize_residual_policy(
                runner.alg.policy, std=args_cli.recovery_residual_std,
            )
    if args_cli.upright_warm_start is not None:
        from upright_warm_start import load_training_config, validate_upright_motion_contract, warm_start_upright_motion

        if agent_cfg.resume or agent_cfg.algorithm.class_name != "PPO" or "Upright-Motion" not in args_cli.task:
            raise ValueError("upright_warm_start仅用于新motion PPO，不能与resume或蒸馏混用")
        source_path = Path(args_cli.upright_warm_start).expanduser().resolve()
        source_params = source_path.parent / "params"
        # 标签白名单避免执行任意 YAML 对象；比较完整预处理而非仅看字段维数/函数简称。
        source_agent = load_training_config(source_params / "agent.yaml")
        source_env = load_training_config(source_params / "env.yaml")
        contract_metadata = validate_upright_motion_contract(
            source_env, env_cfg.to_dict(), source_agent, agent_cfg.to_dict(),
            env.unwrapped.observation_manager.active_terms,
        )
        source_policy = source_agent["policy"]
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        warm_start_metadata = warm_start_upright_motion(
            checkpoint["model_state_dict"], runner.alg.policy,
            source_actor_obs_normalization=source_policy["actor_obs_normalization"],
            source_critic_obs_normalization=source_policy["critic_obs_normalization"],
            source_activation=source_policy["activation"],
        )
        warm_start_metadata.update(contract_metadata)
        warm_start_metadata["source_checkpoint"] = str(source_path)
        with source_path.open("rb") as stream:
            warm_start_metadata["source_checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if vision_warm_start_enabled:
        from vision_warm_start import initialize_teacher_from_blind

        if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
            raise ValueError("视觉 warm start 必须作用于迭代为零且 optimizer 尚未更新的新 runner")
        manager = env.unwrapped.observation_manager
        warm_start_metadata = initialize_teacher_from_blind(
            args_cli.vision_teacher_warm_start, runner.alg.policy,
            target_env=env_cfg.to_dict(), target_agent=vision_target_agent_cfg,
            active_terms=manager.active_terms, term_dims=manager.group_obs_term_dim,
            task=args_cli.task, seed=agent_cfg.seed,
        )
        print("[INFO]: Initialized teacher actor 45→232 and unchanged critic235; optimizer/iteration remain new.")
    resume_lineage = None
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name in DISTILLATION_ALGORITHMS:
        resume_lineage = checkpoint_source_metadata(
            resume_path,
            load_mode="resume" if agent_cfg.resume else "distillation_teacher",
            task=args_cli.task,
            seed=agent_cfg.seed,
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        if agent_cfg.resume and agent_cfg.algorithm.class_name == "DaggerDistillation":
            # checkpoint保存的是从0计数的已完成迭代；从下一轮继续，驱动比例和学习率不重启。
            runner.current_learning_iteration += 1
            runner.alg.num_updates = runner.current_learning_iteration
    if args_cli.resume_learning_rate is not None and (
        not agent_cfg.resume or agent_cfg.algorithm.class_name != "PPO"
    ):
        raise ValueError("--resume_learning_rate 只允许用于 PPO resume")
    if agent_cfg.resume and agent_cfg.algorithm.class_name == "PPO":
        # RSL-RL 只加载 optimizer param group，不恢复 adaptive schedule 使用的内部标量。
        resume_lineage["learning_rate_restore"] = restore_resumed_learning_rate(
            runner.alg, args_cli.resume_learning_rate,
        )
    if args_cli.resume_noise_scale is not None:
        if not agent_cfg.resume or agent_cfg.algorithm.class_name != "PPO":
            raise ValueError("--resume_noise_scale 只允许用于 PPO resume")
        resume_lineage["exploration_noise_reset"] = scale_policy_exploration_noise(
            runner.alg.policy, runner.alg.optimizer, args_cli.resume_noise_scale,
        )

    recovery_protection = None
    if args_cli.recovery_protected:
        if not args_cli.resume or args_cli.recovery_demo_data is None:
            raise ValueError("--recovery_protected 必须与 --resume 和 --recovery_demo_data 一起使用")
        from recovery_protected import attach_recovery_protection

        recovery_protection = attach_recovery_protection(
            runner, task=args_cli.task, demo_path=args_cli.recovery_demo_data,
            env=env_cfg.to_dict(), agent=agent_cfg.to_dict(),
            source_checkpoint=resume_path,
            warmup_iterations=args_cli.recovery_warmup_iterations,
            anchor_coef=args_cli.recovery_anchor_coef,
            anchor_batch_size=args_cli.recovery_anchor_batch_size,
            anchor_transition_weight=args_cli.recovery_anchor_transition_weight,
            anchor_transition_window=args_cli.recovery_anchor_transition_window,
            seed=agent_cfg.seed,
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    save_contract(Path(log_dir) / "params", preview_contract)
    if recovery_control is not None:
        save_control_contract(Path(log_dir) / "params", recovery_control)
        save_physics_evidence(Path(log_dir) / "params", env_cfg, recovery_profile)
    if resume_lineage is not None:
        write_checkpoint_source_metadata(Path(log_dir) / "params", resume_lineage)
    if recovery_protection is not None:
        with open(os.path.join(log_dir, "params", "recovery_protection.json"), "w", encoding="utf-8") as stream:
            json.dump(recovery_protection, stream, ensure_ascii=False, indent=2, allow_nan=False)
        # 此快照确实在learn之前；model_0是首轮更新后，不可当成BC初始化。
        runner.logger_type = agent_cfg.logger
        runner.save(os.path.join(log_dir, "recovery_pre_update.pt"), infos={"snapshot": "before_learn"})
    failure_pool = getattr(env.unwrapped, "_failure_state_pool", None)
    if args_cli.task and "Recovery" in args_cli.task and "Parallelogram-Rover" in args_cli.task:
        from robot_lab.tasks.manager_based.recovery.mdp.failure_states import snapshot_training_pools

        # 普通恢复阶段也继承来源；FailureReplay→Grounded→FailureReplay 不得遗忘旧池。
        source_checkpoint = (resume_path if agent_cfg.resume or agent_cfg.algorithm.class_name in DISTILLATION_ALGORITHMS
                             else args_cli.recovery_residual_warm_start)
        snapshot_training_pools(Path(log_dir) / "params", failure_pool, source_checkpoint)
    if warm_start_metadata is not None:
        if vision_warm_start_enabled:
            from vision_warm_start import save_teacher_initialization

            warm_start_metadata["initialization_snapshot"] = save_teacher_initialization(runner, log_dir)
        with open(os.path.join(log_dir, "params", "warm_start.json"), "w", encoding="utf-8") as stream:
            json.dump(warm_start_metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if recovery_residual_metadata is not None:
        recovery_residual_metadata.update({
            "prior_data": str(args_cli.recovery_prior_data.expanduser().resolve()),
            "prior_program_id": args_cli.recovery_prior_program_id,
            "residual_profile": args_cli.recovery_residual_profile,
            "task": args_cli.task,
        })
        with open(os.path.join(log_dir, "params", "recovery_residual_initialization.json"), "w", encoding="utf-8") as stream:
            json.dump(recovery_residual_metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if residual_task:
        # profile 是动作合同的一部分；play/evaluate 在缺省时从此文件自动恢复，
        # 避免同一 checkpoint 被错误地用两种 wrapper 回放。
        from recovery_prior import prior_runtime_contract

        runtime_contract = prior_runtime_contract(
            args_cli.recovery_prior_data, program_id=args_cli.recovery_prior_program_id,
            residual_profile=args_cli.recovery_residual_profile, task=args_cli.task,
        )
        with open(os.path.join(log_dir, "params", "recovery_residual_runtime.json"), "w", encoding="utf-8") as stream:
            json.dump(runtime_contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
        if recovery_residual_metadata is not None:
            # model_0是第一次PPO更新之后；单独保留真初始化供迁移前后闭环对照。
            runner.logger_type = agent_cfg.logger
            runner.save(os.path.join(log_dir, "recovery_pre_update.pt"), infos={
                "snapshot": "before_learn", "recovery_residual_initialization": recovery_residual_metadata,
                "recovery_residual_runtime": runtime_contract,
            })
    if args_cli.task and "Parallelogram-Rover-Vision-Student" in args_cli.task:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import vision_depth_contract

        with open(os.path.join(log_dir, "params", "depth_contract.json"), "w", encoding="utf-8") as stream:
            json.dump(vision_depth_contract(env_cfg), stream, ensure_ascii=False, indent=2, allow_nan=False)

    # run training
    # recovery 有自由落体时钟，getup 有起立宽限期；只随机 episode_length_buf
    # 会让两个时间基准分离，甚至让四轮起点在首步就终止。
    events = env_cfg.events
    has_freefall = getattr(events, "freefall_init", None) is not None
    reset_pose = getattr(events, "reset_pose", None)
    has_getup = reset_pose is not None and reset_pose.params.get("quad_prob", 0.0) > 0.0
    has_motion = getattr(getattr(env_cfg, "commands", None), "motion_command", None) is not None
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=not (has_freefall or has_getup or has_motion),
    )

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
