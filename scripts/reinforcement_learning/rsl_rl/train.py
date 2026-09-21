# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.13.0
# @last_modified 2026-09-17
# @changelog
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

    failure_replay_task = bool(args_cli.task and "Recovery-FailureReplay" in args_cli.task)
    if failure_replay_task:
        reset_fall_pose = getattr(getattr(env_cfg, "events", None), "reset_fall_pose", None)
        if reset_fall_pose is None or not hasattr(reset_fall_pose, "params"):
            raise ValueError("FailureReplay 环境缺少 reset_fall_pose 参数入口")
        configured_path = reset_fall_pose.params.get("failure_state_path", "")
        if args_cli.failure_states is None and not configured_path:
            raise ValueError("FailureReplay 训练必须通过 --failure_states 或 Hydra 指定失败状态池")
        failure_states_path = Path(args_cli.failure_states or configured_path).expanduser().resolve()
        if configured_path and Path(configured_path).expanduser().resolve() != failure_states_path:
            raise ValueError("--failure_states 与 Hydra 失败池路径冲突，拒绝静默覆盖训练来源")
        if not failure_states_path.is_file():
            raise FileNotFoundError(f"FailureReplay 失败状态池不存在: {failure_states_path}")
        reset_fall_pose.params["failure_state_path"] = str(failure_states_path)
    elif args_cli.failure_states is not None:
        raise ValueError("--failure_states 只允许用于 Recovery-FailureReplay 任务")

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        requested_checkpoint = os.path.expanduser(str(agent_cfg.load_checkpoint))
        if os.path.isfile(requested_checkpoint):
            # 起立第二阶段与视觉蒸馏需要跨 experiment 加载教师/前一阶段权重。
            resume_path = retrieve_file_path(requested_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

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
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_lineage = checkpoint_source_metadata(
            resume_path,
            load_mode="resume" if agent_cfg.resume else "distillation_teacher",
            task=args_cli.task,
            seed=agent_cfg.seed,
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
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

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if resume_lineage is not None:
        write_checkpoint_source_metadata(Path(log_dir) / "params", resume_lineage)
    failure_pool = getattr(env.unwrapped, "_failure_state_pool", None)
    if args_cli.task and "Recovery" in args_cli.task and "Parallelogram-Rover" in args_cli.task:
        from robot_lab.tasks.manager_based.recovery.mdp.failure_states import snapshot_training_pools

        # 普通恢复阶段也继承来源；FailureReplay→Grounded→FailureReplay 不得遗忘旧池。
        source_checkpoint = resume_path if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation" else None
        snapshot_training_pools(Path(log_dir) / "params", failure_pool, source_checkpoint)
    if warm_start_metadata is not None:
        if vision_warm_start_enabled:
            from vision_warm_start import save_teacher_initialization

            warm_start_metadata["initialization_snapshot"] = save_teacher_initialization(runner, log_dir)
        with open(os.path.join(log_dir, "params", "warm_start.json"), "w", encoding="utf-8") as stream:
            json.dump(warm_start_metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
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
