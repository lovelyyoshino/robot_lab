# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.3.0
# @last_modified 2026-09-30
# @changelog
#   - v0.3.0 (2026-09-30): 注册前瞻抬轮课程，保持旧视觉任务可复现。
#   - v0.2.0 (2026-09-16): 注册独立深度时序增强student，保留原连续帧任务。
#   - v0.1.0 (2026-09-16): 注册视觉 height-scan teacher 与 depth student 任务。

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vision_preview_env_cfg:ParallelogramRoverVisionTeacherPreviewEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_cfg:ParallelogramRoverVisionTeacherPreviewPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vision_preview_env_cfg:ParallelogramRoverVisionStudentPreviewEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_vision_cfg:ParallelogramRoverVisionStudentPreviewDistillationRunnerCfg"
        ),
    },
)

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ParallelogramRoverFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverFlatPPORunnerCfg"),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vision_env_cfg:ParallelogramRoverVisionTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_vision_cfg:ParallelogramRoverVisionTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vision_env_cfg:ParallelogramRoverVisionStudentEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_vision_cfg:ParallelogramRoverVisionStudentDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ParallelogramRoverRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverRoughPPORunnerCfg"),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vision_env_cfg:ParallelogramRoverVisionStudentTemporalEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_vision_cfg:ParallelogramRoverVisionStudentTemporalDistillationRunnerCfg"
        ),
    },
)
