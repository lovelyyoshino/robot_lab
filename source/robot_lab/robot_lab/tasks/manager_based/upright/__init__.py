# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.3.0
# @last_modified 2026-09-16
# @changelog
#   - v0.3.0 (2026-09-16): 注册独立 44/49 维两轮运动及运动起立任务。
#   - v0.2.0 (2026-09-16): 注册独立的四轮到双轮起立第二阶段任务。
#   - v0.1.0 (2026-08-25): 注册直立任务。

"""平行四边形轮足车直立（双轮平衡）任务族。

**独立任务族**（PLAN §3.0-A）：不属于 locomotion/velocity 体系，
不继承其基类、不修改其任何文件。

42/47 维任务提供平衡与主动起立；44/49 维任务额外提供地面速度/yaw 指令。
当前运动训练为平地，窄缝/斜坡/台阶不能仅凭任务注册宣称已经支持。
"""

import gymnasium as gym

from .config import agents

gym.register(
    id="RobotLab-Isaac-Upright-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.upright_env_cfg:ParallelogramRoverUprightEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverUprightPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.getup_env_cfg:ParallelogramRoverUprightGetupEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverUprightGetupPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Upright-Motion-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.motion_env_cfg:ParallelogramRoverUprightMotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_motion_cfg:ParallelogramRoverUprightMotionPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Upright-Motion-Getup-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.motion_env_cfg:ParallelogramRoverUprightMotionGetupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_motion_cfg:ParallelogramRoverUprightMotionGetupPPORunnerCfg",
    },
)
