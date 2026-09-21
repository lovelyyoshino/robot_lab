# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.6.0
# @last_modified 2026-09-17
# @changelog
#   - v0.6.0 (2026-09-17): 独立注册人工连续横滚即时控制预训练，不替代正式倒姿验收。
#   - v0.5.0 (2026-09-17): 注册独立姿态/高度进展 shaping 恢复任务。
#   - v0.4.0 (2026-09-17): 注册独立失败状态池即时接管任务，必须显式提供采集数据。
#   - v0.3.0 (2026-09-17): 注册显式准备制动与真实四轮奖励的 grounded 恢复实验。
#   - v0.2.0 (2026-09-16): 注册独立的真实沉降后立即接管任务，保留默认下落协议。
#   - v0.1.0 (2026-08-25): 注册恢复任务（method1 主方案）。

"""平行四边形轮足车倒地恢复任务族。

**独立任务族**（PLAN §3.0-A）：不属于 locomotion/velocity 体系，
不继承其基类、不修改其任何文件。

注册的任务：
  - ``RobotLab-Isaac-Recovery-Parallelogram-Rover-v0``  method1（EDRS，主方案）

  - ``RobotLab-Isaac-Recovery-Getup-Parallelogram-Rover-v0``  method2（对照基线，无 ED/CW）
  - ``RobotLab-Isaac-Recovery-Settled-Parallelogram-Rover-v0`` 真实地面沉降后立即接管
"""

import gymnasium as gym

from .config import agents

gym.register(
    id="RobotLab-Isaac-Recovery-ControlledTilt-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.settled_ground.tilt_env_cfg:ParallelogramRoverControlledTiltRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverControlledTiltRecoveryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Recovery-FailureReplay-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.settled_ground.replay_env_cfg:ParallelogramRoverFailureReplayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverFailureReplayPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.settled_ground.grounded_env_cfg:ParallelogramRoverGroundedRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverGroundedRecoveryPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Recovery-Grounded-Progress-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.settled_ground.progress_env_cfg:"
            "ParallelogramRoverGroundedProgressRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ParallelogramRoverGroundedProgressRecoveryPPORunnerCfg"
        ),
    },
)

##
# 注册 Gym 环境
##

gym.register(
    id="RobotLab-Isaac-Recovery-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.method1_deng.env_cfg:ParallelogramRoverRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverRecoveryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Recovery-Getup-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.method2_getup.env_cfg:ParallelogramRoverRecoveryGetupEnvCfg"
        ),
        # 共用同一 PPO runner —— 保证 A/B 对比只差在 events/rewards
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverRecoveryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Recovery-Settled-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.config.settled_ground.env_cfg:ParallelogramRoverSettledRecoveryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:ParallelogramRoverSettledRecoveryPPORunnerCfg"
        ),
    },
)
