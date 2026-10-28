"""
supermimic 任务注册 — 与现有 PPO / 视觉蒸馏任务分开，不复用它们的实验名或 checkpoint。

@author pony
@date 2026-10-26
@version v0.3.0
@last_modified 2026-10-28
@changelog
    - v0.3.0 (2026-10-28): 注册从零训练楼梯任务 Scratch-Stairs（方案 D）。
    - v0.2.0 (2026-10-26): 注册越障任务 Terrain-Preview。
    - v0.1.0 (2026-10-26): 初始创建
"""

import gymnasium as gym

gym.register(
    id="RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:ParallelogramRoverSuperMimicFlatEnvCfg"},
)

gym.register(
    id="RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.terrain_env_cfg:ParallelogramRoverSuperMimicTerrainEnvCfg"},
)

gym.register(
    id="RobotLab-Isaac-SuperMimic-Scratch-Stairs-Parallelogram-Rover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.scratch_env_cfg:ParallelogramRoverSuperMimicScratchEnvCfg"},
)
