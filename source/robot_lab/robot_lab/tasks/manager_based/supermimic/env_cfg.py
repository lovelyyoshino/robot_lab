"""
env_cfg.py — SuperMimic 专用平地速度任务：在 rover 平地配置上去掉不可建模的扰动。

与 `ParallelogramRoverFlatEnvCfg` 的差别只有三处，均为世界模型服务：
1. 关掉随机推车和外力：世界模型不观测外力，这类跳变只会变成拟合噪声；
2. 关掉 policy 观测噪声：让 `rover_task.policy_observation` 可以逐列核对仿真观测；
3. 加入倾角终止（1.0 rad，与 RoverRewardCfg.tilt_limit 一致）：rough 基线没有翻车终止，
   翻车后的数据对平地速度任务没有价值，并会被软存活权重放大。
动作、关节顺序、PD 增益、摩擦/质量随机化保持不变，导出的 actor 仍走 45→12 部署接口。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from isaaclab.envs import mdp as mdp_core
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.flat_env_cfg import (
    ParallelogramRoverFlatEnvCfg,
)

TILT_LIMIT_RAD = 1.0


@configclass
class ParallelogramRoverSuperMimicFlatEnvCfg(ParallelogramRoverFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        self.observations.policy.enable_corruption = False
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp_core.bad_orientation, params={"limit_angle": TILT_LIMIT_RAD}
        )
        # 父类只对精确类名清理零权重奖励；子类需自己调用，否则空 joint selector 会解析失败。
        self.disable_zero_weight_rewards()
