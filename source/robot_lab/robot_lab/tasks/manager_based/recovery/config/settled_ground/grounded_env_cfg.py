"""显式制动准备与接地奖励的恢复实验；不覆盖旧 EDRS 基线。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 新增侧躺/仰躺接管、连续朝向引导和真实四轮支撑任务。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...mdp import grounded
from ...mdp.method1_deng import rewards as recovery_rewards
from .env_cfg import ParallelogramRoverSettledRecoveryEnvCfg


@configclass
class GroundedRecoveryRewardsCfg:
    """量纲归一的任务信号；无高速联动奖励，避免持续抖动胜过完成恢复。"""

    step_counter = RewTerm(func=recovery_rewards.recovery_step_counter, weight=1e-6)
    orientation = RewTerm(func=grounded.recovery_orientation_score, weight=4.0)
    oriented_height = RewTerm(func=grounded.recovery_oriented_height, weight=3.0)
    oriented_joints = RewTerm(func=grounded.recovery_oriented_joints, weight=1.0)
    wheel_support = RewTerm(func=grounded.recovery_four_wheel_support, weight=4.0)
    stable_hold = RewTerm(func=grounded.recovery_stable_hold, weight=8.0)
    joint_velocity = RewTerm(func=recovery_rewards.recovery_joint_velocity, weight=-1e-3)
    joint_acceleration = RewTerm(func=recovery_rewards.recovery_joint_acceleration, weight=-2.5e-8)
    torques = RewTerm(func=recovery_rewards.recovery_torques, weight=-5e-6)
    wheel_velocity = RewTerm(func=recovery_rewards.recovery_wheel_velocity, weight=-1e-3)


@configclass
class ParallelogramRoverGroundedRecoveryEnvCfg(ParallelogramRoverSettledRecoveryEnvCfg):
    """先训两侧侧躺和仰躺；随机SO(3)、动量及真实失败状态池另行验收。"""

    rewards: GroundedRecoveryRewardsCfg = GroundedRecoveryRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.freefall_init.params["preparation_leg_damping"] = 0.2
        self.events.reset_fall_pose.params["pose_weights"] = {"side_lying": 0.5, "prone": 0.5}
