"""44/49 维两轮行走任务；balance→getup 保持网络和动作合同一致。

@author pony
@date 2026-09-16
@version v0.7.0
@last_modified 2026-09-17
@changelog
    - v0.7.0 (2026-09-17): 显式开放四轮带速起立参数；默认零速，不改变旧训练分布。
    - v0.6.0 (2026-09-17): 用非负零前进质量奖励替代负误差，移除 actor 不可观测的 XY 奖励。
    - v0.5.0 (2026-09-17): 固定 motion 俯仰目标，并加入零前进稠密速度误差。
    - v0.4.0 (2026-09-17): 延长 motion 训练 episode，连续成功门提高至 2 秒。
    - v0.3.0 (2026-09-17): 将世界 XY 漂移锚定惩罚接入 balance/motion-getup。
    - v0.2.0 (2026-09-17): 运动阶段混入 20% 静止指令，并显式导出静止跟踪尺度。
    - v0.1.0 (2026-09-16): 原 42/47 维前缀完全保留，末尾追加地面前进和 yaw 指令。
"""

import math

import isaaclab.envs.mdp as mdp_core
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..mdp.motion import (
    MotionCommandCfg,
    motion_balance_hold,
    motion_lateral_slip,
    motion_velocity_tracking,
    motion_zero_forward_velocity_quality,
)
from .getup_env_cfg import GetupRewardsCfg, ParallelogramRoverUprightGetupEnvCfg
from .upright_env_cfg import (
    ParallelogramRoverUprightEnvCfg,
    UprightCommandsCfg,
    UprightObservationsCfg,
    UprightRewardsCfg,
)


@configclass
class MotionObservationsCfg(UprightObservationsCfg):
    """不修改任何已训练字段的函数、顺序、尺度或噪声；critic 命令在 priv_* 之后。"""

    @configclass
    class PolicyCfg(UprightObservationsCfg.PolicyCfg):
        motion_command = ObsTerm(func=mdp_core.generated_commands, params={"command_name": "motion_command"})

    @configclass
    class CriticCfg(UprightObservationsCfg.CriticCfg):
        motion_command = ObsTerm(func=mdp_core.generated_commands, params={"command_name": "motion_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MotionCommandsCfg(UprightCommandsCfg):
    motion_command: MotionCommandCfg = MotionCommandCfg(
        stationary_fraction=0.2,
        stationary_drift_limit_m=0.30,
        success_time_s=2.0,
    )


@configclass
class MotionRewardsCfg(UprightRewardsCfg):
    balance_hold = RewTerm(func=motion_balance_hold, weight=2.0)
    velocity_tracking = RewTerm(
        func=motion_velocity_tracking,
        weight=2.0,
        params={"velocity_sigma": 0.25, "yaw_sigma": 0.5, "stationary_velocity_sigma": 0.04, "stationary_yaw_sigma": 0.10},
    )
    lateral_slip = RewTerm(func=motion_lateral_slip, weight=-0.5)
    # 非负质量分数避免策略通过提前跌倒缩短负奖励累计；只补零前进/原地转向。
    zero_forward_velocity_quality = RewTerm(
        func=motion_zero_forward_velocity_quality,
        weight=2.0,
        params={"forward_scale": 0.10, "lateral_scale": 0.10, "yaw_scale": 0.20},
    )
    # actor 不含世界位置/里程计，XY 位移不能作为马尔可夫奖励；只在课程门和独立评测使用。


@configclass
class MotionGetupRewardsCfg(GetupRewardsCfg):
    # 原 getup_success 也限制总角速度，必须一起替换，否则转弯会失去起立成功奖励。
    balance_hold = RewTerm(func=motion_balance_hold, weight=2.0)
    getup_success = RewTerm(func=motion_balance_hold, weight=4.0)
    velocity_tracking = RewTerm(
        func=motion_velocity_tracking,
        weight=2.0,
        params={"velocity_sigma": 0.25, "yaw_sigma": 0.5, "stationary_velocity_sigma": 0.04, "stationary_yaw_sigma": 0.10},
    )
    lateral_slip = RewTerm(func=motion_lateral_slip, weight=-0.5)
    zero_forward_velocity_quality = RewTerm(
        func=motion_zero_forward_velocity_quality,
        weight=2.0,
        params={"forward_scale": 0.10, "lateral_scale": 0.10, "yaw_scale": 0.20},
    )


@configclass
class ParallelogramRoverUprightMotionEnvCfg(ParallelogramRoverUprightEnvCfg):
    """先学零速真实双轮支撑，再按成功门扩展直行和转向；轮动作仍为 0.8。"""

    observations: MotionObservationsCfg = MotionObservationsCfg()
    commands: MotionCommandsCfg = MotionCommandsCfg()
    rewards: MotionRewardsCfg = MotionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 最终 stationary/stop 验收固定为 90 度；随机改变倒立摆目标会要求轮子平移，
        # 与无世界位置观测的原地合同冲突，因此 motion 全阶段保持同一名义姿态。
        self.commands.pitch_command.pitch_range = (math.pi / 2, math.pi / 2)
        self.commands.pitch_command.resampling_time_range = (1.0e9, 1.0e9)
        # 8 秒只用于训练：给 2 秒连续成功门和 3～5 秒命令窗口留出余量。
        # 长时 30 秒保持仍由独立评测协议负责，不能用训练 episode 代替。
        self.episode_length_s = 8.0


@configclass
class ParallelogramRoverUprightMotionGetupEnvCfg(ParallelogramRoverUprightGetupEnvCfg):
    """35% 四轮起点与已有保持/行走混训；用 44 维 motion checkpoint 续训。"""

    observations: MotionObservationsCfg = MotionObservationsCfg()
    commands: MotionCommandsCfg = MotionCommandsCfg()
    rewards: MotionGetupRewardsCfg = MotionGetupRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 可用 Hydra 覆盖为 [0.1,0.3] 或 [-0.3,-0.1]；与运动命令独立采样。
        self.events.reset_pose.params["quad_forward_velocity_range"] = (0.0, 0.0)
        self.commands.pitch_command.pitch_range = (math.pi / 2, math.pi / 2)
        self.commands.pitch_command.resampling_time_range = (1.0e9, 1.0e9)
        # 起立阶段额外保留两秒稳定/接管余量，和 motion 使用同一成功门。
        self.episode_length_s = 8.0
