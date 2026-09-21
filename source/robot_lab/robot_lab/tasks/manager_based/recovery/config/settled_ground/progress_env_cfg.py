"""带有符号进展 shaping 的接地恢复实验配置。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 改为有界势能差，倒退计负分并在reset清历史。
    - v0.1.0 (2026-09-17): 保留 Grounded 合同并增加姿态/高度进展奖励。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...mdp import grounded
from .grounded_env_cfg import GroundedRecoveryRewardsCfg, ParallelogramRoverGroundedRecoveryEnvCfg


@configclass
class GroundedProgressRecoveryRewardsCfg(GroundedRecoveryRewardsCfg):
    """新增有符号进展项，最终仍要求四轮支撑和稳定保持。"""

    orientation_progress = RewTerm(func=grounded.recovery_orientation_progress, weight=2.0)
    height_progress = RewTerm(func=grounded.recovery_height_progress, weight=1.0)


@configclass
class ParallelogramRoverGroundedProgressRecoveryEnvCfg(ParallelogramRoverGroundedRecoveryEnvCfg):
    """Grounded 的独立 shaping 消融；58/70 维观测和 12 维动作不变。"""

    rewards: GroundedProgressRecoveryRewardsCfg = GroundedProgressRecoveryRewardsCfg()
