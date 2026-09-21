"""显式失败状态池的即时恢复任务；与人工倒姿沉降实验分开记录。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 复用grounded奖励/58维网络，替换为真实失败末态与零准备接管。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ...mdp.failure_replay import initialize_failure_takeover, reset_failure_state
from .env_cfg import SettledRecoveryEventsCfg
from .grounded_env_cfg import ParallelogramRoverGroundedRecoveryEnvCfg


@configclass
class FailureReplayEventsCfg(SettledRecoveryEventsCfg):
    reset_fall_pose = EventTerm(func=reset_failure_state, mode="reset", params={"failure_state_path": ""})
    freefall_init = EventTerm(func=initialize_failure_takeover, mode="reset")
    freefall_hold = None


@configclass
class ParallelogramRoverFailureReplayEnvCfg(ParallelogramRoverGroundedRecoveryEnvCfg):
    """必须由命令行指定采集池；没有真实数据时禁止退化为人工倒姿。"""

    def __post_init__(self):
        super().__post_init__()
        # 父类 __post_init__ 设置人工姿态参数，之后整体替换以免参数误传到重放函数。
        self.events = FailureReplayEventsCfg()
