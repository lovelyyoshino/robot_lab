"""失败末态重放的 program31 residual 恢复环境。

@author pony
@date 2026-09-30
@version v0.1.0
@last_modified 2026-09-30
@changelog
    - v0.1.0 (2026-09-30): 复用60/72维Side-Residual观测和奖励，替换为失败末态即时接管。

失败池只保存运动学末态，不包含 PhysX 接触求解器缓存；该环境用于针对性 PPO
微调，不代表跨引擎或实机可复现。主训练仍使用 Side-Residual 的 settled_ground
协议，只有确认失败类别后才进入本环境。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ...mdp.failure_replay import reset_failure_state
from .replay_env_cfg import FailureReplayEventsCfg
from .side_env_cfg import ParallelogramRoverSidePhaseRecoveryEnvCfg


@configclass
class FailureReplayResidualEventsCfg(FailureReplayEventsCfg):
    """Residual微调允许重放超时失败；原FailureReplay仍只收提前终止失败。"""

    reset_fall_pose = EventTerm(
        func=reset_failure_state,
        mode="reset",
        params={"failure_state_path": "", "failure_kind": "all"},
    )


@configclass
class ParallelogramRoverFailureReplayResidualEnvCfg(ParallelogramRoverSidePhaseRecoveryEnvCfg):
    """60/72维 residual Actor 从真实失败末态开始学习纠偏。"""

    events: FailureReplayResidualEventsCfg = FailureReplayResidualEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 父类会为 settled_ground 写入人工倒姿参数；这里整体替换，拒绝把它们
        # 误传给 failure_replay 的 reset 函数，且保持 Phase 的奖励/观测合同。
        self.events = FailureReplayResidualEventsCfg()
