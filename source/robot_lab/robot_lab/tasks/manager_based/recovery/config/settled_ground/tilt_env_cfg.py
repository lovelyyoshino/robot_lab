"""连续横滚倾角的即时控制预训练，不等同于静止倒地恢复验收。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 独立35–55度横滚任务，复用Grounded奖励及58/70→12合同。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...mdp.failure_replay import initialize_failure_takeover
from ...mdp.method1_deng.events import reset_with_fall_pose
from .env_cfg import SettledRecoveryEventsCfg
from .grounded_env_cfg import ParallelogramRoverGroundedRecoveryEnvCfg


@configclass
class ControlledTiltRecoveryEventsCfg(SettledRecoveryEventsCfg):
    """保留reset_obs_cache的事件顺序，几何复位后在首个物理步前恢复驱动。"""

    reset_fall_pose = EventTerm(
        func=reset_with_fall_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_weights": None,
            "roll_tilt_range_deg": (35.0, 55.0),
            "leg_joint_noise": 0.3,
            "root_velocity_range": {},
            "collision_clearance": 0.002,
        },
    )
    # 这里只复用即时驱动/时钟初始化，不加载失败池，也不宣称初态已经被动静止。
    freefall_init = EventTerm(func=initialize_failure_takeover, mode="reset")
    freefall_hold = None


@configclass
class ParallelogramRoverControlledTiltRecoveryEnvCfg(ParallelogramRoverGroundedRecoveryEnvCfg):
    """固定角段分阶段预训练；阶段参数不进入actor/critic，也不放宽正式准备门。"""

    initialization_protocol: str = "controlled_tilt"
    acceptance_eligible: bool = False

    def __post_init__(self):
        super().__post_init__()
        # 父类会填写沉降制动和姿态池；之后整体替换，防止把它们误传给即时初始化。
        self.events = ControlledTiltRecoveryEventsCfg()
        self.episode_length_s = 5.0
