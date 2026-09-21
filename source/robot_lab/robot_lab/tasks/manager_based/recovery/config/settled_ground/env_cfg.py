# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.3.0
# @last_modified 2026-09-17
# @changelog
#   - v0.3.0 (2026-09-17): 声明可配置准备腿阻尼；默认0保留零力矩对照，非零需独立记录。
#   - v0.2.0 (2026-09-17): 默认由无穿透近地倒姿准备接管，保留原稳定门和控制预算。
#   - v0.1.0 (2026-09-16): 保持58维观测，增加物理沉降、独立受控时钟与超时失败门。

"""settled_ground：正常 vector step 中准备物理倒姿，稳定后无额外等待接管。

准备最多 6 s，不计入 5 s 的控制预算。未成功准备的尝试会显式终止而非伪装就绪。
默认控制动作、观测、网络和奖励权重沿用 recovery；准备期所有奖励由阶段门归零。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.recovery.config.method1_deng.env_cfg import (
    ParallelogramRoverRecoveryEnvCfg,
    RecoveryEventsCfg,
)
from robot_lab.tasks.manager_based.recovery.mdp.settled_ground import (
    activate_settled_recovery,
    initialize_settled_recovery,
    settled_control_time_out,
    update_settled_initialization,
)


@configclass
class SettledRecoveryEventsCfg(RecoveryEventsCfg):
    """保留 reset_fall_pose 接口；近地几何复位后按真实稳定条件结束准备阶段。"""

    # 保留此字段名，训练入口会据此禁止随机 episode 时钟。
    freefall_init = EventTerm(
        func=initialize_settled_recovery,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "preparation_leg_damping": 0.0},
    )
    freefall_hold = EventTerm(
        func=activate_settled_recovery,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "takeover_velocity_range": {}},
    )


@configclass
class SettledRecoveryTerminationsCfg:
    """初始化采样必须先于控制 timeout 和评测快照，保证 control_steps 表示已执行的物理步。"""

    initialization_failure = DoneTerm(
        func=update_settled_initialization,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "settle_time_s": 0.3,
            "max_settle_time_s": 6.0,
            "max_lin_speed": 0.10,
            "max_ang_speed": 0.20,
            "max_leg_speed": 0.30,
            "min_contact_force": 1.0,
            "require_fallen": True,
            "fallen_gz_min": -0.7,
        },
    )
    time_out = DoneTerm(func=settled_control_time_out, time_out=True)


@configclass
class ParallelogramRoverSettledRecoveryEnvCfg(ParallelogramRoverRecoveryEnvCfg):
    """原地形/关节合同不变，增加可独立训练或由 recovery checkpoint 继续训练的协议。"""

    events: SettledRecoveryEventsCfg = SettledRecoveryEventsCfg()
    terminations: SettledRecoveryTerminationsCfg = SettledRecoveryTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0
        # 显式参数便于通过 Hydra 设置带初速度的下落；真正接管动量另用 takeover_velocity_range。
        self.events.reset_fall_pose.params["root_velocity_range"] = {}
        # 已倒地接管不应默认从 1.1m 砸向地面；仅平地有效，仍由接触和连续稳定门确认。
        self.events.reset_fall_pose.params["collision_clearance"] = 0.002
