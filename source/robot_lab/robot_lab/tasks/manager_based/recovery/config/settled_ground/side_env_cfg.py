"""侧躺优先的恢复课程；不修改旧任务、机械限位或全局机器人资产。

@author pony
@date 2026-09-28
@version v0.2.0
@last_modified 2026-09-29
@changelog
    - v0.3.0 (2026-09-30): 阶段/Residual恢复加入弹射窗口和arm-wheel协同奖励。
    - v0.2.0 (2026-09-29): 新增60/72维阶段课程，修正进展量级和速度分组，保留旧Side。
    - v0.1.0 (2026-09-28): 分离侧躺/翻覆，按物理单位归一动作并取消静止侧躺朝向收益。
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.parallelogram_rover import PARALLELOGRAM_ROVER_MAX_WHEEL_SPEED_RAD_S
from robot_lab.tasks.manager_based.recovery.config.method1_deng.env_cfg import RecoveryObservationsCfg
from robot_lab.tasks.manager_based.recovery.mdp.method1_deng import rewards as recovery_rewards
from robot_lab.tasks.manager_based.recovery.mdp import grounded, phase

from .progress_env_cfg import (
    GroundedProgressRecoveryRewardsCfg,
    ParallelogramRoverGroundedProgressRecoveryEnvCfg,
)


@configclass
class ParallelogramRoverSideRecoveryEnvCfg(ParallelogramRoverGroundedProgressRecoveryEnvCfg):
    """58/70→12维不变，但动作单位改变，旧checkpoint不能直接加载。

    仍使用未标定的仿真驱动基线，不将侧躺任务命名或注册当作硬件可行性证明。
    wheel scale=10使有效轮速目标能被初始探索覆盖，最终目标仍被原机械速度界裁剪。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.copy()
        self.events.reset_fall_pose.params["pose_weights"] = {"side_lying": 1.0}
        self.actions.joint_pos.scale = {".*_arm_joint": 0.6109, ".*_qudong_joint": 1.57}
        self.actions.joint_pos.clip = {".*_arm_joint": (-0.6109, 0.6109), ".*_qudong_joint": (-1.57, 1.57)}
        self.actions.joint_vel.scale = 10.0
        self.actions.joint_vel.clip = {
            ".*": (-PARALLELOGRAM_ROVER_MAX_WHEEL_SPEED_RAD_S, PARALLELOGRAM_ROVER_MAX_WHEEL_SPEED_RAD_S),
        }
        # 已有有符号进展项；不再让保持侧躺本身每步获得固定朝向收益。
        self.rewards.orientation.weight = 0.0


@configclass
class ParallelogramRoverUpsideRecoveryEnvCfg(ParallelogramRoverSideRecoveryEnvCfg):
    """完全翻覆专项实验，沿用prone历史标签；结构可达性与成功率尚未验证。"""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_fall_pose.params["pose_weights"] = {"prone": 1.0}


@configclass
class PhaseRecoveryObservationsCfg:
    """原58/70维前缀保持原顺序，分别在末尾追加阶段和初始倒向。"""

    @configclass
    class PolicyCfg(RecoveryObservationsCfg.PolicyCfg):
        recovery_phase = ObsTerm(func=phase.recovery_phase, params={"duration_s": 3.0})
        recovery_initial_side = ObsTerm(func=phase.recovery_initial_side)

    @configclass
    class CriticCfg(RecoveryObservationsCfg.CriticCfg):
        recovery_phase = ObsTerm(func=phase.recovery_phase, params={"duration_s": 3.0})
        recovery_initial_side = ObsTerm(func=phase.recovery_initial_side)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SideResidualRecoveryRewardsCfg(GroundedProgressRecoveryRewardsCfg):
    """程序先验/Residual 共用的短时动力学 shaping。"""

    arm_wheel_coord = RewTerm(
        func=recovery_rewards.recovery_arm_wheel_coord, weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    launch_roll_rate = RewTerm(
        func=grounded.recovery_launch_roll_rate, weight=2.0,
        params={"duration_s": 1.2, "max_rate": 4.0},
    )
    launch_vertical_speed = RewTerm(
        func=grounded.recovery_launch_vertical_speed, weight=0.5,
        params={"duration_s": 1.2, "max_speed": 1.0},
    )


@configclass
class ParallelogramRoverSidePhaseRecoveryEnvCfg(ParallelogramRoverSideRecoveryEnvCfg):
    """独立60/72→12任务；旧58维模型必须经过显式BC转换，不能直接resume。"""

    observations: PhaseRecoveryObservationsCfg = PhaseRecoveryObservationsCfg()
    rewards: SideResidualRecoveryRewardsCfg = SideResidualRecoveryRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_progress.params = {"gain": 1.0, "as_rate": True}
        self.rewards.orientation_progress.weight = 2.0
        self.rewards.height_progress.params = {"as_rate": True}
        self.rewards.height_progress.weight = 0.2
        self.rewards.joint_velocity.func = grounded.recovery_leg_velocity
        self.rewards.joint_velocity.weight = -2e-4
        self.rewards.wheel_velocity.func = grounded.recovery_phase_wheel_velocity
        self.rewards.wheel_velocity.weight = -1e-4
