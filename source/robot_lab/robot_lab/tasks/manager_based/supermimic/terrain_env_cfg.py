"""
terrain_env_cfg.py — SuperMimic 越障任务：在 Teacher-Preview（232 维、8→12→16 cm 楼梯课程）上去掉想象中无法复现的扰动。

与 `ParallelogramRoverVisionTeacherPreviewEnvCfg` 的差别，均为世界模型/想象一致性服务：
1. 关掉推车和外力：世界模型不观测外力；
2. 关掉 policy 观测噪声（含高度扫描噪声）：想象观测可逐列核对；
3. 加 bad_orientation 终止（1.2 rad，与 PreviewRewardCfg.tilt_limit 一致）：Preview 原本只有超时/成功截断，
   翻车后的数据对越障无用，且软存活需要对应的真实终止。
课程、地形、奖励和观测语义不变，因此 Teacher-Preview 的 actor 权重可直接热启动。

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

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview_env_cfg import (
    ParallelogramRoverVisionTeacherPreviewEnvCfg,
)

TILT_LIMIT_RAD = 1.2


@configclass
class ParallelogramRoverSuperMimicTerrainEnvCfg(ParallelogramRoverVisionTeacherPreviewEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.noise = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp_core.bad_orientation, params={"limit_angle": TILT_LIMIT_RAD}
        )
