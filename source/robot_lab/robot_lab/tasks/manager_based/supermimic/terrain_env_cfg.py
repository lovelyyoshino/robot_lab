"""
terrain_env_cfg.py — SuperMimic 越障任务：在 Teacher-Preview（232 维、8→12→16 cm 楼梯课程）上去掉想象中无法复现的扰动。

与 `ParallelogramRoverVisionTeacherPreviewEnvCfg` 的差别，均为世界模型/想象一致性服务：
1. 关掉推车和外力：世界模型不观测外力；
2. 关掉 policy 观测噪声（含高度扫描噪声）：想象观测可逐列核对；
3. 加 bad_orientation 终止（1.2 rad，与 PreviewRewardCfg.tilt_limit 一致）：Preview 原本只有超时/成功截断，
   翻车后的数据对越障无用，且软存活需要对应的真实终止。
4. 轮胎摩擦固定为 WHEEL_FRICTION（静 1.2 / 动 1.0）：用户 2026-11-07 定为 μ=1.0，取代 vision_env_cfg 里"待实车标定"的
   随机范围（静 0.5–1.2 / 动 0.4–1.0）。摩擦 A/B：8 cm 楼梯 μ=1.0 成功 98%、μ=0.5 只有 3%。本类派生出训练任务
   （Obstacle）与平地验收任务（Terrain-Preview）；单台阶/楼梯验收走 Teacher-Preview，由 evaluate_parallelogram_vision.py
   在 --supermimic_terrain 下读同一个常量。其他 PPO 视觉任务不受影响。
课程、地形、奖励和观测语义不变，因此 Teacher-Preview 的 actor 权重可直接热启动。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-11-07
@changelog
    - v0.2.0 (2026-11-07): 轮胎摩擦固定为 WHEEL_FRICTION = (1.2, 1.0)（静, 动）。
    - v0.1.0 (2026-10-26): 初始创建
"""

from isaaclab.envs import mdp as mdp_core
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview_env_cfg import (
    ParallelogramRoverVisionTeacherPreviewEnvCfg,
)

TILT_LIMIT_RAD = 1.2
WHEEL_FRICTION = (1.2, 1.0)  # (静摩擦, 动摩擦)，与摩擦 A/B 的 μ=1.0 一致


@configclass
class ParallelogramRoverSuperMimicTerrainEnvCfg(ParallelogramRoverVisionTeacherPreviewEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.height_scan.noise = None
        # float32传感器时钟约16秒后可能漏掉20ms更新；想象状态必须对应当前物理步的扫描。
        self.scene.height_scanner.update_period = 0.0
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp_core.bad_orientation, params={"limit_angle": TILT_LIMIT_RAD}
        )
        material = self.events.randomize_wheel_rigid_body_material.params
        material["static_friction_range"] = (WHEEL_FRICTION[0],) * 2
        material["dynamic_friction_range"] = (WHEEL_FRICTION[1],) * 2
