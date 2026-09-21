"""两轮行走 runner：仅新增实验目录，不破坏 44 维 balance/getup 权重复用。

@author pony
@date 2026-09-16
@version v0.1.0
@last_modified 2026-09-16
@changelog
    - v0.1.0 (2026-09-16): 分离 motion 实验；旧 42 维 checkpoint 需迁移后加载。
"""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import ParallelogramRoverUprightGetupPPORunnerCfg, ParallelogramRoverUprightPPORunnerCfg


@configclass
class ParallelogramRoverUprightMotionPPORunnerCfg(ParallelogramRoverUprightPPORunnerCfg):
    experiment_name = "parallelogram_rover_upright_motion"
    run_name = "success_curriculum"


@configclass
class ParallelogramRoverUprightMotionGetupPPORunnerCfg(ParallelogramRoverUprightGetupPPORunnerCfg):
    experiment_name = "parallelogram_rover_upright_motion_getup"
    run_name = "motion_getup"
