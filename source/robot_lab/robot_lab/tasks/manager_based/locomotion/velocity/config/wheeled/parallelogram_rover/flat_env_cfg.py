# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-12
# @version v0.2.0
# @last_modified 2026-08-12
# @changelog
#   - v0.2.0 (2026-08-12): 与标准 rough velocity 配置保持直接控制语义。

"""Flat-terrain velocity configuration for the parallelogram rover."""

from isaaclab.utils import configclass

from .rough_env_cfg import ParallelogramRoverRoughEnvCfg


@configclass
class ParallelogramRoverFlatEnvCfg(ParallelogramRoverRoughEnvCfg):
    """移除地形生成器和射线传感器的平地训练配置。"""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "ParallelogramRoverFlatEnvCfg":
            self.disable_zero_weight_rewards()
