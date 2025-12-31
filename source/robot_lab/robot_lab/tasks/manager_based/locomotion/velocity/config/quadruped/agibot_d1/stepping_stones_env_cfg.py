# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Stepping Stones Environment Configuration for Agibot D1.

This module defines the environment configuration for training the Agibot D1
quadruped robot to navigate stepping stones (梅花桩) terrain. The stepping
stones present a challenging locomotion task that requires precise foot
placement and dynamic balance.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.agibot_d1.rough_env_cfg import (
    AgibotD1RoughEnvCfg,
)


# Stepping stones only terrain configuration
# Uses Isaac Lab's built-in HfSteppingStonesTerrainCfg
STEPPING_STONES_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.2,
            stone_width_range=(0.4, 1.2),
            stone_distance_range=(0.05, 0.3),
            holes_depth=-1.0,
            platform_width=1.5,
        ),
    },
)


# =============================================================================
# Competition-style terrain configuration (similar to Isaac Gym competition)
# This creates a sequential track with 9 terrain types, matching the competition format:
#   Row 0: Flat terrain (starting area)
#   Row 1: Pyramid slope (slope=-0.3)
#   Row 2: Random rough terrain (height: -0.15~0.15)
#   Row 3: Discrete obstacles (height: 0.15)
#   Row 4: Wave terrain (amplitude: 0.2)
#   Row 5: Stairs up (step_height: 0.15)
#   Row 6: Stairs down (step_height: -0.15)
#   Row 7: Stepping stones (stone_size: 1.0, distance: 0.25)
#   Row 8: Flat terrain (ending area)
# =============================================================================
COMPETITION_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(12.0, 12.0),  # Match competition: 12m x 12m per terrain cell
    border_width=25.0,
    num_rows=9,  # 9 sequential terrain types (like competition track)
    num_cols=1,  # Single column = single track (all robots on same path)
    horizontal_scale=0.25,  # Match competition: 0.25m resolution
    vertical_scale=0.005,
    slope_threshold=1.5,  # Higher threshold for competition terrains
    difficulty_range=(0.0, 1.0),  # 启用难度递增 curriculum
    use_cache=False,
    curriculum=True,  # 启用 curriculum 让难度逐渐增加
    sub_terrains={
        # Flat start area (proportion determines which rows get this terrain)
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.11,  # ~1/9 = row 0
        ),
        # Pyramid slope terrain - 增加坡度
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.11,  # row 1
            slope_range=(-0.3, -0.3),  # 增加坡度 (原: -0.2)
            platform_width=2.5,
            inverted=False,
        ),
        # Random rough terrain - 增加粗糙度
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.11,  # row 2
            noise_range=(-0.15, 0.15),  # 增加高度变化 (原: -0.10~0.10)
            noise_step=0.05,
            border_width=0.0,
        ),
        # Discrete obstacles - 增加障碍物高度
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.11,  # row 3
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.8, 2.5),  # 更小的障碍物
            obstacle_height_range=(0.15, 0.15),  # 增加高度 (原: 0.10)
            num_obstacles=20,  # 更多障碍物 (原: 15)
            platform_width=0.0,
        ),
        # Wave terrain - 增加振幅
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.11,  # row 4
            amplitude_range=(0.20, 0.20),  # 增加振幅 (原: 0.15)
            num_waves=3,  # 更多波浪 (原: 2)
            border_width=0.0,
        ),
        # Stairs up - 增加台阶高度
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.11,  # row 5
            step_height_range=(0.15, 0.15),  # 增加台阶高度 (原: 0.12)
            step_width=0.6,  # 更窄的台阶 (原: 0.75)
            platform_width=0.0,
            inverted=False,
        ),
        # Stairs down - 增加台阶高度
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.11,  # row 6
            step_height_range=(0.15, 0.15),  # 增加台阶高度 (原: 0.12)
            step_width=0.6,  # 更窄的台阶 (原: 0.75)
            platform_width=0.0,
            inverted=True,
        ),
        # Stepping stones (main challenge) - 增加难度
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.12,  # row 7 (slightly higher to ensure inclusion)
            stone_height_max=0.20,  # 增加石头高度变化 (原: 0.15)
            stone_width_range=(0.5, 1.0),  # 更小的石头 (原: 0.8~1.2)
            stone_distance_range=(0.20, 0.30),  # 更大的间隙 (原: 0.15~0.20)
            holes_depth=-1.0,
            platform_width=0.8,  # 更小的起始平台 (原: 1.0)
        ),
    },
)


@configclass
class AgibotD1SteppingStonesEnvCfg(AgibotD1RoughEnvCfg):
    """
    Environment configuration for Agibot D1 on stepping stones terrain.

    This configuration extends the rough terrain configuration with:
    - Stepping stones terrain generator
    - Enhanced rewards for precise foot placement
    - Increased feet air time bonus for jumping between stones
    - Height scan enabled for terrain awareness
    """

    def __post_init__(self):
        # Call parent post init
        super().__post_init__()

        # ------------------------------Scene------------------------------
        # Use stepping stones terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STEPPING_STONES_TERRAINS_CFG

        # ------------------------------Observations------------------------------
        # Enable height scan for terrain awareness (critical for stepping stones)
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------Rewards------------------------------
        # Stepping stones specific reward tuning

        # Increase velocity tracking for forward progress
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Enable and boost feet air time reward (crucial for jumping between stones)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=2.0,  # Increased weight
            params={
                "command_name": "base_velocity",
                "threshold": 0.4,  # Slightly lower threshold for faster stepping
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            },
        )

        # Encourage higher foot lifting for clearing gaps
        self.rewards.feet_height = RewTerm(
            func=mdp.feet_height,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=self.foot_link_name),
                "tanh_mult": 2.0,
                "target_height": 0.08,  # Higher target for stepping stones
                "command_name": "base_velocity",
            },
        )

        # Reduce foot height body penalty to allow more dynamic movement
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.25

        # Enable feet slide penalty (important for precise placement)
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "asset_cfg": SceneEntityCfg("robot", body_names=self.foot_link_name),
            },
        )

        # Slightly reduce action rate penalty to allow more dynamic movements
        self.rewards.action_rate_l2.weight = -0.005

        # Increase upward orientation reward
        self.rewards.upward.weight = 1.5

        # Reduce stand still penalty (robot needs to be more dynamic)
        self.rewards.stand_still.weight = -1.0

        # ------------------------------Events------------------------------
        # Reduce initial pose randomization for stepping stones
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (0.0, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        }

        # ------------------------------Commands------------------------------
        # Slightly reduce command ranges for more controlled movement
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # Longer resampling time for stable traversal
        self.commands.base_velocity.resampling_time_range = (12.0, 12.0)

        # ------------------------------Episode------------------------------
        # Longer episodes for stepping stones traversal
        self.episode_length_s = 25.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AgibotD1SteppingStonesEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class AgibotD1CompetitionEnvCfg(AgibotD1RoughEnvCfg):
    """
    Environment configuration for Agibot D1 on competition terrain.

    This configuration matches the Isaac Gym legged robot competition format:
    - Mixed terrain types with curriculum learning
    - Forward-only velocity commands (no lateral/turning)
    - Competition-tuned reward weights
    """

    def __post_init__(self):
        # Call parent post init
        super().__post_init__()

        # ------------------------------Simulation------------------------------
        # Increase GPU collision stack size for complex terrains
        self.sim.physx.gpu_collision_stack_size = 2**27  # 128MB (default is 64MB)

        # ------------------------------Scene------------------------------
        # Use competition terrain with curriculum learning
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = COMPETITION_TERRAINS_CFG
        # Enable curriculum - start from easier terrains
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.scene.terrain.max_init_terrain_level = 5

        # ------------------------------Observations------------------------------
        # Enable height scan for terrain awareness
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------Rewards------------------------------
        # Competition-tuned rewards (matching Isaac Gym go2_config.py)

        # Velocity tracking - IMPORTANT: use larger std for smoother learning curve
        # When error=2.0, std=0.5: exp(-16)≈0 (too harsh)
        # When error=2.0, std=1.0: exp(-4)≈0.018 (more gradual)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # Competition: 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 1.0  # Increased from 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5  # Competition: 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 1.0  # Increased from 0.5

        # Root penalties - match competition
        self.rewards.lin_vel_z_l2.weight = -2.0  # Penalize vertical velocity
        self.rewards.ang_vel_xy_l2.weight = -0.05  # Slightly higher than competition
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=-0.5,  # Increased to encourage stable posture
            params={},
        )

        # DISABLE upward reward - causes robot to just stand still!
        self.rewards.upward = None

        # Joint penalties - match competition
        self.rewards.joint_torques_l2.weight = -0.0002  # Competition: -0.0002
        self.rewards.joint_acc_l2.weight = -2.5e-7  # Competition: -2.5e-7

        # Action rate - match competition
        self.rewards.action_rate_l2.weight = -0.01  # Increased to reduce jitter

        # DISABLE feet_air_time - still produces negative rewards
        # The reward formula (last_air_time - threshold) can be negative
        # Will rely on velocity tracking instead
        self.rewards.feet_air_time = None

        # DISABLE feet_contact_without_cmd - this rewards standing still!
        self.rewards.feet_contact_without_cmd = None

        # Disable feet height reward (not in competition config)
        self.rewards.feet_height = None
        self.rewards.feet_height_body = None

        # Collision penalty - match competition
        self.rewards.undesired_contacts.weight = -1.0  # Competition: -1.0

        # DISABLE stand_still penalty - competition uses different logic
        # (only penalizes at zero command, but our commands are always > 0.5)
        self.rewards.stand_still = None

        # Joint position limits - match competition
        self.rewards.joint_pos_limits.weight = -0.01  # Competition: -0.01

        # Disable joint mirror for competition (not needed for forward running)
        self.rewards.joint_mirror = None

        # Disable joint_pos_penalty (relies on stand_still logic)
        self.rewards.joint_pos_penalty = None

        # Keep joint power penalty
        self.rewards.joint_power.weight = -2e-5

        # ------------------------------Events------------------------------
        # Disable domain randomization for competition (like go2_config)
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        # Minimal reset randomization for competition
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.1),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.1, 0.1),  # Small yaw variation
            },
            "velocity_range": {
                "x": (0.0, 0.5),  # Start with some forward velocity
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ------------------------------Commands------------------------------
        # Competition: forward-only velocity commands
        # 增加速度范围，匹配 go2_config 的 (0.5, 2.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 2.0)  # 增加速度 (原: 0.3~1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # No turning
        self.commands.base_velocity.heading_command = False  # Disable heading command
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        # ------------------------------Curriculum------------------------------
        # Enable terrain curriculum
        # (terrain_levels is inherited from parent)

        # ------------------------------Terminations------------------------------
        # CRITICAL: Re-enable illegal_contact termination (was disabled in rough_env_cfg)
        # Robot should terminate if base or thigh contacts ground (like competition)
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BASE_LINK", ".*_HIP_LINK"]),
                "threshold": 1.0,
            },
        )

        # ------------------------------Episode------------------------------
        # Match competition episode length
        self.episode_length_s = 25.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AgibotD1CompetitionEnvCfg":
            self.disable_zero_weight_rewards()
