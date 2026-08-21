# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-12
# @version v0.3.13
# @last_modified 2026-08-15
# @changelog
#   - v0.3.13 (2026-08-15): 轮子接触摩擦在当前高摩擦基线之上再提高 3 倍，继续抑制楼梯打滑。
#   - v0.3.12 (2026-08-15): 轮子接触摩擦在当前高摩擦基线之上再提高 3 倍，继续抑制打滑。
#   - v0.3.11 (2026-08-15): 将轮子接触摩擦随机化范围提高约 3 倍，降低高速打滑。
#   - v0.3.10 (2026-08-13): 撤回仅完成 smoke 的 command curriculum，恢复同类轮足任务的完整命令训练基线。
#   - v0.3.9 (2026-08-13): 启用 RobotLab 原生线速度 command curriculum，逐步扩展到目标速度范围。
#   - v0.3.8 (2026-08-13): action-rate A/B 未改善 rough 高速跟随，恢复已验证的 -0.005 基线。
#   - v0.3.7 (2026-08-13): 对齐 Go2W 的动作变化惩罚，抑制主动关节与轮端目标频繁波动。
#   - v0.3.6 (2026-08-13): 固定 rough 评测未通过，恢复主动抬轮关节的标准回正惩罚。
#   - v0.3.5 (2026-08-13): 恢复小范围横向命令，并降低主动抬轮关节的常驻回正惩罚。
#   - v0.3.4 (2026-08-13): 恢复标准线速度跟踪权重，并将 rough 训练命令对齐为纯前进加偏航修正。
#   - v0.3.3 (2026-08-12): 对齐 Tita 的线速度跟踪权重，并恢复框架默认误差带宽。
#   - v0.3.2 (2026-08-12): 收紧线速度跟踪误差带宽，减少 rough 地形高速命令下的保守欠跟随。
#   - v0.3.1 (2026-08-12): 移除静止命令样本并提高最低前进速度，强化前进跟随学习信号。
#   - v0.3.0 (2026-08-12): 前进优先命令分布，并重平衡速度跟踪、动作变化和抬轮位姿奖励。
#   - v0.2.2 (2026-08-12): 按 Go2W 标准降低转向步长，增加转向速度/加速度抑制，并降低 upward 常驻奖励。
#   - v0.2.1 (2026-08-12): 降低常驻正立奖励，避免静止策略掩盖速度跟踪学习。
#   - v0.2.0 (2026-08-12): 回归标准 rough velocity 控制，抬轮由 arm 控制量和真实动力学自然产生。
#   - v0.1.5 (2026-08-12): 同一持续堵转不重复发放抬轮进度，解除后才重新武装。
#   - v0.1.4 (2026-08-12): 主动抬轮进度单次封顶 0.25，保留离地事件与 clear 的主奖励。
#   - v0.1.3 (2026-08-12): 抬轮奖励要求主动 master 目标驱动，继续排除 8 个被动 follower。
#   - v0.1.2 (2026-08-12): 提高堵转后抬轮的过渡奖励，减少稀疏 clear 奖励的探索瓶颈。
#   - v0.1.1 (2026-08-12): 统一三个越障项的堵转速度门槛，确保状态机从同一事件启动。
#   - v0.1.0 (2026-08-12): 保持 RobotLab rough 地形与 45->12 接口，加入可验证的越障奖励参数。

"""Rough-terrain velocity configuration for the parallelogram rover."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_CFG,
    PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_NAMES,
    PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
    assert_exact_policy_action_names,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import mdp as rover_mdp


@configclass
class ParallelogramRoverActionsCfg(ActionsCfg):
    """策略控制四个抬轮 master、四轮转向位置和四个轮端速度。"""

    joint_pos = velocity_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[""],
        scale=0.5,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )
    joint_vel = velocity_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[""],
        scale=5.0,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class ParallelogramRoverRewardsCfg(RewardsCfg):
    """将轮端项与低速关节项分开，避免轮速尺度主导正则项。"""

    joint_vel_steering_l2 = RewTerm(
        func=velocity_mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    joint_acc_steering_l2 = RewTerm(
        func=velocity_mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )

    joint_vel_wheel_l2 = RewTerm(
        func=velocity_mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    joint_acc_wheel_l2 = RewTerm(
        func=velocity_mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    joint_torques_wheel_l2 = RewTerm(
        func=velocity_mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )


@configclass
class ParallelogramRoverEventsCfg(EventCfg):
    """Separate wheel contact material randomization from the rest of the body."""

    randomize_wheel_rigid_body_material = EventTerm(
        func=velocity_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "static_friction_range": (8.1, 27.0),
            "dynamic_friction_range": (8.1, 21.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )


@configclass
class ParallelogramRoverRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """与 RL-SAR 45->12 部署接口一致的 rough-terrain 基线任务。"""

    actions: ParallelogramRoverActionsCfg = ParallelogramRoverActionsCfg()
    events: ParallelogramRoverEventsCfg = ParallelogramRoverEventsCfg()
    rewards: ParallelogramRoverRewardsCfg = ParallelogramRoverRewardsCfg()

    base_link_name = "base_link"
    foot_link_name = r"^(fl|rl|fr|rr)_wheel$"
    wheel_body_names = ["fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel"]
    non_wheel_body_pattern = r"^(?!.*_wheel$).*$"
    suspension_joint_names = list(PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES)
    steering_joint_names = list(PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES)
    wheel_joint_names = list(PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES)
    policy_joint_names = list(PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES)
    independent_joint_names = list(PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_NAMES)
    leg_joint_names = suspension_joint_names + steering_joint_names
    joint_names = independent_joint_names

    def __post_init__(self):
        super().__post_init__()

        # Scene：URDF 是训练源，PhysX 在 spawn 阶段临时派生 USD 并创建 MimicJointAPI。
        self.scene.robot = PARALLELOGRAM_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.base_link_name}"
        self.scene.height_scanner_base.prim_path = f"{{ENV_REGEX_NS}}/Robot/{self.base_link_name}"
        # 保持标准混合 rough curriculum，从最低两行开始，让初始台阶高度可达。
        self.scene.terrain.max_init_terrain_level = 1

        # Policy 固定 45-D；policy 和 critic 都只读取 12 个主动关节，排除 8 个 follower。
        policy_wheel_asset_cfg = SceneEntityCfg("robot", joint_names=self.wheel_joint_names, preserve_order=True)
        critic_wheel_asset_cfg = SceneEntityCfg("robot", joint_names=self.wheel_joint_names, preserve_order=True)
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.func = rover_mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.policy_joint_names
        self.observations.policy.joint_pos.params["asset_cfg"].preserve_order = True
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = policy_wheel_asset_cfg
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.policy_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].preserve_order = True
        self.observations.policy.joint_vel.scale = 0.05

        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.func = rover_mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.independent_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].preserve_order = True
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = critic_wheel_asset_cfg
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.independent_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].preserve_order = True
        self.observations.critic.joint_vel.scale = 0.05

        # Actions：沿用 Go2W/Tita 的位置腿动作加速度轮动作，并固定 RL-SAR 顺序与尺度。
        self.actions.joint_pos.scale = {
            r"^(fl|fr|rl|rr)_arm_joint$": 0.25,
            # Match the established wheeled RobotLab configs.  The wheel
            # velocity action remains the primary locomotion channel; steering
            # can still change angle for obstacle recovery, but with smaller
            # target steps to avoid rapid left-right oscillation.
            r"^(fl|fr|rl|rr)_qudong_joint$": 0.25,
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-31.4159265359, 31.4159265359)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names
        # Action 顺序必须与 RL-SAR 的 joint_mapping 和 action_scale 完全一致。
        assert_exact_policy_action_names(
            (*self.actions.joint_pos.joint_names, *self.actions.joint_vel.joint_names),
            context="ParallelogramRoverRoughEnvCfg actions",
        )

        # Reset 是唯一允许写完整关节 state 的入口；函数会从 master 派生 follower。
        self.events.randomize_reset_joints.func = rover_mdp.reset_parallelogram_joints
        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.independent_joint_names, preserve_order=True
        )
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.independent_joint_names, preserve_order=True
        )
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [self.non_wheel_body_pattern]
        self.events.randomize_wheel_rigid_body_material.params["asset_cfg"].body_names = [self.foot_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # Reward 的 joint selector 仅使用主动 master 和转向关节，避免重复计算 follower。
        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0.0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0.0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.suspension_joint_names
        self.rewards.joint_vel_steering_l2.weight = -2.0e-3
        self.rewards.joint_vel_steering_l2.params["asset_cfg"].joint_names = self.steering_joint_names
        self.rewards.joint_acc_steering_l2.weight = -5.0e-7
        self.rewards.joint_acc_steering_l2.params["asset_cfg"].joint_names = self.steering_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -0.25
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.suspension_joint_names
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_steering_l1", -0.1, self.steering_joint_names
        )
        # 固定 rough benchmark 表明 -0.01 会增加高速 arm/steering 摆动且不改善
        # 速度误差；保留已验证的 RobotLab rover 基线，避免用正则项替代前进能力。
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.wheel_vel_penalty.weight = 0.0
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [self.non_wheel_body_pattern]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        # Keep the standard wheeled RobotLab tracking scale.  The fixed
        # benchmark showed that increasing this weight only raised the
        # aggregate return while rough-terrain high-speed error stayed flat.
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.50
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # upward 的原始值在完全正立时为 4；0.10 仅保留姿态稳定 shaping，
        # 避免站起本身成为主要得分来源。
        self.rewards.upward.weight = 0.10

        self.terminations.illegal_contact = None
        # 与现有 wheeled rough 配置一致，直接训练完整命令范围。此前的
        # command curriculum 只通过启动 smoke，尚未通过同预算固定地形 A/B。
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # 前进速度是主任务；保留小范围横移和转向用于 rough 地形修正。
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = None

        if self.__class__.__name__ == "ParallelogramRoverRoughEnvCfg":
            self.disable_zero_weight_rewards()
