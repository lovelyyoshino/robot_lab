# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.2
# @last_modified 2026-09-17
# @changelog
#   - v0.2.2 (2026-09-17): 撤销接触力合并归属的旧误判，说明传感器索引与严格支撑门。
#   - v0.2.1 (2026-09-17): 修正旧几何探针可行域注释；保持当前训练数值不变。
#   - v0.2.0 (2026-09-16): 根据整车碰撞几何复位，修正后轮穿地和奖励支撑漏洞。
#   - v0.1.0 (2026-08-25): 建立 standalone 直立环境；单 cfg 覆盖平衡+起立+行进。

"""平行四边形轮足车直立（双轮平衡）环境 —— **standalone，不继承 velocity 任务族**。

**独立性（PLAN §3.0-A 硬规则）**：从 `ManagerBasedRLEnvCfg` 直接继承；
现有 `rough_env_cfg.py` / `flat_env_cfg.py` / `velocity_env_cfg.py` 一行不改，
仅**只读引用** `PARALLELOGRAM_ROVER_CFG`。摩擦、地形、奖励、终止全部本地声明。

本配置只提供42维姿态/平衡基础；motion_env_cfg.py 另追加速度指令到44维。
当前75°–105°采样区间是实验参数。旧探针漏算碰撞几何，不能再据其力矩值
断言可保持范围、机械余量或20°–60°物理不可保持；真实双后轮验收才是能力证据。
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_core
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_CFG,
    PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
)
from robot_lab.tasks.manager_based.upright import mdp as upright_mdp

# ── 命名（全部来自资产常量）──
LEG_JOINT_NAMES = [
    *PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
]
WHEEL_JOINT_NAMES = list(PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES)
BASE_LINK_NAME = "base_link"
WHEEL_BODY_REGEX = r"^(fl|rl|fr|rr)_wheel$"
UNDESIRED_CONTACT_BODIES = [BASE_LINK_NAME]
"""保留旧基线仅惩罚 base_link 的配置，便于比较既有训练。

旧探针用 articulation.body_names 标注 contact sensor 的力数组，二者顺序不同；
所谓轮力归属到臂或转向体、碰撞体合并的解释已撤销。该历史选择不代表其他
非轮接触应被允许；当前 balance_hold 等严格支撑门按 sensor.body_names 对齐，
要求双后轮承重、前轮离地且所有非轮体均无接触。此处仅纠正文档，不改奖励数值。
"""


# ── 场景 ──

@configclass
class UprightSceneCfg(InteractiveSceneCfg):
    """平地 + 机器人 + 接触传感器。

    **不含 height_scanner**：直立是倒立摆平衡任务，靠本体高频反馈
    （时间常数 0.24 s），地形高度信息用不上；且省渲染开销。
    窄缝/台阶场景在后续课程里通过 `sub_terrains` 加入。
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = PARALLELOGRAM_ROVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, texture_file=None),
    )


# ── 指令 ──

@configclass
class UprightCommandsCfg:
    """pitch 指令。采样区间受 probe2 实测可保持区硬约束。"""

    pitch_command = upright_mdp.PitchCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        # 初始只在最优点 90° 邻域采样（1.45-1.70 ≈ 83°-97°）
        pitch_range=(1.45, 1.70),
        # 硬上下界 = 可保持区 75°-105°，课程不得越界
        hard_limits=(upright_mdp.HOLDABLE_PITCH_MIN, upright_mdp.HOLDABLE_PITCH_MAX),
        debug_vis=False,
    )


# ── 动作（12-D，顺序同资产合同）──

@configclass
class UprightActionsCfg:
    """腿位置控制 + 轮速度控制。

    **轮 scale 0.8（<1.0）**：论文 §C —— `s_v > 1.0` 时轮速动作对指令变化
    高度敏感，产生轮子振荡。倒立摆本就对轮速敏感，更需要小 scale。
    现有 locomotion 用 5.0 是行驶用途，直立必须降。
    """

    joint_pos = mdp_core.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp_core.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=0.8,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


# ── 观测（asymmetric）──

@configclass
class UprightObservationsCfg:
    """actor 42 维；critic 加 priv_*。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """部署可得观测，42 维。

        **含 base_lin_vel**（与 recovery 相反）：倒立摆时间常数仅 0.24 s，
        没有线速度反馈无法判断"正在往哪倒"，只靠角速度会滞后半个周期。
        轮速也不能替代——打滑或接触切换时轮速与机体速度解耦。
        """

        pitch_command = ObsTerm(
            func=mdp_core.generated_commands,
            params={"command_name": "pitch_command"},
        )
        projected_gravity = ObsTerm(
            func=mdp_core.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(
            func=mdp_core.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp_core.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos_legs = ObsTerm(
            func=upright_mdp.joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_legs = ObsTerm(
            func=upright_mdp.joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        wheel_vel = ObsTerm(
            func=upright_mdp.wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp_core.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """actor 全量（无噪声）+ priv_*。"""

        pitch_command = ObsTerm(
            func=mdp_core.generated_commands, params={"command_name": "pitch_command"}
        )
        projected_gravity = ObsTerm(
            func=mdp_core.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_lin_vel = ObsTerm(func=mdp_core.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp_core.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos_legs = ObsTerm(func=upright_mdp.joint_pos_legs, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel_legs = ObsTerm(func=upright_mdp.joint_vel_legs, params={"asset_cfg": SceneEntityCfg("robot")})
        wheel_vel = ObsTerm(func=upright_mdp.wheel_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp_core.last_action)

        priv_base_height = ObsTerm(
            func=upright_mdp.priv_base_height, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        priv_wheel_contact = ObsTerm(
            func=upright_mdp.priv_wheel_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_REGEX)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── 事件 ──

@configclass
class UprightEventsCfg:
    """启动期 DR + 双峰复位 + 推力扰动。"""

    randomize_friction = EventTerm(
        func=mdp_core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 本地声明真实轮胎-地量级；**不继承**现有 rough 的 8.1..27.0
            # （那是为抑制楼梯打滑逐次提高的调参结果，超真实值一个量级）
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    # 双峰复位：默认 quad_prob=0.0（先纯学平衡），课程后期提高以加入起立
    reset_pose = EventTerm(
        func=upright_mdp.reset_upright_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "quad_prob": 0.0,
            "collision_clearance": 0.002,
            "pitch_range": (1.45, 1.70),
            "leg_joint_noise": 0.1,
            "base_height": upright_mdp.UPRIGHT_BASE_HEIGHT,
        },
    )

    # 推力扰动：测试抗扰动平衡（PLAN §11.1 场景 4）
    push_robot = EventTerm(
        func=mdp_core.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)},
        },
    )


# ── 奖励 ──

@configclass
class UprightRewardsCfg:
    """当前奖励初值；横滚与能耗权重需要结合真实双后轮训练结果校准。"""

    # ── 任务奖励 ──
    pitch_tracking = RewTerm(
        func=upright_mdp.pitch_tracking,
        weight=3.0,  # 主驱动项
        params={"command_name": "pitch_command", "asset_cfg": SceneEntityCfg("robot"), "sigma": 0.25},
    )
    balance_hold = RewTerm(
        func=upright_mdp.balance_hold,
        weight=2.0,  # 离散"已稳住"信号，防高频振荡穿越目标角骗分
        params={"command_name": "pitch_command", "asset_cfg": SceneEntityCfg("robot")},
    )
    upright_height = RewTerm(
        func=upright_mdp.upright_height,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": upright_mdp.UPRIGHT_BASE_HEIGHT},
    )
    transit_speed = RewTerm(
        func=upright_mdp.transit_speed,
        weight=0.5,  # 在20°–60°实验过渡窗口内鼓励正向起立进展
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── 惩罚 ──
    roll_penalty = RewTerm(
        func=upright_mdp.roll_penalty,
        weight=-10.0,  # 限制横滚；此权重不代表已证明机械不可恢复
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    wheel_effort = RewTerm(
        func=upright_mdp.wheel_effort_penalty,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp_core.action_rate_l2, weight=-1e-2)
    joint_acc = RewTerm(
        func=mdp_core.joint_acc_l2,
        weight=-2.5e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    undesired_contacts = RewTerm(
        func=mdp_core.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACT_BODIES),
            "threshold": 1.0,
        },
    )


# ── 终止 ──

@configclass
class UprightTerminationsCfg:
    """timeout + 横滚早终止。

    早终止将平衡训练集中在当前姿态范围内；超限被定义为本任务失败，
    不等于该状态在机械上永远不可恢复，倒地恢复由独立任务训练。
    """

    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)

    # ⚠️ **不可用内置 `mdp_core.bad_orientation`**：它测机体 z 轴与世界竖直的
    # 总夹角（`acos(-g_z) > limit`）。直立 90° 时 g_z≈0 → acos(0)=1.57，
    # 任何 limit<1.57 都会在机器人刚站起来时立即终止 episode。
    # 故用只测横滚、与 pitch 解耦的自定义判据。
    excessive_roll = DoneTerm(
        func=upright_mdp.excessive_roll,
        params={"max_gy": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )

    # 塌回近水平即终止（避免在已注定失败的 rollout 上浪费样本）
    # ⚠️ 启用 quad_prob>0（从四轮起步训起立）时必须放宽或关闭此项
    # ⚠️ 前后阈值**不对称**（probe16 实测：对称判据留出 105°-148° 死区，
    #    策略会后仰躺在背上把 episode 跑满，base_link 触地率 60.3%）
    fell_over = DoneTerm(
        func=upright_mdp.fell_over,
        params={
            "max_gz_forward": 0.85,    # pitch < 32°  前塌
            # 后翻阈值 = 可保持区上界本身，**余量为 0**：初版留 15° 余量(0.5/120°)后，
            # probe19 实测策略 70% 的贴地样本正好落在 105°-120° 余量内、
            # 最大值顶在 120.0，且「接触 ∩ 终止」=0% —— 余量本身成了新作弊区。
            # 取 `BACKWARD_GZ_LIMIT`(= -cos(HOLDABLE_PITCH_MAX)) 而非字面量 0.26，
            # 避免同一边界的两套参数化（rad / g_z）各改一处后静默产生 0.22° 缝隙。
            "max_gz_backward": upright_mdp.BACKWARD_GZ_LIMIT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ── 主配置 ──

@configclass
class ParallelogramRoverUprightEnvCfg(ManagerBasedRLEnvCfg):
    """standalone 直立环境：10 s episode、50 Hz 控制、平地。"""

    scene: UprightSceneCfg = UprightSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: UprightObservationsCfg = UprightObservationsCfg()
    actions: UprightActionsCfg = UprightActionsCfg()
    commands: UprightCommandsCfg = UprightCommandsCfg()
    events: UprightEventsCfg = UprightEventsCfg()
    rewards: UprightRewardsCfg = UprightRewardsCfg()
    terminations: UprightTerminationsCfg = UprightTerminationsCfg()

    def __post_init__(self):
        # 50 Hz 控制（倒立摆时间常数 0.24 s → 每常数约 12 步，足够）
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # 4 s = 200 步。**由 L3 实测决定，非按目标反推**：
        #
        # 初版设 10 s（按 PLAN §6 的"平衡 ≥30 s"目标反推），但实测随机策略
        # episode 只有 **49 步（≈1 s）** 就触发 fell_over —— 500 步的预算
        # 永远用不到，利用率仅 9.8%（对照 recovery 是 100%）。
        # 更糟的是 `num_steps_per_env=24` 对 49 步 episode 意味着
        # **约半数 rollout 跨越 done 边界**，削弱 GAE 估计质量。
        #
        # 4 s 的依据：倒立摆时间常数 0.24 s，200 步 ≈ 17 个时间常数，
        # 足够容纳"学会稳住"的行为；且训练后期 episode 变长时利用率会上升。
        # PLAN §6 的 ≥30 s 是**评测指标**，评测时单独用长 episode 配置。
        self.episode_length_s = 4.0

        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
