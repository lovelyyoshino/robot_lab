# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.0
# @last_modified 2026-09-17
# @changelog
#   - v0.2.0 (2026-09-17): 更正接触传感器索引误标解释，保留旧基线的 base_link 惩罚。
#   - v0.1.0 (2026-08-25): 建立 standalone 恢复环境；权重按 10 N·m 下调，摩擦本地声明。

"""平行四边形轮足车倒地恢复环境（EDRS method1）—— **standalone，不继承 velocity 任务族**。

参考：`Kitjesen/recovery` 的 `thunder_recovery/config/method1_deng/env_cfg.py`
（Deng et al., arXiv:2506.05516）。

**独立性（PLAN §3.0-A 硬规则）**：
  - 从 `ManagerBasedRLEnvCfg` 直接继承，**不继承** `LocomotionVelocityRoughEnvCfg`
  - 现有 `rough_env_cfg.py` / `flat_env_cfg.py` 一行不改，仅**只读引用**
    `PARALLELOGRAM_ROVER_CFG`（机器人本体定义）
  - 摩擦、地形、奖励、终止全部本地声明
  - 理由：velocity 基类强加的命令管理器/locomotion 奖励/课程/终止对恢复任务全是错的；
    且现有 locomotion 已训通（`rough_env_cfg.py` 迭代至 v0.3.13），任何改动风险不对称

**设计要点**：
  - 场景：平地（恢复不用粗糙地形；台阶/斜坡留作进阶，见 PLAN §4 P2 地形课程）
  - 无命令（恢复是 command-free 任务）
  - 无课程（论文不用）
  - 终止：仅 timeout（恢复不提前结束——提前终止会截断正在进行的翻正动作）
  - 观测：asymmetric（actor 58 维本体感受；critic 加 priv_*）
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

# 只读引用机器人本体定义（遵守 §3.0-A：不修改该文件）
from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_CFG,
    PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
)
from robot_lab.tasks.manager_based.recovery import mdp as recovery_mdp


# ── 关节/体命名（全部来自资产常量，不硬编码字符串）──

# 8 个主动腿关节：顺序 = (fl,rl,fr,rr)_arm + (fl,rl,fr,rr)_qudong
LEG_JOINT_NAMES = [
    *PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
]
WHEEL_JOINT_NAMES = list(PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES)

BASE_LINK_NAME = "base_link"
WHEEL_BODY_REGEX = r".*_wheel$"
"""轮体正则。URDF 中轮体名为 fl_wheel/rl_wheel/fr_wheel/rr_wheel。"""

BODY_COLLISION_BODIES = [BASE_LINK_NAME]
"""`body_collision` 惩罚的 body —— **只含 base_link**。

保留历史 EDRS 基线的数值选择；它不等于最终恢复成功门。
旧探针误用 articulation.body_names 标注传感器 force 索引，导致将真实轮体
接触写成转向/驱动/连杆承重；不能据此认定 PhysX 合并了接触归属。
Grounded 的严格四轮支撑另由 sensor.body_names 对齐，并排除所有非轮承重。
"""


# ── 场景 ──

@configclass
class RecoverySceneCfg(InteractiveSceneCfg):
    """平地 + 机器人 + 接触传感器。刻意保持最小。

    **不含 height_scanner**：恢复是本体感受 + 接触任务（probe2 证实靠 arm+轮联动完成），
    地形高度信息对它无用；且省下渲染开销。
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

    # 只读引用本体定义 + 覆写 prim_path 以获得 per-env 路径
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


# ── 动作（12-D，顺序 = PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES）──

@configclass
class RecoveryActionsCfg:
    """腿位置控制 + 轮速度控制，共 12 维。

    **动作顺序**：`joint_pos`(8: arm×4 + qudong×4) 在前，`joint_vel`(4: wheel×4) 在后，
    配合 `preserve_order=True`，得到与 `PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES`
    一致的 12-D 顺序 —— 即 idx 0-3 arm、4-7 qudong、8-11 wheel，corner 序为 (fl,rl,fr,rr)。

    **scale 依据**：
      - arm 0.25：±35° 全行程约 0.61 rad，scale 0.25 使单步动作幅度合理
      - qudong 0.25：转向同量级
      - wheel **0.8（<1.0）**：论文 §C 明确 —— `s_v > 1.0` 时轮速动作对指令变化
        高度敏感，产生明显轮子振荡。现有 locomotion 用 5.0 是行驶用途，恢复必须降下来。
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


# ── 观测（asymmetric actor-critic）──

@configclass
class RecoveryObservationsCfg:
    """actor 58 维本体感受；critic 同源干净版 + priv_* 特权。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """部署可得观测，58 维。

        **不含 base_lin_vel** —— 参考实现明确：倒地后 IMU 积分漂移过于严重，
        线速度不可用。这是与 locomotion 观测的关键差异（后者 critic 有线速度）。

        噪声量级取自 InstinctMJ perceptive 配置的分项噪声（比参考实现更细）。
        """

        projected_gravity = ObsTerm(
            func=mdp_core.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(
            func=mdp_core.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos_legs = ObsTerm(
            func=recovery_mdp.joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_legs = ObsTerm(
            func=recovery_mdp.joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        wheel_vel = ObsTerm(
            func=recovery_mdp.wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        prev_joint_pos_legs = ObsTerm(
            func=recovery_mdp.prev_joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        prev_joint_vel_legs = ObsTerm(
            func=recovery_mdp.prev_joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        prev_wheel_vel = ObsTerm(
            func=recovery_mdp.prev_wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        last_action = ObsTerm(func=mdp_core.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """特权观测：actor 全量（**无噪声**）+ priv_*。

        critic 用真值加速价值估计；actor 只看部署可得量。这与现有 locomotion
        的 asymmetric 模式一致（`velocity_env_cfg.py` critic 含 height_scan、policy 不含）。
        """

        projected_gravity = ObsTerm(
            func=mdp_core.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel = ObsTerm(func=mdp_core.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos_legs = ObsTerm(func=recovery_mdp.joint_pos_legs, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel_legs = ObsTerm(func=recovery_mdp.joint_vel_legs, params={"asset_cfg": SceneEntityCfg("robot")})
        wheel_vel = ObsTerm(func=recovery_mdp.wheel_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        prev_joint_pos_legs = ObsTerm(
            func=recovery_mdp.prev_joint_pos_legs, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        prev_joint_vel_legs = ObsTerm(
            func=recovery_mdp.prev_joint_vel_legs, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        prev_wheel_vel = ObsTerm(func=recovery_mdp.prev_wheel_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp_core.last_action)

        # 特权信号
        priv_base_height = ObsTerm(
            func=recovery_mdp.priv_base_height, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        priv_base_lin_vel = ObsTerm(
            func=recovery_mdp.priv_base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        priv_base_ang_vel = ObsTerm(
            func=recovery_mdp.priv_base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        priv_wheel_contact = ObsTerm(
            func=recovery_mdp.priv_wheel_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_REGEX)},
        )
        priv_body_contact_force = ObsTerm(
            func=recovery_mdp.priv_body_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODY_COLLISION_BODIES)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── 事件 ──

@configclass
class RecoveryEventsCfg:
    """启动期域随机化 + 复位（加权跌倒姿态）+ 自由落体零力矩维持。"""

    # startup：摩擦本地声明为真实量级
    randomize_friction = EventTerm(
        func=mdp_core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # ⚠️ 现有 rough 配置用 8.1..27.0（为抑制楼梯打滑逐次提高的调参结果）。
            # 那超出真实轮胎-地系数一个量级；恢复任务必须用真实值，否则学到的
            # 翻正策略依赖不存在的抓地力。本地声明 0.3..1.0，不继承、不修改现有配置。
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    # startup：机体质量随机化（±10%，参考实现同量级）
    randomize_base_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # reset：加权跌倒姿态（PLAN §11.2 真实跌倒分布）
    reset_fall_pose = EventTerm(
        func=recovery_mdp.reset_with_fall_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "drop_height": 1.1,
            "leg_joint_noise": 0.3,
            "pose_weights": recovery_mdp.DEFAULT_FALL_POSE_WEIGHTS,
        },
    )

    # reset：清 prev_* 观测缓存（**必须**，否则首步读到上一 episode 末帧）
    reset_obs_cache = EventTerm(
        func=recovery_mdp.reset_prev_obs_cache,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # reset：立刻零力矩（使首个物理步就处于自由落体）
    # reset：立即零力矩，使**首个物理步**就已是自由落体状态
    # （只绑 interval 不够：interval 在首步之后才触发，首步会带着正常增益产生一次
    #  非预期的 PD 力矩，破坏"完全松弛塌落"的初始条件）
    freefall_init = EventTerm(
        func=recovery_mdp.zero_action_freefall,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # interval：每步维持零力矩直到退出窗口（内部按 step_count 判断）
    freefall_hold = EventTerm(
        func=recovery_mdp.zero_action_freefall,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # 每控制步
        is_global_time=False,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ── 奖励（13 项：3 任务×ED + 2 支持/协同 + 2 行为×CW + 5 常驻 + 1 日志）──

@configclass
class RecoveryRewardsCfg:
    """EDRS 奖励集，权重按本机器人 10 N·m 执行器下调。

    **权重来源与调整链**：
      论文(KYON) → 参考实现(Thunder, 已降) → 本实现(平行四边形, 再降)

    参考实现 README 明确："论文权重是在 KYON 上调的，对 Thunder 过度惩罚"，
    其把 `joint_velocity` 从 -2e-2 降到 -3e-3、`body_collision` 从 -5e-2 降到 -1e-2。
    本机器人腿部执行器 **10 N·m vs Thunder 120 N·m（差 12 倍）**，故行为/常驻惩罚
    再降一个量级。

    ⚠️ 这些权重是**待 L3 短训校准的初值**，不是最终值。校准判据见 PLAN §5 L3。
    """

    # 副作用项：推进 ED 时钟。权重 1e-6（微小但非零——某些 Isaac Lab 版本会剪除权重为 0 的项）
    step_counter = RewTerm(func=recovery_mdp.recovery_step_counter, weight=1e-6)

    # ── 任务奖励（× ED，后期主导）──
    base_height = RewTerm(
        func=recovery_mdp.recovery_base_height,
        weight=120.0,  # 主驱动项，与参考实现同量级
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": recovery_mdp.NOMINAL_BASE_HEIGHT},
    )
    base_orientation = RewTerm(
        func=recovery_mdp.recovery_base_orientation,
        weight=50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stand_joint_pos = RewTerm(
        func=recovery_mdp.recovery_stand_joint_pos,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── 支持状态与协同 ──
    # 旧基线保留姿态(gz<-0.7) + 高度(>0.26)奖励，不表示真实四轮支撑。
    # 旧探针误用 articulation 的体顺序，不能据此断言四轮接触恒为 False。
    # Grounded 的严格支撑另按 sensor.body_names 识别四轮并排除非轮承重。
    support_state = RewTerm(
        func=recovery_mdp.recovery_support_state,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_base_height": recovery_mdp.MIN_BASE_HEIGHT,
            "max_gz": recovery_mdp.UPRIGHT_GZ_THRESHOLD,
        },
    )
    arm_wheel_coord = RewTerm(
        func=recovery_mdp.recovery_arm_wheel_coord,
        weight=0.5,  # probe2：翻正主力是 arm 摆重心，轮转辅助
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── 行为惩罚（× CW，早期约束、随迭代衰减）──
    body_collision = RewTerm(
        func=recovery_mdp.recovery_body_collision,
        weight=-1e-3,  # 参考 -1e-2 → 本机再降（弱执行器不应因大幅动作被重罚）
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BODY_COLLISION_BODIES)},
    )
    action_rate = RewTerm(func=recovery_mdp.recovery_action_rate, weight=-1e-3)

    # ── 常驻惩罚（自由落体后恒生效）──
    joint_deviation = RewTerm(
        func=recovery_mdp.recovery_joint_deviation,
        weight=-2e-3,  # 参考实现额外加的项：提供 ED≈0 早期的"朝站姿走"梯度
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_velocity = RewTerm(
        func=recovery_mdp.recovery_joint_velocity,
        weight=-1e-3,  # ⚠️ 论文 -2e-2 会逼出"不敢动"局部最优；参考降至 -3e-3，本机 -1e-3
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acceleration = RewTerm(
        func=recovery_mdp.recovery_joint_acceleration,
        weight=-2.5e-8,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torques = RewTerm(
        func=recovery_mdp.recovery_torques,
        weight=-5e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    wheel_velocity = RewTerm(
        func=recovery_mdp.recovery_wheel_velocity,
        weight=-1e-3,  # ⚠️ 必须小：轮转是 probe2 验证的唯一可行翻正手段之一
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── 日志项（权重 1e-6，不影响优化）──
    success_rate = RewTerm(
        func=recovery_mdp.recovery_success_rate,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ── 终止 ──

@configclass
class RecoveryTerminationsCfg:
    """**仅 timeout**。

    恢复任务不提前终止：提前结束会截断正在进行的翻正动作，使策略无法学到
    "先大幅翻滚、再精细站立"的两段式行为（这正是 ED 整形要鼓励的）。
    参考实现同样只用 timeout（其注释："recovery never ends early"）。

    特别地**不设 illegal_contact**：恢复过程中机体/腿必然大量触地，
    那是任务的一部分而非失败。非期望接触已由 `body_collision` 惩罚项处理。
    """

    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)


# ── 主环境配置 ──

@configclass
class ParallelogramRoverRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """standalone 恢复环境：5 s episode、50 Hz 控制、平地。"""

    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=4096, env_spacing=2.5)

    observations: RecoveryObservationsCfg = RecoveryObservationsCfg()
    actions: RecoveryActionsCfg = RecoveryActionsCfg()
    events: RecoveryEventsCfg = RecoveryEventsCfg()
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
    terminations: RecoveryTerminationsCfg = RecoveryTerminationsCfg()

    # 无 commands（恢复是 command-free）、无 curriculum（论文不用）

    def __post_init__(self):
        # 50 Hz 控制 = decimation 4 × 0.005 s 物理步
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # 5 s episode = 250 控制步。前 100 步（2 s）是自由落体窗口，
        # 剩余 150 步用于翻正 —— 与 FREEFALL_STEPS/ED 时间线一致。
        self.episode_length_s = 5.0

        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
