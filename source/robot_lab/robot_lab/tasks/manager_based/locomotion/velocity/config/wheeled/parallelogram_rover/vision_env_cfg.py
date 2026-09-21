# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.3.1
# @last_modified 2026-09-17
# @changelog
#   - v0.3.1 (2026-09-17): 拒绝相机刷新周期与 policy 控制周期不一致，防止时序深度合同将重复/漏采样当作新帧。
#   - v0.3.0 (2026-09-16): 新增独立 Temporal student 变体和实时计算的部署预处理元数据，保留原合同。
#   - v0.2.0 (2026-09-16): 视觉任务独立使用常规摩擦范围、加入窄沟课程，并排除已知接触归属歧义。
#   - v0.1.0 (2026-09-16): 新增 height-scan teacher 与 depth-history student 环境。

"""平行四边形轮足车视觉行走的 teacher/student 环境。

V1 teacher 使用 187 点 height scan；V2 student 使用 32x16 深度有效区域的 8 帧历史。
两者共享动作、奖励、地形和 teacher 观测合同，student 因而可由 RSL-RL 原生蒸馏器训练。
当前 RSL-RL 3.1.2 的 ``StudentTeacher`` 只接受一维观测，所以第一版将深度展平；
CNN 版本必须配套自定义 runner，不能仅靠环境配置宣称已经支持。
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp_core
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import MeshGapTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg, ObservationsCfg

from . import mdp as rover_mdp
from .rough_env_cfg import ParallelogramRoverRoughEnvCfg
from .vision_mdp import TemporalDepthObservation, normalized_depth_image
from .vision_temporal import DepthTemporalCfg


def _configure_vision_terrain(env_cfg) -> None:
    """teacher/student 使用同一组地形与动力学；保留普通 blind 任务的已有训练配置。"""
    terrain = env_cfg.scene.terrain.terrain_generator
    terrain.sub_terrains["pyramid_stairs"].proportion = 0.15
    terrain.sub_terrains["pyramid_stairs_inv"].proportion = 0.15
    terrain.sub_terrains["gaps"] = MeshGapTerrainCfg(
        proportion=0.1, gap_width_range=(0.05, 0.20), platform_width=3.0,
    )
    # 这是待实车标定的初始范围，不能沿用 blind 任务 8.1..27 的摩擦补偿。
    material = env_cfg.events.randomize_wheel_rigid_body_material.params
    material["static_friction_range"] = (0.5, 1.2)
    material["dynamic_friction_range"] = (0.4, 1.0)
    material["restitution_range"] = (0.0, 0.1)
    # 本配置只惩罚底盘触地；索引必须来自接触传感器，不能复用 articulation 的 body_ids。
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link"]


def _height_scan_term(*, noisy: bool) -> ObsTerm:
    """构造 teacher 的特权地形观测，确保训练任务与蒸馏任务合同一致。"""
    return ObsTerm(
        func=mdp_core.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1) if noisy else None,
        clip=(-1.0, 1.0),
        scale=1.0,
    )


def _configure_rover_teacher(group, env_cfg: ParallelogramRoverRoughEnvCfg) -> None:
    """把通用 velocity policy 收紧为与 rover teacher 完全一致的主动关节布局。"""
    wheel_cfg = SceneEntityCfg("robot", joint_names=env_cfg.wheel_joint_names, preserve_order=True)
    group.base_lin_vel = None
    group.base_ang_vel.scale = 0.25
    group.joint_pos.func = rover_mdp.joint_pos_rel_without_wheel
    group.joint_pos.params["asset_cfg"].joint_names = env_cfg.policy_joint_names
    group.joint_pos.params["asset_cfg"].preserve_order = True
    group.joint_pos.params["wheel_asset_cfg"] = wheel_cfg
    group.joint_pos.scale = 1.0
    group.joint_vel.params["asset_cfg"].joint_names = env_cfg.policy_joint_names
    group.joint_vel.params["asset_cfg"].preserve_order = True
    group.joint_vel.scale = 0.05
    group.height_scan = _height_scan_term(noisy=False)
    group.enable_corruption = False
    group.concatenate_terms = True


def _validate_depth_camera_timing(env_cfg) -> None:
    """深度历史以 policy 步为采样单位，相机必须每个控制步产生一帧。

    Temporal 缓存会在每个 policy 步推进；若相机周期不同，重复渲染或跳过渲染会被
    静默当作新的 raw frame，从而破坏导出的时间跨度和延迟合同。连续历史同样依赖
    相同假设，故在两个 student 变体上统一拒绝此类 Hydra 覆盖。
    """
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    camera_dt = env_cfg.scene.depth_camera.update_period
    if (
        not math.isfinite(control_dt)
        or not math.isfinite(camera_dt)
        or control_dt <= 0.0
        or camera_dt <= 0.0
        or not math.isclose(camera_dt, control_dt, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise ValueError(
            "depth_camera.update_period 必须等于 policy 控制周期 sim.dt * decimation；"
            "否则深度历史的帧间隔、延迟和部署合同不成立"
        )


@configclass
class ParallelogramRoverVisionSceneCfg(MySceneCfg):
    """在 base_link 前缘挂载低分辨率深度相机。"""

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/depth_camera",
        update_period=0.02,
        height=20,
        width=32,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=2.5,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.5),
        ),
        # world convention 的 +X 为视线方向；绕 +Y 约 15 度使视野覆盖近处台阶。
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.08),
            rot=(0.991445, 0.0, 0.130526, 0.0),
            convention="world",
        ),
    )


@configclass
class VisionStudentPolicyCfg(ObservationsCfg.PolicyCfg):
    """45 维本体观测加 8 帧深度历史；不向 student 暴露 height scan。"""

    depth_image = ObsTerm(
        func=normalized_depth_image,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "distance_to_image_plane",
            "min_depth": 0.1,
            "max_depth": 2.5,
            "crop_top": 2,
            "crop_bottom": 2,
        },
        noise=Unoise(n_min=-0.02, n_max=0.02),
        clip=(-1.0, 1.0),
        history_length=8,
        flatten_history_dim=True,
    )


@configclass
class VisionStudentObservationsCfg(ObservationsCfg):
    """蒸馏所需 student、teacher 与训练日志 critic 三组观测。"""

    policy: VisionStudentPolicyCfg = VisionStudentPolicyCfg()
    teacher: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()


@configclass
class ParallelogramRoverVisionTeacherEnvCfg(ParallelogramRoverRoughEnvCfg):
    """V1：将 height scan 加入 actor，训练可通过障碍的特权 teacher。"""

    def __post_init__(self):
        super().__post_init__()
        _configure_vision_terrain(self)
        self.observations.policy.height_scan = _height_scan_term(noisy=True)
        # rough 基类仅对精确类名清理零权重项；子类必须显式执行，否则空 body selector 会解析失败。
        self.disable_zero_weight_rewards()


@configclass
class ParallelogramRoverVisionStudentEnvCfg(ParallelogramRoverRoughEnvCfg):
    """V2：student 读深度历史，teacher 读同一时刻的 height scan。"""

    scene: ParallelogramRoverVisionSceneCfg = ParallelogramRoverVisionSceneCfg(num_envs=512, env_spacing=2.5)
    observations: VisionStudentObservationsCfg = VisionStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_vision_terrain(self)
        _configure_rover_teacher(self.observations.teacher, self)
        _validate_depth_camera_timing(self)
        self.num_rerenders_on_reset = 2
        self.disable_zero_weight_rewards()


@configclass
class ParallelogramRoverVisionStudentTemporalEnvCfg(ParallelogramRoverVisionStudentEnvCfg):
    """时序增强 student，仍为 4141 维，但必须配套本变体的预处理及独立 checkpoint。

    stride=5 时 8 帧覆盖 0.7 秒。教师的 232 维合同、地形和动力学沿用原 student。
    原 student 的每步连续历史与本配置语义不同，不能仅按网络输入维度选择部署器。
    """

    def __post_init__(self):
        super().__post_init__()
        term = self.observations.policy.depth_image
        term.func = TemporalDepthObservation
        term.history_length = 0
        term.noise = None
        term.params.update(
            history_length=8,
            history_stride=5,
            delay_steps=(0, 1),
            drop_probability=0.05,
            freeze_probability=0.01,
            freeze_steps=(2, 5),
            noise_half_range=0.02,
        )


def vision_depth_contract(env_cfg: ParallelogramRoverVisionStudentEnvCfg) -> dict:
    """从最终有效配置生成 JSON 兼容的部署合同，调用方应与 checkpoint 一起导出。

    不在 __post_init__ 缓存，避免 Hydra 覆盖参数后元数据仍描述旧预处理。
    本函数描述配置合同，不宣称相机、模型或 MuJoCo 部署已经经过验收。
    """
    term = env_cfg.observations.policy.depth_image
    camera = env_cfg.scene.depth_camera
    params = term.params
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    _validate_depth_camera_timing(env_cfg)
    if getattr(env_cfg.observations.policy, "history_length", None) is not None:
        raise ValueError("视觉 student 只允许深度项有历史，不能额外叠加整组 policy 历史")
    height = camera.height - params["crop_top"] - params["crop_bottom"]
    if height <= 0 or camera.width <= 0:
        raise ValueError("深度有效区域不能为空")
    if term.func is TemporalDepthObservation:
        defaults = DepthTemporalCfg()
        temporal = DepthTemporalCfg(**{
            name: params.get(name, getattr(defaults, name))
            for name in DepthTemporalCfg.__dataclass_fields__
        })
        if term.history_length != 0 or term.noise is not None:
            raise ValueError("Temporal student 不允许额外的 manager 历史或噪声")
        contract = temporal.contract(control_dt)
        history_length = temporal.history_length
    elif term.func is normalized_depth_image:
        history_length = term.history_length
        contract = {
            "contract_id": "rover_depth_contiguous_v1",
            "history_owner": "isaaclab.observation_manager",
            "history_order": "oldest_to_newest",
            "history_fill": "repeat_first_received_frame_after_reset",
            "control_dt_s": control_dt,
            "history_span_s": (history_length - 1) * control_dt,
            "frame_age_steps_oldest_to_newest": list(range(history_length - 1, -1, -1)),
            "noise_uniform_range": [term.noise.n_min, term.noise.n_max] if term.noise is not None else None,
        }
    else:
        raise ValueError("未知深度观测函数，拒绝导出可能不匹配的部署合同")
    contract.update({
        "student_proprio_dim": 45,
        "student_obs_dim": 45 + history_length * height * camera.width,
        "teacher_obs_dim": 232,
        "action_dim": 12,
        "camera": {
            "raw_height": camera.height,
            "raw_width": camera.width,
            "update_period_s": camera.update_period,
            "offset_pos_m": list(camera.offset.pos),
            "offset_rot_wxyz": list(camera.offset.rot),
            "offset_convention": camera.offset.convention,
            "focal_length": camera.spawn.focal_length,
            "horizontal_aperture": camera.spawn.horizontal_aperture,
            "clipping_range_m": list(camera.spawn.clipping_range),
        },
        "preprocessing": {
            "data_type": params["data_type"],
            "crop_top": params["crop_top"],
            "crop_bottom": params["crop_bottom"],
            "effective_shape_hw": [height, camera.width],
            "depth_range_m": [params["min_depth"], params["max_depth"]],
            "normalized_range": [-1.0, 1.0],
            "nan_and_positive_inf": "max_depth",
            "negative_inf": "min_depth",
            "flatten_order": "history_then_row_then_column",
        },
        "corruption_enabled": env_cfg.observations.policy.enable_corruption,
        "requires_observation_update_every_policy_step": term.func is TemporalDepthObservation,
    })
    return contract
