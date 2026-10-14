"""独立的前瞻越障课程；保留旧 teacher232/student4141 任务及其语义。

@author pony
@date 2026-09-30
@version v0.3.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 同一任务支持8/12/16cm直楼梯、2/3/4级、接近段及顶端平台。
    - (2026-10-13): 接近速度收为0.2–0.4m/s，保留现有12项奖励与最终10cm验收。
    - (2026-10-12): Teacher、Student及critic统一使用车体系路线命令，维数及特权输入边界不变。
    - (2026-10-12): Preview显式选择12项有效奖励，删除无命令项、常驻正分及重复关节正则。
    - (2026-10-12): 单一Preview任务改三阶段路线课程，保留左右窄障碍、全宽、楼梯与方块。
    - (2026-10-11): 仅Preview替换静止仍可获正分的速度跟踪，不追加停滞计时器或改变观测。
    - (2026-10-11): 保持187点，重新分配前轮近场扫描密度，Teacher与蒸馏Teacher共用新语义。
    - v0.3.0 (2026-10-08): 提高单障碍高度和前后宽度，增加全宽障碍比例，让提前抬轮成为必要动作。
    - v0.2.0 (2026-10-08): 主动抬轮使用可reset的封顶事件奖励，保留单一任务入口。
    - v0.1.0 (2026-09-30): 新增单障碍、低台阶课程及前瞻抬轮 teacher/student。
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import trimesh

import isaaclab.envs.mdp as mdp_core
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass

from .vision_env_cfg import ParallelogramRoverVisionStudentTemporalEnvCfg, ParallelogramRoverVisionTeacherEnvCfg
from .vision_preview import (
    PREVIEW_SCAN_X, PREVIEW_SCAN_Y, PreviewCfg, near_field_height_pattern,
    stairs_profile, step_front_x_for_difficulty, step_lateral_bounds,
)
from .vision_preview_reward import (
    AnticipatoryWheelLift, preview_route_command, preview_route_curriculum, preview_route_success, preview_velocity_tracking,
    preview_yaw_rate_l2,
)


def single_step_terrain(difficulty: float, cfg):
    """沿 +X 路线放置单个矮横障碍，明确前缘以区分碰撞前和碰撞后抬轮。"""
    if not 0 <= difficulty <= 1 or not 0 < cfg.step_height_range[0] <= cfg.step_height_range[1]:
        raise ValueError("单障碍难度或高度范围无效")
    height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    lane_min, lane_max = step_lateral_bounds(cfg.lane_center_y, cfg.lane_width)
    if not (0 < cfg.step_front_x_range[0] < cfg.step_front_x_range[1] < cfg.size[0] / 2 - cfg.step_width
            and -cfg.size[1] / 2 <= lane_min < lane_max <= cfg.size[1] / 2 and cfg.step_width > 0):
        raise ValueError("单障碍必须位于地形内部且尺寸为正")
    front_x = step_front_x_for_difficulty(difficulty, cfg.step_front_x_range, cfg.seed)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    ground = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.10))
    ground.apply_translation(origin + np.array([0.0, 0.0, -0.05]))
    step = trimesh.creation.box(extents=(cfg.step_width, cfg.lane_width, height))
    step.apply_translation(origin + np.array([front_x + cfg.step_width / 2, cfg.lane_center_y, height / 2]))
    return [ground, step], origin


@configclass
class RoverSingleStepTerrainCfg(SubTerrainBaseCfg):
    function = single_step_terrain
    # 轮胎半径约10.85cm；8–12cm课程能减少直接滚上矮台阶的捷径，
    # 同时保留足够可达余量，不把训练变成纯碰撞失败。
    step_height_range: tuple[float, float] = (0.08, 0.12)
    step_front_x_range: tuple[float, float] = (1.20, 1.70)
    # 沿行驶方向的前后宽度；增加动作准备时间和真实越障约束。
    step_width: float = 0.40
    lane_width: float = 3.0
    lane_center_y: float = 0.0


def ascending_stairs_terrain(difficulty: float, cfg):
    """沿+X连续上楼，预留真实接近段；避免倒金字塔平台取整后出生就碰到首级。"""
    profile = stairs_profile(difficulty, cfg.height_levels, cfg.step_counts, cfg.first_edge_m, cfg.step_width)
    if (profile["target_distance_m"] + 0.50 >= cfg.size[0] / 2
            or not 0 < cfg.lane_width <= cfg.size[1]):
        raise ValueError("楼梯地形须容纳后轮登顶及终点平台")
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    ground = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.10))
    ground.apply_translation(origin + np.array([0., 0., -0.05]))
    meshes = [ground]
    # 逐层向终点延伸，顶端平台与最后一级同高；所有台阶均为真实碰撞几何。
    for k in range(profile["num_steps"]):
        edge = cfg.first_edge_m + k * cfg.step_width
        end = cfg.size[0] / 2
        step = trimesh.creation.box(extents=(end - edge, cfg.lane_width, profile["step_height_m"]))
        step.apply_translation(origin + np.array([
            (edge + end) / 2, 0., (k + 0.5) * profile["step_height_m"],
        ]))
        meshes.append(step)
    return meshes, origin


@configclass
class RoverStairsTerrainCfg(SubTerrainBaseCfg):
    function = ascending_stairs_terrain
    height_levels: tuple[float, ...] = (0.08, 0.12, 0.16)
    step_counts: tuple[int, ...] = (2, 3, 4)
    first_edge_m: float = 1.20
    step_width: float = 0.40
    lane_width: float = 3.0


@configclass
class PreviewHeightPatternCfg(PatternBaseCfg):
    func = near_field_height_pattern
    x_positions: tuple[float, ...] = PREVIEW_SCAN_X
    y_positions: tuple[float, ...] = PREVIEW_SCAN_Y


def configure_preview_rewards(rewards):
    """显式选择路线、稳定性与安全项，避免rough基类新增项悄悄改变Preview目标。"""
    weights = {
        "track_lin_vel_xy_exp": 5.0,
        "lin_vel_z_l2": -0.5,
        "ang_vel_xy_l2": -0.05,
        "joint_torques_l2": -2.5e-5,
        "joint_pos_limits": -5.0,
        "action_rate_l2": -0.005,
        "joint_deviation_steering_l1": -0.1,
        "undesired_contacts": -1.0,
        "contact_forces": -1.5e-4,
    }
    for name, term in vars(rewards).items():
        if term is not None and hasattr(term, "weight"):
            if name in weights:
                term.weight = weights[name]
            else:
                setattr(rewards, name, None)
    # 抬轮允许正常俯仰/侧倾；弱姿态负项只抑制失稳，不奖励停着保持正立。
    rewards.flat_orientation_l2 = RewardTermCfg(func=mdp_core.flat_orientation_l2, weight=-0.1)
    rewards.preview_yaw_rate_l2 = RewardTermCfg(func=preview_yaw_rate_l2, weight=-0.05)
    rewards.anticipatory_wheel_lift = RewardTermCfg(func=AnticipatoryWheelLift, weight=2.0)
    rewards.track_lin_vel_xy_exp.func = preview_velocity_tracking


def configure_preview_observations(observations):
    """所有控制/价值组共享路线语义，蒸馏不能只改Student而遗漏Teacher输入。"""
    for name in ("policy", "critic", "teacher"):
        group = getattr(observations, name, None)
        if group is not None:
            group.velocity_commands.func = preview_route_command


def configure_preview(env_cfg, *, student: bool):
    """只修改新任务实例；两阶段共用课程与奖励，student 不获得特权地形输入。"""
    env_cfg.vision_preview = asdict(PreviewCfg())
    env_cfg.vision_preview_objective_version = "active_wheel_prelift"
    env_cfg.scene.height_scanner = env_cfg.scene.height_scanner.copy()
    env_cfg.scene.height_scanner.offset.pos = (0.30, 0.0, 20.0)
    env_cfg.scene.height_scanner.pattern_cfg = PreviewHeightPatternCfg()
    terrain = env_cfg.scene.terrain.terrain_generator.copy()
    stairs = RoverStairsTerrainCfg(proportion=6 / 24)
    boxes = terrain.sub_terrains["boxes"].copy()
    boxes.proportion = 3 / 24
    boxes.grid_height_range = (0.06, 0.12)
    terrain.sub_terrains = {
        # 全部列预生成；reset按路线晋级阶段重抽列，保留难度和地形种类两个轴。
        "single_step": RoverSingleStepTerrainCfg(proportion=5 / 24),
        "single_step_left": RoverSingleStepTerrainCfg(
            proportion=4 / 24, lane_width=0.36, lane_center_y=0.30,
        ),
        "single_step_right": RoverSingleStepTerrainCfg(
            proportion=4 / 24, lane_width=0.36, lane_center_y=-0.30,
        ),
        "pyramid_stairs_inv": stairs,
        "boxes": boxes,
        "flat": MeshPlaneTerrainCfg(proportion=2 / 24),
    }
    terrain.num_rows = 9
    terrain.num_cols = 24
    env_cfg.scene.terrain.terrain_generator = terrain
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum.terrain_levels.func = preview_route_curriculum
    # 旧课程的 ±0.5m 随机起点可能让前轮出生时已越过单障碍前缘。
    env_cfg.events.randomize_reset_base.params["pose_range"].update(
        x=(-0.15, 0.0), y=(-0.10, 0.10), yaw=(-0.10, 0.10),
    )
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.40)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    configure_preview_observations(env_cfg.observations)
    configure_preview_rewards(env_cfg.rewards)
    env_cfg.terminations.preview_success = TerminationTermCfg(func=preview_route_success, time_out=True)
    if student:
        # 25度俯视使低矮障碍更早进入有效裁剪区；相机改变必须随新深度合同保存。
        env_cfg.scene.depth_camera.offset.rot = (0.9762960071, 0.0, 0.2164396139, 0.0)
    else:
        env_cfg.observations.policy.height_scan.noise.n_min = -0.02
        env_cfg.observations.policy.height_scan.noise.n_max = 0.02


@configclass
class ParallelogramRoverVisionTeacherPreviewEnvCfg(ParallelogramRoverVisionTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_preview(self, student=False)


@configclass
class ParallelogramRoverVisionStudentPreviewEnvCfg(ParallelogramRoverVisionStudentTemporalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_preview(self, student=True)
