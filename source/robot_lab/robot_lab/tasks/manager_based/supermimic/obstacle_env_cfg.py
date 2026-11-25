"""
obstacle_env_cfg.py — 从零 SuperMimic 越障任务：楼梯 ≥45%（立面 10→20 cm、踏面 0.26–0.40 m）、
单障碍 40%（全宽/左窄/右窄 6→20 cm）、平地 15%，按地形类型课程。

与 terrain_env_cfg（无推车、无观测噪声、扫描每物理步刷新）相比：
1. 地形与课程见 obstacle_terrain.py；成功截断 = 前进 3.6 m 且 |y| ≤ 0.15 m；|y| > 0.40 m 或到终点时 |y| > 0.15 m
   为出车道真终止；
2. 倾角终止 0.8 rad（与 PreviewRewardCfg.tilt_limit 一致）；
3. 关节软限位系数 0.95（只改本任务，资产默认 0.9 供其他任务保持原样）；
4. 去掉 PPO 时代的 anticipatory_wheel_lift 事件奖励（训练回报由 train_terrain 用 PreviewTask 重算，env 奖励只作日志）；
5. 回合 40 s：楼梯课程不再因超时吃亏，0.10 m/s 命令也有机会走完 3.6 m；
6. 命令 0.10–0.40 m/s，另有约 10% 静止命令（仅本任务；共享 PPO 配置不变）。父类的 UniformThresholdVelocityCommand
   会把 |v| ≤ 0.2 m/s 的命令清零（0.10–0.20 段会全部变成静止），这里换成 Isaac Lab 的 UniformVelocityCommand。
7. 停滞截断 stall_timeout（time_out）：stall_truncate_s > 0 且训练循环维护 env.obstacle_stall_steps 时生效。
观测/动作仍是 232 → 12，部署合同不变。

@author pony
@date 2026-11-05
@version v1.4.0
@last_modified 2026-11-23
@changelog
    - v1.4.0 (2026-11-23): column_explore（每列是否常驻探索）；add_short_stairs 从楼梯比例里拿出一部分做短踏面楼梯列
      （立面/踏面单独给、不用 tread_rule），插在楼梯之后并重算 column_kinds / column_explore。默认不加 = 原行为。
    - v1.3.0 (2026-11-10): 楼梯子地形 tread_rule 字段（默认 None）。
    - v1.2.0 (2026-11-09): 停滞截断终止项 stall_timeout（time_out；obstacle_curriculum_cfg["stall_truncate_s"]，默认 0 关闭，
      只有训练循环写入停滞步数时才会触发）。
    - v1.1.0 (2026-11-05): 回合 25→40 s；命令 0.10–0.40 m/s + 10% 静止（UniformVelocityCommand）；终止按横偏收紧。
    - v1.0.0 (2026-11-05): 初始创建
"""

from __future__ import annotations

from isaaclab.envs import mdp as mdp_core
from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview_env_cfg import (
    RoverSingleStepTerrainCfg,
)

from . import obstacle_terrain as ot
from .terrain_env_cfg import ParallelogramRoverSuperMimicTerrainEnvCfg

TILT_LIMIT_RAD = 0.8
SOFT_JOINT_LIMIT_FACTOR = 0.95
EPISODE_LENGTH_S = 40.0
COMMAND_X_RANGE = (0.10, 0.40)
STANDING_FRACTION = 0.10


@configclass
class ObstacleStairsTerrainCfg(SubTerrainBaseCfg):
    function = ot.straight_stairs_terrain
    riser_range: tuple[float, float] = ot.STAIR_RISER_M
    tread_range: tuple[float, float] = ot.STAIR_TREAD_M
    num_steps: int = ot.STAIR_STEPS
    first_edge_m: float = ot.STAIR_FIRST_EDGE_M
    lane_width: float = 3.0
    # 踏面下限随立面提高 (r0, r1, t1)，见 obstacle_terrain.min_tread_for_riser；None = 原行为
    tread_rule: tuple[float, float, float] | None = None


def obstacle_sub_terrains() -> dict:
    single = dict(step_height_range=ot.SINGLE_HEIGHT_M, step_front_x_range=ot.SINGLE_FRONT_X_M,
                  step_width=ot.SINGLE_LENGTH_M)
    narrow = dict(single, lane_width=ot.NARROW_WIDTH_M)
    return {
        "single_step": RoverSingleStepTerrainCfg(proportion=ot.MIX["single_step"], **single),
        "single_step_left": RoverSingleStepTerrainCfg(proportion=ot.MIX["single_step_left"],
                                                      lane_center_y=ot.NARROW_CENTER_Y_M, **narrow),
        "single_step_right": RoverSingleStepTerrainCfg(proportion=ot.MIX["single_step_right"],
                                                       lane_center_y=-ot.NARROW_CENTER_Y_M, **narrow),
        ot.STAIRS: ObstacleStairsTerrainCfg(proportion=ot.MIX[ot.STAIRS]),
        "flat": MeshPlaneTerrainCfg(proportion=ot.MIX["flat"]),
    }


def refresh_columns(env_cfg) -> None:
    """按当前 sub_terrains 重算每列的地形类型与是否常驻探索（改过子地形后调用）。"""
    terrain = env_cfg.scene.terrain.terrain_generator
    names, proportions = list(terrain.sub_terrains), [t.proportion for t in terrain.sub_terrains.values()]
    env_cfg.column_kinds = ot.column_kinds(names, proportions, terrain.num_cols)
    env_cfg.column_explore = ot.column_explore(names, proportions, terrain.num_cols)


def add_short_stairs(env_cfg, fraction: float, riser_range, tread_range) -> None:
    """从楼梯比例里拿出 fraction 做短踏面楼梯列（常驻探索：随机等级、不改课程、只记录），立面/踏面范围单独给，
    不用 tread_rule。须在楼梯的 tread_rule 设好之后调用（短踏面列不继承它）。"""
    terrain = env_cfg.scene.terrain.terrain_generator
    stairs = terrain.sub_terrains[ot.STAIRS]
    split = ot.split_short_stairs({name: t.proportion for name, t in terrain.sub_terrains.items()}, fraction)
    short = stairs.replace(proportion=split[ot.STAIRS_SHORT], riser_range=tuple(riser_range),
                           tread_range=tuple(tread_range), tread_rule=None)
    stairs.proportion = split[ot.STAIRS]
    subs = {}
    for name, sub in terrain.sub_terrains.items():
        subs[name] = sub
        if name == ot.STAIRS:
            subs[ot.STAIRS_SHORT] = short
    terrain.sub_terrains = subs
    refresh_columns(env_cfg)


@configclass
class ParallelogramRoverSuperMimicObstacleEnvCfg(ParallelogramRoverSuperMimicTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain.terrain_generator
        terrain.sub_terrains = obstacle_sub_terrains()
        terrain.num_rows, terrain.num_cols = ot.NUM_ROWS, ot.NUM_COLS
        terrain.difficulty_range, terrain.curriculum = (0.0, 1.0), True
        self.scene.terrain.max_init_terrain_level = 0
        refresh_columns(self)
        self.obstacle_curriculum_cfg = dict(ot.CURRICULUM_DEFAULTS)
        self.curriculum.terrain_levels.func = ot.obstacle_curriculum
        self.rewards.anticipatory_wheel_lift = None
        self.terminations.preview_success = None
        self.terminations.reached_goal = TerminationTermCfg(func=ot.reached_goal, time_out=True)
        self.terminations.out_of_lane = TerminationTermCfg(func=ot.out_of_lane)
        self.terminations.stall_timeout = TerminationTermCfg(func=ot.stall_timeout, time_out=True)
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp_core.bad_orientation, params={"limit_angle": TILT_LIMIT_RAD})
        self.scene.robot = self.scene.robot.replace(soft_joint_pos_limit_factor=SOFT_JOINT_LIMIT_FACTOR)
        self.episode_length_s = EPISODE_LENGTH_S
        command = self.commands.base_velocity
        self.commands.base_velocity = mdp_core.UniformVelocityCommandCfg(
            asset_name=command.asset_name, resampling_time_range=command.resampling_time_range,
            debug_vis=command.debug_vis, heading_command=False, rel_heading_envs=0.0,
            rel_standing_envs=STANDING_FRACTION,
            ranges=mdp_core.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=COMMAND_X_RANGE, lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        )
