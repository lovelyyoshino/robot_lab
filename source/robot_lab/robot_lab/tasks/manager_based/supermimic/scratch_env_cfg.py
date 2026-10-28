"""
scratch_env_cfg.py — 方案 D：不用 Teacher、从零用 SuperMimic 训练楼梯的任务（参考 go2w-supermimic 的配方）。

与越障微调任务（terrain_env_cfg）的区别：
1. 课程从 2 cm 起步：9 行台阶高度 2→16 cm 连续变化（4 级、40 cm 踏面，几何与验收楼梯一致），从零学必须先学会矮台阶；
2. 晋级只看"往前越过终点、没翻、没出车道"，不依赖 Preview 的抬轮事件状态机（那是给 PPO 的稀疏奖励，想象里复现不了）；
3. 训练用的奖励由 train_terrain.py 在真实状态上用 PreviewTask 重算，env 自带 RewardManager 只作日志，
   这样 critic 拟合的真实回报与 actor 在想象里优化的目标是同一个函数（go2w 的做法）。
观测/动作与 Teacher-Preview 相同（232 → 12），训练产物沿用现有 play / 验收。

@author pony
@date 2026-10-28
@version v1.0.0
@last_modified 2026-10-28
@changelog
    - v1.0.0 (2026-10-28): 初始创建
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview_env_cfg import (
    RoverStairsTerrainCfg,
)

from .terrain_env_cfg import ParallelogramRoverSuperMimicTerrainEnvCfg

SCRATCH_HEIGHT_RANGE = (0.02, 0.16)
SCRATCH_STEPS = 4
FIRST_EDGE_M = 1.20
STEP_WIDTH_M = 0.40
# 终点 = 最后一级前缘 + 0.6 m（后轮登顶的余量），与验收协议的 target_distance_m 一致
TARGET_M = FIRST_EDGE_M + (SCRATCH_STEPS - 1) * STEP_WIDTH_M + 0.60
# 训练期车道放宽到 0.5 m（验收是 0.15 m）：从零阶段先学会爬，横向精度交给奖励塑形
TRAIN_LANE_M = 0.5
FAIL_LANE_M = 1.0


def scratch_step_height(difficulty: float) -> float:
    lo, hi = SCRATCH_HEIGHT_RANGE
    return lo + float(np.clip(difficulty, 0.0, 1.0)) * (hi - lo)


def scratch_stairs_terrain(difficulty: float, cfg):
    """沿 +X 连续上楼，4 级；台阶高度随难度在 2–16 cm 间连续变化。"""
    height = scratch_step_height(difficulty)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    ground = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.10))
    ground.apply_translation(origin + np.array([0.0, 0.0, -0.05]))
    meshes = [ground]
    end = cfg.size[0] / 2
    for k in range(SCRATCH_STEPS):
        edge = FIRST_EDGE_M + k * STEP_WIDTH_M
        step = trimesh.creation.box(extents=(end - edge, cfg.lane_width, height))
        step.apply_translation(origin + np.array([(edge + end) / 2, 0.0, (k + 0.5) * height]))
        meshes.append(step)
    return meshes, origin


@configclass
class ScratchStairsTerrainCfg(RoverStairsTerrainCfg):
    # 保留 RoverStairsTerrainCfg 的字段，Preview 的抬轮诊断代码按名字 pyramid_stairs_inv 识别楼梯仍可用
    function = scratch_stairs_terrain
    step_counts: tuple[int, ...] = (SCRATCH_STEPS,) * 3


def _offsets(env, ids=None):
    robot = env.scene["robot"]
    rel = robot.data.root_pos_w - env.scene.env_origins
    return rel if ids is None else rel[ids]


def scratch_curriculum(env, env_ids):
    """越过终点、直立、在 0.5 m 车道内 → 升一级；翻车 / 出车道 / 几乎没往前走 → 降一级。"""
    terrain = env.scene.terrain
    device = terrain.terrain_levels.device
    ids = torch.arange(env.num_envs, device=device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(
        env_ids, device=device, dtype=torch.long)
    if ids.numel():
        rel = _offsets(env, ids)
        upright = -env.scene["robot"].data.projected_gravity_b[ids, 2] >= 0.85
        active = env.episode_length_buf[ids] > 0
        passed = active & upright & (rel[:, 0] >= TARGET_M) & (rel[:, 1].abs() <= TRAIN_LANE_M)
        failed = active & ~passed & (~upright | (rel[:, 1].abs() > FAIL_LANE_M) | (rel[:, 0] < 0.8))
        levels = (terrain.terrain_levels[ids] + passed.long() - failed.long()).clamp(0, terrain.max_terrain_level - 1)
        terrain.terrain_levels[ids] = levels
        terrain.env_origins[ids] = terrain.terrain_origins[levels, terrain.terrain_types[ids]]
    levels = terrain.terrain_levels.float()
    rows = terrain.cfg.terrain_generator.num_rows
    lo, hi = SCRATCH_HEIGHT_RANGE
    return {"mean_level": levels.mean(), "max_level": levels.max(),
            "mean_step_height_cm": (lo + (levels + 0.5) / rows * (hi - lo)).mean() * 100.0,
            "top_level_fraction": (levels >= rows - 1).float().mean()}


def scratch_done(env):
    """越过终点即截断（成功，time_out 语义）；横向出 1 m 车道也截断，避免在台阶侧面学到绕行。"""
    rel = _offsets(env)
    return (rel[:, 0] >= TARGET_M + 0.3) | (rel[:, 1].abs() > FAIL_LANE_M)


@configclass
class ParallelogramRoverSuperMimicScratchEnvCfg(ParallelogramRoverSuperMimicTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain.terrain_generator
        terrain.sub_terrains = {
            "pyramid_stairs_inv": ScratchStairsTerrainCfg(proportion=0.85),
            "flat": MeshPlaneTerrainCfg(proportion=0.15),
        }
        terrain.num_rows = 9
        terrain.num_cols = 20
        terrain.difficulty_range = (0.0, 1.0)
        terrain.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels.func = scratch_curriculum
        self.terminations.preview_success = None
        self.terminations.scratch_done = TerminationTermCfg(func=scratch_done, time_out=True)
