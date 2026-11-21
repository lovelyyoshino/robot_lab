"""预抬教师和相机学生共用的混合地形与真实事件课程。

@author pony
@date 2026-11-03
@version v0.1.0
@last_modified 2026-11-03
@changelog
    - v0.1.0 (2026-11-03): 加入全宽、左右窄障碍；单障碍晋级同时要求真实预抬和通行。
"""

from __future__ import annotations

import torch

from isaaclab.managers import TerminationTermCfg
from isaaclab.terrains import MeshPlaneTerrainCfg

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview import (
    terrain_column_names,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview_env_cfg import (
    RoverSingleStepTerrainCfg,
)
from .scratch_env_cfg import ScratchStairsTerrainCfg, enable_strict_lane, scratch_out_of_lane


def _column_codes(env):
    cfg = env.scene.terrain.cfg.terrain_generator
    names = terrain_column_names(tuple(cfg.sub_terrains), tuple(t.proportion for t in cfg.sub_terrains.values()),
                                 cfg.num_cols, curriculum=cfg.curriculum)
    return torch.tensor([0 if name.startswith("single_step") else 1 if name == "pyramid_stairs_inv" else 2
                         for name in names], device=env.device)[env.scene.terrain.terrain_types]


def prelift_curriculum(env, env_ids):
    """保持地形类型；滚过单障碍但未完成预抬不能升高课程。"""
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if ids.numel():
        rel = env.scene["robot"].data.root_pos_w - terrain.env_origins
        upright = -env.scene["robot"].data.projected_gravity_b[:, 2] >= 0.85
        active = env.episode_length_buf > 0
        code = _column_codes(env)
        state = env.reward_manager.get_term_cfg("anticipatory_wheel_lift").func.state
        # front_paid 由真实起抬、保持到沿、稳定落轮依次锁存；几何收腿不能置位。
        passed = active & upright & (rel[:, 0] >= 3.0) & (rel[:, 1].abs() <= 0.15)
        passed &= (code != 0) | state.front_paid
        failed = active & (~upright | (rel[:, 1].abs() > 0.75) | (rel[:, 0] < 0.8))
        level = terrain.terrain_levels[ids] + passed[ids].long() - failed[ids].long()
        level = level.clamp_min(0)
        top = level >= terrain.max_terrain_level
        level[top] = torch.randint(0, terrain.max_terrain_level, (int(top.sum()),), device=env.device)
        terrain.terrain_levels[ids] = level
        terrain.env_origins[ids] = terrain.terrain_origins[level, terrain.terrain_types[ids]]
    return {"mean_level": terrain.terrain_levels.float().mean()}


def prelift_done(env):
    """三种地形统一保留至少3m行程，避免窄障碍绕行立即终止。"""
    rel = env.scene["robot"].data.root_pos_w - env.scene.terrain.env_origins
    return rel[:, 0] >= 3.3


def configure_prelift_training(env_cfg, student: bool = False):
    """仅更改地形/课程/终止；保留调用方教师或深度学生的观测、动作和相机合同。"""
    terrain = env_cfg.scene.terrain.terrain_generator
    common = dict(step_height_range=(0.04, 0.12), step_front_x_range=(1.20, 1.70))
    terrain.sub_terrains = {
        "single_step": RoverSingleStepTerrainCfg(proportion=0.25, **common),
        "single_step_left": RoverSingleStepTerrainCfg(proportion=0.20, lane_width=0.36,
                                                      lane_center_y=0.30, **common),
        "single_step_right": RoverSingleStepTerrainCfg(proportion=0.20, lane_width=0.36,
                                                       lane_center_y=-0.30, **common),
        "pyramid_stairs_inv": ScratchStairsTerrainCfg(proportion=0.20),
        "flat": MeshPlaneTerrainCfg(proportion=0.15),
    }
    terrain.num_rows, terrain.num_cols = 9, 20
    terrain.difficulty_range, terrain.curriculum = (0.0, 1.0), True
    # 教师和学生均从低台阶起步；训练入口可在调用后显式设置初始等级。
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum.terrain_levels.func = prelift_curriculum
    env_cfg.terminations.preview_success = None
    env_cfg.terminations.scratch_done = TerminationTermCfg(func=prelift_done, time_out=True)
    enable_strict_lane(env_cfg)
    env_cfg.terminations.scratch_out_of_lane = TerminationTermCfg(func=scratch_out_of_lane)
    return env_cfg
