# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-12
# @version v0.2.0
# @last_modified 2026-08-12
# @changelog
#   - v0.2.0 (2026-08-12): 回归标准 rough velocity 控制；删除专用堵转/落高状态机。

"""MDP helpers for the parallelogram rover.

The rover follows the standard RobotLab velocity task.  The only robot-specific
state handling is the reset-time derivation of the eight passive parallelogram
follower joints from their four arm masters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_lab.assets.parallelogram_rover import (
    PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_NAMES,
    PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _joint_ids_as_tuple(joint_ids, num_joints: int) -> tuple[int, ...]:
    """Normalize SceneEntityCfg joint selectors to global joint indices."""
    if isinstance(joint_ids, slice):
        return tuple(range(*joint_ids.indices(num_joints)))
    if isinstance(joint_ids, int):
        return (joint_ids,)
    if isinstance(joint_ids, torch.Tensor):
        return tuple(int(index) for index in joint_ids.detach().cpu().tolist())
    return tuple(int(index) for index in joint_ids)


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected joint positions and mask continuous wheel positions."""
    if asset_cfg.name != wheel_asset_cfg.name:
        raise ValueError("asset_cfg and wheel_asset_cfg must address the same articulation")

    asset: Articulation = env.scene[asset_cfg.name]
    selected_ids = _joint_ids_as_tuple(asset_cfg.joint_ids, asset.num_joints)
    wheel_ids = set(_joint_ids_as_tuple(wheel_asset_cfg.joint_ids, asset.num_joints))
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    wheel_columns = [column for column, joint_id in enumerate(selected_ids) if joint_id in wheel_ids]
    if wheel_columns:
        joint_pos_rel[:, wheel_columns] = 0.0
    return joint_pos_rel


def _mimic_relation_ids(asset: Articulation) -> tuple[tuple[int, int, float, float], ...]:
    """Resolve and cache follower/master joint indices."""
    cache_name = "_parallelogram_rover_mimic_relation_ids"
    cached = getattr(asset, cache_name, None)
    if cached is not None:
        return cached

    relations = []
    for follower_name, master_name, multiplier, offset in PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS:
        joint_ids, _ = asset.find_joints([follower_name, master_name], preserve_order=True)
        relations.append((int(joint_ids[0]), int(joint_ids[1]), multiplier, offset))
    result = tuple(relations)
    setattr(asset, cache_name, result)
    return result


def reset_parallelogram_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the 12 active joints and derive all eight passive followers."""
    asset: Articulation = env.scene[asset_cfg.name]
    selected_ids = _joint_ids_as_tuple(asset_cfg.joint_ids, asset.num_joints)
    selected_names = tuple(asset.joint_names[index] for index in selected_ids)
    if selected_names != PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_NAMES:
        raise ValueError(
            "parallelogram reset must use the canonical 12 active joints; "
            f"got {list(selected_names)}"
        )

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    independent_pos = joint_pos[:, selected_ids]
    independent_vel = joint_vel[:, selected_ids]
    independent_pos *= math_utils.sample_uniform(*position_range, independent_pos.shape, independent_pos.device)
    independent_vel *= math_utils.sample_uniform(*velocity_range, independent_vel.shape, independent_vel.device)

    position_limits = asset.data.soft_joint_pos_limits[env_ids][:, selected_ids]
    velocity_limits = asset.data.soft_joint_vel_limits[env_ids][:, selected_ids]
    independent_pos.clamp_(position_limits[..., 0], position_limits[..., 1])
    independent_vel.clamp_(-velocity_limits, velocity_limits)
    joint_pos[:, selected_ids] = independent_pos
    joint_vel[:, selected_ids] = independent_vel

    # Follower state is always derived from the master state; it is never
    # randomized or controlled as an independent policy degree of freedom.
    for follower_id, master_id, multiplier, offset in _mimic_relation_ids(asset):
        joint_pos[:, follower_id] = multiplier * joint_pos[:, master_id] + offset
        joint_vel[:, follower_id] = multiplier * joint_vel[:, master_id]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
