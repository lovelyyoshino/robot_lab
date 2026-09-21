"""从独立运动策略真实失败末态开始恢复；保留动量，明确不重建求解器热启动。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 显式数据路径、合同校验、局部复位及首步立即控制。
"""

import torch

from ._utils import _ensure_step_counter
from .failure_states import file_sha256, load_failure_states
from .method1_deng.events import _mimic_relations, zero_action_freefall
from .settled_ground import initialize_settled_recovery


def validate_replay_limits(positions, hard_limits):
    """soft限位是奖励区间；真实失败可以触及硬止挡，只允许2mrad数值约束误差。"""
    if ((positions < hard_limits[:, 0] - .002) | (positions > hard_limits[:, 1] + .002)).any():
        raise ValueError("失败池关节位置超出当前模型物理硬限位")


def reset_failure_state(env, env_ids, failure_state_path: str = ""):
    """只平移记录的世界水平位置；高度/姿态/根与关节速度不变，不插入沉降。

    失败末态可能仍在倾倒途中，正是即时接管需要的分布。数据来自同URDF的
    平地episode，但摩擦随机化和接触求解器缓存未存储，因此称为运动学重放。
    """
    if not failure_state_path:
        raise ValueError("FailureReplay任务必须显式指定 events.reset_fall_pose.params.failure_state_path")
    if len(env_ids) == 0:
        return
    robot = env.scene["robot"]
    if env.cfg.scene.terrain.terrain_type != "plane":
        raise ValueError("失败状态重放当前只支持与采集一致的平地")
    cache = getattr(env, "_failure_state_pool", None)
    if cache is None:
        pool = load_failure_states(failure_state_path, robot.joint_names,
                                   asset_sha256=file_sha256(robot.cfg.spawn.asset_path), control_dt_s=env.step_dt,
                                   failure_kind="terminated")
        cache = {key: torch.as_tensor(pool[key], dtype=robot.data.joint_pos.dtype, device=env.device)
                 for key in ("root_state", "joint_pos", "joint_vel")}
        validate_replay_limits(cache["joint_pos"], robot.data.joint_pos_limits[0])
        for follower, master, multiplier, offset in _mimic_relations(robot):
            if (cache["joint_pos"][:, follower] - multiplier * cache["joint_pos"][:, master] - offset).abs().max() > .01:
                raise ValueError("失败池违反当前mimic几何约束")
        from robot_lab.tasks.manager_based.upright.mdp.geometry import RoverCollisionGeometry

        geometry = RoverCollisionGeometry(robot.cfg.spawn.asset_path, robot.joint_names)
        minimum_z = geometry.root_height(cache["joint_pos"], cache["root_state"][:, 3:7], 0.)
        if (cache["root_state"][:, 2] < minimum_z - .01).any():
            raise ValueError("失败池出现超过1cm的几何穿透，拒绝通过抬高初态隐藏问题")
        cache["path"] = failure_state_path
        cache["sha256"] = pool["sha256"]
        env._failure_state_pool = cache
    elif cache["path"] != failure_state_path:
        raise ValueError("不能在局部reset时更换失败状态池")
    selected = torch.randint(len(cache["root_state"]), (len(env_ids),), device=env.device)
    roots = cache["root_state"][selected].clone()
    roots[:, :2] = 0.0
    roots[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_state_to_sim(roots, env_ids)
    robot.write_joint_state_to_sim(cache["joint_pos"][selected], cache["joint_vel"][selected], env_ids=env_ids)
    _ensure_step_counter(env)
    env._recovery_step_count[env_ids] = -1
    env.action_manager.reset(env_ids)


def initialize_failure_takeover(env, env_ids):
    """复用接管时钟和增益缓存，但在任何物理步前恢复驱动，不消耗准备时间。"""
    initialize_settled_recovery(env, env_ids, preparation_leg_damping=0.0)
    robot = env.scene["robot"]
    env._recovery_control_active[env_ids] = True
    for name, value in (("root_state_w", robot.data.root_state_w), ("joint_pos", robot.data.joint_pos),
                        ("joint_vel", robot.data.joint_vel), ("gravity_b", robot.data.projected_gravity_b)):
        getattr(env, f"_recovery_takeover_{name}")[env_ids] = value[env_ids]
    zero_action_freefall(env, env_ids)
