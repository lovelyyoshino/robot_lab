# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.4.0
# @last_modified 2026-09-17
# @changelog
 #   - v0.4.0 (2026-09-17): 同模式保留训练期 raw last_action，仅跨轮速尺度时换算实际目标。
#   - v0.3.0 (2026-09-17): 上一轮动作改用实际裁剪后的轮速目标，并在 reset 同步清空 hybrid/recovery 历史。
#   - v0.2.0 (2026-09-17): 明确 44 维 motion 与 42 维 upright 共用低轮速动作合同。
#   - v0.1.0 (2026-09-16): 同一物理环境按技能切换轮速尺度，并转换上一动作的观测单位。

"""Isaac Lab 单车多策略执行适配，不包含硬件接口。"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointVelocityAction


def last_action_for_skill(env, wheel_scale: float) -> torch.Tensor:
    """保持训练期 raw last_action；仅跨轮速尺度交接时以实际目标反算新单位。"""
    actions = env.action_manager.action.clone()
    previous_scale = getattr(env, "_hybrid_applied_wheel_scale", None)
    if previous_scale == wheel_scale:
        # 同一策略连续控制时必须与训练期 mdp.last_action 完全相同，即使轮速目标被 clip。
        return actions
    previous_wheel_target = getattr(env, "_hybrid_last_wheel_target", None)
    if previous_scale is None or previous_wheel_target is None:
        # 首步/reset 后不存在 t-1 控制目标；零值与训练时 ActionTerm.reset 的语义一致。
        actions[:, 8:] = 0.0
    else:
        if previous_wheel_target.shape != actions[:, 8:].shape or not torch.isfinite(previous_wheel_target).all():
            raise RuntimeError("hybrid 上一轮实际轮速历史无效")
        actions[:, 8:] = previous_wheel_target / wheel_scale
    return actions


def reset_hybrid_history(env, env_ids, asset_cfg=None) -> None:
    """reset 时同时清空恢复 `prev_*` 与实际轮速历史，避免跨 episode 泄漏控制量。"""
    from robot_lab.tasks.manager_based.recovery.mdp.observations import reset_prev_obs_cache

    reset_prev_obs_cache(env, env_ids, asset_cfg=asset_cfg) if asset_cfg is not None else reset_prev_obs_cache(env, env_ids)
    history = getattr(env, "_hybrid_last_wheel_target", None)
    if history is not None:
        history[slice(None) if env_ids is None else env_ids] = 0.0
    env._hybrid_applied_wheel_scale = None


class HybridWheelVelocityAction(JointVelocityAction):
    """raw action 仍是策略输出，物理轮速按当前模式选择 5.0 或 0.8。

    42 维 upright 和 44 维 motion 都使用 0.8；两者差别只在观测末尾的
    ``[ground vx, world yaw]`` 指令，不能借轮速尺度来推断策略类型。
    """

    def process_actions(self, actions: torch.Tensor):
        scale = float(getattr(self._env, "_hybrid_requested_wheel_scale", 5.0))
        if scale not in (0.8, 5.0):
            raise ValueError(f"不支持的 rover 轮速尺度：{scale}")
        self._scale = scale
        # 与两个训练任务的处理后动作上限一致，避免切换时继续沿用旧模式上限。
        limit = 31.4159265359 if scale == 5.0 else 100.0
        self._clip[..., 0] = -limit
        self._clip[..., 1] = limit
        super().process_actions(actions)
        self._env._hybrid_applied_wheel_scale = scale
        # `processed_actions` 已包含 scale、offset 与 clip，才是物理执行的轮速目标。
        processed = getattr(self, "_processed_actions", getattr(self, "processed", None))
        if processed is None or not torch.isfinite(processed).all():
            raise RuntimeError("HybridWheelVelocityAction 未产生有限的实际轮速目标")
        self._env._hybrid_last_wheel_target = processed.detach().clone()
