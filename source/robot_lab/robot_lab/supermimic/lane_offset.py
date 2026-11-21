"""
lane_offset.py — 越障方案 B：给 SuperMimic actor 追加 1 维"相对车道中心线的横向偏移"观测。

为什么只给 SuperMimic actor 加、不改 Teacher：
- 16 cm 侧漂发生在爬台阶期间（Teacher 也偏 0.5 m），策略原观测看不到横向位置（扫描 ±0.5 m、车道 3 m、无 base_lin_vel），
  旧方案只在奖励里罚偏移，策略看不到就学不会纠正；
- 新列权重初始化为 0（pad_first_layer），训练起点与 Teacher 完全相同，纠偏能力只靠世界模型梯度学出来。

约定：
- 观测值 = clamp((root_y − env_origin_y) × LANE_OBS_SCALE, ±LANE_OBS_CLIP)，作为 policy 组最后一列（第 233 维）；
  想象侧由 PreviewTask.lane_obs 按最近车道中心线计算，同一公式；
- 训练 run 的 params/lane_offset_obs.json 是标记：评测 / 回放 / 诊断据此自动追加该观测，Teacher 的 vision_contract.json 不变
  （合同描述的是前 232 维，第 233 维单独由本标记描述）。

本文件只依赖 torch；Isaac 侧的 ObsTerm 在函数内延迟导入，单测可脱离仿真器。

@author pony
@date 2026-10-28
@version v1.0.0
@last_modified 2026-10-28
@changelog
    - v1.0.0 (2026-10-28): 初始创建
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

LANE_OBS_SCALE = 4.0  # 0.25 m 偏移 → 1.0，与其他观测量级相当
LANE_OBS_CLIP = 4.0  # 对应 ±1.0 m；验收出界线 0.75 m
MARKER = "lane_offset_obs.json"
TERM_NAME = "lane_offset"


def lane_offset_obs(env) -> torch.Tensor:
    """[N, 1] Isaac 观测项：车体到本环境地形块中心线（env_origins 的 y）的横向偏移。"""
    robot = env.scene["robot"]
    y = robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return (y * LANE_OBS_SCALE).clamp(-LANE_OBS_CLIP, LANE_OBS_CLIP).unsqueeze(-1)


def add_lane_offset_term(env_cfg) -> None:
    """把 lane_offset 追加为 policy 组最后一项；必须在合同计算之后、gym.make 之前调用。"""
    from isaaclab.managers import ObservationTermCfg

    env_cfg.observations.policy.lane_offset = ObservationTermCfg(func=lane_offset_obs)


def policy_uses_lane_offset(policy_path) -> bool:
    """policy_path 约定为 <run>/exported/policy.pt（或 <run>/ckpt_*.pt）；按 run 目录的标记判断。"""
    path = Path(policy_path).resolve()
    for run in (path.parent.parent, path.parent):
        if (run / "params" / MARKER).is_file():
            return True
    return False


def write_marker(run_dir, base_obs_dim: int) -> None:
    (Path(run_dir) / "params").mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "params" / MARKER).write_text(json.dumps({
        "term": TERM_NAME, "index": base_obs_dim, "obs_dim": base_obs_dim + 1,
        "scale": LANE_OBS_SCALE, "clip": LANE_OBS_CLIP,
        "definition": "clamp((root_pos_w.y - env_origins.y) * scale, ±clip)，policy 组最后一列",
    }, ensure_ascii=False, indent=2))


def pad_first_layer(actor_state: dict, extra: int = 1) -> dict:
    """rsl_rl / SuperMimic actor 的 actor.0.weight 末尾补 extra 列 0：新输入不影响输出。"""
    out = dict(actor_state)
    w = out["actor.0.weight"]
    out["actor.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], extra, dtype=w.dtype, device=w.device)), dim=1)
    return out
