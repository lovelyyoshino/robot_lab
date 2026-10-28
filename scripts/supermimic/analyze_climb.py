#!/usr/bin/env python3
"""
analyze_climb.py — 离线分析 climb_search.py --record 录下的楼梯回放：逐轮越级事件、卡停形态、每级侧漂。

不启动 Isaac；轮心用 RoverKinematics 由状态重算，楼梯几何来自录制时的 stairs_profile（沿 +x 逐级、全车道宽）。
用法：python scripts/supermimic/analyze_climb.py logs/supermimic_terrain/analysis/teacher/rollout_d0.0.npz [更多 npz]

@author pony
@date 2026-10-27
@version v1.0.0
@last_modified 2026-10-27
@changelog
    - v1.0.0 (2026-10-27): 初始创建；定位 v3 在 8 cm 上后轮卡首级、16 cm 侧漂的机制。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab"))
from supermimic.kinematics import RoverKinematics  # noqa: E402
from supermimic.quat import projected_gravity  # noqa: E402
from supermimic.state import ANG, JPOS, JVEL, LIN, POS, QUAT  # noqa: E402

WHEELS = ("fl", "rl", "fr", "rr")  # 与动作/关节列顺序一致
WHEEL_RADIUS = 0.1085
WHEEL_ACTION_SCALE = 5.0


def load(path: Path):
    z = np.load(path, allow_pickle=False)
    profile = json.loads(str(z["profile"]))
    # stairs_profile 不含级宽，由首末前缘反推（单级时取地形默认 0.40 m）
    n = profile["num_steps"]
    profile["step_width_m"] = (profile["last_edge_m"] - profile["first_edge_m"]) / (n - 1) if n > 1 else 0.40
    state = torch.from_numpy(z["state"]).float()  # [T, N, 37]
    origins = torch.from_numpy(z["origins"]).float()  # [N, 3]
    return state, torch.from_numpy(z["action"]).float(), origins, z["success"], float(z["step_dt"]), profile


def wheel_centers(kin, state):
    jp = state[..., JPOS]
    return kin.wheel_centers_world(state[..., POS], state[..., QUAT], jp[..., 0:4], jp[..., 4:8])  # [T, N, 4, 3]


def ground_height(x_rel, profile):
    """楼梯下的地面高度（相对起点）；每级前缘 first_edge + k·step_width，最后一级延伸到终点平台。"""
    h = torch.zeros_like(x_rel)
    for k in range(profile["num_steps"]):
        h = torch.where(x_rel >= profile["first_edge_m"] + k * profile["step_width_m"],
                        torch.full_like(h, (k + 1) * profile["step_height_m"]), h)
    return h


def analyze(path: Path) -> dict:
    kin = RoverKinematics()
    state, action, origins, success, dt, profile = load(path)
    T, N = state.shape[:2]
    rel = state[..., POS] - origins
    wc = wheel_centers(kin, state) - origins[None, :, None]
    cos_tilt = -projected_gravity(state[..., QUAT])[..., 2]
    fallen = (cos_tilt < 0.0).cumsum(0) > 0  # 翻车后的数据不分析
    valid = ~fallen
    H, S = profile["step_height_m"], profile["num_steps"]
    # 每个轮子"登上第 k 级"的时刻：轮底高度首次 ≥ 该级高度 − 1 cm
    bottom = wc[..., 2] - WHEEL_RADIUS
    climb_t = torch.full((S, N, 4), float("nan"))
    for k in range(S):
        reached = (bottom >= (k + 1) * H - 0.01) & valid[..., None]
        first = reached.float().argmax(0)
        climb_t[k] = torch.where(reached.any(0), first.float() * dt, torch.tensor(float("nan")))
    # 卡停：最后 3 s 前进 < 2 cm 且未成功
    tail = int(3.0 / dt)
    stalled = ((rel[-1, :, 0] - rel[-tail, :, 0]).abs() < 0.02) & torch.from_numpy(~success.astype(bool))
    final = state[-1]
    wheel_x_final = wc[-1, ..., 0]
    edges = torch.tensor([profile["first_edge_m"] + k * profile["step_width_m"] for k in range(S)])
    # 每个轮子最后停在哪级前缘之前（距前缘的水平距离，负 = 在前缘前）
    gap_to_next = torch.full((N, 4), float("nan"))
    for n in range(N):
        for w in range(4):
            ahead = edges[edges > wheel_x_final[n, w] - 0.02]
            if ahead.numel():
                gap_to_next[n, w] = wheel_x_final[n, w] - ahead[0]
    wheel_cmd = action[-tail:, :, 8:12].mean(0) * WHEEL_ACTION_SCALE  # 最后 3 s 平均轮速目标 rad/s
    wheel_vel = state[-tail:, :, JVEL][..., 8:12].abs().mean(0)
    arm = state[..., JPOS][..., 0:4]
    # 每级侧漂：车体 y 在"前轮登上该级"到"后轮登上该级"之间的变化
    drift = []
    for k in range(S):
        t0 = torch.fmin(climb_t[k][:, 0], climb_t[k][:, 2])  # 先上该级的前轮
        t1 = torch.fmax(climb_t[k][:, 1], climb_t[k][:, 3])
        ok = ~t0.isnan() & ~t1.isnan()
        i0, i1 = (t0[ok] / dt).long(), (t1[ok] / dt).long()
        idx = ok.nonzero().squeeze(-1)
        dy = rel[i1, idx, 1] - rel[i0, idx, 1]
        drift.append({"step": k + 1, "n": int(ok.sum()), "dy_mean": dy.mean().item() if ok.any() else None,
                      "dy_std": dy.std().item() if ok.sum() > 1 else None,
                      "duration_s_mean": (t1[ok] - t0[ok]).mean().item() if ok.any() else None})
    yaw = torch.atan2(2 * (state[..., QUAT][..., 0] * state[..., QUAT][..., 3] + state[..., QUAT][..., 1] * state[..., QUAT][..., 2]),
                      1 - 2 * (state[..., QUAT][..., 2] ** 2 + state[..., QUAT][..., 3] ** 2))

    def nanmed(x):
        x = x[~x.isnan()]
        return round(x.median().item(), 3) if x.numel() else None

    res = {"file": str(path), "profile": {k: profile[k] for k in ("step_height_m", "num_steps", "first_edge_m")},
           "N": N, "success_rate": float(success.mean()), "stalled_rate": stalled.float().mean().item()}
    # 越级时序：登上第 1 级的先后（中位时刻）和前/后轴左右差
    res["climb_time_median_s"] = {f"step{k + 1}": {w: nanmed(climb_t[k][:, i]) for i, w in enumerate(WHEELS)} for k in range(S)}
    res["climb_fraction"] = {f"step{k + 1}": {w: round((~climb_t[k][:, i].isnan()).float().mean().item(), 3)
                                              for i, w in enumerate(WHEELS)} for k in range(S)}
    lr = climb_t[0][:, 3] - climb_t[0][:, 1]  # 后轴：rr − rl（正 = 左后先上）
    res["rear_axle_lr_gap_step1_s"] = nanmed(lr)
    fr = climb_t[0][:, 2] - climb_t[0][:, 0]
    res["front_axle_lr_gap_step1_s"] = nanmed(fr)
    # 后轮接近首级前缘时的 arm 角（后轮中心距前缘 0.15 m 内）
    rear_near = ((edges[0] - wc[..., [1, 3], 0]).abs() < 0.15) & valid[..., None]
    for name, mask_set in (("success", torch.from_numpy(success.astype(bool))), ("stalled", stalled)):
        m = rear_near & mask_set[None, :, None]
        res[f"rear_arm_deg_near_edge1_{name}"] = (
            round(math.degrees(arm[..., [1, 3]][m].mean().item()), 1) if m.any() else None)
        mf = ((edges[0] - wc[..., [0, 2], 0]).abs() < 0.15) & valid[..., None] & mask_set[None, :, None]
        res[f"front_arm_deg_near_edge1_{name}"] = (
            round(math.degrees(arm[..., [0, 2]][mf].mean().item()), 1) if mf.any() else None)
    if stalled.any():
        s = stalled
        res["stalled"] = {
            "count": int(s.sum()),
            "body_x_median": nanmed(rel[-1, s, 0]),
            "gap_to_next_edge_m": {w: nanmed(gap_to_next[s, i]) for i, w in enumerate(WHEELS)},
            "wheel_bottom_height_m": {w: nanmed(bottom[-1, s, i] - 0) for i, w in enumerate(WHEELS)},
            "arm_deg": {w: round(math.degrees(final[s][:, JPOS][:, i].median().item()), 1) for i, w in enumerate(WHEELS)},
            "steer_deg": {w: round(math.degrees(final[s][:, JPOS][:, 4 + i].median().item()), 1) for i, w in enumerate(WHEELS)},
            "wheel_cmd_rad_s": {w: nanmed(wheel_cmd[s, i]) for i, w in enumerate(WHEELS)},
            "wheel_speed_abs_rad_s": {w: nanmed(wheel_vel[s, i]) for i, w in enumerate(WHEELS)},
            "yaw_deg": round(math.degrees(yaw[-1, s].median().item()), 1),
            "pitch_proxy_deg": round(math.degrees(torch.acos(cos_tilt[-1, s].clamp(-1, 1)).median().item()), 1),
            "lin_vel_x": nanmed(state[-tail:, s][..., LIN][..., 0].mean(0)),
            "ang_vel_z_abs": nanmed(state[-tail:, s][..., ANG][..., 2].abs().mean(0)),
        }
    res["drift_per_step"] = drift
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(analyze(Path(p)), ensure_ascii=False, indent=1))
