"""
terrain_map.py — 全局地形高度图、可微双线性查询，以及按想象位姿重建 Teacher 的高度扫描与世界模型地形特征。

高度图由 isaac_adapter 在启动时对地形网格做一次竖直射线投射得到（世界系，栅格 res）。
查询用双线性插值，对 xy 可导：想象中"往前走一步，看到的地形会怎么变"需要这条梯度，
go2-supermimic 注释明确指出把传感器 detach 会让策略在地形附近原地打转。
代价是台阶边一格（res）内被插值成斜坡；verify_terrain 会报告这部分偏差。

高度扫描公式与 isaaclab.envs.mdp.height_scan 相同：
  scan = base_z − hit_z − 0.5，射线起点 = base_pos + yaw_rot(pattern + offset)，ray_alignment="yaw"。
起点的竖直分量不影响竖直射线的命中点，因此只需 xy。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import torch

from .quat import quat_apply, yaw_quat
from .state import JPOS, POS, QUAT

SCAN_OFFSET = 0.5  # isaaclab height_scan 的默认 offset
SCAN_CLIP = (-1.0, 1.0)  # Teacher 观测的 clip
# 世界模型地形特征：每个轮心沿航向前后的采样距离（m）与车体周围格点（机体航向系）
WHEEL_PROBE_X = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
BODY_GRID_X = (-0.40, -0.20, 0.0, 0.20, 0.40)
BODY_GRID_Y = (-0.30, 0.0, 0.30)
RELIEF_DIM = 4 * len(WHEEL_PROBE_X) + len(BODY_GRID_X) * len(BODY_GRID_Y)


class HeightMap(torch.nn.Module):
    """heights[i, j] = 世界 (x0 + i·res, y0 + j·res) 处的地面高度。"""

    def __init__(self, heights: torch.Tensor, origin_xy: tuple[float, float], res: float):
        super().__init__()
        self.register_buffer("heights", heights.float())
        self.x0, self.y0 = float(origin_xy[0]), float(origin_xy[1])
        self.res = float(res)

    def query(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: [..., 2] 世界坐标 → [...] 高度；超出范围按边界值，对 xy 可导（分段线性）。"""
        nx, ny = self.heights.shape
        u = ((xy[..., 0] - self.x0) / self.res).clamp(0.0, nx - 1)
        v = ((xy[..., 1] - self.y0) / self.res).clamp(0.0, ny - 1)
        # 下标单独夹到 n-2：float32 下 n-1-ε 会被舍入回 n-1，再 +1 就越界；边界处 fu=1 取到最后一格。
        i0 = u.floor().long().clamp(0, nx - 2)
        j0 = v.floor().long().clamp(0, ny - 2)
        fu, fv = u - i0, v - j0
        h = self.heights
        h00, h10 = h[i0, j0], h[i0 + 1, j0]
        h01, h11 = h[i0, j0 + 1], h[i0 + 1, j0 + 1]
        return (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv


def scan_points_xy(pos_w: torch.Tensor, quat_w: torch.Tensor, pattern_b: torch.Tensor) -> torch.Tensor:
    """pattern_b: [P, 3] 射线起点在 base 系（已含传感器 offset）→ 世界 xy [..., P, 2]，按 yaw 对齐。"""
    q = yaw_quat(quat_w).unsqueeze(-2).expand(pos_w.shape[:-1] + (pattern_b.shape[0], 4))
    pts = quat_apply(q, pattern_b.expand(q.shape[:-1] + (3,)))
    return pos_w[..., None, :2] + pts[..., :2]


def height_scan(hmap: HeightMap, state: torch.Tensor, pattern_b: torch.Tensor) -> torch.Tensor:
    """按 state 的世界位姿重建 Teacher 的 187 维扫描（state 位置须为世界系）。"""
    hit_z = hmap.query(scan_points_xy(state[..., POS], state[..., QUAT], pattern_b))
    return (state[..., POS][..., 2:3] - hit_z - SCAN_OFFSET).clamp(*SCAN_CLIP)


def relief_features(hmap: HeightMap, kin, state: torch.Tensor) -> torch.Tensor:
    """世界模型的地形输入（RELIEF_DIM 维），全部是"相对高度"，对平移不变：

    - 每个轮心沿车体航向前后 6 点的地面高度减轮心高度（轮子离地净空 + 前方台阶）；
    - 车体航向系 5×3 格点的地面高度减 base 高度（车身底部是否会托底）。
    """
    pos, quat = state[..., POS], state[..., QUAT]
    jpos = state[..., JPOS]
    wheels = kin.wheel_centers_world(pos, quat, jpos[..., 0:4], jpos[..., 4:8])  # [..., 4, 3]
    yq = yaw_quat(quat)
    fwd = quat_apply(yq, torch.tensor([1.0, 0.0, 0.0], device=state.device).expand(yq.shape[:-1] + (3,)))[..., :2]
    probe = torch.tensor(WHEEL_PROBE_X, device=state.device)
    wheel_xy = wheels[..., None, :2] + probe[:, None] * fwd[..., None, None, :]  # [..., 4, K, 2]
    wheel_rel = hmap.query(wheel_xy) - wheels[..., 2:3]
    gx, gy = torch.meshgrid(torch.tensor(BODY_GRID_X, device=state.device),
                            torch.tensor(BODY_GRID_Y, device=state.device), indexing="ij")
    grid_b = torch.stack((gx.flatten(), gy.flatten(), torch.zeros_like(gx.flatten())), dim=-1)
    body_rel = hmap.query(scan_points_xy(pos, quat, grid_b)) - pos[..., 2:3]
    return torch.cat((wheel_rel.flatten(-2), body_rel), dim=-1)


def near_edge(hmap: HeightMap, kin, state: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    """任一轮心前后探针范围内地面高差超过 threshold → 该样本处于台阶边附近。"""
    rel = relief_features(hmap, kin, state)[..., : 4 * len(WHEEL_PROBE_X)].reshape(state.shape[:-1] + (4, -1))
    return (rel.amax(-1) - rel.amin(-1) > threshold).any(-1)
