"""
kinematics.py — 平行四边形车的可微正运动学：arm/steer 角 → 四个轮心在机体系/世界系的位置。

直接解析 URDF 的关节原点与轴，不手抄常数。每个角的链为
  base_link →(arm_joint θ) arm →(zhuanxiang_joint −θ，mimic) zhuanxiang →(qudong_joint φ) qudong
  →(wheel_joint，连续，不影响轮心) wheel；
轮心 = wheel 连杆原点加上碰撞圆柱的局部偏移（URDF collision origin）。
lower_arm 只是平行四边形的另一根杆，不在轮子的运动链上，因此 FK 只需 arm 与 steer 两个自由度。
角顺序与动作合同一致：(fl, rl, fr, rr)，arm 在前 4 列、steer 在后 4 列。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from .quat import quat_apply

CORNERS = ("fl", "rl", "fr", "rr")
DEFAULT_URDF = Path(__file__).resolve().parents[2] / "data/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"


def _rpy_matrix(r: float, p: float, y: float) -> torch.Tensor:
    # URDF 约定：R = Rz(y) · Ry(p) · Rx(r)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=torch.float64)


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues，angle: [...]，axis: [3] 单位向量 → [..., 3, 3]，对 angle 可导。"""
    x, y, z = axis.unbind()
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1.0 - c
    rows = [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


class RoverKinematics(torch.nn.Module):
    def __init__(self, urdf_path: str | Path = DEFAULT_URDF):
        super().__init__()
        root = ET.parse(str(urdf_path)).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        links = {lk.get("name"): lk for lk in root.findall("link")}

        def origin(elem):
            o = elem.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            return torch.tensor(xyz, dtype=torch.float64), _rpy_matrix(*rpy)

        def axis(elem):
            a = elem.find("axis")
            v = torch.tensor([float(x) for x in (a.get("xyz") if a is not None else "1 0 0").split()], dtype=torch.float64)
            return v / v.norm()

        names = ("arm", "zhuanxiang", "qudong", "wheel")
        tensors = {f"{n}_{k}": [] for n in names for k in ("t", "R")}
        tensors.update({f"{n}_axis": [] for n in names[:3]})
        mimic = []
        wheel_offset = []
        for corner in CORNERS:
            for n in names:
                j = joints[f"{corner}_{n}_joint"]
                t, R = origin(j)
                tensors[f"{n}_t"].append(t)
                tensors[f"{n}_R"].append(R)
                if n != "wheel":
                    tensors[f"{n}_axis"].append(axis(j))
            m = joints[f"{corner}_zhuanxiang_joint"].find("mimic")
            if m is None or m.get("joint") != f"{corner}_arm_joint":
                raise ValueError(f"{corner}_zhuanxiang_joint 不是 arm 的 mimic，平行四边形假设失效")
            mimic.append((float(m.get("multiplier", "1")), float(m.get("offset", "0"))))
            coll = links[f"{corner}_wheel"].find("collision")
            wheel_offset.append(origin(coll)[0] if coll is not None else torch.zeros(3, dtype=torch.float64))
        for key, value in tensors.items():
            self.register_buffer(key, torch.stack(value).float())
        self.register_buffer("mimic", torch.tensor(mimic, dtype=torch.float32))
        self.register_buffer("wheel_offset", torch.stack(wheel_offset).float())

    def _rot(self, axes, angle):
        # axes: [4,3]，angle: [..., 4] → [..., 4, 3, 3]
        return torch.stack([_axis_angle_matrix(axes[i], angle[..., i]) for i in range(4)], dim=-3)

    def wheel_centers_base(self, arm: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """arm/steer: [..., 4] 绝对关节角 → 轮心在 base_link 系 [..., 4, 3]。"""
        zx = self.mimic[:, 0] * arm + self.mimic[:, 1]
        # 自下而上累乘：p = t_arm + R_arm·Rot(θ)·(t_zx + R_zx·Rot(−θ)·(t_qd + R_qd·Rot(φ)·(t_w + R_w·offset)))
        p = self.wheel_t + (self.wheel_R @ self.wheel_offset.unsqueeze(-1)).squeeze(-1)
        p = p.expand(arm.shape + (3,)) if arm.dim() > 1 else p
        for name, angle in (("qudong", steer), ("zhuanxiang", zx), ("arm", arm)):
            rot = self._rot(getattr(self, f"{name}_axis"), angle)
            frame = getattr(self, f"{name}_R") @ rot
            p = getattr(self, f"{name}_t") + (frame @ p.unsqueeze(-1)).squeeze(-1)
        return p

    def wheel_centers_world(self, pos_w, quat_w, arm, steer) -> torch.Tensor:
        local = self.wheel_centers_base(arm, steer)
        q = quat_w.unsqueeze(-2).expand(local.shape[:-1] + (4,))
        return pos_w.unsqueeze(-2) + quat_apply(q, local)
