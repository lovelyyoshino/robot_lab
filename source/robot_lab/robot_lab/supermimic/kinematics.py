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
@version v0.3.0
@last_modified 2026-11-06
@changelog
    - v0.3.0 (2026-11-06): com_base：由 URDF 质量得到的质心近似（三轮支撑稳定裕度用）。
    - v0.2.0 (2026-11-05): FK 改用预计算的 R·K / R·K² 缓冲（R(θ)=I+sinθ·K+(1−cosθ)·K²），每个关节只剩
      sin/cos 与一次小矩阵乘，前向+反向 aten 算子从 ~440 降到 ~90；旧的逐元素 Rodrigues 保留为参考实现。
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


def _skew(axis: torch.Tensor) -> torch.Tensor:
    """axis: [..., 3] → 叉乘矩阵 K [..., 3, 3]（K·v = axis × v）。"""
    x, y, z = axis.unbind(dim=-1)
    o = torch.zeros_like(x)
    return torch.stack((torch.stack((o, -z, y), -1), torch.stack((z, o, -x), -1), torch.stack((-y, x, o), -1)), -2)


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues，angle: [...]，axis: [3] 单位向量 → [..., 3, 3]，对 angle 可导。

    逐元素写法（9 次 stack），只作 wheel_centers_base_reference 的参考实现；训练路径用预计算缓冲。"""
    x, y, z = axis.unbind(dim=-1)
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
        stacked = {key: torch.stack(value) for key, value in tensors.items()}
        for key, value in stacked.items():
            self.register_buffer(key, value.float())
        self.register_buffer("mimic", torch.tensor(mimic, dtype=torch.float32))
        self.register_buffer("wheel_offset", torch.stack(wheel_offset).float())
        # Rodrigues 展开：frame(θ) = R·Rot(θ) = R + sinθ·(R·K) + (1−cosθ)·(R·K²)。三块在 float64 里预乘后拼成
        # [4, 9, 3]，每个关节一次小矩阵乘得到三组基向量，再按 (1, sinθ, 1−cosθ) 加权；对 θ 可导。
        for name in names[:3]:
            R, K = stacked[f"{name}_R"], _skew(stacked[f"{name}_axis"])
            self.register_buffer(f"{name}_basis", torch.cat((R, R @ K, R @ K @ K), dim=-2).float())
        wheel_center = stacked["wheel_t"] + (stacked["wheel_R"] @ torch.stack(wheel_offset).unsqueeze(-1)).squeeze(-1)
        self.register_buffer("wheel_center_link", wheel_center.float())

        # 质量模型（URDF inertial）：车体质量放在其惯性原点；每条腿 驱动+轮 的质量记在轮心，
        # 摆臂/转向/下摆臂的质量记在髋（arm 关节原点）与轮心的中点。只用于质心投影，误差 ~1 cm。
        def mass(name):
            inertial = links[name].find("inertial") if name in links else None
            if inertial is None:
                return 0.0, torch.zeros(3, dtype=torch.float64)
            return float(inertial.find("mass").get("value")), origin(inertial)[0]

        base_mass, base_com = mass("base_link")
        wheel_assy = [mass(f"{c}_qudong")[0] + mass(f"{c}_wheel")[0] for c in CORNERS]
        arm_assy = [sum(mass(f"{c}_{part}")[0] for part in ("arm", "zhuanxiang", "lower_arm")) for c in CORNERS]
        self.register_buffer("base_mass", torch.tensor(base_mass, dtype=torch.float32))
        self.register_buffer("base_com", base_com.float())
        self.register_buffer("wheel_assy_mass", torch.tensor(wheel_assy, dtype=torch.float32))
        self.register_buffer("arm_assy_mass", torch.tensor(arm_assy, dtype=torch.float32))
        self.register_buffer("total_mass", torch.tensor(base_mass + sum(wheel_assy) + sum(arm_assy), dtype=torch.float32))

    def _rot(self, axes, angle):
        # axes: [4,3]，angle: [..., 4] → [..., 4, 3, 3]
        return _axis_angle_matrix(axes, angle)

    def wheel_centers_base(self, arm: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """arm/steer: [..., 4] 绝对关节角 → 轮心在 base_link 系 [..., 4, 3]。

        p ← t + R·Rot(θ)·p，自下而上 qudong(steer) → zhuanxiang(mimic) → arm；与 wheel_centers_base_reference
        数值等价（float32 下 <1e-6 m），算子数约为其 1/5。"""
        zx = self.mimic[:, 0] * arm + self.mimic[:, 1]
        p = self.wheel_center_link
        for name, angle in (("qudong", steer), ("zhuanxiang", zx), ("arm", arm)):
            # [4,9,3] @ [...,4,3,1] → [...,4,9,1] → [...,4,3(基),3(xyz)]
            v = torch.matmul(getattr(self, f"{name}_basis"), p.unsqueeze(-1)).unflatten(-2, (3, 3)).squeeze(-1)
            p = (getattr(self, f"{name}_t") + v[..., 0, :] + angle.sin().unsqueeze(-1) * v[..., 1, :]
                 + (1.0 - angle.cos()).unsqueeze(-1) * v[..., 2, :])
        return p

    def wheel_centers_base_reference(self, arm: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """v0.1 的逐元素 Rodrigues 实现，仅供数值核对（单测与基准）。"""
        zx = self.mimic[:, 0] * arm + self.mimic[:, 1]
        # 自下而上累乘：p = t_arm + R_arm·Rot(θ)·(t_zx + R_zx·Rot(−θ)·(t_qd + R_qd·Rot(φ)·(t_w + R_w·offset)))
        p = self.wheel_t + (self.wheel_R @ self.wheel_offset.unsqueeze(-1)).squeeze(-1)
        p = p.expand(arm.shape + (3,)) if arm.dim() > 1 else p
        for name, angle in (("qudong", steer), ("zhuanxiang", zx), ("arm", arm)):
            rot = self._rot(getattr(self, f"{name}_axis"), angle)
            frame = getattr(self, f"{name}_R") @ rot
            p = getattr(self, f"{name}_t") + (frame @ p.unsqueeze(-1)).squeeze(-1)
        return p

    def com_base(self, wheel_centers_base: torch.Tensor) -> torch.Tensor:
        """wheel_centers_base [..., 4, 3] → 整车质心在 base 系 [..., 3]（见 __init__ 的质量模型）。"""
        legs = (self.wheel_assy_mass[:, None] * wheel_centers_base
                + self.arm_assy_mass[:, None] * 0.5 * (self.arm_t + wheel_centers_base)).sum(-2)
        return (self.base_mass * self.base_com + legs) / self.total_mass

    def wheel_centers_world(self, pos_w, quat_w, arm, steer) -> torch.Tensor:
        local = self.wheel_centers_base(arm, steer)
        q = quat_w.unsqueeze(-2).expand(local.shape[:-1] + (4,))
        return pos_w.unsqueeze(-2) + quat_apply(q, local)
