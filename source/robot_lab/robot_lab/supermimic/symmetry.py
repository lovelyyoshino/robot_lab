"""
symmetry.py — 平行四边形车关于机体 xz 平面的左右镜像：状态 / 动作 / 命令 / 策略观测 / ξ / 地形特征。

@author pony
@date 2026-11-05
@version v1.1.0
@last_modified 2026-11-16

镜像 S = diag(1, −1, 1) 作用在世界系与机体系上（世界系取 y → −y；地形相关量见下文"镜像世界"）：
- 位置 (x, −y, z)；四元数 (w, −x, y, −z)（S·R·S 的四元数）；线速度 (vx, −vy, vz)；角速度是伪向量 (−wx, wy, −wz)；
- 关节：角顺序 = 动作合同 (fl, rl, fr, rr) × (arm, steer=qudong, wheel)，左右角互换（fl↔fr, rl↔rr），符号由 URDF
  零位时各关节在 base 系的转轴推出（joint_mirror_signs）：本机 arm/wheel 轴均为 +y → 镜像同号，steer 轴为 +z → 反号；
- 45 维本体观测（rover_task.policy_observation / preview_task.proprio_observation 同布局）：
  ang_vel·0.25(3) | projected_gravity(3) | 车体系命令(3) | joint_pos(12) | joint_vel·0.05(12) | last_action(12)；
- 187 点高度扫描：按射线起点坐标匹配 (x, y) ↔ (x, −y)（Preview 图样 17×11，meshgrid "xy" 展平，y 对称）；
  车道观测（lane_obs，1 维横向偏移）取反；
- ξ（isaac_adapter.RoverStateReader.context，25/37 维）：腿 Kp 比(8) | Kd 比(12) | 质量比 | base 质量增量 |
  base 质心(3, y 取反) | 可选四轮材料(4×3)；
- 世界模型地形特征（terrain_map.relief_features，39 维）：四轮 × 6 个沿航向探针（轮互换）| 5×3 车体格点（y 互换）。

"镜像世界"：把 M(s) 放在关于 y=0 镜像的地形上；任何地形相关量 g 都满足 g_mirror_world(M s) = M_out(g(s))，
因此镜像样本的地形特征可以用 M_out(g(M(s'))) 在真实地形上算出（M 是对合），不需要镜像网格或射线。

URDF 并非严格对称（左右关节原点差 4–8 mm），镜像是近似对称；单测给出 FK 偏差上界。

@changelog
    - v1.1.0 (2026-11-16): lead_obs（观测末维先抬侧标记 ±1，镜像取反；命令第 4 维同理）。
    - v1.0.0 (2026-11-05): 初始创建（镜像增强与 actor 镜像一致性损失）。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import torch

from .kinematics import CORNERS, DEFAULT_URDF, _rpy_matrix
from .state import ANG, JPOS, JVEL, LIN, NUM_JOINTS, NUM_LEG_JOINTS, POS, QUAT, STATE_DIM
from . import terrain_map

PROPRIO_DIM = 45
JOINT_GROUPS = ("arm", "qudong", "wheel")  # 动作合同中的三组，各 4 个角
# isaac_adapter.CONTEXT_DIM / CONTACT_CONTEXT_DIM 的布局（该模块依赖 Isaac Lab，这里按同一公式复写，单测核对）
CONTEXT_DIM = NUM_LEG_JOINTS + 12 + 1 + 1 + 3
CONTACT_CONTEXT_DIM = CONTEXT_DIM + 4 * 3


def corner_mirror_permutation() -> list[int]:
    """(fl, rl, fr, rr) → 镜像后每个位置取原来的哪个角：fl↔fr, rl↔rr。"""
    swap = {"l": "r", "r": "l"}
    return [CORNERS.index(corner[0] + swap[corner[1]]) for corner in CORNERS]


@lru_cache(maxsize=4)
def joint_mirror_signs(urdf_path: str = str(DEFAULT_URDF)) -> tuple[float, float, float]:
    """(arm, steer, wheel) 三组的镜像符号。

    关于 xz 平面镜像：左侧绕 base 系轴 a_L 转 θ，镜像后等价于绕 −S·a_L 转 θ。右侧轴 a_R = −S·a_L → 右角 = θ（+1）；
    a_R = S·a_L → 右角 = −θ（−1）；都不满足说明左右不是镜像结构，拒绝增强。
    """
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    chain = ("arm", "zhuanxiang", "qudong", "wheel")
    flip = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))

    def base_axes(corner):
        rotation, axes = torch.eye(3, dtype=torch.float64), {}
        for name in chain:
            joint = joints[f"{corner}_{name}_joint"]
            origin = joint.find("origin")
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            rotation = rotation @ _rpy_matrix(*rpy)
            element = joint.find("axis")
            axis = torch.tensor([float(v) for v in (element.get("xyz") if element is not None else "1 0 0").split()],
                                dtype=torch.float64)
            axes[name] = rotation @ (axis / axis.norm())
        return axes

    axes = {corner: base_axes(corner) for corner in CORNERS}
    signs = []
    for group in JOINT_GROUPS:
        group_signs = set()
        for left, right in (("fl", "fr"), ("rl", "rr")):
            a_left, a_right = axes[left][group], axes[right][group]
            if torch.allclose(a_right, -flip @ a_left, atol=1e-6):
                group_signs.add(1.0)
            elif torch.allclose(a_right, flip @ a_left, atol=1e-6):
                group_signs.add(-1.0)
            else:
                raise ValueError(f"{left}/{right}_{group}_joint 转轴不是镜像关系，不能做左右镜像增强")
        if len(group_signs) != 1:
            raise ValueError(f"{group} 前后两对关节的镜像符号不一致")
        signs.append(group_signs.pop())
    return tuple(signs)


def joint_mirror(urdf_path: str | Path = DEFAULT_URDF) -> tuple[list[int], list[float]]:
    """12 个主动关节（动作合同顺序）：mirrored[i] = sign[i] · original[index[i]]。"""
    corner = corner_mirror_permutation()
    signs = joint_mirror_signs(str(urdf_path))
    index = [4 * g + corner[c] for g in range(len(JOINT_GROUPS)) for c in range(4)]
    sign = [signs[g] for g in range(len(JOINT_GROUPS)) for _ in range(4)]
    return index, sign


def mirror_point_index(xy: torch.Tensor, tol: float = 1e-5) -> list[int]:
    """xy: [P, 2] 点集 → 每个点的镜像点 (x, −y) 在点集中的下标；图样不对称时拒绝。"""
    xy = xy.detach().double().cpu()
    target = xy * torch.tensor([1.0, -1.0], dtype=torch.float64)
    distance = (target[:, None, :] - xy[None, :, :]).abs().amax(-1)
    best = distance.argmin(-1)
    if distance.gather(-1, best[:, None]).max() > tol:
        raise ValueError("扫描/格点图样关于 y=0 不对称，无法镜像")
    return best.tolist()


def _state_layout(joint_index, joint_sign):
    index, sign = list(range(STATE_DIM)), [1.0] * STATE_DIM
    for i, s in zip(range(POS.start, POS.stop), (1.0, -1.0, 1.0)):
        sign[i] = s
    for i, s in zip(range(QUAT.start, QUAT.stop), (1.0, -1.0, 1.0, -1.0)):
        sign[i] = s
    for i, s in zip(range(LIN.start, LIN.stop), (1.0, -1.0, 1.0)):
        sign[i] = s
    for i, s in zip(range(ANG.start, ANG.stop), (-1.0, 1.0, -1.0)):
        sign[i] = s
    for block in (JPOS, JVEL):
        for k in range(NUM_JOINTS):
            index[block.start + k] = block.start + joint_index[k]
            sign[block.start + k] = joint_sign[k]
    return index, sign


def _proprio_layout(joint_index, joint_sign):
    index, sign = [], []
    for offset, signs in ((0, (-1.0, 1.0, -1.0)), (3, (1.0, -1.0, 1.0)), (6, (1.0, -1.0, -1.0))):
        index += [offset + k for k in range(3)]
        sign += list(signs)
    for offset in (9, 21, 33):  # joint_pos, joint_vel, last_action
        index += [offset + j for j in joint_index]
        sign += list(joint_sign)
    return index, sign


def _context_layout(context_dim, joint_index):
    if context_dim == 0:
        return [], []
    if context_dim not in (CONTEXT_DIM, CONTACT_CONTEXT_DIM):
        raise ValueError(f"未知的 ξ 布局（{context_dim} 维），只支持 isaac_adapter 的 {CONTEXT_DIM}/{CONTACT_CONTEXT_DIM} 维")
    index = list(joint_index[:NUM_LEG_JOINTS])  # Kp 比：腿关节，增益非负，只互换
    index += [NUM_LEG_JOINTS + j for j in joint_index]  # Kd 比：12 个关节
    base = NUM_LEG_JOINTS + 12
    index += [base, base + 1, base + 2, base + 3, base + 4]  # 质量比、质量增量、质心 xyz
    sign = [1.0] * len(index)
    sign[base + 3] = -1.0  # 质心 y
    if context_dim == CONTACT_CONTEXT_DIM:
        corner = corner_mirror_permutation()
        index += [CONTEXT_DIM + 3 * corner[c] + k for c in range(4) for k in range(3)]
        sign += [1.0] * 12
    return index, sign


def _relief_layout():
    corner = corner_mirror_permutation()
    probes = len(terrain_map.WHEEL_PROBE_X)
    index = [corner[c] * probes + k for c in range(4) for k in range(probes)]
    gx, gy = torch.meshgrid(torch.tensor(terrain_map.BODY_GRID_X), torch.tensor(terrain_map.BODY_GRID_Y), indexing="ij")
    grid = mirror_point_index(torch.stack((gx.flatten(), gy.flatten()), -1))
    index += [4 * probes + g for g in grid]
    return index, [1.0] * len(index)


class RoverMirror:
    """预先算好的带符号置换；每个映射只有一次 index_select 与一次乘法，可放进想象/世界模型热路径。"""

    def __init__(self, device, obs_dim: int = PROPRIO_DIM, scan_pattern: torch.Tensor | None = None,
                 lane_obs: bool = False, context_dim: int = 0, extra_dim: int = 0,
                 urdf_path: str | Path = DEFAULT_URDF, lead_obs: bool = False):
        self.device = torch.device(device)
        joint_index, joint_sign = joint_mirror(urdf_path)
        layouts = {"state": _state_layout(joint_index, joint_sign),
                   "action": (joint_index, joint_sign),
                   "command": ([0, 1, 2, 3], [1.0, -1.0, -1.0, -1.0]) if lead_obs else ([0, 1, 2], [1.0, -1.0, -1.0]),
                   "context": _context_layout(context_dim, joint_index)}
        obs_index, obs_sign = _proprio_layout(joint_index, joint_sign)
        scan_dim = 0 if scan_pattern is None else scan_pattern.shape[0]
        if scan_dim:
            obs_index += [PROPRIO_DIM + i for i in mirror_point_index(scan_pattern[:, :2])]
            obs_sign += [1.0] * scan_dim
        if lane_obs:
            obs_index.append(PROPRIO_DIM + scan_dim)
            obs_sign.append(-1.0)
        if lead_obs:  # 先抬侧标记：镜像后另一侧先抬
            obs_index.append(len(obs_index))
            obs_sign.append(-1.0)
        if len(obs_index) != obs_dim:
            raise ValueError(f"观测 {obs_dim} 维与镜像布局 {len(obs_index)} 维（45 本体 + {scan_dim} 扫描"
                             f"{' + 1 车道' if lane_obs else ''}{' + 1 先抬侧' if lead_obs else ''}）不一致")
        layouts["obs"] = (obs_index, obs_sign)
        if extra_dim == terrain_map.RELIEF_DIM:
            layouts["relief"] = _relief_layout()
        elif extra_dim:
            raise ValueError(f"未知的世界模型地形特征布局（{extra_dim} 维）")
        else:
            layouts["relief"] = ([], [])
        for name, (index, sign) in layouts.items():
            if sorted(index) != list(range(len(index))):
                raise ValueError(f"{name} 镜像下标不是置换")
            setattr(self, f"_{name}_index", torch.tensor(index, dtype=torch.long, device=self.device))
            setattr(self, f"_{name}_sign", torch.tensor(sign, dtype=torch.float32, device=self.device))

    @classmethod
    def for_task(cls, task, device, context_dim: int = 0, extra_dim: int = 0, **kwargs) -> "RoverMirror":
        """从 learner 任务读取观测布局：平地 45 维；Preview 45 + 扫描图样 + 可选车道。"""
        pattern = getattr(task, "pattern", None)
        return cls(device, task.obs_dim, pattern, bool(getattr(task, "lane_obs", False)), context_dim, extra_dim,
                   lead_obs=bool(getattr(task, "lead_obs", False)), **kwargs)

    def _apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        index, sign = getattr(self, f"_{name}_index"), getattr(self, f"_{name}_sign")
        if x.shape[-1] != index.numel():
            raise ValueError(f"{name} 镜像需要 {index.numel()} 维，实际 {x.shape[-1]}")
        return x.index_select(-1, index) * sign.to(x.dtype)

    def state(self, x):
        return self._apply("state", x)

    def action(self, x):
        return self._apply("action", x)

    def command(self, x):
        return self._apply("command", x)

    def obs(self, x):
        return self._apply("obs", x)

    def context(self, x):
        return self._apply("context", x) if x.shape[-1] else x

    def relief(self, x):
        return self._apply("relief", x)

    def windows(self, batch: dict) -> dict:
        """回放窗口的镜像副本（世界模型训练用）；aux（Σacc²）与 valid 对镜像不变。"""
        out = {"state": self.state(batch["state"]), "action": self.action(batch["action"]),
               "context": self.context(batch["context"]), "valid": batch["valid"]}
        if "aux" in batch:
            out["aux"] = batch["aux"]
        return out


def half_mirrored_extra_fn(extra_fn, mirror: RoverMirror, half: int):
    """前 half 行是真实状态、其余行是镜像世界状态时的地形特征：镜像行先映回真实地形查询，再把特征镜像回去。"""
    def extra(state):
        real = torch.cat((state[:half], mirror.state(state[half:])), dim=0)
        features = extra_fn(real)
        return torch.cat((features[:half], mirror.relief(features[half:])), dim=0)
    return extra

