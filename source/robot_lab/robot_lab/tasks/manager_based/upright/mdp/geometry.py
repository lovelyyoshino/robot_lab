"""用 URDF 原始碰撞体计算复位高度；不依赖一步仿真后的穿透修正。

@author pony
@date 2026-09-16
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 复用 URDF 正向运动学计算四轮直行初速，拒绝侧滑/倾斜轮轴。
    - v0.1.0 (2026-09-16): 批量正向运动学及 box/cylinder 精确垂向支撑范围。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import torch


def rotation_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """wxyz → 旋转矩阵；共享训练复位和无仿真几何测试的同一约定。"""
    if not torch.isfinite(quat).all() or torch.any(quat.norm(dim=-1) < 1e-8):
        raise ValueError("四元数必须有限且非零")
    w, x, y, z = (quat / quat.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*quat.shape[:-1], 3, 3)


def _origin(element):
    node = element.find("origin")
    xyz = tuple(float(x) for x in node.get("xyz", "0 0 0").split()) if node is not None else (0., 0., 0.)
    r, p, y = (float(x) for x in node.get("rpy", "0 0 0").split()) if node is not None else (0., 0., 0.)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rotation = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))
    return xyz, rotation


class RoverCollisionGeometry:
    """仅接受当前 rover 使用的原始 box/cylinder；未知几何应失败，不能猜尺寸。"""

    def __init__(self, urdf_path: str, joint_names: list[str]):
        root = ET.parse(urdf_path).getroot()
        self.joint_names = tuple(joint_names)
        index = {name: i for i, name in enumerate(joint_names)}
        self.joints = []
        pending = list(root.findall("joint"))
        reached = {"base_link"}
        while pending:
            ready = [j for j in pending if self._element(j, "parent").attrib["link"] in reached]
            if not ready:
                raise ValueError("URDF 不是以 base_link 为根的连通关节树")
            for joint in ready:
                kind, name = joint.attrib["type"], joint.attrib["name"]
                if kind not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"未支持的几何关节类型：{kind}")
                if kind != "fixed" and name not in index:
                    raise ValueError(f"状态中缺少关节 {name}")
                axis_node = joint.find("axis")
                axis = tuple(float(x) for x in axis_node.get("xyz", "1 0 0").split()) if axis_node is not None else (1., 0., 0.)
                parent = self._element(joint, "parent").attrib["link"]
                child = self._element(joint, "child").attrib["link"]
                self.joints.append((parent, child, None if kind == "fixed" else index[name], _origin(joint), axis))
                reached.add(child)
                pending.remove(joint)
        self.collisions = []
        for link in root.findall("link"):
            for collision in link.findall("collision"):
                geometry = self._element(collision, "geometry")
                box, cylinder = geometry.find("box"), geometry.find("cylinder")
                if box is not None:
                    values = tuple(float(x) / 2 for x in box.attrib["size"].split())
                    kind = "box"
                elif cylinder is not None:
                    values, kind = (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]) / 2), "cylinder"
                else:
                    raise ValueError(f"未知碰撞几何：{link.get('name')}")
                self.collisions.append((link.attrib["name"], _origin(collision), kind, values))
        if not self.collisions:
            raise ValueError("URDF 没有碰撞体")

    @staticmethod
    def _element(parent: ET.Element, tag: str) -> ET.Element:
        node = parent.find(tag)
        if node is None:
            raise ValueError(f"URDF {parent.tag} 缺少 {tag}")
        return node

    def link_transforms(self, joint_pos: torch.Tensor, root_quat: torch.Tensor):
        """相对 root 位置的世界轴 link 变换；高度与带速复位共享同一运动学。"""
        if joint_pos.ndim != 2 or root_quat.shape != (len(joint_pos), 4):
            raise ValueError("关节和四元数必须有相同批量维度")
        if not torch.isfinite(joint_pos).all():
            raise ValueError("关节角必须有限")
        convert = lambda values: joint_pos.new_tensor(values)
        rotations = {"base_link": rotation_from_quat(root_quat)}
        positions = {"base_link": joint_pos.new_zeros(len(joint_pos), 3)}
        for parent, child, index, (xyz, rotation), axis in self.joints:
            positions[child] = positions[parent] + (rotations[parent] @ convert(xyz).view(3, 1)).squeeze(-1)
            initial_rotation = rotations[parent] @ convert(rotation)
            if index is None:
                rotations[child] = initial_rotation
            else:
                direction = convert(axis)
                direction = direction / direction.norm()
                half = joint_pos[:, index] / 2
                quat = torch.cat((half.cos()[:, None], half.sin()[:, None] * direction), -1)
                rotations[child] = initial_rotation @ rotation_from_quat(quat)
        return positions, rotations

    def straight_wheel_velocities(self, joint_pos, root_quat, root_velocity_w, wheel_joint_names):
        """平地纯平移的无纵向滑动轮速；不模拟行走策略接管或接触求解器状态。

        必须先把转向角置直；倾斜轮轴、侧向/垂向运动直接拒绝，不能用投影掩盖侧滑。
        轮半径和转动符号来自 URDF 的圆柱碰撞体与 joint axis，不硬编码左右轮符号。
        """
        if root_velocity_w.shape != (len(joint_pos), 3) or not torch.isfinite(root_velocity_w).all():
            raise ValueError("四轮直行初速度必须是有限 [N,3]")
        _, rotations = self.link_transforms(joint_pos, root_quat)
        normal = joint_pos.new_tensor([0., 0., 1.]).expand(len(joint_pos), 3)
        values = []
        for name in wheel_joint_names:
            index = self.joint_names.index(name)
            links = [joint for joint in self.joints if joint[2] == index]
            if len(links) != 1:
                raise ValueError(f"无法唯一解析轮关节 {name}")
            _, child, _, _, local_axis = links[0]
            collisions = [collision for collision in self.collisions if collision[0] == child]
            if len(collisions) != 1 or collisions[0][2] != "cylinder":
                raise ValueError(f"轮 {name} 必须有唯一圆柱碰撞体")
            _, (center, rotation), _, (radius, _) = collisions[0]
            axis = joint_pos.new_tensor(local_axis)
            axis = axis / axis.norm()
            collision_axis = joint_pos.new_tensor(rotation)[:, 2]
            offset = joint_pos.new_tensor(center)
            if (not math.isfinite(radius) or radius <= 0
                    or not torch.isclose(torch.dot(axis, collision_axis).abs(), axis.new_tensor(1.), atol=1e-6)
                    or torch.linalg.cross(axis, offset).norm() > 1e-6):
                raise ValueError(f"轮 {name} 的圆柱必须与关节轴同轴且半径有效")
            axis_w = (rotations[child] @ axis.view(3, 1)).squeeze(-1)
            if (torch.any(axis_w[:, 2].abs() > 1e-5)
                    or torch.any((root_velocity_w * axis_w).sum(-1).abs() > 1e-5)
                    or torch.any(root_velocity_w[:, 2].abs() > 1e-5)):
                raise ValueError("直行复位要求水平轮轴、无侧滑及无垂向初速度")
            # v + omega * (axis × -radius*normal) = 0，轮轴正负决定真实轮速符号。
            tangent = torch.linalg.cross(axis_w, normal)
            values.append((root_velocity_w * tangent).sum(-1) / radius)
        return torch.stack(values, dim=-1)

    def lowest_points(self, joint_pos: torch.Tensor, root_quat: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回每个碰撞部件相对 root 的世界 z 最低点；支持任意姿态和真实关节角。"""
        positions, rotations = self.link_transforms(joint_pos, root_quat)
        convert = lambda values: joint_pos.new_tensor(values)
        lows: dict[str, torch.Tensor] = {}
        for name, (xyz, rotation), kind, values in self.collisions:
            local_rotation = rotations[name] @ convert(rotation)
            center = positions[name] + (rotations[name] @ convert(xyz).view(3, 1)).squeeze(-1)
            if kind == "box":
                support = (local_rotation[:, 2].abs() * convert(values)).sum(-1)
            else:
                axis_z = local_rotation[:, 2, 2].clamp(-1, 1)
                radius, half_length = values
                support = radius * (1 - axis_z.square()).clamp_min(0).sqrt() + half_length * axis_z.abs()
            low = center[:, 2] - support
            lows[name] = torch.minimum(lows[name], low) if name in lows else low
        return lows

    def root_height(self, joint_pos: torch.Tensor, root_quat: torch.Tensor, clearance: float = 0.002):
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("离地间隙必须为有限非负数")
        lows = self.lowest_points(joint_pos, root_quat)
        return clearance - torch.stack(list(lows.values()), -1).amin(-1)
