#!/usr/bin/env python3
"""固定倒置车身，扫描当前URDF运动部件的触地间隙；不代替动力学验收。

@author pony
@date 2026-09-30
@version v0.1.0
@last_modified 2026-09-30
@changelog
    - v0.1.0 (2026-09-30): 复用复位运动学，记录限位内碰撞支点采样证据。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "source/robot_lab/data/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"
GEOMETRY = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/upright/mdp/geometry.py"


def scan_inverted_clearance(urdf: Path = URDF, samples_per_joint: int = 81) -> dict:
    """采样完整关节限位，不把静态间隙等同于动态翻转不可能。

    四条独立支链各受一个arm及一个qudong驱动；同一二维网格可同时穷举各支链。
    这不是8维全身组合搜索，不能用于推出全身质心可达域或碰撞自锁结论。
    """
    if type(samples_per_joint) is not int or not 3 <= samples_per_joint <= 401:
        raise ValueError("samples_per_joint必须为3到401的整数")
    spec = importlib.util.spec_from_file_location("rover_recovery_geometry_probe", GEOMETRY)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载原始运动学模块: {GEOMETRY}")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    tree = ET.parse(urdf).getroot()
    joints = [joint for joint in tree.findall("joint") if joint.get("type") != "fixed"]
    names = [joint.attrib["name"] for joint in joints]
    index = {name: i for i, name in enumerate(names)}
    fractions = torch.linspace(0., 1., samples_per_joint, dtype=torch.float64)
    arm, steering = torch.meshgrid(fractions, fractions, indexing="ij")
    q = torch.zeros(samples_per_joint**2, len(names), dtype=torch.float64)
    mimics = []
    for joint in joints:
        name = joint.attrib["name"]
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics.append((name, mimic.attrib["joint"], float(mimic.get("multiplier", "1")),
                           float(mimic.get("offset", "0"))))
        elif name.endswith(("_arm_joint", "_qudong_joint")):
            limits = joint.find("limit")
            if limits is None:
                raise ValueError(f"关节缺少机械限位: {name}")
            lower, upper = float(limits.attrib["lower"]), float(limits.attrib["upper"])
            fraction = arm if name.endswith("_arm_joint") else steering
            q[:, index[name]] = lower + (upper - lower) * fraction.flatten()
        elif not (name.endswith("_wheel_joint") and joint.get("type") == "continuous"):
            raise ValueError(f"未支持的独立关节: {name}")
    pending = list(mimics)
    resolved = set(names) - {item[0] for item in pending}
    while pending:
        ready = [item for item in pending if item[1] in resolved]
        if not ready:
            raise ValueError("mimic存在循环或来源缺失")
        for name, parent, multiplier, offset in ready:
            q[:, index[name]] = multiplier * q[:, index[parent]] + offset
            resolved.add(name)
            pending.remove((name, parent, multiplier, offset))
    model = geometry.RoverCollisionGeometry(str(urdf), names)
    quat = q.new_tensor([0., 1., 0., 0.]).expand(len(q), 4)
    lows = model.lowest_points(q, quat)
    base_low = lows["base_link"]
    records = []
    for link, low in lows.items():
        if link == "base_link":
            continue
        gap = low - base_low
        witness = int(gap.argmin())
        records.append({
            "link": link, "min_gap_m": float(gap.min()), "max_gap_m": float(gap.max()),
            "sampled_ground_reachable": bool((gap <= 0).any()),
            "witness_joint_pos": {name: float(q[witness, i]) for i, name in enumerate(names)
                                  if name.startswith(link[:3])},
        })
    missing = [link.attrib["name"] for link in tree.findall("link")
               if link.find("visual") is not None and link.find("collision") is None]
    return {
        "scope": "fixed_inverted_base_collision_clearance_grid_not_dynamic_recovery_acceptance",
        "urdf": str(urdf.resolve()), "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "samples_per_joint": samples_per_joint, "samples_per_branch": len(q),
        "root_quat_wxyz": [0., 1., 0., 0.], "root_height_m": float(-base_low[0]),
        "gap_reference": "flat_ground_touching_base_collision_box",
        "joint_sampling": "arm_x_qudong_grid_each_branch_with_exact_mimic_wheel_angle_zero",
        "collision_records": records, "missing_collision_visual_links": missing,
        "dynamic_reachability_proven": False, "hardware_feasibility_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_per_joint", type=int, default=81)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output已存在；请保留旧证据并选择新文件名")
    result = scan_inverted_clearance(samples_per_joint=args.samples_per_joint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "min_gaps_m": {
        row["link"]: row["min_gap_m"] for row in result["collision_records"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
