"""倒地恢复物理配置的严格校验与 URDF 审计工具。

@author pony
@date 2026-09-28
@version v0.1.0
@last_modified 2026-09-28
@changelog
    - v0.1.0 (2026-09-28): 新增电机标定合同校验、运行时执行器覆盖与 URDF 审计。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Any


SCHEMA = "rover_recovery_motor_profile_v1"
GROUPS = ("suspension", "steering", "wheels")
GROUP_FIELDS = (
    "stiffness",
    "damping",
    "effort_limit_nm",
    "saturation_effort_nm",
    "no_load_speed_rad_s",
    "velocity_limit_rad_s",
)
UNITS = {
    "stiffness": "N*m/rad",
    "damping": "N*m*s/rad",
    "effort_limit_nm": "N*m",
    "saturation_effort_nm": "N*m",
    "no_load_speed_rad_s": "rad/s",
    "velocity_limit_rad_s": "rad/s",
}


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是正有限数")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正有限数")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是非负有限数")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须是非负有限数")
    return value


def _iso_date(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("measurement_date 必须是 ISO 日期或日期时间")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("measurement_date 必须是 ISO 日期或日期时间") from exc
    return value


def validate_motor_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """加载并严格校验已测量的关节输出端电机合同。

    模板中的 ``unverified``/``null`` 值故意不能通过，以免仿真静默采用假参数。
    """
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise ValueError(f"schema 必须为 {SCHEMA}")
    if profile.get("status") != "measured":
        raise ValueError("标定合同必须 status=measured")
    if profile.get("parameter_frame") != "joint_output":
        raise ValueError("parameter_frame 必须为 joint_output")
    evidence = profile.get("source_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("source_evidence 不能为空")
    _iso_date(profile.get("measurement_date"))
    units = profile.get("units")
    if units != UNITS:
        raise ValueError(f"units 必须精确声明为 {UNITS}")
    normalized = copy.deepcopy(profile)
    groups = normalized.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError(f"groups 必须恰好包含 {GROUPS}")
    for group_name in GROUPS:
        group = groups[group_name]
        if not isinstance(group, dict) or set(group) != set(GROUP_FIELDS):
            raise ValueError(f"{group_name} 必须恰好包含 {GROUP_FIELDS}")
        for field in GROUP_FIELDS:
            checker = _finite_nonnegative if field in ("stiffness", "damping") else _finite_positive
            group[field] = checker(group[field], f"{group_name}.{field}")
        if group["saturation_effort_nm"] < group["effort_limit_nm"]:
            raise ValueError(f"{group_name}.saturation_effort_nm 不得低于 effort_limit_nm")
        if group["velocity_limit_rad_s"] > group["no_load_speed_rad_s"]:
            raise ValueError(f"{group_name}.velocity_limit_rad_s 不得高于 no_load_speed_rad_s")
    return normalized


def load_motor_profile(path: str | Path) -> dict[str, Any]:
    """加载并严格校验已测量的关节输出端电机合同。

    模板中的 ``unverified``/``null`` 值故意不能通过，以免仿真静默采用假参数。
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            profile = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"无法读取严格 JSON 标定文件：{path}") from exc
    return validate_motor_profile(profile)


def profile_digest(profile: dict[str, Any]) -> str:
    """返回规范化合同摘要；摘要不替代原始证据文件哈希。"""
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_motor_profile(env_cfg: Any, profile: dict[str, Any]) -> None:
    """仅覆盖 recovery 环境副本的三组执行器，不触碰 URDF 限位或 mimic。"""
    if isinstance(profile, (str, Path)):
        profile = load_motor_profile(profile)
    elif isinstance(profile, dict):
        profile = validate_motor_profile(profile)
    else:
        raise ValueError("profile 必须是标定文件路径或已校验字典")
    from isaaclab.actuators import DCMotorCfg

    robot = copy.deepcopy(env_cfg.scene.robot)
    for group_name in GROUPS:
        old = robot.actuators[group_name]
        values = profile["groups"][group_name]
        joint_names_expr = copy.deepcopy(old.joint_names_expr)
        robot.actuators[group_name] = DCMotorCfg(
            joint_names_expr=joint_names_expr,
            effort_limit=values["effort_limit_nm"],
            saturation_effort=values["saturation_effort_nm"],
            velocity_limit=values["no_load_speed_rad_s"],
            effort_limit_sim=values["effort_limit_nm"],
            velocity_limit_sim=values["velocity_limit_rad_s"],
            stiffness=values["stiffness"],
            damping=values["damping"],
            armature=getattr(old, "armature", None),
            friction=getattr(old, "friction", 0.0),
        )
    env_cfg.scene.robot = robot


def audit_urdf(path: str | Path) -> dict[str, Any]:
    """审计质量、碰撞覆盖、mimic 和工具不支持的碰撞几何；不判断硬件可行性。"""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"无法解析 URDF：{path}") from exc
    links = root.findall("link")
    collision_links, missing = [], []
    unsupported = []
    total_mass = 0.0
    invalid_mass_links = []
    for link in links:
        name = link.get("name", "")
        inertial = link.find("inertial/mass")
        if inertial is not None and inertial.get("value") is not None:
            try:
                mass = float(inertial.get("value"))
                if math.isfinite(mass) and mass > 0:
                    total_mass += mass
                else:
                    invalid_mass_links.append(name)
            except (TypeError, ValueError):
                invalid_mass_links.append(name)
        elif inertial is None:
            invalid_mass_links.append(name)
        collisions = link.findall("collision")
        if collisions:
            collision_links.append(name)
            for collision in collisions:
                geometry = collision.find("geometry")
                supported = geometry is not None and any(geometry.find(kind) is not None for kind in ("box", "cylinder"))
                if not supported:
                    unsupported.append({"link": name, "geometry": geometry[0].tag if geometry is not None and len(geometry) else None})
        elif link.find("visual") is not None:
            missing.append(name)
    mimics = []
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics.append({"joint": joint.get("name"), "parent_joint": mimic.get("joint"),
                           "multiplier": mimic.get("multiplier"), "offset": mimic.get("offset")})
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "link_count": len(links),
        "total_mass_kg": total_mass,
        "invalid_mass_links": invalid_mass_links,
        "collision_links": collision_links,
        "missing_collision_visual_links": missing,
        "mimic_joints": mimics,
        "unsupported_collision_geometries": unsupported,
        "hardware_feasibility_verified": False,
    }
