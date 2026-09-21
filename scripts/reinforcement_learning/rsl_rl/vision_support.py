"""视觉通行 v4 的地形法向承重合同；不改变训练奖励或其他技能。

@author pony
@date 2026-09-17
@version v0.1.1
@last_modified 2026-09-17
@changelog
    - v0.1.1 (2026-09-17): 聚合溢出失败关闭，记录实际逐体质量，拒绝损坏固定关节。
    - v0.1.0 (2026-09-17): 单刚体地形过滤、实际质量和终点轮支撑失败关闭。
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


TRAVERSAL_PROTOCOL = "vision_traversal_v4_terminal_ground_support"
WHEEL_NAMES = ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel")
SENSOR_PREFIX = "vision_ground_"
FORCE_THRESHOLD_N = 1.0
MINIMUM_WHEELS = 3
MINIMUM_WEIGHT_FRACTION = 0.5


def support_contract() -> dict:
    """这是工程验收门，不将法向支撑等同于静态稳定性或硬件安全。"""
    return {
        "force_semantics": "terrain_only_net_normal_force_on_robot_body_world_frame",
        "wheel_names": list(WHEEL_NAMES), "minimum_supporting_wheels": MINIMUM_WHEELS,
        "per_wheel_upward_force_strictly_gt_n": FORCE_THRESHOLD_N,
        "minimum_total_wheel_upward_weight_fraction": MINIMUM_WEIGHT_FRACTION,
        "nonwheel_net_normal_force_norm_must_be_le_n": FORCE_THRESHOLD_N,
        "mass_source": "current_all_robot_rigid_body_masses_including_fixed_attachments",
        "sampling": "last_physics_substep_before_reset_each_policy_tick",
        "intermediate_airborne_allowed": True,
    }


def _terrain_filter(cfg) -> str:
    if cfg.scene.terrain.terrain_type != "generator":
        raise ValueError("视觉 v4 只支持已知 generator terrain 碰撞路径")
    # 路径来自已安装 IsaacLab 的 TerrainImporter 和 test_contact_sensor，不过滤整个 /World。
    return cfg.scene.terrain.prim_path.rstrip("/") + "/terrain/mesh"


def configure_support_sensors(cfg) -> dict:
    """每体单独过滤地形，避免 ContactSensor 不支持 many-to-many 的归属歧义。"""
    from isaaclab.sensors import ContactSensorCfg

    spawn = cfg.scene.robot.spawn
    asset = Path(spawn.asset_path)
    if asset.suffix != ".urdf" or not asset.is_file() or not spawn.activate_contact_sensors:
        raise ValueError("视觉 v4 需要本地 URDF 和全部刚体 contact reporter")
    root = ET.parse(asset).getroot()
    names = [link.attrib["name"] for link in root.findall("link")]
    if getattr(spawn, "merge_fixed_joints", False):
        merged = set()
        for joint in root.findall("joint"):
            if joint.attrib.get("type") != "fixed":
                continue
            child = joint.find("child")
            if child is None or child.get("link") not in names:
                raise ValueError("URDF 固定关节必须指向存在的 child link")
            merged.add(child.attrib["link"])
        names = [name for name in names if name not in merged]
    if len(set(names)) != len(names) or not set(WHEEL_NAMES).issubset(names) or "base_link" not in names:
        raise ValueError("URDF 缺少唯一的底盘/四轮刚体")
    ground = _terrain_filter(cfg)
    for name in names:
        key = SENSOR_PREFIX + name
        if hasattr(cfg.scene, key):
            raise ValueError(f"视觉地形传感器名称已占用：{key}")
        setattr(cfg.scene, key, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{name}", update_period=0.0, history_length=0,
            filter_prim_paths_expr=[ground], max_contact_data_count_per_prim=32,
            track_air_time=False, track_contact_points=False, debug_vis=False,
        ))
    # history=0/lazy 避免每个物理子步拷贝历史，终止回调读取最新控制步末力。
    return {"body_names": names, "terrain_filter": ground, "sensor_count_per_env": len(names),
            "history_length": 0, "update_period_s": 0.0}


def ground_support_masks(forces, masses, body_names, gravity_z: float):
    """力顺序由 sensor.body_names 对齐；质量按 articulation 全体刚体求和而不借用传感器索引。"""
    names = list(body_names)
    if len(set(names)) != len(names) or not set(WHEEL_NAMES).issubset(names) or "base_link" not in names:
        raise ValueError("承重合同要求唯一底盘/四轮 body 名称")
    if forces.ndim != 3 or forces.shape[1:] != (len(names), 3):
        raise ValueError("地形法向力必须与 body_names 对齐为 [N,B,3]")
    if masses.shape != forces.shape[:2]:
        raise ValueError("实际质量必须覆盖全部机器人刚体 [N,B]")
    if not math.isfinite(gravity_z) or gravity_z >= 0.0:
        raise ValueError("视觉承重合同要求有限向下重力")
    masses = masses.to(device=forces.device, dtype=forces.dtype)
    valid = torch.isfinite(forces).all(dim=(1, 2)) & torch.isfinite(masses).all(dim=1)
    valid &= (masses >= 0.0).all(dim=1) & (masses.sum(dim=1) > 0.0)
    wheel_ids = [names.index(name) for name in WHEEL_NAMES]
    other_ids = [i for i, name in enumerate(names) if name not in WHEEL_NAMES]
    upward = forces[:, wheel_ids, 2]
    supported = upward > FORCE_THRESHOLD_N
    # 使用带符号合力，压顶接触不能被 abs/clamp 伪造成地面向上支撑。
    total_upward = upward.sum(dim=1)
    weight = masses.sum(dim=1) * -gravity_z
    nonwheel_max = forces[:, other_ids].norm(dim=-1).amax(dim=1)
    valid &= torch.isfinite(total_upward) & torch.isfinite(weight) & torch.isfinite(nonwheel_max)
    stable = valid & (supported.sum(dim=1) >= MINIMUM_WHEELS)
    stable &= total_upward >= MINIMUM_WEIGHT_FRACTION * weight
    stable &= nonwheel_max <= FORCE_THRESHOLD_N
    return {"valid": valid, "supported": stable, "wheel_upward_force_n": upward,
            "supporting_wheels": supported, "total_wheel_upward_force_n": total_upward,
            "robot_weight_n": weight, "nonwheel_max_normal_force_n": nonwheel_max,
            "body_names": names, "body_mass_kg": masses}


def capture_ground_support(env):
    """只读取本物理步末的过滤接触，不可回退到包含自接触的 net_forces_w。"""
    robot = env.scene["robot"]
    names = list(robot.body_names)
    expected = {SENSOR_PREFIX + name for name in names}
    present = {key for key in env.scene.sensors if key.startswith(SENSOR_PREFIX)}
    if present != expected:
        raise RuntimeError("视觉地形传感器没有完整覆盖当前机器人刚体")
    rows = []
    ground = _terrain_filter(env.cfg)
    for name in names:
        sensor = env.scene.sensors[SENSOR_PREFIX + name]
        if list(sensor.body_names) != [name] or list(sensor.cfg.filter_prim_paths_expr) != [ground]:
            raise RuntimeError(f"视觉传感器 body/地形过滤归属不一致：{name}")
        if sensor.contact_physx_view.filter_count != 1:
            raise RuntimeError(f"视觉地形过滤必须命中唯一碰撞 mesh：{name}")
        matrix = sensor.data.force_matrix_w
        if matrix is None or tuple(matrix.shape) != (env.num_envs, 1, 1, 3):
            raise RuntimeError(f"视觉地形过滤力缺失或形状错误：{name}")
        rows.append(matrix[:, 0, 0])
    gravity = tuple(env.cfg.sim.gravity)
    if gravity[:2] != (0.0, 0.0):
        raise ValueError("视觉承重合同只支持 world-Z 向下重力")
    # get_masses 读取随机化后的 PhysX 真值；default_mass/URDF 质量均不能用于验收。
    return ground_support_masks(torch.stack(rows, dim=1), robot.root_physx_view.get_masses(), names, gravity[2])


def support_rows(sample) -> list[dict]:
    """保存 reset 前的承重证据；非有限诊断值用 null，不能让 JSON 写出 NaN。"""
    rows = []
    for index, valid in enumerate(sample["valid"].tolist()):
        def finite_scalar(key):
            value = float(sample[key][index])
            return value if math.isfinite(value) else None

        rows.append({
            "valid": valid, "supported": bool(sample["supported"][index]),
            "supporting_wheels": [name for name, yes in zip(WHEEL_NAMES, sample["supporting_wheels"][index].tolist()) if yes],
            "wheel_upward_force_n": [float(x) if math.isfinite(float(x)) else None
                                      for x in sample["wheel_upward_force_n"][index].tolist()],
            "total_wheel_upward_force_n": finite_scalar("total_wheel_upward_force_n"),
            "robot_weight_n": finite_scalar("robot_weight_n"),
            "nonwheel_max_normal_force_n": finite_scalar("nonwheel_max_normal_force_n"),
            "body_names": sample["body_names"],
            "body_mass_kg": [float(x) if math.isfinite(float(x)) else None
                             for x in sample["body_mass_kg"][index].tolist()],
        })
    return rows
