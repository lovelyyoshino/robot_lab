#!/usr/bin/env python3
#
# @author        pony
# @date          2026-08-11
# @version       v1.5.0
# @last_modified 2026-08-31
# @changelog     - v1.5.0 (2026-08-31): 将初始基座高度调到轮子接触地面的无自由落体状态
#                - v1.4.0 (2026-08-31): 同步 self_collision=False 的 MuJoCo 碰撞掩码并统一摩擦基线
#                - v1.3.0 (2026-08-31): 增加波浪道路验证场景和 RL-SAR GUI 按键桥接
#                - v1.2.0 (2026-08-31): 增加 MuJoCo 光照、材质、固定相机和轻量验证场景
#                - v1.1.0 (2026-08-31): 统一生成 RL-SAR 所需的 12 个 effort actuator
#                - v1.0.0 (2026-08-11): 从 parallelogram rover URDF 生成 MJCF
#
"""Generate a MuJoCo MJCF model from the parallelogram rover URDF.

The Isaac Lab asset is intentionally kept as URDF.  MuJoCo's native URDF
reader does not preserve the floating root, actuator definitions, or URDF
``mimic`` tags used by this robot, so this small converter writes an explicit
MJCF model with those pieces declared.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "source/robot_lab/data/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "source/robot_lab/data/Robots/custom/parallelogram_rover/mjcf/parallelogram_rover.xml"

ACTIVE_ORDER = (
    "fl_arm_joint",
    "rl_arm_joint",
    "fr_arm_joint",
    "rr_arm_joint",
    "fl_qudong_joint",
    "rl_qudong_joint",
    "fr_qudong_joint",
    "rr_qudong_joint",
    "fl_wheel_joint",
    "rl_wheel_joint",
    "fr_wheel_joint",
    "rr_wheel_joint",
)

MIMIC_ORDER = (
    ("fl_zhuanxiang_joint", "fl_arm_joint", -1.0),
    ("rl_zhuanxiang_joint", "rl_arm_joint", -1.0),
    ("fr_zhuanxiang_joint", "fr_arm_joint", -1.0),
    ("rr_zhuanxiang_joint", "rr_arm_joint", -1.0),
    ("fl_lower_arm_joint", "fl_arm_joint", 1.0),
    ("rl_lower_arm_joint", "rl_arm_joint", 1.0),
    ("fr_lower_arm_joint", "fr_arm_joint", 1.0),
    ("rr_lower_arm_joint", "rr_arm_joint", 1.0),
)

# 轮子半径 0.1085 m；由当前零姿态的轮心高度反推，使四轮初始轻触地面，
# 避免 MuJoCo 先自由落体再产生一次过大的接触冲量。
INITIAL_BASE_HEIGHT = 0.3535


def _vec(text: str | None, size: int, default: float = 0.0) -> list[float]:
    if not text:
        return [default] * size
    values = [float(value) for value in text.split()]
    if len(values) != size:
        raise ValueError(f"expected {size} values, got {len(values)} in {text!r}")
    return values


def _fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _rpy_to_quat(rpy: list[float]) -> list[float]:
    """Convert URDF roll-pitch-yaw to an MJCF wxyz quaternion."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _set_pose(element: ET.Element, xyz: list[float], rpy: list[float]) -> None:
    if any(abs(value) > 1.0e-12 for value in xyz):
        element.set("pos", _fmt(xyz))
    if any(abs(value) > 1.0e-12 for value in rpy):
        element.set("quat", _fmt(_rpy_to_quat(rpy)))


def _origin(element: ET.Element | None) -> tuple[list[float], list[float]]:
    if element is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return _vec(element.get("xyz"), 3), _vec(element.get("rpy"), 3)


def _add_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        return
    origin_xyz, origin_rpy = _origin(inertial.find("origin"))
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError(f"link {link.get('name')} has incomplete inertial data")
    mj_inertial = ET.SubElement(body, "inertial", mass=mass.get("value", "0"))
    _set_pose(mj_inertial, origin_xyz, origin_rpy)
    mj_inertial.set(
        "fullinertia",
        _fmt(
            [
                float(inertia.get("ixx", "0")),
                float(inertia.get("iyy", "0")),
                float(inertia.get("izz", "0")),
                float(inertia.get("ixy", "0")),
                float(inertia.get("ixz", "0")),
                float(inertia.get("iyz", "0")),
            ]
        ),
    )


def _mesh_name(filename: str) -> str:
    return "mesh_" + Path(filename).name.replace(".", "_").replace("-", "_")


def _add_geometry(parent: ET.Element, link: ET.Element, assets: ET.Element, seen_meshes: set[str]) -> None:
    # Keep robot-to-ground contact while disabling robot self-collision, matching
    # RobotLab's self_collision=False.  MuJoCo tests both cross terms:
    # robot (1,2) vs ground (2,1) collides, robot (1,2) vs robot (1,2) does not.
    for kind, contype, conaffinity, group in (("visual", "0", "0", "2"), ("collision", "1", "2", "1")):
        for index, geometry_container in enumerate(link.findall(kind)):
            origin_xyz, origin_rpy = _origin(geometry_container.find("origin"))
            geometry = geometry_container.find("geometry")
            if geometry is None or len(geometry) == 0:
                continue
            shape = next(iter(geometry))
            geom = ET.SubElement(
                parent,
                "geom",
                name=f"{link.get('name')}_{kind}_{index}",
                contype=contype,
                conaffinity=conaffinity,
                group=group,
            )
            _set_pose(geom, origin_xyz, origin_rpy)
            if shape.tag == "mesh":
                filename = shape.get("filename")
                if not filename:
                    raise ValueError(f"mesh in link {link.get('name')} has no filename")
                name = _mesh_name(filename)
                if name not in seen_meshes:
                    ET.SubElement(
                        assets,
                        "mesh",
                        name=name,
                        file=f"../meshes/{Path(filename).name}",
                        scale=_fmt(_vec(shape.get("scale"), 3, default=1.0)),
                    )
                    seen_meshes.add(name)
                geom.set("type", "mesh")
                geom.set("mesh", name)
            elif shape.tag == "box":
                size = _vec(shape.get("size"), 3)
                geom.set("type", "box")
                geom.set("size", _fmt([value / 2.0 for value in size]))
            elif shape.tag == "cylinder":
                radius = float(shape.get("radius", "0"))
                length = float(shape.get("length", "0"))
                geom.set("type", "cylinder")
                geom.set("size", _fmt([radius, length / 2.0]))
            elif shape.tag == "sphere":
                geom.set("type", "sphere")
                geom.set("size", shape.get("radius", "0"))
            else:
                raise ValueError(f"unsupported URDF geometry <{shape.tag}> in {link.get('name')}")
            if kind == "visual":
                # 可视材质只影响渲染，不改变碰撞/质量；碰撞体继续保持独立的 group。
                geom.set("material", "wheel_visual" if link.get("name", "").endswith("_wheel") else "robot_visual")
            if kind == "collision" and link.get("name", "").endswith("_wheel"):
                geom.set("class", "wheel_contact")


def _add_joint(body: ET.Element, joint: ET.Element) -> None:
    joint_type = joint.get("type")
    if joint_type == "fixed":
        return
    if joint_type not in {"revolute", "continuous"}:
        raise ValueError(f"unsupported joint type {joint_type!r} for {joint.get('name')}")
    axis_element = joint.find("axis")
    axis = _vec(axis_element.get("xyz") if axis_element is not None else None, 3)
    mj_joint = ET.SubElement(body, "joint", name=joint.get("name", ""), type="hinge", axis=_fmt(axis))
    limit = joint.find("limit")
    if joint_type == "revolute":
        if limit is None:
            raise ValueError(f"revolute joint {joint.get('name')} has no limits")
        mj_joint.set("range", _fmt([float(limit.get("lower", "0")), float(limit.get("upper", "0"))]))
        mj_joint.set("limited", "true")
    mj_joint.set("damping", "0")
    mj_joint.set("frictionloss", "0")


def _build_body(
    parent: ET.Element,
    link_name: str,
    links: dict[str, ET.Element],
    joints_by_parent: dict[str, list[ET.Element]],
    assets: ET.Element,
    seen_meshes: set[str],
    joint_by_child: dict[str, ET.Element],
) -> None:
    link = links[link_name]
    if link_name == "base_link":
        # 训练策略不观测 base z；这里优先保证四轮出生即接触地面，避免
        # MuJoCo 从 0.37 m 自由落体后产生额外弹跳冲量。
        body = ET.SubElement(parent, "body", name=link_name, pos=_fmt([0.0, 0.0, INITIAL_BASE_HEIGHT]))
        ET.SubElement(body, "freejoint", name="base_freejoint")
    else:
        body = ET.SubElement(parent, "body", name=link_name)
        joint = joint_by_child[link_name]
        xyz, rpy = _origin(joint.find("origin"))
        _set_pose(body, xyz, rpy)
        _add_joint(body, joint)
    _add_inertial(body, link)
    _add_geometry(body, link, assets, seen_meshes)
    if link_name == "base_link":
        # RL-SAR 用该 site 作为 IMU 参考坐标，不使用 free-joint 的隐式传感器。
        ET.SubElement(body, "site", name="imu", type="sphere", pos="0 0 0", size="0.005", rgba="0 0 0 0")
    for joint in joints_by_parent.get(link_name, []):
        child = joint.find("child")
        if child is None:
            raise ValueError(f"joint {joint.get('name')} has no child link")
        _build_body(
            parent=body,
            link_name=child.get("link"),
            links=links,
            joints_by_parent=joints_by_parent,
            assets=assets,
            seen_meshes=seen_meshes,
            joint_by_child=joint_by_child,
        )


def _add_scene_visuals(assets: ET.Element, worldbody: ET.Element, validation: bool, waves: bool) -> None:
    """Add rendering-only setup and a small contact test rig.

    Isaac Lab's DomeLight intensity is not numerically portable to MuJoCo.
    The equivalent here is an explicit ambient headlight plus a directional
    key/fill pair, so GUI visibility is deterministic without changing RL
    dynamics.
    """
    # 轻量内置纹理/材质，不引入 HDRI 或额外 RobotLab 资产拷贝。
    ET.SubElement(
        assets,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.28 0.42 0.62",
        rgb2="0.02 0.03 0.05",
        width="512",
        height="3072",
    )
    ET.SubElement(
        assets,
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        mark="edge",
        rgb1="0.22 0.24 0.27",
        rgb2="0.10 0.11 0.13",
        markrgb="0.42 0.44 0.48",
        width="300",
        height="300",
    )
    ET.SubElement(assets, "material", name="groundplane", texture="groundplane", texuniform="true", texrepeat="6 6", reflectance="0.15")
    ET.SubElement(assets, "material", name="robot_visual", rgba="0.32 0.42 0.54 1", specular="0.35", shininess="0.35")
    ET.SubElement(assets, "material", name="wheel_visual", rgba="0.06 0.07 0.09 1", specular="0.20", shininess="0.20")
    if validation or waves:
        ET.SubElement(assets, "material", name="validation_visual", rgba="0.55 0.28 0.08 1", specular="0.10", shininess="0.15")
    if waves:
        ET.SubElement(assets, "material", name="wave_visual", rgba="0.18 0.36 0.22 1", specular="0.05", shininess="0.10")

    # 与 RobotLab DomeLight(750) 的视觉意图对齐；MuJoCo 中不使用相同的流明数值。
    ET.SubElement(
        worldbody,
        "light",
        name="sun_key",
        directional="true",
        pos="2 -3 5",
        dir="-0.30 0.45 -1",
        diffuse="0.95 0.90 0.78",
        specular="0.25 0.25 0.25",
        castshadow="true",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="sky_fill",
        directional="true",
        pos="-2 3 3",
        dir="0.25 -0.35 -1",
        diffuse="0.35 0.42 0.55",
        specular="0 0 0",
        castshadow="false",
    )
    # 固定相机通过 MuJoCo Simulate 的 ] 键切换；不修改 RL-SAR 自由相机逻辑。
    ET.SubElement(
        worldbody,
        "camera",
        name="overview",
        pos="1.65 -2.55 1.40",
        xyaxes="0.838 0.545 0 -0.177 0.272 0.946",
    )

    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="20 20 0.1",
        pos="0 0 0",
        material="groundplane",
        friction="1 0.02 0.001",
        contype="2",
        conaffinity="1",
    )
    if validation:
        # 只放两个低矮刚性台阶，用于验证 wheel/arm 接触；不参与训练场景的动力学。
        for name, pos, size in (
            ("validation_pad", "1.00 0 0.035", "0.45 1.20 0.035"),
            ("validation_curb", "1.90 0 0.075", "0.12 1.20 0.075"),
        ):
            ET.SubElement(
                worldbody,
                "geom",
                name=name,
                type="box",
                pos=pos,
                size=size,
                material="validation_visual",
                friction="1 0.02 0.001",
                contype="2",
                conaffinity="1",
            )
    if waves:
        # 用少量相邻斜盒近似正弦路面，避免为一个验证场景引入 hfield 二进制文件。
        start_x = 0.90
        segment_count = 48
        segment_length = 0.12
        amplitude = 0.045
        wavelength = 1.20
        baseline = 0.055
        road_width = 1.60
        thickness = 0.035
        wave_number = 2.0 * math.pi / wavelength
        for index in range(segment_count):
            center_x = start_x + (index + 0.5) * segment_length
            phase_x = center_x - start_x
            profile_z = baseline + amplitude * math.sin(wave_number * phase_x)
            slope = amplitude * wave_number * math.cos(wave_number * phase_x)
            angle = math.atan(slope)
            ET.SubElement(
                worldbody,
                "geom",
                name=f"wave_road_{index:02d}",
                type="box",
                pos=_fmt([center_x, 0.0, profile_z - thickness / 2.0]),
                quat=_fmt([math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0]),
                size=_fmt([segment_length * 0.52, road_width, thickness / 2.0]),
                material="wave_visual",
                friction="1 0.02 0.001",
                contype="2",
                conaffinity="1",
            )


def generate(urdf_path: Path, output_path: Path, validation: bool = False, waves: bool = False) -> None:
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = list(root.findall("joint"))
    joint_by_child = {joint.find("child").get("link"): joint for joint in joints if joint.find("child") is not None}
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')} has no parent/child")
        joints_by_parent.setdefault(parent.get("link"), []).append(joint)
    roots = sorted(set(links) - set(joint_by_child))
    if roots != ["base_link"]:
        raise ValueError(f"expected base_link as the only root link, got {roots}")

    mujoco = ET.Element("mujoco", model="parallelogram_rover")
    ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true")
    # MuJoCo 2 ms physics步与 RL-SAR 的 5 ms control loop解耦，和原先已
    # 通过合同验证的场景一致；策略仍按 base.yaml 的 dt*decimation 推理。
    ET.SubElement(mujoco, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(mujoco, "size", njmax="2000", nconmax="2000")
    ET.SubElement(mujoco, "statistic", center="0 0 0.25", extent="2.5")
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.8 0.8 0.8",
        ambient="0.35 0.35 0.35",
        specular="0.15 0.15 0.15",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.20 0.28 1")
    ET.SubElement(visual, "global", azimuth="-135", elevation="-25")
    ET.SubElement(visual, "quality", shadowsize="2048")
    defaults = ET.SubElement(mujoco, "default")
    ET.SubElement(defaults, "joint", damping="0.5", armature="0.002")
    ET.SubElement(defaults, "geom", condim="4", friction="1 0.02 0.001")
    wheel_defaults = ET.SubElement(defaults, "default", {"class": "wheel_contact"})
    ET.SubElement(
        wheel_defaults,
        "geom",
        friction="1.0 0.05 0.005",
        condim="4",
        solref="0.004 1",
        solimp="0.9 0.95 0.001",
    )
    ET.SubElement(
        defaults,
        "motor",
        ctrllimited="true",
        ctrlrange="-10 10",
        forcelimited="true",
        forcerange="-10 10",
    )

    assets = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")
    _add_scene_visuals(assets, worldbody, validation, waves)
    seen_meshes: set[str] = set()
    _build_body(worldbody, "base_link", links, joints_by_parent, assets, seen_meshes, joint_by_child)

    equality = ET.SubElement(mujoco, "equality")
    for follower, master, multiplier in MIMIC_ORDER:
        ET.SubElement(
            equality,
            "joint",
            name=f"mimic_{follower}",
            joint1=follower,
            joint2=master,
            polycoef=_fmt([0.0, multiplier, 0.0, 0.0, 0.0]),
            # 使用较硬的约束参数，避免在 RL-SAR 的 5 ms 控制周期内累积
            # mimic 误差；验证合同要求 1000 步后残差仍小于 1e-3 rad。
            solref="0.001 1",
            solimp="0.9 0.95 0.001",
        )

    # RL-SAR 在 RL_Sim::SetCommand 中先由 q/dq/kp/kd 计算完整力矩，再写入
    # data->ctrl，因此策略关节必须全部使用 effort actuator；若使用 position
    # actuator，已经计算出的力矩会被错误解释成位置目标。
    actuator = ET.SubElement(mujoco, "actuator")
    for joint_name in ACTIVE_ORDER:
        ET.SubElement(
            actuator,
            "motor",
            name=f"effort_{joint_name}",
            joint=joint_name,
            ctrlrange="-10 10",
            ctrllimited="true",
            forcerange="-10 10",
            forcelimited="true",
        )

    # 传感器数量和顺序与 RL-SAR 的 SensorLayout 合同保持一致；读取通过
    # 名称完成，因此 XML 中 body/joint 的排列不会改变策略映射。
    sensor = ET.SubElement(mujoco, "sensor")
    for joint_name in ACTIVE_ORDER:
        stem = joint_name.removesuffix("_joint")
        ET.SubElement(sensor, "jointpos", name=f"q_{stem}", joint=joint_name)
    for joint_name in ACTIVE_ORDER:
        stem = joint_name.removesuffix("_joint")
        ET.SubElement(sensor, "jointvel", name=f"dq_{stem}", joint=joint_name)
    for joint_name in ACTIVE_ORDER:
        stem = joint_name.removesuffix("_joint")
        ET.SubElement(sensor, "jointactuatorfrc", name=f"tau_{stem}", joint=joint_name)
    ET.SubElement(sensor, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation",
        action="store_true",
        help="add low contact-test steps while keeping the same 12-DOF policy contract",
    )
    parser.add_argument(
        "--waves",
        action="store_true",
        help="add a procedural sinusoidal road for MuJoCo contact/locomotion validation",
    )
    args = parser.parse_args()
    generate(args.urdf.resolve(), args.output.resolve(), validation=args.validation, waves=args.waves)
    print(f"generated {args.output.resolve()}")


if __name__ == "__main__":
    main()
