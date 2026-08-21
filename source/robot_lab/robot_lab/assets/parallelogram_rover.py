# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-11
# @version v0.1.0
# @last_modified 2026-08-12
# @changelog
#   - v0.1.0 (2026-08-12): 固化 PhysX mimic 约束、12-D 主动关节合同与 5 转/秒轮速上限。

"""Configuration for the parallelogram rover."""

import math
from collections.abc import Iterable

from pxr import Sdf, Usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_urdf as _spawn_from_urdf
from isaaclab.sim.utils import clone

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

SUSPENSION_JOINT_PATTERN = r"^(fl|fr|rl|rr)_arm_joint$"
STEERING_JOINT_PATTERN = r"^(fl|fr|rl|rr)_qudong_joint$"
WHEEL_JOINT_PATTERN = r"^(fl|fr|rl|rr)_wheel_joint$"
INDEPENDENT_JOINT_PATTERN = r"^(fl|fr|rl|rr)_(arm|qudong|wheel)_joint$"
MIMIC_FOLLOWER_JOINT_EXPR = r"^(fl|fr|rl|rr)_(zhuanxiang|lower_arm)_joint$"
MIMIC_HARD_CONSTRAINT_ATTRIBUTE_NAMES = (
    "physxMimicJoint:rotZ:naturalFrequency",
    "physxMimicJoint:rotZ:dampingRatio",
)

PARALLELOGRAM_ROVER_CORNER_NAMES = ("fl", "rl", "fr", "rr")
PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES = tuple(f"{corner}_arm_joint" for corner in PARALLELOGRAM_ROVER_CORNER_NAMES)
PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES = tuple(
    f"{corner}_qudong_joint" for corner in PARALLELOGRAM_ROVER_CORNER_NAMES
)
PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES = tuple(f"{corner}_wheel_joint" for corner in PARALLELOGRAM_ROVER_CORNER_NAMES)
PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES = (
    *PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
    *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    *PARALLELOGRAM_ROVER_WHEEL_JOINT_NAMES,
)
PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_NAMES = (
    *PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES,
)
PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS = tuple(
    relation
    for corner in PARALLELOGRAM_ROVER_CORNER_NAMES
    for relation in (
        (f"{corner}_zhuanxiang_joint", f"{corner}_arm_joint", -1.0, 0.0),
        (f"{corner}_lower_arm_joint", f"{corner}_arm_joint", 1.0, 0.0),
    )
)
PARALLELOGRAM_ROVER_MIMIC_FOLLOWER_JOINT_NAMES = tuple(
    follower_name for follower_name, _, _, _ in PARALLELOGRAM_ROVER_MIMIC_JOINT_RELATIONS
)


def assert_exact_policy_action_names(joint_names: Iterable[str], context: str = "policy action") -> tuple[str, ...]:
    """Check the exact 12-D action order shared with RL-SAR."""
    names = tuple(joint_names)
    if names != PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES:
        raise ValueError(
            f"{context} must use the canonical order {list(PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES)}; "
            f"got {list(names)}"
        )
    return names


@clone
def spawn_parallelogram_rover_from_urdf(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the URDF and make its PhysX mimic joints hard constraints."""
    # The public URDF spawner is clone-decorated.  Use its wrapped function so
    # the hard-constraint attributes are authored before environment cloning.
    prim = _spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    expected = set(PARALLELOGRAM_ROVER_MIMIC_FOLLOWER_JOINT_NAMES)
    hardened = set()
    for joint_prim in Usd.PrimRange(prim):
        if joint_prim.GetName() not in expected or "PhysxMimicJointAPI:rotZ" not in joint_prim.GetAppliedSchemas():
            continue
        for attribute_name in MIMIC_HARD_CONSTRAINT_ATTRIBUTE_NAMES:
            attribute = joint_prim.GetAttribute(attribute_name)
            if not attribute.IsValid():
                attribute = joint_prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float)
            attribute.Set(0.0)
        hardened.add(joint_prim.GetName())

    missing = expected - hardened
    if missing:
        raise RuntimeError(f"Missing PhysX mimic joints in imported URDF: {sorted(missing)}")
    return prim


PARALLELOGRAM_ROVER_SUSPENSION_JOINT_EXPR = SUSPENSION_JOINT_PATTERN
PARALLELOGRAM_ROVER_STEERING_JOINT_EXPR = STEERING_JOINT_PATTERN
PARALLELOGRAM_ROVER_WHEEL_JOINT_EXPR = WHEEL_JOINT_PATTERN
PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_EXPR = INDEPENDENT_JOINT_PATTERN

# Isaac Lab and URDF express angular velocity in rad/s.  Five revolutions per
# second is therefore 2*pi*5 rad/s; this is the wheel mechanical limit, not the
# policy action scale.
PARALLELOGRAM_ROVER_MAX_WHEEL_SPEED_RAD_S = 10.0 * math.pi


PARALLELOGRAM_ROVER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_parallelogram_rover_from_urdf,
        fix_base=False,
        merge_fixed_joints=True,
        # This field is forwarded to the URDF importer's set_parse_mimic().
        convert_mimic_joints_to_normal_joints=True,
        force_usd_conversion=False,
        collision_from_visuals=False,
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        asset_path=(f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/custom/parallelogram_rover/urdf/parallelogram_rover.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.37),
        joint_pos={PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_EXPR: 0.0},
        joint_vel={PARALLELOGRAM_ROVER_INDEPENDENT_JOINT_EXPR: 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "suspension": ImplicitActuatorCfg(
            joint_names_expr=[PARALLELOGRAM_ROVER_SUSPENSION_JOINT_EXPR],
            effort_limit_sim=10.0,
            velocity_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.0,
        ),
        # Register all PhysX DOFs without adding a drive to mimic followers.
        "mimic_followers": ImplicitActuatorCfg(
            joint_names_expr=[MIMIC_FOLLOWER_JOINT_EXPR],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[PARALLELOGRAM_ROVER_STEERING_JOINT_EXPR],
            effort_limit_sim=10.0,
            velocity_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.0,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[PARALLELOGRAM_ROVER_WHEEL_JOINT_EXPR],
            effort_limit=10.0,
            saturation_effort=10.0,
            velocity_limit=PARALLELOGRAM_ROVER_MAX_WHEEL_SPEED_RAD_S,
            velocity_limit_sim=PARALLELOGRAM_ROVER_MAX_WHEEL_SPEED_RAD_S,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
