"""恢复训练/回放的物理与观测动作合同，防止同维数模型静默错配。

@author pony
@date 2026-09-28
@version v0.5.0
@last_modified 2026-09-30
@changelog
    - v0.5.0 (2026-09-30): 集中Residual/FailureReplay任务分类，供训练、评估和回放共用。
    - v0.4.0 (2026-09-30): 注册FailureReplay-Residual恢复任务控制合同。
    - v0.3.0 (2026-09-30): 注册Side-Residual恢复任务控制合同。
    - v0.2.0 (2026-09-29): 阶段恢复共用严格四轮验收，观测扩维仍执行语义校验。
    - v0.1.0 (2026-09-28): 保存资产哈希及控制合同，显式拒绝不兼容的恢复checkpoint。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from recovery_demonstrations import canonical_config


GROUNDED_TASKS = (
    "RobotLab-Isaac-Recovery-Side-Phase-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-Grounded-Progress-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-Upside-Parallelogram-Rover-v0",
)

RESIDUAL_RECOVERY_TASKS = (
    "RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0",
)

FAILURE_REPLAY_RECOVERY_TASKS = (
    "RobotLab-Isaac-Recovery-FailureReplay-Parallelogram-Rover-v0",
    "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0",
)


def is_residual_recovery_task(task):
    """判断任务是否需要 program prior + residual action wrapper。"""
    return task in RESIDUAL_RECOVERY_TASKS


def is_failure_replay_recovery_task(task):
    """判断任务是否必须在创建 Isaac 环境前注入失败状态池路径。"""
    return task in FAILURE_REPLAY_RECOVERY_TASKS


def configure_failure_replay_env(env_cfg, task, failure_states=None):
    """在创建 Gym 环境前绑定并校验 FailureReplay 状态池。

    训练和回放都必须经过同一条路径：显式 CLI 路径不能静默覆盖 Hydra 路径，
    且不存在的文件要在启动 PhysX 前失败。返回解析后的绝对路径；普通任务返回 None。
    """
    replay = is_failure_replay_recovery_task(task)
    if not replay:
        if failure_states is not None:
            raise ValueError("failure_states只允许用于FailureReplay任务")
        return None
    reset_fall_pose = getattr(getattr(env_cfg, "events", None), "reset_fall_pose", None)
    if reset_fall_pose is None or not hasattr(reset_fall_pose, "params"):
        raise ValueError("FailureReplay环境缺少reset_fall_pose参数入口")
    configured = reset_fall_pose.params.get("failure_state_path", "")
    if failure_states is None and not configured:
        raise ValueError("FailureReplay必须通过--failure_states或Hydra指定失败状态池")
    selected = Path(failure_states or configured).expanduser().resolve()
    if configured and Path(configured).expanduser().resolve() != selected:
        raise ValueError("--failure_states与Hydra失败池路径冲突，拒绝静默覆盖训练/回放来源")
    if not selected.is_file():
        raise FileNotFoundError(f"FailureReplay失败状态池不存在: {selected}")
    reset_fall_pose.params["failure_state_path"] = str(selected)
    return selected


def is_rover_recovery(task):
    return bool(task and "Recovery-" in task and "Parallelogram-Rover" in task)


def control_contract(env, agent):
    """忽略并行规模/种子/倒姿课程，但不忽略尺度、周期、驱动或碰撞。

    同一控制合同可以从侧躺继续训练到翻覆；改变动作尺度则必须重新初始化。
    eval关闭观测噪声允许，噪声定义及预处理函数不允许悄悄改变。
    """
    robot = env["scene"]["robot"]
    observations = copy.deepcopy(env["observations"])
    for group in observations.values():
        if isinstance(group, dict):
            group.pop("enable_corruption", None)
    spawn = robot["spawn"]
    asset = Path(spawn["asset_path"])
    if not asset.is_file():
        raise FileNotFoundError(f"恢复合同资产不存在: {asset}")
    return canonical_config({
        "schema": "rover_recovery_control_v1",
        "actions": env["actions"], "observations": observations,
        "action_term_order": list(env["actions"]),
        "observation_term_order": {
            name: [term for term, cfg in group.items() if isinstance(cfg, dict) and "func" in cfg]
            for name, group in observations.items() if isinstance(group, dict)
        },
        "physics_dt": env["sim"]["dt"], "decimation": env["decimation"],
        "policy": agent["policy"], "wrapper_raw_clip": agent.get("clip_actions"),
        "obs_groups": agent.get("obs_groups", {}),
        "default_joint_state": {key: robot["init_state"][key] for key in ("joint_pos", "joint_vel")},
        "actuators": robot["actuators"], "soft_joint_pos_limit_factor": robot.get("soft_joint_pos_limit_factor"),
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "spawn_physics": {key: spawn.get(key) for key in (
            "collision_from_visuals", "collider_type", "self_collision", "replace_cylinders_with_capsules",
            "rigid_props", "articulation_props", "joint_drive", "fix_base", "merge_fixed_joints",
        )},
    })


def validate_control_contract(source, target):
    source, target = canonical_config(source), canonical_config(target)
    different = sorted(key for key in source.keys() | target.keys() if source.get(key) != target.get(key))
    if different:
        raise ValueError(
            f"恢复控制合同不匹配: {', '.join(different)}。不能仅凭维数或任务名称直接resume/play；"
            "请使用相同任务/电机配置，或为新控制合同重新采集示范并训练。"
        )


def save_control_contract(params_dir, contract):
    path = Path(params_dir) / "recovery_control_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"contract": canonical_config(contract), "hardware_feasibility_verified": False},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_checkpoint_contract(checkpoint, target):
    params = Path(checkpoint).resolve().parent / "params"
    snapshot = params / "recovery_control_contract.json"
    if snapshot.is_file():
        source = json.loads(snapshot.read_text())["contract"]
        provenance = "snapshot"
    else:
        from upright_warm_start import load_training_config

        source = control_contract(load_training_config(params / "env.yaml"),
                                  load_training_config(params / "agent.yaml"))
        provenance = "legacy_yaml_current_asset_hash_not_historical_verification"
    validate_control_contract(source, target)
    return {"source": "snapshot" if provenance == "snapshot" else "legacy_yaml",
            "provenance": provenance, "compatible": True}


def configure_physics(env_cfg, task, profile_path=None, checkpoint=None):
    """显式标定优先；回放自动使用源run快照，未知硬件不生成假定标定。"""
    if not is_rover_recovery(task):
        if profile_path is not None:
            raise ValueError("recovery_motor_profile只用于平行四边形恢复任务")
        return None
    if profile_path is None and checkpoint is not None:
        saved = Path(checkpoint).resolve().parent / "params/recovery_motor_profile.json"
        if saved.is_file():
            profile_path = saved
    if profile_path is None:
        return None
    from recovery_physics import apply_motor_profile, load_motor_profile

    profile = load_motor_profile(profile_path)
    apply_motor_profile(env_cfg, profile)
    return profile


def save_physics_evidence(params_dir, env_cfg, profile):
    from recovery_physics import audit_urdf

    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_urdf(env_cfg.scene.robot.spawn.asset_path)
    audit["motor_profile_provided"] = profile is not None
    audit["hardware_feasibility_verified"] = False
    with (params_dir / "recovery_physics_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if profile is not None:
        with (params_dir / "recovery_motor_profile.json").open("x", encoding="utf-8") as stream:
            json.dump(profile, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return audit
