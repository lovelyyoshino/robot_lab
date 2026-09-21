#!/usr/bin/env python3
"""记录 Grounded recovery 从沉降到策略接管的真实 PhysX 状态。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 明确逐环境接管/响应时钟并拒绝漏帧、倒退与跨episode取证。
    - v0.1.1 (2026-09-17): 修正torch.where索引、按实名主动腿诊断并保留关闭前异常。
    - v0.1.0 (2026-09-17): 新增接管重力、PhysX 增益和小动作响应诊断。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _finite_rows(value, name: str):
    if value is None:
        raise RuntimeError(f"真实 PhysX 诊断缺少 {name}")
    if not value.isfinite().all():
        raise RuntimeError(f"真实 PhysX 诊断出现非有限 {name}")
    return value.detach().cpu().tolist()


def classify_takeover_records(records: list[dict], num_envs: int, response_steps: int) -> list[dict]:
    """逐环境冻结首个 ``pending -> active -> response`` 链。

    ``common_step`` 是全局物理时钟，必须逐步连续递增；``control_steps`` 是每个
    环境在接管后的局部时钟。接管函数把局部时钟置零，但 termination 快照在
    下一次物理步采样，因此首次 active 记录通常是 ``control_steps == 1``，而
    不是一个可观测的零步快照。响应采用明确的旧合同：首个
    ``control_steps >= response_steps`` 的 active 记录。
    """
    if num_envs < 1:
        raise ValueError("num_envs 必须为正")
    if response_steps < 1:
        raise ValueError("response_steps 必须为正")
    if not records:
        raise ValueError("records 不能为空")

    required = ("common_step", "control_active", "control_steps", "takeover_pending", "initialization_failed")
    previous_common_step = None
    for record_index, record in enumerate(records):
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"记录 {record_index} 缺少字段: {', '.join(missing)}")
        common_step = record["common_step"]
        if type(common_step) is not int:
            raise ValueError(f"记录 {record_index} 的 common_step 必须为整数")
        if previous_common_step is not None and common_step != previous_common_step + 1:
            raise ValueError(
                "common_step 必须连续递增："
                f"记录 {record_index} 为 {common_step}，前一条为 {previous_common_step}"
            )
        previous_common_step = common_step
        for name in ("control_active", "control_steps", "takeover_pending", "initialization_failed"):
            values = record[name]
            if not isinstance(values, list) or len(values) != num_envs:
                raise ValueError(f"记录 {record_index} 的 {name} 必须包含 num_envs 个值")

    results = []
    for index in range(num_envs):
        terminal = first_active = response = None
        outcome = "not_terminal"
        previous_control_steps = None
        for record in records:
            active = bool(record["control_active"][index])
            control_steps = record["control_steps"][index]
            if type(control_steps) is not int or control_steps < 0:
                raise ValueError(f"env {index} 的 control_steps 必须为非负整数")

            if terminal is None:
                if active:
                    raise ValueError(f"env {index} 跳过 pending/activation 边界")
                if record["initialization_failed"][index]:
                    terminal, outcome = record, "initialization_failed"
                    break
                if record["takeover_pending"][index]:
                    terminal, outcome = record, "awaiting_control"
                continue

            # pending 后的下一条快照必须已受控；中断时冻结首个 episode，不能等新一轮接管。
            if first_active is None:
                if not active or record["initialization_failed"][index] or record["takeover_pending"][index]:
                    outcome = "reset_before_activation"
                    break
                if control_steps > 1:
                    raise ValueError(
                        f"env {index} 跳过 activation 边界：首次 active control_steps={control_steps}"
                    )
                first_active = record
                outcome = "controlling"

            # 局部 reset 合法地把控制时钟清零；只有持续 active 中的时钟倒退才是错误。
            if not active or record["takeover_pending"][index] or record["initialization_failed"][index]:
                outcome = "control_ended_before_response"
                break
            if previous_control_steps is not None and control_steps < previous_control_steps:
                raise ValueError(
                    f"env {index} 的 control_steps 倒退：{control_steps} < {previous_control_steps}"
                )
            previous_control_steps = control_steps
            if active and control_steps >= response_steps:
                response, outcome = record, "response_captured"
                break
        results.append({
            "env_id": index,
            "outcome": outcome,
            "pre_activation": terminal,
            "first_active": first_active,
            "response": response,
            "activation_common_step": first_active["common_step"] if first_active else None,
            "activation_control_steps": first_active["control_steps"][index] if first_active else None,
            "response_common_step": response["common_step"] if response else None,
            "response_control_steps": response["control_steps"][index] if response else None,
            "response_elapsed_control_steps": (
                response["control_steps"][index] - first_active["control_steps"][index]
                if first_active and response else None
            ),
            "clock_convention": (
                "response_is_first_active_record_with_control_steps_ge_response_steps; "
                "activation_snapshot_is_post_physics_step_and_may_start_at_1"
            ),
        })
    return results


def _active_leg_values(record: dict | None, index: int, key: str, joint_ids: list[int]) -> list[float] | None:
    if record is None:
        return None
    return [record[key][index][joint_id] for joint_id in joint_ids]


def _compact_record(record: dict | None, index: int, joint_ids: list[int]) -> dict | None:
    """只导出判断接管和主动腿响应所需字段，不把所有逐步状态塞进结果。"""
    if record is None:
        return None
    return {
        "common_step": record["common_step"],
        "control_active": record["control_active"][index],
        "control_steps": record["control_steps"][index],
        "settle_steps": record["settle_steps"][index],
        "initialization_failed": record["initialization_failed"][index],
        "gravity_b": record["gravity_b"][index],
        "cached_takeover_gravity_b": record["cached_takeover_gravity_b"][index],
        "active_leg_joint_pos": _active_leg_values(record, index, "joint_pos", joint_ids),
        "active_leg_joint_pos_target": _active_leg_values(record, index, "joint_pos_target", joint_ids),
        "active_leg_actuator_estimated_torque": _active_leg_values(record, index, "actuator_estimated_torque", joint_ids),
        "active_leg_physx_actuation_force": _active_leg_values(record, index, "physx_actuation_forces", joint_ids),
        "active_leg_physx_stiffness": _active_leg_values(record, index, "physx_stiffness", joint_ids),
        "active_leg_physx_damping": _active_leg_values(record, index, "physx_damping", joint_ids),
    }


def main(args) -> None:
    import gymnasium as gym
    import torch

    from isaaclab.managers import TerminationTermCfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401
    from robot_lab.assets.parallelogram_rover import (
        PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES,
        PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES,
    )

    task = "RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = False
    cfg.events.reset_fall_pose.params["pose_weights"] = {args.scenario: 1.0}
    records = []

    def snapshot(env):
        robot = env.scene["robot"]
        physics = robot.root_physx_view
        if physics is None:
            raise RuntimeError("robot.root_physx_view 不存在，不能用 mock 增益代替真实 PhysX")
        records.append({
            "common_step": int(env.common_step_counter),
            "control_active": env._recovery_control_active.cpu().tolist(),
            "control_steps": _finite_rows(env._recovery_control_steps, "control_steps"),
            "settle_steps": _finite_rows(env._recovery_settle_steps, "settle_steps"),
            "takeover_pending": env._recovery_takeover_pending.cpu().tolist(),
            "initialization_failed": env._recovery_initialization_failed.cpu().tolist(),
            "gravity_b": _finite_rows(robot.data.projected_gravity_b, "gravity_b"),
            "root_state_w": _finite_rows(robot.data.root_state_w, "root_state_w"),
            "joint_pos": _finite_rows(robot.data.joint_pos, "joint_pos"),
            "joint_vel": _finite_rows(robot.data.joint_vel, "joint_vel"),
            "joint_pos_target": _finite_rows(robot.data.joint_pos_target, "joint_pos_target"),
            "actuator_estimated_torque": _finite_rows(robot.data.applied_torque, "actuator_estimated_torque"),
            "physx_actuation_forces": _finite_rows(
                physics.get_dof_actuation_forces(), "physx_actuation_forces"
            ),
            "physx_stiffness": _finite_rows(physics.get_dof_stiffnesses(), "physx_stiffness"),
            "physx_damping": _finite_rows(physics.get_dof_dampings(), "physx_damping"),
            "manager_action": _finite_rows(env.action_manager.action, "manager_action"),
            "cached_takeover_gravity_b": _finite_rows(
                env._recovery_takeover_gravity_b, "cached_takeover_gravity_b"
            ),
        })
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    cfg.terminations.takeover_probe_snapshot = TerminationTermCfg(func=snapshot)
    env = None
    try:
        print("TAKEOVER_PROBE_STAGE create_env", flush=True)
        env = gym.make(task, cfg=cfg)
        base = env.unwrapped
        print("TAKEOVER_PROBE_STAGE reset_begin", flush=True)
        env.reset()
        print("TAKEOVER_PROBE_STAGE reset_complete", flush=True)
        robot = base.scene["robot"]
        active_leg_names = (*PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES, *PARALLELOGRAM_ROVER_STEERING_JOINT_NAMES)
        active_leg_ids, resolved_leg_names = robot.find_joints(list(active_leg_names), preserve_order=True)
        if tuple(resolved_leg_names) != active_leg_names:
            raise RuntimeError(
                f"主动腿实名解析失败：期望 {active_leg_names}，实际 {tuple(resolved_leg_names)}"
            )
        action = torch.zeros(args.num_envs, 12, device=args.device)
        action[:, :8] = args.leg_action_amplitude
        for _ in range(args.steps):
            _, _, _, _, _ = env.step(action)
            activation = classify_takeover_records(records, args.num_envs, args.response_steps)
            if all(item["outcome"] in (
                "response_captured", "initialization_failed", "reset_before_activation",
                "control_ended_before_response",
            ) for item in activation):
                break
        print("TAKEOVER_PROBE_STAGE step_complete", flush=True)

        activation = classify_takeover_records(records, args.num_envs, args.response_steps)
        for item in activation:
            index = item["env_id"]
            first_active = item["first_active"]
            response_record = item["response"]
            response_delta = None
            max_physx_actuation_force = None
            if first_active is not None and response_record is not None:
                response_delta = max(
                    abs(after - before)
                    for after, before in zip(
                        (response_record["joint_pos"][index][joint_id] for joint_id in active_leg_ids),
                        (first_active["joint_pos"][index][joint_id] for joint_id in active_leg_ids),
                    )
                )
                max_physx_actuation_force = max(
                    abs(response_record["physx_actuation_forces"][index][joint_id]) for joint_id in active_leg_ids
                )
            item.update({
                "first_active_common_step": first_active["common_step"] if first_active else None,
                "pre_activation": _compact_record(item["pre_activation"], index, active_leg_ids),
                "first_active": _compact_record(first_active, index, active_leg_ids),
                "response": _compact_record(response_record, index, active_leg_ids),
                "max_leg_joint_delta_rad": response_delta,
                "max_leg_physx_actuation_force": max_physx_actuation_force,
                "response_threshold_rad": args.min_joint_response,
                "response_observed": response_delta is not None and response_delta >= args.min_joint_response,
            })
        result = {
            "scope": "real_isaac_takeover_and_actuator_diagnostic_not_recovery_acceptance",
            "task": task,
            "seed": args.seed,
            "scenario": args.scenario,
            "configured_pose_weights": cfg.events.reset_fall_pose.params["pose_weights"],
            "preparation_leg_damping_nm_s_rad": cfg.events.freefall_init.params["preparation_leg_damping"],
            "step_dt": base.step_dt,
            "requested_leg_action_amplitude": args.leg_action_amplitude,
            "response_control_steps_threshold": args.response_steps,
            "joint_names": list(robot.joint_names),
            "active_leg_joint_ids": list(active_leg_ids),
            "active_leg_joint_names": list(resolved_leg_names),
            "activation": activation,
            "record_count": len(records),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("TAKEOVER_PROBE " + json.dumps({
            "output": str(args.output), "activation": activation, "records": len(records),
        }), flush=True)
    finally:
        if env is not None:
            env.close()


def parse_args():
    """先注册 AppLauncher，再标记必填参数，避免它内部提前解析失败。"""
    parser = argparse.ArgumentParser()
    required = [
        parser.add_argument("--scenario", choices=("side_lying", "prone")),
        parser.add_argument("--output", type=Path),
    ]
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--response-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--leg-action-amplitude", type=float, default=0.15)
    parser.add_argument("--min-joint-response", type=float, default=1e-4)
    AppLauncher.add_app_launcher_args(parser)
    for argument in required:
        argument.required = True
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.response_steps < 1:
        parser.error("环境数、总步数和响应步数必须为正")
    if args.response_steps > args.steps:
        parser.error("响应步数不能超过总步数")
    if not 0.0 < args.leg_action_amplitude <= 1.0:
        parser.error("leg-action-amplitude 必须在 (0, 1] 内")
    if not 0.0 <= args.min_joint_response < 1.0:
        parser.error("min-joint-response 必须在 [0, 1) 内")
    return args


if __name__ == "__main__":
    args = parse_args()
    app = AppLauncher(args).app
    status = 2
    try:
        main(args)
        status = 0
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(status)
        app.close()
    raise SystemExit(status)
