#!/usr/bin/env python3
"""记录倒姿准备期的真实关节/接触，不把初始化通过当作恢复成功。

@author pony
@date 2026-09-17
@version v0.1.0
@last_modified 2026-09-17
@changelog
    - v0.1.0 (2026-09-17): 按关节名保存零驱动沉降轨迹，支持步长/求解器单变量对照。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def main(args):
    import gymnasium as gym
    import torch

    from isaaclab.managers import TerminationTermCfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.manager_based.recovery.mdp._utils import _get_joint_groups
    from robot_lab.tasks.manager_based.recovery.mdp.method1_deng.events import _mimic_relations

    task = "RobotLab-Isaac-Recovery-Settled-Parallelogram-Rover-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    if args.half_dt:
        cfg.sim.dt /= 2
        cfg.decimation *= 2
    if args.solver_iterations:
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.solver_iterations
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.solver_iterations
    cfg.events.reset_fall_pose.params["pose_weights"] = {args.scenario: 1.0}
    cfg.events.freefall_init.params["preparation_leg_damping"] = args.preparation_leg_damping
    cfg.observations.policy.enable_corruption = False
    records = []
    finished = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

    def snapshot(env):
        robot = env.scene["robot"]
        preparing = ~finished & ~env._recovery_control_active
        if preparing.any():
            ids = torch.where(preparing)[0]
            # 终止项在自动 reset 前读数，失败轨迹不会被下一次初态覆盖。
            records.append({
                "step": env.common_step_counter, "env_ids": ids.cpu().tolist(),
                "preparation_time_s": (env._recovery_settle_steps[ids] * env.step_dt).cpu().tolist(),
                "ready": env._recovery_takeover_pending[ids].cpu().tolist(),
                "failed": env._recovery_initialization_failed[ids].cpu().tolist(),
                "joint_pos": robot.data.joint_pos[ids].cpu().tolist(),
                "joint_vel": robot.data.joint_vel[ids].cpu().tolist(),
                "applied_torque": robot.data.applied_torque[ids].cpu().tolist(),
                "root_state_w": robot.data.root_state_w[ids].cpu().tolist(),
                "contact_forces_w": env.scene.sensors["contact_forces"].data.net_forces_w[ids].cpu().tolist(),
                "metrics": {name: value[ids].cpu().tolist()
                            for name, value in env._recovery_preparation_metrics.items()},
            })
        finished.logical_or_(env._recovery_takeover_pending | env._recovery_initialization_failed)
        return torch.zeros_like(finished)

    cfg.terminations.preparation_snapshot = TerminationTermCfg(func=snapshot)
    env = gym.make(task, cfg=cfg)
    try:
        base = env.unwrapped
        env.reset()
        robot = base.scene["robot"]
        initial_gains = {
            "stiffness": robot.root_physx_view.get_dof_stiffnesses().cpu().tolist(),
            "damping": robot.root_physx_view.get_dof_dampings().cpu().tolist(),
        }
        groups = _get_joint_groups(base, robot)
        for _ in range(args.steps):
            with torch.inference_mode():
                env.step(torch.zeros(args.num_envs, 12, device=args.device))
            if finished.all():
                break
        summaries = []
        for index in range(args.num_envs):
            track = [(record, record["env_ids"].index(index)) for record in records if index in record["env_ids"]]
            last, row = track[-1]
            leg_ids = groups["active_leg"].cpu().tolist()
            worst = max(leg_ids, key=lambda j: abs(last["joint_vel"][row][j]))
            summaries.append({
                "env_id": index, "ready": last["ready"][row], "failed": last["failed"][row],
                "preparation_time_s": last["preparation_time_s"][row],
                "worst_leg_joint": robot.joint_names[worst],
                "worst_leg_position_rad": last["joint_pos"][row][worst],
                "worst_leg_velocity_rad_s": last["joint_vel"][row][worst],
                "max_preparation_applied_torque_nm": max(abs(t) for rec, r in track for t in rec["applied_torque"][r]),
                "metrics": {name: value[row] for name, value in last["metrics"].items()},
            })
        result = {
            "scope": "passive_preparation_diagnostic_not_recovery_acceptance",
            "task": task, "seed": args.seed, "scenario": args.scenario,
            "sim_dt": cfg.sim.dt, "decimation": cfg.decimation,
            "preparation_leg_damping_nm_s_rad": args.preparation_leg_damping,
            "solver_position_iterations": cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,
            "solver_velocity_iterations": cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count,
            "joint_names": robot.joint_names,
            "joint_groups": {name: ids.cpu().tolist() for name, ids in groups.items()},
            "mimic_relations": _mimic_relations(robot),
            "joint_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
            "contact_body_names": base.scene.sensors["contact_forces"].body_names,
            "initial_physics_gains": initial_gains, "summaries": summaries, "records": records,
        }
        # 连续轮关节的无限软限位不属于非有限仿真状态，用 null 明确表示无限界。
        import math
        result["joint_limits"] = [[v if math.isfinite(v) else None for v in pair] for pair in result["joint_limits"]]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("SETTLING_PROBE " + json.dumps(summaries), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=310)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--scenario", choices=("side_lying", "prone", "upright_fail", "uniform_so3"), default="side_lying")
    parser.add_argument("--half-dt", action="store_true")
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--preparation-leg-damping", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or (args.solver_iterations is not None and args.solver_iterations < 1):
        parser.error("环境数、步数和求解迭代数必须为正")
    app = AppLauncher(args).app
    status = 2
    try:
        main(args)
        status = 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(status)
        app.close()
    raise SystemExit(status)
