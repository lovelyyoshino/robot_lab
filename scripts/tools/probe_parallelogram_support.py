#!/usr/bin/env python3
"""核对直立支撑的真实接触路径、传感器排序与碰撞几何，不以姿态代替成功。

@author pony
@date 2026-09-16
@version v0.1.0
@last_modified 2026-09-16
@changelog
    - v0.1.0 (2026-09-16): 保存逐体接触与碰撞盒离地间隙，支持零动作和策略对照。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


def main(args):
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils.math import matrix_from_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401

    task = "RobotLab-Isaac-Upright-Parallelogram-Rover-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = agent_cfg.device = args.device
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.reset_pose.params.update(
        quad_prob=float(args.initial_state == "quad"), leg_joint_noise=0.0,
        pitch_range=(1.5707963267948966, 1.5707963267948966), base_height=args.base_height,
    )
    # 固定高度用于控制变量复现；--geometry-reset 则测试正式训练的无穿透复位。
    cfg.events.reset_pose.params["collision_clearance"] = 0.002 if args.geometry_reset else None
    cfg.commands.pitch_command.pitch_range = (1.5707963267948966,) * 2
    cfg.episode_length_s = max(4.0, args.steps * cfg.sim.dt * cfg.decimation + 1.0)
    records = []
    # 尺寸来自当前 URDF，不能使用旧探针中猜测的盒半长。
    urdf = ET.parse(cfg.scene.robot.spawn.asset_path)
    collision = urdf.find("link[@name='base_link']/collision")
    box = collision.find("geometry/box")
    origin = collision.find("origin")
    if origin is not None and any(float(x) != 0 for x in origin.get("rpy", "0 0 0").split()):
        raise ValueError("碰撞盒含旋转；需扩展探针，不能误用轴对齐公式")
    half_size = torch.tensor([float(x) / 2 for x in box.get("size").split()], device=args.device)
    center = torch.tensor([float(x) for x in origin.get("xyz", "0 0 0").split()], device=args.device)

    def snapshot(env):
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        base_id = robot.body_names.index("base_link")
        base_pos = robot.data.body_pos_w[:, base_id]
        rotation = matrix_from_quat(robot.data.body_quat_w[:, base_id])
        clearance = (
            base_pos[:, 2] - env.scene.env_origins[:, 2]
            + (rotation[:, 2] * center).sum(-1) - (rotation[:, 2].abs() * half_size).sum(-1)
        )
        records.append({
            "step": len(records) + 1,
            "episode_steps": env.episode_length_buf.cpu().tolist(),
            "gravity_b": robot.data.projected_gravity_b.cpu().tolist(),
            "base_pos_w": base_pos.cpu().tolist(),
            "base_box_clearance_m": clearance.cpu().tolist(),
            "body_positions_w": robot.data.body_pos_w.cpu().tolist(),
            "joint_pos": robot.data.joint_pos.cpu().tolist(),
            "contact_net_forces_w": sensor.data.net_forces_w.cpu().tolist(),
        })
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    cfg.terminations.support_snapshot = TerminationTermCfg(func=snapshot)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = None
        if args.checkpoint is not None:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(str(args.checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
        robot = env.unwrapped.scene["robot"]
        sensor = env.unwrapped.scene.sensors["contact_forces"]
        contact_paths = list(sensor.contact_physx_view.sensor_paths)
        body_paths = list(sensor.body_physx_view.prim_paths)
        obs, _ = env.reset()
        for _ in range(args.steps):
            with torch.inference_mode():
                action = policy(obs) if runner else torch.zeros(args.num_envs, 12, device=args.device)
                obs, _, dones, _ = env.step(action)
                if runner:
                    runner.alg.policy.reset(dones)
        result = {
            "seed": args.seed, "task": task, "initial_state": args.initial_state,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
            "step_dt": env.unwrapped.step_dt,
            "body_names": robot.body_names, "joint_names": robot.joint_names,
            "contact_body_names": sensor.body_names,
            "rigid_body_view_paths": body_paths, "contact_view_paths": contact_paths,
            "contact_order_matches": contact_paths == body_paths,
            "records": records,
            "scope": "support_geometry_diagnostic_not_skill_acceptance",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("SUPPORT_PROBE " + json.dumps({
            "output": str(args.output), "samples": len(records),
            "contact_order_matches": result["contact_order_matches"],
            "last_clearance_m": records[-1]["base_box_clearance_m"],
        }), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-state", choices=("quad", "upright"), default="upright")
    parser.add_argument("--base-height", type=float, default=0.35)
    parser.add_argument("--geometry-reset", action="store_true")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--output", type=Path, default=Path("/tmp/rover_support_probe.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps 和 num_envs 必须为正")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("checkpoint 不存在")
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
