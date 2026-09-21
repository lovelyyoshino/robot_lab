#!/usr/bin/env python3
"""固定地形、朝向和命令的视觉通行评测，兼容 blind/teacher/student。

@author pony
@date 2026-09-16
@version v0.6.0
@last_modified 2026-09-17
@changelog
    - v0.6.0 (2026-09-17): v4 终点要求地形法向三轮承重、半体重及非轮接触拒绝，保留跨沟腾空。
    - v0.5.0 (2026-09-17): 允许窄沟跨越时合法未命中，终点仍须完整地面扫描，更新至 v3 协议。
    - v0.4.0 (2026-09-17): 修复高差地形误判，拒绝无效扫描，保留窄沟平台掉坑门并版本化协议。
    - v0.3.0 (2026-09-17): 提取自动reset前通行采样入口，支持地形高度判据回归。
    - v0.2.0 (2026-09-16): 增加时序student与通信扰动评测，输出实际深度预处理合同。
    - v0.1.0 (2026-09-16): 增加楼梯、方块、坡面、窄沟与深度失效对照评测。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

from skill_evaluation import SkillEvaluation
from vision_support import (
    TRAVERSAL_PROTOCOL, capture_ground_support, configure_support_sensors, support_contract, support_rows,
)


TASKS = {
    "blind": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0",
    "teacher": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-v0",
    "student": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-v0",
    "student-temporal": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-v0",
}
SCENARIOS = ("pyramid_stairs", "pyramid_stairs_inv", "boxes", "hf_pyramid_slope", "gaps")


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_arg = parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", choices=tuple(TASKS), default="student")
    parser.add_argument("--scenario", choices=SCENARIOS, default="pyramid_stairs")
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes_per_env", type=int, default=2)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--depth_mode", choices=("live", "blank", "frozen"), default="live")
    parser.add_argument("--camera_corruption", action="store_true", help="开启Temporal深度的延迟/丢帧/冻结测试。")
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    checkpoint_arg.required = True
    args = parser.parse_args()
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("difficulty 必须在 [0,1] 内")
    if args.num_envs < 1 or args.episodes_per_env < 1:
        parser.error("环境数与每环境配额必须为正")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint 不存在：{args.checkpoint}")
    if not args.policy.startswith("student") and args.depth_mode != "live":
        parser.error("深度失效对照仅适用于 student")
    if args.camera_corruption and args.policy != "student-temporal":
        parser.error("camera_corruption只适用于student-temporal")
    if args.policy.startswith("student"):
        args.enable_cameras = True
    return args


def capture_traversal(env, *, scenario: str):
    """在物理步末、自动 reset 前记录通行状态；供实际终止项和回归测试共用。"""
    import torch

    if scenario not in SCENARIOS:
        raise ValueError(f"未知视觉评测地形：{scenario}")
    robot = env.scene["robot"]
    relative = robot.data.root_pos_w - env.scene.env_origins
    finite = torch.isfinite(robot.data.root_state_w).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_pos).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel).all(dim=1)
    finite &= torch.isfinite(relative).all(dim=1)
    finite &= torch.isfinite(robot.data.projected_gravity_b).all(dim=1)
    finite &= torch.isfinite(robot.data.root_lin_vel_b).all(dim=1)
    hits = env.scene.sensors["height_scanner_base"].data.ray_hits_w
    hit_finite = torch.isfinite(hits).all(dim=-1)
    # 无底沟的正常射线未命中是 xyz 全 +inf；NaN、-inf 或半损坏坐标仍拒绝。
    hit_valid = hit_finite | (torch.isposinf(hits).all(dim=-1) & (scenario == "gaps"))
    finite &= hit_valid.all(dim=-1)
    support = capture_ground_support(env)
    finite &= support["valid"]
    upright = robot.data.projected_gravity_b[:, 2] < -0.5
    in_lane = (relative[:, 1].abs() <= 1.0) & (relative[:, 0] <= 3.75)
    fallen = ~finite | ~upright | ~in_lane
    # 楼梯/坡面的 origin 是地形中心，正常下行也会低于它；不能当作局部地面。
    # 窄沟两岸与中心平台等高，必须保留平台高度门，防止沿沟底行驶也算通过。
    if scenario == "gaps":
        fallen |= relative[:, 2] < -0.2
    # 支撑只约束末端保持，不因跨沟腾空或越障时短暂非轮接触提前终止。
    stable = (relative[:, 0] >= 2.5) & ~fallen & hit_finite.all(dim=-1) & support["supported"]
    squared_error = torch.square(robot.data.root_lin_vel_b[:, 0] - 0.4)
    env._vision_eval_sample = (
        stable.cpu().tolist(), finite.cpu().tolist(),
        relative[:, 0].cpu().tolist(), squared_error.cpu().tolist(),
    )
    env._vision_eval_support = support_rows(support)
    return fallen


def main(args, simulation_app):
    import gymnasium as gym
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.managers import TerminationTermCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import (
        _configure_vision_terrain,
        vision_depth_contract,
    )

    task = TASKS[args.policy]
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    # 三种策略在相同摩擦和地形下比较；blind 训练时的特殊摩擦不带入验收。
    _configure_vision_terrain(cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = agent_cfg.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = 8.0
    cfg.observations.policy.enable_corruption = False
    if args.camera_corruption:
        # 只保留自管深度通信扰动，消除本体观测噪声对消融的混淆。
        for name, term in vars(cfg.observations.policy).items():
            if name != "depth_image" and hasattr(term, "noise"):
                term.noise = None
        cfg.observations.policy.enable_corruption = True
    terrain = cfg.scene.terrain.terrain_generator
    selected = terrain.sub_terrains[args.scenario]
    selected.proportion = 1.0
    terrain.sub_terrains = {args.scenario: selected}
    terrain.num_rows = 1
    terrain.num_cols = 4
    terrain.difficulty_range = (args.difficulty, args.difficulty)
    terrain.curriculum = False
    terrain.seed = args.seed
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = None
    for name in ("randomize_push_robot", "push_robot", "randomize_apply_external_force_torque"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {}
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (0.4, 0.4)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.resampling_time_range = (10.0, 10.0)
    support_sensors = configure_support_sensors(cfg)

    tracker = SkillEvaluation(args.num_envs, args.episodes_per_env, cfg.sim.dt * cfg.decimation, 0.5, 7.5)
    tracking_error_sum = 0.0
    tracking_samples = 0
    terminal_progress = []
    terminal_support = []

    cfg.terminations.traversal_failure = TerminationTermCfg(
        func=capture_traversal, params={"scenario": args.scenario},
    )
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    try:
        from pxr import UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage

        ground_prim = get_current_stage().GetPrimAtPath(support_sensors["terrain_filter"])
        if not ground_prim.IsValid() or not ground_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("视觉承重过滤路径不存在或不具有 CollisionAPI")
        runner_class = DistillationRunner if args.policy.startswith("student") else OnPolicyRunner
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.reset()
        frozen_depth = obs["policy"][:, 45:].clone() if args.policy.startswith("student") else None
        while not tracker.complete and simulation_app.is_running():
            with torch.inference_mode():
                if args.policy.startswith("student") and args.depth_mode != "live":
                    obs = obs.clone()
                    obs["policy"][:, 45:] = 1.0 if args.depth_mode == "blank" else frozen_depth
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("策略输出非有限动作，终止评测")
                obs, _, dones, _ = env.step(actions)
            stable, finite, progress, errors = env.unwrapped._vision_eval_sample
            terminated = env.unwrapped.reset_terminated.cpu().tolist()
            truncated = env.unwrapped.reset_time_outs.cpu().tolist()
            for index in range(args.num_envs):
                if tracker.counts[index] >= tracker.quota:
                    continue
                if finite[index] and math.isfinite(errors[index]):
                    tracking_error_sum += errors[index]
                    tracking_samples += 1
                if terminated[index] or truncated[index]:
                    terminal_progress.append(progress[index] if math.isfinite(progress[index]) else None)
                    terminal_support.append({"env_id": index, "episode_index": tracker.counts[index],
                                             **env.unwrapped._vision_eval_support[index]})
            tracker.update(stable, terminated, truncated, finite)
            runner.alg.policy.reset(dones)
            if frozen_depth is not None:
                reset_ids = dones.bool()
                frozen_depth[reset_ids] = obs["policy"][reset_ids, 45:]
    finally:
        env.close()

    result = tracker.summary(0.85)
    with args.checkpoint.open("rb") as checkpoint_file:
        checksum = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result.update({
        "protocol": TRAVERSAL_PROTOCOL,
        "task": task, "policy": args.policy, "scenario": args.scenario,
        "difficulty": args.difficulty, "seed": args.seed, "depth_mode": args.depth_mode,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checksum,
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
        "velocity_rmse_m_s": math.sqrt(tracking_error_sum / tracking_samples) if tracking_samples else None,
        "terminal_progress_m": terminal_progress,
        "target_distance_m": 2.5, "command_velocity_m_s": 0.4,
        "episode_length_s": 8.0, "step_dt": tracker.dt,
        "terrain_height_gate": "platform_relative_z_below_-0.2m" if args.scenario == "gaps" else "none",
        "requires_finite_base_height_scan": args.scenario != "gaps",
        "requires_finite_terminal_base_height_scan": True,
        "terminal_wheel_support_verified": tracker.complete,
        "terminal_support_contract": support_contract(),
        "support_sensor_configuration": support_sensors,
        "terminal_support": terminal_support,
        "base_height_scan_policy": "finite_xyz_or_all_positive_inf_during_gap_crossing; finite_xyz_at_terminal",
        "scope": "fixed_lane_traversal; does not establish general navigation or parkour",
        "camera_corruption": args.camera_corruption,
        "depth_contract": vision_depth_contract(cfg) if args.policy.startswith("student") else None,
    })
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    args = parse_args()
    app = AppLauncher(args).app
    exit_code = 2
    try:
        result = main(args, app)
        exit_code = 0 if result["passed"] else 1
    finally:
        # 显式通知 Kit，避免 fast shutdown 把失败评测退出码覆盖为 0。
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(exit_code)
        app.close()
    raise SystemExit(exit_code)
