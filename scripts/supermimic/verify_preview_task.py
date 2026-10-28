#!/usr/bin/env python3
"""
verify_preview_task.py — 越障版 SuperMimic 训练前的合同核对（第 3 步的前置门），不产出策略。

在 SuperMimic-Terrain-Preview 任务上由 Teacher-Preview 驱动，逐步核对：
1. 想象观测（PreviewTask.policy_obs，直通扫描）vs env 232 维 policy 观测；
2. 解析奖励各项 vs RewardManager 同名项（track/lin_vel_z/ang_vel_xy/action_rate/steering/joint_pos_limits/
   flat_orientation/yaw_rate）；anticipatory_wheel_lift 无法复现，只报告它与几何塑形 wheel_lift 的相关性；
3. 从 Teacher 热启动的 actor（clamp 输出）与 Teacher 导出 TorchScript 在同一观测上的动作差。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify SuperMimic terrain task contract")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0")
parser.add_argument("--teacher", default="logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--output", default="logs/supermimic/verify_preview_task.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic.isaac_adapter import (  # noqa: E402
    RoverStateReader, build_height_map, exact_scan_fn, leg_soft_limits, scan_pattern_base)
from robot_lab.supermimic.kinematics import RoverKinematics  # noqa: E402
from robot_lab.supermimic.networks import Actor  # noqa: E402
from robot_lab.supermimic.preview_task import PreviewTask  # noqa: E402

NAME_MAP = {
    "track_lin_vel_xy_exp": "track_lin_vel", "lin_vel_z_l2": "lin_vel_z", "ang_vel_xy_l2": "ang_vel_xy",
    "action_rate_l2": "action_rate", "joint_deviation_steering_l1": "steering_deviation",
    "joint_pos_limits": "joint_pos_limits", "flat_orientation_l2": "flat_orientation",
    "preview_yaw_rate_l2": "yaw_rate",
}


def warm_start_actor(teacher_ckpt: str, device) -> Actor:
    sd = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    actor = Actor(sd["actor.0.weight"].shape[1], (512, 256, 128), squash="clamp")
    actor.load_state_dict({k: v for k, v in sd.items() if k.startswith("actor.")})
    return actor.to(device).eval()


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    device = uenv.device
    reader = RoverStateReader(env, world_frame=True)
    kin = RoverKinematics().to(device)
    hmap = build_height_map(env)
    pattern = scan_pattern_base(env)
    task = PreviewTask(hmap, kin, pattern, leg_soft_limits(reader), exact_scan_fn=exact_scan_fn(env, pattern))
    rm = uenv.reward_manager
    names = list(rm.active_terms)
    weights = {n: rm.get_term_cfg(n).weight for n in names}
    ours_weights = {n: getattr(task.cfg, NAME_MAP[n]) for n in NAME_MAP if n in names}
    # joint_pos_limits 的 joint selector 必须恰好是 8 个腿关节，解析版只算这 8 个。
    jpl_ids = rm.get_term_cfg("joint_pos_limits").params["asset_cfg"].joint_ids
    leg_ids = reader.joint_ids[:8].tolist()
    actor = warm_start_actor(args_cli.teacher, device)
    teacher = torch.jit.load(str(Path(args_cli.teacher).parent / "exported" / "policy.pt"), map_location=device).eval()

    obs, _ = env.reset()
    obs_err = {"proprio": 0.0, "scan": 0.0, "scan_smooth_p99": 0.0}
    reward_err = {n: 0.0 for n in NAME_MAP if n in names}
    act_err, n_kept = 0.0, 0
    lift_env, lift_ours = [], []
    for step in range(args_cli.steps):
        s, prev = reader.state(), reader.last_action()
        cmd = uenv.command_manager.get_command("base_velocity").clone()
        with torch.no_grad():
            ours_obs = task.policy_obs(s, cmd, prev)
            obs_err["proprio"] = max(obs_err["proprio"], (ours_obs[:, :45] - obs["policy"][:, :45]).abs().max().item())
            obs_err["scan"] = max(obs_err["scan"], (ours_obs[:, 45:] - obs["policy"][:, 45:]).abs().max().item())
            if step % 20 == 0:
                from robot_lab.supermimic.terrain_map import height_scan
                d = (height_scan(hmap, s, pattern) - obs["policy"][:, 45:]).abs().flatten()
                obs_err["scan_smooth_p99"] = max(obs_err["scan_smooth_p99"], torch.quantile(d, 0.99).item())
            action = actor(obs["policy"])
            act_err = max(act_err, (action - teacher(obs["policy"]).clamp(-actor.action_bound, actor.action_bound))
                          .abs().max().item())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        keep = ~done & (uenv.command_manager.get_command("base_velocity") == cmd).all(-1)
        if keep.any() and step > 0:
            with torch.no_grad():
                terms = task.reward_terms(s, reader.state(), cmd, action, prev)
            for env_name, ours in NAME_MAP.items():
                if env_name in names:
                    ref = rm._step_reward[:, names.index(env_name)]
                    reward_err[env_name] = max(reward_err[env_name], (terms[ours] - ref)[keep].abs().max().item())
            lift_env.append(rm._step_reward[:, names.index("anticipatory_wheel_lift")][keep])
            lift_ours.append(terms["wheel_lift"][keep])
            n_kept += 1
    le, lo = torch.cat(lift_env), torch.cat(lift_ours)
    report = {
        "obs_max_abs_err": obs_err,
        "reward_term_max_abs_err": reward_err,
        "weights_env_vs_ours": {n: [weights[n], ours_weights[n]] for n in ours_weights},
        "joint_pos_limits_selector_is_legs": sorted(int(i) for i in (jpl_ids if not isinstance(jpl_ids, slice) else []))
        == sorted(leg_ids),
        "warm_start_action_max_abs_err": act_err,
        "wheel_lift_shaping": {"env_anticipatory_mean": le.mean().item(), "ours_mean": lo.mean().item(),
                               "corr": torch.corrcoef(torch.stack((le, lo)))[0, 1].item(),
                               "ours_positive_fraction": (lo > 1e-3).float().mean().item()},
        "steps_compared": n_kept,
    }
    Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # 异常后调用 simulation_app.close() 会卡死且吞掉报错（实测），先打印再直接退出进程。
        import os
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
