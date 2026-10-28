#!/usr/bin/env python3
"""
verify_scan.py — 训练前校验深度→扫描学生链路的观测与 Teacher 重建，不产出策略。

1. teacher_actor_from_rsl_rl 重建的 actor 与 rsl_rl 导出的 TorchScript（exported/policy.pt）逐元素一致；
2. 关闭 policy 噪声后，学生 4141 维前 45 列与 teacher 232 维前 45 列逐列一致（本体合同对齐）；
3. 深度段取值在 [-1, 1]，扫描段在 [-1, 1]，与 tanh 输出域一致；
4. teacher 纯驱动 N 步，报告每步奖励作为后续学生奖励的参照。

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

parser = argparse.ArgumentParser(description="Scan student pipeline verification")
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0")
parser.add_argument("--teacher", default="logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import scan_student as ss  # noqa: E402


def main():
    report = {}
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    ckpt = torch.load(args_cli.teacher, map_location=device, weights_only=False)
    teacher = ss.teacher_actor_from_rsl_rl(ckpt["model_state_dict"]).to(device)
    exported = torch.jit.load(str(Path(args_cli.teacher).parent / "exported" / "policy.pt"), map_location=device).eval()

    obs, _ = env.reset()
    s_obs, t_obs = obs["policy"], obs["teacher"]
    report["dims"] = {"student": s_obs.shape[-1], "teacher": t_obs.shape[-1]}
    max_proprio, max_teacher, rewards = 0.0, 0.0, []
    depth_rng, scan_rng = [1e9, -1e9], [1e9, -1e9]
    with torch.no_grad():
        for _ in range(args_cli.steps):
            s_obs, t_obs = obs["policy"], obs["teacher"]
            max_proprio = max(max_proprio, (s_obs[:, :45] - t_obs[:, :45]).abs().max().item())
            action = teacher(t_obs)
            max_teacher = max(max_teacher, (action - exported(t_obs)).abs().max().item())
            depth, scan = s_obs[:, 45:], t_obs[:, 45:]
            depth_rng = [min(depth_rng[0], depth.min().item()), max(depth_rng[1], depth.max().item())]
            scan_rng = [min(scan_rng[0], scan.min().item()), max(scan_rng[1], scan.max().item())]
            obs, rew, _, _, _ = env.step(action)
            rewards.append(rew.mean().item())
    report.update({
        "proprio_45_max_abs_diff_student_vs_teacher": max_proprio,
        "teacher_actor_vs_exported_max_abs_diff": max_teacher,
        "depth_range": depth_rng,
        "scan_range": scan_rng,
        "teacher_mean_step_reward": sum(rewards) / len(rewards),
    })
    print(json.dumps(report, indent=2))
    Path("logs/supermimic_scan").mkdir(parents=True, exist_ok=True)
    Path("logs/supermimic_scan/verify_scan.json").write_text(json.dumps(report, indent=2))
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
