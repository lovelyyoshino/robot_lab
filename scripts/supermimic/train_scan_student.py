#!/usr/bin/env python3
"""
train_scan_student.py — 深度→高度扫描 latent 学生：冻结 Teacher-Preview actor，只训深度编码器。

数据：在 Student-Preview 任务里，按回合以衰减概率由 teacher（干净 232 维观测）或学生驱动，
每轮采 `--steps_per_iter` 步，在这批样本上做若干 epoch 小批量更新。
产物：logs/supermimic_scan/<run>/ 下 ckpt_XXXXXX.pt（编码器 + 配置）与 exported/policy.pt
（TorchScript，输入 4141 维，与现有 Student 部署接口相同），以及深度合同 depth_contract.json。

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

parser = argparse.ArgumentParser(description="Depth → height-scan latent student")
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0")
parser.add_argument("--teacher", required=True, help="Teacher-Preview rsl_rl checkpoint（model_XXXX.pt）")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=4000)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume", default=None)
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--action_loss_coef", type=float, default=1.0)
parser.add_argument("--teacher_drive_decay_updates", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import scan_student as ss  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import (  # noqa: E402
    vision_depth_contract,
)


def linear(start, end, progress, horizon):
    return end if horizon <= 0 else start + (end - start) * min(progress / horizon, 1.0)


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    teacher_ckpt = torch.load(args_cli.teacher, map_location=device, weights_only=False)
    teacher = ss.teacher_actor_from_rsl_rl(teacher_ckpt["model_state_dict"]).to(device)
    encoder = ss.DepthScanEncoder(args_cli.latent_dim).to(device)
    policy = ss.ScanStudentPolicy(encoder, teacher)
    cfg = ss.ScanStudentCfg(lr=args_cli.lr, action_loss_coef=args_cli.action_loss_coef,
                            teacher_drive_decay_updates=args_cli.teacher_drive_decay_updates)
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    start_iter = 0
    if args_cli.resume:
        data = torch.load(args_cli.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(data["encoder"])
        opt.load_state_dict(data["optimizer"])
        start_iter = data["iteration"]

    run = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs/supermimic_scan") / run
    (log_dir / "exported").mkdir(parents=True, exist_ok=True)
    (log_dir / "depth_contract.json").write_text(json.dumps(vision_depth_contract(env.unwrapped.cfg), indent=2))
    writer = SummaryWriter(str(log_dir))
    csv_file = open(log_dir / "train_log.csv", "a", newline="")
    csv_writer = None

    obs, _ = env.reset()
    if obs["policy"].shape[-1] != ss.STUDENT_OBS_DIM or obs["teacher"].shape[-1] != ss.TEACHER_OBS_DIM:
        raise RuntimeError(f"观测维度不符: policy={obs['policy'].shape}, teacher={obs['teacher'].shape}")
    n = args_cli.num_envs
    teacher_driven = torch.rand(n, device=device) < cfg.teacher_drive_start
    for it in range(start_iter, args_cli.iterations):
        p_teacher = linear(cfg.teacher_drive_start, cfg.teacher_drive_end, it, cfg.teacher_drive_decay_updates)
        student_buf, teacher_buf, rewards = [], [], []
        with torch.no_grad():
            for _ in range(args_cli.steps_per_iter):
                s_obs, t_obs = obs["policy"], obs["teacher"]
                student_buf.append(s_obs)
                teacher_buf.append(t_obs)
                # 回合级驱动者：同一回合内不切换，避免拼出两个策略都不会访问的状态。
                action = torch.where(teacher_driven[:, None], teacher(t_obs), policy(s_obs))
                obs, rew, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                rewards.append(rew[~teacher_driven].mean() if (~teacher_driven).any() else rew.new_zeros(()))
                if done.any():
                    teacher_driven[done] = torch.rand(int(done.sum()), device=device) < p_teacher
        student_data, teacher_data = torch.cat(student_buf), torch.cat(teacher_buf)

        logs = {"loss": 0.0, "scan_mse": 0.0, "action_mse": 0.0}
        updates = 0
        for _ in range(cfg.epochs):
            for idx in torch.randperm(student_data.shape[0], device=device).split(cfg.minibatch):
                loss, scan_loss, action_loss = ss.student_losses(policy, student_data[idx], teacher_data[idx], cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
                opt.step()
                logs["loss"] += loss.item()
                logs["scan_mse"] += scan_loss.item()
                logs["action_mse"] += action_loss.item()
                updates += 1
        logs = {k: v / max(updates, 1) for k, v in logs.items()}
        logs["teacher_drive_p"] = p_teacher
        logs["student_step_reward"] = torch.stack(rewards).mean().item()

        if it % args_cli.log_interval == 0:
            for k, v in logs.items():
                writer.add_scalar(f"scan_student/{k}", v, it)
            row = {"iteration": it, **logs}
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
                if csv_file.tell() == 0:
                    csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()
            print(f"[it {it}] " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()), flush=True)
        if (it + 1) % args_cli.save_interval == 0 or it + 1 == args_cli.iterations:
            torch.save({"iteration": it + 1, "encoder": encoder.state_dict(), "optimizer": opt.state_dict(),
                        "cfg": asdict(cfg), "latent_dim": args_cli.latent_dim, "teacher": str(args_cli.teacher)},
                       log_dir / f"ckpt_{it + 1:06d}.pt")
            torch.jit.script(copy.deepcopy(policy).cpu().eval()).save(str(log_dir / "exported" / "policy.pt"))

    csv_file.close()
    writer.close()
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
