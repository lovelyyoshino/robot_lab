#!/usr/bin/env python3
"""
train_rover.py — 在 Isaac Lab 平地任务上训练平行四边形车的 SuperMimic（世界模型 + BPTT）策略。

每轮：仿真采 `--steps_per_iter` 步写回放 → 世界模型多步拟合 → 预热结束后想象 BPTT 更新 actor/critic。
`--check_obs` 每轮核对解析观测与仿真 45 维 policy 观测，任一列偏差都意味着想象与部署输入不一致。
`--driver` 让一部分环境由固定 PPO 策略驱动，持续给世界模型提供 actor 分布之外的数据。
日志写在 logs/supermimic/<run>/：train_log.csv、TensorBoard、ckpt_XXXXXX.pt、exported/policy.pt。
重点看 wm_reward_bias：持续为正说明模型高估奖励，actor 正在钻模型空子。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-10-26
@changelog
    - v0.2.0 (2026-10-26): 写入 ξ 与关节加速度真值；新增 driver 混合采样与 actor 更新间隔参数。
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SuperMimic rover training")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument("--steps_per_iter", type=int, default=1, help="每轮仿真步数；论文 UTD 0.4 ≈ 2.5 步/轮")
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume", default=None, help="checkpoint 路径；继续迭代号、网络与优化器")
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--check_obs", action="store_true", help="每轮核对解析观测与仿真观测")
parser.add_argument("--seed", type=int, default=42)
# 以下覆盖 SuperMimicCfg；省略时用默认值
for name, typ in (("buffer_len", int), ("horizon", int), ("imag_batch", int), ("wm_batch", int),
                  ("wm_updates", int), ("warmup_iterations", int), ("actor_lr", float), ("wm_lr", float),
                  ("action_noise", float), ("warmup_action_noise", float), ("actor_update_every", int),
                  ("aux_loss_coef", float)):
    parser.add_argument(f"--{name}", type=typ, default=None)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=None)
parser.add_argument("--driver", default=None, help="可选 45→12 TorchScript 数据采集策略（如 rough blind PPO 导出）")
parser.add_argument("--driver_fraction", type=float, default=0.1, help="由 driver 驱动的环境比例")
parser.add_argument("--driver_noise", type=float, default=0.3)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import csv  # noqa: E402
import time  # noqa: E402
from dataclasses import fields  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import rover_task  # noqa: E402
from robot_lab.supermimic.isaac_adapter import CONTEXT_DIM, RoverStateReader  # noqa: E402
from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from robot_lab.supermimic.state import LIN, ANG  # noqa: E402


def build_cfg() -> SuperMimicCfg:
    cfg = SuperMimicCfg(num_envs=args_cli.num_envs, context_dim=CONTEXT_DIM, aux_dim=rover_task.AUX_DIM)
    for f in fields(cfg):
        value = getattr(args_cli, f.name, None)
        if value is not None:
            setattr(cfg, f.name, tuple(value) if isinstance(value, list) else value)
    return cfg


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    reader = RoverStateReader(env)
    device = env.unwrapped.device

    learner = SuperMimicLearner(build_cfg(), device)
    if args_cli.resume:
        learner.load_state_dict(torch.load(args_cli.resume, map_location=device, weights_only=False))
        print(f"[SuperMimic] 续训 {args_cli.resume} @ iteration {learner.iteration}")
    run = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs/supermimic") / run
    (log_dir / "exported").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    csv_file = open(log_dir / "train_log.csv", "a", newline="")
    csv_writer = None

    obs, _ = env.reset()
    min_rows = learner.cfg.wm_window + 2
    n_envs = args_cli.num_envs
    # driver 环境：由固定 PPO 策略 + 噪声驱动，只为世界模型持续提供"actor 之外"的数据，
    # 不计入 actor 的仿真指标；actor 指标只看其余环境。
    n_driver = int(round(args_cli.driver_fraction * n_envs)) if args_cli.driver else 0
    driver_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    driver_mask[:n_driver] = True
    actor_mask = ~driver_mask
    driver = torch.jit.load(args_cli.driver, map_location=device).eval() if n_driver else None
    if n_driver:
        print(f"[SuperMimic] {n_driver}/{n_envs} 个环境由 driver 驱动（噪声 {args_cli.driver_noise}）")
    episode_return = torch.zeros(n_envs, device=device)
    finished_returns, start_time = [], time.time()
    start_iter = learner.iteration
    for it in range(start_iter, args_cli.iterations):
        sim = {"env_reward": 0.0, "terminated": 0.0, "lin_err": 0.0, "ang_err": 0.0, "obs_max_err": 0.0}
        for _ in range(args_cli.steps_per_iter):
            s, cmd, prev, ctx = reader.state(), reader.command(), reader.last_action(), reader.context()
            policy_obs = rover_task.policy_observation(s, cmd, prev)
            if args_cli.check_obs:
                sim["obs_max_err"] = max(sim["obs_max_err"], (policy_obs - obs["policy"]).abs().max().item())
            action = learner.act(policy_obs)
            if n_driver:
                with torch.no_grad():
                    drive = driver(policy_obs[:n_driver])
                    drive = drive + args_cli.driver_noise * torch.randn_like(drive)
                bound = learner.cfg.action_bound
                action[:n_driver] = drive.clamp(-bound, bound)
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            learner.observe(s, action, prev, cmd, done, ctx, reader.acc_aux())
            episode_return += rew
            if (done & actor_mask).any():
                finished_returns += episode_return[done & actor_mask].tolist()
            episode_return[done] = 0.0
            a = actor_mask
            sim["env_reward"] += rew[a].mean().item() / args_cli.steps_per_iter
            sim["terminated"] += terminated[a].float().mean().item() / args_cli.steps_per_iter
            sim["lin_err"] += (cmd[a, :2] - s[a][:, LIN][:, :2]).norm(dim=-1).mean().item() / args_cli.steps_per_iter
            sim["ang_err"] += (cmd[a, 2] - s[a][:, ANG][:, 2]).abs().mean().item() / args_cli.steps_per_iter

        if learner.replay.size >= min_rows:
            logs = learner.update()
        else:
            # 回放尚不足一个窗口时只采样；迭代号照常推进，保证续训与 checkpoint 编号一致。
            logs = {}
            learner.iteration += 1
        logs.update(sim)
        if finished_returns:
            logs["episode_return"] = sum(finished_returns) / len(finished_returns)
            finished_returns.clear()
        if it % args_cli.log_interval == 0:
            logs["elapsed_min"] = (time.time() - start_time) / 60
            for k, v in logs.items():
                writer.add_scalar(f"supermimic/{k}", v, it)
            row = {"iteration": it, **logs}
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(set(row) | {
                    "wm_loss", "wm_err_1step", "wm_err_last", "wm_aux_loss", "wm_reward_bias", "wm_reward_abs_gap",
                    "actor_loss", "critic_loss", "imag_reward", "imag_survival", "episode_return",
                    "actor_skipped"}), extrasaction="ignore")
                if csv_file.tell() == 0:
                    csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()
            print(f"[it {it}] " + " ".join(f"{k}={v:.4g}" for k, v in sorted(logs.items())), flush=True)
        if (it + 1) % args_cli.save_interval == 0 or it + 1 == args_cli.iterations:
            torch.save(learner.state_dict(), log_dir / f"ckpt_{it + 1:06d}.pt")
            learner.export_actor(log_dir / "exported" / "policy.pt")

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
