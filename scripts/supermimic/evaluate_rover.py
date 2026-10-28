#!/usr/bin/env python3
"""
evaluate_rover.py — SuperMimic checkpoint 验收：确定性策略的速度跟踪 + 世界模型开环误差。

开环误差是 plan.md 第 ③ 步的过门指标：在真实轨迹上，从第 t 步状态出发只喂真实动作，
用世界模型推 k 步，与仿真第 t+k 步比较（不跨回合）。报物理单位：位置 m、线速度 m/s、
倾角 rad、轮速 rad/s，并给出常速度基线作对照。结果打印并写 JSON 到 checkpoint 同级目录。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-10-26
@changelog
    - v0.2.0 (2026-10-26): 开环误差改用 diagnostics 共用实现，新增常速度基线对照。
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SuperMimic rover evaluation")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0")
parser.add_argument("--ckpt", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32])
parser.add_argument("--seed", type=int, default=7)
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
from robot_lab.supermimic import rover_task  # noqa: E402
from robot_lab.supermimic.isaac_adapter import RoverStateReader  # noqa: E402
from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from robot_lab.supermimic.diagnostics import open_loop_errors  # noqa: E402
from robot_lab.supermimic.state import ANG, LIN, constant_velocity_step  # noqa: E402


def main():
    torch.manual_seed(args_cli.seed)
    data = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = dict(data["cfg"])
    cfg_dict["reward"] = rover_task.RoverRewardCfg(**cfg_dict["reward"])
    cfg = SuperMimicCfg(**{**cfg_dict, "num_envs": args_cli.num_envs, "buffer_len": 2})
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    reader = RoverStateReader(env)
    device = env.unwrapped.device
    learner = SuperMimicLearner(cfg, device)
    learner.load_state_dict(data, load_optimizers=False)

    env.reset()
    states, actions, dones, terms, lin_err, ang_err, contexts = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(args_cli.steps):
            s, cmd, prev = reader.state(), reader.command(), reader.last_action()
            a = learner.act(rover_task.policy_observation(s, cmd, prev), explore=False)
            _, _, terminated, truncated, _ = env.step(a)
            states.append(s)
            contexts.append(reader.context())
            actions.append(a)
            dones.append(terminated | truncated)
            terms.append(terminated)
            lin_err.append((cmd[:, :2] - s[:, LIN][:, :2]).norm(dim=-1))
            ang_err.append((cmd[:, 2] - s[:, ANG][:, 2]).abs())
        S, A, D = torch.stack(states), torch.stack(actions), torch.stack(dones)
        C = torch.stack(contexts) if cfg.context_dim else None

        result = {
            "ckpt": args_cli.ckpt,
            "iteration": data["iteration"],
            "lin_vel_err_mps": torch.stack(lin_err).mean().item(),
            "ang_vel_err_radps": torch.stack(ang_err).mean().item(),
            "terminations_per_env_minute": torch.stack(terms).float().sum().item()
            / args_cli.num_envs / (args_cli.steps * rover_task.CONTROL_DT / 60),
            "open_loop": {},
        }
        horizons = tuple(args_cli.horizons)
        result["open_loop"] = open_loop_errors(S, A, D, learner.world_model, horizons, contexts=C)
        # 世界模型必须明显优于"速度不变"基线，否则开环数字再小也没有意义。
        result["open_loop_constant_velocity_baseline"] = open_loop_errors(
            S, A, D, lambda s, a, c: constant_velocity_step(s, rover_task.CONTROL_DT), horizons)
    print(json.dumps(result, indent=2))
    out = Path(args_cli.ckpt).with_name(Path(args_cli.ckpt).stem + "_eval.json")
    out.write_text(json.dumps(result, indent=2))
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
