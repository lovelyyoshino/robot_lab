#!/usr/bin/env python3
"""
probe_flat.py — 平地抖动探针：纯平地、固定命令下测车体上下起伏、俯仰抖动与 arm 来回摆动。

用于排查"平地上一直上下跳"：Preview 任务只在楼梯/障碍上验收，平地平稳性没有指标。
每个命令档单独跑一次（同一进程内改命令），前 3 s 视为起步瞬态不计。不改任何训练配置。

@author pony
@date 2026-10-27
@version v1.0.0
@last_modified 2026-10-27
@changelog
    - v1.0.1 (2026-10-28): 支持方案 B 策略（车道偏移观测）。
    - v1.0.0 (2026-10-27): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="平地抖动探针")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0")
parser.add_argument("--policy", action="append", required=True, help="name=path/to/exported/policy.pt，可重复")
parser.add_argument("--commands", type=float, nargs="+", default=[0.0, 0.15, 0.30])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic.isaac_adapter import RoverStateReader  # noqa: E402
from robot_lab.supermimic.quat import projected_gravity  # noqa: E402
from robot_lab.supermimic.state import ANG, JPOS, JVEL, LIN, POS, QUAT  # noqa: E402


def configure(env_cfg):
    terrain = env_cfg.scene.terrain.terrain_generator
    flat = terrain.sub_terrains["flat"].copy()
    flat.proportion = 1.0
    terrain.sub_terrains = {"flat": flat}
    terrain.num_rows, terrain.num_cols = 1, min(args_cli.num_envs, 16)
    terrain.curriculum = False
    terrain.seed = args_cli.seed
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum = None
    for name in ("preview_success", "bad_orientation"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)
    env_cfg.episode_length_s = args_cli.seconds + 5.0
    env_cfg.events.randomize_reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.randomize_reset_base.params["velocity_range"] = {}
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (0.0, 0.0), (0.0, 0.0)
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1e3, 1e3)


def run(env, reader, policy, command):
    uenv = env.unwrapped
    obs, _ = env.reset()
    uenv.command_manager.get_term("base_velocity").vel_command_b[:, 0] = command
    obs = uenv.observation_manager.compute()
    steps = int(args_cli.seconds / uenv.step_dt)
    settle = int(args_cli.settle_s / uenv.step_dt)
    rec = {k: [] for k in ("z", "vz", "pitch_rate", "arm_vel", "arm_pos", "arm_act", "tilt")}
    for t in range(steps):
        uenv.command_manager.get_term("base_velocity").vel_command_b[:, 0] = command
        with torch.no_grad():
            action = policy(obs["policy"])
        obs, _, _, _, _ = env.step(action)
        if t < settle:
            continue
        s = reader.state()
        rec["z"].append(s[:, POS][:, 2])
        rec["vz"].append(s[:, LIN][:, 2])
        rec["pitch_rate"].append(s[:, ANG][:, 1])
        rec["arm_vel"].append(s[:, JVEL][:, 0:4])
        rec["arm_pos"].append(s[:, JPOS][:, 0:4])
        rec["arm_act"].append(action[:, 0:4])
        rec["tilt"].append(torch.acos((-projected_gravity(s[:, QUAT])[:, 2]).clamp(-1, 1)))
    r = {k: torch.stack(v) for k, v in rec.items()}  # [T, N, ...]
    da = (r["arm_act"][1:] - r["arm_act"][:-1]).abs()
    # 主频：车体 vz 的过零次数 / 2 / 时长
    zc = ((r["vz"][1:] * r["vz"][:-1]) < 0).float().sum(0) / 2 / (r["vz"].shape[0] * uenv.step_dt)
    return {
        "command": command,
        "base_z_std_cm": (r["z"].std(0) * 100).median().item(),
        "base_z_ptp_cm": ((r["z"].max(0).values - r["z"].min(0).values) * 100).median().item(),
        "vz_rms_cm_s": (r["vz"].square().mean(0).sqrt() * 100).median().item(),
        "vz_freq_hz": zc.median().item(),
        "pitch_rate_rms_deg_s": torch.rad2deg(r["pitch_rate"].square().mean(0).sqrt()).median().item(),
        "tilt_ptp_deg": torch.rad2deg(r["tilt"].max(0).values - r["tilt"].min(0).values).median().item(),
        "arm_vel_rms_rad_s": r["arm_vel"].square().mean((0, 2)).sqrt().median().item(),
        "arm_pos_ptp_deg": torch.rad2deg((r["arm_pos"].max(0).values - r["arm_pos"].min(0).values).amax(-1)).median().item(),
        "arm_action_delta_med": da.median().item(),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    configure(env_cfg)
    from robot_lab.supermimic.lane_offset import add_lane_offset_term, policy_uses_lane_offset

    specs = [spec.split("=", 1) for spec in args_cli.policy]
    lane = {name: policy_uses_lane_offset(path) for name, path in specs}
    if any(lane.values()):
        add_lane_offset_term(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    reader = RoverStateReader(env, world_frame=False)
    out = {}
    for name, path in specs:
        scripted = torch.jit.load(path, map_location=env.unwrapped.device).eval()
        # 环境追加了车道偏移列时，不用该列的策略只取前 232 维
        policy = scripted if lane[name] or not any(lane.values()) else (lambda o, f=scripted: f(o[:, :-1]))
        out[name] = [run(env, reader, policy, c) for c in args_cli.commands]
        for row in out[name]:
            print(f"[flat] {name} " + " ".join(f"{k}={v:.3g}" for k, v in row.items()), flush=True)
    print("[flat_json]", json.dumps(out), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    simulation_app.close()
