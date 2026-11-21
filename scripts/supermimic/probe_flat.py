#!/usr/bin/env python3
"""
probe_flat.py — 平地行驶探针：accept.py 的平地档（姿态、平顺、偏航/横向/速度跟踪）。

每个策略 × 种子 × 命令速度跑一遍；稳态从 --settle_s 开始统计。逐环境先在时间上取统计量，
输出 per_env 数组供 accept.py 跨种子合并后取 P95（验收规格：跨环境 P95，不用中位数）。

@author pony
@date 2026-10-27
@version v1.2.0
@last_modified 2026-11-05
@changelog
    - v1.2.0 (2026-11-05): 逐环境 roll/pitch/俯仰角速度/摆臂速度/垂向速度/动作变化率与偏航、横向、速度 RMSE；
      --seeds 多种子同进程；232/233 维按 actor 权重切换车道偏移；删除旧中位数抖动指标（旧 accept_terrain 专用）。
    - v1.1.0 (2026-11-02): 可选记录逐步物理轨迹和上下文，定位持续行驶的偏航、减速与模型误差。
    - (2026-11-02): 增加稳态偏航和横向偏移统计，避免起步端正掩盖持续转圈。
    - v1.0.3 (2026-11-02): 记录每个摆臂实际姿态和PD目标，区分单轮持续限位与整体统计。
    - v1.0.2 (2026-11-02): 起步/稳态姿态、路线速度与四轮独立接触力检查。
    - v1.0.1 (2026-10-28): 支持方案 B 策略（车道偏移观测）。
    - v1.0.0 (2026-10-27): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher
from pathlib import Path

parser = argparse.ArgumentParser(description="平地抖动探针")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0")
parser.add_argument("--policy", action="append", required=True, help="name=path/to/exported/policy.pt，可重复")
parser.add_argument("--commands", type=float, nargs="+", default=[0.0, 0.15, 0.30])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--seeds", type=int, nargs="+", default=None, help="多种子依次复位评测；默认只用 --seed")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--record_dir", type=Path, default=None,
                    help="保存各策略/速度的动作前状态、观测、动作和动力学上下文，供离线复核")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic.isaac_adapter import RoverStateReader  # noqa: E402
from robot_lab.supermimic.quat import projected_gravity, quat_apply  # noqa: E402
from robot_lab.supermimic.kinematics import RoverKinematics  # noqa: E402
from robot_lab.supermimic.state import ANG, JPOS, JVEL, LIN, POS, QUAT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reinforcement_learning/rsl_rl"))
from acceptance_metrics import LANE_OFFSET_OBS_DIM, actor_input_dim, quat_to_roll_pitch  # noqa: E402


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


def run(env, reader, policy, command, name, seed):
    uenv = env.unwrapped
    # 各策略/速度使用相同reset随机数，避免执行顺序改变增益随机化而污染对照。
    obs, _ = env.reset(seed=seed)
    uenv.command_manager.get_term("base_velocity").vel_command_b[:, 0] = command
    obs = uenv.observation_manager.compute()
    steps = int(args_cli.seconds / uenv.step_dt)
    settle = int(args_cli.settle_s / uenv.step_dt)
    rec = {k: [] for k in ("vz", "route_vx", "pitch_rate", "arm_vel", "arm_pos", "action", "tilt",
                           "roll", "pitch", "yaw", "lateral")}
    startup_tilt, support, support_loss = [], [], []
    trace = {k: [] for k in ("state", "obs", "action", "context", "command", "done")} if args_cli.record_dir else None
    kin = RoverKinematics().to(uenv.device)
    force_sensor = uenv.scene.sensors["contact_forces"]
    wheel_ids, _ = force_sensor.find_bodies(["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"], preserve_order=True)
    if len(wheel_ids) != 4:
        raise RuntimeError("平地四轮支撑检查需要四个轮子的独立接触力")
    for t in range(steps):
        uenv.command_manager.get_term("base_velocity").vel_command_b[:, 0] = command
        with torch.no_grad():
            action = policy(obs["policy"])
        if trace is not None:
            # 统一记录动作前的状态，避免用后继状态解释同一步动作或拟合错误的动力学转移。
            for key, value in (("state", reader.state()), ("obs", obs["policy"]), ("action", action),
                               ("context", reader.context()), ("command", reader.command())):
                trace[key].append(value.detach().cpu().clone())
        obs, _, terminated, truncated, _ = env.step(action)
        if trace is not None:
            trace["done"].append((terminated | truncated).cpu().clone())
        s = reader.state()
        tilt = torch.acos((-projected_gravity(s[:, QUAT])[:, 2]).clamp(-1, 1))
        if t * uenv.step_dt < 1.0:
            startup_tilt.append(tilt)
        if t < settle:
            continue
        jp = s[:, JPOS]
        wheels = kin.wheel_centers_world(s[:, POS], s[:, QUAT], jp[:, :4], jp[:, 4:8])
        support.append((wheels[:, :, 2] - 0.1085).clamp_min(0.0).amax(-1))
        support_loss.append((force_sensor.data.net_forces_w[:, wheel_ids, 2] <= 1.0).any(-1).float())
        rec["vz"].append(s[:, LIN][:, 2])
        rec["route_vx"].append(quat_apply(s[:, QUAT], s[:, LIN])[:, 0])
        rec["pitch_rate"].append(s[:, ANG][:, 1])
        rec["arm_vel"].append(s[:, JVEL][:, 0:4])
        rec["arm_pos"].append(s[:, JPOS][:, 0:4])
        rec["action"].append(action)
        rec["tilt"].append(tilt)
        q = s[:, QUAT]
        roll, pitch = quat_to_roll_pitch(q)
        rec["roll"].append(roll)
        rec["pitch"].append(pitch)
        rec["yaw"].append(torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                       1 - 2 * (q[:, 2].square() + q[:, 3].square())))
        rec["lateral"].append(s[:, POS][:, 1])
    r = {k: torch.stack(v) for k, v in rec.items()}  # [T, N, ...]
    if trace is not None:
        args_cli.record_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args_cli.record_dir / f"{name}_s{seed}_v{command:.2f}.npz",
                            **{k: torch.stack(v).numpy() for k, v in trace.items()},
                            origins=uenv.scene.env_origins.cpu().numpy(), step_dt=uenv.step_dt)
    deg = torch.rad2deg
    da = r["action"][1:] - r["action"][:-1]
    # 逐环境：先在时间上取统计量，accept.py 再跨环境/种子取 P95。
    per_env = {
        "roll_abs_p95_deg": deg(r["roll"].abs().quantile(0.95, dim=0)),
        "roll_abs_max_deg": deg(r["roll"].abs().amax(0)),
        "pitch_abs_p95_deg": deg(r["pitch"].abs().quantile(0.95, dim=0)),
        "pitch_abs_max_deg": deg(r["pitch"].abs().amax(0)),
        "pitch_rate_rms_deg_s": deg(r["pitch_rate"].square().mean(0).sqrt()),
        "arm_vel_rms_rad_s": r["arm_vel"].square().mean((0, 2)).sqrt(),
        "base_vz_rms_m_s": r["vz"].square().mean(0).sqrt(),
        "action_delta_rms": da.square().mean((0, 2)).sqrt(),
        "arm_action_delta_rms": da[..., :4].square().mean((0, 2)).sqrt(),
        "action_rate_rms_per_s": da.square().mean((0, 2)).sqrt() / uenv.step_dt,
        "yaw_abs_p95_deg": deg(r["yaw"].abs().quantile(0.95, dim=0)),
        "lateral_abs_p95_m": r["lateral"].abs().quantile(0.95, dim=0),
        "route_velocity_rmse_m_s": (r["route_vx"] - command).square().mean(0).sqrt(),
    }
    per_env = {k: v.cpu().tolist() for k, v in per_env.items()}
    return {
        "command": command, "seed": seed, "envs": len(per_env["roll_abs_p95_deg"]), "per_env": per_env,
        "p95_across_envs": {k: float(np.percentile(v, 95)) for k, v in per_env.items()},
        # 以下为诊断，不设门
        "route_velocity_mean_m_s": r["route_vx"].mean().item(),
        "startup_tilt_p95_deg": torch.rad2deg(torch.stack(startup_tilt)).quantile(0.95).item(),
        "tilt_p95_deg": torch.rad2deg(r["tilt"]).quantile(0.95).item(),
        "arm_pose_p95_deg": torch.rad2deg(r["arm_pos"].abs()).quantile(0.95).item(),
        "arm_pose_mean_deg_per_joint": torch.rad2deg(r["arm_pos"].mean((0, 1))).tolist(),
        "arm_target_mean_deg_per_joint": torch.rad2deg(0.25 * r["action"][..., :4].mean((0, 1))).tolist(),
        "wheel_clearance_p95_cm": (torch.stack(support) * 100).quantile(0.95).item(),
        "four_wheel_support_loss_fraction": torch.stack(support_loss).mean().item(),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    configure(env_cfg)
    specs = [spec.split("=", 1) for spec in args_cli.policy]
    # 233 维为旧车道偏移基线：环境追加该列；当前越障策略为 232 维。
    lane = {name: actor_input_dim(path) == LANE_OFFSET_OBS_DIM for name, path in specs}
    if any(lane.values()):
        from robot_lab.supermimic.lane_offset import add_lane_offset_term

        add_lane_offset_term(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    reader = RoverStateReader(env, world_frame=False, contact_context=args_cli.record_dir is not None)
    out = {}
    for name, path in specs:
        scripted = torch.jit.load(path, map_location=env.unwrapped.device).eval()
        # 环境追加了车道偏移列时，不用该列的策略只取前 232 维
        policy = scripted if lane[name] or not any(lane.values()) else (lambda o, f=scripted: f(o[:, :-1]))
        out[name] = [run(env, reader, policy, c, name, seed)
                     for seed in (args_cli.seeds or [args_cli.seed]) for c in args_cli.commands]
        for row in out[name]:
            print(f"[flat] {name} seed={row['seed']} v={row['command']:.2f} "
                  + " ".join(f"{k}={v:.3g}" for k, v in row["p95_across_envs"].items()), flush=True)
    print("[flat_json]", json.dumps(out), flush=True)
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(out, indent=2) + "\n")
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
