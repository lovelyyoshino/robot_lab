#!/usr/bin/env python3
"""
verify_terrain.py — SuperMimic 地形版的开发实验：plan 第 1、2 步的过门检查，不产出策略。

第 1 步（一致性）：
  scan — 高度图重建的 187 点扫描 vs Teacher 观测里 RayCaster 的真实扫描（逐射线，分"台阶边/非台阶边"）；
  fk   — 可微 FK 轮心 vs 仿真 wheel 连杆 + 碰撞圆柱偏移；同时报 mimic 从动关节对 arm 的跟随误差。
第 2 步（世界模型过门）：Teacher-Preview 策略 + 分档噪声在 8/12/16 cm 楼梯与单台阶上采数据，
  离线训练"无地形输入"与"有地形输入（relief_features）"两个世界模型，在留出环境上分"台阶边起点 / 平地起点"
  比较 k=1/8/32 开环误差（含轮心高度、俯仰），并与常速度基线对照。
过门标准在运行前写死在 GATES 里，结果写入 logs/supermimic/verify_terrain.json。

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

parser = argparse.ArgumentParser(description="SuperMimic terrain dev experiment")
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0")
parser.add_argument("--teacher_policy",
                    default="logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/exported/policy.pt")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--heldout_envs", type=int, default=256)
parser.add_argument("--collect_steps", type=int, default=1000)
parser.add_argument("--wm_updates", type=int, default=6000)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=[1024, 1024, 1024])
parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.3])
parser.add_argument("--map_res", type=float, default=0.02)
parser.add_argument("--output", default="logs/supermimic/verify_terrain.json")
parser.add_argument("--seed", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import rover_task, terrain_map  # noqa: E402
from robot_lab.supermimic.diagnostics import open_loop_errors  # noqa: E402
from robot_lab.supermimic.isaac_adapter import (  # noqa: E402
    CONTEXT_DIM, RoverStateReader, build_height_map, scan_pattern_base)
from robot_lab.supermimic.kinematics import CORNERS, RoverKinematics  # noqa: E402
from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from robot_lab.supermimic.quat import projected_gravity, quat_apply  # noqa: E402
from robot_lab.supermimic.state import JPOS, POS, QUAT, constant_velocity_step  # noqa: E402

# 运行前写死的过门标准；不根据结果事后调整。
GATES = {
    "scan_nonedge_within_5mm_fraction_min": 0.98,
    "fk_max_err_m_max": 1e-3,
    "wm_edge_k8_wheel_z_vs_no_relief_ratio_max": 0.7,
    "wm_edge_k8_wheel_z_vs_const_vel_ratio_max": 0.5,
    "wm_edge_k8_pitch_vs_const_vel_ratio_max": 0.5,
    "wm_flat_k8_wheel_z_vs_no_relief_ratio_max": 1.1,
}


def configure(env_cfg):
    terrain = env_cfg.scene.terrain.terrain_generator
    stairs, step = terrain.sub_terrains["pyramid_stairs_inv"].copy(), terrain.sub_terrains["single_step"].copy()
    stairs.proportion, step.proportion = 2 / 3, 1 / 3
    terrain.sub_terrains = {"pyramid_stairs_inv": stairs, "single_step": step}
    # 6 行 × 6 列：行难度 0→1，对应 8/8/12/12/16/16 cm 楼梯；起始行随机，覆盖三档。
    terrain.num_rows, terrain.num_cols = 6, 6
    terrain.difficulty_range = (0.0, 1.0)
    terrain.curriculum = True
    terrain.seed = args_cli.seed
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.observations.policy.enable_corruption = False
    for name in ("randomize_apply_external_force_torque", "randomize_push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def stats(err: torch.Tensor) -> dict:
    err = err.flatten().float()
    if err.numel() == 0:
        return {"count": 0}
    q = torch.quantile(err[torch.randperm(err.numel(), device=err.device)[:2_000_000]],
                       torch.tensor([0.5, 0.95, 0.99], device=err.device))
    return {"count": err.numel(), "median": q[0].item(), "p95": q[1].item(), "p99": q[2].item(),
            "max": err.max().item(), "within_1mm": (err < 1e-3).float().mean().item(),
            "within_5mm": (err < 5e-3).float().mean().item()}


def pitch(state):
    g = projected_gravity(state[..., QUAT])
    return torch.atan2(g[..., 0], -g[..., 2])


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    configure(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    device = uenv.device
    reader = RoverStateReader(env, world_frame=True)
    robot = reader.robot
    kin = RoverKinematics().to(device)
    report = {"gates": GATES}

    t0 = time.time()
    hmap = build_height_map(env, res=args_cli.map_res)
    pattern = scan_pattern_base(env)
    report["height_map"] = {"shape": list(hmap.heights.shape), "res_m": hmap.res, "build_s": time.time() - t0,
                            "num_rays": pattern.shape[0]}
    teacher = torch.jit.load(args_cli.teacher_policy, map_location=device).eval()
    obs, _ = env.reset()
    if obs["policy"].shape[-1] != 45 + pattern.shape[0]:
        raise RuntimeError(f"Teacher 观测 {obs['policy'].shape[-1]} 维，预期 45+{pattern.shape[0]}")

    # FK 用的仿真真值：wheel 连杆位姿 + 碰撞圆柱偏移；mimic 从动关节名
    wheel_body = [robot.find_bodies(f"{c}_wheel")[0][0] for c in CORNERS]
    arm_ids = [robot.find_joints(f"{c}_arm_joint")[0][0] for c in CORNERS]
    zx_ids = [robot.find_joints(f"{c}_zhuanxiang_joint")[0][0] for c in CORNERS]
    default_pos = robot.data.default_joint_pos[0, reader.joint_ids[:8]].abs().max().item()

    levels = torch.tensor(args_cli.noise_levels, device=device)
    env_noise = levels[torch.arange(args_cli.num_envs, device=device) % len(levels)]
    S, A, D, C = [], [], [], []
    scan_err_edge, scan_err_flat, fk_err, mimic_err = [], [], [], []
    for step in range(args_cli.collect_steps):
        s, ctx = reader.state(), reader.context()
        if step % 5 == 0:
            with torch.no_grad():
                pred = terrain_map.height_scan(hmap, s, pattern)
                true = obs["policy"][:, 45:]
                xy = terrain_map.scan_points_xy(s[:, POS], s[:, QUAT], pattern)
                # 射线周围 ±1 格高差 > 1 cm 视为台阶边：双线性插值在那里必然偏离阶跃真值。
                r = hmap.res
                nb = torch.stack([hmap.query(xy + torch.tensor(d, device=device) * r)
                                  for d in ((1, 0), (-1, 0), (0, 1), (0, -1))], -1)
                edge = (nb.amax(-1) - nb.amin(-1)) > 0.01
                err = (pred - true).abs()
                scan_err_edge.append(err[edge])
                scan_err_flat.append(err[~edge])
                jp = robot.data.joint_pos
                fk = kin.wheel_centers_world(robot.data.root_pos_w, robot.data.root_quat_w, jp[:, arm_ids],
                                             jp[:, reader.joint_ids[4:8]])
                bq = robot.data.body_link_quat_w[:, wheel_body]
                truth = robot.data.body_link_pos_w[:, wheel_body] + quat_apply(bq, kin.wheel_offset.expand(bq.shape[:-1] + (3,)))
                fk_err.append((fk - truth).norm(dim=-1))
                mimic_err.append((jp[:, zx_ids] + jp[:, arm_ids]).abs())
        with torch.no_grad():
            action = teacher(obs["policy"]) + env_noise[:, None] * torch.randn(args_cli.num_envs, 12, device=device)
            action = action.clamp(-SuperMimicCfg().action_bound, SuperMimicCfg().action_bound)
        obs, _, terminated, truncated, _ = env.step(action)
        for buf, v in zip((S, A, D, C), (s, action, terminated | truncated, ctx)):
            buf.append(v)
    S, A, D, C = (torch.stack(v) for v in (S, A, D, C))
    report["collect_seconds"] = time.time() - t0
    report["scan"] = {"nonedge": stats(torch.cat(scan_err_flat)), "edge": stats(torch.cat(scan_err_edge))}
    report["fk"] = {"wheel_center_err_m": stats(torch.cat(fk_err)), "mimic_follower_abs_err_rad": stats(torch.cat(mimic_err)),
                    "default_leg_joint_pos_abs_max": default_pos}

    # 数据覆盖：台阶边起点比例；车下地面相对本回合起点地面的抬升（证明确实爬上了台阶）
    edge_mask = torch.cat([terrain_map.near_edge(hmap, kin, S[t]).unsqueeze(0) for t in range(S.shape[0])])
    ground = torch.cat([hmap.query(S[t, :, :2]).unsqueeze(0) for t in range(S.shape[0])])
    start_ground = ground[0].clone()
    rise = torch.empty_like(ground)
    for t in range(S.shape[0]):
        if t > 0:
            start_ground = torch.where(D[t - 1], ground[t], start_ground)
        rise[t] = ground[t] - start_ground
    max_rise = rise.amax(0)
    report["coverage"] = {
        "edge_fraction": edge_mask.float().mean().item(),
        "rise_over_6cm_fraction": (rise > 0.06).float().mean().item(),
        "envs_max_rise_m_quantiles": torch.quantile(max_rise, torch.tensor([0.25, 0.5, 0.75, 1.0], device=device)).tolist(),
        "envs_reached_16cm_fraction": (max_rise > 0.15).float().mean().item(),
        "episodes_finished": int(D.sum().item()),
    }

    # 世界模型过门
    n_train = args_cli.num_envs - args_cli.heldout_envs
    held = slice(n_train, None)
    prev_actions = torch.cat((torch.zeros_like(A[:1]), A[:-1]))

    def wheel_z(pred, target):
        f = lambda x: kin.wheel_centers_world(x[:, POS], x[:, QUAT], x[:, JPOS][:, :4], x[:, JPOS][:, 4:8])[..., 2]  # noqa: E731
        return (f(pred) - f(target)).abs().mean(-1)

    extra = {"wheel_z_m": wheel_z, "pitch_rad": lambda p, t: (pitch(p) - pitch(t)).abs()}
    const_vel = lambda s, a, c: constant_velocity_step(s, rover_task.CONTROL_DT)  # noqa: E731
    report["wm_probe"] = {}
    for variant in ("no_relief", "relief"):
        torch.manual_seed(args_cli.seed)
        cfg = SuperMimicCfg(num_envs=n_train, buffer_len=args_cli.collect_steps, wm_hidden=tuple(args_cli.wm_hidden),
                            wm_updates=1, context_dim=CONTEXT_DIM, aux_dim=0)
        extra_fn = (lambda s: terrain_map.relief_features(hmap, kin, s)) if variant == "relief" else None
        learner = SuperMimicLearner(cfg, device, extra_fn=extra_fn,
                                    extra_dim=terrain_map.RELIEF_DIM if extra_fn else 0)
        for t in range(args_cli.collect_steps):
            learner.observe(S[t, :n_train], A[t, :n_train], prev_actions[t, :n_train], torch.zeros(n_train, 3, device=device),
                            D[t, :n_train], C[t, :n_train])
        curve, t1 = [], time.time()
        for u in range(args_cli.wm_updates):
            logs = learner.update_world_model()
            if u % 500 == 0 or u == args_cli.wm_updates - 1:
                curve.append({"update": u, **{k: logs[k] for k in ("wm_loss", "wm_err_1step", "wm_err_last")}})
                print(f"[wm {variant} {u}] " + " ".join(f"{k}={v:.4g}" for k, v in curve[-1].items()), flush=True)
        res = {"train_seconds": time.time() - t1, "curve": curve}
        for subset, mask in (("edge", edge_mask[:, held]), ("flat", ~edge_mask[:, held])):
            res[subset] = open_loop_errors(S[:, held], A[:, held], D[:, held], learner.wm_step, contexts=C[:, held],
                                           start_mask=mask, extra_metrics=extra, chunk=4096)
            if variant == "no_relief":
                report.setdefault("const_vel", {})[subset] = open_loop_errors(
                    S[:, held], A[:, held], D[:, held], const_vel, start_mask=mask, extra_metrics=extra, chunk=4096)
        report["wm_probe"][variant] = res
        del learner
        torch.cuda.empty_cache()

    wp, cv = report["wm_probe"], report["const_vel"]

    def ratio(a, b):
        return a / b if b > 0 else math.inf

    edge8 = {k: wp[k]["edge"]["k8"] for k in wp}
    report["gate_results"] = {
        "scan_nonedge_within_5mm": report["scan"]["nonedge"]["within_5mm"] >= GATES["scan_nonedge_within_5mm_fraction_min"],
        "fk_max_err": report["fk"]["wheel_center_err_m"]["max"] <= GATES["fk_max_err_m_max"],
        "edge_wheel_z_vs_no_relief": ratio(edge8["relief"]["wheel_z_m"], edge8["no_relief"]["wheel_z_m"])
        <= GATES["wm_edge_k8_wheel_z_vs_no_relief_ratio_max"],
        "edge_wheel_z_vs_const_vel": ratio(edge8["relief"]["wheel_z_m"], cv["edge"]["k8"]["wheel_z_m"])
        <= GATES["wm_edge_k8_wheel_z_vs_const_vel_ratio_max"],
        "edge_pitch_vs_const_vel": ratio(edge8["relief"]["pitch_rad"], cv["edge"]["k8"]["pitch_rad"])
        <= GATES["wm_edge_k8_pitch_vs_const_vel_ratio_max"],
        "flat_no_regression": ratio(wp["relief"]["flat"]["k8"]["wheel_z_m"], wp["no_relief"]["flat"]["k8"]["wheel_z_m"])
        <= GATES["wm_flat_k8_wheel_z_vs_no_relief_ratio_max"],
    }
    Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("height_map", "scan", "fk", "coverage", "gate_results")}, indent=2))
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
