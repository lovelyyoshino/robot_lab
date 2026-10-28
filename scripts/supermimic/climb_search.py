#!/usr/bin/env python3
"""
climb_search.py — 16 cm 楼梯可行性搜索：参数化的特权攀爬程序 + 交叉熵（CEM）搜索，产出参考动作与示范数据。

不是策略，不用于部署。目的只有两个：
1. 回答"这台车（当前执行器/限位）能不能爬上 16 cm×4 级楼梯"；
2. 若能，录下成功回合（232 维观测、12 维原始动作、状态），作为第 5 步跟踪器/PPO 示范的数据源。

程序（每个轮子独立，按特权高度图 + 可微 FK 判断台阶）：
- 轮子前方 trigger 距离内出现 >5 cm 的升高 → 该轮进入"抬"状态，arm 转向抬轮方向 lift；
- 轮下地面已比进入时高出 4 cm（轮子站上新台阶）→ 回到"支撑"状态；
- 支撑状态的轮子按 stance_push 向下伸，抬高车身；arm 目标加俯仰反馈抑制车身前后翻；
- 轮速分接近/抬轮两档。arm 符号由 FK 确定：前轮 arm<0 抬、后轮 arm>0 抬（见 kinematics）。
场景与 evaluate_parallelogram_vision 的 Preview 楼梯验收一致（同一地形生成与终点 3.0 m）；
成败判据复现其 route_passed（前进 ≥ 终点、横向 ≤ 0.15 m、正立、朝向），不含承重力检查。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-10-27
@changelog
    - v0.2.1 (2026-10-28): 识别方案 B 策略并追加车道偏移观测。
    - v0.2.0 (2026-10-27): --record 保存逐步状态；途中倾角门取 stairs_acceptance.CLIMB_TILT_DEG，与评测一致；输出各倾角门内比例；最大倾角只统计出界前的时段（此前把冲出平台后的掉落算进去了）；登顶判据按本档总高（原硬编码 0.62 m）；沿路线带符号探针。
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="16 cm stairs feasibility search (CEM over a privileged climb program)")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0")
parser.add_argument("--teacher_policy",
                    default="logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/exported/policy.pt")
parser.add_argument("--difficulty", type=float, default=1.0, help="1.0 = 16 cm × 4 级")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--generations", type=int, default=12)
parser.add_argument("--episode_s", type=float, default=16.0)
parser.add_argument("--elite_frac", type=float, default=0.1)
parser.add_argument("--validate_envs", type=int, default=256)
parser.add_argument("--output", default="logs/supermimic_climb")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--teacher_only", action="store_true", help="只跑 Teacher 基线与失败诊断，不搜索")
parser.add_argument("--record", action="store_true",
                    help="teacher_only 时把逐步状态/动作/成功标记存成 <output>/rollout_d<难度>.npz，供 analyze_climb.py 离线分析")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic.isaac_adapter import RoverStateReader, build_height_map  # noqa: E402
from robot_lab.supermimic.kinematics import RoverKinematics  # noqa: E402
from robot_lab.supermimic.quat import projected_gravity, quat_apply, yaw_quat  # noqa: E402
from robot_lab.supermimic.state import JPOS, LIN, POS, QUAT  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview import (  # noqa: E402
    stairs_profile,
)

ARM_SCALE, ARM_LIMIT = 0.25, 0.55
FRONT = torch.tensor([True, False, True, False])  # 角顺序 (fl, rl, fr, rr)
# 参数：名称、下界、上界、初值
PARAMS = [
    ("front_lift", 0.0, 0.55, 0.45), ("rear_lift", 0.0, 0.55, 0.45),
    ("front_trigger", 0.00, 0.30, 0.10), ("rear_trigger", 0.00, 0.30, 0.10),
    ("stance_push", 0.0, 0.55, 0.20), ("v_approach", 0.3, 4.0, 1.3),
    ("v_lift", 0.0, 6.28, 1.5), ("pitch_gain", -1.0, 1.0, 0.0),
    ("ramp_tau", 0.02, 0.40, 0.10), ("yaw_gain", -3.0, 3.0, 0.0),
    # 车身调平：按俯仰对前后 arm 反向修正（前短后长抵消楼梯坡度）；arm 行程 ±0.55 rad 约可抵 ±16°
    ("level_gain", -2.0, 2.0, 0.0), ("lateral_gain", -6.0, 6.0, 0.0),
]
LO = torch.tensor([p[1] for p in PARAMS])
HI = torch.tensor([p[2] for p in PARAMS])
INIT = torch.tensor([p[3] for p in PARAMS])


def configure(env_cfg, num_envs, seed):
    terrain = env_cfg.scene.terrain.terrain_generator
    stairs = terrain.sub_terrains["pyramid_stairs_inv"].copy()
    stairs.proportion = 1.0
    terrain.sub_terrains = {"pyramid_stairs_inv": stairs}
    # 最多 32 列：多台车共用一列时环境间碰撞已被过滤，与训练时相同；避免生成数百块地形。
    terrain.num_rows, terrain.num_cols = 1, min(num_envs, 32)
    terrain.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    terrain.curriculum = False
    terrain.seed = seed
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum = None
    env_cfg.terminations.preview_success = None
    env_cfg.episode_length_s = args_cli.episode_s + 1.0
    env_cfg.events.randomize_reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.randomize_reset_base.params["velocity_range"] = {}
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (0.30, 0.30), (0.0, 0.0), (0.0, 0.0)
    cmd.resampling_time_range = (1e3, 1e3)
    return stairs


class ClimbProgram:
    def __init__(self, theta, hmap, kin, wheel_sign, dt):
        self.p = {name: theta[:, i] for i, (name, *_) in enumerate(PARAMS)}
        self.hmap, self.kin, self.sign, self.dt = hmap, kin, wheel_sign, dt
        n = theta.shape[0]
        self.lifting = torch.zeros(n, 4, dtype=torch.bool, device=theta.device)
        self.ground_at_lift = torch.zeros(n, 4, device=theta.device)
        self.lift_level = torch.zeros(n, 4, device=theta.device)
        self.front = FRONT.to(theta.device)

    def reset(self, ids):
        self.lifting[ids] = False
        self.lift_level[ids] = 0.0

    def __call__(self, s, yaw0, lateral=None):
        p, dev = self.p, s.device
        self.lateral = torch.zeros(s.shape[0], device=dev) if lateral is None else lateral
        jp = s[:, JPOS]
        wheels = self.kin.wheel_centers_world(s[:, POS], s[:, QUAT], jp[:, :4], jp[:, 4:8])
        fwd = quat_apply(yaw_quat(s[:, QUAT]), torch.tensor([1.0, 0, 0], device=dev).expand(s.shape[0], 3))[:, :2]
        ground = self.hmap.query(wheels[..., :2])
        look = torch.arange(0.0, 0.32, 0.02, device=dev)
        ahead = self.hmap.query(wheels[..., None, :2] + look[:, None] * fwd[:, None, None, :]) - ground[..., None]
        rise = ahead > 0.05
        dist = torch.where(rise.any(-1), look[rise.float().argmax(-1)], torch.full_like(ground, 9.0))
        trigger = torch.where(self.front, p["front_trigger"][:, None], p["rear_trigger"][:, None])
        start = ~self.lifting & (dist <= trigger)
        self.ground_at_lift = torch.where(start, ground, self.ground_at_lift)
        self.lifting = (self.lifting | start) & ~(ground > self.ground_at_lift + 0.04)
        alpha = self.dt / (p["ramp_tau"][:, None] + self.dt)
        self.lift_level += alpha * (self.lifting.float() - self.lift_level)
        # 抬轮方向：前轮 arm 负、后轮 arm 正；支撑下伸方向相反
        lift_dir = torch.where(self.front, -1.0, 1.0)
        lift_mag = torch.where(self.front, p["front_lift"][:, None], p["rear_lift"][:, None])
        g = projected_gravity(s[:, QUAT])
        pitch = torch.atan2(g[:, 0], -g[:, 2])  # 机头下俯为正
        arm = lift_dir * (self.lift_level * lift_mag - (1 - self.lift_level) * p["stance_push"][:, None])
        arm = arm + p["pitch_gain"][:, None] * pitch[:, None] * lift_dir
        # 调平项作用于全部轮子：前轮与后轮反向（lift_dir 已含前负后正），不依赖抬轮状态
        arm = arm + p["level_gain"][:, None] * pitch[:, None] * torch.where(self.front, 1.0, -1.0)
        arm = arm.clamp(-ARM_LIMIT, ARM_LIMIT)
        any_lift = self.lift_level.max(-1).values
        v = p["v_approach"] + (p["v_lift"] - p["v_approach"]) * any_lift
        q = s[:, QUAT]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        yaw_err = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
        # 横向偏移也折算进航向误差：只纠航向会平行偏出车道（Teacher 越线时横向 0.5 m）
        yaw_err = yaw_err + p["lateral_gain"] * 0.1 * self.lateral
        left = torch.tensor([1.0, 1.0, -1.0, -1.0], device=dev)  # fl, rl 在左
        # 轮速参数直接是原始动作（×5 = 目标 rad/s）。轮子是 DCMotor（damping 0.5），扭矩 ≈ 0.5×(目标−实际)，
        # 目标设成"想要的速度"几乎没有扭矩；Teacher 实测平均原始动作 1.26（目标 6.3、实际 2.9 rad/s）。
        wheel_raw = v[:, None] + p["yaw_gain"][:, None] * yaw_err[:, None] * left * 0.1
        action = torch.zeros(s.shape[0], 12, device=dev)
        action[:, 0:4] = arm / ARM_SCALE
        action[:, 8:12] = (self.sign * wheel_raw).clamp(-2 * math.pi, 2 * math.pi)
        return action, self.lift_level


def rollout(env, reader, origins, target, act_fn, steps, record=False, hmap=None, diag=None, climb_tilt_deg=31.8,
            top_height=0.64):
    """返回每环境 (score, success, max_progress, flipped) 以及可选的录制；diag 字典收集爬升/横向诊断。"""
    uenv = env.unwrapped
    n = uenv.num_envs
    obs, _ = env.reset()
    yaw0 = torch.zeros(n, device=uenv.device)
    best_prog = torch.zeros(n, device=uenv.device)
    flipped = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    stable = torch.zeros(n, device=uenv.device)
    success = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    rec = {"obs": [], "action": [], "state": []} if record else None
    rise = torch.zeros(n, device=uenv.device)
    lat = torch.zeros(n, device=uenv.device)
    ground0 = hmap.query(origins[:, :2]) if hmap is not None else None
    at_target = {k: torch.full((n,), float("nan"), device=uenv.device) for k in ("lateral", "alignment", "rise", "yaw_deg")}
    crossed = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    # 沿路线若干 x 处记录带符号的横向/航向，定位偏航在哪一级台阶开始累积
    probe_x = (0.6, 1.2, 1.6, 2.0, 2.4, 2.8)
    probe = {x: {k: torch.full((n,), float("nan"), device=uenv.device) for k in ("y", "yaw", "vel_dir", "steer_f", "steer_r")}
             for x in probe_x}
    tilt_max = torch.zeros(n, device=uenv.device)
    eval_fallen = torch.zeros(n, dtype=torch.bool, device=uenv.device)
    for _ in range(steps):
        s = reader.state()
        act_fn.lateral = (s[:, POS] - origins)[:, 1]
        action = act_fn(s, yaw0, obs)
        if record:
            rec["obs"].append(obs["policy"].cpu())
            rec["action"].append(action.cpu())
            rec["state"].append(s.cpu())
        obs, _, terminated, truncated, _ = env.step(action)
        s1 = reader.state()
        rel = s1[:, POS] - origins
        q = s1[:, QUAT]
        align = 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        cos_tilt = -projected_gravity(q)[:, 2]
        upright = cos_tilt >= 0.85
        eval_fallen_before = eval_fallen | (rel[:, 0] > 3.75) | (rel[:, 1].abs() > 0.75)
        # 与评测 capture_traversal 相同：途中倾角超过本档 climb_tilt_deg、横向>0.75 或冲过 3.75 m 即判失败；
        # 终点稳定仍要求 31.8° 内
        eval_fallen |= (cos_tilt < math.cos(math.radians(climb_tilt_deg))) | (rel[:, 1].abs() > 0.75) | (rel[:, 0] > 3.75)
        ok = (rel[:, 0] >= target) & (rel[:, 1].abs() <= 0.15) & upright & (align >= 0.8) & ~flipped & ~eval_fallen
        stable = torch.where(ok, stable + uenv.step_dt, torch.zeros_like(stable))
        success |= stable >= 0.5
        flipped |= terminated
        best_prog = torch.where(flipped, best_prog, torch.maximum(best_prog, rel[:, 0]))
        if hmap is not None:
            h = hmap.query(s1[:, :2]) - ground0
            rise = torch.maximum(rise, h)
            lat = torch.maximum(lat, rel[:, 1].abs())
            first = (rel[:, 0] >= target) & ~crossed & ~flipped
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            for k, v in (("lateral", rel[:, 1]), ("alignment", align), ("rise", h), ("yaw_deg", torch.rad2deg(yaw))):
                at_target[k] = torch.where(first, v, at_target[k])
            crossed |= first
            for x in probe_x:
                hit = (rel[:, 0] >= x) & probe[x]["y"].isnan() & ~flipped & ~eval_fallen
                probe[x]["y"] = torch.where(hit, rel[:, 1], probe[x]["y"])
                probe[x]["yaw"] = torch.where(hit, torch.rad2deg(yaw), probe[x]["yaw"])
                # 实际行进方向（世界系速度角）与航向的差 = 侧滑/蟹行；转向关节顺序 fl,rl,fr,rr
                v_w = quat_apply(q, s1[:, LIN])
                probe[x]["vel_dir"] = torch.where(hit, torch.rad2deg(torch.atan2(v_w[:, 1], v_w[:, 0].clamp_min(1e-3))),
                                                  probe[x]["vel_dir"])
                jp = s1[:, JPOS]
                probe[x]["steer_f"] = torch.where(hit, torch.rad2deg(0.5 * (jp[:, 4] + jp[:, 6])), probe[x]["steer_f"])
                probe[x]["steer_r"] = torch.where(hit, torch.rad2deg(0.5 * (jp[:, 5] + jp[:, 7])), probe[x]["steer_r"])
            # 只统计评测仍在计的时段：冲出 3.75 m 后车会从平台边缘掉下，不能算进爬升倾角
            tilt_now = torch.acos((-projected_gravity(q)[:, 2]).clamp(-1, 1))
            tilt_max = torch.where(eval_fallen_before, tilt_max, torch.maximum(tilt_max, tilt_now))
    score = best_prog / target + 2.0 * success.float() - 1.0 * flipped.float()
    if hmap is not None:
        # 爬升高度计分（只靠前进会奖励绕开楼梯）；倾角超过评测门限按超出量扣分，横向偏移扣分
        score = score + 1.5 * rise.clamp(0, 0.7)
        score = score - 2.0 * (tilt_max - math.radians(climb_tilt_deg)).clamp_min(0) - 1.0 * lat.clamp(0, 1.0)
    if diag is not None:
        diag.update({"max_rise_m_median": rise.median().item(), "max_rise_m_max": rise.max().item(),
                     "max_lateral_m_median": lat.median().item(),
                     "reached_top_fraction": (rise >= top_height - 0.02).float().mean().item(),
                     "crossed_target_fraction": crossed.float().mean().item(),
                     "tilt_max_deg_median": torch.rad2deg(tilt_max).median().item(),
                     **{f"tilt_max_within_{g}deg": (torch.rad2deg(tilt_max) <= g).float().mean().item()
                        for g in (31.8, 45.0, 55.0, 65.0)},
                     "eval_fallen_fraction": eval_fallen.float().mean().item()})
        if crossed.any():
            c = crossed
            diag.update({f"at_target_{k}_" + stat: getattr(v[c].abs() if k in ("lateral", "yaw_deg") else v[c], stat)().item()
                         for k, v in at_target.items() for stat in ("median", "max")})
            diag["at_target_lateral_signed_mean"] = at_target["lateral"][c].mean().item()
            diag["at_target_yaw_deg_signed_mean"] = at_target["yaw_deg"][c].mean().item()
        diag["route_probe"] = {f"x{x}": {k: [round(v[~v.isnan()].mean().item(), 3), round(v[~v.isnan()].std().item(), 3)]
                                         for k, v in probe[x].items()} for x in probe_x}
        if crossed.any():
            diag["at_target_lateral_within_0.15"] = (at_target["lateral"][c].abs() <= 0.15).float().mean().item()
    return score, success, best_prog, flipped, rec


class _wrap:
    """把 ClimbProgram 包成 rollout 的 act_fn，并把横向偏移传进去。"""

    def __init__(self, prog):
        self.prog, self.lateral = prog, None

    def __call__(self, s, yaw0, obs):
        return self.prog(s, yaw0, self.lateral)[0]


class _teacher_wrap:
    def __init__(self, fn):
        self.fn, self.lateral = fn, None

    def __call__(self, s, yaw0, obs):
        return self.fn(s, yaw0, obs)


def main():
    torch.manual_seed(args_cli.seed)
    out = Path(args_cli.output)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    stairs = configure(env_cfg, args_cli.num_envs, args_cli.seed)
    from robot_lab.supermimic.lane_offset import add_lane_offset_term, policy_uses_lane_offset

    if policy_uses_lane_offset(args_cli.teacher_policy):
        add_lane_offset_term(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    dev = uenv.device
    reader = RoverStateReader(env, world_frame=True)
    kin = RoverKinematics().to(dev)
    hmap = build_height_map(env)
    profile = stairs_profile(args_cli.difficulty, stairs.height_levels, stairs.step_counts, stairs.first_edge_m,
                             stairs.step_width)
    target = profile["target_distance_m"]
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.stairs_acceptance import climb_tilt_deg
    profile["climb_tilt_deg"] = climb_tilt_deg(profile)
    steps = int(args_cli.episode_s / uenv.step_dt)
    teacher = torch.jit.load(args_cli.teacher_policy, map_location=dev).eval()
    report = {"profile": profile, "steps": steps}

    def origins():
        return uenv.scene.env_origins.clone()

    # 0) Teacher 基线 + 轮速符号标定（前进时各轮原始动作的符号）
    signs = []

    def teacher_fn(s, yaw0, obs):
        with torch.no_grad():
            a = teacher(obs["policy"]).clamp(-2 * math.pi, 2 * math.pi)
        if len(signs) < 50:
            signs.append(a[:, 8:12].mean(0))
        return a

    t0 = time.time()
    d = {}
    sc, ok, prog, flip, rec = rollout(env, reader, origins(), target, _teacher_wrap(teacher_fn), steps, hmap=hmap, diag=d,
                                      climb_tilt_deg=profile["climb_tilt_deg"], top_height=profile["total_height_m"],
                                      record=args_cli.record and args_cli.teacher_only)
    wheel_sign = torch.sign(torch.stack(signs).mean(0))
    wheel_sign = torch.where(wheel_sign == 0, torch.ones_like(wheel_sign), wheel_sign)
    report["teacher"] = {"success_rate": ok.float().mean().item(), "progress_median_m": prog.median().item(),
                         "flip_rate": flip.float().mean().item(), "seconds": time.time() - t0, **d}
    report["wheel_sign"] = wheel_sign.tolist()
    print("[teacher]", json.dumps(report["teacher"]), "wheel_sign", report["wheel_sign"], flush=True)
    if args_cli.teacher_only:
        (out / f"teacher_d{args_cli.difficulty}.json").write_text(json.dumps(report, indent=2))
        if rec is not None:
            # state 为世界系（含位置），按 env_origins 还原相对楼梯坐标；轮心几何用 RoverKinematics 离线重算
            np.savez_compressed(out / f"rollout_d{args_cli.difficulty}.npz",
                                state=torch.stack(rec["state"]).numpy(), action=torch.stack(rec["action"]).numpy(),
                                obs=torch.stack(rec["obs"]).numpy(), origins=origins().cpu().numpy(), success=ok.cpu().numpy(), flipped=flip.cpu().numpy(),
                                step_dt=uenv.step_dt, profile=json.dumps(profile))
        env.close()
        return

    # 1) CEM 搜索
    mean, std = INIT.clone(), (HI - LO) * 0.3
    history = []
    for gen in range(args_cli.generations):
        theta = (mean + std * torch.randn(args_cli.num_envs, len(PARAMS))).clamp(LO, HI)
        theta[0] = mean.clamp(LO, HI)  # 每代保留当前均值
        prog_fn = ClimbProgram(theta.to(dev), hmap, kin, wheel_sign, uenv.step_dt)
        d = {}
        sc, ok, prog, flip, _ = rollout(env, reader, origins(), target, _wrap(prog_fn), steps, hmap=hmap, diag=d,
                                    climb_tilt_deg=profile["climb_tilt_deg"], top_height=profile["total_height_m"])
        sc = sc.cpu()
        k = max(int(args_cli.elite_frac * args_cli.num_envs), 4)
        elite = theta[sc.topk(k).indices]
        mean, std = elite.mean(0), torch.maximum(elite.std(0), (HI - LO) * 0.03)
        rec = {"generation": gen, "success_rate": ok.float().mean().item(), "best_score": sc.max().item(),
               "elite_score": sc.topk(k).values.mean().item(), "progress_median_m": prog.median().item(),
               "flip_rate": flip.float().mean().item(), **d,
               "mean": dict(zip([p[0] for p in PARAMS], mean.tolist()))}
        history.append(rec)
        print(f"[gen {gen}] tilt_med={rec['tilt_max_deg_median']:.1f} evalfall={rec['eval_fallen_fraction']:.2f} "
              f"lat_med={rec['max_lateral_m_median']:.2f} top={rec['reached_top_fraction']:.2f} succ={rec['success_rate']:.3f} best={rec['best_score']:.3f} elite={rec['elite_score']:.3f} "
              f"prog_med={rec['progress_median_m']:.2f} flip={rec['flip_rate']:.2f} rise_med={rec['max_rise_m_median']:.3f} "
              f"rise_max={rec['max_rise_m_max']:.3f}", flush=True)
    report["cem"] = history

    # 2) 用最终均值在全部环境上重复验证（同一参数，环境间只差域随机化），并录制成功回合
    theta = mean.clamp(LO, HI).expand(args_cli.num_envs, -1).clone()
    prog_fn = ClimbProgram(theta.to(dev), hmap, kin, wheel_sign, uenv.step_dt)
    d = {}
    sc, ok, prog, flip, rec = rollout(env, reader, origins(), target, _wrap(prog_fn), steps, record=True, hmap=hmap, climb_tilt_deg=profile["climb_tilt_deg"], top_height=profile["total_height_m"],
                                      diag=d)
    report["validation"] = {"params": dict(zip([p[0] for p in PARAMS], mean.tolist())),
                            "success_rate": ok.float().mean().item(), "progress_median_m": prog.median().item(),
                            "flip_rate": flip.float().mean().item(), "num_envs": args_cli.num_envs, **d}
    if ok.any():
        idx = ok.nonzero().squeeze(-1).cpu()
        np.savez_compressed(out / f"demos_d{args_cli.difficulty}.npz",
                            obs=torch.stack(rec["obs"])[:, idx].numpy(), action=torch.stack(rec["action"])[:, idx].numpy(),
                            state=torch.stack(rec["state"])[:, idx].numpy(), dt=uenv.step_dt,
                            params=json.dumps(report["validation"]["params"]))
        report["demos"] = {"file": str(out / f"demos_d{args_cli.difficulty}.npz"), "episodes": int(idx.numel())}
    (out / f"climb_search_d{args_cli.difficulty}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("profile", "teacher", "validation")}, indent=2))
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
