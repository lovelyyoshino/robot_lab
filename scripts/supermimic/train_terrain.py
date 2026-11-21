#!/usr/bin/env python3
"""
train_terrain.py — SuperMimic 越障训练：从零（无 PPO Teacher、无监督初始化、无锚定）用世界模型 + BPTT 优化。

@author pony
@date 2026-10-26
@version v0.22.0
@last_modified 2026-11-19

流程：
1. 前 --warmup_iterations 轮只拟合世界模型（地形输入 relief_features），之后每轮 actor 在想象中按 PreviewTask 的
   越障解析目标做 BPTT；critic 拟合真实回报（或想象 λ-return，取决于 learner 配置）。
2. 真实回报与想象目标是同一个函数 PreviewTask.reward；翻车/出车道终止另加 terminated×dt。车道惩罚用地形块中心
   的横向偏移（训练特权，--lane≠0 或 --critic_lane 时启用），actor 观测仍为 232 维。
3. 地形/课程见 tasks/manager_based/supermimic/obstacle_terrain.py：训练循环逐步记录每个环境本回合的立面碰撞、
   姿态超限与命令距离，课程在 reset 时按地形类型晋级/降级。
4. --check_obs_interval N：每 N 轮核对一次想象观测与 env 观测（>1e-4 即中止并保存现场）。

日志：logs/supermimic_terrain/<run>/ 下 train_log.csv、TensorBoard、ckpt_XXXXXX.pt、exported/policy.pt|onnx
（232→12）、params/vision_contract.json（部署合同）、params/supermimic_terrain.json（全部参数与实现哈希）。

@changelog
    - v0.22.0 (2026-11-19): --post_climb_relief / --post_climb_window_m（preview_task v0.23.0 登顶后回正窗口）。
    - v0.21.0 (2026-11-19): --tilt_temperature（软存活 sigmoid 的温度；与 --terminated 一起决定想象里翻车代价从多大倾角开始计入）。
    - v0.20.0 (2026-11-18): --leg_action_bound：腿关节动作上界取 min(2π, 饱和界)（摆臂 ≈ ±2.72，转向与轮子不变；networks v0.5.0），
      摆臂动作不再落到关节限位之外的零梯度区；--stall_linear_cap（preview_task v0.21.0，线性停滞代价放开到倒车区间）。
    - v0.19.0 (2026-11-17): --stairs_success_lane_schedule START END：楼梯成功车道（课程晋级的横偏门）单独按这条时间表
      从 fail_lane_m 收紧到 success_lane_m；给了它时 --stairs_lane_schedule 只管楼梯区车道代价。r22 第 32000 轮 16×40
      楼梯 5/8 回合登顶但横偏 17–27 cm（验收门 15 cm）：代价可以早开，晋级门要等楼梯高度上来再收紧。
    - v0.18.0 (2026-11-16): --stairs_climb_trail_weight / --stairs_alt_lift / --stairs_alt_lift_frac / --stairs_contact_climb_m（preview_task v0.20.0，楼梯左右轮替）。
    - v0.17.0 (2026-11-16): --lead_side_obs：每回合随机先抬侧标记 ±1，作为命令第 4 维进回放、作为观测第 233 维进 actor
      （镜像取反，preview_task v0.19.0 / symmetry v1.1.0 / learner v0.14.0 / networks v0.4.0）。r20 第 30000 轮 16×40
      楼梯 4/4 回合后轴顶第一级卡死、左右动作几乎相同（|a_rl − a_rr| 0.2），第 24000 轮靠左右不对称一次上一个后轮
      （|a_rl − a_rr| 1.0–1.5）全部登顶；镜像一致性在近似对称的状态上把动作推向对称。导出 policy.pt/onnx 固定 +1，
      仍是 232 维输入，验收/部署链路不变。
    - v0.16.0 (2026-11-13): --stairs_lane_schedule START END：楼梯车道约束从宽到严。START 前楼梯成功车道 = fail_lane_m、
      楼梯区车道项与越障横向速度额外倍数 ×0，START→END 线性收紧到 success_lane_m / ×1（obstacle_terrain v1.6.0、
      preview_task v0.18.0），日志 stairs_lane_scale。不给时与原行为相同。
    - v0.15.0 (2026-11-13): --stall_climb_min_clearance / --stall_climb_lead_m / --riser_spin / --riser_spin_lead_m
      （preview_task v0.17.0）。
    - v0.14.0 (2026-11-12): --stall_linear / --stairs_climb_pair_sum（preview_task v0.16.0）。
    - v0.13.0 (2026-11-11): --obstacle_lateral_vel_scale（preview_task v0.15.0）。
    - v0.12.0 (2026-11-11): --trailing_contact_climb（preview_task v0.14.0）。
    - v0.11.0 (2026-11-11): --stairs_stall_climb_scale / --stairs_riser_contact_scale / --stairs_riser_approach_m
      （preview_task v0.13.0）；--curriculum_stairs_promote_streak / --curriculum_stairs_demote_streak /
      --curriculum_freeze_non_stairs / --curriculum_stairs_min_level（obstacle_terrain v1.5.0，续训恢复等级后同样套用下限）。
    - v0.10.0 (2026-11-10): --stall_climb_scale / --drop_relief（preview_task v0.12.0）、--actor_ema_decay /
      --critic_warmup_iterations / --warm_start_reset（learner v0.13.0）、--curriculum_freeze / --curriculum_promote_streak /
      --curriculum_demote_streak 与 --stairs_tread_rule（obstacle_terrain v1.4.0）。
    - v0.9.0 (2026-11-09): --action_saturation_coef / --action_saturation_margin（腿关节动作超出软限位外 margin 的 tanh 前
      代价，界由 env 动作项 scale 与软限位换算）、--imag_flat_fraction；课程 --curriculum_stairs_demote_ratio /
      --curriculum_stall_truncate_s（训练循环把停滞步数写入 env.obstacle_stall_steps，供 stall_timeout 截断）；
      日志 stall_truncated、actor_mirror_loss、actor_saturation_loss、actor_saturated_fraction、imag_flat_start_fraction。
    - v0.8.0 (2026-11-08): 奖励参数 --flat_steer / --wheel_sync / --wheel_sync_delta / --prelift_margin_m /
      --prelift_shift_zone（preview_task v0.10.0）。
    - v0.7.0 (2026-11-08): 奖励参数 --single_slant_deg / --prelift_front / --toe / --wheel_slip / --wheel_slip_delta /
      --stand_yaw_rate / --stand_yaw_rate_delta（preview_task v0.9.0）。
    - v0.6.1 (2026-11-07): 启动时总打印实际生效的轮胎静/动摩擦（固定 μ=1.0 后核对用）。
    - v0.6.0 (2026-11-07): --curriculum_stairs_explore_frac（楼梯探索回合，随机等级、不改课程）；日志与存档用课程等级，
      另记 curriculum/stairs_explore[_high]_pass_rate；--contact_context（ξ 追加四轮摩擦，世界模型与 critic 用）。
    - v0.5.0 (2026-11-07): 奖励参数 --yaw_rate / --stairs_slant_deg / --stairs_slant_zone_m。
    - v0.4.0 (2026-11-07): 奖励参数 --heading_huber / --heading_huber_delta。
    - v0.3.0 (2026-11-05): 车道惩罚/横向 Huber/预抬与姿态参考新参数；课程 success_lane_m、楼梯碰撞门；
      --episode_length_s、--command_lin_vel_x、--command_standing_fraction；日志 lateral_abs_m。
    - v0.2.0 (2026-11-05): 只保留从零越障路线：统一目标、四轮立面碰撞、按类型课程、观测核对按间隔；
      删除 Teacher 热启动/驾驶、锚定、动作参考、预抬探索、平地校准、车道观测、旧预抬目标等路径。
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

DEFAULT_TASK = "RobotLab-Isaac-SuperMimic-Obstacle-Parallelogram-Rover-v0"
# PreviewRewardCfg 字段（名字与类型一一对应；未传时取 dataclass 默认值）
REWARD_FLAGS = (
    ("track_lin_vel", float), ("track_lin_vel_std", float), ("slow_band_m", float), ("slow_band_speed", float),
    ("progress", float), ("stall", float), ("stall_speed_ratio", float), ("stall_climb_scale", float),
    ("stairs_stall_climb_scale", float), ("stall_linear", float), ("stall_linear_cap", float),
    ("stall_climb_min_clearance", float), ("stall_climb_lead_m", float), ("riser_spin", float), ("riser_spin_lead_m", float), ("stairs_lane_scale", float),
    ("drop_relief", float), ("post_climb_relief", float), ("post_climb_window_m", float),
    ("terminated", float), ("tilt_limit", float), ("tilt_temperature", float), ("roll", float), ("roll_excess", float), ("roll_excess_deg", float),
    ("pitch", float), ("pitch_tol_deg", float),
    ("pitch_ref_window", float), ("pitch_ref_deadband_deg", float), ("ang_vel", float), ("arm_vel", float),
    ("action_rate", float), ("lin_vel_z", float), ("heading", float), ("heading_huber", float),
    ("heading_huber_delta", float), ("yaw_rate", float), ("stairs_slant_deg", float), ("stairs_slant_zone_m", float),
    ("single_slant_deg", float), ("lateral_vel", float),
    ("lateral_vel_delta", float), ("obstacle_lateral_vel_scale", float),
    ("lane", float), ("steering", float), ("flat_neutral", float),
    ("rest_clearance", float), ("flat_lookahead", float), ("toe", float), ("flat_steer", float),
    ("wheel_sync", float), ("wheel_sync_delta", float), ("wheel_slip", float),
    ("wheel_slip_delta", float), ("stand_yaw_rate", float), ("stand_yaw_rate_delta", float),
    ("riser_min_height", float), ("riser_contact_gap", float),
    ("riser_contact_speed", float), ("riser_safe_speed", float), ("riser_approach_m", float),
    ("stairs_riser_contact_scale", float), ("stairs_riser_approach_m", float), ("trailing_contact_climb", float),
    ("stairs_climb_pair_sum", float), ("stairs_climb_trail_weight", float), ("stairs_alt_lift", float),
    ("stairs_alt_lift_frac", float), ("stairs_contact_climb_m", float),
    ("prelift_track", float), ("prelift_window", float), ("prelift_front", float), ("early_lift", float),
    ("early_lift_lead", float),
    ("support_margin", float), ("support_margin_m", float), ("prelift_margin_m", float),
    ("prelift_shift_zone", float), ("multi_lift", float),
    ("ready_stroke", float), ("ready_zone_m", float),
    ("min_step", float), ("joint_pos_limits", float),
)
# SuperMimicCfg 字段（含 learner 新增：actor 学习率余弦衰减、镜像一致性、诊断间隔、torch.compile）；
# 显式传入而 learner 没有该字段时启动即报错，避免静默忽略。想象 λ-return critic = --no-critic_real_returns。
LEARNER_FLAGS = (
    ("buffer_len", int), ("horizon", int), ("imag_batch", int), ("wm_window", int), ("wm_batch", int),
    ("wm_updates", int), ("wm_lr", float), ("warmup_iterations", int), ("actor_lr", float), ("critic_lr", float),
    ("action_noise", float), ("warmup_action_noise", float), ("actor_update_every", int), ("real_return_window", int),
    ("real_return_batch", int), ("gamma", float), ("lam", float), ("priority_fraction", float),
    ("actor_value_grad_scale", float), ("actor_lr_final", float), ("actor_lr_decay_iterations", int),
    ("mirror_loss_coef", float), ("diagnostics_interval", int), ("imag_flat_fraction", float),
    ("action_saturation_coef", float), ("actor_ema_decay", float), ("critic_warmup_iterations", int),
)
LEARNER_SWITCHES = ("critic_real_returns", "actor_value_grad", "compile_wm", "compile_imagination", "mirror_aug",
                    "warm_start_reset")
CURRICULUM_FLAGS = ("promote_m", "success_lane_m", "demote_ratio", "posture_bad_frac", "collision_frac",
                    "stairs_collision_frac", "fail_lane_m", "roll_limit_deg", "pitch_dev_limit_deg",
                    "stairs_explore_frac", "stairs_demote_ratio", "stall_truncate_s", "freeze", "promote_streak",
                    "demote_streak", "stairs_promote_streak", "stairs_demote_streak", "freeze_non_stairs",
                    "stairs_min_level", "stairs_success_lane_m")

parser = argparse.ArgumentParser(description="SuperMimic obstacle terrain training (from scratch)")
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", dest="iterations", type=int, default=40000)
parser.add_argument("--steps_per_iter", type=int, default=2)
parser.add_argument("--stairs_lane_schedule", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="楼梯车道约束：START 轮前放松（成功车道 = fail_lane_m，楼梯区车道代价 ×0），START→END 线性收紧")
parser.add_argument("--stairs_success_lane_schedule", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="楼梯成功车道单独的收紧时间表（fail_lane_m → success_lane_m）；给了时 --stairs_lane_schedule 只管车道代价")
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume", action="store_true", help="从 --load_run 的 --checkpoint（默认最新）续训")
parser.add_argument("--load_run", default=None)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("--map_res", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--init_terrain_level", type=int, default=0)
parser.add_argument("--check_obs_interval", type=int, default=0, help="每 N 轮核对一次想象/真实观测；0 关闭")
parser.add_argument("--critic_lane", action="store_true", help="critic 额外看车道横向偏移（训练特权，actor 不看）")
parser.add_argument("--episode_length_s", type=float, default=None, help="覆盖 env 回合长度（秒）")
parser.add_argument("--command_lin_vel_x", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="覆盖前进命令范围 (m/s)")
parser.add_argument("--command_standing_fraction", type=float, default=None, help="覆盖静止命令比例 rel_standing_envs")
parser.add_argument("--critic_route_features", action="store_true", help="critic 额外看世界路线航向")
parser.add_argument("--contact_context", action="store_true",
                    help="世界模型与 critic 的上下文 ξ 追加四轮摩擦/恢复系数（25→37 维，训练特权，actor 不看）")
parser.add_argument("--wm_hidden", type=int, nargs="+", default=None)
parser.add_argument("--stairs_tread_rule", type=float, nargs=3, default=None, metavar=("R0", "R1", "T1"),
                    help="楼梯踏面下限随立面提高：立面 ≤R0 不变、≥R1 时下限 T1（米），中间线性；默认不启用")
parser.add_argument("--lead_side_obs", action="store_true",
                    help="actor 观测追加每回合随机的先抬侧标记 ±1（镜像取反）；导出时固定 +1，部署仍 232 维")
parser.add_argument("--leg_action_bound", action="store_true",
                    help="腿关节动作上界取 min(2π, 软限位外放 --action_saturation_margin 后的幅值)，默认关 = 统一 2π")
parser.add_argument("--action_saturation_margin", type=float, default=0.10,
                    help="动作饱和代价的界 = 腿关节软限位外放该值（rad）对应的动作值")
for _name, _type in REWARD_FLAGS + LEARNER_FLAGS:
    parser.add_argument(f"--{_name}", type=_type, default=None)
for _name in LEARNER_SWITCHES:
    parser.add_argument(f"--{_name}", action=argparse.BooleanOptionalAction, default=None)
for _name in CURRICULUM_FLAGS:
    parser.add_argument(f"--curriculum_{_name}", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.stairs_success_lane_schedule is not None and args_cli.stairs_lane_schedule is None:
    parser.error("--stairs_success_lane_schedule 须与 --stairs_lane_schedule 一起给")
if args_cli.check_obs_interval < 0 or args_cli.steps_per_iter < 1:
    parser.error("--check_obs_interval 须 ≥0，--steps_per_iter 须 ≥1")
if args_cli.episode_length_s is not None and args_cli.episode_length_s <= 0:
    parser.error("--episode_length_s 须为正")
if args_cli.command_lin_vel_x is not None and not 0.0 <= args_cli.command_lin_vel_x[0] <= args_cli.command_lin_vel_x[1]:
    parser.error("--command_lin_vel_x 须满足 0 ≤ LO ≤ HI")
if args_cli.command_standing_fraction is not None and not 0.0 <= args_cli.command_standing_fraction <= 1.0:
    parser.error("--command_standing_fraction 须在 [0, 1]")
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import terrain_map  # noqa: E402
from robot_lab.supermimic.isaac_adapter import (  # noqa: E402
    CONTEXT_DIM, RoverStateReader, build_height_map, exact_riser_fn, exact_scan_fn, leg_soft_limits, scan_pattern_base)
from robot_lab.supermimic.kinematics import RoverKinematics  # noqa: E402
from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from robot_lab.supermimic.preview_task import PreviewRewardCfg, PreviewTask  # noqa: E402
from robot_lab.supermimic.quat import quat_apply  # noqa: E402
from robot_lab.supermimic.rover_task import CONTROL_DT  # noqa: E402
from robot_lab.supermimic.schedule import restore_terrain_levels  # noqa: E402
from robot_lab.supermimic.state import LIN, QUAT  # noqa: E402
from robot_lab.tasks.manager_based.supermimic import obstacle_terrain  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = Path("logs/supermimic_terrain")
# 从零配方（scratch_40k_r8k 最优存档 + 上游式优化）：每轮更新 actor、价值梯度 1.0、4096 想象批量、
# 8 步 × 2048 × 10 次世界模型窗口、tanh 输出、事件优先回放 0.25、critic 拟合 64 步真实回报。
LEARNER_DEFAULTS = dict(buffer_len=600, imag_batch=4096, wm_window=8, wm_batch=2048, wm_updates=10, horizon=32,
                        warmup_iterations=200, critic_warmup_iterations=0, actor_lr=1e-4, action_noise=0.1,
                        warmup_action_noise=0.3, action_squash="tanh", aux_dim=0, context_dim=CONTEXT_DIM,
                        critic_real_returns=True, real_return_window=64, priority_fraction=0.25,
                        actor_update_every=1, actor_value_grad=True, actor_value_grad_scale=1.0)
LEARNER_LOG_KEYS = ("wm_loss", "wm_err_1step", "wm_err_last", "wm_reward_bias", "wm_reward_abs_gap",
                    "wm_event_reward_abs_gap", "wm_flat_reward_abs_gap", "wm_grad_norm", "actor_loss",
                    "actor_model_loss", "actor_grad_norm", "critic_loss", "imag_reward", "imag_survival",
                    "actor_skipped", "actor_update_applied", "actor_lr", "actor_mirror_loss", "actor_saturation_loss",
                    "actor_saturated_fraction", "imag_flat_start_fraction")
IMPLEMENTATION_FILES = (
    "source/robot_lab/robot_lab/supermimic/preview_task.py", "source/robot_lab/robot_lab/supermimic/learner.py",
    "source/robot_lab/robot_lab/supermimic/networks.py", "source/robot_lab/robot_lab/supermimic/kinematics.py",
    "source/robot_lab/robot_lab/supermimic/isaac_adapter.py", "source/robot_lab/robot_lab/supermimic/terrain_map.py",
    "source/robot_lab/robot_lab/supermimic/replay.py", "source/robot_lab/robot_lab/supermimic/state.py",
    "source/robot_lab/robot_lab/supermimic/quat.py",
    "source/robot_lab/robot_lab/tasks/manager_based/supermimic/terrain_env_cfg.py",
    "source/robot_lab/robot_lab/tasks/manager_based/supermimic/obstacle_env_cfg.py",
    "source/robot_lab/robot_lab/tasks/manager_based/supermimic/obstacle_terrain.py",
    "scripts/supermimic/train_terrain.py",
)


def build_cfg() -> SuperMimicCfg:
    cfg = SuperMimicCfg(num_envs=args_cli.num_envs, **LEARNER_DEFAULTS)
    known = {f.name for f in fields(cfg)}
    requested = {name: getattr(args_cli, name) for name, _ in LEARNER_FLAGS}
    requested.update({name: getattr(args_cli, name) for name in LEARNER_SWITCHES})
    requested["wm_hidden"] = tuple(args_cli.wm_hidden) if args_cli.wm_hidden else None
    missing = sorted(name for name, value in requested.items() if value is not None and name not in known)
    if missing:
        raise SystemExit(f"learner.SuperMimicCfg 没有这些字段：{missing}（需与 learner.py 对齐后再启动）")
    for name, value in requested.items():
        if value is not None:
            setattr(cfg, name, value)
    if args_cli.diagnostics_interval is None and "diagnostics_interval" in known:
        cfg.diagnostics_interval = args_cli.log_interval  # reward_gap 等诊断只在记日志的轮次计算
    return cfg


def build_reward_cfg() -> PreviewRewardCfg:
    cfg = PreviewRewardCfg()
    for name, _ in REWARD_FLAGS:
        if getattr(args_cli, name) is not None:
            setattr(cfg, name, getattr(args_cli, name))
    return cfg


def resolve_resume() -> Path | None:
    if not args_cli.resume:
        return None
    if args_cli.checkpoint and Path(args_cli.checkpoint).is_file():
        return Path(args_cli.checkpoint)
    if not args_cli.load_run:
        raise SystemExit("--resume 需要 --load_run")
    run_dir = LOG_ROOT / args_cli.load_run
    if args_cli.checkpoint:
        return run_dir / args_cli.checkpoint
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"{run_dir} 下没有 ckpt_*.pt")
    return ckpts[-1]


def leg_action_limits(uenv, reader: RoverStateReader, margin: float) -> tuple:
    """12 个动作的 (lo, hi)：腿关节（默认位 + scale·a）软限位外放 margin 处对应的动作值；轮速 4 个不限（None）。"""
    term = uenv.action_manager.get_term("joint_pos")
    ids = reader.joint_ids[:8].tolist()
    names = [reader.robot.joint_names[i] for i in ids]
    if list(term._joint_names) != names:
        raise SystemExit(f"腿关节动作顺序 {term._joint_names} 与状态读取顺序 {names} 不一致")
    scale = term._scale[0] if isinstance(term._scale, torch.Tensor) else torch.full((8,), float(term._scale))
    if not bool(term.cfg.use_default_offset) or (scale <= 0).any():
        raise SystemExit("动作饱和界假设 腿关节目标 = 默认位 + scale·a（scale > 0）")
    soft, scale = leg_soft_limits(reader).cpu(), scale.cpu()
    lows = ((soft[:, 0] - margin) / scale).tolist()
    highs = ((soft[:, 1] + margin) / scale).tolist()
    return tuple(zip(lows, highs)) + (None,) * 4


def export_policy(learner: SuperMimicLearner, out: Path, obs_dim: int) -> None:
    """policy.pt（TorchScript）+ policy.onnx；观测/动作语义见 params/vision_contract.json。"""
    learner.export_actor(out / "policy.pt")
    cpu = learner.deploy_actor()
    torch.onnx.export(cpu, torch.zeros(1, obs_dim), str(out / "policy.onnx"), input_names=["obs"],
                      output_names=["actions"], dynamic_axes={"obs": {0: "batch"}}, opset_version=17)


def write_deployment_contract(env_cfg, log_dir: Path) -> None:
    """部署合同来自 Teacher-Preview 部署任务的观测配置（与越障任务观测逐列相同），不依赖任何训练产物。"""
    sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
    from vision_contract import contract_for_cfg

    task = "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0"
    deployment_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    deployment_cfg.scene.height_scanner.update_period = env_cfg.scene.height_scanner.update_period
    contract = contract_for_cfg(task, deployment_cfg, load_cfg_from_registry(task, "rsl_rl_cfg_entry_point"))
    (log_dir / "params" / "vision_contract.json").write_text(json.dumps(contract, indent=2))


def check_observation(task, s, cmd, prev, env_obs, log_dir: Path, uenv) -> float:
    with torch.no_grad():
        difference = (task.policy_obs(s, cmd, prev) - env_obs).abs()
    worst = difference.max().item()
    if worst > 1e-4:
        scanner = uenv.scene.sensors["height_scanner"]
        torch.save({"state": s, "command": cmd, "prev_action": prev, "obs": env_obs, "difference": difference,
                    "sensor_hits": scanner.data.ray_hits_w, "episode_length": uenv.episode_length_buf},
                   log_dir / "observation_mismatch.pt")
        raise RuntimeError(f"想象与真实观测误差 {worst:.6g} > 1e-4（本体 {difference[:, :45].max().item():.3g}，"
                           f"扫描 {difference[:, 45:].max().item():.3g}）；现场已存 observation_mismatch.pt")
    return worst


def main():
    torch.manual_seed(args_cli.seed)
    resume_path = resolve_resume()
    resume_data = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    task_features = {"critic_lane": args_cli.critic_lane, "critic_route_features": args_cli.critic_route_features}
    if args_cli.contact_context:
        # 只在开启时写入，旧存档（无此键）仍可按原配置续训；ξ 维数不同的存档不能互相续训
        task_features["contact_context"] = True
    if args_cli.lead_side_obs:
        task_features["lead_side_obs"] = True  # actor 输入 233 维，不能和 232 维存档互相续训
    if resume_data is not None and resume_data.get("task_features") != task_features:
        raise SystemExit(f"续训存档的 critic 输入 {resume_data.get('task_features')} 与本次 {task_features} 不同")
    cfg = build_cfg()
    reward_cfg = build_reward_cfg()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.init_terrain_level
    for name in CURRICULUM_FLAGS:
        if getattr(args_cli, f"curriculum_{name}") is not None:
            env_cfg.obstacle_curriculum_cfg[name] = getattr(args_cli, f"curriculum_{name}")
    curriculum_cfg = {**obstacle_terrain.CURRICULUM_DEFAULTS, **env_cfg.obstacle_curriculum_cfg}
    if args_cli.stairs_tread_rule is not None:
        stairs_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains[obstacle_terrain.STAIRS]
        stairs_cfg.tread_rule = tuple(args_cli.stairs_tread_rule)
        obstacle_terrain.min_tread_for_riser(stairs_cfg.riser_range[1], stairs_cfg.tread_range[0], stairs_cfg.tread_rule)
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    command_cfg = env_cfg.commands.base_velocity
    if args_cli.command_lin_vel_x is not None:
        command_cfg.ranges.lin_vel_x = tuple(args_cli.command_lin_vel_x)
    if args_cli.command_standing_fraction is not None:
        command_cfg.rel_standing_envs = args_cli.command_standing_fraction
    command_summary = {"class": type(command_cfg).__name__, "lin_vel_x": tuple(command_cfg.ranges.lin_vel_x),
                       "rel_standing_envs": command_cfg.rel_standing_envs,
                       "resampling_time_range": tuple(command_cfg.resampling_time_range),
                       "episode_length_s": env_cfg.episode_length_s}
    print(f"[supermimic] 命令与回合：{command_summary}", flush=True)
    if abs(env_cfg.terminations.bad_orientation.params["limit_angle"] - reward_cfg.tilt_limit) > 1e-6:
        raise SystemExit("env 倾角终止与 --tilt_limit 不一致，软存活与真实终止会分叉")
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    device = uenv.device
    tracker = obstacle_terrain.EpisodeTracker(uenv.num_envs, device)
    uenv.obstacle_tracker, uenv.obstacle_curriculum_log = tracker, {}
    kinds = torch.as_tensor(env_cfg.column_kinds, device=device)

    # 摩擦摘要（第七轮诊断）：同一策略在 8 cm 楼梯上 μ=1.0 成功 98%、μ=0.5 只有 3%（后轮顶住第一级），
    # 摩擦是后轴能否翻上立面的主变量；不给世界模型/critic 摩擦，它们只能对 μ∈[0.4,1.0] 取平均。
    reader = RoverStateReader(env, world_frame=True, contact_context=args_cli.contact_context)
    cfg.context_dim = reader.context_dim
    # 打印实际生效的轮胎摩擦（2026-11-07 起固定 μ=1.0，见 terrain_env_cfg.WHEEL_FRICTION）
    material = reader._read_wheel_materials().view(-1, 4, 3)
    print(f"[supermimic] 轮胎摩擦：静 {material[..., 0].min():.2f}–{material[..., 0].max():.2f}，"
          f"动 {material[..., 1].min():.2f}–{material[..., 1].max():.2f}"
          + ("（ξ 含四轮摩擦）" if args_cli.contact_context else ""), flush=True)
    kin = RoverKinematics().to(device)
    hmap = build_height_map(env, res=args_cli.map_res)
    pattern = scan_pattern_base(env)
    # 车道中心 = 各地形块原点的 y（与终止/课程用的 env_origins 同源）；只进奖励与 critic，不进 actor。
    lane_center_y = (torch.unique(uenv.scene.terrain.terrain_origins[..., 1].flatten()).sort().values
                     if args_cli.critic_lane or reward_cfg.lane else None)
    task = PreviewTask(hmap, kin, pattern, leg_soft_limits(reader), reward_cfg, exact_scan_fn=exact_scan_fn(env, pattern),
                       exact_riser_fn=exact_riser_fn(env), lane_center_y=lane_center_y,
                       critic_route_features=args_cli.critic_route_features, critic_lane=args_cli.critic_lane,
                       lead_obs=args_cli.lead_side_obs)
    if cfg.action_saturation_coef or args_cli.leg_action_bound:
        cfg.action_limits = leg_action_limits(uenv, reader, args_cli.action_saturation_margin)
    if args_cli.leg_action_bound:
        # 动作上界逐维：腿关节取 min(原上界, 饱和界的幅值)，轮子不变；超过关节限位的动作物理上没有区别，梯度为 0
        base = float(cfg.action_bound)
        cfg.action_bound = tuple(base if lim is None else min(base, max(abs(float(lim[0])), abs(float(lim[1]))))
                                 for lim in cfg.action_limits)
        print("[supermimic] 动作上界（逐维）：" + " ".join(f"{b:.2f}" for b in cfg.action_bound), flush=True)
    if cfg.action_saturation_coef:
        print(f"[supermimic] 动作饱和界（动作单位，腿关节软限位外放 {args_cli.action_saturation_margin:.2f} rad）："
              + " ".join(f"[{lo:+.2f},{hi:+.2f}]" for lo, hi in cfg.action_limits[:8]), flush=True)
    learner = SuperMimicLearner(cfg, device, extra_fn=lambda s: terrain_map.relief_features(hmap, kin, s),
                                extra_dim=terrain_map.RELIEF_DIM, task=task)
    terrain = uenv.scene.terrain
    if resume_data is not None:
        learner.load_state_dict(resume_data)
        levels = restore_terrain_levels(resume_data["terrain_levels"], uenv.num_envs, terrain.max_terrain_level).to(device)
        # 楼梯补练：存档里的楼梯等级低于 stairs_min_level 时直接抬到下限（其它类型不变）
        levels = obstacle_terrain.apply_level_floor(levels, kinds[terrain.terrain_types], curriculum_cfg,
                                                    terrain.max_terrain_level)
        terrain.terrain_levels[:] = levels
        terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
        if getattr(uenv, "obstacle_base_levels", None) is not None:
            uenv.obstacle_base_levels[:] = levels  # 课程等级与存档一致（探索回合的随机等级不入档）
        print(f"[supermimic] 续训 {resume_path} @ iteration {learner.iteration}，课程均值 {levels.float().mean():.2f}", flush=True)
    else:
        print("[supermimic] 从零训练：actor 随机初始化", flush=True)

    run = args_cli.run_name or (resume_path.parent.name if resume_path else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = LOG_ROOT / run
    if resume_data is None and log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有目录：{log_dir}")
    (log_dir / "exported").mkdir(parents=True, exist_ok=True)
    (log_dir / "params").mkdir(exist_ok=True)
    write_deployment_contract(env_cfg, log_dir)
    implementation_sha256 = {Path(p).name: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                             for p in IMPLEMENTATION_FILES}
    (log_dir / "params" / "supermimic_terrain.json").write_text(json.dumps(
        {"objective": "obstacle", "actor_source": str(resume_path) if resume_path else "random",
         "actor_optimizer": "world_model_bptt", "ppo_updates": 0, "task": args_cli.task, "task_features": task_features,
         "implementation_sha256": implementation_sha256, "cfg": asdict(cfg), "reward": asdict(reward_cfg),
         "curriculum": curriculum_cfg, "command": command_summary, "arguments": vars(args_cli),
         "map_res": args_cli.map_res},
        indent=2, default=str))
    writer = SummaryWriter(str(log_dir))
    csv_file = open(log_dir / "train_log.csv", "a", newline="")
    csv_writer = None

    n = uenv.num_envs
    roll_limit = math.radians(curriculum_cfg["roll_limit_deg"])
    pitch_limit = math.radians(curriculum_cfg["pitch_dev_limit_deg"])
    stall_steps = torch.zeros(n, dtype=torch.long, device=device)
    stall_limit = round(1.0 / CONTROL_DT)  # 停滞 >1 s 记入日志
    uenv.obstacle_stall_steps = stall_steps  # obstacle_terrain.stall_timeout 读取（stall_truncate_s > 0 时截断回合）
    obs, _ = env.reset()
    # 先抬侧标记：每回合 ±1 随机，回合结束（done）时重抽；只在 --lead_side_obs 时进命令与 actor 观测。
    lead = torch.randint(0, 2, (n,), device=device).float() * 2.0 - 1.0

    def with_lead(x):
        return torch.cat((x, lead[:, None]), dim=-1) if args_cli.lead_side_obs else x

    min_rows = max(cfg.wm_window, cfg.real_return_window if cfg.critic_real_returns else 0) + 2
    start_time = time.time()
    lane_strict = float(curriculum_cfg["stairs_success_lane_m"]) or float(curriculum_cfg["success_lane_m"])
    for it in range(learner.iteration, args_cli.iterations):
        log_now = it % args_cli.log_interval == 0
        sums: dict[str, torch.Tensor] = {}
        if args_cli.stairs_lane_schedule is not None:
            # 先爬上去、后期再收紧车道：楼梯成功车道 fail_lane_m → lane_strict，楼梯区车道代价 ×0 → ×1
            lane_scale = obstacle_terrain.lane_schedule_scale(it, *args_cli.stairs_lane_schedule)
            task.set_stairs_lane_scale(lane_scale, device=device)
            success_scale = (lane_scale if args_cli.stairs_success_lane_schedule is None
                             else obstacle_terrain.lane_schedule_scale(it, *args_cli.stairs_success_lane_schedule))
            uenv.cfg.obstacle_curriculum_cfg["stairs_success_lane_m"] = (
                curriculum_cfg["fail_lane_m"] + (lane_strict - curriculum_cfg["fail_lane_m"]) * success_scale)

        def add(key, value):
            value = value.float().mean()
            sums[key] = sums[key] + value if key in sums else value

        for step in range(args_cli.steps_per_iter):
            s, prev, ctx = reader.state(), reader.last_action(), reader.context()
            cmd = with_lead(uenv.command_manager.get_command("base_velocity").clone())
            policy_obs = with_lead(obs["policy"])
            if args_cli.check_obs_interval and it % args_cli.check_obs_interval == 0 and step == 0:
                sums["obs_max_err"] = torch.tensor(check_observation(task, s, cmd, prev, policy_obs, log_dir, uenv))
            with torch.no_grad():
                # 课程统计属于当前回合：在 env.step（其中会 reset）之前记录步前状态。
                contacts = task.riser_contacts(s)
                roll, pitch_dev = task.posture_errors(s)
                tracker.update(contacts.any(-1), (roll > roll_limit) | (pitch_dev > pitch_limit), cmd[:, 0], CONTROL_DT)
                priority = task.event_priority(s)
            # 截断判据在 env.step 内按截至上一步的连续停滞步数计算。
            uenv.obstacle_stall_steps = stall_steps
            stall_cut = obstacle_terrain.stall_steps_exceeded(stall_steps, curriculum_cfg["stall_truncate_s"], CONTROL_DT)
            action = learner.act(policy_obs)
            if not torch.isfinite(action).all():
                raise FloatingPointError("动作含非有限值，拒绝送入仿真")
            obs, env_rew, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            with torch.no_grad():
                # done 的步后状态已被 reset，没有物理后继：用步前状态代替，该步只剩终止惩罚有意义。
                s_next = torch.where(done[:, None], s, reader.state())
                terms = task.reward_terms(s, s_next, cmd, action, prev)
                rew = torch.stack(list(terms.values()), dim=-1).sum(-1) * CONTROL_DT
                rew = rew + reward_cfg.terminated * CONTROL_DT * terminated.float()
                priority = torch.maximum(priority, terminated.float())
                stall_steps = torch.where(task.progressing(s, cmd) | done, 0, stall_steps + 1)
            learner.observe(s, action, prev, cmd, done, ctx, reward=rew, terminated=terminated, priority=priority)
            if args_cli.lead_side_obs:
                lead = torch.where(done, torch.randint_like(lead, 0, 2) * 2.0 - 1.0, lead)
            if log_now:
                with torch.no_grad():
                    for key, value in terms.items():
                        add(f"r_{key}", value)
                    add("shared_reward", rew)
                    add("env_reward", env_rew)
                    add("terminated", terminated)
                    add("priority_rate", priority > 0)
                    add("riser_contact_front", contacts[:, [0, 2]].any(-1))
                    add("riser_contact_rear", contacts[:, [1, 3]].any(-1))
                    add("roll_deg", roll * 180.0 / math.pi)
                    add("pitch_dev_deg", pitch_dev * 180.0 / math.pi)
                    add("pitch_rate_rms", s[:, 11].square())
                    add("stall_1s_fraction", stall_steps > stall_limit)
                    add("stall_truncated", stall_cut)
                    add("route_vel_err", (cmd[:, 0] - quat_apply(s[:, QUAT], s[:, LIN])[:, 0]).abs())
                    if lane_center_y is not None:
                        add("lateral_abs_m", task.lateral_offset(s).abs())
        if learner.replay.size >= min_rows:
            logs = learner.update()
        else:
            logs = {}
            learner.iteration += 1
        if log_now:
            logs.update({k: v.item() / (1 if k == "obs_max_err" else args_cli.steps_per_iter) for k, v in sums.items()})
            if args_cli.stairs_lane_schedule is not None:
                logs["stairs_lane_scale"] = lane_scale
                logs["stairs_success_lane_m"] = uenv.cfg.obstacle_curriculum_cfg["stairs_success_lane_m"]
            logs["pitch_rate_rms"] = math.sqrt(logs["pitch_rate_rms"]) * 180.0 / math.pi
            logs.update({f"curriculum/{k}": v.item() for k, v in obstacle_terrain.curriculum_summary(
                obstacle_terrain.curriculum_levels(uenv), kinds[terrain.terrain_types],
                terrain.cfg.terrain_generator.num_rows).items()})
            decisions, uenv.obstacle_curriculum_log = uenv.obstacle_curriculum_log, {}
            for name in obstacle_terrain.KIND_NAMES.values():
                episodes = max(decisions.get(f"{name}_episodes", 0), 1)
                for outcome in ("promoted", "demoted", "collided"):
                    logs[f"curriculum/{name}_{outcome}_rate"] = decisions.get(f"{name}_{outcome}", 0) / episodes
            if curriculum_cfg["stairs_explore_frac"] > 0:
                # 探索回合（随机楼梯等级）的通过率；high = ≥12 cm（验收的高楼梯区间）。回合数少时噪声大，看趋势
                for key in ("explore", "explore_high"):
                    logs[f"curriculum/stairs_{key}_episodes"] = decisions.get(f"stairs_{key}_episodes", 0)
                    logs[f"curriculum/stairs_{key}_pass_rate"] = (decisions.get(f"stairs_{key}_passed", 0)
                                                                  / max(decisions.get(f"stairs_{key}_episodes", 0), 1))
            logs["elapsed_min"] = (time.time() - start_time) / 60
            for k, v in logs.items():
                writer.add_scalar(f"supermimic_terrain/{k}", v, it)
            row = {"iteration": it, **logs}
            if csv_writer is None:
                csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(set(row) | set(LEARNER_LOG_KEYS)),
                                            extrasaction="ignore")
                if csv_file.tell() == 0:
                    csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()
            print(f"[it {it}] " + " ".join(f"{k}={v:.4g}" for k, v in sorted(logs.items())), flush=True)
        if (it + 1) % args_cli.save_interval == 0 or it + 1 == args_cli.iterations:
            torch.save({**learner.state_dict(), "terrain_levels": obstacle_terrain.curriculum_levels(uenv).cpu(),
                        "task_features": task_features, "task_reward": asdict(reward_cfg),
                        "curriculum": curriculum_cfg, "implementation_sha256": implementation_sha256},
                       log_dir / f"ckpt_{it + 1:06d}.pt")
            export_policy(learner, log_dir / "exported", task.obs_dim - int(task.lead_obs))
    csv_file.close()
    writer.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # 异常后调用 simulation_app.close() 会卡死且吞掉报错（实测），先打印再直接退出进程。
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
