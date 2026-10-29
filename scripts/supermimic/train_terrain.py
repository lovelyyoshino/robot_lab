#!/usr/bin/env python3
"""
train_terrain.py — SuperMimic 越障训练：从 Teacher-Preview 热启动，在 8→12→16 cm 楼梯课程上用世界模型 + BPTT 优化。

流程：
1. actor 直接加载 Teacher-Preview 的 rsl_rl 权重（232→512/256/128→12，clamp 输出，界内动作与 Teacher 完全相同）；
2. 前 --warmup_iterations 轮只拟合世界模型（有地形输入 relief_features），接着 --critic_warmup_iterations 轮只训 critic；
3. 之后 actor 在想象中按 PreviewTask 的解析奖励 + 几何抬轮塑形做 BPTT；
4. --driver_fraction 的环境始终由冻结 Teacher + 噪声驱动，给世界模型持续提供 actor 分布之外（含卡台阶）的数据。
课程沿用 Preview 原生 preview_route_curriculum（按真实通行晋级），日志里的 stage/stairs 比例即为课程进度。

日志：logs/supermimic_terrain/<run>/ 下 train_log.csv、TensorBoard、ckpt_XXXXXX.pt、exported/policy.pt
（232→12 TorchScript）、params/vision_contract.json（与 Teacher 同一合同，供 play/评测校验）。

@author pony
@date 2026-10-26
@version v0.9.0
@last_modified 2026-10-29
@changelog
    - v0.9.0 (2026-10-29): --init_from（继承 actor + 世界模型）、--strict_lane（车道对齐验收）；从零可配 --lane_obs。
    - v0.8.0 (2026-10-28): --from_scratch（方案 D：从零、共用奖励、翻车惩罚、事件优先回放、2→16 cm 课程）。
    - v0.7.0 (2026-10-28): --lane_obs（方案 B：actor 追加车道横向偏移观测，新列 0 初始化）。
    - v0.6.0 (2026-10-28): --max_iterations 默认 6000→1000（长训退化）；默认开启平地平稳惩罚 --smooth_flat_arm -0.5 / --smooth_flat_ang -5.0。
    - v0.5.0 (2026-10-27): 导出同时写 policy.onnx；新增 --lateral_pos、--riser_anchor、--anchor_final（PPO 约束衰减）；--wheel_heading 默认 0（v4 实测有害）；默认值回到 v3 基线。
    - v0.4.0 (2026-10-27): 参数与 rsl_rl/train.py 对齐：--max_iterations、--resume --load_run --checkpoint。
    - v0.3.0 (2026-10-26): 航向/横向速度/倾角软惩罚（想象与真实回报同口径）。
    - v0.2.0 (2026-10-26): v1 退化后改为 critic 拟合真实回报 + Teacher 信赖域；抬轮塑形 ×10；初始地形覆盖全课程。
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SuperMimic terrain (stairs) training")
# 方案 D：不用 Teacher，从零在 2→16 cm 课程上训练（参考 go2w-supermimic）。下列默认值只在未显式传参时生效。
parser.add_argument("--from_scratch", action="store_true",
                    help="不热启动、不锚定 Teacher、无 Teacher 驾驶；任务改为 Scratch-Stairs，奖励两边共用同一实现")
parser.add_argument("--terminated", type=float, default=None, help="翻车惩罚权重（想象与真实同口径）；从零默认 −20")
parser.add_argument("--priority_fraction", type=float, default=None, help="事件优先回放比例；从零默认 0.25")
# d2：在 d1 已学会的爬升上继续，但横向要求对齐验收。只继承 actor 与世界模型；critic 输入维度/回报口径都变了，重新训
parser.add_argument("--init_from", default=None,
                    help="从另一个越障 ckpt 继承 actor 与世界模型（不继承 critic、优化器、迭代号）；actor 维度不足时补 0 列")
parser.add_argument("--strict_lane", action="store_true",
                    help="从零任务的晋级/出界车道对齐验收（0.15 / 0.75 m），出车道为真终止")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0")
parser.add_argument("--teacher", default="logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt")
parser.add_argument("--num_envs", type=int, default=2048)
# 与 rsl_rl/train.py 同名：--max_iterations；--iterations 为旧名保留
# 默认 1000：2026-10-28 同配置 checkpoint 验收 8/12 cm——第 1000 轮 67%/81%，第 2000 轮 6%/0%，第 4000 轮 3%/0%。
# 越训越偏离 Teacher（anchor_mse 0.85→1.35），8/12 cm 登顶后侧漂冲出终点；车道横向位置不在想象目标里，长训会丢掉。
parser.add_argument("--max_iterations", "--iterations", dest="iterations", type=int, default=1000)
parser.add_argument("--steps_per_iter", type=int, default=2)
parser.add_argument("--run_name", default=None)
# 与 rsl_rl/train.py 同口径：--resume [--load_run RUN] [--checkpoint ckpt_XXXXXX.pt]；仍接受旧写法 --resume <路径>
parser.add_argument("--resume", nargs="?", const=True, default=None)
parser.add_argument("--load_run", default=None, help="logs/supermimic_terrain 下的 run 名；省略时取最新 run")
parser.add_argument("--checkpoint", default=None, help="ckpt 文件名或路径；省略时取该 run 最新 ckpt")
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--driver_fraction", type=float, default=0.15)
parser.add_argument("--driver_noise", type=float, default=0.2)
parser.add_argument("--map_res", type=float, default=0.02)
parser.add_argument("--check_obs", action="store_true", help="每轮核对想象观测与 env 观测")
parser.add_argument("--seed", type=int, default=42)
for name, typ in (("buffer_len", int), ("horizon", int), ("imag_batch", int), ("wm_batch", int), ("wm_updates", int),
                  ("warmup_iterations", int), ("critic_warmup_iterations", int), ("actor_lr", float), ("wm_lr", float),
                  ("action_noise", float), ("warmup_action_noise", float), ("actor_update_every", int),
                  ("anchor_coef", float), ("real_return_window", int)):
    parser.add_argument(f"--{name}", type=typ, default=None)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=None)
parser.add_argument("--wheel_lift", type=float, default=10.0,
                    help="几何抬轮塑形权重；1.0 时只有跟踪项的 ~1%（v1 实测无效），默认 10")
parser.add_argument("--idle_lift", type=float, default=None)
parser.add_argument("--imagined_critic", action="store_true", help="回到 v1：critic 只在想象轨迹上拟合")
parser.add_argument("--init_terrain_level", type=int, default=8,
                    help="初始地形等级上限（0–8 均匀抽）；Teacher 已学会全课程，从 0 起会把 12/16 cm 留到很晚")
# v3：v2 失败主要是越线横向 >0.15 m（航向偏 ~13°）和 16 cm 倾角 >31.8°，env 奖励对二者都不约束。
# 13° 航向时 heading 项 ≈ -0.5/s、tilt_soft 在 62° 时 ≈ -2.4/s，对比跟踪项满分 5/s。传 0 回到 v2。
parser.add_argument("--heading", type=float, default=-10.0, help="× sin²(航向)，惩罚取负")
parser.add_argument("--lateral_vel", type=float, default=-20.0, help="× 世界系 vy²，惩罚取负")
# 默认值 = v3（2026-10-27 用户选定的基线：8/12/16 cm 86%/98%/0%）。去掉 tilt_soft 的 v3b 是 78%/94%/0%，
# 16 cm 上它不起作用（途中倾角仅 ~20°），但 8/12 cm 略好，保留。
parser.add_argument("--tilt_soft", type=float, default=-6.0, help="× relu(cos25° − cosθ)，惩罚取负")
# v4：策略习惯性蟹行（车头左偏、轮子右转），上 16 cm 台阶时沿轮子方向侧漂；惩罚轮子世界系朝向
# v4 实测有害：蟹行是策略斜着上台阶的手段，压掉后 8 cm 50%、12 cm 19–44%（后轮卡在首级前），默认关闭
parser.add_argument("--wheel_heading", type=float, default=0.0, help="× 四轮平均 sin²(航向+转向角)，惩罚取负")
# PPO 指导逐步让位给世界模型梯度：actor 开始更新后，anchor_coef 在 --anchor_decay_iterations 轮内线性降到
# --anchor_final（默认不衰减 = v3）；riser_anchor 的局部加权不受影响，难点处仍保持 Teacher 约束。
parser.add_argument("--anchor_final", type=float, default=None, help="anchor_coef 衰减终值；省略则不衰减")
parser.add_argument("--anchor_decay_iterations", type=int, default=500)
# 平地平稳：terrain_v3 训到 4000 轮后平地 arm 2.9 Hz 来回摆、车体上下跳。默认开启；传 0 回到 v3。
parser.add_argument("--smooth_flat_arm", type=float, default=-0.5, help="× 平地处 Σ arm 关节速度²，惩罚取负")
# 量级：抖动时 arm 项 ≈ −2.9/s、角速度项 ≈ −0.85/s；平稳行驶时两项合计 < −0.05/s（跟踪满分 5/s）
parser.add_argument("--smooth_flat_ang", type=float, default=-5.0, help="× 平地处俯仰/滚转角速度²，惩罚取负")
# 方案 B：actor 追加 1 维车道横向偏移观测（Teacher 不动，新列权重初始化为 0），配合 --lateral_pos 让世界模型梯度学纠偏
parser.add_argument("--lane_obs", action="store_true", help="actor 观测追加车道横向偏移（233 维）；见 supermimic/lane_offset.py")
parser.add_argument("--riser_anchor", type=float, default=0.0,
                    help="后轮贴台阶立面时 Teacher 锚定加权倍数（1+此值）；0 关闭")
parser.add_argument("--lateral_pos", type=float, default=0.0,
                    help="× 相对车道中心线横向偏移²，惩罚取负；非 0 时 critic 额外看到偏移量")
parser.add_argument("--tilt_soft_start_deg", type=float, default=25.0)
AppLauncher.add_app_launcher_args(parser)
_explicit = {a.split("=")[0].lstrip("-") for a in sys.argv[1:] if a.startswith("--")}
args_cli, hydra_args = parser.parse_known_args()
SCRATCH_TASK = "RobotLab-Isaac-SuperMimic-Scratch-Stairs-Parallelogram-Rover-v0"
# 从零：task / 地形起点 / Teacher 驾驶 / 塑形的默认值改掉；用户显式传的参数一律保留
SCRATCH_ARG_DEFAULTS = {"task": SCRATCH_TASK, "init_terrain_level": 0, "driver_fraction": 0.0,
                        "tilt_soft": 0.0, "terminated": -20.0, "priority_fraction": 0.25}
if args_cli.from_scratch:
    for _k, _v in SCRATCH_ARG_DEFAULTS.items():
        if _k not in _explicit:
            setattr(args_cli, _k, _v)
    if args_cli.riser_anchor or args_cli.anchor_final is not None:
        parser.error("--from_scratch 不与 --riser_anchor / --anchor_final 组合（它们依赖 Teacher 锚定）")
if args_cli.init_from and "critic_warmup_iterations" not in _explicit:
    # actor 已会爬但 critic 是新的：随机 critic 的自举值会把第一批 BPTT 梯度推坏，先只训 critic
    args_cli.critic_warmup_iterations = 300
if args_cli.strict_lane and not args_cli.from_scratch:
    parser.error("--strict_lane 只用于 --from_scratch")
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import terrain_map  # noqa: E402
from robot_lab.supermimic.isaac_adapter import (  # noqa: E402
    CONTEXT_DIM, RoverStateReader, build_height_map, exact_scan_fn, leg_soft_limits, scan_pattern_base)
from robot_lab.supermimic.kinematics import RoverKinematics  # noqa: E402
from robot_lab.supermimic.lane_offset import add_lane_offset_term, pad_first_layer, write_marker  # noqa: E402
from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from robot_lab.supermimic.preview_task import PreviewRewardCfg, PreviewTask  # noqa: E402
from robot_lab.supermimic.rover_task import CONTROL_DT  # noqa: E402
from robot_lab.supermimic.quat import quat_apply  # noqa: E402
from robot_lab.supermimic.state import LIN, QUAT  # noqa: E402

# 越障默认值：比平地更小的想象批量（每步含 FK + 地形查询 + 射线投射），热启动后降低探索噪声。
# v2（2026-10-26）：v1 在 100 次 actor 更新内把 8 cm 楼梯从 100% 降到 9%；改为 critic 拟合真实回报 + Teacher 信赖域。
TERRAIN_DEFAULTS = dict(buffer_len=600, imag_batch=2048, wm_batch=2048, wm_updates=10, warmup_iterations=200,
                        critic_warmup_iterations=300, actor_lr=5e-5, action_noise=0.05, warmup_action_noise=0.1,
                        action_squash="clamp", aux_dim=0, context_dim=CONTEXT_DIM, critic_real_returns=True,
                        real_return_window=64, anchor_coef=1.0)


# 从零（go2w 配方）：tanh 输出、无 Teacher 锚定、无 critic 预热；探索加大；事件优先回放；actor lr 介于平地 2e-4 与微调 5e-5 之间
SCRATCH_DEFAULTS = dict(TERRAIN_DEFAULTS, imag_batch=4096, warmup_iterations=200, critic_warmup_iterations=0,
                        actor_lr=1e-4, action_noise=0.1, warmup_action_noise=0.3, action_squash="tanh", anchor_coef=0.0)


def build_cfg() -> SuperMimicCfg:
    cfg = SuperMimicCfg(num_envs=args_cli.num_envs, **(SCRATCH_DEFAULTS if args_cli.from_scratch else TERRAIN_DEFAULTS))
    if args_cli.priority_fraction is not None:
        cfg.priority_fraction = args_cli.priority_fraction
    for f in fields(cfg):
        value = getattr(args_cli, f.name, None)
        if value is not None:
            setattr(cfg, f.name, tuple(value) if isinstance(value, list) else value)
    if args_cli.imagined_critic:
        cfg.critic_real_returns = False
    return cfg


def export_policy(learner: SuperMimicLearner, out: Path, obs_dim: int) -> None:
    """policy.pt（TorchScript）+ policy.onnx，与 play_rover / rsl_rl play 的导出物一致；
    观测/动作语义沿用 params/vision_contract.json（与热启动 Teacher 相同）。"""
    learner.export_actor(out / "policy.pt")
    cpu = copy.deepcopy(learner.actor).cpu().eval()
    torch.onnx.export(cpu, torch.zeros(1, obs_dim), str(out / "policy.onnx"), input_names=["obs"],
                      output_names=["actions"], dynamic_axes={"obs": {0: "batch"}}, opset_version=17)


def resolve_resume() -> str | None:
    if args_cli.resume is None:
        return None
    if isinstance(args_cli.resume, str):
        return args_cli.resume
    if args_cli.checkpoint and Path(args_cli.checkpoint).is_file():
        return args_cli.checkpoint
    root = Path("logs/supermimic_terrain")
    run_dir = root / args_cli.load_run if args_cli.load_run else max(
        (d for d in root.iterdir() if any(d.glob("ckpt_*.pt"))), key=lambda d: d.stat().st_mtime, default=None)
    if run_dir is None or not run_dir.is_dir():
        raise FileNotFoundError(f"找不到续训 run：{run_dir or root}")
    if args_cli.checkpoint:
        return str(run_dir / args_cli.checkpoint)
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"{run_dir} 下没有 ckpt_*.pt")
    return str(ckpts[-1])


def teacher_actor_state(teacher_ckpt: str, obs_dim: int) -> dict:
    """Teacher 的 actor.*；任务多出的输入列（方案 B 车道偏移）补 0 权重，输出与原 Teacher 逐位相同。"""
    sd = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    actor_sd = {k: v for k, v in sd.items() if k.startswith("actor.")}
    extra = obs_dim - actor_sd["actor.0.weight"].shape[1]
    return pad_first_layer(actor_sd, extra) if extra > 0 else actor_sd


def warm_start(learner: SuperMimicLearner, teacher_ckpt: str) -> None:
    actor_sd = teacher_actor_state(teacher_ckpt, learner.task.obs_dim)
    if actor_sd["actor.0.weight"].shape[1] != learner.task.obs_dim:
        raise ValueError(f"Teacher 输入 {actor_sd['actor.0.weight'].shape[1]} 维，任务观测 {learner.task.obs_dim} 维")
    learner.actor.load_state_dict(actor_sd)
    print(f"[SuperMimic-Terrain] actor 从 Teacher 热启动：{teacher_ckpt}")


def curriculum_stats(uenv) -> dict:
    t = uenv.scene.terrain
    levels = t.terrain_levels.float()
    if args_cli.from_scratch:
        from robot_lab.tasks.manager_based.supermimic.scratch_env_cfg import SCRATCH_HEIGHT_RANGE
        lo, hi = SCRATCH_HEIGHT_RANGE
        rows = t.cfg.terrain_generator.num_rows
        return {"terrain_level_mean": levels.mean().item(), "terrain_level_max": levels.max().item(),
                "step_height_cm_mean": ((lo + (levels + 0.5) / rows * (hi - lo)).mean() * 100).item(),
                "top_level_fraction": (levels >= rows - 1).float().mean().item()}
    return {"terrain_level_mean": levels.mean().item(), "stage_1": (levels < 3).float().mean().item(),
            "stage_2": ((levels >= 3) & (levels < 6)).float().mean().item(), "stage_3": (levels >= 6).float().mean().item()}


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.init_terrain_level
    if args_cli.lane_obs:
        add_lane_offset_term(env_cfg)
    if args_cli.strict_lane:
        from robot_lab.tasks.manager_based.supermimic.scratch_env_cfg import enable_strict_lane
        enable_strict_lane(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    device = uenv.device
    reader = RoverStateReader(env, world_frame=True)
    kin = RoverKinematics().to(device)
    hmap = build_height_map(env, res=args_cli.map_res)
    pattern = scan_pattern_base(env)
    reward_cfg = PreviewRewardCfg(**{k: getattr(args_cli, k) for k in ("wheel_lift", "idle_lift", "heading",
                                                                       "lateral_vel", "tilt_soft", "wheel_heading", "lateral_pos", "riser_anchor",
                                                                       "smooth_flat_arm", "smooth_flat_ang", "terminated")
                                     if getattr(args_cli, k) is not None})
    reward_cfg.tilt_soft_start = math.radians(args_cli.tilt_soft_start_deg)
    lane_center_y = None
    if reward_cfg.lateral_pos != 0.0 or args_cli.lane_obs:
        # 所有地形块原点的世界 y；楼梯与全宽单障碍的车道中心就是块原点（lane_center_y=0）
        lane_center_y = torch.unique(uenv.scene.terrain.terrain_origins[..., 1].flatten()).sort().values
    task = PreviewTask(hmap, kin, pattern, leg_soft_limits(reader), reward_cfg, exact_scan_fn=exact_scan_fn(env, pattern),
                       lane_center_y=lane_center_y, lane_obs=args_cli.lane_obs)
    cfg = build_cfg()
    learner = SuperMimicLearner(cfg, device, extra_fn=lambda s: terrain_map.relief_features(hmap, kin, s),
                                extra_dim=terrain_map.RELIEF_DIM, task=task)
    resume_path = resolve_resume()
    if resume_path:
        learner.load_state_dict(torch.load(resume_path, map_location=device, weights_only=False))
        print(f"[SuperMimic-Terrain] 续训 {resume_path} @ iteration {learner.iteration}")
    elif args_cli.init_from:
        src = torch.load(args_cli.init_from, map_location=device, weights_only=False)
        actor_sd = src["actor"]
        extra = task.obs_dim - actor_sd["actor.0.weight"].shape[1]
        learner.actor.load_state_dict(pad_first_layer(actor_sd, extra) if extra > 0 else actor_sd)
        learner.world_model.load_state_dict(src["world_model"])
        print(f"[SuperMimic-Terrain] 从 {args_cli.init_from} 继承 actor（补 {max(extra, 0)} 列 0）与世界模型；critic 重新训练",
              flush=True)
    elif not args_cli.from_scratch:
        warm_start(learner, args_cli.teacher)
    if not args_cli.from_scratch:
        # 信赖域参考 = Teacher 本身（续训时也锚定 Teacher，而不是上一个 checkpoint）
        anchor = copy.deepcopy(learner.actor)
        anchor.load_state_dict(teacher_actor_state(args_cli.teacher, task.obs_dim))
        learner.set_anchor(anchor)
    elif not args_cli.init_from:
        print("[SuperMimic-Terrain] 从零训练：actor 随机初始化，不加载、不锚定 Teacher", flush=True)

    # 续训未给 --run_name 时写回原 run，与 rsl_rl 续训习惯一致
    run = args_cli.run_name or (Path(resume_path).parent.name if resume_path else datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = Path("logs/supermimic_terrain") / run
    (log_dir / "exported").mkdir(parents=True, exist_ok=True)
    (log_dir / "params").mkdir(exist_ok=True)
    # 与 Teacher 相同的观测/动作合同：play/评测按它校验，防止同维度不同语义的误加载。
    teacher_contract = Path(args_cli.teacher).parent / "params" / "vision_contract.json"
    if teacher_contract.is_file():
        shutil.copy(teacher_contract, log_dir / "params" / "vision_contract.json")
    if args_cli.lane_obs:
        # 合同仍描述前 232 维；第 233 维由此标记描述，评测/回放据此追加观测
        write_marker(log_dir, task.base_obs_dim)
    (log_dir / "params" / "supermimic_terrain.json").write_text(json.dumps(
        {"teacher": str(Path(args_cli.teacher).resolve()), "task": args_cli.task, "cfg": asdict(cfg),
         "reward": asdict(reward_cfg), "map_res": args_cli.map_res}, indent=2, default=str))
    writer = SummaryWriter(str(log_dir))
    csv_file = open(log_dir / "train_log.csv", "a", newline="")
    csv_writer = None

    n = args_cli.num_envs
    n_driver = int(round(args_cli.driver_fraction * n))
    actor_mask = torch.ones(n, dtype=torch.bool, device=device)
    actor_mask[:n_driver] = False
    teacher = (torch.jit.load(str(Path(args_cli.teacher).parent / "exported" / "policy.pt"), map_location=device).eval()
               if n_driver else None)
    bound = cfg.action_bound
    obs, _ = env.reset()
    min_rows = cfg.wm_window + 2
    start_time = time.time()
    anchor_start = cfg.anchor_coef
    actor_begin = cfg.warmup_iterations + cfg.critic_warmup_iterations
    for it in range(learner.iteration, args_cli.iterations):
        if args_cli.anchor_final is not None:
            frac = min(max(it - actor_begin, 0) / max(args_cli.anchor_decay_iterations, 1), 1.0)
            cfg.anchor_coef = anchor_start + frac * (args_cli.anchor_final - anchor_start)
        sim = {"env_reward": 0.0, "shaping_reward": 0.0, "route_vel_err": 0.0, "terminated": 0.0,
               "obs_max_err": 0.0}
        for _ in range(args_cli.steps_per_iter):
            s, prev, ctx = reader.state(), reader.last_action(), reader.context()
            cmd = uenv.command_manager.get_command("base_velocity").clone()
            if args_cli.check_obs:
                with torch.no_grad():
                    sim["obs_max_err"] = max(sim["obs_max_err"],
                                             (task.policy_obs(s, cmd, prev) - obs["policy"]).abs().max().item())
            # actor 直接吃 env 观测（与部署一致）；想象观测与它逐列一致已由 verify_preview_task 核对。
            action = learner.act(obs["policy"])
            if n_driver:
                with torch.no_grad():
                    drive = teacher(obs["policy"][:n_driver, :task.base_obs_dim])
                action[:n_driver] = (drive + args_cli.driver_noise * torch.randn_like(drive)).clamp(-bound, bound)
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            # v3 塑形补到真实回报上（critic 拟合真实回报，不补则想象目标与 critic 口径不一致）。
            # 用步前状态 s：步后状态在 reset 环境里已被重置；与想象里用 next_state 差一步，可忽略。
            with torch.no_grad():
                shaping = task.shaping_reward(s)
                priority = task.event_priority(s) if cfg.priority_fraction > 0 else None
                if priority is not None:
                    priority = torch.maximum(priority, terminated.float())
                if args_cli.from_scratch:
                    # go2w：真实回报与想象目标用同一个奖励函数，critic 与 actor 不再各看一套。
                    # done 的 env 步后状态已被 reset，没有物理后继，用步前状态代替（该步只剩翻车惩罚有意义）。
                    s_next = reader.state()
                    s_next = torch.where(done[:, None], s, s_next)
                    env_rew = rew
                    rew = task.reward(s, s_next, cmd, action, prev)
                    if reward_cfg.terminated != 0.0:
                        rew = rew + reward_cfg.terminated * CONTROL_DT * terminated.float()
            sim["shaping_reward"] += shaping[actor_mask].mean().item() / args_cli.steps_per_iter
            if not args_cli.from_scratch:
                env_rew = rew
                rew = rew + shaping
            if priority is not None:
                sim["priority_rate"] = sim.get("priority_rate", 0.0) + priority.mean().item() / args_cli.steps_per_iter
            sim["shared_reward"] = sim.get("shared_reward", 0.0) + rew[actor_mask].mean().item() / args_cli.steps_per_iter
            learner.observe(s, action, prev, cmd, done, ctx, reward=rew, terminated=terminated, priority=priority)
            a = actor_mask
            v_w = quat_apply(s[a][:, QUAT], s[a][:, LIN])
            sim["env_reward"] += env_rew[a].mean().item() / args_cli.steps_per_iter
            sim["route_vel_err"] += (cmd[a, 0] - v_w[:, 0]).abs().mean().item() / args_cli.steps_per_iter
            # terminated 只剩 bad_orientation（成功/超时/出界都是 truncated），即翻车率
            sim["terminated"] += terminated[a].float().mean().item() / args_cli.steps_per_iter
        if learner.replay.size >= min_rows:
            logs = learner.update()
        else:
            logs = {}
            learner.iteration += 1
        logs.update(sim)
        logs["anchor_coef"] = cfg.anchor_coef
        if it % args_cli.log_interval == 0:
            logs.update(curriculum_stats(uenv))
            logs["elapsed_min"] = (time.time() - start_time) / 60
            for k, v in logs.items():
                writer.add_scalar(f"supermimic_terrain/{k}", v, it)
            row = {"iteration": it, **logs}
            if csv_writer is None:
                keys = set(row) | {"wm_loss", "wm_err_1step", "wm_err_last", "wm_reward_bias", "wm_reward_abs_gap",
                                   "actor_loss", "anchor_mse", "actor_grad_norm", "critic_loss", "imag_reward",
                                   "imag_survival", "actor_skipped", "terrain_level_mean", "stage_1", "stage_2",
                                   "stage_3"}
                csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(keys), extrasaction="ignore")
                if csv_file.tell() == 0:
                    csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()
            print(f"[it {it}] " + " ".join(f"{k}={v:.4g}" for k, v in sorted(logs.items())), flush=True)
        if (it + 1) % args_cli.save_interval == 0 or it + 1 == args_cli.iterations:
            torch.save(learner.state_dict(), log_dir / f"ckpt_{it + 1:06d}.pt")
            export_policy(learner, log_dir / "exported", task.obs_dim)
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
