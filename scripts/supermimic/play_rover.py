#!/usr/bin/env python3
"""
play_rover.py — 回放 SuperMimic 平地策略，并导出部署用 policy.pt / policy.onnx。

用法与 scripts/reinforcement_learning/rsl_rl/play.py 保持一致：--checkpoint 指定 ckpt_XXXXXX.pt，
省略时取 logs/supermimic/<--load_run 或最新 run>/ 下编号最大的 checkpoint；支持 --keyboard、
--video、--real-time、--max_steps，默认 64 台车。回放关闭推车/外力/观测噪声（与 play.py 相同）。
导出写到 checkpoint 同级 exported/：45 维观测 → 12 维原始动作（含 tanh 界），与 PPO blind 接口
和 RL-SAR 的 45→12 合同相同，可直接替换。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a SuperMimic rover checkpoint")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0")
parser.add_argument("--checkpoint", default=None, help="ckpt_XXXXXX.pt；省略时自动选最新")
parser.add_argument("--load_run", default=None, help="logs/supermimic 下的 run 名；省略时取最新 run")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=None, help="可选的回放步数，适合无界面导出检查。")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--keyboard", action="store_true", default=False, help="单车键盘遥控（与 play.py 相同按键）")
parser.add_argument("--export_only", action="store_true", help="只导出 policy.pt/onnx，不启动仿真")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.export_only:
    args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.assets.parallelogram_rover import PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES  # noqa: E402
from robot_lab.supermimic import rover_task  # noqa: E402
from robot_lab.supermimic.networks import Actor  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning")))
from rl_utils import camera_follow  # noqa: E402

DEFAULT_PLAY_ENVS = 64


def resolve_checkpoint() -> Path:
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = Path("logs/supermimic")
    runs = [root / args_cli.load_run] if args_cli.load_run else sorted(
        (p for p in root.iterdir() if p.is_dir() and any(p.glob("ckpt_*.pt"))), key=lambda p: p.stat().st_mtime)
    if not runs or not runs[-1].is_dir():
        raise FileNotFoundError(f"{root} 下没有含 ckpt_*.pt 的 run")
    ckpts = sorted(runs[-1].glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"{runs[-1]} 下没有 ckpt_*.pt")
    return ckpts[-1].resolve()


def load_actor(path: Path, device) -> Actor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = data["cfg"]
    actor = Actor(rover_task.OBS_DIM, tuple(cfg["actor_hidden"]), cfg["action_bound"])
    actor.load_state_dict(data["actor"])
    print(f"[INFO] SuperMimic checkpoint: {path}（iteration {data['iteration']}）")
    return actor.to(device).eval()


def export(actor: Actor, ckpt: Path) -> Path:
    out = ckpt.parent / "exported"
    out.mkdir(parents=True, exist_ok=True)
    cpu = copy.deepcopy(actor).cpu().eval()
    torch.jit.script(cpu).save(str(out / "policy.pt"))
    torch.onnx.export(cpu, torch.zeros(1, rover_task.OBS_DIM), str(out / "policy.onnx"),
                      input_names=["obs"], output_names=["actions"], dynamic_axes={"obs": {0: "batch"}}, opset_version=17)
    # 写出部署合同：维度相同不代表语义相同，RL-SAR 侧据此核对关节顺序和动作尺度。
    contract = {
        "source": "supermimic", "checkpoint": str(ckpt), "obs_dim": rover_task.OBS_DIM, "action_dim": 12,
        "obs_layout": ["base_ang_vel*0.25 (3)", "projected_gravity (3)", "velocity_commands (3)",
                       "joint_pos_rel, wheel cols=0 (12)", "joint_vel*0.05 (12)", "last_action (12)"],
        "action_joint_order": list(PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES),
        "action_scale": {"arm": 0.25, "steering": 0.25, "wheel_velocity": 5.0},
        "action_output_bound": actor.action_bound,
    }
    (out / "policy_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    print(f"[INFO] 已导出 {out / 'policy.pt'}、policy.onnx、policy_contract.json")
    return out


def main():
    ckpt = resolve_checkpoint()
    device = args_cli.device or "cuda:0"
    actor = load_actor(ckpt, device)
    export(actor, ckpt)
    if args_cli.export_only:
        return

    env_cfg = parse_env_cfg(args_cli.task, device=device,
                            num_envs=args_cli.num_envs if args_cli.num_envs is not None else DEFAULT_PLAY_ENVS)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    # 与 play.py 相同：回放不加观测噪声和推扰（本任务训练时也已关闭，这里显式写出）。
    env_cfg.observations.policy.enable_corruption = False
    for name in ("randomize_apply_external_force_torque", "randomize_push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        ranges = env_cfg.commands.base_velocity.ranges
        controller = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=ranges.lin_vel_x[1], v_y_sensitivity=ranges.lin_vel_y[1],
            omega_z_sensitivity=ranges.ang_vel_z[1]))
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {"video_folder": str(ckpt.parent / "videos" / "play"), "step_trigger": lambda step: step == 0,
                        "video_length": args_cli.video_length, "disable_logger": True}
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 直接用 env 的 45 维 policy 观测（与部署一致）；解析观测与它逐列一致已由 verify_rover 核对。
    obs, _ = env.reset()
    dt = env.unwrapped.step_dt
    timestep = 0
    try:
        while simulation_app.is_running():
            start = time.time()
            with torch.inference_mode():
                obs, _, _, _, _ = env.step(actor(obs["policy"]))
            timestep += 1
            if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                break
            if args_cli.video and timestep == args_cli.video_length:
                break
            if args_cli.keyboard:
                camera_follow(env)
            sleep_time = dt - (time.time() - start)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
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
