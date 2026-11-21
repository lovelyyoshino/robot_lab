#!/usr/bin/env python3
"""
train_scan_student.py — 深度→高度扫描 latent 学生：冻结 Teacher-Preview actor，只训深度编码器。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-11-03
@changelog
    - v0.2.0 (2026-11-03): 接入预抬课程与加权蒸馏，保护采样标签、续训参数、传感合同和更新有限性。
    - v0.1.0 (2026-10-26): 初始创建

数据：在 Student-Preview 任务里，按回合以衰减概率由 teacher（干净232/233维观测）或学生驱动，
每轮采 `--steps_per_iter` 步，在这批样本上做若干 epoch 小批量更新。
产物：logs/supermimic_scan/<run>/ 下 ckpt_XXXXXX.pt（编码器 + 配置）与 exported/policy.pt
（TorchScript，输入4141维；带车道定位时4142维），以及深度合同 depth_contract.json。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Depth → height-scan latent student")
parser.add_argument("--task", default=(
    "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0"))
parser.add_argument("--teacher", required=True,
                    help="rsl_rl Teacher或原生SuperMimic地形ckpt（保留tanh及233维车道观测）")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=4000)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume", default=None)
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--latent_dim", type=int, default=None)
parser.add_argument("--prelift_training", action=argparse.BooleanOptionalAction, default=None,
                    help="使用预抬训练地形分布；续训默认沿用存档")
parser.add_argument("--clean_proprio", action=argparse.BooleanOptionalAction, default=None,
                    help="只关闭本体观测噪声，保留相机扰动；续训沿用保存的传感合同")
parser.add_argument("--camera_corruption", action=argparse.BooleanOptionalAction, default=None,
                    help="相机噪声/延迟/丢帧；关闭时仍保留确定性历史采样步长")
for name, typ in (("lr", float), ("action_loss_coef", float), ("teacher_drive_decay_updates", int),
                  ("scan_step_weight", float), ("scan_edge_weight", float), ("scan_edge_threshold", float),
                  ("arm_action_weight", float), ("epochs", int), ("minibatch", int), ("max_grad_norm", float),
                  ("teacher_drive_start", float), ("teacher_drive_end", float)):
    parser.add_argument(f"--{name}", type=typ, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from dataclasses import asdict, fields  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import robot_lab.tasks  # noqa: F401, E402
import torch  # noqa: E402
from robot_lab.supermimic import scan_student as ss  # noqa: E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import (  # noqa: E402
    vision_depth_contract,
)
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def linear(start, end, progress, horizon):
    return end if horizon <= 0 else start + (end - start) * min(progress / horizon, 1.0)


def main():
    torch.manual_seed(args_cli.seed)
    resume = torch.load(args_cli.resume, map_location="cpu", weights_only=False) if args_cli.resume else None
    overrides = {field.name: getattr(args_cli, field.name) for field in fields(ss.ScanStudentCfg)
                 if getattr(args_cli, field.name, None) is not None}
    cfg = ss.resolve_student_config(resume.get("cfg") if resume else None, **overrides)
    latent_dim = args_cli.latent_dim if args_cli.latent_dim is not None else (resume or {}).get("latent_dim", 128)
    if latent_dim <= 0 or (resume is not None and latent_dim != resume["latent_dim"]):
        raise ValueError("latent_dim 必须为正，且续训时必须与存档相同")
    prelift_training = (args_cli.prelift_training if args_cli.prelift_training is not None
                        else (resume or {}).get("prelift_training", False))
    saved_noise = (resume or {}).get("observation_noise", {"clean_proprio": False, "camera_corruption": True})
    observation_noise = {name: saved_noise[name] if getattr(args_cli, name) is None else getattr(args_cli, name)
                         for name in ("clean_proprio", "camera_corruption")}
    if resume is not None and observation_noise != saved_noise:
        raise ValueError("续训不可改变本体/相机扰动合同；请沿用保存配置")
    if min(args_cli.iterations, args_cli.steps_per_iter, args_cli.save_interval, args_cli.log_interval) <= 0:
        raise ValueError("训练/采样步数与保存/日志间隔必须为正数")
    start_iter = resume["iteration"] if resume else 0
    if args_cli.iterations <= start_iter:
        raise ValueError("iterations 必须大于续训起点，避免生成没有更新的新实验")
    run = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ss.create_student_run_directory("logs/supermimic_scan", run)
    (log_dir / "exported").mkdir()
    (log_dir / "params").mkdir()

    teacher_ckpt = torch.load(args_cli.teacher, map_location="cpu", weights_only=False)
    with open(args_cli.teacher, "rb") as stream:
        teacher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    teacher, lane_obs = ss.teacher_actor_from_checkpoint(teacher_ckpt)
    if resume is not None and (resume.get("teacher_sha256") != teacher_sha256
                               or resume.get("lane_obs", False) != lane_obs):
        raise ValueError("续训必须使用相同教师权重和车道观测")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # 校验共同的关节顺序、扫描坐标与动作语义，不能仅凭232/233维相同就换教师。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reinforcement_learning/rsl_rl"))
    from vision_contract import contract_for_cfg, restore_height_scan_period, validate_contract

    source_contract = Path(args_cli.teacher).parent / "params/vision_contract.json"
    teacher_contract = json.loads(source_contract.read_text())
    restore_height_scan_period(env_cfg, teacher_contract)
    target_contract = contract_for_cfg(
        args_cli.task, env_cfg, load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"))
    validate_contract(teacher_contract, target_contract, teacher_source=True)
    if prelift_training:
        from robot_lab.tasks.manager_based.supermimic.prelift_env_cfg import configure_prelift_training

        # 教师保存的是部署接口合同；先严格验证观测/动作，再显式切换已授权的学生训练分布。
        # 实际分布另存training_vision_contract，不能拿24列部署合同冒充20列训练配置。
        configure_prelift_training(env_cfg, student=True)
    ss.configure_student_lane(env_cfg, lane_obs)
    ss.configure_student_observation_noise(env_cfg, **observation_noise)
    depth_contract = ss.student_depth_contract(vision_depth_contract(env_cfg), lane_obs, observation_noise)
    if resume is not None and resume.get("depth_contract", depth_contract) != depth_contract:
        raise ValueError("续训深度合同与存档不同：请核对相机、历史步长和扰动配置")
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    teacher = teacher.to(device)
    encoder = ss.DepthScanEncoder(latent_dim).to(device)
    policy = ss.ScanStudentPolicy(encoder, teacher, lane_obs=lane_obs)
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    if resume is not None:
        encoder.load_state_dict(resume["encoder"])
        ss.restore_student_optimizer(opt, resume["optimizer"], cfg)

    (log_dir / "depth_contract.json").write_text(json.dumps(depth_contract, indent=2))
    training_contract = contract_for_cfg(
        args_cli.task, env_cfg, load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"))
    source_paths = {
        "scan_student.py": Path(ss.__file__), "train_scan_student.py": Path(__file__),
        "prelift_env_cfg.py": Path(ss.__file__).parents[1] / "tasks/manager_based/supermimic/prelift_env_cfg.py",
    }
    implementation_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    metadata = {
        "teacher": str(Path(args_cli.teacher).resolve()), "teacher_sha256": teacher_sha256,
        "teacher_method": "supermimic_bptt" if "world_model" in teacher_ckpt else "rsl_rl",
        "student_method": "depth_scan_and_action_distillation", "lane_obs": lane_obs,
        "cfg": asdict(cfg), "latent_dim": latent_dim, "prelift_training": prelift_training,
        "observation_noise": observation_noise, "depth_contract": depth_contract,
        "arguments": vars(args_cli), "resume_from": str(Path(args_cli.resume).resolve()) if resume else None,
        "resume_config_overrides": overrides, "start_iteration": start_iter,
        "implementation_sha256": implementation_sha256,
        "deployment_contract": "params/vision_contract.json",
        "training_contract": "params/training_vision_contract.json",
        "training_objective": training_contract.get("shared", {}).get("training_objective", {}),
        "training_terrain_proportions": {name: terrain.proportion for name, terrain
                                         in env_cfg.scene.terrain.terrain_generator.sub_terrains.items()},
    }
    (log_dir / "lineage.json").write_text(json.dumps(metadata, indent=2))
    (log_dir / "params" / "student_training.json").write_text(json.dumps(metadata, indent=2))
    (log_dir / "params" / "env.json").write_text(json.dumps(env_cfg.to_dict(), indent=2, default=str))
    # 此文件只描述已核验的部署接口；实际训练目标/地形见相邻的training_vision_contract与env.json。
    (log_dir / "params" / "vision_contract.json").write_text(json.dumps(target_contract, indent=2))
    (log_dir / "params" / "training_vision_contract.json").write_text(json.dumps(training_contract, indent=2))
    writer = SummaryWriter(str(log_dir))
    with open(log_dir / "train_log.csv", "a", newline="") as csv_file:
        train_iterations(env, policy, opt, cfg, start_iter, depth_contract, metadata, log_dir, writer, csv_file)
    writer.close()
    env.close()


def train_iterations(env, policy, opt, cfg, start_iter, depth_contract, metadata, log_dir, writer, csv_file):
    """采样与更新共用配置；文件资源由调用方持有并在异常退出时关闭。"""
    teacher, encoder = policy.teacher_actor, policy.encoder
    device = env.unwrapped.device
    csv_writer = None

    obs, _ = env.reset()
    if (obs["policy"].shape[-1] != depth_contract["student_obs_dim"]
            or obs["teacher"].shape[-1] != depth_contract["teacher_obs_dim"]):
        raise RuntimeError(f"观测维度不符: policy={obs['policy'].shape}, teacher={obs['teacher'].shape}")
    n = args_cli.num_envs
    teacher_driven = torch.rand(n, device=device) < cfg.teacher_drive_start
    for it in range(start_iter, args_cli.iterations):
        p_teacher = linear(cfg.teacher_drive_start, cfg.teacher_drive_end, it, cfg.teacher_drive_decay_updates)
        student_buf, teacher_buf, rewards = [], [], []
        with torch.no_grad():
            for _ in range(args_cli.steps_per_iter):
                s_obs, t_obs = ss.clone_student_observations(obs)
                student_buf.append(s_obs)
                teacher_buf.append(t_obs)
                # 回合级驱动者：同一回合内不切换，避免拼出两个策略都不会访问的状态。
                action = torch.where(teacher_driven[:, None], teacher(t_obs), policy(s_obs))
                if not torch.isfinite(action).all():
                    raise FloatingPointError("深度学生/教师动作非有限，已拒绝推进仿真")
                obs, rew, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                rewards.append(rew[~teacher_driven].mean() if (~teacher_driven).any() else rew.new_zeros(()))
                if done.any():
                    teacher_driven[done] = torch.rand(int(done.sum()), device=device) < p_teacher
        student_data, teacher_data = torch.cat(student_buf), torch.cat(teacher_buf)
        del student_buf, teacher_buf

        logs = {"loss": 0.0, "scan_mse": 0.0, "action_mse": 0.0}
        updates = 0
        for _ in range(cfg.epochs):
            for idx in torch.randperm(student_data.shape[0], device=device).split(cfg.minibatch):
                loss, scan_loss, action_loss = ss.train_student_batch(
                    policy, opt, student_data[idx], teacher_data[idx], cfg)
                logs["loss"] += loss.item()
                logs["scan_mse"] += scan_loss.item()
                logs["action_mse"] += action_loss.item()
                updates += 1
        logs = {k: v / max(updates, 1) for k, v in logs.items()}
        logs["teacher_drive_p"] = p_teacher
        logs["lr"] = opt.param_groups[0]["lr"]
        logs["student_step_reward"] = torch.stack(rewards).mean().item()
        with torch.no_grad():
            probe = student_data[:128]
            live = policy(probe)
            blank = probe.clone()
            blank[:, ss.PROPRIO_DIM:ss.STUDENT_OBS_DIM] = 1.0
            logs["blank_depth_action_delta"] = (live - policy(blank)).square().mean().sqrt().item()
            frozen = probe.clone()
            depth = frozen[:, ss.PROPRIO_DIM:ss.STUDENT_OBS_DIM].reshape(-1, ss.DEPTH_FRAMES, ss.DEPTH_H * ss.DEPTH_W)
            depth[:, 1:] = depth[:, :1].clone()
            logs["constant_history_action_delta"] = (live - policy(frozen)).square().mean().sqrt().item()
            # 诊断只取有限批量，避免整轮图像再次前向造成相机训练显存峰值。
            scan_pred = encoder(probe)
            scan_target = teacher_data[:len(probe), ss.PROPRIO_DIM:ss.TEACHER_OBS_DIM]
            weights = ss.scan_loss_weights(scan_target, cfg)
            error = (scan_pred - scan_target).square()
            important = weights > 1.0
            logs["scan_raw_mse"] = error.mean().item()
            edges = ss.scan_edge_mask(scan_target, cfg)
            logs["scan_edge_fraction"] = edges.float().mean().item()
            logs["scan_edge_mse"] = error[edges].mean().item() if edges.any() else 0.0
            logs["scan_priority_fraction"] = important.float().mean().item()
            logs["scan_priority_mse"] = (error[important].mean().item() if important.any() else 0.0)

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
            torch.save({**metadata, "iteration": it + 1, "encoder": encoder.state_dict(),
                        "optimizer": opt.state_dict()},
                       log_dir / f"ckpt_{it + 1:06d}.pt")
            torch.jit.script(copy.deepcopy(policy).cpu().eval()).save(str(log_dir / "exported" / "policy.pt"))

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
