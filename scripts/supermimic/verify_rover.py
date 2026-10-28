#!/usr/bin/env python3
"""
verify_rover.py — 训练前校验 SuperMimic 平地链路，不产出可部署策略。

三项检查（全部基于真实仿真）：
1. contract：解析观测 vs 仿真 45 维观测；解析奖励各项 vs RewardManager 同名项；
   并列出 env 中有、解析奖励没有的项及其量级（知道想象里少了什么）。
2. wheel：用 PPO 驱动策略实测有效轮半径，检查 actor 动作界能否覆盖命令速度。
3. wm_probe：PPO 驱动 + 分档噪声采数据，离线训练世界模型，在留出环境上与常速度基线比较
   k=1/8/32 开环误差——这是 plan.md 第 ③ 步的过门实验。

@author pony
@date 2026-10-26
@version v0.2.0
@last_modified 2026-10-26
@changelog
    - v0.2.0 (2026-10-26): 关节加速度三项改为逐项核对；世界模型探针对照有/无 ξ，报 aux 校准与 PPO 基线。
    - v0.1.0 (2026-10-26): 初始创建
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SuperMimic rover pipeline verification")
parser.add_argument("--task", default="RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0")
parser.add_argument("--driver", default="logs/rsl_rl/parallelogram_rover_rough/2026-08-13_17-35-42/exported/policy.pt",
                    help="45→12 TorchScript 驱动策略（rough blind PPO 导出）")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--heldout_envs", type=int, default=256)
parser.add_argument("--collect_steps", type=int, default=1200)
parser.add_argument("--wm_updates", type=int, default=6000)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=[1024, 1024, 1024])
parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.6])
parser.add_argument("--output", default="logs/supermimic/verify_rover.json")
parser.add_argument("--seed", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402
from robot_lab.supermimic import rover_task  # noqa: E402
from robot_lab.supermimic.diagnostics import open_loop_errors  # noqa: E402
from robot_lab.supermimic.isaac_adapter import CONTEXT_DIM, RoverStateReader  # noqa: E402
from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from robot_lab.supermimic.state import ANG, JVEL, LIN, constant_velocity_step  # noqa: E402

# RewardManager 项名 → rover_task.reward_terms 项名
REWARD_NAME_MAP = {
    "track_lin_vel_xy_exp": "track_lin_vel",
    "track_ang_vel_z_exp": "track_ang_vel",
    "lin_vel_z_l2": "lin_vel_z",
    "ang_vel_xy_l2": "ang_vel_xy",
    "action_rate_l2": "action_rate",
    "joint_pos_penalty": "suspension_pos",
    "joint_deviation_steering_l1": "steering_deviation",
    "joint_vel_steering_l2": "steering_vel",
    "upward": "upward",
}
# 关节加速度三项：解析奖励直接吃 env 真值 acc_sq，因此也应逐项一致（核对的是权重与组划分）。
REWARD_NAME_MAP.update({
    "joint_acc_l2": "suspension_acc",
    "joint_acc_steering_l2": "steering_acc",
    "joint_acc_wheel_l2": "wheel_acc",
})


def main():
    torch.manual_seed(args_cli.seed)
    report = {}
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    reader = RoverStateReader(env)
    device = uenv.device
    driver = torch.jit.load(args_cli.driver, map_location=device).eval()
    reward_cfg = rover_task.RoverRewardCfg()
    rm = uenv.reward_manager
    term_names = list(rm.active_terms)
    weights = {name: rm.get_term_cfg(name).weight for name in term_names}

    # 每个环境固定一个噪声档，覆盖从"纯 PPO"到"大扰动"的状态分布。
    levels = torch.tensor(args_cli.noise_levels, device=device)
    env_noise = levels[torch.arange(args_cli.num_envs, device=device) % len(levels)]

    bound = SuperMimicCfg().action_bound
    raw_p99 = []
    obs, _ = env.reset()
    S, A, D, C, X, R, CMD = [], [], [], [], [], [], []
    obs_err, reward_err, reward_count = 0.0, {}, 0
    missing_mag = {n: 0.0 for n in term_names if n not in REWARD_NAME_MAP and n not in rover_task.ACC_AUX_NAMES}
    t0 = time.time()
    for step in range(args_cli.collect_steps):
        s, cmd, prev, ctx = reader.state(), reader.command(), reader.last_action(), reader.context()
        policy_obs = rover_task.policy_observation(s, cmd, prev)
        obs_err = max(obs_err, (policy_obs - obs["policy"]).abs().max().item())
        with torch.no_grad():
            action = driver(policy_obs) + env_noise[:, None] * torch.randn(args_cli.num_envs, 12, device=device)
            raw_p99.append(action.abs().quantile(0.99, dim=0))
            # PPO 原始轮动作可达 ±100+；截到 actor 的输出界，世界模型只学 actor 能到达的动作区。
            action = action.clamp(-bound, bound)
        obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        acc = reader.acc_aux()
        for buf, v in zip((S, A, D, C, X, R, CMD), (s, action, done, ctx, acc, rew, cmd)):
            buf.append(v)
        # 奖励核对：只取未 reset 且命令未重采样的环境（RewardManager 在 reset/重采样之前计算）。
        keep = ~done & (reader.command() == cmd).all(-1)
        if keep.any() and step > 0:
            terms = rover_task.reward_terms(s, reader.state(), cmd, action, prev, reward_cfg, acc)
            for env_name, ours in REWARD_NAME_MAP.items():
                if env_name not in term_names:
                    continue
                ref = rm._step_reward[:, term_names.index(env_name)]
                err = (terms[ours] - ref)[keep].abs().max().item()
                reward_err[env_name] = max(reward_err.get(env_name, 0.0), err)
            for name in missing_mag:
                missing_mag[name] += rm._step_reward[:, term_names.index(name)][keep].mean().item()
            reward_count += 1
    S, A, D, C, X, R, CMD = (torch.stack(v) for v in (S, A, D, C, X, R, CMD))
    # 旧做法（控制步差分代理）与 env 真值的比值：说明为什么必须用 aux 头。
    nxt_ok = ~D[:-1]
    proxy = rover_task.proxy_acc_sq(S[:-1], S[1:])[nxt_ok].mean(0)
    truth = X[:-1][nxt_ok].mean(0)
    clean = env_noise == 0.0
    report["contract"] = {
        "obs_max_abs_err": obs_err,
        "reward_term_max_abs_err": reward_err,
        "joint_acc_proxy_over_env_ratio": dict(zip(rover_task.ACC_AUX_NAMES, (proxy / truth).tolist())),
        "reward_weights_env": {k: weights[k] for k in REWARD_NAME_MAP if k in weights},
        "env_terms_not_modeled_mean_weighted": {k: v / max(reward_count, 1) for k, v in missing_mag.items()},
        "context_std_across_envs": {
            "kp_ratio": C[0, :, :8].std(0).mean().item(), "kd_ratio": C[0, :, 8:20].std(0).mean().item(),
            "mass_and_com": C[0, :, 20:].std(0).tolist()},
        "collect_seconds": time.time() - t0,
    }
    # PPO 基线（纯驱动、无噪声环境）：SuperMimic 训练指标要对照的目标。
    report["ppo_baseline"] = {
        "lin_err_mps": (CMD[:, clean, :2] - S[:, clean][..., LIN][..., :2]).norm(dim=-1).mean().item(),
        "ang_err_radps": (CMD[:, clean, 2] - S[:, clean][..., ANG][..., 2]).abs().mean().item(),
        "env_reward_per_step": R[:, clean].mean().item(),
    }

    # 有效轮半径：PPO 纯驱动环境里，直线行驶时 v_x / 平均 |轮速|。
    v = S[..., LIN][..., 0][:, clean]
    w = S[..., JVEL][..., 8:].abs().mean(-1)[:, clean]
    moving = (v.abs() > 0.3) & (w > 1.0)
    radius = (v[moving].abs() / w[moving]).median().item() if moving.any() else float("nan")
    raw = torch.stack(raw_p99).mean(0)
    report["wheel"] = {
        "effective_radius_m": radius,
        "wheel_speed_needed_for_1mps_radps": 1.0 / radius,
        "actor_wheel_speed_bound_radps": bound * 5.0,  # 轮速动作 scale=5
        "driver_raw_action_abs_p99_leg": raw[:8].max().item(),
        "driver_raw_action_abs_p99_wheel": raw[8:].max().item(),
        "max_wheel_speed_reached_radps": S[..., JVEL][..., 8:][:, clean].abs().quantile(0.99).item(),
        "max_lin_vel_reached_mps": v.abs().quantile(0.99).item(),
    }

    # 世界模型探针：前 N-heldout 个环境训练，留出环境评测；对照"有/无 ξ"两个模型。
    n_train = args_cli.num_envs - args_cli.heldout_envs
    held = slice(n_train, None)
    prev_actions = torch.cat((torch.zeros_like(A[:1]), A[:-1]))
    report["wm_probe"] = {}
    for variant, context_dim in (("no_context", 0), ("context", CONTEXT_DIM)):
        torch.manual_seed(args_cli.seed)
        cfg = SuperMimicCfg(num_envs=n_train, buffer_len=args_cli.collect_steps, wm_hidden=tuple(args_cli.wm_hidden),
                            wm_updates=1, context_dim=context_dim, aux_dim=rover_task.AUX_DIM)
        learner = SuperMimicLearner(cfg, device)
        for t in range(args_cli.collect_steps):
            learner.observe(S[t, :n_train], A[t, :n_train], prev_actions[t, :n_train], CMD[t, :n_train],
                            D[t, :n_train], C[t, :n_train, :context_dim], X[t, :n_train])
        curve, t0 = [], time.time()
        for u in range(args_cli.wm_updates):
            logs = learner.update_world_model()
            if u % 500 == 0 or u == args_cli.wm_updates - 1:
                curve.append({"update": u, **logs})
                print(f"[wm {variant} {u}] " + " ".join(f"{k}={v:.4g}" for k, v in logs.items()), flush=True)
        wm = learner.world_model
        ctx_all = C[..., :context_dim]
        by_noise = {}
        for level in args_cli.noise_levels:
            cols = (env_noise[held] == level).nonzero(as_tuple=False).squeeze(-1) + n_train
            by_noise[f"noise_{level}"] = {
                "world_model": open_loop_errors(S[:, cols], A[:, cols], D[:, cols], wm, contexts=ctx_all[:, cols]),
                "constant_velocity": open_loop_errors(
                    S[:, cols], A[:, cols], D[:, cols], lambda s, a, c: constant_velocity_step(s, rover_task.CONTROL_DT)),
            }
        # aux 头校准：留出环境上预测 Σacc² 的均值与 env 真值之比、逐样本相关系数。
        with torch.no_grad():
            pa = wm.aux(S[:, held], A[:, held], ctx_all[:, held]).reshape(-1, rover_task.AUX_DIM)
        ta = X[:, held].reshape(-1, rover_task.AUX_DIM)
        corr = [torch.corrcoef(torch.stack((pa[:, i], ta[:, i])))[0, 1].item() for i in range(rover_task.AUX_DIM)]
        report["wm_probe"][variant] = {
            "train_seconds": time.time() - t0, "curve": curve, "heldout": by_noise,
            "aux_mean_ratio_pred_over_env": dict(zip(rover_task.ACC_AUX_NAMES, (pa.mean(0) / ta.mean(0)).tolist())),
            "aux_corr": dict(zip(rover_task.ACC_AUX_NAMES, corr)),
        }

    Path(args_cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("contract", "ppo_baseline", "wheel")}, indent=2))
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
