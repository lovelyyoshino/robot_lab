# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 新增保持相同网络合同的四轮起立阶段 runner。
#   - v0.1.0 (2026-08-25): 直立任务 PPO 配置；gamma 提高以适应倒立摆长时序信用分配。

"""直立策略的 RSL-RL PPO 配置。

**与 recovery PPO 配置的三处差异，各有依据**：

1. `entropy_coef = 0.005`（recovery 用 0.0）。
   recovery 靠 ED 整形提供探索（早期任务奖励≈0，策略被迫大幅试错）；
   直立没有 ED 机制，需要显式熵激励。且平衡任务的动作噪声不像行驶那样有害。

2. `gamma = 0.995`（recovery 用 0.99）。
   倒立摆时间常数 √(L/g) ≈ 0.24 s，50 Hz 下约 12 步。要学"提前预判倒向"
   需要更长的信用分配窗口；0.99 的有效视野约 100 步（2 s），0.995 约 200 步（4 s）。

3. `num_steps_per_env = 16`（recovery 用 48）。
   **由 L3 实测调整**：初版设 24，但实测随机策略 episode 仅 49 步
   （早终止在 ≈1 s 触发），24/49 意味着约半数 rollout 跨越 `done` 边界，
   截断的轨迹会削弱 GAE 估计。降到 16 使 rollout/episode ≈ 0.33，
   跨界比例明显下降；同时更短的 rollout 提高更新频率，适合早期快速收敛。

   ⚠️ 这一项与 `episode_length_s` 相互约束：训练后期 episode 变长后，
   可回调 `num_steps_per_env` 以摊薄更新开销。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ParallelogramRoverUprightPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """asymmetric actor-critic PPO（actor 42 维、critic 47 维）。"""

    seed = 42
    num_steps_per_env = 16  # 见模块 docstring 差异 3（由 L3 实测调整）
    max_iterations = 6000
    save_interval = 200
    experiment_name = "parallelogram_rover_upright"
    run_name = ""
    logger = "tensorboard"
    resume = False

    class_name = "OnPolicyRunner"

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 见模块 docstring 差异 1
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,  # 见模块 docstring 差异 2
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ParallelogramRoverUprightGetupPPORunnerCfg(ParallelogramRoverUprightPPORunnerCfg):
    """从纯平衡策略继续训练的四轮起立阶段。"""

    max_iterations = 8000
    experiment_name = "parallelogram_rover_upright_getup"
    run_name = "stage2_getup"
