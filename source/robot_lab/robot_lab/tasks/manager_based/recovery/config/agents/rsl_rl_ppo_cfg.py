# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-25
# @version v0.8.0
# @last_modified 2026-09-28
# @changelog
#   - v0.10.0 (2026-09-30): 增加失败末态重放的60/72维 residual 微调 runner。
#   - v0.9.0 (2026-09-30): 增加program31先验+有界residual的直接PPO runner。
#   - v0.8.0 (2026-09-28): 侧躺/翻覆专用非ED探索配置；不改变已有任务超参数。
#   - v0.7.0 (2026-09-17): 连续横滚即时控制预训练独立日志，网络/PPO保持Grounded基线合同。
#   - v0.6.0 (2026-09-17): Progress沿用基线PPO参数，只改变shaping以隔离实验变量。
#   - v0.5.0 (2026-09-17): 增加独立 grounded progress shaping 实验 runner。
#   - v0.4.0 (2026-09-17): 隔离真实失败状态池实验目录，网络合同不变。
#   - v0.3.0 (2026-09-17): grounded奖励使用独立日志目录，网络与原58/70维相同。
#   - v0.2.0 (2026-09-16): settled_ground 使用独立实验目录但保持58维网络兼容。
#   - v0.1.0 (2026-08-25): 恢复任务 PPO 配置；num_steps_per_env 必须与 CW 时间基准一致。

"""恢复策略的 RSL-RL PPO 配置（method1 / method2 共用）。

参考：`Kitjesen/recovery` 的 `thunder_recovery/config/recovery_ppo_cfg.py`。

**关键约束（易错）**：`num_steps_per_env` **必须**等于
`recovery/mdp/_utils.py::RECOVERY_STEPS_PER_ITER`（48）。
CW 衰减用 `common_step_counter / RECOVERY_STEPS_PER_ITER` 近似训练迭代数；
两者不一致会使 CW 衰减速度偏离预期——行为惩罚过早或过晚失效，
且**不会报错**，只表现为训练曲线异常。本文件用断言在导入时守住这一点。

**entropy_coef=0.0 的理由**：ED 整形本身就是探索机制（早期任务奖励≈0，
策略被鼓励大幅试错）。再加熵激励会与 ED 叠加，导致收敛期动作仍然抖动。
参考实现同样取 0。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from robot_lab.tasks.manager_based.recovery.mdp import RECOVERY_STEPS_PER_ITER


@configclass
class ParallelogramRoverRecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """asymmetric actor-critic PPO。

    网络规模 128×3 relu：参考实现按其已发布 checkpoint 的结构固定为此值。
    本机器人观测更小（actor 58 维 vs Thunder 78 维），128×3 仍充裕。
    """

    seed = 42
    num_steps_per_env = RECOVERY_STEPS_PER_ITER  # 必须与 CW 基准一致，见模块 docstring
    max_iterations = 10000
    save_interval = 200
    experiment_name = "parallelogram_rover_recovery"
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
        activation="relu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,  # ED 即探索机制，见模块 docstring
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=10.0,
    )


@configclass
class ParallelogramRoverSettledRecoveryPPORunnerCfg(ParallelogramRoverRecoveryPPORunnerCfg):
    """与原恢复 checkpoint 结构兼容；只隔离新协议的训练日志。"""

    experiment_name = "parallelogram_rover_recovery_settled"
    run_name = "settled_ground"


@configclass
class ParallelogramRoverGroundedRecoveryPPORunnerCfg(ParallelogramRoverRecoveryPPORunnerCfg):
    """隔离新奖励实验，避免与旧模型/旧成功率混报。"""

    experiment_name = "parallelogram_rover_recovery_grounded"
    run_name = "grounded"


@configclass
class ParallelogramRoverGroundedProgressRecoveryPPORunnerCfg(ParallelogramRoverRecoveryPPORunnerCfg):
    """有符号进展 shaping 的独立实验；PPO参数继承基线以隔离实验变量。"""

    experiment_name = "parallelogram_rover_recovery_grounded_progress"
    run_name = "orientation_height_progress"


@configclass
class ParallelogramRoverControlledTiltRecoveryPPORunnerCfg(ParallelogramRoverRecoveryPPORunnerCfg):
    """只隔离人工倾角预训练产物；不能与静止倒姿或失败池重放统计混用。"""

    experiment_name = "parallelogram_rover_recovery_controlled_tilt"
    run_name = "controlled_tilt"


@configclass
class ParallelogramRoverFailureReplayPPORunnerCfg(ParallelogramRoverRecoveryPPORunnerCfg):
    """真实失败动量的恢复实验不能与人工倒姿成功率混报。"""

    experiment_name = "parallelogram_rover_recovery_failure_replay"
    run_name = "failure_replay"


@configclass
class ParallelogramRoverSideRecoveryPPORunnerCfg(ParallelogramRoverRecoveryPPORunnerCfg):
    """非ED任务的独立探索起点，参数需闭环消融验证，不修改参考EDRS基线。"""

    experiment_name = "parallelogram_rover_recovery_side"
    run_name = "side_progress"

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.005
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "fixed"


@configclass
class ParallelogramRoverUpsideRecoveryPPORunnerCfg(ParallelogramRoverSideRecoveryPPORunnerCfg):
    """与侧躺保持控制兼容，隔离尚未验证可达性的翻覆实验产物。"""

    experiment_name = "parallelogram_rover_recovery_upside"
    run_name = "upside_experimental"


@configclass
class ParallelogramRoverSidePhasePPORunnerCfg(ParallelogramRoverSideRecoveryPPORunnerCfg):
    """阶段BC的保守微调起点；不把探索噪声奖励当作恢复任务奖励。"""

    experiment_name = "parallelogram_rover_recovery_side_phase"
    run_name = "side_phase"
    save_interval = 50

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3e-5
        self.algorithm.entropy_coef = 0.0


@configclass
class ParallelogramRoverSideResidualPPORunnerCfg(ParallelogramRoverSidePhasePPORunnerCfg):
    """program31 先验 + 有界 residual 的直接 PPO 起点。

    该任务不从 BC checkpoint resume：脚本先验在 action wrapper 中执行，Actor 的
    均值由训练入口置零，PPO 只学习 tanh residual。保留 phase/initial_side 观测，
    但不再使用 open-loop BC 的状态屏蔽逻辑。
    """

    experiment_name = "parallelogram_rover_recovery_side_residual"
    run_name = "side_residual"
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.schedule = "fixed"


@configclass
class ParallelogramRoverFailureReplayResidualPPORunnerCfg(ParallelogramRoverSideResidualPPORunnerCfg):
    """从真实失败末态继续优化 residual；保持与Side-Residual同一网络合同。"""

    experiment_name = "parallelogram_rover_recovery_failure_replay_residual"
    run_name = "failure_replay_residual"
    save_interval = 50
