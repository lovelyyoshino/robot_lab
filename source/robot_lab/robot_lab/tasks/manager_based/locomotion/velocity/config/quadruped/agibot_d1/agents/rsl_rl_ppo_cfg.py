# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AgibotD1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "agibot_d1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # 降低初始噪声，提高策略稳定性
        actor_obs_normalization=True,  # 启用观察值归一化
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 降低熵系数，减少探索，更快收敛
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # 降低学习率，提高稳定性
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,  # 降低目标KL，更保守的更新
        max_grad_norm=1.0,
    )


@configclass
class AgibotD1FlatPPORunnerCfg(AgibotD1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "agibot_d1_flat"


@configclass
class AgibotD1SteppingStonesPPORunnerCfg(AgibotD1RoughPPORunnerCfg):
    """PPO configuration for Agibot D1 on stepping stones terrain.

    This configuration is optimized for learning precise foot placement
    on discrete platforms with gaps between them.
    """

    def __post_init__(self):
        super().__post_init__()

        # Experiment settings
        self.max_iterations = 30000  # Longer training for challenging terrain
        self.experiment_name = "agibot_d1_stepping_stones"
        self.save_interval = 200

        # Slightly larger network for complex terrain navigation
        self.policy.actor_hidden_dims = [512, 256, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 256, 128]

        # Slightly higher initial noise for exploration
        self.policy.init_noise_std = 1.2

        # Algorithm tuning for stepping stones
        self.algorithm.entropy_coef = 0.008  # Slightly lower for more exploitation
        self.algorithm.num_learning_epochs = 6
        self.algorithm.learning_rate = 8.0e-4  # Slightly lower for stability


@configclass
class AgibotD1CompetitionPPORunnerCfg(AgibotD1RoughPPORunnerCfg):
    """PPO configuration for Agibot D1 on competition terrain.

    This configuration matches the Isaac Gym legged robot competition:
    - num_steps_per_env = 48 (competition uses 48)
    - max_iterations = 6000 (competition default)
    - Network architecture matches competition
    """

    def __post_init__(self):
        super().__post_init__()

        # Experiment settings - match competition
        self.num_steps_per_env = 48  # Competition: 48 (was 24)
        self.max_iterations = 6000  # Competition: 6000
        self.experiment_name = "agibot_d1_competition"
        self.save_interval = 50  # Competition: 50

        # Network architecture - match competition
        self.policy.actor_hidden_dims = [512, 256, 128]  # Competition default
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.init_noise_std = 1.0  # Competition: 1.0

        # Algorithm tuning - match competition
        self.algorithm.entropy_coef = 0.01  # Competition: 0.01
        self.algorithm.num_learning_epochs = 5  # Competition: 5
        self.algorithm.num_mini_batches = 4  # Competition: 4
        self.algorithm.learning_rate = 1.0e-3  # Competition: 1.e-3
