# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.2.0
# @last_modified 2026-09-16
# @changelog
#   - v0.2.0 (2026-09-16): 为显式时序增强 student 增加独立实验名，防止误续训旧预处理合同。
#   - v0.1.0 (2026-09-16): 新增视觉 teacher PPO 与 depth student 蒸馏配置。

"""平行四边形轮足车视觉行走的 RSL-RL 配置。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from .rsl_rl_ppo_cfg import ParallelogramRoverRoughPPORunnerCfg


@configclass
class ParallelogramRoverVisionTeacherPPORunnerCfg(ParallelogramRoverRoughPPORunnerCfg):
    """height-scan teacher 从头训练，不能直接加载 45 维 blind actor。"""

    max_iterations = 8000
    save_interval = 100
    experiment_name = "parallelogram_rover_vision_teacher"
    run_name = "height_scan"


@configclass
class ParallelogramRoverVisionStudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """将 height-scan teacher 蒸馏到深度历史 student。"""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "parallelogram_rover_vision_student"
    run_name = "depth_distillation"
    logger = "tensorboard"
    resume = False
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    clip_actions = None

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[1024, 512, 256],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )


@configclass
class ParallelogramRoverVisionStudentTemporalDistillationRunnerCfg(
    ParallelogramRoverVisionStudentDistillationRunnerCfg
):
    """教师架构不变；student 必须使用 Temporal 变体，不将输入同为 4141 当作兼容依据。"""

    experiment_name = "parallelogram_rover_vision_student_temporal"
    run_name = "depth_stride5_latency_drop_freeze"
