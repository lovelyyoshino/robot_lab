# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-09-16
# @version v0.4.0
# @last_modified 2026-10-09
# @changelog
#   - v0.4.0 (2026-10-09): Preview student 改用 DAgger 混合蒸馏，每轮更新3→24次并线性衰减学习率。
#   - v0.3.0 (2026-09-30): 为前瞻抬轮课程提供独立teacher/student实验名，不改旧任务。
#   - v0.2.0 (2026-09-16): 为显式时序增强 student 增加独立实验名，防止误续训旧预处理合同。
#   - v0.1.0 (2026-09-16): 新增视觉 teacher PPO 与 depth student 蒸馏配置。

"""平行四边形轮足车视觉行走的 RSL-RL 配置。"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from .dagger_distillation import register as register_dagger_distillation
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


@configclass
class ParallelogramRoverVisionTeacherPreviewPPORunnerCfg(ParallelogramRoverVisionTeacherPPORunnerCfg):
    experiment_name = "parallelogram_rover_vision_teacher_preview"
    run_name = "preview_teacher"


@configclass
class RslRlDaggerDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """原生蒸馏参数之外增加 teacher 驱动比例与学习率的线性衰减。"""

    class_name: str = "DaggerDistillation"
    teacher_drive_start: float = 0.5
    teacher_drive_end: float = 0.0
    teacher_drive_decay_updates: int = 4000
    learning_rate_end: float | None = None
    learning_rate_decay_updates: int = 0


@configclass
class ParallelogramRoverVisionStudentPreviewDistillationRunnerCfg(
    ParallelogramRoverVisionStudentTemporalDistillationRunnerCfg
):
    experiment_name = "parallelogram_rover_vision_student_preview"
    run_name = "preview_student"
    # 旧配置每轮只有约3次Adam更新（48个小批 // gradient_length 15），student连平地轮速都欠拟合。
    # 每4个时间步更新一次、4个epoch，每轮24次更新；单次更新样本变少、梯度更噪，学习率减半。
    # 75%→10%的回合由teacher驱动，前期覆盖难状态、后期仍保留少量参考轨迹；
    # 验收只用student，混合训练的reward不能冒充student独立成绩。
    algorithm = RslRlDaggerDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=4,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
        teacher_drive_start=0.75,
        teacher_drive_end=0.10,
        teacher_drive_decay_updates=4000,
        learning_rate_end=1.0e-4,
        learning_rate_decay_updates=8000,
    )


# Preview student 的 runner 通过 eval(class_name) 查找算法类；导入配置即完成注册。
register_dagger_distillation()
