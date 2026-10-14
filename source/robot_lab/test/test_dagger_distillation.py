"""DAgger 混合蒸馏的调度与回合级驱动者切换回归，不依赖仿真器。

@author pony
@date 2026-10-09
@version v1.0.0
@last_modified 2026-10-09
@changelog
    - v1.0.0 (2026-10-09): 覆盖线性调度、回合内驱动者不变、标签恒为teacher、学习率衰减与注册。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / (
    "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/agents/dagger_distillation.py"
)


@pytest.fixture(scope="module")
def dagger():
    # 直接按文件加载，避免导入 robot_lab.tasks 触发 Isaac Sim 依赖。
    spec = importlib.util.spec_from_file_location("dagger_distillation_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeStudentTeacher(torch.nn.Module):
    """student 输出恒0、teacher 输出恒1，便于区分驱动者与监督标签。"""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.student = torch.nn.Linear(2, 3)

    def act(self, obs):
        return torch.zeros(obs.shape[0], 3)

    def evaluate(self, obs):
        return torch.ones(obs.shape[0], 3)

    def update_normalization(self, obs):
        pass

    def reset(self, dones=None, hidden_states=None):
        pass


def make(dagger, **kwargs):
    alg = dagger.DaggerDistillation(FakeStudentTeacher(), learning_rate=5e-4, **kwargs)

    class Storage:
        def add_transitions(self, transition):
            self.last = (transition.actions.clone(), transition.privileged_actions.clone())

    alg.storage = Storage()
    return alg


def test_linear_schedule_clamps_and_supports_no_decay(dagger):
    assert dagger.linear_schedule(0.5, 0.0, 0, 4000) == 0.5
    assert dagger.linear_schedule(0.5, 0.0, 2000, 4000) == pytest.approx(0.25)
    assert dagger.linear_schedule(0.5, 0.0, 9000, 4000) == 0.0
    assert dagger.linear_schedule(0.5, 0.0, 9000, 0) == 0.5


def test_labels_are_always_teacher_and_driver_fixed_within_episode(dagger):
    torch.manual_seed(0)
    alg = make(dagger, teacher_drive_start=0.5)
    obs = torch.zeros(256, 2)
    no_done = torch.zeros(256)
    alg.act(obs)
    driven = alg.teacher_driven.clone()
    assert 0.3 < driven.float().mean().item() < 0.7
    for _ in range(5):
        actions = alg.act(obs)
        alg.process_env_step(obs, None, no_done, {})
        actions, labels = alg.storage.last
        assert torch.equal(labels, torch.ones_like(labels))
        assert torch.equal(actions[:, 0].bool(), driven)
        assert torch.equal(alg.teacher_driven, driven)


def test_reset_resamples_only_finished_envs(dagger):
    alg = make(dagger, teacher_drive_start=1.0, teacher_drive_end=0.0, teacher_drive_decay_updates=1)
    obs = torch.zeros(8, 2)
    alg.act(obs)
    assert alg.teacher_driven.all()
    alg.num_updates = 1  # 进度到头后驱动比例为0
    dones = torch.zeros(8)
    dones[:3] = 1
    alg.process_env_step(obs, None, dones, {})
    assert not alg.teacher_driven[:3].any()
    assert alg.teacher_driven[3:].all()


def test_learning_rate_decays_with_updates(dagger):
    alg = make(dagger, learning_rate_end=1e-4, learning_rate_decay_updates=10)
    alg.storage.generator = lambda: iter(())
    alg.storage.clear = lambda: None
    alg.policy.get_hidden_states = lambda: (None, None)
    alg.policy.detach_hidden_states = lambda dones=None: None
    alg.num_updates = 10
    # 空生成器会让原生 update 除零；这里只验证学习率在调用父类前已按进度设置。
    with pytest.raises(ZeroDivisionError):
        alg.update()
    assert alg.learning_rate == pytest.approx(1e-4)
    assert all(group["lr"] == pytest.approx(1e-4) for group in alg.optimizer.param_groups)


def test_invalid_schedule_rejected(dagger):
    with pytest.raises(ValueError):
        make(dagger, teacher_drive_start=0.2, teacher_drive_end=0.5)
    with pytest.raises(ValueError):
        make(dagger, learning_rate_end=1e-2)


def test_register_exposes_class_to_runner_eval(dagger):
    from rsl_rl.runners import distillation_runner

    dagger.register()
    assert distillation_runner.DaggerDistillation is dagger.DaggerDistillation


def test_real_update_changes_only_student_and_reports_actual_driver_fraction(dagger):
    from rsl_rl.modules import StudentTeacher
    from tensordict import TensorDict

    torch.manual_seed(142)
    obs = TensorDict({"policy": torch.randn(8, 3), "teacher": torch.randn(8, 4)}, [8])
    policy = StudentTeacher(obs, {"policy": ["policy"], "teacher": ["teacher"]}, 2,
                            student_hidden_dims=[8], teacher_hidden_dims=[8])
    alg = dagger.DaggerDistillation(policy, teacher_drive_start=0.75, teacher_drive_end=0.10,
                                    teacher_drive_decay_updates=4000, gradient_length=2,
                                    num_learning_epochs=2, learning_rate=5e-4,
                                    learning_rate_end=1e-4, learning_rate_decay_updates=8000)
    alg.num_updates = 4000
    alg.init_storage("distillation", 8, 4, obs, (2,))
    teacher_before = {k: v.clone() for k, v in policy.teacher.state_dict().items()}
    student_before = {k: v.clone() for k, v in policy.student.state_dict().items()}
    actual_samples = 0
    for _ in range(4):
        with torch.inference_mode():
            alg.act(obs)
            actual_samples += int(alg.teacher_driven.sum())
            alg.process_env_step(obs, torch.zeros(8), torch.zeros(8, dtype=torch.bool), {})
    result = alg.update()
    assert result["teacher_drive_probability"] == pytest.approx(0.10)
    assert result["teacher_driven_fraction"] == pytest.approx(actual_samples / 32)
    assert alg.learning_rate == pytest.approx(3e-4)
    assert alg.num_updates == 4001
    assert alg._total_samples == 0
    assert all(torch.equal(teacher_before[k], v) for k, v in policy.teacher.state_dict().items())
    assert all(p.grad is None for p in policy.teacher.parameters())
    assert any(not torch.equal(student_before[k], v) for k, v in policy.student.state_dict().items())
    assert torch.isfinite(torch.tensor(result["behavior"]))
