"""两轮运动的真实几何、成功课程、观测前缀及评测回归；不启动 GPU。

@author pony
@date 2026-09-16
@version v0.14.0
@last_modified 2026-09-17
@changelog
    - v0.14.0 (2026-09-17): 回归 wrapper构造reset失败时必须释放底层环境。
    - v0.13.0 (2026-09-17): 带速起立独立评测身份、命令边界及逐回合实际初态快照。
    - v0.12.0 (2026-09-17): 覆盖速度迭代单变量诊断的原子参数校验与训练默认不变性。
    - v0.11.0 (2026-09-17): 验证逐刚体COM姿态与惯量实测、坐标合同及损坏快照拒绝。
    - v0.10.0 (2026-09-17): 隔离腿限速诊断不得改变默认配置或冒充正式验收。
    - v0.9.0 (2026-09-17): 覆盖真实评测质量/材料及初态快照、形状顺序与非有限参数拒绝。
    - v0.8.0 (2026-09-17): 将零前进回归改为非负质量分数，防止终止规避负奖励。
    - v0.7.0 (2026-09-17): 覆盖固定俯仰合同、零前进稠密误差和扩大后的漂移上限。
    - v0.6.0 (2026-09-17): 验证累计位移失败不可擦除、命令边界计时及缺失锚点拒绝。
    - v0.5.0 (2026-09-17): 覆盖静止 XY 锚点、运动转停车重基准和局部 reset 隔离。
    - v0.4.0 (2026-09-17): 覆盖静止指令采样、收紧课程门和专用跟踪尺度。
    - v0.3.0 (2026-09-17): 覆盖静止命令真实XY漂移门、阶段首步和reset隔离。
    - v0.2.0 (2026-09-16): 核实两种 motion 任务注册，避免仅有配置但无法启动。
    - v0.1.0 (2026-09-16): 覆盖 90 度/世界 yaw、退化、支撑与连续成功门。
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def motion(monkeypatch):
    for name in ("isaaclab", "isaaclab.managers", "isaaclab.utils", "motion_test_package"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    class CommandTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device
            self.metrics = {}

        def reset(self, env_ids):
            self._resample_command(env_ids)
            return {}

    sys.modules["isaaclab.managers"].CommandTerm = CommandTerm
    sys.modules["isaaclab.managers"].CommandTermCfg = object
    sys.modules["isaaclab.utils"].configclass = lambda cls: cls
    for name in ("support", "motion"):
        path = Path(__file__).parents[1] / f"robot_lab/tasks/manager_based/upright/mdp/{name}.py"
        spec = importlib.util.spec_from_file_location("motion_test_package." + name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return sys.modules["motion_test_package.motion"]


def quaternion(yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return torch.tensor([[cy * cp, -sy * sp, cy * sp, sy * cp]])


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, -1.2])
@pytest.mark.parametrize("pitch", [0.0, -math.pi / 2, -1.83])
def test_heading_and_motion_are_consistent_at_vertical_pitch(motion, yaw, pitch):
    q = quaternion(yaw, pitch)
    forward = torch.tensor([[math.cos(yaw), math.sin(yaw), 0.]])
    lateral = torch.tensor([[-math.sin(yaw), math.cos(yaw), 0.]])
    actual_forward, actual_lateral, valid = motion.ground_heading(q)
    torch.testing.assert_close(actual_forward, forward, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_lateral, lateral, atol=1e-6, rtol=1e-6)
    assert valid.all()
    speed, tilt, valid = motion.ground_motion(q, .3 * forward + .1 * lateral, torch.tensor([[0., 0., .8]]))
    torch.testing.assert_close(speed, torch.tensor([[.3, .1, .8]]))
    assert tilt.item() == 0 and valid.all()


@pytest.mark.parametrize("q", [[[math.sqrt(.5), math.sqrt(.5), 0., 0.]], [[0., 0., 0., 0.]], [[float("nan"), 0., 0., 0.]]])
def test_roll_degenerate_or_invalid_quaternion_has_no_fake_heading(motion, q):
    forward, lateral, valid = motion.ground_heading(torch.tensor(q))
    assert not valid.any()
    assert torch.isfinite(forward).all() and torch.isfinite(lateral).all()
    assert forward.count_nonzero() == 0 and lateral.count_nonzero() == 0


def environment(num_envs=2):
    class Scene(dict):
        pass

    data = SimpleNamespace(
        root_quat_w=quaternion(0., -math.pi / 2).repeat(num_envs, 1),
        root_lin_vel_w=torch.tensor([[.3, 0., 0.]]).repeat(num_envs, 1),
        root_ang_vel_w=torch.tensor([[0., 0., .8]]).repeat(num_envs, 1),
        projected_gravity_b=torch.tensor([[-1., 0., 0.]]).repeat(num_envs, 1),
        root_pos_w=torch.tensor([[0., 0., .46]]).repeat(num_envs, 1),
        joint_pos=torch.zeros(num_envs, 20), joint_vel=torch.zeros(num_envs, 20),
    )
    names = ["rr_wheel", "base_link", "fl_wheel", "rl_qudong", "fr_wheel", "rl_wheel"]
    forces = torch.zeros(num_envs, len(names), 3)
    forces[:, [0, 5], 2] = 50.
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(num_envs, 3)
    scene.sensors = {"contact_forces": SimpleNamespace(body_names=names, data=SimpleNamespace(net_forces_w=forces))}
    commands = {"pitch_command": torch.full((num_envs, 1), math.pi / 2),
                "motion_command": torch.tensor([[.3, .8]]).repeat(num_envs, 1)}
    return SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=commands.__getitem__),
                           commands=commands, num_envs=num_envs, device="cpu", step_dt=.1,
                           common_step_counter=1, episode_length_buf=torch.ones(num_envs, dtype=torch.long))


def test_world_yaw_is_allowed_and_horizontal_rotation_is_not(motion):
    env = environment()
    assert motion.motion_state(env)["stable"].all()
    assert motion.motion_balance_hold(env).tolist() == [1., 1.]
    env.scene["robot"].data.root_ang_vel_w[0, 0] = 1.
    assert motion.motion_balance_hold(env).tolist() == [0., 1.]


@pytest.mark.parametrize("failure", ["front", "body", "one_rear", "air", "velocity", "yaw", "lateral", "nonfinite"])
def test_motion_gate_rejects_non_two_wheel_or_tracking_failures(motion, failure):
    env = environment()
    forces = env.scene.sensors["contact_forces"].data.net_forces_w
    data = env.scene["robot"].data
    if failure in ("front", "body"):
        forces[:, 2 if failure == "front" else 1, 2] = 5.
    elif failure == "one_rear":
        forces[:, 0] = 0.
    elif failure == "air":
        forces.zero_()
    elif failure == "velocity":
        data.root_lin_vel_w[:, 0] = 0.
    elif failure == "lateral":
        data.root_lin_vel_w[:, 1] = .5
    elif failure == "yaw":
        data.root_ang_vel_w[:, 2] = 0.
    else:
        data.joint_pos[:, 0] = float("nan")
    assert not motion.motion_state(env)["stable"].any()


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_command_ranges_and_zero_lateral(motion, stage):
    commands = motion.sample_motion_commands(4096, stage)
    for column, (low, high) in enumerate(motion.MOTION_RANGES[stage]):
        assert (commands[:, column] >= low).all() and (commands[:, column] <= high).all()
    env = environment()
    cfg = SimpleNamespace(initial_stage=0, fixed_stage=stage, fixed_command=None, stationary_fraction=0., success_time_s=.3, success_fraction=1.)
    term = motion.MotionCommand(cfg, env)
    term.reset()
    assert term.command.shape == (2, 2)
    assert term.planar_command[:, 1].count_nonzero() == 0


@pytest.mark.parametrize("stage", [1, 2])
def test_motion_stages_mix_exact_stationary_commands_without_changing_ranges(motion, stage):
    torch.manual_seed(7)
    commands = motion.sample_motion_commands(10000, stage, stationary_fraction=.2)
    stationary = motion.stationary_command_mask(commands)
    assert stationary.float().mean().item() == pytest.approx(.2, abs=.02)
    moving = commands[~stationary]
    for column, (low, high) in enumerate(motion.MOTION_RANGES[stage]):
        assert (moving[:, column] >= low).all() and (moving[:, column] <= high).all()


def test_stage_zero_and_default_sampling_keep_existing_contract(motion):
    assert not motion.sample_motion_commands(256, 0, stationary_fraction=.8).count_nonzero()
    torch.manual_seed(8)
    commands = motion.sample_motion_commands(10000, 2)
    assert not motion.stationary_command_mask(commands).any()
    with pytest.raises(ValueError):
        motion.sample_motion_commands(1, 1, stationary_fraction=1.1)


def test_curriculum_requires_contiguous_current_success_and_deduplicates(motion):
    course = motion.MotionCurriculum(2, .1, .3, 1.)
    yes = torch.ones(2, dtype=torch.bool)
    no = ~yes
    assert not course.update(yes, 1)
    assert not course.update(yes, 1)
    assert course.streak.tolist() == [1, 1]
    assert not course.update(yes, 2)
    assert not course.update(no, 3)
    assert not course.update(yes, 4)
    assert not course.update(yes, 5)
    assert course.update(yes, 6)
    assert course.stage == 1 and course.streak.count_nonzero() == 0
    assert not course.update(yes, 8)  # 漏采样不能补足连续成功时间。
    assert course.streak.tolist() == [1, 1]


def test_reset_breaks_only_reset_environment_streak(motion):
    course = motion.MotionCurriculum(2, .1, .3, 1.)
    yes = torch.ones(2, dtype=torch.bool)
    course.update(yes, 1)
    course.update(yes, 2)
    course.reset(torch.tensor([0]))
    assert not course.update(yes, 3)
    assert course.streak.tolist() == [1, 3]
    assert course.stage == 0


def test_no_training_iteration_progression_and_fixed_stage_never_advances(motion):
    course = motion.MotionCurriculum(2, .1, .1, 1.)
    assert not course.update(torch.zeros(2, dtype=torch.bool), 100000)
    assert course.stage == 0
    fixed = motion.MotionCurriculum(2, .1, .1, 1., fixed_stage=1)
    for step in range(10):
        assert not fixed.update(torch.ones(2, dtype=torch.bool), step)
    assert fixed.stage == 1


def test_command_curriculum_uses_real_state_and_skips_reset_samples(motion):
    env = environment()
    env.scene["robot"].data.root_lin_vel_w.zero_()
    env.scene["robot"].data.root_ang_vel_w.zero_()
    cfg = SimpleNamespace(initial_stage=0, fixed_stage=None, fixed_command=None, stationary_fraction=0., success_time_s=.3, success_fraction=1.)
    term = motion.MotionCommand(cfg, env)
    env.commands["motion_command"] = term.command
    term.reset()
    term._update_metrics()
    term._update_metrics()
    assert term.curriculum.streak.tolist() == [1, 1]
    env.episode_length_buf[0] = 0
    env.common_step_counter = 2
    term._update_metrics()
    assert term.curriculum.streak.tolist() == [0, 2]
    for step in (3, 4, 5):
        env.episode_length_buf.fill_(step)
        env.common_step_counter = step
        term._update_metrics()
    assert term.curriculum.stage == 1
    assert (term.command[:, 0] >= 0).all() and (term.command[:, 0] <= .3).all()


def test_fixed_command_is_reproducible_and_validated(motion):
    env = environment()
    cfg = SimpleNamespace(initial_stage=0, fixed_stage=2, fixed_command=(.3, .8), stationary_fraction=1., success_time_s=.3, success_fraction=1.)
    term = motion.MotionCommand(cfg, env)
    term.reset()
    torch.testing.assert_close(term.command, torch.tensor([[.3, .8], [.3, .8]]))
    cfg.fixed_command = (1., 0.)
    with pytest.raises(ValueError):
        motion.MotionCommand(cfg, env)


def test_resampling_only_changes_selected_commands_and_fixed_command_ignores_stationary_fraction(motion):
    env = environment()
    sampled_cfg = SimpleNamespace(
        initial_stage=1, fixed_stage=1, fixed_command=None, stationary_fraction=1., success_time_s=.3, success_fraction=1.,
    )
    sampled = motion.MotionCommand(sampled_cfg, env)
    sampled.command[:] = torch.tensor([[.2, .3], [.1, -.2]])
    sampled._resample_command(torch.tensor([0]))
    torch.testing.assert_close(sampled.command, torch.tensor([[0., 0.], [.1, -.2]]))

    fixed_cfg = SimpleNamespace(
        initial_stage=0, fixed_stage=2, fixed_command=(.3, .8), stationary_fraction=1., success_time_s=.3, success_fraction=1.,
    )
    fixed = motion.MotionCommand(fixed_cfg, env)
    fixed.reset(torch.tensor([0]))
    torch.testing.assert_close(fixed.command[0], torch.tensor([.3, .8]))


def test_stationary_training_gate_rejects_slow_drift_without_changing_generic_state(motion):
    env = environment()
    env.commands["motion_command"] = torch.zeros(2, 2)
    env.scene["robot"].data.root_lin_vel_w[:, 0] = .04
    env.scene["robot"].data.root_ang_vel_w.zero_()
    assert motion.motion_state(env)["stable"].all()
    assert not motion.motion_curriculum_state(env)["stable"].any()
    assert motion.motion_balance_hold(env).tolist() == [0., 0.]


def test_stationary_velocity_reward_uses_narrow_sigmas_and_is_finite_safe(motion):
    env = environment()
    env.commands["motion_command"] = torch.zeros(2, 2)
    env.scene["robot"].data.root_lin_vel_w.zero_()
    env.scene["robot"].data.root_ang_vel_w.zero_()
    stopped = motion.motion_velocity_tracking(env)
    env.scene["robot"].data.root_lin_vel_w[:, 0] = .04
    drifting = motion.motion_velocity_tracking(env)
    assert (stopped > drifting).all()
    env.commands["motion_command"] = torch.tensor([[.3, .8], [.3, .8]])
    env.scene["robot"].data.root_lin_vel_w[:, 0] = .3
    env.scene["robot"].data.root_ang_vel_w[:, 2] = .8
    torch.testing.assert_close(motion.motion_velocity_tracking(env), torch.full((2,), 2.))
    env.scene["robot"].data.root_lin_vel_w[0, 0] = float("nan")
    assert motion.motion_velocity_tracking(env)[0].item() == 0.


def test_zero_forward_velocity_quality_is_dense_and_does_not_change_forward_motion(motion):
    env = environment()
    env.commands["motion_command"] = torch.zeros(2, 2)
    env.scene["robot"].data.root_lin_vel_w[:] = torch.tensor([[.10, .05, 0.]])
    env.scene["robot"].data.root_ang_vel_w.zero_()
    torch.testing.assert_close(
        motion.motion_zero_forward_velocity_quality(env),
        torch.full((2,), 1.0 / 2.25),
    )
    env.commands["motion_command"][:] = torch.tensor([.3, 0.])
    assert not motion.motion_zero_forward_velocity_quality(env).any()


def test_zero_forward_velocity_quality_tracks_in_place_yaw_and_rejects_invalid_state(motion):
    env = environment()
    env.commands["motion_command"][:] = torch.tensor([0., .8])
    env.scene["robot"].data.root_lin_vel_w.zero_()
    env.scene["robot"].data.root_ang_vel_w[:, 2] = .6
    torch.testing.assert_close(
        motion.motion_zero_forward_velocity_quality(env),
        torch.full((2,), .5),
    )
    env.scene["robot"].data.joint_pos[0, 0] = float("nan")
    torch.testing.assert_close(
        motion.motion_zero_forward_velocity_quality(env),
        torch.tensor([0., .5]),
    )
    with pytest.raises(ValueError, match="有限正数"):
        motion.motion_zero_forward_velocity_quality(env, forward_scale=0.)


def anchored_motion(motion):
    env = environment()
    cfg = SimpleNamespace(initial_stage=0, fixed_stage=2, fixed_command=(0., 0.),
                          stationary_fraction=0., success_time_s=.3, success_fraction=1.)
    term = motion.MotionCommand(cfg, env)
    env.commands["motion_command"] = term.command
    term.reset()
    return env, term


def test_xy_drift_includes_first_step_and_zero_command_resampling_cannot_erase_it(motion):
    env, term = anchored_motion(motion)
    env.scene["robot"].data.root_pos_w[:, :2] += torch.tensor([.1, .2])
    actual = motion.motion_xy_drift(env)
    torch.testing.assert_close(actual, torch.full((2,), .05))
    actual.fill_(999)  # 返回值不持有内部缓存，调用者改动不能影响下次读取。
    term._resample_command(torch.arange(2))
    term._update_command()
    torch.testing.assert_close(motion.motion_xy_drift(env), torch.full((2,), .05))
    term.cfg.fixed_command = (0., .8)
    term._resample_command(torch.arange(2))
    torch.testing.assert_close(motion.motion_xy_drift(env), torch.full((2,), .05))


def test_xy_drift_rebases_before_stop_physics_and_preserves_deceleration_distance(motion):
    env, term = anchored_motion(motion)
    term.cfg.fixed_command = (.2, 0.)
    term._resample_command(torch.arange(2))
    env.scene["robot"].data.root_pos_w[:, 0] = 5.
    assert not motion.motion_xy_drift(env).any()
    term.cfg.fixed_command = (0., 0.)
    term._resample_command(torch.arange(2))
    assert not motion.motion_xy_drift(env).any()
    env.scene["robot"].data.root_pos_w[:, 0] += .1
    torch.testing.assert_close(motion.motion_xy_drift(env), torch.full((2,), .01))


def test_xy_drift_partial_reset_anchors_new_position_before_first_step(motion):
    env, term = anchored_motion(motion)
    env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([.3, .1])
    torch.testing.assert_close(motion.motion_xy_drift(env), torch.tensor([.09, .01]))
    env.scene["robot"].data.root_pos_w[0, 0] = 10.  # 模拟 Isaac reset event 先写根位置。
    term.reset(torch.tensor([0]))
    env.scene["robot"].data.root_pos_w[:, 0] += .1
    torch.testing.assert_close(motion.motion_xy_drift(env), torch.tensor([.01, .04]))


def test_xy_drift_invalid_position_is_penalized_without_rebasing_and_small_commands_move(motion):
    env, term = anchored_motion(motion)
    env.scene["robot"].data.root_pos_w[0, 0] = float("nan")
    assert motion.motion_xy_drift(env).tolist() == [9., 0.]
    term._update_command()
    env.scene["robot"].data.root_pos_w[0, 0] = .2
    assert motion.motion_xy_drift(env)[0] == pytest.approx(.04)
    term.cfg.fixed_command = (1e-7, 0.)
    term._resample_command(torch.arange(2))
    assert not motion.motion_xy_drift(env).any()


def test_xy_drift_keeps_gradient_beyond_one_meter_and_validates_cap(motion):
    env, _ = anchored_motion(motion)
    env.scene["robot"].data.root_pos_w[:, 0] = torch.tensor([2., 4.])
    assert motion.motion_xy_drift(env).tolist() == [4., 9.]
    with pytest.raises(ValueError, match="有限正数"):
        motion.motion_xy_drift(env, maximum_displacement_m=float("nan"))


def test_stationary_curriculum_rejects_accumulated_drift_even_after_return(motion):
    env, term = anchored_motion(motion)
    data = env.scene["robot"].data
    data.root_lin_vel_w.zero_()
    data.root_ang_vel_w.zero_()
    data.root_pos_w[:, 0] = torch.tensor([.29, .31])
    assert motion.motion_curriculum_state(env)["stable"].tolist() == [True, False]
    data.root_pos_w[:, 0] = 0.
    term._resample_command(torch.arange(2))
    assert motion.motion_curriculum_state(env)["stable"].tolist() == [True, False]
    term.reset(torch.tensor([1]))
    assert motion.motion_curriculum_state(env)["stable"].all()


def test_missing_or_nonfinite_anchor_never_grants_stationary_success(motion):
    env = environment()
    env.commands["motion_command"].zero_()
    env.scene["robot"].data.root_lin_vel_w.zero_()
    env.scene["robot"].data.root_ang_vel_w.zero_()
    assert not motion.motion_curriculum_state(env)["stable"].any()
    env, term = anchored_motion(motion)
    data = env.scene["robot"].data
    data.root_lin_vel_w.zero_()
    data.root_ang_vel_w.zero_()
    data.root_pos_w[0, 0] = float("nan")
    assert motion.motion_curriculum_state(env)["stable"].tolist() == [False, True]
    data.root_pos_w[0, 0] = 0.
    assert motion.motion_curriculum_state(env)["stable"].tolist() == [False, True]


def test_command_boundary_breaks_only_changed_streak_and_preserves_same_target(motion):
    env, term = anchored_motion(motion)
    term.curriculum.streak[:] = 12
    term._resample_command(torch.tensor([0]))
    assert term.curriculum.streak.tolist() == [12, 12]
    term.cfg.fixed_command = (.2, 0.)
    term._resample_command(torch.tensor([0]))
    assert term.curriculum.streak.tolist() == [0, 12]
    term.curriculum.streak[:] = 12
    term.command[0] = 0.  # 停止评测子类在更新钩子内直接改命令。
    env.scene["robot"].data.root_pos_w[0, 0] = 5.
    term._update_command()
    assert term.curriculum.streak.tolist() == [0, 12]
    env.scene["robot"].data.root_pos_w[0, 0] += .1
    assert motion.motion_xy_drift(env)[0] == pytest.approx(.01, abs=1e-7)


def test_pitch_change_cannot_reuse_old_target_success_or_prechange_physics(motion):
    env, term = anchored_motion(motion)
    env.scene["robot"].data.root_lin_vel_w.zero_()
    env.scene["robot"].data.root_ang_vel_w.zero_()
    term.curriculum.streak[:] = 12
    term.curriculum.last_step = env.common_step_counter - 1
    env.commands["pitch_command"][0, 0] += .01
    term._update_metrics()
    assert term.curriculum.streak.tolist() == [0, 13]


@pytest.mark.parametrize("limit", [0., -1., float("nan"), float("inf")])
def test_stationary_drift_limit_is_validated_at_construction(motion, limit):
    cfg = SimpleNamespace(stationary_fraction=.2, stationary_drift_limit_m=limit)
    with pytest.raises(ValueError, match="位移门"):
        motion.MotionCommand(cfg, environment())


def test_observation_inheritance_only_appends_commands_to_both_prefixes():
    path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/upright/config/motion_env_cfg.py"
    tree = ast.parse(path.read_text())
    observations = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionObservationsCfg")
    for name in ("PolicyCfg", "CriticCfg"):
        group = next(node for node in observations.body if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.unparse(group.bases[0]) == "UprightObservationsCfg." + name
        fields = [node.targets[0].id for node in group.body if isinstance(node, ast.Assign)]
        assert fields == ["motion_command"]


def test_motion_envs_fix_pitch_target_to_the_evaluation_contract():
    path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/upright/config/motion_env_cfg.py"
    tree = ast.parse(path.read_text())
    for class_name in ("ParallelogramRoverUprightMotionEnvCfg", "ParallelogramRoverUprightMotionGetupEnvCfg"):
        config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        post_init = next(node for node in config.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
        assignments = {
            ast.unparse(node.targets[0]): ast.unparse(node.value)
            for node in ast.walk(post_init)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }
        assert assignments["self.commands.pitch_command.pitch_range"] == "(math.pi / 2, math.pi / 2)"
        assert assignments["self.commands.pitch_command.resampling_time_range"] == "(1000000000.0, 1000000000.0)"


def test_motion_tasks_are_registered_with_distinct_balance_and_getup_entrypoints():
    path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/upright/__init__.py"
    tree = ast.parse(path.read_text())
    registrations = {
        next(keyword.value.value for keyword in node.keywords if keyword.arg == "id"): ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "gym.register"
    }
    for task_suffix, cfg_suffix in (("", ""), ("-Getup", "Getup")):
        call = registrations[f"RobotLab-Isaac-Upright-Motion{task_suffix}-Parallelogram-Rover-v0"]
        assert f"motion_env_cfg:ParallelogramRoverUprightMotion{cfg_suffix}EnvCfg" in call
        assert f"rsl_rl_motion_cfg:ParallelogramRoverUprightMotion{cfg_suffix}PPORunnerCfg" in call


@pytest.fixture
def evaluation(monkeypatch):
    scripts = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("motion_evaluation_test", scripts / "evaluate_parallelogram_motion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dynamics_environment():
    masses = torch.tensor([[8., 1.], [9., 1.]])
    materials = torch.tensor([[[.8, .6, .0], [.7, .5, .1], [.9, .7, .2]],
                              [[.6, .4, .1], [.9, .7, .0], [.8, .5, .1]]])
    inertias = torch.diag(torch.tensor([.1, .2, .25])).reshape(1, 1, 9).repeat(2, 2, 1)
    coms = torch.tensor([.01, .02, .03, 0., 0., .6, .8]).repeat(2, 2, 1)
    view = SimpleNamespace(get_masses=lambda: masses, get_material_properties=lambda: materials,
                           get_inertias=lambda: inertias, get_coms=lambda: coms)
    data = SimpleNamespace(root_state_w=torch.tensor([[1., 2., .45, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).repeat(2, 1),
                           joint_pos=torch.zeros(2, 3), joint_vel=torch.zeros(2, 3))
    robot = SimpleNamespace(body_names=["base_link", "rear_wheel"], root_physx_view=view,
                            joint_names=["leg", "follower", "wheel"], data=data)
    class Scene(dict):
        pass
    scene = Scene(robot=robot)
    scene.env_origins = torch.tensor([[1., 2., 0.], [3., 4., 0.]])
    return SimpleNamespace(num_envs=2, scene=scene), masses, materials


def test_dynamics_snapshot_records_actual_per_environment_parameters_without_guessing_shape_bodies(evaluation):
    env, masses, materials = dynamics_environment()
    snapshot = evaluation.capture_dynamics_parameters(env)
    assert snapshot["body_names"] == ["base_link", "rear_wheel"]
    assert snapshot["material_shape_order"] == "physx_articulation_shape_index_not_body_index"
    assert snapshot["body_mass_kg"] == masses.tolist()
    assert snapshot["shape_material_properties"] == materials.tolist()
    assert snapshot["material_columns"] == ["static_friction", "dynamic_friction", "restitution"]
    masses.zero_()
    materials.zero_()
    assert snapshot["body_mass_kg"][0][0] == 8.0
    assert snapshot["shape_material_properties"][0][0][0] == pytest.approx(.8)


def test_dynamics_snapshot_captures_replayable_initial_state_without_claiming_cross_engine_identity(evaluation):
    env, _, _ = dynamics_environment()
    snapshot = evaluation.capture_dynamics_parameters(env)
    initial = snapshot["initial_kinematic_state"]
    assert initial["root_state_layout"] == "world_position_xyz_quaternion_wxyz_world_linear_xyz_world_angular_xyz"
    assert initial["joint_names"] == ["leg", "follower", "wheel"]
    assert initial["env_origins_w"] == [[1., 2., 0.], [3., 4., 0.]]
    assert initial["root_state_w"] == env.scene["robot"].data.root_state_w.tolist()
    assert initial["joint_pos"] == [[0., 0., 0.], [0., 0., 0.]]
    assert not initial["contains_contact_solver_state"]
    env.scene["robot"].data.root_state_w.zero_()
    assert initial["root_state_w"][0][3] == 1.0


def test_dynamics_snapshot_keeps_com_frame_and_xyzw_without_mutating_backend(evaluation):
    env, _, _ = dynamics_environment()
    view = env.scene["robot"].root_physx_view
    before = view.get_coms().clone()
    snapshot = evaluation.capture_dynamics_parameters(env)
    assert snapshot["body_inertial_contract"] == "physx_link_axes_com_inertia_v2"
    assert snapshot["body_com_pose_xyzw"] == before.tolist()
    assert snapshot["body_inertia_link_kg_m2"] == view.get_inertias().tolist()
    assert torch.equal(view.get_coms(), before)
    view.get_coms().zero_()
    view.get_inertias().zero_()
    assert snapshot["body_com_pose_xyzw"][0][0][-1] == pytest.approx(.8)
    assert snapshot["body_inertia_link_kg_m2"][0][0][0] == pytest.approx(.1)


@pytest.mark.parametrize("fault", ["nan_com", "bad_quaternion", "com_shape", "nan_inertia", "asymmetric", "negative"])
def test_dynamics_snapshot_rejects_invalid_inertial_parameters(evaluation, fault):
    env, _, _ = dynamics_environment()
    view = env.scene["robot"].root_physx_view
    if fault == "nan_com":
        view.get_coms()[0, 0, 0] = float("nan")
    elif fault == "bad_quaternion":
        view.get_coms()[0, 0, 3:] = 0.
    elif fault == "com_shape":
        view.get_coms = lambda: torch.zeros(2, 2, 6)
    elif fault == "nan_inertia":
        view.get_inertias()[0, 0, 0] = float("nan")
    elif fault == "asymmetric":
        view.get_inertias()[0, 0, 1] = .05
    else:
        view.get_inertias()[0, 0, 0] = -.1
    with pytest.raises(ValueError):
        evaluation.capture_dynamics_parameters(env)


@pytest.mark.parametrize("fault", ["root_nan", "root_shape", "joint_shape", "origin_nan", "duplicate_joints"])
def test_dynamics_snapshot_rejects_incomplete_initial_state(evaluation, fault):
    env, _, _ = dynamics_environment()
    data = env.scene["robot"].data
    if fault == "root_nan":
        data.root_state_w[0, 0] = float("nan")
    elif fault == "root_shape":
        data.root_state_w = torch.zeros(2, 12)
    elif fault == "joint_shape":
        data.joint_vel = torch.zeros(2, 2)
    elif fault == "origin_nan":
        env.scene.env_origins[1, 0] = float("inf")
    else:
        env.scene["robot"].joint_names = ["leg", "leg", "wheel"]
    with pytest.raises(ValueError):
        evaluation.capture_dynamics_parameters(env)


@pytest.mark.parametrize("fault", ["nan_mass", "negative_mass", "zero_mass", "nan_material", "negative_material",
                                  "bad_restitution", "mass_shape", "material_shape", "duplicate_names"])
def test_dynamics_snapshot_rejects_invalid_or_misaligned_parameters(evaluation, fault):
    env, masses, materials = dynamics_environment()
    if fault == "nan_mass":
        masses[0, 0] = float("nan")
    elif fault == "negative_mass":
        masses[0, 0] = -1.
    elif fault == "zero_mass":
        masses[0].zero_()
    elif fault == "nan_material":
        materials[0, 0, 1] = float("inf")
    elif fault == "negative_material":
        materials[0, 0, 0] = -1.
    elif fault == "bad_restitution":
        materials[0, 0, 2] = 1.1
    elif fault == "mass_shape":
        env.scene["robot"].root_physx_view.get_masses = lambda: torch.ones(2, 3)
    elif fault == "material_shape":
        env.scene["robot"].root_physx_view.get_material_properties = lambda: torch.ones(1, 3, 3)
    else:
        env.scene["robot"].body_names = ["base_link", "base_link"]
    with pytest.raises(ValueError):
        evaluation.capture_dynamics_parameters(env)


def test_evaluation_records_motion_and_requires_terminal_contiguous_tracking(evaluation):
    tracker = evaluation.MotionEvaluation(1, 1, .1, .3, .2)
    for step in range(4):
        tracker.update_motion(
            [True], [False], [step == 3], [True], [[.2, 0., .5]], [[.2, .5]], [True],
            world_positions=[[(step + 1) * .02, 0.]], previous_world_positions=[[step * .02, 0.]],
        )
    record = tracker.records[0]
    assert record["success"]
    assert record["ground_forward_distance_m"] == pytest.approx(.08)
    assert record["yaw_integral_rad"] == pytest.approx(.2)
    assert record["velocity_rmse_m_s"] == 0
    assert record["two_rear_wheel_support_fraction"] == 1


def test_evaluation_rejects_stationary_pose_under_motion_command(evaluation):
    tracker = evaluation.MotionEvaluation(1, 1, .1, .3, .2)
    for step in range(4):
        tracker.update_motion(
            [True], [False], [step == 3], [True], [[0., 0., 0.]], [[.2, .5]], [True],
            world_positions=[[0., 0.]], previous_world_positions=[[0., 0.]],
        )
    assert not tracker.records[0]["success"]
    assert not tracker.records[0]["progress_gate_passed"]


def test_evaluation_quotas_reset_statistics_and_reject_nonfinite(evaluation):
    tracker = evaluation.MotionEvaluation(1, 2, .1, .2, .2)
    for episode in range(2):
        for step in range(3):
            velocity = [.2, 0., 0.] if episode == 0 else [float("nan"), 0., 0.]
            tracker.update_motion(
                [True], [False], [step == 2], [True], [velocity], [[.2, 0.]], [True],
                world_positions=[[episode * 10. + step * .02, 0.]],
                previous_world_positions=[[episode * 10. + max(0, step - 1) * .02, 0.]],
            )
    assert tracker.complete
    assert tracker.records[0]["success"] and not tracker.records[1]["success"]
    assert tracker.records[1]["velocity_rmse_m_s"] is None


def test_stationary_command_rejects_displacement_and_resets_reference(evaluation):
    tracker = evaluation.MotionEvaluation(1, 2, .1, .2, .2, stationary_drift_limit_m=.30)
    episodes = ((10.0, (.20, .40, .40)), (100.0, (.05, .10, .10)))
    for origin, offsets in episodes:
        previous = origin
        for step, offset in enumerate(offsets):
            current = origin + offset
            tracker.update_motion(
                [True], [False], [step == 2], [True], [[0., 0., 0.]], [[0., 0.]], [True],
                world_positions=[[current, 0.]], previous_world_positions=[[previous, 0.]],
            )
            previous = current
    assert not tracker.records[0]["success"]
    assert tracker.records[0]["maximum_horizontal_displacement_m"] == pytest.approx(.4)
    assert tracker.records[1]["success"]
    assert tracker.records[1]["reference_xy_m"] == [100., 0.]
    assert tracker.records[1]["maximum_horizontal_displacement_m"] == pytest.approx(.1)


def test_custom_hold_is_smoke_not_standard_acceptance(evaluation):
    assert evaluation.evaluation_scope("balance", "stationary", None) == "standard_evaluation"
    assert evaluation.evaluation_scope("balance", "stationary", 1.0) == "custom_duration_smoke"
    assert evaluation.evaluation_scope("balance", "stationary", 30.0) == "standard_evaluation"
    assert evaluation.evaluation_scope("getup", "stationary", 5.0) == "custom_duration_smoke"
    assert evaluation.evaluation_scope("getup", "stop", 5.0) == "standard_evaluation"


def test_moving_getup_has_separate_scope_and_preserves_smoke_and_diagnostic_gates(evaluation):
    assert not evaluation.validate_getup_initial_speed("balance", (0., 0.))
    assert evaluation.validate_getup_initial_speed("getup", (-.3, .3))
    assert evaluation.evaluation_scope("getup", "forward", None, moving_getup=True) == "synthetic_moving_getup_evaluation"
    assert evaluation.evaluation_scope("getup", "forward", .2, moving_getup=True) == "custom_duration_smoke"
    assert evaluation.evaluation_scope("getup", "forward", None, moving_getup=True,
                                       diagnostic_drive=True) == "diagnostic_dynamics_override"


@pytest.mark.parametrize("phase,speeds", [("balance", (.1, .1)), ("getup", (.2, -.2)),
                                        ("getup", (float("nan"), .1)), ("getup", (-.6, 0.)),
                                        ("getup", (0., float("inf"))), ("getup", (.1,))])
def test_invalid_initial_speed_protocol_rejected(evaluation, phase, speeds):
    with pytest.raises(ValueError):
        evaluation.validate_getup_initial_speed(phase, speeds)


def test_actual_episode_initial_snapshots_are_independent_across_resets(evaluation):
    env, _, _ = dynamics_environment()
    commands = torch.tensor([[.25, 0.], [0., 0.]])
    env.command_manager = SimpleNamespace(get_command=lambda name: commands)
    data = env.scene["robot"].data
    data.root_state_w[0, 7] = .2
    initial = evaluation.capture_episode_initial_states(env)
    old = initial[0]
    assert old["root_state_w"][7] == pytest.approx(.2)
    assert old["command_forward_yaw"] == [.25, 0.]
    assert old["sampling"] == "after_reset_before_first_policy_action"
    data.root_state_w[0, 7] = -.1
    commands[0] = 0.
    initial.update(evaluation.capture_episode_initial_states(env, [0]))
    assert initial[0]["root_state_w"][7] == pytest.approx(-.1)
    assert initial[0]["command_forward_yaw"] == [0., 0.]
    assert old["root_state_w"][7] == pytest.approx(.2)
    assert old["command_forward_yaw"] == [.25, 0.]
    assert initial[1]["root_state_w"][7] == 0.
    data.joint_vel[0, 1] = float("nan")
    with pytest.raises(ValueError):
        evaluation.capture_episode_initial_states(env, [0])


def test_wrapper_reset_failure_closes_raw_environment_before_app_shutdown(evaluation, monkeypatch):
    raw = SimpleNamespace(closed=False)
    raw.close = lambda: setattr(raw, "closed", True)
    gym = ModuleType("gymnasium")
    gym.make = lambda task, cfg: raw
    wrapper_module = ModuleType("isaaclab_rl.rsl_rl")

    def failing_wrapper(env, clip_actions):
        assert env is raw
        raise ValueError("reset failure")

    wrapper_module.RslRlVecEnvWrapper = failing_wrapper
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "isaaclab_rl", ModuleType("isaaclab_rl"))
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", wrapper_module)
    with pytest.raises(ValueError, match="reset failure"):
        evaluation.create_motion_environment("task", SimpleNamespace(), None)
    assert raw.closed


def test_drive_override_is_explicit_leg_only_and_not_acceptance(evaluation):
    actuators = {name: SimpleNamespace(velocity_limit_sim=value)
                 for name, value in (("suspension", 5.), ("steering", 5.), ("mimic_followers", 10.), ("wheels", 31.4))}
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators=actuators)))
    assert evaluation.apply_diagnostic_drive_override(cfg, None) == {}
    result = evaluation.apply_diagnostic_drive_override(cfg, 1000.)
    assert result["original_leg_velocity_limit_sim_rad_s"] == {"suspension": 5., "steering": 5.}
    assert result["leg_velocity_limit_sim_rad_s"] == 1000.
    assert actuators["suspension"].velocity_limit_sim == 1000.
    assert actuators["steering"].velocity_limit_sim == 1000.
    assert actuators["mimic_followers"].velocity_limit_sim == 10.
    assert actuators["wheels"].velocity_limit_sim == 31.4
    assert evaluation.evaluation_scope("balance", "stationary", None, diagnostic_drive=True) == "diagnostic_dynamics_override"
    for bad in (0., -1., float("inf"), float("nan")):
        with pytest.raises(ValueError):
            evaluation.apply_diagnostic_drive_override(cfg, bad)
    assert actuators["suspension"].velocity_limit_sim == 1000.


def test_solver_iteration_diagnostic_is_atomic_and_retains_other_physics(evaluation):
    props = SimpleNamespace(solver_position_iteration_count=8, solver_velocity_iteration_count=2)
    actuators = {name: SimpleNamespace(velocity_limit_sim=5.) for name in ("suspension", "steering")}
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(
        actuators=actuators, spawn=SimpleNamespace(articulation_props=props))))
    for bad in (-1, 256, 1.5, True):
        with pytest.raises(ValueError):
            evaluation.apply_diagnostic_drive_override(cfg, 1000., solver_velocity_iterations=bad)
        assert actuators["suspension"].velocity_limit_sim == 5.
        assert props.solver_velocity_iteration_count == 2
    result = evaluation.apply_diagnostic_drive_override(cfg, None, solver_velocity_iterations=0)
    assert props.solver_velocity_iteration_count == 0
    assert props.solver_position_iteration_count == 8
    assert result["original_solver_velocity_iterations"] == 2
    assert result["solver_velocity_iterations"] == 0
    assert actuators["suspension"].velocity_limit_sim == 5.


@pytest.mark.parametrize("first_position", [[.31, 0.], [0., .31], [float("nan"), 0.]])
def test_first_step_excursion_or_nan_cannot_be_erased_by_returning(evaluation, first_position):
    tracker = evaluation.MotionEvaluation(1, 1, .1, .2, .2)
    for step in range(3):
        tracker.update_motion(
            [True], [False], [step == 2], [True], [[0., 0., 0.]], [(0., 0.)], [True],
            world_positions=[first_position if step == 0 else [0., 0.]],
            previous_world_positions=[[0., 0.]],
        )
    assert not tracker.records[0]["success"]
    assert not tracker.records[0]["stationary_drift_gate_passed"]


def test_motion_scenarios_and_separate_stop_protocol_exist(evaluation):
    assert evaluation.SCENARIOS["backward"][0] < 0
    assert evaluation.SCENARIOS["turn-left"][1] > 0
    assert evaluation.SCENARIOS["turn-right"][1] < 0
    assert evaluation.SCENARIOS["stop"][0] > 0
    assert hasattr(evaluation, "StopEvaluation")
