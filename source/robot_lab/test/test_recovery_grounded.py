"""用倒姿、高度投机与错误接触归属验证恢复奖励，不启动仿真。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 验证有符号进展、双项局部 reset、时钟间断和非有限物理状态隔离。
    - v0.1.0 (2026-09-17): 覆盖连续翻正信号、四轮真实承重和准备期隔离。
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/recovery/mdp/grounded.py"


@pytest.fixture
def grounded():
    spec = importlib.util.spec_from_file_location("grounded_recovery_rewards", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(rolls):
    class Scene(dict):
        pass

    n = len(rolls)
    gravity = torch.tensor([[0., math.sin(r), -math.cos(r)] for r in rolls])
    root = torch.zeros(n, 13)
    root[:, 2] = .332
    root[:, 3] = 1.0
    data = SimpleNamespace(
        projected_gravity_b=gravity,
        root_pos_w=root[:, :3],
        root_state_w=root,
        root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3),
        joint_pos=torch.zeros(n, 20), default_joint_pos=torch.zeros(n, 20),
        joint_vel=torch.zeros(n, 20),
    )
    # 顺序刻意不等于articulation顺序；必须按传感器自己的名字解析。
    names = ["rr_wheel", "fl_qudong", "base_link", "rl_wheel", "fr_wheel", "fl_wheel"]
    forces = torch.zeros(n, len(names), 3)
    forces[:, [0, 3, 4, 5], 2] = 26.0
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(n, 3)
    scene.sensors = {"contact_forces": SimpleNamespace(body_names=names, data=SimpleNamespace(net_forces_w=forces))}
    return SimpleNamespace(scene=scene, num_envs=n, common_step_counter=0,
                           reset_buf=torch.zeros(n, dtype=torch.bool),
                           _recovery_control_active=torch.ones(n, dtype=torch.bool),
                           _recovery_joint_groups={"active_leg": torch.arange(8)})


def test_orientation_has_signal_between_supine_side_and_upright(grounded):
    env = environment([math.pi, 2.5, math.pi / 2, 0.8, 0.0])
    reward = grounded.recovery_orientation_score(env)
    assert torch.all(reward[1:] > reward[:-1])
    assert reward[0] == 0 and reward[-1] == 1


def test_side_lying_at_target_height_cannot_collect_standing_rewards(grounded):
    env = environment([math.pi / 2, math.pi, 0.0])
    for func in (grounded.recovery_oriented_height, grounded.recovery_oriented_joints,
                 grounded.recovery_four_wheel_support, grounded.recovery_stable_hold):
        reward = func(env)
        torch.testing.assert_close(reward[:2], torch.zeros(2), rtol=0, atol=1e-12)
        assert reward[2] > 0


def test_contact_support_requires_all_four_wheels_and_no_nonwheel(grounded):
    env = environment([0.0] * 4)
    forces = env.scene.sensors["contact_forces"].data.net_forces_w
    forces[0] = 0  # 空中水平姿态
    forces[1, 5] = 0  # 少一个支撑轮
    forces[2, 1, 2] = 2  # 连杆承重，即便四轮也接触
    assert grounded.recovery_four_wheel_support(env).tolist() == [0, 0, 0, 1]


def test_hold_rejects_moving_platform_or_active_leg_but_not_free_wheel_spin(grounded):
    env = environment([0.0] * 4)
    data = env.scene["robot"].data
    data.root_lin_vel_w[0, 0] = 0.3
    data.root_ang_vel_w[1, 1] = 0.8
    data.joint_vel[2, 3] = 0.6
    data.joint_vel[3, 18] = 10
    assert grounded.recovery_stable_hold(env).tolist() == [0, 0, 0, 1]


def test_rewards_invariant_to_world_origin_and_no_preparation_or_nan_reward(grounded):
    env = environment([0.0] * 3)
    funcs = (grounded.recovery_orientation_score, grounded.recovery_oriented_height,
             grounded.recovery_oriented_joints, grounded.recovery_four_wheel_support, grounded.recovery_stable_hold)
    baseline = [f(env) for f in funcs]
    env.scene.env_origins[:, 2] += 20
    env.scene["robot"].data.root_pos_w[:, 2] += 20
    for f, expected in zip(funcs, baseline):
        torch.testing.assert_close(f(env), expected)
    env._recovery_control_active[0] = False
    env.scene["robot"].data.projected_gravity_b[1] = float("nan")
    for f in funcs:
        result = f(env)
        assert torch.isfinite(result).all()
        assert result[:2].tolist() == [0, 0]


def test_progress_is_signed_deduplicated_and_resettable(grounded):
    env = environment([math.pi / 2])

    # 首样本没有前态，不能凭初始姿态得到进展奖励。
    env.common_step_counter = 0
    assert grounded.recovery_orientation_progress(env).item() == 0.0

    env.common_step_counter = 1
    env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0.0, 0.0, -1.0]])
    positive = grounded.recovery_orientation_progress(env)
    assert positive.item() > 0.0

    # 重复读取返回同一变化量，且调用者不能通过修改返回值污染内部缓存。
    torch.testing.assert_close(grounded.recovery_orientation_progress(env), positive)
    positive.fill_(99)
    assert grounded.recovery_orientation_progress(env).item() == pytest.approx(8.0)

    env.common_step_counter = 2
    env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0.0, 1.0, 0.0]])
    assert grounded.recovery_orientation_progress(env).item() < 0.0

    grounded.reset_grounded_progress(env, torch.tensor([0]))
    env.common_step_counter = 3
    # 下一 episode 首样本不能继承上一轮差分。
    assert grounded.recovery_orientation_progress(env).item() == 0.0


def test_progress_nonfinite_state_is_finite_and_does_not_poison_history(grounded):
    env = environment([0.0, 0.0])
    env.common_step_counter = 0
    grounded.recovery_orientation_progress(env)
    env.common_step_counter = 1
    env.scene["robot"].data.root_state_w[0, 0] = float("nan")
    result = grounded.recovery_orientation_progress(env)
    assert torch.isfinite(result).all()
    assert result.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("reward_name", ["recovery_orientation_progress", "recovery_height_progress"])
@pytest.mark.parametrize("interruption", ["inactive", "terminal", "missed_step", "rewind", "nonfinite"])
def test_progress_breaks_history_at_episode_or_sampling_boundaries(grounded, reward_name, interruption):
    env = environment([math.pi / 2, math.pi / 2])
    reward = getattr(grounded, reward_name)
    reward(env)
    env.common_step_counter = 1
    env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0., .6, -.8]])
    if interruption == "inactive":
        env._recovery_control_active[0] = False
    elif interruption == "terminal":
        env.reset_buf[0] = True
    elif interruption == "nonfinite":
        env.scene.sensors["contact_forces"].data.net_forces_w[0, 0, 0] = float("nan")
    else:
        env.common_step_counter = 3 if interruption == "missed_step" else -1
    current = reward(env)
    assert torch.isfinite(current).all()
    if interruption in ("missed_step", "rewind"):
        assert not current.any()
    elif interruption != "terminal":
        assert current[0] == 0
    else:
        assert current[0] > 0  # 终止前最后一次真实进展仍可计入。
    env.common_step_counter += 1
    env._recovery_control_active.fill_(True)
    env.reset_buf.zero_()
    env.scene.sensors["contact_forces"].data.net_forces_w[0, 0, 0] = 0.
    env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0., 0., -1.]])
    next_reward = reward(env)
    if interruption in ("inactive", "terminal", "nonfinite"):
        assert next_reward[0] == 0 and next_reward[1] > 0
    else:
        assert (next_reward > 0).all()


def test_height_progress_is_relative_oriented_bounded_and_signed(grounded):
    env = environment([0., math.pi / 2])
    data = env.scene["robot"].data
    env.scene.env_origins[:, 2] = 10.
    data.root_pos_w[:, 2] = 10.1
    grounded.recovery_height_progress(env)
    total = torch.zeros(2)
    for step, height in enumerate((.2, .332, .8, .1), 1):
        env.common_step_counter = step
        data.root_pos_w[:, 2] = 10. + height
        delta = grounded.recovery_height_progress(env)
        total += delta
        assert delta[1].item() == pytest.approx(0., abs=1e-10)
        if step == 3:
            # 世界高度减大原点存在 float32 消减误差，不能对势能差用精确相等。
            assert delta[0].item() == pytest.approx(0., abs=1e-5)
        if step == 4:
            assert delta[0] < 0
    torch.testing.assert_close(total, torch.zeros(2), atol=1e-5, rtol=0)
