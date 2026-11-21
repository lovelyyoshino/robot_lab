"""SuperMimic 核心（四元数、状态增量、观测合同、奖励、回放、世界模型、learner、扫描学生）回归，不依赖仿真器。

@author pony
@date 2026-10-26
@version v0.4.0
@last_modified 2026-11-09
@changelog
    - v0.4.0 (2026-11-09): actor 动作饱和代价（tanh 前原始输出超出动作界）只作用在受限维、梯度把输出拉回界内。
    - (2026-11-03): 续训优化器保留动量，同时使用本次配置中的学习率。
    - v0.3.0 (2026-11-02): 参考约束梯度审计覆盖关闭和启用情况，不修改模型与参数梯度。
    - v0.2.0 (2026-10-26): 覆盖 ξ 上下文、aux 头、奖励偏差诊断与 actor 更新间隔。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import ast
import copy
import math
import sys
from pathlib import Path

import pytest
import torch

# 把 robot_lab/ 加到路径，以顶层包 supermimic 导入，避开 robot_lab/__init__ 对 Isaac Sim 的依赖。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

from supermimic import quat, rover_task, scan_student, state  # noqa: E402
from supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from supermimic.networks import RunningNorm, WorldModel  # noqa: E402
from supermimic.replay import ReplayBuffer  # noqa: E402


def random_quat(n):
    return quat.quat_normalize(torch.randn(n, 4))


def random_state(n, scale=1.0):
    return state.make_state(
        torch.randn(n, 3), random_quat(n), torch.randn(n, 3) * scale, torch.randn(n, 3) * scale,
        torch.randn(n, 12) * 0.3, torch.randn(n, 12) * scale,
    )


# ---------------- 四元数 ----------------
def test_axis_angle_round_trip_and_small_angle_gradient():
    rotvec = torch.randn(64, 3)
    rotvec = rotvec / rotvec.norm(dim=-1, keepdim=True) * torch.rand(64, 1) * 3.0
    assert torch.allclose(quat.quat_to_axis_angle(quat.axis_angle_to_quat(rotvec)), rotvec, atol=1e-5)
    zero = torch.zeros(1, 3, requires_grad=True)
    quat.axis_angle_to_quat(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()


def test_projected_gravity_matches_isaac_convention():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(quat.projected_gravity(identity), torch.tensor([[0.0, 0.0, -1.0]]))
    # 绕 +Y 转 90°（机头朝下）后，重力在机体系指向 +X。
    pitch = quat.axis_angle_to_quat(torch.tensor([[0.0, torch.pi / 2, 0.0]]))
    assert torch.allclose(quat.projected_gravity(pitch), torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


# ---------------- 状态增量 ----------------
def test_integrate_inverts_compute_delta():
    s0, s1 = random_state(128), random_state(128)
    s1[:, state.QUAT] = quat.quat_mul(s0[:, state.QUAT], quat.axis_angle_to_quat(torch.randn(128, 3) * 0.5))
    rebuilt = state.integrate(s0, state.compute_delta(s0, s1))
    assert torch.allclose(rebuilt, s1, atol=1e-5)
    assert torch.allclose(state.state_error(rebuilt, s1), torch.zeros(128, state.DELTA_DIM), atol=1e-5)


def test_features_and_delta_are_heading_and_translation_invariant():
    s0, s1 = random_state(32), random_state(32)
    yaw = quat.axis_angle_to_quat(torch.tensor([[0.0, 0.0, 1.3]])).expand(32, 4)
    shift = torch.tensor([5.0, -2.0, 0.0])

    def transform(s):
        out = s.clone()
        out[:, state.POS] = quat.quat_apply(yaw, s[:, state.POS]) + shift
        out[:, state.QUAT] = quat.quat_mul(yaw, s[:, state.QUAT])
        return out

    assert torch.allclose(state.features(transform(s0)), state.features(s0), atol=1e-5)
    assert torch.allclose(state.compute_delta(transform(s0), transform(s1)), state.compute_delta(s0, s1), atol=1e-4)


def test_wheel_positions_never_enter_state():
    s = random_state(4)
    assert torch.count_nonzero(s[:, state.JPOS][:, state.NUM_LEG_JOINTS:]) == 0


def test_isaac_state_reader_preserves_native_sensor_quaternion_bits():
    from types import SimpleNamespace

    path = Path(__file__).parents[1] / "robot_lab/supermimic/isaac_adapter.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "RoverStateReader")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "state")
    scope = {"torch": torch, "make_state": state.make_state}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    # 2048环境现场的原生float32姿态；再次归一化会改变扫描边缘射线的命中表面。
    raw = torch.tensor([[0.9946059584617615, 0.06997364014387131,
                         0.03470316529273987, -0.06825200468301773]])
    assert not torch.equal(raw, quat.quat_normalize(raw))
    data = SimpleNamespace(root_pos_w=torch.zeros(1, 3), root_quat_w=raw,
                           root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
                           joint_pos=torch.zeros(1, 12), default_joint_pos=torch.zeros(1, 12),
                           joint_vel=torch.zeros(1, 12))
    reader = SimpleNamespace(robot=SimpleNamespace(data=data), joint_ids=torch.arange(12), world_frame=True)
    observed = scope["state"](reader)
    assert torch.equal(observed[:, state.QUAT], raw)
    # 想象积分仍将姿态归一化，不能将读取精度修复变成长期数值漂移。
    predicted = state.integrate(observed, torch.zeros(1, state.DELTA_DIM))
    assert torch.allclose(predicted[:, state.QUAT].norm(dim=-1), torch.ones(1), atol=1e-7)


# ---------------- 观测与奖励 ----------------
def test_open_loop_errors_zero_for_exact_model_and_constant_velocity_baseline():
    """常速度轨迹上：真动力学和常速度基线开环误差都应为 0；跨 done 的起点不计入。"""
    from supermimic.diagnostics import open_loop_errors

    dt, T, N = rover_task.CONTROL_DT, 12, 4
    s = random_state(N, 0.5)
    s[:, state.JVEL] = 0.0
    states = [s]
    for _ in range(T - 1):
        states.append(state.constant_velocity_step(states[-1], dt))
    S, A = torch.stack(states), torch.zeros(T, N, 12)
    D = torch.zeros(T, N, dtype=torch.bool)
    D[5, 0] = True
    step = lambda x, a, c: state.constant_velocity_step(x, dt)  # noqa: E731
    err = open_loop_errors(S, A, D, step, horizons=(1, 4))
    for k in ("k1", "k4"):
        assert all(v < 1e-4 for name, v in err[k].items() if name != "samples")
    # k=4：env0 在起点 2..5 跨过 done[5]，共 8 个起点里少 4 个。
    assert err["k4"]["samples"] == 8 * N - 4


def test_policy_observation_layout_matches_45d_contract():
    s = random_state(3)
    cmd, last = torch.randn(3, 3), torch.randn(3, 12)
    obs = rover_task.policy_observation(s, cmd, last)
    assert obs.shape == (3, 45)
    assert torch.allclose(obs[:, 0:3], s[:, state.ANG] * 0.25)
    assert torch.allclose(obs[:, 3:6], quat.projected_gravity(s[:, state.QUAT]))
    assert torch.allclose(obs[:, 6:9], cmd)
    assert torch.allclose(obs[:, 9:17], s[:, state.JPOS][:, :8])
    assert torch.count_nonzero(obs[:, 17:21]) == 0
    assert torch.allclose(obs[:, 21:33], s[:, state.JVEL] * 0.05)
    assert torch.allclose(obs[:, 33:45], last)


def _upright_state(lin_vel, ang_z=0.0):
    s = state.make_state(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), torch.tensor([lin_vel]),
                         torch.tensor([[0.0, 0.0, ang_z]]), torch.zeros(1, 12), torch.zeros(1, 12))
    return s


def test_reward_prefers_tracking_and_gates_when_flipped():
    cfg = rover_task.RoverRewardCfg()
    cmd, zero = torch.tensor([[0.8, 0.0, 0.0]]), torch.zeros(1, 12)
    good = rover_task.reward(_upright_state([0.8, 0.0, 0.0]), _upright_state([0.8, 0.0, 0.0]), cmd, zero, zero, cfg)
    bad = rover_task.reward(_upright_state([0.0, 0.0, 0.0]), _upright_state([0.0, 0.0, 0.0]), cmd, zero, zero, cfg)
    assert good > bad
    flipped = _upright_state([0.8, 0.0, 0.0])
    flipped[:, state.QUAT] = quat.axis_angle_to_quat(torch.tensor([[torch.pi, 0.0, 0.0]]))
    terms = rover_task.reward_terms(flipped, flipped, cmd, zero, zero, cfg)
    assert terms["track_lin_vel"].item() == pytest.approx(0.0)


def test_survival_is_soft_and_consistent_with_hard_termination():
    cfg = rover_task.RoverRewardCfg()
    s = _upright_state([0.0, 0.0, 0.0]).repeat(3, 1)
    angles = torch.tensor([0.2, cfg.tilt_limit, 2.0])
    s[:, state.QUAT] = quat.axis_angle_to_quat(torch.stack((angles, 0 * angles, 0 * angles), -1))
    p = rover_task.survival(s, cfg)
    assert p[0] > 0.99 and p[1] == pytest.approx(0.5, abs=1e-3) and p[2] < 0.01
    assert rover_task.hard_terminated(s, cfg).tolist() == [False, False, True]


# ---------------- 回放 ----------------
def test_replay_windows_mask_episode_boundaries_and_wrap():
    buf = ReplayBuffer(capacity=6, num_envs=1, device="cpu")
    for t in range(9):  # 写满后再绕回 3 行
        s = torch.full((1, state.STATE_DIM), float(t))
        buf.add(s, torch.zeros(1, 12), torch.zeros(1, 12), torch.zeros(1, 3), torch.tensor([t == 5]))
    assert buf.size == 6
    torch.manual_seed(0)
    batch = buf.sample_windows(256, 2)
    first = batch["state"][:, 0, 0]
    assert first.min() >= 3 and first.max() <= 6  # 逻辑最旧行是 t=3，最后起点 t=6
    # 起点 t=5 的第一个转移跨过 done → 全部无效；起点 t=4 第二个转移无效。
    assert (batch["valid"][first == 5] == 0).all()
    assert (batch["valid"][first == 4] == torch.tensor([1.0, 0.0])).all()
    assert (batch["valid"][first == 3] == 1).all()


# ---------------- 网络 ----------------
def test_running_norm_matches_full_batch_statistics():
    norm, data = RunningNorm(4, eps=0.0), torch.randn(1000, 4) * 3 + 2
    for chunk in data.split(137):
        norm.update(chunk)
    assert torch.allclose(norm.mean, data.mean(0), atol=1e-4)
    assert torch.allclose(norm.var, data.var(0, unbiased=False), atol=1e-3)


def test_world_model_fits_simple_dynamics():
    """合成动力学：关节速度 ← 动作，其余静止。世界模型应能把单步误差压到很低。"""
    torch.manual_seed(0)
    wm = WorldModel(hidden=(128, 128))
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    s0 = random_state(2048, 0.3)

    def true_step(s, a):
        nxt = s.clone()
        nxt[:, state.JVEL] = 0.5 * s[:, state.JVEL] + a
        return nxt

    a = torch.randn(2048, 12)
    target = true_step(s0, a)
    wm.update_normalizers(s0, a, state.compute_delta(s0, target))
    for _ in range(400):
        loss = (state.state_error(wm(s0, a), target) / wm.delta_norm.std).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02


# ---------------- learner ----------------
def _tiny_learner(**overrides):
    params = dict(num_envs=8, buffer_len=40, wm_hidden=(32,), actor_hidden=(32,), critic_hidden=(32,),
                  wm_window=4, wm_batch=16, wm_updates=2, horizon=5, imag_batch=16, warmup_iterations=0,
                  context_dim=5, aux_dim=rover_task.AUX_DIM)
    params.update(overrides)
    return SuperMimicLearner(SuperMimicCfg(**params), "cpu")


def _fill(learner, steps=30):
    n = learner.cfg.num_envs
    s = random_state(n, 0.3)
    prev = torch.zeros(n, 12)
    ctx = torch.rand(n, learner.cfg.context_dim)
    for _ in range(steps):
        a = torch.randn_like(prev)
        learner.observe(s, a, prev, torch.rand(n, 3), torch.rand(n) < 0.05, ctx,
                        torch.rand(n, learner.cfg.aux_dim) * 1e4)
        s = state.integrate(s, torch.randn(n, state.DELTA_DIM) * 0.01)
        prev = a


def test_actor_update_changes_only_actor_and_critic():
    torch.manual_seed(0)
    learner = _tiny_learner()
    _fill(learner)
    learner.update_world_model()
    wm_before = {k: v.clone() for k, v in learner.world_model.state_dict().items()}
    actor_before = [p.clone() for p in learner.actor.parameters()]
    logs = learner.update_actor_critic()
    assert all(torch.equal(wm_before[k], v) for k, v in learner.world_model.state_dict().items())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, learner.actor.parameters()))
    assert all(p.requires_grad for p in learner.world_model.parameters())
    assert all(math.isfinite(v) for v in logs.values())
    assert logs["actor_update_applied"] == 1.0
    assert logs["actor_mirror_loss"] == 0.0


def test_action_saturation_penalizes_pre_squash_excess_on_limited_dims_only():
    torch.manual_seed(3)
    limits = ((-2.0, 2.0),) * 8 + (None,) * 4
    with pytest.raises(ValueError):
        _tiny_learner(action_saturation_coef=1.0)  # 没给界
    learner = _tiny_learner(action_saturation_coef=1.0, action_limits=limits)
    obs = torch.randn(6, learner.task.obs_dim)
    last = learner.actor.actor[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    loss, fraction = learner._saturation(obs)
    assert loss.item() == 0.0 and fraction.item() == 0.0
    with torch.no_grad():
        last.bias[0] = -30.0  # 摆臂远超界（tanh 后 ≈ −6.0，界 ±6.28 内但关节早到限位）
        last.bias[9] = 30.0  # 轮速不限
    loss, fraction = learner._saturation(obs)
    raw_limit = learner.cfg.action_bound * math.atanh(2.0 / learner.cfg.action_bound)
    assert loss.item() == pytest.approx((30.0 - raw_limit) ** 2, rel=1e-5)
    assert fraction.item() == pytest.approx(1.0 / 8.0)
    grad = torch.autograd.grad(loss, last.bias)[0]
    assert grad[0] < 0 and grad[9] == 0 and grad[1:8].abs().max() == 0  # 梯度下降把原始输出拉回界内
    _fill(learner)
    learner.update_world_model()
    logs = learner.update_actor_critic()
    assert logs["actor_saturation_loss"] > 0 and logs["actor_saturated_fraction"] == pytest.approx(1.0 / 8.0)


def test_actor_receives_reward_gradient_through_world_model_without_anchor(monkeypatch):
    torch.manual_seed(11)
    learner = _tiny_learner(aux_dim=0)
    learner.world_model.requires_grad_(False)
    start = {"state": random_state(4, 0.1), "command": torch.zeros(4, 3),
             "prev_action": torch.zeros(4, 12), "context": torch.zeros(4, 5)}
    # 回报仅依赖模型预测速度，排除动作正则项或模仿标签直接更新actor的旁路。
    monkeypatch.setattr(learner.task, "reward", lambda s, nxt, *args: nxt[:, state.LIN].square().sum(-1))
    loss = learner.imagine(start)["reward"].sum()
    grads = torch.autograd.grad(loss, tuple(learner.actor.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0
    step = learner.wm_step
    monkeypatch.setattr(learner, "wm_step", lambda *args: step(*args).detach())
    assert not learner.imagine(start)["reward"].requires_grad


def test_model_gradient_audit_preserves_forward_values_and_parameters():
    torch.manual_seed(11)
    learner = _tiny_learner(aux_dim=0)
    start = {"state": random_state(4, 0.1), "command": torch.zeros(4, 3),
             "prev_action": torch.zeros(4, 12), "context": torch.zeros(4, 5)}
    before = {name: {k: v.clone() for k, v in getattr(learner, name).state_dict().items()}
              for name in ("actor", "world_model", "critic")}
    report = learner.model_gradient_diagnostics(start)
    assert all(math.isfinite(v) for v in report.values())
    assert report["full_gradient_norm"] > 0 and report["through_dynamics_gradient_norm"] > 0
    assert report["forward_loss_abs_difference"] < 1e-6
    assert not any(key.startswith(("anchor", "motion_reference")) for key in report)
    for name, values in before.items():
        module = getattr(learner, name)
        assert all(torch.equal(v, module.state_dict()[key]) for key, v in values.items())
        assert all(p.grad is None for p in module.parameters())


def test_nonfinite_actor_skip_count_survives_resume_and_defaults_for_old_checkpoints():
    learner = _tiny_learner(aux_dim=0)
    learner.skipped_actor_updates = 3
    checkpoint = learner.state_dict()
    resumed = _tiny_learner(aux_dim=0)
    resumed.load_state_dict(checkpoint, load_optimizers=False)
    assert resumed.skipped_actor_updates == 3
    checkpoint.pop("skipped_actor_updates")
    resumed.load_state_dict(checkpoint, load_optimizers=False)
    assert resumed.skipped_actor_updates == 0


def test_actor_value_detach_preserves_forward_and_model_reward_gradient(monkeypatch):
    learner = _tiny_learner(aux_dim=0, actor_value_grad=False)
    start = {"state": random_state(4, 0.1), "command": torch.zeros(4, 3),
             "prev_action": torch.zeros(4, 12), "context": torch.zeros(4, 5)}
    monkeypatch.setattr(learner.task, "survival", lambda s: torch.ones(s.shape[0]))
    monkeypatch.setattr(learner.task, "reward", lambda s, nxt, *args: nxt[:, state.LIN].sum(-1) * 0.0)
    monkeypatch.setattr(learner, "_values", lambda critic, traj: traj["state"][:, :, state.LIN].sum(-1))
    traj = learner.imagine(start)
    detached = learner._actor_returns(traj)
    full = learner._actor_returns(traj, detach_value=False)
    assert torch.equal(detached, full)
    params = tuple(learner.actor.parameters())
    grads = torch.autograd.grad(detached.sum(), params)
    assert all(g.count_nonzero() == 0 for g in grads)
    monkeypatch.setattr(learner.task, "reward", lambda s, nxt, *args: nxt[:, state.LIN].square().sum(-1))
    grads = torch.autograd.grad(learner._actor_returns(learner.imagine(start)).sum(), params)
    assert all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0


def test_checkpoint_optimizer_rates_follow_active_config():
    original = _tiny_learner(actor_lr=1e-4, wm_lr=3e-4, critic_lr=3e-4)
    resumed = _tiny_learner(actor_lr=3e-5, wm_lr=1e-5, critic_lr=1e-4)
    resumed.load_state_dict(original.state_dict())
    assert resumed.actor_opt.param_groups[0]["lr"] == 3e-5
    assert resumed.wm_opt.param_groups[0]["lr"] == 1e-5
    assert resumed.critic_opt.param_groups[0]["lr"] == 1e-4


def test_legacy_checkpoint_reference_fields_are_ignored():
    """v0.10 存档里的 anchor_actor / motion_reference 字段在当前 learner 里被忽略，其余状态照常恢复。"""
    learner = _tiny_learner()
    with torch.no_grad():
        learner.actor.actor[-1].bias.add_(0.5)
    data = copy.deepcopy(learner.state_dict())
    assert not any(key.startswith(("anchor", "motion_reference")) for key in data)
    data.update(anchor_actor=copy.deepcopy(learner.actor.state_dict()), motion_reference=None,
                motion_reference_rng=torch.Generator().get_state())
    other = _tiny_learner()
    other.load_state_dict(data)
    obs = torch.randn(3, learner.task.obs_dim)
    assert torch.equal(other.actor(obs), learner.actor(obs))
    assert not hasattr(other, "anchor_actor") and not hasattr(other, "motion_reference")


@pytest.mark.parametrize("aux_dim", [0, 2])
def test_world_model_context_expansion_preserves_old_prediction_and_relief(aux_dim):
    torch.manual_seed(18)
    old_context, added, relief = 5, 12, 7
    old = WorldModel((32,), old_context + relief, aux_dim=aux_dim)
    s, a = random_state(4, 0.2), torch.randn(4, 12)
    ctx, terrain = torch.randn(4, old_context), torch.randn(4, relief)
    old.update_normalizers(s, a, torch.randn(4, state.DELTA_DIM), torch.cat((ctx, terrain), -1))
    new = WorldModel((32,), old_context + added + relief, aux_dim=aux_dim)
    new.load_context_expanded_state_dict(old.state_dict(), extra_dim=relief)
    material = torch.randn(4, added, requires_grad=True)
    expected = old(s, a, torch.cat((ctx, terrain), -1))
    actual = new(s, a, torch.cat((ctx, material, terrain), -1))
    assert torch.allclose(expected, actual, atol=1e-6)
    assert torch.count_nonzero(torch.autograd.grad(actual.sum(), material)[0]) == 0
    assert torch.equal(new.input_norm.mean[-relief:], old.input_norm.mean[-relief:])
    assert torch.equal(new.input_norm.var[-relief:], old.input_norm.var[-relief:])
    if aux_dim:
        assert torch.allclose(old.aux(s, a, torch.cat((ctx, terrain), -1)),
                              new.aux(s, a, torch.cat((ctx, material, terrain), -1)), atol=1e-6)
    new.load_context_expanded_state_dict(new.state_dict(), extra_dim=relief)
    with pytest.raises(ValueError, match="context"):
        old.load_context_expanded_state_dict(new.state_dict(), extra_dim=relief)


def test_world_model_uses_context_to_resolve_hidden_dynamics():
    """隐藏增益 ξ 决定关节速度响应；有 ξ 的模型应明显优于无 ξ 的模型（对应 randomize_actuator_gains）。"""
    torch.manual_seed(0)
    n = 4096
    s0 = random_state(n, 0.3)
    a = torch.randn(n, 12)
    gain = torch.rand(n, 1) * 1.5 + 0.5

    def true_step(s):
        nxt = s.clone()
        nxt[:, state.JVEL] = gain * a
        return nxt

    target = true_step(s0)
    losses = {}
    for ctx_dim in (0, 1):
        torch.manual_seed(1)
        wm = WorldModel(hidden=(128, 128), context_dim=ctx_dim)
        ctx = gain if ctx_dim else None
        opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
        wm.update_normalizers(s0, a, state.compute_delta(s0, target), ctx)
        for _ in range(300):
            loss = (state.state_error(wm(s0, a, ctx), target) / wm.delta_norm.std).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses[ctx_dim] = loss.item()
    assert losses[1] < 0.3 * losses[0]


def test_world_model_requires_declared_context():
    wm = WorldModel(hidden=(8,), context_dim=3)
    with pytest.raises(ValueError):
        wm(random_state(2), torch.zeros(2, 12))


def test_aux_head_learns_conditional_mean_and_feeds_reward():
    """aux 头用 MSE 回归 Σacc²，应收敛到条件均值；reward 使用 acc_sq 时三项等于权重×输入。"""
    torch.manual_seed(0)
    wm = WorldModel(hidden=(16,), aux_dim=3, aux_hidden=(64,))
    s0, a = random_state(4096, 0.3), torch.randn(4096, 12)
    target = torch.stack((a[:, 0].square(), a[:, 4].abs(), a[:, 8].square() * 10), -1) * 1e4
    wm.update_normalizers(s0, a, torch.zeros(4096, state.DELTA_DIM), aux=target)
    opt = torch.optim.Adam(wm.aux_net.parameters(), lr=3e-3)
    for _ in range(600):
        loss = (wm.aux_normalized(s0, a) - wm.aux_norm.normalize(target)).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred = wm.aux(s0, a)
    assert torch.allclose(pred.mean(0), target.mean(0), rtol=0.05)
    assert (pred >= 0).all()
    cfg = rover_task.RoverRewardCfg()
    acc = torch.tensor([[1e4, 2e4, 3e4]])
    s = _upright_state([0.0, 0.0, 0.0])
    terms = rover_task.reward_terms(s, s, torch.zeros(1, 3), torch.zeros(1, 12), torch.zeros(1, 12), cfg, acc)
    assert terms["suspension_acc"].item() == pytest.approx(cfg.suspension_acc * 1e4)
    assert terms["steering_acc"].item() == pytest.approx(cfg.steering_acc * 2e4)
    assert terms["wheel_acc"].item() == pytest.approx(cfg.wheel_acc * 3e4)


def test_world_model_update_reports_reward_gap_and_aux_loss():
    torch.manual_seed(0)
    for ctx_dim in (5, 0):  # context_dim=0 曾在 aux 展平时因空张量 reshape(-1, 0) 报错
        learner = _tiny_learner(context_dim=ctx_dim)
        _fill(learner)
        logs = learner.update_world_model()
        for key in ("wm_aux_loss", "wm_reward_bias", "wm_reward_abs_gap"):
            assert math.isfinite(logs[key])
        assert logs["wm_reward_abs_gap"] >= abs(logs["wm_reward_bias"]) - 1e-9


def test_actor_update_every_skips_intermediate_iterations():
    torch.manual_seed(0)
    learner = _tiny_learner(actor_update_every=3)
    _fill(learner)
    updated = ["actor_loss" in learner.update() for _ in range(6)]
    assert updated == [True, False, False, True, False, False]


def test_lambda_returns_reduce_to_discounted_sum_when_lambda_is_one():
    learner = _tiny_learner(lam=1.0, gamma=0.5)
    reward, survival = torch.ones(3, 1), torch.ones(3, 1)
    values = torch.tensor([[0.0], [0.0], [0.0], [8.0]])
    returns = learner.lambda_returns(reward, survival, values)
    assert returns[0].item() == pytest.approx(1 + 0.5 + 0.25 + 0.125 * 8)


def test_bptt_gradient_finite_from_exact_upright_rest():
    """回归：reset 后状态精确正立静止；survival 曾用 acos，cos=1 处梯度 -inf 使 actor 变 NaN。"""
    torch.manual_seed(0)
    learner = _tiny_learner()
    n = learner.cfg.num_envs
    rest = state.make_state(torch.zeros(n, 3), torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1), torch.zeros(n, 3),
                            torch.zeros(n, 3), torch.zeros(n, 12), torch.zeros(n, 12))
    for _ in range(10):
        learner.observe(rest, torch.zeros(n, 12), torch.zeros(n, 12), torch.rand(n, 3), torch.zeros(n, dtype=torch.bool),
                        torch.ones(n, learner.cfg.context_dim), torch.zeros(n, learner.cfg.aux_dim))
    learner.update_world_model()
    # 世界模型清零输出层后预测增量 = 归一化均值 ≈ 0，rollout 停在精确正立点上。
    torch.nn.init.zeros_(learner.world_model.net[-1].weight)
    torch.nn.init.zeros_(learner.world_model.net[-1].bias)
    learner.update_actor_critic()
    assert learner.skipped_actor_updates == 0
    assert all(torch.isfinite(p).all() for p in learner.actor.parameters())


def test_checkpoint_and_export_round_trip(tmp_path):
    torch.manual_seed(0)
    learner = _tiny_learner()
    _fill(learner)
    learner.update()
    other = _tiny_learner()
    other.load_state_dict(learner.state_dict())
    obs = torch.randn(5, 45)
    assert torch.allclose(other.act(obs, explore=False), learner.act(obs, explore=False))
    learner.export_actor(tmp_path / "policy.pt")
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    assert torch.allclose(scripted(obs), learner.act(obs, explore=False), atol=1e-6)
    assert scripted(obs * 1e3).abs().max() <= learner.cfg.action_bound + 1e-5


# ---------------- 扫描学生 ----------------
def _fake_teacher_state_dict():
    actor = scan_student.mlp(scan_student.TEACHER_OBS_DIM, (512, 256, 128), 12)
    return {f"actor.{k}": v for k, v in actor.state_dict().items()} | {"std": torch.ones(12)}


def test_scan_student_shapes_and_frozen_teacher():
    torch.manual_seed(0)
    teacher = scan_student.teacher_actor_from_rsl_rl(_fake_teacher_state_dict())
    policy = scan_student.ScanStudentPolicy(scan_student.DepthScanEncoder(latent_dim=32), teacher)
    student_obs = torch.rand(6, scan_student.STUDENT_OBS_DIM) * 2 - 1
    teacher_obs = torch.cat((student_obs[:, :45], torch.rand(6, 187) * 2 - 1), dim=-1)
    assert policy(student_obs).shape == (6, 12)
    loss, scan_loss, action_loss = scan_student.student_losses(policy, student_obs, teacher_obs,
                                                               scan_student.ScanStudentCfg())
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None for p in policy.encoder.parameters())
    # 预测扫描等于真值时，动作项应为 0。
    class ExactEncoder(torch.nn.Module):
        def forward(self, obs):
            return teacher_obs[:, 45:]

    exact = scan_student.ScanStudentPolicy(ExactEncoder(), teacher)
    _, s_loss, a_loss = scan_student.student_losses(exact, student_obs, teacher_obs, scan_student.ScanStudentCfg())
    assert s_loss.item() == pytest.approx(0.0) and a_loss.item() == pytest.approx(0.0, abs=1e-10)


def test_scan_student_exports_to_torchscript(tmp_path):
    torch.manual_seed(0)
    teacher = scan_student.teacher_actor_from_rsl_rl(_fake_teacher_state_dict())
    policy = scan_student.ScanStudentPolicy(scan_student.DepthScanEncoder(latent_dim=16), teacher).eval()
    torch.jit.script(policy).save(str(tmp_path / "student.pt"))
    scripted = torch.jit.load(str(tmp_path / "student.pt"))
    obs = torch.rand(3, scan_student.STUDENT_OBS_DIM) * 2 - 1
    assert torch.allclose(scripted(obs), policy(obs), atol=1e-6)


@pytest.mark.parametrize("lane_obs", [False, True])
@pytest.mark.parametrize("squash", ["tanh", "clamp"])
def test_scan_student_keeps_native_supermimic_actions_and_lane(tmp_path, lane_obs, squash):
    from supermimic.networks import Actor

    torch.manual_seed(12)
    teacher = Actor(232 + int(lane_obs), (32, 16), squash=squash)
    checkpoint = {"actor": teacher.state_dict(), "cfg": {
        "actor_hidden": (32, 16), "action_bound": teacher.action_bound, "action_squash": squash}}
    loaded, has_lane = scan_student.teacher_actor_from_checkpoint(checkpoint)
    obs = torch.randn(3, 232 + int(lane_obs))
    assert has_lane is lane_obs and torch.equal(loaded(obs), teacher(obs))
    policy = scan_student.ScanStudentPolicy(scan_student.DepthScanEncoder(latent_dim=8), loaded, lane_obs)
    student_obs = torch.randn(3, scan_student.STUDENT_OBS_DIM + int(lane_obs))
    if lane_obs:
        student_obs[:, -1] = obs[:, -1]
    scan = policy.encoder(student_obs)
    assembled = policy.actor_observation(student_obs, scan)
    if lane_obs:
        assert torch.equal(assembled[:, -1], obs[:, -1])
        changed = student_obs.clone()
        changed[:, -1] += 1.0
        assert torch.equal(policy.encoder(changed), scan)
    loss, _, _ = scan_student.student_losses(policy, student_obs, obs, scan_student.ScanStudentCfg())
    loss.backward()
    assert all(p.grad is None for p in loaded.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.encoder.parameters())
    scripted = torch.jit.script(policy.eval())
    scripted.save(str(tmp_path / "student.pt"))
    restored = torch.jit.load(str(tmp_path / "student.pt"))
    assert torch.allclose(restored(student_obs), policy(student_obs))


def test_lane_depth_contract_preserves_depth_and_adds_separate_localization():
    base = {"student_obs_dim": 4141, "teacher_obs_dim": 232, "history_order": "oldest_to_newest"}
    contract = scan_student.student_depth_contract(base, True)
    assert base["student_obs_dim"] == 4141 and contract["student_obs_dim"] == 4142
    assert contract["teacher_obs_dim"] == 233 and contract["extra_observations"][0]["index"] == 4141
    assert scan_student.student_depth_contract(base, False) == base


@pytest.mark.parametrize("lane_obs", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_evaluation_depth_freeze_reset_uses_only_depth_columns(lane_obs, reset):
    import ast

    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py"
    tree = ast.parse(path.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "frozen_depth is not None")
    obs = torch.randn(2, scan_student.STUDENT_OBS_DIM + int(lane_obs))
    frozen = torch.zeros(2, 4096)
    dones = torch.tensor([reset, False])
    scope = {"frozen_depth": frozen, "dones": dones, "obs": {"policy": obs}, "depth_end": 4141}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), str(path), "exec"), scope)
    assert torch.equal(frozen[dones], obs[dones, 45:4141])
    assert frozen[~dones].count_nonzero() == 0


def test_depth_reshape_is_oldest_to_newest_row_major():
    """第 f 帧第 (r,c) 像素必须在 45 + f*512 + r*32 + c，与 DepthTemporalBuffer 输出一致。"""
    enc = scan_student.DepthScanEncoder(latent_dim=8)
    captured = {}
    # hook 返回非 None 会替换模块输出，这里必须显式返回 None。
    def hook(module, inputs, output):
        captured["x"] = inputs[0]

    enc.cnn.register_forward_hook(hook)
    obs = torch.zeros(1, scan_student.STUDENT_OBS_DIM)
    obs[0, 45 + 3 * 512 + 5 * 32 + 7] = 1.0
    enc(obs)
    frames = captured["x"].reshape(8, 16, 32)
    assert frames[3, 5, 7] == 1.0 and frames.sum() == 1.0


def test_critic_warmup_freezes_actor_but_trains_critic():
    torch.manual_seed(0)
    learner = _tiny_learner(warmup_iterations=1, critic_warmup_iterations=2)
    _fill(learner)
    actor0 = [p.clone() for p in learner.actor.parameters()]
    critic0 = [p.clone() for p in learner.critic.parameters()]
    logs = [learner.update() for _ in range(3)]  # it0: 只世界模型；it1、it2: 只 critic
    assert "actor_loss" not in logs[0] and "critic_loss" in logs[1] and "critic_loss" in logs[2]
    assert all(torch.equal(a, b) for a, b in zip(actor0, learner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic0, learner.critic.parameters()))
    learner.update()  # it3: actor 开始更新
    assert any(not torch.equal(a, b) for a, b in zip(actor0, learner.actor.parameters()))


def test_real_critic_warmup_does_not_run_actor_imagination(monkeypatch):
    torch.manual_seed(2)
    learner = _tiny_learner(critic_real_returns=True, real_return_window=8, real_return_batch=16)
    _fill(learner)
    actor_before = [p.detach().clone() for p in learner.actor.parameters()]
    critic_before = [p.detach().clone() for p in learner.critic.parameters()]

    def unnecessary_imagination(*args):
        raise AssertionError("真实回报critic预热不应调用想象")

    monkeypatch.setattr(learner, "imagine", unnecessary_imagination)
    logs = learner.update_actor_critic(update_actor=False)
    assert logs["actor_update_applied"] == 0 and math.isfinite(logs["critic_loss"])
    assert all(torch.equal(a, b) for a, b in zip(actor_before, learner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters()))


def test_real_critic_updates_each_iteration_when_actor_is_throttled(monkeypatch):
    learner = _tiny_learner(warmup_iterations=2, critic_warmup_iterations=4, actor_update_every=3,
                            critic_real_returns=True)
    calls = []
    monkeypatch.setattr(learner, "update_world_model", lambda: {})
    monkeypatch.setattr(learner, "update_actor_critic", lambda update_actor: calls.append((learner.iteration, update_actor)) or {})
    for _ in range(12):
        learner.update()
    assert [iteration for iteration, _ in calls] == list(range(2, 12))
    assert [iteration for iteration, actor in calls if actor] == [8, 11]


def test_real_lambda_returns_handle_termination_and_truncation():
    learner = _tiny_learner(gamma=0.5, lam=1.0)
    r = torch.ones(3, 3)
    done = torch.tensor([[False, False, False], [False, True, False], [False, True, False]])
    term = torch.tensor([[False, False, False], [False, True, False], [False, False, False]])
    values = torch.tensor([[0.0, 0.0, 0.0, 8.0], [0.0, 4.0, 0.0, 8.0], [0.0, 4.0, 0.0, 8.0]])
    G = learner.real_lambda_returns({"reward": r, "done": done, "terminated": term}, values)
    assert G[0, 0].item() == pytest.approx(1 + 0.5 + 0.25 + 0.125 * 8)  # 窗口末端自举
    assert G[1, 1].item() == pytest.approx(1.0)  # 真终止不自举
    assert G[1, 0].item() == pytest.approx(1 + 0.5 * 1.0)
    assert G[2, 1].item() == pytest.approx(1 + 0.5 * 4.0)  # 截断用本步价值近似自举


def test_critic_real_returns_fit_replay_lambda_returns():
    torch.manual_seed(0)
    learner = _tiny_learner(critic_real_returns=True, real_return_window=4, real_return_batch=16, critic_lr=1e-2)
    n = learner.cfg.num_envs
    s = random_state(n, 0.3)
    for _ in range(30):
        learner.observe(s, torch.randn(n, 12), torch.zeros(n, 12), torch.rand(n, 3), torch.rand(n) < 0.05,
                        torch.rand(n, 5), torch.rand(n, 3), reward=torch.ones(n), terminated=torch.zeros(n, dtype=torch.bool))
        s = state.integrate(s, torch.randn(n, state.DELTA_DIM) * 0.01)
    start = learner.replay.sample_states(64)

    def value():
        with torch.no_grad():
            return learner.critic(start["state"], start["command"], start["prev_action"],
                                  learner.critic_context(start["state"], start["context"])).mean().item()

    before = value()
    losses = [learner.update()["critic_loss"] for _ in range(20)]
    # 回报全为 +1：真实 λ-return 目标为正，critic 应明显上移
    assert all(math.isfinite(v) for v in losses)
    assert value() > before + 0.5


