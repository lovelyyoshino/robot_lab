"""SuperMimic：Rodrigues FK 等价性、左右镜像映射、镜像增强/一致性损失、诊断间隔、actor 学习率余弦衰减、
torch.compile 回退。不依赖仿真器。

@author pony
@date 2026-11-05
@version v1.0.0
@last_modified 2026-11-05
@changelog
    - v1.0.0 (2026-11-05): 初始创建。
"""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

from supermimic import kinematics, quat, rover_task, state, symmetry, terrain_map  # noqa: E402
from supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402
from supermimic.preview_task import PreviewRewardCfg, PreviewTask, proprio_observation  # noqa: E402

_VP = Path(__file__).resolve().parents[1] / (
    "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview.py")


def _preview_pattern():
    """Teacher-Preview 的真实射线图样（vision_preview.near_field_height_pattern + 传感器 offset 0.30 m）。"""
    spec = importlib.util.spec_from_file_location("vision_preview_symmetry_test", _VP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    cfg = type("Cfg", (), {"x_positions": mod.PREVIEW_SCAN_X, "y_positions": mod.PREVIEW_SCAN_Y})
    starts, _ = mod.near_field_height_pattern(cfg, "cpu")
    return starts + torch.tensor([0.30, 0.0, 20.0]), len(mod.PREVIEW_SCAN_X), len(mod.PREVIEW_SCAN_Y)


def _random_state(n, scale=0.5, generator=None):
    g = generator
    return state.make_state(torch.randn(n, 3, generator=g), quat.quat_normalize(torch.randn(n, 4, generator=g)),
                            torch.randn(n, 3, generator=g) * scale, torch.randn(n, 3, generator=g) * scale,
                            torch.randn(n, 12, generator=g) * 0.3, torch.randn(n, 12, generator=g) * scale)


# ---------------- FK：Rodrigues 缓冲 vs 旧逐元素公式 ----------------
@pytest.mark.parametrize("shape", [(4,), (1, 4), (4096, 4), (128, 32, 4), (2, 3, 5, 4)])
def test_fast_fk_matches_reference_formula(shape):
    torch.manual_seed(0)
    kin = kinematics.RoverKinematics()
    arm = (torch.rand(shape) * 2 - 1) * 0.7
    steer = (torch.rand(shape) * 2 - 1) * 1.6
    assert kin.wheel_centers_base(arm, steer).shape == shape + (3,)
    diff = (kin.wheel_centers_base(arm, steer) - kin.wheel_centers_base_reference(arm, steer)).abs().max()
    assert diff <= 1e-6
    pos, q = torch.randn(shape[:-1] + (3,)), quat.quat_normalize(torch.randn(shape[:-1] + (4,)))
    world = kin.wheel_centers_world(pos, q, arm, steer)
    reference = pos.unsqueeze(-2) + quat.quat_apply(q.unsqueeze(-2).expand(shape + (4,)),
                                                    kin.wheel_centers_base_reference(arm, steer))
    assert (world - reference).abs().max() <= 1e-6


def test_fast_fk_gradient_matches_reference_formula():
    torch.manual_seed(1)
    kin = kinematics.RoverKinematics()
    arm = ((torch.rand(64, 4) * 2 - 1) * 0.7).requires_grad_(True)
    steer = ((torch.rand(64, 4) * 2 - 1) * 1.6).requires_grad_(True)
    weight = torch.randn(64, 4, 3)
    fast = torch.autograd.grad((kin.wheel_centers_base(arm, steer) * weight).sum(), (arm, steer))
    slow = torch.autograd.grad((kin.wheel_centers_base_reference(arm, steer) * weight).sum(), (arm, steer))
    for a, b in zip(fast, slow):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_fast_fk_float64_module_is_exact():
    kin = kinematics.RoverKinematics().double()
    arm, steer = torch.rand(256, 4, dtype=torch.float64) - 0.5, torch.rand(256, 4, dtype=torch.float64) - 0.5
    assert (kin.wheel_centers_base(arm, steer) - kin.wheel_centers_base_reference(arm, steer)).abs().max() < 1e-7


# ---------------- 镜像映射 ----------------
def test_joint_mirror_signs_follow_urdf_axes():
    # arm 与 wheel 轴两侧都是 base +y（俯仰/滚动同号），steer 轴是 +z（偏航反号）
    assert symmetry.joint_mirror_signs() == (1.0, -1.0, 1.0)
    index, sign = symmetry.joint_mirror()
    assert index == [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
    assert sign == [1.0] * 4 + [-1.0] * 4 + [1.0] * 4


def test_mirrored_joint_angles_mirror_wheel_positions():
    """镜像关节角下的 FK = 原 FK 关于 xz 平面镜像再左右互换；误差只来自 URDF 的毫米级不对称。"""
    torch.manual_seed(2)
    kin = kinematics.RoverKinematics()
    mirror = symmetry.RoverMirror("cpu")
    joints = torch.zeros(512, 12)
    joints[:, :8] = (torch.rand(512, 8) * 2 - 1) * torch.tensor([0.55] * 4 + [1.4] * 4)
    mirrored = mirror.action(joints)
    wheels = kin.wheel_centers_base(joints[:, :4], joints[:, 4:8])
    expected = wheels[:, symmetry.corner_mirror_permutation()] * torch.tensor([1.0, -1.0, 1.0])
    error = (kin.wheel_centers_base(mirrored[:, :4], mirrored[:, 4:8]) - expected).abs().max()
    wrong = joints.clone()
    wrong[:, 4:8] = joints[:, [6, 7, 4, 5]]  # 不翻 steer 符号时的错误映射
    wrong_error = (kin.wheel_centers_base(wrong[:, :4], wrong[:, 4:8]) - expected).abs().max()
    assert error < 0.03 and wrong_error > 5 * error


@pytest.mark.parametrize("lane_obs", [False, True])
@pytest.mark.parametrize("context_dim", [0, symmetry.CONTEXT_DIM, symmetry.CONTACT_CONTEXT_DIM])
def test_mirror_is_an_involution(lane_obs, context_dim):
    pattern, _, _ = _preview_pattern()
    obs_dim = 45 + pattern.shape[0] + int(lane_obs)
    mirror = symmetry.RoverMirror("cpu", obs_dim, pattern, lane_obs, context_dim, terrain_map.RELIEF_DIM)
    for name, dim in (("state", state.STATE_DIM), ("action", 12), ("command", 3), ("obs", obs_dim),
                      ("context", context_dim), ("relief", terrain_map.RELIEF_DIM)):
        x = torch.randn(7, 3, dim)
        fn = getattr(mirror, name)
        assert torch.equal(fn(fn(x)), x), name
        if dim:
            assert not torch.equal(fn(x), x), name
    flat = symmetry.RoverMirror("cpu")
    x = torch.randn(5, 45)
    assert torch.equal(flat.obs(flat.obs(x)), x)


def test_context_layout_matches_isaac_adapter_formula():
    source = (Path(symmetry.__file__).parent / "isaac_adapter.py").read_text()
    assert "CONTEXT_DIM = NUM_LEG_JOINTS + 12 + 1 + 1 + 3" in source
    assert "CONTACT_CONTEXT_DIM = CONTEXT_DIM + 4 * 3" in source
    assert symmetry.CONTEXT_DIM == 25 and symmetry.CONTACT_CONTEXT_DIM == 37
    mirror = symmetry.RoverMirror("cpu", context_dim=37)
    ctx = torch.arange(37.0)[None]
    out = mirror.context(ctx)[0]
    assert out[:8].tolist() == [2, 3, 0, 1, 6, 7, 4, 5]  # Kp：fl↔fr, rl↔rr
    assert out[23].item() == -23.0 and out[22].item() == 22.0  # 质心 y 取反
    assert out[25:28].tolist() == [31, 32, 33]  # fl 位置拿到 fr 的材料


def _symmetric_state(n=6):
    torch.manual_seed(3)
    s = torch.zeros(n, state.STATE_DIM)
    s[:, 0], s[:, 2] = torch.randn(n), 0.37
    pitch = torch.randn(n) * 0.2
    s[:, 3:7] = torch.stack((torch.cos(pitch / 2), torch.zeros(n), torch.sin(pitch / 2), torch.zeros(n)), -1)
    s[:, 7], s[:, 9], s[:, 11] = torch.randn(n), torch.randn(n), torch.randn(n)  # vx, vz, wy
    arm, steer, wheel = torch.randn(n, 2) * 0.3, torch.randn(n, 2) * 0.3, torch.randn(n, 2)
    for block, (a, st, w) in ((state.JPOS, (arm, steer, torch.zeros(n, 2))), (state.JVEL, (arm, steer, wheel))):
        joints = torch.cat((a, a, st, -st, w, w), -1)  # (fl, rl) = (fr, rr)；steer 左右相反
        s[:, block] = joints
    return s


def test_symmetric_configuration_maps_to_itself():
    pattern, _, _ = _preview_pattern()
    mirror = symmetry.RoverMirror("cpu", 45 + pattern.shape[0], pattern, False, 25, terrain_map.RELIEF_DIM)
    s = _symmetric_state()
    assert torch.allclose(mirror.state(s), s)
    command = torch.tensor([[0.3, 0.0, 0.0]]).expand(len(s), 3)
    last = torch.cat((torch.ones(len(s), 4) * 0.2, torch.tensor([[0.1, 0.1, -0.1, -0.1]]).expand(len(s), 4),
                      torch.ones(len(s), 4)), -1)
    proprio = proprio_observation(s, command, last)
    assert torch.allclose(mirror.obs(torch.cat((proprio, torch.zeros(len(s), pattern.shape[0])), -1))[:, :45], proprio)
    assert torch.allclose(mirror.command(command), command)


def test_mirror_state_matches_reflected_rotation_and_velocities():
    s = _random_state(32)
    m = symmetry.RoverMirror("cpu").state(s)
    flip = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    v = torch.randn(32, 3)
    # R' = S R S：机体系向量先镜像再旋转 = 旋转后镜像
    assert torch.allclose(quat.quat_apply(m[:, state.QUAT], v @ flip), quat.quat_apply(s[:, state.QUAT], v) @ flip,
                          atol=1e-6)
    assert torch.allclose(quat.projected_gravity(m[:, state.QUAT]), quat.projected_gravity(s[:, state.QUAT]) @ flip,
                          atol=1e-6)
    # 世界系线速度镜像；角速度是伪向量（−S）
    lin_w = quat.quat_apply(s[:, state.QUAT], s[:, state.LIN])
    assert torch.allclose(quat.quat_apply(m[:, state.QUAT], m[:, state.LIN]), lin_w @ flip, atol=1e-6)
    ang_w = quat.quat_apply(s[:, state.QUAT], s[:, state.ANG])
    assert torch.allclose(quat.quat_apply(m[:, state.QUAT], m[:, state.ANG]), -(ang_w @ flip), atol=1e-6)


def test_scan_grid_mirror_uses_preview_pattern_layout():
    pattern, nx, ny = _preview_pattern()
    assert pattern.shape[0] == 187 and (nx, ny) == (17, 11)
    index = symmetry.mirror_point_index(pattern[:, :2])
    # meshgrid(indexing="xy") 展平：k = iy·nx + ix，镜像只翻 iy
    expected = [(ny - 1 - k // nx) * nx + k % nx for k in range(nx * ny)]
    assert index == expected
    asymmetric = pattern.clone()
    asymmetric[:, 1] += 0.05
    with pytest.raises(ValueError):
        symmetry.mirror_point_index(asymmetric[:, :2])


def _random_map(res=0.05, half=3.0):
    g = torch.Generator().manual_seed(4)
    xs = torch.arange(-half, half + 1e-9, res)
    ys = torch.arange(-half, half + 1e-9, res)
    heights = torch.randn(len(xs), len(ys), generator=g).cumsum(0).cumsum(1) * 1e-3
    return terrain_map.HeightMap(heights, (-half, -half), res), terrain_map.HeightMap(heights.flip(1), (-half, -half), res)


def test_height_scan_in_mirrored_world_equals_mirrored_scan():
    """镜像状态在镜像地形上的扫描 = 原扫描按格点镜像（扫描不经过 FK，应逐点相等）。"""
    pattern, _, _ = _preview_pattern()
    hmap, flipped = _random_map()
    mirror = symmetry.RoverMirror("cpu", 45 + 187, pattern)
    s = _random_state(16, 0.3)
    s[:, :2] = s[:, :2].clamp(-0.8, 0.8)
    s[:, 2] = 0.37
    scan = terrain_map.height_scan(hmap, s, pattern)
    scan_m = terrain_map.height_scan(flipped, mirror.state(s), pattern)
    obs = torch.cat((torch.zeros(16, 45), scan), -1)
    assert torch.allclose(scan_m, mirror.obs(obs)[:, 45:], atol=2e-5)


class _SymmetricKinematics:
    """严格左右对称的轮心（URDF 有毫米级不对称），只用于核对 relief 布局。"""

    offsets = torch.tensor([[0.3, 0.3, -0.2], [-0.3, 0.3, -0.2], [0.3, -0.3, -0.2], [-0.3, -0.3, -0.2]])

    def wheel_centers_world(self, pos, q, arm, steer):
        lift = torch.stack((torch.zeros_like(arm), torch.zeros_like(arm), 0.1 * arm), -1)
        local = self.offsets + lift
        return pos.unsqueeze(-2) + quat.quat_apply(q.unsqueeze(-2).expand(local.shape[:-1] + (4,)), local)


def test_relief_features_in_mirrored_world_equal_mirrored_features():
    hmap, flipped = _random_map()
    kin = _SymmetricKinematics()
    mirror = symmetry.RoverMirror("cpu", extra_dim=terrain_map.RELIEF_DIM)
    s = _random_state(16, 0.3)
    s[:, :2] = s[:, :2].clamp(-0.8, 0.8)
    s[:, 2] = 0.37
    relief = terrain_map.relief_features(hmap, kin, s)
    relief_m = terrain_map.relief_features(flipped, kin, mirror.state(s))
    assert torch.allclose(relief_m, mirror.relief(relief), atol=2e-5)
    # 半镜像包装：镜像行映回真实地形查询，再镜像特征
    extra = symmetry.half_mirrored_extra_fn(lambda x: terrain_map.relief_features(hmap, kin, x), mirror, 16)
    both = extra(torch.cat((s, mirror.state(s))))
    assert torch.allclose(both[:16], relief) and torch.allclose(both[16:], mirror.relief(relief), atol=1e-6)


# ---------------- learner 集成 ----------------
def _learner(**overrides):
    params = dict(num_envs=8, buffer_len=40, wm_hidden=(32,), actor_hidden=(32,), critic_hidden=(32,),
                  wm_window=4, wm_batch=16, wm_updates=2, horizon=4, imag_batch=8, warmup_iterations=0,
                  context_dim=symmetry.CONTEXT_DIM, aux_dim=rover_task.AUX_DIM)
    params.update(overrides)
    return SuperMimicLearner(SuperMimicCfg(**params), "cpu")


def _fill(learner, steps=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = learner.cfg.num_envs
    s = _random_state(n, 0.3, g)
    prev = torch.zeros(n, 12)
    ctx = torch.rand(n, learner.cfg.context_dim, generator=g)
    for _ in range(steps):
        a = torch.randn(n, 12, generator=g)
        learner.observe(s, a, prev, torch.rand(n, 3, generator=g), torch.rand(n, generator=g) < 0.05, ctx,
                        torch.rand(n, learner.cfg.aux_dim, generator=g) * 1e4, reward=torch.randn(n, generator=g),
                        terminated=torch.zeros(n, dtype=torch.bool))
        s = state.integrate(s, torch.randn(n, state.DELTA_DIM, generator=g) * 0.01)
        prev = a


def test_conjugate_policy_acts_on_mirrored_observations():
    learner = _learner(mirror_aug=True)
    obs = torch.randn(6, 45)
    out = learner._imagined_action(obs, 3)
    assert torch.allclose(out[:3], learner.actor(obs[:3]))
    assert torch.allclose(out[3:], learner.mirror.action(learner.actor(learner.mirror.obs(obs[3:]))))


def test_mirror_aug_doubles_world_model_windows_and_imagination(monkeypatch):
    torch.manual_seed(0)
    learner = _learner(mirror_aug=True, mirror_loss_coef=0.5)
    _fill(learner)
    sizes = {"wm": [], "imag": []}
    rollout, imagine = learner.model_rollout_loss, learner.imagine
    monkeypatch.setattr(learner, "model_rollout_loss",
                        lambda batch, **kw: sizes["wm"].append(batch["state"].shape[0]) or rollout(batch, **kw))
    monkeypatch.setattr(learner, "imagine",
                        lambda start, **kw: sizes["imag"].append(start["state"].shape[0]) or imagine(start, **kw))
    logs = learner.update()
    assert sizes == {"wm": [32, 32], "imag": [16]}
    assert all(math.isfinite(v) for v in logs.values())
    assert logs["actor_mirror_loss"] > 0
    # 镜像副本同样进入归一化统计：横向速度均值被对称化到 ~0
    lin_y = learner.world_model.input_norm.mean[4]
    assert abs(lin_y.item()) < 1e-4


def test_mirror_consistency_loss_drives_policy_toward_symmetry():
    torch.manual_seed(5)
    learner = _learner(mirror_loss_coef=1.0, actor_lr=3e-3)
    _fill(learner)
    monkeypatch_returns = lambda traj, detach_value=None: traj["reward"] * 0.0  # noqa: E731
    learner._actor_returns = monkeypatch_returns
    start = learner.replay.sample_states(64)
    before = learner._mirror_consistency(start).item()
    for _ in range(30):
        learner.update_actor_critic()
    assert learner._mirror_consistency(start).item() < 0.5 * before


def test_mirror_aug_rejects_unknown_layouts():
    with pytest.raises(ValueError):
        _learner(mirror_aug=True, context_dim=5)


def test_preview_task_mirror_layout_from_task():
    pattern, _, _ = _preview_pattern()
    hmap, _ = _random_map()
    task = PreviewTask(hmap, kinematics.RoverKinematics(), pattern, torch.tensor([[-1.0, 1.0]] * 8),
                       PreviewRewardCfg(), lane_center_y=torch.tensor([-4.0, 4.0]))
    mirror = symmetry.RoverMirror.for_task(task, "cpu", symmetry.CONTACT_CONTEXT_DIM, terrain_map.RELIEF_DIM)
    s = _random_state(3, 0.3)
    obs = task.policy_obs(s, torch.tensor([[0.3, 0.0, 0.0]]).expand(3, 3), torch.randn(3, 12))
    assert task.obs_dim == 232 == obs.shape[-1] and torch.equal(mirror.obs(mirror.obs(obs)), obs)
    # 镜像后的本体观测 = 镜像状态的本体观测（命令/上一动作同时镜像）
    proprio_m = proprio_observation(mirror.state(s), torch.tensor([[0.3, 0.0, 0.0]]).expand(3, 3),
                                    mirror.action(obs[:, 33:45]))
    assert torch.allclose(mirror.obs(obs)[:, :45], proprio_m, atol=1e-5)
    lane = symmetry.RoverMirror("cpu", 233, pattern, lane_obs=True)
    x = torch.randn(3, 233)
    assert torch.equal(lane.obs(x)[:, -1], -x[:, -1])


# ---------------- 诊断间隔 / 学习率 / compile 回退 ----------------
def _train(interval, iterations=4):
    torch.manual_seed(7)
    learner = _learner(diagnostics_interval=interval, horizon=3)
    _fill(learner, seed=7)
    logs = [learner.update() for _ in range(iterations)]
    return learner, logs


def test_diagnostics_interval_skips_reward_gap_without_changing_training():
    every, logs_every = _train(1)
    sparse, logs_sparse = _train(3)
    assert ["wm_reward_bias" in row for row in logs_sparse] == [True, False, False, True]
    assert all("wm_reward_bias" in row for row in logs_every)
    for name in ("world_model", "actor", "critic"):
        a, b = getattr(every, name).state_dict(), getattr(sparse, name).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a), name
    for row_a, row_b in zip(logs_every, logs_sparse):
        assert all(row_a[k] == v for k, v in row_b.items())


def test_actor_lr_cosine_decay_from_actor_start():
    learner = _learner(actor_lr=1e-4, actor_lr_final=1e-5, actor_lr_decay_iterations=100, warmup_iterations=10,
                       critic_warmup_iterations=5)
    values = {}
    for iteration in (0, 15, 65, 115, 500):
        learner.iteration = iteration
        values[iteration] = learner._actor_lr()
    assert values[0] == pytest.approx(1e-4) and values[15] == pytest.approx(1e-4)
    assert values[65] == pytest.approx(5.5e-5) and values[115] == pytest.approx(1e-5) and values[500] == pytest.approx(1e-5)
    assert _learner(actor_lr=2e-4)._actor_lr() == 2e-4


def test_actor_lr_schedule_is_applied_to_optimizer():
    torch.manual_seed(0)
    learner = _learner(actor_lr=1e-3, actor_lr_final=1e-4, actor_lr_decay_iterations=4)
    _fill(learner)
    learner.iteration = 2
    logs = learner.update_actor_critic()
    assert logs["actor_lr"] == pytest.approx(5.5e-4)
    assert learner.actor_opt.param_groups[0]["lr"] == pytest.approx(5.5e-4)


def test_compile_failure_falls_back_to_identical_eager(monkeypatch):
    def broken_compile(fn, **kwargs):
        def run(*args):
            raise RuntimeError("compiler unavailable")
        return run

    reference, _ = _train(1, iterations=2)
    monkeypatch.setattr(torch, "compile", broken_compile)
    torch.manual_seed(7)
    learner = _learner(diagnostics_interval=1, horizon=3, compile_wm=True, compile_imagination=True)
    _fill(learner, seed=7)
    with pytest.warns(UserWarning, match="torch.compile"):
        for _ in range(2):
            learner.update()
    for name in ("world_model", "actor", "critic"):
        a, b = getattr(reference, name).state_dict(), getattr(learner, name).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a), name


def test_nonfinite_compiled_imagination_falls_back_to_eager(monkeypatch):
    torch.manual_seed(0)
    learner = _learner(compile_imagination=True)
    _fill(learner)

    def fake_compile(fn, **kwargs):
        # "编译版"替身：前向不变、反传梯度为 NaN（复现 Isaac 越障 A/B 中观察到的现象）
        def run(*args):
            action, next_state, reward, survival = fn(*args)
            reward = reward * 1.0
            reward.register_hook(lambda grad: grad * float("nan"))
            return action, next_state, reward, survival
        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    with pytest.warns(UserWarning, match="退回 eager"):
        logs = learner.update_actor_critic()
    assert learner._eager_imagination and learner.skipped_actor_updates == 0
    assert math.isfinite(logs["actor_loss"]) and logs["actor_update_applied"] == 1.0


@pytest.mark.skipif(os.environ.get("SUPERMIMIC_COMPILE_TEST") != "1", reason="真实编译较慢，设 SUPERMIMIC_COMPILE_TEST=1 运行")
def test_compiled_steps_match_eager_to_rounding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for compiled in (False, True):
        torch.manual_seed(9)
        learner = SuperMimicLearner(SuperMimicCfg(
            num_envs=8, buffer_len=40, wm_hidden=(64,), actor_hidden=(32,), critic_hidden=(32,), wm_window=4,
            wm_batch=16, wm_updates=1, horizon=3, imag_batch=8, warmup_iterations=0, context_dim=0, aux_dim=0,
            wm_reanchor_error=10.0, compile_wm=compiled, compile_imagination=compiled), device)
        g = torch.Generator().manual_seed(9)
        s = _random_state(8, 0.3, g)
        for _ in range(30):
            learner.observe(s.to(device), torch.randn(8, 12, generator=g).to(device), torch.zeros(8, 12, device=device),
                            torch.rand(8, 3, generator=g).to(device), torch.zeros(8, dtype=torch.bool, device=device))
            s = state.integrate(s, torch.randn(8, state.DELTA_DIM, generator=g) * 0.01)
        results.append([learner.update() for _ in range(2)])
        assert not learner._compile_failed
    for eager, compiled in zip(*results):
        for key in ("wm_loss", "actor_loss", "critic_loss"):
            assert compiled[key] == pytest.approx(eager[key], rel=1e-3, abs=1e-6), key
