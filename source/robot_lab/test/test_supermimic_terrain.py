"""SuperMimic 地形扩展（可微 FK、高度图、想象高度扫描、地形特征、learner extra_fn）回归，不依赖仿真器。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

from supermimic import kinematics, quat, state, terrain_map  # noqa: E402
from supermimic.learner import SuperMimicCfg, SuperMimicLearner  # noqa: E402


# ---------------- FK：与独立的 4×4 齐次变换实现对拍 ----------------
def _homogeneous_reference(urdf, corner, arm, steer):
    root = ET.parse(str(urdf)).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    def T(elem, angle=0.0, axis=None):
        o = elem.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        r, p, y = (float(v) for v in o.get("rpy").split())
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = Rz @ Ry @ Rx, xyz
        if axis is not None:
            a = np.asarray(axis, float) / np.linalg.norm(axis)
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            J = np.eye(4)
            J[:3, :3] = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
            M = M @ J
        return M

    def ax(j):
        return [float(v) for v in j.find("axis").get("xyz").split()]

    a, z, q, w = (joints[f"{corner}_{n}_joint"] for n in ("arm", "zhuanxiang", "qudong", "wheel"))
    mult = float(z.find("mimic").get("multiplier"))
    M = T(a, arm, ax(a)) @ T(z, mult * arm, ax(z)) @ T(q, steer, ax(q)) @ T(w)
    coll = next(lk for lk in root.findall("link") if lk.get("name") == f"{corner}_wheel").find("collision")
    off = np.array([float(v) for v in coll.find("origin").get("xyz").split()] + [1.0])
    return (M @ off)[:3]


def test_fk_matches_independent_homogeneous_chain():
    kin = kinematics.RoverKinematics()
    rng = np.random.default_rng(0)
    arm = torch.tensor(rng.uniform(-0.8, 0.8, (5, 4)), dtype=torch.float32)
    steer = torch.tensor(rng.uniform(-0.8, 0.8, (5, 4)), dtype=torch.float32)
    out = kin.wheel_centers_base(arm, steer)
    for b in range(5):
        for c, corner in enumerate(kinematics.CORNERS):
            ref = _homogeneous_reference(kinematics.DEFAULT_URDF, corner, arm[b, c].item(), steer[b, c].item())
            assert np.allclose(out[b, c].numpy(), ref, atol=1e-5), (corner, out[b, c], ref)


def test_fk_geometry_is_plausible_and_differentiable():
    kin = kinematics.RoverKinematics()
    zero = torch.zeros(1, 4)
    w = kin.wheel_centers_base(zero, zero)[0]
    # 左轮 y>0、右轮 y<0；前轮 x>后轮 x；轮心在 base 下方。
    assert (w[[0, 1], 1] > 0).all() and (w[[2, 3], 1] < 0).all()
    assert w[0, 0] > w[1, 0] and w[2, 0] > w[3, 0]
    assert (w[:, 2] < 0).all()
    arm = torch.zeros(1, 4, requires_grad=True)
    kin.wheel_centers_base(arm, zero)[..., 2].sum().backward()
    assert torch.isfinite(arm.grad).all() and arm.grad.abs().sum() > 0


def test_fk_world_applies_base_pose():
    kin = kinematics.RoverKinematics()
    zero = torch.zeros(1, 4)
    yaw = quat.axis_angle_to_quat(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    base = kin.wheel_centers_base(zero, zero)
    world = kin.wheel_centers_world(torch.tensor([[1.0, 2.0, 0.3]]), yaw, zero, zero)
    # 绕 z 转 90°：(x, y) → (−y, x)
    assert torch.allclose(world[0, :, 0], 1.0 - base[0, :, 1], atol=1e-6)
    assert torch.allclose(world[0, :, 1], 2.0 + base[0, :, 0], atol=1e-6)


# ---------------- 高度图 / 扫描 ----------------
def _step_map(res=0.02, step_x=1.0, height=0.12):
    xs = torch.arange(0, 3.0 + 1e-9, res)
    ys = torch.arange(-1.0, 1.0 + 1e-9, res)
    h = torch.where(xs[:, None] >= step_x, height, 0.0).expand(len(xs), len(ys))
    return terrain_map.HeightMap(h.clone(), (0.0, -1.0), res)


def test_heightmap_exact_on_grid_and_linear_between():
    hm = _step_map()
    pts = torch.tensor([[0.5, 0.0], [2.0, 0.3], [1.0, 0.0], [0.99, 0.0], [-5.0, 0.0], [9.0, 0.0]])
    out = hm.query(pts)
    assert torch.allclose(out, torch.tensor([0.0, 0.12, 0.12, 0.06, 0.0, 0.12]), atol=1e-5)
    xy = torch.tensor([[0.99, 0.0]], requires_grad=True)
    hm.query(xy).sum().backward()
    assert xy.grad[0, 0].item() == pytest.approx(0.12 / 0.02, rel=1e-3)
    # 回归：正好落在最后一行/列（float32 下 n-1-ε 曾舍入成 n-1 导致越界）
    nx, ny = hm.heights.shape
    corner = torch.tensor([[hm.x0 + (nx - 1) * hm.res, hm.y0 + (ny - 1) * hm.res]])
    assert hm.query(corner).item() == pytest.approx(0.12)


def _flat_state(x, z, yaw=0.0):
    return state.make_state(torch.tensor([[x, 0.0, z]]), quat.axis_angle_to_quat(torch.tensor([[0.0, 0.0, yaw]])),
                            torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 12), torch.zeros(1, 12))


def test_height_scan_formula_and_yaw_alignment():
    hm = _step_map()
    pattern = torch.tensor([[0.0, 0.0, 20.0], [0.6, 0.0, 20.0]])  # 起点含 20 m 竖直 offset，不影响命中
    s = _flat_state(0.5, 0.40)
    scan = terrain_map.height_scan(hm, s, pattern)
    assert torch.allclose(scan, torch.tensor([[0.40 - 0.0 - 0.5, 0.40 - 0.12 - 0.5]]), atol=1e-5)
    # 掉头 180°：前方点落到 x=-0.1 的平地
    back = terrain_map.height_scan(hm, _flat_state(0.5, 0.40, math.pi), pattern)
    assert torch.allclose(back, torch.tensor([[-0.10, -0.10]]), atol=1e-5)


def test_relief_features_shape_gradient_and_edge_detection():
    hm, kin = _step_map(), kinematics.RoverKinematics()
    far = _flat_state(0.0, 0.37)
    s = _flat_state(0.55, 0.37)
    s.requires_grad_(True)
    feat = terrain_map.relief_features(hm, kin, s)
    assert feat.shape == (1, terrain_map.RELIEF_DIM)
    feat.sum().backward()
    assert torch.isfinite(s.grad).all() and s.grad[0, 0].abs() > 0  # 对 x 有梯度
    assert not terrain_map.near_edge(hm, kin, far).item()
    # 前轮距台阶约 0.1 m 时应判为台阶边
    front_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 0, 0].item()
    edge = _flat_state(1.0 - front_x - 0.08, 0.37)
    assert terrain_map.near_edge(hm, kin, edge).item()


# ---------------- learner extra_fn ----------------
def test_learner_extra_fn_feeds_world_model_and_gradients():
    torch.manual_seed(0)
    hm, kin = _step_map(), kinematics.RoverKinematics()
    cfg = SuperMimicCfg(num_envs=4, buffer_len=30, wm_hidden=(16,), actor_hidden=(16,), critic_hidden=(16,),
                        wm_window=3, wm_batch=8, wm_updates=1, horizon=3, imag_batch=8, warmup_iterations=0)
    learner = SuperMimicLearner(cfg, "cpu", extra_fn=lambda s: terrain_map.relief_features(hm, kin, s),
                                extra_dim=terrain_map.RELIEF_DIM)
    assert learner.world_model.in_dim == state.FEATURE_DIM + 12 + terrain_map.RELIEF_DIM
    s = state.make_state(torch.rand(4, 3) + torch.tensor([0.3, 0.0, 0.3]), torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1),
                         torch.zeros(4, 3), torch.zeros(4, 3), torch.zeros(4, 12), torch.zeros(4, 12))
    for _ in range(20):
        learner.observe(s, torch.randn(4, 12), torch.zeros(4, 12), torch.rand(4, 3), torch.zeros(4, dtype=torch.bool))
        s = state.integrate(s, torch.randn(4, state.DELTA_DIM) * 0.01)
    logs = learner.update()
    assert math.isfinite(logs["wm_loss"]) and "actor_loss" in logs
    assert learner.skipped_actor_updates == 0


# ---------------- 越障任务 PreviewTask ----------------
import importlib.util  # noqa: E402

from supermimic import preview_task  # noqa: E402
from supermimic.networks import Actor  # noqa: E402

_VP = Path(__file__).resolve().parents[1] / (
    "robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/vision_preview.py")


def _vision_preview():
    # 纯 torch 文件，直接按路径加载，避开 robot_lab.tasks 对 Isaac Sim 的依赖。
    spec = importlib.util.spec_from_file_location("vision_preview_under_test", _VP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass 解析注解时要从 sys.modules 找到模块
    spec.loader.exec_module(mod)
    return mod


def _preview(**cfg):
    hm, kin = _step_map(step_x=1.5), kinematics.RoverKinematics()
    pattern = torch.tensor([[x, y, 20.0] for x in (0.0, 0.3, 0.6) for y in (-0.2, 0.2)])
    limits = torch.tensor([[-1.0, 1.0]] * 8)
    return preview_task.PreviewTask(hm, kin, pattern, limits, preview_task.PreviewRewardCfg(**cfg)), hm, kin


def test_centered_tracking_matches_env_implementation():
    vp = _vision_preview()
    cmd = torch.tensor([[0.3, 0.0], [0.05, 0.0], [0.4, 0.0]])
    vel = torch.tensor([[0.25, 0.02], [0.0, 0.0], [-0.1, 0.1]])
    ours = preview_task._centered_tracking(cmd, vel, 0.5)
    assert torch.allclose(ours, vp.centered_velocity_tracking(cmd, vel, 0.5), atol=1e-6)


def test_route_command_rotates_world_command_into_body_frame():
    s = _flat_state(0.0, 0.37, yaw=math.pi / 2)
    body = preview_task.route_command_body(s, torch.tensor([[0.3, 0.0, 0.0]]))
    assert torch.allclose(body, torch.tensor([[0.0, -0.3, 0.0]]), atol=1e-6)


def test_preview_obs_layout_and_scan_straight_through():
    task, hm, _ = _preview()
    s = _flat_state(0.5, 0.37)
    s.requires_grad_(True)
    obs = task.policy_obs(s, torch.tensor([[0.3, 0.0, 0.0]]), torch.zeros(1, 12))
    assert obs.shape == (1, 45 + 6)
    smooth = terrain_map.height_scan(hm, s, task.pattern)
    task.exact_scan_fn = lambda st: torch.full((st.shape[0], 6), 0.123)
    st_scan = task.scan(s)
    assert torch.allclose(st_scan, torch.full((1, 6), 0.123))  # 前向取真实射线
    g_smooth = torch.autograd.grad(smooth.sum(), s, retain_graph=True)[0]
    g_st = torch.autograd.grad(st_scan.sum(), s)[0]
    assert torch.allclose(g_smooth, g_st)  # 反向仍是双线性梯度


def test_wheel_lift_rewards_lifting_only_when_step_ahead():
    task, _, kin = _preview()
    front_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 0, 0].item()
    z_rest = 0.37
    flat = _flat_state(0.0, z_rest)
    near = _flat_state(1.5 - front_x - 0.10, z_rest)
    zero = torch.zeros(1, 12)
    cmd = torch.tensor([[0.3, 0.0, 0.0]])
    lifted = near.clone()
    lifted[:, state.JPOS][:, 0] = 0.0
    # 前轮抬起：直接平移整车 +0.08 m 模拟轮心抬升（只验证奖励随净空单调）
    lifted[:, 2] += 0.08
    r_near = task.reward_terms(near, near, cmd, zero, zero)
    r_lift = task.reward_terms(lifted, lifted, cmd, zero, zero)
    r_flat_lift = task.reward_terms(flat, flat.clone().index_add_(1, torch.tensor([2]), torch.tensor([[0.08]])),
                                    cmd, zero, zero)
    assert r_lift["wheel_lift"] > r_near["wheel_lift"] >= 0
    # 平地无台阶：has_step=sigmoid(-0.03/0.005)≈0.0025，抬轮塑形几乎为 0，抬轮反被扣分
    assert r_flat_lift["idle_lift"] < 0 and r_flat_lift["wheel_lift"].abs() < 0.01


def test_shaping_terms_values_and_finite_grad_at_upright():
    task, _, _ = _preview()
    assert task.shaping_terms(_flat_state(0.0, 0.37)) == {}  # 默认关闭，不改 v2 奖励
    task, _, _ = _preview(heading=-10.0, lateral_vel=-20.0, tilt_soft=-6.0)
    # 正立且航向 0：acos/atan2 在此处梯度为无穷，这里必须有限且三项为 0
    s = _flat_state(0.0, 0.37).requires_grad_(True)
    terms = task.shaping_terms(s)
    g = torch.autograd.grad(sum(v.sum() for v in terms.values()), s)[0]
    assert torch.isfinite(g).all() and all(v.abs().item() < 1e-6 for v in terms.values())
    yawed = _flat_state(0.0, 0.37, yaw=math.radians(13.0))
    yawed[:, state.LIN] = torch.tensor([[0.4, 0.0, 0.0]])  # 车体系前进 → 世界系有 vy
    t = task.shaping_terms(yawed)
    assert math.isclose(t["heading"].item(), -10.0 * math.sin(math.radians(13.0)) ** 2, rel_tol=1e-4)
    assert math.isclose(t["lateral_vel"].item(), -20.0 * (0.4 * math.sin(math.radians(13.0))) ** 2, rel_tol=1e-4)
    # 俯仰 40°：超过 25° 起罚点
    pitched = state.make_state(torch.tensor([[0.0, 0.0, 0.37]]),
                               quat.axis_angle_to_quat(torch.tensor([[0.0, math.radians(40.0), 0.0]])),
                               torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 12), torch.zeros(1, 12))
    expected = -6.0 * (math.cos(math.radians(25.0)) - math.cos(math.radians(40.0)))
    assert math.isclose(task.shaping_terms(pitched)["tilt_soft"].item(), expected, rel_tol=1e-4)
    assert torch.allclose(task.shaping_reward(pitched), sum(task.shaping_terms(pitched).values()) * 0.02)


def test_wheel_heading_penalizes_crab_not_aligned_steering():
    task, _, _ = _preview(wheel_heading=-10.0)
    yaw = math.radians(20.0)
    # 车头 +20°、四轮 −20°：轮子沿路线滚，不罚
    s = _flat_state(0.0, 0.37, yaw=yaw)
    jp = s[:, state.JPOS].clone(); jp[:, 4:8] = -yaw; s[:, state.JPOS] = jp
    assert task.shaping_terms(s)["wheel_heading"].abs().item() < 1e-6
    # 车头 +20°、四轮 −35°（诊断里的蟹行）：轮子朝 −15° 滚
    jp[:, 4:8] = math.radians(-35.0); s[:, state.JPOS] = jp
    expected = -10.0 * math.sin(math.radians(-15.0)) ** 2
    assert math.isclose(task.shaping_terms(s)["wheel_heading"].item(), expected, rel_tol=1e-4)
    s.requires_grad_(True)
    g = torch.autograd.grad(task.shaping_terms(s)["wheel_heading"].sum(), s)[0]
    assert torch.isfinite(g).all()


def test_lateral_pos_uses_nearest_lane_center_and_extends_critic():
    hm, kin = _step_map(step_x=1.5), kinematics.RoverKinematics()
    pattern = torch.tensor([[x, y, 20.0] for x in (0.0, 0.3, 0.6) for y in (-0.2, 0.2)])
    limits = torch.tensor([[-1.0, 1.0]] * 8)
    centers = torch.tensor([-8.0, 0.0, 8.0])
    task = preview_task.PreviewTask(hm, kin, pattern, limits, preview_task.PreviewRewardCfg(lateral_pos=-5.0),
                                    lane_center_y=centers)
    assert task.critic_extra_dim == terrain_map.RELIEF_DIM + 1
    s = state.make_state(torch.tensor([[0.0, -0.4, 0.37], [0.0, 8.3, 0.37], [0.0, -7.9, 0.37]]),
                         quat.axis_angle_to_quat(torch.zeros(3, 3)), torch.zeros(3, 3), torch.zeros(3, 3),
                         torch.zeros(3, 12), torch.zeros(3, 12))
    assert torch.allclose(task.lateral_offset(s), torch.tensor([-0.4, 0.3, 0.1]), atol=1e-5)
    assert torch.allclose(task.shaping_terms(s)["lateral_pos"], -5.0 * torch.tensor([0.16, 0.09, 0.01]), atol=1e-5)
    assert task.critic_extra(s).shape == (3, terrain_map.RELIEF_DIM + 1)
    # 未给车道中心时 critic 维度与 v2–v4 相同，旧 checkpoint 仍能续训
    plain, _, _ = _preview()
    assert plain.critic_extra_dim == terrain_map.RELIEF_DIM


def test_riser_anchor_weights_only_rear_wheels_at_riser():
    task, _, kin = _preview(riser_anchor=20.0)
    rear_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 1, 0].item()
    # 台阶前缘 x=1.5、高 0.12：后轮心距前缘 0.10 m → 加权；离得远 / 前轮贴立面 → 不加权
    at = _flat_state(1.5 - rear_x - 0.10, 0.37)
    far = _flat_state(0.0, 0.37)
    front_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 0, 0].item()
    front_only = _flat_state(1.5 - front_x - 0.10, 0.37)
    w = task.anchor_weight(torch.cat((at, far, front_only)))
    assert torch.allclose(w, torch.tensor([21.0, 1.0, 1.0]))
    plain, _, _ = _preview()
    assert plain.anchor_weight(at) is None


def test_preview_learner_runs_imagination_with_terrain_critic():
    torch.manual_seed(0)
    task, hm, kin = _preview()
    cfg = SuperMimicCfg(num_envs=4, buffer_len=30, wm_hidden=(16,), actor_hidden=(16,), critic_hidden=(16,),
                        wm_window=3, wm_batch=8, wm_updates=1, horizon=3, imag_batch=8, warmup_iterations=0,
                        action_squash="clamp")
    learner = SuperMimicLearner(cfg, "cpu", extra_fn=lambda s: terrain_map.relief_features(hm, kin, s),
                                extra_dim=terrain_map.RELIEF_DIM, task=task)
    assert learner.critic.in_dim == state.FEATURE_DIM + 3 + 12 + terrain_map.RELIEF_DIM
    s = state.make_state(torch.rand(4, 3) * 0.5 + torch.tensor([0.5, 0.0, 0.37]), torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1),
                         torch.zeros(4, 3), torch.zeros(4, 3), torch.zeros(4, 12), torch.zeros(4, 12))
    for _ in range(20):
        learner.observe(s, torch.randn(4, 12), torch.zeros(4, 12), torch.tensor([[0.3, 0.0, 0.0]]).repeat(4, 1),
                        torch.zeros(4, dtype=torch.bool))
        s = state.integrate(s, torch.randn(4, state.DELTA_DIM) * 0.01)
    logs = learner.update()
    assert "actor_loss" in logs and learner.skipped_actor_updates == 0


def test_clamp_actor_reproduces_rsl_rl_linear_output_within_bound():
    torch.manual_seed(0)
    actor = Actor(232, (32,), 6.2831853, squash="clamp")
    x = torch.randn(64, 232)
    raw = actor.actor(x) * 5.0
    with torch.no_grad():
        actor.actor[-1].weight.mul_(5.0)
        actor.actor[-1].bias.mul_(5.0)
    out = actor(x)
    inside = raw.abs() < 6.28
    assert torch.allclose(out[inside], raw[inside], atol=1e-5)
    assert out.abs().max() <= 6.2831853 + 1e-6
    torch.jit.script(actor)


def test_clamp_actor_state_dict_loads_into_rsl_rl_mlp_unchanged():
    # export_terrain_teacher 的前提：越障 actor 的 actor.* 键与 rsl_rl MLP（Linear/ELU 交替）逐键同形，
    # 界内输出逐位相同，所以转换只需替换权重
    torch.manual_seed(0)
    a = Actor(232, (512, 256, 128), 6.2831853, squash="clamp")
    layers, dims = [], (232, 512, 256, 128, 12)
    for i in range(4):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < 3:
            layers.append(torch.nn.ELU())
    rsl = torch.nn.Sequential(*layers)
    rsl.load_state_dict({k[len("actor."):]: v for k, v in a.state_dict().items()})
    x = torch.randn(64, 232) * 0.1
    assert torch.allclose(rsl(x).clamp(-6.2831853, 6.2831853), a(x))


def test_smooth_flat_penalizes_arm_motion_only_away_from_steps():
    task, _, kin = _preview(smooth_flat_arm=-0.5, smooth_flat_ang=-5.0)
    front_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 0, 0].item()
    flat = _flat_state(0.0, 0.37)
    near = _flat_state(1.5 - front_x - 0.10, 0.37)  # 前轮 0.10 m 外有 12 cm 台阶
    for st in (flat, near):
        jv = st[:, state.JVEL].clone(); jv[:, 0:4] = 1.0; st[:, state.JVEL] = jv
        an = st[:, state.ANG].clone(); an[:, 1] = 0.4; st[:, state.ANG] = an
    t_flat, t_near = task.shaping_terms(flat), task.shaping_terms(near)
    assert math.isclose(t_flat["smooth_flat_arm"].item(), -0.5 * 4.0, rel_tol=1e-2)
    assert math.isclose(t_flat["smooth_flat_ang"].item(), -5.0 * 0.16, rel_tol=1e-2)
    assert t_near["smooth_flat_arm"].abs().item() < 1e-3 and t_near["smooth_flat_ang"].abs().item() < 1e-3
    s = flat.clone().requires_grad_(True)
    g = torch.autograd.grad(sum(v.sum() for v in task.shaping_terms(s).values()), s)[0]
    assert torch.isfinite(g).all()


# ---------------- 方案 B：车道横向偏移观测 ----------------
from supermimic import lane_offset  # noqa: E402


def test_lane_obs_appends_scaled_offset_and_padded_actor_matches_teacher():
    hm, kin = _step_map(step_x=1.5), kinematics.RoverKinematics()
    pattern = torch.tensor([[x, y, 20.0] for x in (0.0, 0.3, 0.6) for y in (-0.2, 0.2)])
    task = preview_task.PreviewTask(hm, kin, pattern, torch.tensor([[-1.0, 1.0]] * 8), preview_task.PreviewRewardCfg(),
                                    lane_center_y=torch.tensor([-8.0, 0.0, 8.0]), lane_obs=True)
    assert task.obs_dim == task.base_obs_dim + 1
    s = state.make_state(torch.tensor([[0.0, -0.10, 0.37], [0.0, 8.5, 0.37]]),
                         quat.axis_angle_to_quat(torch.zeros(2, 3)), torch.zeros(2, 3), torch.zeros(2, 3),
                         torch.zeros(2, 12), torch.zeros(2, 12)).requires_grad_(True)
    cmd, prev = torch.tensor([[0.3, 0.0, 0.0]] * 2), torch.zeros(2, 12)
    obs = task.policy_obs(s, cmd, prev)
    # 与 env 侧 lane_offset_obs 同一公式：(y − 中心线) × 4，±4 截断（8.5 → +0.5 m → 2.0）
    assert torch.allclose(obs[:, -1], torch.tensor([-0.4, 2.0]), atol=1e-5)
    g = torch.autograd.grad(obs[:, -1].sum(), s)[0]
    assert torch.allclose(g[:, 1], torch.full((2,), lane_offset.LANE_OBS_SCALE))  # 梯度经车体 y 回传
    # 0 列补齐后的 actor 与原 Teacher 输出逐位相同，新输入不影响动作
    torch.manual_seed(0)
    teacher = Actor(task.base_obs_dim, (16,), 6.2831853, squash="clamp")
    padded = Actor(task.obs_dim, (16,), 6.2831853, squash="clamp")
    padded.load_state_dict(lane_offset.pad_first_layer(teacher.state_dict()))
    with torch.no_grad():
        assert torch.equal(padded(obs), teacher(obs[:, :-1]))
    with pytest.raises(ValueError):
        preview_task.PreviewTask(hm, kin, pattern, torch.tensor([[-1.0, 1.0]] * 8), lane_obs=True)


def test_lane_marker_round_trip(tmp_path):
    run = tmp_path / "run"
    (run / "exported").mkdir(parents=True)
    assert not lane_offset.policy_uses_lane_offset(run / "exported" / "policy.pt")
    lane_offset.write_marker(run, 232)
    assert lane_offset.policy_uses_lane_offset(run / "exported" / "policy.pt")
    assert lane_offset.policy_uses_lane_offset(run / "ckpt_001000.pt")


# ---------------- 方案 D：从零训练 ----------------
from supermimic.replay import ReplayBuffer  # noqa: E402


def test_replay_priority_sampling_prefers_flagged_rows():
    torch.manual_seed(0)
    rb = ReplayBuffer(50, 8, "cpu")
    for t in range(50):
        flag = (torch.rand(8) < 0.05).float()  # 约 5% 的行是关键事件
        rb.add(torch.zeros(8, state.STATE_DIM), torch.zeros(8, 12), torch.zeros(8, 12), torch.zeros(8, 3),
               torch.zeros(8, dtype=torch.bool), priority=flag)
    assert rb.sample_windows(400, 4, priority_fraction=0.5)["state"].shape[0] == 400

    def window_priority(fraction):
        start, env = rb._pick(2000, rb.size - 4, 4, fraction)
        rows = rb._physical(start[:, None] + torch.arange(4)[None, :])
        return rb.priority[rows, env[:, None].expand_as(rows)].amax(-1).mean().item()
    # 均匀时约 1−0.95⁴≈19% 的窗口含事件；优先那一半从 4 倍候选里取前 1/4，候选中含事件的约 19%，
    # 所以优先样本里约 0.19×4≈76% 含事件，整体约 (76%+19%)/2，即至少翻倍
    assert window_priority(0.5) > 2.0 * window_priority(0.0)
    assert rb.sample_states(64, priority_fraction=0.25)["state"].shape == (64, state.STATE_DIM)


def test_terminated_penalty_and_event_priority():
    task, _, kin = _preview(terminated=-20.0)
    upright, cmd, zero = _flat_state(0.0, 0.37), torch.tensor([[0.3, 0.0, 0.0]]), torch.zeros(1, 12)
    tipped = state.make_state(torch.tensor([[0.0, 0.0, 0.37]]), quat.axis_angle_to_quat(torch.tensor([[0.0, 1.5, 0.0]])),
                              torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 12), torch.zeros(1, 12))
    assert task.reward_terms(upright, upright, cmd, zero, zero)["terminated"].abs().item() < 0.1
    assert task.reward_terms(upright, tipped, cmd, zero, zero)["terminated"].item() < -19.0
    plain, _, _ = _preview()
    assert "terminated" not in plain.reward_terms(upright, upright, cmd, zero, zero)
    front_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 0, 0].item()
    near = _flat_state(1.5 - front_x - 0.10, 0.37)
    assert task.event_priority(torch.cat((upright, near))).tolist() == [0.0, 1.0]
