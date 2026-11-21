"""SuperMimic 地形扩展（可微 FK、高度图、想象高度扫描、地形特征、learner extra_fn）回归，不依赖仿真器。

@author pony
@date 2026-10-26
@version v0.5.0
@last_modified 2026-11-05
@changelog
    - (2026-11-05): _preview 关闭车道惩罚 lane（需要训练特权 lane_center_y），其余目标仍取默认。
    - (2026-11-05): 删除锚定、旧预抬、平地姿态/支撑、车道观测等旧奖励的测试；越障统一目标测试见 test_supermimic_obstacle.py。
    - (2026-11-03): 覆盖平地轮速目标正则的动作梯度、接近台阶和跨阶释放边界。
    - (2026-11-03): 验证想象平地比例、上下文索引一致性及缺少越障状态时的回退。
    - (2026-11-02): 平地动作锚定须允许完整协调修改；路线critic须能区分相同本体状态的不同航向。
    - v0.4.0 (2026-11-02): 覆盖承重轮接触压入时预抬梯度与旧奖励兼容。
    - v0.3.0 (2026-10-30): 课程到顶随机回落、课程饱和判据
    - v0.2.0 (2026-10-29): 续训课程等级恢复与横向惩罚爬坡
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


def test_preview_scan_refresh_survives_float32_clock_boundary():
    """执行真实配置入口的扫描部分，并复现SensorBase的时钟比较，覆盖16秒后的漏更新。"""
    import ast
    from types import SimpleNamespace

    path = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/supermimic/terrain_env_cfg.py"
    config_class = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))

    class Base(SimpleNamespace):
        def __post_init__(self):
            pass

    friction = next(ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "WHEEL_FRICTION")
    scanner = SimpleNamespace(update_period=0.02, offset=SimpleNamespace(pos=(0., 0., 20.)))
    scope = {"configclass": lambda cls: cls, "ParallelogramRoverVisionTeacherPreviewEnvCfg": Base,
             "mdp_core": SimpleNamespace(bad_orientation=object()), "TILT_LIMIT_RAD": 1.2,
             "TerminationTermCfg": lambda **kwargs: kwargs, "WHEEL_FRICTION": friction}
    exec(compile(ast.Module(body=[config_class], type_ignores=[]), str(path), "exec"), scope)
    material = SimpleNamespace(params={"static_friction_range": (0.5, 1.2), "dynamic_friction_range": (0.4, 1.0)})
    events = SimpleNamespace(randomize_wheel_rigid_body_material=material)
    cfg = scope[config_class.name](scene=SimpleNamespace(height_scanner=scanner), events=events,
                                   observations=SimpleNamespace(policy=SimpleNamespace(height_scan=SimpleNamespace())),
                                   terminations=SimpleNamespace())
    cfg.__post_init__()

    def missed_control_samples(period):
        timestamp, last = torch.zeros(1), torch.zeros(1)
        missed = 0
        for _ in range(1200):
            outdated = False
            for _ in range(4):
                timestamp += 0.005
                outdated |= bool(timestamp - last + 1e-6 >= period)
            if outdated:
                last.copy_(timestamp)
            else:
                missed += 1
        return missed

    assert missed_control_samples(0.02) > 0  # 原更新周期确实在边界后失效。
    assert missed_control_samples(cfg.scene.height_scanner.update_period) == 0
    # 轮胎摩擦固定为 μ=1.0（用户 2026-11-07 决定）：训练与平地验收不再随机
    assert friction == (1.2, 1.0)
    assert material.params["static_friction_range"] == (1.2, 1.2)
    assert material.params["dynamic_friction_range"] == (1.0, 1.0)


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
    # 这些测试不提供车道中心：关掉车道惩罚（它需要训练特权 lane_center_y），其余目标保持默认
    return preview_task.PreviewTask(hm, kin, pattern, limits, preview_task.PreviewRewardCfg(**{"lane": 0.0, **cfg})), hm, kin


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


def test_route_critic_features_distinguish_heading_while_legacy_layout_is_preserved():
    task, hm, kin = _preview(heading=-10.0)
    straight = _flat_state(-2.0, 0.37)
    turned = _flat_state(-2.0, 0.37, yaw=math.pi / 2)
    states = torch.cat((straight, turned))
    # 旧critic完全不看航向，但世界路线奖励随航向变化，不能继续把它们映成同一输入。
    assert torch.equal(state.features(states)[0], state.features(states)[1])
    assert torch.allclose(task.critic_extra(states)[0], task.critic_extra(states)[1], atol=1e-6)
    updated = preview_task.PreviewTask(hm, kin, task.pattern, task.limits, task.cfg, critic_route_features=True)
    extra = updated.critic_extra(states)
    assert updated.critic_extra_dim == task.critic_extra_dim + 2
    assert torch.allclose(extra[:, :-2], task.critic_extra(states), atol=1e-6)
    assert torch.allclose(extra[:, -2:], torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6)
    states.requires_grad_(True)
    gradient = torch.autograd.grad(updated.critic_extra(states)[:, -1].sum(), states)[0]
    assert torch.isfinite(gradient).all() and gradient[:, state.QUAT].abs().sum() > 0


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


def test_balanced_imagination_keeps_requested_flat_fraction_and_matching_context():
    torch.manual_seed(42)
    task, _, kin = _preview()
    cfg = SuperMimicCfg(num_envs=2, buffer_len=2, wm_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,),
                        imag_batch=20, imag_flat_fraction=0.4, context_dim=1)
    learner = SuperMimicLearner(cfg, "cpu", task=task)
    front_x = kin.wheel_centers_base(torch.zeros(1, 4), torch.zeros(1, 4))[0, 0, 0]
    states = torch.cat((_flat_state(0.0, 0.37), _flat_state(1.5 - front_x - 0.20, 0.37)))
    context = torch.tensor([[1.0], [2.0]])
    learner.observe(states, torch.zeros(2, 12), torch.zeros(2, 12), torch.zeros(2, 3),
                    torch.zeros(2, dtype=torch.bool), context)
    batch = learner._sample_imagination_start()
    flat = task.imagination_flat_mask(batch["state"])
    assert flat.sum().item() == 8
    assert batch["state"].shape == (20, 37)
    assert torch.equal(batch["context"][:, 0], torch.where(flat, 1.0, 2.0))


def test_balanced_imagination_uses_available_states_when_obstacles_are_absent():
    task, _, _ = _preview()
    cfg = SuperMimicCfg(num_envs=1, buffer_len=2, wm_hidden=(8,), actor_hidden=(8,), critic_hidden=(8,),
                        imag_batch=20, imag_flat_fraction=0.4)
    learner = SuperMimicLearner(cfg, "cpu", task=task)
    learner.observe(_flat_state(0.0, 0.37), torch.zeros(1, 12), torch.zeros(1, 12), torch.zeros(1, 3),
                    torch.zeros(1, dtype=torch.bool))
    batch = learner._sample_imagination_start()
    assert batch["state"].shape == (20, 37)
    assert task.imagination_flat_mask(batch["state"]).all()


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


# ---------------- 方案 B：车道横向偏移观测 ----------------
from supermimic import lane_offset  # noqa: E402


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


def test_scratch_lane_gates_follow_cfg():
    # scratch_env_cfg 依赖 Isaac 配置类；这里只核对门限读取与出车道判据的纯函数部分
    spec = importlib.util.spec_from_file_location(
        "scratch_under_test", Path(__file__).resolve().parents[1] / "robot_lab/tasks/manager_based/supermimic/scratch_env_cfg.py")
    src = spec.loader.get_source(spec.name)
    assert "scratch_pass_lane_m" in src and "def scratch_out_of_lane" in src
    ns = {"torch": torch, "getattr": getattr}
    exec(compile("TRAIN_LANE_M = 0.5\nFAIL_LANE_M = 1.0\n" + src[src.index("def _lanes"):src.index("def scratch_step_height")],
                 "lanes", "exec"), ns)
    d1 = type("E", (), {"cfg": type("C", (), {})()})()
    d2 = type("E", (), {"cfg": type("C", (), {"scratch_pass_lane_m": 0.15, "scratch_fail_lane_m": 0.75})()})()
    assert ns["_lanes"](d1) == (0.5, 1.0) and ns["_lanes"](d2) == (0.15, 0.75)


from supermimic.schedule import linear_ramp, next_terrain_levels, restore_terrain_levels, saturation_reached  # noqa: E402


def test_linear_ramp_clamps_and_interpolates():
    assert linear_ramp(4.0, 5.0, 10.0, -5.0, -15.0) == -5.0
    assert linear_ramp(7.5, 5.0, 10.0, -5.0, -15.0) == pytest.approx(-10.0)
    assert linear_ramp(12.0, 5.0, 10.0, -5.0, -15.0) == -15.0
    assert linear_ramp(5.0, 5.0, 5.0, -5.0, -15.0) == -15.0  # 区间退化时到起点即取终值


def test_restore_terrain_levels_keeps_progress():
    saved = torch.tensor([0, 2, 3, 8, 5])
    # 环境数相同：逐个还原，越界的裁到最高级
    assert restore_terrain_levels(saved, 5, max_level=6).tolist() == [0, 2, 3, 5, 5]
    # 环境数不同：从存档分布抽样，不回到第 0 级
    big = restore_terrain_levels(torch.full((64,), 4), 512, max_level=9, generator=torch.Generator().manual_seed(0))
    assert big.shape == (512,) and bool((big == 4).all())


def test_next_terrain_levels_resamples_top_graduates():
    torch.manual_seed(0)
    levels = torch.tensor([0, 3, 8, 8, 8, 8, 8, 8, 8, 8])
    passed = torch.tensor([False, True] + [True] * 8)
    failed = torch.tensor([True, False] + [False] * 8)
    out = next_terrain_levels(levels, passed, failed, max_level=9)
    assert out[0] == 0 and out[1] == 4                           # 降级不低于 0、普通升级 +1
    assert bool(((out[2:] >= 0) & (out[2:] < 9)).all()) and len(set(out[2:].tolist())) > 1  # 通关最高级随机回落
    kept = next_terrain_levels(levels, passed, failed, max_level=9, top_resample=False)
    assert kept[2:].tolist() == [8] * 8                          # 旧行为：夹在最高级


def test_saturation_reached_accumulates():
    ever = torch.zeros(4, dtype=torch.bool)
    ever, hit = saturation_reached(ever, torch.tensor([8, 0, 0, 0]), max_level=9, threshold=0.5)
    assert not hit
    ever, hit = saturation_reached(ever, torch.tensor([2, 8, 0, 0]), max_level=9, threshold=0.5)
    assert hit and ever.tolist() == [True, True, False, False]  # 第 0 个已回落，仍计入
