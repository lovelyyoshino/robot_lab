"""SuperMimic 越障统一目标、预抬物理边界（三轮支撑/错开/轻碰/准备行程）、按类型课程与地形配置的回归测试，不依赖仿真器。

@author pony
@date 2026-11-05
@version v1.15.0
@last_modified 2026-11-20
@changelog
    - v1.15.0 (2026-11-20): launcher 断言第二十七轮楼梯车道代价早开、晋级横偏门不收紧：
      --stairs_lane_schedule 2000 8000 --stairs_success_lane_schedule 90000 100000。
    - v1.14.0 (2026-11-19): launcher 断言第二十六轮登顶后回正窗口 --post_climb_relief 1.0 --post_climb_window_m 1.0。
    - v1.13.0 (2026-11-19): launcher 断言回退到第二十二轮配置：--stall -5.0、--progress 1.0、--stairs_slant_deg 6.0、
      --prelift_front 25.0、--stall_linear -3.0 --stall_linear_cap 1.0、--stairs_lane_schedule 90000 100000（不再有
      晋级横偏门 schedule）、去掉 --leg_action_bound。
    - v1.12.0 (2026-11-19): tilt_temperature 也是 CLI 参数、launcher 显式写出；第二十五轮翻车代价 --terminated -10000.0。
    - v1.11.0 (2026-11-18): launcher 断言跟随第二十四轮：晋级横偏门不收紧（--stairs_success_lane_schedule 90000 100000）、
      --stall 0.0、--stall_linear -20.0 --stall_linear_cap 5.0、--progress 12.0、--prelift_front 10.0、--leg_action_bound。
    - v1.10.0 (2026-11-17): launcher 断言跟随第二十三轮：楼梯不斜行 --stairs_slant_deg 0.0、车道代价 --stairs_lane_schedule 2000 8000、
      晋级横偏门 --stairs_success_lane_schedule 16000 26000（旧断言靠 changelog 注释里的同名字符串才通过）。
    - v1.9.0 (2026-11-13): 楼梯车道约束先放松后收紧：stairs_success_lane_m（课程与终点终止按地形类型）、lane_schedule_scale、
      楼梯区车道代价系数 set_stairs_lane_scale 的测试。
    - v1.8.0 (2026-11-13): 贴立面停滞折扣只给真离地的轮 stall_climb_min_clearance（复刻第十六轮 it18000 左后轮顶箱姿态的
      抬轮单调性）、贴立面着地轮空转 riser_spin、楼梯课程迟滞 2/3 的测试。
    - v1.7.0 (2026-11-12): 楼梯同轴两轮都在翻越时误差相加 stairs_climb_pair_sum（复刻第十五轮 it26000 楼梯死锁姿态）、
      停滞线性代价 stall_linear、楼梯课程迟滞 3/3 的测试。
    - v1.6.0 (2026-11-09): 先移质心/支撑净空门控只对单台阶（楼梯前轴回到窗口内、门槛 0.5 cm、两者平均）；楼梯降级比例
      stairs_demote_ratio、停滞截断 stall_timeout 与 launcher 新参数的测试。
    - v1.5.0 (2026-11-08): 平地转向平方 flat_steer、四轮同速 wheel_sync、toe/打滑新权重；该抬的轮从立面前 0.25 m 起裕度门槛
      0.8 cm；接触前支撑净空按裕度门控（质心压在对角线上时要求实际净空也达标）的测试。
    - v1.4.0 (2026-11-08): 整宽单台阶斜行、前轴领先轮预抬 prelift_front、平地 toe/打滑、零速偏航 Huber、launcher 新参数的测试。
    - v1.3.0 (2026-11-06): 支撑裕度计入预抬窗口内该抬的轮（先移质心可清零）、翻越中的轮不算悬空、净空跟踪 5.0 的测试。
    - v1.2.0 (2026-11-06): 删除立面接近代价的测试；新增轻碰、三轮支撑裕度、两轮离地、每根轴一个轮预抬、
      准备行程、横滚超限、预抬只在单台阶/翻越对所有立面、减速带 min(cmd, 0.12) 的测试。
    - v1.1.0 (2026-11-05): 车道惩罚/横向 Huber、成功与晋级横偏门、楼梯无碰撞门、命令速度项、减速带离地门、
      预抬窗口 0.14 m、θ_ref 扣除 6°、命令与回合配置的测试。
    - v1.0.0 (2026-11-05): 初始创建
"""

from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PKG = Path(__file__).resolve().parents[1] / "robot_lab"
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG))

from supermimic import kinematics, quat, state  # noqa: E402
from supermimic.preview_task import WHEEL_RADIUS, PreviewRewardCfg, PreviewTask, first_crossing  # noqa: E402
from supermimic.terrain_map import RELIEF_DIM, HeightMap  # noqa: E402

R = WHEEL_RADIUS
KIN = kinematics.RoverKinematics()
NEUTRAL = KIN.wheel_centers_base(torch.zeros(4), torch.zeros(4))
FRONT_X, REAR_X, WHEEL_Z = NEUTRAL[0, 0].item(), NEUTRAL[1, 0].item(), NEUTRAL[0, 2].item()
GROUND_Z = -WHEEL_Z + R  # 四轮轮底刚好贴地的车体高度
CMD = torch.tensor([[0.3, 0.0, 0.0]])
ZERO = torch.zeros(1, 12)
LANES = torch.tensor([-8.0, 0.0, 8.0])  # 车道中心（训练特权）；测试状态默认 y=0 在车道中心


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, PKG / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ot = _load("obstacle_terrain_under_test", "tasks/manager_based/supermimic/obstacle_terrain.py")


def _map(profile, res=0.02, x_max=4.0):
    xs = torch.arange(0.0, x_max + 1e-9, res)
    ys = torch.arange(-1.0, 1.0 + 1e-9, res)
    heights = torch.tensor([profile(float(x)) for x in xs]).float()[:, None].expand(len(xs), len(ys))
    return HeightMap(heights.clone(), (0.0, -1.0), res)


PLATFORM = lambda x: 0.12 if x >= 1.5 else 0.0  # noqa: E731
SINGLE = lambda x: 0.12 if 1.5 <= x <= 1.9 else 0.0  # noqa: E731
STAIRS = lambda x: 0.12 * min(max(math.floor((x - 1.5) / 0.30) + 1, 0), 6)  # noqa: E731


def _box_riser_fn(edge=1.5, height=0.12, calls=None):
    """精确网格射线的解析替身：x ≥ edge 处的实体台阶，只命中朝 −x 的立面。"""
    def fn(starts, directions):
        if calls is not None:
            calls.append(starts.shape)
        hit = (directions[..., 0] > 0.5) & (starts[..., 0] < edge) & (starts[..., 2] < height)
        distance = (edge - starts[..., 0]) / directions[..., 0].clamp_min(1e-6)
        return torch.where(hit & (distance <= 0.75), distance, torch.inf)
    return fn


def _task(profile=PLATFORM, riser_fn=None, **cfg):
    limits = torch.tensor([[-0.58, 0.58]] * 4 + [[-1.3, 1.3]] * 4)
    pattern = torch.tensor([[x, y, 20.0] for x in (0.0, 0.3) for y in (-0.2, 0.2)])
    return PreviewTask(_map(profile), KIN, pattern, limits, PreviewRewardCfg(**cfg), exact_riser_fn=riser_fn,
                       lane_center_y=LANES)


def _state(x, z=GROUND_Z, vx=0.0, pitch=0.0, roll=0.0, yaw=0.0, arm=(0.0, 0.0, 0.0, 0.0), y=0.0, vy=0.0):
    rotation = quat.quat_mul(quat.axis_angle_to_quat(torch.tensor([[0.0, 0.0, yaw]])),
                             quat.quat_mul(quat.axis_angle_to_quat(torch.tensor([[0.0, -pitch, 0.0]])),
                                           quat.axis_angle_to_quat(torch.tensor([[roll, 0.0, 0.0]]))))
    joints = torch.zeros(1, 12)
    joints[0, :4] = torch.tensor(arm)
    return state.make_state(torch.tensor([[x, y, z]]), rotation, torch.tensor([[vx, vy, 0.0]]),
                            torch.zeros(1, 3), joints, torch.zeros(1, 12))


def _terms(task, s, cmd=CMD):
    return task.reward_terms(s, s, cmd, ZERO, ZERO)


# ---------------- 配置：旧路径已删除，每个权重都是 CLI 参数 ----------------
def test_reward_cfg_has_only_current_terms_and_every_weight_is_a_cli_flag():
    names = {f.name for f in fields(PreviewRewardCfg)}
    for legacy in ("riser_anchor", "lateral_pos", "idle_lift", "timed_prelift", "wheel_lift", "flat_posture",
                   "obstacle_anchor", "prelift_objective_version", "smooth_flat_arm", "riser_collision",
                   "riser_cmd_speed", "riser_margin", "riser_depth_scale", "slow_band_ratio", "slow_band_clearance",
                   "stagger", "stagger_m", "stagger_zone_m"):
        assert legacy not in names
    tree = ast.parse((REPO / "scripts/supermimic/train_terrain.py").read_text())
    flags = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and getattr(node.targets[0], "id", "") == "REWARD_FLAGS")
    assert {pair.elts[0].value for pair in flags.elts} == names
    launcher = (REPO / "scripts/supermimic/run_train.sh").read_text()
    for name in names:
        assert re.search(rf"--{name} ", launcher), name
    for removed in ("--lane_obs", "--teacher", "anchor", "motion_reference", "prelift_exploration",
                    "flat_calibration", "--init_from"):
        assert removed not in launcher
    assert "--max_iterations" in launcher and "setsid" in launcher and ".pid" in launcher
    # 课程参数：每个 CLI 名字都有默认值，launcher 显式写出新增的横偏/碰撞门/命令/回合参数
    curriculum = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                      and getattr(node.targets[0], "id", "") == "CURRICULUM_FLAGS")
    names = {element.value for element in curriculum.elts}
    assert names <= set(ot.CURRICULUM_DEFAULTS) and {"success_lane_m", "stairs_collision_frac", "collision_frac"} <= names
    for needle in ("--curriculum_success_lane_m 0.15", "--curriculum_fail_lane_m 0.40", "--curriculum_collision_frac 0.10",
                   "--curriculum_stairs_collision_frac 1.0", "--episode_length_s 40", "--command_lin_vel_x 0.10 0.40",
                   "--command_standing_fraction 0.10", "--lane -30.0", "--lateral_vel -15.0", "--lateral_vel_delta 0.02",
                   "--heading_huber -5.0", "--heading_huber_delta 0.02", "--flat_neutral -5.0",
                   "--prelift_track 5.0", "--prelift_window 0.12", "--early_lift -2.0", "--early_lift_lead 0.12",
                   "--slow_band_speed 0.12", "--pitch_ref_deadband_deg 6.0", "--support_margin -10.0", "--support_margin_m 0.005",
                   "--riser_safe_speed 0.15", "--roll_excess -2000.0", "--stall -5.0 ", "--progress 1.0 ", "--yaw_rate -2.0",
                   "--stairs_slant_deg 6.0 ", "--stairs_slant_zone_m 0.60",
                   "--single_slant_deg 6.0", "--obstacle_lateral_vel_scale 3.0", "--prelift_front 25.0\n", "--toe -20.0", "--wheel_slip -20.0",
                   "--flat_steer -10.0", "--wheel_sync -50.0", "--wheel_sync_delta 0.05",
                   "--prelift_margin_m 0.008", "--prelift_shift_zone 0.25",
                   "--stand_yaw_rate -5.0", "--action_rate -0.02", "FROM_RUN",
                   "--curriculum_stairs_demote_ratio 0.8", "--curriculum_stall_truncate_s 12",
                   "--mirror_loss_coef 1.0", "--action_saturation_coef 0.02", "--action_saturation_margin 0.10",
                   "--imag_flat_fraction 0.35",
                   "--stall_climb_scale 0.2", "--drop_relief 0.8", "--post_climb_relief 1.0 --post_climb_window_m 1.0\n", "--riser_contact_speed -400.0",
                   "--riser_approach_m 0.15", "--curriculum_promote_streak 2", "--curriculum_demote_streak 2",
                   "--stairs_tread_rule 0.12 0.15 0.36", "--actor_ema_decay 0.999",
                   "--stairs_stall_climb_scale 1.0", "--stairs_riser_contact_scale 0.5", "--stairs_riser_approach_m 0.05",
                   "--curriculum_stairs_promote_streak 2", "--curriculum_stairs_demote_streak 3",
                   "--stall_climb_min_clearance 0.01", "--stall_climb_lead_m 0.05", "--riser_spin -10.0",
                   "--trailing_contact_climb 1.0", "--stairs_climb_pair_sum 1.0", "--stall_linear -3.0 --stall_linear_cap 1.0\n",
                   "--critic_lane", "--stairs_lane_schedule 2000 8000 --stairs_success_lane_schedule 90000 100000\n",
                   "TRAIN_ARGS"):
        assert needle in launcher, needle


def test_default_weights_match_recipe():
    cfg = PreviewRewardCfg()
    assert (cfg.track_lin_vel, cfg.terminated, cfg.tilt_limit, cfg.heading) == (5.0, -20.0, 0.8, -30.0)
    assert (cfg.lateral_vel, cfg.lateral_vel_delta, cfg.lane) == (-15.0, 0.02, -30.0)
    assert (cfg.heading_huber, cfg.heading_huber_delta, cfg.flat_neutral) == (-5.0, 0.02, -5.0)
    assert (cfg.ang_vel, cfg.arm_vel, cfg.action_rate) == (-2.0, -0.5, -0.005)
    assert cfg.pitch_tol_deg == 3.0 and cfg.pitch_ref_deadband_deg == 6.0
    assert (cfg.slow_band_m, cfg.slow_band_speed) == (0.30, 0.12)
    assert (cfg.prelift_track, cfg.prelift_window, cfg.early_lift, cfg.early_lift_lead) == (5.0, 0.12, -2.0, 0.12)
    assert (cfg.riser_contact_speed, cfg.riser_safe_speed, cfg.riser_approach_m) == (-200.0, 0.15, 0.05)
    assert (cfg.support_margin, cfg.support_margin_m, cfg.multi_lift) == (-10.0, 0.005, -5.0)
    assert (cfg.ready_stroke, cfg.ready_zone_m) == (-1.0, 0.30)
    assert (cfg.roll_excess, cfg.roll_excess_deg) == (-2000.0, 4.5)
    assert (cfg.stall, cfg.yaw_rate, cfg.stairs_slant_deg, cfg.stairs_slant_zone_m) == (-5.0, -2.0, 12.0, 0.60)
    assert (cfg.single_slant_deg, cfg.prelift_front) == (12.0, 15.0)
    assert (cfg.toe, cfg.wheel_slip, cfg.wheel_slip_delta) == (-20.0, -20.0, 0.05)
    assert (cfg.flat_steer, cfg.wheel_sync, cfg.wheel_sync_delta) == (-10.0, -50.0, 0.05)
    assert (cfg.prelift_margin_m, cfg.prelift_shift_zone) == (0.008, 0.25)
    assert (cfg.stand_yaw_rate, cfg.stand_yaw_rate_delta) == (-5.0, 0.02)


# ---------------- 5 车道 / 横向速度 ----------------
def test_lane_penalty_uses_privileged_offset_and_keeps_actor_input():
    task = _task(lambda x: 0.0)
    for y in (0.0, 0.1, -0.2):
        assert _terms(task, _state(1.0, y=y))["lane"].item() == pytest.approx(-30.0 * y * y, abs=1e-6)
    assert _terms(task, _state(1.0, y=7.9))["lane"].item() == pytest.approx(-30.0 * 0.01, rel=1e-3)  # 最近车道
    s = _state(1.0, y=0.1).requires_grad_(True)
    grad = torch.autograd.grad(_terms(task, s)["lane"].sum(), s)[0]
    assert grad[0, 1].item() == pytest.approx(-60.0 * 0.1, rel=1e-4)  # 梯度把车推回车道中心
    assert task.policy_obs(s, CMD, ZERO).shape[-1] == task.obs_dim == 45 + 4  # actor 输入不含车道偏移
    no_critic = PreviewTask(_map(lambda x: 0.0), KIN, torch.zeros(4, 3), torch.zeros(8, 2), lane_center_y=LANES,
                            critic_lane=False)
    assert no_critic.critic_extra_dim == RELIEF_DIM and "lane" in _terms(no_critic, _state(1.0, y=0.1))
    with pytest.raises(ValueError):
        PreviewTask(_map(lambda x: 0.0), KIN, torch.zeros(4, 3), torch.zeros(8, 2))  # lane≠0 却没有车道中心
    off = PreviewTask(_map(lambda x: 0.0), KIN, torch.zeros(4, 3), torch.zeros(8, 2), PreviewRewardCfg(lane=0.0))
    assert "lane" not in _terms(off, _state(1.0)) and off.critic_extra_dim == RELIEF_DIM


def test_lateral_velocity_penalty_is_huber():
    task = _task(lambda x: 0.0)
    delta, w = task.cfg.lateral_vel_delta, task.cfg.lateral_vel

    def cost(vy):
        return _terms(task, _state(1.0, vy=vy))["lateral_vel"].item()

    assert w == -15.0
    assert cost(0.01) == pytest.approx(w * 0.5 * 0.01**2 / delta, rel=1e-4)  # |v|<δ：二次
    assert cost(0.10) == pytest.approx(w * (0.10 - 0.5 * delta), rel=1e-4)  # |v|≥δ：线性
    assert cost(-0.10) == pytest.approx(cost(0.10), rel=1e-6)
    assert cost(delta - 1e-6) == pytest.approx(cost(delta + 1e-6), abs=5e-5)  # δ 处连续
    assert cost(0.05) < -20.0 * 0.05**2  # 小漂移比旧的 −20·v² 罚得重


# ---------------- 3 姿态：θ_ref 来自前后轮下地形 ----------------
def test_pitch_reference_follows_stair_slope_minus_deadband_and_is_zero_on_flat():
    task = _task(STAIRS)
    slope = math.atan2(0.12, 0.30)
    s = _state(2.6, z=GROUND_Z + 0.36, pitch=slope)
    ctx = task.terrain_context(s)
    geo, ref = ctx["pitch_geo"].item(), ctx["pitch_ref"].item()
    assert abs(geo - slope) < math.radians(5.0)
    assert ref == pytest.approx(geo - math.radians(6.0), abs=1e-6)  # θ_ref = θ_geo − 6°
    assert not ctx["pitch_ref"].requires_grad
    assert abs(_task(lambda x: 0.0).terrain_context(_state(1.0))["pitch_ref"].item()) < 1e-6
    no_deadband = _task(STAIRS, pitch_ref_deadband_deg=0.0).terrain_context(s)
    assert no_deadband["pitch_ref"].item() == pytest.approx(geo, abs=1e-6)


def test_pitch_reference_deadband_is_symmetric_and_clips_small_slopes():
    # 12 cm 单障碍：前轮在顶面、后轮在地面时 θ_geo≈atan(0.12/0.694)≈9.8°，θ_ref≈3.8°（"单台阶俯仰 ≤8°"门内）
    single = _task(PLATFORM)
    s = _state(1.5 + 0.2 - FRONT_X, z=GROUND_Z + 0.06, pitch=math.radians(9.8))
    ctx = single.terrain_context(s)
    geo = ctx["pitch_geo"].item()
    assert math.radians(8.0) < geo < math.radians(11.0)
    assert ctx["pitch_ref"].item() == pytest.approx(geo - math.radians(6.0), abs=1e-6)
    low = _task(lambda x: 0.04 if x >= 1.5 else 0.0)  # 4 cm：θ_geo≈3.3° < 6° → θ_ref = 0
    assert low.terrain_context(_state(1.5 + 0.2 - FRONT_X, z=GROUND_Z + 0.02))["pitch_ref"].item() == 0.0
    down = _task(lambda x: 0.12 if x < 1.5 else 0.0)  # 下台阶：θ_geo < 0，θ_ref 对称地取负
    ctx = down.terrain_context(_state(1.5 + 0.2 - FRONT_X, z=GROUND_Z + 0.06))
    assert ctx["pitch_geo"].item() < -math.radians(8.0)
    assert ctx["pitch_ref"].item() == pytest.approx(ctx["pitch_geo"].item() + math.radians(6.0), abs=1e-6)


def test_pitch_penalty_has_tolerance_band_and_finite_gradient():
    flat = _task(lambda x: 0.0)
    assert _terms(flat, _state(1.0, pitch=math.radians(2.0)))["pitch"].item() == pytest.approx(0.0, abs=1e-7)
    tilted = _terms(flat, _state(1.0, pitch=math.radians(6.0)))["pitch"].item()
    assert tilted == pytest.approx(-50.0 * math.radians(3.0) ** 2, rel=0.02)
    stairs = _task(STAIRS)
    slope = math.atan2(0.12, 0.30)
    ref = stairs.terrain_context(_state(2.6, z=GROUND_Z + 0.36, pitch=slope))["pitch_ref"].item()
    on_ref = _terms(stairs, _state(2.6, z=GROUND_Z + 0.36, pitch=ref))["pitch"].item()
    level = _terms(stairs, _state(2.6, z=GROUND_Z + 0.36, pitch=0.0))["pitch"].item()
    assert on_ref > -0.01 and level < -0.5  # 楼梯上贴参考俯仰不罚，强行水平反而罚
    s = _state(1.0, pitch=math.radians(10.0), roll=math.radians(5.0)).requires_grad_(True)
    terms = _terms(flat, s)
    assert terms["roll"].item() == pytest.approx(-50.0 * math.sin(math.radians(5.0)) ** 2, rel=0.05)
    grad = torch.autograd.grad(terms["pitch"].sum() + terms["roll"].sum(), s)[0]
    assert torch.isfinite(grad).all() and grad[:, state.QUAT].abs().sum() > 0


# ---------------- 7 越障：轻碰（四轮同一公式），几何每状态只算一次 ----------------
def _front_gap_state(gap, vx=0.3, **kw):
    return _state(1.5 - R - gap - FRONT_X, vx=vx, **kw)


def test_contact_speed_cost_matches_formula_and_spares_slow_contact():
    task = _task(PLATFORM, _box_riser_fn())
    cfg = task.cfg
    closeness = (cfg.riser_approach_m - 0.02) / cfg.riser_approach_m
    cost = _terms(task, _front_gap_state(0.02))["riser_contact_speed"].item()
    expected = cfg.riser_contact_speed * 2 * closeness * (0.3 - cfg.riser_safe_speed) ** 2
    assert cost == pytest.approx(expected, rel=0.02) and cost < 0.0
    assert _terms(task, _front_gap_state(0.02, vx=0.10))["riser_contact_speed"].item() == 0.0  # 低于安全速度
    assert _terms(task, _front_gap_state(cfg.riser_approach_m + 0.02))["riser_contact_speed"].item() == 0.0
    assert "riser_collision" not in _terms(task, _front_gap_state(0.02))


def test_contact_speed_cost_applies_to_rear_wheels_and_vanishes_when_lifted():
    task = _task(PLATFORM, _box_riser_fn())
    lifted = KIN.wheel_centers_base(torch.tensor([-0.61, 0.0, -0.61, 0.0]), torch.zeros(4))
    rear = _state(1.5 - R - 0.02 - REAR_X, vx=0.3, arm=(-0.61, 0.0, -0.61, 0.0))
    ctx = task.terrain_context(rear)
    assert lifted[0, 2] - WHEEL_Z > 0.11 and not ctx["face_valid"][0, [0, 2]].any()
    assert ctx["face_valid"][0, [1, 3]].all()
    front = _terms(task, _front_gap_state(0.02))["riser_contact_speed"].item()
    assert _terms(task, rear)["riser_contact_speed"].item() == pytest.approx(front, rel=1e-3)
    assert task.riser_contacts(_state(1.5 - R - REAR_X, arm=(-0.61, 0.0, -0.61, 0.0)))[0].tolist() == [False, True, False, True]
    assert _terms(_task(PLATFORM), _front_gap_state(0.02, z=GROUND_Z + 0.10))["riser_contact_speed"].item() == 0.0


# ---------------- 7 越障：三轮支撑 / 两轮离地 ----------------
def test_com_model_and_rim_clearance():
    com = KIN.com_base(NEUTRAL)
    assert abs(com[0].item()) < 0.01 and abs(com[1].item()) < 0.01 and KIN.total_mass.item() == pytest.approx(10.6, abs=0.3)
    task = _task(PLATFORM)
    ctx = task.terrain_context(_state(0.5))
    assert ctx["rim_clear"].abs().max().item() < 0.003 and ctx["lifted"].max().item() < 0.01
    # 前轮压在立面角上（轮心在立面前 3 cm，轮底在地面上方 2 cm）：轮下地面算离地，轮廓-地形间隙仍为 0（被立面角托住）
    corner = _state(1.5 - 0.03 - FRONT_X, z=GROUND_Z + 0.02)
    c = task.terrain_context(corner)
    assert c["clearance"][0, 0].item() == pytest.approx(0.02, abs=1e-3)
    assert c["rim_clear"][0, 0].item() < 0.0 and c["lifted"][0, 0].item() < 0.01


def test_support_margin_penalizes_unstable_single_wheel_lift():
    task = _task(lambda x: 0.0)
    grounded = _terms(task, _state(0.5))
    assert grounded["support_margin"].item() > -0.05 and grounded["multi_lift"].item() == pytest.approx(0.0, abs=1e-3)
    # 只抬左前轮（摆臂 −0.58 rad，轮心升高约 11 cm）：质心几乎落在 fr–rl 对角线上，裕度不足 0.5 cm
    s = _state(0.5, arm=(-0.58, 0.0, 0.0, 0.0)).requires_grad_(True)
    ctx = task.terrain_context(s)
    assert ctx["lifted"][0].round().tolist() == [1.0, 0.0, 0.0, 0.0]
    assert ctx["support_margin"][0, 0].item() < 0.005
    terms = _terms(task, s)
    deficit = (0.005 - ctx["support_margin"][0, 0].item()) / 0.01
    expected = -10.0 * (0.5 * deficit**2 if deficit < 1.0 else deficit - 0.5)
    assert terms["support_margin"].item() == pytest.approx(expected, rel=0.02) and expected < -0.5
    assert terms["multi_lift"].item() == pytest.approx(0.0, abs=1e-3)
    grad = torch.autograd.grad(terms["support_margin"].sum(), s)[0]
    assert torch.isfinite(grad).all() and grad[0, state.JPOS].abs().sum() > 0  # 摆臂可以调整支撑三角形
    # 四轮同时离地（车体抬高 5 cm）：两轮以上离地的代价
    assert _terms(task, _state(0.5, z=GROUND_Z + 0.05))["multi_lift"].item() == pytest.approx(-5.0 * 9.0, rel=0.01)


# ---------------- 7 越障：每根轴一个轮预抬 ----------------
def _narrow_task():
    """只在左侧（y ≥ 0）有 12 cm 单台阶（x ∈ [1.5, 1.9]）。"""
    xs, ys = torch.arange(0.0, 4.0 + 1e-9, 0.02), torch.arange(-1.0, 1.0 + 1e-9, 0.02)
    heights = torch.tensor([[0.12 if (1.5 <= float(x) <= 1.9 and float(y) >= 0.0) else 0.0 for y in ys] for x in xs])
    return PreviewTask(HeightMap(heights.float(), (0.0, -1.0), 0.02), KIN, torch.zeros(4, 3),
                       torch.tensor([[-0.58, 0.58]] * 4 + [[-1.3, 1.3]] * 4), PreviewRewardCfg(), lane_center_y=LANES)


def _fl_arm_for_lift(lift):
    """左前摆臂角：轮心相对车体升高 lift（前腿 θ 负向抬轮）。"""
    thetas = torch.linspace(-0.58, 0.0, 581)
    z = KIN.wheel_centers_base(torch.stack((thetas, torch.zeros_like(thetas), torch.zeros_like(thetas),
                                            torch.zeros_like(thetas)), -1), torch.zeros(len(thetas), 4))[:, 0, 2]
    return thetas[(z - WHEEL_Z - lift).abs().argmin()].item()


def test_prelift_requires_one_wheel_per_axle_on_full_width_step():
    """整宽单台阶：两前轮都贴地只算一个轮的误差；左前轮抬到目标、右前轮贴地（之后滚上去）不罚。"""
    full = _task(SINGLE)
    x = 1.5 - R - 0.08 - FRONT_X
    grounded = _terms(full, _state(x))["prelift_track"].item()
    assert grounded == pytest.approx(-5.0 * 0.75, rel=0.02)
    arm = _fl_arm_for_lift(0.035)
    one_up = _state(x, arm=(arm, 0.0, 0.0, 0.0))
    assert full.terrain_context(one_up)["clearance"][0, 0].item() == pytest.approx(0.035, abs=0.003)
    assert _terms(full, one_up)["prelift_track"].item() == pytest.approx(0.0, abs=1e-4)
    s = _state(x).requires_grad_(True)
    grad = torch.autograd.grad(_terms(full, s)["prelift_track"].sum(), s)[0]
    assert torch.isfinite(grad).all()


def test_prelift_on_narrow_step_requires_the_facing_wheel():
    narrow = _narrow_task()
    x = 1.5 - R - 0.08 - FRONT_X
    ctx = narrow.terrain_context(_state(x))
    assert ctx["single"][0].tolist() == [True, False, False, False]  # 只有左前轮面对窄台阶
    assert _terms(narrow, _state(x))["prelift_track"].item() == pytest.approx(-5.0 * 0.75, rel=0.02)
    assert _terms(narrow, _state(x, arm=(_fl_arm_for_lift(0.035), 0.0, 0.0, 0.0)))["prelift_track"].item() == \
        pytest.approx(0.0, abs=1e-4)


def test_support_margin_counts_the_wheel_due_to_lift_before_it_lifts():
    """立面前 prelift_shift_zone 内该抬的轮不论是否已抬都计支撑裕度（门槛 prelift_margin_m）：不抬逃不掉代价；
    先倾斜车体移质心可以清零。"""
    narrow = _narrow_task()
    x = 1.5 - R - 0.08 - FRONT_X
    weight, _, target = narrow._support_weights(narrow.terrain_context(_state(x)))
    assert weight[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-3)  # 窄台阶：面对立面的左前轮
    assert target[0].tolist() == pytest.approx([0.008, 0.005, 0.005, 0.005])
    level = _terms(narrow, _state(x))["support_margin"].item()
    assert level == pytest.approx(-10.0 * 0.5 * 0.8**2, rel=0.05)  # 水平姿态质心落在 fr–rl 对角线上，差 0.8 cm
    tilts = [_state(x, roll=sr * math.radians(4.0), pitch=sp * math.radians(3.0)) for sr in (-1, 1) for sp in (-1, 1)]
    margins = [narrow.terrain_context(t)["support_margin"][0, 0].item() for t in tilts]
    k = max(range(4), key=lambda i: margins[i])
    assert margins[k] >= 0.008  # 横滚 4° + 车头抬 3° 即有 ≥0.8 cm 裕度
    assert _terms(narrow, tilts[k])["support_margin"].item() > 0.05 * level  # 只剩着地轮 sigmoid 尾巴
    full = _task(SINGLE)
    w_full, _, _ = full._support_weights(full.terrain_context(_state(x)))
    assert w_full[0, [0, 2]].sum().item() == pytest.approx(1.0, abs=1e-3) and w_full[0, [1, 3]].sum().item() < 1e-3  # 整宽：每轴只计一个轮
    stairs = _task(STAIRS)
    w_stairs, _, t_stairs = stairs._support_weights(stairs.terrain_context(_state(x)))
    assert w_stairs[0, [0, 2]].sum().item() == pytest.approx(1.0, abs=1e-3) and w_stairs[0, [1, 3]].sum().item() < 5e-3  # 楼梯：前轴一个轮
    assert t_stairs[0].tolist() == pytest.approx([0.005] * 4)  # 楼梯前轴门槛仍是 support_margin_m（不要求先移质心）
    stairs_shift = stairs._support_weights(stairs.terrain_context(_state(1.5 - R - 0.20 - FRONT_X)))[0]
    assert stairs_shift.sum().item() < 5e-3  # 楼梯：预抬窗口外不计（移质心区只对单台阶）
    shift = _state(1.5 - R - 0.20 - FRONT_X)  # 预抬窗口（0.12 m）外、移质心区（0.25 m）内：已经要求裕度
    assert narrow._support_weights(narrow.terrain_context(shift))[0][0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-3)
    far = _state(1.5 - R - 0.35 - FRONT_X)
    assert narrow._support_weights(narrow.terrain_context(far))[0].sum().item() < 5e-3  # 区外不计


def test_climbing_wheels_do_not_count_as_airborne():
    """整宽立面上两前轮同时翻越（由立面角托住）不算两轮离地；同样抬法离立面远时仍算。"""
    arm = _fl_arm_for_lift(0.035)
    lifted = KIN.wheel_centers_base(torch.tensor([arm, 0.0, arm, 0.0]), torch.zeros(4))
    front_x = lifted[0, 0].item()
    stairs = _task(STAIRS)
    climbing = _state(1.5 - R + 0.01 - front_x, arm=(arm, 0.0, arm, 0.0))
    ctx = stairs.terrain_context(climbing)
    assert ctx["lead"][0, [0, 2]].max().item() <= 0.0 and ctx["lead"][0, [0, 2]].min().item() > -R
    assert ctx["lifted"][0, [0, 2]].min().item() > 0.99  # 轮廓已离开地形：旧判据算两轮离地
    terms = _terms(stairs, climbing)
    assert terms["multi_lift"].item() == pytest.approx(0.0, abs=1e-4)
    assert terms["support_margin"].item() > -0.01  # 只剩着地轮 sigmoid 尾巴（约 3e-4 × 代价）
    flat = _task(lambda x: 0.0)
    assert _terms(flat, _state(0.5, arm=(arm, 0.0, arm, 0.0)))["multi_lift"].item() == pytest.approx(-5.0, rel=0.02)


# ---------------- 7 越障：准备行程 / 横滚超限 ----------------
def test_ready_stroke_requires_arm_travel_before_riser():
    task = _task(PLATFORM)
    assert task.max_stroke.min().item() > 0.2
    neutral = _terms(task, _state(1.5 - R - 0.20 - FRONT_X))["ready_stroke"].item()
    stroke = task.wheel_top_z[0].item() - WHEEL_Z
    deficit = (0.12 + 0.01 - stroke) / 0.03
    assert neutral == pytest.approx(-1.0 * 2 * 0.5 * deficit**2, rel=0.05) and neutral < 0.0
    ready = _terms(task, _state(1.5 - R - 0.20 - FRONT_X, arm=(0.3, 0.0, 0.3, 0.0)))["ready_stroke"].item()
    assert ready == pytest.approx(0.0, abs=1e-6)  # 前臂下压（轮相对车体降低）后行程足够
    assert _terms(task, _state(0.3))["ready_stroke"].item() == 0.0  # 离立面远


def test_roll_excess_only_beyond_threshold():
    flat = _task(lambda x: 0.0)
    assert _terms(flat, _state(0.5, roll=math.radians(3.0)))["roll_excess"].item() == 0.0
    big = _terms(flat, _state(0.5, roll=math.radians(10.0)))["roll_excess"].item()
    assert big == pytest.approx(-2000.0 * (math.sin(math.radians(10.0)) - math.sin(math.radians(4.5))) ** 2, rel=0.01)


def test_riser_geometry_is_cast_once_per_state_and_shared():
    calls = []
    task = _task(PLATFORM, _box_riser_fn(calls=calls))
    s = _front_gap_state(0.05)
    task.reward_terms(s, s, CMD, ZERO, ZERO)
    with torch.no_grad():
        task.event_priority(s)
        task.riser_contacts(s)
        task.posture_errors(s)
    assert len(calls) == 1 and calls[0] == (1, 4, 3)  # 四个轮一次批量投射
    task.reward_terms(s, s.clone(), CMD, ZERO, ZERO)
    assert len(calls) == 2
    with torch.no_grad():
        fresh = _front_gap_state(0.06)
        task.event_priority(fresh)
    task.reward_terms(fresh, fresh, CMD, ZERO, ZERO)  # 无梯度缓存不能给需要梯度的调用复用
    assert len(calls) == 4


def test_first_crossing_interpolates_and_reports_missing():
    offsets = torch.tensor([0.0, 0.1, 0.2, 0.3])
    values = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]])
    out = first_crossing(values, offsets, 0.05)
    assert out[0].item() == pytest.approx(0.15) and math.isinf(out[1].item())


# ---------------- 早抬：只在单障碍，不在楼梯 ----------------
def test_early_lift_penalty_on_single_steps_and_front_axle_on_stairs():
    raised = dict(z=GROUND_Z + 0.03)
    far = _state(1.5 - R - 0.40 - FRONT_X, **raised)
    single, stairs = _task(SINGLE), _task(STAIRS)
    assert single.terrain_context(far)["single"][0, [0, 2]].all()
    assert not stairs.terrain_context(far)["single"].any()
    expected = -2.0 * 2 * ((0.03 - 0.003) / 0.02) ** 2
    assert _terms(single, far)["early_lift"].item() == pytest.approx(expected, rel=0.05)
    assert _terms(stairs, far)["early_lift"].item() == pytest.approx(expected, rel=0.05)  # 楼梯：前轴同样不许早抬
    # 预抬/早抬的轮：单台阶四轮都算，楼梯只有前轴
    assert stairs._prelift_eligible(stairs.terrain_context(far))[0].tolist() == [True, False, True, False]
    rear_near = single.terrain_context(_state(1.5 - R - 0.08 - REAR_X))  # 后轮面对单台阶
    assert rear_near["single"][0, [1, 3]].all() and single._prelift_eligible(rear_near)[0].all()
    before_window = _state(1.5 - R - 0.16 - FRONT_X, **raised)  # 0.16 m > early_lift_lead 0.14：仍算早抬
    assert _terms(single, before_window)["early_lift"].item() == pytest.approx(expected, rel=0.05)
    near = _state(1.5 - R - 0.10 - FRONT_X, **raised)
    assert _terms(single, near)["early_lift"].item() == 0.0  # 起抬窗口（8–14 cm）内不罚


def test_prelift_tracking_window_matches_acceptance():
    """窗口 0.12 m：lead 0.16 m 不跟踪；lead 0.08 m 时前轮贴地被罚、抬到目标净空后不罚。"""
    task = _task(SINGLE)
    outside = _terms(task, _state(1.5 - R - 0.16 - FRONT_X))["prelift_track"].item()
    grounded = _terms(task, _state(1.5 - R - 0.08 - FRONT_X))["prelift_track"].item()
    # 贴地：每根轴只要求一个轮，差目标净空 3 cm → 5.0 × smoothL1(1; β=0.5)=0.75 → −3.75（后轮离立面远）
    assert outside == 0.0 and grounded == pytest.approx(-3.75, rel=0.02)
    arm = (_fl_arm_for_lift(0.04), 0.0, 0.0, 0.0)
    lifted = _terms(task, _state(1.5 - R - 0.08 - FRONT_X, arm=arm))["prelift_track"].item()
    assert lifted == pytest.approx(0.0, abs=1e-4)  # 净空 4 cm 落在 [目标, 目标 + 1.5 cm] 内
    # 整车抬高 4 cm（四轮都离地）不是抬轮：实际净空达标，支撑净空为 0；水平姿态质心在对角线上（裕度≈0），
    # 两者取大 → 与贴地同样的误差
    raised = _terms(task, _state(1.5 - R - 0.08 - FRONT_X, z=GROUND_Z + 0.04))["prelift_track"].item()
    assert raised == pytest.approx(grounded, rel=0.02)


def test_support_clearance_ignores_body_pose_and_follows_the_arm():
    """支撑净空 = 其余三轮着地时的轮底净空：对车身平移/倾斜不变，梯度只经摆臂；单收左前臂时左前、右后为正。"""
    flat = _task(lambda x: 0.0)
    arm = _fl_arm_for_lift(0.03)
    base = flat.terrain_context(_state(0.5, arm=(arm, 0.0, 0.0, 0.0)))["support_clearance"][0]
    assert base[0].item() == pytest.approx(0.03, abs=0.003) and base[3].item() == pytest.approx(0.03, abs=0.004)
    assert base[1].item() == pytest.approx(-0.03, abs=0.004) and base[2].item() == pytest.approx(-0.03, abs=0.004)
    for kw in (dict(z=GROUND_Z + 0.05), dict(roll=math.radians(4.0)), dict(pitch=math.radians(-4.0)),
               dict(z=GROUND_Z + 0.02, roll=math.radians(-3.0), pitch=math.radians(3.0))):
        moved = flat.terrain_context(_state(0.5, arm=(arm, 0.0, 0.0, 0.0), **kw))["support_clearance"][0]
        assert (moved - base).abs().max().item() < 0.003, kw
    s = _state(0.5).requires_grad_(True)
    g = torch.autograd.grad(flat.terrain_context(s)["support_clearance"][0, 0], s)[0][0]
    assert g[state.POS].abs().max().item() < 1e-4 and g[state.QUAT].abs().max().item() < 0.02
    assert g[state.JPOS][0].abs().item() > 0.05  # 左前摆臂


def test_climb_tracking_one_wheel_per_axle_at_a_time():
    """整宽立面两前轮都贴住时只算一个轮的翻越误差；左前轮翻上去后右前轮单独面对立面，必须翻。"""
    stairs = _task(STAIRS)
    x = 1.5 - R + 0.001 - FRONT_X
    both = _terms(stairs, _state(x))["prelift_track"].item()
    narrow = _narrow_task()
    one = _terms(narrow, _state(x))["prelift_track"].item()
    assert both < -1.0 and both == pytest.approx(one, rel=0.05)


def test_prelift_on_single_steps_and_front_axle_on_stairs_climb_tracking_on_all_risers():
    stairs = _task(STAIRS)
    # 楼梯接触前：前轴每级一个轮预抬（与单台阶同一误差）；后轴不要求
    assert _terms(stairs, _state(1.5 - R - 0.08 - FRONT_X))["prelift_track"].item() == pytest.approx(-3.75, rel=0.02)
    pressed = _terms(stairs, _state(1.5 - R + 0.001 - FRONT_X))["prelift_track"].item()
    assert pressed < -1.0  # 贴住立面（lead≈0）不上：翻越跟踪
    platform = _task(PLATFORM)  # 平台（立面后不回落）不是单台阶
    assert not platform.terrain_context(_state(0.5))["single"].any()


# ---------------- 1 跟踪 / 前进 / 停滞 ----------------
def test_slow_band_before_riser_allows_safe_speed_and_stall_penalty():
    task = _task(PLATFORM)
    near_slow = _terms(task, _front_gap_state(0.20, vx=0.15))
    near_safe = _terms(task, _front_gap_state(0.20, vx=0.12))
    too_slow = _terms(task, _front_gap_state(0.20, vx=0.06))
    far_slow = _terms(task, _state(0.0, vx=0.15))
    assert near_slow["track_lin_vel"].item() == pytest.approx(5.0, rel=1e-4)
    assert near_safe["track_lin_vel"].item() == pytest.approx(5.0, rel=1e-4)  # min(cmd, 0.12) 在带内
    assert too_slow["track_lin_vel"].item() < 4.9
    assert far_slow["track_lin_vel"].item() < 4.5
    stopped = _terms(task, _front_gap_state(0.20, vx=0.0))
    assert stopped["track_lin_vel"].item() < 4.0 and stopped["stall"].item() < -1.9
    assert near_slow["stall"].item() > -0.1 and near_slow["progress"].item() == pytest.approx(0.5, rel=1e-4)
    assert near_safe["stall"].item() > -0.1 and stopped["stall"].item() < -4.9  # 减速带下限几乎不罚，停住罚满
    assert task.progressing(_state(0.0, vx=0.2), CMD).item() and not task.progressing(_state(0.0, vx=0.0), CMD).item()



def test_stairs_zone_stall_and_contact_scales_only_change_stairs():
    """第十五轮：楼梯区单独的贴立面停滞系数与轻碰系数/接近距离；单台阶完全不受影响，默认 = 原行为。"""
    base = dict(stall_climb_scale=0.2, riser_contact_speed=-400.0, riser_approach_m=0.15)
    new = dict(base, stairs_stall_climb_scale=1.0, stairs_riser_contact_scale=0.5, stairs_riser_approach_m=0.05)
    climbing = _front_gap_state(0.0, vx=0.0)  # 前轮贴着立面翻越（−r < lead ≤ 0），车速 0
    approach = _front_gap_state(0.05, vx=0.3)
    old_task, new_task = _task(STAIRS, **base), _task(STAIRS, **new)
    assert new_task.terrain_context(climbing)["stairs_zone"].item()
    old_stall, new_stall = _terms(old_task, climbing)["stall"].item(), _terms(new_task, climbing)["stall"].item()
    assert new_stall < -4.9 and old_stall == pytest.approx(0.2 * new_stall, rel=1e-5)
    gap = new_task.terrain_context(approach)["gap"][0, 0].item()
    assert 0.0 < gap < 0.05
    ratio = 0.5 * ((0.05 - gap) / 0.05) / ((0.15 - gap) / 0.15)
    old_c = _terms(old_task, approach)["riser_contact_speed"].item()
    new_c = _terms(new_task, approach)["riser_contact_speed"].item()
    assert old_c < -1.0 and new_c == pytest.approx(ratio * old_c, rel=1e-4)
    for s in (climbing, approach):
        assert not _task(SINGLE, **new).terrain_context(s)["stairs_zone"].item()
        old_t, new_t = _terms(_task(SINGLE, **base), s), _terms(_task(SINGLE, **new), s)
        for key in ("stall", "riser_contact_speed"):
            assert new_t[key].item() == pytest.approx(old_t[key].item(), rel=1e-6, abs=1e-9), key
    defaults = PreviewRewardCfg()
    assert (defaults.stairs_stall_climb_scale, defaults.stairs_riser_contact_scale,
            defaults.stairs_riser_approach_m) == (-1.0, 1.0, 0.0)



def test_trailing_wheel_climbs_on_contact_once_partner_is_a_level_up():
    """第十五轮：同轴另一个轮已站上更高一级时，本轮不再是接触前预抬对象（跟踪/支撑裕度"该抬的轮"），碰到立面后照常翻越。

    场景复刻第十四轮 it40000 12 cm×40 cm 楼梯卡住的状态：左前轮已在第二级、右前轮在第二级立面前约 5 cm（预抬窗口内）。"""
    stairs40 = lambda x: 0.12 * min(max(math.floor((x - 1.5) / 0.40) + 1, 0), 6)  # noqa: E731
    s = _state(1.5055, z=GROUND_Z + 0.12, yaw=-0.308)
    fl, fr = 0, 2  # 关节顺序 (fl, rl, fr, rr)
    old, new = _task(stairs40), _task(stairs40, trailing_contact_climb=1.0)
    ctx_old, ctx_new = old.terrain_context(s), new.terrain_context(s)
    assert ctx_new["ground"][0, fl].item() - ctx_new["ground"][0, fr].item() == pytest.approx(0.12, abs=1e-3)
    lead_fr = ctx_new["lead"][0, fr].item()
    assert 0.0 < lead_fr <= old.cfg.prelift_window
    assert old._prelift_eligible(ctx_old)[0, fr] and not new._prelift_eligible(ctx_new)[0, fr]
    assert new._prelift_eligible(ctx_new)[0, fl]  # 先上去的轮（同轴另一轮更低）仍是预抬对象
    assert old._prelift_errors(ctx_old)[1][0, fr] and not new._prelift_errors(ctx_new)[1][0, fr]
    assert _terms(old, s)["prelift_track"].item() < -1.0 and _terms(new, s)["prelift_track"].item() == 0.0
    assert old._support_weights(ctx_old)[0][0, fr] == 1.0 and new._support_weights(ctx_new)[0][0, fr] < 0.01  # 只剩离地程度（≈0）
    # 碰到立面后（−r < lead ≤ 0）翻越跟踪不受影响
    touching = _state(1.5055 + lead_fr + 0.01, z=GROUND_Z + 0.12, yaw=-0.308)
    c = new.terrain_context(touching)
    assert -WHEEL_RADIUS < c["lead"][0, fr].item() <= 0.0 and new._prelift_errors(c)[1][0, fr]
    # 默认关闭 = 原行为
    assert PreviewRewardCfg().trailing_contact_climb == 0.0



def test_stairs_pair_climb_lifting_trailing_wheel_beats_standing_still():
    """第十六轮：复刻第十五轮 it26000 在 12 cm×40 cm 楼梯的死锁姿态——左前轮搭在第二级台阶角上（翻越中、误差≈0），
    右前轮顶着第二级立面，后轴在地面。原行为下抬右前臂让翻越跟踪更差（取两轮较小误差 + 支撑净空假设失效）；
    stairs_climb_pair_sum 下抬右前臂明显更好。单台阶不受影响，默认关闭。"""
    stairs40 = lambda x: 0.12 * min(max(math.floor((x - 1.5) / 0.40) + 1, 0), 6)  # noqa: E731
    fl, fr = 0, 2  # 关节顺序 (fl, rl, fr, rr)
    stuck_arm, lifted_arm = (-0.58, -0.25, 0.064, -0.333), (-0.58, -0.25, -0.58, -0.333)
    pose = dict(z=0.442, pitch=0.110, yaw=-0.152)
    stuck, lifted = _state(1.48, arm=stuck_arm, **pose), _state(1.48, arm=lifted_arm, **pose)
    common = dict(trailing_contact_climb=1.0)
    old, new = _task(stairs40, **common), _task(stairs40, stairs_climb_pair_sum=1.0, **common)
    ctx = new.terrain_context(stuck)
    lead = ctx["lead"][0]
    assert ctx["stairs_zone"].item()
    assert -WHEEL_RADIUS < lead[fl].item() <= 0.0 and -WHEEL_RADIUS < lead[fr].item() <= 0.0  # 两前轮都在翻越
    assert ctx["ground"][0, fr].item() == pytest.approx(0.12, abs=1e-3)
    track = lambda task, s: _terms(task, s)["prelift_track"].item()  # noqa: E731
    assert track(old, lifted) < track(old, stuck) - 1.0  # 原行为：抬右轮反而更差（死锁成因）
    assert track(new, stuck) < -5.0  # 右轮不翻不再被左轮的小误差吃掉
    assert track(new, lifted) > track(new, stuck) + 3.0
    # 单台阶与非成对翻越不受影响
    for s in (_front_gap_state(0.0, vx=0.0), _front_gap_state(0.05, vx=0.3)):
        a, b = _task(SINGLE, **common), _task(SINGLE, stairs_climb_pair_sum=1.0, **common)
        assert track(b, s) == pytest.approx(track(a, s), rel=1e-6, abs=1e-9)
    assert PreviewRewardCfg().stairs_climb_pair_sum == 0.0


def test_stall_linear_gives_restart_gradient_when_stopped():
    """第十六轮：低于停滞门槛的速度差线性代价——车速 0 处停滞 sigmoid 饱和、几乎无梯度，线性项有；与停滞同样乘贴立面系数。"""
    task = _task(PLATFORM, stall_linear=-3.0)
    gate = task.cfg.stall_speed_ratio * CMD[0, 0].item()
    t0, t1 = _terms(task, _state(0.0, vx=0.0)), _terms(task, _state(0.0, vx=0.01))
    assert t0["stall_linear"].item() == pytest.approx(-3.0, rel=1e-4)
    assert t1["stall_linear"].item() == pytest.approx(-3.0 * (gate - 0.01) / gate, rel=1e-3)
    assert abs(t1["stall"].item() - t0["stall"].item()) < 0.05  # sigmoid 饱和
    assert t1["stall_linear"].item() - t0["stall_linear"].item() > 0.3
    assert _terms(task, _state(0.0, vx=gate + 0.01))["stall_linear"].item() == 0.0
    climbing = _front_gap_state(0.0, vx=0.0)
    scaled = _terms(_task(SINGLE, stall_linear=-3.0, stall_climb_scale=0.2), climbing)
    assert scaled["stall_linear"].item() == pytest.approx(0.2 * -3.0, rel=1e-4)
    assert _terms(task, _state(0.0), cmd=torch.zeros(1, 3))["stall_linear"].item() == 0.0  # 静止命令不计
    assert PreviewRewardCfg().stall_linear == 0.0 and "stall_linear" not in _terms(_task(PLATFORM), _state(0.0))


def test_stall_climb_discount_only_for_lifted_wheel():
    """第十七轮：贴立面停滞折扣只给立面前/贴面且真离地的轮（对净空可导）；着地顶着立面按全额停滞计。"""
    common = dict(stall_climb_scale=0.2, stall_linear=-3.0)
    old = _task(PLATFORM, **common)
    new = _task(PLATFORM, stall_climb_min_clearance=0.01, stall_climb_lead_m=0.05, **common)
    pressed = _front_gap_state(0.0, vx=0.0)
    lifted = _front_gap_state(0.0, vx=0.0, arm=(_fl_arm_for_lift(0.04), 0.0, 0.0, 0.0))
    c = new.terrain_context(lifted)
    assert c["clearance"][0, 0].item() > 0.03 and -R < c["lead"][0, 0].item() <= 0.05
    assert _terms(old, pressed)["stall"].item() == pytest.approx(0.2 * -5.0, rel=0.01)  # 原行为：顶着就打折
    assert _terms(new, pressed)["stall"].item() == pytest.approx(-5.0, rel=0.01)
    assert _terms(new, pressed)["stall_linear"].item() == pytest.approx(-3.0, rel=0.01)
    assert _terms(new, lifted)["stall"].item() == pytest.approx(0.2 * -5.0, rel=0.02)
    assert _terms(new, lifted)["stall_linear"].item() == pytest.approx(0.2 * -3.0, rel=0.02)
    assert _terms(new, _state(0.0, vx=0.0))["stall"].item() == pytest.approx(-5.0, rel=0.01)  # 远离立面不变
    # 抬到一半（净空≈门槛）：停滞对前摆臂有梯度，往抬轮方向（θ 更负）代价变小
    half = _front_gap_state(0.0, vx=0.0, arm=(_fl_arm_for_lift(0.012), 0.0, 0.0, 0.0)).requires_grad_(True)
    grad = torch.autograd.grad(_terms(new, half)["stall"].sum(), half)[0][0, state.JPOS][0].item()
    assert grad < -1.0
    cfg = PreviewRewardCfg()
    assert cfg.stall_climb_min_clearance < 0.0 and cfg.stall_climb_lead_m == 0.0


def test_riser_spin_penalizes_grounded_wheel_spinning_against_riser():
    """第十七轮：贴立面且着地的轮空转（轮缘速度 > 轮心沿轮向速度）计代价；平地纯滚动、停转、已离地的轮不计。"""
    task = _task(PLATFORM, riser_spin=-10.0)
    delta = task.cfg.wheel_slip_delta

    def spinning(s, wheel):
        s[0, state.JVEL.start + 8: state.JVEL.start + 12] = torch.tensor(wheel)
        return s

    pressed = _terms(task, spinning(_front_gap_state(0.0, vx=0.0), (2.5, 0.0, 2.5, 0.0)))["riser_spin"].item()
    grounded = 1.0 / (1.0 + math.exp(-0.01 / 0.003))
    assert pressed == pytest.approx(-10.0 * 2 * (2.5 * R - 0.5 * delta) * grounded, rel=0.05)
    assert _terms(task, spinning(_front_gap_state(0.0, vx=0.0), (0.0,) * 4))["riser_spin"].item() == 0.0
    assert _terms(task, _rolling_state(x=0.0))["riser_spin"].item() == 0.0
    one_lifted = spinning(_front_gap_state(0.0, vx=0.0, arm=(_fl_arm_for_lift(0.04), 0.0, 0.0, 0.0)), (2.5, 0.0, 2.5, 0.0))
    assert _terms(task, one_lifted)["riser_spin"].item() == pytest.approx(0.5 * pressed, rel=0.05)
    assert PreviewRewardCfg().riser_spin == 0.0 and "riser_spin" not in _terms(_task(PLATFORM), _state(0.0))


def test_stairs_lane_scale_relaxes_lane_costs_only_in_stairs_zone():
    """第十八轮：楼梯区车道项与越障横向速度额外倍数 × stairs_lane_scale；楼梯区外不变，默认 1 = 原行为。"""
    x = 1.5 - R - 0.30 - FRONT_X
    stairs = _task(STAIRS, obstacle_lateral_vel_scale=3.0)
    near = _state(x, y=0.2, vy=0.1)
    assert stairs.terrain_context(near)["stairs_zone"].item()
    full = _terms(stairs, near)
    stairs.set_stairs_lane_scale(0.0)
    relaxed = _terms(stairs, near)
    assert relaxed["lane"].item() == 0.0 and full["lane"].item() == pytest.approx(-30.0 * 0.04, rel=1e-4)
    flat_gate = stairs.terrain_context(near)["flat"].item()
    assert relaxed["lateral_vel"].item() == pytest.approx(full["lateral_vel"].item() / (1.0 + 2.0 * (1.0 - flat_gate)),
                                                          rel=1e-4)
    stairs.set_stairs_lane_scale(0.5)
    assert _terms(stairs, near)["lane"].item() == pytest.approx(0.5 * full["lane"].item(), rel=1e-4)
    far = _state(1.5 - R - 0.90 - FRONT_X, y=0.2, vy=0.1)  # 离第一级 >0.6 m：不在楼梯区
    assert not stairs.terrain_context(far)["stairs_zone"].item()
    stairs.set_stairs_lane_scale(0.0)
    assert _terms(stairs, far)["lane"].item() == pytest.approx(-30.0 * 0.04, rel=1e-4)
    single = _task(SINGLE)
    single.set_stairs_lane_scale(0.0)
    assert _terms(single, _state(x, y=0.2))["lane"].item() == pytest.approx(-30.0 * 0.04, rel=1e-4)  # 单台阶不放松
    with pytest.raises(ValueError):
        stairs.set_stairs_lane_scale(1.5)
    assert PreviewRewardCfg().stairs_lane_scale == 1.0


def test_stairs_success_lane_and_schedule():
    """第十八轮：楼梯单独的成功车道（课程晋级与终点终止都按类型取），默认 0 = 同 success_lane_m；收紧系数线性。"""
    progress = torch.tensor([3.7, 3.7, 3.7, 3.7])
    lateral = torch.tensor([0.30, 0.30, 0.10, 0.45])
    upright = active = torch.ones(4, dtype=torch.bool)
    kinds = torch.tensor([ot.KIND_STAIRS, ot.KIND_SINGLE, ot.KIND_STAIRS, ot.KIND_STAIRS])
    strict = ot.curriculum_decision(progress, lateral, upright, active, kinds, None)[0]
    assert strict.tolist() == [False, False, True, False]
    loose_cfg = {**ot.CURRICULUM_DEFAULTS, "stairs_success_lane_m": 0.40}
    loose = ot.curriculum_decision(progress, lateral, upright, active, kinds, None, loose_cfg)[0]
    assert loose.tolist() == [True, False, True, False]  # 只放松楼梯；0.45 > 出车道 0.40 仍不过
    rel = torch.stack((progress, lateral), dim=-1)
    reached, out = ot.goal_and_lane(rel, loose_cfg, kinds)
    assert reached.tolist() == [True, False, True, False] and out.tolist() == [False, True, False, True]
    assert torch.equal(ot.goal_and_lane(rel, loose_cfg)[0], ot.goal_and_lane(rel)[0])  # 无 kinds = 原行为
    assert [ot.lane_schedule_scale(i, 26000, 36000) for i in (14000, 26000, 31000, 36000, 44000)] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert ot.lane_schedule_scale(9, 10, 10) == 0.0 and ot.lane_schedule_scale(10, 10, 10) == 1.0
    assert ot.CURRICULUM_DEFAULTS["stairs_success_lane_m"] == 0.0


def test_obstacle_lateral_velocity_scale_only_off_flat():
    """第十五轮：越障区（1 − 平地门）横向速度代价放大；平地不变，默认 = 原行为。"""
    old, new = _task(PLATFORM), _task(PLATFORM, obstacle_lateral_vel_scale=3.0)
    near, far = _front_gap_state(0.20, vy=0.1), _state(0.0, vx=0.3, vy=0.1)
    assert new.terrain_context(near)["flat"].item() < 0.01 and new.terrain_context(far)["flat"].item() > 0.99
    assert _terms(new, near)["lateral_vel"].item() == pytest.approx(3.0 * _terms(old, near)["lateral_vel"].item(), rel=1e-3)
    # 平地门是 sigmoid（平地约 0.995），平地上只多约 1%
    assert _terms(new, far)["lateral_vel"].item() == pytest.approx(_terms(old, far)["lateral_vel"].item(), rel=0.02)
    assert _terms(old, near)["lateral_vel"].item() < -1.0
    assert PreviewRewardCfg().obstacle_lateral_vel_scale == 1.0


# ---------------- 6 平地中立 / 2 终止 / 优先级 ----------------
def test_flat_neutral_only_on_flat_and_termination_penalty():
    task = _task(PLATFORM)
    bent = (0.3, 0.3, 0.3, 0.3)
    assert _terms(task, _state(0.0, arm=bent))["flat_neutral"].item() < -0.3
    assert _terms(task, _front_gap_state(0.20, arm=bent))["flat_neutral"].item() > -0.01
    tipped = _state(0.0, roll=1.2)
    assert _terms(task, tipped)["terminated"].item() < -19.0
    assert _terms(task, _state(0.0))["terminated"].item() > -0.1
    priority = task.event_priority(torch.cat((_state(0.0), _front_gap_state(0.20), _state(1.5 - R - FRONT_X))))
    assert priority.tolist() == [0.0, 1.0, 2.0]


def test_critic_lane_is_critic_only():
    lanes = torch.tensor([0.0, 8.0])
    task = PreviewTask(_map(PLATFORM), KIN, torch.zeros(6, 3), torch.zeros(8, 2), lane_center_y=lanes)
    s = _state(0.0)
    s[0, 1] = 0.25
    assert task.critic_extra_dim == RELIEF_DIM + 1 and task.obs_dim == 45 + 6
    assert task.critic_extra(s)[0, -1].item() == pytest.approx(0.25)
    assert task.policy_obs(s, CMD, ZERO).shape == (1, 51)


# ---------------- 地形配置 ----------------
def test_stairs_ranges_and_tread_distribution():
    risers, treads = zip(*(ot.stairs_dimensions(d, seed=7) for d in torch.linspace(0, 1, 400).tolist()))
    assert min(risers) == pytest.approx(0.04) and max(risers) == pytest.approx(0.20)
    assert 0.26 <= min(treads) < 0.28 and 0.38 < max(treads) <= 0.40
    assert ot.SINGLE_HEIGHT_M == (0.03, 0.20) and ot.STAIR_RISER_M == (0.04, 0.20)


def test_terrain_mix_columns_match_isaac_generator():
    names, proportions = list(ot.MIX), list(ot.MIX.values())
    kinds = ot.column_kinds(names, proportions, ot.NUM_COLS)
    assert kinds.count(ot.KIND_STAIRS) / ot.NUM_COLS >= 0.45
    assert kinds.count(ot.KIND_SINGLE) / ot.NUM_COLS == pytest.approx(0.40)
    assert kinds.count(ot.KIND_FLAT) / ot.NUM_COLS == pytest.approx(0.15)
    vp = _load("vision_preview_for_obstacle", "tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/"
               "vision_preview.py")
    assert ot.column_names(names, proportions, ot.NUM_COLS) == vp.terrain_column_names(
        tuple(names), tuple(proportions), ot.NUM_COLS, curriculum=True)
    cm = ot.equivalent_height_cm(torch.tensor([0, 9]), ot.KIND_STAIRS)
    assert cm.tolist() == pytest.approx([4.8, 19.2])


def test_stairs_mesh_geometry():
    pytest.importorskip("trimesh")
    cfg = SimpleNamespace(size=(8.0, 8.0), riser_range=ot.STAIR_RISER_M, tread_range=ot.STAIR_TREAD_M,
                          num_steps=ot.STAIR_STEPS, first_edge_m=ot.STAIR_FIRST_EDGE_M, lane_width=3.0, seed=3)
    meshes, origin = ot.straight_stairs_terrain(1.0, cfg)
    riser, tread = ot.stairs_dimensions(1.0, seed=3)
    assert len(meshes) == 1 + ot.STAIR_STEPS and riser == pytest.approx(0.20)
    tops = sorted(m.bounds[1][2] for m in meshes[1:])
    assert tops == pytest.approx([riser * (k + 1) for k in range(ot.STAIR_STEPS)])
    edges = sorted(m.bounds[0][0] - origin[0] for m in meshes[1:])
    assert edges == pytest.approx([1.2 + k * tread for k in range(ot.STAIR_STEPS)])


def test_obstacle_env_cfg_wires_limits_curriculum_and_commands():
    src = (PKG / "tasks/manager_based/supermimic/obstacle_env_cfg.py").read_text()
    assert "TILT_LIMIT_RAD = 0.8" in src and "SOFT_JOINT_LIMIT_FACTOR = 0.95" in src
    for needle in ("ot.obstacle_curriculum", "ot.reached_goal", "ot.out_of_lane", "anticipatory_wheel_lift = None",
                   "TerminationTermCfg(func=ot.stall_timeout, time_out=True)",
                   "self.episode_length_s = EPISODE_LENGTH_S", "rel_standing_envs=STANDING_FRACTION",
                   "lin_vel_x=COMMAND_X_RANGE", "mdp_core.UniformVelocityCommandCfg("):
        assert needle in src, needle
    tree = ast.parse(src)
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper()}
    assert constants["EPISODE_LENGTH_S"] == 40.0 and constants["COMMAND_X_RANGE"] == (0.10, 0.40)
    assert constants["STANDING_FRACTION"] == 0.10
    # 父类的阈值命令会把 |v|≤0.2 清零，本任务不能再用它
    assert "UniformThreshold" not in src.split('"""', 2)[2]
    registry = (PKG / "tasks/manager_based/supermimic/__init__.py").read_text()
    assert 'id="RobotLab-Isaac-SuperMimic-Obstacle-Parallelogram-Rover-v0"' in registry
    assert "obstacle_env_cfg:ParallelogramRoverSuperMimicObstacleEnvCfg" in registry


# ---------------- 课程：按类型晋级/降级 ----------------
def _stats(collisions, posture_bad, steps, commanded):
    return {"collisions": torch.tensor(collisions), "posture_bad": torch.tensor(posture_bad),
            "steps": torch.tensor(steps), "commanded_m": torch.tensor(commanded)}


def test_curriculum_decision_promotes_clean_traversals_and_demotes_stalls():
    progress = torch.tensor([3.7, 3.7, 3.7, 1.5, 2.5, 3.7, 0.5, 3.7, 3.7, 1.0])
    lateral = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.2, -0.3])
    upright = torch.tensor([True, True, True, True, True, True, False, True, True, True])
    active = torch.ones(10, dtype=torch.bool)
    kinds = torch.tensor([ot.KIND_STAIRS, ot.KIND_SINGLE, ot.KIND_FLAT, ot.KIND_STAIRS, ot.KIND_STAIRS,
                          ot.KIND_STAIRS, ot.KIND_STAIRS, ot.KIND_STAIRS, ot.KIND_SINGLE, ot.KIND_STAIRS])
    stats = _stats([0, 200, 2, 0, 0, 0, 0, 400, 0, 0], [0] * 10, [900] * 10,
                   [5.0, 5.0, 5.0, 5.0, 4.0, 5.0, 1.0, 5.0, 5.0, 1.0])
    passed, failed = ot.curriculum_decision(progress, lateral, upright, active, kinds, stats)
    # 0 干净通过；1 单障碍碰立面 22% 不升；2 平地无立面；3 停在半程以下（期望 3.6 m）降级；4 慢但过半保持；
    # 5 出车道（0.5 > 0.40）；6 翻车；7 楼梯碰立面 44% 仍晋级（楼梯无碰撞门）；8 到终点但 |y|=0.2 > 0.15 不升不降；
    # 9 |y|=0.3 仍在 0.40 车道内、前进过半：保持
    assert passed.tolist() == [True, False, True, False, False, False, False, True, False, False]
    assert failed.tolist() == [False, False, False, True, False, True, True, False, False, False]
    gated = dict(ot.CURRICULUM_DEFAULTS, stairs_collision_frac=0.10)  # 参数可恢复旧的楼梯碰撞门
    assert not ot.curriculum_decision(progress[7:8], lateral[7:8], upright[7:8], active[7:8], kinds[7:8],
                                      _stats([400], [0], [900], [5.0]), gated)[0].item()
    assert ot.CURRICULUM_DEFAULTS["fail_lane_m"] == 0.40 and ot.CURRICULUM_DEFAULTS["success_lane_m"] == 0.15
    assert ot.CURRICULUM_DEFAULTS["collision_frac"] == 0.10 and ot.CURRICULUM_DEFAULTS["stairs_collision_frac"] == 1.0
    bad_posture = _stats([0], [100], [900], [5.0])
    assert not ot.curriculum_decision(progress[:1], lateral[:1], upright[:1], active[:1], kinds[:1], bad_posture)[0].item()


def test_stairs_demote_ratio_demotes_episodes_stuck_on_the_stairs_only():
    """楼梯单独的降级比例：卡在楼梯中段（前进 > 0.5×期望但 < 0.8×期望）的回合降级；单障碍仍按 demote_ratio；
    默认值与 demote_ratio 相同（原行为）。"""
    progress = torch.tensor([2.5, 2.5, 3.0, 0.9])
    lateral = torch.zeros(4)
    upright = active = torch.ones(4, dtype=torch.bool)
    kinds = torch.tensor([ot.KIND_STAIRS, ot.KIND_SINGLE, ot.KIND_STAIRS, ot.KIND_STAIRS])
    stats = _stats([0] * 4, [0] * 4, [900] * 4, [5.0, 5.0, 5.0, 1.0])  # 期望 = min(命令距离, 3.6 m)
    assert ot.CURRICULUM_DEFAULTS["stairs_demote_ratio"] == ot.CURRICULUM_DEFAULTS["demote_ratio"] == 0.5
    assert ot.curriculum_decision(progress, lateral, upright, active, kinds, stats)[1].tolist() == [False] * 4
    strict = dict(ot.CURRICULUM_DEFAULTS, stairs_demote_ratio=0.8)
    passed, failed = ot.curriculum_decision(progress, lateral, upright, active, kinds, stats, strict)
    # 0 楼梯 2.5 < 0.8×3.6 降级；1 单障碍同样距离保持；2 楼梯 3.0 ≥ 2.88 保持；3 命令只走了 1 m、前进 0.9 m 保持
    assert failed.tolist() == [True, False, False, False] and not passed.any()
    legacy = {k: v for k, v in ot.CURRICULUM_DEFAULTS.items() if k != "stairs_demote_ratio"}  # 旧存档的课程配置
    assert ot.curriculum_decision(progress, lateral, upright, active, kinds, stats, legacy)[1].tolist() == [False] * 4


def test_stall_timeout_truncates_only_when_enabled_and_counted():
    steps = torch.tensor([0, 599, 600, 1000])
    assert ot.stall_steps_exceeded(steps, 12.0, 0.02).tolist() == [False, False, True, True]
    assert not ot.stall_steps_exceeded(steps, 0.0, 0.02).any()
    env = SimpleNamespace(cfg=SimpleNamespace(obstacle_curriculum_cfg={"stall_truncate_s": 12.0}), num_envs=4,
                          device="cpu", step_dt=0.02)
    assert not ot.stall_timeout(env).any()  # 训练循环没写停滞步数（验收等）：从不触发
    env.obstacle_stall_steps = steps
    assert ot.stall_timeout(env).tolist() == [False, False, True, True]
    env.cfg.obstacle_curriculum_cfg = {}
    assert ot.CURRICULUM_DEFAULTS["stall_truncate_s"] == 0.0 and not ot.stall_timeout(env).any()  # 默认关闭


def test_obstacle_curriculum_updates_levels_per_type_and_resamples_top():
    torch.manual_seed(0)
    n, rows, cols = 4, ot.NUM_ROWS, ot.NUM_COLS
    kinds = ot.column_kinds(list(ot.MIX), list(ot.MIX.values()), cols)
    types = torch.tensor([kinds.index(ot.KIND_STAIRS), kinds.index(ot.KIND_SINGLE), kinds.index(ot.KIND_STAIRS),
                          kinds.index(ot.KIND_FLAT)])
    origins = torch.zeros(rows, cols, 3)
    origins[..., 0] = torch.arange(rows)[:, None] * 8.0
    terrain = SimpleNamespace(terrain_levels=torch.tensor([3, 3, rows - 1, 0]), terrain_types=types,
                              terrain_origins=origins, env_origins=origins[torch.tensor([3, 3, rows - 1, 0]), types].clone(),
                              max_terrain_level=rows, cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_rows=rows)))
    root = terrain.env_origins.clone()
    root[:, 0] += torch.tensor([3.7, 1.0, 3.7, 3.7])
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root, projected_gravity_b=torch.tensor([[0., 0., -1.]] * n)))

    class Scene:
        def __init__(self):
            self.terrain = terrain

        def __getitem__(self, key):
            return robot

    tracker = ot.EpisodeTracker(n, "cpu")
    for _ in range(100):
        tracker.update(torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), torch.full((n,), 0.3), 0.02)
    env = SimpleNamespace(scene=Scene(), num_envs=n, episode_length_buf=torch.full((n,), 100),
                          cfg=SimpleNamespace(column_kinds=kinds, obstacle_curriculum_cfg={}), obstacle_tracker=tracker,
                          obstacle_curriculum_log={})
    summary = ot.obstacle_curriculum(env, torch.arange(n))
    levels = terrain.terrain_levels.tolist()
    assert levels[0] == 4 and levels[1] == 3 and 0 <= levels[2] < rows and levels[3] == 1
    assert tracker.steps.sum().item() == 0  # 回合统计已取走
    assert env.obstacle_curriculum_log["stairs_promoted"] == 2 and env.obstacle_curriculum_log["single_episodes"] == 1
    assert torch.equal(terrain.env_origins, origins[terrain.terrain_levels, types])
    assert {"stairs_level", "stairs_height_cm", "single_level", "single_height_cm", "flat_level"} <= set(summary)


def _curriculum_env(levels, kinds_per_env, explore_frac, x_progress):
    rows, cols = ot.NUM_ROWS, ot.NUM_COLS
    kinds = ot.column_kinds(list(ot.MIX), list(ot.MIX.values()), cols)
    types = torch.tensor([kinds.index(k) for k in kinds_per_env])
    origins = torch.zeros(rows, cols, 3)
    origins[..., 0] = torch.arange(rows)[:, None] * 8.0
    lv = torch.tensor(levels)
    terrain = SimpleNamespace(terrain_levels=lv.clone(), terrain_types=types, terrain_origins=origins,
                              env_origins=origins[lv, types].clone(), max_terrain_level=rows,
                              cfg=SimpleNamespace(terrain_generator=SimpleNamespace(num_rows=rows)))
    root = terrain.env_origins.clone()
    root[:, 0] += torch.tensor(x_progress)
    n = len(levels)
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root, projected_gravity_b=torch.tensor([[0., 0., -1.]] * n)))

    class Scene:
        def __init__(self):
            self.terrain = terrain

        def __getitem__(self, key):
            return robot

    return SimpleNamespace(scene=Scene(), num_envs=n, episode_length_buf=torch.full((n,), 100),
                           cfg=SimpleNamespace(column_kinds=kinds,
                                               obstacle_curriculum_cfg={"stairs_explore_frac": explore_frac}),
                           obstacle_curriculum_log={})


def test_stairs_explore_episodes_sample_random_levels_without_moving_the_curriculum():
    torch.manual_seed(0)
    n = 400
    kinds = [ot.KIND_STAIRS] * (n // 2) + [ot.KIND_SINGLE] * (n // 2)
    env = _curriculum_env([1] * n, kinds, 0.5, [3.7] * n)  # 全部干净通过
    ot.obstacle_curriculum(env, torch.arange(n))
    terrain = env.scene.terrain
    stairs = torch.arange(n) < n // 2
    # 首次 reset 前没有探索回合：所有环境按课程晋级到 2
    assert env.obstacle_base_levels.tolist() == [2] * n
    explore = env.obstacle_explore
    assert not explore[~stairs].any() and 0.3 < explore[stairs].float().mean().item() < 0.7
    # 探索回合的地形等级随机（覆盖高等级），非探索回合等于课程等级；env 原点跟随回合等级
    assert (terrain.terrain_levels[stairs & explore] >= ot.EXPLORE_HIGH_LEVEL).any()
    assert torch.equal(terrain.terrain_levels[~explore], env.obstacle_base_levels[~explore])
    assert torch.equal(terrain.env_origins, terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
    assert torch.equal(ot.curriculum_levels(env), env.obstacle_base_levels)
    # 第二次 reset：上一回合是探索的环境即使通过也不升级；课程日志不计探索回合
    was_explore = explore.clone()
    env.obstacle_curriculum_log = {}
    root = env.scene["robot"].data.root_pos_w
    root[:] = terrain.env_origins
    root[:, 0] += 3.7  # 新回合同样干净通过
    ot.obstacle_curriculum(env, torch.arange(n))
    assert env.obstacle_base_levels[was_explore].tolist() == [2] * int(was_explore.sum())
    assert env.obstacle_base_levels[~was_explore].tolist() == [3] * int((~was_explore).sum())
    log = env.obstacle_curriculum_log
    assert log["stairs_episodes"] + log["stairs_explore_episodes"] == n // 2
    assert log["stairs_explore_passed"] == log["stairs_explore_episodes"] == int(was_explore.sum())
    assert 0 < log["stairs_explore_high_episodes"] < log["stairs_explore_episodes"]


def test_stairs_explore_off_keeps_original_behaviour():
    env = _curriculum_env([3, 3], [ot.KIND_STAIRS, ot.KIND_STAIRS], 0.0, [3.7, 1.0])
    ot.obstacle_curriculum(env, torch.arange(2))
    assert env.scene.terrain.terrain_levels.tolist() == [4, 2] and not env.obstacle_explore.any()
    assert ot.CURRICULUM_DEFAULTS["stairs_explore_frac"] == 0.0
    with pytest.raises(ValueError):
        ot.obstacle_curriculum(_curriculum_env([1], [ot.KIND_STAIRS], 1.0, [3.7]), torch.arange(1))



def test_curriculum_streaks_per_kind_freeze_non_stairs_and_stairs_floor():
    """第十五轮楼梯补练：楼梯迟滞单独设定、只冻结非楼梯、楼梯等级下限；第十四轮的通用迟滞照旧。"""
    kinds = [ot.KIND_STAIRS, ot.KIND_SINGLE, ot.KIND_STAIRS, ot.KIND_SINGLE]
    env = _curriculum_env([3, 3, 1, 3], kinds, 0.0, [3.7] * 4)  # 全部干净通过
    env.cfg.obstacle_curriculum_cfg.update(promote_streak=2.0, demote_streak=2.0, stairs_promote_streak=1.0,
                                           stairs_min_level=4.0)
    ot.obstacle_curriculum(env, torch.arange(4))
    # 楼梯迟滞 1：一次通过即升；等级 1→2 后抬到下限 4；单台阶迟滞 2：第一次通过不升
    assert env.obstacle_base_levels.tolist() == [4, 3, 4, 3]
    root = env.scene["robot"].data.root_pos_w
    root[:] = env.scene.terrain.env_origins
    root[:, 0] += 3.7
    ot.obstacle_curriculum(env, torch.arange(4))
    assert env.obstacle_base_levels.tolist() == [5, 4, 5, 4]
    # 只冻结非楼梯：单台阶通过也不升，楼梯照常
    frozen = _curriculum_env([3, 3], [ot.KIND_STAIRS, ot.KIND_SINGLE], 0.0, [3.7, 3.7])
    frozen.cfg.obstacle_curriculum_cfg.update(freeze_non_stairs=1.0)
    ot.obstacle_curriculum(frozen, torch.arange(2))
    assert frozen.obstacle_base_levels.tolist() == [4, 3]
    levels = torch.tensor([0, 2, 7, 1])
    floor = ot.apply_level_floor(levels, torch.tensor(kinds), {"stairs_min_level": 4.0}, ot.NUM_ROWS)
    assert floor.tolist() == [4, 2, 7, 1]
    assert ot.apply_level_floor(levels, torch.tensor(kinds), ot.CURRICULUM_DEFAULTS, ot.NUM_ROWS) is levels


# ---------------- 终止：成功截断与出车道 ----------------
def test_goal_requires_lane_and_off_lane_goal_terminates():
    rel = torch.tensor([[3.7, 0.10], [3.7, -0.20], [3.7, 0.50], [1.0, 0.45], [1.0, 0.30], [3.59, 0.0]])
    reached, out = ot.goal_and_lane(rel)
    assert reached.tolist() == [True, False, False, False, False, False]
    # 到终点 |y|>0.15 也必须结束（楼梯顶端在块边是断崖），按出车道真终止；0.45 > 0.40 出车道
    assert out.tolist() == [False, True, True, True, False, False]
    assert not (reached & out).any()
    loose = ot.goal_and_lane(rel, {"success_lane_m": 0.40, "fail_lane_m": 0.40})
    assert loose[0].tolist() == [True, True, False, False, False, False]

    class Scene(dict):
        env_origins = torch.zeros(len(rel), 3)

    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.cat((rel, torch.zeros(len(rel), 1)), dim=-1)))
    env = SimpleNamespace(scene=Scene(robot=robot), cfg=SimpleNamespace(obstacle_curriculum_cfg={}))
    assert torch.equal(ot.reached_goal(env), reached) and torch.equal(ot.out_of_lane(env), out)


def test_stairs_top_edge_leaves_no_room_after_goal():
    """3.6 m 时前轮已在块边（4 m）附近：到终点后继续行驶会从楼梯顶端掉下，因此终点必须结束回合。"""
    assert ot.PROMOTE_M + FRONT_X > 3.9 and ot.PROMOTE_M + FRONT_X < 4.0


def test_heading_huber_has_gradient_at_small_yaw():
    """sin² 在 2° 偏航处的梯度只有 Huber 项的约 1/10：小角度修正靠 Huber 项。"""
    flat = _task(lambda x: 0.0)
    terms = _terms(flat, _state(1.0, yaw=math.radians(2.0)))
    s2 = math.sin(math.radians(2.0))
    assert terms["heading_huber"].item() == pytest.approx(-5.0 * (s2 - 0.01), rel=0.02)
    assert terms["heading"].item() == pytest.approx(-30.0 * s2**2, rel=0.02)
    assert _terms(flat, _state(1.0))["heading_huber"].item() == pytest.approx(0.0, abs=1e-8)


def test_stairs_zone_heading_target_is_symmetric_slant():
    """楼梯区航向目标 |ψ|=12°（左右任取）：±12° 不罚、0° 被罚；关掉整宽单台阶斜行后单台阶与平地仍以 0° 为目标。"""
    stairs, single, flat = _task(STAIRS), _task(SINGLE, single_slant_deg=0.0), _task(lambda x: 0.0)
    x = 1.5 - R - 0.30 - FRONT_X
    assert stairs.terrain_context(_state(x))["stairs_zone"].item()
    assert stairs.terrain_context(_state(2.6, z=GROUND_Z + 0.36))["stairs_zone"].item()  # 楼梯上
    assert not stairs.terrain_context(_state(1.5 - R - 0.70 - FRONT_X))["stairs_zone"].item()  # 离第一级 >0.6 m
    assert not single.terrain_context(_state(x))["stairs_zone"].item()
    for yaw in (12.0, -12.0):
        terms = _terms(stairs, _state(x, yaw=math.radians(yaw)))
        assert terms["heading"].item() > -1e-3 and terms["heading_huber"].item() > -1e-3, yaw
    err = 1e-3 - math.sin(math.radians(12.0))
    straight = _terms(stairs, _state(x))
    assert straight["heading"].item() == pytest.approx(-30.0 * err**2, rel=1e-3)
    assert straight["heading_huber"].item() == pytest.approx(-5.0 * (abs(err) - 0.01), rel=1e-3)
    for task in (single, flat):
        assert _terms(task, _state(x))["heading"].item() == pytest.approx(0.0, abs=1e-8)
        assert _terms(task, _state(x, yaw=math.radians(12.0)))["heading"].item() < -1.0


def test_dropping_wheels_do_not_count_as_airborne():
    """整宽块后沿：两前轮刚越过下行边沿、离开低处地面时不算两轮离地；同样抬法在平地上仍算。"""
    arm = _fl_arm_for_lift(0.04)
    lifted = KIN.wheel_centers_base(torch.tensor([arm, 0.0, arm, 0.0]), torch.zeros(4))
    s = _state(2.00 - lifted[0, 0].item(), arm=(arm, 0.0, arm, 0.0))  # 前轮心在块后沿（1.9 m）外 0.10 m
    task = _task(SINGLE)
    ctx = task.terrain_context(s)
    assert ctx["dropping"][0].tolist() == [True, False, True, False]
    assert ctx["lifted"][0, [0, 2]].min().item() > 0.9
    _, airborne, _ = task._support_weights(ctx)
    assert airborne[0, [0, 2]].max().item() == 0.0
    assert _terms(task, s)["multi_lift"].item() == pytest.approx(0.0, abs=1e-4)
    flat = _task(lambda x: 0.0)
    assert not flat.terrain_context(s)["dropping"].any()
    assert _terms(flat, s)["multi_lift"].item() == pytest.approx(-5.0, rel=0.02)


def test_yaw_rate_penalty():
    flat = _task(lambda x: 0.0)
    s = _state(1.0)
    assert _terms(flat, s)["yaw_rate"].item() == 0.0
    s[0, state.ANG.start + 2] = 0.5
    assert _terms(flat, s)["yaw_rate"].item() == pytest.approx(-2.0 * 0.25, rel=1e-6)


# ---------------- 第十一轮：整宽单台阶斜行 / 前轴领先轮预抬 / 平地滚动 / 零速偏航 ----------------
def test_wide_single_step_slants_but_narrow_step_and_far_approach_stay_straight():
    """整宽单台阶（同轴两轮都在单台阶区）航向目标 |ψ|=12°；窄台阶只有一侧的轮看到立面、离立面 >0.6 m，仍以 0° 为目标。"""
    single, narrow = _task(SINGLE), _narrow_task()
    x = 1.5 - R - 0.30 - FRONT_X
    ctx = single.terrain_context(_state(x))
    assert ctx["wide_single_zone"].item() and not ctx["stairs_zone"].item()
    for yaw in (12.0, -12.0):
        assert _terms(single, _state(x, yaw=math.radians(yaw)))["heading"].item() > -1e-3, yaw
    assert _terms(single, _state(x))["heading"].item() < -1.0  # −30·sin²12° ≈ −1.3
    assert not narrow.terrain_context(_state(x))["wide_single_zone"].item()
    assert _terms(narrow, _state(x))["heading"].item() == pytest.approx(0.0, abs=1e-8)
    assert not single.terrain_context(_state(1.5 - R - 0.70 - FRONT_X))["wide_single_zone"].item()


def test_prelift_front_tracks_the_leading_front_wheel():
    """整宽单台阶两前轮都在窗口内：只算 lead 较小（先到立面）的轮。原 prelift_track 取误差较小者 = 落后的那个轮，
    先到的轮贴地撞上去几乎不罚；领先轮抬到目标净空后 prelift_front 清零，落后的轮不抬也不罚。"""
    full = _task(SINGLE)
    s = _state(1.5 - R - 0.10 - FRONT_X, yaw=math.radians(-5.0))
    ctx = full.terrain_context(s)
    lead = ctx["lead"][0]
    assert lead[0] < lead[2] <= 0.12  # 右偏航：左前轮先到
    error, active, _ = full._prelift_errors(ctx)
    assert active[0, [0, 2]].all() and error[0, 0] > error[0, 2] > 0
    assert _terms(full, s)["prelift_front"].item() == pytest.approx(-15.0 * error[0, 0].item(), rel=1e-5)
    assert _terms(full, s)["prelift_track"].item() == pytest.approx(-5.0 * error[0, 2].item(), rel=1e-5)
    lifted = _state(1.5 - R - 0.10 - FRONT_X, yaw=math.radians(-5.0), arm=(_fl_arm_for_lift(0.04), 0.0, 0.0, 0.0))
    assert _terms(full, lifted)["prelift_front"].item() == pytest.approx(0.0, abs=1e-4)
    sg = s.clone().requires_grad_(True)
    grad = torch.autograd.grad(_terms(full, sg)["prelift_front"].sum(), sg)[0][0, state.JPOS]
    assert grad[0].item() < -1.0  # 推左前摆臂往负 θ（抬轮）方向；右前臂的梯度来自支撑净空的三轮平面
    trailing = _state(1.5 - R - 0.10 - FRONT_X, yaw=math.radians(-5.0), arm=(0.0, 0.0, _fl_arm_for_lift(0.04), 0.0))
    assert _terms(full, trailing)["prelift_front"].item() <= _terms(full, s)["prelift_front"].item() + 1e-6  # 抬落后的轮没用
    # 窄台阶：面对立面的那个前轮；楼梯：不算（楼梯前轴预抬仍由 prelift_track 负责）
    x = 1.5 - R - 0.08 - FRONT_X
    assert _terms(_narrow_task(), _state(x))["prelift_front"].item() == pytest.approx(-15.0 * 0.75, rel=0.02)
    assert _terms(_task(STAIRS), _state(x))["prelift_front"].item() == 0.0
    assert "prelift_front" not in _terms(_task(SINGLE, prelift_front=0.0), _state(x))


def _rolling_state(vx=0.3, steer=(0.0, 0.0, 0.0, 0.0), wheel=None, x=1.0, **kw):
    s = _state(x, vx=vx, **kw)
    s[0, state.JPOS.start + 4: state.JPOS.start + 8] = torch.tensor(steer)
    wheel = (vx / R,) * 4 if wheel is None else wheel
    s[0, state.JVEL.start + 8: state.JVEL.start + 12] = torch.tensor(wheel)
    return s


def test_flat_rolling_penalizes_toe_and_wheel_slip_only_on_flat():
    """平地：同轴两轮反向打角（toe）与轮缘/轮心速度差（打滑）计代价；纯滚动、蟹行不罚；台阶附近（平地门外）不罚。"""
    flat = _task(lambda x: 0.0)
    rolling = _terms(flat, _rolling_state())
    for name in ("toe", "flat_steer", "wheel_sync", "wheel_slip"):
        assert rolling[name].item() == pytest.approx(0.0, abs=1e-6), name
    crab = _terms(flat, _rolling_state(steer=(0.2, 0.2, 0.2, 0.2), wheel=(0.3 * math.cos(0.2) / R,) * 4))
    assert crab["toe"].item() == pytest.approx(0.0, abs=1e-8) and crab["wheel_slip"].item() == pytest.approx(0.0, abs=1e-6)
    assert crab["wheel_sync"].item() == pytest.approx(0.0, abs=1e-6)
    assert crab["flat_steer"].item() == pytest.approx(-10.0 * 4 * 0.2**2, rel=0.01)  # 平地蟹行也有转向代价（平地门 ≈0.995）
    # 第十轮探针：后轮 ±43° 八字
    toe = _terms(flat, _rolling_state(steer=(0.0, 0.75, 0.0, -0.75)))["toe"].item()
    assert toe == pytest.approx(-20.0 * 1.5**2, rel=0.01)
    # 前轮两倍转速：每个前轮打滑 0.3 m/s → smoothL1 = 0.3 − 0.025；四轮轮缘速度偏离均值 ±0.15 → 4 × (0.15 − 0.025)
    spun = _terms(flat, _rolling_state(wheel=(0.6 / R, 0.3 / R, 0.6 / R, 0.3 / R)))
    assert spun["wheel_slip"].item() == pytest.approx(-20.0 * 2 * 0.275, rel=0.01)
    assert spun["wheel_sync"].item() == pytest.approx(-50.0 * 4 * 0.125, rel=0.01)
    # 四轮同速只看关节速度，不经车身速度（世界模型的车速预测）
    sg = _rolling_state(wheel=(0.6 / R, 0.3 / R, 0.6 / R, 0.3 / R)).requires_grad_(True)
    g_sync = torch.autograd.grad(_terms(flat, sg)["wheel_sync"].sum(), sg)[0][0]
    assert g_sync[state.LIN].abs().max().item() == 0.0 and g_sync[state.JVEL][8].item() < 0 < g_sync[state.JVEL][9].item()
    # 零速前后轴对顶：前轮正转、后轮反转
    fight = _terms(flat, _rolling_state(vx=0.0, wheel=(2.3, -1.0, 2.3, -1.0)), torch.zeros(1, 3))["wheel_slip"].item()
    assert fight < -2.0
    sg = _rolling_state(wheel=(0.6 / R, 0.3 / R, 0.6 / R, 0.3 / R)).requires_grad_(True)
    grad = torch.autograd.grad(_terms(flat, sg)["wheel_slip"].sum(), sg)[0][0, state.JVEL]
    assert grad[8].item() < 0 and grad[9].abs().item() < 1e-5  # 推前轮减速，后轮已在滚动
    # 台阶附近：平地门关闭
    near = _terms(_task(SINGLE), _rolling_state(steer=(0.0, 0.75, 0.0, -0.75), wheel=(0.6 / R, 0.3 / R, 0.6 / R, 0.3 / R),
                                                 x=1.5 - R - 0.30 - FRONT_X))
    for name in ("toe", "flat_steer", "wheel_sync", "wheel_slip"):
        assert near[name].item() == pytest.approx(0.0, abs=1e-3), name
    off = _terms(_task(lambda x: 0.0, toe=0.0, wheel_slip=0.0, flat_steer=0.0, wheel_sync=0.0), _rolling_state())
    assert not {"toe", "wheel_slip", "flat_steer", "wheel_sync"} & set(off)


def test_prelift_support_clearance_needs_com_inside_the_support_triangle():
    """接触前：质心不在其余三轮三角形内（裕度 ≤ 0）时支撑净空不作数，实际净空也得达标（两者取大）；
    裕度 ≥ prelift_margin_m 时回到两者平均；接触后翻越中不门控。"""
    task = _task(SINGLE)

    def errors(lead, margin, clearance=0.0, support=0.04, single=True):
        ctx = {"lead": torch.tensor([[lead, 1.0, 1.0, 1.0]]), "riser_rise": torch.full((1, 4), 0.12),
               "single": torch.tensor([[single, False, False, False]]),
               "clearance": torch.tensor([[clearance, 0.0, 0.0, 0.0]]),
               "support_clearance": torch.tensor([[support, 0.0, 0.0, 0.0]]),
               "support_margin": torch.tensor([[margin, 0.0, 0.0, 0.0]])}
        return task._prelift_errors(ctx)[0][0, 0].item()

    full = errors(0.06, margin=0.0, support=0.0)  # 贴地：目标 3 cm → smoothL1(1; 0.5) = 0.75
    assert full == pytest.approx(0.75, rel=1e-4)
    # 第十一轮：摆臂抬了（支撑净空 4 cm）、车身倾倒（实际净空 0）
    assert errors(0.06, margin=0.0) == pytest.approx(full, rel=1e-4)  # 质心在对角线上：不给分
    assert errors(0.06, margin=-0.01) == pytest.approx(full, rel=1e-4)
    assert errors(0.06, margin=0.008) == pytest.approx(0.5 * full, rel=1e-4)  # 质心进了三角形：支撑净空算一半
    assert 0.5 * full < errors(0.06, margin=0.004) < full
    assert errors(0.06, margin=0.0, clearance=0.04) == pytest.approx(0.0, abs=1e-6)  # 真的离地：不罚
    # 翻越中（lead ≤ 0，立面角托住；目标 = 立面圆周包络约 7.9 cm）：不门控，支撑净空达标仍是两者平均
    climb_full = errors(-0.02, margin=0.0, support=0.0)
    assert errors(-0.02, margin=0.0, support=0.085) == pytest.approx(0.5 * climb_full, rel=1e-4)
    # 楼梯前轴（非单台阶）接触前不门控：支撑净空仍算一半（第十一轮口径）
    assert errors(0.06, margin=0.0, single=False) == pytest.approx(0.5 * full, rel=1e-4)


def test_stand_yaw_rate_only_for_zero_command():
    flat = _task(lambda x: 0.0)
    s = _state(1.0)
    s[0, state.ANG.start + 2] = 0.04
    assert _terms(flat, s, torch.zeros(1, 3))["stand_yaw_rate"].item() == pytest.approx(-5.0 * (0.04 - 0.01), rel=1e-5)
    assert _terms(flat, s)["stand_yaw_rate"].item() == 0.0  # 有前进命令时由航向项负责
