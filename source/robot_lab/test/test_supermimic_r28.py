"""
test_supermimic_r28.py — 第二十八轮：偏离只扣分、不判失败。

- obstacle_terrain v1.7.0 goal_lane_termination：0 时到终点一律成功截断，只有 |y| > fail_lane_m 终止；默认 1 = 原行为；
- 晋级横偏门 0.40 m、出车道 0.75 m 时的课程判定，楼梯晋级门的时间表 0 0；
- preview_task v0.24.0 course = w·sin²(ψ + δ̄)：只看四轮转向平均（同相），反相不计，左右镜像不变；
- 转向角正 = 轮头朝左：蟹行且车体速度沿轮向时不算打滑；
- 验收软判（acceptance_metrics v1.3.0）：soft_ok 第一次为真即软成功，按软成功时 |y| 扣分，旧样本没有 soft_ok 也能跑。

@author pony
@date 2026-11-21
@version v1.0.0
@last_modified 2026-11-21

@changelog
    - v1.0.0 (2026-11-21): 初始创建
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

import test_supermimic_obstacle as T  # noqa: E402

ot = T.ot
FLAT = lambda x: 0.0  # noqa: E731
R28 = {"success_lane_m": 0.40, "fail_lane_m": 0.75, "goal_lane_termination": 0.0}


def _rel(*xy):
    return torch.tensor(xy, dtype=torch.float32)


def test_goal_lane_termination_default_keeps_old_behaviour():
    rel = _rel((3.7, 0.20), (3.7, 0.10), (2.0, 0.50))
    reached, out = ot.goal_and_lane(rel)
    assert reached.tolist() == [False, True, False] and out.tolist() == [True, False, True]
    assert ot.CURRICULUM_DEFAULTS["goal_lane_termination"] == 1.0


def test_goal_lane_termination_off_truncates_at_goal_regardless_of_lateral():
    rel = _rel((3.7, 0.20), (3.7, 0.60), (3.7, -0.80), (2.0, 0.60), (2.0, 0.80))
    reached, out = ot.goal_and_lane(rel, R28)
    assert reached.tolist() == [True, True, False, False, False]
    assert out.tolist() == [False, False, True, False, True]
    assert not (reached & out).any()
    # 楼梯单独的成功车道不影响终止（只影响晋级）
    kinds = torch.full((5,), ot.KIND_STAIRS)
    assert torch.equal(ot.goal_and_lane(rel, {**R28, "stairs_success_lane_m": 0.15}, kinds)[0], reached)


def test_curriculum_with_wide_lanes_promotes_by_040_and_demotes_only_off_platform():
    cfg = {**ot.CURRICULUM_DEFAULTS, **R28}
    progress = torch.tensor([3.7, 3.7, 3.7, 0.5])
    lateral = torch.tensor([0.30, 0.50, 0.80, 0.10])
    ones = torch.ones(4, dtype=torch.bool)
    kinds = torch.full((4,), ot.KIND_SINGLE)
    passed, failed = ot.curriculum_decision(progress, lateral, ones, ones, kinds, None, cfg)
    assert passed.tolist() == [True, False, False, False]
    assert failed.tolist() == [False, False, True, True]  # 0.50 m 到顶：不升不降；0.80 m 开出台面、0.5 m 进度不足：降


def test_stairs_success_lane_schedule_zero_means_strict_from_start():
    fail, strict = R28["fail_lane_m"], R28["success_lane_m"]
    scale = ot.lane_schedule_scale(0, 0, 0)
    assert scale == 1.0 and fail + (strict - fail) * scale == pytest.approx(0.40)
    assert ot.lane_schedule_scale(39999, 90000, 100000) == 0.0  # r27 的写法在 fail 0.75 下会把楼梯门放到 0.75


def _course_state(yaw, steer):
    s = T._state(1.0, vx=0.3, yaw=yaw)
    s[0, T.state.JPOS.start + 4: T.state.JPOS.start + 8] = torch.tensor(steer)
    return s


def test_course_error_uses_heading_plus_mean_steer_only():
    task = T._task(FLAT)
    aligned = task.course_error(_course_state(0.2, (-0.2,) * 4))
    assert aligned.item() == pytest.approx(0.0, abs=1e-5)
    # 反相（前后反向打，转车头）不进来：前轮 fl/fr +0.1、后轮 rl/rr −0.1，平均 0
    counter = task.course_error(_course_state(0.2, (0.1, -0.1, 0.1, -0.1)))
    assert counter.item() == pytest.approx(math.sin(0.2), rel=1e-3)
    assert task.course_error(_course_state(0.1, (0.1,) * 4)).item() == pytest.approx(math.sin(0.2), rel=1e-3)
    # 左右镜像（ψ、δ 同时取反）平方不变
    a = task.course_error(_course_state(0.15, (0.05, 0.02, 0.07, -0.01))).square()
    b = task.course_error(_course_state(-0.15, (-0.05, -0.02, -0.07, 0.01))).square()
    assert a.item() == pytest.approx(b.item(), rel=1e-5)


def test_course_term_off_by_default_and_weighted_when_on():
    s = _course_state(0.15, (0.0,) * 4)
    assert T.PreviewRewardCfg().course == 0.0 and "course" not in T._terms(T._task(FLAT), s)
    terms = T._terms(T._task(FLAT, course=-20.0), s)
    assert terms["course"].item() == pytest.approx(-20.0 * math.sin(0.15) ** 2, rel=1e-3)
    crab = T._terms(T._task(FLAT, course=-20.0), _course_state(0.15, (-0.15,) * 4))
    assert crab["course"].item() == pytest.approx(0.0, abs=1e-6)


def test_course_gradient_points_steering_against_heading():
    task = T._task(FLAT, course=-20.0)
    s = _course_state(0.15, (0.0,) * 4).requires_grad_(True)
    T._terms(task, s)["course"].sum().backward()
    grad = s.grad[0, T.state.JPOS.start + 4: T.state.JPOS.start + 8]
    assert (grad < 0).all()  # 车头偏左（ψ > 0）时，转向往负（朝右）打奖励更高


def test_wheel_slip_treats_positive_steer_as_left():
    """转向角正 = 轮头朝左（URDF 正运动学：+0.2 rad 时滚动方向 +11.5°）。车体速度沿轮向蟹行、轮缘速度 = 车速时不算打滑。"""
    flat = T._task(FLAT)
    d, v = 0.2, 0.3
    crab = T._terms(flat, T._rolling_state(vx=v * math.cos(d), vy=v * math.sin(d), steer=(d,) * 4, wheel=(v / T.R,) * 4))
    assert crab["wheel_slip"].item() == pytest.approx(0.0, abs=1e-4)
    wrong = T._terms(flat, T._rolling_state(vx=v * math.cos(d), vy=-v * math.sin(d), steer=(d,) * 4,
                                             wheel=(v / T.R,) * 4))
    assert wrong["wheel_slip"].item() < -0.1  # 车往右滑、轮子朝左：沿轮向速度 v·cos2δ，打滑 ≈ 0.024 m/s × 4 轮


def test_run_train_r28_recipe():
    text = (Path(__file__).resolve().parents[3] / "scripts" / "supermimic" / "run_train.sh").read_text()
    for flag in ("--curriculum_success_lane_m 0.40 --curriculum_fail_lane_m 0.75\n", "--curriculum_goal_lane_termination 0\n",
                 "--stairs_lane_schedule 2000 8000 --stairs_success_lane_schedule 0 0\n", "--course -20.0\n",
                 "--lane -30.0 --steering -0.1\n"):
        assert flag in text, flag
    train = (Path(__file__).resolve().parents[3] / "scripts" / "supermimic" / "train_terrain.py").read_text()
    assert '("course", float)' in train and '"goal_lane_termination")' in train


# ---------------------------------------------------------------- 验收软判（只记录，原判不变）
RSL = Path(__file__).resolve().parents[3] / "scripts" / "reinforcement_learning" / "rsl_rl"


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(RSL))
    import acceptance_metrics

    return acceptance_metrics


def _steps(ys, soft):
    return {"y": list(ys), "soft_ok": list(soft)}


@pytest.mark.parametrize("y,score", [(0.10, 1.0), (-0.15, 1.0), (0.20, 0.90), (0.40, 0.50), (0.70, 0.50)])
def test_soft_score_deducts_002_per_cm_beyond_015(metrics, y, score):
    out = metrics.soft_fields(_steps([0.0, y, y], [False, True, True]), 0.02)
    assert out["soft_success"] and out["soft_score"] == pytest.approx(score)
    assert out["soft_y_m"] == y and out["soft_time_s"] == pytest.approx(0.04)


def test_soft_fields_without_soft_column_or_success(metrics):
    assert metrics.soft_fields({"y": [0.1, -0.3]}, 0.02) == {"max_abs_y_m": 0.3}
    out = metrics.soft_fields(_steps([0.1, 0.2], [False, False]), 0.02)
    assert out["soft_success"] is False and out["soft_score"] == 0.0 and out["max_abs_y_m"] == pytest.approx(0.2)


def _sample(y, soft=None):
    base = {"roll": [0.0], "pitch": [0.0], "pitch_rate": [0.0], "vz": [0.0], "arm_vel": [[0.0] * 4],
            "action": [[0.0] * 12], "riser_contact": [[False] * 4], "tilt_deg": [1.0], "x": [3.0], "y": [y],
            "finite": [True]}
    return base if soft is None else {**base, "soft_ok": [soft]}


def test_episode_metrics_records_soft_success_and_aggregates(monkeypatch, metrics):
    monkeypatch.syspath_prepend(str(RSL))
    from skill_evaluation import SkillEvaluation

    traversal = SkillEvaluation(1, 2, 0.02, 0.02, 2)
    recorder = metrics.EpisodeMetrics(1, 0.02, climb_tilt_deg=45.0, target_distance_m=2.5)
    # 第一回合：横偏 0.25 m 稳住（软成功）后冲出终止（严格失败）
    for y, soft, term in ((0.2, False, False), (0.25, True, False), (0.3, True, True)):
        traversal.update([False], [term], [False], [True])
        recorder.update(_sample(y, soft), traversal)
    # 第二回合：没稳住就超时
    for y, trunc in ((0.1, False), (0.1, True)):
        traversal.update([False], [False], [trunc], [True])
        recorder.update(_sample(y, False), traversal)
    first, second = recorder.records
    assert first["success"] is False and first["soft_success"] is True
    assert first["soft_y_m"] == 0.25 and first["soft_score"] == pytest.approx(0.8) and first["max_abs_y_m"] == 0.3
    assert second["soft_success"] is False and second["soft_score"] == 0.0
    agg = metrics.aggregate_episodes(recorder.records)
    assert agg["soft_successes"] == 1 and agg["soft_score_mean"] == pytest.approx(0.4) and agg["successes"] == 0


def test_episode_metrics_without_soft_column_keeps_old_records(monkeypatch, metrics):
    monkeypatch.syspath_prepend(str(RSL))
    from skill_evaluation import SkillEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    recorder = metrics.EpisodeMetrics(1, 0.02)
    traversal.update([False], [True], [False], [True])
    recorder.update(_sample(0.1), traversal)
    assert "soft_success" not in recorder.records[0]
    assert "soft_successes" not in metrics.aggregate_episodes(recorder.records)
