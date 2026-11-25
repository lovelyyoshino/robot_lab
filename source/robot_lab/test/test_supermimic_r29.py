"""
test_supermimic_r29.py — 第二十九轮：卡住两项。

- obstacle_terrain v1.8.0 短踏面楼梯列（STAIRS_SHORT）：算楼梯、常驻探索（随机等级、不改课程、不进楼梯课程统计），
  列分配与 Isaac Lab 一致（楼梯 9 列 = 8 + 1）；
- ProgressStallTracker：按本回合最远前进 x 计时，倒车、原地、前后蹭不清零，再前进 ≥ gain_m、未命令前进、回合结束清零；
- stall_hold_scale：宽限后线性升到 1；
- train_terrain v0.24.0 / run_train v2.20.0 的开关与配方。

@author pony
@date 2026-11-23
@version v1.1.0
@last_modified 2026-11-23

@changelog
    - v1.1.0 (2026-11-23): 第三十轮去掉短踏面列（--stairs_short_frac 0），launcher 配方断言不再钉 0.05（见 test_supermimic_r30）
    - v1.0.0 (2026-11-23): 初始创建
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))

import test_supermimic_obstacle as T  # noqa: E402

ot = T.ot
REPO = Path(__file__).resolve().parents[3]


def _mix29():
    return ot.split_short_stairs(dict(ot.MIX), 0.05)


def test_split_short_stairs_keeps_total_and_order():
    mix = _mix29()
    assert list(mix) == ["single_step", "single_step_left", "single_step_right", ot.STAIRS, ot.STAIRS_SHORT, "flat"]
    assert sum(mix.values()) == pytest.approx(sum(ot.MIX.values()))
    assert mix[ot.STAIRS] == pytest.approx(0.40) and mix[ot.STAIRS_SHORT] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ot.split_short_stairs(mix, 0.05)  # 已经加过
    with pytest.raises(ValueError):
        ot.split_short_stairs(dict(ot.MIX), 0.5)  # 超过楼梯比例


def test_short_stairs_column_is_stairs_and_forced_explore():
    mix = _mix29()
    names, props = list(mix), list(mix.values())
    cols = ot.column_names(names, props, ot.NUM_COLS)
    kinds = ot.column_kinds(names, props, ot.NUM_COLS)
    explore = ot.column_explore(names, props, ot.NUM_COLS)
    assert cols.count(ot.STAIRS) == 8 and cols.count(ot.STAIRS_SHORT) == 1
    assert kinds.count(ot.KIND_STAIRS) == 9  # 楼梯总列数与第二十八轮相同
    assert kinds.count(ot.KIND_SINGLE) == 8 and kinds.count(ot.KIND_FLAT) == 3
    assert sum(explore) == 1 and kinds[explore.index(True)] == ot.KIND_STAIRS
    # 原配置：没有常驻探索列；加短踏面列只把楼梯的最后一列换掉，其余列的类型不变
    base = ot.column_kinds(list(ot.MIX), list(ot.MIX.values()), ot.NUM_COLS)
    assert not any(ot.column_explore(list(ot.MIX), list(ot.MIX.values()), ot.NUM_COLS))
    assert base == kinds


def test_short_stairs_dimensions_without_tread_rule():
    for d in (0.0, 0.3, 0.7, 0.999):
        riser, tread = ot.stairs_dimensions(d, (0.14, 0.175), (0.26, 0.30), seed=7, tread_rule=None)
        assert 0.14 <= riser <= 0.175 and 0.26 <= tread <= 0.30
    # 对照：主楼梯的踏面规则让 ≥15 cm 立面踏面 ≥36 cm
    _, tread = ot.stairs_dimensions(0.9, ot.STAIR_RISER_M, ot.STAIR_TREAD_M, seed=7, tread_rule=(0.12, 0.15, 0.36))
    assert tread >= 0.36


def _short_env(n_short, n_stairs, x_progress):
    """短踏面列 + 普通楼梯列的假环境（复用 test_supermimic_obstacle 的构造，再换成 r29 的列配置）。"""
    mix = _mix29()
    names, props = list(mix), list(mix.values())
    kinds = ot.column_kinds(names, props, ot.NUM_COLS)
    explore = ot.column_explore(names, props, ot.NUM_COLS)
    short_col, stairs_col = explore.index(True), kinds.index(ot.KIND_STAIRS)
    n = n_short + n_stairs
    env = T._curriculum_env([3] * n, [ot.KIND_STAIRS] * n, 0.0, x_progress)
    terrain = env.scene.terrain
    terrain.terrain_types = torch.tensor([short_col] * n_short + [stairs_col] * n_stairs)
    terrain.env_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types].clone()
    root = env.scene["robot"].data.root_pos_w
    root[:] = terrain.env_origins
    root[:, 0] += torch.tensor(x_progress)
    env.cfg.column_kinds, env.cfg.column_explore = kinds, explore
    return env


def test_short_column_episodes_never_move_the_curriculum_and_are_logged_apart():
    torch.manual_seed(0)
    n_short, n_stairs = 200, 4
    env = _short_env(n_short, n_stairs, [3.7] * (n_short + n_stairs))  # 全部干净通过
    short = torch.arange(n_short + n_stairs) < n_short
    assert torch.equal(ot.env_forced_explore(env), short)
    ot.obstacle_curriculum(env, torch.arange(n_short + n_stairs))
    # 第一回合就算探索：短踏面列不升级，普通楼梯升级；短踏面列抽到随机等级
    assert env.obstacle_base_levels[short].tolist() == [3] * n_short
    assert env.obstacle_base_levels[~short].tolist() == [4] * n_stairs
    assert env.obstacle_explore[short].all() and not env.obstacle_explore[~short].any()
    levels = env.scene.terrain.terrain_levels
    assert levels[short].unique().numel() >= 5
    log = env.obstacle_curriculum_log
    assert log["stairs_episodes"] == n_stairs and log["stairs_short_episodes"] == n_short
    assert log["stairs_short_passed"] == n_short and log.get("stairs_explore_episodes", 0) == 0
    # 课程统计不含短踏面列
    summary = ot.curriculum_summary(ot.curriculum_levels(env), ot.summary_kinds(env))
    assert summary["stairs_level"].item() == pytest.approx(4.0)
    assert summary["mean_level"].item() == pytest.approx(4.0)


def test_curriculum_summary_ignores_negative_kinds_only():
    levels = torch.tensor([2, 4, 9])
    kinds = torch.tensor([ot.KIND_STAIRS, ot.KIND_STAIRS, -1])
    out = ot.curriculum_summary(levels, kinds)
    assert out["stairs_level"].item() == pytest.approx(3.0) and out["mean_level"].item() == pytest.approx(3.0)
    assert out["stairs_top_fraction"].item() == 0.0
    same = ot.curriculum_summary(levels, torch.tensor([ot.KIND_STAIRS] * 3))
    assert same["mean_level"].item() == pytest.approx(5.0)


def _run(tracker, xs, commanded=None, done=None):
    out = []
    for i, x in enumerate(xs):
        c = torch.tensor([True if commanded is None else commanded[i]])
        d = torch.tensor([False if done is None else done[i]])
        out.append(int(tracker.update(torch.tensor([x]), c, d)[0]))
    return out


def test_progress_stall_tracker_counts_from_farthest_x():
    tr = ot.ProgressStallTracker(1, "cpu", gain_m=0.01)
    # 前进 → 停住 → 倒车 → 顶回原处（没超过最远处 1 cm）→ 超过最远处
    xs = [0.0, 0.02, 0.04, 0.04, 0.04, 0.02, 0.0, 0.03, 0.045, 0.051]
    assert _run(tr, xs) == [0, 0, 0, 1, 2, 3, 4, 5, 6, 0]


def test_progress_stall_tracker_resets_on_no_command_and_done():
    tr = ot.ProgressStallTracker(1, "cpu", gain_m=0.01)
    assert _run(tr, [0.0, 0.0, 0.0], commanded=[True, True, False]) == [0, 1, 0]
    assert _run(tr, [0.0, 0.0], commanded=[True, True]) == [1, 2]
    # 回合结束：计时清零，下一回合从新起点重算最远处
    assert _run(tr, [0.0, 5.0, 5.0], done=[True, False, False]) == [0, 0, 1]
    with pytest.raises(ValueError):
        ot.ProgressStallTracker(1, "cpu", gain_m=0.0)


def test_stall_hold_scale_ramps_after_grace():
    steps = torch.tensor([0, 50, 75, 100, 150, 400])
    out = ot.stall_hold_scale(steps, 0.02, 1.0, 2.0)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.25, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError):
        ot.stall_hold_scale(steps, 0.02, 1.0, 0.0)


def test_run_train_keeps_r29_stall_hold_and_flags():
    text = (REPO / "scripts" / "supermimic" / "run_train.sh").read_text()
    for flag in ("--stall_hold -20.0 --stall_hold_grace_s 1.0 --stall_hold_ramp_s 2.0 --stall_hold_gain_m 0.01\n",
                 "--stairs_tread_rule 0.12 0.15 0.36\n", "--curriculum_goal_lane_termination 0\n", "--course -20.0\n",
                 "--stall -5.0 --stall_speed_ratio 0.25\n", "--stall_climb_scale 0.2 --drop_relief 0.8\n"):
        assert flag in text, flag
    train = (REPO / "scripts" / "supermimic" / "train_terrain.py").read_text()
    for needle in ('"--stairs_short_frac"', '"--stall_hold"', "add_short_stairs(env_cfg", "ProgressStallTracker(",
                   'terms["stall_hold"]', "summary_kinds(uenv)", '"stairs_short"'):
        assert needle in train, needle
    cfg = (REPO / "source" / "robot_lab" / "robot_lab" / "tasks" / "manager_based" / "supermimic"
           / "obstacle_env_cfg.py").read_text()
    assert "def add_short_stairs" in cfg and "refresh_columns(self)" in cfg and "tread_rule=None" in cfg
