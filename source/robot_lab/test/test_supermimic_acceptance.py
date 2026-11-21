"""SuperMimic 越障验收指标单测：全部用合成数组，不启动 Isaac。

覆盖 P95 聚合（非中位数）、坡度相对俯仰门、四轮踢面碰撞计数、预抬 8–14 cm 窗口与无碰撞联合判定、
232/233 维识别、留出种子与 accept.py 计划/汇总。

@author pony
@date 2026-11-05
@version v1.0.0
@last_modified 2026-11-05
@changelog
    - v1.0.0 (2026-11-05): 初始创建
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).parents[3]
RSL = REPO / "scripts/reinforcement_learning/rsl_rl"


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(RSL))
    import acceptance_metrics

    return acceptance_metrics


@pytest.fixture
def stairs(metrics):
    return metrics.load_stairs_acceptance()


@pytest.fixture
def accept(monkeypatch, metrics):
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))  # accept.py 导入时改 sys.path，测试后还原
    spec = importlib.util.spec_from_file_location("accept_under_test", REPO / "scripts/supermimic/accept.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- P95
def test_percentile_matches_numpy_and_ignores_nonfinite(metrics):
    rng = np.random.default_rng(0)
    values = rng.normal(size=257).tolist()
    for q in (0, 5, 50, 95, 100):
        assert metrics.percentile(values, q) == pytest.approx(np.percentile(values, q))
    assert metrics.p95(values + [float("nan"), None, float("inf")]) == pytest.approx(np.percentile(values, 95))
    assert metrics.p95([]) is None and metrics.p95([None, float("nan")]) is None
    assert metrics.rms([3.0, -4.0]) == pytest.approx(math.sqrt(12.5))


def test_aggregate_uses_p95_not_median(metrics):
    # 80% 回合很稳，20% 回合大幅侧倾：中位数会掩盖，P95 必须暴露
    episodes = [{"roll_abs_p95_deg": 1.0, "success": True, "outcome": "success"} for _ in range(80)]
    episodes += [{"roll_abs_p95_deg": 12.0, "success": False, "outcome": "rollover"} for _ in range(20)]
    agg = metrics.aggregate_episodes(episodes, metrics=("roll_abs_p95_deg",))
    assert agg["p95"]["roll_abs_p95_deg"] == pytest.approx(12.0)
    assert agg["p95_success_only"]["roll_abs_p95_deg"] == pytest.approx(1.0)
    assert agg["outcomes"] == {"success": 80, "rollover": 20}
    rows = metrics.tilt_smooth_checks("flat", agg["p95"], case="c")
    assert next(r for r in rows if r["metric"] == "roll_abs_p95_deg")["passed"] is False
    # 缺值的门判失败，非门项不判
    assert next(r for r in rows if r["metric"] == "arm_vel_rms_rad_s")["passed"] is False
    assert next(r for r in rows if r["metric"] == "base_vz_rms_m_s")["passed"] is None


# ---------------------------------------------------------------- 楼梯坡度相对俯仰门
@pytest.mark.parametrize("riser,tread,slope", [(0.175, 0.28, 32.0054), (0.08, 0.40, 11.3099), (0.16, 0.40, 21.8014)])
def test_stairs_slope_relative_pitch_gate(stairs, metrics, riser, tread, slope):
    gates = stairs.stairs_pitch_gates(riser, tread)
    assert gates["slope_deg"] == pytest.approx(slope, abs=1e-3)
    assert gates["pitch_abs_p95_deg"] == pytest.approx(slope - 5.0, abs=1e-3)
    assert gates["pitch_abs_max_deg"] == pytest.approx(slope + 3.0, abs=1e-3)
    good = {"pitch_abs_p95_deg": slope - 5.1, "pitch_abs_max_deg": slope + 2.9}
    bad = {"pitch_abs_p95_deg": slope - 4.9, "pitch_abs_max_deg": slope + 3.1}
    for values, expected in ((good, True), (bad, False)):
        rows = metrics.tilt_smooth_checks("stairs", values, case="s", pitch_limits=gates)
        assert all(r["passed"] is expected for r in rows if r["metric"].startswith("pitch_abs"))


def test_stairs_geometry_and_climb_tilt(stairs):
    assert stairs.climb_tilt_for_riser(0.08) == stairs.climb_tilt_for_riser(0.12) == 31.8
    assert stairs.climb_tilt_for_riser(0.16) == stairs.climb_tilt_for_riser(0.175) == 45.0
    assert len(stairs.ACCEPTANCE_STAIR_RISERS_M) * len(stairs.ACCEPTANCE_STAIR_TREADS_M) == 8
    with pytest.raises(ValueError):
        stairs.stairs_pitch_gates(0.02, 0.40)  # 坡度 2.9° < 5°，门为负
    with pytest.raises(ValueError):
        stairs.stair_slope_deg(0.1, 0.0)


def test_step_pitch_gate_only_up_to_12cm_and_arm_thresholds(metrics):
    values = {k: 0.0 for k in metrics.EPISODE_METRICS}
    values["pitch_abs_p95_deg"] = 9.0
    low = metrics.tilt_smooth_checks("step", values, case="s10", step_height_m=0.10)
    high = metrics.tilt_smooth_checks("step", values, case="s14", step_height_m=0.14)
    assert next(r for r in low if r["metric"] == "pitch_abs_p95_deg")["passed"] is False
    assert next(r for r in high if r["metric"] == "pitch_abs_p95_deg")["gating"] is False
    values["arm_vel_rms_rad_s"] = 0.6
    flat = metrics.tilt_smooth_checks("flat", values, case="f")
    step = metrics.tilt_smooth_checks("step", values, case="s", step_height_m=0.10)
    assert next(r for r in flat if r["metric"] == "arm_vel_rms_rad_s")["passed"] is False
    assert next(r for r in step if r["metric"] == "arm_vel_rms_rad_s")["passed"] is True


# ---------------------------------------------------------------- 踢面碰撞
def contact(metrics, centers, resisting, risers, radius=0.1085):
    return metrics.riser_contact_mask(torch.tensor(centers, dtype=torch.float32), torch.full((len(centers[0]),), radius),
                                      torch.tensor(resisting, dtype=torch.float32), risers,
                                      **metrics.RISER_CONTACT).tolist()


def test_single_step_riser_contact_requires_force_low_bottom_and_proximity(metrics):
    r = 0.1085
    risers = metrics.single_step_risers(1.5, 0.14, (-1.5, 1.5))
    touching = 1.5 - r - 0.002          # 轮前沿距立面 2 mm（在 1.5 cm 接触余量内）
    centers = [[[touching, 0.3, r], [touching, 0.3, r + 0.15], [1.2, 0.3, r], [touching, 0.3, r]]]
    # 轮0 顶住；轮1 轮底已高于踢面顶；轮2 离立面 19 cm；轮3 阻力不足
    assert contact(metrics, centers, [[30.0, 30.0, 30.0, 2.0]], risers) == [[True, False, False, False]]
    # 轮心越过立面（已在台阶顶上）不算
    assert contact(metrics, [[[1.55, 0.3, r + 0.14]] * 4], [[30.0] * 4], risers) == [[False] * 4]
    # 非有限测量不算碰撞
    assert contact(metrics, [[[float("nan"), 0.3, r]] * 4], [[30.0] * 4], risers) == [[False] * 4]


def test_narrow_step_only_counts_wheels_inside_lane(metrics):
    r = 0.1085
    risers = metrics.single_step_risers(1.5, 0.10, (0.12, 0.48))  # 左窄台阶
    x = 1.5 - r
    centers = [[[x, 0.30, r], [x, 0.30, r], [x, -0.30, r], [x, -0.30, r]]]  # fl, rl 在车道内；fr, rr 在右侧
    assert contact(metrics, centers, [[20.0] * 4], risers) == [[True, True, False, False]]


def test_stairs_contact_uses_next_riser_top(metrics):
    r = 0.1085
    risers = metrics.stairs_risers(1.2, 0.28, 0.175, 4, 3.0)
    assert risers["x"] == pytest.approx([1.2, 1.48, 1.76, 2.04])
    assert risers["top_z"] == pytest.approx([0.175, 0.35, 0.525, 0.70])
    x = 1.48 - r
    # 站在第 1 级（0.175）上顶第 2 级立面：轮底 0.175 < 0.35 → 碰撞；轮底 0.36 → 已越过高度
    centers = [[[x, 0.0, 0.175 + r], [x, 0.0, 0.36 + r], [x, 0.0, 0.175 + r], [x, 0.0, 0.175 + r]]]
    assert contact(metrics, centers, [[10.0, 10.0, 0.0, 10.0]], risers) == [[True, False, False, True]]


def test_episode_statistics_counts_contact_events_and_skips_warmup(metrics):
    n = 10
    steps = {
        "roll": [math.radians(20.0)] * 2 + [math.radians(1.0)] * (n - 2),   # 出生 2 步大侧倾不计
        "pitch": [0.0] * n, "pitch_rate": [math.radians(4.0)] * n, "vz": [0.01] * n,
        "arm_vel": [[0.1, -0.1, 0.1, -0.1]] * n,
        "action": [[0.0] * 12 if t % 2 == 0 else [0.1] * 12 for t in range(n)],
        "riser_contact": [[t in (1, 2, 5), False, t == 9, False] for t in range(n)],
        "tilt_deg": [1.0] * n, "x": [0.0] * n, "y": [0.0] * n, "finite": [True] * n,
    }
    stats = metrics.episode_statistics(steps, 0.02, warmup_steps=2)
    assert stats["roll_abs_max_deg"] == pytest.approx(1.0)
    assert stats["pitch_rate_rms_deg_s"] == pytest.approx(4.0)
    assert stats["arm_vel_rms_rad_s"] == pytest.approx(0.1)
    assert stats["action_delta_rms"] == pytest.approx(0.1)
    assert stats["action_rate_rms_per_s"] == pytest.approx(5.0)
    # 碰撞从第 1 步起算：轮0 两段（1–2、5），轮2 一段
    assert stats["riser_contact_events"] == [2, 0, 1, 0]
    assert stats["riser_contact_steps"] == [3, 0, 1, 0]
    steps["finite"][5] = False
    assert metrics.episode_statistics(steps, 0.02, 2)["riser_contact_events"] == [1, 0, 1, 0]


def test_contact_impact_speed_and_hold_time(metrics):
    """接触前 5 步内的最大前进速度、连续贴面时长；四轮所有接触段取回合最大值，无接触为 None。"""
    n = 20
    x = [0.0]
    for t in range(1, n):
        x.append(x[-1] + (0.3 if t <= 6 else 0.1) * 0.02)  # 前 6 步 0.3 m/s，之后 0.1 m/s
    contact = [[6 <= t <= 10, False, 15 <= t <= 16, False] for t in range(n)]
    steps = {"roll": [0.0] * n, "pitch": [0.0] * n, "pitch_rate": [0.0] * n, "vz": [0.0] * n,
             "arm_vel": [[0.0] * 4] * n, "action": [[0.0] * 12] * n, "riser_contact": contact,
             "tilt_deg": [0.0] * n, "x": x, "y": [0.0] * n, "finite": [True] * n}
    stats = metrics.episode_statistics(steps, 0.02, warmup_steps=0)
    assert stats["riser_impact_speed_max_m_s"] == pytest.approx(0.3)  # 轮0 在 0.3 m/s 时撞上
    assert stats["riser_contact_hold_max_s"] == pytest.approx(5 * 0.02)  # 轮0 贴面 5 步
    steps["riser_contact"] = [[False] * 4] * n
    quiet = metrics.episode_statistics(steps, 0.02, warmup_steps=0)
    assert quiet["riser_impact_speed_max_m_s"] is None and quiet["riser_contact_hold_max_s"] is None
    rows = metrics.contact_checks({"riser_impact_speed_max_m_s": 0.12, "riser_contact_hold_max_s": None}, case="s")
    assert [r["passed"] for r in rows] == [True, True] and all(r["requirement"] == 3 for r in rows)
    assert metrics.contact_checks({"riser_impact_speed_max_m_s": 0.2, "riser_contact_hold_max_s": 0.7},
                                  case="s")[1]["passed"] is False
    # 楼梯贴面时长门 1.0 s（同轴第二个轮碰到后再翻）；单台阶仍 0.5 s
    assert metrics.GATES["stairs_riser_contact_hold_s"] == 1.0 and metrics.GATES["riser_contact_hold_s"] == 0.5
    assert metrics.contact_checks({"riser_impact_speed_max_m_s": 0.1, "riser_contact_hold_max_s": 0.7}, case="s",
                                  hold_gate=metrics.GATES["stairs_riser_contact_hold_s"])[1]["passed"] is True


# ---------------------------------------------------------------- 逐回合记录与预抬联合判定
def sample(metrics, *, tilt=1.0, x=1.0, y=0.0, contact=(False,) * 4, finite=True):
    return {"roll": [0.0], "pitch": [0.0], "pitch_rate": [0.0], "vz": [0.0], "arm_vel": [[0.0] * 4],
            "action": [[0.0] * 12], "riser_contact": [list(contact)], "tilt_deg": [tilt], "x": [x], "y": [y],
            "finite": [finite]}


@pytest.mark.parametrize("ending,outcome", [
    ({"tilt": 50.0}, "rollover"), ({"y": 0.8}, "out_of_lane"), ({"x": 3.8}, "overshoot"), ({}, "terminated_other"),
])
def test_episode_outcome_classification(monkeypatch, metrics, ending, outcome):
    monkeypatch.syspath_prepend(str(RSL))
    from skill_evaluation import SkillEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    recorder = metrics.EpisodeMetrics(1, 0.02, climb_tilt_deg=45.0, target_distance_m=3.0)
    traversal.update([False], [False], [False], [True])
    recorder.update(sample(metrics), traversal)
    traversal.update([False], [True], [False], [True])
    recorder.update(sample(metrics, **ending), traversal)
    assert recorder.records[0]["outcome"] == outcome
    # 配额满后不再记录
    traversal.update([False], [False], [True], [True])
    recorder.update(sample(metrics), traversal)
    assert len(recorder.records) == 1


def test_timeout_split_into_stall_and_unstable(metrics):
    last = {"tilt_deg": 2.0, "x": 1.0, "y": 0.0, "finite": True}
    assert metrics.classify_outcome(False, False, last, climb_tilt_deg=45, target_distance_m=2.5) == "stall_timeout"
    assert metrics.classify_outcome(False, False, {**last, "x": 2.7}, climb_tilt_deg=45,
                                    target_distance_m=2.5) == "unstable_at_target"
    assert metrics.classify_outcome(True, False, last, climb_tilt_deg=45, target_distance_m=2.5) == "success"
    assert metrics.classify_outcome(False, True, {**last, "finite": False}, climb_tilt_deg=45,
                                    target_distance_m=2.5) == "nonfinite"


@pytest.mark.parametrize("takeoff_lead,rear_collision,expected", [
    (0.10, False, True),    # 3–12 cm 窗口内起抬，保持到前缘，落轮
    (0.05, False, True),    # 5 cm 起抬：仍在窗口内
    (0.13, False, False),   # 13 cm 先离地：早抬，窗口外
    (0.10, True, True),     # 前轮合格，后轮接触后滚上去：允许（接触轻重由 contact_checks 设门）
])
def test_prelift_window_and_rolling_partner_allowed(monkeypatch, metrics, takeoff_lead, rear_collision, expected):
    monkeypatch.syspath_prepend(str(RSL))
    from skill_evaluation import SkillEvaluation
    from vision_prelift_evaluation import PreliftEvaluation

    traversal = SkillEvaluation(1, 1, 0.02, 0.02, 2)
    prelift = PreliftEvaluation(1, 0.02, require_hold_to_edge=True, **metrics.PRELIFT_WINDOW)
    assert prelift.summary(traversal.summary(0.70))["gates"]["min_lead_distance_m"] == 0.03
    recorder = metrics.EpisodeMetrics(1, 0.02)

    def advance(lead, high=True, *, landed=False, done=False, rear=False):
        traversal.update([True], [False], [done], [True])
        prelift.update([{
            "lead_m": [lead, lead], "lift_m": [0.03 if high else 0.] * 2, "clearance_m": [0.03 if high else 0.] * 2,
            "reward_clearance_m": [0. if landed else 0.03] * 2,
            "wheel_support_force_n": [20. if landed or not high else 0.] * 2,
            "eligible": True, "hold_eligible": True,
        }], traversal)
        recorder.update(sample(metrics, contact=(False, rear, False, False)), traversal)

    advance(0.3, high=False)
    advance(takeoff_lead)
    for lead in (0.12, 0.10, 0.06, 0.04, 0.02):
        advance(min(lead, takeoff_lead))
    advance(-0.01)
    advance(-0.15, landed=True, rear=rear_collision)
    advance(-0.15, landed=True)
    advance(-0.15, landed=True, done=True)
    rows = metrics.prelift_acceptance_records(prelift.records, recorder.records)
    assert rows[0]["strict_prelift_success"] is (takeoff_lead <= 0.12)
    assert rows[0]["riser_collision_free"] is (not rear_collision)
    assert rows[0]["success"] is expected


def test_missing_metrics_record_is_fail_closed(metrics):
    rows = metrics.prelift_acceptance_records([{"env_id": 0, "episode_index": 0, "success": True}], [])
    assert rows == [{"env_id": 0, "episode_index": 0, "strict_prelift_success": True, "riser_collision_free": False,
                     "riser_contact_events": None, "success": False}]  # 没有逐回合指标就无法检查接触，判失败


# ---------------------------------------------------------------- 232/233 维与种子
class _Actor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.actor = torch.nn.Sequential(torch.nn.Linear(dim, 8), torch.nn.ELU(), torch.nn.Linear(8, 12))

    def forward(self, obs):
        return self.actor(obs)


@pytest.mark.parametrize("dim", [232, 233])
def test_actor_obs_dim_from_torchscript_and_raw_checkpoint(metrics, tmp_path, dim):
    scripted = tmp_path / f"policy_{dim}.pt"
    torch.jit.script(_Actor(dim)).save(str(scripted))
    assert metrics.actor_input_dim(scripted) == dim
    raw = tmp_path / f"ckpt_{dim}.pt"
    torch.save({"actor": _Actor(dim).state_dict(), "cfg": {}}, raw)
    assert metrics.actor_input_dim(raw) == dim


def test_actor_obs_dim_rejects_unknown(metrics, tmp_path):
    path = tmp_path / "policy.pt"
    torch.jit.script(_Actor(4141)).save(str(path))
    with pytest.raises(ValueError):
        metrics.actor_input_dim(path)


def test_training_seed_read_from_run_params(metrics, tmp_path):
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    (run / "exported").mkdir()
    (run / "params/supermimic_terrain.json").write_text(json.dumps({"arguments": {"seed": 52}}))
    assert metrics.training_seeds(run / "exported/policy.pt") == [52]
    assert metrics.training_seeds(tmp_path / "other/exported/policy.pt") == []


# ---------------------------------------------------------------- accept.py 计划与汇总
def test_plan_covers_all_cases_and_seed_modes(accept):
    args = accept.parse_args(["--checkpoint", "x.pt"])
    cases = accept.plan_cases(args)
    assert sum(c["suite"] == "prelift" for c in cases) == 9
    assert sum(c["suite"] == "stairs" for c in cases) == 8
    assert {c["id"] for c in cases} >= {"step_17.5cm_left", "stairs_17.5cm_tread28cm", "stairs_8cm_tread40cm"}
    assert accept.seed_plan([9001, 9002, 9003], "split", 64, 0) == [(9001, 22), (9002, 22), (9003, 22)]
    assert accept.seed_plan([9001, 9002, 9003], "rotate", 64, 4) == [(9002, 64)]
    cmd = accept.eval_command(cases[0], 9001, 22, Path("p.pt"), Path("o.json"))
    assert "--acceptance_metrics" in cmd and cmd[cmd.index("--step_height_m") + 1] == "0.1"


def test_held_out_seed_clash_is_execution_error(accept, tmp_path, capsys):
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    (run / "params/supermimic_terrain.json").write_text(json.dumps({"arguments": {"seed": 9001}}))
    assert accept.main(["--checkpoint", str(run / "exported/policy.pt"), "--dry_run"]) == 2
    assert accept.main(["--checkpoint", str(run / "exported/policy.pt"), "--dry_run", "--seeds", "142"]) == 2
    assert accept.main(["--checkpoint", str(run / "exported/policy.pt"), "--dry_run", "--seeds", "9002"]) == 0
    assert accept.main(["--checkpoint", str(run / "exported/policy.pt"), "--dry_run", "--seeds", "142",
                        "--seed_role", "selection"]) == 0


def _metrics_episode(env_id, *, success=True, pitch=5.0, roll=1.0, arm=0.3, contact=(0, 0, 0, 0), outcome=None):
    return {"env_id": env_id, "episode_index": 0, "success": success, "outcome": outcome or ("success" if success else "stall_timeout"),
            "roll_abs_p95_deg": roll, "roll_abs_max_deg": roll + 1, "pitch_abs_p95_deg": pitch,
            "pitch_abs_max_deg": pitch + 4, "pitch_rate_rms_deg_s": 6.0, "arm_vel_rms_rad_s": arm,
            "base_vz_rms_m_s": 0.02, "action_delta_rms": 0.01, "arm_action_delta_rms": 0.01,
            "action_rate_rms_per_s": 0.5, "riser_contact_events": list(contact),
            "riser_impact_speed_max_m_s": 0.10, "riser_contact_hold_max_s": 0.3}


def test_summaries_and_verdicts(accept, metrics):
    args = accept.parse_args(["--checkpoint", "x.pt"])
    cases = {c["id"]: c for c in accept.plan_cases(args)}
    stair = cases["stairs_12cm_tread28cm"]  # 坡度 23.2°：P95 门 18.2°
    data = {"complete": True, "acceptance_metrics": {"episodes": [_metrics_episode(i, success=i % 4 != 0) for i in range(22)]}}
    runs = [{"seed": s, "envs": 22, "data": data, "error": None} for s in (9001, 9002, 9003)]
    row = accept.summarize_eval_case(stair, runs)
    assert row["episodes"] == 66 and row["successes"] == 48
    assert all(c["passed"] for c in row["checks"] if c["gating"])
    step = cases["step_10cm_full"]
    eps = [_metrics_episode(i, contact=(0, 0, 0, 1) if i < 4 else (0, 0, 0, 0)) for i in range(22)]
    prelift = {"episode_results": [{"env_id": i, "episode_index": 0, "success": i < 14} for i in range(22)],
               "takeoff_distance_m": {}}
    data = {"complete": True, "episode_results": [{"success": True}] * 22, "prelift": prelift,
            "acceptance_metrics": {"episodes": eps}}
    row_step = accept.summarize_eval_case(step, [{"seed": s, "envs": 22, "data": data, "error": None}
                                                for s in (9001, 9002, 9003)])
    # 前 4 个回合后轮碰撞（允许，只进诊断）、后 8 个严格预抬失败 → 14/22
    assert row_step["successes"] == 42 and row_step["riser_collision_free_rate"] == pytest.approx(18 / 22)
    assert next(c for c in row_step["checks"] if c["metric"] == "success_rate")["passed"] is False
    broken = accept.summarize_eval_case(step, [{"seed": 9001, "envs": 22, "data": None, "error": "exit 2"}])
    assert any(c["metric"] == "execution" and c["passed"] is False for c in broken["checks"])
    flat_rows = [{"command": c, "seed": s, "per_env": {k: [0.01] * 64 for k in metrics.FLAT_METRICS},
                  "startup_tilt_p95_deg": 1.0, "tilt_p95_deg": 1.0, "arm_pose_p95_deg": 3.0,
                  "wheel_clearance_p95_cm": 0.1, "four_wheel_support_loss_fraction": 0.0}
                 for s in (9001, 9002, 9003) for c in (0.0, 0.15, 0.30)]
    flat = accept.summarize_flat({"policy": flat_rows}, None)
    assert flat["by_command"]["0.30"]["envs"] == 192
    verdict = accept.verdicts([flat, row, row_step], ["flat", "prelift", "stairs"])
    assert verdict["3"]["passed"] is False and verdict["4"]["passed"] is True and verdict["5"]["passed"] is True
    partial = accept.verdicts([row], ["stairs"])
    assert partial["1"]["fully_covered"] is False and partial["1"]["passed"] is False and partial["5"]["passed"] is None
    summary = {"passed": False, "policy": "p", "policy_sha256": "0" * 64, "actor_obs_dim": 232, "seeds": [9001],
               "seed_role": "held_out", "seed_mode": "split", "forbidden_seeds": [], "finished_at": "t",
               "elapsed_s": 60.0, "requirements": verdict, "cases": [flat, row, row_step]}
    markdown = accept.render_markdown(summary)
    assert "stairs_12cm_tread28cm" in markdown and "**FAIL**" in markdown


# ---------------------------------------------------------------- Isaac 侧采样胶水（假环境）
def test_quat_roll_pitch_signs(metrics):
    a = math.radians(10.0)
    pitch_quat = torch.tensor([[math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]])
    roll_quat = torch.tensor([[math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]])
    roll, pitch = metrics.quat_to_roll_pitch(torch.cat((pitch_quat, roll_quat)))
    assert torch.allclose(torch.rad2deg(pitch), torch.tensor([10.0, 0.0]), atol=1e-4)
    assert torch.allclose(torch.rad2deg(roll), torch.tensor([0.0, 10.0]), atol=1e-4)


def test_capture_acceptance_sample_from_fake_env(metrics):
    from types import SimpleNamespace

    r = 0.1085
    arm_names = ["fl_arm_joint", "rl_arm_joint", "fr_arm_joint", "rr_arm_joint"]
    data = SimpleNamespace(
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]), root_pos_w=torch.tensor([[11.0, 5.2, 0.3]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]), root_ang_vel_b=torch.tensor([[0.0, 0.2, 0.0]]),
        root_lin_vel_w=torch.tensor([[0.3, 0.0, -0.05]]), joint_vel=torch.arange(12.0).unsqueeze(0),
    )
    robot = SimpleNamespace(data=data, find_joints=lambda names, preserve_order: ([4, 5, 6, 7], list(names)))
    scene = {"robot": robot}
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.full((1, 12), 0.5)),
        cfg=SimpleNamespace(acceptance_metrics={"risers": metrics.single_step_risers(1.5, 0.14, (-1.5, 1.5)),
                                        "arm_joint_names": arm_names}),
    )
    env.scene = type("Scene", (), {"__getitem__": lambda self, key: scene[key],
                                   "env_origins": torch.tensor([[10.0, 5.0, 0.0]])})()
    centers = torch.tensor([[[11.5 - r, 5.3, r], [10.8, 5.3, r], [11.5 - r, 4.7, r + 0.2], [10.8, 4.7, r]]])
    preview = {"wheel_centers_w": centers, "wheel_radii": torch.full((4,), r),
               "wheel_resisting_force_n": torch.tensor([[12.0, 12.0, 12.0, 0.0]])}
    out = metrics.capture_acceptance_sample(env, preview, torch.tensor([True]))
    assert out["riser_contact"] == [[True, False, False, False]]
    assert out["arm_vel"] == [[4.0, 5.0, 6.0, 7.0]]
    assert out["x"] == pytest.approx([1.0]) and out["y"] == pytest.approx([0.2])
    assert out["pitch_rate"] == pytest.approx([0.2]) and out["vz"] == pytest.approx([-0.05])
    assert out["tilt_deg"] == pytest.approx([0.0], abs=1e-3) and out["action"] == [[0.5] * 12]
    assert set(out) == set(metrics.STEP_KEYS)
