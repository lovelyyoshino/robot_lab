#!/usr/bin/env python3
"""
accept.py — SuperMimic 四轮腿越障验收唯一入口：平地 + 单台阶预抬 + 楼梯，按 5 项需求出 JSON 与 Markdown。

需求：1 不倾斜（roll/pitch/俯仰角速度）；2 平顺（摆臂速度，另报垂向速度与动作变化率）；
3 预抬：单台阶（10/14/17.5 cm × 全宽/左窄/右窄）至少一个前轮严格预抬（3–12 cm 起抬、保持到前缘、落轮）并通行
  ≥70%，同轴另一个轮可以接触后滚上去；楼梯上四个轮不可能每级都离地预抬。两者都要求轻碰快上
  （接触前速度 P95 ≤0.15 m/s、贴面时长 P95 单台阶 ≤0.5 s、楼梯 ≤1.0 s）；
4 楼梯（8/12/16/17.5 cm × 踏面 0.40/0.28 m，严格登顶 ≥70%）；5 平地行驶（偏航/横向/速度跟踪）。
统计一律为跨回合/环境 P95，口径见 acceptance_metrics.py。种子默认 9001–9003（留出，不得用于训练与挑选）。

每个（档位, 种子）起一个 Isaac 进程（一个进程只能建一次场景），全部顺序执行，并对 logs/.isaac_eval.lock 加锁，
保证同一时刻只有一个 Isaac 评测。本进程不 import isaaclab。

输出：<run 或 ckpt_eval/itN>/acceptance/{cases/*.json,*.log, flat.json, summary.json, summary.md}
退出码：0 全部需求通过；1 有需求未通过；2 执行异常（评测崩溃/超时/种子冲突等，结果不可用于结论）。

@author pony
@date 2026-11-05
@version v1.4.0
@last_modified 2026-11-21
@changelog
    - v1.4.0 (2026-11-21): 单台阶与楼梯表加"软判"列（软成功率 / 平均得分，acceptance_metrics v1.3.0：横偏不判失败、
      超过 0.15 m 每厘米扣 0.02、最多扣 0.5）；summary.json 每档加 soft_successes、soft_score_mean。门与结论不变。
    - v1.3.0 (2026-11-11): 楼梯贴面时长门用 stairs_riser_contact_hold_s（1.0 s，acceptance_metrics v1.2.0）；单台阶仍 0.5 s。
    - v1.2.0 (2026-11-07): 轮胎摩擦固定 μ=1.0（evaluate 自动对齐）；汇总记录 wheel_friction，复用结果须带摩擦记录。
    - v1.1.0 (2026-11-06): 需求 3 覆盖楼梯（轻碰快上）；预抬起抬窗口 3–12 cm，一个轮先抬、另一个滚上去即可。
    - v1.0.0 (2026-11-05): 初始创建；取代 accept_terrain.py 与 watch_accept 的旧预抬分支。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RSL = ROOT / "scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, str(RSL))
import acceptance_metrics as m  # noqa: E402

EVAL = RSL / "evaluate_parallelogram_vision.py"
FLAT = ROOT / "scripts/supermimic/probe_flat.py"
EXPORT = ROOT / "scripts/supermimic/export_ckpt.py"
LOCK = ROOT / "logs/.isaac_eval.lock"
STAIRS = m.load_stairs_acceptance()
STEP_WIDTHS = {"full": "single_step", "left": "single_step_left", "right": "single_step_right"}
PRELIFT_HEIGHTS_M = (0.10, 0.14, 0.175)
FLAT_COMMANDS = (0.0, 0.15, 0.30)
REQUIREMENTS = {1: "不倾斜", 2: "平顺", 3: "预抬（单台阶）/轻碰快上（楼梯）", 4: "楼梯", 5: "平地行驶"}
SUITE_REQUIREMENTS = {"flat": (1, 2, 5), "prelift": (1, 2, 3), "stairs": (1, 2, 3, 4)}
# 实测（2026-11-04）：32 环境 × 2 回合的楼梯评测约 55 s/进程（含 Isaac 启动），平地 3 速度约 72 s/种子。
SECONDS_PER_EVAL = 55
SECONDS_PER_FLAT_SEED = 72


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="导出的 TorchScript（<run>/exported/policy.pt 或 ckpt_eval/itN/exported/policy.pt）")
    src.add_argument("--load_run", help="logs/supermimic_terrain 下的 run 名")
    p.add_argument("--iteration", type=int, help="与 --load_run 合用：验收 ckpt_XXXXXX.pt（自动导出到 ckpt_eval/itN）")
    p.add_argument("--suites", nargs="+", choices=tuple(SUITE_REQUIREMENTS), default=list(SUITE_REQUIREMENTS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(m.HELD_OUT_SEEDS))
    p.add_argument("--seed_role", choices=("held_out", "selection"), default="held_out",
                   help="held_out：最终验收，拒绝训练/挑选用过的种子；selection：训练中挑存档（watch_accept）")
    p.add_argument("--seed_mode", choices=("split", "rotate"), default="split",
                   help="split：每档跑全部种子、回合数均分；rotate：每档一个种子轮换（快约 3 倍）")
    p.add_argument("--min_episodes", type=int, default=m.GATES["min_episodes"])
    p.add_argument("--flat_envs", type=int, default=64, help="平地每种子环境数（每环境一个样本）")
    p.add_argument("--prelift_heights", type=float, nargs="+", default=list(PRELIFT_HEIGHTS_M))
    p.add_argument("--step_widths", nargs="+", choices=tuple(STEP_WIDTHS), default=list(STEP_WIDTHS))
    p.add_argument("--stair_risers", type=float, nargs="+", default=list(STAIRS.ACCEPTANCE_STAIR_RISERS_M))
    p.add_argument("--stair_treads", type=float, nargs="+", default=list(STAIRS.ACCEPTANCE_STAIR_TREADS_M))
    p.add_argument("--out", type=Path, help="输出目录；默认策略 run 目录下 acceptance/")
    p.add_argument("--timeout_s", type=int, default=1800, help="单个 Isaac 进程超时")
    p.add_argument("--reuse", action="store_true", help="复用同一策略 sha256/种子已完成的档位结果（中断后续跑）")
    p.add_argument("--summarize_only", action="store_true", help="只用已有结果重算汇总，不启动 Isaac")
    p.add_argument("--dry_run", action="store_true", help="只打印计划与命令")
    args = p.parse_args(argv)
    if args.iteration is not None and args.load_run is None:
        p.error("--iteration 需要 --load_run")
    if args.min_episodes < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error("--min_episodes 必须为正，--seeds 不能为空或重复")
    return args


def resolve_policy(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    run = ROOT / "logs/supermimic_terrain" / args.load_run
    if args.iteration is None:
        return (run / "exported/policy.pt").resolve()
    policy = run / "ckpt_eval" / f"it{args.iteration}" / "exported/policy.pt"
    if not policy.is_file() and not args.dry_run:
        subprocess.run([sys.executable, str(EXPORT), "--load_run", args.load_run, "--iterations", str(args.iteration)],
                       cwd=ROOT, check=True)
    return policy.resolve()


def cm(value_m: float) -> str:
    return f"{value_m * 100:g}"


def plan_cases(args) -> list[dict]:
    cases = []
    if "prelift" in args.suites:
        for height in args.prelift_heights:
            for width in args.step_widths:
                cases.append({"id": f"step_{cm(height)}cm_{width}", "suite": "prelift", "scenario": STEP_WIDTHS[width],
                              "step_height_m": height, "width": width})
    if "stairs" in args.suites:
        for riser in args.stair_risers:
            for tread in args.stair_treads:
                cases.append({"id": f"stairs_{cm(riser)}cm_tread{cm(tread)}cm", "suite": "stairs",
                              "scenario": "pyramid_stairs_inv", "riser_m": riser, "tread_m": tread,
                              "pitch_limits": STAIRS.stairs_pitch_gates(riser, tread),
                              "climb_tilt_deg": STAIRS.climb_tilt_for_riser(riser)})
    return cases


def seed_plan(seeds, mode, min_episodes, index) -> list[tuple[int, int]]:
    """(种子, 环境数)；每环境 1 回合。split 均分到每个种子，rotate 每档一个种子。"""
    if mode == "rotate":
        return [(seeds[index % len(seeds)], min_episodes)]
    per_seed = math.ceil(min_episodes / len(seeds))
    return [(seed, per_seed) for seed in seeds]


def eval_command(case, seed, num_envs, policy, output) -> list[str]:
    cmd = [sys.executable, str(EVAL), "--headless", "--policy", "teacher-preview", "--supermimic_terrain",
           "--checkpoint", str(policy), "--scenario", case["scenario"], "--num_envs", str(num_envs),
           "--episodes_per_env", "1", "--seed", str(seed), "--acceptance_metrics", "--output", str(output)]
    if case["suite"] == "prelift":
        # 难度只影响前缘位置哈希；高度由覆盖固定
        cmd += ["--difficulty", "0.5", "--step_height_m", str(case["step_height_m"])]
    else:
        cmd += ["--difficulty", "0.0", "--stair_riser_m", str(case["riser_m"]), "--stair_tread_m", str(case["tread_m"])]
    return cmd


def flat_command(policy, seeds, num_envs, output) -> list[str]:
    return [sys.executable, str(FLAT), "--headless", "--policy", f"policy={policy}", "--seeds", *map(str, seeds),
            "--num_envs", str(num_envs), "--commands", *map(str, FLAT_COMMANDS), "--output", str(output)]


def run_logged(cmd, log: Path, timeout_s: int) -> int:
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[accept] 超时 {timeout_s}s\n")
            return 124


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def valid_case_json(path: Path, policy_sha: str, seed: int):
    """可复用 = 同一策略、同一种子、配额完整且含逐回合验收指标。"""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return None
    # 2026-11-07 前的结果是随机摩擦（无 wheel_friction 记录），不能与固定 μ=1.0 的新结果混用
    ok = (data.get("checkpoint_sha256") == policy_sha and data.get("seed") == seed and data.get("complete")
          and data.get("acceptance_metrics") is not None and "wheel_friction" in (data.get("terrain_override") or {}))
    return data if ok else None


# ---------------------------------------------------------------- 汇总
def summarize_eval_case(case, runs) -> dict:
    """runs: [{seed, envs, data|None, error|None}]；跨种子合并后出指标与门。"""
    errors = [r["error"] for r in runs if r.get("error")]
    datas = [r for r in runs if r.get("data") is not None]
    episodes = [{**e, "seed": r["seed"]} for r in datas for e in r["data"]["acceptance_metrics"]["episodes"]]
    expected = sum(r["envs"] for r in runs)
    complete = not errors and len(datas) == len(runs) and all(r["data"]["complete"] for r in datas)
    row = {"id": case["id"], "suite": case["suite"], "seeds": [r["seed"] for r in runs],
           "expected_episodes": expected, "complete": complete, "errors": errors,
           "aggregate": m.aggregate_episodes(episodes)}
    if case["suite"] == "prelift":
        joined = [{**j, "seed": r["seed"]} for r in datas
                  for j in m.prelift_acceptance_records(r["data"]["prelift"]["episode_results"],
                                                r["data"]["acceptance_metrics"]["episodes"])]
        traversal = sum(e["success"] for r in datas for e in r["data"]["episode_results"])
        row.update({"step_height_m": case["step_height_m"], "width": case["width"],
                    "episodes": len(joined), "successes": sum(j["success"] for j in joined),
                    "strict_prelift_rate": sum(j["strict_prelift_success"] for j in joined) / len(joined) if joined else 0.0,
                    "riser_collision_free_rate": sum(j["riser_collision_free"] for j in joined) / len(joined) if joined else 0.0,
                    "traversal_rate": traversal / len(joined) if joined else 0.0,
                    "takeoff_distance_m": [r["data"]["prelift"]["takeoff_distance_m"] for r in datas]})
        row["checks"] = (m.success_checks(3, case["id"], successes=row["successes"], episodes=row["episodes"],
                                          complete=complete)
                         + m.contact_checks(row["aggregate"]["p95"], case=case["id"])
                         + m.tilt_smooth_checks("step", row["aggregate"]["p95"], case=case["id"],
                                                step_height_m=case["step_height_m"]))
    else:
        successes = sum(e["success"] for e in episodes)
        row.update({"riser_m": case["riser_m"], "tread_m": case["tread_m"], "pitch_limits": case["pitch_limits"],
                    "climb_tilt_deg": case["climb_tilt_deg"], "episodes": len(episodes), "successes": successes})
        row["checks"] = (m.success_checks(4, case["id"], successes=successes, episodes=len(episodes), complete=complete)
                         + m.tilt_smooth_checks("stairs", row["aggregate"]["p95"], case=case["id"],
                                                pitch_limits=case["pitch_limits"])
                         + m.contact_checks(row["aggregate"]["p95"], case=case["id"],
                                           hold_gate=m.GATES["stairs_riser_contact_hold_s"]))
    if errors:
        for req in SUITE_REQUIREMENTS[case["suite"]]:
            row["checks"].append(m.check(req, case["id"], "execution", 0.0, 1.0, op=">=", note="; ".join(errors)))
    return row


def summarize_flat(data, error) -> dict:
    if error is not None:
        return {"id": "flat", "suite": "flat", "error": error,
                "checks": [m.check(r, "flat", "execution", 0.0, 1.0, op=">=", note=error) for r in (1, 2, 5)]}
    rows = data["policy"]
    pooled = m.pool_flat_rows(rows)
    diagnostics = {k: max(r[k] for r in rows) for k in ("startup_tilt_p95_deg", "tilt_p95_deg", "arm_pose_p95_deg",
                                                         "wheel_clearance_p95_cm", "four_wheel_support_loss_fraction")}
    return {"id": "flat", "suite": "flat", "by_command": {k: {kk: vv for kk, vv in v.items() if kk != "per_env"}
                                                          for k, v in pooled.items()},
            "diagnostics_worst": diagnostics, "checks": m.flat_checks(pooled)}


def verdicts(case_rows, suites) -> dict:
    checks = [c for row in case_rows for c in row["checks"]]
    out = {}
    for req, name in REQUIREMENTS.items():
        evaluated = any(req in SUITE_REQUIREMENTS[s] for s in suites)
        gating = [c for c in checks if c["requirement"] == req and c["gating"]]
        failed = [c for c in gating if not c["passed"]]
        # 需求 1/2 跨三类地形；只跑部分地形时不能宣布整项通过
        full = all(s in suites for s, reqs in SUITE_REQUIREMENTS.items() if req in reqs)
        out[str(req)] = {"name": name, "evaluated": evaluated, "fully_covered": full,
                         "passed": (not failed and bool(gating) and full) if evaluated else None,
                         "failed_checks": [f"{c['case']}:{c['metric']}" for c in failed]}
    return out


def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def cell(row_checks, metric):
    c = next((c for c in row_checks if c["metric"] == metric), None)
    if c is None:
        return "-"
    if not c["gating"]:
        return fmt(c["value"])
    return f"{fmt(c['value'])} (≤{c['threshold']:.3g}) {'PASS' if c['passed'] else '**FAIL**'}"


def soft_cell(row) -> str:
    """软判：软成功率 / 平均得分（acceptance_metrics.SOFT_RULE）；旧结果没有软判字段时为 -。"""
    agg = row["aggregate"]
    if "soft_successes" not in agg or not agg["episodes"]:
        return "-"
    return f"{agg['soft_successes'] / agg['episodes']:.0%} / {agg['soft_score_mean']:.2f}"


def render_markdown(summary) -> str:
    v = summary["requirements"]
    lines = [f"# SuperMimic 越障 验收：{'PASS' if summary['passed'] else 'FAIL'}", "",
             f"- 策略：`{summary['policy']}`（sha256 {summary['policy_sha256'][:12]}，actor 输入 {summary['actor_obs_dim']} 维）",
             f"- 种子：{summary['seeds']}（{summary['seed_role']}，{summary['seed_mode']}；禁用 {summary['forbidden_seeds']}）",
             f"- 轮胎摩擦（静, 动）：{summary.get('wheel_friction') or '随机（旧协议）'}",
             f"- 时间：{summary['finished_at']}，用时 {summary['elapsed_s'] / 60:.1f} min", "",
             "| 需求 | 结论 | 未通过项 |", "|---|---|---|"]
    for req, item in v.items():
        verdict = "未评测" if item["passed"] is None else ("PASS" if item["passed"] else "**FAIL**")
        if item["evaluated"] and not item["fully_covered"]:
            verdict += "（地形未覆盖全）"
        lines.append(f"| {req} {item['name']} | {verdict} | {', '.join(item['failed_checks'][:8]) or '-'}"
                     f"{' …' if len(item['failed_checks']) > 8 else ''} |")
    rows = summary["cases"]
    lines += ["", "## 1 不倾斜 / 2 平顺（跨回合 P95；max 为逐回合最大值的 P95）", "",
              "| 档位 | roll P95° | roll max° | pitch P95° | pitch max° | 俯仰角速度 RMS °/s | 摆臂速度 RMS rad/s "
              "| 垂向速度 RMS m/s | 动作变化率 RMS /s |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        groups = {}
        for c in row["checks"]:
            if c["requirement"] in (1, 2):
                groups.setdefault(c["case"], []).append(c)
        for case_id, cs in groups.items():
            lines.append(f"| {case_id} | " + " | ".join(cell(cs, k) for k in (
                "roll_abs_p95_deg", "roll_abs_max_deg", "pitch_abs_p95_deg", "pitch_abs_max_deg",
                "pitch_rate_rms_deg_s", "arm_vel_rms_rad_s", "base_vz_rms_m_s", "action_rate_rms_per_s")) + " |")
    pre = [r for r in rows if r["suite"] == "prelift"]
    if pre:
        lines += ["", "## 3 单台阶预抬（至少一个前轮 3–12 cm 起抬的严格时序 + 通行；其余轮轻碰快上）", "",
                  "| 档位 | 回合 | 成功率 | 软判通行 / 得分 | 严格预抬 | 四轮无碰撞（诊断） | 通行 | 接触前速度 m/s | 贴面时长 s "
                  "| 碰撞回合率 fl/rl/fr/rr | 结论 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in pre:
            col = r["aggregate"].get("riser_collision_episode_rate", {})
            ok = all(c["passed"] for c in r["checks"] if c["requirement"] == 3 and c["gating"])
            lines.append(f"| {r['id']} | {r['episodes']}/{r['expected_episodes']} | "
                         f"{r['successes'] / r['episodes'] if r['episodes'] else 0:.0%} | {soft_cell(r)} | "
                         f"{r['strict_prelift_rate']:.0%} | "
                         f"{r['riser_collision_free_rate']:.0%} | {r['traversal_rate']:.0%} | "
                         f"{cell(r['checks'], 'riser_impact_speed_max_m_s')} | {cell(r['checks'], 'riser_contact_hold_max_s')} | "
                         + "/".join(f"{col.get(w, 0):.0%}" for w in m.WHEEL_LABELS)
                         + f" | {'PASS' if ok else '**FAIL**'} |")
    stairs = [r for r in rows if r["suite"] == "stairs"]
    if stairs:
        lines += ["", "## 4 楼梯（严格登顶）+ 3 轻碰快上（接触前速度 / 贴面时长，跨回合 P95）", "",
                  "| 档位 | 坡度° | 回合 | 成功率 | 软判成功 / 得分 | 停住/超时 | 到终点未稳 | 翻车 | 出车道 | 冲出 | 其他 "
                  "| 接触前速度 m/s | 贴面时长 s | 碰撞回合率 fl/rl/fr/rr | 结论 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in stairs:
            o = r["aggregate"]["outcomes"]
            col = r["aggregate"].get("riser_collision_episode_rate", {})
            other = sum(n for k, n in o.items() if k not in (
                "success", "stall_timeout", "unstable_at_target", "rollover", "out_of_lane", "overshoot"))
            ok = all(c["passed"] for c in r["checks"] if c["requirement"] in (3, 4) and c["gating"])
            lines.append(f"| {r['id']} | {r['pitch_limits']['slope_deg']:.1f} | {r['episodes']}/{r['expected_episodes']} | "
                         f"{r['successes'] / r['episodes'] if r['episodes'] else 0:.0%} | {soft_cell(r)} | "
                         f"{o.get('stall_timeout', 0)} | "
                         f"{o.get('unstable_at_target', 0)} | {o.get('rollover', 0)} | {o.get('out_of_lane', 0)} | "
                         f"{o.get('overshoot', 0)} | {other} | {cell(r['checks'], 'riser_impact_speed_max_m_s')} | "
                         f"{cell(r['checks'], 'riser_contact_hold_max_s')} | "
                         + "/".join(f"{col.get(w, 0):.0%}" for w in m.WHEEL_LABELS)
                         + f" | {'PASS' if ok else '**FAIL**'} |")
    flat = next((r for r in rows if r["suite"] == "flat"), None)
    if flat is not None and "by_command" in flat:
        lines += ["", "## 5 平地行驶（跨环境 P95）", "", "| 速度 m/s | 环境数 | 偏航 P95° | 横向 P95 m | 速度 RMSE m/s |",
                  "|---|---|---|---|---|"]
        for key, entry in flat["by_command"].items():
            cs = [c for c in flat["checks"] if c["case"] == f"flat_v{entry['command']:.2f}"]
            lines.append(f"| {key} | {entry['envs']} | {cell(cs, 'yaw_abs_p95_deg')} | {cell(cs, 'lateral_abs_p95_m')} | "
                         f"{cell(cs, 'route_velocity_rmse_m_s')} |")
        d = flat["diagnostics_worst"]
        lines.append(f"\n诊断（不设门，三速度最差）：起步倾角 P95 {d['startup_tilt_p95_deg']:.1f}°，摆臂角 P95 "
                     f"{d['arm_pose_p95_deg']:.1f}°，轮底净空 P95 {d['wheel_clearance_p95_cm']:.2f} cm，"
                     f"四轮支撑不足 {d['four_wheel_support_loss_fraction']:.1%}")
    errs = [(r["id"], r.get("error") or "; ".join(r.get("errors", []))) for r in rows if r.get("error") or r.get("errors")]
    if errs:
        lines += ["", "## 执行异常（结论无效）", ""] + [f"- {i}: {e}" for i, e in errs]
    return "\n".join(lines) + "\n"


def acquire_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[accept] 另一个 Isaac 评测持有 {LOCK}，排队等待……", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    policy = resolve_policy(args)
    if not policy.is_file() and not args.dry_run:
        print(f"[accept] 找不到策略：{policy}", file=sys.stderr)
        return 2
    out = (args.out or policy.parent.parent / "acceptance").resolve()
    forbidden = sorted(set(m.SELECTION_SEEDS) | set(m.training_seeds(policy))) if args.seed_role == "held_out" else []
    clash = sorted(set(args.seeds) & set(forbidden))
    if clash:
        print(f"[accept] 种子 {clash} 用于训练或挑选，不能作为留出种子；挑存档请用 --seed_role selection",
              file=sys.stderr)
        return 2
    cases = plan_cases(args)
    plans = [(case, seed_plan(args.seeds, args.seed_mode, args.min_episodes, i)) for i, case in enumerate(cases)]
    n_eval = sum(len(p) for _, p in plans)
    estimate = n_eval * SECONDS_PER_EVAL + ("flat" in args.suites) * len(args.seeds) * SECONDS_PER_FLAT_SEED
    print(f"[accept] {len(cases)} 个档位、{n_eval} 个评测进程" + ("＋平地" if "flat" in args.suites else "")
          + f"，预计 {estimate / 60:.0f} min；输出 {out}", flush=True)
    cases_dir = out / "cases"
    if args.dry_run:
        if "flat" in args.suites:
            print(" ".join(flat_command(policy, args.seeds, args.flat_envs, out / "flat.json")))
        for case, plan in plans:
            for seed, envs in plan:
                print(" ".join(eval_command(case, seed, envs, policy, cases_dir / f"{case['id']}_s{seed}.json")))
        return 0
    cases_dir.mkdir(parents=True, exist_ok=True)
    policy_sha = sha256(policy)
    actor_obs_dim = m.actor_input_dim(policy)
    lock = None if args.summarize_only else acquire_lock()
    frictions = set()
    try:
        rows = []
        if "flat" in args.suites:
            result, log = out / "flat.json", out / "flat.log"
            data, error = None, None
            if not args.summarize_only and not (args.reuse and result.is_file()):
                result.unlink(missing_ok=True)
                code = run_logged(flat_command(policy, args.seeds, args.flat_envs, result), log, args.timeout_s)
                if code != 0:
                    error = f"平地评测 exit {code}，见 {log}"
            if error is None:
                try:
                    data = json.loads(result.read_text())
                    assert sorted({r["seed"] for r in data["policy"]}) == sorted(args.seeds)
                except (OSError, ValueError, KeyError, AssertionError):
                    error = f"平地结果缺失或与种子不符，见 {log}"
            rows.append(summarize_flat(data, error))
            print(f"[accept] flat: {error or 'done'}", flush=True)
        for case, plan in plans:
            runs = []
            for seed, envs in plan:
                result, log = cases_dir / f"{case['id']}_s{seed}.json", cases_dir / f"{case['id']}_s{seed}.log"
                data = valid_case_json(result, policy_sha, seed) if (args.reuse or args.summarize_only) else None
                error = None
                if data is None and not args.summarize_only:
                    result.unlink(missing_ok=True)
                    t0 = time.time()
                    code = run_logged(eval_command(case, seed, envs, policy, result), log, args.timeout_s)
                    # evaluate：0 通过、1 未达旧门（有效结果）、其他为执行异常
                    data = valid_case_json(result, policy_sha, seed) if code in (0, 1) else None
                    if data is None:
                        error = f"seed {seed} exit {code}，见 {log}"
                    print(f"[accept] {case['id']} seed {seed}: {error or 'ok'}（{time.time() - t0:.0f} s）", flush=True)
                elif data is None:
                    error = f"seed {seed} 无可用结果 {result}"
                if data is not None:
                    frictions.add(tuple(data["terrain_override"]["wheel_friction"]))
                runs.append({"seed": seed, "envs": envs, "data": data, "error": error})
            rows.append(summarize_eval_case(case, runs))
    finally:
        if lock is not None:
            lock.close()
    req = verdicts(rows, args.suites)
    broken = [r["id"] for r in rows if r.get("error") or r.get("errors")]
    summary = {
        "protocol": "supermimic_acceptance", "policy": str(policy), "policy_sha256": policy_sha,
        "actor_obs_dim": actor_obs_dim, "lane_offset_obs": actor_obs_dim == m.LANE_OFFSET_OBS_DIM,
        "seeds": args.seeds, "seed_role": args.seed_role, "seed_mode": args.seed_mode, "forbidden_seeds": forbidden,
        "suites": args.suites, "wheel_friction": [list(f) for f in sorted(frictions)],
        "thresholds": m.GATES, "riser_contact": m.RISER_CONTACT,
        "statistics": "per-episode time statistic, then P95 across episodes/envs; *_max = P95 of per-episode max; "
                      "warmup 0.5 s excluded; failed episodes included",
        "requirements": req, "passed": all(item["passed"] for item in req.values()) and not broken,
        "broken": broken, "cases": rows,
        "finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_s": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    markdown = render_markdown(summary)
    (out / "summary.md").write_text(markdown)
    print(markdown, flush=True)
    return 2 if broken else (0 if summary["passed"] else 1)


if __name__ == "__main__":
    sys.exit(main())
