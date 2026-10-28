#!/usr/bin/env python3
"""
accept_terrain.py — SuperMimic 越障一键验收：8/12/16 cm 三档楼梯正式评测 + 失败诊断 + 汇总表。

另跑一次平地平稳检查；每档各起一个 Isaac 进程（Isaac Sim 一个进程只能建一次场景），顺序执行，不并行
（两个进程同时导入 URDF 会冲突，实测报 mimic 关节缺失）。不启动 Kit，自身不 import isaaclab。
倾角门按 stairs_acceptance.CLIMB_TILT_DEG（8/12 cm 31.8°，16 cm 45°）。

输出：logs/supermimic_terrain/<run>/acceptance/{d0.0,d0.5,d1.0}.json、diag_*.log、summary.json、summary.md。
退出码：0 = 全部档位都跑完（不代表通过，通过与否看 summary）；2 = 有档位执行失败（非验收失败）。

@author pony
@date 2026-10-27
@version v1.1.0
@last_modified 2026-10-28
@changelog
    - v1.1.0 (2026-10-28): 增加平地平稳检查（probe_flat.py），门为 arm 速度与俯仰角速度，不看高度慢漂。
    - v1.0.0 (2026-10-27): 初始创建；替代手写 bash 循环（曾因合同失配全部报错却无提示）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py"
DIAG = ROOT / "scripts/supermimic/climb_search.py"
FLAT = ROOT / "scripts/supermimic/probe_flat.py"
# 平地平稳门（0.3 m/s、前 3 s 不计）：Teacher / v3 为 arm 速度 rms 0.13 rad/s、车体起伏 0.1 cm；
# 用户训到 4000 轮的 terrain_v3 为 1.21 rad/s、2.7 cm（明显上下跳）。门取二者之间。
# 门只看"抖"（arm 来回摆、车身俯仰抖），不看车体高度的缓慢漂移：第 4000 轮新 terrain_v3 在 0.15 m/s 下
# 车体高度慢漂 1.18 cm 但 arm 0.11 rad/s、俯仰 0.5°/s，并不抖；真抖的旧 terrain_v3 为俯仰 24°/s。
FLAT_GATES = {"arm_vel_rms_rad_s": 0.4, "pitch_rate_rms_deg_s": 8.0}
LEVELS = {"0.0": "8 cm", "0.5": "12 cm", "1.0": "16 cm"}
# evaluate_parallelogram_vision：0 通过、1 未达验收门、2 执行异常
EVAL_OK_CODES = (0, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--load_run", help="logs/supermimic_terrain 下的 run 名")
    p.add_argument("--checkpoint", type=Path, help="直接给 exported/policy.pt；与 --load_run 二选一")
    p.add_argument("--difficulties", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--episodes_per_env", type=int, default=2)
    p.add_argument("--seed", type=int, default=142)
    p.add_argument("--diag_envs", type=int, default=128, help="失败诊断的环境数；0 跳过诊断")
    p.add_argument("--timeout_s", type=int, default=1500)
    p.add_argument("--skip_flat", action="store_true", help="跳过平地平稳检查")
    args = p.parse_args()
    if (args.load_run is None) == (args.checkpoint is None):
        p.error("--load_run 与 --checkpoint 必须且只能给一个")
    return args


def run(cmd, log: Path, timeout_s: int) -> int:
    with open(log, "w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[accept_terrain] 超时 {timeout_s}s\n")
            return 124


def parse_diag(log: Path) -> dict:
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("[teacher]"):
            return json.loads(line[line.index("{"):line.index("} wheel_sign") + 1])
    return {}


def main() -> int:
    args = parse_args()
    policy = (args.checkpoint or ROOT / "logs/supermimic_terrain" / args.load_run / "exported/policy.pt").resolve()
    if not policy.is_file():
        print(f"[accept_terrain] 找不到策略：{policy}", file=sys.stderr)
        return 2
    out = policy.parent.parent / "acceptance"
    out.mkdir(exist_ok=True)
    rows, broken = [], []
    for d in args.difficulties:
        result = out / f"d{d}.json"
        code = run([sys.executable, str(EVAL), "--headless", "--policy", "teacher-preview", "--supermimic_terrain",
                    "--checkpoint", str(policy), "--scenario", "pyramid_stairs_inv", "--difficulty", d,
                    "--num_envs", str(args.num_envs), "--episodes_per_env", str(args.episodes_per_env),
                    "--seed", str(args.seed), "--output", str(result)], out / f"d{d}.log", args.timeout_s)
        row = {"difficulty": d, "level": LEVELS[d], "eval_exit": code}
        if code not in EVAL_OK_CODES or not result.is_file():
            # 执行异常（合同失配、崩溃、超时）必须显式报出，不能当成 0% 通过
            broken.append(d)
            row["error"] = f"评测执行失败（exit {code}），见 {out / f'd{d}.log'}"
        else:
            j = json.loads(result.read_text())
            er = j["episode_results"]
            tp = j["terminal_progress_m"]
            fails = [i for i, e in enumerate(er) if not e["success"]]
            row.update({
                "success_rate": j["success_rate"], "episodes": j["episodes"], "passed": j["passed"],
                "climb_tilt_deg": j["climb_protocol"].get("climb_tilt_deg"),
                # 失败形态：冲出终点（≥3.75 m）还是中途停住（超时且未到终点）
                "fail_overshoot": sum(tp[i] >= 3.74 for i in fails),
                "fail_stalled": sum(tp[i] < j["target_distance_m"] and er[i]["duration_s"] >= j["episode_length_s"] - 0.05
                                    for i in fails),
            })
            row["fail_other"] = len(fails) - row["fail_overshoot"] - row["fail_stalled"]
        if args.diag_envs > 0:
            log = out / f"diag_d{d}.log"
            code = run([sys.executable, str(DIAG), "--headless", "--teacher_only", "--teacher_policy", str(policy),
                        "--difficulty", d, "--num_envs", str(args.diag_envs), "--episode_s", "18",
                        "--output", str(out / "diag")], log, args.timeout_s)
            diag = parse_diag(log) if code == 0 else {}
            row["diag"] = {k: diag[k] for k in ("reached_top_fraction", "tilt_max_deg_median", "at_target_lateral_median",
                                               "at_target_lateral_signed_mean", "at_target_lateral_within_0.15",
                                               "at_target_yaw_deg_signed_mean", "route_probe") if k in diag}
            if not diag:
                row["diag_error"] = f"诊断失败（exit {code}），见 {log}"
        rows.append(row)
        print(f"[accept_terrain] {LEVELS[d]}: " + (row.get("error") or f"{row['success_rate']:.0%}"), flush=True)

    flat = None
    if not args.skip_flat:
        log = out / "flat.log"
        code = run([sys.executable, str(FLAT), "--headless", "--policy", f"policy={policy}"], log, args.timeout_s)
        flat = {"error": f"平地检查失败（exit {code}），见 {log}"}
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("[flat_json]"):
                rows_flat = json.loads(line[len("[flat_json]"):])["policy"]
                worst = {k: max(r[k] for r in rows_flat) for k in FLAT_GATES}
                flat = {"by_command": rows_flat, "worst": worst,
                        "passed": all(worst[k] <= g for k, g in FLAT_GATES.items())}
        if "error" in flat:
            broken.append("flat")
        print(f"[accept_terrain] 平地平稳: " + (flat.get("error") or
              f"{'通过' if flat['passed'] else '❌ 抖动'} arm 速度 rms {flat['worst']['arm_vel_rms_rad_s']:.2f} rad/s，"
              f"俯仰角速度 rms {flat['worst']['pitch_rate_rms_deg_s']:.1f}°/s"), flush=True)
    summary = {"policy": str(policy), "seed": args.seed, "rows": rows, "flat": flat, "broken": broken}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ["| 楼梯 | 通过率 | 倾角门 | 失败：冲出终点 / 中途停住 / 其他 | 登顶 | 越线横向 ≤0.15 m |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['level']} | ❌ {r['error']} | | | | |")
            continue
        dg = r.get("diag", {})
        fmt = lambda k: f"{dg[k]:.0%}" if k in dg else "-"  # noqa: E731
        lines.append(f"| {r['level']} | {r['success_rate']:.0%}（{r['episodes']}） | {r['climb_tilt_deg']}° | "
                     f"{r['fail_overshoot']} / {r['fail_stalled']} / {r['fail_other']} | "
                     f"{fmt('reached_top_fraction')} | {fmt('at_target_lateral_within_0.15')} |")
    if flat and "error" not in flat:
        w = flat["worst"]
        lines.append("")
        lines.append(f"平地平稳（0/0.15/0.3 m/s 中最差）：arm 速度 rms {w['arm_vel_rms_rad_s']:.2f} rad/s（门 "
                     f"{FLAT_GATES['arm_vel_rms_rad_s']}），俯仰角速度 rms {w['pitch_rate_rms_deg_s']:.1f}°/s（门 {FLAT_GATES['pitch_rate_rms_deg_s']}）"
                     f" → {'通过' if flat['passed'] else '❌ 平地抖动'}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 2 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
