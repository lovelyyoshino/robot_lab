#!/usr/bin/env python3
"""
export_terrain_teacher.py — 把 SuperMimic 越障策略（train_terrain.py 产物）转成 rsl_rl 格式的 Teacher checkpoint，
供 rsl_rl/train.py 的 Student DAgger 蒸馏直接用 --checkpoint 加载。

为什么能转：越障 actor 从 Teacher-Preview 热启动，网络结构（232→512/256/128→12、ELU）和 rsl_rl 的 actor.* 键名完全一致，
观测/动作合同也相同（run 目录的 params/vision_contract.json 就是从 Teacher 复制的）。
差别只有输出上限：SuperMimic 部署时把动作 clamp 到 ±2π，rsl_rl Teacher 没有 clamp（agent clip_actions=None）。
本脚本报告越界比例；越界时 Student 学到的标签会和 SuperMimic 部署行为不一致，需要用 --verify_obs 实测确认。

产物：logs/supermimic_terrain/<run>/rsl_rl_teacher/model_<iter>.pt + params/（agent.yaml、env.yaml、vision_contract.json，
复制自热启动 Teacher）+ exported/policy.pt（原 TorchScript）+ lineage.json。不启动 Isaac。

@author pony
@date 2026-10-27
@version v1.0.0
@last_modified 2026-10-27
@changelog
    - v1.0.0 (2026-10-27): 初始创建；让 Student 能以世界模型训出的越障策略为老师。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robot_lab/robot_lab"))
from supermimic.networks import Actor  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="SuperMimic 越障策略 → rsl_rl Teacher checkpoint")
    p.add_argument("--load_run", required=True, help="logs/supermimic_terrain 下的 run 名")
    p.add_argument("--checkpoint", default=None, help="ckpt 文件名；省略取该 run 最新 ckpt_*.pt")
    p.add_argument("--verify_obs", type=Path, default=None,
                   help="可选：climb_search.py --record 产出的 npz（需含 obs），统计真实观测上的越界比例")
    args = p.parse_args()

    run = ROOT / "logs/supermimic_terrain" / args.load_run
    if (run / "params" / "lane_offset_obs.json").is_file():
        raise SystemExit("该 run 是方案 B（233 维，含车道偏移观测）；Student 观测尚无此项，暂不能转换")
    ckpt = run / args.checkpoint if args.checkpoint else sorted(run.glob("ckpt_*.pt"))[-1]
    meta = json.loads((run / "params/supermimic_terrain.json").read_text())
    teacher_ckpt = Path(meta["teacher"])
    sm = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = sm["cfg"]
    if cfg["action_squash"] != "clamp":
        raise ValueError(f"只支持 clamp 输出的越障 actor（实际 {cfg['action_squash']}），tanh 无法等价成 rsl_rl 线性输出")

    src = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    state = dict(src["model_state_dict"])
    actor_keys = [k for k in state if k.startswith("actor.")]
    if sorted(actor_keys) != sorted(sm["actor"]):
        raise ValueError(f"越障 actor 键与 Teacher 不一致：{sorted(sm['actor'])} vs {sorted(actor_keys)}")
    for k in actor_keys:
        if state[k].shape != sm["actor"][k].shape:
            raise ValueError(f"{k} 形状不一致：{tuple(sm['actor'][k].shape)} vs {tuple(state[k].shape)}")
        state[k] = sm["actor"][k].clone()
    # critic / std 保留 Teacher 原值：蒸馏只读 actor.*；std 只影响 PPO 采样，不进入 Student 标签
    out_dir = run / "rsl_rl_teacher"
    (out_dir / "params").mkdir(parents=True, exist_ok=True)
    (out_dir / "exported").mkdir(exist_ok=True)
    out = out_dir / f"model_{sm['iteration']}.pt"
    torch.save({"model_state_dict": state, "optimizer_state_dict": src.get("optimizer_state_dict", {}),
                "iter": int(sm["iteration"]), "infos": {"source": "supermimic_terrain", "run": args.load_run}}, out)
    for name in ("agent.yaml", "env.yaml", "vision_contract.json"):
        shutil.copy(teacher_ckpt.parent / "params" / name, out_dir / "params" / name)
    shutil.copy(run / "exported/policy.pt", out_dir / "exported/policy.pt")

    # 等价性：rsl_rl actor（无 clamp）与 SuperMimic actor（clamp）在界内输出逐位相同
    actor = Actor(232, tuple(cfg["actor_hidden"]), cfg["action_bound"], squash="clamp")
    actor.load_state_dict(sm["actor"])
    actor.eval()
    report = {"source_ckpt": str(ckpt), "source_sha256": sha256(ckpt), "teacher_template": str(teacher_ckpt),
              "teacher_template_sha256": sha256(teacher_ckpt), "output": str(out), "output_sha256": sha256(out),
              "action_bound": cfg["action_bound"]}
    if args.verify_obs is not None:
        import numpy as np

        z = np.load(args.verify_obs)
        obs = torch.from_numpy(z["obs"]).float().reshape(-1, 232)
        with torch.no_grad():
            raw = actor.actor(obs)
        over = raw.abs() > cfg["action_bound"]
        report["verify_obs"] = str(args.verify_obs)
        report["samples"] = int(obs.shape[0])
        report["out_of_bound_fraction"] = over.float().mean().item()
        report["out_of_bound_fraction_per_joint"] = [round(v, 5) for v in over.float().mean(0).tolist()]
        report["max_abs_raw"] = raw.abs().max().item()
    (out_dir / "lineage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[export_terrain_teacher] Student 蒸馏用：--checkpoint {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
