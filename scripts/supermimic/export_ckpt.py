#!/usr/bin/env python3
"""
export_ckpt.py — 把越障 run 的某个 ckpt_XXXXXX.pt 导出成可验收的 TorchScript，放在 <run>/ckpt_eval/itXXXX/。

一个 run 训练中会退化（2026-10-28：第 1000 轮 67%/81%，第 2000 轮 6%/0%），验收要能挑中间存档；
params/ 整个复制过去，方案 B 的 lane_offset_obs.json 标记随之生效。不启动 Isaac。

@author pony
@date 2026-10-28
@version v1.0.0
@last_modified 2026-10-28
@changelog
    - v1.0.0 (2026-10-28): 初始创建
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robot_lab/robot_lab"))
from supermimic.networks import Actor  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
p.add_argument("--load_run", required=True)
p.add_argument("--iterations", type=int, nargs="+", required=True)
args = p.parse_args()
run = ROOT / "logs/supermimic_terrain" / args.load_run
for it in args.iterations:
    ck = torch.load(run / f"ckpt_{it:06d}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    obs_dim = ck["actor"]["actor.0.weight"].shape[1]
    actor = Actor(obs_dim, tuple(cfg["actor_hidden"]), cfg["action_bound"], squash=cfg["action_squash"])
    actor.load_state_dict(ck["actor"])
    out = run / "ckpt_eval" / f"it{it}"
    (out / "exported").mkdir(parents=True, exist_ok=True)
    shutil.copytree(run / "params", out / "params", dirs_exist_ok=True)
    torch.jit.script(actor.eval()).save(str(out / "exported" / "policy.pt"))
    print(f"[export_ckpt] {out / 'exported' / 'policy.pt'}（{obs_dim} 维输入）")
