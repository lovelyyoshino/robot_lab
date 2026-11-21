#!/usr/bin/env python3
"""
export_ckpt.py — 把越障 run 的某个 ckpt_XXXXXX.pt 导出成可验收的 TorchScript，放在 <run>/ckpt_eval/itXXXX/。

一个 run 训练中会退化（2026-10-28：第 1000 轮 67%/81%，第 2000 轮 6%/0%），验收要能挑中间存档；
params/ 整个复制过去，方案 B 的 lane_offset_obs.json 标记随之生效。不启动 Isaac。

@author pony
@date 2026-10-28
@version v1.2.0
@last_modified 2026-11-16
@changelog
    - v1.2.0 (2026-11-16): 存档 task_features 带 lead_side_obs（r21 起，actor 233 维 = 232 + 先抬侧标记）时包 LeadSideActor，
      标记固定 --lead（默认 +1），导出仍是 232 维输入。原来直接导出 233 维 actor，accept.py 把第 233 维当成车道偏移观测喂进去，
      r21 第 2000–14000 轮的验收都用错了输入
    - v1.1.0 (2026-11-10): 存档带 actor_ema 时默认导出 EMA 权重（--raw 导出原始 actor）
    - v1.0.0 (2026-10-28): 初始创建
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robot_lab/robot_lab"))
from supermimic.networks import Actor, LeadSideActor  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
p.add_argument("--load_run", required=True)
p.add_argument("--iterations", type=int, nargs="+", required=True)
p.add_argument("--raw", action="store_true", help="存档带 actor_ema 时仍导出原始 actor（默认导出 EMA 权重）")
p.add_argument("--lead", type=float, default=1.0, choices=(1.0, -1.0),
               help="先抬侧标记存档（lead_side_obs）导出时固定的标记值，默认 +1，与训练脚本导出一致")
args = p.parse_args()
run = ROOT / "logs/supermimic_terrain" / args.load_run
for it in args.iterations:
    ck = torch.load(run / f"ckpt_{it:06d}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    obs_dim = ck["actor"]["actor.0.weight"].shape[1]
    actor = Actor(obs_dim, tuple(cfg["actor_hidden"]), cfg["action_bound"], squash=cfg["action_squash"])
    use_ema = "actor_ema" in ck and not args.raw
    actor.load_state_dict(ck["actor_ema"] if use_ema else ck["actor"])
    lead = bool((ck.get("task_features") or {}).get("lead_side_obs"))
    if lead:
        # 最后一维是先抬侧标记，不是车道偏移；部署接口不含它
        actor = LeadSideActor(actor.eval(), args.lead)
        obs_dim -= 1
    out = run / "ckpt_eval" / f"it{it}"
    (out / "exported").mkdir(parents=True, exist_ok=True)
    shutil.copytree(run / "params", out / "params", dirs_exist_ok=True)
    torch.jit.script(actor.eval()).save(str(out / "exported" / "policy.pt"))
    print(f"[export_ckpt] {out / 'exported' / 'policy.pt'}（{obs_dim} 维输入，{'EMA' if use_ema else '原始'} 权重"
          f"{f'，先抬侧标记固定 {args.lead:+.0f}' if lead else ''}）")
