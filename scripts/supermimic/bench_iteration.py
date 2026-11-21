#!/usr/bin/env python3
"""
bench_iteration.py — SuperMimic 单轮训练耗时基准与 A/B 记录。

@author pony
@date 2026-11-05
@version v1.0.0
@last_modified 2026-11-05

两种模式：
- synthetic：不启动 Isaac。按某个 run 的 params/supermimic_terrain.json 复现 learner 配置，用合成台阶高度图、
  合成回放和模拟射线（含设备同步）跑 N 次 learner.update()，逐轮记录耗时和损失。可在两份代码树间对比
  （PYTHONPATH 指向不同 source/robot_lab），检查优化是否改变数值。
- train：原样运行 train_terrain.py，但在 learner 构造时注入 SuperMimicCfg 覆盖项（--overrides JSON），并把每次
  learner.update() 的耗时、整轮间隔和损失逐行写入 --record（jsonl）。用于在 train_terrain.py 尚未接线新参数前做
  Isaac A/B；不修改 train_terrain.py 的任何逻辑。

用法：
  python scripts/supermimic/bench_iteration.py synthetic --params <run>/params/supermimic_terrain.json \
      --iters 12 --overrides '{"diagnostics_interval": 20}' --record /tmp/a.jsonl
  python scripts/supermimic/bench_iteration.py train --overrides '{"compile_wm": true}' --record rec.jsonl -- \
      --headless --from_scratch ...（train_terrain.py 的参数）
  python scripts/supermimic/bench_iteration.py compare a.jsonl b.jsonl

@changelog
    - (2026-11-05): 当前代码树的车道惩罚 lane≠0 时传入车道中心；critic_lane 按存档参数传入。
    - v1.0.0 (2026-11-05): 初始创建（越障训练提速：reward_gap 按日志间隔、Rodrigues FK、torch.compile、镜像增强）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

LOG_KEYS = ("wm_loss", "wm_err_1step", "wm_err_last", "wm_clearance_loss", "wm_grad_norm", "wm_rehearsal_loss",
            "wm_reanchor_fraction", "wm_reward_bias", "actor_loss", "actor_model_loss", "actor_grad_norm",
            "critic_loss", "imag_reward", "imag_survival", "actor_update_applied", "actor_lr",
            "actor_mirror_loss", "wm_mirror_loss")


def _row(logs: dict) -> dict:
    return {k: float(logs[k]) for k in LOG_KEYS if k in logs}


# ======================================================================================
# train 模式：注入 cfg 覆盖 + 逐轮记录
# ======================================================================================
def _install_learner_hook(overrides: dict, record: Path, compat_legacy: bool = False) -> None:
    import importlib.abc
    import importlib.util

    record.parent.mkdir(parents=True, exist_ok=True)
    out = open(record, "a", buffering=1)

    def patch(module):
        cfg_cls = module.SuperMimicCfg
        cfg_init = cfg_cls.__init__ if compat_legacy else None

        def patched_cfg_init(self, *args, **kwargs):
            # 兼容统一目标重写之前的 train_terrain.py：已删除的字段（锚定/动作参考等）不进 dataclass，只挂成普通属性。
            dropped = {k: kwargs.pop(k) for k in list(kwargs) if k not in cfg_cls.__dataclass_fields__}
            cfg_init(self, *args, **kwargs)
            for key, value in dropped.items():
                setattr(self, key, value)

        if compat_legacy:
            cfg_cls.__init__ = patched_cfg_init
        cls = module.SuperMimicLearner
        init, update = cls.__init__, cls.update
        last = {"t": None}

        def patched_init(self, cfg, *args, **kwargs):
            for key, value in overrides.items():
                if not hasattr(cfg, key):
                    raise AttributeError(f"SuperMimicCfg 没有字段 {key}（当前代码树不支持该覆盖项）")
                setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
            init(self, cfg, *args, **kwargs)
            if compat_legacy:
                # 重写之前的 train_terrain.py 仍读取已删除的锚定/动作参考字段；这里补成"关闭"。
                for name in ("anchor_coef", "motion_reference_coef"):
                    if not hasattr(cfg, name):
                        setattr(cfg, name, 0.0)
                for name in ("anchor_actor", "motion_reference"):
                    if not hasattr(self, name):
                        setattr(self, name, None)
            out.write(json.dumps({"event": "cfg", "overrides": overrides}) + "\n")

        def patched_update(self, *args, **kwargs):
            import torch
            iteration = self.iteration
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            logs = update(self, *args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            row = {"iteration": iteration, "update_s": end - start,
                   "loop_s": (end - last["t"]) if last["t"] is not None else None, **_row(logs)}
            last["t"] = end
            out.write(json.dumps(row) + "\n")
            return logs

        cls.__init__, cls.update = patched_init, patched_update

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "robot_lab.supermimic.learner":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            exec_module = spec.loader.exec_module

            def exec_and_patch(module):
                exec_module(module)
                patch(module)

            spec.loader.exec_module = exec_and_patch
            return spec

    sys.meta_path.insert(0, Hook())


def run_train(args, passthrough):
    import runpy

    overrides = json.loads(args.overrides)
    _install_learner_hook(overrides, Path(args.record), args.compat_legacy)
    script = Path(__file__).resolve().parent / "train_terrain.py"
    sys.argv = [str(script)] + passthrough
    runpy.run_path(str(script), run_name="__main__")


# ======================================================================================
# synthetic 模式：离线复现 learner.update()
# ======================================================================================
def _import_core():
    """绕过 robot_lab/__init__（它会注册 Isaac 任务），只加载纯 torch 的 supermimic 包。"""
    import importlib.util
    spec = importlib.util.find_spec("robot_lab")
    roots = list(spec.submodule_search_locations) if spec is not None else []
    pkg = types.ModuleType("robot_lab")
    pkg.__path__ = roots
    sys.modules["robot_lab"] = pkg
    from robot_lab import supermimic  # noqa: F401
    return roots


def build_synthetic(params: dict, overrides: dict, device, seed: int = 0, sync_raycast: bool = True):
    import torch
    from collections import Counter

    from robot_lab.supermimic import terrain_map
    from robot_lab.supermimic.kinematics import RoverKinematics
    from robot_lab.supermimic.learner import SuperMimicCfg, SuperMimicLearner
    from robot_lab.supermimic.preview_task import WHEEL_RADIUS, PreviewRewardCfg, PreviewTask
    from robot_lab.supermimic.quat import quat_apply, yaw_quat
    from robot_lab.supermimic.state import JPOS, POS, QUAT, STATE_DIM

    torch.manual_seed(seed)
    raycasts = Counter()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    res, n = 0.02, 3200
    xs = torch.arange(n, device=device) * res
    heights = (((xs % 4.0) > 2.0).float() * 0.08).unsqueeze(1).expand(n, n).clone()
    hmap = terrain_map.HeightMap(heights, (0.0, 0.0), res).to(device)
    kin = RoverKinematics().to(device)
    from_y = torch.tensor([i / 10 for i in range(-5, 6)])
    from_x = torch.tensor((-0.80, -0.60, -0.40, -0.20, 0.0, 0.10, 0.16, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
                           0.32, 0.40, 0.60, 0.80))
    gx, gy = torch.meshgrid(from_x, from_y, indexing="xy")
    pattern = torch.stack((gx.flatten() + 0.3, gy.flatten(), torch.full((187,), 20.0)), -1).to(device)
    limits = torch.tensor([[-0.55, 0.55]] * 4 + [[-1.41, 1.41]] * 4, device=device)

    def fake_raycast(starts, tag):
        """Isaac raycast_mesh 的替身：相近算子数 + wp.synchronize()（设备同步）。"""
        raycasts[tag] += 1
        s = starts.reshape(-1, starts.shape[-1]).contiguous()
        out = torch.full((s.shape[0], 3), float("inf"), device=device)
        out[:, 2] = hmap.query(s[:, :2])
        if sync_raycast:
            sync()
        return out

    def exact_scan(state):
        q = yaw_quat(state[:, QUAT]).unsqueeze(1).expand(-1, pattern.shape[0], -1)
        starts = quat_apply(q, pattern.expand(state.shape[0], -1, -1)) + state[:, None, POS]
        hits = fake_raycast(starts, "scan").reshape(state.shape[0], pattern.shape[0], 3)
        return (state[:, None, 2] - hits[..., 2] - 0.5).clamp(-1, 1)

    @torch.no_grad()
    def exact_edge(state):
        shape = state.shape[:-1]
        flat = state.reshape(-1, state.shape[-1])
        jp = flat[:, JPOS]
        wheels = kin.wheel_centers_world(flat[:, POS], flat[:, QUAT], jp[:, :4], jp[:, 4:8])[:, [0, 2]]
        starts = wheels.clone()
        starts[..., 2] += 2.0
        ground = fake_raycast(starts, "edge").reshape(starts.shape)
        starts[..., 2] = ground[..., 2] + 0.015
        valid = torch.isfinite(starts).all(-1)
        safe = torch.where(valid[..., None], starts, wheels)
        dist = fake_raycast(safe, "edge").reshape(safe.shape)[..., 2].abs()
        dist = torch.where(valid, dist, torch.inf)
        return (dist - WHEEL_RADIUS).reshape(*shape, 2)

    def fake_riser(starts, directions):
        raycasts["riser"] += 1
        if sync_raycast:
            sync()
        return torch.full(starts.shape[:-1], float("inf"), device=device)

    def make_task(reward_cfg, **kwargs):
        """按当前代码树 PreviewTask 的签名传参（重写前后接口不同：lane_obs / exact_prelift_edge_fn / exact_riser_fn）。"""
        import inspect
        accepted = inspect.signature(PreviewTask.__init__).parameters
        if "exact_riser_fn" in accepted:
            kwargs["exact_riser_fn"] = fake_riser
        return PreviewTask(hmap, kin, pattern, limits, reward_cfg, exact_scan_fn=exact_scan,
                           **{k: v for k, v in kwargs.items() if k in accepted})

    rcfg = dict(params["reward"])
    for key in ("lift_lookahead", "riser_gap"):
        if key in rcfg:
            rcfg[key] = tuple(rcfg[key])
    known = PreviewRewardCfg.__dataclass_fields__
    reward_cfg = PreviewRewardCfg(**{k: v for k, v in rcfg.items() if k in known})
    args = params.get("arguments", {})
    lane_obs = bool(args.get("lane_obs", False))
    lanes = torch.arange(0.0, 64.0, 8.0, device=device) + 4.0
    needs_lanes = lane_obs or getattr(reward_cfg, "lateral_pos", 0.0) or getattr(reward_cfg, "lane", 0.0)
    task = make_task(reward_cfg, lane_center_y=lanes if needs_lanes else None, lane_obs=lane_obs,
                     critic_lane=bool(args.get("critic_lane", False)),
                     critic_route_features=bool(args.get("critic_route_features", False)),
                     exact_prelift_edge_fn=exact_edge if getattr(reward_cfg, "prelift_edge_source", "") == "mesh" else None)
    c = dict(params["cfg"])
    c.pop("reward", None)
    for key in ("wm_hidden", "actor_hidden", "critic_hidden"):
        c[key] = tuple(c[key])
    known = SuperMimicCfg.__dataclass_fields__
    c = {k: v for k, v in c.items() if k in known}
    # 当前代码已删除 Teacher 锚定与动作参考；旧代码树里也关掉，保证 A/B 计算图一致。
    for key in ("anchor_coef", "motion_reference_coef"):
        if key in c:
            c[key] = 0.0
    for key, value in overrides.items():
        if key not in known:
            raise AttributeError(f"SuperMimicCfg 没有字段 {key}")
        c[key] = tuple(value) if isinstance(value, list) else value
    cfg = SuperMimicCfg(**c)
    learner = SuperMimicLearner(cfg, device, extra_fn=lambda s: terrain_map.relief_features(hmap, kin, s),
                                extra_dim=terrain_map.RELIEF_DIM, task=task)

    def make_traj(count, T, generator):
        s = torch.zeros(count, T + 1, STATE_DIM, device=device)
        s[:, 0, 0] = torch.rand(count, device=device, generator=generator) * 50 + 5
        s[:, 0, 1] = torch.rand(count, device=device, generator=generator) * 50 + 5
        s[:, 0, 2] = 0.35
        s[:, 0, 3] = 1.0
        for t in range(T):
            s[:, t + 1] = s[:, t] + 0.01 * torch.randn(count, STATE_DIM, device=device, generator=generator)
            s[:, t + 1, 3:7] = torch.nn.functional.normalize(s[:, t + 1, 3:7], dim=-1)
            s[:, t + 1, 13 + 8:13 + 12] = 0
        return s

    data_gen = torch.Generator(device=device).manual_seed(seed + 7)

    class FakeBundle:
        def __init__(self):
            count, T = 64, 48
            self.states = make_traj(count, T, data_gen)
            self.actions = torch.rand(count, T, 12, device=device, generator=data_gen) - 0.5
            self.context = torch.ones(count, T, c["context_dim"], device=device)
            self.hmap, self.kin = hmap, kin
            self.task = make_task(PreviewRewardCfg(**{k: v for k, v in rcfg.items()
                                                      if k in PreviewRewardCfg.__dataclass_fields__}),
                                  lane_center_y=lanes, lane_obs=True, exact_prelift_edge_fn=exact_edge)

        def sample(self, count, length, *, generator=None):
            ids = torch.randint(self.actions.shape[0], (count,), device=device, generator=generator)
            starts = torch.randint(self.actions.shape[1] - length + 1, (count,), device=device, generator=generator)
            time_ = starts[:, None] + torch.arange(length, device=device)
            state_time = starts[:, None] + torch.arange(length + 1, device=device)
            return {"state": self.states[ids[:, None], state_time], "action": self.actions[ids[:, None], time_],
                    "context": self.context[ids[:, None], time_], "valid": torch.ones(count, length, device=device)}

        def extra_fn(self, state):
            return terrain_map.relief_features(self.hmap, self.kin, state)

    if getattr(cfg, "wm_rehearsal_coef", 0.0):  # 重写之前的代码树才有保留回放
        learner.wm_rehearsal = [FakeBundle(), FakeBundle()]
    rb = learner.replay
    traj = make_traj(cfg.num_envs, cfg.buffer_len - 1, data_gen).transpose(0, 1)
    for t in range(cfg.buffer_len):
        rb.add(traj[t], torch.rand(cfg.num_envs, 12, device=device, generator=data_gen) - 0.5,
               torch.rand(cfg.num_envs, 12, device=device, generator=data_gen) - 0.5,
               torch.tensor([0.3, 0.0, 0.0], device=device).expand(cfg.num_envs, 3),
               torch.rand(cfg.num_envs, device=device, generator=data_gen) < 0.005,
               torch.ones(cfg.num_envs, c["context_dim"], device=device),
               reward=torch.randn(cfg.num_envs, device=device, generator=data_gen) * 0.01,
               terminated=torch.zeros(cfg.num_envs, dtype=torch.bool, device=device),
               priority=(torch.rand(cfg.num_envs, device=device, generator=data_gen) < 0.1).float())
    return learner, raycasts, sync


def run_synthetic(args):
    import torch

    roots = _import_core()
    device = torch.device(args.device)
    params = json.loads(Path(args.params).read_text())
    overrides = json.loads(args.overrides)
    learner, raycasts, sync = build_synthetic(params, overrides, device, seed=args.seed,
                                              sync_raycast=not args.no_sync)
    learner.iteration = args.start_iteration
    out = open(args.record, "w") if args.record else None
    print(f"[bench] code={roots} device={device} overrides={overrides}", flush=True)
    times = []
    for i in range(args.iters):
        sync()
        start = time.perf_counter()
        logs = learner.update()
        sync()
        dt = time.perf_counter() - start
        times.append(dt)
        row = {"iteration": learner.iteration - 1, "update_s": dt, **_row(logs)}
        if out:
            out.write(json.dumps(row) + "\n")
        print(f"[bench] it {row['iteration']} {dt * 1000:8.1f} ms  " +
              " ".join(f"{k}={v:.6g}" for k, v in row.items() if k not in ("iteration", "update_s")), flush=True)
    steady = times[args.skip:] or times
    print(f"[bench] mean {sum(steady) / len(steady) * 1000:.1f} ms/update over {len(steady)} (skip {args.skip}); "
          f"raycasts/iter={sum(raycasts.values()) / max(len(times), 1):.0f}", flush=True)


# ======================================================================================
# compare：两份 jsonl 的逐轮差异
# ======================================================================================
def _load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "iteration" in row:
            rows[row["iteration"]] = row
    return rows


def run_compare(args):
    a, b = _load(args.a), _load(args.b)
    common = sorted(set(a) & set(b))
    print(f"common iterations: {len(common)}")

    def mean_time(rows, key):
        values = [r[key] for i, r in rows.items() if r.get(key) is not None and i >= args.skip]
        return sum(values) / len(values) if values else float("nan")

    for key in ("update_s", "loop_s"):
        print(f"{key:10s} A={mean_time(a, key):.3f}s  B={mean_time(b, key):.3f}s")
    for key in LOG_KEYS:
        pairs = [(a[i][key], b[i][key]) for i in common if key in a[i] and key in b[i]]
        if not pairs:
            continue
        diffs = [abs(x - y) for x, y in pairs]
        rel = [abs(x - y) / max(abs(x), abs(y), 1e-12) for x, y in pairs]
        first = next((i for i, (x, y) in zip([i for i in common if key in a[i] and key in b[i]], pairs) if x != y), None)
        ma = sum(x for x, _ in pairs) / len(pairs)
        mb = sum(y for _, y in pairs) / len(pairs)
        print(f"{key:22s} n={len(pairs):4d} meanA={ma:.6g} meanB={mb:.6g} max|d|={max(diffs):.3g} "
              f"max_rel={max(rel):.3g} first_diff_it={first}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    syn = sub.add_parser("synthetic")
    syn.add_argument("--params", required=True)
    syn.add_argument("--iters", type=int, default=10)
    syn.add_argument("--skip", type=int, default=2)
    syn.add_argument("--start_iteration", type=int, default=8000)
    syn.add_argument("--overrides", default="{}")
    syn.add_argument("--record", default=None)
    syn.add_argument("--device", default="cuda")
    syn.add_argument("--seed", type=int, default=0)
    syn.add_argument("--no_sync", action="store_true", help="模拟射线不做设备同步")
    tr = sub.add_parser("train")
    tr.add_argument("--overrides", default="{}")
    tr.add_argument("--record", required=True)
    tr.add_argument("--compat_legacy", action="store_true",
                    help="在统一目标重写之前的代码树里运行旧 train_terrain.py（吞掉已删除的锚定/动作参考字段），仅用于 A/B")
    cmp = sub.add_parser("compare")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--skip", type=int, default=0)
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        index = argv.index("--")
        argv, passthrough = argv[:index], argv[index + 1:]
    args = parser.parse_args(argv)
    if args.mode == "train":
        run_train(args, passthrough)
    elif args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
