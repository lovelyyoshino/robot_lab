"""
obstacle_terrain.py — SuperMimic 越障训练地形几何与按地形类型的课程（纯 numpy/torch，单测可按路径加载，不依赖 Isaac）。

地形（每块 8×8 m，机器人从块中心出发沿 +X）：
- 楼梯 pyramid_stairs_inv：5 级，立面 10→20 cm 随难度连续变化，踏面每块在 0.26–0.40 m 均匀抽取；
- 单障碍 single_step / _left / _right：0.40 m 长的方块，6→20 cm，左右窄道宽 0.36 m；
- 平地 flat。
课程（每个环境的地形类型固定，等级按类型的等效高度解释）：
- 晋级：前进 ≥ promote_m（越过全部障碍、约半块长）且 |y| ≤ success_lane_m、姿态超限步数 ≤ posture_bad_frac；
  碰立面步数 ≤ collision_frac（单障碍）/ stairs_collision_frac（楼梯，默认 1.0 即不设门）；
- 降级：翻车 / 出车道（|y| > fail_lane_m）/ 前进距离 < demote_ratio × 期望距离（期望 = 命令速度积分，停滞计入；
  楼梯用 stairs_demote_ratio）；
- 停滞截断（stall_truncate_s > 0）：命令前进时连续停滞超过该秒数即截断回合（time_out），按上一条判定升降级；
- 通关最高级随机回落到 [0, max_level)；
- 楼梯探索（stairs_explore_frac）：每次 reset 时楼梯环境按该比例抽成"探索回合"，地形等级在 [0, max_level) 均匀抽取；
  探索回合不改课程等级（课程等级另存于 env.obstacle_base_levels），只记录各高度的通过情况。
  前七轮楼梯课程一直停在 6–8 cm（10 级里的第 1–2 级），12–17.5 cm 的立面在训练里几乎从未出现，世界模型没有
  这些高度上后轴翻越的数据，想象梯度也就无从指向。
终止：x ≥ promote_m 时回合必然结束——|y| ≤ success_lane_m 为成功截断（time_out），否则与 |y| > fail_lane_m
一样按出车道真终止（楼梯顶端平台在 x≈4 m 处是块边断崖，到终点后不能继续行驶）。
goal_lane_termination = 0 时到终点一律成功截断，只有 |y| > fail_lane_m 才终止（横偏交给稠密代价，不再判失败）。

@author pony
@date 2026-11-05
@version v1.7.0
@last_modified 2026-11-21
@changelog
    - v1.7.0 (2026-11-21): goal_lane_termination（默认 1 = 原行为）：0 时到终点不看横偏一律成功截断，出车道只剩
      |y| > fail_lane_m。r27@36k 的 critic 探针：单台阶、平地 46–48% 的回合因到终点 |y| > 0.15 m 被真终止，
      楼梯爬升段 critic 给横偏的估值反而更高（继续开的价值约 −23，终止只扣 0.4）。终止判据计算时把相对块原点的
      位置存到 env.obstacle_goal_rel（训练循环按它记到终点横偏的分布）。
    - v1.6.0 (2026-11-13): 楼梯单独的成功车道 stairs_success_lane_m（0 = 同 success_lane_m，原行为），课程晋级与终点
      截断/出车道终止都按地形类型取；lane_schedule_scale(it, start, end) 给训练循环逐步收紧用。第十七轮 it12000 楼梯
      12 cm×40 cm 98% 爬上顶，但斜 9–11° 没有蟹行、横偏 0.21–0.32 m，验收 0%；训练里同样到终点 |y|>0.15 判出车道终止，
      楼梯课程 it10000 起停在 8.2–8.6 cm，16 cm 以上从未练过。
    - v1.5.0 (2026-11-11): 楼梯单独的课程迟滞 stairs_promote_streak / stairs_demote_streak（0 = 与通用值相同）、
      非楼梯课程冻结 freeze_non_stairs、楼梯最低等级 stairs_min_level（默认均为原行为）。第十四轮迟滞 2/2 下楼梯
      升级率 0.02–0.12（第十三轮 0.15–0.48），楼梯课程从 it6000 到 it38000 一直停在 8.0–8.6 cm，没练过 12 cm 以上。
    - v1.4.0 (2026-11-10): 楼梯踏面下限随立面提高（tread_rule，默认关闭）；课程冻结 freeze 与迟滞 promote_streak /
      demote_streak（默认 1 = 原行为）。第十三轮楼梯降级率常年 0.2–0.5、课程在 8–9.5 cm 来回，部分来自几何上不可行的
      "高立面 + 短踏面"地块；热启动打磨阶段需要固定课程。
    - v1.3.0 (2026-11-09): stairs_demote_ratio（楼梯单独的降级距离比例）与 stall_truncate_s / stall_timeout（连续停滞截断，
      默认关闭）。第十二轮（第十轮同样）楼梯课程每次升到 9–11 cm 都引发停滞比例 ×10、世界模型末步误差 ×2、actor 损失 ×5，
      已学会的平地/单台阶行为一起退化，课程回落后才恢复。
    - v1.2.0 (2026-11-07): 楼梯探索回合 stairs_explore_frac（默认 0 = 原行为）；课程等级与回合地形等级分开存放。
    - v1.1.0 (2026-11-05): 成功/晋级要求 |y| ≤ success_lane_m，出车道 0.75→0.40 m；楼梯取消碰立面晋级门。
    - v1.0.0 (2026-11-05): 初始创建（高楼梯、随机踏面、按类型课程）
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

STAIRS = "pyramid_stairs_inv"  # 沿用旧名，评测/诊断按此名识别楼梯
STAIR_RISER_M = (0.04, 0.20)  # 从零训练须从可滚过的低台阶起步（10 cm≈轮半径时第 0 级无法学习）
STAIR_TREAD_M = (0.26, 0.40)
STAIR_STEPS = 5
STAIR_FIRST_EDGE_M = 1.20
SINGLE_HEIGHT_M = (0.03, 0.20)
SINGLE_FRONT_X_M = (1.20, 1.70)
SINGLE_LENGTH_M = 0.40
NARROW_WIDTH_M = 0.36
NARROW_CENTER_Y_M = 0.30
NUM_ROWS = 10
NUM_COLS = 20
# 列按 Isaac Lab 课程生成器的比例边界分配：单障碍 8 列（40%）、楼梯 9 列（45%）、平地 3 列（15%）。
MIX = {"single_step": 0.16, "single_step_left": 0.12, "single_step_right": 0.12, STAIRS: 0.45, "flat": 0.15}
KIND_SINGLE, KIND_STAIRS, KIND_FLAT = 0, 1, 2
KIND_NAMES = {KIND_SINGLE: "single", KIND_STAIRS: "stairs", KIND_FLAT: "flat"}
PROMOTE_M = 3.60  # 楼梯最后一级前缘 ≤ 1.2 + 4×0.40 = 2.8 m；3.6 m 时后轮已登顶。也是成功截断距离
# 楼梯第 k 级立面 = 4 + (k + U)/10 × 16 cm：第 5 级起 ≥12 cm，即验收的 12/16/17.5 cm 区间（探索日志分档用）
EXPLORE_HIGH_LEVEL = 5
# success_lane_m 与验收"横移 ≤0.15 m"一致；楼梯碰立面比例恒 ≤1，stairs_collision_frac=1.0 即楼梯不设碰撞门
# （碰撞门曾让楼梯课程停在约 6 cm）。
# stairs_demote_ratio：楼梯的降级距离比例（默认同 demote_ratio = 原行为）。第十二轮楼梯回合 30–45% 既不晋级也不降级——
# 卡在第 3–5 级立面前、前进量 > 0.5×期望的环境留在同一等级反复卡满 40 s，课程一升高停滞比例就涨 10 倍、把回放灌满。
# stall_truncate_s：命令前进时连续停滞（世界系前进速度 < stall_speed_ratio×命令）超过该秒数即截断回合（time_out 语义，
# 0 = 关闭）；停滞步数由训练循环写入 env.obstacle_stall_steps，验收/其他任务不写该属性，不受影响。
CURRICULUM_DEFAULTS = {"promote_m": PROMOTE_M, "success_lane_m": 0.15, "demote_ratio": 0.5, "posture_bad_frac": 0.05,
                       "collision_frac": 0.10, "stairs_collision_frac": 1.0, "fail_lane_m": 0.40, "upright_cos": 0.85,
                       "roll_limit_deg": 10.0, "pitch_dev_limit_deg": 12.0, "stairs_explore_frac": 0.0,
                       "stairs_demote_ratio": 0.5, "stall_truncate_s": 0.0,
                       # 课程冻结（1 = 等级不再升降，打磨阶段用）与迟滞（连续 N 次通过才升、连续 M 次失败才降；1 = 原行为）
                       "freeze": 0.0, "promote_streak": 1.0, "demote_streak": 1.0,
                       # 楼梯单独的迟滞（0 = 同上）、只冻结单台阶/平地（楼梯补练用）、楼梯课程等级下限
                       "stairs_promote_streak": 0.0, "stairs_demote_streak": 0.0, "freeze_non_stairs": 0.0,
                       "stairs_min_level": 0.0,
                       # 楼梯的成功车道（0 = 同 success_lane_m）；训练循环可按 lane_schedule_scale 从宽到严逐步收紧
                       "stairs_success_lane_m": 0.0,
                       # 到终点且 |y| > 成功车道时是否按出车道真终止（1 = 原行为；0 = 一律成功截断，只有 |y| > fail_lane_m 终止）
                       "goal_lane_termination": 1.0}


def tile_fraction(seed, difficulty: float, salt: str) -> float:
    """按难度派生每块的确定性随机数，不消耗地形生成器的 RNG。"""
    digest = hashlib.blake2b(f"{seed}:{salt}:{difficulty:.9f}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def min_tread_for_riser(riser: float, tread_lo: float, tread_rule) -> float:
    """踏面下限随立面高度提高：tread_rule = (r0, r1, t1)，立面 ≤ r0 时下限不变，≥ r1 时为 t1，中间线性。

    依据（第十四轮静态可行性，URDF 尺寸、横滚 ≤8°、三轮支撑裕度）：立面 16–17.5 cm 时踏面 ≤32 cm 前轮无法在
    三轮支撑下抬过立面，踏面 28 cm 时前后轴同时顶在立面上、没有轴能发力，训练里这类地块必然失败、反复触发降级。"""
    if tread_rule is None:
        return tread_lo
    r0, r1, t1 = (float(v) for v in tread_rule)
    if not (math.isfinite(r0) and math.isfinite(r1) and math.isfinite(t1)) or r1 <= r0 or t1 < tread_lo:
        raise ValueError("tread_rule 须为 (r0 < r1, t1 ≥ 踏面下限)")
    return tread_lo + (t1 - tread_lo) * min(max((riser - r0) / (r1 - r0), 0.0), 1.0)


def stairs_dimensions(difficulty: float, riser_range=STAIR_RISER_M, tread_range=STAIR_TREAD_M, seed=None,
                      tread_rule=None):
    """(立面高, 踏面深)：高度随难度线性，踏面每块在 [下限(立面), 上限] 内均匀抽取（tread_rule=None 时下限固定）。"""
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("楼梯难度必须位于[0,1]")
    riser = riser_range[0] + difficulty * (riser_range[1] - riser_range[0])
    lo = min(min_tread_for_riser(riser, tread_range[0], tread_rule), tread_range[1])
    tread = lo + tile_fraction(seed, difficulty, "tread") * (tread_range[1] - lo)
    return riser, tread


def straight_stairs_terrain(difficulty: float, cfg):
    """沿 +X 连续上楼；每级是一直延伸到块边的实体盒，顶端平台与最后一级同高。"""
    import trimesh

    riser, tread = stairs_dimensions(difficulty, cfg.riser_range, cfg.tread_range, cfg.seed,
                                     getattr(cfg, "tread_rule", None))
    last_edge = cfg.first_edge_m + (cfg.num_steps - 1) * cfg.tread_range[1]
    if last_edge + 0.8 >= cfg.size[0] / 2 or not 0 < cfg.lane_width <= cfg.size[1]:
        raise ValueError("楼梯须容纳后轮登顶与终点平台")
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    ground = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.10))
    ground.apply_translation(origin + np.array([0.0, 0.0, -0.05]))
    meshes, end = [ground], cfg.size[0] / 2
    for k in range(cfg.num_steps):
        edge = cfg.first_edge_m + k * tread
        step = trimesh.creation.box(extents=(end - edge, cfg.lane_width, riser))
        step.apply_translation(origin + np.array([(edge + end) / 2, 0.0, (k + 0.5) * riser]))
        meshes.append(step)
    return meshes, origin


def column_names(names, proportions, num_cols):
    """与 Isaac Lab 课程生成器（vision_preview.terrain_column_names）相同的列分配。"""
    bounds = np.cumsum(np.asarray(proportions, dtype=np.float64)) / float(sum(proportions))
    positions = np.arange(num_cols, dtype=np.float64) / num_cols + 0.001
    return tuple(names[i] for i in np.searchsorted(bounds, positions, side="right"))


def column_kinds(names, proportions, num_cols) -> tuple[int, ...]:
    return tuple(KIND_STAIRS if name == STAIRS else KIND_SINGLE if name.startswith("single_step") else KIND_FLAT
                 for name in column_names(names, proportions, num_cols))


def equivalent_height_cm(levels: torch.Tensor, kind: int, rows: int = NUM_ROWS) -> torch.Tensor:
    """课程等级对应的平均障碍高度（块内难度 = (行 + U)/行数）。"""
    lo, hi = STAIR_RISER_M if kind == KIND_STAIRS else SINGLE_HEIGHT_M if kind == KIND_SINGLE else (0.0, 0.0)
    return (lo + (levels.float() + 0.5) / rows * (hi - lo)) * 100.0


class EpisodeTracker:
    """每个环境本回合的立面碰撞步数、姿态超限步数、总步数与命令距离；由训练循环逐步更新、课程在 reset 时取走。"""

    def __init__(self, num_envs: int, device):
        self.collisions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.posture_bad = torch.zeros_like(self.collisions)
        self.steps = torch.zeros_like(self.collisions)
        self.commanded_m = torch.zeros(num_envs, device=device)

    @torch.no_grad()
    def update(self, collided, posture_bad, command_x, dt: float) -> None:
        self.collisions += collided.long()
        self.posture_bad += posture_bad.long()
        self.steps += 1
        self.commanded_m += command_x.clamp_min(0.0) * dt

    def pop(self, ids) -> dict[str, torch.Tensor]:
        stats = {name: getattr(self, name)[ids].clone() for name in ("collisions", "posture_bad", "steps", "commanded_m")}
        for name in stats:
            getattr(self, name)[ids] = 0
        return stats


def success_lane(kinds, cfg, like):
    """每个环境的成功车道 |y| 上限：楼梯用 stairs_success_lane_m（>0 时），其它类型用 success_lane_m。"""
    get = lambda key: float(cfg.get(key, CURRICULUM_DEFAULTS[key]))  # noqa: E731
    base, stairs = get("success_lane_m"), get("stairs_success_lane_m")
    if kinds is None or stairs <= 0.0:
        return like.new_tensor(base)
    return torch.where(kinds == KIND_STAIRS, like.new_tensor(stairs), like.new_tensor(base))


def lane_schedule_scale(iteration: int, start: int, end: int) -> float:
    """车道约束收紧系数：iteration ≤ start 为 0（放松），之后线性升到 end 时的 1（完全收紧）。"""
    if end <= start:
        return 0.0 if iteration < start else 1.0
    return min(max((iteration - start) / (end - start), 0.0), 1.0)


def curriculum_decision(progress_m, lateral_m, upright, active, kinds, stats, cfg=CURRICULUM_DEFAULTS):
    """(passed, failed) 布尔张量。stats 为 None（无跟踪器）时只按距离/横偏/翻车判定。"""
    get = lambda key: cfg.get(key, CURRICULUM_DEFAULTS[key])  # noqa: E731
    reached = (progress_m >= cfg["promote_m"]) & (lateral_m.abs() <= success_lane(kinds, cfg, progress_m))
    in_lane = lateral_m.abs() <= cfg["fail_lane_m"]
    if stats is None:
        clean = posture_ok = torch.ones_like(reached)
        expected = torch.full_like(progress_m, cfg["promote_m"])
    else:
        # 碰立面步数 ≤ collision_frac：零碰撞是验收标准，用作课程门槛会让低级别永远拿不到晋级信号。
        # 楼梯每级都要碰/抬，按单障碍的 10% 门会把课程卡在低级（默认 stairs_collision_frac=1.0 不设门）。
        frac = torch.where(kinds == KIND_STAIRS, get("stairs_collision_frac"), get("collision_frac"))
        clean = (stats["collisions"].float() <= frac * stats["steps"].clamp_min(1).float()) | (kinds == KIND_FLAT)
        posture_ok = stats["posture_bad"].float() <= cfg["posture_bad_frac"] * stats["steps"].clamp_min(1).float()
        expected = stats["commanded_m"].clamp(max=cfg["promote_m"])
    passed = active & upright & in_lane & reached & clean & posture_ok
    ratio = torch.where(kinds == KIND_STAIRS, progress_m.new_tensor(float(get("stairs_demote_ratio"))),
                        progress_m.new_tensor(float(cfg["demote_ratio"])))
    failed = active & ~passed & (~upright | ~in_lane | (progress_m < ratio * expected))
    return passed, failed


def _apply_streaks(env, ids, passed, failed, cfg, kinds=None):
    """课程冻结与迟滞：返回实际用于升降级的 (passed, failed)。

    冻结时等级不变（含通关回落）；freeze_non_stairs 只冻结单台阶/平地。迟滞时本回合判定计入连续计数，计满才升/降并清零，
    其它结果清零对应计数；楼梯可用 stairs_promote_streak / stairs_demote_streak 单独设定（0 = 与通用值相同）。
    kinds 为本批环境的地形类型（None 时不区分类型）。"""
    if float(cfg.get("freeze", 0.0)) > 0.5:
        return torch.zeros_like(passed), torch.zeros_like(failed)
    streak = lambda key: int(round(float(cfg.get(key, 0.0))))  # noqa: E731
    need_up, need_down = streak("promote_streak") or 1, streak("demote_streak") or 1
    stairs_up, stairs_down = streak("stairs_promote_streak") or need_up, streak("stairs_demote_streak") or need_down
    if min(need_up, need_down, stairs_up, stairs_down) < 1:
        raise ValueError("promote_streak / demote_streak（含楼梯）须 ≥1")
    if kinds is not None and float(cfg.get("freeze_non_stairs", 0.0)) > 0.5:
        keep = kinds == KIND_STAIRS
        passed, failed = passed & keep, failed & keep
    if need_up == need_down == stairs_up == stairs_down == 1:
        return passed, failed
    if getattr(env, "obstacle_pass_streak", None) is None:
        env.obstacle_pass_streak = torch.zeros_like(env.scene.terrain.terrain_levels)
        env.obstacle_fail_streak = torch.zeros_like(env.scene.terrain.terrain_levels)
    is_stairs = (kinds == KIND_STAIRS) if kinds is not None else torch.zeros_like(passed)
    up = torch.where(passed, env.obstacle_pass_streak[ids] + 1, torch.zeros_like(env.obstacle_pass_streak[ids]))
    down = torch.where(failed, env.obstacle_fail_streak[ids] + 1, torch.zeros_like(env.obstacle_fail_streak[ids]))
    need_up_t = torch.where(is_stairs, torch.full_like(up, stairs_up), torch.full_like(up, need_up))
    need_down_t = torch.where(is_stairs, torch.full_like(down, stairs_down), torch.full_like(down, need_down))
    promote, demote = up >= need_up_t, down >= need_down_t
    env.obstacle_pass_streak[ids] = torch.where(promote, torch.zeros_like(up), up)
    env.obstacle_fail_streak[ids] = torch.where(demote, torch.zeros_like(down), down)
    return promote, demote


def apply_level_floor(levels, kinds, cfg, max_level: int) -> torch.Tensor:
    """楼梯课程等级不低于 stairs_min_level（≤ max_level−1）；其它类型不变。"""
    floor = min(int(round(float(cfg.get("stairs_min_level", 0.0)))), max_level - 1)
    if floor <= 0:
        return levels
    return torch.where(kinds == KIND_STAIRS, levels.clamp_min(floor), levels)


def next_levels(levels, passed, failed, max_level: int) -> torch.Tensor:
    raw = levels + passed.long() - failed.long()
    return torch.where(raw >= max_level, torch.randint_like(raw, max_level), raw.clamp_min(0))


def _env_kinds(env) -> torch.Tensor:
    terrain = env.scene.terrain
    return torch.as_tensor(env.cfg.column_kinds, device=terrain.terrain_levels.device)[terrain.terrain_types]


def curriculum_levels(env) -> torch.Tensor:
    """课程等级（不含探索回合的随机地形等级）；没有探索时就是 terrain_levels。存档与日志都用它。"""
    base = getattr(env, "obstacle_base_levels", None)
    return env.scene.terrain.terrain_levels if base is None else base


def obstacle_curriculum(env, env_ids):
    """Isaac 课程项：按地形类型晋级/降级，返回每类平均等级与等效高度（cm）。

    stairs_explore_frac > 0 时，楼梯环境每次 reset 按该比例抽为探索回合（地形等级均匀随机），探索回合的结果
    不改课程等级，只记到 stairs_explore_* 日志里（≥ EXPLORE_HIGH_LEVEL 另计，对应验收的 12–17.5 cm）。"""
    terrain = env.scene.terrain
    device = terrain.terrain_levels.device
    ids = (torch.arange(env.num_envs, device=device)[env_ids] if isinstance(env_ids, slice)
           else torch.as_tensor(env_ids, device=device, dtype=torch.long))
    cfg = {**CURRICULUM_DEFAULTS, **getattr(env.cfg, "obstacle_curriculum_cfg", {})}
    kinds = _env_kinds(env)
    explore_frac = float(cfg["stairs_explore_frac"])
    if not 0.0 <= explore_frac < 1.0:
        raise ValueError("stairs_explore_frac 须在 [0, 1)")
    if getattr(env, "obstacle_base_levels", None) is None:
        env.obstacle_base_levels = terrain.terrain_levels.clone()
        env.obstacle_explore = torch.zeros_like(terrain.terrain_levels, dtype=torch.bool)
    base, explore = env.obstacle_base_levels, env.obstacle_explore
    if ids.numel():
        robot = env.scene["robot"]
        rel = robot.data.root_pos_w[ids] - terrain.env_origins[ids]
        upright = -robot.data.projected_gravity_b[ids, 2] >= cfg["upright_cos"]
        active = env.episode_length_buf[ids] > 0
        tracker = getattr(env, "obstacle_tracker", None)
        stats = tracker.pop(ids) if tracker is not None else None
        passed, failed = curriculum_decision(rel[:, 0], rel[:, 1], upright, active, kinds[ids], stats, cfg)
        passed, failed = _apply_streaks(env, ids, passed, failed, cfg, kinds[ids])
        was_explore = explore[ids].clone()
        ran_level = terrain.terrain_levels[ids].clone()
        # 探索回合的地形等级是随机抽的，不能据此升降课程等级。
        levels = torch.where(was_explore, base[ids], next_levels(base[ids], passed, failed, terrain.max_terrain_level))
        levels = apply_level_floor(levels, kinds[ids], cfg, terrain.max_terrain_level)
        base[ids] = levels
        draw = (kinds[ids] == KIND_STAIRS) & (torch.rand(ids.numel(), device=device) < explore_frac)
        explore[ids] = draw
        episode_levels = torch.where(draw, torch.randint_like(levels, terrain.max_terrain_level), levels)
        terrain.terrain_levels[ids] = episode_levels
        terrain.env_origins[ids] = terrain.terrain_origins[episode_levels, terrain.terrain_types[ids]]
        log = getattr(env, "obstacle_curriculum_log", None)
        if log is not None:
            for kind, name in KIND_NAMES.items():
                mask = active & (kinds[ids] == kind) & ~was_explore
                log[f"{name}_episodes"] = log.get(f"{name}_episodes", 0) + int(mask.sum())
                log[f"{name}_promoted"] = log.get(f"{name}_promoted", 0) + int((passed & mask).sum())
                log[f"{name}_demoted"] = log.get(f"{name}_demoted", 0) + int((failed & mask).sum())
                if stats is not None:
                    log[f"{name}_collided"] = log.get(f"{name}_collided", 0) + int(((stats["collisions"] > 0) & mask).sum())
            probe = active & was_explore
            high = probe & (ran_level >= EXPLORE_HIGH_LEVEL)
            for key, mask in (("explore", probe), ("explore_high", high)):
                log[f"stairs_{key}_episodes"] = log.get(f"stairs_{key}_episodes", 0) + int(mask.sum())
                log[f"stairs_{key}_passed"] = log.get(f"stairs_{key}_passed", 0) + int((passed & mask).sum())
    return curriculum_summary(curriculum_levels(env), kinds, terrain.cfg.terrain_generator.num_rows)


def curriculum_summary(levels, kinds, rows: int = NUM_ROWS) -> dict[str, torch.Tensor]:
    out = {"mean_level": levels.float().mean()}
    for kind, name in KIND_NAMES.items():
        mask = kinds == kind
        if mask.any():
            out[f"{name}_level"] = levels[mask].float().mean()
            out[f"{name}_top_fraction"] = (levels[mask] >= rows - 1).float().mean()
            if kind != KIND_FLAT:
                out[f"{name}_height_cm"] = equivalent_height_cm(levels[mask], kind, rows).mean()
    return out


def goal_and_lane(rel_xy, cfg=CURRICULUM_DEFAULTS, kinds=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(reached_goal, out_of_lane)：rel_xy 为相对块原点的 (x, y)，kinds 为各环境地形类型（楼梯成功车道单独取）。两者互斥。

    goal_lane_termination = 0：到终点只要没出 fail_lane_m 就是成功截断，成功车道只用于课程晋级。"""
    cfg = {**CURRICULUM_DEFAULTS, **cfg}
    at_goal = rel_xy[:, 0] >= cfg["promote_m"]
    lateral = rel_xy[:, 1].abs()
    out_lane = lateral > cfg["fail_lane_m"]
    if float(cfg["goal_lane_termination"]) > 0.0:
        reached = at_goal & (lateral <= success_lane(kinds, cfg, lateral))
        return reached, out_lane | (at_goal & ~reached)
    return at_goal & ~out_lane, out_lane


def _goal_and_lane(env):
    rel = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    env.obstacle_goal_rel = rel  # reset 之前的最终位置，训练循环记到终点横偏用（只读）
    kinds = _env_kinds(env) if hasattr(env.cfg, "column_kinds") else None
    return goal_and_lane(rel, getattr(env.cfg, "obstacle_curriculum_cfg", {}), kinds)


def reached_goal(env):
    """前进到 promote_m 且 |y| ≤ success_lane_m（goal_lane_termination = 0 时为 |y| ≤ fail_lane_m）：成功截断（time_out 语义）。"""
    return _goal_and_lane(env)[0]


def out_of_lane(env):
    """|y| > fail_lane_m，或到达 promote_m 时 |y| > success_lane_m（goal_lane_termination = 1 时）：真终止
    （训练用特权判据，不进入观测）。"""
    return _goal_and_lane(env)[1]


def stall_steps_exceeded(stall_steps, limit_s: float, step_dt: float):
    """连续停滞步数 ≥ limit_s / step_dt；limit_s ≤ 0 时全 False。"""
    if limit_s <= 0:
        return torch.zeros_like(stall_steps, dtype=torch.bool)
    return stall_steps >= max(1, round(limit_s / step_dt))


def stall_timeout(env):
    """命令前进时连续停滞超过 stall_truncate_s 的回合截断（time_out，课程按前进距离判定）。停滞步数由训练循环维护
    （env.obstacle_stall_steps，见 train_terrain）；没有该属性或 stall_truncate_s = 0 时从不触发。"""
    limit_s = float({**CURRICULUM_DEFAULTS, **getattr(env.cfg, "obstacle_curriculum_cfg", {})}["stall_truncate_s"])
    steps = getattr(env, "obstacle_stall_steps", None)
    if steps is None or limit_s <= 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return stall_steps_exceeded(steps, limit_s, env.step_dt)
