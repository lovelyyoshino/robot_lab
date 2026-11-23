"""越障验收的指标、门限与观测维度开关；评测进程与 accept.py 共用，模块级不导入 Isaac/torch。

统计口径（用户确认的验收规格）：
- 每回合先在时间上取统计量（|roll|/|pitch| 的 P95 与最大值、俯仰角速度 RMS、摆臂速度 RMS ……），
  再对全部回合/环境取 P95 与门比较，不用中位数；"最大值 ≤ X" 即"逐回合最大值的跨回合 P95 ≤ X"。
- 前 0.5 s 出生落地不计入姿态/平顺统计；踢面碰撞从第 1 步开始计数。
- 失败回合也计入姿态统计（翻车就是倾斜），另给只看成功回合的诊断值。

@author pony
@date 2026-11-05
@version v1.3.0
@last_modified 2026-11-21
@changelog
    - v1.3.0 (2026-11-21): 软判（只记录，原判与门不变）：评测进程给出逐步 soft_ok（去掉 |y| ≤ 0.15 m 的稳定条件后连续保持
      0.5 s），每回合记 soft_success / soft_time_s / soft_y_m / soft_score / max_abs_y_m。得分按 SOFT_RULE：软成功记 1 分，
      软成功时 |y| 每超 0.15 m 一厘米扣 0.02，最多扣 0.5；|y| > 0.75 m 仍是出车道终止、不会软成功。aggregate_episodes
      另给 soft_successes 与 soft_score_mean。用户 11-21 定：验收以软判为主，偏离扣分、不判失败。
    - v1.2.0 (2026-11-11): 楼梯贴面时长门 0.5→1.0 s（stairs_riser_contact_hold_s，用户决定）：楼梯同轴第二个轮按
      "碰到立面后再翻"上去（立面高于轮半径时只能接触翻越），最快约 0.7 s；单台阶仍为 0.5 s。
    - v1.1.0 (2026-11-06): 单台阶预抬起抬窗口 3–12 cm（PRELIFT_WINDOW），成功 = 至少一个前轮严格预抬并通行，
      不再要求四轮无踢面碰撞（同轴另一个轮可以轻碰后滚上去）；"轻碰快上"：逐次踢面接触的接触前速度与贴面时长
      （接触到越过踢面顶），按回合取最大值，单台阶与楼梯都设门 0.15 m/s / 0.5 s（需求 3）。
    - v1.0.0 (2026-11-05): 初始创建；逐回合统计、P95 聚合、坡度相对俯仰门、四轮踢面碰撞计数、232/233 维开关。
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAIRS_ACCEPTANCE = ROOT / (
    "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/parallelogram_rover/"
    "stairs_acceptance.py"
)

# 留出种子：从不参与训练和存档挑选；挑选用种子（watch_accept 等）与之分开。
HELD_OUT_SEEDS = (9001, 9002, 9003)
SELECTION_SEEDS = (7, 142, 143, 144, 242)
# 232 = 当前越障策略（45 本体 + 187 扫描）；233 = 旧方案 B 基线（末尾追加车道偏移列）。
BASE_OBS_DIM = 232
LANE_OFFSET_OBS_DIM = 233
WARMUP_S = 0.5
WHEEL_LABELS = ("fl", "rl", "fr", "rr")  # 与 vision_preview.WHEEL_NAMES 顺序一致
# 踢面碰撞：轮子沿行进反方向受力 ≥5 N（与 Preview collision_resisting_force_n 一致），且轮底低于该踢面顶 5 mm 以上、
# 轮前沿距立面 ≤1.5 cm（几何上接触只发生在 lead∈[−r,0]，留数值余量）、横向落在台阶车道内。
RISER_CONTACT = {"force_threshold_n": 5.0, "contact_margin_m": 0.015, "height_tolerance_m": 0.005,
                 "lateral_margin_m": 0.05}
# 单台阶预抬：一次只抬一个轮（两前轮不可能同时离地），起抬越晚三轮支撑时间越短；PreliftEvaluation 的构造参数。
PRELIFT_WINDOW = {"lead_window_m": (0.03, 0.12), "target_lead_m": 0.08}
# 接触前速度：踢面接触起点之前这么多步内车体前进速度的最大值。
IMPACT_LOOKBACK_STEPS = 5

GATES = {
    "min_episodes": 64,
    "min_success_rate": 0.70,
    "roll_abs_p95_deg": 5.0,
    "roll_abs_max_deg": 8.0,
    "flat_pitch_abs_p95_deg": 5.0,
    "step_pitch_abs_p95_deg": 8.0,
    "step_pitch_max_height_m": 0.12,
    "flat_pitch_rate_rms_deg_s": 8.0,
    "stairs_pitch_rate_rms_deg_s": 15.0,
    "flat_arm_vel_rms_rad_s": 0.4,
    "obstacle_arm_vel_rms_rad_s": 0.8,
    "flat_yaw_abs_p95_deg": 10.0,
    "flat_lateral_abs_p95_m": 0.15,
    "flat_route_velocity_rmse_m_s": 0.08,
    "riser_impact_speed_m_s": 0.15,
    "riser_contact_hold_s": 0.5,
    # 楼梯：同轴第二个轮碰到立面后再翻（用户 2026-11-11 决定），贴面时长门放宽
    "stairs_riser_contact_hold_s": 1.0,
}
EPISODE_METRICS = ("roll_abs_p95_deg", "roll_abs_max_deg", "pitch_abs_p95_deg", "pitch_abs_max_deg",
                   "pitch_rate_rms_deg_s", "arm_vel_rms_rad_s", "base_vz_rms_m_s",
                   "action_delta_rms", "arm_action_delta_rms", "action_rate_rms_per_s")
FLAT_METRICS = EPISODE_METRICS + ("yaw_abs_p95_deg", "lateral_abs_p95_m", "route_velocity_rmse_m_s")
# 踢面接触（楼梯需求 3"轻碰快上"）：无接触的回合为 None，不参与 P95
CONTACT_METRICS = ("riser_impact_speed_max_m_s", "riser_contact_hold_max_s")
STEP_KEYS = ("roll", "pitch", "pitch_rate", "vz", "arm_vel", "action", "riser_contact", "tilt_deg", "x", "y", "finite")
# 评测进程可选给出的逐步键：soft_ok = 软判稳定已连续保持 hold_time_s（v1.3.0）；旧结果、单测样本没有这一列
OPTIONAL_STEP_KEYS = ("soft_ok",)
# 软判得分：每回合 1 分，软成功时 |y| 超过 lane_m 的部分每米扣 deduct_per_m（每厘米 0.02），最多扣 max_deduct
SOFT_RULE = {"lane_m": 0.15, "deduct_per_m": 2.0, "max_deduct": 0.5}


def soft_fields(steps: dict, dt: float, rule=SOFT_RULE) -> dict:
    """逐步序列 → 软判字段。soft_ok 第一次为真即软成功；没有 soft_ok 列时只给 max_abs_y_m。"""
    ys = [abs(y) for y in steps.get("y", []) if isinstance(y, (int, float)) and math.isfinite(y)]
    out = {"max_abs_y_m": max(ys) if ys else None}
    if "soft_ok" not in steps:
        return out
    hit = next((i for i, ok in enumerate(steps["soft_ok"]) if ok), None)
    if hit is None:
        return {**out, "soft_success": False, "soft_time_s": None, "soft_y_m": None, "soft_score": 0.0}
    y = steps["y"][hit]
    excess = max(abs(y) - rule["lane_m"], 0.0)
    score = 1.0 - min(rule["max_deduct"], rule["deduct_per_m"] * excess)
    return {**out, "soft_success": True, "soft_time_s": (hit + 1) * dt, "soft_y_m": y, "soft_score": score}


def load_stairs_acceptance():
    """按文件路径加载，避免导入 robot_lab.tasks 包（会拉起 Isaac）。"""
    spec = importlib.util.spec_from_file_location("stairs_acceptance", STAIRS_ACCEPTANCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 统计
def _finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def percentile(values, q: float):
    """与 numpy.percentile(method='linear') 一致；空或全非有限返回 None。"""
    data = sorted(_finite(values))
    if not data:
        return None
    if not 0 <= q <= 100:
        raise ValueError(f"分位数必须位于 [0,100]：{q}")
    rank = (len(data) - 1) * q / 100.0
    low = math.floor(rank)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (rank - low)


def p95(values):
    return percentile(values, 95.0)


def rms(values):
    data = _finite(values)
    return math.sqrt(sum(v * v for v in data) / len(data)) if data else None


# ---------------------------------------------------------------- 观测维度与种子
def actor_input_dim(policy_path) -> int:
    """读 actor 第一层输入维数：232 为当前越障策略，233 为带车道偏移的旧基线；其他维数拒绝。"""
    import torch

    path = Path(policy_path)
    try:
        state = torch.jit.load(str(path), map_location="cpu").state_dict()
    except RuntimeError:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        state = checkpoint.get("actor", checkpoint.get("model_state_dict", checkpoint))
    weight = state.get("actor.0.weight")
    if weight is None:
        weight = next((v for v in state.values() if getattr(v, "ndim", 0) == 2), None)
    if weight is None:
        raise ValueError(f"无法从 {path} 读出 actor 第一层权重")
    dim = int(weight.shape[1])
    if dim not in (BASE_OBS_DIM, LANE_OFFSET_OBS_DIM):
        raise ValueError(f"actor 输入 {dim} 维，只支持 {BASE_OBS_DIM}（当前）或 {LANE_OFFSET_OBS_DIM}（旧车道偏移基线）")
    return dim


def training_seeds(policy_path) -> list[int]:
    """run 目录 params/supermimic_terrain.json 记录的训练种子；找不到返回空表。"""
    path = Path(policy_path).resolve()
    for run in (path.parent.parent, path.parent):
        params = run / "params" / "supermimic_terrain.json"
        if params.is_file():
            seed = json.loads(params.read_text()).get("arguments", {}).get("seed")
            return [int(seed)] if seed is not None else []
    return []


# ---------------------------------------------------------------- 逐步采样（Isaac 进程内，torch）
def quat_to_roll_pitch(quat):
    """wxyz 四元数 → (roll, pitch) 弧度，ZYX 欧拉角；pitch 正为低头。"""
    import torch

    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def single_step_risers(front_x_m, height_m, lane_y):
    """单台阶：一个立面（相对 env origin），台阶只占 lane_y 横向区间。"""
    return {"x": [float(front_x_m)], "top_z": [float(height_m)], "lane_y": [float(lane_y[0]), float(lane_y[1])]}


def stairs_risers(first_edge_m, tread_m, riser_m, num_steps, lane_width_m):
    """连续上楼：第 k 级立面在 first_edge + k·tread，顶高 (k+1)·riser。"""
    return {"x": [first_edge_m + k * tread_m for k in range(num_steps)],
            "top_z": [(k + 1) * riser_m for k in range(num_steps)],
            "lane_y": [-lane_width_m / 2, lane_width_m / 2]}


def riser_contact_mask(wheel_centers, wheel_radii, resisting_n, risers, *, force_threshold_n=5.0,
                       contact_margin_m=0.015, height_tolerance_m=0.005, lateral_margin_m=0.05):
    """[N,W] 布尔：轮子正顶在前方踢面上（水平阻力且轮底低于踢面顶）。

    wheel_centers 为相对 env origin 的 [N,W,3]；只看轮心前方最近的立面，越过立面的轮子在台阶顶上，不再算碰撞。
    """
    import torch

    n, w = resisting_n.shape
    if risers is None or not risers["x"]:
        return torch.zeros(n, w, dtype=torch.bool, device=resisting_n.device)
    x = torch.as_tensor(risers["x"], dtype=wheel_centers.dtype, device=wheel_centers.device)
    top = torch.as_tensor(risers["top_z"], dtype=wheel_centers.dtype, device=wheel_centers.device)
    radii = torch.as_tensor(wheel_radii, dtype=wheel_centers.dtype, device=wheel_centers.device)
    cx, cy, cz = wheel_centers.unbind(-1)
    index = (x < cx[..., None]).sum(-1)  # 立面升序：已越过的个数 = 前方最近立面下标
    ahead = index < x.numel()
    index = index.clamp(max=x.numel() - 1)
    lead = x[index] - cx - radii
    below = cz - radii < top[index] - height_tolerance_m
    lateral = (cy >= risers["lane_y"][0] - lateral_margin_m) & (cy <= risers["lane_y"][1] + lateral_margin_m)
    pushing = resisting_n >= force_threshold_n
    finite = torch.isfinite(wheel_centers).all(-1) & torch.isfinite(resisting_n)
    return ahead & (lead <= contact_margin_m) & below & lateral & pushing & finite


def capture_acceptance_sample(env, preview, finite) -> dict:
    """自动 reset 前的逐步验收采样；preview 为 vision_preview.collect_preview 的同步结果。"""
    import torch

    spec = env.cfg.acceptance_metrics
    robot = env.scene["robot"]
    data = robot.data
    if not hasattr(env, "_acceptance_arm_ids"):
        ids, names = robot.find_joints(list(spec["arm_joint_names"]), preserve_order=True)
        if list(names) != list(spec["arm_joint_names"]):
            raise RuntimeError(f"摆臂关节不匹配：{names}")
        env._acceptance_arm_ids = ids
    origins = env.scene.env_origins
    roll, pitch = quat_to_roll_pitch(data.root_quat_w)
    relative = data.root_pos_w - origins
    contact = riser_contact_mask(preview["wheel_centers_w"] - origins[:, None, :], preview["wheel_radii"],
                                 preview["wheel_resisting_force_n"], spec["risers"], **RISER_CONTACT)
    tilt = torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
    columns = {"roll": roll, "pitch": pitch, "pitch_rate": data.root_ang_vel_b[:, 1], "vz": data.root_lin_vel_w[:, 2],
               "arm_vel": data.joint_vel[:, env._acceptance_arm_ids], "action": env.action_manager.action,
               "riser_contact": contact, "tilt_deg": tilt, "x": relative[:, 0], "y": relative[:, 1], "finite": finite}
    return {key: value.detach().cpu().tolist() for key, value in columns.items()}


# ---------------------------------------------------------------- 逐回合统计
def episode_statistics(steps: dict, dt: float, warmup_steps: int) -> dict:
    """steps 为本回合逐步序列（键见 STEP_KEYS）；姿态/平顺跳过出生 warmup，碰撞全程计数。"""
    n = len(steps["roll"])
    start = warmup_steps if n > warmup_steps else 0
    ok = [bool(f) for f in steps["finite"]]
    window = [t for t in range(start, n) if ok[t]]
    deg = math.degrees
    roll = [abs(deg(steps["roll"][t])) for t in window]
    pitch = [abs(deg(steps["pitch"][t])) for t in window]
    deltas, arm_deltas = [], []
    for t in window:
        if t - 1 >= start and ok[t - 1]:
            diff = [a - b for a, b in zip(steps["action"][t], steps["action"][t - 1])]
            deltas += diff
            arm_deltas += diff[:4]
    action_delta = rms(deltas)
    pitch_rate = rms([steps["pitch_rate"][t] for t in window])
    contacts = [[bool(row[k]) and ok[t] for t, row in enumerate(steps["riser_contact"])] for k in range(len(WHEEL_LABELS))]
    impact, hold = contact_profile(contacts, steps["x"], ok, dt)
    return {
        "steps": n, "metric_steps": len(window),
        "roll_abs_p95_deg": p95(roll), "roll_abs_max_deg": max(roll) if roll else None,
        "pitch_abs_p95_deg": p95(pitch), "pitch_abs_max_deg": max(pitch) if pitch else None,
        "pitch_rate_rms_deg_s": deg(pitch_rate) if pitch_rate is not None else None,
        "arm_vel_rms_rad_s": rms([v for t in window for v in steps["arm_vel"][t]]),
        "base_vz_rms_m_s": rms([steps["vz"][t] for t in window]),
        "action_delta_rms": action_delta,
        "arm_action_delta_rms": rms(arm_deltas),
        "action_rate_rms_per_s": action_delta / dt if action_delta is not None else None,
        "riser_contact_events": [sum(1 for t in range(n) if c[t] and (t == 0 or not c[t - 1])) for c in contacts],
        "riser_contact_steps": [sum(c) for c in contacts],
        "riser_impact_speed_max_m_s": max(impact) if impact else None,
        "riser_contact_hold_max_s": max(hold) if hold else None,
    }


def contact_profile(contacts, x, ok, dt):
    """逐次踢面接触：(接触前 IMPACT_LOOKBACK_STEPS 步内的最大前进速度, 连续贴面时长)，四轮所有接触段汇总。"""
    impact, hold = [], []
    for c in contacts:
        t = 0
        while t < len(c):
            if not c[t]:
                t += 1
                continue
            start = t
            while t < len(c) and c[t]:
                t += 1
            hold.append((t - start) * dt)
            speeds = [(x[k] - x[k - 1]) / dt for k in range(max(1, start - IMPACT_LOOKBACK_STEPS), start + 1)
                      if ok[k] and ok[k - 1]]
            if speeds:
                impact.append(max(speeds))
    return impact, hold


def classify_outcome(success, terminated, last, *, climb_tilt_deg, target_distance_m,
                     lane_half_width_m=0.75, max_progress_m=3.75) -> str:
    """按 capture_traversal 的终止条件归类失败；超时分"没到终点"与"到了但没稳住"。"""
    if success:
        return "success"
    if not last["finite"]:
        return "nonfinite"
    if terminated:
        if last["tilt_deg"] > climb_tilt_deg - 1e-6:
            return "rollover"
        if abs(last["y"]) > lane_half_width_m:
            return "out_of_lane"
        if last["x"] > max_progress_m:
            return "overshoot"
        return "terminated_other"
    return "stall_timeout" if last["x"] < target_distance_m else "unstable_at_target"


class EpisodeMetrics:
    """在 SkillEvaluation.update 之后调用；与 PreliftEvaluation 相同的配额与 reset 隔离规则。"""

    def __init__(self, num_envs, dt, *, climb_tilt_deg=31.8, target_distance_m=2.5, warmup_s=WARMUP_S):
        if num_envs < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("环境数和控制周期必须为正")
        self.dt = dt
        self.warmup_steps = math.ceil(warmup_s / dt - 1e-9)
        self.climb_tilt_deg = climb_tilt_deg
        self.target_distance_m = target_distance_m
        self.buffers = [self._empty() for _ in range(num_envs)]
        self.records = []
        self._seen = 0

    @staticmethod
    def _empty():
        return {key: [] for key in STEP_KEYS}

    def update(self, sample, traversal):
        endings = {r["env_id"]: r for r in traversal.records[self._seen:]}
        self._seen = len(traversal.records)
        for i, buffer in enumerate(self.buffers):
            ending = endings.get(i)
            if traversal.counts[i] >= traversal.quota and ending is None:
                continue
            for key in STEP_KEYS:
                buffer[key].append(sample[key][i])
            for key in OPTIONAL_STEP_KEYS:
                if key in sample:
                    buffer.setdefault(key, []).append(sample[key][i])
            if ending is None:
                continue
            last = {key: buffer[key][-1] for key in ("tilt_deg", "x", "y", "finite")}
            self.records.append({
                "env_id": i, "episode_index": ending["episode_index"], "success": ending["success"],
                "terminated": ending["terminated"], "truncated": ending["truncated"],
                "duration_s": ending["duration_s"],
                "outcome": classify_outcome(ending["success"], ending["terminated"], last,
                                            climb_tilt_deg=self.climb_tilt_deg,
                                            target_distance_m=self.target_distance_m),
                "terminal_x_m": last["x"], "terminal_y_m": last["y"], "terminal_tilt_deg": last["tilt_deg"],
                **soft_fields(buffer, self.dt),
                **episode_statistics(buffer, self.dt, self.warmup_steps),
            })
            self.buffers[i] = self._empty()

    def summary(self) -> dict:
        return {"protocol": "supermimic_acceptance_episode_metrics", "warmup_s": self.warmup_steps * self.dt,
                "climb_tilt_deg": self.climb_tilt_deg, "wheel_labels": list(WHEEL_LABELS),
                "riser_contact": RISER_CONTACT, "episodes": self.records}


# ---------------------------------------------------------------- 跨回合聚合与门
def aggregate_episodes(episodes, metrics=EPISODE_METRICS + CONTACT_METRICS) -> dict:
    """逐回合统计 → 跨回合 P95（门用）、最差值与只看成功回合的 P95（诊断）。"""
    succeeded = [e for e in episodes if e.get("success")]
    result = {"episodes": len(episodes), "successes": len(succeeded),
              "p95": {k: p95([e.get(k) for e in episodes]) for k in metrics},
              "worst": {k: max(_finite([e.get(k) for e in episodes]), default=None) for k in metrics},
              "p95_success_only": {k: p95([e.get(k) for e in succeeded]) for k in metrics},
              "outcomes": dict(Counter(e.get("outcome", "unknown") for e in episodes))}
    if episodes and all("soft_success" in e for e in episodes):
        result["soft_successes"] = sum(bool(e["soft_success"]) for e in episodes)
        result["soft_score_mean"] = sum(e["soft_score"] for e in episodes) / len(episodes)
    if episodes and "riser_contact_events" in episodes[0]:
        result["riser_collision_episode_rate"] = {
            label: sum(e["riser_contact_events"][k] > 0 for e in episodes) / len(episodes)
            for k, label in enumerate(WHEEL_LABELS)}
        result["riser_collision_events_mean"] = {
            label: sum(e["riser_contact_events"][k] for e in episodes) / len(episodes)
            for k, label in enumerate(WHEEL_LABELS)}
    return result


def check(requirement, case, metric, value, threshold, *, op="<=", gating=True, note=None) -> dict:
    """单项门；缺值的门判失败（fail-closed），非门项 passed 为 None。"""
    passed = None
    if gating:
        ok = value is not None and math.isfinite(value)
        passed = ok and (value <= threshold if op == "<=" else value >= threshold)
    row = {"requirement": requirement, "case": case, "metric": metric, "value": value,
           "op": op, "threshold": threshold, "gating": gating, "passed": passed}
    if note:
        row["note"] = note
    return row


def tilt_smooth_checks(kind, p95_metrics, *, case, step_height_m=None, pitch_limits=None) -> list[dict]:
    """需求 1（不倾斜）与 2（平顺）。kind ∈ flat/step/stairs；stairs 需 stairs_pitch_gates 的结果。"""
    if kind not in ("flat", "step", "stairs"):
        raise ValueError(f"未知地形类别：{kind}")
    g = GATES
    v = p95_metrics.get
    rows = [check(1, case, "roll_abs_p95_deg", v("roll_abs_p95_deg"), g["roll_abs_p95_deg"]),
            check(1, case, "roll_abs_max_deg", v("roll_abs_max_deg"), g["roll_abs_max_deg"])]
    if kind == "flat":
        rows += [check(1, case, "pitch_abs_p95_deg", v("pitch_abs_p95_deg"), g["flat_pitch_abs_p95_deg"]),
                 check(1, case, "pitch_abs_max_deg", v("pitch_abs_max_deg"), None, gating=False),
                 check(1, case, "pitch_rate_rms_deg_s", v("pitch_rate_rms_deg_s"), g["flat_pitch_rate_rms_deg_s"])]
    elif kind == "step":
        if step_height_m is None:
            raise ValueError("单台阶需要 step_height_m")
        low = step_height_m <= g["step_pitch_max_height_m"] + 1e-9
        rows += [check(1, case, "pitch_abs_p95_deg", v("pitch_abs_p95_deg"), g["step_pitch_abs_p95_deg"] if low else None,
                       gating=low, note=None if low else "规格只对 ≤12 cm 单台阶设俯仰门"),
                 check(1, case, "pitch_abs_max_deg", v("pitch_abs_max_deg"), None, gating=False),
                 check(1, case, "pitch_rate_rms_deg_s", v("pitch_rate_rms_deg_s"), None, gating=False,
                       note="规格只对平地/楼梯设俯仰角速度门")]
    else:
        if pitch_limits is None:
            raise ValueError("楼梯需要坡度相对俯仰门")
        rows += [check(1, case, "pitch_abs_p95_deg", v("pitch_abs_p95_deg"), pitch_limits["pitch_abs_p95_deg"],
                       note=f"坡度 {pitch_limits['slope_deg']:.1f}° − 5°"),
                 check(1, case, "pitch_abs_max_deg", v("pitch_abs_max_deg"), pitch_limits["pitch_abs_max_deg"],
                       note=f"坡度 {pitch_limits['slope_deg']:.1f}° + 3°"),
                 check(1, case, "pitch_rate_rms_deg_s", v("pitch_rate_rms_deg_s"), g["stairs_pitch_rate_rms_deg_s"])]
    arm = g["flat_arm_vel_rms_rad_s"] if kind == "flat" else g["obstacle_arm_vel_rms_rad_s"]
    rows += [check(2, case, "arm_vel_rms_rad_s", v("arm_vel_rms_rad_s"), arm),
             check(2, case, "base_vz_rms_m_s", v("base_vz_rms_m_s"), None, gating=False),
             check(2, case, "action_rate_rms_per_s", v("action_rate_rms_per_s"), None, gating=False),
             check(2, case, "arm_action_delta_rms", v("arm_action_delta_rms"), None, gating=False)]
    return rows


def contact_checks(p95_metrics, *, case, hold_gate=None) -> list[dict]:
    """需求 3 轻碰快上（单台阶的同轴滚上轮与楼梯）：接触前速度与贴面时长的跨回合 P95；整档没有任何接触时按 0 计（没碰就是轻碰）。

    hold_gate：贴面时长门（默认 riser_contact_hold_s；楼梯传 stairs_riser_contact_hold_s）。"""
    v = {k: (0.0 if p95_metrics.get(k) is None else p95_metrics[k]) for k in CONTACT_METRICS}
    return [check(3, case, "riser_impact_speed_max_m_s", v["riser_impact_speed_max_m_s"], GATES["riser_impact_speed_m_s"]),
            check(3, case, "riser_contact_hold_max_s", v["riser_contact_hold_max_s"], GATES["riser_contact_hold_s"] if hold_gate is None else hold_gate)]


def success_checks(requirement, case, *, successes, episodes, complete) -> list[dict]:
    """需求 3/4：完整配额、≥64 回合、成功率 ≥70%。"""
    rate = successes / episodes if episodes else 0.0
    return [check(requirement, case, "success_rate", rate, GATES["min_success_rate"], op=">="),
            check(requirement, case, "episodes", float(episodes), float(GATES["min_episodes"]), op=">="),
            check(requirement, case, "complete", 1.0 if complete else 0.0, 1.0, op=">=")]


def prelift_acceptance_records(prelift_records, metrics_records) -> list[dict]:
    """成功 = 至少一个前轮严格预抬（时序 + 通行）；同轴另一个轮可以接触后滚上去，接触的轻重快慢由
    contact_checks 在档位层面设门。四轮无碰撞只作诊断。缺逐回合指标记录判失败（fail-closed）。"""
    by_key = {(r["env_id"], r["episode_index"]): r for r in metrics_records}
    rows = []
    for record in prelift_records:
        metrics = by_key.get((record["env_id"], record["episode_index"]))
        events = metrics["riser_contact_events"] if metrics is not None else None
        collision_free = events is not None and not any(events)
        rows.append({"env_id": record["env_id"], "episode_index": record["episode_index"],
                     "strict_prelift_success": bool(record["success"]),
                     "riser_collision_free": collision_free,
                     "riser_contact_events": events,
                     "success": bool(record["success"]) and metrics is not None})
    return rows


def pool_flat_rows(rows) -> dict:
    """probe_flat 各（种子, 速度）行的逐环境数组按速度合并，跨环境取 P95。"""
    pooled = {}
    for row in rows:
        entry = pooled.setdefault(f"{row['command']:.2f}", {"command": row["command"], "seeds": [], "per_env": {}})
        entry["seeds"].append(row["seed"])
        for key in FLAT_METRICS:
            entry["per_env"].setdefault(key, []).extend(row["per_env"][key])
    for entry in pooled.values():
        entry["envs"] = len(entry["per_env"]["roll_abs_p95_deg"])
        entry["p95"] = {k: p95(v) for k, v in entry["per_env"].items()}
    return pooled


def flat_checks(pooled) -> list[dict]:
    """平地：需求 1/2 + 需求 5（偏航、横向、速度跟踪，门值沿用旧验收）。"""
    rows = []
    for entry in pooled.values():
        case = f"flat_v{entry['command']:.2f}"
        rows += tilt_smooth_checks("flat", entry["p95"], case=case)
        rows += [check(5, case, "yaw_abs_p95_deg", entry["p95"]["yaw_abs_p95_deg"], GATES["flat_yaw_abs_p95_deg"]),
                 check(5, case, "lateral_abs_p95_m", entry["p95"]["lateral_abs_p95_m"], GATES["flat_lateral_abs_p95_m"]),
                 check(5, case, "route_velocity_rmse_m_s", entry["p95"]["route_velocity_rmse_m_s"],
                       GATES["flat_route_velocity_rmse_m_s"]),
                 check(5, case, "envs", float(entry["envs"]), float(GATES["min_episodes"]), op=">=")]
    return rows
