"""单障碍事件验收：目标轮必须在首次到达前缘前完成避障，且最终稳定通行。

@author pony
@date 2026-09-30
@version v0.3.0
@last_modified 2026-11-06
@changelog
    - v0.3.0 (2026-11-06): 起抬窗口改为构造参数 lead_window_m / target_lead_m（默认仍为 8–14 cm），
      SuperMimic 越障验收传入 3–12 cm。
    - (2026-10-14): 实际全宽验收允许一侧完整主动动作，保留双侧独立诊断及历史默认门槛。
    - (2026-10-13): 独立建立稳定接近基线，排除出生冲击并输出冻结及采样门。
    - (2026-10-11): 仅在65cm障碍接近范围内记录首次离地，避免出生颠簸误消耗整回合资格。
    - (2026-10-11): 独立记录5mm真实离地与高度达标位置，拒绝早离地后进窗补领。
    - (2026-10-10): 独立验收约10cm起抬（8–14cm），保留完整保持与落轮门槛。
    - (2026-10-10): 增加独立持续保持/落轮验收，瞬时事件不再冒充完整避障动作。
    - (2026-10-09): 对齐窄障碍接地承重基线，输出基线采样诊断并保留严格抬轮独立判据。
    - (2026-10-09): 窄障碍允许目标轮提前卸载并保留对角支撑，不强求2cm净空。
    - v0.2.0 (2026-10-08): 输出左右有效事件率，几何势能不再标为实际训练奖励。
    - v0.1.0 (2026-09-30): 双前轮独立计时、自动reset隔离及完整配额验收。
"""

from __future__ import annotations

import math
import statistics

PRELIFT_MINIMUM_SUCCESS_RATE = 0.70
# 视觉可以提前发现障碍，但有效动作必须落在目标前缘约10cm处。
PRELIFT_ACTION_MIN_ADVANCE_M = 0.08
PRELIFT_ACTION_MAX_ADVANCE_M = 0.14
PRELIFT_ACTION_TARGET_ADVANCE_M = 0.10
PRELIFT_TAKEOFF_CLEARANCE_M = 0.005
PRELIFT_PERCEPTION_MAX_ADVANCE_M = 0.65


def wheel_lane_pixel_counts(points, candidate_mask, wheel_center_y_m, lane_half_width_m):
    """Count valid obstacle pixels inside each front wheel's lateral corridor."""
    if points.ndim != 2 or points.shape[1] != 3 or candidate_mask.shape != points.shape[:1]:
        raise ValueError("points和candidate_mask必须分别为[N,3]及[N]")
    if len(wheel_center_y_m) != 2 or not math.isfinite(lane_half_width_m) or lane_half_width_m <= 0:
        raise ValueError("必须提供两个有限轮迹和正的有限轮迹半宽")
    counts = []
    for center_y in wheel_center_y_m:
        if not math.isfinite(center_y):
            raise ValueError("轮迹横向位置必须有限")
        in_lane = (points[:, 1] - center_y).abs() <= lane_half_width_m
        counts.append(int((candidate_mask & in_lane).sum().item()))
    return counts


class PreliftEvaluation:
    """固定单障碍协议；不是多障碍检测器，也不证明视觉因果性。"""

    def __init__(self, num_envs, dt, *, required_front_wheels=(0, 1), allow_load_drop=False,
                 load_drop_fraction=0.35, min_force_drop_n=3.0, min_clearance_m=0.005, unload_hold_s=0.06,
                 require_hold_to_edge=False, prelift_hold_s=0.06, release_hold_s=0.06,
                 baseline_settle_s=0.40, baseline_stable_s=0.12,
                 preparation_max_advance_m=0.25, required_active_count=None,
                 lead_window_m=(PRELIFT_ACTION_MIN_ADVANCE_M, PRELIFT_ACTION_MAX_ADVANCE_M),
                 target_lead_m=PRELIFT_ACTION_TARGET_ADVANCE_M):
        if not required_front_wheels or len(set(required_front_wheels)) != len(required_front_wheels) or any(
            wheel not in (0, 1) for wheel in required_front_wheels
        ):
            raise ValueError("required_front_wheels 必须是非空且不重复的前轮索引")
        if not math.isfinite(dt) or dt <= 0 or not 0 < load_drop_fraction < 1:
            raise ValueError("控制周期和卸载比例无效")
        if any(not math.isfinite(x) or x <= 0 for x in (
            min_force_drop_n, min_clearance_m, unload_hold_s, prelift_hold_s, release_hold_s,
            baseline_settle_s, baseline_stable_s, preparation_max_advance_m,
        )):
            raise ValueError("窄障碍净空、减载力和保持时长必须为有限正数")
        self.min_lead_m, self.max_lead_m = (float(v) for v in lead_window_m)
        self.target_lead_m = float(target_lead_m)
        if not 0 < self.min_lead_m <= self.target_lead_m <= self.max_lead_m:
            raise ValueError("起抬窗口必须满足 0 < min ≤ target ≤ max")
        if not (
            self.max_lead_m <= preparation_max_advance_m < PRELIFT_PERCEPTION_MAX_ADVANCE_M
        ):
            raise ValueError("稳定基线容差或准备距离无效")
        self.dt = dt
        self.required_front_wheels = tuple(required_front_wheels)
        self.required_active_count = len(required_front_wheels) if required_active_count is None else required_active_count
        if (not isinstance(self.required_active_count, int)
                or not 1 <= self.required_active_count <= len(required_front_wheels)):
            raise ValueError("required_active_count必须在目标前轮数量范围内")
        self.allow_load_drop = bool(allow_load_drop)
        self.load_drop_fraction = load_drop_fraction
        self.min_force_drop_n = min_force_drop_n
        self.min_clearance_m = min_clearance_m
        self.unload_hold_steps = math.ceil(unload_hold_s / dt - 1e-9)
        self.baseline_settle_s = baseline_settle_s
        self.baseline_steps = math.ceil(baseline_stable_s / dt - 1e-9)
        self.preparation_max_advance_m = preparation_max_advance_m
        # 历史事件分析调用可保留原语义；实际单障碍CLI必须显式启用完整时序门。
        self.require_hold_to_edge = bool(require_hold_to_edge)
        self.prelift_hold_steps = math.ceil(prelift_hold_s / dt - 1e-9)
        self.release_hold_steps = math.ceil(release_hold_s / dt - 1e-9)
        self.pending = [self._empty() for _ in range(num_envs)]
        self.records = []

    @staticmethod
    def _empty():
        return {
            "steps": 0,
            "contacted": [False, False],
            "first_contact_lead_m": [None, None],
            "low_seen": [False, False],
            "low_seen_once": [False, False],
            "lifted": [None, None],
            "takeoff": [None, None],
            "takeoff_valid": [False, False],
            "baseline_force_n": [None, None],
            "baseline_samples": [0, 0],
            "baseline_locked": [False, False],
            "baseline_history": [[], []],
            "unload_steps": [0, 0],
            "strict_lifted": [None, None],
            "collision_seen": [False, False],
            "hold_steps": [0, 0],
            "hold_broken": [False, False],
            "held_to_edge": [False, False],
            "release_steps": [0, 0],
            "released": [False, False],
            "diagnostics": {
                "preview_steps": [0, 0],
                "reward_eligible_preview_steps": [0, 0],
                "evaluation_eligible_preview_steps": [0, 0],
                "lead_window_steps": [0, 0],
                "eligible_lead_window_steps": [0, 0],
                "max_lift_m_in_window": [None, None],
                "max_clearance_m_in_window": [None, None],
                "max_reward_clearance_m_in_window": [None, None],
                "max_load_drop_fraction_in_window": [0.0, 0.0],
                "unloaded_clearance_steps_in_window": [0, 0],
                "early_lift_steps": [0, 0],
                "first_early_lift_lead_m": [None, None],
                "joint_threshold_steps_in_window": [0, 0],
                "eligible_joint_threshold_steps_in_window": [0, 0],
                "ineligible_threshold_reasons": {},
                "geometric_potential_sum": 0.0,
                "geometric_potential_max": 0.0,
            },
        }

    def update(self, samples, traversal):
        """在 SkillEvaluation.update 之后调用；采样本身必须来自自动reset之前。"""
        new_records = {(r["env_id"], r["episode_index"]): r for r in traversal.records[len(self.records):]}
        for i, sample in enumerate(samples):
            ending = next((r for (env_id, _), r in new_records.items() if env_id == i), None)
            if traversal.counts[i] >= traversal.quota and ending is None:
                continue
            state = self.pending[i]
            state["steps"] += 1
            diagnostics = state["diagnostics"]
            preview = sample.get("preview", [False, False])
            reward_eligible = sample.get("reward_eligible", False)
            reasons = diagnostics["ineligible_threshold_reasons"]
            for name in sample.get("eligibility_gates", {}):
                reasons.setdefault(name, [0, 0])
            diagnostics["geometric_potential_sum"] += sample.get("preview_potential_raw", 0.0)
            diagnostics["geometric_potential_max"] = max(
                diagnostics["geometric_potential_max"], sample.get("preview_potential_raw", 0.0)
            )
            for wheel in range(2):
                lead = sample["lead_m"][wheel]
                valid = all(math.isfinite(sample[key][wheel]) for key in ("lead_m", "lift_m", "clearance_m"))
                if preview[wheel]:
                    diagnostics["preview_steps"][wheel] += 1
                    if reward_eligible:
                        diagnostics["reward_eligible_preview_steps"][wheel] += 1
                    if sample["eligible"]:
                        diagnostics["evaluation_eligible_preview_steps"][wheel] += 1
                # 一旦抵达前缘，之后后退再抬轮也不能冒充首次前瞻动作。
                if not valid or lead <= 0:
                    if valid and lead <= 0 and state["first_contact_lead_m"][wheel] is None:
                        state["first_contact_lead_m"][wheel] = lead
                    state["contacted"][wheel] = True
                collision = sample.get("blocking_collision", [False, False])[wheel]
                state["collision_seen"][wheel] |= bool(collision)
                state["contacted"][wheel] |= bool(collision)
                in_window = (valid and not state["contacted"][wheel]
                             and self.min_lead_m <= lead <= self.max_lead_m)
                if in_window:
                    diagnostics["lead_window_steps"][wheel] += 1
                    diagnostics["max_lift_m_in_window"][wheel] = max(
                        diagnostics["max_lift_m_in_window"][wheel] or float("-inf"), sample["lift_m"][wheel]
                    )
                    diagnostics["max_clearance_m_in_window"][wheel] = max(
                        diagnostics["max_clearance_m_in_window"][wheel] or float("-inf"),
                        sample["clearance_m"][wheel],
                    )
                    reward_clearance = sample.get("reward_clearance_m", sample["clearance_m"])[wheel]
                    if math.isfinite(reward_clearance):
                        current = diagnostics["max_reward_clearance_m_in_window"][wheel]
                        diagnostics["max_reward_clearance_m_in_window"][wheel] = max(
                            current if current is not None else float("-inf"), reward_clearance
                        )
                    if sample["eligible"]:
                        diagnostics["eligible_lead_window_steps"][wheel] += 1
                    if sample["lift_m"][wheel] >= 0.02 and sample["clearance_m"][wheel] >= 0.02:
                        diagnostics["joint_threshold_steps_in_window"][wheel] += 1
                        if sample["eligible"]:
                            diagnostics["eligible_joint_threshold_steps_in_window"][wheel] += 1
                        else:
                            for name, passed in sample.get("eligibility_gates", {}).items():
                                if not passed:
                                    reasons[name][wheel] += 1
                force = sample.get("wheel_support_force_n", [None, None])[wheel]
                force_valid = isinstance(force, (int, float)) and math.isfinite(force) and force >= 0
                # 与训练一致：相对车身抬高不等于离地，窄障碍低位只看实际接地承重。
                narrow_grounded = (
                    valid and force_valid and force > 1.0
                    and sample["clearance_m"][wheel] < self.min_clearance_m
                    and not state["contacted"][wheel]
                )
                # 独立于训练缓存：先排除出生冲击，再在准备区外连续稳定承重。
                # 进入25cm后锁存；缺少合格基线则不能在卸载过程临时补采。
                if valid and lead <= self.preparation_max_advance_m:
                    state["baseline_locked"][wheel] = True
                if (not state["baseline_locked"][wheel] and narrow_grounded
                        and force > self.min_force_drop_n + 1.0
                        and self.preparation_max_advance_m < lead <= PRELIFT_PERCEPTION_MAX_ADVANCE_M
                        and state["steps"] * self.dt >= self.baseline_settle_s
                        and state["takeoff"][wheel] is None
                        and sample.get("reward_eligible", sample["eligible"])):
                    history = state["baseline_history"][wheel]
                    history.append(force)
                    del history[:-self.baseline_steps]
                    state["baseline_samples"][wheel] = len(history)
                    if len(history) >= self.baseline_steps:
                        state["baseline_force_n"][wheel] = statistics.median(history)
                elif not state["baseline_locked"][wheel]:
                    state["baseline_samples"][wheel] = 0
                    state["baseline_history"][wheel].clear()
                baseline = state["baseline_force_n"][wheel]
                load_drop = (
                    max(0.0, min(1.0, (baseline - force) / max(baseline, 1.0)))
                    if baseline is not None and force_valid else 0.0
                )
                if in_window:
                    diagnostics["max_load_drop_fraction_in_window"][wheel] = max(
                        diagnostics["max_load_drop_fraction_in_window"][wheel], load_drop,
                    )
                rising = valid and sample["lift_m"][wheel] >= 0.02 and sample["clearance_m"][wheel] >= 0.02
                unloading = (
                    valid and baseline is not None and force_valid
                    and state["baseline_samples"][wheel] >= self.baseline_steps
                    and load_drop >= self.load_drop_fraction and baseline - force >= self.min_force_drop_n
                    and sample["clearance_m"][wheel] >= self.min_clearance_m
                )
                # 真实离地与关节达标分开计时；不读取训练状态机，保留独立验收。
                airborne = (valid and force_valid and sample["clearance_m"][wheel] >= PRELIFT_TAKEOFF_CLEARANCE_M
                            and (force <= 1.0 or (self.allow_load_drop and unloading)))
                if (airborne and state["takeoff"][wheel] is None
                        and lead <= PRELIFT_PERCEPTION_MAX_ADVANCE_M
                        and (state["low_seen"][wheel] or preview[wheel])):
                    state["takeoff"][wheel] = {"lead_distance_m": lead, "time_s": state["steps"] * self.dt}
                    state["takeoff_valid"][wheel] = bool(
                        state["low_seen"][wheel] and in_window and sample["eligible"]
                    )
                if state["lifted"][wheel] is None and (
                    narrow_grounded or not valid or not force_valid or not sample.get("hold_eligible", sample["eligible"])
                ):
                    state["takeoff_valid"][wheel] = False
                developing = (state["takeoff_valid"][wheel] and valid and not state["contacted"][wheel]
                              and lead > 0 and sample["eligible"])
                state["unload_steps"][wheel] = (
                    state["unload_steps"][wheel] + 1 if unloading and developing else 0
                )
                if unloading and in_window:
                    diagnostics["unloaded_clearance_steps_in_window"][wheel] += 1
                avoidance = unloading if self.allow_load_drop else rising
                if (airborne and state["takeoff"][wheel] is not None and not state["contacted"][wheel]
                        and lead > self.max_lead_m):
                    diagnostics["early_lift_steps"][wheel] += 1
                    if diagnostics["first_early_lift_lead_m"][wheel] is None:
                        diagnostics["first_early_lift_lead_m"][wheel] = lead
                held = not self.allow_load_drop or state["unload_steps"][wheel] >= self.unload_hold_steps
                if (rising and developing
                        and state["strict_lifted"][wheel] is None):
                    state["strict_lifted"][wheel] = {
                        **state["takeoff"][wheel], "height_threshold_lead_m": lead,
                        "height_threshold_time_s": state["steps"] * self.dt,
                    }
                event = avoidance and held and developing and state["lifted"][wheel] is None
                if event:
                    state["lifted"][wheel] = {
                        **state["takeoff"][wheel],
                        "height_threshold_lead_m": lead, "height_threshold_time_s": state["steps"] * self.dt,
                        "event_type": "unloaded_clearance" if self.allow_load_drop else "articulation_lift",
                        "load_drop_fraction": load_drop if self.allow_load_drop else None,
                        "clearance_m": sample["clearance_m"][wheel],
                    }
                if self.require_hold_to_edge:
                    self._update_hold(state, sample, wheel, valid, force, force_valid, rising, unloading, event)
                # 首次真实离地消费本障碍资格，落下再抬不能改写首次起抬时机。
                if not valid or not force_valid or airborne or state["takeoff"][wheel] is not None:
                    state["low_seen"][wheel] = False
                elif narrow_grounded:
                    state["low_seen"][wheel] = True
                    state["low_seen_once"][wheel] = True
            if ending is not None:
                both = all(lift is not None for lift in state["lifted"])
                required = sum(state["lifted"][wheel] is not None for wheel in self.required_front_wheels) >= self.required_active_count
                # 同一轮必须完成起抬、保持和落轮，不能拼接左轮起抬与右轮落地。
                sequence_complete = sum(
                    state["lifted"][wheel] is not None and state["released"][wheel]
                    for wheel in self.required_front_wheels
                ) >= self.required_active_count
                self.records.append({
                    "env_id": i, "episode_index": ending["episode_index"],
                    "front_wheels": state["lifted"], "both_front_wheels_prelift": both,
                    "front_wheel_takeoffs": state["takeoff"],
                    "strict_front_wheel_lifts": state["strict_lifted"],
                    "blocking_collisions": state["collision_seen"],
                    "required_front_wheels_prelift": required,
                    "front_wheels_held_to_edge": state["held_to_edge"],
                    "front_wheels_released": state["released"],
                    "front_wheel_hold_failures": state["hold_broken"],
                    "required_front_wheels_sequence_complete": sequence_complete,
                    "traversal_success": ending["success"],
                    "success": required and ending["success"] and (
                        sequence_complete or not self.require_hold_to_edge),
                    "diagnostics": {
                        **diagnostics,
                        "low_seen_before_lift": state["low_seen_once"],
                        "baseline_samples": state["baseline_samples"],
                        "baseline_force_n": state["baseline_force_n"],
                        "first_contact_lead_m": state["first_contact_lead_m"],
                    },
                })
                self.pending[i] = self._empty()

    def _update_hold(self, state, sample, wheel, valid, force, force_valid, rising, unloading, event):
        if state["lifted"][wheel] is None or state["hold_broken"][wheel] or state["released"][wheel]:
            return
        local_clearance = sample.get("reward_clearance_m", sample["clearance_m"])[wheel]
        safe = sample.get("hold_eligible", sample["eligible"])
        valid = valid and math.isfinite(local_clearance)
        if not state["held_to_edge"][wheel]:
            holds = valid and safe and (unloading if self.allow_load_drop else rising)
            holds = holds and not sample.get("blocking_collision", [False, False])[wheel]
            if not holds:
                state["hold_broken"][wheel] = True
                return
            state["hold_steps"][wheel] += 1
            if event and self.allow_load_drop:
                state["hold_steps"][wheel] = state["unload_steps"][wheel]
            if sample["lead_m"][wheel] <= 0:
                if state["hold_steps"][wheel] < self.prelift_hold_steps:
                    state["hold_broken"][wheel] = True
                    return
                state["held_to_edge"][wheel] = True
        if state["held_to_edge"][wheel]:
            # 高度相对进场平地适合验证保持，但站在障碍顶面落轮必须使用局部净空。
            release_threshold = self.min_clearance_m if self.allow_load_drop else 0.01
            release = (valid and safe and force_valid and force > 1.0
                       and -0.02 <= local_clearance < release_threshold and sample["lead_m"][wheel] <= 0)
            state["release_steps"][wheel] = state["release_steps"][wheel] + 1 if release else 0
            state["released"][wheel] = state["release_steps"][wheel] >= self.release_hold_steps

    def summary(self, traversal_summary):
        successes = sum(r["success"] for r in self.records)
        rate = successes / len(self.records) if self.records else 0.0
        complete = traversal_summary["complete"] and len(self.records) == traversal_summary["expected_episodes"]
        return {
            "protocol": "rover_single_step_prelift", "complete": complete,
            "success_rate": rate, "passed": complete and rate >= PRELIFT_MINIMUM_SUCCESS_RATE,
            "front_wheel_event_rates": {
                label: sum(r["front_wheels"][wheel] is not None for r in self.records) / len(self.records)
                if self.records else 0.0 for wheel, label in enumerate(("fl", "fr"))
            },
            **{
                f"front_wheel_{label}_rates": {
                    name: sum(r[key][wheel] for r in self.records) / len(self.records)
                    if self.records else 0.0 for wheel, name in enumerate(("fl", "fr"))
                }
                for label, key in (("held_to_edge", "front_wheels_held_to_edge"),
                                   ("release", "front_wheels_released"),
                                   ("hold_failure", "front_wheel_hold_failures"))
            },
            "strict_front_wheel_lift_rates": {
                label: sum(r["strict_front_wheel_lifts"][wheel] is not None for r in self.records) / len(self.records)
                if self.records else 0.0 for wheel, label in enumerate(("fl", "fr"))
            },
            "front_wheel_early_lift_rates": {
                label: sum(r["diagnostics"]["early_lift_steps"][wheel] > 0 for r in self.records) / len(self.records)
                if self.records else 0.0 for wheel, label in enumerate(("fl", "fr"))
            },
            "takeoff_distance_m": {
                label: self._distance_stats(wheel, "front_wheel_takeoffs", "lead_distance_m")
                for wheel, label in enumerate(("fl", "fr"))
            },
            "height_threshold_distance_m": {
                label: self._distance_stats(wheel, "front_wheels", "height_threshold_lead_m")
                for wheel, label in enumerate(("fl", "fr"))
            },
            "gates": {"min_lead_distance_m": self.min_lead_m,
                      "max_lead_distance_m": self.max_lead_m,
                      "target_lead_distance_m": self.target_lead_m,
                      "timing_reference": "first_ground_clearance_with_unloading",
                      "takeoff_clearance_m": PRELIFT_TAKEOFF_CLEARANCE_M,
                      "min_articulation_lift_m": None if self.allow_load_drop else 0.02,
                      "min_ground_clearance_m": self.min_clearance_m if self.allow_load_drop else 0.02,
                      "allow_target_wheel_load_drop": self.allow_load_drop,
                      "min_load_drop_fraction": self.load_drop_fraction if self.allow_load_drop else None,
                      "min_load_drop_n": self.min_force_drop_n if self.allow_load_drop else None,
                      "min_unload_hold_s": self.unload_hold_steps * self.dt if self.allow_load_drop else None,
                      "requires_low_to_high_transition": True,
                      "requires_hold_to_edge_and_release": self.require_hold_to_edge,
                      "min_prelift_hold_s": self.prelift_hold_steps * self.dt,
                      "min_release_hold_s": self.release_hold_steps * self.dt,
                      "low_lift_threshold_m": None,
                      "low_ground_clearance_threshold_m": self.min_clearance_m,
                      "low_support_force_threshold_n": 1.0,
                      "baseline_min_support_force_n": self.min_force_drop_n + 1.0 if self.allow_load_drop else None,
                      "baseline_required_samples": self.baseline_steps if self.allow_load_drop else None,
                      "baseline_settle_s": self.baseline_settle_s,
                      "baseline_approach_distance_m": [self.preparation_max_advance_m, PRELIFT_PERCEPTION_MAX_ADVANCE_M],
                      "baseline_estimator": "rolling_contact_median",
                      "required_front_wheels": list(self.required_front_wheels),
                      "required_active_count": self.required_active_count,
                      "minimum_success_rate": PRELIFT_MINIMUM_SUCCESS_RATE},
            "episode_results": self.records,
        }

    def _distance_stats(self, wheel, key, field):
        values = [r[key][wheel][field] for r in self.records if r[key][wheel] is not None]
        return {"count": len(values), "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None, "max": max(values) if values else None}
