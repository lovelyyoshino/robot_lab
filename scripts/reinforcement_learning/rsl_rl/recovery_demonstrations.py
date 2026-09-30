"""将严格成功的恢复脚本回合保存为可追溯、无pickle的动作前观测示范。

@author pony
@date 2026-09-17
@version v0.3.0
@last_modified 2026-09-28
@changelog
    - v0.3.0 (2026-09-28): 支持独立侧躺/翻覆与进展任务，保持原严格四轮成功门。
    - v0.2.0 (2026-09-17): 重建来源分段动作及限位合同，防止改写示范动作后伪装原始执行。
    - v0.1.0 (2026-09-17): 逐控制步执行确认、完整成功回合筛选与来源/合同校验。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "rover_recovery_demonstrations_v1"
OBSERVATION_ORDER = ["projected_gravity", "base_ang_vel", "joint_pos_legs", "joint_vel_legs", "wheel_vel",
                     "prev_joint_pos_legs", "prev_joint_vel_legs", "prev_wheel_vel", "last_action"]
OBSERVATION_DIMS = [3, 3, 8, 8, 4, 8, 8, 4, 12]
STATE_DIMS = {"root_state_w": 13, "joint_pos": 20, "joint_vel": 20}


def canonical_config(value):
    """只规范数据类型，不执行对象构造；slice明确编码，函数应已由cfg.to_dict转为名字。"""
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("合同字典键必须为字符串")
        return {key: canonical_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_config(item) for item in value]
    if isinstance(value, slice):
        return {"__slice__": [value.start, value.stop, value.step]}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"合同包含不支持或非有限数据类型: {type(value).__name__}")


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_contract(contract):
    actor_dim, critic_dim = contract.get("actor_observation_dim"), contract.get("critic_observation_dim")
    if (actor_dim, critic_dim, contract.get("action_dim")) != (58, 70, 12):
        raise ValueError("示范只支持58/70→12恢复合同")
    expected_order = OBSERVATION_ORDER
    expected_dims = [[n] for n in OBSERVATION_DIMS]
    if contract.get("observation_order") != expected_order or contract.get("observation_term_dims") != expected_dims:
        raise ValueError("恢复观测语义或顺序不匹配")
    if not math.isfinite(contract.get("step_dt", 0)) or contract["step_dt"] <= 0:
        raise ValueError("step_dt必须有限且为正")
    names = contract.get("action_joint_names", [])
    if len(names) != 12 or len(set(names)) != 12 or not all(isinstance(name, str) for name in names):
        raise ValueError("需要12个唯一的实名动作关节")
    for name in ("action_scales", "action_offsets"):
        values = np.asarray(contract.get(name))
        if values.shape != (12,) or not np.isfinite(values).all() or (name == "action_scales" and (values == 0).any()):
            raise ValueError(f"{name}必须为12维有限值且scale非零")
    canonical_config(contract)


def _validate_source(source):
    if (source.get("complete") is not True or source.get("skill") != "recovery"
                or source.get("task") not in (
                "RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0",
                "RobotLab-Isaac-Recovery-Grounded-Progress-Parallelogram-Rover-v0",
                "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0",
                "RobotLab-Isaac-Recovery-Upside-Parallelogram-Rover-v0",
            )
            or source.get("protocol") != "settled_ground"
            or source.get("recovery_support_protocol") != "four_wheel_stable_v1"
            or source.get("action_source") != "segmented_joint_targets_with_active_pd"
            or source.get("evaluation_scope") != "bounded_action_program_search_not_policy_acceptance"
            or source.get("checkpoint_policy_executed") is not False
            or source.get("acceptance_eligible") is not False or source.get("passed") is not False):
        raise ValueError("示范来源必须是完整严格Grounded脚本诊断，不能冒充策略验收")
    if source["task"] in (
        "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0",
        "RobotLab-Isaac-Recovery-Upside-Parallelogram-Rover-v0",
    ) and not isinstance(source.get("control_contract"), dict):
        raise ValueError("新恢复任务必须携带物理控制合同，不能只凭相同维度认定来源兼容")
    if (source.get("episode_length_s") != 5. or source.get("gates", {}).get("maximum_time_to_stable_s") != 4.
            or source.get("gates", {}).get("minimum_terminal_stable_time_s") != .5):
        raise ValueError("恢复成功门或受控时间预算已改变")
    records = source.get("episode_results", [])
    count = source.get("expected_episodes")
    if count != len(records) or source.get("episodes") != count or source.get("episodes_by_env") != [1] * count:
        raise ValueError("采集要求每环境完整单回合")
    if sorted(record["env_id"] for record in records) != list(range(count)):
        raise ValueError("回合环境身份缺失或重复")


def _validate_success(record, steps, dt):
    if (record.get("success") is not True or record.get("truncated") is not True
            or record.get("terminated") is not False or record.get("nonfinite_seen") is not False
            or record.get("initialization_failed") is not False or record.get("takeover_valid") is not True
            or record.get("episode_index") != 0):
        raise ValueError("回合未通过完整恢复门")
    hold, start = record.get("terminal_stable_time_s"), record.get("time_to_stable_s")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (hold, start)):
        raise ValueError("成功时钟缺失或非有限")
    if hold < .5 or not 0 < start <= 4. + 1e-9 or not math.isclose(steps * dt, 5., abs_tol=1e-8):
        raise ValueError("回合未完整保持至原5秒终态")
    if not math.isclose(record.get("duration_s", -1), steps * dt, abs_tol=1e-8):
        raise ValueError("采集样本数与受控时间不一致")
    if not math.isclose(hold + start - dt, steps * dt, abs_tol=1e-8):
        raise ValueError("成功保持区间没有连续延伸至终态")


class RecoveryDemonstrationCollector:
    """先暂存动作前状态，再于termination快照确认执行；reset后的状态不能入库。"""

    def __init__(self, num_envs, max_steps, contract):
        _validate_contract(contract)
        if type(num_envs) is not int or num_envs < 1 or type(max_steps) is not int or max_steps < 1:
            raise ValueError("环境数及步数必须为正整数")
        if not math.isclose(max_steps * contract["step_dt"], 5., abs_tol=1e-8):
            raise ValueError("示范须保留完整5秒控制预算")
        self.num_envs, self.max_steps = num_envs, max_steps
        self.contract = copy.deepcopy(contract)
        self.rows = [[] for _ in range(num_envs)]
        self.finished = np.zeros(num_envs, dtype=bool)
        self.pending = None

    def before_step(self, observations, actions, control_steps, active, **state):
        if self.pending is not None:
            raise ValueError("上一物理步尚未确认执行")
        clock, active = _array(control_steps), _array(active)
        if clock.shape != (self.num_envs,) or active.shape != clock.shape or clock.dtype.kind not in "iu":
            raise ValueError("接管时钟维度或类型不匹配")
        ids = np.flatnonzero(active & ~self.finished)
        values = {"observations": _array(observations), "actions": _array(actions),
                  **{key: _array(state[key]) for key in STATE_DIMS}}
        for key, width in {"observations": 58, "actions": 12, **STATE_DIMS}.items():
            if values[key].shape != (self.num_envs, width) or not np.isfinite(values[key][ids]).all():
                raise ValueError(f"动作前{key}的维度或数值损坏")
        if any(clock[i] != len(self.rows[i]) or clock[i] >= self.max_steps for i in ids):
            raise ValueError("每回合控制时钟必须从0连续采集，不能漏帧或拼接reset")
        self.pending = {int(i): {"control_step": int(clock[i]),
                                **{key: value[i].astype(np.float32, copy=True) for key, value in values.items()}}
                        for i in ids}

    def confirm_step(self, control_steps, active, actual_actions, **state):
        if self.pending is None:
            raise ValueError("未记录动作前观测，不能确认物理执行")
        clock, active, actual = _array(control_steps), _array(active), _array(actual_actions)
        after = {key: _array(state[key]) for key in STATE_DIMS}
        for index, row in self.pending.items():
            if not active[index] or clock[index] != row["control_step"] + 1:
                raise ValueError("执行确认必须采自reset前的下一控制步")
            if not np.array_equal(actual[index], row["actions"]):
                raise ValueError("记录动作与实际action manager动作不同，可能发生额外clip/reset")
            for key, width in STATE_DIMS.items():
                if after[key].shape != (self.num_envs, width) or not np.isfinite(after[key][index]).all():
                    raise ValueError(f"执行后{key}损坏")
        for index, row in self.pending.items():
            row.update({f"next_{key}": value[index].astype(np.float32, copy=True) for key, value in after.items()})
            self.rows[index].append(row)
        self.pending = None

    def finish_episode(self, record):
        index = record["env_id"]
        if self.pending is not None or self.finished[index]:
            raise ValueError("完成回合前须确认物理步，不能重复完成")
        if record["success"]:
            _validate_success(record, len(self.rows[index]), self.contract["step_dt"])
        else:
            self.rows[index].clear()
        self.finished[index] = True

    def arrays(self, source):
        _validate_source(source)
        if self.pending is not None or not self.finished.all():
            raise ValueError("采集未结束，不能导出部分成功样本")
        accepted = sorted((copy.deepcopy(r) for r in source["episode_results"] if r["success"]), key=lambda r: r["env_id"])
        if not accepted:
            raise ValueError("没有完整成功回合，不生成看似可训练的空示范")
        arrays, episodes = {}, []
        for episode_id, record in enumerate(accepted):
            rows = self.rows[record["env_id"]]
            _validate_success(record, len(rows), self.contract["step_dt"])
            for key in rows[0]:
                arrays.setdefault(key, []).append(np.asarray([row[key] for row in rows]))
            arrays.setdefault("episode_id", []).append(np.full(len(rows), episode_id, dtype=np.int64))
            record.update(program_id=record["env_id"], trajectory_index=episode_id)
            episodes.append(record)
        arrays = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
        arrays["control_step"] = arrays["control_step"].astype(np.int64)
        return arrays, episodes

    def save(self, path, source_path, source):
        path, source_path = Path(path).resolve(), Path(source_path).resolve()
        if path.exists():
            raise FileExistsError(f"示范文件已存在，拒绝覆盖: {path}")
        arrays, episodes = self.arrays(source)
        source_bytes = source_path.read_bytes()
        if json.loads(source_bytes) != source:
            raise ValueError("源评测JSON与本次采集来源不一致")
        metadata = {"schema": SCHEMA, "source_evaluation_path": str(source_path),
                    "source_evaluation_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "contract": self.contract, "source": source, "episodes": episodes,
                    "sampling": "pre_step_observation_and_raw_action_confirmed_before_auto_reset",
                    "acceptance_scope": "successful_script_episodes_not_trained_policy"}
        _validate_arrays(arrays, metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "successful_episodes": len(episodes), "samples": len(arrays["observations"])}


def _validate_arrays(arrays, metadata):
    if metadata.get("schema") != SCHEMA:
        raise ValueError("不支持的恢复示范schema")
    contract, source, episodes = metadata["contract"], metadata["source"], metadata["episodes"]
    _validate_contract(contract)
    _validate_source(source)
    if source.get("control_contract") != contract.get("control_contract"):
        raise ValueError("示范和评测的物理控制合同不一致")
    if source["step_dt"] != contract["step_dt"] or not episodes:
        raise ValueError("示范控制周期不符或没有成功回合")
    size = len(arrays["observations"])
    dimensions = {"observations": 58, "actions": 12, **STATE_DIMS,
                  **{f"next_{key}": width for key, width in STATE_DIMS.items()}}
    if set(arrays) != set(dimensions) | {"control_step", "episode_id"}:
        raise ValueError("示范字段缺失或未知")
    for key, width in dimensions.items():
        if arrays[key].shape != (size, width) or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"示范{key}形状、类型或有限值不匹配")
    for key in ("control_step", "episode_id"):
        if arrays[key].shape != (size,) or arrays[key].dtype != np.int64:
            raise ValueError("示范索引须为逐帧int64")
    if not np.array_equal(np.unique(arrays["episode_id"]), np.arange(len(episodes))):
        raise ValueError("示范episode身份缺失/多余")
    accepted = {r["env_id"]: r for r in source["episode_results"] if r["success"]}
    if sorted(r["env_id"] for r in episodes) != sorted(accepted):
        raise ValueError("示范与来源成功回合集合不一致")
    program = source.get("action_program", {})
    if program.get("joint_names") != contract["action_joint_names"] or program.get("step_dt") != contract["step_dt"]:
        raise ValueError("来源动作程序与示范关节/周期合同不同")
    targets = np.asarray(program.get("targets"), dtype=np.float32)
    defaults = np.asarray(program.get("default_targets"), dtype=np.float32)
    scales = np.asarray(program.get("action_scales"), dtype=np.float32)
    segment_steps, active_steps = program.get("segment_steps"), program.get("active_steps")
    if (type(segment_steps) is not int or segment_steps < 1 or type(active_steps) is not int
            or active_steps < 1 or active_steps % segment_steps != 0):
        raise ValueError("来源动作程序分段时钟损坏")
    expected_shape = (source["expected_episodes"], active_steps // segment_steps, 12)
    identities = program.get("source_program_ids")
    mirrors = program.get("mirrored")
    if identities is not None or mirrors is not None:
        if (not isinstance(identities, list) or len(identities) != expected_shape[0]
                or any(type(value) is not int or not 0 <= value < expected_shape[0] for value in identities)
                or not isinstance(mirrors, list) or len(mirrors) != expected_shape[0]
                or any(type(value) is not bool for value in mirrors)):
            raise ValueError("来源程序或镜像身份损坏")
    if targets.shape != expected_shape or defaults.shape != (expected_shape[0], 12) or scales.shape != defaults.shape:
        raise ValueError("来源动作程序维度损坏")
    if not all(np.isfinite(value).all() for value in (targets, defaults, scales)) or (scales == 0).any():
        raise ValueError("来源动作程序非有限或不可逆")
    if not (np.array_equal(scales, np.broadcast_to(np.asarray(contract["action_scales"], np.float32), scales.shape))
            and np.array_equal(defaults, np.broadcast_to(np.asarray(contract["action_offsets"], np.float32), defaults.shape))):
        raise ValueError("来源程序的scale/offset与声明不匹配")
    for index, record in enumerate(episodes):
        row = {k: v for k, v in record.items() if k not in ("trajectory_index", "program_id")}
        if row != accepted[record["env_id"]] or record["trajectory_index"] != index or record["program_id"] != row["env_id"]:
            raise ValueError("回合来源身份被修改")
        mask = arrays["episode_id"] == index
        count = int(mask.sum())
        _validate_success(record, count, contract["step_dt"])
        if not np.array_equal(arrays["control_step"][mask], np.arange(count)):
            raise ValueError("回合时钟缺失、重复或乱序")
        if not np.array_equal(arrays["observations"][mask][1:, -12:], arrays["actions"][mask][:-1]):
            raise ValueError("观测last_action与前一步执行动作不一致，疑似串帧")
        if arrays["observations"][mask][0, -12:].any():
            raise ValueError("首个接管观测未清零动作历史")
        expected_actions = np.zeros((count, 12), dtype=np.float32)
        if active_steps > count:
            raise ValueError("程序超过受控预算")
        expected_actions[:active_steps] = (
            np.repeat(targets[record["program_id"]], segment_steps, axis=0) - defaults[record["program_id"]]
        ) / scales[record["program_id"]]
        if not np.allclose(arrays["actions"][mask], expected_actions, rtol=1e-6, atol=1e-6):
            raise ValueError("示范动作与来源分段程序不一致")
        for key in STATE_DIMS:
            actual = arrays[key][mask]
            if not np.allclose(actual[0], record[f"takeover_{key}"], rtol=0, atol=1e-6):
                raise ValueError("首个动作前状态与真实接管状态不一致")
            if not np.allclose(actual[1:], arrays[f"next_{key}"][mask][:-1], rtol=0, atol=1e-6):
                raise ValueError("前后物理状态不连续，可能包含reset或错位")


def load_demonstrations(path):
    """训练入口必须使用此loader；拒绝pickle、损坏来源、非有限值和不完整回合。"""
    with np.load(Path(path), allow_pickle=False) as archive:
        raw = archive["metadata"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("metadata必须为无pickle的标量JSON字符串")
        metadata = json.loads(raw.item())
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
    source_bytes = Path(metadata["source_evaluation_path"]).read_bytes()
    if (hashlib.sha256(source_bytes).hexdigest() != metadata["source_evaluation_sha256"]
            or json.loads(source_bytes) != metadata["source"]):
        raise ValueError("示范来源评测文件已改变")
    _validate_arrays(arrays, metadata)
    return arrays, metadata
