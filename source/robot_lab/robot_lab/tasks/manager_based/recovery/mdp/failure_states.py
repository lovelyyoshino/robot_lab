"""保存真实策略失败前的状态窗口，供独立恢复任务采样；不声称重建接触求解器状态。

@author pony
@date 2026-09-17
@version v0.3.0
@last_modified 2026-09-17
@changelog
    - v0.3.0 (2026-09-17): 校验非空采集身份并继承全部续训池快照，未知历史不能声称数据分离。
    - v0.2.0 (2026-09-17): 比较评测池与训练快照/采集轨迹，拒绝同池或重打包轨迹冒充独立评测。
    - v0.1.0 (2026-09-17): 增加 reset 前轨迹采集、无 pickle 数据合同和按名称重放末态。
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _collection_identity(path):
    """来源缺失允许离线诊断，但空标签/伪哈希不能获得独立评测资格。"""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        offsets = data["episode_offsets"]
    if not isinstance(meta, dict):
        return None
    task, seed = meta.get("task"), meta.get("seed")
    labels = meta.get("labels")
    if (not isinstance(task, str) or not task.strip() or not _valid_sha(meta.get("checkpoint_sha256"))
            or type(seed) is not int or seed < 0 or not isinstance(labels, list) or not labels
            or offsets.ndim != 1 or len(offsets) != len(labels) + 1):
        return None
    identities = set()
    for label in labels:
        if not isinstance(label, dict):
            return None
        key = (label.get("env_id"), label.get("episode_index"))
        if any(type(value) is not int or value < 0 for value in key) or key in identities:
            return None
        identities.add(key)
    return (task, meta["checkpoint_sha256"], seed), identities


def _training_pool_history(checkpoint):
    """只读取实验目录内可校验快照；旧实验没有完整谱系时保留未知状态。"""
    params = Path(checkpoint).resolve().parent / "params"
    manifest = params / "failure_pool_lineage.json"
    current = params / "failure_states.npz"
    if not manifest.is_file():
        return ([current] if current.is_file() else []), False
    history = json.loads(manifest.read_text(encoding="utf-8"))
    if (history.get("schema") != "rover_failure_pool_lineage_v1"
            or type(history.get("complete")) is not bool or not isinstance(history.get("pools"), list)):
        raise ValueError("失败池训练谱系格式错误")
    paths = []
    for entry in history["pools"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("snapshot"), str)
                or not _valid_sha(entry.get("sha256"))):
            raise ValueError("失败池训练谱系条目错误")
        path = (params / entry["snapshot"]).resolve()
        if not path.is_relative_to(params) or not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError("失败池训练谱系快照缺失、越界或哈希不匹配")
        paths.append(path)
    if current.is_file() and file_sha256(current) not in {file_sha256(path) for path in paths}:
        raise ValueError("当前失败池未包含在训练谱系中")
    return paths, history["complete"]


def snapshot_training_pools(params, failure_pool=None, source_checkpoint=None):
    """随训练保存当前与历史池；跨实验续训不能遗忘已训练过的数据。

    无来源的新训练建立完整空谱系；来源实验无谱系则永久保留未知标志，
    不能通过再续训一次洗掉来源缺失。快照路径相对实验目录，支持整体搬迁。
    """
    params = Path(params)
    params.mkdir(parents=True, exist_ok=True)
    paths, complete = ([], True) if source_checkpoint is None else _training_pool_history(source_checkpoint)
    entries = {}
    if failure_pool is not None:
        source = Path(failure_pool["path"])
        sha = failure_pool["sha256"]
        if file_sha256(source) != sha:
            raise ValueError("失败池在加载后发生变化，拒绝训练来源不一致")
        target = params / "failure_states.npz"
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if file_sha256(target) != sha:
            raise ValueError("失败池实验快照哈希不匹配")
        entries[sha] = {"source": str(source.resolve()), "snapshot": target.name, "sha256": sha}
        with (params / "failure_states.json").open("x", encoding="utf-8") as stream:
            json.dump({**entries[sha], "states": len(failure_pool["root_state"]),
                       "protocol": "immediate_failure_state_replay_v1"}, stream, indent=2)
    for source in paths:
        sha = file_sha256(source)
        if sha in entries:
            continue
        target = params / "failure_pool_ancestors" / (sha + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if file_sha256(target) != sha:
            raise ValueError("历史失败池复制期间发生变化")
        entries[sha] = {"source": str(source), "snapshot": str(target.relative_to(params)), "sha256": sha}
    history = {"schema": "rover_failure_pool_lineage_v1", "complete": complete,
               "source_checkpoint": str(Path(source_checkpoint).resolve()) if source_checkpoint is not None else None,
               "pools": list(entries.values())}
    with (params / "failure_pool_lineage.json").open("x", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2, allow_nan=False)
    return history


def evaluation_pool_status(checkpoint, evaluation_path, *, allow_training_pool_smoke=False):
    """核对全部已知训练池；采集元数据分离不等于统计独立或仿真泛化。"""
    paths, complete = _training_pool_history(checkpoint)
    evaluation_sha = file_sha256(evaluation_path)
    eval_identity = _collection_identity(evaluation_path)
    status = {"training_snapshots": [str(path) for path in paths], "evaluation_sha256": evaluation_sha,
              "training_sha256": [file_sha256(path) for path in paths], "training_history_complete": complete,
              "data_separation_verified": False, "scope": "unverified_pool_smoke"}
    known = complete and eval_identity is not None
    overlap = evaluation_sha in status["training_sha256"]
    for path in paths:
        train_identity = _collection_identity(path)
        known &= train_identity is not None
        if train_identity is not None and eval_identity is not None:
            overlap |= train_identity[0] == eval_identity[0] and bool(train_identity[1] & eval_identity[1])
    if overlap:
        status["scope"] = "training_pool_smoke"
    elif known:
        status["scope"] = "distinct_collection_pool"
        status["data_separation_verified"] = True
    if not status["data_separation_verified"] and not allow_training_pool_smoke:
        raise ValueError("无法确认评测池与全部训练数据分离；同池/缺来源只允许 --allow_training_pool_smoke，不能作为验收")
    if allow_training_pool_smoke:
        status["data_separation_verified"] = False
        status["scope"] = "training_pool_smoke" if overlap else "pool_smoke"
    return status


class FailureStateRecorder:
    """只存配额内失败 episode 的末段；成功样本与非有限状态不能进入恢复初态池。"""

    def __init__(self, num_envs, joint_names, metadata, window_steps=25):
        if num_envs < 1 or window_steps < 1 or len(set(joint_names)) != len(joint_names):
            raise ValueError("轨迹环境数/窗口须为正且关节名唯一")
        self.names = list(joint_names)
        self.metadata = dict(metadata)
        self.buffers = [deque(maxlen=window_steps) for _ in range(num_envs)]
        self.trajectories = []
        self.labels = []
        self.invalid_episodes = 0
        self.invalid = [False] * num_envs

    def append(self, roots_local, positions, velocities, actions, episode_steps):
        arrays = [np.asarray(value, dtype=np.float64) for value in (roots_local, positions, velocities, actions)]
        n, joints = len(self.buffers), len(self.names)
        if [a.shape for a in arrays] != [(n, 13), (n, joints), (n, joints), (n, 12)]:
            raise ValueError("失败轨迹必须覆盖全部环境的根13维、实名关节和12维动作")
        if len(episode_steps) != n:
            raise ValueError("失败轨迹时钟必须覆盖全部环境")
        for index, step in enumerate(episode_steps):
            if step < 1 or int(step) != step:
                raise ValueError("失败轨迹必须在物理步后、自动reset前采集")
            if self.buffers[index] and step != self.buffers[index][-1][-1] + 1:
                raise ValueError("失败轨迹控制步必须连续，reset后须清空该环境窗口")
            frame = tuple(array[index].copy() for array in arrays) + (int(step),)
            self.invalid[index] |= not all(np.isfinite(a[index]).all() for a in arrays)
            self.buffers[index].append(frame)

    def finish(self, env_id, *, failed, label):
        if failed and self.buffers[env_id]:
            if self.invalid[env_id]:
                self.invalid_episodes += 1
            else:
                self.trajectories.append(list(self.buffers[env_id]))
                self.labels.append(dict(label))
        self.buffers[env_id].clear()
        self.invalid[env_id] = False

    def save(self, path):
        frames = [frame for trajectory in self.trajectories for frame in trajectory]
        offsets = np.cumsum([0] + [len(trajectory) for trajectory in self.trajectories])
        metadata = {**self.metadata, "schema": "rover_failure_states_v1", "joint_names": self.names,
                    "root_frame": "position_relative_to_env_origin_orientation_wxyz_velocities_world",
                    "capture": "after_physics_before_auto_reset", "labels": self.labels,
                    "invalid_episodes_excluded": self.invalid_episodes,
                    "scope": "kinematic_failure_replay_not_solver_or_material_state_reproduction"}
        arrays = {}
        for index, (name, width) in enumerate((("root_state", 13), ("joint_pos", len(self.names)),
                                             ("joint_vel", len(self.names)), ("action", 12))):
            arrays[name] = np.stack([frame[index] for frame in frames]) if frames else np.empty((0, width))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # exclusive 创建避免复测静默覆盖原始失败池；空池保留诊断但 loader 明确拒绝训练。
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays, episode_offsets=offsets,
                                episode_step=np.asarray([frame[-1] for frame in frames], dtype=np.int64),
                                metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
        return {"path": str(path), "sha256": file_sha256(path), "episodes": len(self.trajectories),
                "frames": len(frames), "invalid_episodes_excluded": self.invalid_episodes}


def load_failure_states(path, joint_names, *, asset_sha256, control_dt_s, failure_kind="all"):
    """验证模型、周期、数据形状与姿态后返回真实失败末态；只允许明确的关节名称重排。"""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta.get("schema") != "rover_failure_states_v1" or meta.get("capture") != "after_physics_before_auto_reset":
            raise ValueError("失败状态池来源/版本不受支持")
        if meta.get("root_frame") != "position_relative_to_env_origin_orientation_wxyz_velocities_world":
            raise ValueError("失败状态坐标合同不匹配")
        if meta.get("terrain") != "plane" or meta.get("asset_sha256") != asset_sha256:
            raise ValueError("失败状态池机器人资产或平地合同不匹配")
        if not math.isclose(meta.get("control_dt_s", -1), control_dt_s, abs_tol=1e-10, rel_tol=0):
            raise ValueError("失败状态池控制周期不匹配")
        saved_names = meta.get("joint_names", [])
        if len(saved_names) != len(set(saved_names)) or len(joint_names) != len(set(joint_names)) or set(saved_names) != set(joint_names):
            raise ValueError("失败状态池关节名称不匹配")
        arrays = {key: np.array(data[key], copy=True) for key in ("root_state", "joint_pos", "joint_vel", "action")}
        offsets, steps = data["episode_offsets"], data["episode_step"]
        count = len(arrays["root_state"])
        if count == 0:
            raise ValueError("失败状态池为空，不能用于训练")
        for key, width in (("root_state", 13), ("joint_pos", len(saved_names)),
                           ("joint_vel", len(saved_names)), ("action", 12)):
            if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"失败状态池{key}维度错误或含非有限值")
        if (offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer) or len(offsets) < 2
                or offsets[0] != 0 or offsets[-1] != count or (np.diff(offsets) <= 0).any()
                or len(meta.get("labels", [])) != len(offsets) - 1):
            raise ValueError("失败状态池episode边界错误")
        if steps.shape != (count,) or not np.issubdtype(steps.dtype, np.integer) or (steps < 1).any():
            raise ValueError("失败状态池时钟错误")
        for start, end in zip(offsets[:-1], offsets[1:]):
            if (np.diff(steps[start:end]) != 1).any():
                raise ValueError("失败状态池存在漏采样")
        if not np.allclose(np.linalg.norm(arrays["root_state"][:, 3:7], axis=-1), 1, atol=1e-4):
            raise ValueError("失败状态池四元数不是单位四元数")
        if failure_kind not in ("all", "terminated"):
            raise ValueError("failure_kind必须为all或terminated")
        final = offsets[1:] - 1
        if failure_kind == "terminated":
            # 只选择源任务真实提前失败；超时漂移属于另一分布，不能冒称倒地接管。
            selected = np.asarray([label.get("terminated") is True for label in meta["labels"]])
            final = final[selected]
            if len(final) == 0:
                raise ValueError("失败池没有真实提前终止样本；漂移超时池不能冒充失稳池")
        order = [saved_names.index(name) for name in joint_names]
        return {"root_state": arrays["root_state"][final], "joint_pos": arrays["joint_pos"][final][:, order],
                "joint_vel": arrays["joint_vel"][final][:, order], "metadata": meta,
                "sha256": file_sha256(path), "failure_kind": failure_kind}
