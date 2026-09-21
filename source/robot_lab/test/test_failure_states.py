"""失败轨迹不能被自动reset污染，数据重排和即时接管须保留速度与局部环境隔离。

@author pony
@date 2026-09-17
@version v0.3.0
@last_modified 2026-09-17
@changelog
    - v0.3.0 (2026-09-17): 拒绝空/错误采集来源并验证多次续训、未知历史继承和快照篡改。
    - v0.2.0 (2026-09-17): 覆盖同池改名、重打包轨迹重叠、缺来源及不同seed采集池。
    - v0.1.0 (2026-09-17): 验证真实采集窗口、无pickle合同、错误资产拒绝和重放边界。
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from test_recovery_initialization import recovery, env  # noqa: F401


@pytest.fixture
def states():
    source = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/recovery/mdp/failure_states.py"
    spec = importlib.util.spec_from_file_location("failure_states_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(n=2, joints=2):
    root = np.zeros((n, 13))
    root[:, 2:4] = [.4, 1.]
    root[:, 7:10] = [.4, -.2, .1]
    return root, np.tile(np.arange(joints), (n, 1)), np.ones((n, joints)), np.zeros((n, 12))


def create_pool(states, tmp_path):
    recorder = states.FailureStateRecorder(2, ["a", "b"], {
        "asset_sha256": "asset", "terrain": "plane", "control_dt_s": .02,
    }, window_steps=2)
    for step in range(1, 4):
        recorder.append(*sample(), [step, step])
    recorder.finish(0, failed=True, label={"env_id": 0, "terminated": False})
    recorder.finish(1, failed=False, label={})
    path = tmp_path / "states.npz"
    recorder.save(path)
    return path


def test_failure_pool_keeps_only_pre_reset_failed_window_and_reorders(states, tmp_path):
    path = create_pool(states, tmp_path)
    pool = states.load_failure_states(path, ["b", "a"], asset_sha256="asset", control_dt_s=.02)
    assert pool["joint_pos"].tolist() == [[1., 0.]]
    assert pool["root_state"][0, 7:10].tolist() == [.4, -.2, .1]
    with np.load(path, allow_pickle=False) as data:
        assert data["episode_step"].tolist() == [2, 3]
        assert data["episode_offsets"].tolist() == [0, 2]
    with pytest.raises(ValueError, match="漂移超时池"):
        states.load_failure_states(path, ["a", "b"], asset_sha256="asset", control_dt_s=.02,
                                   failure_kind="terminated")


@pytest.mark.parametrize("change", ["asset", "dt", "joints", "quaternion", "nonfinite", "offsets", "clock", "terrain"])
def test_pool_rejects_incompatible_or_corrupt_input(states, tmp_path, change):
    path = create_pool(states, tmp_path)
    args = {"asset_sha256": "asset", "control_dt_s": .02}
    names = ["a", "b"]
    if change == "asset":
        args["asset_sha256"] = "different"
    elif change == "dt":
        args["control_dt_s"] = .01
    elif change == "joints":
        names = ["a", "c"]
    else:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        if change == "quaternion":
            arrays["root_state"][:, 3] = 0
        elif change == "nonfinite":
            arrays["root_state"][:, 7] = np.nan
        elif change == "offsets":
            arrays["episode_offsets"][-1] = 99
        elif change == "clock":
            arrays["episode_step"][-1] += 1
        else:
            metadata = json.loads(str(arrays["metadata"].item()))
            arrays["metadata"] = np.asarray(json.dumps({**metadata, "terrain": "stairs"}))
        np.savez(path, **arrays)
    with pytest.raises(ValueError):
        states.load_failure_states(path, names, **args)


def test_local_reset_invalid_episode_and_no_silent_overwrite(states, tmp_path):
    recorder = states.FailureStateRecorder(2, ["a", "b"], {})
    recorder.append(*sample(), [1, 1])
    recorder.finish(0, failed=False, label={})
    values = list(sample())
    values[0][0, 7] = np.nan
    recorder.append(*values, [1, 2])
    recorder.finish(0, failed=True, label={})
    recorder.finish(1, failed=True, label={"env_id": 1})
    result = recorder.save(tmp_path / "states.npz")
    assert result["episodes"] == 1 and result["invalid_episodes_excluded"] == 1
    with pytest.raises(FileExistsError):
        recorder.save(tmp_path / "states.npz")


def test_recorder_refuses_a_missed_control_step(states):
    recorder = states.FailureStateRecorder(2, ["a", "b"], {})
    recorder.append(*sample(), [1, 1])
    with pytest.raises(ValueError, match="连续"):
        recorder.append(*sample(), [3, 2])


def provenance_pool(states, path, seed, extra=None):
    recorder = states.FailureStateRecorder(1, ["a", "b"], {
        "task": "motion", "checkpoint_sha256": "a" * 64, "seed": seed, **(extra or {}),
    })
    recorder.append(*sample(n=1), [1])
    recorder.finish(0, failed=True, label={"env_id": 0, "episode_index": 0, "terminated": True})
    recorder.save(path)
    return {"path": str(path), "sha256": states.file_sha256(path), "root_state": [sample(n=1)[0][0]]}


def test_evaluation_pool_separation_requires_traceable_distinct_collection(states, tmp_path):
    checkpoint = tmp_path / "model.pt"
    train = tmp_path / "params" / "failure_states.npz"
    evaluation = tmp_path / "evaluation.npz"
    source = provenance_pool(states, tmp_path / "source.npz", 43)
    states.snapshot_training_pools(tmp_path / "params", source)
    # 换名字不能绕过哈希；重打包改变文件字节也不能绕过采集episode身份。
    evaluation.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="数据分离"):
        states.evaluation_pool_status(checkpoint, evaluation)
    assert states.evaluation_pool_status(checkpoint, evaluation, allow_training_pool_smoke=True)["scope"] == "training_pool_smoke"
    repacked = tmp_path / "repacked.npz"
    provenance_pool(states, repacked, 43, {"note": "repacked"})
    with pytest.raises(ValueError, match="数据分离"):
        states.evaluation_pool_status(checkpoint, repacked)
    heldout = tmp_path / "heldout.npz"
    provenance_pool(states, heldout, 143)
    assert states.evaluation_pool_status(checkpoint, heldout)["data_separation_verified"]
    assert not states.evaluation_pool_status(checkpoint, heldout, allow_training_pool_smoke=True)["data_separation_verified"]
    with pytest.raises(ValueError, match="数据分离"):
        states.evaluation_pool_status(tmp_path / "unknown" / "model.pt", heldout)


@pytest.mark.parametrize("change", ["null_seed", "boolean_seed", "empty_task", "null_sha", "bad_sha",
                                   "empty_labels", "missing_episode", "null_env", "boolean_env"])
def test_incomplete_collection_identity_cannot_claim_separation(states, tmp_path, change):
    source = provenance_pool(states, tmp_path / "source.npz", 43)
    states.snapshot_training_pools(tmp_path / "params", source)
    evaluation = tmp_path / "evaluation.npz"
    provenance_pool(states, evaluation, 143)
    with np.load(evaluation, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = json.loads(str(arrays["metadata"].item()))
    if change == "null_seed":
        meta["seed"] = None
    elif change == "boolean_seed":
        meta["seed"] = True
    elif change == "empty_task":
        meta["task"] = "  "
    elif change in ("null_sha", "bad_sha"):
        meta["checkpoint_sha256"] = None if change == "null_sha" else "weights"
    elif change == "empty_labels":
        meta["labels"] = []
    elif change == "missing_episode":
        del meta["labels"][0]["episode_index"]
    else:
        meta["labels"][0]["env_id"] = None if change == "null_env" else False
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez(evaluation, **arrays)
    with pytest.raises(ValueError, match="数据分离"):
        states.evaluation_pool_status(tmp_path / "model.pt", evaluation)
    smoke = states.evaluation_pool_status(tmp_path / "model.pt", evaluation, allow_training_pool_smoke=True)
    assert not smoke["data_separation_verified"]


def test_resume_keeps_all_training_pools_across_non_replay_stage(states, tmp_path):
    first_pool = provenance_pool(states, tmp_path / "first.npz", 43)
    second_pool = provenance_pool(states, tmp_path / "second.npz", 44)
    checkpoints = [tmp_path / name / "model.pt" for name in ("first", "plain", "second")]
    states.snapshot_training_pools(checkpoints[0].parent / "params", first_pool)
    states.snapshot_training_pools(checkpoints[1].parent / "params", source_checkpoint=checkpoints[0])
    lineage = states.snapshot_training_pools(checkpoints[2].parent / "params", second_pool, checkpoints[1])
    assert lineage["complete"] and len(lineage["pools"]) == 2
    for trained in (first_pool, second_pool):
        with pytest.raises(ValueError, match="数据分离"):
            states.evaluation_pool_status(checkpoints[2], trained["path"])
    heldout = provenance_pool(states, tmp_path / "heldout.npz", 143)
    assert states.evaluation_pool_status(checkpoints[2], heldout["path"])["data_separation_verified"]
    inherited = next(entry for entry in lineage["pools"] if entry["sha256"] == first_pool["sha256"])
    (checkpoints[2].parent / "params" / inherited["snapshot"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="快照"):
        states.evaluation_pool_status(checkpoints[2], heldout["path"])


def test_legacy_unknown_lineage_cannot_be_laundered_by_resume(states, tmp_path):
    legacy = tmp_path / "legacy" / "model.pt"
    provenance_pool(states, legacy.parent / "params" / "failure_states.npz", 43)
    resumed = tmp_path / "resumed" / "model.pt"
    result = states.snapshot_training_pools(resumed.parent / "params", source_checkpoint=legacy)
    assert not result["complete"] and len(result["pools"]) == 1
    heldout = provenance_pool(states, tmp_path / "heldout.npz", 143)
    with pytest.raises(ValueError, match="数据分离"):
        states.evaluation_pool_status(resumed, heldout["path"])


def test_replay_preserves_momentum_and_other_environments(recovery, env, states, monkeypatch):
    package = "recovery_initialization_test_package"
    monkeypatch.setitem(sys.modules, package + ".failure_states", states)
    source = Path(__file__).parents[1] / "robot_lab/tasks/manager_based/recovery/mdp/failure_replay.py"
    spec = importlib.util.spec_from_file_location(package + ".failure_replay", source)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay.validate_replay_limits(torch.tensor([[.61]]), torch.tensor([[-.6109, .6109]]))
    with pytest.raises(ValueError, match="物理硬限位"):
        replay.validate_replay_limits(torch.tensor([[.62]]), torch.tensor([[-.6109, .6109]]))
    robot = env.scene["robot"]
    env.cfg = SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_type="plane")))
    roots = robot.data.root_state_w[:1].clone()
    roots[0, :3] = torch.tensor([8., 9., .32])
    roots[0, 7:] = torch.tensor([.4, -.2, .1, .2, -.3, .5])
    env._failure_state_pool = {"root_state": roots, "joint_pos": torch.zeros(1, 20),
                               "joint_vel": torch.full((1, 20), .1), "path": "pool"}
    env.scene.env_origins[1] = torch.tensor([10., 20., 0.])
    robot.write_root_state_to_sim = lambda value, ids: robot.data.root_state_w.__setitem__(ids, value)
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: (
        robot.data.joint_pos.__setitem__(env_ids, pos), robot.data.joint_vel.__setitem__(env_ids, vel))
    env.action_manager.reset = lambda ids: None
    initial = robot.data.root_state_w.clone()
    replay.reset_failure_state(env, torch.tensor([1]), "pool")
    torch.testing.assert_close(robot.data.root_state_w[[0, 2]], initial[[0, 2]])
    torch.testing.assert_close(robot.data.root_state_w[1, :3], torch.tensor([10., 20., .32]))
    torch.testing.assert_close(robot.data.root_state_w[1, 7:], roots[0, 7:])
    replay.initialize_failure_takeover(env, torch.tensor([1]))
    assert env._recovery_control_active.tolist() == [False, True, False]
    assert env._recovery_control_steps.tolist() == [0, 0, 0]
    assert env._recovery_settle_steps.tolist() == [0, 0, 0]
    assert robot.actuators["legs"].stiffness[1].tolist() == [20.] * 8
    assert env._recovery_takeover_root_state_w[1, 7:].tolist() == roots[0, 7:].tolist()
