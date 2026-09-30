"""恢复控制合同隔离：维度相同不能掩盖动作尺度或驱动模型改变。

@author pony
@date 2026-09-28
@version v0.1.0
@last_modified 2026-09-28
@changelog
    - v0.1.0 (2026-09-28): 覆盖新旧控制合同、评测噪声差异及资产内容变更。
"""

import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def contracts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"))
    import recovery_contract

    return recovery_contract


@pytest.fixture
def configs(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="fixture"/>')
    env = {
        "sim": {"dt": .005}, "decimation": 4,
        "actions": {"joint_pos": {"scale": .25}, "joint_vel": {"scale": .8}},
        "observations": {"policy": {"enable_corruption": True, "gravity": {"func": "gravity"}},
                         "critic": {"enable_corruption": False, "gravity": {"func": "gravity"}}},
        "scene": {"robot": {"init_state": {"joint_pos": {".*": 0.}, "joint_vel": {".*": 0.}},
                            "actuators": {"suspension": {"stiffness": 20.}},
                            "spawn": {"asset_path": str(urdf), "self_collision": False}}},
    }
    agent = {"policy": {"class_name": "ActorCritic"}, "clip_actions": None, "obs_groups": {}}
    return env, agent


def test_control_contract_ignores_run_geometry_and_evaluation_corruption(contracts, configs):
    env, agent = configs
    source = contracts.control_contract(env, agent)
    target = copy.deepcopy(env)
    target["observations"]["policy"]["enable_corruption"] = False
    target["scene"]["num_envs"] = 4
    target["seed"] = 900
    contracts.validate_control_contract(source, contracts.control_contract(target, agent))


@pytest.mark.parametrize("mutation", [
    lambda env: env["actions"]["joint_vel"].update(scale=10.),
    lambda env: env["scene"]["robot"]["actuators"]["suspension"].update(stiffness=40.),
    lambda env: env["scene"]["robot"]["spawn"].update(self_collision=True),
    lambda env: env["sim"].update(dt=.01),
    lambda env: env["observations"]["policy"]["gravity"].update(func="wrong"),
    lambda env: env.update(actions=dict(reversed(list(env["actions"].items())))),
])
def test_semantic_mismatch_rejected(contracts, configs, mutation):
    env, agent = configs
    source = contracts.control_contract(env, agent)
    mutation(env)
    with pytest.raises(ValueError, match="恢复控制合同"):
        contracts.validate_control_contract(source, contracts.control_contract(env, agent))


def test_same_asset_path_changed_bytes_rejected(contracts, configs):
    env, agent = configs
    source = contracts.control_contract(env, agent)
    Path(env["scene"]["robot"]["spawn"]["asset_path"]).write_text('<robot name="changed"/>')
    with pytest.raises(ValueError, match="恢复控制合同"):
        contracts.validate_control_contract(source, contracts.control_contract(env, agent))


def test_snapshot_roundtrip_and_no_silent_missing_params(contracts, configs, tmp_path):
    env, agent = configs
    current = contracts.control_contract(env, agent)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    with pytest.raises(FileNotFoundError):
        contracts.validate_checkpoint_contract(checkpoint, current)
    params = tmp_path / "params"
    contracts.save_control_contract(params, current)
    assert json.loads((params / "recovery_control_contract.json").read_text())["contract"] == current
    result = contracts.validate_checkpoint_contract(checkpoint, current)
    assert result["source"] == "snapshot"


def test_recovery_task_classification_is_shared_across_entrypoints(contracts):
    assert contracts.is_residual_recovery_task(
        "RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0"
    )
    assert contracts.is_residual_recovery_task(
        "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0"
    )
    assert contracts.is_failure_replay_recovery_task(
        "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0"
    )
    assert not contracts.is_failure_replay_recovery_task(
        "RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0"
    )
    assert not contracts.is_residual_recovery_task(
        "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0"
    )


def test_failure_replay_path_is_bound_before_environment_creation(contracts, tmp_path):
    failure_pool = tmp_path / "failure.npz"
    failure_pool.write_bytes(b"pool")

    class Reset:
        params = {"failure_state_path": ""}

    class Events:
        reset_fall_pose = Reset()

    class Config:
        events = Events()

    task = "RobotLab-Isaac-Recovery-FailureReplay-Residual-Parallelogram-Rover-v0"
    selected = contracts.configure_failure_replay_env(Config(), task, failure_pool)
    assert selected == failure_pool.resolve()
    assert Config.events.reset_fall_pose.params["failure_state_path"] == str(selected)


def test_failure_replay_path_rejects_missing_and_conflicting_sources(contracts, tmp_path):
    class Reset:
        params = {"failure_state_path": ""}

    class Events:
        reset_fall_pose = Reset()

    class Config:
        events = Events()

    task = "RobotLab-Isaac-Recovery-FailureReplay-Parallelogram-Rover-v0"
    with pytest.raises(ValueError, match="失败状态池"):
        contracts.configure_failure_replay_env(Config(), task)
    configured = tmp_path / "configured.npz"
    configured.write_bytes(b"pool")
    Config.events.reset_fall_pose.params["failure_state_path"] = str(configured)
    with pytest.raises(ValueError, match="路径冲突"):
        contracts.configure_failure_replay_env(Config(), task, tmp_path / "other.npz")
