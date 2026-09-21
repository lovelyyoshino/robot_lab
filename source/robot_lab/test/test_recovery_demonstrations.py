"""恢复示范必须是动作前观测、真实执行动作及完整严格成功回合。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 示范动作重建及终态保持时钟必须匹配来源程序。
    - v0.1.0 (2026-09-17): 验证异步接管、reset前确认、失败排除、来源与时序防串帧。
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[3]
PATH = ROOT / "scripts/reinforcement_learning/rsl_rl/recovery_demonstrations.py"
spec = importlib.util.spec_from_file_location("recovery_demonstrations", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def contract():
    return dict(actor_observation_dim=58, critic_observation_dim=70, action_dim=12, step_dt=.02,
                observation_order=module.OBSERVATION_ORDER, observation_term_dims=[[n] for n in module.OBSERVATION_DIMS],
                action_joint_names=[f"joint_{i}" for i in range(12)], policy_config={}, wrapper_raw_clip=None,
                observation_group={}, action_config={}, default_joint_state={}, action_scales=[.25]*8+[.8]*4,
                action_offsets=[0.]*12)


def state(n=2):
    root = np.zeros((n, 13), dtype=np.float32)
    root[:, 3] = 1
    return {"root_state_w": root, "joint_pos": np.zeros((n, 20), np.float32),
            "joint_vel": np.zeros((n, 20), np.float32)}


def record(i, success):
    initial = state(1)
    return dict(env_id=i, episode_index=0, success=success, duration_s=5., preparation_time_s=.3,
                initialization_failed=False, terminated=False, truncated=True, nonfinite_seen=False,
                takeover_valid=True, takeover_root_state_w=initial["root_state_w"][0].tolist(),
                takeover_joint_pos=initial["joint_pos"][0].tolist(), takeover_joint_vel=initial["joint_vel"][0].tolist(),
                terminal_stable_time_s=1.02 if success else 0., time_to_stable_s=4. if success else None)


def result(records):
    scales = np.asarray([.25]*8+[.8]*4, dtype=np.float32)
    program = {"joint_names": contract()["action_joint_names"], "step_dt": .02,
               "segment_steps": 25, "active_steps": 150,
               "targets": np.broadcast_to(scales * .2, (2, 6, 12)).tolist(),
               "default_targets": np.zeros((2, 12)).tolist(), "action_scales": np.broadcast_to(scales, (2, 12)).tolist()}
    return dict(complete=True, episodes=2, expected_episodes=2, episodes_by_env=[1, 1], episode_results=records,
                action_program=program,
                task="RobotLab-Isaac-Recovery-Grounded-Parallelogram-Rover-v0", skill="recovery", seed=742,
                action_source="segmented_joint_targets_with_active_pd", checkpoint_policy_executed=False,
                acceptance_eligible=False, passed=False, recovery_support_protocol="four_wheel_stable_v1",
                protocol="settled_ground", step_dt=.02, episode_length_s=5.,
                evaluation_scope="bounded_action_program_search_not_policy_acceptance",
                checkpoint_sha256="a"*64, gates={"minimum_success_rate": .85,
                "maximum_time_to_stable_s": 4., "minimum_terminal_stable_time_s": .5})


def make_collected():
    collector = module.RecoveryDemonstrationCollector(2, 250, contract())
    physical = state()
    obs, actions = np.zeros((2, 58), np.float32), np.zeros((2, 12), np.float32)
    # 第一个环境准备期间，另一个也不能制造控制样本。
    collector.before_step(obs, actions, [0, 0], [False, False], **physical)
    collector.confirm_step([0, 0], [False, False], actions, **physical)
    for step in range(250):
        actions[:] = .2 if step < 150 else 0
        collector.before_step(obs, actions, [step, step], [True, True], **physical)
        collector.confirm_step([step+1, step+1], [True, True], actions, **physical)
        obs[:, -12:] = actions
    records = [record(0, True), record(1, False)]
    for item in records:
        collector.finish_episode(item)
    return collector, result(records)


def write_data(tmp_path):
    collector, evaluation = make_collected()
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps(evaluation))
    path = tmp_path / "demonstrations.npz"
    info = collector.save(path, source, evaluation)
    return path, source, info


def test_success_only_roundtrip_preserves_action_before_obs_and_copies_buffers(tmp_path):
    path, source, info = write_data(tmp_path)
    arrays, metadata = module.load_demonstrations(path)
    assert info["samples"] == 250 and info["successful_episodes"] == 1
    assert arrays["observations"].shape == (250, 58)
    np.testing.assert_array_equal(arrays["control_step"], np.arange(250))
    np.testing.assert_array_equal(arrays["observations"][1:, -12:], arrays["actions"][:-1])
    assert not arrays["observations"][0, -12:].any()
    assert arrays["actions"][149, 0] == pytest.approx(.2)
    assert arrays["actions"][150, 0] == 0
    assert metadata["episodes"][0]["program_id"] == 0
    assert metadata["source_evaluation_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        assert archive["metadata"].dtype.kind == "U"


def test_capture_requires_post_step_clock_and_real_action_confirmation():
    for bad_clock, bad_action in [([0, 1], False), ([1, 1], True)]:
        collector = module.RecoveryDemonstrationCollector(2, 250, contract())
        zeros = np.zeros((2, 12), np.float32)
        collector.before_step(np.zeros((2, 58), np.float32), zeros, [0, 0], [True, True], **state())
        with pytest.raises(ValueError):
            collector.confirm_step(bad_clock, [True, True], zeros+int(bad_action), **state())


def test_missing_confirmation_or_skipped_step_cannot_be_silently_accepted():
    collector = module.RecoveryDemonstrationCollector(2, 250, contract())
    obs, actions = np.zeros((2, 58), np.float32), np.zeros((2, 12), np.float32)
    with pytest.raises(ValueError, match="连续"):
        collector.before_step(obs, actions, [2, 0], [True, False], **state())
    collector.before_step(obs, actions, [0, 0], [True, False], **state())
    with pytest.raises(ValueError, match="确认"):
        collector.before_step(obs, actions, [1, 0], [True, False], **state())


def test_source_changes_overwrite_and_false_success_rejected(tmp_path):
    path, source, _ = write_data(tmp_path)
    collector, evaluation = make_collected()
    with pytest.raises(FileExistsError):
        collector.save(path, source, evaluation)
    source.write_text("{}")
    with pytest.raises(ValueError, match="来源"):
        module.load_demonstrations(path)
    collector, evaluation = make_collected()
    evaluation["episode_results"][0]["terminated"] = True
    with pytest.raises(ValueError):
        collector.arrays(evaluation)


@pytest.mark.parametrize("key", ["observations", "actions", "control_step", "episode_id"])
def test_misaligned_or_nonfinite_arrays_rejected(tmp_path, key):
    path, _, _ = write_data(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    if key in ("control_step", "episode_id"):
        payload[key][10] = 999
    else:
        payload[key][10, 0] = np.nan
    corrupt = tmp_path / f"bad_{key}.npz"
    np.savez_compressed(corrupt, **payload)
    with pytest.raises(ValueError):
        module.load_demonstrations(corrupt)


def test_contract_canonicalization_rejects_nan_and_preserves_slice():
    assert module.canonical_config({"x": (slice(None, 4, 2),)}) == {"x": [{"__slice__": [None, 4, 2]}]}
    with pytest.raises(ValueError):
        module.canonical_config({"x": float("nan")})
    invalid = copy.deepcopy(contract())
    invalid["observation_order"] = list(reversed(invalid["observation_order"]))
    with pytest.raises(ValueError):
        module.RecoveryDemonstrationCollector(2, 250, invalid)


def test_actions_and_history_cannot_be_changed_together_to_fake_a_program(tmp_path):
    path, _, _ = write_data(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["actions"][10] += 1
    payload["observations"][11, -12:] += 1
    altered = tmp_path / "altered_actions.npz"
    np.savez_compressed(altered, **payload)
    with pytest.raises(ValueError, match="来源分段程序"):
        module.load_demonstrations(altered)
