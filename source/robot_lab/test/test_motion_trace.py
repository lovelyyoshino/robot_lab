"""运动诊断逐步记录的时间边界与坐标合同回归，不启动仿真。

@author pony
@date 2026-09-17
@version v0.5.0
@last_modified 2026-09-17
@changelog
    - v0.5.0 (2026-09-17): 全连杆位姿按名字记录、四元数转换及损坏输入拒绝。
    - v0.4.0 (2026-09-17): 子步缓存与PhysX直接查询对照，区分过期采样和求解器差异。
    - v0.3.0 (2026-09-17): 子步元数据保存PhysX真实逐关节速度限制，避免只记录配置意图。
    - v0.2.0 (2026-09-17): 验证子步驱动/状态采样时序、只读接触查询和异常后钩子恢复。
    - v0.1.0 (2026-09-17): 覆盖动作前/自动reset前快照、首回合隔离及原始证据防覆盖。
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def trace_module():
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/motion_trace.py"
    spec = importlib.util.spec_from_file_location("motion_trace_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    class Scene(dict):
        pass

    root = torch.zeros(2, 13)
    root[:, 3] = 1
    data = SimpleNamespace(
        root_link_state_w=root.clone(), root_com_state_w=root.clone(),
        joint_pos=torch.zeros(2, 20), joint_vel=torch.zeros(2, 20), default_joint_pos=torch.zeros(2, 20),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
    )
    poses = torch.zeros(2, 21, 7)
    poses[:, :, 6] = 1
    robot = SimpleNamespace(data=data, joint_names=[f"joint_{i}" for i in range(20)],
                            body_names=["base_link"] + [f"body_{i}" for i in range(20)],
                            root_physx_view=SimpleNamespace(get_link_transforms=lambda: poses.clone()))
    scene = Scene(robot=robot)
    scene.env_origins = torch.tensor([[0., 0., 0.], [2., 3., 0.]])
    return SimpleNamespace(scene=scene, num_envs=2, episode_length_buf=torch.zeros(2, dtype=torch.long),
                           action_manager=SimpleNamespace(action=torch.zeros(2, 12)))


def append_step(recorder, env, terminated=False):
    recorder.begin(env, torch.zeros(2, 44), torch.ones(2, 12))
    env.episode_length_buf += 1
    env.scene["robot"].data.root_link_state_w[:, 0] += .1
    env.action_manager.action.fill_(.7)
    recorder.capture_post(env)
    recorder.finish(terminated, False)


def test_trace_keeps_postphysics_before_reset_and_stops_first_episode(trace_module, tmp_path):
    recorder = trace_module.MotionTraceRecorder(1, 12)
    env = environment()
    recorder.begin(env, torch.arange(88.).reshape(2, 44), torch.ones(2, 12))
    env.episode_length_buf += 1
    env.scene["robot"].data.root_link_state_w[1, 2] = .25
    env.scene["robot"].data.root_com_state_w[1, 2] = .23
    env.action_manager.action.fill_(.5)
    recorder.capture_post(env)
    env.scene["robot"].data.root_link_state_w[1, 2] = 999
    env.episode_length_buf[1] = 0
    recorder.finish(True, False)
    recorder.begin(env, torch.zeros(2, 44), torch.zeros(2, 12))
    recorder.capture_post(env)
    recorder.finish(False, False)
    path = tmp_path / "trace.json"
    recorder.save(path, {"control_dt_s": .02})
    payload = json.loads(path.read_text())
    assert payload["trace_contract"] == "rover_motion_step_trace_v1"
    assert payload["stop_reason"] == "first_episode_ended" and len(payload["steps"]) == 1
    step = payload["steps"][0]
    assert step["pre"]["episode_step"] == 0 and step["post"]["episode_step"] == 1
    assert step["post"]["root_link_state_w"][2] == .25
    assert step["post"]["root_com_state_w"][2] == pytest.approx(.23)
    assert step["pre"]["env_origin_w"] == [2., 3., 0.]
    assert step["policy_observation"] == list(range(44, 88))
    assert step["policy_action"] == [1.] * 12 and step["applied_action"] == [.5] * 12
    assert payload["contains_contact_solver_state"] is False
    with pytest.raises(FileExistsError):
        recorder.save(path, {})


def test_trace_records_only_requested_prefix_and_never_changes_state(trace_module):
    recorder, env = trace_module.MotionTraceRecorder(0, 2), environment()
    append_step(recorder, env)
    append_step(recorder, env)
    append_step(recorder, env)
    assert len(recorder.steps) == 2 and recorder.stop_reason == "requested_prefix_complete"
    assert env.scene["robot"].data.root_link_state_w[0, 0] == pytest.approx(.3)


@pytest.mark.parametrize("fault", ["wrong_obs", "nan_action", "duplicate_joints", "nan_state", "wrong_step"])
def test_trace_rejects_unreplayable_inputs(trace_module, fault):
    recorder, env = trace_module.MotionTraceRecorder(0, 2), environment()
    obs, action = torch.zeros(2, 44), torch.zeros(2, 12)
    if fault == "wrong_obs":
        obs = torch.zeros(2, 42)
    elif fault == "nan_action":
        action[0, 0] = float("nan")
    elif fault == "duplicate_joints":
        env.scene["robot"].joint_names[1] = "joint_0"
    elif fault == "nan_state":
        env.scene["robot"].data.root_link_state_w[0, 0] = float("inf")
    else:
        env.episode_length_buf[0] = 1
    with pytest.raises(ValueError):
        recorder.begin(env, obs, action)


def test_trace_requires_post_snapshot_before_finish_and_no_pending_save(trace_module, tmp_path):
    recorder, env = trace_module.MotionTraceRecorder(0, 2), environment()
    recorder.begin(env, torch.zeros(2, 44), torch.zeros(2, 12))
    with pytest.raises(RuntimeError):
        recorder.finish(False, False)
    with pytest.raises(RuntimeError):
        recorder.save(tmp_path / "incomplete.json", {})


def test_production_trace_is_hooked_before_action_and_before_reset():
    path = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_motion.py"
    source = path.read_text()
    assert source.index("trace.begin(") < source.index("obs, _, dones, _ = env.step(actions)")
    terminal = source[source.index("def capture_terminal_state(env):"):source.index("env_cfg.terminations.motion_snapshot")]
    assert "trace.capture_post(env)" in terminal
    assert "trace.finish(" in source


def physics_environment():
    env = environment()
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005), decimation=4)
    robot = env.scene["robot"]
    for key in ("computed_torque", "applied_torque"):
        setattr(robot.data, key, torch.zeros(2, 20))
    for key in ("_joint_pos_target_sim", "_joint_vel_target_sim", "_joint_effort_target_sim"):
        setattr(robot, key, torch.zeros(2, 20))
    robot.actuators = {"legs": SimpleNamespace(joint_indices=list(range(16)), is_implicit_model=True),
                       "wheels": SimpleNamespace(joint_indices=list(range(16, 20)), is_implicit_model=False)}
    robot.root_physx_view = SimpleNamespace(
        get_link_transforms=robot.root_physx_view.get_link_transforms,
        get_dof_positions=lambda: robot.data.joint_pos.clone(),
        get_dof_velocities=lambda: robot.data.joint_vel.clone(),
        get_dof_armatures=lambda: torch.zeros(2, 20),
        get_dof_max_velocities=lambda: torch.full((2, 20), 5.),
        get_dof_projected_joint_forces=lambda: torch.full((2, 20), 7.),
    )
    sensor = SimpleNamespace(body_names=["base_link", "rl_wheel"], contact_physx_view=SimpleNamespace(
        get_net_contact_forces=lambda dt: torch.full((4, 3), dt * 100),
    ))
    env.scene["contact_forces"] = sensor
    env.scene.write_data_to_sim = lambda: robot.data.applied_torque.fill_(3.)

    def update(dt):
        robot.data.joint_pos.add_(dt)

    env.scene.update = update
    return env


def test_trace_link_poses_are_named_world_frames_and_do_not_alias_backend(trace_module):
    env, recorder = environment(), trace_module.MotionTraceRecorder(1, 1)
    poses = env.scene["robot"].root_physx_view.get_link_transforms()
    poses[1, 4] = torch.tensor([2., 3., .7, .6, 0., 0., .8])
    env.scene["robot"].root_physx_view.get_link_transforms = lambda: poses
    recorder.begin(env, torch.zeros(2, 44), torch.zeros(2, 12))
    state = recorder.pending["pre"]
    assert state["body_link_pose_contract"] == "world_xyz_wxyz_link_v1"
    assert state["body_names"][4] == "body_3"
    assert state["body_link_pose_w"][4] == pytest.approx([2., 3., .7, .8, .6, 0., 0.])
    assert poses[1, 4].tolist() == pytest.approx([2., 3., .7, .6, 0., 0., .8])
    poses[1, 4, 0] = 999
    assert state["body_link_pose_w"][4][0] == 2.


@pytest.mark.parametrize("fault", ["duplicate", "missing", "shape", "nan", "quaternion"])
def test_trace_rejects_unusable_body_fk(trace_module, fault):
    env, recorder = environment(), trace_module.MotionTraceRecorder(0, 1)
    robot = env.scene["robot"]
    poses = robot.root_physx_view.get_link_transforms()
    if fault == "duplicate":
        robot.body_names[1] = "base_link"
    elif fault == "missing":
        robot.body_names[0] = "not_base_link"
    elif fault == "shape":
        poses = poses[:, :20]
    elif fault == "nan":
        poses[0, 0, 0] = float("nan")
    else:
        poses[0, 0, 6] = 2.
    robot.root_physx_view.get_link_transforms = lambda: poses
    with pytest.raises(ValueError):
        recorder.begin(env, torch.zeros(2, 44), torch.zeros(2, 12))


def test_physics_hooks_capture_each_substep_and_restore_on_error(trace_module, tmp_path):
    env, recorder = physics_environment(), trace_module.MotionTraceRecorder(1, 1)
    original_write, original_update = env.scene.write_data_to_sim, env.scene.update
    with pytest.raises(RuntimeError, match="caller error"):
        with recorder.physics_hooks(env):
            recorder.begin(env, torch.zeros(2, 44), torch.ones(2, 12))
            for _ in range(4):
                env.scene.write_data_to_sim()
                env.scene.update(.005)
            env.episode_length_buf += 1
            recorder.capture_post(env)
            recorder.finish(False, False)
            raise RuntimeError("caller error")
    assert env.scene.write_data_to_sim is original_write and env.scene.update is original_update
    recorder.save(tmp_path / "physics.json", {})
    payload = json.loads((tmp_path / "physics.json").read_text())
    assert payload["physics_trace"]["contract"] == "rover_motion_physics_trace_v1"
    assert payload["physics_trace"]["implicit_joint_names"] == [f"joint_{i}" for i in range(16)]
    assert payload["physics_trace"]["joint_velocity_limit_sim_rad_s"] == [5.] * 20
    samples = payload["steps"][0]["physics_substeps"]
    assert len(samples) == 4
    assert samples[0]["pre"]["joint_pos"] == [0.] * 20
    assert samples[0]["post"]["joint_pos"] == pytest.approx([.005] * 20)
    assert samples[-1]["post"]["joint_pos"] == pytest.approx([.02] * 20)
    assert samples[0]["drive"]["actuator_model_clipped_torque_nm"] == [3.] * 20
    assert samples[0]["contact"]["net_normal_force_world_n"] == [[.5] * 3] * 2
    assert samples[0]["projected_incoming_joint_force_nm"] == [7.] * 20
    assert samples[0]["joint_cache_max_abs_error"] == {"position_rad": 0., "velocity_rad_s": 0.}
    # 不能把隐式执行器的近似PD输出伪装成求解器实测电机力矩。
    assert payload["physics_trace"]["implicit_drive_force_measured"] is False


def test_physics_trace_detects_stale_joint_cache_without_changing_it(trace_module):
    env, recorder = physics_environment(), trace_module.MotionTraceRecorder(0, 1)
    robot = env.scene["robot"]
    robot.root_physx_view.get_dof_positions = lambda: robot.data.joint_pos + .1
    robot.root_physx_view.get_dof_velocities = lambda: robot.data.joint_vel + .2
    with recorder.physics_hooks(env):
        recorder.begin(env, torch.zeros(2, 44), torch.zeros(2, 12))
        env.scene.write_data_to_sim()
        env.scene.update(.005)
        errors = recorder.pending["physics_substeps"][0]["joint_cache_max_abs_error"]
        assert errors["position_rad"] == pytest.approx(.1)
        assert errors["velocity_rad_s"] == pytest.approx(.2)
        assert robot.data.joint_pos[0, 0] == pytest.approx(.005)


def test_physics_trace_rejects_missing_substep_or_wrong_dt(trace_module):
    for fault in ("missing", "wrong_dt", "duplicate_write"):
        env, recorder = physics_environment(), trace_module.MotionTraceRecorder(0, 1)
        with recorder.physics_hooks(env):
            recorder.begin(env, torch.zeros(2, 44), torch.zeros(2, 12))
            env.scene.write_data_to_sim()
            with pytest.raises((ValueError, RuntimeError)):
                if fault == "wrong_dt":
                    env.scene.update(.01)
                elif fault == "duplicate_write":
                    env.scene.write_data_to_sim()
                else:
                    env.scene.update(.005)
                    env.episode_length_buf += 1
                    recorder.capture_post(env)
