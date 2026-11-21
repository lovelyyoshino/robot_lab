"""深度学生的预抬蒸馏、输入合同和训练恢复回归；使用真实PyTorch网络，不启动Isaac。

@author pony
@date 2026-11-03
@version v0.1.0
@last_modified 2026-11-03
@changelog
    - v0.1.0 (2026-11-03): 覆盖深度动作梯度、边缘/摆臂权重、冻结教师、标签快照和恢复保护。
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# 仅导入纯torch包，避免robot_lab顶层初始化引入仿真依赖。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot_lab"))
from supermimic import scan_student as ss
from supermimic.networks import Actor


class FixedScanEncoder(torch.nn.Module):
    def __init__(self, scan):
        super().__init__()
        self.scan = torch.nn.Parameter(scan.clone())

    def forward(self, obs):
        return self.scan.expand(obs.shape[0], -1)


def scan_teacher(lane_obs=False):
    teacher = torch.nn.Linear(ss.TEACHER_OBS_DIM + int(lane_obs), 12, bias=False)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.weight[:, ss.PROPRIO_DIM:ss.PROPRIO_DIM + 12] = torch.eye(12)
    return teacher


def test_real_depth_encoder_changes_actions_and_preserves_scripted_inference(tmp_path):
    torch.manual_seed(19)
    policy = ss.ScanStudentPolicy(ss.DepthScanEncoder(8, (16,)), scan_teacher(True), lane_obs=True)
    observation = torch.zeros(2, ss.STUDENT_OBS_DIM + 1)
    observation[:, -1] = 0.4
    changed = observation.clone()
    changed[:, ss.PROPRIO_DIM:ss.STUDENT_OBS_DIM] = 0.8
    assert (policy(changed) - policy(observation)).abs().max() > 1e-5
    observation.requires_grad_(True)
    gradient = torch.autograd.grad(policy(observation).sum(), observation)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, ss.PROPRIO_DIM:ss.STUDENT_OBS_DIM].abs().sum() > 0
    assert gradient[:, -1].count_nonzero() == 0
    scripted = torch.jit.script(policy.eval())
    scripted.save(str(tmp_path / "student.pt"))
    deployed = torch.jit.load(str(tmp_path / "student.pt"))
    assert torch.equal(deployed(changed), policy(changed))
    assert torch.isfinite(deployed(changed)).all()
    with pytest.raises((ValueError, torch.jit.Error), match="输入维度"):
        deployed(changed[:, :-1])


def test_edges_weight_both_sides_and_respect_lateral_step_boundary():
    cfg = ss.ScanStudentCfg(scan_step_weight=2.0, scan_edge_weight=7.0)
    scan = torch.zeros(1, 11, 17)
    scan[:, :, 8:] = -0.12
    weights = ss.scan_loss_weights(scan.flatten(1), cfg).reshape_as(scan)
    assert (weights[:, :, 7:9] == 7).all()
    assert (weights[:, :, :7] == 1).all()
    assert (weights[:, :, 9:] == 2).all()
    side_scan = torch.zeros_like(scan)
    side_scan[:, 5:, :] = -0.12
    edges = ss.scan_edge_mask(side_scan.flatten(1), cfg).reshape_as(scan)
    assert edges[:, 4:6].all()
    assert edges[:, :4].count_nonzero() == edges[:, 6:].count_nonzero() == 0
    assert torch.equal(ss.scan_loss_weights(torch.zeros(2, 187), cfg), torch.ones(2, 187))


def test_edge_and_arm_errors_receive_the_requested_gradient_weight():
    target = torch.zeros(1, 11, 17)
    target[:, :, 8:] = -0.12
    encoder = FixedScanEncoder(target.flatten(1)[0] + 0.1)
    policy = ss.ScanStudentPolicy(encoder, scan_teacher())
    student = torch.zeros(1, ss.STUDENT_OBS_DIM)
    teacher = torch.cat((student[:, :45], target.flatten(1)), -1)
    cfg = ss.ScanStudentCfg(action_loss_coef=0, scan_edge_weight=7)
    _, scan_loss, _ = ss.student_losses(policy, student, teacher, cfg)
    gradient = torch.autograd.grad(scan_loss, encoder.scan)[0].reshape(11, 17)
    assert gradient[0, 7] / gradient[0, 0] == pytest.approx(7.0)
    assert gradient[0, 8] / gradient[0, 0] == pytest.approx(7.0, rel=1e-5)

    with torch.no_grad():
        encoder.scan.zero_()
        encoder.scan[[0, 8]] = 0.1
    _, _, action_loss = ss.student_losses(policy, student, torch.zeros(1, 232),
                                         ss.ScanStudentCfg(arm_action_weight=5))
    gradient = torch.autograd.grad(action_loss, encoder.scan)[0]
    assert gradient[0] / gradient[8] == pytest.approx(5.0)


def test_legacy_loss_matches_unweighted_mse_and_teacher_stays_frozen():
    torch.manual_seed(7)
    encoder = ss.DepthScanEncoder(8, (16,))
    teacher = scan_teacher()
    policy = ss.ScanStudentPolicy(encoder, teacher)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    observations = torch.randn(3, ss.STUDENT_OBS_DIM) * 0.2
    labels = torch.randn(3, ss.TEACHER_OBS_DIM) * 0.2
    cfg = ss.ScanStudentCfg()
    _, scan_loss, action_loss = ss.student_losses(policy, observations, labels, cfg)
    prediction = encoder(observations)
    assert torch.allclose(scan_loss, (prediction - labels[:, 45:]).square().mean())
    expected_action = teacher(policy.actor_observation(observations, prediction))
    assert torch.allclose(action_loss, (expected_action - teacher(labels)).square().mean())
    before = copy.deepcopy(teacher.state_dict())
    encoder_before = [parameter.clone() for parameter in encoder.parameters()]
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    ss.train_student_batch(policy, optimizer, observations, labels, cfg)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(torch.equal(value, before[name]) for name, value in teacher.state_dict().items())
    assert any(not torch.equal(value, old) for value, old in zip(encoder.parameters(), encoder_before))


def test_nonfinite_loss_and_gradient_never_reach_optimizer():
    encoder = FixedScanEncoder(torch.zeros(187))
    policy = ss.ScanStudentPolicy(encoder, scan_teacher())
    optimizer = torch.optim.Adam(encoder.parameters())
    student, teacher = torch.zeros(1, 4141), torch.zeros(1, 232)
    invalid = teacher.clone()
    invalid[0, 45] = torch.nan
    with pytest.raises(FloatingPointError, match="loss非有限"):
        ss.train_student_batch(policy, optimizer, student, invalid, ss.ScanStudentCfg())
    assert encoder.scan.count_nonzero() == 0 and not optimizer.state
    teacher[:, 45:] = 0.1
    hook = encoder.scan.register_hook(lambda gradient: gradient * float("inf"))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            ss.train_student_batch(policy, optimizer, student, teacher, ss.ScanStudentCfg())
    finally:
        hook.remove()
    assert encoder.scan.count_nonzero() == 0 and not optimizer.state


def test_collection_clones_observations_before_environment_reuses_buffers():
    observations = {"policy": torch.full((2, 4142), 0.1), "teacher": torch.full((2, 233), 0.2)}
    student, teacher = ss.clone_student_observations(observations)
    observations["policy"].fill_(0.8)
    observations["teacher"].fill_(0.9)
    assert (student == 0.1).all() and (teacher == 0.2).all()
    assert student.data_ptr() != observations["policy"].data_ptr()
    assert teacher.data_ptr() != observations["teacher"].data_ptr()


def fake_noise_environment():
    policy = SimpleNamespace(
        enable_corruption=True,
        base_ang_vel=SimpleNamespace(noise=object()),
        joint_pos=SimpleNamespace(noise=object()),
        depth_image=SimpleNamespace(noise=None, params={
            "history_stride": 5, "noise_half_range": 0.02, "drop_probability": 0.05,
            "freeze_probability": 0.01, "delay_steps": (0, 1),
        }),
    )
    return SimpleNamespace(observations=SimpleNamespace(policy=policy))


def test_clean_proprio_keeps_camera_perturbations_and_roundtrips_contract():
    cfg = fake_noise_environment()
    depth_before = copy.deepcopy(cfg.observations.policy.depth_image.params)
    noise = {"clean_proprio": True, "camera_corruption": True}
    ss.configure_student_observation_noise(cfg, **noise)
    assert cfg.observations.policy.base_ang_vel.noise is None
    assert cfg.observations.policy.joint_pos.noise is None
    assert cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.depth_image.params == depth_before
    base = {"student_obs_dim": 4141, "teacher_obs_dim": 232}
    contract = ss.student_depth_contract(base, False, noise)
    assert json.loads(json.dumps(contract))["observation_noise"] == noise
    assert "observation_noise" not in base
    ss.configure_student_observation_noise(cfg, camera_corruption=False)
    params = cfg.observations.policy.depth_image.params
    assert params["history_stride"] == 5
    assert params["delay_steps"] == (0, 0)
    assert params["noise_half_range"] == params["drop_probability"] == params["freeze_probability"] == 0


def test_resume_restores_loss_configuration_and_explicit_lr_keeps_adam_momentum():
    saved_cfg = asdict(ss.ScanStudentCfg(lr=1e-4, scan_step_weight=3, scan_edge_weight=8,
                                       arm_action_weight=4, epochs=2))
    restored = ss.resolve_student_config(saved_cfg)
    assert asdict(restored) == saved_cfg
    changed = ss.resolve_student_config(saved_cfg, lr=2e-5, action_loss_coef=None)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    old = torch.optim.Adam([parameter], lr=saved_cfg["lr"])
    parameter.square().backward()
    old.step()
    new = torch.optim.Adam([parameter], lr=changed.lr)
    ss.restore_student_optimizer(new, copy.deepcopy(old.state_dict()), changed)
    assert new.param_groups[0]["lr"] == changed.lr
    assert torch.equal(new.state[parameter]["exp_avg"], old.state[parameter]["exp_avg"])
    assert changed.scan_edge_weight == 8 and changed.arm_action_weight == 4
    legacy = ss.resolve_student_config({"lr": 1e-4, "epochs": 2})
    assert legacy.scan_edge_weight == legacy.arm_action_weight == 1


def test_deployment_restores_saved_sensor_noise_contract(tmp_path, monkeypatch):
    cfg = fake_noise_environment()
    noise = {"clean_proprio": True, "camera_corruption": False}
    contract = ss.student_depth_contract({"student_obs_dim": 4141, "teacher_obs_dim": 232}, False, noise)
    (tmp_path / "depth_contract.json").write_text(json.dumps(contract))
    module_name = ("robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled."
                   "parallelogram_rover.vision_env_cfg")
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(
        vision_depth_contract=lambda config: {"student_obs_dim": 4141, "teacher_obs_dim": 232},
    ))
    restored = ss.configure_saved_student(cfg, tmp_path / "exported/policy.pt")
    assert restored == contract
    assert cfg.observations.policy.base_ang_vel.noise is None
    assert cfg.observations.policy.depth_image.params["drop_probability"] == 0


def test_existing_run_is_never_overwritten(tmp_path):
    path = ss.create_student_run_directory(tmp_path, "existing")
    marker = path / "depth_contract.json"
    marker.write_text("preserved")
    with pytest.raises(FileExistsError):
        ss.create_student_run_directory(tmp_path, "existing")
    assert marker.read_text() == "preserved"
    with pytest.raises(ValueError):
        ss.create_student_run_directory(tmp_path, "../outside")


@pytest.mark.parametrize("kwargs", [
    {"lr": float("nan")}, {"scan_edge_weight": -1}, {"arm_action_weight": float("inf")},
    {"scan_grid_shape": (-11, -17)}, {"teacher_drive_start": 1.1}, {"minibatch": 0},
])
def test_invalid_training_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ss.ScanStudentCfg(**kwargs)


def test_training_main_uses_cloned_labels_and_exports_effective_configuration(tmp_path, monkeypatch):
    """执行真实main与真实编码器更新，仅用会原地复用buffer的最小环境替代渲染/物理。"""
    import csv

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    teacher = Actor(232, (8,), squash="tanh")
    teacher_path = tmp_path / "teacher" / "ckpt.pt"
    teacher_path.parent.mkdir()
    (teacher_path.parent / "params").mkdir()
    torch.save({"actor": teacher.state_dict(), "world_model": {}, "cfg": {
        "actor_hidden": (8,), "action_bound": teacher.action_bound, "action_squash": "tanh",
    }}, teacher_path)
    (teacher_path.parent / "params/vision_contract.json").write_text("{}")

    class FakeEnvironment:
        def __init__(self):
            self.unwrapped = self
            self.device = "cpu"
            self.buffer = {"policy": torch.zeros(1, 4141), "teacher": torch.zeros(1, 232)}
            self.steps = 0

        def reset(self):
            return self.buffer, {}

        def step(self, action):
            assert torch.isfinite(action).all()
            self.steps += 1
            self.buffer["policy"][:, 45:] = 0.1 * self.steps
            self.buffer["teacher"][:, 45:] = -0.1 * self.steps
            return self.buffer, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

        def close(self):
            pass

    class FakeWriter:
        def __init__(self, *args):
            pass

        def add_scalar(self, *args):
            pass

        def close(self):
            pass

    config = fake_noise_environment()
    config.to_dict = lambda: {"fake_environment": True}
    config.seed = 0
    config.scene = SimpleNamespace(terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={})))
    calls = []
    original_update = ss.train_student_batch

    def observed_update(policy, optimizer, student_obs, teacher_obs, cfg):
        calls.extend(zip(student_obs[:, 45].tolist(), teacher_obs[:, 45].tolist()))
        return original_update(policy, optimizer, student_obs, teacher_obs, cfg)

    monkeypatch.setattr(ss, "train_student_batch", observed_update)
    def deployment_configuration(*args, **kwargs):
        config.prelift = False
        config.scene.terrain.terrain_generator.sub_terrains = {"flat": SimpleNamespace(proportion=1.0)}
        return config

    def validate_deployment(source, target, **kwargs):
        assert target["prelift"] is False

    def configure_prelift(config, student=False):
        assert student is True
        config.prelift = True
        config.scene.terrain.terrain_generator.sub_terrains = {"single_step": SimpleNamespace(proportion=1.0)}

    monkeypatch.setitem(sys.modules, "vision_contract", SimpleNamespace(
        contract_for_cfg=lambda task, config, agent: {"prelift": config.prelift},
        restore_height_scan_period=lambda *args: None, validate_contract=validate_deployment,
    ))
    monkeypatch.setitem(sys.modules, "robot_lab.tasks.manager_based.supermimic.prelift_env_cfg", SimpleNamespace(
        configure_prelift_training=configure_prelift,
    ))
    script = Path(__file__).resolve().parents[3] / "scripts/supermimic/train_scan_student.py"
    tree = ast.parse(script.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    arguments = dict(
        seed=42, resume=None, latent_dim=8, prelift_training=True, clean_proprio=True,
        camera_corruption=True, iterations=1, steps_per_iter=2, save_interval=1, log_interval=1,
        run_name="fake", teacher=str(teacher_path), task="fake", device="cpu", num_envs=1,
    )
    arguments.update({name: None for name in asdict(ss.ScanStudentCfg()) if name not in arguments})
    arguments.update(epochs=1, minibatch=2, scan_step_weight=2.0, scan_edge_weight=6.0, arm_action_weight=4.0)
    scope = {
        "__file__": str(script), "torch": torch, "ss": ss, "args_cli": SimpleNamespace(**arguments),
        "Path": Path, "datetime": datetime, "asdict": asdict, "fields": fields,
        "sys": sys, "hashlib": hashlib, "json": json, "copy": copy, "csv": csv,
        "parse_env_cfg": deployment_configuration,
        "load_cfg_from_registry": lambda *args: None,
        "vision_depth_contract": lambda cfg: {"student_obs_dim": 4141, "teacher_obs_dim": 232},
        "gym": SimpleNamespace(make=lambda *args, **kwargs: FakeEnvironment()), "SummaryWriter": FakeWriter,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(script), "exec"), scope)
    scope["main"]()
    assert len(calls) == 2
    assert sorted(round(value[0], 4) for value in calls) == [0.0, 0.1]
    assert all(student == pytest.approx(-label) for student, label in calls)
    run = tmp_path / "logs/supermimic_scan/fake"
    checkpoint = torch.load(run / "ckpt_000001.pt", weights_only=False)
    assert checkpoint["teacher_sha256"] == hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    assert checkpoint["cfg"]["scan_edge_weight"] == 6.0
    assert checkpoint["training_terrain_proportions"] == {"single_step": 1.0}
    assert set(checkpoint["implementation_sha256"]) == {
        "scan_student.py", "train_scan_student.py", "prelift_env_cfg.py",
    }
    assert checkpoint["observation_noise"]["clean_proprio"] is True
    assert checkpoint["depth_contract"]["observation_noise"]["camera_corruption"] is True
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == checkpoint["cfg"]["lr"]
    assert (run / "params/student_training.json").is_file()
    assert (run / "params/env.json").is_file()
    assert json.loads((run / "params/vision_contract.json").read_text())["prelift"] is False
    assert json.loads((run / "params/training_vision_contract.json").read_text())["prelift"] is True
    deployed = torch.jit.load(str(run / "exported/policy.pt"))
    assert torch.isfinite(deployed(torch.zeros(1, 4141))).all()
    assert all(torch.equal(value, teacher.state_dict()[name])
               for name, value in deployed.teacher_actor.state_dict().items())
    # 第二次执行main验证续训分支真正使用存档配置，并覆盖Adam恢复进来的旧lr。
    scope["args_cli"].resume = str(run / "ckpt_000001.pt")
    scope["args_cli"].run_name = "resumed"
    scope["args_cli"].iterations = 2
    scope["args_cli"].lr = 1e-5
    for name in ("scan_step_weight", "scan_edge_weight", "arm_action_weight", "clean_proprio"):
        setattr(scope["args_cli"], name, None)
    scope["main"]()
    resumed = torch.load(tmp_path / "logs/supermimic_scan/resumed/ckpt_000002.pt", weights_only=False)
    assert resumed["cfg"]["scan_edge_weight"] == 6.0
    assert resumed["cfg"]["arm_action_weight"] == 4.0
    assert resumed["cfg"]["lr"] == resumed["optimizer"]["param_groups"][0]["lr"] == 1e-5
    assert resumed["observation_noise"] == checkpoint["observation_noise"]
    scope["args_cli"].camera_corruption = False
    with pytest.raises(ValueError, match="扰动合同"):
        scope["main"]()
