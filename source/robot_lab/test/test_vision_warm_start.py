"""视觉 teacher warm start 的 CPU 合同测试，不以函数保持冒充通行成功。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 保存首次PPO更新前快照，不覆盖证据、不推进随机数或优化器。
    - v0.1.0 (2026-09-17): 验证 blind actor 扩维、critic 保持及显式安全迁移。
"""

from __future__ import annotations

import importlib
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import pytest
from torch import nn


SCRIPT_DIR = Path(__file__).parents[3] / "scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, str(SCRIPT_DIR))


class ActorCriticMLP(nn.Module):
    """真实执行与 RSL 相同的 Linear/ELU 网络，并保留 optimizer 引用的 Parameter。"""

    is_recurrent = False
    state_dependent_std = False
    actor_obs_normalization = False
    critic_obs_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    def __init__(self, actor_dim):
        super().__init__()
        activation = nn.ELU()
        self.actor = nn.Sequential(nn.Linear(actor_dim, 16), activation, nn.Linear(16, 8), activation, nn.Linear(8, 12))
        self.critic = nn.Sequential(nn.Linear(235, 16), activation, nn.Linear(16, 8), activation, nn.Linear(8, 1))
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.std = nn.Parameter(torch.ones(12))


def policy_config():
    return {"class_name": "ActorCritic", "activation": "elu", "noise_std_type": "scalar",
            "actor_obs_normalization": False, "critic_obs_normalization": False,
            "actor_hidden_dims": [16, 8], "critic_hidden_dims": [16, 8]}


def test_initialization_snapshot_is_evaluable_immutable_and_not_a_training_checkpoint(tmp_path):
    module = importlib.import_module("vision_warm_start")
    target = ActorCriticMLP(232)
    optimizer = torch.optim.Adam(target.parameters())
    runner = SimpleNamespace(alg=SimpleNamespace(policy=target, optimizer=optimizer), current_learning_iteration=0)
    rng = torch.get_rng_state().clone()
    metadata = module.save_teacher_initialization(runner, tmp_path)
    path = Path(metadata["checkpoint"])
    assert path.name == "teacher_initialization.pt" and not path.match("model_*.pt")
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    payload = torch.load(path, weights_only=True)
    assert payload["iter"] == 0 and payload["optimizer_state_dict"]["state"] == {}
    assert payload["infos"]["scope"] == "initialization_before_ppo_not_skill_acceptance"
    torch.testing.assert_close(payload["model_state_dict"]["actor.0.weight"], target.actor[0].weight)
    assert len(metadata["sha256"]) == 64
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.save_teacher_initialization(runner, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["iteration", "optimizer"])
def test_initialization_snapshot_refuses_already_trained_runner(tmp_path, fault):
    module = importlib.import_module("vision_warm_start")
    target = ActorCriticMLP(232)
    optimizer = torch.optim.Adam(target.parameters())
    runner = SimpleNamespace(alg=SimpleNamespace(policy=target, optimizer=optimizer), current_learning_iteration=0)
    if fault == "iteration":
        runner.current_learning_iteration = 1
    else:
        target.actor(torch.zeros(1, 232)).sum().backward()
        optimizer.step()
    with pytest.raises(ValueError):
        module.save_teacher_initialization(runner, tmp_path)
    assert not list(tmp_path.iterdir())


def test_blind_actor_function_is_preserved_for_arbitrary_height_scan_and_critic_is_unchanged():
    module = importlib.import_module("vision_warm_start")
    torch.manual_seed(142)
    old, target = ActorCriticMLP(45), ActorCriticMLP(232)
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
    parameters = tuple(target.parameters())
    source_before = {key: value.clone() for key, value in old.state_dict().items()}
    metadata = module.warm_start_vision_teacher(old.state_dict(), target, source_policy_cfg=policy_config())
    assert all(a is b for a, b in zip(parameters, target.parameters()))
    assert all(a is b for a, b in zip(parameters, optimizer.param_groups[0]["params"]))
    assert len(optimizer.state) == 0
    proprio, critic = torch.randn(13, 45), torch.randn(13, 235)
    for magnitude in (0., 1., 1e6):
        height = torch.randn(13, 187) * magnitude
        torch.testing.assert_close(target.actor(torch.cat((proprio, height), -1)), old.actor(proprio), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(target.critic(critic), old.critic(critic), rtol=0, atol=0)
    assert target.actor[0].weight[:, 45:].count_nonzero() == 0
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, source_before[name], rtol=0, atol=0)
    assert not metadata["optimizer_transferred"] and not metadata["iteration_transferred"]
    assert not metadata["skill_success_verified"]


def config_fixture():
    # 测试自含 schema，不要求另一台机器具有本机未跟踪的历史训练日志。
    prefix = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    def term(name, *, noise=None):
        return {"func": "isaaclab.envs.mdp.observations:" + name, "params": {}, "modifiers": None,
                "noise": noise, "clip": (-1., 1.), "scale": 1., "history_length": 0, "flatten_history_dim": True}
    policy = {"concatenate_terms": True, "concatenate_dim": -1, "enable_corruption": True,
              "history_length": None, "flatten_history_dim": True, "base_lin_vel": None}
    policy.update({name: term(name) for name in prefix})
    policy["joint_vel"]["scale"] = .05
    policy["base_ang_vel"]["noise"] = {"func": "isaaclab.utils.noise.noise_model:uniform_noise",
                                         "operation": "add", "n_min": -.2, "n_max": .2}
    policy["height_scan"] = None
    critic = {"concatenate_terms": True, "enable_corruption": False, "history_length": None,
              "base_lin_vel": term("base_lin_vel")}
    critic.update({name: term(name) for name in prefix})
    critic["base_lin_vel"]["scale"] = 2.
    critic["height_scan"] = term("height_scan")
    critic["height_scan"]["params"] = {"sensor_cfg": {"name": "height_scanner", "body_ids": slice(None)}}
    source = {"observations": {"policy": policy, "critic": critic},
              "actions": {"joint_pos": {"scale": .25}, "joint_vel": {"scale": 5.}},
              "scene": {"height_scanner": {"update_period": .02, "pattern_cfg": {"size": (1.6, 1.)}},
                        "robot": {"init_state": {"joint_pos": {".*": 0.}, "joint_vel": {".*": 0.}}}},
              "sim": {"dt": .005}, "decimation": 4}
    target = copy.deepcopy(source)
    height = copy.deepcopy(critic["height_scan"])
    height["noise"] = {"func": "isaaclab.utils.noise.noise_model:uniform_noise", "operation": "add",
                       "n_min": -.1, "n_max": .1}
    target["observations"]["policy"]["height_scan"] = height
    agent = {"class_name": "OnPolicyRunner", "algorithm": {"class_name": "PPO"}, "policy": policy_config(),
             "resume": False, "obs_groups": {}, "clip_actions": None, "empirical_normalization": None}
    return source, agent, target, copy.deepcopy(agent)


def active_contract():
    prefix = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    terms = {"policy": prefix + ["height_scan"], "critic": ["base_lin_vel"] + prefix + ["height_scan"]}
    dimensions = {"policy": [(3,), (3,), (3,), (12,), (12,), (12,), (187,)],
                  "critic": [(3,), (3,), (3,), (3,), (12,), (12,), (12,), (187,)]}
    return terms, dimensions


def test_compatible_blind_and_teacher_configs_have_exact_contract():
    module = importlib.import_module("vision_warm_start")
    source_env, source_agent, target_env, target_agent = config_fixture()
    terms, dimensions = active_contract()
    result = module.validate_blind_teacher_contract(source_env, target_env, source_agent, target_agent, terms, dimensions)
    assert result["configuration_contract_checked"]
    assert not result["source_code_behavior_equivalence_verified"]


@pytest.mark.parametrize("fault", ["scale", "noise", "prefix_order", "actions", "action_order", "joint_default",
    "normalization", "group_history", "critic", "height_sensor", "height_term", "active_dims", "active_order", "routing", "dt"])
def test_semantic_mismatch_is_rejected(fault):
    module = importlib.import_module("vision_warm_start")
    source_env, source_agent, target_env, target_agent = config_fixture()
    terms, dimensions = active_contract()
    policy = target_env["observations"]["policy"]
    if fault == "scale": policy["joint_vel"]["scale"] = 1.
    elif fault == "noise": policy["base_ang_vel"]["noise"] = None
    elif fault == "prefix_order": policy["base_ang_vel"] = policy.pop("base_ang_vel")
    elif fault == "actions": target_env["actions"]["joint_vel"]["scale"] = 50.
    elif fault == "action_order": target_env["actions"] = dict(reversed(list(target_env["actions"].items())))
    elif fault == "joint_default": target_env["scene"]["robot"]["init_state"]["joint_pos"] = {}
    elif fault == "normalization": target_agent["policy"]["actor_obs_normalization"] = True
    elif fault == "group_history": policy["history_length"] = 2
    elif fault == "critic": target_env["observations"]["critic"]["base_lin_vel"]["scale"] = 1.
    elif fault == "height_sensor": target_env["scene"]["height_scanner"]["update_period"] = .1
    elif fault == "height_term": policy["height_scan"]["clip"] = (-2., 2.)
    elif fault == "active_dims": dimensions["policy"][3] = (8,)
    elif fault == "active_order": terms["policy"].reverse()
    elif fault == "routing": target_agent["obs_groups"] = {"policy": ["critic"], "critic": ["policy"]}
    elif fault == "dt": target_env["sim"]["dt"] = .002
    with pytest.raises(ValueError):
        module.validate_blind_teacher_contract(source_env, target_env, source_agent, target_agent, terms, dimensions)


@pytest.mark.parametrize("fault", ["missing", "extra", "wrong_actor", "wrong_critic", "nan", "dtype", "zero_std", "activation", "target_normalizer"])
def test_invalid_parameters_are_atomic_and_do_not_modify_target(fault):
    module = importlib.import_module("vision_warm_start")
    old, target = ActorCriticMLP(45), ActorCriticMLP(232)
    state, cfg = copy.deepcopy(old.state_dict()), policy_config()
    if fault == "missing": del state["critic.4.bias"]
    elif fault == "extra": state["unknown"] = torch.ones(1)
    elif fault == "wrong_actor": state["actor.0.weight"] = torch.ones(16, 42)
    elif fault == "wrong_critic": state["critic.0.weight"] = torch.ones(16, 232)
    elif fault == "nan": state["critic.4.bias"].fill_(float("nan"))
    elif fault == "dtype": state["std"] = state["std"].double()
    elif fault == "zero_std": state["std"].zero_()
    elif fault == "activation": cfg["activation"] = "relu"
    elif fault == "target_normalizer": target.actor_obs_normalizer = nn.LayerNorm(232)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError):
        module.warm_start_vision_teacher(state, target, source_policy_cfg=cfg)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_checkpoint_initialization_records_lineage_without_loading_optimizer_or_iteration(tmp_path):
    import yaml

    module = importlib.import_module("vision_warm_start")
    source_env, source_agent, target_env, target_agent = config_fixture()
    source_agent["policy"].update(policy_config())
    target_agent["policy"].update(policy_config())
    params = tmp_path / "params"
    params.mkdir()
    for name, value in (("env.yaml", source_env), ("agent.yaml", source_agent)):
        (params / name).write_text(yaml.dump(value, sort_keys=False), encoding="utf-8")
    source, target = ActorCriticMLP(45), ActorCriticMLP(232)
    checkpoint = tmp_path / "model_19999.pt"
    torch.save({"model_state_dict": source.state_dict(), "optimizer_state_dict": {"must_not_load": True}, "iter": 19999}, checkpoint)
    terms, dimensions = active_contract()
    metadata = module.initialize_teacher_from_blind(checkpoint, target, target_env=target_env,
        target_agent=target_agent, active_terms=terms, term_dims=dimensions, task=module.TEACHER_TASK, seed=142)
    assert metadata["source_iteration"] == 19999
    assert len(metadata["source_checkpoint_sha256"]) == 64
    assert len(metadata["source_env_sha256"]) == 64
    assert len(metadata["source_agent_sha256"]) == 64
    assert metadata["load_mode"] == "vision_teacher_warm_start"
    assert metadata["target_initial_iteration"] == 0
    assert not metadata["optimizer_transferred"] and not metadata["iteration_transferred"]


@pytest.mark.parametrize("fault", [None, "disabled", "resume", "student", "upright", "checkpoint", "load_run", "distillation", "noise", "learning_rate"])
def test_request_is_opt_in_teacher_only_and_rejects_conflicting_load_modes(fault):
    module = importlib.import_module("vision_warm_start")
    args = SimpleNamespace(vision_teacher_warm_start="/tmp/source.pt", task=module.TEACHER_TASK,
        upright_warm_start=None, checkpoint=None, load_run=None, resume_noise_scale=None, resume_learning_rate=None)
    agent = {"resume": False, "class_name": "OnPolicyRunner", "algorithm": {"class_name": "PPO"},
             "load_checkpoint": "model_.*.pt", "load_run": ".*"}
    if fault == "disabled": args.vision_teacher_warm_start = None
    elif fault == "resume": agent["resume"] = True
    elif fault == "student": args.task = "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-v0"
    elif fault == "upright": args.upright_warm_start = "/tmp/upright.pt"
    elif fault == "checkpoint": args.checkpoint = "/tmp/resume.pt"
    elif fault == "load_run": agent["load_run"] = "a_hydra_run"
    elif fault == "distillation": agent["algorithm"]["class_name"] = "Distillation"
    elif fault == "noise": args.resume_noise_scale = .1
    elif fault == "learning_rate": args.resume_learning_rate = 1e-4
    if fault in (None, "disabled"):
        assert module.validate_vision_warm_start_request(args, agent) is (fault is None)
    else:
        with pytest.raises(ValueError):
            module.validate_vision_warm_start_request(args, agent)
