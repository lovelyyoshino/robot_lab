"""离线导出已验证合同的 rover RSL-RL checkpoint 为 TorchScript。

@author pony
@date 2026-09-17
@version v0.2.0
@last_modified 2026-09-17
@changelog
    - v0.2.0 (2026-09-17): 固定校验观测函数、动作映射、默认姿态和控制周期，并记录 exporter/rsl_rl 来源。
    - v0.1.0 (2026-09-17): 严格校验 44/49→12 motion 合同，并使用官方 ActorCritic 与 exporter。

不创建 Isaac App、不加载环境或仿真。未知网络/观测合同必须拒绝，不能手工拼接网络。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]
WARM_START = ROOT / "scripts/reinforcement_learning/rsl_rl/upright_warm_start.py"
OFFICIAL_EXPORTER = Path("/media/bigdisk/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py")
POLICY_TERMS = ("pitch_command", "projected_gravity", "base_lin_vel", "base_ang_vel", "joint_pos_legs", "joint_vel_legs", "wheel_vel", "last_action", "motion_command")
CRITIC_TERMS = POLICY_TERMS[:-1] + ("priv_base_height", "priv_wheel_contact", "motion_command")
ASSET_FUNCTIONS = {
    "projected_gravity": "isaaclab.envs.mdp.observations:projected_gravity",
    "base_lin_vel": "isaaclab.envs.mdp.observations:base_lin_vel",
    "base_ang_vel": "isaaclab.envs.mdp.observations:base_ang_vel",
    "joint_pos_legs": "robot_lab.tasks.manager_based.upright.mdp.observations:joint_pos_legs",
    "joint_vel_legs": "robot_lab.tasks.manager_based.upright.mdp.observations:joint_vel_legs",
    "wheel_vel": "robot_lab.tasks.manager_based.upright.mdp.observations:wheel_vel",
    "priv_base_height": "robot_lab.tasks.manager_based.upright.mdp.observations:priv_base_height",
}
JOINT_POS = ("fl_arm_joint", "rl_arm_joint", "fr_arm_joint", "rr_arm_joint", "fl_qudong_joint", "rl_qudong_joint", "fr_qudong_joint", "rr_qudong_joint")
JOINT_VEL = ("fl_wheel_joint", "rl_wheel_joint", "fr_wheel_joint", "rr_wheel_joint")
DEFAULT_JOINT_STATE = {"^(fl|fr|rl|rr)_(arm|qudong|wheel)_joint$": 0.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法按路径加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _term_names(group: dict[str, Any]) -> tuple[str, ...]:
    return tuple(name for name, value in group.items() if isinstance(value, dict) and "func" in value)


def _validate_term(name: str, term: dict[str, Any]) -> None:
    """验证 MuJoCo 输入重建依赖的函数、尺度、历史及传感器/命令语义。"""
    if term.get("scale") is not None or term.get("history_length") != 0 or term.get("flatten_history_dim") is not True:
        raise ValueError(f"观测 {name} 的 scale/history 合同不支持")
    if name in ("pitch_command", "motion_command"):
        expected_command = name
        if term.get("func") != "isaaclab.envs.mdp.observations:generated_commands" or term.get("params") != {"command_name": expected_command}:
            raise ValueError(f"观测 {name} 的命令函数或参数不匹配")
    elif name == "last_action":
        if term.get("func") != "isaaclab.envs.mdp.observations:last_action" or term.get("params") != {}:
            raise ValueError("last_action 观测合同不匹配")
    elif name == "priv_wheel_contact":
        sensor = term.get("params", {}).get("sensor_cfg", {})
        if term.get("func") != "robot_lab.tasks.manager_based.upright.mdp.observations:priv_wheel_contact" or sensor.get("name") != "contact_forces" or sensor.get("body_names") != "^(fl|rl|fr|rr)_wheel$":
            raise ValueError("priv_wheel_contact 传感器合同不匹配")
    else:
        asset = term.get("params", {}).get("asset_cfg", {})
        if term.get("func") != ASSET_FUNCTIONS[name] or asset.get("name") != "robot" or asset.get("preserve_order") is not False:
            raise ValueError(f"观测 {name} 的函数或机器人参数不匹配")


def validate_motion_contract(agent: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    """只放行本地 MuJoCo 已实现的标准非循环 44/49→12 motion 合同。"""
    try:
        policy = agent["policy"]
        if agent["class_name"] != "OnPolicyRunner" or policy["class_name"] != "ActorCritic":
            raise ValueError("仅支持 OnPolicyRunner 的非蒸馏 ActorCritic")
        if policy["activation"] != "elu" or policy["noise_std_type"] not in ("scalar", "log"):
            raise ValueError("仅支持 ELU 与状态无关 scalar/log 动作标准差")
        if not isinstance(policy["actor_obs_normalization"], bool) or not isinstance(policy["critic_obs_normalization"], bool):
            raise ValueError("观测归一化字段必须为 bool")
        if agent.get("empirical_normalization") is not None:
            raise ValueError("旧 empirical_normalization 必须为 null；normalizer 由 policy 字段和 checkpoint 严格加载")
        if agent.get("obs_groups", {}) not in ({}, {"policy": ["policy"], "critic": ["critic"]}):
            raise ValueError("不支持自定义观测组、recurrent 或 distillation 合同")
        if policy.get("state_dependent_std", False) is not False:
            raise ValueError("不支持 state_dependent_std")
        if tuple(policy["actor_hidden_dims"]) != (128, 128, 128) or tuple(policy["critic_hidden_dims"]) != (128, 128, 128):
            raise ValueError("当前工具只支持已验证的 128×3 motion MLP")
        observations = env["observations"]
        if _term_names(observations["policy"]) != POLICY_TERMS or _term_names(observations["critic"]) != CRITIC_TERMS:
            raise ValueError("观测项顺序不是已验证的 motion 44/49 维合同")
        for group in ("policy", "critic"):
            for name in _term_names(observations[group]):
                _validate_term(name, observations[group][name])
        actions = env["actions"]
        if tuple(actions) != ("joint_pos", "joint_vel") or tuple(actions["joint_pos"]["joint_names"]) != JOINT_POS or tuple(actions["joint_vel"]["joint_names"]) != JOINT_VEL:
            raise ValueError("动作关节顺序与 MuJoCo 映射不匹配")
        if actions["joint_pos"].get("scale") != 0.25 or actions["joint_vel"].get("scale") != 0.8 or actions["joint_pos"].get("offset") != 0.0 or actions["joint_vel"].get("offset") != 0.0:
            raise ValueError("动作尺度或偏置与部署合同不匹配")
        if env["scene"]["robot"]["init_state"]["joint_pos"] != DEFAULT_JOINT_STATE or env["scene"]["robot"]["init_state"]["joint_vel"] != DEFAULT_JOINT_STATE:
            raise ValueError("默认关节状态与部署合同不匹配")
        if env["sim"]["dt"] != 0.005 or env["decimation"] != 4:
            raise ValueError("物理周期或控制降采样与部署合同不匹配")
        if sum(len(value["joint_names"]) for value in actions.values()) != 12:
            raise ValueError("动作合同必须按 joint_pos、joint_vel 顺序且恰为 12 维")
    except KeyError as error:
        raise ValueError(f"训练快照缺少导出合同字段: {error}") from error
    return {"actor_observation_dim": 44, "critic_observation_dim": 49, "action_dim": 12, "control_dt_s": 0.02,
            "actor_normalization": policy["actor_obs_normalization"], "critic_normalization": policy["critic_obs_normalization"]}


def build_policy(agent: dict[str, Any], contract: dict[str, Any]):
    """使用本机 rsl_rl 官方 ActorCritic 构造模型，不复制网络实现。"""
    from rsl_rl.modules import ActorCritic

    config = agent["policy"]
    observations = TensorDict({"policy": torch.zeros(1, 44), "critic": torch.zeros(1, 49)}, batch_size=[1])
    return ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, 12,
                       actor_obs_normalization=contract["actor_normalization"], critic_obs_normalization=contract["critic_normalization"],
                       actor_hidden_dims=config["actor_hidden_dims"], critic_hidden_dims=config["critic_hidden_dims"],
                       activation=config["activation"], init_noise_std=config["init_noise_std"], noise_std_type=config["noise_std_type"])


def load_checkpoint(policy, path: Path) -> int:
    """严格加载 state_dict；normalizer、循环层或多余键均视为不支持的合同。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"无法以 weights_only 读取 checkpoint: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("checkpoint 必须含 model_state_dict")
    try:
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"checkpoint 与标准 ActorCritic 合同不一致: {error}") from error
    iteration = checkpoint.get("iter")
    if type(iteration) is not int or iteration < 0 or policy.is_recurrent or policy.state_dependent_std:
        raise ValueError("不支持无 iter、recurrent 或 state-dependent std checkpoint")
    return iteration


def export_policy(run_dir: Path, checkpoint: Path, output_dir: Path, samples=32) -> dict[str, Any]:
    """CPU 导出并以随机输入比较 eager actor 与官方 TorchScript 输出。"""
    run_dir, checkpoint, output_dir = run_dir.resolve(), checkpoint.resolve(), output_dir.resolve()
    if checkpoint.parent != run_dir or not checkpoint.is_file():
        raise ValueError("checkpoint 必须是指定训练目录内的现有文件")
    safe_yaml = _module(WARM_START, "rover_export_safe_yaml")
    agent_path, env_path = run_dir / "params/agent.yaml", run_dir / "params/env.yaml"
    agent, env = safe_yaml.load_training_config(agent_path), safe_yaml.load_training_config(env_path)
    contract = validate_motion_contract(agent, env)
    policy = build_policy(agent, contract).cpu().eval()
    iteration = load_checkpoint(policy, checkpoint)
    if not OFFICIAL_EXPORTER.is_file():
        raise RuntimeError(f"缺少 Isaac Lab 官方 exporter: {OFFICIAL_EXPORTER}")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / f"{checkpoint.stem}_policy.pt"
    _module(OFFICIAL_EXPORTER, "rover_official_exporter").export_policy_as_jit(policy, policy.actor_obs_normalizer, str(output_dir), policy_path.name)
    scripted = torch.jit.load(str(policy_path), map_location="cpu").eval()
    inputs = torch.randn(samples, 44, generator=torch.Generator().manual_seed(20260917))
    with torch.no_grad():
        eager, jit = policy.actor(policy.actor_obs_normalizer(inputs)), scripted(inputs)
    max_abs_error = float((eager - jit).abs().max())
    if not torch.isfinite(jit).all() or max_abs_error > 1e-6:
        raise RuntimeError(f"导出前后 actor 不等价或输出非有限: {max_abs_error}")
    import rsl_rl
    evidence = {"schema_version": 2, "source": {"run_dir": str(run_dir), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "agent_yaml_sha256": sha256(agent_path), "env_yaml_sha256": sha256(env_path), "iteration": iteration, "rsl_rl_version": getattr(rsl_rl, "__version__", "unknown")}, "output": {"policy": str(policy_path), "policy_sha256": sha256(policy_path)}, "implementation": {"official_exporter": str(OFFICIAL_EXPORTER), "official_exporter_sha256": sha256(OFFICIAL_EXPORTER)}, "contract": contract, "verification": {"device": "cpu", "random_samples": samples, "max_abs_error": max_abs_error, "eager_jit_equivalent": True}}
    evidence_path = output_dir / f"{checkpoint.stem}_export_evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence["output"]["evidence"] = str(evidence_path)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="离线导出已验证的 rover motion 策略")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("随机对比样本数必须为正")
    print(json.dumps(export_policy(args.run_dir, args.checkpoint, args.output_dir or args.run_dir / "exported_offline", args.samples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
