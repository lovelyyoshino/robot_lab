"""58维成功示范到阶段BC的显式转换；原NPZ和物理控制语义保持不变。

@author pony
@date 2026-09-29
@version v0.1.0
@last_modified 2026-09-29
@changelog
    - v0.1.0 (2026-09-29): 校验固定三秒脚本时钟，追加阶段/锁定倒向并迁移首层。
"""

import copy

import numpy as np
import torch

PHASE_TASK = "RobotLab-Isaac-Recovery-Side-Phase-Parallelogram-Rover-v0"
SIDE_TASK = "RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0"
PHASE_DURATION_S = 3.0
PHASE_TERMS = ("recovery_phase", "recovery_initial_side")


def phase_observation_group(group):
    result = copy.deepcopy(group)
    for name in PHASE_TERMS:
        if name in result:
            raise ValueError("观测已经包含阶段项，拒绝重复扩维")
        result[name] = {
            "func": f"robot_lab.tasks.manager_based.recovery.mdp.phase:{name}",
            "params": {"duration_s": PHASE_DURATION_S} if name == "recovery_phase" else {},
            "modifiers": None, "noise": None, "clip": None, "scale": None,
            "history_length": 0, "flatten_history_dim": True,
        }
    return result


def phase_configs(source_env, source_agent):
    """只修改声明的两列观测和训练奖励；真实运行时仍以完整控制合同核验。"""
    env, agent = copy.deepcopy(source_env), copy.deepcopy(source_agent)
    for group in ("policy", "critic"):
        env["observations"][group] = phase_observation_group(env["observations"][group])
    rewards = env["rewards"]
    rewards["orientation_progress"].update(params={"gain": 1.0, "as_rate": True}, weight=2.0)
    rewards["height_progress"].update(params={"as_rate": True}, weight=0.2)
    prefix = "robot_lab.tasks.manager_based.recovery.mdp.grounded:"
    rewards["joint_velocity"].update(func=prefix + "recovery_leg_velocity", weight=-2e-4)
    rewards["wheel_velocity"].update(func=prefix + "recovery_phase_wheel_velocity", weight=-1e-4)
    agent.update(experiment_name="parallelogram_rover_recovery_side_phase", run_name="side_phase", save_interval=50)
    agent["algorithm"].update(learning_rate=3e-5, entropy_coef=0.0, schedule="fixed")
    return env, agent


def phase_demonstrations(arrays, metadata):
    """阶段来自动作前control_step，倒向只从同回合接管观测取一次，避免标签泄漏。"""
    source, contract = metadata["source"], metadata["contract"]
    program = source.get("action_program", {})
    if (source.get("task") != SIDE_TASK or contract["actor_observation_dim"] != 58
            or contract["critic_observation_dim"] != 70
            or program.get("active_steps", 0) * contract["step_dt"] != PHASE_DURATION_S):
        raise ValueError("阶段BC只接受原Side的58/70维完整三秒动作脚本示范")
    x, steps, ids = arrays["observations"], arrays["control_step"], arrays["episode_id"]
    if x.shape[1] != 58:
        raise ValueError("阶段转换源输入必须为58维")
    sides = np.zeros(len(x), dtype=np.float32)
    for episode in np.unique(ids):
        rows = np.flatnonzero(ids == episode)
        if len(rows) != 250 or not np.array_equal(steps[rows], np.arange(250)):
            raise ValueError("阶段转换需要每回合250步连续时钟")
        initial_y = float(x[rows[0], 1])
        if not np.isfinite(initial_y) or abs(initial_y) < 0.7:
            raise ValueError("接管状态不是可确定倒向的侧躺")
        sides[rows] = np.sign(initial_y)
    clock = np.clip(steps.astype(np.float32) * contract["step_dt"] / PHASE_DURATION_S, 0., 1.)
    result = dict(arrays)
    result["observations"] = np.concatenate((x, clock[:, None], sides[:, None]), axis=1).astype(np.float32)
    info = copy.deepcopy(metadata)
    target = info["contract"]
    target.update(actor_observation_dim=60, critic_observation_dim=72)
    target["observation_order"] += list(PHASE_TERMS)
    target["observation_term_dims"] += [[1], [1]]
    target["observation_group"] = phase_observation_group(target["observation_group"])
    if "control_contract" in target:
        for group in ("policy", "critic"):
            cc = target["control_contract"]
            cc["observations"][group] = phase_observation_group(cc["observations"][group])
            cc["observation_term_order"][group] += list(PHASE_TERMS)
    info["derived_features"] = {"duration_s": PHASE_DURATION_S, "initial_side": "sign(first_observation.gravity_y)",
                                "source_actor_dim": 58, "target_actor_dim": 60, "source_npz_unchanged": True}
    return result, info


def phase_policy_state(source_state):
    """新增列置零，转换瞬间的actor/critic函数不变；不迁移旧优化器。"""
    result = {name: value.detach().clone() for name, value in source_state.items()}
    for key, old_width in (("actor.0.weight", 58), ("critic.0.weight", 70)):
        weight = result[key]
        if weight.ndim != 2 or weight.shape[1] != old_width:
            raise ValueError(f"{key}不是受支持的旧输入维数")
        result[key] = torch.cat((weight, weight.new_zeros(weight.shape[0], 2)), dim=1)
    return result
