#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
# @author pony
# @date 2026-08-26
# @version v0.4.0
# @last_modified 2026-09-17
# @changelog
#   - v0.4.0 (2026-09-17): 恢复交还和两轮完成改为显式接触/速度门，生产路径缺测 fail closed。
#   - v0.3.0 (2026-09-17): 声明 42/44 维直立与连续/Temporal 视觉的不可互换部署合同。
#   - v0.2.0 (2026-09-16): 修复策略键映射与恢复门，加入起立过渡态和显式恢复命令。
#   - v0.1.0 (2026-08-26): 建立三策略姿态状态机与观测合同。

"""三策略切换器（部署态）。

`parallelogram_rover` 是轮腿混合四足：四轮行驶（`loco`）、双轮直立（`upright`）、
倒地爬起（`recover`）由三个**独立训练**的策略各自承担。切换器根据机体姿态
（IMU 重力投影 + 高度）在三个模式间自动切换，不依赖策略梯度。

三个策略的 actor 观测互不兼容（见 `OBS_LAYOUT` 表），但部署时必须读取同一台车的
同一时刻状态。协调器从一份传感器快照分别组装观测，只执行当前模式对应的策略；
不能用三个各自演化的仿真 env 代替同一物理状态。

**观测布局权威来源**：`OBS_LAYOUT` 已按各 env 的 `group_obs_term_dim['policy']` 实测核对：
- `loco` 45 维 = base_ang_vel(3)+gravity(3)+velocity_commands(3)+joint_pos(12)+joint_vel(12)+actions(12)
  （flat 的 policy 不含 `height_scan`（置 None），也**不含 base_lin_vel**）
- `upright` 42 维（含 `base_lin_vel` 与 `pitch_command`）
- `recover` 58 维（含 `prev_*` 历史 3 项，无 `base_lin_vel`）

> 拼接器必须按 `OBS_LAYOUT` 的**项名**组装——loco 用 `joint_pos`/`actions`，
> upright/recover 用 `joint_pos_legs`/`last_action`，不能混用。


**关键约束**（§3.0-A 硬规则 + 本轮实测）：
  - 不改动已训通的 locomotion / velocity 任何文件
  - 三个策略的动作空间均为 **12 维**（loco 也是 12 个主动关节，非 20）
  - actor 观测布局各不相同（见 `_OBS`），切换器按策略各自组装
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "StanceMode",
    "StanceStateMachine",
    "PlannerMode",
    "PlannerConfig",
    "Planner",
    "HYBRID_MANIFEST",
    "STANCE_KINEMATICS",
    "contact_state_from_forces",
]

# ════════════════════════════════════════════════════════════════════════════
# Stance state machine（纯函数，IL 无关，可单测）
# ════════════════════════════════════════════════════════════════════════════

StanceMode = Literal["LOCO", "UPRIGHT", "RECOVER"]
StanceCommand = Literal["lo", "up", "rec"]


@dataclass
class StanceConfig:
    """姿态阈值，全部来自实测/已定义常量（非估计）。

    - `q_gz_max`：横滚判据。`abs(g_y) > 0.7` ≈ 横滚 44°，用 `excessive_roll` 同值。
    - `upright_gz_max`：直立判据 `abs(g_z) < 0.3`，覆盖可保持区
      [75°,105°] → g_z ∈ [-0.259, +0.256]。
    - `upright_h_min`：直立高度门。实测四轮平放 root=0.332、独轮站立≈0.30、
      塌落<0.229。故取 0.24 排除塌落，但**与恢复完成门区分**。
    - `h_fall`：高度塌落门（低于则必须恢复）。
    - `loco_gz`：四轮行驶中判定倾倒的姿态门；正常四轮态约为 -1。
    - `recovery_h_min`：恢复完成高度门，与 recovery support-state 保持一致。
    - `rise_timeout`：起立策略允许穿越不可保持区的最大控制步数。
    - `min_hold`：滞回最小驻留（步）。
    """

    q_gz_max: float = 0.70
    upright_gz_max: float = 0.30
    upright_h_min: float = 0.24
    h_fall: float = 0.15
    loco_gz: float = -0.5
    loco_h: float = 0.25
    recovery_h_min: float = 0.26
    recovery_h_max: float = 0.45
    recovery_angular_speed_max: float = 0.75
    recovery_linear_speed_max: float = 0.15
    recovery_leg_speed_max: float = 0.50
    upright_linear_speed_max: float = 1.00
    upright_leg_speed_max: float = 0.50
    rise_timeout: int = 150
    min_hold: int = 18


def contact_state_from_forces(body_names, forces, threshold: float = 1.0) -> dict[str, bool]:
    """以接触传感器自身名称解析四轮/双后轮支撑；缺名称或非有限力一律不支持。"""
    wheels = ("fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel")
    names = list(body_names)
    if len(names) != len(set(names)) or not all(name in names for name in wheels):
        raise ValueError("接触传感器必须包含唯一的 fl/fr/rl/rr_wheel 名称")
    if len(forces) != len(names):
        raise ValueError("接触力数量与接触传感器 body_names 不一致")
    magnitudes = []
    for force in forces:
        components = tuple(float(value) for value in force)
        if len(components) != 3 or not all(math.isfinite(value) for value in components):
            return {"valid": False, "four_wheel": False, "rear_wheel": False, "non_wheel": True}
        magnitudes.append(math.sqrt(sum(value * value for value in components)))
    contacts = {name: magnitudes[index] > threshold for index, name in enumerate(names)}
    non_wheel = any(active for name, active in contacts.items() if name not in wheels)
    four_wheel = all(contacts[name] for name in wheels) and not non_wheel
    rear_wheel = contacts["rl_wheel"] and contacts["rr_wheel"]
    rear_wheel &= not contacts["fl_wheel"] and not contacts["fr_wheel"] and not non_wheel
    return {"valid": True, "four_wheel": four_wheel, "rear_wheel": rear_wheel, "non_wheel": non_wheel}


@dataclass
class StanceStateMachine:
    """姿态感知的切换状态机。带滞回，避免在阈值边界抖动。"""

    cfg: StanceConfig = field(default_factory=StanceConfig)
    mode: StanceMode = "LOCO"
    _holds: int = 0
    _rising: bool = False
    _rise_steps: int = 0
    last_gz: float = field(default=-1.0, init=False)
    last_gy: float = field(default=0.0, init=False)
    last_h: float = field(default=0.0, init=False)
    last_gx: float | None = field(default=None, init=False)
    strict_measurements: bool = False
    last_measurement_diagnostics: str = field(default="legacy_pose_only", init=False)

    @property
    def is_upright(self) -> bool:
        return (
            abs(self.last_gz) < self.cfg.upright_gz_max
            and self.last_h >= self.cfg.upright_h_min
            and (self.last_gx is None or self.last_gx < 0.0)
        )

    def _measurement_gate(self, *, four_wheel_support, rear_wheel_support, linear_speed, leg_speed,
                          recovery: bool) -> bool:
        """生产 strict 模式缺任何接触/速度量即失败；旧调用仅保留为姿态诊断兼容。"""
        values = (four_wheel_support, rear_wheel_support, linear_speed, leg_speed)
        if any(value is None for value in values):
            self.last_measurement_diagnostics = "missing_contact_or_velocity_measurement"
            return not self.strict_measurements
        if not all(isinstance(value, bool) for value in values[:2]) or not all(math.isfinite(float(value)) for value in values[2:]):
            self.last_measurement_diagnostics = "invalid_contact_or_velocity_measurement"
            return False
        if recovery:
            passed = bool(four_wheel_support) and linear_speed <= self.cfg.recovery_linear_speed_max
            passed &= leg_speed <= self.cfg.recovery_leg_speed_max
        else:
            passed = bool(rear_wheel_support) and linear_speed <= self.cfg.upright_linear_speed_max
            passed &= leg_speed <= self.cfg.upright_leg_speed_max
        self.last_measurement_diagnostics = "strict_measurements_passed" if passed else "support_or_speed_gate_failed"
        return passed

    def _enter(self, mode: StanceMode, *, rising: bool = False) -> StanceMode:
        """统一清理滞回计数，避免跨模式残留触发错误切换。"""
        self.mode = mode
        self._holds = 0
        self._rising = rising
        self._rise_steps = 0
        return self.mode

    def select(
        self,
        stance_cmd: StanceCommand,
        g_z: float,
        g_y: float,
        base_height: float,
        *,
        g_x: float | None = None,
        angular_speed: float = 0.0,
        four_wheel_support: bool | None = None,
        rear_wheel_support: bool | None = None,
        linear_speed: float | None = None,
        leg_speed: float | None = None,
    ) -> StanceMode:
        """根据当前模式 + 传感器，返回进入的模式。

        `stance_cmd` 是用户/上层指令（"lo"|"up"|"rec");`g_z`/`g_y` 是重力在
        机体坐标的投影,`base_height` 是机体离地高度。
        """
        state = (g_z, g_y, base_height, angular_speed) + (() if g_x is None else (g_x,))
        if not all(math.isfinite(value) for value in state):
            raise ValueError("切换器收到非有限机器人状态")
        self.last_gx = g_x
        self.last_gz = g_z
        self.last_gy = g_y
        self.last_h = base_height

        if stance_cmd not in {"lo", "up", "rec"}:
            raise ValueError(f"未知姿态指令: {stance_cmd!r}; 期望 'lo'、'up' 或 'rec'")

        # rec 是显式接管命令；保持 rec 时不自动交回其它策略。
        if stance_cmd == "rec":
            if self.mode != "RECOVER":
                return self._enter("RECOVER")
            return self.mode

        # 恢复完成判据：四轮着地且机体近水平，持续 T 步 → 交还 LOCO。
        # 与 recovery 的 `support_state` 同源：MIN_BASE_HEIGHT=0.26、GZ<-0.7。
        if self.mode == "RECOVER":
            if (
                self.cfg.recovery_h_min <= base_height <= self.cfg.recovery_h_max
                and g_z < -0.7 and angular_speed < self.cfg.recovery_angular_speed_max
                and self._measurement_gate(
                    four_wheel_support=four_wheel_support, rear_wheel_support=rear_wheel_support,
                    linear_speed=linear_speed, leg_speed=leg_speed, recovery=True,
                )
            ):
                self._holds += 1
                if self._holds >= self.cfg.min_hold:
                    return self._enter("LOCO")
            else:
                self._holds = 0
            return self.mode

        # 横滚/侧躺 → RECOVER（任何模式都可能侧翻）
        if abs(g_y) > self.cfg.q_gz_max:
            return self._enter("RECOVER")

        # 高度塌落 → RECOVER
        if base_height < self.cfg.h_fall:
            return self._enter("RECOVER")

        if self.mode == "UPRIGHT":
            # 从双轮态回四轮态也必须经过受控过渡，不能直接交给四轮行驶策略。
            if stance_cmd == "lo":
                return self._enter("RECOVER")

            if self._rising:
                self._rise_steps += 1
                if self.is_upright and self._measurement_gate(
                    four_wheel_support=four_wheel_support, rear_wheel_support=rear_wheel_support,
                    linear_speed=linear_speed, leg_speed=leg_speed, recovery=False,
                ):
                    self._rising = False
                    self._rise_steps = 0
                elif self._rise_steps >= self.cfg.rise_timeout:
                    return self._enter("RECOVER")
                return self.mode

            # 仅在已经完成起立后，偏离可保持区才按倾倒处理。
            if not self.is_upright or not self._measurement_gate(
                four_wheel_support=four_wheel_support, rear_wheel_support=rear_wheel_support,
                linear_speed=linear_speed, leg_speed=leg_speed, recovery=False,
            ):
                return self._enter("RECOVER")
            # 保持直立
            return self.mode

        # LOCO 模式
        if stance_cmd == "up":
            already_upright = self.is_upright and self._measurement_gate(
                four_wheel_support=four_wheel_support, rear_wheel_support=rear_wheel_support,
                linear_speed=linear_speed, leg_speed=leg_speed, recovery=False,
            )
            return self._enter("UPRIGHT", rising=not already_upright)

        # 正常四轮态 g_z≈-1 且高度约 0.33；姿态或高度任一越界都需恢复。
        if g_z > self.cfg.loco_gz or base_height < self.cfg.loco_h:
            return self._enter("RECOVER")
        return self.mode

    def reset(self, mode: StanceMode = "LOCO") -> None:
        self._enter(mode)


# ════════════════════════════════════════════════════════════════════════════
# 混合 Planner
# ════════════════════════════════════════════════════════════════════════════

# 三个策略的 actor 观测布局
# loco:  base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_legs, joint_vel_legs, actions, height_scan
#        （height_scan 是射线, 8 条; 空 list 时由 env 决定——需动态）
# upright: pitch_command, projected_gravity, base_lin_vel, base_ang_vel, joint_pos_legs, joint_vel_legs, wheel_vel, last_action
# recover: projected_gravity, base_ang_vel, joint_pos_legs, joint_vel_legs, wheel_vel, prev_joint_pos_legs, prev_joint_vel_legs, prev_wheel_vel, last_action

# 注意：这三套观测维度不同，切换器**不能**用一个通用 ObservationManager 喂。
# 故每个策略的 obs 维度是静态的（训练时固定），切换器按《各自策略的 obs_spec》组装。

# 策略元数据：[名称, 观测维度, 动作维度]
POLICY_SPECS = {
    "loco": {"obs": 45, "act": 12},
    "upright": {"obs": 42, "act": 12},
    "recover": {"obs": 58, "act": 12},
}

# 这些规格是 hybrid 入口选择任务的唯一白名单。输入维度相同不代表可互换：
# Temporal 与连续帧 student 都是 4141 维，但历史采样、时延和扰动语义不同。
UPRIGHT_POLICY_SPECS = {
    "upright": {
        "task": "RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0",
        "observation_group": "upright",
        "actor_obs_dim": 42,
        "critic_obs_dim": 47,
        "wheel_scale": 0.8,
    },
    "motion": {
        "task": "RobotLab-Isaac-Upright-Motion-Getup-Parallelogram-Rover-v0",
        "observation_group": "motion",
        "actor_obs_dim": 44,
        "critic_obs_dim": 49,
        "wheel_scale": 0.8,
        "command_layout": "append:[ground_forward_velocity_m_s,world_yaw_rate_rad_s]",
    },
}

LOCOMOTION_POLICY_SPECS = {
    "blind": {
        "task": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0",
        "observation_group": "policy",
        "actor_obs_dim": 45,
        "wheel_scale": 5.0,
    },
    "student": {
        "task": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-v0",
        "observation_group": "policy",
        "actor_obs_dim": 4141,
        "wheel_scale": 5.0,
        "depth_contract_id": "rover_depth_contiguous_v1",
    },
    "student-temporal": {
        "task": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-v0",
        "observation_group": "policy",
        "actor_obs_dim": 4141,
        "wheel_scale": 5.0,
        "depth_contract_id": "rover_depth_temporal_v2",
    },
}

PlannerMode = StanceMode


@dataclass
class PlannerConfig:
    """Planner 配置。

    与 `StanceConfig` 一致，`stance_cmd` 是用户/上层指令。
    """

    stance_cmd: str = "lo"  # "lo" | "up" | "rec"
    stance: StanceConfig = field(default_factory=StanceConfig)


@dataclass
class Planner:
    """三策略切换协调者（不含训练）。负责：
    1. 根据姿态选择合适的策略
    2. 为选中的策略组装观测（各策略自有 env / obs manager）
    3. 把动作映射到 12 维动作空间

    生产路径只维护一份实时机器人状态。`poll()` 接收由这份状态按三个合同分别
    组装的观测；仿真评测也只能有一个真正推进动力学的受控 env。
    """

    cfg: PlannerConfig = field(default_factory=PlannerConfig)

    def __post_init__(self):
        self._envs: dict[str, object] = {}
        self._policies: dict[str, object] = {}
        self._state: StanceStateMachine | None = None
        self._stance_cmd: StanceCommand = self.cfg.stance_cmd

    # —— 接入 env / policy ——
    def register_env(self, name: str, env) -> None:
        self._envs[name] = env

    def register_policy(self, name: str, policy) -> None:
        self._policies[name] = policy

    @property
    def mode(self) -> StanceMode:
        return self._state.mode if self._state else "LOCO"

    def setup(self, env, policy_upright, policy_loco, policy_recover, stance_cmd="lo"):
        """配置基本环境 + 三个策略 + 状态机（单 env 演示用）。"""
        self._env = env
        self._policies = {
            "upright": policy_upright,
            "loco": policy_loco,
            "recover": policy_recover,
        }
        self._state = StanceStateMachine(cfg=self.cfg.stance)
        self.set_stance_command(stance_cmd)

    def set_stance_command(self, stance_cmd: StanceCommand) -> None:
        """更新上层姿态指令，并在输入错误时立即失败。"""
        if stance_cmd not in {"lo", "up", "rec"}:
            raise ValueError(f"未知姿态指令: {stance_cmd!r}; 期望 'lo'、'up' 或 'rec'")
        self._stance_cmd = stance_cmd

    # ── 生产路径：每步调用 ──
    def poll(self, g_z, g_y, base_height, obs_by_policy):
        """生产协调者接口。

        Args:
            g_z: 机体重力投影 z（world 竖直投影）
            g_y: 机体重力投影 y（横滚）
            base_height: 机体离地高度
            obs_by_policy: dict，每策略 env 各自 compute 好的 policy 观测
                （tensor [1, n_obs]，维度与 POLICY_SPECS 一致）

        Returns:
            (mode, action)。mode 是选中的策略名。
        """
        if self._state is None:
            self._state = StanceStateMachine()
        mode = self._state.select(self._stance_cmd, g_z, g_y, base_height)
        policy_name = mode.lower()
        self._last_policy = policy_name
        obs = obs_by_policy[policy_name]
        policy = self._policies[policy_name]
        action = policy(obs)
        return mode, action

    # ── 单 env 演示：用该 env 的观测组切片 ──
    def _assemble(self, mode: str, obs_universe):
        """从单一 env 的 policy 观测组切片（演示用，非生产）。

        生产下三策略各自的完整观测已经由各自 env 算好（`poll` 的 `obs_by_policy`），
        此处仅用于 `play_parallelogram_hybrid.py` 单 env 走通维度。
        """
        policy_name = mode.lower()
        obs = obs_universe.view(-1)[: POLICY_SPECS[policy_name]["obs"]]
        return obs


# ── 观测布局（每个策略的训练顺序）──
# 这些从 3 个 env 的 `observation_manager.group_obs_term_dim['policy']` 动态取得，
# 但为单测，这里给出静态布局。

OBS_LAYOUT = {
    "loco": [
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_commands", 3),
        ("joint_pos", 12),
        ("joint_vel", 12),
        ("actions", 12),
    ],
    "upright": [
        ("pitch_command", 1),
        ("projected_gravity", 3),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("joint_pos_legs", 8),
        ("joint_vel_legs", 8),
        ("wheel_vel", 4),
        ("last_action", 12),
    ],
    "recover": [
        ("projected_gravity", 3),
        ("base_ang_vel", 3),
        ("joint_pos_legs", 8),
        ("joint_vel_legs", 8),
        ("wheel_vel", 4),
        ("prev_joint_pos_legs", 8),
        ("prev_joint_vel_legs", 8),
        ("prev_wheel_vel", 4),
        ("last_action", 12),
    ],
}

# 静态维度核对：loco = 3+3+3+3+12+12+9 = 45（若 height_scan=0）；upright = 1+3+3+3+8+8+4+12=42；recover=58
assert sum(d for _, d in OBS_LAYOUT["upright"]) == 42, "upright obs != 42"
assert sum(d for _, d in OBS_LAYOUT["recover"]) == 58, "recover obs != 58"

# 与 POLICY_SPECS 一致
assert POLICY_SPECS["loco"]["obs"] == sum(d for _, d in OBS_LAYOUT["loco"])
assert POLICY_SPECS["upright"]["obs"] == 42
assert POLICY_SPECS["recover"]["obs"] == 58


# ════════════════════════════════════════════════════════════════════════════
# 三策略混合部署的 manifest（不发散单一份）
# ════════════════════════════════════════════════════════════════════════════

HYBRID_MANIFEST = {
    "upright": {
        "policy": POLICY_SPECS["upright"],
        "obs_layout": OBS_LAYOUT["upright"],
    },
    "loco": {
        "policy": POLICY_SPECS["loco"],
        "obs_layout": OBS_LAYOUT["loco"],
    },
    "recover": {
        "policy": POLICY_SPECS["recover"],
        "obs_layout": OBS_LAYOUT["recover"],
    },
}


# 姿态运动学（从重力投影反算姿态角，供状态机/奖励用）
STANCE_KINEMATICS = {
    # 直立：机体 z 轴关于横滚轴的角度（用于 pitch 反解）
    "upright_pitch": lambda gx, gy, gz: math.atan2(-gx, -gz) if math.sqrt(gx * gx + gz * gz) > 1e-3 else 0.0,
    "roll": lambda gy: math.asin(max(-1.0, min(1.0, gy))),
}


def build_planner(env, policy_upright, policy_loco, policy_recover, stance_cmd="lo"):
    """便捷构造完整的 Planner，供 play/评测用。"""

    planner = Planner()
    planner.setup(env, policy_upright, policy_loco, policy_recover, stance_cmd)
    return planner
