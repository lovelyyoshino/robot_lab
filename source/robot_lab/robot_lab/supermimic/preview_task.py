"""
preview_task.py — Teacher-Preview 越障任务在想象中的可导版本：232 维观测、解析奖励、几何抬轮塑形、软存活。

与 env 的关系（verify_preview_task 逐项核对）：
- 观测 = 45 维本体（命令按 preview_route_command 把世界系路线命令转到车体系）⊕ 187 维高度扫描；
- 解析奖励逐项复现 env 的 track_lin_vel_xy_exp(preview 版)、lin_vel_z、ang_vel_xy、action_rate、
  joint_deviation_steering、joint_pos_limits、flat_orientation、preview_yaw_rate；
- env 的 anticipatory_wheel_lift 是带隐藏状态机的稀疏脉冲（实测占回报方差约 100%），在想象里无法复现，
  这里换成可导的几何塑形 wheel_lift：前方有台阶时，按轮心抬升占所需高度的比例给分；
  平地无台阶时抬轮扣分（对应 env 的 idle_lift_penalty）。时机按用户要求放宽，不强制 8–14 cm 窗口。
- 未复现的 joint_torques / contact_forces / undesired_contacts 在 Teacher 数据上合计 |r|≈3e-5/步，忽略。

@author pony
@date 2026-10-26
@version v0.4.0
@last_modified 2026-10-28
@changelog
    - v0.4.0 (2026-10-28): 新增 terminated 翻车惩罚、event_priority（事件优先回放）、lane_obs（方案 B 车道横向偏移观测）、smooth_flat_arm / smooth_flat_ang（平地 arm 摆动与俯仰抖动惩罚）。
    - v0.3.0 (2026-10-27): 新增 riser_anchor（后轮贴立面时加重 Teacher 锚定）、lateral_pos（critic 可见车道偏移）；新增 wheel_heading（轮子世界系朝向偏离路线），针对蟹行导致的台阶侧漂。
    - v0.2.0 (2026-10-26): 新增航向/横向速度/倾角软惩罚 shaping_terms（默认关闭）。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import terrain_map
from .quat import projected_gravity, quat_apply, yaw_quat
from .lane_offset import LANE_OBS_CLIP, LANE_OBS_SCALE
from .rover_task import ANG_VEL_SCALE, CONTROL_DT, JOINT_VEL_SCALE, OBS_CLIP, STEERING
from .state import ANG, JPOS, JVEL, LIN, NUM_LEG_JOINTS, POS, QUAT

PROPRIO_DIM = 45
WHEEL_RADIUS = 0.1085  # URDF 碰撞圆柱半径


@dataclass
class PreviewRewardCfg:
    # 与 configure_preview_rewards 相同的权重；verify_preview_task 会与 RewardManager 的实际权重比对。
    track_lin_vel: float = 5.0
    track_lin_vel_std: float = 0.5
    lin_vel_z: float = -0.5
    ang_vel_xy: float = -0.05
    action_rate: float = -0.005
    steering_deviation: float = -0.1
    joint_pos_limits: float = -5.0
    flat_orientation: float = -0.1
    yaw_rate: float = -0.05
    # 几何抬轮塑形（替代 anticipatory_wheel_lift）
    wheel_lift: float = 1.0
    idle_lift: float = -2.0
    lift_lookahead: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
    lift_margin: float = 0.01
    min_step: float = 0.03
    rest_clearance: float = 0.005
    # 软存活：与 env 新增的 bad_orientation 阈值一致
    tilt_limit: float = 1.2
    tilt_temperature: float = 0.05
    # v3 直行/姿态塑形（env 原奖励没有，默认 0 以保持 verify_preview_task 的逐项核对）。
    # 验收要求越线时横向 ≤0.15 m、倾角 ≤31.8°；v2 越线航向偏 ~13°、16 cm 倾角中位 62°。
    heading: float = 0.0  # × sin²(航向角)，路线方向固定为世界 +x
    lateral_vel: float = 0.0  # × 世界系 vy²
    tilt_soft: float = 0.0  # × relu(cos(tilt_soft_start) − cosθ)，31.8° 处约 0.056、60° 处约 0.41
    tilt_soft_start: float = math.radians(25.0)
    # × 四轮平均 sin²(航向 + 转向角)：Teacher/v2 都是"车头左偏 15–25°、四轮右转 20–35°"的蟹行，
    # 平地上相互抵消；上 16 cm 台阶时车沿轮子滚动方向爬，每级向 −y 偏约 0.12 m（2026-10-27 诊断）。
    wheel_heading: float = 0.0
    # × 相对车道中心线的横向偏移²。actor 观测里没有横向位置（扫描 ±0.5 m、车道 3 m），
    # 但 16 cm 侧漂是系统性的（v2/v3b/v4 越线时均 −0.37~−0.51 m，标准差 <0.1 m），可按地形学开环补偿；
    # critic 通过 critic_extra 看到偏移量。需要构造时传 lane_center_y。
    lateral_pos: float = 0.0
    # 后轮贴台阶立面时对 Teacher 的额外锚定倍数（1 + riser_anchor）。2026-10-27 回放分析：
    # v3 在 8 cm 上 12% 回合后轮停在首级立面前（每回合贴立面 0.95 s，Teacher 仅 0.13 s），
    # 此时 v3 后轮扭矩中位 3.5 N·m、后 arm −14°，Teacher 为 5.1 N·m、−23°；越过前缘要穿过接触，
    # 想象梯度学不到"再用力推一下"，BPTT 反而把这里的动作从 Teacher 推开。0 = 关闭。
    riser_anchor: float = 0.0
    # × 平地处（四轮前方都无台阶）arm 关节速度²之和；以及俯仰/滚转角速度²。2026-10-28：用户训到 4000 轮的
    # terrain_v3 在平地 0.3 m/s 下 arm 以 2.9 Hz 来回摆（arm 速度 rms 1.21 rad/s，第 3000 轮时 0.14），
    # 车体上下 2.7 cm、俯仰角速度 24°/s；原有 lin_vel_z −0.5 / ang_vel_xy −0.05 太弱压不住。台阶前后不罚，不影响抬轮。
    smooth_flat_arm: float = 0.0
    smooth_flat_ang: float = 0.0
    # × 翻车：想象里是 1 − 软存活概率，真实里是 terminated（train_terrain 同口径补上）。go2w 用 −20；
    # 热启动任务原来没有翻车惩罚，默认 0。
    terminated: float = 0.0
    riser_gap: tuple[float, float] = (0.06, 0.16)  # 后轮心到前方立面的水平距离窗口（半径 0.1085）


def route_command_body(state: torch.Tensor, command_w: torch.Tensor) -> torch.Tensor:
    """preview_route_command：世界系 (vx, vy) 按车体航向转到车体系，yaw 分量原样。"""
    q = state[..., QUAT]
    yaw = torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2].square() + q[..., 3].square()))
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * command_w[..., 0] + s * command_w[..., 1],
                        -s * command_w[..., 0] + c * command_w[..., 1], command_w[..., 2]), dim=-1)


def proprio_observation(state, command_w, last_action) -> torch.Tensor:
    joint_pos = state[..., JPOS].clone()
    joint_pos[..., NUM_LEG_JOINTS:] = 0.0
    return torch.cat((
        state[..., ANG] * ANG_VEL_SCALE,
        projected_gravity(state[..., QUAT]),
        route_command_body(state, command_w),
        joint_pos,
        state[..., JVEL] * JOINT_VEL_SCALE,
        last_action,
    ), dim=-1).clamp(-OBS_CLIP, OBS_CLIP)


def _centered_tracking(command_xy, velocity_xy, std):
    # vision_preview.centered_velocity_tracking 的可导版本（不含 nan 处理）
    command_sq = command_xy.square().sum(-1)
    score = torch.exp(-(command_xy - velocity_xy).square().sum(-1) / std**2)
    rest = torch.exp(-command_sq / std**2)
    centered = ((score - rest) / (1.0 - rest).clamp_min(0.01)).clamp(-1.0, 1.0)
    return torch.where(command_sq >= 0.1**2, centered, score)


def _alignment(q):
    return (1.0 - 2.0 * (q[..., 2].square() + q[..., 3].square())).clamp(0.0, 1.0)


class PreviewTask:
    """learner 的任务接口：policy_obs / reward_terms / reward / survival / critic_extra。"""

    obs_dim = PROPRIO_DIM + 187

    def __init__(self, hmap, kin, scan_pattern, joint_soft_limits, cfg: PreviewRewardCfg | None = None,
                 exact_scan_fn=None, lane_center_y: torch.Tensor | None = None, lane_obs: bool = False):
        """joint_soft_limits: [8, 2] 腿关节软限位（相对默认位，默认位为 0）。
        exact_scan_fn(state) → [..., 187]：可选，用真实射线投射给出前向值（直通，梯度仍走双线性）。
        lane_obs=True：actor 观测末尾追加 1 维车道横向偏移（方案 B，见 lane_offset.py），需同时给 lane_center_y。"""
        if lane_obs and lane_center_y is None:
            raise ValueError("lane_obs 需要 lane_center_y")
        self.hmap, self.kin, self.pattern = hmap, kin, scan_pattern
        self.limits = joint_soft_limits
        self.cfg = cfg or PreviewRewardCfg()
        self.exact_scan_fn = exact_scan_fn
        # 各地形块车道中心的世界 y（升序）；回放状态不带 env id，按最近中心线归属
        self.lane_center_y = lane_center_y
        self.critic_extra_dim = terrain_map.RELIEF_DIM + (1 if lane_center_y is not None else 0)
        self.lane_obs = lane_obs
        self.base_obs_dim = PROPRIO_DIM + scan_pattern.shape[0]
        self.obs_dim = self.base_obs_dim + (1 if lane_obs else 0)

    # ---------------- 观测 ----------------
    def scan(self, state):
        smooth = terrain_map.height_scan(self.hmap, state, self.pattern)
        if self.exact_scan_fn is None:
            return smooth
        with torch.no_grad():
            exact = self.exact_scan_fn(state.reshape(-1, state.shape[-1])).reshape(smooth.shape)
        # 直通：前向用真实射线（台阶边不再是斜坡），反向用双线性插值的梯度。
        return smooth + (exact - smooth).detach()

    def policy_obs(self, state, command, last_action):
        parts = [proprio_observation(state, command, last_action), self.scan(state)]
        if self.lane_obs:
            # 与 env 侧 lane_offset_obs 同一公式；梯度经车体 y 穿过世界模型，actor 由此学到"往哪边打能回中线"
            parts.append((self.lateral_offset(state) * LANE_OBS_SCALE).clamp(-LANE_OBS_CLIP, LANE_OBS_CLIP)[..., None])
        return torch.cat(parts, dim=-1)

    def critic_extra(self, state):
        relief = terrain_map.relief_features(self.hmap, self.kin, state)
        if self.lane_center_y is None:
            return relief
        return torch.cat((relief, self.lateral_offset(state)[..., None]), dim=-1)

    def lateral_offset(self, state) -> torch.Tensor:
        """车体到最近车道中心线的带符号横向距离；块间距 8 m，验收出界 0.75 m，不会归错块。"""
        y = state[..., POS][..., 1]
        centers = self.lane_center_y.to(y.device)
        idx = torch.bucketize(y.detach(), centers).clamp(1, centers.numel() - 1)
        lo, hi = centers[idx - 1], centers[idx]
        nearest = torch.where((y - lo).abs() <= (hi - y).abs(), lo, hi)
        return y - nearest

    # ---------------- 奖励 ----------------
    @torch.no_grad()
    def event_priority(self, state) -> torch.Tensor:
        """[B] 0/1：任一轮前方 lookahead 内有 ≥ min_step 的台阶，或轮底已离开轮下地面（正在越级）。
        这些是世界模型最难、从零训练又最稀少的片段，供回放优先采样。"""
        step, clearance = self.step_ahead(state)
        return ((step >= self.cfg.min_step) | (clearance > 0.02)).any(-1).float()

    def anchor_weight(self, state):
        """[B] 锚定权重：后轮（rl, rr）前方 riser_gap 内有 ≥ min_step 的立面且轮底仍贴地时为 1 + riser_anchor。"""
        cfg = self.cfg
        if cfg.riser_anchor == 0.0:
            return None
        with torch.no_grad():
            jp = state[..., JPOS]
            wheels = self.kin.wheel_centers_world(state[..., POS], state[..., QUAT], jp[..., 0:4], jp[..., 4:8])[..., [1, 3], :]
            fwd = quat_apply(yaw_quat(state[..., QUAT]),
                             torch.tensor([1.0, 0.0, 0.0], device=state.device).expand(state.shape[:-1] + (3,)))[..., :2]
            ground = self.hmap.query(wheels[..., :2])
            lo, hi = cfg.riser_gap
            ahead = torch.linspace(lo, hi, 6, device=state.device)
            front = self.hmap.query(wheels[..., None, :2] + ahead[:, None] * fwd[..., None, None, :]).amax(-1)
            at_riser = ((front - ground) >= cfg.min_step) & ((wheels[..., 2] - WHEEL_RADIUS - ground) < 0.02)
            return 1.0 + cfg.riser_anchor * at_riser.any(-1).float()

    def step_ahead(self, state):
        """每个轮心前方 lookahead 内的最高地面与轮下地面之差 [..., 4]，以及轮心离地 [..., 4]。"""
        jp = state[..., JPOS]
        wheels = self.kin.wheel_centers_world(state[..., POS], state[..., QUAT], jp[..., 0:4], jp[..., 4:8])
        yq = yaw_quat(state[..., QUAT])
        fwd = quat_apply(yq, torch.tensor([1.0, 0.0, 0.0], device=state.device).expand(yq.shape[:-1] + (3,)))[..., :2]
        ahead = torch.tensor(self.cfg.lift_lookahead, device=state.device)
        ground_here = self.hmap.query(wheels[..., :2])
        ground_ahead = self.hmap.query(wheels[..., None, :2] + ahead[:, None] * fwd[..., None, None, :]).amax(-1)
        clearance = wheels[..., 2] - WHEEL_RADIUS - ground_here
        return ground_ahead - ground_here, clearance

    def reward_terms(self, state, next_state, command, action, last_action, acc_sq=None) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        q = next_state[..., QUAT]
        gravity = projected_gravity(q)
        upright = (-gravity[..., 2]).clamp(0.0, 0.7) / 0.7
        lin_b, ang_b = next_state[..., LIN], next_state[..., ANG]
        lin_w = quat_apply(q, lin_b)
        ang_w = quat_apply(q, ang_b)
        command_w = torch.stack((command[..., 0], torch.zeros_like(command[..., 0])), dim=-1)
        align = _alignment(q)
        track = _centered_tracking(command_w, lin_w[..., :2], cfg.track_lin_vel_std)
        track = (track.clamp_min(0.0) * align + track.clamp_max(0.0) - 0.25 * (1.0 - align)) * upright
        leg = next_state[..., JPOS][..., :NUM_LEG_JOINTS]
        out_of_limits = (self.limits[:, 0] - leg).clamp_min(0.0) + (leg - self.limits[:, 1]).clamp_min(0.0)
        step, clearance = self.step_ahead(next_state)
        has_step = torch.sigmoid((step - cfg.min_step) / 0.005)
        lift_frac = (clearance / (step + cfg.lift_margin).clamp_min(cfg.min_step)).clamp(0.0, 1.0)
        idle = (clearance - cfg.rest_clearance).clamp_min(0.0)
        return {
            "track_lin_vel": cfg.track_lin_vel * track,
            "lin_vel_z": cfg.lin_vel_z * lin_b[..., 2].square() * upright,
            "ang_vel_xy": cfg.ang_vel_xy * ang_b[..., :2].square().sum(-1) * upright,
            "action_rate": cfg.action_rate * (action - last_action).square().sum(-1),
            "steering_deviation": cfg.steering_deviation * next_state[..., JPOS][..., STEERING].abs().sum(-1),
            "joint_pos_limits": cfg.joint_pos_limits * out_of_limits.sum(-1),
            "flat_orientation": cfg.flat_orientation * gravity[..., :2].square().sum(-1),
            "yaw_rate": cfg.yaw_rate * ang_w[..., 2].square(),
            "wheel_lift": cfg.wheel_lift * (has_step * lift_frac).mean(-1) * upright,
            "idle_lift": cfg.idle_lift * ((1.0 - has_step) * idle).sum(-1),
            **({"terminated": cfg.terminated * (1.0 - self.survival(next_state))} if cfg.terminated != 0.0 else {}),
            **self.shaping_terms(next_state),
        }

    def shaping_terms(self, state) -> dict[str, torch.Tensor]:
        """只依赖状态的直行/姿态塑形；train_terrain 也用它补到真实回报上，使 critic 与想象目标一致。
        不用 acos/atan2：正立或航向为 0 处梯度为无穷（见 BPTT acos NaN 教训）。"""
        cfg = self.cfg
        if cfg.heading == 0.0 and cfg.lateral_vel == 0.0 and cfg.tilt_soft == 0.0 and cfg.wheel_heading == 0.0 \
                and cfg.lateral_pos == 0.0 and cfg.smooth_flat_arm == 0.0 and cfg.smooth_flat_ang == 0.0:
            return {}
        q = state[..., QUAT]
        fwd = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=state.device).expand(q.shape[:-1] + (3,)))
        norm = (fwd[..., :2].square().sum(-1) + 1e-6).sqrt()
        sin_yaw, cos_yaw = fwd[..., 1] / norm, fwd[..., 0] / norm
        sin2_yaw = sin_yaw.square()
        # 轮子世界系朝向 = 航向 + 转向角（绕车体 z，逆时针为正）；sin(ψ+δ) 展开避免 atan2
        steer = state[..., JPOS][..., STEERING]
        sin_wheel = sin_yaw[..., None] * steer.cos() + cos_yaw[..., None] * steer.sin()
        vy_w = quat_apply(q, state[..., LIN])[..., 1]
        cos_tilt = -projected_gravity(q)[..., 2]
        return {
            "heading": cfg.heading * sin2_yaw,
            "lateral_vel": cfg.lateral_vel * vy_w.square(),
            "tilt_soft": cfg.tilt_soft * (math.cos(cfg.tilt_soft_start) - cos_tilt).clamp_min(0.0),
            "wheel_heading": cfg.wheel_heading * sin_wheel.square().mean(-1),
            **({"lateral_pos": cfg.lateral_pos * self.lateral_offset(state).square()}
               if cfg.lateral_pos != 0.0 else {}),
            **(self._smooth_flat(state) if cfg.smooth_flat_arm != 0.0 or cfg.smooth_flat_ang != 0.0 else {}),
        }

    def _smooth_flat(self, state) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        step, _ = self.step_ahead(state)
        # 任一轮前方有台阶（或后轮贴立面）就不罚；门控本身不求梯度，避免策略靠"看起来像台阶"逃罚
        flat = (1.0 - torch.sigmoid((step.amax(-1) - cfg.min_step) / 0.005)).detach()
        arm_vel = state[..., JVEL][..., 0:4]
        return {
            "smooth_flat_arm": cfg.smooth_flat_arm * flat * arm_vel.square().sum(-1),
            "smooth_flat_ang": cfg.smooth_flat_ang * flat * state[..., ANG][..., :2].square().sum(-1),
        }

    def shaping_reward(self, state) -> torch.Tensor:
        terms = self.shaping_terms(state)
        if not terms:
            return torch.zeros(state.shape[:-1], device=state.device)
        return torch.stack(list(terms.values()), dim=-1).sum(-1) * CONTROL_DT

    def reward(self, state, next_state, command, action, last_action, acc_sq=None) -> torch.Tensor:
        terms = self.reward_terms(state, next_state, command, action, last_action)
        return torch.stack(list(terms.values()), dim=-1).sum(-1) * CONTROL_DT

    def survival(self, state) -> torch.Tensor:
        cos_tilt = -projected_gravity(state[..., QUAT])[..., 2]
        return torch.sigmoid((cos_tilt - math.cos(self.cfg.tilt_limit))
                             / (self.cfg.tilt_temperature * math.sin(self.cfg.tilt_limit)))
