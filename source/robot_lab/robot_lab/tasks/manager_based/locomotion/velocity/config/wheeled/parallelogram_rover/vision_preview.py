"""只用于新视觉任务的前瞻抬轮奖励；特权地形不能进入 student Actor。

@author pony
@date 2026-09-30
@version v0.4.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 8/12/16cm爬阶课程、单侧主动完成与新前缘重新武装，路线通行独立计奖。
    - (2026-10-13): 稳定接近建立减载基线，保留首次尝试信用与悬架行程，按左右起抬再完整动作晋级。
    - (2026-10-12): 补齐5mm前的窗口净空引导，单障碍通行奖励/晋级绑定完整动作，左右窄障碍依次过关。
    - (2026-10-12): 抬轮运动门对齐世界路线，删除闲置时机配置，碰撞/漏抬缓存按诊断语义命名。
    - (2026-10-12): 合并完整动作结算，删除重复时机/落轮及漏抬/碰撞罚分，增加一次性通行奖励。
    - (2026-10-12): Preview专属路线晋级与阶段地形抽样，侧移或原地转向不能升级。
    - (2026-10-11): 接地减载提供封顶引导，速度跟踪以静止为零基线；清理计时罚项、重复状态和预瞄前误锁。
    - (2026-10-11): 真实离地锁存起抬位置，分离高度达标，补全宽碰撞/漏抬与近场扫描。
    - (2026-10-10): 感知与10cm动作窗口分离；时机奖延后至落轮，补早抬/失败重抬惩罚。
    - (2026-10-10): 单障碍奖励用轮迹水平射线精定位前缘，不将10cm扫描偏移当真实距离。
    - (2026-10-09): 窄障碍按接地承重建立基线，悬架收缩不再阻止左右目标轮卸载计分。
    - (2026-10-09): 按required_mask发放窄障碍目标轮完成奖励，保留全宽同前缘双轮约束。
    - (2026-10-09): 窄障碍按提前减载与真实净空计事件，加入阻挡碰撞惩罚和分地形证据。
    - (2026-10-10): 独立记录提前动作、持续保持和稳定落轮，取消提前落轮收益并抑制窗外跺脚。
    - v0.3.0 (2026-10-08): 抬轮进度/事件只发给前轮，后轮仍计悬轮惩罚；双前轮完成额度1.0→1.5。
    - v0.2.0 (2026-10-08): 区分几何势能与时序奖励，封顶首次障碍事件并允许顺序抬轮。
    - v0.1.0 (2026-09-30): 增加轮迹预瞄、URDF 中立几何和防跳跃/停滞刷分门。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

WHEEL_NAMES = ("fl_wheel", "rl_wheel", "fr_wheel", "rr_wheel")
# 提前抬轮目标只针对前轴。四轮顺序与URDF共享不可改，这里显式给出计分子集，
# 防止后轮反应替代前轮准备；后轮仍参与几何检测，并受非窗口悬轮惩罚约束。
FRONT_WHEEL_INDICES = (0, 2)
STEP_TERRAIN_KINDS = {"single_step": 0, "single_step_left": 1, "single_step_right": 2}
STAIRS_TERRAIN_KIND = 3
PREVIEW_STAGE_WEIGHTS = (
    {"flat": 0.10, "single_step": 0.25, "single_step_left": 0.20, "single_step_right": 0.20,
     "pyramid_stairs_inv": 0.20, "boxes": 0.05},
    {"flat": 0.05, "single_step": 0.20, "single_step_left": 0.15, "single_step_right": 0.15,
     "pyramid_stairs_inv": 0.40, "boxes": 0.05},
    {"flat": 0.05, "single_step": 0.10, "single_step_left": 0.10, "single_step_right": 0.10,
     "pyramid_stairs_inv": 0.55, "boxes": 0.10},
)


def stairs_profile(difficulty, height_levels=(0.08, 0.12, 0.16), step_counts=(2, 3, 4),
                   first_edge_m=1.20, step_width_m=0.40):
    """训练、play和独立验收共用三档几何；终点留出后轮登顶及稳定承重的空间。"""
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("楼梯难度必须位于[0,1]")
    if (len(height_levels) != 3 or len(step_counts) != 3
            or any(not math.isfinite(h) or h <= 0 for h in height_levels)
            or any(type(n) is not int or n < 1 for n in step_counts)
            or not 0 < first_edge_m or not 0 < step_width_m):
        raise ValueError("楼梯必须提供三档正高度、级数及有效接近距离")
    stage = min(int(difficulty * 3), 2)
    height, count = height_levels[stage], step_counts[stage]
    last_edge = first_edge_m + (count - 1) * step_width_m
    return {"stage": stage + 1, "step_height_m": height, "num_steps": count,
            "first_edge_m": first_edge_m, "last_edge_m": last_edge,
            "total_height_m": height * count, "target_distance_m": max(2.5, last_edge + 0.60)}


def preview_route_targets(env):
    """按本环境实际地形行和种类给出终点；评测单行固定难度沿用同一几何。"""
    terrain = env.scene.terrain
    cfg = terrain.cfg.terrain_generator
    names = terrain_column_names(tuple(cfg.sub_terrains),
                                 tuple(c.proportion for c in cfg.sub_terrains.values()),
                                 cfg.num_cols, curriculum=cfg.curriculum)
    levels = terrain.terrain_levels
    targets = torch.full_like(levels, 2.5, dtype=torch.float)
    if "pyramid_stairs_inv" not in cfg.sub_terrains:
        return targets
    stairs = cfg.sub_terrains["pyramid_stairs_inv"]
    if not hasattr(stairs, "height_levels"):
        return targets
    cols = torch.tensor([i for i, name in enumerate(names) if name == "pyramid_stairs_inv"], device=levels.device)
    low, high = cfg.difficulty_range
    difficulty = low + (levels.float() + 0.5) / cfg.num_rows * (high - low)
    stage = (difficulty * 3).long().clamp(0, 2)
    counts = levels.new_tensor(stairs.step_counts)[stage]
    stair_targets = (stairs.first_edge_m + (counts - 1) * stairs.step_width + 0.60).clamp_min(2.5)
    return torch.where(torch.isin(terrain.terrain_types, cols), stair_targets, targets)

# 仍为17×11点，传感器前移0.30m后在车体x=0.50–0.62m每2cm采样。
# 这是Teacher高度图，不是给Student增加真实前缘距离；保留近后轮及远场覆盖。
PREVIEW_SCAN_X = (-0.80, -0.60, -0.40, -0.20, 0.0, 0.10, 0.16,
                  0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.40, 0.60, 0.80)
PREVIEW_SCAN_Y = tuple(i / 10 for i in range(-5, 6))


def near_field_height_pattern(cfg, device):
    """保持xy展平顺序与187点接口；非均匀坐标必须随观测合同保存。"""
    x, y = torch.meshgrid(torch.tensor(cfg.x_positions, device=device),
                          torch.tensor(cfg.y_positions, device=device), indexing="xy")
    starts = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()), -1)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.
    return starts, directions


def terrain_column_names(names, proportions, num_cols, *, curriculum):
    """按Isaac Lab课程生成器的列边界还原地形名。"""
    if len(names) != len(proportions) or not names or num_cols < 1:
        raise ValueError("地形名称、比例或列数无效")
    if any(not math.isfinite(p) or p < 0 for p in proportions) or sum(proportions) <= 0:
        raise ValueError("地形比例必须有限、非负且总和为正")
    if len(names) == 1:
        return tuple(names[0] for _ in range(num_cols))
    if not curriculum:
        raise ValueError("Preview混合地形需要curriculum=True以准确记录地形类别，不能猜测随机列")
    bounds = torch.tensor(proportions, dtype=torch.float64).cumsum(0) / sum(proportions)
    positions = torch.arange(num_cols, dtype=torch.float64) / num_cols + 0.001
    indices = torch.searchsorted(bounds, positions, right=True).tolist()
    return tuple(names[index] for index in indices)


def terrain_column_kinds(names, proportions, num_cols, *, curriculum, device="cpu"):
    """严格匹配Isaac Lab列分配；非单障碍地形标记为-1。"""
    columns = terrain_column_names(names, proportions, num_cols, curriculum=curriculum)
    return torch.tensor([STAIRS_TERRAIN_KIND if name == "pyramid_stairs_inv"
                         else STEP_TERRAIN_KINDS.get(name, -1) for name in columns], device=device)


def preview_route_outcome(positions_w, origins_w, upright, forward_alignment=None, target_distance_m=2.5):
    """只有沿地形+X穿过障碍且留在行车带内才晋级。"""
    progress = positions_w[:, 0] - origins_w[:, 0]
    lateral = (positions_w[:, 1] - origins_w[:, 1]).abs()
    aligned = torch.ones_like(upright) if forward_alignment is None else forward_alignment >= 0.8
    passed = (progress >= target_distance_m) & (lateral <= 0.15) & upright & aligned
    failed = (progress < 0.8) | (lateral > 0.75) | ~upright
    if forward_alignment is not None:
        failed |= forward_alignment < 0.5
    return passed, failed & ~passed


def sample_preview_terrain_columns(levels, column_names):
    """行数控制难度，阶段控制可抽到的障碍种类；每种地形先按类别权重抽样。"""
    if not column_names or levels.ndim != 1 or ((levels < 0) | (levels > 8)).any():
        raise ValueError("Preview课程需要9行及有效地形列")
    stage = (levels >= 3).long() + (levels >= 6).long()
    weights = torch.zeros((3, len(column_names)), device=levels.device)
    for stage_index, families in enumerate(PREVIEW_STAGE_WEIGHTS):
        for name, weight in families.items():
            columns = [index for index, column_name in enumerate(column_names) if column_name == name]
            if not columns:
                raise ValueError(f"Preview课程缺少地形列: {name}")
            weights[stage_index, columns] = weight / len(columns)
    return torch.multinomial(weights[stage], 1).squeeze(-1)


def preview_route_promotion_mask(levels, terrain_types, column_names):
    """通行决定升级；跨8→12→16cm阶段必须真的通过本档楼梯，平地不能晋级。"""
    if levels.shape != terrain_types.shape:
        raise ValueError("课程行与地形列必须逐环境对齐")
    obstacles = torch.tensor([i for i, name in enumerate(column_names) if name != "flat"], device=levels.device)
    stairs = torch.tensor([i for i, name in enumerate(column_names) if name == "pyramid_stairs_inv"],
                          device=levels.device)
    boundary = (levels == 2) | (levels == 5)
    return torch.isin(terrain_types, obstacles) & (~boundary | torch.isin(terrain_types, stairs))


def narrow_support_mask(wheel_upward_force_n, robot_weight_n):
    """窄障碍动作允许三轮或两轮对角承重；同侧两轮和全车腾空不能提供卸载信用。"""
    supported = wheel_upward_force_n > 1.0
    diagonal = (supported[:, 0] & supported[:, 3]) | (supported[:, 1] & supported[:, 2])
    enough_wheels = (supported.sum(-1) >= 3) | ((supported.sum(-1) == 2) & diagonal)
    valid = torch.isfinite(wheel_upward_force_n).all(-1) & torch.isfinite(robot_weight_n)
    return valid & (robot_weight_n > 0) & enough_wheels & (wheel_upward_force_n.sum(-1) >= 0.5 * robot_weight_n)


def step_lateral_bounds(center_y: float, width: float) -> tuple[float, float]:
    """World-aligned step lane bounds relative to its terrain origin."""
    if not math.isfinite(center_y) or not math.isfinite(width) or width <= 0:
        raise ValueError("单障碍横向位置和宽度必须有限且宽度为正")
    return center_y - width / 2, center_y + width / 2


def step_front_x_for_difficulty(difficulty: float, front_range: tuple[float, float], seed: int | None) -> float:
    """Derive a per-tile step position without consuming the terrain generator's RNG."""
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("单障碍难度必须位于[0,1]")
    near, far = front_range
    if not all(math.isfinite(value) for value in front_range) or not 0.0 < near < far:
        raise ValueError("单障碍前缘范围必须为递增正数")
    digest = hashlib.blake2b(f"{seed}:{difficulty:.9f}".encode(), digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    return near + fraction * (far - near)


@dataclass(frozen=True)
class PreviewCfg:
    """初始训练超参数，不是实测硬件能力或已达到的成功率。"""

    lookahead_m: float = 0.65
    min_advance_m: float = 0.08
    # 感知可提前到 65cm，但真正的主动抬轮只在障碍前 8--14cm 结算。
    # 两个距离分开，避免策略因为“看见障碍”就过早抬腿。
    action_max_advance_m: float = 0.14
    lane_half_width_m: float = 0.09
    ground_sample_radius_m: float = 0.13
    min_rise_m: float = 0.035
    max_rise_m: float = 0.18
    clearance_margin_m: float = 0.01
    min_lift_m: float = 0.02
    takeoff_clearance_m: float = 0.005
    min_velocity_m_s: float = 0.05
    min_command_m_s: float = 0.10
    min_upright_cos: float = 0.85
    min_support_wheels: int = 2
    low_lift_threshold_m: float = 0.01
    scan_edge_margin_m: float = 0.10
    progress_credit: float = 1.0
    wheel_event_credit: float = 0.50
    # 准备阶段可以接地减载，但不能提前离地；只使用进度总额度的一部分，且不重复付费。
    preparation_max_advance_m: float = 0.25
    preparation_fraction: float = 0.35
    baseline_settle_s: float = 0.40
    baseline_stable_s: float = 0.12
    # 只限制远处接地深蹲，不锁关节、不干扰25cm内的协同动作。
    approach_max_contraction_m: float = 0.04
    approach_stance_penalty: float = 0.50
    # 任一目标前轮保持到前缘并稳定落轮才完成；一抬即放不能领取这项收益。
    front_completion_credit: float = 2.0
    route_completion_credit: float = 4.0
    idle_lift_penalty: float = 0.50
    prelift_hold_s: float = 0.06
    release_hold_s: float = 0.06
    # 保留初学有效起抬的信用；失败锁存及一次性额度阻止重复跺脚，完成大奖仍不给。
    hold_violation_penalty: float = 0.25
    passage_margin_m: float = 0.45
    # 窄障碍的可实现验收以目标轮卸载为主；这里的比例和绝对力门槛
    # 用于训练事件与评测保持同一口径，不把轻微接触抖动算成避障。
    narrow_load_drop_fraction: float = 0.35
    narrow_min_force_drop_n: float = 3.0
    narrow_min_clearance_m: float = 0.005
    narrow_hold_s: float = 0.06
    collision_resisting_force_n: float = 5.0
    new_edge_min_spacing_m: float = 0.20
    rearm_ground_hold_s: float = 0.06

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        if (self.min_advance_m >= self.action_max_advance_m
                or self.action_max_advance_m > self.lookahead_m
                or self.min_rise_m >= self.max_rise_m):
            raise ValueError("预瞄距离和高度上下界必须递增")
        if not 0 < self.min_upright_cos < 1 or type(self.min_support_wheels) is not int:
            raise ValueError("正立门或承重轮数无效")
        if not 1 <= self.min_support_wheels <= 4:
            raise ValueError("承重轮数必须位于1到4")
        if self.low_lift_threshold_m >= self.min_lift_m:
            raise ValueError("低位武装阈值必须低于抬起阈值")
        if self.takeoff_clearance_m >= self.min_lift_m:
            raise ValueError("真实离地阈值必须低于高度达标阈值")
        if not self.action_max_advance_m <= self.preparation_max_advance_m <= self.lookahead_m:
            raise ValueError("接地准备距离须覆盖起抬窗口且位于预瞄范围内")
        if self.preparation_fraction >= 1:
            raise ValueError("准备额度比例必须小于1")
        if self.narrow_load_drop_fraction >= 1 or self.narrow_min_clearance_m >= self.min_lift_m:
            raise ValueError("窄障碍减载比例必须小于1、净空门必须低于全宽门")
        if self.hold_violation_penalty >= self.wheel_event_credit:
            raise ValueError("保持失败不能抹掉首次有效起抬的全部事件信用")
        if self.front_completion_credit <= self.progress_credit + self.wheel_event_credit:
            raise ValueError("单侧完整动作收益须高于单轮瞬时事件收益")


def centered_velocity_tracking(command_xy, velocity_xy, std):
    """行驶命令下静止为0、准确跟踪为1；不另设停滞倒计时，零速指令保留原跟踪语义。"""
    if not math.isfinite(std) or std <= 0:
        raise ValueError("速度跟踪std必须为有限正数")
    valid = torch.isfinite(command_xy).all(-1) & torch.isfinite(velocity_xy).all(-1)
    command, velocity = torch.nan_to_num(command_xy), torch.nan_to_num(velocity_xy)
    command_squared = command.square().sum(-1)
    score = torch.exp(-(command - velocity).square().sum(-1) / std**2)
    rest = torch.exp(-command_squared / std**2)
    # 归一化维持满分幅度；有界负分避免小命令下误差放大，不将倒退视为有用的前进。
    centered = ((score - rest) / (1. - rest).clamp_min(0.01)).clamp(-1., 1.)
    result = torch.where(command_squared >= 0.1**2, centered, score)
    return torch.where(valid, result, 0.)


def rotate_vectors(quat: torch.Tensor, vectors: torch.Tensor, *, inverse=False) -> torch.Tensor:
    """对齐批量维度的 wxyz 旋转；调用方负责提供单位四元数。"""
    xyz = -quat[..., 1:] if inverse else quat[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(xyz, vectors, dim=-1)
    return vectors + quat[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def preview_measurements(
    *, ray_hits_w, root_pos_w, root_quat_w, wheel_centers_w, neutral_centers_b,
    wheel_radii, command_x, velocity_x, supported_wheels, nonwheel_contact, cfg: PreviewCfg = PreviewCfg(),
    ground_hits_w=None,
):
    """按实际轮迹检测前方正高差；忽略侧边障碍、未知地面与超过课程范围的高差。

    轮子相对车体的抬升与相对地面的净空必须同时增加，防止整车弹跳或俯仰冒充主动抬轮。
    地形只用于训练奖励和评测，函数不生成 Actor 观测。O(N×4×ray_count)。
    """
    n = root_pos_w.shape[0]
    if root_pos_w.shape != (n, 3) or root_quat_w.shape != (n, 4) or wheel_centers_w.shape != (n, 4, 3):
        raise ValueError("机器人和四轮状态批量维度不匹配")
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[0] != n or ray_hits_w.shape[-1] != 3:
        raise ValueError("ray_hits_w必须为[N,R,3]")
    if neutral_centers_b.shape != (4, 3) or wheel_radii.shape != (4,):
        raise ValueError("中立轮心和半径必须遵循固定四轮顺序")
    ray_valid = torch.isfinite(ray_hits_w).all(-1)
    root_valid = torch.isfinite(root_pos_w).all(-1) & torch.isfinite(root_quat_w).all(-1)
    root_valid &= (root_quat_w.square().sum(-1) - 1.0).abs() < 1e-3
    root_valid &= torch.isfinite(wheel_centers_w).all(dim=(-1, -2))
    root_valid &= torch.isfinite(command_x) & torch.isfinite(velocity_x)
    # 非法四元数不能用最大浮点数代替inf后参与叉积；NaN乘零仍会污染PPO奖励。
    root = torch.where(root_valid[:, None], root_pos_w, torch.zeros_like(root_pos_w))
    identity = root_quat_w.new_tensor([1.0, 0.0, 0.0, 0.0])
    quat = torch.where(root_valid[:, None], root_quat_w, identity)
    wheels = torch.where(root_valid[:, None, None], wheel_centers_w, torch.zeros_like(wheel_centers_w))
    rays = torch.nan_to_num(ray_hits_w, nan=0.0, posinf=0.0, neginf=0.0)
    yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                      1 - 2 * (quat[:, 2].square() + quat[:, 3].square()))
    c, s = yaw.cos(), yaw.sin()
    delta = rays[:, None, :, :2] - wheels[:, :, None, :2]
    dx = delta[..., 0] * c[:, None, None] + delta[..., 1] * s[:, None, None]
    dy = -delta[..., 0] * s[:, None, None] + delta[..., 1] * c[:, None, None]
    distances = delta.square().sum(-1).masked_fill(~ray_valid[:, None], float("inf"))
    ground_distance, ground_index = distances.min(-1)
    ground_z = rays[:, :, 2].gather(1, ground_index)
    ground_valid = ground_distance <= cfg.ground_sample_radius_m**2
    if ground_hits_w is not None:
        if ground_hits_w.shape != wheel_centers_w.shape:
            raise ValueError("局部地面射线须与四轮维度一致")
        ground_valid = torch.isfinite(ground_hits_w).all(-1)
        ground_z = torch.nan_to_num(ground_hits_w[..., 2], nan=0., posinf=0., neginf=0.)
    advance = dx - wheel_radii[None, :, None]
    rises = rays[:, None, :, 2] - ground_z[:, :, None]
    height_candidates = (
        ray_valid[:, None] & (dy.abs() <= cfg.lane_half_width_m)
        & (advance >= -cfg.passage_margin_m)
        & (rises >= cfg.min_rise_m) & (rises <= cfg.max_rise_m)
    )
    height_candidates &= ground_valid[:, :, None]
    unbounded_advance_m = advance.masked_fill(~height_candidates, float("inf")).min(-1).values
    candidates = height_candidates & (advance >= cfg.min_advance_m) & (advance <= cfg.lookahead_m)
    advance_m, next_index = advance.masked_fill(~candidates, float("inf")).min(-1)
    preview = candidates.any(-1) & ground_valid & root_valid[:, None]
    rise_m = rises.gather(2, next_index[:, :, None]).squeeze(-1)
    target = (rise_m + cfg.clearance_margin_m).clamp(cfg.min_lift_m, cfg.max_rise_m + cfg.clearance_margin_m)
    body_centers = rotate_vectors(quat[:, None].expand(-1, 4, -1), wheels - root[:, None], inverse=True)
    articulation_lift = body_centers[:, :, 2] - neutral_centers_b[None, :, 2]
    clearance = wheels[:, :, 2] - wheel_radii[None] - ground_z
    up_cos = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
    eligible_gates = {
        "valid_state": root_valid,
        "upright": up_cos >= cfg.min_upright_cos,
        "supported": supported_wheels >= cfg.min_support_wheels,
        "body_clear": ~nonwheel_contact,
        "forward_command": command_x > cfg.min_command_m_s,
        "forward_motion": velocity_x > cfg.min_velocity_m_s,
    }
    eligible = torch.stack(tuple(eligible_gates.values()), dim=-1).all(-1)
    lift_score = torch.minimum((articulation_lift / target).clamp(0, 1), (clearance / target).clamp(0, 1))
    progress = (torch.nan_to_num(velocity_x) / torch.nan_to_num(command_x).clamp_min(cfg.min_command_m_s)).clamp(0, 1)
    prelift = preview & eligible[:, None] & (articulation_lift >= cfg.min_lift_m) & (clearance >= cfg.min_lift_m)

    # 仅供诊断的瞬时几何势能，不能直接作为训练奖励：静态悬轮也会有此值。
    potential = ((lift_score * preview + 0.25 * prelift).sum(-1) / 2.0) * progress * eligible
    heading = torch.stack((c, s), dim=-1)
    obstacle_xy = rays[..., :2].gather(1, next_index[:, :, None].expand(-1, -1, 2))
    # 扫描点位于障碍顶面而非精确前缘；向前缘外保守退一个栅格，避免碰后计分。
    obstacle_xy = obstacle_xy - cfg.scan_edge_margin_m * heading[:, None]
    return {
        "potential": potential, "preview": preview, "prelift": prelift, "eligible": eligible,
        "eligible_gates": eligible_gates, "ground_valid": ground_valid,
        "advance_m": torch.where(preview, advance_m, torch.zeros_like(advance_m)),
        # 诊断用未截断距离：有效预瞄窗口外也保留最近高差，避免把“尚未看到”显示成0米。
        "unbounded_advance_m": torch.where(
            torch.isfinite(unbounded_advance_m), unbounded_advance_m, torch.zeros_like(unbounded_advance_m),
        ),
        "rise_m": torch.where(preview, rise_m, torch.zeros_like(rise_m)),
        "articulation_lift_m": articulation_lift, "clearance_m": clearance,
        "obstacle_xy": obstacle_xy, "heading_xy": heading, "target_lift_m": target,
        "wheel_centers_w": wheels, "wheel_radii": wheel_radii,
    }


def refine_step_edges(sample, edge_hits_w, cfg=PreviewCfg()):
    """单障碍及楼梯以真实轮迹的水平射线定位竖直前缘，不改Actor输入。

    稀疏高度扫描仍决定可处理的正高差；额外每轮一条射线消除扫描栅格的10cm偏移。
    单障碍/楼梯射线未命中时拒绝新事件，不能把未知距离伪装为10cm；方块保留扫描估计。
    """
    centers = sample["wheel_centers_w"]
    if edge_hits_w.shape != centers.shape:
        raise ValueError("前缘射线必须与四轮中心具有相同维度")
    step = (sample["terrain_kind"] >= 0)[:, None]
    valid = torch.isfinite(edge_hits_w).all(-1) & sample["ground_valid"]
    valid &= sample["eligible_gates"]["valid_state"][:, None]
    delta = torch.nan_to_num(edge_hits_w[..., :2]) - centers[..., :2]
    lead = (delta * sample["heading_xy"][:, None]).sum(-1) - sample["wheel_radii"][None]
    valid &= torch.isfinite(lead)
    detected = valid & (lead >= cfg.min_advance_m) & (lead <= cfg.lookahead_m)
    sample["preview"] &= ~step | detected
    sample["prelift"] &= sample["preview"]
    sample["obstacle_xy"] = torch.where((step & valid)[..., None], edge_hits_w[..., :2], sample["obstacle_xy"])
    sample["advance_m"] = torch.where(step & sample["preview"], lead, sample["advance_m"])
    sample["advance_m"] = torch.where(sample["preview"], sample["advance_m"], 0.)
    sample["edge_distance_m"] = torch.where(step & valid, lead, float("nan"))
    sample["edge_distance_valid"] = step & valid
    return sample


class PreliftRewardState:
    """每个前缘每轮一次前瞻尝试；楼梯通过旧前缘且重新承重后处理新一级。

    只保存训练用状态，不增加Actor输入。两个前轮的尝试独立记忆，允许一个
    已经落轮后另一个才抬起；一次失败尝试不能通过反复抬落重新领取奖励。
    新回合必须显式reset，后退或扫描消失不能重领。
    """

    def __init__(self, num_envs, device, cfg=PreviewCfg()):
        self.cfg = cfg
        self.seen = torch.zeros((num_envs, 4), dtype=torch.bool, device=device)
        self.low_seen = torch.zeros_like(self.seen)
        self.contacted = torch.zeros_like(self.seen)
        self.completed = torch.zeros_like(self.seen)
        self.prelift_active = torch.zeros_like(self.seen)
        self.passed_front_edge = torch.zeros_like(self.seen)
        self.release_paid = torch.zeros_like(self.seen)
        self.hold_steps = torch.zeros_like(self.seen, dtype=torch.long)
        self.hold_broken = torch.zeros_like(self.seen)
        self.release_steps = torch.zeros_like(self.seen, dtype=torch.long)
        self.best = torch.zeros((num_envs, 4), device=device)
        self.early_lift_seen = torch.zeros_like(self.seen)
        self.takeoff_seen = torch.zeros_like(self.seen)
        self.takeoff_valid = torch.zeros_like(self.seen)
        self.takeoff_lead = torch.full_like(self.best, float("nan"))
        self.height_lead = torch.full_like(self.best, float("nan"))
        self.missed_seen = torch.zeros_like(self.seen)
        self.edge_xy = torch.zeros((num_envs, 4, 2), device=device)
        self.heading = torch.zeros_like(self.edge_xy)
        self.approach_ground_z = torch.zeros_like(self.best)
        self.target = torch.ones_like(self.best)
        self.front_paid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.route_paid = torch.zeros_like(self.front_paid)
        self.front_mask = torch.zeros((num_envs, 4), dtype=torch.bool, device=device)
        self.front_mask[:, FRONT_WHEEL_INDICES] = True
        self.required_mask = self.front_mask.clone()
        self.narrow_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.baseline_force = torch.zeros_like(self.best)
        self.baseline_samples = torch.zeros_like(self.best, dtype=torch.long)
        self.baseline_locked = torch.zeros_like(self.seen)
        self.elapsed_s = torch.zeros(num_envs, device=device)
        self.baseline_history = None
        self.unload_steps = torch.zeros_like(self.best, dtype=torch.long)
        self.unprepared_collision_seen = torch.zeros_like(self.completed)
        self.terrain_kind = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.preparation_best = torch.zeros_like(self.best)
        self.rearm_steps = torch.zeros_like(self.hold_steps)
        self.completion_edge_x = torch.full((num_envs,), -float("inf"), device=device)
        self.route_hold_s = torch.zeros(num_envs, device=device)
        self.stair_opportunities = torch.zeros_like(self.hold_steps)
        self.stair_events = torch.zeros_like(self.hold_steps)
        self.stair_completions = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buffer in (self.seen, self.low_seen, self.contacted, self.completed,
                       self.prelift_active, self.passed_front_edge, self.release_paid,
                       self.hold_steps, self.hold_broken, self.release_steps, self.best,
                       self.early_lift_seen, self.takeoff_seen, self.takeoff_valid,
                       self.missed_seen,
                       self.edge_xy, self.heading, self.approach_ground_z, self.front_paid, self.route_paid,
                       self.baseline_force, self.baseline_samples, self.baseline_locked, self.elapsed_s,
                       self.unload_steps, self.unprepared_collision_seen,
                       self.preparation_best, self.rearm_steps, self.route_hold_s,
                       self.stair_opportunities, self.stair_events, self.stair_completions):
            buffer[ids] = 0
        self.target[ids] = 1
        self.narrow_mask[ids] = False
        self.required_mask[ids] = self.front_mask[ids]
        self.terrain_kind[ids] = -1
        self.takeoff_lead[ids] = float("nan")
        self.height_lead[ids] = float("nan")
        self.completion_edge_x[ids] = -float("inf")
        if self.baseline_history is not None:
            self.baseline_history[ids] = 0

    def _rearm_wheels(self, mask):
        """仅清除新前缘的轮缓存；路线总额、前缘去重和回合统计不能跟随轮子清空。"""
        for buffer in (self.seen, self.low_seen, self.contacted, self.completed, self.prelift_active,
                       self.passed_front_edge, self.release_paid, self.hold_steps, self.hold_broken,
                       self.release_steps, self.best, self.early_lift_seen, self.takeoff_seen,
                       self.takeoff_valid, self.missed_seen, self.approach_ground_z,
                       self.baseline_force, self.baseline_samples, self.baseline_locked, self.unload_steps,
                       self.unprepared_collision_seen, self.preparation_best, self.rearm_steps):
            buffer[mask] = 0
        self.takeoff_lead[mask] = self.height_lead[mask] = float("nan")
        if self.baseline_history is not None:
            self.baseline_history[mask] = 0

    def update(self, sample):
        cfg = self.cfg
        dt = sample.get("control_dt_s", 0.02)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("控制周期必须为有限正数")
        self.elapsed_s += dt
        baseline_steps = math.ceil(cfg.baseline_stable_s / dt - 1e-9)
        if self.baseline_history is None:
            self.baseline_history = self.best.new_zeros((*self.best.shape, baseline_steps))
        elif self.baseline_history.shape[-1] != baseline_steps:
            raise ValueError("基线采样期间不能改变控制周期")
        terrain_kind = sample.get("terrain_kind")
        if terrain_kind is not None:
            terrain_kind = terrain_kind.to(self.narrow_mask.device).view(-1)
            # 种类在首次前轮检测后锁存，跨列或后轮后续检测不能改变本回合目标。
            acquire = ~self.seen[:, FRONT_WHEEL_INDICES].any(-1)
            self.terrain_kind[acquire] = terrain_kind[acquire]
        left, right = self.terrain_kind == 1, self.terrain_kind == 2
        self.narrow_mask = left | right
        self.required_mask = self.front_mask.clone()
        self.required_mask[left, 2] = False
        self.required_mask[right, 0] = False
        force = sample.get("wheel_support_force_n", torch.full_like(self.best, float("nan")))
        wheel_x = sample["wheel_centers_w"][..., 0]
        new_edge = sample["obstacle_xy"][..., 0]
        # 无论上一轮主动还是被动上阶，都允许面对下一级。必须越过旧前缘、
        # 在顶面重新承重并看见更前方的新前缘；回退、扫描抖动和同阶跺脚不能重新武装。
        can_rearm = (
            (self.terrain_kind == STAIRS_TERRAIN_KIND)[:, None] & self.seen & sample["preview"]
            & (new_edge > self.edge_xy[..., 0] + cfg.new_edge_min_spacing_m)
            & (wheel_x > self.edge_xy[..., 0] + 0.10)
            & torch.isfinite(force) & (force > 1.)
            & (sample["clearance_m"].abs() < cfg.low_lift_threshold_m)
            & sample["eligible_gates"]["valid_state"][:, None]
            & sample["eligible_gates"]["upright"][:, None]
            & sample["eligible_gates"]["body_clear"][:, None]
        )
        self.rearm_steps = torch.where(can_rearm, self.rearm_steps + 1, 0)
        self._rearm_wheels(self.rearm_steps >= math.ceil(cfg.rearm_ground_hold_s / dt - 1e-9))
        fresh = sample["preview"] & ~self.seen
        self.stair_opportunities += (fresh & self.front_mask &
                                     (self.terrain_kind == STAIRS_TERRAIN_KIND)[:, None]).long()
        self.edge_xy[fresh] = sample["obstacle_xy"][fresh]
        headings = sample["heading_xy"][:, None].expand(-1, 4, -1)
        self.heading[fresh] = headings[fresh]
        self.target[fresh] = sample["target_lift_m"][fresh]
        ground_z = sample["wheel_centers_w"][..., 2] - sample["wheel_radii"][None] - sample["clearance_m"]
        self.approach_ground_z[fresh] = ground_z[fresh]
        self.seen |= fresh
        raw_force = sample.get("wheel_support_force_n", torch.full_like(self.best, float("nan")))
        force_valid = torch.isfinite(raw_force) & (raw_force >= 0)
        support_force = torch.nan_to_num(raw_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        lead = ((self.edge_xy - sample["wheel_centers_w"][..., :2]) * self.heading).sum(-1)
        lead -= sample["wheel_radii"][None]
        lift, clearance = sample["articulation_lift_m"], sample["clearance_m"]
        finite = (sample["eligible_gates"]["valid_state"][:, None] & sample["ground_valid"]
                  & torch.isfinite(lift) & torch.isfinite(clearance) & torch.isfinite(lead))
        approach_clearance = sample["wheel_centers_w"][..., 2] - sample["wheel_radii"][None]
        approach_clearance -= self.approach_ground_z
        resisting = sample.get("wheel_resisting_force_n", torch.zeros_like(self.best))
        actual_collision = (
            finite & force_valid & torch.isfinite(resisting) & (resisting >= cfg.collision_resisting_force_n)
            & (resisting > 0.5 * support_force)
            & (lead <= cfg.scan_edge_margin_m) & (lead >= -cfg.passage_margin_m)
            # 台阶顶面着地时局部净空接近0，不等于轮底仍低于竖直前缘。
            & (approach_clearance < self.target - cfg.clearance_margin_m)
        )
        actual_collision &= self.seen & self.required_mask
        unprepared = actual_collision & ~self.prelift_active & ~self.completed & ~self.unprepared_collision_seen
        self.unprepared_collision_seen |= unprepared
        # 抵达前缘锁死后退重领；碰撞仅作诊断，并阻止撞后抬轮冒充提前动作。
        self.contacted |= self.seen & ((lead <= 0) | ~finite)
        self.contacted |= actual_collision
        # 预瞄仍由 collect_preview 保留到 lookahead_m；这里是动作/奖励窗口。
        window = (self.seen & ~self.contacted & finite
                  & (lead >= cfg.min_advance_m) & (lead <= cfg.action_max_advance_m))
        high = finite & (lift >= cfg.min_lift_m) & (clearance >= cfg.min_lift_m)
        eligible = window & sample["eligible"][:, None]
        # 接地准备允许收缩悬架/侧倾；两种障碍的起抬武装都必须来自真实承重接地。
        narrow_grounded = (
            finite & force_valid & (support_force > 1.0)
            & (clearance < cfg.narrow_min_clearance_m) & ~self.contacted
        )
        # 落地冲击不能当作卸载参考。出生至少0.4s后，在25–65cm稳定接近段
        # 连续接地采样取滑动中位数，不用20%力变化门阻断PPO探索。
        # 进入准备区即冻结，后退也不能重写基线。
        self.baseline_locked |= self.seen & (lead <= cfg.preparation_max_advance_m)
        baseline_zone = self.seen & ~self.baseline_locked & (lead <= cfg.lookahead_m)
        baseline_update = (baseline_zone & narrow_grounded & ~self.takeoff_seen
                           & (support_force > cfg.narrow_min_force_drop_n + 1.0)
                           & sample["eligible"][:, None]
                           & (self.elapsed_s[:, None] >= cfg.baseline_settle_s))
        count = (self.baseline_samples + 1).clamp_max(baseline_steps)
        shifted = torch.cat((self.baseline_history[..., 1:], support_force[..., None]), dim=-1)
        self.baseline_history = torch.where(baseline_update[..., None], shifted,
                                             torch.where(self.baseline_locked[..., None], self.baseline_history, 0.))
        # 与独立验收同用偶数样本的中间两值均值；不足时不发布可用基线。
        ordered = self.baseline_history.sort(-1).values
        baseline = (ordered[..., (baseline_steps - 1) // 2] + ordered[..., baseline_steps // 2]) / 2
        self.baseline_force = torch.where(baseline_update & (count >= baseline_steps), baseline, self.baseline_force)
        self.baseline_samples = torch.where(baseline_update, count,
                                            torch.where(self.baseline_locked, self.baseline_samples, 0))
        baseline_ready = self.baseline_samples >= baseline_steps
        force_drop = ((self.baseline_force - support_force) / self.baseline_force.clamp_min(1.0)).clamp(0.0, 1.0)
        narrow_avoidance = (
            force_drop >= cfg.narrow_load_drop_fraction
        ) & ((self.baseline_force - support_force) >= cfg.narrow_min_force_drop_n)
        narrow_avoidance &= force_valid & finite & baseline_ready
        narrow_avoidance &= clearance >= cfg.narrow_min_clearance_m
        narrow_support = sample.get("narrow_supported", torch.zeros_like(self.narrow_mask))
        gates = sample["eligible_gates"]
        safe = gates["valid_state"] & gates["upright"] & gates["supported"] & gates["body_clear"]
        safe = torch.where(self.narrow_mask, safe & narrow_support, safe)
        eligible = torch.where(self.narrow_mask[:, None], eligible & narrow_support[:, None], eligible)
        # 起抬看离地/减载，而非相对车身的2cm高度。首次起抬锁存，不允许
        # 30cm处侧倾悬轮后在10cm收腿补领，也不允许落下再抬重写起抬距离。
        airborne = finite & force_valid & (clearance >= cfg.takeoff_clearance_m)
        airborne &= (support_force <= 1.0) | (self.narrow_mask[:, None] & narrow_avoidance)
        # 只消耗已预瞄障碍的资格；此前平地颠簸落回地面不应锁死后续尝试。
        # 若远处悬轮一直不落，进入预瞄时仍会锁为早抬，不能进10cm窗口补领。
        takeoffs = airborne & ~self.takeoff_seen & self.required_mask & self.seen
        self.takeoff_lead = torch.where(takeoffs & self.seen, lead, self.takeoff_lead)
        self.takeoff_valid |= takeoffs & eligible & self.low_seen
        self.takeoff_seen |= takeoffs
        self.early_lift_seen |= takeoffs & (lead > cfg.action_max_advance_m)
        self.takeoff_valid &= ~((narrow_grounded | ~finite | ~force_valid | ~safe[:, None]) & ~self.completed)
        # 窗口约束首次离地；后续逐渐达到2cm可发生在最后8cm内，仍须在前缘前。
        developing = (self.takeoff_valid & self.seen & ~self.contacted & finite
                      & sample["eligible"][:, None] & (lead > 0))
        developing &= ~self.narrow_mask[:, None] | narrow_support[:, None]
        self.unload_steps = torch.where(developing & narrow_avoidance, self.unload_steps + 1, 0)
        unload_hold_steps = math.ceil(cfg.narrow_hold_s / dt - 1e-9)
        minimum_hold_steps = math.ceil(cfg.prelift_hold_s / dt - 1e-9)
        release_hold_steps = math.ceil(cfg.release_hold_s / dt - 1e-9)
        event_state = torch.where(self.narrow_mask[:, None], narrow_avoidance, high)
        held = ~self.narrow_mask[:, None] | (self.unload_steps >= unload_hold_steps)
        events = (developing & event_state & held & ~self.completed
               & self.required_mask & ~self.contacted)
        self.height_lead = torch.where(events, lead, self.height_lead)
        # 接地减载必须也有学习信号，不能先要求已经离地才教离地；仍须有承重基线和安全支撑。
        # 准备可以短暂停车，封顶增量防止原地反复卸载刷分；提前悬轮不能领取准备信用。
        preparing = (self.seen & self.required_mask & finite & force_valid & baseline_ready
                     & self.low_seen & ~self.takeoff_seen & ~self.contacted & safe[:, None]
                     & gates["forward_command"][:, None] & (lead >= cfg.min_advance_m)
                     & (lead <= cfg.preparation_max_advance_m)
                     & (clearance >= -cfg.min_lift_m) & (clearance < cfg.takeoff_clearance_m))
        # 离地5mm前也须有净空学习信号；否则只减载不离地即可耗尽准备额度。
        # 纯减载只用准备额度的1/4，其余须在8–14cm窗口增长净空并同步卸载。
        # 窗外仍可接地协调，但不能因远处弹性形变或有净空却未减载而多领准备分。
        clearance_progress = torch.where(window, (clearance / cfg.takeoff_clearance_m).clamp(0, 1), 0.)
        preparation_score = torch.where(
            preparing, cfg.preparation_fraction * force_drop * (0.25 + 0.75 * clearance_progress), 0.,
        )
        # 连续台阶之间未必有25–65cm的承重基线区间。非窄障碍使用真实净空
        # 与关节抬升的共同增量引导5mm之前的动作，纯侧倾或悬架深蹲均不能领取。
        clearance_guidance = cfg.preparation_fraction * torch.minimum(
            (clearance / cfg.takeoff_clearance_m).clamp(0, 1), (lift / cfg.min_lift_m).clamp(0, 1),
        )
        guidance_valid = (window & self.required_mask & self.low_seen & ~self.takeoff_seen
                          & safe[:, None] & ~self.narrow_mask[:, None] & force_valid
                          & (clearance < cfg.takeoff_clearance_m) & (clearance >= 1e-5))
        preparation_score = torch.maximum(preparation_score, torch.where(guidance_valid, clearance_guidance, 0.))
        self.preparation_best = torch.maximum(self.preparation_best, preparation_score)
        # 准备与真实抬升共用单轮总额度；只支付历史最高值增量，抖动、停留、反复放下都不能刷分。
        score = torch.minimum(lift / self.target, clearance / self.target).clamp(0, 1)
        unload_score = force_drop * (clearance / cfg.narrow_min_clearance_m).clamp(0, 1)
        unload_score = torch.where(force_valid & baseline_ready, unload_score, 0.0)
        score = torch.where(self.narrow_mask[:, None], unload_score, score)
        score = cfg.preparation_fraction + (1 - cfg.preparation_fraction) * score
        score = torch.where(
            developing & ~self.hold_broken & self.required_mask,
            score,
            0.0,
        )
        score = torch.maximum(score, preparation_score)
        delta = (score - self.best).clamp_min(0)
        self.best = torch.maximum(self.best, score)
        # completed保留“有效提前动作事件”的原有语义，避免事件率与最终完成率混淆。
        self.completed |= events
        self.stair_events += (events & (self.terrain_kind == STAIRS_TERRAIN_KIND)[:, None]).long()
        self.prelift_active |= events
        # 保持不再受首次获得事件的8–14cm窗口或前进速度门限制。局部地面会
        # 跳到障碍顶面，因此这里相对首次预瞄的地面计算净空，落轮则用当前地面。
        strict_hold = (lift >= cfg.min_lift_m) & (approach_clearance >= cfg.min_lift_m)
        narrow_hold = (force_valid & baseline_ready
                       & (force_drop >= cfg.narrow_load_drop_fraction)
                       & (self.baseline_force - support_force >= cfg.narrow_min_force_drop_n)
                       & (approach_clearance >= cfg.narrow_min_clearance_m))
        hold_condition = (finite & safe[:, None] & ~actual_collision
                          & torch.where(self.narrow_mask[:, None], narrow_hold, strict_hold))
        pending_hold = self.prelift_active & ~self.passed_front_edge & ~self.hold_broken
        self.hold_steps = torch.where(pending_hold & hold_condition, self.hold_steps + 1, self.hold_steps)
        # 窄障碍首次事件已经包含连续卸载时长，不再丢掉这段有效保持。
        self.hold_steps = torch.where(events & self.narrow_mask[:, None], self.unload_steps, self.hold_steps)
        at_edge = pending_hold & finite & (lead <= 0)
        hold_violation = pending_hold & (
            ~hold_condition | (at_edge & (self.hold_steps < minimum_hold_steps))
        )
        self.hold_broken |= hold_violation
        held_to_edge = at_edge & hold_condition & ~self.hold_broken
        self.passed_front_edge |= held_to_edge
        release_condition = (
            finite & safe[:, None] & force_valid & (support_force > 1.0)
            & (clearance < torch.where(self.narrow_mask[:, None], cfg.narrow_min_clearance_m,
                                        cfg.low_lift_threshold_m)) & (clearance >= -cfg.min_lift_m)
            & (lead <= 0)
        )
        pending_release = self.passed_front_edge & ~self.release_paid & ~self.hold_broken
        self.release_steps = torch.where(pending_release & release_condition, self.release_steps + 1, 0)
        releases = pending_release & (self.release_steps >= release_hold_steps)
        self.release_paid |= releases
        self.prelift_active &= ~self.hold_broken & ~self.release_paid
        # 预瞄前短暂离地不耗尝试，但必须重新承重落地才能武装；持续悬轮不能保留旧接地证据。
        self.low_seen = torch.where(~finite | ~force_valid | airborne | self.takeoff_seen,
                                    False, self.low_seen | narrow_grounded)
        # 位置仅用于记录漏抬，不把位置当成实际碰撞，也不再发漏抬罚分。
        missed = (self.seen & self.required_mask & finite & (lead <= 0)
                  & ~self.completed & ~self.missed_seen)
        self.missed_seen |= missed
        # 全宽允许任一前轮完成主动动作，另一轮可滚上；窄障碍只计受阻侧。
        # 按世界+X前缘去重，左右先后完成同一级只领一次，下一真实前缘可重新结算。
        released = self.release_paid & self.required_mask
        front_done = released.any(-1)
        released_edge = self.edge_xy[..., 0].masked_fill(~released, -float("inf")).max(-1).values
        front_completion = front_done & (released_edge > self.completion_edge_x + cfg.new_edge_min_spacing_m)
        self.completion_edge_x = torch.where(front_completion, released_edge, self.completion_edge_x)
        self.front_paid |= front_done
        self.stair_completions += (front_completion & (self.terrain_kind == STAIRS_TERRAIN_KIND)).long()
        route_passed = sample.get("route_passed", torch.zeros_like(self.route_paid)) & safe
        self.route_hold_s = torch.where(route_passed, self.route_hold_s + dt, 0.)
        route_completion = (self.route_hold_s >= 0.30 - 1e-6) & ~self.route_paid
        self.route_paid |= route_completion
        passage = self.seen & (lead >= -cfg.passage_margin_m) & (lead <= cfg.action_max_advance_m)
        # 只惩罚真实悬轮，不能把仍承重接地的悬架准备当成早抬。
        raised = ((clearance - cfg.takeoff_clearance_m) / cfg.min_lift_m).clamp(0, 1)
        # 越障时非目标轮/后轮的协同悬架动作是必要的，不按单轮预瞄窗口误罚。
        allowed = passage.any(-1) & safe
        excess = ((clearance - self.target - cfg.min_lift_m) / cfg.min_lift_m).clamp(0, 1)
        # 协同豁免不能使尚未进窗的另一目标前轮抢跑；失败后的重抬同样不能无限跺脚。
        # 非目标轮仍可在实际越障期间协同，不能为抑制早抬锁死整车悬架。
        early = self.seen & self.required_mask & ~self.contacted & (lead > cfg.action_max_advance_m)
        repeated = self.hold_broken & self.required_mask & (lead > 0)
        unwanted = early | repeated
        per_wheel_idle = torch.where(allowed[:, None], excess, raised)
        per_wheel_idle = torch.where(unwanted, torch.maximum(per_wheel_idle, raised), per_wheel_idle)
        idle = torch.where(finite, per_wheel_idle, 0.0).mean(-1)
        # 四腿提前收缩会消耗车身高度而非产生离地；远处接地时保留行程即可，
        # 不要求精确回中，进入准备/越缘区后允许三轮支撑及俯仰侧倾。
        coordinating = (self.seen & self.front_mask & (lead <= cfg.preparation_max_advance_m)
                         & (lead >= -cfg.passage_margin_m)).any(-1)
        contraction = ((lift - cfg.approach_max_contraction_m) / cfg.approach_max_contraction_m).clamp(0, 1)
        grounded = finite & force_valid & (support_force > 1.) & (clearance < cfg.takeoff_clearance_m)
        stance = torch.where(grounded & ~coordinating[:, None], contraction.square(), 0.).mean(-1)
        # 通行和主动准备分开结算，第二轮被动滚上不剥夺整车通行收益。
        credit = (cfg.progress_credit * delta.sum(-1) + cfg.wheel_event_credit * events.sum(-1)
                  + cfg.front_completion_credit * front_completion
                  + cfg.route_completion_credit * route_completion
                  - cfg.hold_violation_penalty * hold_violation.sum(-1))
        return {"credit": credit, "idle_penalty": idle, "stance_penalty": stance, "events": events,
                "held_to_edge": held_to_edge, "releases": releases,
                "hold_broken": self.hold_broken.clone(),
                "prelift_active": self.prelift_active.clone(),
                "passed_front_edge": self.passed_front_edge.clone(),
                "front_completion": front_completion, "route_completion": route_completion, "lead_m": lead,
                "action_window": window & self.required_mask, "early_lift": early & airborne,
                "takeoffs": takeoffs, "takeoff_lead_m": self.takeoff_lead.clone(),
                "height_lead_m": self.height_lead.clone(), "missed_wheels": missed,
                "force_drop_fraction": force_drop, "unprepared_collision": unprepared}


class _PreviewGeometry:
    """固定拓扑只解析一次；碰撞轮心和中立高度来自实际 URDF，不猜关节正负号。"""

    def __init__(self, env):
        from robot_lab.tasks.manager_based.upright.mdp.geometry import RoverCollisionGeometry

        robot = env.scene["robot"]
        self.body_ids, names = robot.find_bodies(list(WHEEL_NAMES), preserve_order=True)
        if tuple(names) != WHEEL_NAMES:
            raise ValueError("四轮刚体顺序不匹配")
        sensor = env.scene.sensors["contact_forces"]
        self.sensor_ids, sensor_names = sensor.find_bodies(list(WHEEL_NAMES), preserve_order=True)
        if tuple(sensor_names) != WHEEL_NAMES:
            raise ValueError("四轮接触传感器顺序不匹配")
        self.other_ids = [i for i, name in enumerate(sensor.body_names) if name not in WHEEL_NAMES]
        geometry = RoverCollisionGeometry(robot.cfg.spawn.asset_path, robot.joint_names)
        neutral = robot.data.default_joint_pos[:1]
        # 当前资产的默认 master/follower 都为零；新中立姿态必须单独审核 mimic 一致性。
        if not torch.allclose(neutral, torch.zeros_like(neutral)):
            raise ValueError("前瞻几何当前只支持已审核的零关节中立姿态")
        identity = neutral.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        positions, rotations = geometry.link_transforms(neutral, identity)
        offsets, radii, centers = [], [], []
        for name in WHEEL_NAMES:
            collisions = [item for item in geometry.collisions if item[0] == name]
            if len(collisions) != 1 or collisions[0][2] != "cylinder":
                raise ValueError("前瞻奖励要求每轮唯一圆柱碰撞体")
            _, (offset, _), _, (radius, _) = collisions[0]
            offsets.append(offset)
            radii.append(radius)
            centers.append(positions[name][0] + rotations[name][0] @ neutral.new_tensor(offset))
        self.offsets = neutral.new_tensor(offsets)
        self.radii = neutral.new_tensor(radii)
        self.neutral_centers = torch.stack(centers)
        mass = robot.root_physx_view.get_masses().to(device=env.device)
        self.robot_weight_n = mass.sum(-1) * -env.cfg.sim.gravity[2]
        terrain = env.cfg.scene.terrain.terrain_generator
        self.column_kinds = terrain_column_kinds(
            list(terrain.sub_terrains), [cfg.proportion for cfg in terrain.sub_terrains.values()],
            terrain.num_cols, curriculum=terrain.curriculum, device=env.device,
        )

    def sample(self, env):
        from isaaclab.utils.warp import raycast_mesh

        robot = env.scene["robot"]
        data = robot.data
        offsets = self.offsets[None].expand(env.num_envs, -1, -1)
        centers = data.body_pos_w[:, self.body_ids] + rotate_vectors(data.body_quat_w[:, self.body_ids], offsets)
        forces = env.scene.sensors["contact_forces"].data.net_forces_w
        supported = (forces[:, self.sensor_ids, 2] > 1.0).sum(-1)
        body_contact = (forces[:, self.other_ids].norm(dim=-1) > 5.0).any(-1)
        body_contact |= ~torch.isfinite(forces).all(dim=(-1, -2))
        scanner = env.scene.sensors["height_scanner"]
        # 稀疏高度图在台阶边缘可能把轮旁顶面当脚下地面。四条竖直射线提供
        # 真实局部地面，仅用于奖励和验收，Teacher/Student输入维数及权限不变。
        ground_starts = centers.clone()
        ground_starts[..., 2] += 0.30
        ground_directions = torch.zeros_like(ground_starts)
        ground_directions[..., 2] = -1.
        ground_hits = raycast_mesh(
            ground_starts, ground_directions, mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]], max_dist=2.0,
        )[0]
        result = preview_measurements(
            ray_hits_w=env.scene.sensors["height_scanner"].data.ray_hits_w,
            root_pos_w=data.root_pos_w, root_quat_w=data.root_quat_w,
            wheel_centers_w=centers, neutral_centers_b=self.neutral_centers, wheel_radii=self.radii,
            command_x=env.command_manager.get_command("base_velocity")[:, 0],
            velocity_x=data.root_lin_vel_w[:, 0], supported_wheels=supported, nonwheel_contact=body_contact,
            cfg=PreviewCfg(**env.cfg.vision_preview), ground_hits_w=ground_hits,
        )
        # 课程在reset重抽列；逐步读取当前真实分配，避免缓存旧地形类别。
        result["terrain_kind"] = self.column_kinds[env.scene.terrain.terrain_types]
        cfg = PreviewCfg(**env.cfg.vision_preview)
        # 只访问地形静态mesh，不射机器人自身；高度低于课程最小台阶，避免命中顶面。
        # 每轮一条水平射线，复用扫描器mesh，额外数据不进入teacher/student观测。
        starts = centers.clone()
        starts[..., 2] -= self.radii[None] + result["clearance_m"] - 0.5 * cfg.min_rise_m
        directions = torch.zeros_like(starts)
        directions[..., :2] = result["heading_xy"][:, None]
        edge_hits = raycast_mesh(
            torch.nan_to_num(starts), directions,
            mesh=scanner.meshes[scanner.cfg.mesh_prim_paths[0]],
            max_dist=cfg.lookahead_m + cfg.ground_sample_radius_m,
        )[0]
        result = refine_step_edges(result, edge_hits, cfg)
        wheel_forces = forces[:, self.sensor_ids]
        result["wheel_support_force_n"] = wheel_forces[:, :, 2]
        result["wheel_resisting_force_n"] = -(wheel_forces[:, :, :2] * result["heading_xy"][:, None]).sum(-1)
        result["narrow_supported"] = narrow_support_mask(wheel_forces[:, :, 2], self.robot_weight_n)
        quat = data.root_quat_w
        alignment = 1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square())
        upright = -data.projected_gravity_b[:, 2] >= 0.85
        result["route_passed"] = preview_route_outcome(
            data.root_pos_w, env.scene.terrain.env_origins, upright, alignment, preview_route_targets(env),
        )[0]
        result["route_passed"] &= (supported >= 3) & (wheel_forces[..., 2].sum(-1) >= 0.5 * self.robot_weight_n)
        result["control_dt_s"] = env.step_dt
        return result


def collect_preview(env):
    """每次读取当前物理状态，不复用 reset 前缓存；只缓存静态几何和名称映射。"""
    if not hasattr(env, "_vision_preview_geometry"):
        env._vision_preview_geometry = _PreviewGeometry(env)
    return env._vision_preview_geometry.sample(env)
