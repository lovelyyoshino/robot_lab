"""
isaac_adapter.py — 从 Isaac Lab ManagerBasedRLEnv 读出 SuperMimic 扁平状态、命令、上一动作、
动力学上下文 ξ 与关节加速度监督量。

@author pony

必须在 AppLauncher 启动之后导入。关节按动作合同顺序（arm4, steer4, wheel4）取 12 个
主动关节，8 个 mimic 从动关节不读；位置减 default_joint_pos，与 joint_pos_rel 观测同口径。

ξ 的组成（CONTEXT_DIM=25）：
- 8 个腿关节 Kp / 默认 Kp、12 个关节 Kd / 默认 Kd：randomize_actuator_gains 每次 reset 在 0.5–2 倍间
  重抽，是 rover 平地动力学里最大的未观测变量；
- 整车质量 / 默认质量、base 质量增量 (kg)、base 质心 (3)：startup 随机化，启动后缓存。
可选追加四轮静摩擦/动摩擦/恢复系数（12维）；视觉任务摩擦较低，不能沿用普通轮车的高摩擦假设。

@date 2026-10-26
@version v0.4.0
@last_modified 2026-11-05
@changelog
    - (2026-11-05): exact_riser_fn 一次批量投射四轮水平立面射线，替换只看前轮的预抬边缘函数。
    - v0.4.0 (2026-11-02): 保留PhysX原生四元数精度，与RayCaster使用同一姿态计算扫描。
    - v0.3.1 (2026-10-26): 新增 exact_scan_fn（直通前向的真实射线扫描）与 leg_soft_limits。
    - v0.3.0 (2026-10-26): world_frame 选项；新增 build_height_map / scan_pattern_base（地形任务）。
    - v0.2.0 (2026-10-26): 新增 context()（ξ）与 acc_aux()（env 关节加速度项真值）。
    - v0.1.0 (2026-10-26): 初始创建
"""

from __future__ import annotations

import torch

from robot_lab.assets.parallelogram_rover import PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES

from .rover_task import ACC_AUX_NAMES
from .state import NUM_LEG_JOINTS, make_state

CONTEXT_DIM = NUM_LEG_JOINTS + 12 + 1 + 1 + 3
CONTACT_CONTEXT_DIM = CONTEXT_DIM + 4 * 3


class RoverStateReader:
    def __init__(self, env, asset_name: str = "robot", command_name: str = "base_velocity",
                 base_body: str = "base_link", world_frame: bool = False, contact_context: bool = False):
        """world_frame=True：位置保留世界坐标（地形任务需要它去查全局高度图）；
        平地任务默认减 env 原点，只为数值小。"""
        self.env = env.unwrapped
        self.robot = self.env.scene[asset_name]
        self.command_name = command_name
        self.world_frame = world_frame
        ids, names = self.robot.find_joints(list(PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES), preserve_order=True)
        if tuple(names) != PARALLELOGRAM_ROVER_POLICY_ACTION_JOINT_NAMES:
            raise ValueError(f"关节顺序与动作合同不符: {names}")
        self.joint_ids = torch.tensor(ids, device=self.env.device)
        self._static_context = self._read_static_context(base_body)
        self.material_context = self._read_wheel_materials() if contact_context else None
        self.context_dim = CONTACT_CONTEXT_DIM if contact_context else CONTEXT_DIM

    # ---------------- 状态 ----------------
    def state(self) -> torch.Tensor:
        data = self.robot.data
        ids = self.joint_ids
        # 平地：减 env 原点只为数值不过大，世界模型本身对平移不变；地形：必须是世界系才能查高度图。
        pos = data.root_pos_w if self.world_frame else data.root_pos_w - self.env.scene.env_origins
        # RayCaster直接使用PhysX姿态；再次归一化造成的1e-7误差会在台阶边切换命中表面。
        return make_state(pos, data.root_quat_w, data.root_lin_vel_b, data.root_ang_vel_b,
                          data.joint_pos[:, ids] - data.default_joint_pos[:, ids], data.joint_vel[:, ids],
                          normalize_quat=False)

    def command(self) -> torch.Tensor:
        return self.env.command_manager.get_command(self.command_name).clone()

    def last_action(self) -> torch.Tensor:
        # ActionManager.action 是送进各 term 前的原始动作，reset 时清零，与 last_action 观测一致。
        return self.env.action_manager.action.clone()

    # ---------------- 动力学上下文 ξ ----------------
    def _read_static_context(self, base_body: str) -> torch.Tensor:
        # startup 随机化在 env 构造时已执行，这里读到的就是本次训练全程不变的质量/质心。
        view = self.robot.root_physx_view
        masses = view.get_masses().to(self.env.device)
        default = self.robot.data.default_mass.to(self.env.device)
        base = self.robot.find_bodies(base_body)[0][0]
        com = view.get_coms().to(self.env.device)[:, base, :3]
        return torch.cat((
            (masses.sum(-1) / default.sum(-1)).unsqueeze(-1),
            (masses[:, base] - default[:, base]).unsqueeze(-1),
            com,
        ), dim=-1)

    def _gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.robot.num_joints
        kp = torch.zeros(self.env.num_envs, n, device=self.env.device)
        kd = torch.zeros_like(kp)
        for actuator in self.robot.actuators.values():
            idx = actuator.joint_indices
            if isinstance(idx, slice):
                idx = torch.arange(n, device=self.env.device)[idx]
            kp[:, idx] = actuator.stiffness
            kd[:, idx] = actuator.damping
        return kp[:, self.joint_ids], kd[:, self.joint_ids]

    def _read_wheel_materials(self) -> torch.Tensor:
        from .kinematics import CORNERS

        term = self.env.event_manager.get_term_cfg("randomize_wheel_rigid_body_material").func
        counts = term.num_shapes_per_body
        view = self.robot.root_physx_view
        if counts is None or sum(counts) != view.max_shapes:
            raise RuntimeError("逐轮接触上下文需要经过校验的body到碰撞shape映射")
        ids, _ = self.robot.find_bodies([f"{corner}_wheel" for corner in CORNERS], preserve_order=True)
        if len(ids) != 4 or any(counts[i] != 1 for i in ids):
            raise RuntimeError("逐轮接触上下文要求四个轮子各有一个碰撞shape")
        shapes = [sum(counts[:i]) for i in ids]
        # 复用Isaac事件的shape映射读取实际抽样值；startup材料仅在此读取一次。
        return view.get_material_properties()[:, shapes, :].flatten(1).to(self.env.device).clone()

    def context(self) -> torch.Tensor:
        kp, kd = self._gains()
        ids = self.joint_ids
        kp0 = self.robot.data.default_joint_stiffness[:, ids]
        kd0 = self.robot.data.default_joint_damping[:, ids]
        # 轮子 Kp 恒为 0，只取 8 个腿关节的 Kp 比；默认值为 0 的 Kd 比记 1。
        kp_ratio = (kp / kp0.clamp_min(1e-6))[:, :NUM_LEG_JOINTS]
        kd_ratio = torch.where(kd0 > 1e-6, kd / kd0.clamp_min(1e-6), torch.ones_like(kd))
        parts = [kp_ratio, kd_ratio, self._static_context]
        if self.material_context is not None:
            parts.append(self.material_context)
        return torch.cat(parts, dim=-1)

    # ---------------- 关节加速度监督量 ----------------
    def acc_aux(self) -> torch.Tensor:
        """刚结束的 step 中 RewardManager 算出的三项 Σjoint_acc²（除去权重与 dt 之前的原值）。

        必须在 env.step 之后、下一次 step 之前调用；reset 发生在奖励计算之后，不影响该值。
        """
        rm = self.env.reward_manager
        names = list(rm.active_terms)
        cols = []
        for name in ACC_AUX_NAMES:
            if name not in names:
                raise ValueError(f"任务缺少奖励项 {name}，无法提供关节加速度监督；该任务应设 aux_dim=0")
            weight = rm.get_term_cfg(name).weight
            cols.append(rm._step_reward[:, names.index(name)] / weight)
        return torch.stack(cols, dim=-1)

    def has_acc_aux(self) -> bool:
        names = set(self.env.reward_manager.active_terms)
        return all(name in names for name in ACC_AUX_NAMES)


# ---------------- 地形 ----------------
def build_height_map(env, sensor_name: str = "height_scanner", res: float = 0.02, margin: float = 0.5,
                     chunk: int = 2_000_000):
    """对 RayCaster 使用的同一张地形网格做竖直射线栅格化，得到全局 HeightMap（世界系）。

    用 RayCaster 自己的 warp mesh，保证想象扫描与真实扫描看的是同一几何。竖直射线只命中朝上的面，
    所以格点上的值与 RayCaster 在该 xy 的命中高度相同；格点之间是双线性插值。
    """
    import warp as wp

    from isaaclab.utils.warp import raycast_mesh

    from .terrain_map import HeightMap

    sensor = env.unwrapped.scene.sensors[sensor_name]
    mesh_path = sensor.cfg.mesh_prim_paths[0]
    mesh = sensor.meshes[mesh_path]
    device = env.unwrapped.device
    pts = wp.to_torch(mesh.points)
    lo = pts.min(0).values[:2] - margin
    hi = pts.max(0).values[:2] + margin
    top = float(pts[:, 2].max()) + 5.0
    xs = torch.arange(float(lo[0]), float(hi[0]) + res, res, device=device)
    ys = torch.arange(float(lo[1]), float(hi[1]) + res, res, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    starts = torch.stack((gx.flatten(), gy.flatten(), torch.full_like(gx.flatten(), top)), dim=-1)
    down = torch.tensor([0.0, 0.0, -1.0], device=device)
    heights = torch.empty(starts.shape[0], device=device)
    for i in range(0, starts.shape[0], chunk):
        s = starts[i:i + chunk].unsqueeze(0)
        hits = raycast_mesh(s, down.expand_as(s), mesh=mesh, max_dist=1e3)[0][0]
        heights[i:i + chunk] = hits[:, 2]
    # 未命中（inf）出现在网格外缘之外；按最低地面填，避免 NaN 传进梯度。
    finite = torch.isfinite(heights)
    heights = torch.where(finite, heights, heights[finite].min())
    return HeightMap(heights.reshape(len(xs), len(ys)), (float(xs[0]), float(ys[0])), res).to(device)


def scan_pattern_base(env, sensor_name: str = "height_scanner") -> torch.Tensor:
    """RayCaster 射线起点在 base 系的坐标（pattern + offset，环境间相同），[P, 3]。"""
    sensor = env.unwrapped.scene.sensors[sensor_name]
    if sensor.cfg.ray_alignment != "yaw":
        raise ValueError(f"{sensor_name} 的 ray_alignment={sensor.cfg.ray_alignment}，想象扫描只实现了 yaw 对齐")
    if any(abs(v) > 0 for rng in sensor.cfg.ray_cast_drift_range.values() for v in rng) or any(
            abs(v) > 0 for v in sensor.cfg.drift_range):
        raise ValueError(f"{sensor_name} 启用了 drift，想象扫描无法复现")
    return sensor.ray_starts[0].clone()


def exact_scan_fn(env, pattern_b: torch.Tensor, sensor_name: str = "height_scanner"):
    """返回 f(state) → [B, P]：按想象位姿对真实地形网格做射线投射，公式同 height_scan，供直通前向使用。"""
    from isaaclab.utils.math import quat_apply_yaw
    from isaaclab.utils.warp import raycast_mesh

    from .state import POS, QUAT
    from .terrain_map import SCAN_CLIP, SCAN_OFFSET

    sensor = env.unwrapped.scene.sensors[sensor_name]
    mesh = sensor.meshes[sensor.cfg.mesh_prim_paths[0]]

    def fn(state: torch.Tensor) -> torch.Tensor:
        # 前向必须复用RayCaster的归一化、旋转算子和offset；台阶边缘的微小舍入差也会切换命中面。
        starts = quat_apply_yaw(state[:, QUAT].repeat(1, pattern_b.shape[0]),
                                pattern_b.expand(state.shape[0], -1, -1))
        starts += state[:, None, POS]
        down = sensor.ray_directions[0].expand_as(starts).contiguous()
        hits = raycast_mesh(starts.contiguous(), down, mesh=mesh, max_dist=sensor.cfg.max_distance)[0]
        return (state[:, None, 2] - hits[..., 2] - SCAN_OFFSET).clamp(*SCAN_CLIP)

    return fn


def exact_riser_fn(env, sensor_name: str = "height_scanner", max_dist: float = 0.75):
    """返回 f(starts [..., 3], directions [..., 3]) → 水平射线到地形网格的距离 [...]（未命中为 inf）。

    PreviewTask 每个状态只调用一次，四个轮同批：起点 = 轮心 xy、轮底 + riser_min_height，方向 = 航向。
    仅用于训练奖励/统计，不进入 Actor 观测；不反传。"""
    from isaaclab.utils.warp import raycast_mesh

    sensor = env.unwrapped.scene.sensors[sensor_name]
    mesh = sensor.meshes[sensor.cfg.mesh_prim_paths[0]]

    @torch.no_grad()
    def fn(starts, directions):
        shape = starts.shape[:-1]
        starts, directions = starts.reshape(-1, 3), directions.reshape(-1, 3)
        # 非有限起点不能送入 Warp；置零后按未命中处理。
        valid = torch.isfinite(starts).all(-1)
        safe = torch.where(valid[:, None], starts, torch.zeros_like(starts)).contiguous()
        # Warp 射线接口要求 [B, N, 3]；这里把全部射线当作一个批次送入，再取回 [N]。
        distance = raycast_mesh(safe[None], directions.contiguous()[None], mesh=mesh, max_dist=max_dist,
                                return_distance=True)[1][0]
        return torch.where(valid, distance, torch.inf).reshape(shape)

    return fn


def leg_soft_limits(reader: RoverStateReader) -> torch.Tensor:
    """8 个腿关节的软限位，减去默认位（与 state 的 joint_pos_rel 同口径），[8, 2]。"""
    ids = reader.joint_ids[:NUM_LEG_JOINTS]
    lim = reader.robot.data.soft_joint_pos_limits[0, ids]
    return lim - reader.robot.data.default_joint_pos[0, ids, None]
