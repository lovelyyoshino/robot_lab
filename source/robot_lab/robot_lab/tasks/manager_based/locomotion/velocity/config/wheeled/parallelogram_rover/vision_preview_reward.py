"""主动抬轮时序奖励的Isaac Lab生命周期适配。

@author pony
@date 2026-10-08
@version v0.2.0
@last_modified 2026-10-14
@changelog
    - (2026-10-14): 路线连续稳定后晋级，跨阶段必须通过楼梯，移除双侧主动硬门。
    - (2026-10-13): 课程区分起抬学习与完整验收，远处接地深蹲计入行程保护成本。
    - (2026-10-12): 在奖励reset前读取完整动作证据决定课程晋级，分开记录动作完成与通行加分。
    - (2026-10-12): 路线命令按实时偏航转换为车体观测，Teacher/Student可辨别朝向偏差。
    - (2026-10-12): 用偏航速度负项替代零偏航常驻正分，保留直行奖励的静止零基线。
    - (2026-10-12): 速度收益改对齐世界+X路线，按有效通行切换三阶段地形课程。
    - (2026-10-11): Preview速度跟踪采用静止零基线，分地形记录准备减载，不叠加停滞计时规则。
    - (2026-10-11): 记录真实起抬/高度达标距离与逐地形漏抬率，不以收腿位置冒充离地时机。
    - (2026-10-10): 记录逐轮早抬回合率，区分通行、有效起抬与窗口外抢跑。
    - (2026-10-09): 分全宽/左窄/右窄记录目标事件、机会数及未准备阻挡碰撞率。
    - (2026-10-10): 分开统计首次事件、保持到前缘、落轮与中途失败，不混淆动作和通行。
    - v0.1.0 (2026-10-08): 按环境reset事件缓存并记录左右命中率，不重复发放悬轮奖励。
"""

import math

import torch

from isaaclab.managers import ManagerTermBase

from .vision_preview import (
    PreliftRewardState, PreviewCfg, centered_velocity_tracking, collect_preview,
    preview_route_outcome, preview_route_promotion_mask, preview_route_targets,
    sample_preview_terrain_columns, terrain_column_names,
)


def preview_route_command(env, command_name):
    """保留三维命令接口，用车体坐标表达世界路线，不原位修改命令管理器。"""
    command = env.command_manager.get_command(command_name)
    quat = env.scene["robot"].data.root_quat_w
    yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                      1 - 2 * (quat[:, 2].square() + quat[:, 3].square()))
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * command[:, 0] + s * command[:, 1],
                        -s * command[:, 0] + c * command[:, 1], command[:, 2]), dim=-1)


def preview_velocity_tracking(env, std, command_name, asset_cfg=None):
    """沿世界+X路线行驶，车体转向后不能继续靠本体前进速度刷分。"""
    robot = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    command_w = torch.stack((command_x, torch.zeros_like(command_x)), dim=-1)
    reward = centered_velocity_tracking(command_w, robot.data.root_lin_vel_w[:, :2], std)
    quat = robot.data.root_quat_w
    alignment = (1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square())).clamp(0.0, 1.0)
    reward = reward.clamp_min(0.0) * alignment + reward.clamp_max(0.0) - 0.25 * (1.0 - alignment)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0, 0.7) / 0.7
    return reward * upright


def preview_yaw_rate_l2(env):
    """只约束旋转速度，不为原地保持零偏航持续发放正奖励。"""
    return env.scene["robot"].data.root_ang_vel_w[:, 2].square()


def preview_route_success(env):
    """稳定通行后截断回合，避免驶出顶端平台才reset而丢失晋级证据。"""
    state = env.reward_manager.get_term_cfg("anticipatory_wheel_lift").func.state
    # 终止管理器先于奖励更新；验证当前状态，不能沿用上一步承重而接受本步跌倒。
    return (state.route_hold_s >= 0.30 - 1e-6) & collect_preview(env)["route_passed"]


def preview_route_curriculum(env, env_ids):
    """9行分8/12/16cm三阶段；本档楼梯通行后才可跨阶段。"""
    terrain = env.scene.terrain
    device = terrain.terrain_levels.device
    ids = torch.arange(env.num_envs, device=device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(
        env_ids, device=device, dtype=torch.long,
    )
    cfg = terrain.cfg.terrain_generator
    columns = terrain_column_names(
        tuple(cfg.sub_terrains), tuple(item.proportion for item in cfg.sub_terrains.values()),
        cfg.num_cols, curriculum=cfg.curriculum,
    )
    if ids.numel():
        robot = env.scene["robot"]
        quat = robot.data.root_quat_w[ids]
        alignment = 1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square())
        upright = -robot.data.projected_gravity_b[ids, 2] >= 0.85
        passed, failed = preview_route_outcome(
            robot.data.root_pos_w[ids], terrain.env_origins[ids], upright, alignment,
            preview_route_targets(env)[ids],
        )
        active = env.episode_length_buf[ids] > 0
        # 晋级不依赖双侧主动事件，但须在终点真实承重并连续稳定；不能用单帧跳过终点冒充通过。
        state = env.reward_manager.get_term_cfg("anticipatory_wheel_lift").func.state
        promotable = preview_route_promotion_mask(
            terrain.terrain_levels[ids], terrain.terrain_types[ids], columns,
        )
        passed &= state.route_hold_s[ids] >= 0.30 - 1e-6
        if not hasattr(env, "_preview_route_success_streak"):
            env._preview_route_success_streak = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        streak = env._preview_route_success_streak
        eligible_episode = active & promotable
        streak[ids] = torch.where(eligible_episode & passed, streak[ids] + 1,
                                 torch.where(eligible_episode | (failed & active), 0, streak[ids]))
        promote = passed & active & promotable & (streak[ids] >= 2)
        levels = (
            terrain.terrain_levels[ids] + promote.long() - (failed & active).long()
        ).clamp(0, 8)
        streak[ids] = torch.where(promote | (failed & active), 0, streak[ids])
        terrain.terrain_levels[ids] = levels
        terrain.terrain_types[ids] = sample_preview_terrain_columns(levels, columns)
        terrain.env_origins[ids] = terrain.terrain_origins[levels, terrain.terrain_types[ids]]
    levels = terrain.terrain_levels
    stair_columns = torch.tensor([i for i, name in enumerate(columns) if name == "pyramid_stairs_inv"], device=device)
    on_stairs = torch.isin(terrain.terrain_types, stair_columns)
    return {"mean_level": levels.float().mean(), "stage_1": (levels < 3).float().mean(),
            "stage_2": ((levels >= 3) & (levels < 6)).float().mean(),
            "stage_3": (levels >= 6).float().mean(),
            "stairs_8cm_fraction": (on_stairs & (levels < 3)).float().mean(),
            "stairs_12cm_fraction": (on_stairs & (levels >= 3) & (levels < 6)).float().mean(),
            "stairs_16cm_fraction": (on_stairs & (levels >= 6)).float().mean()}


class AnticipatoryWheelLift(ManagerTermBase):
    """框架负责reset；几何采样和评测不能意外推进奖励状态机。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.preview_cfg = PreviewCfg(**env.cfg.vision_preview)
        self.state = PreliftRewardState(env.num_envs, env.device, self.preview_cfg)

    def __call__(self, env):
        result = self.state.update(collect_preview(env))
        # RewardManager会乘dt；事件总额应与控制频率无关，常驻惩罚则按秒累计。
        return (
            result["credit"] / env.step_dt
            - self.preview_cfg.idle_lift_penalty * result["idle_penalty"]
            - self.preview_cfg.approach_stance_penalty * result["stance_penalty"]
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        seen, done = self.state.seen[ids], self.state.completed[ids]
        held, released, broken = (
            self.state.passed_front_edge[ids], self.state.release_paid[ids], self.state.hold_broken[ids],
        )
        if seen.numel():
            metrics = self._env.extras.setdefault("log", {})
            stair_opportunities = self.state.stair_opportunities[ids][:, [0, 2]]
            stair_events = self.state.stair_events[ids][:, [0, 2]]
            metrics["Metrics/prelift/stairs/opportunities"] = stair_opportunities.sum()
            metrics["Metrics/prelift/stairs/event_rate"] = (
                stair_events.sum().float() / stair_opportunities.sum().clamp_min(1))
            metrics["Metrics/prelift/stairs/active_steps_completed"] = self.state.stair_completions[ids].sum()
            metrics["Metrics/prelift/route_stable_success_rate"] = (self.state.route_hold_s[ids] >= 0.30 - 1e-6).float().mean()
            for index, label in ((0, "fl"), (2, "fr")):
                metrics[f"Metrics/prelift/{label}_opportunities"] = seen[:, index].float().sum()
                metrics[f"Metrics/prelift/{label}_event_rate"] = (
                    done[:, index].float().sum() / seen[:, index].sum().clamp_min(1))
                metrics[f"Metrics/prelift/{label}_early_lift_rate"] = (
                    (self.state.early_lift_seen[ids][:, index] & seen[:, index]).float().sum()
                    / seen[:, index].sum().clamp_min(1))
                for name, distances in (("takeoff", self.state.takeoff_lead), ("height", self.state.height_lead)):
                    values = distances[ids][:, index]
                    valid = values.isfinite() & seen[:, index]
                    metrics[f"Metrics/prelift/{label}_{name}_count"] = valid.sum()
                    metrics[f"Metrics/prelift/{label}_{name}_distance_m"] = (
                        values.nan_to_num().masked_fill(~valid, 0.).sum() / valid.sum().clamp_min(1))
                for name, values in (("held_to_edge", held), ("release", released), ("hold_failure", broken)):
                    metrics[f"Metrics/prelift/{label}_{name}_rate"] = (
                        (seen[:, index] & values[:, index]).float().sum() / seen[:, index].sum().clamp_min(1))
            both = seen[:, [0, 2]].all(-1)
            metrics["Metrics/prelift/both_front_event_rate"] = (
                (done[:, [0, 2]].all(-1) & both).float().sum() / both.sum().clamp_min(1))
            kinds = self.state.terrain_kind[ids]
            collisions = self.state.unprepared_collision_seen[ids]
            for kind, name, targets in ((0, "single_step", (0, 2)), (1, "single_step_left", (0,)),
                                        (2, "single_step_right", (2,))):
                subset = kinds == kind
                opportunities = subset & seen[:, targets].all(-1)
                denominator = opportunities.sum().clamp_min(1)
                prefix = f"Metrics/prelift/{name}"
                metrics[f"{prefix}/episodes"] = subset.sum()
                metrics[f"{prefix}/opportunities"] = opportunities.sum()
                metrics[f"{prefix}/event_rate"] = (opportunities & done[:, targets].any(-1)).float().sum() / denominator
                metrics[f"{prefix}/sequence_complete_rate"] = (
                    (opportunities & self.state.front_paid[ids]).float().sum() / denominator)
                metrics[f"{prefix}/route_reward_rate"] = (
                    (opportunities & self.state.route_paid[ids]).float().sum() / denominator)
                for metric, values in (("held_to_edge_rate", held), ("release_rate", released)):
                    metrics[f"{prefix}/{metric}"] = (
                        (opportunities & values[:, targets].any(-1)).float().sum() / denominator)
                metrics[f"{prefix}/hold_failure_rate"] = (
                    (opportunities & broken[:, targets].any(-1)).float().sum() / denominator)
                metrics[f"{prefix}/missed_wheel_rate"] = (
                    (opportunities & self.state.missed_seen[ids][:, targets].any(-1)).float().sum() / denominator)
                metrics[f"{prefix}/unprepared_collision_rate"] = (
                    (subset & collisions[:, targets].any(-1)).float().sum() / subset.sum().clamp_min(1)
                )
                metrics[f"{prefix}/preparation_rate"] = (
                    (opportunities & (self.state.preparation_best[ids][:, targets] > 0).any(-1)).float().sum()
                    / denominator
                )
                ready = self.state.baseline_samples[ids][:, targets] >= math.ceil(
                    self.preview_cfg.baseline_stable_s / self._env.step_dt - 1e-9)
                metrics[f"{prefix}/baseline_ready_rate"] = (
                    (opportunities & ready.all(-1)).float().sum() / denominator)
        self.state.reset(env_ids)
