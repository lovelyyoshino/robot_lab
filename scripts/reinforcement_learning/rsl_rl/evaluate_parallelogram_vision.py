#!/usr/bin/env python3
"""固定地形、朝向和命令的视觉通行评测，兼容 blind/teacher/student。

@author pony
@date 2026-09-16
@version v0.12.0
@last_modified 2026-11-07
@changelog
    - v0.12.0 (2026-11-07): --supermimic_terrain 未给 --wheel_friction 时用 terrain_env_cfg.WHEEL_FRICTION（固定 μ=1.0），
      与 SuperMimic 训练任务一致，并记入 terrain_override。
    - v0.11.0 (2026-11-07): 诊断参数 --wheel_friction STATIC DYNAMIC（固定轮胎摩擦，验证楼梯后轴对摩擦的依赖）。
    - v0.10.0 (2026-11-06): --supermimic_terrain 的单台阶预抬使用 acceptance_metrics.PRELIFT_WINDOW（3–12 cm 起抬）。
    - v0.9.0 (2026-11-05): --acceptance_metrics 逐回合姿态/平顺/四轮踢面碰撞（accept.py）；单台阶/楼梯几何运行时覆盖；
      232/233 维按 actor 权重切换车道偏移列；删除旧的扫描消融 --scan_mode。
    - (2026-10-28): --supermimic_terrain 自动识别方案 B 策略并追加车道偏移观测。
    - (2026-10-27): 楼梯途中跌倒按 stairs_acceptance.CLIMB_TILT_DEG（16 cm 45°），终点稳定仍 31.8°；--climb_tilt_deg 覆盖。
    - (2026-10-26): 新增 --supermimic_terrain，评测 supermimic 越障策略（TorchScript 232→12）。
    - (2026-10-26): 新增 --scan_student，评测 supermimic 深度→扫描学生的 TorchScript，按其 depth_contract.json 校验。
    - (2026-10-14): Preview楼梯按8/12/16cm真实登顶验收，连续稳定后截断，并允许全宽单侧主动。
    - (2026-10-13): 独立评测使用稳定接近基线配置，不放宽10cm起抬或完整通过标准。
    - (2026-10-11): 全宽保留阻挡碰撞检查，分别输出真实离地和高度达标距离。
    - (2026-10-10): 深度关键帧诊断对齐8–14cm动作窗口，与65cm感知预瞄区分。
    - (2026-10-10): 在Kit关闭前打印评测异常，保留执行失败退出码2，避免无提示退出。
    - (2026-10-10): 固定单障碍启用保持到前缘及连续落轮验收，输出分段动作完成率。
    - (2026-10-09): 窄障碍验收采用持续减载与净空，保留终态三轮支撑，Preview门槛改为70%。
    - v0.8.0 (2026-10-08): 统一主动抬轮协议，区分瞬时几何势能与事件奖励并输出简明结果。
    - v0.7.0 (2026-09-30): 新增Preview单障碍双前轮时序验收，保留通行v4独立成绩。
    - v0.6.0 (2026-09-17): v4 终点要求地形法向三轮承重、半体重及非轮接触拒绝，保留跨沟腾空。
    - v0.5.0 (2026-09-17): 允许窄沟跨越时合法未命中，终点仍须完整地面扫描，更新至 v3 协议。
    - v0.4.0 (2026-09-17): 修复高差地形误判，拒绝无效扫描，保留窄沟平台掉坑门并版本化协议。
    - v0.3.0 (2026-09-17): 提取自动reset前通行采样入口，支持地形高度判据回归。
    - v0.2.0 (2026-09-16): 增加时序student与通信扰动评测，输出实际深度预处理合同。
    - v0.1.0 (2026-09-16): 增加楼梯、方块、坡面、窄沟与深度失效对照评测。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import traceback
from pathlib import Path

from skill_evaluation import SkillEvaluation
from vision_contract import contract_for_cfg, validate_checkpoint
from vision_prelift_evaluation import (
    PRELIFT_ACTION_MAX_ADVANCE_M,
    PRELIFT_ACTION_MIN_ADVANCE_M,
    PRELIFT_MINIMUM_SUCCESS_RATE,
    PreliftEvaluation,
    wheel_lane_pixel_counts,
)
from vision_support import (
    TRAVERSAL_PROTOCOL,
    capture_ground_support,
    configure_support_sensors,
    support_contract,
    support_rows,
)

TASKS = {
    "blind": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0",
    "teacher": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-v0",
    "student": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-v0",
    "student-temporal": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-v0",
    "teacher-preview": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0",
    "student-preview": "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0",
}
SCENARIOS = ("pyramid_stairs", "pyramid_stairs_inv", "boxes", "hf_pyramid_slope", "gaps",
             "single_step", "single_step_left", "single_step_right")
STEP_FRONT_WHEELS = {"single_step": (0, 1), "single_step_left": (0,), "single_step_right": (1,)}


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_arg = parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", choices=tuple(TASKS), default="student")
    parser.add_argument("--scenario", choices=SCENARIOS, default="pyramid_stairs")
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes_per_env", type=int, default=2)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--depth_mode", choices=("live", "blank", "frozen"), default="live")
    parser.add_argument("--camera_corruption", action="store_true", help="开启Temporal深度的延迟/丢帧/冻结测试。")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--depth_snapshot", type=Path, help="可选：保存初始裁剪深度PNG，非Actor能力证明。")
    parser.add_argument("--supermimic_terrain", action="store_true",
                        help="checkpoint 是 scripts/supermimic/train_terrain.py 导出的 TorchScript（232→12），"
                             "仅用于 teacher-preview；按 run 目录 params/vision_contract.json 校验。")
    parser.add_argument("--scan_student", action="store_true",
                        help="checkpoint 是 scripts/supermimic/train_scan_student.py 导出的 TorchScript（4141→12），"
                             "仅用于 student-preview；按 run 目录的 depth_contract.json 校验，不走 rsl_rl runner。")
    parser.add_argument("--climb_tilt_deg", type=float, default=None,
                        help="覆盖楼梯途中倾角门；默认按 stairs_acceptance.CLIMB_TILT_DEG（8/12 cm 31.8°，16 cm 45°）。")
    parser.add_argument("--acceptance_metrics", action="store_true",
                        help="记录逐回合姿态、平顺与四轮踢面碰撞（acceptance_metrics.py），供 accept.py 汇总。")
    parser.add_argument("--step_height_m", type=float, default=None,
                        help="单台阶高度覆盖（运行时改 cfg，不改训练配置文件），如 0.10/0.14/0.175。")
    parser.add_argument("--stair_riser_m", type=float, default=None, help="楼梯踢面高度覆盖，需同时给 --stair_tread_m。")
    parser.add_argument("--stair_tread_m", type=float, default=None, help="楼梯踏面深度覆盖，如 0.40/0.28。")
    parser.add_argument("--stair_steps", type=int, default=None,
                        help="楼梯级数覆盖；默认按 stairs_acceptance.ACCEPTANCE_STAIR_STEP_COUNTS。")
    parser.add_argument("--wheel_friction", type=float, nargs=2, default=None, metavar=("STATIC", "DYNAMIC"),
                        help="诊断：把轮胎摩擦固定为给定值（默认按任务随机 static 0.5–1.2 / dynamic 0.4–1.0）。")
    AppLauncher.add_app_launcher_args(parser)
    checkpoint_arg.required = True
    args = parser.parse_args()
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("difficulty 必须在 [0,1] 内")
    if args.num_envs < 1 or args.episodes_per_env < 1:
        parser.error("环境数与每环境配额必须为正")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint 不存在：{args.checkpoint}")
    if not args.policy.startswith("student") and args.depth_mode != "live":
        parser.error("深度失效对照仅适用于 student")
    if args.acceptance_metrics and not args.policy.endswith("-preview"):
        parser.error("--acceptance_metrics 需要 Preview 任务（四轮几何与接触力来自 vision_preview）")
    if args.step_height_m is not None and (args.scenario not in STEP_FRONT_WHEELS or not 0 < args.step_height_m <= 0.25):
        parser.error("--step_height_m 只用于单台阶且须在 (0, 0.25] m")
    if (args.stair_riser_m is None) != (args.stair_tread_m is None) or (
            args.stair_steps is not None and args.stair_riser_m is None):
        parser.error("--stair_riser_m 与 --stair_tread_m 必须同时给出，--stair_steps 依赖二者")
    if args.stair_riser_m is not None and (args.scenario != "pyramid_stairs_inv" or not args.policy.endswith("-preview")
                                           or not 0 < args.stair_riser_m <= 0.25 or not 0.2 <= args.stair_tread_m <= 0.6):
        parser.error("楼梯几何覆盖只用于 Preview 的 pyramid_stairs_inv，踢面 (0,0.25] m、踏面 [0.2,0.6] m")
    if args.depth_snapshot is not None and not args.policy.startswith("student"):
        parser.error("depth_snapshot需要student相机")
    if args.camera_corruption and args.policy not in ("student-temporal", "student-preview"):
        parser.error("camera_corruption只适用于时序student")
    if args.scenario in STEP_FRONT_WHEELS and not args.policy.endswith("-preview"):
        parser.error("single_step提前抬轮协议需要Preview任务")
    if args.scan_student and args.policy != "student-preview":
        parser.error("--scan_student 只适用于 --policy student-preview")
    if args.supermimic_terrain and args.policy != "teacher-preview":
        parser.error("--supermimic_terrain 只适用于 --policy teacher-preview")
    if args.policy.startswith("student"):
        args.enable_cameras = True
    return args


def capture_traversal(env, *, scenario: str):
    """在物理步末、自动 reset 前记录通行状态；供实际终止项和回归测试共用。"""
    import torch

    if scenario not in SCENARIOS:
        raise ValueError(f"未知视觉评测地形：{scenario}")
    robot = env.scene["robot"]
    relative = robot.data.root_pos_w - env.scene.env_origins
    finite = torch.isfinite(robot.data.root_state_w).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_pos).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel).all(dim=1)
    finite &= torch.isfinite(relative).all(dim=1)
    finite &= torch.isfinite(robot.data.projected_gravity_b).all(dim=1)
    finite &= torch.isfinite(robot.data.root_lin_vel_b).all(dim=1)
    hits = env.scene.sensors["height_scanner_base"].data.ray_hits_w
    hit_finite = torch.isfinite(hits).all(dim=-1)
    # 无底沟的正常射线未命中是 xyz 全 +inf；NaN、-inf 或半损坏坐标仍拒绝。
    hit_valid = hit_finite | (torch.isposinf(hits).all(dim=-1) & (scenario == "gaps"))
    finite &= hit_valid.all(dim=-1)
    support = capture_ground_support(env)
    finite &= support["valid"]
    upright = robot.data.projected_gravity_b[:, 2] < -0.5
    in_lane = (relative[:, 1].abs() <= 1.0) & (relative[:, 0] <= 3.75)
    fallen = ~finite | ~upright | ~in_lane
    # 楼梯/坡面的 origin 是地形中心，正常下行也会低于它；不能当作局部地面。
    # 窄沟两岸与中心平台等高，必须保留平台高度门，防止沿沟底行驶也算通过。
    if scenario == "gaps":
        fallen |= relative[:, 2] < -0.2
    # 支撑只约束末端保持，不因跨沟腾空或越障时短暂非轮接触提前终止。
    stable = (relative[:, 0] >= 2.5) & ~fallen & hit_finite.all(dim=-1) & support["supported"]
    squared_error = torch.square(robot.data.root_lin_vel_b[:, 0] - 0.4)
    protocol = getattr(env.cfg, "vision_evaluation_protocol", None)
    if protocol is not None:
        finite &= torch.isfinite(robot.data.root_lin_vel_w).all(-1)
        quat = robot.data.root_quat_w
        alignment = 1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square())
        cos_tilt = -robot.data.projected_gravity_b[:, 2]
        upright = cos_tilt >= 0.85
        # 途中跌倒按楼梯档位的倾角门（16 cm 放宽）；终点稳定仍要求 31.8° 内，顶端平台是平地。
        climb_upright = cos_tilt >= math.cos(math.radians(protocol.get("climb_tilt_deg", 31.8)))
        in_lane = (relative[:, 1].abs() <= 0.75) & (relative[:, 0] <= 3.75)
        fallen = ~finite | ~climb_upright | ~in_lane
        stable = (relative[:, 0] >= protocol["target_distance_m"]) & ~fallen & upright
        stable &= (relative[:, 1].abs() <= 0.15) & (alignment >= 0.8)
        stable &= hit_finite.all(-1) & support["supported"]
        if "total_height_m" in protocol:
            # 后轮与车身须真正登顶；绕楼梯从侧边平地通过不能冒充爬楼成功。
            ground_height = hits[..., 2] - env.scene.env_origins[:, None, 2]
            stable &= ground_height.amin(-1) >= protocol["total_height_m"] - 0.02
        squared_error = (robot.data.root_lin_vel_w[:, 0] - protocol["command_velocity_m_s"]).square()
        if not hasattr(env, "_vision_eval_hold_steps"):
            env._vision_eval_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=relative.device)
            env._vision_eval_max_progress = torch.zeros(env.num_envs, device=relative.device)
        fresh = env.episode_length_buf <= 1
        env._vision_eval_hold_steps[fresh] = 0
        safe_progress = torch.nan_to_num(relative[:, 0], nan=0., posinf=0., neginf=0.)
        env._vision_eval_max_progress[fresh] = safe_progress[fresh]
        env._vision_eval_hold_steps = torch.where(stable, env._vision_eval_hold_steps + 1, 0)
        env._vision_eval_max_progress = torch.maximum(env._vision_eval_max_progress, safe_progress)
        env._vision_eval_confirmed = env._vision_eval_hold_steps >= math.ceil(protocol["hold_time_s"] / env.step_dt - 1e-9)
    env._vision_eval_sample = (
        stable.cpu().tolist(), finite.cpu().tolist(),
        relative[:, 0].cpu().tolist(), squared_error.cpu().tolist(),
    )
    env._vision_eval_support = support_rows(support)
    acceptance_spec = getattr(env.cfg, "acceptance_metrics", None)
    if scenario in STEP_FRONT_WHEELS or acceptance_spec is not None:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover import (
            vision_preview,
        )

        sample = vision_preview.collect_preview(env)
    if acceptance_spec is not None:
        from acceptance_metrics import capture_acceptance_sample

        env._acceptance_sample = capture_acceptance_sample(env, sample, finite)
    if scenario in STEP_FRONT_WHEELS:
        # 固定地形前缘比离散扫描命中的栅格更准确；必须在自动reset前测量。
        terrain = env.cfg.scene.terrain.terrain_generator.sub_terrains[scenario]
        front_x = vision_preview.step_front_x_for_difficulty(
            env.cfg.scene.terrain.terrain_generator.difficulty_range[0], terrain.step_front_x_range,
            env.cfg.scene.terrain.terrain_generator.seed,
        )
        centers = sample["wheel_centers_w"][:, [0, 2]]
        lead = (env.scene.env_origins[:, None, 0] + front_x
                - centers[:, :, 0] - sample["wheel_radii"][[0, 2]])
        clearance = centers[:, :, 2] - sample["wheel_radii"][[0, 2]] - env.scene.env_origins[:, None, 2]
        reward_clearance = sample["clearance_m"][:, [0, 2]]
        lift = sample["articulation_lift_m"][:, [0, 2]]
        evaluation_gates = {**sample["eligible_gates"], "finite": finite,
                            "in_lane": in_lane, "started": relative[:, 0] > 0.02}
        if scenario != "single_step":
            evaluation_gates["supported"] = vision_preview.narrow_support_mask(
                support["wheel_upward_force_n"], support["robot_weight_n"],
            )
        gate_lists = {name: values.cpu().tolist() for name, values in evaluation_gates.items()}
        gate_rows = [{name: values[i] for name, values in gate_lists.items()} for i in range(env.num_envs)]
        eligible = torch.stack(tuple(evaluation_gates.values()), dim=-1).all(-1)
        resisting = sample["wheel_resisting_force_n"][:, [0, 2]]
        upward = support["wheel_upward_force_n"][:, [0, 2]]
        preview_cfg = env.cfg.vision_preview
        difficulty = env.cfg.scene.terrain.terrain_generator.difficulty_range[0]
        step_height = terrain.step_height_range[0] + difficulty * (
            terrain.step_height_range[1] - terrain.step_height_range[0]
        )
        blocking = (
            (resisting >= preview_cfg["collision_resisting_force_n"]) & (resisting > 0.5 * upward.clamp_min(0))
            & (lead <= preview_cfg["scan_edge_margin_m"]) & (lead >= -preview_cfg["passage_margin_m"])
            & (clearance < step_height)
        )
        # 全宽也检查未准备撞前缘，不能把滚上去混为提前抬轮。
        env._vision_prelift_sample = [
            {"lead_m": a, "lift_m": b, "clearance_m": c, "wheel_support_force_n": force,
             "blocking_collision": collision,
             "eligible": d,
             "preview": e, "reward_eligible": f, "reward_clearance_m": g,
             "hold_eligible": all(passed for name, passed in gates.items()
                                  if name not in {"forward_command", "forward_motion", "started"}),
             "preview_potential_raw": h, "eligibility_gates": gates}
            for a, b, c, force, collision, d, e, f, g, h, gates in zip(
                lead.cpu().tolist(), lift.cpu().tolist(), clearance.cpu().tolist(),
                upward.cpu().tolist(), blocking.cpu().tolist(),
                eligible.cpu().tolist(), sample["preview"][:, [0, 2]].cpu().tolist(),
                sample["eligible"].cpu().tolist(),
                reward_clearance.cpu().tolist(), sample["potential"].cpu().tolist(), gate_rows, strict=True,
            )
        ]
    return fallen


def traversal_complete(env):
    """failure回调先采样，本项仅消费同一步的连续稳定证据。"""
    return env._vision_eval_confirmed


def capture_depth_probe(env, args, *, save_snapshot=True):
    """用真实渲染深度反投影，核对单障碍是否在裁剪后的有效图像中。"""
    import torch

    from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_apply, quat_mul

    camera = env.scene.sensors["depth_camera"].data
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies("base_link")
    if len(body_ids) != 1 or env.cfg.scene.depth_camera.offset.convention != "world":
        raise ValueError("深度诊断要求已审核的base_link相机和world外参约定")
    base_pos = robot.data.body_pos_w[0, body_ids[0]]
    base_quat = robot.data.body_quat_w[0, body_ids[0]]
    offset = env.cfg.scene.depth_camera.offset
    # Fabric模式下XFormPrim元数据可能仍是USD初始位姿；使用真实刚体状态组合固定外参。
    camera_pos = base_pos + quat_apply(base_quat, base_pos.new_tensor(offset.pos))
    camera_quat = quat_mul(base_quat, base_quat.new_tensor(offset.rot))
    camera_ros = convert_camera_frame_orientation_convention(camera_quat[None], origin="world", target="ros")[0]
    raw = camera.output["distance_to_image_plane"][0, :, :, 0]
    params = env.cfg.observations.policy.depth_image.params
    top, bottom = params["crop_top"], raw.shape[0] - params["crop_bottom"]
    cropped = raw[top:bottom]
    finite = torch.isfinite(cropped)
    valid = finite & (cropped > params["min_depth"]) & (cropped < params["max_depth"])
    result = {"scope": "render_env0_not_visual_policy_acceptance",
              "finite_fraction": finite.float().mean().item(), "valid_fraction": valid.float().mean().item(),
              "valid_depth_min_m": cropped[valid].min().item() if valid.any() else None,
              "valid_depth_max_m": cropped[valid].max().item() if valid.any() else None,
              "camera_position_relative_origin_m": (camera_pos - env.scene.env_origins[0]).tolist(),
              "camera_quat_wxyz_world": camera_quat.tolist(),
              "pose_source": "live_base_link_plus_fixed_camera_offset",
              "intrinsics": camera.intrinsic_matrices[0].tolist()}
    if args.scenario in STEP_FRONT_WHEELS:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover import (
            vision_preview,
        )

        y, x = torch.meshgrid(torch.arange(top, bottom, device=raw.device),
                              torch.arange(raw.shape[1], device=raw.device), indexing="ij")
        k = camera.intrinsic_matrices[0]
        points = torch.stack(((x - k[0, 2]) / k[0, 0] * cropped,
                              (y - k[1, 2]) / k[1, 1] * cropped, cropped), -1).reshape(-1, 3)
        camera_rotations = camera_ros.reshape(1, 4).expand(points.shape[0], 4)
        points = quat_apply(camera_rotations, points)
        points += camera_pos - env.scene.env_origins[0]
        terrain = env.cfg.scene.terrain.terrain_generator.sub_terrains[args.scenario]
        front_x = vision_preview.step_front_x_for_difficulty(args.difficulty, terrain.step_front_x_range, args.seed)
        height = terrain.step_height_range[0] + args.difficulty * (
            terrain.step_height_range[1] - terrain.step_height_range[0])
        lane_min, lane_max = vision_preview.step_lateral_bounds(terrain.lane_center_y, terrain.lane_width)
        near_step = ((points[:, 0] >= front_x - 0.015)
                     & (points[:, 0] <= front_x + terrain.step_width + 0.015)
                     & (points[:, 1] >= lane_min) & (points[:, 1] <= lane_max)
                     & (points[:, 2] > 0.015) & (points[:, 2] <= height + 0.015))
        obstacle_pixels = near_step & valid.flatten()
        result["pixels_consistent_with_known_step"] = obstacle_pixels.sum().item()
        result["step_height_m"] = height
        result["step_front_x_m"] = front_x
        result["step_lateral_bounds_m"] = [lane_min, lane_max]
        preview = vision_preview.collect_preview(env)
        front_wheel_centers = preview["wheel_centers_w"][0, [0, 2]] - env.scene.env_origins[0]
        front_wheel_radii = preview["wheel_radii"][[0, 2]]
        wheel_y = front_wheel_centers[:, 1].tolist()
        wheel_lead = (front_x - front_wheel_centers[:, 0] - front_wheel_radii).tolist()
        counts = wheel_lane_pixel_counts(
            points, obstacle_pixels, wheel_y, env.cfg.vision_preview["lane_half_width_m"],
        )
        targets = STEP_FRONT_WHEELS[args.scenario]
        result["front_wheel_labels"] = ["fl", "fr"]
        result["front_wheel_lead_to_step_m"] = wheel_lead
        result["front_wheel_in_prelift_window"] = [
            PRELIFT_ACTION_MIN_ADVANCE_M <= lead <= PRELIFT_ACTION_MAX_ADVANCE_M for lead in wheel_lead
        ]
        result["visible_step_pixels_by_front_wheel"] = dict(zip(("fl", "fr"), counts, strict=True))
        result["target_front_wheels"] = ["fl" if wheel == 0 else "fr" for wheel in targets]
        result["target_wheel_visible_step_pixels"] = [counts[wheel] for wheel in targets]
        scan = env.scene.sensors["height_scanner"].data.ray_hits_w[0] - env.scene.env_origins[0]
        result["height_scan_max_z_m"] = scan[:, 2].max().item()
        result["reconstructed_valid_z_range_m"] = (
            [points[valid.flatten(), 2].min().item(), points[valid.flatten(), 2].max().item()]
            if valid.any() else None)
    if args.depth_snapshot is not None and save_snapshot:
        from PIL import Image

        gray = (torch.nan_to_num(cropped, nan=params["max_depth"], posinf=params["max_depth"])
                .clamp(params["min_depth"], params["max_depth"]) - params["min_depth"])
        gray = (gray / (params["max_depth"] - params["min_depth"]) * 255).byte().cpu().numpy()
        args.depth_snapshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gray).resize((640, 320), Image.Resampling.NEAREST).save(args.depth_snapshot)
        result["depth_snapshot"] = str(args.depth_snapshot.resolve())
    return result


def main(args, simulation_app):
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import torch
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_env_cfg import (
        _configure_vision_terrain,
        vision_depth_contract,
    )
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry

    task = TASKS[args.policy]
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    if args.supermimic_terrain:
        from vision_contract import restore_height_scan_period

        source_path = args.checkpoint.parent.parent / "params/vision_contract.json"
        restore_height_scan_period(cfg, json.loads(source_path.read_text()))
    # 三种策略在相同摩擦和地形下比较；blind 训练时的特殊摩擦不带入验收。
    preview_contract = contract_for_cfg(task, cfg, agent_cfg)
    scan_depth_contract = None
    actor_obs_dim = None
    if args.scan_student:
        from vision_contract import validate_scan_student

        from robot_lab.supermimic.scan_student import configure_saved_student

        scan_depth_contract = configure_saved_student(cfg, args.checkpoint)
        validate_scan_student(args.checkpoint, scan_depth_contract)
    elif args.supermimic_terrain:
        from vision_contract import validate_supermimic_terrain

        validate_supermimic_terrain(args.checkpoint, preview_contract)
        from acceptance_metrics import LANE_OFFSET_OBS_DIM, actor_input_dim

        actor_obs_dim = actor_input_dim(args.checkpoint)
        if actor_obs_dim == LANE_OFFSET_OBS_DIM:
            from robot_lab.supermimic.lane_offset import add_lane_offset_term

            # 旧方案 B 基线（233 维）：合同之外追加车道偏移列；当前越障策略为 232 维，不加。
            add_lane_offset_term(cfg)
    else:
        validate_checkpoint(args.checkpoint, preview_contract)
    if preview_contract is None:
        _configure_vision_terrain(cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = agent_cfg.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = 20.0 if preview_contract is not None else 8.0
    cfg.observations.policy.enable_corruption = False
    if args.camera_corruption:
        # 只保留自管深度通信扰动，消除本体观测噪声对消融的混淆。
        for name, term in vars(cfg.observations.policy).items():
            if name != "depth_image" and hasattr(term, "noise"):
                term.noise = None
        cfg.observations.policy.enable_corruption = True
    terrain = cfg.scene.terrain.terrain_generator
    if args.scenario not in terrain.sub_terrains:
        raise ValueError(f"{args.policy}课程不包含{args.scenario}")
    selected = terrain.sub_terrains[args.scenario]
    selected.proportion = 1.0
    terrain.sub_terrains = {args.scenario: selected}
    # 验收档位在运行时覆盖几何；合同已在上面按原配置计算，训练配置文件不变。
    terrain_override = None
    if args.step_height_m is not None:
        selected.step_height_range = (args.step_height_m, args.step_height_m)
        terrain_override = {"step_height_m": args.step_height_m}
    if args.stair_riser_m is not None:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.stairs_acceptance import (
            ACCEPTANCE_STAIR_STEP_COUNTS,
        )

        steps = args.stair_steps or ACCEPTANCE_STAIR_STEP_COUNTS.get(round(args.stair_riser_m, 3))
        if steps is None:
            raise ValueError(f"踢面 {args.stair_riser_m} m 没有默认级数，请给 --stair_steps")
        selected.height_levels = (args.stair_riser_m,) * 3
        selected.step_counts = (int(steps),) * 3
        selected.step_width = args.stair_tread_m
        terrain_override = {"stair_riser_m": args.stair_riser_m, "stair_tread_m": args.stair_tread_m,
                            "stair_steps": int(steps)}
    terrain.num_rows = 1
    terrain.num_cols = 4
    terrain.difficulty_range = (args.difficulty, args.difficulty)
    terrain.curriculum = False
    terrain.seed = args.seed
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = None
    for name in ("randomize_push_robot", "push_robot", "randomize_apply_external_force_torque"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {}
    wheel_friction = args.wheel_friction
    if wheel_friction is None and args.supermimic_terrain:
        # SuperMimic 训练与平地验收的任务固定了轮胎摩擦；Teacher-Preview 任务仍是随机范围，这里对齐
        from robot_lab.tasks.manager_based.supermimic.terrain_env_cfg import WHEEL_FRICTION

        wheel_friction = WHEEL_FRICTION
    if wheel_friction is not None:
        material = cfg.events.randomize_wheel_rigid_body_material.params
        material["static_friction_range"] = (wheel_friction[0],) * 2
        material["dynamic_friction_range"] = (wheel_friction[1],) * 2
        terrain_override = {**(terrain_override or {}), "wheel_friction": list(wheel_friction)}
    command = cfg.commands.base_velocity
    evaluation_speed = 0.30 if preview_contract is not None else 0.40
    command.ranges.lin_vel_x = (evaluation_speed, evaluation_speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.resampling_time_range = (25.0, 25.0)
    evaluation_protocol = None
    if preview_contract is not None:
        evaluation_protocol = {"target_distance_m": 2.5, "command_velocity_m_s": evaluation_speed,
                               "hold_time_s": 0.5, "episode_length_s": cfg.episode_length_s}
        if args.scenario == "pyramid_stairs_inv":
            from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.vision_preview import stairs_profile
            evaluation_protocol.update(stairs_profile(
                args.difficulty, selected.height_levels, selected.step_counts, selected.first_edge_m, selected.step_width,
            ))
            from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover.stairs_acceptance import (
                climb_tilt_deg,
                climb_tilt_for_riser,
            )
            override = args.climb_tilt_deg
            if override is None and args.stair_riser_m is not None:
                # 档位覆盖后 stage 恒为 1，按踢面取翻车门（≥14 cm 45°）
                override = climb_tilt_for_riser(args.stair_riser_m)
            evaluation_protocol["climb_tilt_deg"] = climb_tilt_deg(evaluation_protocol, override)
        cfg.vision_evaluation_protocol = evaluation_protocol
        cfg.terminations.preview_success = None
    support_sensors = configure_support_sensors(cfg)

    tracker = SkillEvaluation(args.num_envs, args.episodes_per_env, cfg.sim.dt * cfg.decimation,
                              0.5, cfg.episode_length_s - 0.5)
    prelift_window = {}
    if args.supermimic_terrain:
        from acceptance_metrics import PRELIFT_WINDOW

        prelift_window = PRELIFT_WINDOW
    prelift = (
        PreliftEvaluation(
            args.num_envs,
            tracker.dt,
            required_front_wheels=STEP_FRONT_WHEELS[args.scenario],
            required_active_count=1,
            allow_load_drop=args.scenario in {"single_step_left", "single_step_right"},
            load_drop_fraction=cfg.vision_preview["narrow_load_drop_fraction"],
            min_force_drop_n=cfg.vision_preview["narrow_min_force_drop_n"],
            min_clearance_m=cfg.vision_preview["narrow_min_clearance_m"],
            unload_hold_s=cfg.vision_preview["narrow_hold_s"],
            require_hold_to_edge=True,
            prelift_hold_s=cfg.vision_preview["prelift_hold_s"],
            release_hold_s=cfg.vision_preview["release_hold_s"],
            baseline_settle_s=cfg.vision_preview["baseline_settle_s"],
            baseline_stable_s=cfg.vision_preview["baseline_stable_s"],
            preparation_max_advance_m=cfg.vision_preview["preparation_max_advance_m"],
            **prelift_window,
        )
        if args.scenario in STEP_FRONT_WHEELS else None
    )
    acceptance = None
    if args.acceptance_metrics:
        from acceptance_metrics import EpisodeMetrics, single_step_risers, stairs_risers

        from robot_lab.assets.parallelogram_rover import PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES
        from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover import (
            vision_preview,
        )

        risers = None
        if args.scenario in STEP_FRONT_WHEELS:
            low, high = selected.step_height_range
            risers = single_step_risers(
                vision_preview.step_front_x_for_difficulty(args.difficulty, selected.step_front_x_range, terrain.seed),
                low + args.difficulty * (high - low),
                vision_preview.step_lateral_bounds(selected.lane_center_y, selected.lane_width),
            )
        elif args.scenario == "pyramid_stairs_inv" and evaluation_protocol is not None:
            risers = stairs_risers(selected.first_edge_m, selected.step_width, evaluation_protocol["step_height_m"],
                                   evaluation_protocol["num_steps"], selected.lane_width)
        cfg.acceptance_metrics = {"risers": risers, "arm_joint_names": list(PARALLELOGRAM_ROVER_SUSPENSION_JOINT_NAMES)}
        acceptance = EpisodeMetrics(args.num_envs, tracker.dt,
                              climb_tilt_deg=(evaluation_protocol or {}).get("climb_tilt_deg", 31.8),
                              target_distance_m=(evaluation_protocol or {}).get("target_distance_m", 2.5))
    tracking_error_sum = 0.0
    tracking_samples = 0
    terminal_progress = []
    terminal_support = []
    maximum_progress = []

    cfg.terminations.traversal_failure = TerminationTermCfg(
        func=capture_traversal, params={"scenario": args.scenario},
    )
    if evaluation_protocol is not None:
        cfg.terminations.traversal_complete = TerminationTermCfg(func=traversal_complete, time_out=True)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    try:
        from pxr import UsdPhysics

        from isaaclab.sim.utils.stage import get_current_stage

        ground_prim = get_current_stage().GetPrimAtPath(support_sensors["terrain_filter"])
        if not ground_prim.IsValid() or not ground_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("视觉承重过滤路径不存在或不具有 CollisionAPI")
        if args.scan_student or args.supermimic_terrain:
            scripted = torch.jit.load(str(args.checkpoint), map_location=env.unwrapped.device).eval()

            def policy(observation):
                return scripted(observation["policy"])
        else:
            runner_class = DistillationRunner if args.policy.startswith("student") else OnPolicyRunner
            runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(str(args.checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.reset()
        if actor_obs_dim is not None and obs["policy"].shape[-1] != actor_obs_dim:
            raise RuntimeError(f"环境 policy 观测 {obs['policy'].shape[-1]} 维，策略需要 {actor_obs_dim} 维")
        depth_probe_initial = (capture_depth_probe(env.unwrapped, args, save_snapshot=False)
                               if args.policy.startswith("student") else None)
        depth_probe = depth_probe_initial
        depth_probe_prelift_window = None
        evaluation_steps = 0
        depth_end = 45 + 8 * 16 * 32
        frozen_depth = obs["policy"][:, 45:depth_end].clone() if args.policy.startswith("student") else None
        while not tracker.complete and simulation_app.is_running():
            with torch.inference_mode():
                if args.policy.startswith("student") and args.depth_mode != "live":
                    obs = obs.clone()
                    obs["policy"][:, 45:depth_end] = 1.0 if args.depth_mode == "blank" else frozen_depth
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("策略输出非有限动作，终止评测")
                obs, _, dones, _ = env.step(actions)
            evaluation_steps += 1
            if evaluation_steps == 10 and args.policy.startswith("student"):
                depth_probe = capture_depth_probe(env.unwrapped, args)
            if (prelift is not None and args.policy.startswith("student")
                    and depth_probe_prelift_window is None and not bool(dones[0].item())):
                target_wheels = STEP_FRONT_WHEELS[args.scenario]
                leads = env.unwrapped._vision_prelift_sample[0]["lead_m"]
                if any(math.isfinite(leads[wheel])
                       and PRELIFT_ACTION_MIN_ADVANCE_M <= leads[wheel] <= PRELIFT_ACTION_MAX_ADVANCE_M
                       for wheel in target_wheels):
                    depth_probe_prelift_window = capture_depth_probe(
                        env.unwrapped, args, save_snapshot=False,
                    )
            stable, finite, progress, errors = env.unwrapped._vision_eval_sample
            terminated = env.unwrapped.reset_terminated.cpu().tolist()
            truncated = env.unwrapped.reset_time_outs.cpu().tolist()
            for index in range(args.num_envs):
                if tracker.counts[index] >= tracker.quota:
                    continue
                if finite[index] and math.isfinite(errors[index]):
                    tracking_error_sum += errors[index]
                    tracking_samples += 1
                if terminated[index] or truncated[index]:
                    terminal_progress.append(progress[index] if math.isfinite(progress[index]) else None)
                    terminal_support.append({"env_id": index, "episode_index": tracker.counts[index],
                                             **env.unwrapped._vision_eval_support[index]})
                    if evaluation_protocol is not None:
                        maximum_progress.append(float(env.unwrapped._vision_eval_max_progress[index].item()))
            tracker.update(stable, terminated, truncated, finite)
            if prelift is not None:
                prelift.update(env.unwrapped._vision_prelift_sample, tracker)
            if acceptance is not None:
                acceptance.update(env.unwrapped._acceptance_sample, tracker)
            if not (args.scan_student or args.supermimic_terrain):
                # 扫描学生的 GRU 只在 8 帧窗口内展开、跨步无状态，无需 reset。
                runner.alg.policy.reset(dones)
            if frozen_depth is not None:
                reset_ids = dones.bool()
                frozen_depth[reset_ids] = obs["policy"][reset_ids, 45:depth_end]
    finally:
        env.close()

    minimum_rate = PRELIFT_MINIMUM_SUCCESS_RATE if preview_contract is not None else 0.85
    result = tracker.summary(minimum_rate)
    if prelift is not None:
        prelift_result = prelift.summary(result)
        prelift_result["diagnostic_protocol"] = "rover_prelift_trace_summary_v1"
        result["traversal_passed"] = result["passed"]
        result["prelift"] = prelift_result
        result["passed"] = result["passed"] and prelift_result["passed"]
    with args.checkpoint.open("rb") as checkpoint_file:
        checksum = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result.update({
        "protocol": "rover_preview_route_climb" if evaluation_protocol is not None else TRAVERSAL_PROTOCOL,
        "task": task, "policy": args.policy, "scenario": args.scenario,
        "difficulty": args.difficulty, "seed": args.seed, "depth_mode": args.depth_mode,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checksum,
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
        "velocity_rmse_m_s": math.sqrt(tracking_error_sum / tracking_samples) if tracking_samples else None,
        "terminal_progress_m": terminal_progress,
        "target_distance_m": evaluation_protocol["target_distance_m"] if evaluation_protocol is not None else 2.5,
        "command_velocity_m_s": evaluation_speed,
        "episode_length_s": cfg.episode_length_s, "step_dt": tracker.dt,
        "climb_protocol": evaluation_protocol, "maximum_progress_m": maximum_progress,
        "terrain_height_gate": (
            "base_scan_on_top_platform_with_2cm_tolerance"
            if evaluation_protocol is not None and "total_height_m" in evaluation_protocol else
            "platform_relative_z_below_-0.2m" if args.scenario == "gaps" else "none"
        ),
        "requires_finite_base_height_scan": args.scenario != "gaps",
        "requires_finite_terminal_base_height_scan": True,
        "terminal_wheel_support_verified": tracker.complete,
        "terminal_support_contract": support_contract(),
        "support_sensor_configuration": support_sensors,
        "terminal_support": terminal_support,
        "base_height_scan_policy": "finite_xyz_or_all_positive_inf_during_gap_crossing; finite_xyz_at_terminal",
        "scope": "fixed_lane_traversal; does not establish general navigation or parkour",
        "camera_corruption": args.camera_corruption,
        "depth_contract": (scan_depth_contract or vision_depth_contract(cfg)) if args.policy.startswith("student") else None,
        "vision_contract": preview_contract,
        "depth_probe": depth_probe,
        "depth_probe_initial": depth_probe_initial,
        "depth_probe_prelift_window": depth_probe_prelift_window,
        "actor_obs_dim": actor_obs_dim,
        "terrain_override": terrain_override,
        "acceptance_metrics": acceptance.summary() if acceptance is not None else None,
        "acceptance_protocol": ("rover_single_step_prelift" if prelift is not None else
                                "rover_preview_route_climb" if evaluation_protocol is not None else TRAVERSAL_PROTOCOL),
    })
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"[评测结果] 完成={result['complete']}，回合={result['episodes']}，"
          f"通行率={result['success_rate']:.1%}，综合通过={result['passed']}")
    if prelift is not None:
        rates = result["prelift"]["front_wheel_event_rates"]
        print(f"[主动抬轮] 左前有效事件={rates['fl']:.1%}，右前有效事件={rates['fr']:.1%}，"
              f"目标轮提前避障并通行={result['prelift']['success_rate']:.1%}（要求{minimum_rate:.0%}）")
        held = result["prelift"]["front_wheel_held_to_edge_rates"]
        released = result["prelift"]["front_wheel_release_rates"]
        print(f"[动作保持] 左/右保持到前缘={held['fl']:.1%}/{held['fr']:.1%}，"
              f"左/右稳定落轮={released['fl']:.1%}/{released['fr']:.1%}")
        early = result["prelift"]["front_wheel_early_lift_rates"]
        print(f"[起抬时机] 目标约10cm（有效8–14cm），左/右早抬回合率={early['fl']:.1%}/{early['fr']:.1%}")
        for label, name in (("fl", "左前"), ("fr", "右前")):
            onset = result["prelift"]["takeoff_distance_m"][label]
            height = result["prelift"]["height_threshold_distance_m"][label]
            onset_text = f"{100 * onset['mean']:.1f}cm" if onset["mean"] is not None else "无记录"
            height_text = f"{100 * height['mean']:.1f}cm" if height["mean"] is not None else "无记录"
            print(f"[实际距离] {name}首次离地均值={onset_text}（n={onset['count']}），"
                  f"有效高度达标均值={height_text}（n={height['count']}）")
    if args.output is not None:
        print(f"[报告] {args.output.resolve()}")
    return result


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    args = parse_args()
    app = AppLauncher(args).app
    exit_code = 2
    try:
        result = main(args, app)
        exit_code = 0 if result["passed"] else 1
    except Exception:
        # Kit快速退出可能早于未捕获异常的打印；先输出原因，仍以2区分执行失败。
        traceback.print_exc()
    finally:
        # 显式通知 Kit，避免 fast shutdown 把失败评测退出码覆盖为 0。
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(exit_code)
        app.close()
    raise SystemExit(exit_code)
