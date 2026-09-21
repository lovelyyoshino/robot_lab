"""首回合前缀的运动诊断记录；区分link/COM，不改变评测或跨reset拼接轨迹。

@author pony
@date 2026-09-17
@version v0.5.0
@last_modified 2026-09-17
@changelog
    - v0.5.0 (2026-09-17): 直接读取全连杆link位姿，供区分几何映射与动力学分叉。
    - v0.4.0 (2026-09-17): 对照子步状态缓存和PhysX直接查询，不额外刷新物理或接触。
    - v0.3.0 (2026-09-17): 记录PhysX真实逐关节速度上限，供硬限速单变量对照。
    - v0.2.0 (2026-09-17): 可选物理子步采样，区分驱动目标/近似力矩/约束反力，不推进传感器缓存。
    - v0.1.0 (2026-09-17): 记录动作前原始输入和自动reset前物理末态，供跨引擎差分。
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path


class MotionTraceRecorder:
    """仅采集选定环境首回合；结束标记由env.step返回后补齐，状态必须在reset前采集。"""

    def __init__(self, env_id: int, max_steps: int):
        if not isinstance(env_id, int) or env_id < 0 or not isinstance(max_steps, int) or not 1 <= max_steps <= 1000:
            raise ValueError("trace环境索引须非负，步数须为1–1000整数")
        self.env_id = env_id
        self.max_steps = max_steps
        self.steps: list[dict] = []
        self.pending: dict | None = None
        self.stop_reason: str | None = None
        self.physics_spec: dict | None = None
        self._hooks_active = False

    def _row(self, tensor, env, width):
        import torch

        if self.env_id >= env.num_envs or tensor.shape != (env.num_envs, width):
            raise ValueError("trace张量必须与环境数及观测合同一致")
        row = tensor[self.env_id].detach().cpu()
        if not torch.isfinite(row).all():
            raise ValueError("trace不能保存非有限状态")
        return row.tolist()

    def _state(self, env):
        import torch

        robot = env.scene["robot"]
        names = list(robot.joint_names)
        if len(names) != 20 or len(set(names)) != len(names):
            raise ValueError("motion trace必须包含20个唯一主动/被动关节")
        state = {
            "joint_names": names,
            "env_origin_w": self._row(env.scene.env_origins, env, 3),
            "episode_step": int(env.episode_length_buf[self.env_id]),
        }
        # root_state_w混合link位姿与COM速度；显式记录两个API，禁止当成同一参考点。
        for key, width in (
            ("root_link_state_w", 13), ("root_com_state_w", 13),
            ("joint_pos", 20), ("joint_vel", 20), ("default_joint_pos", 20),
            ("projected_gravity_b", 3), ("root_lin_vel_b", 3), ("root_ang_vel_b", 3),
        ):
            state[key] = self._row(getattr(robot.data, key), env, width)
        bodies = list(robot.body_names)
        if len(bodies) != 21 or len(set(bodies)) != 21 or "base_link" not in bodies:
            raise ValueError("全车位姿必须包含21个唯一link名称及base_link")
        # 直接读取已完成物理步的PhysX位姿；不额外调用FK更新或推进传感器历史。
        poses = robot.root_physx_view.get_link_transforms()
        if poses.shape != (env.num_envs, len(bodies), 7):
            raise ValueError("link位姿必须按body_names提供[N,21,7]数据")
        row = poses[self.env_id].detach().cpu()
        if (not torch.isfinite(row).all()
                or not torch.all((row[:, 3:].norm(dim=-1) - 1.0).abs() <= 1e-5)):
            raise ValueError("link位姿必须有限且四元数归一化")
        state.update(body_names=bodies, body_link_pose_contract="world_xyz_wxyz_link_v1",
                     body_link_pose_w=row[:, [0, 1, 2, 6, 3, 4, 5]].tolist())
        return state

    def begin(self, env, observation, action):
        if self.stop_reason is not None:
            return
        if self.pending is not None:
            raise RuntimeError("上一trace物理步尚未完成")
        pre = self._state(env)
        if pre["episode_step"] != len(self.steps):
            raise ValueError("trace只能从首次reset的第0步开始且不得跨reset拼接")
        self.pending = {
            "tick": len(self.steps), "pre": pre,
            "policy_observation": self._row(observation, env, 44),
            "policy_action": self._row(action, env, 12),
        }

    def capture_post(self, env):
        if self.pending is None:
            return
        if "post" in self.pending:
            raise RuntimeError("同一trace物理步不能重复采样")
        if self.physics_spec is not None:
            samples = self.pending.get("physics_substeps", [])
            if len(samples) != self.physics_spec["decimation"] or any("post" not in sample for sample in samples):
                raise RuntimeError("trace物理子步缺失，不能保存不完整控制步")
        post = self._state(env)
        if post["episode_step"] != self.pending["pre"]["episode_step"] + 1:
            raise ValueError("trace末态必须为当前动作后的一个控制步且在reset之前")
        self.pending["post"] = post
        self.pending["applied_action"] = self._row(env.action_manager.action, env, 12)

    @contextmanager
    def physics_hooks(self, env):
        """仅在评测实例包裹已有写入/更新边界；不改Isaac库、不额外步进或刷新接触历史。"""
        if self._hooks_active or self.pending is not None or self.steps:
            raise RuntimeError("子步钩子必须在首次动作之前且不能重复安装")
        dt, decimation = float(env.cfg.sim.dt), env.cfg.decimation
        if not math.isfinite(dt) or dt <= 0 or not isinstance(decimation, int) or decimation < 1:
            raise ValueError("子步采样须有有限正物理步长和正整数decimation")
        robot = env.scene["robot"]
        implicit, explicit = [], []
        for actuator in robot.actuators.values():
            indices = list(range(20))[actuator.joint_indices] if isinstance(actuator.joint_indices, slice) else actuator.joint_indices
            names = [robot.joint_names[index] for index in indices]
            (implicit if actuator.is_implicit_model else explicit).extend(names)
        if len(implicit + explicit) != 20 or len(set(implicit + explicit)) != 20:
            raise ValueError("所有20关节必须恰有一个执行器归属")
        self.physics_spec = {
            "contract": "rover_motion_physics_trace_v1", "physics_dt_s": dt, "decimation": decimation,
            "joint_names": list(robot.joint_names), "implicit_joint_names": implicit, "explicit_joint_names": explicit,
            "joint_armature_kg_m2": self._row(robot.root_physx_view.get_dof_armatures(), env, 20),
            "joint_velocity_limit_sim_rad_s": self._row(robot.root_physx_view.get_dof_max_velocities(), env, 20),
            "drive_sampling": "after_write_data_to_sim_before_sim_step",
            "post_sampling": "after_scene_update_before_episode_counter_increment",
            "implicit_drive_force_measured": False,
            "implicit_torque_semantics": "actuator_model_torque_is_PD_estimate_not_PhysX_solved_drive_force",
            "projected_force_semantics": "incoming_joint_wrench_projected_on_joint_axis_not_isolated_motor_or_mimic_force",
            "contact_semantics": "net_normal_only_world_frame_direct_PhysX_query_no_sensor_history_update",
            "tangential_contact_force_available": False,
            "tangential_unavailable_reason": "existing_sensor_has_no_filtered_friction_data_buffer",
        }
        scene = env.scene
        original_write, original_update = scene.write_data_to_sim, scene.update
        instance_values = {key: scene.__dict__.get(key) for key in ("write_data_to_sim", "update")}
        instance_keys = set(scene.__dict__)

        def write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            self._capture_physics_pre(env)
            return result

        def update(dt, *args, **kwargs):
            if self.pending is not None and not math.isclose(float(dt), self.physics_spec["physics_dt_s"], abs_tol=1e-12):
                raise ValueError("scene更新步长与物理采样合同不同")
            result = original_update(dt, *args, **kwargs)
            self._capture_physics_post(env)
            return result

        scene.write_data_to_sim, scene.update = write, update
        self._hooks_active = True
        try:
            yield
        finally:
            for key in ("write_data_to_sim", "update"):
                if key in instance_keys:
                    setattr(scene, key, instance_values[key])
                else:
                    delattr(scene, key)
            self._hooks_active = False

    def _capture_physics_pre(self, env):
        if self.pending is None or "post" in self.pending:
            return
        samples = self.pending.setdefault("physics_substeps", [])
        if len(samples) >= self.physics_spec["decimation"] or (samples and "post" not in samples[-1]):
            raise RuntimeError("物理子步写入重复或缺失上一次更新")
        robot = env.scene["robot"]
        # 隐式PD的applied_torque是估计量；实际写入effort为零不表示未驱动。
        drive = {
            "actuator_model_computed_torque_nm": self._row(robot.data.computed_torque, env, 20),
            "actuator_model_clipped_torque_nm": self._row(robot.data.applied_torque, env, 20),
            "sim_position_target_rad": self._row(robot._joint_pos_target_sim, env, 20),
            "sim_velocity_target_rad_s": self._row(robot._joint_vel_target_sim, env, 20),
            "sim_effort_target_nm": self._row(robot._joint_effort_target_sim, env, 20),
        }
        samples.append({"index": len(samples), "pre": self._state(env), "drive": drive})

    def _capture_physics_post(self, env):
        import torch

        if self.pending is None or "post" in self.pending:
            return
        samples = self.pending.get("physics_substeps", [])
        if not samples or "post" in samples[-1]:
            raise RuntimeError("物理末态采样必须紧跟一次驱动写入")
        sample = samples[-1]
        sample["post"] = self._state(env)
        view = env.scene["robot"].root_physx_view
        backend_q = self._row(view.get_dof_positions(), env, 20)
        backend_dq = self._row(view.get_dof_velocities(), env, 20)
        sample["joint_cache_max_abs_error"] = {
            "position_rad": max(abs(a - b) for a, b in zip(backend_q, sample["post"]["joint_pos"])),
            "velocity_rad_s": max(abs(a - b) for a, b in zip(backend_dq, sample["post"]["joint_vel"])),
        }
        sensor = env.scene["contact_forces"]
        names = list(sensor.body_names)
        forces = sensor.contact_physx_view.get_net_contact_forces(dt=self.physics_spec["physics_dt_s"])
        if forces.shape != (env.num_envs * len(names), 3) or not torch.isfinite(forces).all():
            raise ValueError("接触法向力须按sensor.body_names排列且有限")
        sample["contact"] = {"body_names": names, "net_normal_force_world_n":
                             forces.reshape(env.num_envs, len(names), 3)[self.env_id].detach().cpu().tolist()}
        sample["projected_incoming_joint_force_nm"] = self._row(
            env.scene["robot"].root_physx_view.get_dof_projected_joint_forces(), env, 20)

    def finish(self, terminated: bool, truncated: bool):
        if self.pending is None:
            return
        if "post" not in self.pending:
            raise RuntimeError("缺少reset前末态，不得拿reset后的状态补写trace")
        self.pending.update(terminated=bool(terminated), truncated=bool(truncated))
        self.steps.append(self.pending)
        self.pending = None
        if terminated or truncated:
            self.stop_reason = "first_episode_ended"
        elif len(self.steps) == self.max_steps:
            self.stop_reason = "requested_prefix_complete"

    def save(self, path: Path, metadata: dict) -> dict:
        if self.pending is not None or not self.steps:
            raise RuntimeError("不能保存未完成的物理步或空trace")
        payload = {
            "metadata": metadata, "trace_contract": "rover_motion_step_trace_v1",
            "scope": "cross_engine_diagnostic_not_skill_acceptance",
            "env_id": self.env_id, "requested_steps": self.max_steps,
            "stop_reason": self.stop_reason or "evaluation_stopped_before_requested_prefix",
            "contains_contact_solver_state": False,
            "state_layout": "position_xyz_quaternion_wxyz_world_linear_xyz_world_angular_xyz",
            "root_velocity_reference": "root_link_state_w=link_origin; root_com_state_w=center_of_mass",
            "policy_input_stage": "observation_manager_output_before_runner_normalization",
            "applied_action_semantics": "action_manager_raw_after_wrapper_clipping_before_action_term_scaling",
            "steps": self.steps,
        }
        if self.physics_spec is not None:
            payload["physics_trace"] = self.physics_spec
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "steps": len(self.steps), "env_id": self.env_id, "stop_reason": payload["stop_reason"]}
