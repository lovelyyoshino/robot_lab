"""回放专用深度角落面板；只读相机，不改变策略观测或时间历史。

@author pony
@date 2026-09-30
@version v0.1.0
@last_modified 2026-09-30
@changelog
    - v0.1.0 (2026-09-30): 1–4 台视觉车自动显示深度；teacher 临时加显示相机。
"""

from __future__ import annotations

import logging
import math
import os

LOGGER = logging.getLogger(__name__)
DEFAULT_PLAY_ENVS = 64
FRAME_NAME = "robot_lab.play.depth_overlay"
DATA_TYPE = "distance_to_image_plane"
TASK_PREFIX = "RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-"


def vision_role(task: str | None) -> str | None:
    """仅对本项目轮式车视觉任务自动开启，避免给其他机器人意外加相机。"""
    task = (task or "").split(":")[-1]
    if not task.startswith(TASK_PREFIX):
        return None
    variant = task[len(TASK_PREFIX) :]
    if variant in ("Teacher-v0", "Teacher-Preview-v0"):
        return "teacher"
    if variant in ("Student-v0", "Student-Temporal-v0", "Student-Temporal-Preview-v0"):
        return "student"
    return None


def depth_overlay_requested(args, environ=None) -> bool:
    """启动 Kit 前判断是否需要渲染；默认 64 台和键盘强制单车与 play 一致。"""
    environ = os.environ if environ is None else environ
    num_envs = 1 if args.keyboard else (args.num_envs if args.num_envs is not None else DEFAULT_PLAY_ENVS)
    # AppLauncher 的 HEADLESS/LIVESTREAM 也能开启无本地窗口模式，不能只看命令行。
    livestream = getattr(args, "livestream", -1)
    if livestream < 0:
        livestream = int(environ.get("LIVESTREAM", "0"))
    headless = args.headless or bool(int(environ.get("HEADLESS", "0"))) or livestream > 0
    return vision_role(args.task) is not None and 1 <= num_envs <= 4 and not headless and not args.disable_depth_overlay


def _teacher_display_camera(task):
    """复用对应 student 的标定，避免 Preview 的 25 度俯角在回放中悄悄变回 15 度。"""
    from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.parallelogram_rover import (
        vision_env_cfg,
        vision_preview_env_cfg,
    )

    if task.split(":")[-1] == TASK_PREFIX + "Teacher-Preview-v0":
        return vision_preview_env_cfg.ParallelogramRoverVisionStudentPreviewEnvCfg().scene.depth_camera.copy()
    return vision_env_cfg.ParallelogramRoverVisionSceneCfg.depth_camera.copy()


def configure_display_camera(env_cfg, task):
    """仅在 play 的独立配置实例补 teacher 相机；student 相机及所有观测保持原样。"""
    role = vision_role(task)
    if role is None:
        raise ValueError("深度面板只支持 Parallelogram Rover 的视觉任务")
    camera = getattr(env_cfg.scene, "depth_camera", None)
    if camera is None:
        if role != "teacher":
            raise ValueError("student 缺少策略必需的 depth_camera，不能用显示相机掩盖配置错误")
        camera = _teacher_display_camera(task)
        camera.update_period = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.scene.depth_camera = camera
        # teacher 原本无 RTX 传感器；复位后重渲染，避免展示上一回合的图像。
        env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 2)
    if DATA_TYPE not in camera.data_types:
        raise ValueError(f"深度面板要求相机输出 {DATA_TYPE}")
    return camera


def depth_to_rgba(depth, min_depth: float, max_depth: float):
    """固定米制标尺：近白远黑、无效紫色；不逐帧拉伸，也不修改原始数组。"""
    import numpy as np

    if not (math.isfinite(min_depth) and math.isfinite(max_depth) and 0 <= min_depth < max_depth):
        raise ValueError("深度显示范围必须为有限、非负且递增的米制范围")
    image = np.asarray(depth)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("单车深度图必须是非空 H×W 数组")
    valid = np.isfinite(image) & (image > 0)
    clipped = np.nan_to_num(image, nan=max_depth, posinf=max_depth, neginf=min_depth)
    gray = np.rint(255 * (1 - (np.clip(clipped, min_depth, max_depth) - min_depth) / (max_depth - min_depth)))
    rgba = np.empty((*image.shape, 4), dtype=np.uint8)
    rgba[..., :3] = gray[..., None].astype(np.uint8)
    rgba[..., 3] = 255
    rgba[~valid, :3] = (210, 60, 210)
    return rgba


def create_depth_overlay(env, task, *, enabled):
    """禁止在 headless/大批量回放导入 UI；无活动视口时明确提示且不阻塞策略。"""
    if not enabled or not 1 <= env.num_envs <= 4 or not env.sim.has_gui():
        return None
    role = vision_role(task)
    if role is None:
        return None
    from omni.kit.viewport.utility import get_active_viewport_window

    viewport = get_active_viewport_window()
    if viewport is None:
        LOGGER.warning("[DepthOverlay] 没有活动视口，跳过深度面板。")
        return None
    overlay = DepthOverlay(viewport, env.scene["depth_camera"], env.num_envs, role=role)
    print(f"[INFO] Depth overlay: {env.num_envs} panels; {role}; live raw depth (not policy history).")
    return overlay


class DepthOverlay:
    """复用主视口的四角，不新开窗口，不注册额外物理/观测更新回调。"""

    def __init__(self, viewport, camera, num_envs: int, *, role: str):
        if not 1 <= num_envs <= 4 or role not in ("teacher", "student"):
            raise ValueError("深度面板需要 1–4 台车及明确的 teacher/student 角色")
        import omni.ui as ui

        self._camera = camera
        self._num_envs = num_envs
        self._min_depth, self._max_depth = camera.cfg.spawn.clipping_range
        if not (0 <= self._min_depth < self._max_depth < math.inf):
            raise ValueError("深度面板需要有限的相机 clipping_range")
        self._role = role
        self._providers: list[ui.ByteImageProvider] = []
        self._frame_labels: list[ui.Label] = []
        self._closed = False
        self._frame = viewport.get_frame(FRAME_NAME)
        self._frame.clear()
        self._frame.visible = True
        self._frame.opaque_for_mouse_events = False
        with self._frame:
            with ui.VStack(spacing=0):
                # 给视口的相机菜单留位置；百分比面板随主视口缩放，中间保持可见。
                ui.Spacer(height=32)
                self._build_row(ui, (0, 1))
                ui.Spacer()
                if num_envs > 2:
                    self._build_row(ui, (2, 3))
                ui.Spacer(height=8)

    def _build_row(self, ui, env_ids):
        with ui.HStack(height=ui.Percent(34)):
            ui.Spacer(width=10)
            self._build_panel(ui, env_ids[0])
            ui.Spacer()
            if env_ids[1] < self._num_envs:
                self._build_panel(ui, env_ids[1])
            ui.Spacer(width=10)

    def _build_panel(self, ui, env_id):
        provider = ui.ByteImageProvider()
        self._providers.append(provider)
        # Kit 默认字体未必覆盖中文，界面短标签用英文，中文含义写入使用说明。
        role_text = "Teacher: display only" if self._role == "teacher" else "Student: live sensor"
        with ui.ZStack(width=ui.Percent(29)):
            ui.Rectangle(style={"background_color": 0xDA181818, "border_radius": 4})
            with ui.VStack(spacing=2, style={"margin": 5}):
                ui.Label(f"Rover {env_id + 1} | env_{env_id}", height=18, style={"font_size": 14})
                ui.Label(role_text, height=16, style={"font_size": 12})
                ui.ImageWithProvider(
                    provider,
                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    opaque_for_mouse_events=False,
                )
                self._frame_labels.append(ui.Label("Waiting for depth...", height=16, style={"font_size": 11}))
                ui.Label(
                    f"White {self._min_depth:g}m / Black {self._max_depth:g}m / Invalid purple",
                    height=28,
                    word_wrap=True,
                    style={"font_size": 11},
                )

    def update(self):
        """在 env.step 后读取；绝不调用 observation_manager.compute 推进 student 历史。"""
        if self._closed:
            return
        depth = self._camera.data.output[DATA_TYPE]
        if depth.ndim != 4 or depth.shape[0] != self._num_envs or depth.shape[-1] != 1:
            raise ValueError("相机深度必须为 N×H×W×1，且 N 等于回放车辆数")
        images = depth.detach().cpu().numpy()
        frames = self._camera.frame.detach().cpu().tolist()
        for env_id, provider in enumerate(self._providers):
            rgba = depth_to_rgba(images[env_id, ..., 0], self._min_depth, self._max_depth)
            height, width = rgba.shape[:2]
            provider.set_bytes_data(rgba.ravel().data, [width, height])
            self._frame_labels[env_id].text = f"RAW {width}x{height} | frame {int(frames[env_id])}"

    def close(self):
        """先释放面板对相机和图像的引用，再由 play 关闭环境；允许重复关闭。"""
        if self._closed:
            return
        self._closed = True
        self._frame.clear()
        self._frame.visible = False
        self._providers.clear()
        self._frame_labels.clear()
        self._camera = None
