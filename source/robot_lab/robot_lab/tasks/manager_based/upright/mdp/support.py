"""以接触传感器自身名称顺序定义真实双后轮支撑。

@author pony
@date 2026-09-16
@version v0.1.0
@last_modified 2026-09-16
@changelog
    - v0.1.0 (2026-09-16): 排除底盘/驱动杆支撑及前轮落地，避免借地平衡获奖。
"""

import torch


def support_masks(forces: torch.Tensor, body_names: list[str], threshold: float = 1.0):
    """返回无非轮支撑、双后轮支撑；禁止用 articulation 的 BFS 索引查询传感器。"""
    wheels = ("fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel")
    if forces.ndim != 3 or forces.shape[1:] != (len(body_names), 3):
        raise ValueError("接触力必须是与 body_names 一一对应的 [N,B,3]")
    if len(set(body_names)) != len(body_names) or not all(name in body_names for name in wheels):
        raise ValueError("接触传感器缺少四轮或包含重复名称")
    contacts = forces.norm(dim=-1) > threshold
    nonwheel = [i for i, name in enumerate(body_names) if name not in wheels]
    clear = ~contacts[:, nonwheel].any(-1)
    rear = contacts[:, [body_names.index("rl_wheel"), body_names.index("rr_wheel")]].all(-1)
    front = contacts[:, [body_names.index("fl_wheel"), body_names.index("fr_wheel")]].any(-1)
    finite = torch.isfinite(forces).all(dim=(-2, -1))
    return clear & finite, clear & rear & ~front & finite


def current_support(env):
    sensor = env.scene.sensors["contact_forces"]
    return support_masks(sensor.data.net_forces_w, sensor.body_names)
