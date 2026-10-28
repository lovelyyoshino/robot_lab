"""
supermimic — 世界模型 + BPTT 的 SuperMimic 训练核心，以及深度→高度扫描 latent 学生。

本包只依赖 torch，不导入 Isaac Lab；仿真适配放在 `isaac_adapter.py`，由脚本在
AppLauncher 之后再导入，保证单测可以脱离仿真器运行。与现有 PPO / DAgger 流程
完全独立，不修改任何既有任务或 checkpoint 合同。

@author pony
@date 2026-10-26
@version v0.1.0
@last_modified 2026-10-26
@changelog
    - v0.1.0 (2026-10-26): 初始创建
"""
