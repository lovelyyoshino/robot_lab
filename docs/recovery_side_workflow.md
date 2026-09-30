# 四轮车侧躺恢复：动作先验 + residual PPO

更新：2026-09-30。本页说明当前实现、实测证据和使用边界。
从零采集、训练、play、续训和最终验收的完整绝对路径命令统一维护在
[README 的侧躺恢复章节](../README.md#平行四边形轮式车从零训练侧躺恢复实验性)，
不再使用已删除的旧模型、旧示范目录或历史成绩作为本轮依据。

## 1. 当前主路线

目标是 **侧躺 → 四轮真实承重 → 稳定保持**。
当前采用 program31 的六段动作先验与有界残差 PPO，不是 AMP，也不需要先训练 BC。

| 阶段 | 入口及产物 | 用途 |
|---|---|---|
| 创建采集源环境 | `train.py`，Side 任务，20 轮 | 生成合法 checkpoint、配置及物理控制合同，不作为最终恢复策略 |
| 采集动作先验 | `evaluate_parallelogram_skills.py --diagnostic_action_program` | 执行 program31，保存完整成功回合及动作表 |
| 从零训练策略 | `train.py`，Side-Residual 任务 | 100 轮短训后，按开发结果决定续训 |
| 回放与开发评估 | `play.py`、`evaluate_parallelogram_skills.py` | 用同一模型、同一先验与 profile 核对实际恢复 |
| 最终验收 | 冻结 checkpoint 后使用未参与调参的 seed | 不把训练 reward 或开发集成功率当作泛化保证 |

本轮产物根目录为 `/media/bigdisk/robot_lab/logs/recovery_side_restart_v1`：
`source/` 保存采集源模型，`demos/` 保存动作数据，`policy/` 保存策略，
`evaluation/` 保存评测报告。日志、NPZ 和模型不随代码提交到 Git。

## 2. 本轮已验证的结果

2026-09-30 重新采集的 `demos/train.json` 显示：seed 1942 下，
program31 完成 64 回合，成功 48 回合（75%），NPZ 保存 48 个完整成功回合。
这是脚本诊断，报告中的 `passed=false` 不代表策略评测失败，也不能作为 PPO 的成功率。

随后对下列同一个策略 checkpoint 做了两组开发评估：

```text
/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy/2026-09-30_10-37-40_constant/model_1800.pt
SHA256: 6b5f3c1894ade3c553ad4711a0efaba3ba4dc299b715bb35d1484b6206a619a7
```

任务为 `RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0`，
使用 `constant` profile、同一 `demos/train.npz` 和 program31，
协议为 `settled_ground`、场景为 `side_lying`。

| 开发 seed | 完成回合 | 成功回合 | 成功率 | 平均进入最终稳定段时间 | passed |
|---:|---:|---:|---:|---:|:---:|
| 3942 | 64 | 64 | 100% | 3.3734 s | true |
| 4051 | 64 | 64 | 100% | 3.3788 s | true |

报告位于本轮根目录下：

- `evaluation/audit_model1800_seed3942_20260930_1245.json`
- `evaluation/audit_model1800_seed4051_20260930_1248.json`

两份报告的 `complete=true`，且 checkpoint 哈希一致。
这些 seed 属于开发集；本轮尚未完成冻结模型的独立最终验收。
有限仿真样本中的 100% 不等于任意姿态或真机成功率 100%。
训练继续生成的新 checkpoint 也不能继承 model_1800 的成绩，必须重新评测。

## 3. 输入、动作与保持期

| 项目 | 维度 | 含义 |
|---|---:|---|
| Actor 观测 | 60 | 原 58 维状态/历史，再追加接管阶段与锁定的初始倒向 |
| Critic 观测 | 72 | 原 70 维特权状态，再追加相同两项 |
| Actor 输出 | 12 | 8 个关节位置通道、4 个轮速通道的归一化残差，不是最终电机目标 |

准备期阶段和倒向观测为零；接管后按各环境独立时钟推进，
阶段为 `clamp(control_steps × 0.02 / 3, 0, 1)`。
初始倒向在接管时锁定，用于左右镜像，不能用翻滚中的当前倒向反复替换。

前 3 秒按六个 0.5 秒分段读取先验，
再叠加 `tanh(actor_output) × residual_scale`。
`constant` 的 raw 残差上限为腿关节 0.20、轮速 0.25；
最终仍经过原环境的动作映射和机械限位。
物理动作 scale 为摆臂 0.6109 rad、转向 1.57 rad、轮速 10 rad/s。

**constant 在 3 秒后将先验与残差都置零，余下保持期由默认姿态 PD 控制。**
因此当前结果不能描述为“神经网络在整个保持期持续闭环纠偏”。
`settle_feedback` 保留 3 秒后的小幅残差通路；它是独立实验选项，
本轮两组成绩没有验证它比 constant 更好，不应直接切换默认值。

## 4. 使用和验收约束

- 原 58 维 Side/Grounded/BC 权重不能直接加载到 60 维 Side-Residual。
- 必须一起保留 checkpoint、同级 `params/` 和动作先验 NPZ；只有一个 `.pt` 不够。
- `recovery_residual_runtime.json` 记录先验 SHA256、program_id、profile 和动作语义。
- 同一 profile 续训使用 `--resume --recovery_residual_profile auto`；
  切换 profile 必须用显式 `--recovery_residual_warm_start`，另建实验并重置优化器/迭代。
- `recovery_pre_update.pt` 是零更新快照；`model_0.pt` 已经过一次 PPO 更新。
- `--max_iterations` 是本次追加预算，不是成功率保证。继续训练前后均应比较实际评测。
- README 的自动选择命令要求评测期间不新增运行或模型；
  若训练仍在后台，必须像本次评估一样显式固定已有 checkpoint，并比较报告哈希。
- 每个验收 seed 都需完整回合、成功率至少 85%，四轮承重、无非轮承重，
  并满足姿态、高度与速度条件。合格稳态须连续延伸到 5 秒终点、
  持续至少 0.5 秒，且稳态开始不晚于 4 秒。
- 报告完整但 `passed=false` 表示技能未达标；traceback、缺失回合或非有限数值才是运行问题。
  `time_out=1` 和 reward 上升均不等价于成功率 100%。

## 5. 可选研究分支，不是必跑步骤

| 分支 | 已有实现 | 注意事项 |
|---|---|---|
| BC | `pretrain_recovery_demonstrations.py` 的阶段扩维、换段加权、独占输出目录 | 离线 loss 不能代替闭环恢复评测；先验证文件和成功示范，再选择新的输出目录 |
| 受保护 PPO | `recovery_protected.py` 的 critic 预热与示范锚点 | 仅适用明确的 Side-Phase BC 初始化流程，不要与 residual 主路线混用 |
| 其他残差 profile | `launch_protected`、`upright_release`、`settle_feedback` | 使用新实验对照，不能冒用 constant 的评估结果 |
| 失败重放 | 失败状态记录与 FailureReplay-Residual | 失败状态不是 BC 专家动作；训练池不能当独立验收池，还需回测原 Side 分布 |
| 几何/电机审计 | `probe_recovery_geometry.py`、`recovery_physics.py` | 几何扫描不证明动态翻转可行；电机模板不是实测标定 |

BC 可以用于后续闭环教师数据研究，但当前主路线不需要重新拟合 program31 的开环动作。
完全倒置、四轮朝天的恢复应单独证明可达轨迹并训练/验收，
不能把 Side 的成功率直接套用到 Upside 任务。

## 6. 物理和部署边界

当前结果仅覆盖既有 Isaac Sim 物理配置下的侧躺恢复。
真实电机模板位于
`source/robot_lab/data/Robots/custom/parallelogram_rover/calibration/actuator_profile.template.json`，
保留 `unverified/null` 是为了避免把仿真假设当作硬件测量。

URDF 的摆臂、转向节和下连杆存在碰撞覆盖缺口，完整碰撞模型和实测执行器仍需审查。
改变碰撞体、电机参数或控制周期后，应重新检查控制合同、生成数据并评测，
不能沿用当前成功率。不要通过超规格力矩或不存在的支点获得虚假恢复能力。

现有 rl_sar/MuJoCo 旧接口不能仅替换网络文件就使用此模型：
还需实现相同的 60 维观测、独立接管时钟、倒向锁定、分段先验、tanh 残差合成、
动作尺度和 reset 行为，再做 sim-to-sim 与真机安全验证。
play 导出的 12 维 Actor 输出不是最终电机目标；本次不宣称已经完成部署。

## 7. 离线回归

在已配置好的 Isaac Lab Python 环境中运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest /media/bigdisk/robot_lab/source/robot_lab/test -q -p no:cacheprovider
```

这里关闭的是无关第三方 pytest 插件的自动加载，不是跳过项目测试；
机器上的 ROS launch_testing 插件可能与 Isaac Lab Python 环境不兼容。
离线测试用于检查观测、先验、合同、动作与评测逻辑，不能替代真实仿真回合。
