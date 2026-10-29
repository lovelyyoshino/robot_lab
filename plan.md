# SuperMimic 迁移计划：复用 / 替换 / 新增清单

更新：本文基于当前 `main`（2d5bfc23a）的代码盘点；外部仓库只读过 README，未读源码。

参考实现：
- `zita-ch/supermimic_vibed`（Isaac Lab 3 + MJWarp，MIT）— 算法主参考
- `Zhengtianjin/go2-supermimic`（mjlab，MIT）— 无示范任务奖励 + 高度扫描地形
- `Motphys/Uni-SuperMimic`（UniLab，Apache-2.0）— 合同/评测组织方式参考

核心判断：SuperMimic 只替换"策略怎么训"（PPO → 世界模型 + BPTT），不替换任务定义、资产、
合同、评测和部署。Teacher-Student 结构保留，蒸馏点从动作级改为 latent 级。

---

## 1. 直接复用（不改或只加接口）

| 模块 | 路径 | 复用方式 |
|---|---|---|
| 车体资产 + 12 维动作合同 | `assets/parallelogram_rover.py` | 原样；`assert_exact_policy_action_names` 继续守住 fl,rl,fr,rr 顺序 |
| mimic 从动关节推导 | `parallelogram_rover/mdp.py` `reset_parallelogram_joints` | 原样；世界模型只看 12 个独立关节 |
| 碰撞几何复位高度 | `upright/mdp/geometry.py` | 原样；replay buffer 初态与 reset 共用 |
| 深度时序缓冲 | `vision_temporal.py` `DepthTemporalBuffer` | 原样，8 帧 stride 5、延迟/丢帧扰动保留 |
| 深度单帧预处理 | `vision_mdp.py` `normalized_depth_image` | 原样 |
| 深度/视觉合同 | `rsl_rl/vision_contract.py`、`depth_contract.json` | 原样；新增字段 `latent_contract`（z 维度、编码器版本） |
| 地形课程 8→12→16 cm | `vision_preview.py` `preview_route_curriculum` | 原样，作为 SuperMimic 采样环境的课程 |
| 失败状态池 | `recovery/mdp/failure_states.py` | 原样；可作为 replay buffer 的额外初态来源 |
| 实验谱系 | `rsl_rl/experiment_lineage.py` | 原样；世界模型 checkpoint 也写谱系 |
| 技能/运动/视觉验收 | `skill_evaluation.py`、`evaluate_parallelogram_{motion,skills,vision}.py` | 原样；只需支持加载新 actor 格式 |
| 运动诊断记录 | `rsl_rl/motion_trace.py` | 原样；兼作 mimic 动作库的录制器（见 §3） |
| 三策略切换器 | `hybrid/planner.py`、`hybrid/runtime.py` | 阶段一原样；阶段二改为"技能选择器→参考片段"（见 §2） |
| 深度回放叠加 | `vision_depth_overlay.py` | 原样 |
| 现有 PPO checkpoint | `logs/` 下 loco / upright / recover / teacher-preview | 当数据采集 driver 和 mimic 参考来源，不再作为最终部署策略 |
| 测试环境 | `env_isaaclab` + `setup_conda_env.sh` + `unset PYTHONPATH` | 原样 |

## 2. 需要替换

| 现有 | 替换为 | 原因 |
|---|---|---|
| `rsl_rl_ppo_cfg.py` 的 PPO runner（各任务） | 新 `supermimic` runner：replay buffer + 世界模型 + BPTT actor + GAE critic | 核心算法替换；PPO 配置保留作基线对照 |
| Student 平铺 MLP（`rsl_rl_vision_cfg.py`，4141→1024/512/256） | CNN+GRU 深度编码器 → z_depth，接冻结 actor | 已知欠拟合根因之一 |
| `agents/dagger_distillation.py` 动作级蒸馏 | latent 对齐（z_depth≈z_terrain）+ 高度扫描重建辅助损失；DAgger 仅保留为末段微调 | 只学"看"，不再从头学"控制" |
| 依赖接触力的奖励/观测（`upright/mdp/support.py`、`upright/mdp/observations.py`、`recovery/mdp/grounded.py`、`settled_ground.py`、`vision_preview.py` 中的接触项） | 世界模型侧的解析版：姿态、高度、轮心高度（可微 FK） | 世界模型不预测接触力；且接触力本身不报在 `*_wheel` 上 |
| 硬终止（姿态/高度阈值） | 软存活权重（sigmoid 距阈值） | BPTT 需要可导；仿真侧终止可保留用于 reset |
| `recovery_prior.py` 分段动作先验 + residual | 起身/恢复片段进入 mimic 动作库，由跟踪器学 | 先验变成参考动作；residual 包装器退役 |
| `hybrid/planner.py` 三策略切换 | 单一跟踪器 + 技能选择器（输出参考片段 ID 与起始帧） | 阶段二再做，阶段一不动 |
| `beyondmimic/` G1 跟踪任务 | 仅保留作参考，不用于车 | 车的跟踪 command 另写（见 §3） |

## 3. 需要新增

| 新模块 | 建议位置 | 内容 | 过门标准 |
|---|---|---|---|
| 可微 FK | `assets/parallelogram_rover_kinematics.py` | arm 角 →（mimic 关系）→ 4 轮心位姿/速度，torch 实现 | 对仿真 link 位姿误差 < 1e-5 |
| 世界模型特征 | `rl/supermimic/features.py` | 航向不变根状态、12 关节（轮子只用速度）、上一动作、DR 上下文 ξ（摩擦、质心偏移、关节偏置） | — |
| 世界模型 | `rl/supermimic/world_model.py` | MLP 预测帧局部增量，k=8 自回归窗口 | 平地 32 步开环误差可接受，侧滑/轮速预测准 |
| 解析奖励/终止头 | `rl/supermimic/rover_heads.py` | 把 §2 里替换后的奖励写成状态函数 | 与仿真侧奖励逐项对齐 |
| BPTT learner | `rl/supermimic/learner.py` | H=32 想象 rollout、critic 自举、梯度裁剪 | go2w 平地速度先收敛 |
| 高度扫描查询 | `rl/supermimic/height_query.py` | 按预测位姿在地形高度图上双线性插值（可导） | 与仿真 RayCaster 187 点一致 |
| mimic 动作库 | `tasks/.../rover_mimic/motion_library.py` | 由 `motion_trace.py` 录专家轨迹：根位姿、arm/steer 角、轮速，50 Hz，1 s 分箱 | 每段可从第 0 帧回放 |
| 跟踪 command/奖励 | `tasks/.../rover_mimic/mdp.py` | 未来 K=10 帧目标、根位姿/倾角/arm/轮心跟踪奖励、按回报重置采样 | 全片段 0 失败 |
| latent 对齐训练 | `rsl_rl/latent_alignment.py` | z_depth↔z_terrain、高度扫描重建、可选 DAgger 微调 | `blank` 明显掉分，`live` 接近 Teacher |
| 训练入口 | `rsl_rl/train_supermimic.py` | 复用 `cli_args.py` 与谱系记录 | — |

## 4. 执行顺序

| 步 | 内容 | 依赖 | 主要改动 |
|---|---|---|---|
| ⑦ | latent 对齐（用现有 PPO Teacher-Preview） | 无 | 替换 Student/DAgger |
| ① | 跑通 go2-supermimic | 无 | 外部仓库，不改本库 |
| ② | 移植 learner 到 robot_lab，先在 go2w 验证 | ① | 新增 `rl/supermimic/` |
| ③ | 车世界模型离线验证 + 可微 FK | ② | 新增 FK、特征、世界模型 |
| ④ | 车平地速度 / upright 用任务奖励跑 SuperMimic | ③ | 替换接触力奖励、软终止 |
| ⑤ | mimic 动作库 + 跟踪器，吸收 recovery_prior | ④ | 新增 rover_mimic，退役 residual |
| ⑥ | 地形 + 高度扫描进想象，楼梯课程 | ⑤ | 新增 height_query；⑦ 的编码器接上 |

③ 是否决点：世界模型对轮地滑移预测不准，就停在 ④ 之前，车上保持 PPO，只保留 ⑦ 的成果。

## 4.1 已实现（分支 `feat/supermimic`，与现有任务完全独立）

| 文件 | 内容 |
|---|---|
| `source/robot_lab/robot_lab/supermimic/` | 纯 torch 核心：`quat` `state` `rover_task` `networks` `replay` `learner` `scan_student`；`isaac_adapter` 仅脚本使用 |
| `tasks/manager_based/supermimic/` | 任务 `RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0`（平地，去推车/外力/观测噪声，加 1 rad 倾角终止） |
| `scripts/supermimic/train_rover.py` | ②③④：世界模型 + BPTT 训练 |
| `scripts/supermimic/evaluate_rover.py` | 速度跟踪 + 世界模型开环误差（k=1/8/32） |
| `scripts/supermimic/play_rover.py` | SuperMimic 回放 + 导出 45→12 policy.pt/onnx/合同 |
| `rsl_rl/play.py --scan_student` | 深度学生回放（深度角落面板、Preview 固定场景照常可用） |
| `scripts/supermimic/train_scan_student.py` | ⑦：深度→187 点扫描→冻结 Teacher-Preview actor |
| `scripts/supermimic/verify_rover.py` / `verify_scan.py` | 训练前校验：观测/奖励合同、轮半径、世界模型探针、Teacher 重建 |
| `evaluate_parallelogram_vision.py --scan_student` | ⑦ 学生的楼梯验收（live/blank/frozen），按 run 的 depth_contract.json 校验 |
| `source/robot_lab/test/test_supermimic_core.py` | 24 项单测 |

## 4.2 训练前校验结果（2026-10-26，第二轮）

| 项 | 结果 | 结论 |
|---|---|---|
| 解析观测 vs 仿真 45 维 | 最大偏差 1.2e-7 | ✅ |
| 12 个奖励项 vs RewardManager（含 3 个 joint_acc） | 最大偏差 4.8e-7 | ✅ |
| joint_acc 根因 | env 每 5 ms 子步刷新加速度，控制步差分只有真值 0.09–0.23 倍；改由世界模型 aux 头回归 env 真值 | ✅ 已解决：留出环境均值比 1.03–1.06 |
| 动力学上下文 ξ（25 维：Kp/Kd 比、质量、质心） | 每回合 Kp/Kd 在 0.5–2 倍间重抽，跨环境标准差 ≈0.43；接入世界模型与 critic，actor 不看 | ✅ 世界模型 k8 训练误差 1.24→0.60 |
| 有效轮半径 | 0.114 m，1 m/s 需 8.8 rad/s；actor 界 2π（轮速目标 ±31.4） | ✅ |
| 扫描学生：本体 45 列 / Teacher 重建 | 偏差均为 0 | ✅ |
| 扫描学生冒烟（40 轮）楼梯 8 cm | live 25%，blank 0% | ✅ 链路通 |
| 全量规模 | 4096 env：约 0.8 s/轮，显存 16 GB；1000 轮 ≈ 13 min | ✅ |
| 训练动态（1000 轮，同参数 A/B） | 旧：100–400 轮想象奖励达真实 4 倍、lin_err 回升到 0.7；新：真实奖励全程 ≥ 想象，wm_reward_bias ≈ +0.0005 | ✅ 钻空子已解决 |
| 训练末期（第 980 轮，训练分布） | lin 0.149 / ang 0.111 / 每步奖励 0.121；旧配置 0.209 / 0.201 / 0.109 | ✅ |
| 确定性验收（768 env × 1200 步，与 PPO 同协议） | SuperMimic 1000 轮：lin 0.263 / ang 0.181；PPO 20000 轮：lin 0.219 / ang 0.204 | 🟡 ang 已优于 PPO，lin 仍差 0.04 |

已知局限：
- aux 头逐样本相关只有 0.3–0.4：子步加速度主要是接触冲击噪声，可预测的只有条件均值；期望惩罚只需要均值，可接受。
- 在 actor 自己的平滑轨迹上，短时线速度/轮速预测略逊于"速度不变"基线（k1 线速度 0.020 vs 0.013 m/s），说明有约 0.02 m/s 的噪声底；倾角、位置仍明显更好。
- aux 损失在 actor 刚开始更新（100–600 轮）时升到 0.5–1.5，第 80 轮出现过一次奖励偏差 -5.8 的离群值；600 轮后回落到 0.07，未影响训练。
- 旧 checkpoint（无 ξ/aux）与新代码不兼容，不能续训。

结论：⑦ 和 SuperMimic 平地都可以开始全量训练。

## 4.3 地形开发实验（第 1、2 步，2026-10-26，`scripts/supermimic/verify_terrain.py`）

数据：Teacher-Preview + 噪声 0/0.1/0.3，1024 env × 1000 步，8/12/16 cm 楼梯与单台阶；48% 样本在台阶边，35% 时间车下地面比起点高 6 cm 以上。过门标准运行前写死在脚本 `GATES` 里。

| 项 | 结果 | 门 |
|---|---|---|
| 高度图扫描 vs RayCaster（非台阶边，3650 万条） | 99.89% 在 5 mm 内 | ✅ |
| 同上（台阶边 ±1 格，180 万条） | 65% 在 1 mm 内，p99 28 cm（插值把台阶变成 2 cm 斜坡） | 只报告 |
| 可微 FK vs 仿真轮心 | 中位 7e-6 m，p99 4.8e-5 m，最大 1.4 mm | ❌（门 1 mm）；最大值与 mimic 约束瞬时违反（最大 5.6e-3 rad × 臂长）同时出现，99.9994% 在 1 mm 内 |
| 世界模型 k8 台阶边轮心高度：有地形 / 无地形 | 1.9 / 2.4 mm = 0.79 | ❌（门 0.7） |
| 同上 k32 | 3.1 / 6.8 mm = 0.46 | 未设门，明显受益 |
| k8 台阶边轮心高度 / 常速度基线 | 1.9 / 7.2 mm = 0.26 | ✅（门 0.5） |
| k8 台阶边俯仰 / 常速度基线 | 0.0045 / 0.024 rad = 0.19 | ✅（门 0.5） |
| 平地不退化（k8 轮心高度，有/无地形） | 0.6 / 0.7 mm | ✅ |

结论：地形世界模型可用，第 ③ 步在楼梯上过门；地形输入的收益主要在长时域（k32 误差减半），k8 未达预设 0.7。FK 门差 0.4 mm，来自仿真 mimic 软约束而非 FK 公式。注意：数据里 16 cm 片段多为 Teacher 失败/卡住，世界模型见过"撞台阶"，没见过"爬上 16 cm"。

## 4.4 第 3 步中规模结果（2026-10-26，`train_terrain.py --num_envs 1024 --iterations 1000`）：❌ 退化

同协议楼梯验收（32 env × 2 回合，seed 142）：

| 策略 | 8 cm 通过 | 12 cm 通过 | 失败形态 |
|---|---|---|---|
| Teacher-Preview（PPO） | 100% | 70% | — |
| 仅热启动未训练（核对评测链路） | 100% | — | 与 Teacher 完全一致 |
| SuperMimic 第 500 轮（actor 只更新 100 次，参数最大变化 0.005） | 9% | — | 卡在第一级台阶前（最远 1.49 m） |
| SuperMimic 第 1000 轮 | 11% | 23% | 8 cm 能越过但未判稳定、继续前冲后失败；12 cm 卡在第一级 |

诊断（同一批回放起点上比较想象回报）：训练后 actor 的 λ-return 7.63 > Teacher 7.43，但想象前进距离反而更小（0.18 vs 0.21 m）。
即想象目标把更差的策略评成更好：
1. 32 步（0.64 s）只够前进约 0.2 m，"卡住"和"爬上去"在视界内几乎同分，回报约 60% 来自 critic 自举，而 critic 只在想象数据上拟合；
2. 几何抬轮塑形量级只有跟踪项的约 1%（0.035 vs 4.6），实际不起作用；
3. 越台阶的梯度要穿过接触，actor 梯度范数长期 2–8、每次被裁剪。
结论：按当前配置，SuperMimic 不能在楼梯上改进 Teacher；楼梯能力以 Teacher 为准。

## 4.5 双线结果（2026-10-26）

**A：SuperMimic 越障 v2**（critic 拟合真实回报 + Teacher 信赖域 anchor_coef=1 + 抬轮塑形 ×10 + 初始地形覆盖全课程；1024 env × 1000 轮）

| 楼梯 | Teacher（PPO） | v1 | **v2** |
|---|---|---|---|
| 8 cm | 100% | 11% | 34% |
| 12 cm | 70% | 23% | **94%** |
| 16 cm | 0% | — | **30%** |

v2 是第一个在 12/16 cm 上超过 Teacher 的版本，但 8 cm 退化。失败诊断（`climb_search.py --teacher_only`）：8 cm 能登顶，越线时航向偏约 13°，横向在 0.15 m 内的只有 34%，于是冲过终点后失败；16 cm 95% 能登顶，但过程中最大倾角中位 62°（评测门限 31.8°）。

**B：16 cm 程序搜索**（特权攀爬程序 + CEM，512 env × 14 代，目标与评测同口径）：❌ 收敛到"只爬一级"（抬升 0.16 m、倾角 29°、不倒不偏）。说明这类简单程序在 31.8° 约束内爬不完 4 级；Teacher 能登顶靠的是大幅俯仰（65°）。

**共同瓶颈**：三档楼梯失败都与"航向/横向漂移"和"倾角超过 31.8°"有关，与能不能爬上去关系不大。下一步（v3）：在 PreviewTask 里加航向保持、横向速度惩罚、超过 25° 的倾角软惩罚；同时需要用户决定 16 cm 楼梯是否沿用 31.8° 门限。

**v3**（v2 + 航向 -10·sin²ψ + 横向速度 -20·vy² + 倾角软惩罚 -6·relu(cos25°−cosθ)；想象与真实回报同口径；`--run_name probe_v3`）

| 楼梯 | Teacher | v2 | **v3** |
|---|---|---|---|
| 8 cm | 100% | 34% | **86%** |
| 12 cm | 70% | 94% | **98%** |
| 16 cm | 0% | 30% | 0% |

诊断（`climb_search.py --teacher_only`，128 env）：

| | 8 cm 越线横向 ≤0.15 m | 8 cm 航向中位 | 16 cm 登顶 | 16 cm 最大倾角中位 | 16 cm 越线横向中位 |
|---|---|---|---|---|---|
| v2 | 34% | 12.9° | 95% | 62° | 0.31 m |
| v3 | **99%** | 9.7° | 75% | 63° | 0.46 m |

评测里任一时刻倾角 >31.8° 即 `traversal_failure` 终止（`evaluate_parallelogram_vision.py` 第 235/450 行）。（后经复核，16 cm 爬升途中实际倾角约 20°，见下方「v3b 与倾角门复核」。）
航向/横向项对 8 cm 有效（越线横向合格 34%→99%）。倾角软惩罚没有把 16 cm 倾角压下来（中位 63°），反而让登顶率 95%→75%、16 cm 通过率 30%→0%：
策略在 −6 的权重下没学会"低倾角爬法"，只是爬得更犹豫。
下一步 v3b：去掉倾角软惩罚（`--tilt_soft 0`），确认航向/横向项单独能否同时保住 8 cm 与 16 cm；16 cm 倾角需要专门的调平手段（或放宽门限，由用户决定）。

**16 cm 倾角门放宽（用户 2026-10-27 决定）**：`parallelogram_rover/stairs_acceptance.py` 的 `CLIMB_TILT_DEG = (31.8, 31.8, 45.0)`，只放宽 16 cm 途中跌倒判据；终点稳定仍要求 31.8°。
依据：轴距 0.694 m 跨 1–2 级 16 cm 台阶时静态俯仰 13–28°，叠加起抬瞬态后 31.8° 过严；45° 仍远低于训练翻车终止 68.75°。
评测（`capture_traversal`）与诊断（`climb_search.py`）取同一值；训练本身没有途中倾角门，不受影响。
不放进 `vision_preview.py`：`vision_contract` 对该文件做 sha256，改动它会让所有已训练 Preview checkpoint 合同失配（第一次放进去时 v3b 验收即因此全部报错）。
放宽前的 16 cm 结果（上表）不可与放宽后直接对比，Teacher/v2/v3/v3b 已按 45° 重测（`eval/*_d1.0_tilt45.json`）。
**v3b 与倾角门复核（2026-10-27）**

| 楼梯 | Teacher | v2 | v3 | v3b（无倾角软惩罚） |
|---|---|---|---|---|
| 8 cm | 100% | 34% | **86%** | 78% |
| 12 cm | 70% | 94% | **98%** | 94% |
| 16 cm（31.8°） | 0% | **30%** | 0% | 0% |
| 16 cm（45°） | 0% | **30%** | 0% | 0% |

**更正：倾角从来不是 16 cm 的瓶颈。** 31.8° 与 45° 两种门下 64 回合结果逐项相同；把门收紧到 10° 时 v2 立刻 0%（卡在 1.0 m），说明门本身生效。
之前诊断的"最大倾角中位 62°"是 `climb_search.py` 的统计错误：它把冲出 3.75 m 后车从平台边缘掉落的时段也算进去了。
修正为只统计出界前时段后，v2 / v3b 在 16 cm 上最大倾角中位均约 20°，100% 在 31.8° 内。
16 cm 的真实失败形态：登顶（v2 95%、v3b 91%），但越线时横向偏移中位 0.31 / 0.46 m，0.5 s 内回不到 0.15 m 内，一路开到 3.75 m 出界。
v3 的倾角软惩罚因此是在优化一个不存在的问题；航向/横向速度项在 8 cm 有效，在 16 cm 反而比 v2 偏得更多（爬台阶时车身被台阶带偏，速度惩罚压不住已经累积的横向位置）。
45° 门保留（用户决定，且无副作用），但对现有策略没有影响。
下一步原计划：惩罚相对路线中心线的横向位置。复核后改为下面的根因修正（横向位置策略看不到，惩罚它学不会）。

**横向漂移根因：蟹行（2026-10-27，`climb_search.py` 路线探针，128 env）**

沿路线 x 处的带符号均值（航向 ψ 逆时针为正；转向角 δ 为前/后轴平均）：

| 策略 / 楼梯 | x=0.6 m（平地） | x=1.2 m（首级） | x=2.0 m | x=2.8 m |
|---|---|---|---|---|
| Teacher 8 cm | y −0.00，ψ +6°，δf −5° / δr −20° | y −0.03，ψ +15°，δ −22° / −27° | y −0.04 | y +0.03 ✅ |
| Teacher 16 cm | y −0.00，ψ +8°，δ −9° / −33° | y −0.11，ψ +23°，δ −29° / −35° | y −0.39 | y −0.54 ❌ |
| v2 16 cm | y +0.01，ψ +9°，δ −6° / −29° | y −0.02，ψ +26°，δ −32° / −29° | y −0.26 | y −0.35 ❌ |

所有策略都是"车头左偏、四轮右转"的蟹行，平地上二者抵消（行进方向 ≈ 0°）。上 16 cm 台阶时车沿轮子滚动方向（ψ+δ ≈ −6 ~ −12°）被台阶推着走，每级向 −y 偏约 0.12 m，4 级累计 0.4–0.5 m。
策略观测里没有横向位置（扫描仅 ±0.5 m，车道宽 3 m；无 base_lin_vel），只能靠航向/转向自身纠正，所以 v3 的航向 + 横向速度惩罚压不住。
v4：加 `wheel_heading`（× 四轮平均 sin²(ψ+δ)），直接惩罚轮子世界系朝向偏离路线；−10（v4）与 −30（v4b）两档。


**v4 结果（wheel_heading −10 / −30）：❌ 全面退化**

| 楼梯 | v3 | v4（−10） | v4b（−30） | v4 失败形态 |
|---|---|---|---|---|
| 8 cm | 86% | 52% | 50% | 23 回合停在 x≈1.40 m（后轮卡首级） |
| 12 cm | 98% | 19% | 44% | v4 冲出终点 45；v4b 停在 1.38 m 36 |
| 16 cm | 0% | 0% | 0% | v4b 登顶率 91%→52% |

平地蟹行确实消失（x=0.6 m 处 ψ≈0°、δ≈−3°），但一到台阶边又恢复（v4 16 cm x=1.2 m：ψ +20°、δ −28°/−33°），侧漂几乎不变（越线 −0.51 m）。
结论：蟹行是策略斜着上台阶、让后轮逐个越过前缘的手段，侧漂是这个手段的代价；压掉它，后轮就卡在首级前。`wheel_heading` 默认改回 0。

**v5（进行中）**：保留斜着爬，改为对相对车道中心线的横向偏移 y² 直接扣分（−5 / −20），critic 额外输入该偏移。
actor 仍看不到 y，但侧漂是系统性的（各版本越线时 −0.37~−0.51 m，标准差 <0.1 m），可以从扫描里的台阶形态学成开环补偿（例如登顶后轻微向 +y 回正）。



## 4.6 深度分析：为什么楼梯难爬（2026-10-27，`climb_search.py --record` + `analyze_climb.py`，Teacher / v3 各 128 env）

基线回到 v3（`train_terrain.py` 默认值 = v3）。

**物理量级**：轮半径 0.1085 m，整车 106 N，单轮静载约 27 N，轮电机 10 N·m，阻尼 0.5（扭矩 ≈ 0.5×(目标−实际)）。
8 cm（=0.74 R）可以靠轮子硬推上去；16 cm（=1.47 R）超过轮半径，单靠滚动上不去，必须靠 arm 抬轮 + 其他轮推。
地面摩擦随机化 8–27，摩擦不是瓶颈，扭矩与时机才是。

**越级时序**（登上第 1 级的中位时刻，s）：

| | fl | fr | rl | rr | 失败形态 |
|---|---|---|---|---|---|
| Teacher 8 cm | 2.96 | 2.46 | 5.02 | 4.64 | 无 |
| v3 8 cm | 2.82 | 2.58 | 5.16 | 4.84 | 12% 后轮停在首级立面前 |
| Teacher 16 cm | 3.14 | 2.22 | 5.18 | 4.56 | 99% 登顶，侧漂出界 |
| v3 16 cm | 2.94 | 2.40 | 5.22 | 4.66 | 25% 后轮停在第 1/2 级立面前 |

1. **右侧总比左侧先上**（前轴差 0.5–0.9 s，后轴 0.3–0.6 s），这就是固定 −y 侧漂的来源：16 cm 每一级 −0.18 ~ −0.29 m，4 级累计约 −0.5 m。Teacher 和 v3 一样，是 Teacher 继承下来的习惯（训练里有大量左/右单侧窄障碍）。
2. **v3 的新失败是后轮推不上去**。后轮贴在首级立面前时：

| | 每回合贴立面时长 | 后轮扭矩中位 | 后 arm 角 |
|---|---|---|---|
| Teacher 8 cm | 0.13 s | 5.1 N·m | −23° |
| v3 8 cm | 0.95 s | 3.5 N·m | −14° |

   Teacher 后轮"顶上去就过"，v3 推得轻、arm 压得浅，就停在立面前（卡住时后轴两轮转向还一正一负，−12° / +14°，互相抵消推力）。
   原因判断：越过前缘需要穿过接触，世界模型给不出"再用力推一下就上去"的梯度，BPTT 在这里反而把动作从 Teacher 推开；抬轮几何塑形对贴立面的后轮净空为 0，也没有梯度。
3. **前面几版为什么都没用**：倾角从来不超门（v3 已证）；压蟹行（v4）让两个后轮同时正撞立面，扭矩需求翻倍，卡得更多；横向速度/航向惩罚管不住"右侧先上"带来的每级侧漂。

**对应的修正**：
- v6（进行中）：v3 + `riser_anchor 20`，后轮贴立面的状态对 Teacher 的锚定加重 21 倍，保住 Teacher 的"顶上去"动作，其余状态照常优化。目标是 8 cm 回到 ≥95%，保住 12 cm 的 98%。
- v5（进行中）：v3 + 车道偏移惩罚，测试策略能否对固定 −y 侧漂学会开环补偿；这是 16 cm 唯一不改观测的路线。
- 如果 v5 不行，16 cm 需要让策略看到横向偏差（改路线命令或观测），要重训 Teacher，需用户决定。

**v5 结果（车道偏移惩罚 −5 / −20）：❌**

| 楼梯 | v3 | v5（−5） | v5b（−20） |
|---|---|---|---|
| 8 cm | 86% | 94% | 11% |
| 12 cm | 98% | 23% | 3% |
| 16 cm | 0% | 0% | 0% |

侧漂完全没有被补偿：越线横向 v5 为 −0.10 / −0.26 / −0.54 m（8/12/16 cm），v5b 为 −0.18 / −0.28 / −0.47 m，和 v3 同量级。12 cm 失败全是"登顶 100% 但冲出终点"。
v5 的 8 cm 94% 是唯一亮点，但 12 cm 退化太多，整体不如 v3。
结论：策略看不到横向位置时，奖励里罚偏移学不出补偿，只会扰乱其他行为；不改观测的路线到此为止。16 cm 要么让策略看到横向偏差（改路线命令/观测并重训 Teacher），要么接受 16 cm 不达标。

**C → Student（2026-10-27）**：`export_terrain_teacher.py` 把越障 actor 写成 rsl_rl Teacher checkpoint（替换 actor.*，其余沿用热启动 Teacher）。
核验：转换后按普通 teacher-preview 评测 8/12 cm 为 86% / 98%，与原 v3 逐回合相同；Student 冒烟 2 轮，checkpoint 内 teacher 权重与 v3 差 0、与原 PPO Teacher 差 0.023；v3 在 16 cm 上 0.009% 动作超 ±2π。

**v7：❌** v3 + PPO 指导只留在难点（全局 anchor 500 轮内 1→0.2，后轮贴立面处 ×21）。8/12/16 cm 为 72% / 75% / 0%（v3 86% / 98% / 0%）。
8 cm 中途停住从 9 增到 17、12 cm 越线横向合格 34%→9%：PPO 约束一放松，世界模型梯度就把 Teacher 的爬台阶动作推坏，局部锚定也保不住。
结论：当前世界模型梯度只适合在 PPO 强约束下做小幅改进（v3 的 anchor_coef=1），不能主导楼梯技能；仍用 v3。

**平地上下跳（2026-10-28，`probe_flat.py`，纯平地、固定命令、前 3 s 不计）**

| 策略 | 0.3 m/s 时 arm 速度 rms | 车体起伏 | 俯仰角速度 rms |
|---|---|---|---|
| Teacher / probe_v3 / v6 / v7 | 0.13–0.14 rad/s | 0.03–0.1 cm | 2.3–2.7°/s |
| 用户 terrain_v3 第 1000 / 2000 / 3000 轮 | 0.13 / 0.13 / 0.14 | 0.05–0.07 cm | 2.1–2.9°/s |
| 用户 terrain_v3 第 4000 轮（play 默认取最新导出） | **1.21**（2.9 Hz） | **2.7 cm** | **24°/s** |

抖动只在 3000→4000 轮之间出现，且只在正常行驶速度下；停车和 0.15 m/s 不抖。同一策略楼梯为 2% / 92% / 39%（8 cm 冲出终点 62 回合）。
原因：Preview 奖励里 lin_vel_z −0.5、ang_vel_xy −0.05 太弱，楼梯验收也不看平地，长训后世界模型梯度把平地 arm 推成周期摆动也不受罚。
修正：`smooth_flat_arm −0.5`（平地 Σarm 速度²）、`smooth_flat_ang −5`（平地俯仰/滚转角速度²），只在四轮前方都没有台阶时生效；
`accept_terrain.py` 加平地平稳门（arm 速度 rms ≤0.4 rad/s、车体起伏 ≤1 cm）。v8 = v3 + 这两项，训 4000 轮验证。

**新 terrain_v3（含平地惩罚，4000 轮，2026-10-28）**：8/12/16 cm 3% / 0% / 0%，登顶 100/98/98%，越线横向 ≤0.15 m 只有 2/0/0%，失败几乎全是冲出终点。
同 run 的存档：第 1000 轮 67% / 81% / 0%，第 2000 轮 6% / 0% / 0%。平地三档都不再抖（arm 0.13–0.14 rad/s）。`--max_iterations` 默认改为 1000。

**侧漂分阶段拆解**（`analysis_drift/`，64 env；横向位移单位 m，越线 y 为越过终点线时的横向）：

| 楼梯 / 策略 | 平地接近 | 爬台阶期间 | 登顶后 | 越线 y |
|---|---|---|---|---|
| 8 cm Teacher | +0.01 | −0.04 | +0.04 | +0.01 |
| 8 cm 第 1000 轮 | −0.02 | −0.08 | +0.04 | −0.06 |
| 8 cm 第 4000 轮 | −0.03 | −0.15 | −0.05 | −0.26 |
| 16 cm Teacher | +0.02 | −0.54 | +0.03 | −0.50 |
| 16 cm 第 1000 轮 | −0.01 | −0.42 | +0.01 | −0.44 |
| 16 cm 第 4000 轮 | −0.02 | −0.51 | −0.02 | −0.59 |

1. 侧漂几乎全发生在"爬台阶期间"（首个前轮上第 1 级到后轮全部登顶）；平地和登顶后都很小。
2. 爬台阶时所有策略都是车头左偏（ψ +5~+19°）、四轮右转（δ −8~−34°），实际行进方向偏右 7~30°；右前轮比左前轮早 0.2–0.9 s 上第 1 级。车是沿轮子方向被台阶推着走的。
3. 16 cm 连 Teacher 也偏 0.5 m：这是 Teacher 带来的爬法，不是 SuperMimic 训坏的。
4. 8 cm 上 Teacher 爬时只偏 0.04 m、登顶后还能拉回 0.04 m；长训后爬时偏 0.15 m、登顶后继续往外偏。越线 0.15 m 横向门不在想象目标里（只有航向、横向速度项），长训偏离 Teacher 后就丢了"拉回来"。
5. 根本限制：策略观测里没有横向位置（扫描 ±0.5 m、车道 3 m，无 base_lin_vel），偏了自己不知道。

**方案 B（2026-10-28，用户选定，进行中）**：Teacher 不动，SuperMimic actor 追加第 233 维车道横向偏移观测（`supermimic/lane_offset.py`，(y−中心线)×4，±4 截断），
新列权重 0 初始化（训练起点与 Teacher 逐位相同），想象里 `--lateral_pos` 罚偏移、critic 也看到偏移。合同仍是 Teacher 的 232 维，第 233 维由 `params/lane_offset_obs.json` 描述，评测/回放/诊断自动识别。
冒烟：想象观测与 env 观测最大误差 2e-6；评测、回放、平地探针、诊断四条链路都能加载 233 维策略。
实验：`laneB_p5`（lateral_pos −5）、`laneB_p20`（−20），各 1000 轮。结果 ❌：

| | 8 cm | 12 cm | 16 cm | 16 cm 越线 y |
|---|---|---|---|---|
| Teacher | 100% | 70% | 0% | −0.50 |
| v3（无车道观测） | 86% | 98% | 0% | — |
| laneB 第 500 轮（两档相同） | 100% | 70% | 0% | — |
| laneB_p5 第 1000 轮 | 81% | 42% | 0% | −0.57 |
| laneB_p20 第 1000 轮 | 34% | 19% | 0% | −0.38 |

第 500 轮与 Teacher 逐项相同：前 500 轮是世界模型/critic 预热，actor 未更新，验证了 0 初始化起点正确。
500 次 actor 更新后，新列权重 |w| 0.008–0.011（其他列 0.038），学到的纠偏很弱：p20 把 16 cm 侧漂从 −0.57 压到 −0.38，登顶后略有回拉（x2.4→2.8：−0.44→−0.40），仍远超 0.15 m；代价是 8/12 cm 大幅退化。
判断：侧漂发生在台阶接触期间，正是世界模型最不准、梯度最弱的地方；车道偏移惩罚的梯度在那里推不动动作，却在别处扰乱了 Teacher 的爬法。纠偏需要能处理接触的学习方法（PPO 在真实仿真里试错）。

**定版（2026-10-27，用户决定）：越障用 v3（`logs/supermimic_terrain/probe_v3`）。** v6（riser_anchor）中途停止未验收。
`train_terrain.py` 默认值 = v3（航向 −10、横向速度 −20、倾角软惩罚 −6，其余新增项默认 0）；
`wheel_heading` / `lateral_pos` / `riser_anchor` 代码保留但默认关闭，实验结论见上。
v3 相对 Teacher：12 cm 70%→98%，8 cm 100%→86%，16 cm 都是 0%。

## 4.7 方案 D：从零训练（2026-10-28，训练中）

依据 go2w-supermimic（见上文分析）：Go2W 从零 40000 轮、37850 次 actor 更新、稠密共用奖励、接触头 + 事件优先回放、2 cm 起步课程。
我们的对应实现（`train_terrain.py --from_scratch`，任务 `Scratch-Stairs`）：

| 项 | 设置 |
|---|---|
| 课程 | 9 行，台阶 2→16 cm 连续，4 级、40 cm 踏面；越过 3.4 m、直立、|y|≤0.5 m 升级；翻车 / |y|>1 m / 前进 <0.8 m 降级 |
| actor | 随机初始化，tanh 输出，lr 1e-4，无 Teacher 锚定，无 Teacher 驾驶 |
| 奖励 | 回放里的奖励用 `PreviewTask.reward` 在真实转移上重算（与想象同一函数），翻车 −20（想象里为 1−软存活） |
| 回放 | priority_fraction 0.25：世界模型窗口与想象起点优先取轮前有 ≥3 cm 台阶 / 轮底离地 >2 cm / 翻车的片段（未加接触头） |
| critic | 拟合真实 λ-return（共用奖励后与想象目标一致） |
| 规模 | 2048 env × 12000 轮，每 1000 轮存档 |

冒烟（256 env × 60 轮）：想象观测与 env 观测误差 1.9e-6，共用奖励从 −0.18 升到 +0.07，无非有限数。

**方案 D 中期结果（2026-10-29，第 10000 轮存档，严格验收）**：8 cm 2%、12 cm 0%（16 cm 测试中）。训练课程平均台阶 9.6 cm、25% 在 16 cm 档，但验收几乎全挂：
失败大多在 3.5 s、x≈1.1 m 被"出 0.75 m 车道"提前终止。沿路线探针：平地 x=0.6 m 已偏 +0.17 m，首级 x=1.2 m 偏 +0.48 m，前轮转向 +28°；
PPO Teacher 同位置 y=0.00 / −0.02（8 cm），前轮 −5°。
原因：训练晋级车道放宽到 0.5 m、失败线 1 m，奖励里只有航向和横向速度项、没有横向位置项，策略学成"一上来就斜着往 +y 开半米再爬"。
这是平地上就发生的偏移（世界模型在平地最准），与方案 B 的"台阶接触处梯度弱"不同，收紧课程 + 罚车道偏移应能纠正。
d1 在第 10200 轮停止（用户决定）。d2（2026-10-29 启动）：`--init_from` d1 第 10000 轮（继承 actor + 世界模型，critic 重训）、`--strict_lane`（0.15 / 0.75 m，出车道真终止）、
`--lane_obs --lateral_pos -5`，2048 env × 8000 轮。冒烟：想象/真实观测误差 1.9e-6，出车道终止项已注册。
d2 ❌（第 1040 轮停止）：critic 只预热 300 轮、actor lr 1e-4、无锚定。放开 actor 后速度误差 0.10→0.31 m/s，第 1000 轮存档平地 x=0.6 m 已偏 −0.28 m（d1 为 +0.17），
8/12 cm 登顶 0%、92–98% 被判出界：纠偏过头，同时丢了爬升能力（新 critic 不准时 BPTT 把继承的 actor 推坏，与热启动 v1 同一现象）。
d3（2026-10-29）：同 d2，但锚定继承来的 d1 actor（anchor_coef 0.5，不是 PPO Teacher）、critic 预热 800 轮、actor lr 3e-5。
d3 第 2000 轮（actor 更新约 1000 次）沿路线诊断：8 cm 成功 12.5%（d1 第 10000 轮 0.8%）、登顶 58%；首级 x=1.2 m 横向 +0.30 m（d1 +0.48，d2 第 1000 轮出界）；
12 cm 登顶 3%（d1 81%），出界 31%。训练课程平均等级 0.05→0.36 回升中，速度误差 0.06–0.09 m/s，没有像 d2 那样崩。方向对，继续训。
d3 第 4000 轮诊断（climb_search，64 env）：8 cm 成功 67%、登顶 100%；12 cm 成功 45%、登顶 88%。横向：首级 +0.20 m、最大 +0.30–0.36 m，
登顶后能拉回（x=2.8 m 处 +0.11 / +0.09 m），越线 ≤0.15 m 的 36–39%。训练课程平均台阶 4.2 cm（门限收紧后从 2 cm 重新爬）。
正式验收（accept_terrain，64 回合/档）：**8 cm 61%、12 cm 31%、16 cm 8%**，平地平稳通过。16 cm 第一次在严格门下非零（PPO Teacher、v3 都是 0%）。
失败：8 cm 冲出终点 19 / 提前终止 6；12 cm 停住到超时 30；16 cm 停住到超时 45、提前终止 12——高台阶主要卡在推不上去（课程还没爬到那一档）。
d3 第 6000 / 8000 轮严格验收：8 cm 69% / 14%、12 cm 31% / 23%、16 cm 0% / 3%；登顶率 100/80/27% → 99/92/55%，越线横向 ≤0.15 m 57/37/7% → 9/13/13%。
课程台阶 4.2→6.5 cm 仍线性上涨，但越训越会爬、越歪：课程曲线不能代替验收判断收敛。另查出 `--resume` 不恢复 terrain_levels（课程从头爬）、
从零续训不设锚定、锚定衰减按绝对轮号算（续训第一轮就到终值）、续训回放空时 critic 真实回报窗口（65 步）采样报错，均已修。
d4（2026-10-29）：从 d3 第 6000 轮续训到 40000 轮；lateral_pos 随课程平均台阶 5→10 cm 从 −5 加重到 −15；锚定续训起点 actor，2000 轮内 0.5→0.1；
每 2000 轮自动严格验收（watch_accept.sh → scratch_d4/acceptance_progress.md），横向持续变差就停。


运行命令（均先 `conda activate env_isaaclab; unset PYTHONPATH; source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh`）：

```bash
# 单测
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider source/robot_lab/test/test_supermimic_core.py
# SuperMimic 平地训练（全量；ξ 与 aux 头自动启用）
python scripts/supermimic/train_rover.py --headless --num_envs 4096 --iterations 10000 --steps_per_iter 2 --run_name flat_v1
# 续训
python scripts/supermimic/train_rover.py --headless --resume logs/supermimic/flat_v1/ckpt_010000.pt --iterations 20000 --run_name flat_v1
# 验收（第 ③ 步过门看 open_loop）
python scripts/supermimic/evaluate_rover.py --headless --ckpt logs/supermimic/flat_v1/ckpt_010000.pt --num_envs 768 --steps 1200 --seed 3
# 深度 latent 学生（全量，可以开始）
python scripts/supermimic/train_scan_student.py --headless --num_envs 512 --iterations 4000 \
  --teacher logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt --run_name scan_v1
# 学生楼梯验收（live 与 blank 对照，difficulty 0/0.5/1 对应 8/12/16 cm）
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless --policy student-preview \
  --scan_student --checkpoint logs/supermimic_scan/scan_v1/exported/policy.pt \
  --scenario pyramid_stairs_inv --difficulty 0.0 --depth_mode live
# 回放（与 rsl_rl/play.py 同习惯：默认 64 车，--keyboard 单车遥控，--video 录像，--real-time，--max_steps）
# SuperMimic：省略 --checkpoint 取最新 run 的最新 ckpt；同时导出 exported/policy.pt、policy.onnx、policy_contract.json
python scripts/supermimic/play_rover.py --load_run flat_v1
python scripts/supermimic/play_rover.py --load_run flat_v1 --keyboard
python scripts/supermimic/play_rover.py --checkpoint logs/supermimic/flat_v1/ckpt_010000.pt --export_only
# 深度学生：走原 play.py，加 --scan_student（省略 --checkpoint 取 logs/supermimic_scan 最新导出）
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --scan_student --checkpoint logs/supermimic_scan/scan_v1/exported/policy.pt --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5
# SuperMimic 越障（参数名与 rsl_rl/train.py、play.py 一致）
python scripts/supermimic/train_terrain.py --headless --num_envs 1024 --max_iterations 1000 --run_name probe_v3b --tilt_soft 0
python scripts/supermimic/train_terrain.py --headless --resume --load_run probe_v3b --max_iterations 3000   # 续训，写回原 run
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --supermimic_terrain --load_run probe_v3b --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 1.0
# 一键验收（三档评测 + 诊断 + 汇总表）
python scripts/supermimic/accept_terrain.py --load_run probe_v4
# 单档楼梯验收（16 cm 途中倾角门 45°，见 stairs_acceptance.CLIMB_TILT_DEG；--climb_tilt_deg 可覆盖）
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless --policy teacher-preview --supermimic_terrain \
  --checkpoint logs/supermimic_terrain/probe_v3b/exported/policy.pt --scenario pyramid_stairs_inv --difficulty 1.0 \
  --num_envs 32 --episodes_per_env 2 --seed 142
# 训练前校验（改了奖励/动作/任务后重跑）
python scripts/supermimic/verify_rover.py --headless
python scripts/supermimic/verify_scan.py --headless
```

## 5. 风险与未验证项

- 外部三仓库仅读 README；世界模型特征、地形任务想象阶段如何处理高度扫描，需读 go2-supermimic 源码确认。
- PhysX 4096 env 的采样速度是否跟得上 UTD 0.4，未测。
- 轮子是连续旋转关节（限位 ±inf），任何按关节限位做的采样/归一化都要跳过轮子。
- H=32（0.64 s）覆盖不了整段楼梯，长时序依赖 critic；16 cm 仍是探索上限。
- 直立可保持区仅约 75°–105°（旧探针结论已被标注为待复核），mimic 参考片段不要超出实测可行域。
