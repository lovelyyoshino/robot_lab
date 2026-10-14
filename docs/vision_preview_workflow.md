# 视觉前瞻抬轮与8→12→16cm楼梯训练

更新：2026-10-14。本文直接描述当前单一Preview任务，不把历史奖励迭代当成多个训练版本。Teacher总预算8000轮，当前合同的一次全量训练已完成；Student全量蒸馏由用户自行执行。最新能力结果与验证边界见第12节。

## 1. 目标、策略与能力边界

目标是保持世界+X前进方向，接近障碍后主动协调轮腿并稳定越过。全宽允许任一前轮先主动抬起，另一轮可以随后主动或被动滚上，不强求同时悬空；窄障碍只验对应侧。两个前轮机会独立，不固定左先。高障碍跳跃不在本轮范围。

| 策略 | 输入与作用 | 训练方式 |
|---|---|---|
| Teacher Preview | 232维：45维本体／路线命令＋187点特权高度扫描 | PPO，学越障与楼梯 |
| Student Temporal Preview | 4141维：45维本体／路线命令＋4096维深度历史 | 固定Teacher提供动作标签，DAgger混合回合蒸馏 |

Teacher仅为训练和诊断策略，最终视觉行走使用Student。当前Student为MLP深度历史输入，未实现CNN＋GRU、共享Actor或同轮交替PPO；本流程不需要恢复示范、AMP或单独BC初始化。动作维度仍为12，执行器与力矩限制没有扩大。

10cm以轮胎最前端到障碍前缘的距离计算，不是车身中心距离。65cm用于预瞄；25cm内可接地减载准备；首次真实离地5mm应发生在8–14cm窗口，之后保持到越缘并稳定落轮。单障碍独立评测分别记录真实离地距离和2cm高度达标距离，不能混用。楼梯评测验证实际登顶通行，暂不宣称独立验证每一级的提前抬轮时序或视觉因果性。

16cm超过这台车约12cm的单轮无载竖直抬升，需协调车身姿态、支撑及滚动。它是探索上限，不保证当前结构或8000轮必然达到；先看8cm与12cm结果再判断。

## 2. 一个网络的三个自动阶段

| 阶段 | 行 | 楼梯每级高度 | 级数／总高 | 楼梯比例 | 全宽／左窄／右窄／方块／平地 |
|---|---|---|---|---|---|
| stage1 | 0–2 | 8cm | 2级／16cm | 20% | 25%／20%／20%／5%／10% |
| stage2 | 3–5 | 12cm | 3级／36cm | 40% | 20%／15%／15%／5%／5% |
| stage3 | 6–8 | 16cm | 4级／64cm | 55% | 10%／10%／10%／10%／5% |

每个环境从stage1开始，完成两次可晋级障碍的稳定通行后升一行；失败会清掉成功连胜并按失败规则降一行。平地成功不能晋级。第2行跨到第3行、第5行跨到第6行时，必须实际通过本档楼梯。全宽、左右窄障碍可以稳定滚上，不再被双侧主动门卡住；提前动作仍有独立信用和验收，通行不能冒充主动抬轮。

8000轮是一个PPO run的总预算，不是每阶段各8000轮，也不需要三个模型。不同环境独立晋级，所以stage1/2/3比例可以同时非零；比例没有变化时要查本档楼梯的真实通行，不能通过play显示16cm楼梯来声称课程已达到stage3。

楼梯沿+X真正逐级上升，首级前缘距出生中心1.20m，踏面40cm，宽3m，最后一级后为同高顶端平台；固定评测终点依次为2.5／2.6／3.0m，给后轮登顶留余量。场景参数 `pyramid_stairs_inv` 沿用原入口，但Preview对应的几何已换为直楼梯。单障碍仍为8–12cm、前后宽40cm，方块为6–12cm，不随楼梯阶段变成16cm单障碍。

每一级轮子通过旧前缘、接地承重至少60ms并看到更前方的新前缘后重新武装；上一阶被动滚上也能学习下一阶。左右完成同一级只付一次完整动作奖，后退或同阶抬落不能重新领。四条额外地面射线只改善奖励净空判定，不增加Teacher或Student输入。

## 3. 奖励与成功判定

保留12项路线、稳定与安全奖励：

| 项 | 权重／规则 |
|---|---|
| 世界+X速度跟踪 | +5，静止零基线，同时检查前进朝向 |
| 竖直速度／横滚俯仰角速度 | −0.5／−0.05，给正常爬升留空间 |
| 姿态／偏航角速度 | −0.1／−0.05，限制失稳与旋转 |
| 转向关节偏置 | −0.1 |
| 力矩／关节限位／动作变化 | −2.5e−5／−5／−0.005 |
| 底盘碰撞／过大接触力 | −1／−1.5e−4 |
| 主动抬轮与通行 | 权重+2：单轮进度额度1、有效事件0.5、任一目标前轮完整动作2、整车稳定通行4 |

净空进度、起抬事件与完整动作分开计算；非窄障碍在5mm离地前也可获得真实净空和关节抬升共同增长的引导。窄障碍采用独立稳定承重基线，要求明显减载和至少5mm净空，动作期允许两轮对角支撑。远处悬轮、失败后的重复抬落及提前接地深蹲有小额成本，窗口内协同动作保持余量。

训练终点须朝前、正立、横偏不超过15cm，至少三轮承重、合力达到半体重，稳定0.30秒后发通行奖励并成功截断；成功截断作为timeout处理，课程在奖励reset前读取本回合证据。碰撞／漏抬指标用于诊断，未额外增加阻挡碰撞扣分。

固定评测比训练多检查地形过滤承重、非轮接触和楼梯顶面高度，连续稳定0.50秒后截断。Preview固定命令0.30m/s、回合上限20秒；旧8秒／0.4m/s协议的结果不能与本轮直接混比。全宽至少一个前轮完整提前动作且整车通行；左／右窄仍必须对应侧完成。双前轮事件率保留为诊断，不再作为全宽成功硬门。

## 4. 从零训练Teacher：8000轮

使用已经能启动Isaac Lab的Python终端，直接执行完整路径命令。训练前保留旧模型和报告；新run不加 `--resume`，不传旧checkpoint。课程、奖励、楼梯几何与终止条件的合同都已改变，维度相同也不能直接续用旧合同。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/teacher \
  --run_name stairs_climb \
  --num_envs 1024 --max_iterations 8000 --seed 42 --headless
```

显存或速度压力大可将1024降为512／256；这会改变每轮样本数，8000轮不再具有相同总样本量。正式训练不要使用诊断用的 `env.scene.terrain.max_init_terrain_level=8`：正式课程默认从0级开始。

日志位于 `/media/bigdisk/robot_lab/logs/vision_preview/teacher/时间戳_stairs_climb/`。启动时记录真实目录，正常跑满后产出 `model_7999.pt`，同时保存 `params/agent.yaml`、`env.yaml` 和 `vision_contract.json`。若只是完成短训，不能拿它正式蒸馏Student。

查看TensorBoard：

```bash
tensorboard --logdir /media/bigdisk/robot_lab/logs/vision_preview/teacher
```

重点看 `Curriculum/terrain_levels/stage_1/2/3`、`mean_level` 和 `stairs_8cm_fraction/stairs_12cm_fraction/stairs_16cm_fraction`。后三项是全部环境中正在对应高度楼梯上训练的比例，不是成功率，其总和可以小于1。重置时的抽样比例是上表权重，实时比例受回合长度影响。

动作指标包括 `Metrics/prelift/stairs/opportunities`、`event_rate`、`active_steps_completed`、`route_stable_success_rate` 以及逐单障碍的 `event_rate/sequence_complete_rate/route_reward_rate`。全宽event_rate改为任一前轮有有效动作，`both_front_event_rate`仍单独保留。没有机会数时0事件率不能证明失败；有大量机会仍不动作、12cm通行长期为0或持续旋转时，应保存checkpoint和报告分析。

可以在1000、2000、4000、6000轮检查保存模型；检查点不是最终能力验收。总reward允许随新场景变难而降低，不能跨奖励合同比较，也不能因为reward涨就判断学会楼梯。

## 5. 选择本次模型

先列出本次正式最终模型：

```bash
find /media/bigdisk/robot_lab/logs/vision_preview/teacher \
  -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort
```

以下表达式只搜索 `*_stairs_climb/model_7999.pt`，不会选旧 `active_prelift` 或短训run。没有输出时不要执行后面的命令。多个新run时，它选最后时间戳，不评选最佳模型；正式一组对照应替换为选定模型的真实绝对路径，核对报告checkpoint和SHA256一致。正在训练的中期模型请手动选已写完的实际路径，不使用最终模型表达式。

## 6. Teacher单障碍独立验收

先确认全宽任一前轮有完整提前动作，再检查左右窄障碍对应侧；另一侧滚上不会让全宽成绩变成0。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy teacher-preview --scenario single_step --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/teacher_full_seed142.json
```


```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy teacher-preview --scenario single_step_left --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/teacher_left_seed142.json
```


```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy teacher-preview --scenario single_step_right --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/teacher_right_seed142.json
```

结果先看 `complete=True`、回合数32和 `passed`。`success_rate` 为整车通行；`prelift.success_rate` 为目标侧主动动作完整且通行，两者不能混同。各场景70%为开发门，降低门槛不能补足缺失动作。全宽 `gates.required_active_count=1`，窄障碍还限制对应侧；查看左右事件／保持／落轮率和takeoff距离，确认没有远处早抬。

## 7. Teacher楼梯：8→12→16cm分别验证

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy teacher-preview --scenario pyramid_stairs_inv --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/teacher_stairs_8cm_seed142.json
```


```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy teacher-preview --scenario pyramid_stairs_inv --difficulty 0.5 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/teacher_stairs_12cm_seed142.json
```


```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy teacher-preview --scenario pyramid_stairs_inv --difficulty 1.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/teacher_stairs_16cm_seed142.json
```

报告 `climb_protocol` 给出实际单级高度、级数、总高和终点；`terminal_progress_m`、`maximum_progress_m`用于区分首级卡住和接近登顶。`success_rate`必须来自真正到达顶端且保持方向和承重，不能只看走过2.5m。楼梯未独立验证逐级主动时序，配合debug回放和训练楼梯事件指标看动作。8cm通过、12cm不通过时，只能声明8cm技能，不能因为生成了16cm地形就说支持16cm。

最终再将seed142改为314／942复测同一个模型，并使用较大回合数；32回合开发门的统计波动不能作为部署保证。方块可用 `--scenario boxes --difficulty 0.5` 另测。

## 8. Teacher play与深度显示

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 4 --max_steps 1000 \
  --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5 --preview_debug
```

将difficulty改为0.0／1.0分别看8cm／16cm。看单障碍用以下命令：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 4 --max_steps 1000 \
  --preview_scenario single_step --preview_difficulty 0.0 --preview_debug
```

`--preview_scenario mixed` 默认展示全宽／左窄／右窄／楼梯／方块，4台车对应前4列；展示难度不代表训练阶段。固定scene＋difficulty才是可比较回放。GUI下1–4台车自动显示对应角落深度，Teacher为显示相机、Student为实时传感器；大于4台或headless不显示面板。原始深度画面不等于Student实际裁剪的历史输入。

`--preview_debug`每5步打印最多4台车的 `edge_lead_m`、`phase`、`lift_m`、`clearance_m`、`support_N`、`takeoff_lead_m` 和 `height_lead_m`。前缘未命中为nan／unknown，不当作0。楼梯会在真正新前缘重新武装，锁存距离也是当前级的记录；稀疏输出不能取代完整时序评测。成功回合会重置重新演示，画面跳回起点是正常成功截断。

## 9. 选择Teacher并训练Student

面向最终提前抬轮目标，先完成单障碍与计划使用的楼梯档位验收再正式蒸馏。当前Teacher已完成8000轮，可用于低障碍通行基线实验，但提前主动动作和16cm楼梯仍未达标（见第12节）；动作蒸馏不能保证补足Teacher未掌握的技能。以下明确使用这次已经评测的Teacher路径，后续若训练了更好的Teacher，应填写实际选定路径。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint /media/bigdisk/robot_lab/logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/student \
  --run_name stairs_depth \
  --num_envs 128 --max_iterations 8000 --seed 42 --headless
```

新建Student不要加 `--resume`，此时checkpoint是固定Teacher监督来源；Student同样使用三阶段地形课程。任务自动开启相机，显存不足可降为64／32，每轮样本量会改变。Teacher输入232、Student输入4141，深度合同和来源哈希会保存。保留整个run目录及同级 `params`，不能只保留 `.pt`。

4141维为45维本体／路线命令＋8×16×32深度历史，历史跨度约0.7秒；Student为MLP，隐藏层1024／512／256，输出12维动作。当前为 `DistillationRunner`＋`DaggerDistillation`，没有CNN＋GRU、共享Actor或同轮交替PPO。

DAgger按每台车、每个回合选择驱动者，Teacher驱动概率前4000轮由75%降到10%，之后保留10%；监督标签始终来自同一状态的固定Teacher。每轮24步、4个epoch、每4步累计更新一次，即每轮24次优化器更新；学习率在8000轮内由5e−4降到1e−4。训练reward不作为Student的PPO目标，reward和阶段日志含Teacher驱动回合，不能当成Student独立能力；最终play／eval全部由Student控制。

查看日志：

```bash
tensorboard --logdir /media/bigdisk/robot_lab/logs/vision_preview/student
```

关注 `Loss/behavior`、`Loss/teacher_drive_probability`、`Loss/teacher_driven_fraction`、`Loss/learning_rate` 和 `Curriculum/terrain_levels/stage_1/2/3`。概率是当前采样参数，实际驱动比例还受各回合长度影响，两者不必逐轮相等。loss下降只能说明动作更接近Teacher，不能免除Student独立评测。

## 10. Student验收与视觉消融

以下表达式只选择完整8000轮后的 `*_stairs_depth/model_7999.pt`。没有文件时不要执行；中期模型请填写已经保存的实际绝对路径。多个run时最后时间戳不代表最佳策略，一组对照固定同一模型并核对报告 `checkpoint_sha256` 一致。

### 10.1 全宽、左右窄障碍的实时深度独立评测

全宽8cm：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_full_live_seed142.json
```

左窄8cm：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step_left --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_left_live_seed142.json
```

右窄8cm：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step_right --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_right_live_seed142.json
```

每份报告先看 `complete=True`、回合数32，再分别看 `success_rate`（稳定通行）、`prelift.success_rate`（目标轮提前动作完整且通行）和 `passed`（综合门）。Student与Teacher使用同一开发门槛70%，能滚过障碍不能冒充提前主动动作。正常完成但未达技能门时脚本退出码为1，需结合JSON区分验收失败和执行异常。

### 10.2 8／12／16cm楼梯分别评测

8cm×2级：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario pyramid_stairs_inv --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_stairs8_live_seed142.json
```

12cm×3级：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario pyramid_stairs_inv --difficulty 0.5 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_stairs12_live_seed142.json
```

16cm×4级探索：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario pyramid_stairs_inv --difficulty 1.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_stairs16_live_seed142.json
```

楼梯目前验证实际登顶和稳定性，暂未独立验收每一级的提前动作时刻。

### 10.3 实时／空白／冻结深度对照

第10.1节全宽live报告为实时基线。保持同一个Student、相同场景／难度／seed／回合数，再测试空白和冻结：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode blank \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_full_blank_seed142.json
```

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode frozen \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_full_frozen_seed142.json
```

`live`使用实时深度历史；`blank`将4096维深度替换为归一化最远距离值+1，并非数值0；`frozen`在每回合固定初始8帧历史，reset后刷新。三种模式的45维本体／路线命令始终实时。默认关闭训练中的观测扰动；额外验证延迟／丢帧／冻结／噪声时加 `--camera_corruption`，三种对照须一致。`depth_probe`始终读取实际相机，blank／frozen报告也可能包含正常障碍图，不能据此认为Student接收了live输入。

| 对照结果 | 可以得出的结论 |
|---|---|
| 实时成功，空白／冻结明显下降 | 支持Student利用视觉，仍需多场景、多seed确认 |
| 三种都成功 | 视觉依赖尚未证明，可能利用固定动作、地形先验或本体反馈 |
| 三种都失败 | 先验证Student独立控制，不能据此判断视觉是否有效 |

在左右窄障碍及楼梯上重复三种对照，改变scenario／difficulty时同时改输出文件名，并补测seed314／942；不能只凭深度截图或单seed结果宣称视觉决策完成。固定场景先验及异常深度输入的分布变化仍限制对照的解释范围。

### 10.4 Student play与抬轮时刻

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 4 --max_steps 1000 \
  --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5 --preview_debug
```

单障碍将scenario改为 `single_step`、`single_step_left` 或 `single_step_right`，difficulty改为0.0；楼梯difficulty0.0／0.5／1.0分别为8／12／16cm。GUI下1–4台车显示角落深度，大于4台或headless不显示；原始画面不等于Student裁剪后的8帧输入。`takeoff_lead_m`为真实离地距离，目标8–14cm；`height_lead_m`为高度达标距离，不能混用。完整动作判读遵循第8节，GUI面板本次未实测。

## 11. 同合同续训

仅允许本次同合同、同策略角色checkpoint续训。Teacher例子追加1000轮：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --resume --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/teacher_resume \
  --run_name stairs_climb_resume --num_envs 1024 --max_iterations 1000 --seed 42 --headless
```

Student完成8000轮后追加1000轮：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --resume \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/student_resume \
  --run_name stairs_depth_resume \
  --num_envs 128 --max_iterations 1000 --seed 42 --headless
```

Student续训恢复Student、内置Teacher、优化器和训练进度，不能再加载Teacher当成Student续训。续训是追加轮数，选择候选时切换到实际续训产物路径；不能继续搜索首轮目录而误测旧模型。当前地形课程状态不存入策略checkpoint，重建环境会从stage1开始；策略权重保留，课程需要重新通过晋级，不代表加载了错误网络。

## 12. 当前验证与历史证据

### 12.1 当前Teacher的8000轮独立结果

已完成当前源码合同的8000轮Teacher训练。模型为 `/media/bigdisk/robot_lab/logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt`，SHA256为 `568d246abd48d7e919f7090be09100ee4d48d0c6ab22d8fbce3a55e7e314aa39`。以下每个场景seed142、16环境×2回合，6份报告均 `complete=True`、32回合且模型SHA一致：

| 场景 | 稳定通行率 | 提前主动动作并通行 | 综合验收 |
|---|---|---|---|
| 全宽8cm | 100% | 18.75% | 未通过 |
| 左窄8cm | 100% | 0% | 未通过 |
| 右窄8cm | 96.875% | 6.25% | 未通过 |
| 8cm×2级楼梯 | 100% | 未单独验收逐级提前动作 | 通过通行门 |
| 12cm×3级楼梯 | 75% | 未单独验收逐级提前动作 | 通过通行门 |
| 16cm×4级楼梯 | 0% | 未单独验收逐级提前动作 | 未通过 |

报告为 `/media/bigdisk/robot_lab/logs/vision_preview_diagnostics/final_7999_review.TbdtwP/` 下的 `full_8cm.json`、`left_8cm.json`、`right_8cm.json`、`stairs_8cm.json`、`stairs_12cm.json` 和 `stairs_16cm.json`。当前Teacher是低障碍稳定通行基线，约10cm提前主动抬轮与16cm攀爬仍未完成，不能将前两档楼梯通行等同于全部目标达标。16cm失败的具体机制仍需诊断，不能仅凭0%认定结构不可行或评测错误。

### 12.2 Student链路小规模验证

193项相关测试通过，覆盖DAgger、深度历史、模型合同、通行和提前动作验收；测试关闭了与本任务无关的全局pytest插件自动加载，未修改训练依赖。4环境Student新训2轮后，模仿loss从0.1392降至0.0795；续训1轮从iteration2继续，loss约0.0815。观测为Student4141／Teacher232／Critic235、动作12，权重有限，Teacher权重与来源actor保持完全一致。此处只证明训练和恢复流程，不能把3轮当成技能训练。

同一个2轮Student在全宽8cm、seed142、4环境×1回合下完成live／blank／frozen评测；三份报告均完整、通行率0%、提前动作通行率0%、未达技能门，无非有限数回合。这不代表全量Student也会失败，也不能据此判断视觉无效。相机探针确认被测全宽场景中障碍可见，进入约14cm窗口时两前轮方向均有障碍像素；未验证所有场景可见或网络利用了深度。

证据目录为 `/media/bigdisk/robot_lab/logs/vision_preview_diagnostics/student_access.T8MpL4/`：`train.log`、`resume.log`、`live.json`、`blank.json`、`frozen.json` 和 `depth_live.png`。模型均为诊断短训产物，不放入正式Student目录当作候选。

### 12.3 历史基线与实现验证

以下保留历史记录，与上述最新8000轮Teacher及Student短训结果分开解释。

2026-10-13旧8000轮模型的真实复测：全宽8cm通行100%、左前主动96.875%、右前0%；按本轮任一前轮完整动作重算31/32合格。左右窄完整主动约53.125%／50%；旧6cm楼梯通行56.25%，旧12cm楼梯0/32。训练一直stage1，因此这些数据不能证明本轮8/12/16cm课程已学会。原始证据保留于 `logs/vision_preview/eval/20261014_new_teacher.1pa2HW/`，未改写旧JSON。

本次验证区分自动课程／奖励正确性和策略能力。484项相关测试通过，覆盖新前缘重新武装、被动上阶、同阶去重、持续承重、两次通行晋级、跨stage楼梯门、单侧完整验收、楼梯顶面判定和深度历史／显示回归。32环境短程PPO正常结束，Actor232／Critic235／动作12，loss有限；诊断初始化主动覆盖三档，只验证各档场景与日志，不把初始化比例当作自然晋级成绩。

使用当前源码合同的8轮模型，8／12／16cm固定楼梯均完成2回合独立评测、产生完整JSON；通行率均为0%，短训模型尚未达到技能门。4台车、12cm楼梯的80步headless play正常结束，场景打印为12cm×3级、终点2.60m，逐步诊断未异常。GUI深度角落面板本轮未实测。

额外实际仿真探针将车放到16cm楼梯顶面，用于单独检查成功判定：四轮承重合力131.52N、实际车重131.22N，连续稳定后正确成功截断。另一组真实reset探针验证两次稳定通行后第2→3行、第5→6行、第7→8行均晋级。摆放探针不能计作策略爬楼成功率。证据目录为 `logs/vision_preview_diagnostics/stairs_multistage/2026-10-14_11-04-14_verified/`，包含三档报告、`top_support_probe.json` 和 `curriculum_reset_probe.json`；临时探针源码不加入训练入口。

上述实现验证时语法编译、注释检查和diff空白检查通过，环境未安装ruff／pyright，未运行这两项静态检查。当前Teacher的正式8000轮已完成，结果见第12.1节；Student全量蒸馏、约10cm主动抬轮达标、16cm能力、GUI面板与实车迁移仍需后续训练和独立验证。

遇到合同不匹配，先核对是不是旧模型或修改后保存的新合同，不修改JSON绕过。NaN／Inf、仿真异常或动作爆振应先排查；正常训练但当前档不通行时保留模型与分场景报告。奖励／任务／楼梯课程入口分别在 `vision_preview.py`、`vision_preview_reward.py`、`vision_preview_env_cfg.py`，评测脚本为 `evaluate_parallelogram_vision.py`。
