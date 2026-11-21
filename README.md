# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

本仓库 fork 自 [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)（Apache-2.0），团队分支为 `feat/supermimic`。
上游的通用内容（安装、Docker、上游机器人任务、添加新机器人、排错、引用）保留在本文对应章节；
本分支新增的内容集中在平行四边形四轮腿车（下称“四轮车”，代码中为 `parallelogram_rover`）。

## 目录

1. [项目简介与当前状态](#intro)
2. [安装与环境](#install)
3. [仓库结构](#layout)
4. [任务总览](#tasks)
5. [通用训练、回放与上游示例](#generic)
6. [四轮车 PPO 流程](#ppo)
7. [SuperMimic 越障训练（当前主线）](#supermimic)
8. [工具与调试](#tools)
9. [添加新机器人](#add-robot)
10. [引用、致谢与许可证](#credits)

<a id="intro"></a>

## 1. 项目简介与当前状态

**robot_lab** 是基于 Isaac Lab 的机器人强化学习扩展库，可以在 Isaac Lab 核心仓库之外独立开发。在上游基础上，本分支新增了以下内容：

- **四轮车资产与任务**：URDF/MJCF 位于 `source/robot_lab/data/Robots/custom/parallelogram_rover/`，资产定义位于 `source/robot_lab/robot_lab/assets/parallelogram_rover.py`。
- **PPO 流程**（rsl_rl）：盲走基线、视觉前瞻抬轮和三阶段楼梯 Teacher、深度 Student 的 DAgger 蒸馏、侧躺恢复（动作先验加 residual PPO）、双轮直立和起立、三策略混合切换，以及相应的验收脚本。
- **SuperMimic 越障**：用世界模型加想象 BPTT 代替 PPO，从零训练越障策略。

### 当前状态

| 流程 | 状态 | 入口 | 现有产物与结论 |
|---|---|---|---|
| SuperMimic 越障 | **当前主线**，训练中 | `scripts/supermimic/run_train.sh`、`watch_accept.sh`、`accept.py` | 2026-11-07 14:10 起训练 `logs/supermimic_terrain/obstacle_stairslant_20261107/`：按预抬的物理边界设计（单台阶每根轴一个轮先抬、另一个滚上去；楼梯前轴每级一个轮先抬、车身斜 12° 让后轴两轮先后翻上立面，见 [docs/supermimic.md §1.1](docs/supermimic.md)）。`obstacle_prelift_20261106/` 第一次学会严格预抬（第 18000 轮 17.5 cm 窄台阶 80–83%），保留 ckpt 12000/16000/18000 作基线；`obstacle_prelift_20261107/` 第 6000 轮窄台阶预抬 52%，但楼梯卡在后轴、整宽台阶停在块上，改动见设计文档"每轮训练后的修正"。更早的 `obstacle_bumpclimb_20261106/` 能碰了再爬，但严格预抬 0%、台阶横滚 8–19°。见[第 7 节](#supermimic) |
| SuperMimic 对照基线 | 只读保留 | `logs/supermimic_terrain/scratch_40k_r8k/`、`normal_drive_rehearsal_20k_20261103/` | 按旧验收协议，楼梯 8/12/16 cm 的通过率分别为 100/100/84%、100/98/89%。按当前验收协议需要重测，见 [7.6](#sm-baselines) |
| 视觉前瞻 Teacher（PPO） | 冻结，作为对照 | `train.py` + `…-Vision-Teacher-Preview-v0` | `logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt`：楼梯 8/12/16 cm 通过率 100/75/0%，单障碍提前抬轮未达标 |
| 深度 Student（DAgger） | 冻结 | `train.py` + `…-Vision-Student-Temporal-Preview-v0` | `logs/vision_preview/student/2026-10-14_19-40-29_stairs_depth/model_7999.pt`：能通行单障碍，未学会提前抬轮 |
| 侧躺恢复（residual PPO） | 冻结，已通过开发评测和一个留出 seed | `train.py` + `…-Recovery-Side-Residual-…-v0` | `logs/recovery_side_restart_v1/policy/2026-09-30_10-37-40_constant/model_5098.pt`：成功率 100%，门槛为 85% |
| 盲走（PPO） | 遗留基线 | `train.py` + `…-Rough-Parallelogram-Rover-v0` | `logs/rsl_rl/parallelogram_rover_rough/2026-08-13_17-35-42/model_19999.pt` |
| 双轮直立、起立、两轮运动 | 遗留，没有保留 checkpoint | `…-Upright-*-v0` | 需要重新训练 |
| 三策略混合切换 | 遗留，只做过仿真集成 | `scripts/play_parallelogram_hybrid.py` | 依赖直立 checkpoint，当前无法直接运行 |
| SuperMimic 平地、深度扫描学生、旧越障路线 | 遗留工具 | 见 [7.8](#sm-other) | 未接入当前越障训练 |
| 上游四足、轮足、人形任务 | 与上游一致，本分支未改动 | 见[第 4 节](#tasks)、[第 5 节](#generic) | — |

> [!NOTE]
> 上游建议在 Gazebo 或实机上运行策略时使用 [rl_sar](https://github.com/fan-ziqi/rl_sar)。上游讨论区：[GitHub Discussion](https://github.com/fan-ziqi/robot_lab/discussions)、[Discord](http://www.robotsfan.com/dc_robot_lab)。

<a id="install"></a>

## 2. 安装与环境

### 2.1 团队训练服务器（已装好）

仓库位于 `/media/bigdisk/robot_lab`，Isaac Lab 位于 `/media/bigdisk/IsaacLab`。本文所有命令都默认在仓库根目录、已加载下面这套环境的终端里执行：

```bash
cd /media/bigdisk/robot_lab
conda activate env_isaaclab
unset PYTHONPATH
source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh
R=/media/bigdisk/robot_lab   # 下文命令中的 $R
```

`train.py` 和 `play.py` 会把 `--experiment_name` 拼接到 `logs/rsl_rl/` 之后，所以要把产物放在其他目录时，`--experiment_name` 必须写**绝对路径**（下文统一用 `$R/logs/...`）。

使用规则：

- 同一时刻只启动一个导入四轮车的 Isaac 进程；需要并行训练时请错开启动，否则会报 mimic 关节缺失。SuperMimic 的验收脚本会对 `logs/.isaac_eval.lock` 加锁，多个验收任务会自动排队执行。
- 评测脚本的退出码：0 表示通过，1 表示评测正常跑完但未达验收门，2 表示执行出错。只有 2 需要排查运行问题。

### 2.2 从零安装（上游步骤）

| robot_lab 版本 | Isaac Lab 版本 | Isaac Sim 版本 |
|---|---|---|
| `main` 分支 | `main` 分支 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.2` | `v2.3.2` | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.2` | `v2.2.1` | Isaac Sim 4.5 / 5.0 |
| `v2.1.1` | `v2.1.1` | Isaac Sim 4.5 |
| `v1.1` | `v1.4.1` | Isaac Sim 4.2 |

1. 按照 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab，推荐使用 conda 安装方式。
2. 在 `IsaacLab` 目录之外克隆本仓库。上游仓库是 `https://github.com/fan-ziqi/robot_lab.git`，团队 fork 是 `git@github.com:lovelyyoshino/robot_lab.git`（分支 `feat/supermimic`）：

   ```bash
   git clone -b feat/supermimic git@github.com:lovelyyoshino/robot_lab.git
   cd robot_lab
   ```

3. 用装有 Isaac Lab 的 Python 安装扩展，然后列出全部环境，确认安装成功：

   ```bash
   python -m pip install -e source/robot_lab
   python scripts/tools/list_envs.py
   ```

<details>
<summary>配置 IDE（可选）</summary>

在 VSCode 中按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，运行 `setup_python_env`，并按提示填入 Isaac Sim 的安装路径。成功后会生成 `.vscode/.python.env`，里面包含 Isaac Sim 和 Omniverse 各扩展的 Python 路径，用于代码补全和索引。

</details>

<details>
<summary>作为 Omniverse 扩展加载（可选）</summary>

示例 UI 扩展定义在 `source/robot_lab/robot_lab/ui_extension_example.py`，启用扩展后会自动加载。启用步骤如下：

1. 打开 `Window` -> `Extensions`，点击汉堡菜单（☰）-> `Settings`，在 `Extension Search Paths` 中加入 `robot_lab/source` 的绝对路径。如果列表中还没有 Isaac Lab 的扩展目录（`IsaacLab/source`），也一并加入。然后点击汉堡菜单 -> `Refresh`。
2. 在 `Third Party` 分类下找到本扩展并启用。

</details>

### 2.3 Docker（上游）

<details>
<summary>展开</summary>

需要先按 [Isaac Lab 部署文档](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html) 在本地构建 `isaac-lab-base` 镜像，构建完成后可以用 `docker images` 确认。之后构建本项目的镜像（名称为 `robot-lab`，可以在 [`docker/docker-compose.yaml`](docker/docker-compose.yaml) 中修改）：

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
docker compose --env-file .env.base --file docker-compose.yaml up -d      # 去掉 -d 则在前台运行
docker exec --interactive --tty -e DISPLAY=${DISPLAY} robot-lab /bin/bash  # 进入容器
docker compose --env-file .env.base --file docker-compose.yaml down       # 停止并删除容器（保留镜像）
```

</details>

<a id="layout"></a>

## 3. 仓库结构

```text
robot_lab/
├── source/robot_lab/
│   ├── robot_lab/
│   │   ├── assets/                 # 机器人资产；parallelogram_rover.py 为本分支新增
│   │   ├── tasks/
│   │   │   ├── manager_based/
│   │   │   │   ├── locomotion/velocity/config/{quadruped,wheeled,humanoid,others}/
│   │   │   │   │   └── wheeled/parallelogram_rover/  # 盲走、视觉 Teacher/Student、Preview 楼梯课程与验收几何
│   │   │   │   ├── recovery/       # 倒地、侧躺恢复任务族
│   │   │   │   ├── upright/        # 双轮直立、起立、两轮运动任务族
│   │   │   │   ├── hybrid/         # 三策略切换器（planner.py / runtime.py，不注册 gym 任务）
│   │   │   │   ├── supermimic/     # SuperMimic 环境：Flat / Terrain-Preview / Scratch-Stairs / Obstacle
│   │   │   │   └── beyondmimic/    # 上游 G1 动作跟踪
│   │   │   └── direct/g1_amp/      # 上游 G1 AMP
│   │   ├── supermimic/             # 世界模型、BPTT learner、解析奖励（preview_task.py）、可微运动学、镜像增强
│   │   └── ui_extension_example.py
│   ├── data/Robots/custom/parallelogram_rover/   # urdf/、mjcf/、meshes/、calibration/
│   └── test/                       # 离线单元测试（pytest）
├── scripts/
│   ├── reinforcement_learning/
│   │   ├── rsl_rl/                 # train.py、play.py，以及本分支的评测、合同和热启动模块
│   │   ├── cusrl/  skrl/           # 上游训练与回放入口
│   ├── supermimic/                 # SuperMimic 训练、导出、验收
│   ├── tools/                      # 资产转换、探针、离线导出、list_envs 等
│   └── play_parallelogram_hybrid.py
├── docs/supermimic.md              # SuperMimic 越障训练设计文档
├── docker/
└── logs/                           # 训练产物，不入 Git
    ├── rsl_rl/<experiment_name>/<时间戳>_<run_name>/   # rsl_rl 默认位置：model_*.pt、params/、exported/
    ├── vision_preview/{teacher,student,eval}/
    ├── recovery_side_restart_v1/{source,demos,policy,evaluation}/
    └── supermimic_terrain/          # SuperMimic 越障 run，布局见 7.5
```

<a id="tasks"></a>

## 4. 任务总览

运行 `python scripts/tools/list_envs.py` 可以列出全部已注册的任务。

### 4.1 四轮车（本分支）

| 用途 | 任务 ID | 观测 → 动作 | 说明 |
|---|---|---|---|
| 盲走 | `RobotLab-Isaac-Velocity-Flat-Parallelogram-Rover-v0`<br>`RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0` | 45 → 12 | 基线，也是混合切换中的 `loco` 策略 |
| 视觉前瞻 Teacher | `RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0` | 232（45 本体 + 187 高度扫描）→ 12 | 当前 PPO 视觉主任务，见 [6.2](#ppo-teacher) |
| 深度 Student | `RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0` | 4141（45 + 8 帧 16×32 深度）→ 12 | DAgger 蒸馏，见 [6.3](#ppo-student) |
| 旧视觉合同 | `…-Vision-Teacher-v0`、`…-Vision-Student-v0`、`…-Vision-Student-Temporal-v0` | — | 旧合同，仍然注册；混合切换和评测脚本可以加载 |
| 侧躺恢复（主线） | `RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0` | 60（critic 72）→ 12 维残差 | program31 先验 + residual PPO，见 [6.4](#ppo-recovery) |
| 侧躺采集源 | `RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0` | 58 → 12 | 用来生成采集环境和控制合同 |
| 倒地恢复（旧主方案） | `RobotLab-Isaac-Recovery-Parallelogram-Rover-v0` | 58 → 12 | 遗留，混合切换中的 `recover` 策略 |
| 其他恢复实验 | `RobotLab-Isaac-Recovery-<变体>-Parallelogram-Rover-v0`，`<变体>` 为 `Getup`、`Settled`、`Grounded`、`Grounded-Progress`、`ControlledTilt`、`FailureReplay`、`FailureReplay-Residual`、`Upside`、`Side-Phase` | 58 / 60 | 遗留的对照或诊断任务 |
| 双轮直立和起立 | `RobotLab-Isaac-Upright-Parallelogram-Rover-v0`<br>`RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0` | 42（critic 47）→ 12 | 遗留，见 [6.5](#ppo-upright) |
| 两轮运动 | `RobotLab-Isaac-Upright-Motion-Parallelogram-Rover-v0`<br>`RobotLab-Isaac-Upright-Motion-Getup-Parallelogram-Rover-v0` | 44（critic 49）→ 12 | 遗留，带地面速度和 yaw 指令，只在平地训练过 |
| SuperMimic 越障 | `RobotLab-Isaac-SuperMimic-Obstacle-Parallelogram-Rover-v0` | 232 → 12 | **当前主线**，由 `train_terrain.py` 使用 |
| SuperMimic 旧环境 | `RobotLab-Isaac-SuperMimic-Flat-Parallelogram-Rover-v0`<br>`RobotLab-Isaac-SuperMimic-Terrain-Preview-Parallelogram-Rover-v0`<br>`RobotLab-Isaac-SuperMimic-Scratch-Stairs-Parallelogram-Rover-v0` | Flat 为 45，其余为 232（旧 scratch 策略多 1 维车道偏移，共 233） | Flat 与 Scratch-Stairs 已无调用方（训练结束后删除）；Terrain-Preview 仍是 `probe_flat.py`（越障验收平地档）的默认环境 |

SuperMimic 任务只注册了环境配置，没有 rsl_rl 配置，所以只能用 `scripts/supermimic/` 下的脚本训练。

### 4.2 上游机器人

| 类别 | 机器人 | 任务 ID（Rough；把 `Rough` 换成 `Flat` 即为平地版） | 截图 |
|---|---|---|---|
| **四足** | [Anymal D](https://www.anybotics.com/robotics/anymal) | `Isaac-Velocity-Flat-Anymal-D-v0`（仅平地，见下方说明） | <img src="./docs/imgs/anymal_d.png" alt="anymal_d" width="75"> |
| | [Unitree Go2](https://www.unitree.com/go2) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 | <img src="./docs/imgs/unitree_go2.png" alt="unitree_go2" width="75"> |
| | [Unitree B2](https://www.unitree.com/b2) | RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0 | <img src="./docs/imgs/unitree_b2.png" alt="unitree_b2" width="75"> |
| | [Unitree A1](https://www.unitree.com/a1) | RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 | <img src="./docs/imgs/unitree_a1.png" alt="unitree_a1" width="75"> |
| | [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="75"> |
| | [Zsibot ZSL1](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0 | <img src="./docs/imgs/zsibot_zsl1.png" alt="zsibot_zsl1" width="75"> |
| | [Magiclab MagicDog](https://www.magiclab.top/dog) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-v0 | <img src="./docs/imgs/magiclab_magicdog.png" alt="magiclab_magicdog" width="75"> |
| | [Agibot D1](https://www.agibot.com/) | RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0 | <img src="./docs/imgs/agibot_d1.png" alt="agibot_d1" width="75"> |
| **轮足** | [Unitree Go2W](https://www.unitree.com/go2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 | <img src="./docs/imgs/unitree_go2w.png" alt="unitree_go2w" width="75"> |
| | [Unitree B2W](https://www.unitree.com/b2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0 | <img src="./docs/imgs/unitree_b2w.png" alt="unitree_b2w" width="75"> |
| | [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="75"> |
| | [DDTRobot Tita](https://directdrive.com/product_TITA) | RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0 | <img src="./docs/imgs/ddtrobot_tita.png" alt="ddtrobot_tita" width="75"> |
| | [Zsibot ZSL1W](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1W-v0 | <img src="./docs/imgs/zsibot_zsl1w.png" alt="zsibot_zsl1w" width="75"> |
| | [Magiclab MagicDog-W](https://www.magiclab.top/dog-w) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-W-v0 | <img src="./docs/imgs/magiclab_magicdog_w.png" alt="magiclab_magicdog_w" width="75"> |
| **人形** | [Unitree G1](https://www.unitree.com/g1) | RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 | <img src="./docs/imgs/unitree_g1.png" alt="unitree_g1" width="75"> |
| | [Unitree H1](https://www.unitree.com/h1) | RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 | <img src="./docs/imgs/unitree_h1.png" alt="unitree_h1" width="75"> |
| | [FFTAI GR1T1](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 | <img src="./docs/imgs/fftai_gr1t1.png" alt="fftai_gr1t1" width="75"> |
| | [FFTAI GR1T2](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0 | <img src="./docs/imgs/fftai_gr1t2.png" alt="fftai_gr1t2" width="75"> |
| | [Booster T1](https://www.boosterobotics.com/) | RobotLab-Isaac-Velocity-Rough-Booster-T1-v0 | <img src="./docs/imgs/booster_t1.png" alt="booster_t1" width="75"> |
| | [RobotEra Xbot](https://www.robotera.com/) | RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0 | <img src="./docs/imgs/robotera_xbot.png" alt="robotera_xbot" width="75"> |
| | [Openloong Loong](https://www.openloong.net/) | RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0 | <img src="./docs/imgs/openloong_loong.png" alt="openloong_loong" width="75"> |
| | [RoboParty ATOM01](https://roboparty.cn/) | RobotLab-Isaac-Velocity-Rough-RoboParty-ATOM01-v0 | <img src="./docs/imgs/roboparty_atom01.png" alt="roboparty_atom01" width="75"> |
| | [Magiclab MagicBot-Gen1](https://www.magiclab.top/human) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Gen1-v0 | <img src="./docs/imgs/magiclab_magicbot_gen1.png" alt="magiclab_magicbot_gen1" width="75"> |
| | [Magiclab MagicBot-Z1](https://www.magiclab.top/z1) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Z1-v0 | <img src="./docs/imgs/magiclab_magicbot_z1.png" alt="magiclab_magicbot_z1" width="75"> |

其他上游任务包括：`RobotLab-Isaac-Velocity-{Flat,Rough}-HandStand-Unitree-A1-v0`、`RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0`、`RobotLab-Isaac-G1-AMP-Dance-Direct-v0`。

> [!NOTE]
> 当前代码中，Anymal D 只注册了 `Isaac-Velocity-Flat-Anymal-D-v0` 和 `Isaac-Velocity-Flat-Anymal-D-Play-v0`，没有 `RobotLab-` 前缀，也没有 Rough 版本。这两个 ID 与 Isaac Lab 自带的同名任务重名，`robot_lab.tasks` 导入后会覆盖 Isaac Lab 的版本。

<a id="generic"></a>

## 5. 通用训练、回放与上游示例

```bash
# RSL-RL（本分支所有 PPO 流程都用它）
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME>

# CusRL（实验性）
python scripts/reinforcement_learning/cusrl/train.py --task=<TASK_NAME> --headless
python scripts/reinforcement_learning/cusrl/play.py --task=<TASK_NAME>

# 零动作和随机动作智能体，用于检查环境配置是否正确
python scripts/tools/zero_agent.py --task=<TASK_NAME>
python scripts/tools/random_agent.py --task=<TASK_NAME>
```

常用参数如下：

- 环境数量：`--num_envs 32`。录像（需要 `ffmpeg`）：`--video --video_length 200`。
- 回放指定 run 或 checkpoint：`--load_run <run 目录名> --checkpoint /PATH/TO/model.pt`。续训：再加 `--resume`。
- `play.py` 会把策略导出到 checkpoint 同级目录的 `exported/policy.pt` 和 `exported/policy.onnx`。
- 回放时用键盘控制**单台**机器人：加 `--keyboard`。按键为：x 方向用 Numpad 8/2 或 ↑/↓，y 方向用 Numpad 4/6 或 →/←，绕 z 旋转用 Numpad 7/9 或 Z/X。
- 单机多卡训练：

  ```bash
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
  ```

  多机训练时，每台机器各启动一个进程。主节点用 `--nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555`，其余节点把 `--node_rank` 改成本机编号，并把 `--rdzv_endpoint` 改成 `<主节点IP>:5555`。端口可以换成任意可用端口。

<details>
<summary>上游示例：BeyondMimic（G1）、AMP、倒立、对称性和蒸馏</summary>

**BeyondMimic（Unitree G1）**：参考动作数据集请遵守各自的原始许可证，格式与 Unitree 数据集的 csv 相同。数据来源包括：[LAFAN1 重定向数据](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)、[KungfuBot](https://kungfu-bot.github.io/) 的侧踢、[ASAP](https://github.com/LeCAR-Lab/ASAP) 的 C 罗庆祝动作、[HuB](https://hub-robot.github.io/) 的平衡动作。

```bash
python scripts/tools/beyondmimic/csv_to_npz.py -f path_to_input.csv --input_fps 60 --headless   # 前向运动学补全位姿/速度/加速度
python scripts/tools/beyondmimic/replay_npz.py -f path_to_motion.npz
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --num_envs 2
```

**AMP 舞蹈（Unitree G1，skrl）**：

```bash
python scripts/reinforcement_learning/skrl/train.py --task=RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --headless
python scripts/reinforcement_learning/skrl/play.py --task=RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --num_envs=32
```

**倒立（Unitree A1）**：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0
```

**Anymal D 对称性数据增强和蒸馏**（任务 ID 见 4.2 的说明）：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=ppo_with_symmetry_data_augmentation agent.algorithm.symmetry_cfg.use_data_augmentation=true
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=ppo_with_symmetry_data_augmentation agent.algorithm.symmetry_cfg.use_data_augmentation=true

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --headless                                     # teacher
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent=rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point
```

</details>

<a id="ppo"></a>

## 6. 四轮车 PPO 流程

本节所有流程都用 rsl_rl，目前都已冻结。保留下来的 checkpoint 用作对照、蒸馏源或部署候选。
需要选择 checkpoint 时，请写**真实的绝对路径**，并在每份评测 JSON 中核对 `checkpoint_sha256`，保证同一组对比用的是同一个文件。
训练 reward 和课程比例都不能代表独立能力，结论一律以评测 JSON 为准。

### 6.1 盲走基线

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-v0 \
  --checkpoint $R/logs/rsl_rl/parallelogram_rover_rough/2026-08-13_17-35-42/model_19999.pt --num_envs 16
```

如果要在 play 之外离线导出 rover checkpoint（不启动 Isaac，会严格校验观测和动作合同），可以用 `scripts/tools/export_rover_policy.py --run-dir <run> --checkpoint <model.pt>`。

<a id="ppo-teacher"></a>

### 6.2 视觉前瞻抬轮与三阶段楼梯（Teacher）

**目标**：保持前进方向，接近障碍后主动抬轮并稳定越过，不训练跳跃。全宽障碍允许左前轮或右前轮任一侧先抬起；左右窄障碍只检查对应一侧。理想的起抬位置是轮胎前缘距障碍约 10 cm（有效范围 8–14 cm）；65 cm 内开始感知，25 cm 内允许接地减载做准备。

**课程**：一个网络、8000 轮的总预算，各环境独立晋级，不按迭代数硬切阶段。每个阶段都同时包含全宽障碍、左右窄障碍、楼梯、方块和少量平地。

| 阶段 | 地形行 | 每级高度 / 级数 | 楼梯抽样占比 |
|---|---|---|---|
| stage1 | 0–2 | 8 cm / 2 级 | 20% |
| stage2 | 3–5 | 12 cm / 3 级 | 40% |
| stage3 | 6–8 | 16 cm / 4 级 | 55% |

晋级需要两次合格的稳定通行，跨 stage 时必须通过本档楼梯，平地不计入晋级。楼梯前有 1.2 m 接近段，踏面 40 cm，顶端有平台；每一级都会重新识别前缘，只给一次动作机会。
`vision_preview.py` 和 `vision_preview_reward.py` 的哈希写在每个 Preview 策略的 `params/vision_contract.json` 中。改动这两个文件后，已有的 Preview 策略都会拒绝评测和续训。

```bash
# 训练（记下打印的时间戳目录）
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --experiment_name $R/logs/vision_preview/teacher --run_name stairs_climb \
  --num_envs 1024 --max_iterations 8000 --seed 42 --headless
TEACHER=$R/logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt   # 现有 Teacher；换成你的

# 楼梯逐档评测：difficulty 0.0/0.5/1.0 = 8/12/16 cm
for d in 0.0 0.5 1.0; do
  python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
    --policy teacher-preview --scenario pyramid_stairs_inv --difficulty $d \
    --checkpoint $TEACHER --num_envs 16 --episodes_per_env 2 --seed 142 \
    --output $R/logs/vision_preview/eval/teacher_stairs_d${d}_seed142_$(date +%Y%m%d).json
done

# 单障碍评测：全宽、左窄、右窄，均为 8 cm
for s in single_step single_step_left single_step_right; do
  python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
    --policy teacher-preview --scenario $s --difficulty 0.0 \
    --checkpoint $TEACHER --num_envs 16 --episodes_per_env 2 --seed 142 \
    --output $R/logs/vision_preview/eval/teacher_${s}_seed142_$(date +%Y%m%d).json
done

# GUI 回放（12 cm 楼梯）
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --checkpoint $TEACHER --num_envs 4 --max_steps 1000 \
  --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5 --preview_debug
```

- 训练中重点看 `Curriculum/terrain_levels/stage_1/2/3`、各档的 `stairs_*cm_fraction`、`Metrics/prelift/route_stable_success_rate`，以及楼梯的机会数和事件数，不要只看 reward。
- 评测 JSON 先确认 `complete=True` 并且回合数满 32，再看 `success_rate`（稳定通行）、`prelift.success_rate`（目标轮提前动作完整并通行）和 `passed`（综合验收）。开发门是 70%，但能滚过障碍不等于学会了提前抬轮。楼梯评测要求真正登顶、保持方向、三轮承重并稳定 0.5 s，目前不单独验收每一级的抬轮时刻。
- `--preview_scenario` 可选 `mixed`、`single_step`、`single_step_left`、`single_step_right`、`pyramid_stairs_inv`、`boxes`。在 1–4 台车的 GUI 回放中，角落会显示实时深度图（Teacher 的面板来自显示相机，策略本身仍读取高度扫描），可以用 `--disable_depth_overlay` 关闭。
- 输出的 `takeoff_lead_m` 是真实离地时轮胎前缘到障碍的距离，目标 8–14 cm；`height_lead_m` 是抬高达标时的距离。两者含义不同，不要混用。

**现有 Teacher 的成绩**（`2026-10-14_11-31-31_stairs_climb/model_7999.pt`，seed 142，每个场景 16 环境 × 2 回合；原始报告目录已清理，需要时用上面的命令重测）：

| 场景 | 稳定通行 | 提前主动动作并通行 | 综合验收 |
|---|---|---|---|
| 全宽 8 cm | 100% | 18.75% | 未通过 |
| 左窄 8 cm | 100% | 0% | 未通过 |
| 右窄 8 cm | 96.9% | 6.25% | 未通过 |
| 楼梯 8 cm × 2 级 | 100% | 不验收 | 通过 |
| 楼梯 12 cm × 3 级 | 75% | 不验收 | 通过 |
| 楼梯 16 cm × 4 级 | 0% | 不验收 | 未通过 |

后来按 64 回合每档的严格楼梯协议复测，8/12/16 cm 的通过率为 100/70/0%。结论是：低障碍通行已有基线，但约 10 cm 处的主动抬轮和 16 cm 楼梯都没有学会。如果目标是这两项能力，应先改进 Teacher（或改用 SuperMimic），不要指望蒸馏补齐 Teacher 本身缺失的技能。

<a id="ppo-student"></a>

### 6.3 Student 蒸馏（DAgger）

Student 的输入是 4141 维：45 维本体和路线命令，加上 8 帧 16×32 的深度历史（4096 维，跨度约 0.7 s）。这里的深度是仿真相机测距经过裁剪和时间堆叠得到的，不是单独训练的深度网络。网络是 MLP（1024/512/256），输出 12 维动作。
训练用 RSL-RL 的 `DistillationRunner` 加 `DaggerDistillation`，Teacher 固定作为动作标签来源。每个回合选择由谁驾驶：Teacher 驾驶的概率在前 4000 轮从 75% 降到 10%，之后保持 10%。每轮 24 步、4 个 epoch，每 4 步累计一次更新；学习率从 5e-4 降到 1e-4。任务会自动开启相机，并把深度合同写入 run 目录下的 `params/depth_contract.json`。

```bash
# 首次训练不加 --resume，此时 --checkpoint 是 Teacher
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint $TEACHER \
  --experiment_name $R/logs/vision_preview/student --run_name stairs_depth \
  --num_envs 128 --max_iterations 8000 --seed 42 --headless          # 显存不足时可以降到 64/32（每轮样本量会随之变化）
STUDENT=$R/logs/vision_preview/student/2026-10-14_19-40-29_stairs_depth/model_7999.pt   # 现有 Student；换成你的

# 单障碍评测（实时深度）
for s in single_step single_step_left single_step_right; do
  python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
    --policy student-preview --scenario $s --difficulty 0.0 --depth_mode live \
    --checkpoint $STUDENT --num_envs 16 --episodes_per_env 2 --seed 142 \
    --output $R/logs/vision_preview/eval/student_${s}_live_seed142_$(date +%Y%m%d).json
done

# 楼梯评测：8/12/16 cm
for d in 0.0 0.5 1.0; do
  python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
    --policy student-preview --scenario pyramid_stairs_inv --difficulty $d --depth_mode live \
    --checkpoint $STUDENT --num_envs 16 --episodes_per_env 2 --seed 142 \
    --output $R/logs/vision_preview/eval/student_stairs_d${d}_live_seed142_$(date +%Y%m%d).json
done

# 深度消融：模型、场景、seed 和回合数保持不变，只把深度换成空白或冻结帧
for m in blank frozen; do
  python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
    --policy student-preview --scenario single_step --difficulty 0.0 --depth_mode $m \
    --checkpoint $STUDENT --num_envs 16 --episodes_per_env 2 --seed 142 \
    --output $R/logs/vision_preview/eval/student_single_step_${m}_seed142_$(date +%Y%m%d).json
done

# GUI 回放（由 Student 独立控制）
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint $STUDENT --num_envs 4 --max_steps 1000 \
  --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5 --preview_debug

# 续训：加载 Student 并加 --resume，会恢复内置 Teacher、优化器和训练进度
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --resume --checkpoint $STUDENT \
  --experiment_name $R/logs/vision_preview/student_resume --run_name stairs_depth_resume \
  --num_envs 128 --max_iterations 1000 --seed 42 --headless
```

- 续训时不能用 Teacher 模型冒充 Student。地形课程状态不保存在 checkpoint 中，续训会从 stage1 重新晋级，策略权重不受影响。合同必须一致，不要手改 JSON 绕过检查。
- 看 TensorBoard 时（`tensorboard --logdir $R/logs/vision_preview/student`）关注 `Loss/behavior`、`Loss/teacher_drive_probability`、`Loss/teacher_driven_fraction`、`Loss/learning_rate`。训练 reward 中包含 Teacher 驾驶的回合，不能代表 Student 的能力。
- 深度消融的判读方法：实时深度成功而空白或冻结深度明显下降，说明策略利用了视觉；三种都成功，说明视觉依赖还没有被证明；三种都失败，先检查 Student 的独立控制能力。

**现有 Student 的成绩**（`2026-10-14_19-40-29_stairs_depth/model_7999.pt`，seed 142，32 回合，报告位于 `logs/vision_preview/eval/student_*_20261015.json`）：

| 场景（8 cm） | 深度 | 稳定通行 | 提前主动动作并通行 | 综合验收 |
|---|---|---|---|---|
| 全宽 | live | 87.5% | 21.9% | 未通过 |
| 左窄 | live | 96.9% | 0% | 未通过 |
| 右窄 | live | 71.9% | 9.4% | 未通过 |
| 全宽 | blank | 0% | 0% | 未通过 |

把深度换成空白后通行率降到 0%，说明 Student 确实依赖深度输入。它和 Teacher 一样没有学会提前抬轮。楼梯档和冻结深度的评测还没有存档。

<a id="ppo-recovery"></a>

### 6.4 侧躺恢复（program31 动作先验 + residual PPO）

目标是**侧躺 → 四轮真实承重 → 稳定保持**。流程为：生成采集源 → 采集 program31 动作 → 用 residual PPO 训练 → 回放和开发评测 → 续训 → 用留出 seed 做最终验收。整个流程不需要 BC，也不依赖历史 checkpoint。
所有产物都放在 `logs/recovery_side_restart_v1/` 下：`source/` 存采集源模型，`demos/` 存动作数据，`policy/` 存策略，`evaluation/` 存评测报告。

```bash
RS=$R/logs/recovery_side_restart_v1
PRIOR="--recovery_prior_data $RS/demos/train.npz --recovery_prior_program_id 31"

# 1. 采集源：Side 任务跑 20 轮，只用来生成合法的 checkpoint、params/ 和控制合同，reward 低是正常的
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0 \
  --experiment_name $RS/source --run_name source_bootstrap --num_envs 1024 --max_iterations 20 --seed 42 --headless
SOURCE=$(find $RS/source -mindepth 2 -maxdepth 2 -type f -path '*_source_bootstrap/model_*.pt' | sort -V | tail -n 1)

# 2. 采集 program31（由脚本程序执行动作，不是源策略）。固定 seed 731、程序编号 31、左右镜像、轮速 30 rad/s
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py --headless \
  --task RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0 --checkpoint "$SOURCE" \
  --skill recovery --scenario side_lying --recovery_protocol settled_ground \
  --diagnostic_action_program --program_seed 731 --program_id 31 --program_mirror --program_wheel_speed 30 \
  --num_envs 64 --episodes_per_env 1 --seed 1942 \
  --output $RS/demos/train.json --demonstration_output $RS/demos/train.npz

# 3. residual PPO 先短训 100 轮（Actor/Critic 分别为 60/72 维，不能加载 58 维的 Side/Grounded/BC 权重）
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  $PRIOR --recovery_residual_profile constant --recovery_residual_std 0.15 \
  --experiment_name $RS/policy --run_name constant --num_envs 1024 --max_iterations 100 --seed 42 --headless

# 4. 回放（需要显示环境，自动选取最新的 *_constant run）
python scripts/reinforcement_learning/rsl_rl/play.py --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --experiment_name $RS/policy --load_run '.*_constant$' $PRIOR --recovery_residual_profile auto --num_envs 8 --max_steps 1250

# 5. 开发评测（seed 3942 和 4051）；7. 最终验收：把 seed 换成 71061 和 81061，num_envs 改为 128，输出前缀改为 holdout
POLICY=$(find $RS/policy -mindepth 2 -maxdepth 2 -type f -path '*_constant/model_*.pt' | sort -V | tail -n 1)
for seed in 3942 4051; do
  python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py --headless \
    --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 --checkpoint "$POLICY" \
    $PRIOR --recovery_residual_profile auto \
    --skill recovery --scenario side_lying --recovery_protocol settled_ground \
    --num_envs 64 --episodes_per_env 1 --seed $seed \
    --output $RS/evaluation/dev_seed${seed}_$(date +%Y%m%d_%H%M%S).json
done

# 6. 续训：每次追加 5000 轮，结果写入新的时间戳目录；完成后重新执行第 4、5 步
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --resume --experiment_name $RS/policy --load_run '.*_constant$' --run_name constant \
  $PRIOR --recovery_residual_profile auto --num_envs 1024 --max_iterations 5000 --seed 42 --headless
```

- 第 2 步的 JSON 中 `passed=false` 是预期的，因为这是动作诊断，不是策略验收。需要检查的是 `complete`、`success_rate` 和 `demonstration_dataset.successful_episodes`。如果成功回合数为 0，先排查动作和物理配置，不要继续往下做。同名的 JSON 和 NPZ 不会被覆盖。
- `recovery_pre_update.pt` 是首次 PPO 更新前的快照，`model_0.pt` 已经过一次更新。续训时不要更换先验、profile、力矩或动作尺度。
- 验收门槛：每个 seed 的报告都要完整，并且 `passed=true`，成功率 ≥85%。成功要求四轮真实承重、没有非轮部件承重、姿态、高度和速度都合格；合格稳态需要至少持续 0.5 s 并一直保持到 5 s 终点，且最晚在 4 s 时开始。最终验收时 checkpoint、数据和参数必须冻结，seed 不能用于调参，两份报告的 `checkpoint_sha256` 必须一致。

**现有结果**：

- `demos/train.json`：program31 跑了 64 回合，成功 48 回合（75%），NPZ 中保存了 48 个回合。
- `policy/2026-09-30_10-37-40_constant/model_1800.pt`：开发 seed 3942 和 4051 均为 64/64。
- `policy/2026-09-30_10-37-40_constant/model_5098.pt`：开发 seed 3942 为 64/64，留出 seed 71061 为 128/128，`passed=true`。留出 seed 81061 还没有报告。

**适用边界**：以上结果只覆盖当前 Isaac Sim 物理配置下的侧躺恢复，不能直接推广到完全倒置（Upside）。URDF 的摆臂、转向节和下连杆存在碰撞覆盖缺口。电机模板 `source/robot_lab/data/Robots/custom/parallelogram_rover/calibration/actuator_profile.template.json` 中的值都是未实测的。部署前需要实现同样的 60 维观测、接管时钟、倒向锁定、分段先验和 tanh 残差合成，再做 sim-to-sim 和实机安全验证；play 导出的 12 维 Actor 输出不是最终的电机目标。

<a id="ppo-upright"></a>

### 6.5 双轮直立、起立与两轮运动（遗留）

目前没有保留任何 checkpoint，日志默认写入 `logs/rsl_rl/parallelogram_rover_upright{,_getup,_motion,_motion_getup}/`。42/47 维任务提供平衡和主动起立；44/49 维的 Motion 任务额外接收地面速度和 yaw 指令，只在平地训练过。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0 --headless
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py --headless \
  --task RobotLab-Isaac-Upright-Getup-Parallelogram-Rover-v0 --skill upright-getup --checkpoint <model.pt>

# 两轮运动：可以用 --upright_warm_start 从 42/47 维的直立权重初始化（这不是 resume）
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Upright-Motion-Getup-Parallelogram-Rover-v0 \
  --upright_warm_start <upright model.pt> --headless
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_motion.py --headless \
  --phase getup --scenario forward --checkpoint <model.pt>   # scenario：stationary/forward/backward/turn-left/turn-right/stop
```

`evaluate_parallelogram_skills.py --skill` 的可选值为 `recovery`、`upright-balance`、`upright-getup`。

### 6.6 三策略混合切换（遗留，仅仿真集成）

`scripts/play_parallelogram_hybrid.py` 在同一台仿真车上按指令表切换三个独立训练的策略：`loco`（四轮行驶，45 维）、`upright`（双轮直立，42 维；或 Motion，44 维）、`recover`（倒地恢复，58 维）。状态机在 `tasks/manager_based/hybrid/planner.py` 中。

```bash
python scripts/play_parallelogram_hybrid.py \
  --loco-checkpoint $R/logs/rsl_rl/parallelogram_rover_rough/2026-08-13_17-35-42/model_19999.pt \
  --upright-checkpoint <Upright-Getup model.pt> --recover-checkpoint <Recovery-Parallelogram-Rover model.pt> \
  --loco-policy blind --upright-policy upright --schedule 0:lo,200:up,600:lo --hybrid-steps 1000
```

`upright` 必须用经过起立阶段训练的模型。当前没有可用的直立 checkpoint，所以需要先完成 6.5 的训练。这个脚本只验证仿真中的策略集成，每个技能都要先单独通过验收。

### 6.7 验收与诊断脚本一览

| 脚本（`scripts/reinforcement_learning/rsl_rl/`） | 用途 |
|---|---|
| `evaluate_parallelogram_vision.py` | 视觉通行评测：固定地形、朝向和命令；支持 blind/teacher/student 以及 SuperMimic 策略（`--supermimic_terrain`、`--scan_student`、`--acceptance_metrics`） |
| `evaluate_parallelogram_skills.py` | 恢复和直立评测；采集 program 动作（`--diagnostic_action_program`） |
| `evaluate_parallelogram_motion.py` | 44 维两轮运动的固定命令验收 |
| `vision_prelift_evaluation.py`、`skill_evaluation.py`、`acceptance_metrics.py` | 上述评测共用的判据和统计模块，不能单独运行 |
| `vision_contract.py`、`recovery_contract.py`、`experiment_lineage.py` | 观测、动作和训练来源的合同校验，防止维数相同但语义不同的模型被静默错配 |
| `motion_trace.py`、`vision_depth_overlay.py` | 运动诊断记录；回放时的深度角落面板 |

<a id="supermimic"></a>

## 7. SuperMimic 越障训练（当前主线）

### 7.1 方法与目标

越障策略完全从零训练，只用 SuperMimic：世界模型加 32 步想象 BPTT，价值梯度全开，想象批量 4096。不使用 PPO Teacher，不做监督初始化，也不加锚定。
目标是：车身不倾斜；遇到台阶时提前抬轮（10/14/17.5 cm 单台阶，分全宽、左窄、右窄，一次只抬一个轮，同轴另一个轮可以滚上去）；能上 8/12/16/17.5 cm 的楼梯（踏面 0.40 m 和 0.28 m），楼梯上轻碰快上；全程平稳；平地直行时偏航和横移都小。
两前轮不能同时离地，楼梯上也不可能每级都让四个轮离地预抬，依据见设计文档 §1.1。
需求门槛、预抬的物理边界、目标函数、地形课程、上一版为什么推倒重来以及每轮训练后的修正，都写在设计文档 [docs/supermimic.md](docs/supermimic.md) 中。代码位于 `source/robot_lab/robot_lab/supermimic/`（`preview_task.py` 为解析目标）和 `tasks/manager_based/supermimic/obstacle_env_cfg.py`、`obstacle_terrain.py`。

### 7.2 训练

```bash
# nohup + setsid 后台运行；参数为 run 名、env 数、轮数。每 1000 轮存档一次，2048 env 实测约 1.3 s/轮
bash scripts/supermimic/run_train.sh <run_name> 2048 40000
tail -f logs/supermimic_terrain/<run_name>.log
tensorboard --logdir logs/supermimic_terrain/<run_name>
kill $(cat logs/supermimic_terrain/<run_name>.pid)        # 停止训练
```

全部目标权重、课程/终止参数、命令与回合长度和优化参数都显式写在 `run_train.sh` 中。如果 run 名已经存在，脚本会拒绝启动。续训需要手动执行：`python scripts/supermimic/train_terrain.py --headless --resume --load_run <run_name>`，后面跟上与 `run_train.sh` 中 `TRAIN_ARGS` 相同的参数。

### 7.3 训练中的过程验收

```bash
nohup bash scripts/supermimic/watch_accept.sh <run_name> 2000 > logs/supermimic_terrain/<run_name>_watch.log 2>&1 &
cat logs/supermimic_terrain/<run_name>/acceptance_progress.md
```

watch 每 2000 轮把新存档导出到 `ckpt_eval/it<N>/`，然后用挑选用 seed（默认 142/143/144，按档轮换，每个存档约 20 分钟）运行 `accept.py`，把结论表追加到 `acceptance_progress.md`。它以最低优先级运行，不会干预训练。已经评测过的存档会跳过，所以可以随时重启。训练退出、并且没有待评存档时，watch 会自动结束。
可以用环境变量调整行为：`ACCEPT_SEEDS`、`ACCEPT_SEED_MODE`（`rotate` 或 `split`）、`ACCEPT_SUITES`（默认 `flat prelift stairs`）、`ACCEPT_POLL_S`、`ACCEPT_IDLE_EXIT_H`。

### 7.4 最终验收

```bash
# 留出 seed 9001–9003（不能用于训练和挑选），约 50–55 分钟
python scripts/supermimic/accept.py --load_run <run_name> --iteration 40000
# 也可以直接指定一个导出的 TorchScript
python scripts/supermimic/accept.py --checkpoint logs/supermimic_terrain/<run_name>/ckpt_eval/it<N>/exported/policy.pt
```

- 加 `--iteration N` 时，结果写入 `logs/supermimic_terrain/<run_name>/ckpt_eval/itN/acceptance/`；不加时验收 `<run_name>/exported/policy.pt`，结果写入 `<run_name>/acceptance/`。输出包括 `summary.md`、`summary.json`、`flat.json` 和 `cases/`。
- 退出码：0 表示 5 项需求全部通过，1 表示有需求未通过，2 表示执行异常（崩溃、超时或 seed 冲突），此时结果不能用来下结论。
- 常用选项：`--suites flat prelift stairs`、`--reuse`（中断后续跑）、`--summarize_only`（不启动 Isaac，只重算汇总）、`--dry_run`（只打印计划）。所有统计都是跨回合或跨环境取 P95，口径见 `scripts/reinforcement_learning/rsl_rl/acceptance_metrics.py`。

回放越障策略（与验收使用同一个 Teacher-Preview 环境和合同校验）：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --supermimic_terrain --load_run <run_name> \
  --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5
```

如果要回放某个中间存档，就把 `--load_run` 换成 `--checkpoint logs/supermimic_terrain/<run_name>/ckpt_eval/it<N>/exported/policy.pt`。这个文件需要先用 `python scripts/supermimic/export_ckpt.py --load_run <run_name> --iterations <N>` 导出。

### 7.5 产物与日志位置

```text
logs/supermimic_terrain/
├── <run>.log  <run>.pid  <run>_watch.log         # 训练日志、训练进程 PID、watch 日志
└── <run>/
    ├── ckpt_XXXXXX.pt  train_log.csv  events.out.tfevents.*
    ├── params/{supermimic_terrain.json, vision_contract.json}
    ├── exported/policy.{pt,onnx}                  # 最近一次存档的导出
    ├── acceptance_progress.md                     # watch 的过程验收结论
    └── ckpt_eval/it<N>/{exported/, params/, acceptance/}
```

<a id="sm-baselines"></a>

### 7.6 对照基线

`obstacle_bumpclimb_20261106/` 的 ckpt_006000 和 ckpt_008000 是上一版奖励（碰了再爬，232 维）。第 6000 轮的过程验收（当时预抬窗口为 8–14 cm）结果是：10 cm 单台阶 98–100%，17.5 cm 正面 36%，8 cm 楼梯 94%，平地通过；严格预抬 0%，台阶横滚 P95 8–19°。

下面两个旧基线是 233 维观测（多一维车道偏移），`accept.py` 和 `play.py` 会自动识别。

| 基线 | 说明 | 旧协议的楼梯结果（8/12/16 cm，64 回合每档） |
|---|---|---|
| `scratch_40k_r8k/ckpt_040000.pt` | 旧的从零训练 | 100 / 100 / 84% |
| `normal_drive_rehearsal_20k_20261103/ckpt_018400.pt` | 楼梯成绩最好，但不会预抬，平地偏航不达标 | 100 / 98 / 89% |
| `normal_drive_rehearsal_20k_20261103/ckpt_020000.pt` | 同上 | 100 / 94 / 78% |

旧协议的逐存档记录在各自目录的 `acceptance_progress.md` 中，`ckpt_eval/it<N>/acceptance/` 里也是旧协议结果。要按当前协议比较，需要用 `accept.py` 重测（导出文件已经存在于 `ckpt_eval/it<N>/` 中），建议用 `--out` 写到别的目录，避免和旧结果混在一起：

```bash
python scripts/supermimic/accept.py --load_run scratch_40k_r8k --iteration 40000
python scripts/supermimic/accept.py --load_run normal_drive_rehearsal_20k_20261103 --iteration 18400
python scripts/supermimic/accept.py --load_run normal_drive_rehearsal_20k_20261103 --iteration 20000
```

### 7.7 风险与待办

完整清单见 [docs/supermimic.md](docs/supermimic.md) 第 5 节。要点如下：

- 三轮支撑裕度、轻碰、横滚超限、车道和镜像的权重是按量级估出来的，需要看训练曲线再校准。
- 三轮支撑的静稳定裕度只有约 2 cm；整宽台阶上同轴两轮谁先抬由策略决定，要看是否两轮都不抬。
- 12 cm 单台阶的几何俯仰约 9.8°，θ_ref 已扣 6°（约 3.8°）；楼梯上 θ_ref 比坡度低 6°，要确认楼梯俯仰门仍满足。
- URDF 左右有 4–8 mm 的不对称，镜像增强只是近似。
- 仿真中摆臂执行器偏软（刚度 20 N·m/rad，力矩上限 10 N·m），需要和实机核对。
- 深度学生要等越障教师过门之后再适配。

<a id="sm-other"></a>

### 7.8 其他 SuperMimic 脚本

| 脚本（`scripts/supermimic/`） | 状态 | 用途 |
|---|---|---|
| `export_ckpt.py` | 在用 | 把指定存档导出到 `ckpt_eval/it<N>/`，watch 和 accept 会自动调用 |
| `probe_flat.py` | 在用 | `accept.py` 的平地档，也可以单独做平地探针 |
| `bench_iteration.py` | 辅助 | 测单轮训练耗时，做优化的 A/B 对比 |
| `train_scan_student.py` | 待适配 | 深度 → 高度扫描学生，等越障教师过门之后再重新适配 |

<a id="tools"></a>

## 8. 工具与调试

### 8.1 scripts/tools

| 脚本 | 用途 |
|---|---|
| `list_envs.py`、`zero_agent.py`、`random_agent.py` | 列出任务；用零动作或随机动作检查环境 |
| `convert_urdf.py`、`convert_mjcf.py` | URDF/MJCF 转 USD（上游） |
| `generate_parallelogram_mjcf.py` | 由四轮车 URDF 生成 MuJoCo MJCF，显式写出浮动基座、执行器和 mimic |
| `export_rover_policy.py` | 离线导出 rover checkpoint 为 TorchScript，并校验合同 |
| `probe_parallelogram_support.py`、`probe_recovery_geometry.py`、`probe_recovery_settling.py`、`probe_recovery_takeover.py` | 接触、几何、沉降和接管过程的物理探针（诊断用，不能代替验收） |
| `beyondmimic/csv_to_npz.py`、`beyondmimic/replay_npz.py` | BeyondMimic 动作数据转换和回放（上游） |
| `clean_trash.py` | 清理工具（上游） |

### 8.2 离线测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest source/robot_lab/test -q -p no:cacheprovider
```

关闭插件自动加载是为了避开机器上 ROS 的 launch_testing 插件与 Isaac Lab Python 环境不兼容的问题，项目测试本身不会被跳过。离线测试只检查观测、合同、先验和评测逻辑，不能代替真实的仿真回合。

### 8.3 Tensorboard

```bash
tensorboard --logdir=logs                                  # 全部
tensorboard --logdir=logs/supermimic_terrain/<run_name>    # 单个 SuperMimic run
```

### 8.4 代码格式

```bash
pip install pre-commit
pre-commit run --all-files
```

### 8.5 常见问题

**Pylance 缺少扩展索引**：在 `.vscode/settings.json` 的 `"python.analysis.extraPaths"` 中加入扩展路径，并把 `<path-to-isaac-lab>` 换成你的 Isaac Lab 路径：

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/robot_lab",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
```

**清理 USD 缓存**：仿真时会在 `/tmp/IsaacLab/usd_{date}_{time}_{random}` 生成临时 USD 文件，占用空间较大，可以用以下命令清理：

```bash
rm -rf /tmp/IsaacLab/usd_*
```

<a id="add-robot"></a>

## 9. 添加新机器人

环境遵循 Gym 注册机制，任务名的格式为 `RobotLab-Isaac-<Task>-<Robot>-v<X>`，其中 `<X>` 表示观测或动作空间的版本。环境用 `configclass` 装饰的 Python 类配置，基础任务配置与各机器人的实例目录放在一起：

```tree
source/robot_lab/robot_lab/assets/
├── __init__.py
└── unitree.py  # <- 定义机器人资产

source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/
├── config
│   └── quadruped/unitree_a1
│       ├── agents  # <- 学习算法配置
│       ├── __init__.py  # <- 向 gym 注册环境
│       ├── flat_env_cfg.py
│       └── rough_env_cfg.py
├── __init__.py
└── velocity_env_cfg.py  # <- 基础任务配置
```

注册写在 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_a1/__init__.py` 中：

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1FlatTrainerCfg",
    },
)
```

Rough 版本的注册方式相同，只需把 `Flat` 换成 `Rough`。本分支的四轮车任务族（`recovery/`、`upright/`、`supermimic/`）不继承 locomotion/velocity 的基类，在各自的 `__init__.py` 中独立注册。

<a id="credits"></a>

## 10. 引用、致谢与许可证

本项目采用 Apache-2.0 许可证（见 [LICENSE](LICENSE)），贡献者见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。使用本代码时请引用上游项目：

```
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```

上游项目使用了以下开源仓库的部分代码：

- [linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)

SuperMimic 部分参考了 `zita-ch/supermimic_vibed`（MIT）、`Zhengtianjin/go2-supermimic`（MIT）和 `Motphys/Uni-SuperMimic`（Apache-2.0）。
