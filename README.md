# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

The table below lists all available environments:

| Category   | Robot Model         | Environment Name (<ENV_NAME>)                                      | Screenshot |
|------------|---------------------|------------------------------------------------------------|------------|
| **Quadruped** | [Anymal D](https://www.anybotics.com/robotics/anymal) | RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 | <img src="./docs/imgs/anymal_d.png" alt="anymal_d" width="75"> |
|            | [Unitree Go2](https://www.unitree.com/go2) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 | <img src="./docs/imgs/unitree_go2.png" alt="unitree_go2" width="75"> |
|            | [Unitree B2](https://www.unitree.com/b2) | RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0 | <img src="./docs/imgs/unitree_b2.png" alt="unitree_b2" width="75"> |
|            | [Unitree A1](https://www.unitree.com/a1) | RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 | <img src="./docs/imgs/unitree_a1.png" alt="unitree_a1" width="75"> |
|            | [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="75"> |
|            | [Zsibot ZSL1](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0 | <img src="./docs/imgs/zsibot_zsl1.png" alt="zsibot_zsl1" width="75"> |
|            | [Magiclab MagicDog](https://www.magiclab.top/dog) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-v0 | <img src="./docs/imgs/magiclab_magicdog.png" alt="magiclab_magicdog" width="75"> |
|            | [Agibot D1](https://www.agibot.com/) | RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0 | <img src="./docs/imgs/agibot_d1.png" alt="magiclab_magicdog" width="75">  |
| **Wheeled** | [Unitree Go2W](https://www.unitree.com/go2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 | <img src="./docs/imgs/unitree_go2w.png" alt="unitree_go2w" width="75"> |
|            | [Unitree B2W](https://www.unitree.com/b2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0 | <img src="./docs/imgs/unitree_b2w.png" alt="unitree_b2w" width="75"> |
|            | [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="75"> |
|            | [DDTRobot Tita](https://directdrive.com/product_TITA) | RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0 | <img src="./docs/imgs/ddtrobot_tita.png" alt="ddtrobot_tita" width="75"> |
|            | [Zsibot ZSL1W](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1W-v0 | <img src="./docs/imgs/zsibot_zsl1w.png" alt="zsibot_zsl1w" width="75"> |
|            | [Magiclab MagicDog-W](https://www.magiclab.top/dog-w) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-W-v0 | <img src="./docs/imgs/magiclab_magicdog_w.png" alt="magiclab_magicdog_w" width="75"> |
| **Humanoid** | [Unitree G1](https://www.unitree.com/g1) | RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 | <img src="./docs/imgs/unitree_g1.png" alt="unitree_g1" width="75"> |
|             | [Unitree H1](https://www.unitree.com/h1) | RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 | <img src="./docs/imgs/unitree_h1.png" alt="unitree_h1" width="75"> |
|             | [FFTAI GR1T1](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 | <img src="./docs/imgs/fftai_gr1t1.png" alt="fftai_gr1t1" width="75"> |
|             | [FFTAI GR1T2](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0 | <img src="./docs/imgs/fftai_gr1t2.png" alt="fftai_gr1t2" width="75"> |
|             | [Booster T1](https://www.boosterobotics.com/) | RobotLab-Isaac-Velocity-Rough-Booster-T1-v0 | <img src="./docs/imgs/booster_t1.png" alt="booster_t1" width="75"> |
|             | [RobotEra Xbot](https://www.robotera.com/) | RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0 | <img src="./docs/imgs/robotera_xbot.png" alt="robotera_xbot" width="75"> |
|             | [Openloong Loong](https://www.openloong.net/) | RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0 | <img src="./docs/imgs/openloong_loong.png" alt="openloong_loong" width="75"> |
|             | [RoboParty ATOM01](https://roboparty.cn/) | RobotLab-Isaac-Velocity-Rough-RoboParty-ATOM01-v0 | <img src="./docs/imgs/roboparty_atom01.png" alt="roboparty_atom01" width="75"> |
|             | [Magiclab MagicBot-Gen1](https://www.magiclab.top/human) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Gen1-v0 | <img src="./docs/imgs/magiclab_magicbot_gen1.png" alt="magiclab_magicbot_gen1" width="75"> |
|             | [Magiclab MagicBot-Z1](https://www.magiclab.top/z1) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Z1-v0 | <img src="./docs/imgs/magiclab_magicbot_z1.png" alt="magiclab_magicbot_z1" width="75"> |

> [!NOTE]
> If you want to run policy in gazebo or real robot, please use [rl_sar](https://github.com/fan-ziqi/rl_sar) project.
>
> Discuss in [Github Discussion](https://github.com/fan-ziqi/robot_lab/discussions) or [Discord](http://www.robotsfan.com/dc_robot_lab).

## Version Dependency

| robot_lab Version | Isaac Lab Version             | Isaac Sim Version         |
|------------------ | ----------------------------- | ------------------------- |
| `main` branch     | `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.2`          | `v2.3.2`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.2`          | `v2.2.1`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.1`          | `v2.1.1`                      | Isaac Sim 4.5             |
| `v1.1`            | `v1.4.1`                      | Isaac Sim 4.2             |

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/fan-ziqi/robot_lab.git
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/robot_lab
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>Set up IDE (Optional, click to expand)</summary>

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

</details>

<details>

<summary>Setup as Omniverse Extension (Optional, click to expand)</summary>

We provide an example UI extension that will load upon enabling your extension defined in `source/robot_lab/robot_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `robot_lab/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Docker setup

<details>

<summary>Click to expand</summary>

### Building Isaac Lab Base Image

Currently, we don't have the Docker for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html).

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building robot_lab Image

Following above, you can build the docker container for this project. It is called `robot-lab`. However,
you can modify this name inside the [`docker/docker-compose.yaml`](docker/docker-compose.yaml).

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
```

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# robot-lab                        latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### Running the container

After building, the usual next step is to start the containers associated with your services. You can do this with:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up
```

This will start the services defined in your `docker-compose.yaml` file, including robot-lab.

If you want to run it in detached mode (in the background), use:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} robot-lab /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

</details>

## Try examples

You can use the following commands to run all environments:

RSL-RL:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME>
```

CusRL (**Experimental**):

```bash
# Train
python scripts/reinforcement_learning/cusrl/train.py --task=<TASK_NAME> --headless

# Play
python scripts/reinforcement_learning/cusrl/play.py --task=<TASK_NAME>
```

Running a task with dummy agents (These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly):

```bash
# Zero-action agent
python scripts/tools/zero_agent.py --task=<TASK_NAME>
# Random-action agent
python scripts/tools/random_agent.py --task=<TASK_NAME>
```

BeyondMimic for Unitree G1:

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of Unitree's dataset

  - Unitree-retargeted LAFAN1 Dataset is available
    on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
  - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
  - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
  - Balance motions are from [HuB](https://hub-robot.github.io/)

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body acceleration) via forward kinematics

  ```bash
  python scripts/tools/beyondmimic/csv_to_npz.py -f path_to_input.csv --input_fps 60 --headless
  ```

- Replaying the motion in Isaac Sim:

  ```bash
  python scripts/tools/beyondmimic/replay_npz.py -f path_to_motion.npz
  ```

- Training and Evaluation

  ```bash
  # Train
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --headless

  # Play
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --num_envs 2
  ```

<a id="rover-vision-preview"></a>

### 平行四边形轮式车：视觉前瞻抬轮与楼梯攀爬（实验性）

当前目标是保持前进方向、接近障碍后主动抬轮并稳定越过。全宽允许左或右前轮先主动抬起，另一轮主动或被动滚上；左右窄障碍仍检查对应侧，不固定左先，不训练跳跃。真正起抬偏好为轮胎前缘距障碍约10cm（有效8–14cm）；65cm用于感知，25cm内允许接地减载准备。

Teacher为232维高度扫描策略；Student为4141维本体信息＋深度历史策略。这里的深度是仿真相机测距经过裁剪和时间堆叠，不是单独训练的深度网络。相机、0.7秒历史、合同号和仍在注册的旧任务见 [深度观测入口](docs/vision_preview_workflow.md#depth-observation)。流程为Teacher PPO → 单障碍动作验收和8/12/16cm楼梯验收 → Student DAgger动作蒸馏 → 独立Student评测与深度消融。一个Teacher网络自动训练多个阶段，8000轮是总预算。

| 阶段 | 地形行 | 每级高度／级数 | 楼梯抽样占比 |
|---|---|---|---|
| stage1 | 0–2 | 8cm／2级 | 20% |
| stage2 | 3–5 | 12cm／3级 | 40% |
| stage3 | 6–8 | 16cm／4级 | 55% |

每阶段同时包含全宽、左右窄障碍、楼梯、方块及少量平地。晋级要求两次合格的稳定通行；跨stage必须通过本档楼梯，平地不晋级。全宽不再用双侧主动动作卡住课程。楼梯有1.2m接近段、40cm踏面及顶端平台，每一级重新识别前缘并提供一次动作机会，禁止同阶跺脚重复领奖。

当前课程、奖励和终止条件与旧合同不同，旧合同模型需要重新训练；保留原模型与报告。2026-10-14已完成当前合同的8000轮Teacher：8cm楼梯通行100%、12cm为75%、16cm为0%，单障碍提前主动动作仍未达标。16cm仍是探索目标。训练继续使用原来的Preview任务名，不增加功能版本。

#### 完整流程总览：Teacher → Student → 回放

所有命令在 `/media/bigdisk/robot_lab` 下、已能启动 Isaac Lab 的终端执行（`conda activate env_isaaclab; unset PYTHONPATH; source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh`）。本节两步都用 rsl_rl：Teacher 是 PPO，Student 是 DAgger 动作蒸馏。

| 步骤 | 做什么 | 产物 |
|---|---|---|
| 1. Teacher | 232 维高度扫描策略，8/12/16 cm 课程，PPO 8000 轮 | `logs/vision_preview/teacher/<时间戳>_stairs_climb/model_7999.pt` |
| 2. Student | 4141 维深度历史策略，DAgger 蒸馏 Teacher 8000 轮 | `logs/vision_preview/student/<时间戳>_stairs_depth/model_7999.pt` |
| 3. 验收 / 回放 | 楼梯逐档评测、单障碍评测、深度消融、GUI 回放 | `logs/vision_preview/eval/*.json` |

```bash
# 1. Teacher：记下训练打印的时间戳目录
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/teacher --run_name stairs_climb \
  --num_envs 1024 --max_iterations 8000 --seed 42 --headless
TEACHER=logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt   # 换成你的
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
  --policy teacher-preview --scenario pyramid_stairs_inv --difficulty 0.5 \
  --checkpoint $TEACHER --num_envs 16 --episodes_per_env 2 --seed 142
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --checkpoint $TEACHER --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5

# 2. Student：首次不加 --resume（--checkpoint 是 Teacher）
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint $TEACHER \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/student --run_name stairs_depth \
  --num_envs 128 --max_iterations 8000 --seed 42 --headless
STUDENT=logs/vision_preview/student/<时间戳>_stairs_depth/model_7999.pt              # 换成你的

# 3. Student 验收与回放（difficulty 0.0/0.5/1.0 = 8/12/16 cm）
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
  --policy student-preview --scenario pyramid_stairs_inv --difficulty 0.5 --depth_mode live \
  --checkpoint $STUDENT --num_envs 16 --episodes_per_env 2 --seed 142
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint $STUDENT --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5
```

- 当前 Teacher 楼梯成绩（64 回合/档）：8/12/16 cm 为 100% / 70% / 0%。
- 评测脚本退出码 1 表示"跑完但未达验收门"，2 才是执行出错。
- 每一步的细节（日志判读、单障碍与深度消融、续训）见下文 Teacher / Student 小节。
- 想在 Teacher 基础上继续提升楼梯能力，见独立的 [SuperMimic 越障优化一节](#rover-supermimic)，它不改动本节的任何训练与产物。

先在已能启动Isaac Lab的终端执行Teacher训练：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/teacher \
  --run_name stairs_climb \
  --num_envs 1024 --max_iterations 8000 --seed 42 --headless
```

训练启动时记录打印的时间戳目录。检查 `Curriculum/terrain_levels/stage_1/2/3`、各档 `stairs_*cm_fraction`、`Metrics/prelift/route_stable_success_rate` 和楼梯机会／事件数，不能只看reward。各环境独立晋级，不按固定迭代数硬切阶段。

8000轮完成后，下方表达式只选择本次 `*_stairs_climb/model_7999.pt`，不会选旧 `active_prelift` 模型；没有文件时先等训练完成。正式对比请将表达式替换为选定的真实绝对路径，保证所有报告使用同一checkpoint和SHA256。

按8、12、16cm依次评测：

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

可视化12cm楼梯；查看8cm或16cm时将difficulty改为0.0或1.0：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/teacher -mindepth 2 -maxdepth 2 -type f -path '*_stairs_climb/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 4 --max_steps 1000 \
  --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5 --preview_debug
```

1–4台车的GUI回放自动显示各车实时深度图；Teacher面板是显示相机，Teacher策略仍读取高度扫描。大于4台或headless不显示角落面板。楼梯评测要求真正登顶、保持方向、三轮承重且稳定0.5秒；70%为开发门，不是硬件安全或16cm能力保证。

#### Student：训练、独立评测与续训

Student接收4141维输入：45维本体／路线命令＋8帧16×32深度历史（4096维，约0.7秒跨度），网络为MLP，隐藏层1024／512／256，输出12维动作。Teacher固定为监督来源，通过RSL-RL `DistillationRunner`＋`DaggerDistillation`训练Student；本流程没有CNN＋GRU、共享Actor或同轮交替PPO，也不需要恢复示范、AMP或单独BC初始化。另一条深度扫描学生属于 SuperMimic 实验，见 [SuperMimic 越障优化一节](#rover-supermimic) 的「其他可选流程」。

**1. 确认Teacher能力，再开始Student训练。** 当前Teacher可用于低障碍通行基线实验；如果目标是约10cm前主动抬轮及16cm楼梯，先继续改善Teacher，不能把动作蒸馏当成补足缺失技能的方法。以下使用已经评测过的明确Teacher路径：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint /media/bigdisk/robot_lab/logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/student \
  --run_name stairs_depth \
  --num_envs 128 --max_iterations 8000 --seed 42 --headless
```

首次Student训练不加 `--resume`，此时checkpoint是Teacher。任务自动开启相机，并把预处理合同写到该run的 `params/depth_contract.json`；显存不足可将128降为64／32，但每轮样本量会改变。每台车每回合选择驱动者，Teacher驱动概率前4000轮从75%降到10%，之后保留10%；动作标签始终由同一状态的固定Teacher提供。每轮24步、4个epoch、每4步累计更新一次，学习率在8000轮内从5e−4降至1e−4。训练reward和课程比例含Teacher驱动回合，不能当作Student独立能力。

**2. 选择Student模型，再分别评测全宽、左右窄障碍。** 以下模型表达式只选择跑满8000轮后的 `*_stairs_depth/model_7999.pt`。没有文件时不要执行；中途评测请填写已保存模型的真实绝对路径。多个run时最后时间戳不代表最佳模型，同一组评测固定同一文件并核对 `checkpoint_sha256`。

全宽8cm：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_full_live_seed142.json
```

左侧窄障碍8cm：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step_left --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_left_live_seed142.json
```

右侧窄障碍8cm：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario single_step_right --difficulty 0.0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_right_live_seed142.json
```

先看 `complete=True` 和32回合，再看 `success_rate`（稳定通行）、`prelift.success_rate`（目标轮提前动作完整且通行）及 `passed`（综合验收）。当前开发门槛70%；能滚过障碍不等于已学会提前抬轮。评测正常完成但未达技能门时，脚本退出码为1，需看JSON判断，不能只据此认定运行异常。

**3. 楼梯逐档评测。** 下例为12cm×3级；8cm／16cm按表修改难度和输出文件名，分别运行：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py \
  --policy student-preview --scenario pyramid_stairs_inv --difficulty 0.5 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 16 --episodes_per_env 2 --seed 142 --headless --depth_mode live \
  --output /media/bigdisk/robot_lab/logs/vision_preview/eval/student_stairs12_live_seed142.json
```

| 楼梯 | `--difficulty` | 输出文件名（同一eval目录） |
|---|---|---|
| 8cm×2级 | `0.0` | `student_stairs8_live_seed142.json` |
| 12cm×3级 | `0.5` | `student_stairs12_live_seed142.json` |
| 16cm×4级 | `1.0` | `student_stairs16_live_seed142.json` |

楼梯评测验证真正登顶与稳定性，暂未独立验收每一级的提前抬轮时刻。补测seed314／942时同时修改输出文件名，保留每次报告。

**4. 实时／空白／冻结深度对照。** 第2步全宽 `live` 为基线；保持同一模型、场景、难度、seed和回合数，再执行：

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


实时成功而空白／冻结明显下降，支持策略利用视觉；三种都成功时视觉依赖尚未证明，可能存在固定动作或本体反馈；三种都失败时先检查Student独立控制能力。固定场景和异常深度输入也有判读限制，需在左右窄障碍及楼梯上重复对照、增加seed，不能仅凭深度截图宣称视觉决策完成。

**5. Student回放。** 下例展示12cm楼梯；单障碍可将scenario改为 `single_step`、`single_step_left` 或 `single_step_right`，difficulty改为0.0：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --num_envs 4 --max_steps 1000 \
  --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5 --preview_debug
```

play／eval均由Student独立控制。GUI下1–4台车显示角落深度面板；原始画面不等于裁剪后的8帧网络输入。`takeoff_lead_m`为真实离地时的轮胎前缘距离，目标8–14cm；`height_lead_m`为抬高达标时的距离，不能混用。逐步打印较稀疏，动作时刻以完整评测时序为准。此次GUI面板未实测。

**6. Student续训。** 完成8000轮后追加1000轮；加载Student checkpoint并加 `--resume`：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --resume \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/vision_preview/student -mindepth 2 -maxdepth 2 -type f -path '*_stairs_depth/model_7999.pt' -print | sort | tail -n 1)" \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/student_resume \
  --run_name stairs_depth_resume \
  --num_envs 128 --max_iterations 1000 --seed 42 --headless
```

恢复Student、内置Teacher、优化器和训练进度；不能再传Teacher模型充当Student续训。后续评测改用 `student_resume` 下的新模型，不能继续搜索首轮目录。当前地形课程状态不保存在checkpoint中，重建环境从stage1重新晋级，策略权重仍保留。合同必须一致，不修改JSON绕过检查。

**7. 日志与当前验证边界。** 查看Student日志：

```bash
tensorboard --logdir /media/bigdisk/robot_lab/logs/vision_preview/student
```

关注 `Loss/behavior`、`Loss/teacher_drive_probability`、`Loss/teacher_driven_fraction`、`Loss/learning_rate` 和 `Curriculum/terrain_levels/stage_1/2/3`。模仿损失下降表示动作更接近Teacher；混合训练reward上涨不等于Student独立成功。最终以分场景评测及视觉对照为准。

2026-10-14当前Teacher `2026-10-14_11-31-31_stairs_climb/model_7999.pt` 的固定场景评测（seed142，每场景16环境×2回合）：

| 场景 | 稳定通行率 | 提前主动动作并通行 | 综合验收 |
|---|---|---|---|
| 全宽8cm | 100% | 18.75% | 未通过 |
| 左窄8cm | 100% | 0% | 未通过 |
| 右窄8cm | 96.875% | 6.25% | 未通过 |
| 8cm×2级楼梯 | 100% | 未单独验收逐级提前动作 | 通过通行门 |
| 12cm×3级楼梯 | 75% | 未单独验收逐级提前动作 | 通过通行门 |
| 16cm×4级楼梯 | 0% | 未单独验收逐级提前动作 | 未通过 |

以上6份JSON均完整且使用同一Teacher SHA256，报告目录为 `/media/bigdisk/robot_lab/logs/vision_preview_diagnostics/final_7999_review.TbdtwP/`。这说明低障碍通行已有基线，约10cm主动抬轮和16cm楼梯仍未完成，单seed的32回合不能代表部署可靠性。

Student本次193项相关测试通过，4环境新训2轮＋续训1轮正常，Teacher权重保持不变；实时／空白／冻结各完成4回合，通行率均0%，无非有限数回合。短训只证明输入、监督、续训和评测链路能运行，尚无全量Student成绩或视觉依赖结论。相机探针确认被测全宽场景能看到障碍，未证明所有场景可见或策略实际利用视觉。证据目录为 `/media/bigdisk/robot_lab/logs/vision_preview_diagnostics/student_access.T8MpL4/`；Student全量训练、GUI面板、实车迁移仍需验证。


### 平行四边形轮式车：从零训练侧躺恢复（实验性）

本流程按**旧模型、旧示范和旧日志都已删除**编写，不依赖任何历史 checkpoint，
也不需要先运行 BC。目标是 **侧躺 → 四轮承重 → 稳定保持**，路线为：

**创建采集源模型 → 生成 program31 动作数据 → 校验数据 → residual PPO 短训 → 回放/评估 → 续训 → 最终验收。**

以下命令在已经配置好的 Isaac Lab Python 环境中运行，所有项目和产物路径均已填全。
控制原理、当前实测结果及部署限制见 [侧躺恢复说明](docs/recovery_side_workflow.md)。

#### 1. 从零生成采集所需的 Side 源模型

不加 `--resume`，不提供任何旧 checkpoint，先运行 20 轮：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0 \
  --experiment_name /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/source \
  --run_name source_bootstrap \
  --num_envs 1024 --max_iterations 20 --seed 42 --headless
```

本步用于生成合法 checkpoint 以及同级 `params/agent.yaml`、`params/env.yaml` 和恢复控制配置。
**这里 reward 低不表示后续脚本采集失败；不要把这 20 轮当作最终恢复训练。**
保留整个运行目录，不能只保留 `.pt`。

等命令正常结束后，确认本轮确实生成了模型：

```bash
find /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/source \
  -mindepth 2 -maxdepth 2 -type f \
  -path '*_source_bootstrap/model_*.pt' -print | sort -V
```

必须看到真实文件路径再进行下一步。下方采集命令会自动选择这里最新运行的最新模型。

#### 2. 重新生成 program31 动作先验

本步加载刚生成的源模型来建立采集环境，但动作由**脚本程序**执行，而不是源策略。
固定程序生成 seed 731、候选编号 31、左右镜像和轮速幅值 30 rad/s；不要随意改成程序 7。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py \
  --task RobotLab-Isaac-Recovery-Side-Parallelogram-Rover-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/source -mindepth 2 -maxdepth 2 -type f -path '*_source_bootstrap/model_*.pt' -print | sort -V | tail -n 1)" \
  --skill recovery --scenario side_lying --recovery_protocol settled_ground \
  --diagnostic_action_program --program_seed 731 --program_id 31 \
  --program_mirror --program_wheel_speed 30 \
  --num_envs 64 --episodes_per_env 1 --seed 1942 \
  --output /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.json \
  --demonstration_output /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --headless
```

预期产物为 `/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.json`
和 `/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz`。
NPZ 只接受完整的成功脚本回合，**本次能生成多少成功回合必须实际测量，不沿用以前的数量或成功率**。

这是动作诊断，JSON 的 `passed=false` 是预期标记，不代表已经执行过策略验收。
检查 `complete`、`success_rate`、`demonstration_dataset.successful_episodes`。
如果成功回合为 0，或 NPZ 没生成，先停在这里排查动作/物理配置，不要继续 BC 或 PPO。
重复采集不会覆盖同名 JSON/NPZ；确认数据有效后进入第 3 节，不必重复本步。

#### 3. 从零训练 residual PPO：先跑 100 轮

这里不加载第 1 节的源模型权重，只使用第 2 节新生成的 program31 动作先验。
任务为 `Side-Residual`，Actor / Critic 观测分别是 **60 / 72 维**，输出 **12 维残差动作**。
60 维是在原 58 维上追加接管阶段和初始倒向，72 维则保留训练用特权信息。
它不能直接加载 58 维 Side/Grounded/BC 权重。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile constant \
  --recovery_residual_std 0.15 \
  --experiment_name /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy \
  --run_name constant \
  --num_envs 1024 --max_iterations 100 --seed 42 --headless
```

本步正常结束后，会在 `/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy/`
下生成新的 `时间戳_constant` 运行目录。
`recovery_pre_update.pt` 是首次 PPO 更新前的初始化快照，`model_0.pt` 已经过第一次更新。
回放和评估只自动选择 `model_*.pt`，不猜测末尾编号。

#### 4. 回放刚训练的模型

无需填写新时间戳，play 自动读取本轮 policy 目录中最新的 `constant` 运行：

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --experiment_name /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy \
  --load_run '.*_constant$' \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile auto \
  --num_envs 8 --max_steps 1250
```

本命令不带 `--headless`，需要可用显示环境；只想做无窗口运行检查时才追加该参数。
核对终端 `Loading model checkpoint from:` 打印的路径属于本轮 `policy/`。
看完整的“侧躺、翻转、四轮支撑、稳定保持”，不能只凭某一帧翻正判定成功。

#### 5. 在两个开发 seed 上评估当前模型

先等训练结束，评估期间不要启动新的训练或修改模型。下面自动选择本轮同一个最新 checkpoint，
输出文件自动带时间戳；同一秒内不要重复启动同一 seed 的评估。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy -mindepth 2 -maxdepth 2 -type f -path '*_constant/model_*.pt' -print | sort -V | tail -n 1)" \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile auto \
  --skill recovery --scenario side_lying --recovery_protocol settled_ground \
  --num_envs 64 --episodes_per_env 1 --seed 3942 \
  --output "/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/evaluation/dev_seed3942_$(date +%Y%m%d_%H%M%S).json" --headless

python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy -mindepth 2 -maxdepth 2 -type f -path '*_constant/model_*.pt' -print | sort -V | tail -n 1)" \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile auto \
  --skill recovery --scenario side_lying --recovery_protocol settled_ground \
  --num_envs 64 --episodes_per_env 1 --seed 4051 \
  --output "/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/evaluation/dev_seed4051_$(date +%Y%m%d_%H%M%S).json" --headless
```

查看 JSON 的 `checkpoint`、`checkpoint_sha256`、`complete`、`success_rate`、`passed`
和 `episode_results` 中的 `failure_state`；两份报告的模型哈希应相同。
开发评估没有达到 85% 不等于脚本运行故障；退出码 1 且报告完整表示技能未达标。
traceback、不完整回合或 NaN 才需要先排查运行问题。

100 轮只是短训检查，不要求必然达到最终门槛；但若一直 0 成功或明显破坏脚本动作，
先停下排查，不能靠继续增加轮数掩盖问题。

#### 6. 继续本轮训练：每次追加 5000 轮

短训、回放和开发评估完成后，用 `--resume` 接着**本轮最新模型**训练。
不要重复第 3 节，否则会另建一个从零开始的策略。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --resume \
  --experiment_name /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy \
  --load_run '.*_constant$' --run_name constant \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile auto \
  --num_envs 1024 --max_iterations 5000 --seed 42 --headless
```

`--max_iterations 5000` 表示本次追加预算，续训保存到新的时间戳运行目录，旧运行不被覆盖。
每次完成后重新执行 **第 4 节回放 + 第 5 节开发评估**；再需要训练就重复本节。
保留同一个 `train.npz` 和 profile，不要在续训中替换先验、扩大力矩或更换动作尺度。
没有 NaN 不代表已学会恢复，mean reward 上升也不能代替真实成功率。

#### 7. 冻结模型后的最终验收

开发结果满足目标后停止训练，不再更改 checkpoint、数据和参数，再做最终测试。
下面给出 seed 71061 / 81061；如果你已经用它们调参，应换成其他未使用的 seed。
评估期间不要新建 `constant` 运行，并核对两份报告的 `checkpoint_sha256` 相同。

```bash
python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy -mindepth 2 -maxdepth 2 -type f -path '*_constant/model_*.pt' -print | sort -V | tail -n 1)" \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile auto \
  --skill recovery --scenario side_lying --recovery_protocol settled_ground \
  --num_envs 128 --episodes_per_env 1 --seed 71061 \
  --output "/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/evaluation/holdout_seed71061_$(date +%Y%m%d_%H%M%S).json" --headless

python /media/bigdisk/robot_lab/scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_skills.py \
  --task RobotLab-Isaac-Recovery-Side-Residual-Parallelogram-Rover-v0 \
  --checkpoint "$(find /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/policy -mindepth 2 -maxdepth 2 -type f -path '*_constant/model_*.pt' -print | sort -V | tail -n 1)" \
  --recovery_prior_data /media/bigdisk/robot_lab/logs/recovery_side_restart_v1/demos/train.npz \
  --recovery_prior_program_id 31 --recovery_residual_profile auto \
  --skill recovery --scenario side_lying --recovery_protocol settled_ground \
  --num_envs 128 --episodes_per_env 1 --seed 81061 \
  --output "/media/bigdisk/robot_lab/logs/recovery_side_restart_v1/evaluation/holdout_seed81061_$(date +%Y%m%d_%H%M%S).json" --headless
```

每个 seed 都需报告完整、`passed=true`，当前侧躺恢复成功率门槛为 85%。
成功还要求四轮真实承重、没有非轮部件承重、姿态/高度/速度合格；
合格稳态需连续延伸到 5 秒终点，持续至少 0.5 秒，且该稳态段开始不晚于 4 秒。
通过有限样本测试也不意味着任意姿态或真机成功率有保证。

<a id="rover-supermimic"></a>

### 平行四边形轮式车：SuperMimic 越障优化（实验性）

用 SuperMimic 世界模型梯度训练 8/12/16 cm 楼梯攀爬：世界模型在楼梯回放上拟合，actor 在世界模型里想象 32 步（0.64 s），
回报梯度穿过世界模型更新 actor，critic 拟合真实回报。不改动上面视觉前瞻一节的训练与产物。

命令都在 `/media/bigdisk/robot_lab` 下、已能启动 Isaac Lab 的终端执行：

```bash
conda activate env_isaaclab; unset PYTHONPATH; source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh
```

#### 两条训练路线

| | 路线一：Teacher 热启动 | 路线二：从零训练 |
|---|---|---|
| 起点 | PPO Teacher-Preview 的 actor | 随机初始化，不用 Teacher |
| 训练量 | 1024 env × 1000 轮（约 1 小时；超过 1000 轮会退化） | 2048 env × 40000 轮 |
| 课程 | 8→12→16 cm | 2→16 cm 连续 9 级，从 2 cm 起步 |
| 观测 | 232 维，与 Teacher 相同 | 233 维（多 1 维车道横向偏移） |
| 能否转成 Student 老师 | 能（第 4 步） | 暂不能（Student 没有车道偏移观测） |

#### 验收结果

严格验收：登顶、越线时横向 ≤0.15 m、对齐 ≥0.8、稳住 0.5 s；途中倾角门 8/12 cm 为 31.8°，16 cm 为 45°。每档 64 回合。

| 策略 | 8 cm | 12 cm | 16 cm |
|---|---|---|---|
| PPO Teacher-Preview | 100% | 70% | 0% |
| 路线一（v3，第 1000 轮；同配置重训一次为 67% / 81% / 0%） | 86% | 98% | 0% |
| 路线二（scratch_d3 第 4000 轮，分阶段训练的旧版本） | 61% | 31% | 8% |
| 路线二 `scratch_40k` 第 6000 轮（一次从零训练） | 91% | 89% | 33% |
| 路线二 `scratch_40k` 第 8000 轮 | 84% | 81% | 34% |
| 路线二 `scratch_40k` 第 10000 轮 / 第 12000 轮起（约第 10500 轮崩溃，已停训） | 5% / 0% | 38% / 0% | 59% / 0% |

16 cm 的主要难点是爬台阶时向一侧漂移，越线横向超差；路线二加了车道偏移观测和随台阶加重的横向惩罚，`scratch_40k` 第 6000 轮起 12 cm 超过 PPO Teacher，16 cm 到 33–34%。
各版本的实验记录与失败分析见 `plan.md` 4.5–4.7。

#### 路线一：Teacher → 越障优化 → Student（6 步）

| 步骤 | 做什么 | 训练方法 | 产物 |
|---|---|---|---|
| 1. Teacher | 232 维高度扫描策略，8/12/16 cm 课程 | PPO | `logs/vision_preview/teacher/<时间戳>_stairs_climb/model_7999.pt` |
| 2. 越障优化 | 从 Teacher 热启动，在楼梯上继续优化 | SuperMimic 世界模型梯度 | `logs/supermimic_terrain/<run>/` |
| 3. 验收与回放 | 8/12/16 cm 评测 + 汇总表，GUI 回放 | — | `<run>/acceptance/summary.md` |
| 4. 转换 | 把越障策略转成 rsl_rl 格式的 Teacher | — | `<run>/rsl_rl_teacher/model_1000.pt` |
| 5. Student | 4141 维深度策略，蒸馏第 4 步的老师 | DAgger | `logs/vision_preview/student_v3/<时间戳>_stairs_depth/model_7999.pt` |
| 6. Student 验收与回放 | 楼梯逐档评测、深度消融、GUI 回放 | — | `eval/*.json` |

```bash
# 1. Teacher（PPO 8000 轮）；已有可跳过
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/teacher --run_name stairs_climb \
  --num_envs 1024 --max_iterations 8000 --seed 42 --headless
TEACHER=logs/vision_preview/teacher/2026-10-14_11-31-31_stairs_climb/model_7999.pt   # 换成你的

# 2. 越障优化（默认参数即 v3；保持 1000 轮，2000 轮后 8/12 cm 会退化到 0–6%）
python scripts/supermimic/train_terrain.py --headless --teacher $TEACHER \
  --num_envs 1024 --max_iterations 1000 --run_name terrain_v3

# 3. 验收（约 40 分钟）与回放
python scripts/supermimic/accept_terrain.py --load_run terrain_v3
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --supermimic_terrain --load_run terrain_v3 --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5

# 4. 转换（几秒，不启动 Isaac）
python scripts/supermimic/export_terrain_teacher.py --load_run terrain_v3

# 5. Student（DAgger 8000 轮；首次不加 --resume）
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint logs/supermimic_terrain/terrain_v3/rsl_rl_teacher/model_1000.pt \
  --experiment_name /media/bigdisk/robot_lab/logs/vision_preview/student_v3 --run_name stairs_depth \
  --num_envs 128 --max_iterations 8000 --seed 42 --headless
STUDENT=logs/vision_preview/student_v3/<时间戳>_stairs_depth/model_7999.pt          # 换成你的

# 6. Student 验收与回放（difficulty 0.0/0.5/1.0 = 8/12/16 cm；depth_mode 换 blank/frozen 做消融）
python scripts/reinforcement_learning/rsl_rl/evaluate_parallelogram_vision.py --headless \
  --policy student-preview --scenario pyramid_stairs_inv --difficulty 0.5 --depth_mode live \
  --checkpoint $STUDENT --num_envs 16 --episodes_per_env 2 --seed 142
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --checkpoint $STUDENT --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5
```

按上面命令重训 1000 轮即可直接接第 3 步。现有的 `logs/supermimic_terrain/terrain_v3` 被训到了 4000 轮，`exported/policy.pt` 是退化版本（3% / 0% / 0%）；
要用其中第 1000 轮（实测 67% / 81% / 0%），先导出再验收、转换：

```bash
python scripts/supermimic/export_ckpt.py --load_run terrain_v3 --iterations 1000
python scripts/supermimic/accept_terrain.py --checkpoint logs/supermimic_terrain/terrain_v3/ckpt_eval/it1000/exported/policy.pt
python scripts/supermimic/export_terrain_teacher.py --load_run terrain_v3 --checkpoint ckpt_001000.pt
```

目前第 5–6 步只跑过冒烟，还没有全量训练过的 Student。

#### 路线二：从零训练（不用 Teacher）

参考 [go2w-supermimic](https://github.com/uwvwko-zzz/go2w-supermimic)（Go2W 轮足从零用 SuperMimic 训到 20 cm 楼梯）的做法，全程是世界模型 + 想象 BPTT，没有 PPO。

| 项 | 内容 |
|---|---|
| 任务 | `RobotLab-Isaac-SuperMimic-Scratch-Stairs-Parallelogram-Rover-v0`：9 级课程，台阶 2→16 cm（4 级台阶、40 cm 踏面），越过终点、直立、在车道内即升级 |
| 奖励 | 真实回报和想象目标用同一个 `PreviewTask.reward`（抬轮塑形、航向/横向、平地平稳），加翻车惩罚 −20 |
| 回放 | 25% 的世界模型窗口和想象起点优先取"轮子前方有台阶 / 正在越级 / 翻车"的片段 |
| `--strict_lane` | 晋级车道 0.15 m、出界 0.75 m，与验收一致；出车道为真终止 |
| `--lane_obs` | actor 多 1 维车道横向偏移（233 → 12） |
| `--lateral_pos -5 --lateral_pos_final -15 --lateral_ramp_cm 5 10` | 课程平均台阶从 5 cm 到 10 cm，横向位置惩罚从 −5 加重到 −15 |
| `--saturation_anchor <ckpt 或 self>` | 课程饱和（≥50% 环境到过最高级）后以该存档为信赖域参考（0.5），防止到顶后策略漂走；`self` 为触发时的当前 actor |

课程中通关最高级（16 cm）的环境会随机回落到各级（与 Isaac Lab 默认一致），矮台阶一直有训练数据。

```bash
# 训练（nohup 后台，2048 env × 40000 轮，每 1000 轮存 ckpt；日志 logs/supermimic_terrain/scratch_40k.log）
bash scripts/supermimic/run_scratch.sh scratch_40k 2048 40000 \
  --strict_lane --lane_obs --lateral_pos -5 --lateral_pos_final -15 --lateral_ramp_cm 5 10 --saturation_anchor self
# 过程验收：每 2000 轮自动导出并严格验收，结果追加到 logs/supermimic_terrain/scratch_40k/acceptance_progress.md
nohup bash scripts/supermimic/watch_accept.sh scratch_40k 2000 40000 2000 > logs/supermimic_terrain/scratch_40k_watch.log 2>&1 &
# 看课程进度（terrain_level_mean / step_height_cm_mean / lateral_pos_coef）
grep "^\[it" logs/supermimic_terrain/scratch_40k.log | tail -3
# 单独验收某个存档
python scripts/supermimic/export_ckpt.py --load_run scratch_40k --iterations 20000
python scripts/supermimic/accept_terrain.py --checkpoint logs/supermimic_terrain/scratch_40k/ckpt_eval/it20000/exported/policy.pt
# 回放（自动识别第 233 维车道偏移观测）
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Teacher-Preview-v0 \
  --supermimic_terrain --load_run scratch_40k --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5
```

- 课程曲线一直涨不代表严格验收在涨，选哪个存档以 `acceptance_progress.md` 为准。`scratch_40k` 在课程到顶（约 8000 轮）后开始退化，约第 10500 轮崩溃成原地不动，过程见 `plan.md`。当时课程把通关者夹在最高级、也没有饱和锚定；现已修复，`scratch_40k_r8k` 从它第 8000 轮存档按新设置重训（锚定该存档）。
- 训练中断后，用原来的参数加 `--resume --load_run scratch_40k` 运行 `train_terrain.py --from_scratch`；存档带地形课程等级，课程不会从头爬。
- 部署需要车载定位提供"离路线中线多远"。

#### 注意事项

- 同一时刻只启动一个 Isaac 进程导入机器人，并行训练请错开启动，否则会报 mimic 关节缺失。
- `vision_preview.py` 与 `vision_preview_reward.py` 的哈希写在每个 Preview 策略的 `params/vision_contract.json` 里，改动它们会让已有策略全部无法评测。
- 越障策略部署时动作限幅 ±2π，rsl_rl Teacher 不限幅；v3 在 16 cm 楼梯上只有 0.009% 的动作超出，对蒸馏没有实际影响。

#### 其他可选流程

不在上面两条路线上，按需使用。

平地 SuperMimic（45 → 12，纯平地速度跟踪，没有地形）：

```bash
python scripts/supermimic/train_rover.py --headless --num_envs 4096 --iterations 10000 --steps_per_iter 2 --run_name flat_v1
python scripts/supermimic/play_rover.py --load_run flat_v1              # 加 --keyboard 遥控，--video --headless 录像
```

深度扫描学生（第 5 步的另一种做法：深度预测高度扫描，再交给冻结的 Teacher actor；目前只支持 rsl_rl 格式 Teacher）：

```bash
python scripts/supermimic/train_scan_student.py --headless --teacher $TEACHER --num_envs 512 --iterations 4000 --run_name scan_v1
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-Parallelogram-Rover-Vision-Student-Temporal-Preview-v0 \
  --scan_student --checkpoint logs/supermimic_scan/scan_v1/exported/policy.pt \
  --num_envs 4 --preview_scenario pyramid_stairs_inv --preview_difficulty 0.5
```

### 其他实验示例

- Train AMP Dance for Unitree G1

  ```bash
  # Train
  python scripts/reinforcement_learning/skrl/train.py --task=RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --headless

  # Play
  python scripts/reinforcement_learning/skrl/play.py --task=RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --num_envs=32
  ```

- Train Handstand for Unitree A1

  ```bash
  # Train
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 --headless

  # Play
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0
  ```

- Train Anymal D with symmetry

  ```bash
  # Train
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 --headless --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=ppo_with_symmetry_data_augmentation agent.algorithm.symmetry_cfg.use_data_augmentation=true

  # Play
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 --agent=rsl_rl_with_symmetry_cfg_entry_point --run_name=ppo_with_symmetry_data_augmentation agent.algorithm.symmetry_cfg.use_data_augmentation=true
  ```

- Training and distilling Anymal D

  ```bash
  # Train the teacher agent
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --headless

  # Distill the teacher agent into a student agent
  python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --headless --agent=rsl_rl_distillation_cfg_entry_point --load_run teacher_run_folder_name

  # Play the student agent
  python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --agent rsl_rl_distillation_cfg_entry_point
  ```

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> ====================== ========================= ========================
> Command                Key (+ve axis)            Key (-ve axis)
> ====================== ========================= ========================
> Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
> Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
> Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
> ====================== ========================= ========================
> ```

* You can change `Rough` to `Flat` in the above configs.
* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint /PATH/TO/model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint /PATH/TO/model.pt`
* To train with multiple GPUs, use the following command, where --nproc_per_node represents the number of available GPUs:
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
    ```
* To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes. To train across multiple nodes/machines, it is required to launch an individual process on each node.

    For the master node, use the following command, where --nproc_per_node represents the number of available GPUs, and --nnodes represents the number of nodes:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
    ```
    Note that the port (`5555`) can be replaced with any other available port.
    For non-master nodes, use the following command, replacing `--node_rank` with the index of each machine:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless --distributed
    ```

## Add your own robot

Using the core framework developed as part of Isaac Lab, we provide various learning environments for robotics research.
These environments follow the `gym.Env` API from OpenAI Gym version `0.21.0`. The environments are registered using
the Gym registry.

Each environment's name is composed of `Isaac-<Task>-<Robot>-v<X>`, where `<Task>` indicates the skill to learn
in the environment, `<Robot>` indicates the embodiment of the acting agent, and `<X>` represents the version of
the environment (which can be used to suggest different observation or action spaces).

The environments are configured using either Python classes (wrapped using `configclass` decorator) or through
YAML files. The template structure of the environment is always put at the same level as the environment file
itself. However, its various instances are included in directories within the environment directory itself.
This looks like as follows:

```tree
source/robot_lab/assets/
├── __init__.py
└── unitree.py  # <- this is where we define robot assets

source/robot_lab/tasks/manager_based/locomotion/
├── __init__.py
└── velocity
    ├── config
    │   └── unitree_a1
    │       ├── agent  # <- this is where we store the learning agent configurations
    │       ├── __init__.py  # <- this is where we register the environment and configurations to gym registry
    │       ├── flat_env_cfg.py
    │       └── rough_env_cfg.py
    ├── __init__.py
    └── velocity_env_cfg.py  # <- this is the base task configuration
```

The environments are then registered in the `source/robot_lab/tasks/manager_based/locomotion/velocity/config/unitree_a1/__init__.py`:

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

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1RoughTrainerCfg",
    },
)
```

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=logs
```

## Code formatting

A pre-commit template is given to automatically format the code.

To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

**Note: Replace `<path-to-isaac-lab>` with your own IsaacLab path.**

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

### Clean USD Caches

Temporary USD files are generated in `/tmp/IsaacLab/usd_{date}_{time}_{random}` during simulation runs. These files can consume significant disk space and can be cleaned by:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

## Citation

Please cite the following if you use this code or parts of it:

```
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```

## Acknowledgements

The project uses some code from the following open-source code repositories:

- [linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
