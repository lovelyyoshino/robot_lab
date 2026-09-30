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
