#!/usr/bin/env bash
# 方案 D：从零 SuperMimic 楼梯训练（无 Teacher），nohup 后台运行。
# 用法：bash scripts/supermimic/run_scratch.sh [run_name] [num_envs] [iterations]
# 日志：logs/supermimic_terrain/<run_name>.log；每 1000 轮存 ckpt，可用 export_ckpt.py + accept_terrain.py 验收中间存档。
# 续训：python scripts/supermimic/train_terrain.py --headless --from_scratch --resume --load_run <run_name> --max_iterations <N>
set -e
RUN=${1:-scratch_d1}
ENVS=${2:-2048}
ITERS=${3:-12000}
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset PYTHONPATH
source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh >/dev/null
mkdir -p logs/supermimic_terrain
nohup python scripts/supermimic/train_terrain.py --headless --from_scratch \
  --num_envs "$ENVS" --max_iterations "$ITERS" --save_interval 1000 --log_interval 20 \
  --run_name "$RUN" > "logs/supermimic_terrain/$RUN.log" 2>&1 &
echo "started $RUN pid=$! log=logs/supermimic_terrain/$RUN.log"
