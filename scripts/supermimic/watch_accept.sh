#!/usr/bin/env bash
# watch_accept.sh — 训练中按间隔等存档落盘，自动导出并跑严格验收，结果汇总到 <run>/acceptance_progress.md。
# 只读训练产物、不干预训练；横向变差时是否停训由人看汇总决定。
# 用法：nohup bash scripts/supermimic/watch_accept.sh <run> <起始轮> <终止轮> [间隔=2000] > logs/supermimic_terrain/<run>_watch.log 2>&1 &
#
# @author pony
# @date 2026-10-29
# @version v1.0.0
# @last_modified 2026-10-29
# @changelog
#   - v1.0.0 (2026-10-29): 初始创建（d4 长训的过程验收）
RUN=$1; FROM=$2; TO=$3; STEP=${4:-2000}
cd "$(dirname "$0")/../.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset PYTHONPATH
source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh >/dev/null
DIR=logs/supermimic_terrain/$RUN
OUT=$DIR/acceptance_progress.md
for ((it=FROM; it<=TO; it+=STEP)); do
  CK=$(printf "%s/ckpt_%06d.pt" "$DIR" "$it")
  # 等存档出现且写完（大小 10 s 内不再变化），避免读到半个文件
  until [ -f "$CK" ] && s1=$(stat -c %s "$CK") && sleep 10 && [ "$s1" = "$(stat -c %s "$CK")" ]; do sleep 60; done
  python scripts/supermimic/export_ckpt.py --load_run "$RUN" --iterations "$it" || { echo "export it$it 失败"; continue; }
  RES=$(python scripts/supermimic/accept_terrain.py --checkpoint "$DIR/ckpt_eval/it$it/exported/policy.pt" 2>&1 \
        | grep -E "^\| (8|12|16) cm|平地平稳（")
  { echo "### 第 $it 轮 ($(date '+%F %T'))"; echo "$RES"; echo; } >> "$OUT"
  echo "it$it done"
done
