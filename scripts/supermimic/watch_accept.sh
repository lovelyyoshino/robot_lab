#!/usr/bin/env bash
# watch_accept.sh — 训练中每 N 轮对新存档自动导出并跑 accept.py（挑选用种子），结论表追加到 <run>/acceptance_progress.md。
# 只读训练产物、不干预训练；nice/ionice 最低优先级；accept.py 对 logs/.isaac_eval.lock 加锁，同一时刻只有一个 Isaac 评测，
# 多个 watch 或手动验收会排队而不是并行。已有 ckpt_eval/itN/acceptance/summary.json 或 .accept_failed 标记的存档跳过，
# 所以重启 watch 不会重复评测。训练进程退出且没有待评存档时自动结束。
#
# 用法：nohup bash scripts/supermimic/watch_accept.sh <run> [每N轮=2000] [起始轮=N] \
#         > logs/supermimic_terrain/<run>_watch.log 2>&1 &
# 环境变量：ACCEPT_SEEDS（默认 "142 143 144"：挑选种子，最终验收的留出种子 9001–9003 不能在这里用）
#           ACCEPT_SEED_MODE（默认 rotate：每档一个种子，约 20 min/存档；split 约 50 min）
#           ACCEPT_SUITES（默认 "flat prelift stairs"）、ACCEPT_POLL_S（默认 120）
#           ACCEPT_IDLE_EXIT_H（没有 <run>.pid 时，连续多久无新存档就退出，默认 6）
#
# @author pony
# @date 2026-10-29
# @version v2.0.0
# @last_modified 2026-11-05
# @changelog
#   - v2.0.0 (2026-11-05): 只跑越障验收 accept.py；轮询新存档、跳过已评、低优先级、全局锁顺序执行；删除旧 accept_terrain/预抬分支。
#   - v1.1.0 (2026-11-02): 可选三类单台阶预抬验收；显式记录执行异常，训练退出后不永久等待。
#   - v1.0.0 (2026-10-29): 初始创建（d4 长训的过程验收）
set -u
RUN=${1:?用法：watch_accept.sh <run> [每N轮] [起始轮]}
EVERY=${2:-2000}
FROM=${3:-$EVERY}
SEEDS=${ACCEPT_SEEDS:-"142 143 144"}
MODE=${ACCEPT_SEED_MODE:-rotate}
SUITES=${ACCEPT_SUITES:-"flat prelift stairs"}
POLL=${ACCEPT_POLL_S:-120}
IDLE_S=$(( ${ACCEPT_IDLE_EXIT_H:-6} * 3600 ))
if (( EVERY < 1 || FROM < 1 )); then
  echo "间隔和起始轮必须为正" >&2
  exit 2
fi
cd "$(dirname "$0")/../.."
set +u  # conda 与 Isaac 的环境脚本会引用未定义变量（如 ZSH_VERSION）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
unset PYTHONPATH
source /media/bigdisk/IsaacLab/_isaac_sim/setup_conda_env.sh >/dev/null
set -u
DIR=logs/supermimic_terrain/$RUN
OUT=$DIR/acceptance_progress.md
PID_FILE=logs/supermimic_terrain/$RUN.pid
LOW="nice -n 19 ionice -c 3"
mkdir -p "$DIR"
last_new=$(date +%s)
while true; do
  pending=0
  evaluated=0
  for CK in $(ls "$DIR"/ckpt_*.pt 2>/dev/null | sort); do
    name=$(basename "$CK" .pt)
    it=$((10#${name#ckpt_}))
    (( it >= FROM && it % EVERY == 0 )) || continue
    EVAL_DIR=$DIR/ckpt_eval/it$it
    if [ -f "$EVAL_DIR/acceptance/summary.json" ] || [ -f "$EVAL_DIR/.accept_failed" ]; then
      continue
    fi
    # 存档 10 s 内大小不变才算写完，避免读到半个文件
    s1=$(stat -c %s "$CK")
    sleep 10
    if [ "$s1" != "$(stat -c %s "$CK")" ]; then
      pending=1
      continue
    fi
    last_new=$(date +%s)
    evaluated=1
    mkdir -p "$EVAL_DIR"
    echo "### 第 $it 轮 ($(date '+%F %T'))" >> "$OUT"
    if ! $LOW python scripts/supermimic/export_ckpt.py --load_run "$RUN" --iterations "$it" > "$EVAL_DIR/export.log" 2>&1; then
      echo "导出失败，见 $EVAL_DIR/export.log" >> "$OUT"
      touch "$EVAL_DIR/.accept_failed"
      continue
    fi
    code=0
    # shellcheck disable=SC2086
    $LOW python scripts/supermimic/accept.py --checkpoint "$EVAL_DIR/exported/policy.pt" \
      --seeds $SEEDS --seed_role selection --seed_mode "$MODE" --suites $SUITES \
      > "$EVAL_DIR/acceptance.log" 2>&1 || code=$?
    if [ -f "$EVAL_DIR/acceptance/summary.md" ]; then
      # 只抄结论表（到第一个二级标题为止），细节看 summary.md
      awk '/^## /{exit} {print}' "$EVAL_DIR/acceptance/summary.md" >> "$OUT"
    fi
    if (( code > 1 )); then
      echo "验收执行异常（exit $code），见 $EVAL_DIR/acceptance.log" >> "$OUT"
      touch "$EVAL_DIR/.accept_failed"
    fi
    echo >> "$OUT"
    echo "it$it done (exit $code)"
  done
  if [ -f "$PID_FILE" ]; then
    # 训练退出后再完整扫一遍（本轮没有新评测）才结束，避免漏掉退出前最后写下的存档
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null && (( pending == 0 && evaluated == 0 )); then
      echo "训练已退出，全部存档已验收" | tee -a "$OUT"
      exit 0
    fi
  elif (( $(date +%s) - last_new > IDLE_S )); then
    echo "超过 $(( IDLE_S / 3600 )) h 没有新存档，退出" | tee -a "$OUT"
    exit 0
  fi
  sleep "$POLL"
done
