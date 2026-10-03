#!/bin/bash
# depth 矩阵补全：把 5 档量化全部跑 depth sweep
#
# ⚠️ 串行，绝不并行：
#   oMLX 是**单驻**的（engine_pool 只保一个 loaded 模型），
#   换档必须 unload + load，两个任务同时跑会互相抢显存并且污染速度读数。
#   温度扫描在跑时**不能**启动这个脚本。
#
# 已完成：4bit（data/bench_depth.json）
# 本脚本补：oq3e（最终推荐档，最重要）、6bit、8bit、ternary
#
# 每档：depth 0-4 × ctx 1k/16k × 3 reps + 回测回归
# 每档约 80 分钟（含模型加载 ~60s/次切换），4 档约 5.5 小时

set -u
LOGDIR=/Users/a1-6/mtp_depth_lab/results
REPO=/Users/a1-6/Documents/mtp-depth-lab
export LANG=en_US.UTF-8

# 单驻硬闸：确认没有别的实验在跑
for pid in 54109; do
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "❌ 温度扫描 (pid $pid) 还在跑，拒绝启动（会污染彼此读数）"
    exit 1
  fi
done

for TIER in oq3e 6bit 8bit ternary; do
  echo "=============== $(date '+%H:%M') 档位 $TIER 开始 ==============="
  cd "$REPO" || exit 1
  python3 -u src/bench_speed.py \
      --tier "$TIER" --depths 0 1 2 3 4 --contexts 1k 16k \
      --reps 3 --cooldown 180 --precool 300 \
      --out "$LOGDIR/bench_depth_$TIER.json" \
      > "$LOGDIR/bench_depth_$TIER.log" 2>&1
  rc=$?
  echo "[$(date '+%H:%M')] $TIER 结束 rc=$rc"
  tail -6 "$LOGDIR/bench_depth_$TIER.log"
  # 档间长静置：上一档刚跑完，模型还在热
  echo "  档间静置 300s …"
  sleep 300
done

echo "=============== $(date '+%H:%M') 全部完成 ==============="
for TIER in oq3e 6bit 8bit ternary; do
  echo "--- $TIER"
  python3 - <<PY
import json
d=json.load(open("$LOGDIR/bench_depth_$TIER.json"))
rows=d['rows'] if isinstance(d,dict) else d
for r in rows:
    print(f"  depth={r['depth']} ctx={r['context']:4s} tps={r['tps_median']:6.2f} IQR={r['tps_iqr']}")
if isinstance(d,dict):
    print(f"  漂移={d.get('drift_pct')}% 判定={d.get('verdict')}")
PY
done
