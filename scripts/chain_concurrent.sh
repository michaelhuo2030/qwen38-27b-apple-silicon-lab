#!/bin/bash
# 并发队列：depth 矩阵之后接并发实验
#
# ⚠️ 顺序是硬要求，不是偏好：
#   1. 并发是**持续满载**，散热条件与单请求完全不同。
#      如果让它和单请求 depth 矩阵交替跑，两边的热历史都不可比。
#   2. oMLX engine_pool 是**单驻**的，两个实验同时跑会抢显存并污染读数。
#   3. 并发实验自带**独立的热纪律**（预冷 300s / 组间 180s / 回测回归 5%），
#      这套纪律要求它前面有一段干净的静置期。
#
# 所以：等 depth 矩阵 4 档全部结束 → 静置 → 跑并发（oq3e，最终推荐档）

set -u
export LANG=en_US.UTF-8
LOGDIR=/Users/a1-6/mtp_depth_lab/results
REPO=/Users/a1-6/Documents/mtp-depth-lab

echo "[$(date '+%H:%M')] 等待 depth 矩阵结束（跑 oq3e/6bit/8bit/ternary）…"
# depth 矩阵的 python 进程特征：bench_speed.py --tier <非4bit>
while pgrep -f "bench_speed.py --tier" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%H:%M')] depth 矩阵已结束"

echo "[$(date '+%H:%M')] 静置 600s 让 GPU 完全冷却（并发实验要求干净热历史）…"
sleep 600

cd "$REPO" || exit 1
echo "[$(date '+%H:%M')] 并发实验开始：档位 oq3e，depth 0-4 × conc 1/2/4"
python3 -u src/bench_concurrent.py \
    --tier oq3e --depths 0,1,2,3,4 --conc 1,2,4 \
    --waves 2 --cooldown 180 --precool 300 \
    --out "$LOGDIR/bench_concurrent_oq3e.json" \
    > "$LOGDIR/bench_concurrent_oq3e.log" 2>&1
echo "[$(date '+%H:%M')] 并发实验结束 rc=$?"
tail -25 "$LOGDIR/bench_concurrent_oq3e.log"
