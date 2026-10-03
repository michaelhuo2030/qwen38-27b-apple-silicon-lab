#!/usr/bin/env python3
"""从 oMLX server.log 提取**按 draft 位置分层**的 MTP 接受率。

为什么需要这个：
  bench_speed.py 只能拿到「整条流的 α」。但 depth-k 的真正故事在**位置衰减**上——
  d1 的接受率必然远高于 d4，因为越靠后的 draft 是在前面 draft 的错误上继续猜的。
  拿到 depth[d1=..,d2=..,d3=..,d4=..] 就能直接看出这条衰减曲线。

配套事实（决定怎么解读）：
  Qwen3.8-27B 的 text_config.mtp_num_hidden_layers = 1
  → 模型只带 **1 层** MTP 头。oMLX 的 depth-k 不是「用更多层 MTP」，
    而是**把同一个头链式跑 depth 次**（batch_generator.py 的 `for j in range(depth)`）。
  所以 depth>1 的每一层 draft 都建立在上一层 draft 的输出之上，
  误差会累积 —— 预期 d2+ 接受率断崖式下降。

用法：
  parse_depth_chain.py --log ~/.omlx/logs/server.log --since "19:30" --depth 2
  parse_depth_chain.py --json      # 全部结构化输出（给决策矩阵取数用）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime

# MTP[0] finish=stop tokens=13 cycles=3 tok/cycle=4.33 accept=8/9 (88.9%)
# depth[d1=2/3,d2=2/2,d3=2/2,d4=2/2] emits[...] timing[...]
LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?"
    r"MTP\[(?P<idx>\d+)\] finish=(?P<finish>\w+) tokens=(?P<tokens>\d+)\s+"
    r"cycles=(?P<cycles>\d+) tok/cycle=(?P<tpc>[\d.]+)\s+"
    r"accept=(?P<an>\d+)/(?P<ad>\d+)"
    r"(?:.*?depth\[(?P<depth>[^\]]*)\])?"
)
POS = re.compile(r"d(?P<pos>\d+)=(?P<num>\d+)/(?P<den>\d+)")


def parse(path: str, since: str | None) -> list[dict]:
    """since 形如 '19:30' 或 '2026-10-03 19:30'，按服务器本地时间过滤。"""
    lo: datetime | None = None
    if since:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%H:%M", "%H:%M:%S"):
            try:
                lo = datetime.strptime(since, fmt)
                break
            except ValueError:
                continue
        if lo is None:
            sys.exit(f"无法解析 --since {since!r}（支持 '19:30' 或 '2026-10-03 19:30'）")

    out: list[dict] = []
    with open(path, "r", errors="replace") as f:
        for raw in f:
            m = LINE.search(raw)
            if not m:
                continue
            ts = datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S")
            if lo is not None and ts.time() < lo.time():
                continue

            pos: dict[int, list[int]] = {}
            if m["depth"]:
                for p in POS.finditer(m["depth"]):
                    pos[int(p["pos"])] = [int(p["num"]), int(p["den"])]

            rec = {
                "ts": m["ts"],
                "idx": int(m["idx"]),
                "finish": m["finish"],
                "tokens": int(m["tokens"]),
                "cycles": int(m["cycles"]),
                "tok_per_cycle": float(m["tpc"]),
                "accept_num": int(m["an"]),
                "accept_den": int(m["ad"]),
                "positions": {k: {"num": v[0], "den": v[1]} for k, v in sorted(pos.items())},
            }
            rec["alpha"] = (rec["accept_num"] / rec["accept_den"]) if rec["accept_den"] else None
            out.append(rec)
    return out


def cfg_depth(r: dict) -> int:
    """从记录里反推**配置**的 MTP depth。

    depth=1 的行只会带 `depth[d1=..]`；depth=k 会带 d1..dk。
    所以「出现的最大位置号」就是这次请求的配置 depth。

    为什么必须先分组再合并（踩过的坑）：
    第一次跑 `--since 19:00` 不带 --depth，把 depth=1 和 depth=4 的记录混在一起，
    算出的 d1 = 5077/6765 —— **分母被 depth=4 的 16 个 cycle 稀释了**，
    而 d2/d3/d4 只有 12/12 的小样本。混算出来的 α 没有任何物理意义。
    正确做法是按 cfg_depth 分组，每组内部再合并。
    """
    return max(r["positions"]) if r["positions"] else 1


def aggregate(recs: list[dict]) -> dict:
    """把所有请求的同位置接受数/接受次数加起来（**先合并再除**，不是逐行平均）。"""
    agg_num: dict[int, int] = defaultdict(int)
    agg_den: dict[int, int] = defaultdict(int)
    a_num = a_den = 0
    cyc = tok = 0
    skipped = 0
    for r in recs:
        if r["cycles"] == 0:      # 预热/空跑，accept=0/0
            skipped += 1
            continue
        a_num += r["accept_num"]
        a_den += r["accept_den"]
        cyc += r["cycles"]
        tok += r["tokens"]
        for k, v in r["positions"].items():
            agg_num[k] += v["num"]
            agg_den[k] += v["den"]

    per_pos = {}
    for k in sorted(agg_num):
        per_pos[k] = {
            "num": agg_num[k],
            "den": agg_den[k],
            "rate": round(agg_num[k] / agg_den[k], 4) if agg_den[k] else None,
        }
    return {
        "n_requests": len(recs),
        "n_cycles": cyc,
        "tokens": tok,
        "skipped_zero_cycle": skipped,
        "alpha_pooled": round(a_num / a_den, 4) if a_den else None,
        "tok_per_cycle_pooled": round(tok / cyc, 3) if cyc else None,
        "per_position": per_pos,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/Users/a1-6/.omlx/logs/server.log")
    ap.add_argument("--since", help="起始时间，如 19:30（服务器本地时间）")
    ap.add_argument("--depth", type=int, help="只看配置 depth 恰为该值的记录")
    ap.add_argument("--group-by-depth", action="store_true",
                    help="按配置 depth 分组输出（推荐：避免混算污染分母）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args()

    recs = parse(a.log, a.since)
    if a.depth:
        recs = [r for r in recs if cfg_depth(r) == a.depth]
    if not recs:
        print("没有匹配的 MTP 记录（检查 --since）", file=sys.stderr)
        return 1

    if a.group_by_depth:
        buckets: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            buckets[cfg_depth(r)].append(r)
        if a.json:
            print(json.dumps({str(k): aggregate(v) for k, v in sorted(buckets.items())},
                             ensure_ascii=False, indent=1))
            return 0
        for k in sorted(buckets):
            agg = aggregate(buckets[k])
            print(f"\n===== 配置 depth={k} "
                  f"（{agg['n_requests']} 请求 / {agg['n_cycles']} cycles / {agg['tokens']} tokens）=====")
            print(f"  合并 α = {agg['alpha_pooled']}   tok/cycle = {agg['tok_per_cycle_pooled']}")
            base = None
            for pos, v in agg["per_position"].items():
                if base is None and v["rate"] is not None:
                    base = v["rate"]
                rel = f"{(v['rate']/base*100):.0f}%" if v["rate"] and base else "-"
                print(f"  d{pos:<3}{v['num']}/{v['den']:<10}{(v['rate'] or 0):<8.3f}相对 d1 {rel}")
        return 0

    agg = aggregate(recs)
    if a.json:
        print(json.dumps({"records": recs, "aggregate": agg}, ensure_ascii=False, indent=1))
        return 0

    print(f"请求 {agg['n_requests']} 个（跳过 {agg['skipped_zero_cycle']} 个零周期预热）"
          f"  cycles={agg['n_cycles']}  tokens={agg['tokens']}")
    print(f"合并 α = {agg['alpha_pooled']}   tok/cycle = {agg['tok_per_cycle_pooled']}")
    print("\n按 draft 位置分层接受率：")
    print(f"  {'位置':<6}{'接受/机会':<16}{'接受率':<10}相对 d1")
    base = None
    for pos, v in agg["per_position"].items():
        if base is None and v["rate"] is not None:
            base = v["rate"]
        rel = f"{(v['rate']/base*100):.0f}%" if v["rate"] and base else "-"
        print(f"  d{pos:<5}{v['num']}/{v['den']:<14}{(v['rate'] or 0):<10.3f}{rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
