#!/usr/bin/env python3
"""
bench_concurrent.py — MTP depth 在**并发**下的表现（单请求结论不外推到这里）

## 为什么必须单独测

2026-10-03 的 depth sweep 全部是**单请求**。但真实使用是并发的，
而并发会改变三件事：

1. **batching**：多个请求的 token 在同一次前向里拼批。单请求测的
   backbone 前向成本不再成立 —— 一次前向要喂 N 条序列的 KV。
2. **MTP 接受率**：draft 是在 batch 内共享/竞争资源时产生的，
   α 未必和单请求一样。
3. **热**：并发是持续满载，散热条件与单请求完全不同，**必须重新建立
   热纪律**（预冷 / 组间静置 / 回测回归），不能沿用单请求的那套读数。

已知必须保持的前提：**TurboQuant 关**。oMLX 0.7.0 有未修 bug
(jundot/omlx#3906：TQ+MTP+并发直接崩)，并发期硬性禁用。

## 测什么

固定 tier，逐 depth 扫，并发度 1 / 2 / 4。
关键是 **scaling 曲线**：单请求最优的 depth，在并发下还最优吗？

- 单请求实测峰值是 depth=3（oQ4e）。如果并发下峰值左移到 1 或 2，
  说明 batching 摊薄了 backbone 成本优势，depth 的最优值**不是模型属性**。
- 同时报每请求延迟（p50/p95），因为吞吐涨了但延迟炸了就毫无意义。

## 用法

  python3 src/bench_concurrent.py --tier oq3e --depths 0 1 2 3 4 \
      --conc 1 2 4 --waves 2 --cooldown 180 --precool 300 \
      --out ../results/bench_concurrent.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ladder27 as L
import omlx_client as C


def one_request(prompt: str, max_tokens: int) -> dict:
    """跑一条并发请求。**不共用 omlx_client 的全局 opener**——
    那个 cookie jar 不是线程安全的，并发下会串状态。"""
    a = C.generate(prompt, max_tokens=max_tokens, temperature=0.0,
                   think=False)
    return {
        "ok": not a.get("error"),
        "tps": a.get("tps"),
        "tokens": a.get("completion_tokens"),
        "alpha": a.get("alpha"),
        "ttft": a.get("ttft"),
        "chars": len(a.get("text") or ""),
    }


def run_wave(prompt: str, max_tokens: int, conc: int) -> dict:
    """一波并发：conc 条请求同时打出去，测墙钟吞吐与每请求延迟。"""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda _: one_request(prompt, max_tokens),
                         range(conc)))
    wall = time.time() - t0

    ok = [r for r in res if r["ok"]]
    toks = [r["tokens"] for r in ok if r["tokens"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"]]
    # 聚合吞吐 = 全部请求的 token / 墙钟，这才是并发下真正该比较的量
    agg_tps = sum(toks) / wall if wall > 0 and toks else None
    return {
        "conc": conc,
        "wall_s": round(wall, 2),
        "n_ok": len(ok),
        "n_err": len(res) - len(ok),
        "agg_tps": round(agg_tps, 2) if agg_tps else None,
        "per_req_tps_median": round(st.median([r["tps"] for r in ok
                                               if r["tps"]]), 2) if ok else None,
        "ttft_p50": round(st.median(ttfts), 3) if ttfts else None,
        "ttft_max": round(max(ttfts), 3) if ttfts else None,
        "alpha_mean": round(st.mean([r["alpha"] for r in ok
                                     if r["alpha"]]), 2) if ok else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="oq3e", choices=list(L.TIERS))
    ap.add_argument("--depths", default="0,1,2,3,4")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--waves", type=int, default=2,
                    help="每格并发度重复几波（取中位数）")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--cooldown", type=float, default=180.0)
    ap.add_argument("--precool", type=float, default=300.0)
    ap.add_argument("--out", default="../results/bench_concurrent.json")
    ap.add_argument("--allow-overload", action="store_true",
                    help="跳过内存闸门（不推荐：root 运行的 omlx-server 崩了救不回来）")
    a = ap.parse_args()

    depths = [int(x) for x in a.depths.split(",")]
    concs = [int(x) for x in a.conc.split(",")]

    print(f"档位 {a.tier}  depths={depths}  conc={concs}  waves={a.waves}")
    print("⚠️ TurboQuant 保持关闭（oMLX 0.7.0 #3906：TQ+MTP+并发会崩）\n")

    if a.precool > 0:
        print(f"开跑前预冷 {a.precool:.0f}s（并发是持续满载，散热条件与单请求不同）…", flush=True)
        time.sleep(a.precool)

    # ---- 内存闸门：必须在激活模型**之前** ----
    #
    # 为什么放在这里：并发压测会把 N 条序列的 KV cache 同时压进
    # Metal cap（本机 85.9GB）。如果机器本来就被别的程序塞满
    # （实测 2026-10-03：swap 已用 93%、load 8.92、其余进程合计 36.9GB），
    # 再加并发就是真实的崩溃风险。而 omlx-server 是 **root 运行**，
    # 崩了 agent 自己救不回来（ps 里 kill 不掉）。
    #
    # 所以先查、后跑，不靠"跑着看看会不会崩"。
    print("\n=== 内存闸门 ===", flush=True)
    import memgate
    ok, reasons, data = memgate.check(strict=not a.allow_overload)
    sw = data.get("swap") or {}
    print(f"  可用内存 {data['mem_free_pct']}%  "
          f"swap 已用 {sw.get('used_pct')}%  剩 {sw.get('free_mb')}MB  "
          f"load {data['load1']}")
    for r in data.get("top_rss", [])[:5]:
        print(f"    {r['rss_gb']:6.2f} GB  {r['proc']}")
    if not ok:
        print("❌ 内存不达标，拒绝启动并发压测：", flush=True)
        for r in reasons:
            print(f"   - {r}")
        print("   关掉占内存的程序后重跑；确需强压加 --allow-overload（不推荐）")
        return 3
    print("✅ 放行\n", flush=True)

    lad = L.Ladder(a.tier, think=False, ane=True, turboquant=0, mtp=True,
                   verbose=False)
    lad.activate()
    C.MODEL_ID = lad.model_id
    lad.assert_single_resident()
    sent = lad.sentinel()
    if sent.failed or not sent.ok:
        print(f"哨兵未通过，终止：{sent.failed}")
        return 2
    print(f"哨兵 {len(sent.passed)}/{len(sent.passed)+len(sent.failed)} 通过\n")

    prompt = C.generate("用一句话说明什么是投机解码。", 32, 0.0) and (
        "请用大约 150 个字解释：投机解码（speculative decoding）为什么能加速"
        "大模型推理，以及它的主要风险是什么。")
    # 预热：并发下第一次同样要付 kernel 编译代价（单请求那次已栽过）
    run_wave(prompt, 64, max(concs))
    print("预热完成\n")

    rows = []
    first_cell = (depths[0], concs[0])
    for d in depths:
        lad.depth = d
        lad.activate()
        for ci, c in enumerate(concs):
            if rows:
                print(f"  静置 {a.cooldown:.0f}s …", flush=True)
                time.sleep(a.cooldown)
            waves = [run_wave(prompt, a.max_tokens, c) for _ in range(a.waves)]
            agg = [w["agg_tps"] for w in waves if w["agg_tps"]]
            row = {
                "tier": a.tier, "depth": d, "conc": c, "waves": a.waves,
                "agg_tps_median": round(st.median(agg), 2) if agg else None,
                "ttft_p50": round(st.median([w["ttft_p50"] for w in waves
                                            if w["ttft_p50"]]), 3),
                "ttft_max": max(w["ttft_max"] for w in waves if w["ttft_max"]),
                "n_err": sum(w["n_err"] for w in waves),
                "wall_s": waves[0]["wall_s"],
            }
            rows.append(row)
            print(f"  depth={d} conc={c}  聚合吞吐中位={row['agg_tps_median']}  "
                  f"每请求={waves[0]['per_req_tps_median']}  "
                  f"ttft_p50={row['ttft_p50']}s  err={row['n_err']}", flush=True)
            json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)

    # 回测第一格：并发下热污染更狠（持续满载），漂移阈值同样 5%
    print(f"\n回测第一格（depth={first_cell[0]} conc={first_cell[1]})…", flush=True)
    time.sleep(a.cooldown)
    lad.depth = first_cell[0]
    lad.activate()
    rw = [run_wave(prompt, a.max_tokens, first_cell[1]) for _ in range(a.waves)]
    re_agg = st.median([w["agg_tps"] for w in rw if w["agg_tps"]])
    f = next(r for r in rows
             if r["depth"] == first_cell[0] and r["conc"] == first_cell[1])
    drift = ((re_agg - f["agg_tps_median"]) / f["agg_tps_median"] * 100
             if f["agg_tps_median"] and re_agg else None)
    verdict = ("✅ 可用" if drift is not None and abs(drift) <= 5
               else "❌ 漂移 >5%，整轮作废（并发下热污染更重）")
    print(f"  首测 {f['agg_tps_median']}  回测 {round(re_agg,2)}  "
          f"漂移 {drift:+.1f}%" if drift is not None else "  无法比较")
    print(f"  判定：{verdict}")

    json.dump({"rows": rows, "regression_agg_tps": re_agg,
               "drift_pct": drift, "verdict": verdict,
               "config": {"tier": a.tier, "depths": depths, "conc": concs,
                          "waves": a.waves, "cooldown": a.cooldown,
                          "turboquant": 0}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
