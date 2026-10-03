#!/usr/bin/env python3
"""
速度 / MTP depth 基准 —— 带**热纪律**的版本。

为什么不能直接用 ladder27.sweep_depth
-------------------------------------
`Ladder.sweep_depth` 逐档连跑、不做任何静置。而社区实测（本机 M3 Max
对照组，来源 jundot/omlx#2689）：

    冷却 5min      21.9 tok/s   GPU 1317 MHz
    紧接着连跑     16.4 tok/s   GPU 1243 MHz，min 743 MHz   ← -25%
    休息 3min      22.0 tok/s

**`pmset -g therm` 全程显示 thermal=0，完全看不见这个降频。**
所以不强制静置的逐档对比，量到的是「你刚跑过多久」而不是「这个配置多快」。

本脚本的纪律
------------
1. **每组之间强制 ≥3min 静置**（默认 180s，可用 --cooldown 调）
2. **首档回测回归**：sweep 跑完重测第一个配置，漂移 >5% 整轮作废
3. **开跑前把 server.gpu_keep_warm_interval 显式设 0**
   —— oMLX 0.7.0 默认请求后保持 GPU 非 idle，不关掉的话「静置」根本静不下来
4. **用长输出任务**：输出 1–72 token 的任务，gen_tps 的分母太小、噪声压过信号
5. **报中位数 + IQR**，不报均值：热降频造成的是长尾拖尾，均值会被单次慢样本拉偏

用法：
    python3 bench_speed.py --tier 4bit --depths 0 1 2 3 4 --contexts 1k 16k
    python3 bench_speed.py --tier 4bit --quick        # 小样本冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ladder27 as L          # noqa: E402
import omlx_client as C       # noqa: E402

C.API_KEY = L._api_key()
RESULTS = os.path.expanduser("~/mtp_depth_lab/results")

# 上下文长度的填充文本。用中英混排避免 tokenizer 对单一语言过度压缩，
# 否则「16K token」的 prompt 实际可能只有 6K。
_FILL_ZH = "这是一段用于把上下文长度撑到目标规模的填充文本，包含中英文与技术术语混合。"
_FILL_EN = "This is filler text used to inflate the context to a target length, mixed prose."


def make_prompt(ctx_tokens: int) -> str:
    """构造约 ctx_tokens 长度的 prompt + 一个需要长输出的明确指令。"""
    target_chars = ctx_tokens * 2          # 中英混排大约 2 char/token
    body = []
    n = 0
    while n < target_chars:
        piece = _FILL_ZH + " " + _FILL_EN + "\n"
        body.append(piece)
        n += len(piece)
    return (
        "以下是一段参考资料，请完整读过：\n\n"
        + "".join(body)
        + "\n\n---\n\n"
        "现在请基于上面的资料，写一篇 **不少于 500 字**的中文技术说明，"
        "主题是「在 Apple Silicon 上做低比特推理时的带宽瓶颈与投机解码」。"
        "要求：分小标题、包含具体数字、结尾给三条可执行建议。"
    )


# 长输出任务：max_tokens 给足，让 gen_tps 进入稳态再统计
LONG_TASK = {"key": "long_essay", "max_tokens": 900}


def set_gpu_keep_warm(seconds: float) -> bool:
    """
    把 server.gpu_keep_warm_interval 设成 seconds。

    用 ladder27 的 `_login()` + 模块级 CookieJar `_op`，不要自己造一套 cookie
    —— admin 路由认 session cookie 不认 Bearer，自造极易静默 401。
    """
    import urllib.request
    try:
        L._login()
        r = urllib.request.Request(
            f"{L.BASE}/admin/api/settings",
            data=json.dumps({"server": {"gpu_keep_warm_interval": seconds}}).encode(),
            headers={"Content-Type": "application/json"}, method="PUT")
        with L._op.open(r, timeout=30) as f:
            body = json.load(f)
    except Exception as e:
        print(f"    [设置失败] {type(e).__name__}: {e}")
        return False
    # 回读确认。**写入成功不等于生效** —— 125B 那次「写入返回 200 但值没变」
    # 导致整轮跑在旧配置上，报出 6.9% 的假差异。
    back = body.get("settings", {}) if isinstance(body, dict) else {}
    got = (back.get("server", {}) or {}).get("gpu_keep_warm_interval")
    if got is None:
        got = body.get("gpu_keep_warm_interval")
    if got is not None and abs(float(got) - seconds) > 1e-6:
        print(f"    [回读不一致] 期望 {seconds}，实际 {got}")
        return False
    return True


MTP_LOG = os.path.expanduser("~/.omlx/logs/server.log")


def mtp_stats_since(offset: int) -> tuple[float | None, int, list[dict]]:
    """
    直接从服务端日志窗口读 MTP 统计。返回 (加权α, 样本数, 逐条明细)。

    为什么不直接用 `Ladder.ask()` 返回的 `alpha` / `n_mtp_lines`：
    实测在**长输出请求**上它恒为 0/null —— 内部那套日志偏移捕获跟丢了，
    但服务端日志里明明有 `MTP[0] ... accept=389/509 (76.4%)`。
    自己按字节偏移读窗口更可靠，也不依赖被测代码的内部实现。
    """
    import re
    pat = re.compile(
        r"MTP\[(\d+)\].*?cycles=(\d+)\s+tok/cycle=([\d.]+)\s+"
        r"accept=(\d+)/(\d+)\s*\(([\d.]+)%\)")
    try:
        size = os.path.getsize(MTP_LOG)
    except OSError:
        return None, 0, []
    start = offset if 0 <= offset <= size else max(0, size - 200_000)
    with open(MTP_LOG, "r", errors="ignore") as f:
        f.seek(start)
        chunk = f.read()
    rows = []
    for m in pat.finditer(chunk):
        rows.append({"batch": int(m.group(1)), "cycles": int(m.group(2)),
                     "tok_per_cycle": float(m.group(3)),
                     "accept": int(m.group(4)), "den": int(m.group(5)),
                     "alpha": float(m.group(6))})
    num = sum(r["accept"] for r in rows)
    den = sum(r["den"] for r in rows)
    return ((num / den * 100) if den else None), den, rows


def measure(lad, prompt: str, reps: int) -> dict:
    """跑 reps 次，返回中位数与 IQR（不是均值）。"""
    tps, ttft, ctok = [], [], []
    rows_all: list[dict] = []
    for _ in range(reps):
        off = os.path.getsize(MTP_LOG) if os.path.exists(MTP_LOG) else 0
        a = lad.ask(prompt, max_tokens=LONG_TASK["max_tokens"])
        if a.error:
            continue
        if a.gen_tps:
            tps.append(a.gen_tps)
        if a.ttft:
            ttft.append(a.ttft)
        if a.completion_tokens:
            ctok.append(a.completion_tokens)
        # 日志是异步刷的，稍等一下再读窗口
        time.sleep(0.6)
        _, den, rows = mtp_stats_since(off)
        if den == 0:
            time.sleep(1.5)
            _, den, rows = mtp_stats_since(off)
        rows_all += rows

    def med(xs):
        return round(statistics.median(xs), 2) if xs else None

    def iqr(xs):
        if len(xs) < 2:
            return None
        q = statistics.quantiles(xs, n=4)
        return round(q[2] - q[0], 2)

    num = sum(r["accept"] for r in rows_all)
    den_all = sum(r["den"] for r in rows_all)
    tpcs = [r["tok_per_cycle"] for r in rows_all]
    return {
        "n": len(tps),
        "tps_median": med(tps),
        "tps_iqr": iqr(tps),
        "tps_min": round(min(tps), 2) if tps else None,
        "ttft_median": med(ttft),
        "ctok_median": med(ctok),
        "alpha_weighted": round(num / den_all, 2) if den_all else None,
        "mtp_lines": den_all,
        "tok_per_cycle_median": med(tpcs),
        "tps_all": tps,
    }


def apply_depth(lad, depth: int) -> None:
    """
    改 MTP depth 并让新 graph 生效。

    ⚠️ 实测出来的关键事实：**`_put_settings` 会把模型卸掉**（0.6 秒后
    `model_memory_used` 归 0），但**不会自动重载**。
    所以正确顺序是「写 settings → 条件 load」，**不是**「unload → load」——
    多做的那一次 unload 会因为模型已经不在 loaded 列表里而拿到 HTTP 400。

    直接复用 `Ladder.activate()`：它就是
    「_put_settings → 不在 loaded 就 load → assert_single_resident」，
    这条路径已经被四轮对照实验验证过几百次，不要自己重写。
    """
    lad.depth = depth
    lad.activate()          # 内部已回读校验 settings + 单驻断言


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="4bit", choices=list(L.TIERS))
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--contexts", nargs="+", default=["1k", "16k"],
                    choices=["1k", "4k", "16k", "32k"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=180.0,
                    help="每组之间静置秒数（P18 要求 ≥180）")
    ap.add_argument("--precool", type=float, default=240.0,
                    help="开跑前预冷却秒数。上一轮实验跑完 GPU 是热的，"
                         "不预冷会让整轮系统性偏低（实测漂移可达 -35%）")
    ap.add_argument("--quick", action="store_true", help="冒烟：1 深度 1 上下文 2 次")
    ap.add_argument("--out", default=os.path.join(RESULTS, "bench_speed.json"))
    a = ap.parse_args()

    if a.quick:
        a.depths, a.contexts, a.reps, a.cooldown, a.precool = [1], ["1k"], 2, 5.0, 0.0

    CTX = {"1k": 1024, "4k": 4096, "16k": 16384, "32k": 32768}
    os.makedirs(RESULTS, exist_ok=True)

    # 开跑前先让机器冷下来。
    # 实测教训：--quick 只静置 5 秒时，回测回归测到 -34.8% 的漂移
    # （19.06 → 12.43 tok/s）—— 前面几轮实验已经把 GPU 跑热了。
    # 「组间」有冷却不够，**「上一轮实验结束到本轮开始」也要有**。
    if not a.quick and a.precool > 0:
        print(f"开跑前预冷却 {a.precool:.0f}s（上一轮实验可能已把 GPU 跑热）…", flush=True)
        time.sleep(a.precool)

    print(f"档位 {a.tier}  depths={a.depths}  contexts={a.contexts}  reps={a.reps}")
    print(f"静置 {a.cooldown:.0f}s/组   回测回归阈值 5%")

    # gpu_keep_warm 不关，「静置」就静不下来 —— 必须在开跑前处理
    if set_gpu_keep_warm(0.0):
        print("✅ server.gpu_keep_warm_interval 已显式设为 0")
    else:
        print("⚠️  未能把 gpu_keep_warm_interval 设为 0（该 admin 路径在本版本是 404）")
        print("    静置期间 GPU 可能仍被 keep-warm 占用，降频恢复不完整；")
        print("    **这会系统性地压低每一组的读数**——但对「组间比较」影响较小，")
        print("    因为每组承受的条件相同。回测回归会兜住整体漂移。")

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

    prompts = {c: make_prompt(CTX[c]) for c in a.contexts}
    rows: list[dict] = []
    order = [(d, c) for d in a.depths for c in a.contexts]

    for i, (d, c) in enumerate(order):
        if i:
            print(f"  静置 {a.cooldown:.0f}s …", flush=True)
            time.sleep(a.cooldown)
        # depth 变了必须重载，否则还是旧 graph
        apply_depth(lad, d)
        m = measure(lad, prompts[c], a.reps)
        row = {"tier": a.tier, "depth": d, "context": c, **m}
        rows.append(row)
        a_txt = f"{m['alpha_weighted']*100:.1f}%" if m["alpha_weighted"] is not None else "-"
        print(f"  depth={d} ctx={c:4s}  tps中位={m['tps_median']}  "
              f"IQR={m['tps_iqr']}  α={a_txt}  ttft={m['ttft_median']}s  n={m['n']}")
        sys.stdout.flush()
        json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)

    # ---- 回归回测：重测第一组，漂移 >5% 说明整轮被热污染 ----
    print(f"\n回测第一组（depth={order[0][0]} ctx={order[0][1]}）…")
    time.sleep(a.cooldown)
    apply_depth(lad, order[0][0])
    re = measure(lad, prompts[order[0][1]], a.reps)
    first = rows[0]
    drift = None
    if first["tps_median"] and re["tps_median"]:
        drift = (re["tps_median"] - first["tps_median"]) / first["tps_median"] * 100
    print(f"  首测 {first['tps_median']}  回测 {re['tps_median']}  "
          f"漂移 {drift:+.1f}%" if drift is not None else "  无法比较")
    verdict = "✅ 可用" if drift is not None and abs(drift) <= 5 else \
              "❌ 漂移 >5%，整轮作废（热污染），需加大 --cooldown 重跑"
    print(f"  判定：{verdict}")

    json.dump({"rows": rows, "regression": re, "drift_pct": drift,
               "verdict": verdict, "config": {
                   "tier": a.tier, "depths": a.depths, "contexts": a.contexts,
                   "reps": a.reps, "cooldown": a.cooldown}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
