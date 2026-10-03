#!/usr/bin/env python3
"""
低比特质量对照 runner —— 拿我们自己的数据，不引用社区结论。

用法：
    python3 run_lowbit.py --tiers ternary 4bit 6bit
    python3 run_lowbit.py --tiers ternary 4bit --reps 3
    python3 run_lowbit.py --tier ternary            # 单档
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ladder27 as L          # noqa: E402
import lowbit_probe as P      # noqa: E402
import omlx_client as C       # noqa: E402

RESULTS = os.path.expanduser("~/mtp_depth_lab/results")

# omlx_client 的 API_KEY 只从 env 读，默认空串。ladder27 有自己的读取函数，
# 这里复用它，避免"请求发出去了但没鉴权 → 401 → 静默当成模型答错"这个坑
# （第一版 runner 就踩了：ctoks=0 / text='' / valid 都没报错，看起来像模型坏了）。
C.API_KEY = L._api_key()


def task_version() -> str:
    """
    任务集的版本指纹。

    为什么必须有：L_TASKS 改过一轮题目措辞（L1 加中间步骤、L8 修自相矛盾），
    而结果文件是**累加**的。结果就是 rep0 出现两次 —— 一次是旧题、一次是新题，
    混在同一个 n 里做统计。旧题的数据不能和新题直接平均。

    指纹 = 8 个任务 (key, prompt, max_tokens) 的 sha256 前 12 位。
    题目一改，指纹就变，历史数据自然分层。
    """
    import hashlib
    h = hashlib.sha256()
    for t in P.L_TASKS:
        h.update(f"{t[0]}|{t[2]}|{t[3]}".encode())
    return h.hexdigest()[:12]


def run_one(task, think: bool) -> dict:
    key, tier_label, prompt, max_tokens, _kind, _arg = task
    # 唯一前缀：否则测的是 prefix cache 命中率不是模型（见 E8 / P22）
    uniq = f"[probe/{time.time_ns()}] {prompt}"
    t0 = time.time()
    try:
        r = C.generate(uniq, max_tokens=max_tokens, think=think)
    except Exception as e:  # 生成失败也要记账，不能当"没跑"
        return {"task": key, "ok": False, "error": f"{type(e).__name__}: {e}",
                "text": "", "wall": round(time.time() - t0, 2)}

    # 先看传输层：失败的话**不能**记成"模型答错"，那是两回事
    if r.get("transport_error"):
        return {"task": key, "ok": False,
                "error": f"transport: {r['transport_error']}",
                "text": r.get("text", ""), "wall": r.get("wall")}

    text = r.get("text", "")
    if not text.strip():
        return {"task": key, "ok": False, "error": "empty_response",
                "text": "", "wall": r.get("wall"),
                "completion_tokens": r.get("completion_tokens")}

    ok, why = P.verify(key, text)
    flags = P.loop_flags(text)
    return {
        "task": key,
        "tier_label": tier_label,
        "ok": ok,                 # 这道题自己问的东西答对没有
        "why": why,
        "degenerate": flags["any_degenerate"],
        "flags": {k: v for k, v in flags.items() if k != "_raw"},
        "metrics": flags["_raw"],
        "completion_tokens": r.get("completion_tokens"),
        "gen_tps": r.get("gen_tps"),
        "ttft": r.get("ttft"),
        # 注意真名是 accept_pct 不是 alpha；valid 是"MTP 样本够不够"的统计标志，
        # 不是对错 —— 两者必须分开记。
        "accept_pct": r.get("accept_pct"),
        "accept_den": r.get("accept_den"),
        "tok_per_cycle": r.get("tok_per_cycle"),
        "mtp_stat_valid": r.get("valid"),
        "wall": r.get("wall"),
        "text": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["ternary", "4bit", "6bit", "8bit"],
                    choices=list(L.TIERS))
    ap.add_argument("--reps", type=int, default=1, help="每题每档重复次数（>1 看方差）")
    ap.add_argument("--think", action="store_true",
                    help="开 thinking（默认 off。社区实测 xhigh 一轮要 20K thinking token）")
    ap.add_argument("--tasks", nargs="*", default=None, help="只跑这些任务 key")
    ap.add_argument("--out", default=os.path.join(RESULTS, "lowbit.json"))
    a = ap.parse_args()

    tasks = P.L_TASKS
    if a.tasks:
        want = set(a.tasks)
        tasks = [t for t in P.L_TASKS if t[0] in want]
        if not tasks:
            print(f"没有匹配的任务，可选：{P.L_TASK_KEYS}")
            return 2

    os.makedirs(RESULTS, exist_ok=True)
    ver = task_version()
    all_rows = []
    # 上次结果用于同键覆盖；**只沿用同任务版本**的行，否则新旧题混在一起算 n
    if os.path.exists(a.out):
        try:
            all_rows = [r for r in json.load(open(a.out))
                        if r.get("task_version") == ver]
        except Exception:
            all_rows = []
    print(f"任务集版本 {ver}；沿用同版本的旧结果 {len(all_rows)} 行")

    for tier in a.tiers:
        print(f"\n{'='*62}\n档位 {tier}  ({L.TIERS[tier]['id']}, {L.TIERS[tier]['gb']} GB)\n{'='*62}")
        lad = L.Ladder(tier, think=a.think, ane=True, turboquant=0,
                       mtp=True, verbose=False)
        try:
            lad.activate()
            # 客户端用的是模块级 MODEL_ID，激活后同步过来
            C.MODEL_ID = lad.model_id
        except Exception as e:
            print(f"  激活失败：{type(e).__name__}: {e}")
            for t in tasks:
                all_rows.append({"tier": tier, "task": t[0], "ok": False,
                                 "error": f"activate failed: {e}", "text": ""})
            continue

        cfg = lad.describe()
        for rep in range(a.reps):
            for t in tasks:
                row = run_one(t, a.think)
                row["tier"] = tier
                row["rep"] = rep
                row["task_version"] = ver
                row["config"] = {k: cfg.get(k) for k in
                                 ("model_id", "weights_gb", "bpw", "sampling",
                                  "mtp_enabled", "turboquant_kv_enabled",
                                  "qwen35_ane_prefill_enabled")}
                all_rows.append(row)
                if row.get("error"):
                    mark, extra = "ERR ", f"  {row['error'][:46]}"
                else:
                    mark = "OK  " if row["ok"] else "FAIL"
                    deg = " [退化]" if row.get("degenerate") else ""
                    extra = (f"  α={row.get('accept_pct') or '-'}"
                             f"  {str(row.get('why'))[:36]}{deg}")
                print(f"  rep{rep} {mark} {t[0]:20s} {str(row.get('gen_tps') or '-'):>6} tok/s{extra}")
                sys.stdout.flush()

    json.dump(all_rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
