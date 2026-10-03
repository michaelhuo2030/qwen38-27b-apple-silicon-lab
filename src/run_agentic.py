#!/usr/bin/env python3
"""
Agentic 收敛探针 runner —— 逐档对照「多步动作是否收敛」。

护的是社区最严重、而我一条都没测过的指控：

  * AGI Hunt：agentic loop 建 ISS 追踪器，"Bonsai 连续 114 次搜索，
    一行代码都没写"；全精度模型正常交付
  * MindStudio："struggled with a basic self-correction loop in code generation"
  * 官方口径：98.2% 是 20 项均值，medium effort 只有 96.0%，
    两项 agentic coding benchmark 只有 ~75%

用法：
    python3 run_agentic.py                      # 四档，每题每档 1 次
    python3 run_agentic.py --reps 3             # 每题每档 3 次
    python3 run_agentic.py --tiers ternary 4bit
    python3 run_agentic.py --tasks A3_emit_10_countdown
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agentic_probe as A       # noqa: E402
import ladder27 as L            # noqa: E402
import omlx_client as C         # noqa: E402

RESULTS = os.path.expanduser("~/mtp_depth_lab/results")

# 复用 ladder27 的取 key 路径：omlx_client.API_KEY 只从 env 读，默认空串。
# 漏设的后果不是报错而是 401 被吞进 transport_error，表现为"模型答错"——
# 第一版 runner 踩过这个坑。
C.API_KEY = L._api_key()

# 10 步逆序那道题输出最长（10 行）。给 512 是为了让"半途卡住"能被
# termination_ok 抓到，而不是被 max_tokens 截断伪装成"模型没走完"。
MAX_TOKENS = 512

# 故意留的余量：唯一前缀 + 指令块。防止测到 prefix cache 而不是模型（E8/P22）
UNIQUIFY = True


def task_version() -> str:
    """
    任务集指纹 = 5 个 (key, prompt, required) 的 sha256 前 12 位。

    改题目即改指纹，历史结果自动分层，不会和旧题混进同一个 n。
    """
    h = hashlib.sha256()
    for key, _label, prompt, required in A.A_TASKS:
        h.update(f"{key}|{required}|{prompt}".encode())
    return h.hexdigest()[:12]


def run_one(task, think: bool) -> dict:
    key, tier_label, prompt, required = task
    body = f"[probe/{time.time_ns()}] {prompt}" if UNIQUIFY else prompt
    t0 = time.time()
    try:
        r = C.generate(body, max_tokens=MAX_TOKENS, think=think)
    except Exception as e:
        # 生成失败也要记账 —— 记成"没跑"，不能默默消失
        return {"task": key, "ok": False,
                "error": f"{type(e).__name__}: {e}", "text": "",
                "wall": round(time.time() - t0, 2)}

    if r.get("transport_error"):
        return {"task": key, "ok": False,
                "error": f"transport: {r['transport_error']}",
                "text": r.get("text", ""), "wall": r.get("wall")}

    text = r.get("text", "")
    if not text.strip():
        return {"task": key, "ok": False, "error": "empty_response",
                "text": "", "wall": r.get("wall"),
                "completion_tokens": r.get("completion_tokens")}

    ok, why = A.verify(key, text)
    flags = A.loop_flags(text, required)
    return {
        "task": key,
        "tier_label": tier_label,
        "required_steps": required,
        "ok": ok,                      # N 步全部按要求产出
        "why": why,
        "flags": {k: v for k, v in flags.items() if k != "_raw"},
        "metrics": flags["_raw"],
        "completion_tokens": r.get("completion_tokens"),
        # 下面三个只作为参考记录。**本 run 的 gen_tps 不可用于档位速度比较**：
        # 无 3 分钟冷却（P18），GPU 热降频会污染；各档负载输出长度也不同。
        # 速度对照另开实验做。
        "gen_tps_uncooled": r.get("gen_tps"),
        "ttft": r.get("ttft"),
        "accept_pct": r.get("accept_pct"),   # 真名 accept_pct，不是 alpha
        "accept_den": r.get("accept_den"),
        "tok_per_cycle": r.get("tok_per_cycle"),
        "mtp_stat_valid": r.get("valid"),    # 统计有效性，不是对错
        "wall": r.get("wall"),
        "text": text,
    }


def summarize(rows: list[dict]) -> None:
    """收尾打印：把「每档正确率」和「收敛失败形态」分开报。"""
    tiers = [t for t in L.TIERS if any(r.get("tier") == t for r in rows)]
    tasks = sorted({r["task"] for r in rows if r.get("task")})
    n_rep = max((r.get("rep", 0) for r in rows), default=-1) + 1

    print(f"\n{'='*78}\n逐档正确率（每格 {n_rep} 次/题）\n{'='*78}")
    hdr = f"{'任务':26s}" + "".join(f"{t:>11s}" for t in tiers)
    print(hdr)
    totals = {t: [0, 0] for t in tiers}
    for tk in tasks:
        line = f"{tk:26s}"
        for t in tiers:
            cell = [r for r in rows if r.get("tier") == t and r.get("task") == tk]
            good = sum(1 for r in cell if r.get("ok"))
            bad = sum(1 for r in cell if r.get("error"))
            tot = [good, len(cell) - bad]
            totals[t][0] += good
            totals[t][1] += tot[1]
            mark = f"{good}/{len(cell)}" if not bad else f"{good}/{len(cell)}!"
            line += f"{mark:>11s}"
        print(line)
    line = f"{'合计':26s}"
    for t in tiers:
        g, n = totals[t]
        line += f"{f'{g}/{n}':>11s}"
    print(line)

    # 收敛失败形态：这是本次真正要看的信号，比对错更贴近社区指控
    print(f"\n{'='*78}\n收敛失败形态（社区指控的直接对应）\n{'='*78}")
    for t in tiers:
        cell = [r for r in rows if r.get("tier") == t and not r.get("error")]
        n = len(cell)
        if not n:
            continue
        def pct(f):
            return sum(1 for r in cell if r["flags"].get(f)) * 100.0 / n
        prem = sum(1 for r in cell if r["flags"].get("premature"))
        worst = min((r["metrics"]["emitted_steps"] / max(r["required_steps"], 1)
                     for r in cell), default=0.0)
        print(f"{t:9s} n={n:3d}  兜圈 {pct('repeat_action'):5.1f}%  "
              f"乱序 {pct('out_of_order'):5.1f}%  少步收工 {prem:2d}次({prem*100.0/n:4.1f}%)  "
              f"末步没收住 {pct('not_terminated'):5.1f}%  "
              f"最长完成度 {worst*100:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["ternary", "4bit", "6bit", "8bit"],
                    choices=list(L.TIERS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--think", action="store_true",
                    help="开 thinking（默认 off）。开 thinking 会让 gen_tps 差几十倍，"
                         "质量结论也可能变，两种口径不要混")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--settle", type=float, default=15.0,
                    help="换档后静置秒数（让卸载/加载和内存回落完成，不是热冷却）")
    ap.add_argument("--no-sentinel", action="store_true",
                    help="跳过每次换档后的哨兵（调试用，会破坏 A/B 纪律）")
    ap.add_argument("--out", default=os.path.join(RESULTS, "agentic.json"))
    a = ap.parse_args()

    tasks = A.A_TASKS
    if a.tasks:
        want = set(a.tasks)
        tasks = [t for t in A.A_TASKS if t[0] in want]
        if not tasks:
            print(f"没有匹配的任务，可选：{A.A_TASK_KEYS}")
            return 2

    os.makedirs(RESULTS, exist_ok=True)
    ver = task_version()
    rows: list[dict] = []
    if os.path.exists(a.out):
        try:
            rows = [r for r in json.load(open(a.out)) if r.get("task_version") == ver]
        except Exception:
            rows = []
    print(f"任务集版本 {ver}；沿用同版本的旧结果 {len(rows)} 行")
    print(f"档位 {a.tiers}｜每题每档 {a.reps} 次｜thinking {'on' if a.think else 'off'}")
    print("注意：本 run 不做热冷却，gen_tps 不可用于速度比较（P18）")

    for ti, tier in enumerate(a.tiers):
        if ti and a.settle:
            print(f"\n静置 {a.settle:.0f}s …")
            time.sleep(a.settle)
        print(f"\n{'='*62}\n档位 {tier}  ({L.TIERS[tier]['id']}, {L.TIERS[tier]['gb']} GB)\n{'='*62}")
        lad = L.Ladder(tier, think=a.think, ane=True, turboquant=0,
                       mtp=True, verbose=False)
        try:
            lad.activate()
            # 客户端用模块级 MODEL_ID，激活后同步过来
            C.MODEL_ID = lad.model_id
            # 硬闸：单次只允许一个模型常驻。并发/换出状态下测出来的
            # 质量差是内存压力造成的，不是精度造成的
            lad.assert_single_resident()
        except Exception as e:
            print(f"  激活/单驻校验失败：{type(e).__name__}: {e}")
            for t in tasks:
                rows.append({"tier": tier, "task": t[0], "ok": False,
                             "error": f"activate failed: {e}", "text": ""})
            continue

        if not a.no_sentinel:
            # 配置变过就必须过哨兵，否则这次数据不能和别的档位相提并论。
            # 注意 sentinel() 返回 SentinelReport（.passed/.failed/.ok），
            # 不是 dict 列表 —— 写成迭代会静默拿到 0 个"元素"从而永远判过。
            sent = lad.sentinel()
            n_sent = len(sent.passed) + len(sent.failed)
            if sent.failed or not sent.ok:
                print(f"  哨兵未通过 {len(sent.failed)}/{n_sent}：{sent.failed}")
                for t in tasks:
                    rows.append({"tier": tier, "task": t[0], "ok": False,
                                 "error": f"sentinel failed: {sent.failed[:2]}",
                                 "text": ""})
                continue
            print(f"  哨兵 {len(sent.passed)}/{n_sent} 通过")

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
                rows.append(row)
                if row.get("error"):
                    mark, extra = "ERR ", f"  {row['error'][:46]}"
                else:
                    mark = "OK  " if row["ok"] else "FAIL"
                    f = row["flags"]
                    tags = "".join(k[0].upper() for k in
                                   ("repeat_action", "out_of_order", "premature",
                                    "not_terminated") if f.get(k))
                    extra = (f"  {row['metrics']['emitted_steps']}/{row['required_steps']}步"
                             f"  {row['why'][:30]}" + (f"  [{tags}]" if tags else ""))
                print(f"  rep{rep} {mark} {t[0]:24s}{extra}")
                sys.stdout.flush()
        # 每档跑完立刻落盘：中断也不丢已跑完的档
        json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)

    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
