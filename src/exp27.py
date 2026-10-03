"""exp27 —— Qwen3.8-27B 实验执行器。

一个入口跑完 TEST_PLAN_27B.md §3 的 E0–E7，每个子命令都能
`--self-check` 在**没有模型、没有 oMLX** 的情况下验证参数与配置逻辑。

    python3 exp27.py --self-check                 # 全部子命令的配置自检
    python3 exp27.py smoke     --tier ternary
    python3 exp27.py sentinel  --tier ternary
    python3 exp27.py depth     --tier 4bit --depths 0 1 2 3 4
    python3 exp27.py ane       --tier 4bit
    python3 exp27.py tqkv      --tier 6bit
    python3 exp27.py quality   --tier 4bit 6bit 8bit
    python3 exp27.py concurrency --tier 4bit

所有子命令共用同一套纪律：
  * 测量前 assert_single_resident()（两个模型服务器同驻 = 数据作废）
  * 配置变更后先过 sentinel()（TurboQuant / ANE 可能静默改输出）
  * settings 写完断言回读（吞异常 = 整轮跑在旧配置上）
  * 结果里带上 n、标准误、截断标记，便于事后判有效性
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ladder27 import (  # noqa: E402
    Answer, Ladder, SAMPLING, TIERS, status, weighted_alpha,
)

OUTDIR = os.path.expanduser("~/mtp_depth_lab/results")
os.makedirs(OUTDIR, exist_ok=True)


def _save(name: str, payload: dict) -> str:
    p = os.path.join(OUTDIR, f"{name}.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return p


def _stdev(xs: list[float]) -> float | None:
    return round(statistics.stdev(xs), 3) if len(xs) > 1 else None


def _note(row: dict) -> dict:
    """给一行结果补上有效性判据需要的字段（TEST_PLAN §4）。"""
    row["n"] = row.get("n_ok", 0)
    row["valid"] = bool(
        row.get("n_ok", 0) > 0
        and not row.get("truncated")
        and not row.get("error"))
    return row


# ================================================================ E0 冒烟

def cmd_smoke(a) -> int:
    L = Ladder(a.tier, think=False, ane=a.ane, turboquant=a.tqkv,
               mtp=True, verbose=True)
    print(f"档位: {L.describe()}")
    t0 = time.time()
    L.activate()
    load_s = time.time() - t0

    st = status()
    ans = L.ask("Reply with exactly: ok", max_tokens=16)
    print(f"\n加载 {load_s:.0f}s   内存 "
          f"{st['model_memory_used']/2**30:.1f}/{st['model_memory_max']/2**30:.1f} GiB")
    print(f"生成: {ans.text!r}  err={ans.error}")

    ok = bool(ans.text) and ans.error is None
    rep = L.sentinel()
    print(f"哨兵: {'通过' if rep.ok else '失败'}  "
          f"passed={rep.passed} failed={rep.failed}")

    p = _save("smoke", {
        "tier": a.tier, "config": L.describe(), "load_seconds": round(load_s, 1),
        "memory_gib": round(st["model_memory_used"] / 2**30, 2),
        "smoke_ok": ok, "sentinel_ok": rep.ok,
        "sentinel_failed": rep.failed, "answer": ans.to_dict(),
        "kernels": st.get("custom_kernels"),
    })
    print(f"\n写入 {p}")
    return 0 if (ok and rep.ok) else 1


# ================================================================ E1 哨兵

def cmd_sentinel(a) -> int:
    L = Ladder(a.tier, think=False, ane=a.ane, turboquant=a.tqkv, verbose=True)
    L.activate()
    rep = L.sentinel()
    print(f"哨兵结果: {'通过' if rep.ok else '失败'}")
    for k, v in rep.outputs.items():
        print(f"  {k:<11} {v[:90]!r}")
    if rep.failed:
        print("失败项:")
        for f in rep.failed:
            print("  -", f)
    print("\n提示：同一配置 T=0 两次输出应当逐字一致。"
          "不一致说明配置里还有变量在漂，先找到它再谈性能。")
    return 0 if rep.ok else 1


# ================================================================ E2 MTP depth

def cmd_depth(a) -> int:
    from tasks import TASKS
    L = Ladder(a.tier, think=False, ane=a.ane, turboquant=a.tqkv,
               mtp=True, verbose=True)
    rows = L.sweep_depth(a.depths, TASKS, reps=a.reps)
    L.activate()
    L.depth = a.depths[-1]
    L._put = None                     # 仅为可读性：depth 已在 sweep 内写完

    # 按 depth 汇总：α 与 tok/s 都要带 n 与 spread
    summary = []
    for d in a.depths:
        sel = [r for r in rows if r["depth"] == d]
        al = [r["alpha"] for r in sel if r["alpha"] is not None]
        tp = [r["gen_tps_mean"] for r in sel if r["gen_tps_mean"]]
        mtp_lines = sum(r["n_mtp_lines"] for r in sel)
        summary.append({
            "depth": d,
            "alpha_weighted": round(statistics.mean(al), 2) if al else None,
            "alpha_spread": _stdev(al),
            "gen_tps_mean": round(statistics.mean(tp), 2) if tp else None,
            "gen_tps_spread": _stdev(tp),
            "n_mtp_lines": mtp_lines,
            # V8：0 行与「MTP 没跑」无法区分
            "valid": mtp_lines > 0 or d == 0,
        })
    for s in summary:
        flag = "" if s["valid"] else "  ⚠️ 无 MTP 日志行，这格 α 不可信"
        print(f"  depth={s['depth']}  α={s['alpha_weighted']} "
              f"±{s['alpha_spread']}  gen={s['gen_tps_mean']} "
              f"±{s['gen_tps_spread']} tok/s  日志行={s['n_mtp_lines']}{flag}")

    p = _save("mtp_depth", {"tier": a.tier, "config": L.describe(),
                            "rows": rows, "summary": summary})
    print(f"\n写入 {p}")
    print("判读：最优 depth 看 gen_tps 的拐点，不是 α 最高的那个。"
          "α 高但 tok/cycle 低的深度可能不划算。")
    return 0


# ================================================================ E3 ANE A/B

def cmd_ane(a) -> int:
    """同一天、同 prompt、只差 ane 一个开关。"""
    LEN_PROMPT = ("请逐行解释下面这段代码的每一行为何这样写：\n" + "\n".join(
        f"def f{i}(x):\n    return x * {i}" for i in range(1, 40)))
    rows = []
    for ane in (False, True):
        L = Ladder(a.tier, think=False, ane=ane, turboquant=a.tqkv,
                   mtp=True, verbose=True)
        L.activate()
        rep = L.sentinel()
        if not rep.ok:
            print(f"  ANE={ane}: 哨兵未通过 {rep.failed}，跳过性能测量")
            rows.append({"ane": ane, "skipped": "sentinel failed",
                         "sentinel_failed": rep.failed})
            continue
        # 长输入测 prefill
        pf = [L.ask(LEN_PROMPT, max_tokens=200) for _ in range(2)]
        # 长输出测稳态 decode
        dg = [L.ask("Write a 600-line Python module implementing a "
                    "thread-safe rate limiter, with docstrings.",
                    max_tokens=1800) for _ in range(a.reps)]
        ttfts = [x.ttft for x in pf if x.ttft]
        pps = [(x.prompt_tokens / x.ttft) if x.ttft else None for x in pf]
        dts = [x.gen_tps for x in dg if x.gen_tps]
        rows.append({
            "ane": ane, "n_ok": len([x for x in dg if x.error is None]),
            "sentinel_ok": True,
            "prefill_tps": round(statistics.mean([p for p in pps if p]), 1)
            if any(pps) else None,
            "prefill_spread": _stdev([p for p in pps if p]),
            "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
            "gen_tps_mean": round(statistics.mean(dts), 2) if dts else None,
            "gen_tps_spread": _stdev(dts),
            "truncated": any(x.truncated for x in dg),
        })
        print(f"  ANE={ane}: prefill {rows[-1]['prefill_tps']} t/s  "
              f"ttft {rows[-1]['ttft_mean']}s  "
              f"gen {rows[-1]['gen_tps_mean']} ±{rows[-1]['gen_tps_spread']} tok/s")

    p = _save("ane_ab", {"tier": a.tier, "rows": rows})
    print(f"\n写入 {p}")
    if len(rows) == 2 and rows[0].get("gen_tps_mean") and rows[1].get("gen_tps_mean"):
        r = rows[1]["gen_tps_mean"] / rows[0]["gen_tps_mean"]
        print(f"ANE decode 增益 {r:.2f}×"
              f"（M4 Max 128GB 上的社区值是 1.11×，M2 Max 的 ANE 只有 16 核，不可外推）")
    return 0


# ================================================================ E4 TurboQuant

def cmd_tqkv(a) -> int:
    rows = []
    for tq in (None, 4, 8):
        L = Ladder(a.tier, think=False, ane=a.ane, turboquant=tq, mtp=True,
                   verbose=True)
        L.activate()
        rep = L.sentinel()
        if not rep.ok:
            # 27B 是混合架构：48 层 GDN 没有 KV，只有 16 层全注意力有。
            # 量化 kernel 若没正确处理这个结构，**不会报错，只会输出变差**。
            print(f"  tqkv={tq}: 哨兵未通过 {rep.failed}")
            print("     → 混合架构下 KV 量化可能不正确，停止该档")
            rows.append({"tqkv": tq, "skipped": "sentinel failed",
                         "sentinel_failed": rep.failed,
                         "sentinel_outputs": rep.outputs})
            continue
        st = status()
        ans = [L.ask("Write a Python function that merges two sorted lists. "
                     "Code only.", max_tokens=400) for _ in range(2)]
        good = [x for x in ans if x.error is None]
        rows.append({
            "tqkv": tq, "n_ok": len(good), "sentinel_ok": True,
            "memory_gib": round(st["model_memory_used"] / 2**30, 2),
            "gen_tps_mean": round(statistics.mean(
                [x.gen_tps for x in good if x.gen_tps]), 2) if good else None,
            "truncated": any(x.truncated for x in ans),
        })
        print(f"  tqkv={tq}: gen {rows[-1]['gen_tps_mean']} tok/s  "
              f"内存 {rows[-1]['memory_gib']} GiB")
    p = _save("tqkv_ab", {"tier": a.tier, "rows": rows})
    print(f"\n写入 {p}")
    return 0


# ================================================================ E5 质量

def cmd_quality(a) -> int:
    from tasks import TASKS
    try:
        from quality_checks import check
    except ImportError:
        print("quality_checks 导入失败，先确认 src/ 在 sys.path")
        return 2
    all_rows = []
    for tier in a.tiers:
        L = Ladder(tier, think=False, ane=a.ane, turboquant=a.tqkv,
                   mtp=True, verbose=True)
        L.activate()
        if not L.sentinel().ok:
            print(f"[{tier}] 哨兵未通过，跳过质量测量")
            continue
        for key, ent, prompt, mt in TASKS:
            ok_n = 0
            errs: list[str] = []
            tps = []
            for _ in range(a.reps):
                ans = L.ask(prompt, max_tokens=mt)
                if ans.error:
                    errs.append(ans.error)
                    continue
                if ans.gen_tps:
                    tps.append(ans.gen_tps)
                try:
                    res = check(key, ans.text)
                    if res.get("passed"):
                        ok_n += 1
                except Exception as e:            # noqa: BLE001
                    errs.append(f"checker: {type(e).__name__}: {e}")
            n = a.reps
            p_hat = ok_n / n if n else 0.0
            # 报比例必须同时报 n 与标准误，否则 1–3 个点的差异没有意义
            se = (p_hat * (1 - p_hat) / n) ** 0.5 if n else None
            all_rows.append(_note({
                "tier": tier, "task": key, "entropy": ent,
                "n": n, "n_ok": ok_n, "rate": round(p_hat, 3),
                "stderr": round(se, 4) if se is not None else None,
                "gen_tps": round(statistics.mean(tps), 2) if tps else None,
                "errors": errs[:3],
            }))
            print(f"  [{tier:<8}] {key:<24} {ok_n}/{n} "
                  f"= {p_hat:.0%} ±{(se or 0)*100:.1f}pt")
    p = _save("quality_ladder", {"tiers": a.tiers, "reps": a.reps,
                                 "rows": all_rows})
    print(f"\n写入 {p}")
    print("判读：差异小于 2×stderr 就是不显著，不要挑显著项报告。"
          "低位元量化最容易掉的是 tool calling / 精确输出，重点看那类任务。")
    return 0


# ================================================================ E7 并发

def cmd_concurrency(a) -> int:
    L = Ladder(a.tier, think=False, ane=a.ane, turboquant=a.tqkv,
               mtp=True, verbose=True)
    L.activate()
    rows = L.sweep_concurrency(
        "Explain what a Bloom filter is and when to use one.",
        max_tokens=400, sizes=tuple(a.sizes))
    p = _save("concurrency", {"tier": a.tier, "rows": rows})
    print(f"\n写入 {p}")
    print("判读：找 aggregate_tps 的拐点。拐点之后只有延迟在涨，"
          "吞吐增益 <10% 就该停在那儿，并据此改 ask.py 的 MAX_WORKERS。")
    print("注意：125B 的 4 是 78GB 权重摊不动的结果，不是原理；"
          "27B 只有 13.9–30GB，同型号社区实测 8 并发有 6.49× 加速。")
    return 0


# ================================================================ 自检

def self_check() -> int:
    """没有模型、没有 oMLX 也能验证参数与配置逻辑。"""
    bad = []
    cases = [
        ("ane", ["--tier", "4bit"], 0),
        ("ane", ["--tier", "3bit"], 1),          # 未知档位
        ("tqkv", ["--tier", "6bit", "--aq", "0"], 0),
        ("tqkv", ["--tier", "6bit", "--aq", "5"], 1),   # tqkv 非法值
        ("depth", ["--tier", "4bit", "--depths", "0", "1"], 0),
        ("depth", ["--tier", "4bit", "--depths", "0", "-1"], 1),  # 负 depth
        ("depth", ["--tier", "4bit", "--depths"], 1),          # 空 depth 列表
        ("quality", ["--tiers", "4bit"], 0),
        ("quality", ["--tiers", "9bit"], 1),
        ("concurrency", ["--tier", "4bit", "--sizes", "1", "2"], 0),
        ("concurrency", ["--tier", "4bit", "--sizes", "0"], 1),  # n=0 非法
        ("smoke", ["--tier", "ternary", "--ane"], 0),
        ("sentinel", ["--tier", "ternary"], 0),
    ]
    for cmd, argv, want in cases:
        try:
            a = _parse(cmd, argv)
            errs = validate_args(a)
            got = 0 if not errs else 1
            detail = f" ({'; '.join(errs)})" if errs else ""
        except SystemExit as e:
            got = e.code or 0
            detail = " (argparse 拒绝)"
        except Exception:                          # noqa: BLE001
            got, detail = 2, " (异常)"
        okk = (got == 0) if want == 0 else (got != 0)
        if not okk:
            bad.append(f"{cmd} {' '.join(argv)}: 期望 "
                       f"{'成功' if want == 0 else '拒绝'}，实际 exit={got}{detail}")

    # 采样表与档位表的一致性
    if set(SAMPLING) != {"frozen", "think_on", "think_off"}:
        bad.append(f"采样组缺失: {set(SAMPLING)}")
    if len(TIERS) != 4:
        bad.append(f"档位数应为 4，实际 {len(TIERS)}")

    # 关键：非 self-check 的命令在真的执行时必须先过配置校验
    try:
        Ladder("4bit", ane=True, oq_a8=True, verbose=False)
        bad.append("互斥配置没被拦下")
    except Exception:                              # noqa: BLE001
        pass

    if bad:
        print("exp27 self-check 失败：")
        for b in bad:
            print("  -", b)
        return 1
    print(f"exp27 self-check clean: {len(cases)} 个参数用例"
          f"（合法接受 / 非法拒绝），配置校验与表一致性通过")
    return 0


def validate_args(a) -> list[str]:
    """参数门禁。返回错误列表，空 = 通过。

    抽成独立函数是为了**能被 self_check 直接测到**。之前这些检查写在
    `main()` 里，自检只 parse 不执行，于是"负 depth 会被拒绝"这条
    自检永远通过 —— 守卫没被证明会开火，等于没有守卫。
    """
    errs = []
    if getattr(a, "aq", 4) == 0:
        a.tqkv = None
    if a.cmd == "depth":
        d = getattr(a, "depths", None)
        if not d:
            errs.append("--depths 不能为空")
        elif any(x < 0 for x in d):
            errs.append(f"depth 不能为负，收到 {d}")
    if a.cmd == "concurrency":
        s = getattr(a, "sizes", None)
        if not s or any(n <= 0 for n in s):
            errs.append(f"并发档位必须全为正，收到 {s}")
    if a.cmd in ("smoke", "sentinel", "depth", "ane", "tqkv", "concurrency"):
        if not getattr(a, "tier", None):
            errs.append(f"{a.cmd} 需要 --tier")
    if a.cmd == "quality" and not getattr(a, "tiers", None):
        errs.append("quality 需要 --tiers")
    if a.reps is not None and a.reps < 1:
        errs.append(f"--reps 必须 ≥1，收到 {a.reps}")
    return errs


def _parse(cmd: str, argv: list[str]):
    ap = build_parser()
    return ap.parse_args([cmd, *argv])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen3.8-27B 实验执行器")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        """四档选择。--tier 用于单档命令，--tiers 用于多档（质量对照）。"""
        p.add_argument("--tier", choices=list(TIERS),
                       help="单档实验用这个")
        p.add_argument("--tiers", nargs="*", choices=list(TIERS),
                       help="多档对照用这个（quality 子命令）")
        p.add_argument("--ane", dest="ane", action="store_true", default=True)
        p.add_argument("--no-ane", dest="ane", action="store_false")
        # TurboQuant KV 默认**关闭**，理由三条：
        #   1. 同型号（M2 Max 38c/96GB）两份 oMLX 官方 benchmark recipe 都是 false
        #   2. 社区一致：MTP + TQ 双开会让 verify 加速失效（omlx#2215/#2782）
        #   3. 0.7.0 仍有活 bug：TQ + MTP + 2 并发会崩（omlx#3906，本地源码已确认未修）
        # 要测它必须显式 --aq 4 / --aq 8。
        p.add_argument("--aq", dest="tqkv", type=int, default=0,
                       choices=[0, 4, 8],
                       help="TurboQuant KV 位宽；0 = 关闭（默认，见上方三条理由）")
        p.add_argument("--reps", type=int, default=3)
        return p

    # 每个子命令只 add_parser 一次；额外参数在这里声明
    EXTRA: dict[str, list[tuple]] = {
        "depth":       [("--depths", {"nargs": "+", "type": int,
                                      "default": [0, 1, 2, 3, 4]})],
        "concurrency": [("--sizes", {"nargs": "+", "type": int,
                                     "default": [1, 2, 4, 8]})],
    }
    for name in ("smoke", "sentinel", "depth", "ane", "tqkv",
                 "quality", "concurrency"):
        p = common(sub.add_parser(name))
        for flag, kw in EXTRA.get(name, []):
            p.add_argument(flag, **kw)
    return ap


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("--self-check", "-h", "--help"):
        if argv and argv[0] == "--self-check":
            return self_check()
        build_parser().print_help()
        return 0

    ap = build_parser()
    a = ap.parse_args(argv)

    errs = validate_args(a)
    if errs:
        print("参数错误：")
        for e in errs:
            print("  -", e)
        return 1

    fn = {"smoke": cmd_smoke, "sentinel": cmd_sentinel, "depth": cmd_depth,
          "ane": cmd_ane, "tqkv": cmd_tqkv, "quality": cmd_quality,
          "concurrency": cmd_concurrency}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
