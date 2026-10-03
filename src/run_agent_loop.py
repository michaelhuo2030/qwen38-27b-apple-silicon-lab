#!/usr/bin/env python3
"""
真实多轮 agent loop runner —— 逐档对照「能不能用工具把活干完」。

用法：
    python3 run_agent_loop.py                          # 四档，1 次
    python3 run_agent_loop.py --tiers ternary 8bit --reps 3
    python3 run_agent_loop.py --max-turns 20

与 run_lowbit / run_agentic 同样的纪律：
  * 单驻硬闸 assert_single_resident()
  * 换档后过哨兵 sentinel()
  * 结果累加带 task_version 指纹
  * transport 错误单列，不记成"模型失败"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_loop as G          # noqa: E402
import ladder27 as L             # noqa: E402
import omlx_client as C          # noqa: E402

RESULTS = os.path.expanduser("~/mtp_depth_lab/results")
ENDPOINT = "http://127.0.0.1:8091/v1/chat/completions"

SYSTEM = (
    "你是一个能使用工具的编程助手。工具会自动执行并把结果返回给你。\n"
    "规则：\n"
    "1. 先用工具收集信息，再动手改，最后用工具验证。\n"
    "2. 需要修改文件时必须调用 write_file，不要只在对话里贴代码。\n"
    "3. 改完必须调用 run_tests 确认通过。\n"
    "4. 确认通过后，用一两句话总结你改了什么，然后停止调用工具。"
)

# 长程场景额外强调：必须先搜、只改真正不一致的地方。
# 「全部改一遍」在这个场景会被对照项判负，所以这句不是提示而是判据。
SYSTEM_LONG = (
    "你是一个能使用工具的编程助手。工具会自动执行并把结果返回给你。\n"
    "规则：\n"
    "1. 先用 search_docs / list_dir / read_file 收集信息，再动手改。\n"
    "2. 需要修改文件时必须调用 write_file，不要只在对话里贴代码。\n"
    "3. **只修改真正与文档不一致的地方**。已经一致的地方不要动。\n"
    "4. 可能有**多处**不一致，要全部找出来并修完。\n"
    "5. 改完必须调用 run_tests 确认全部通过，"
    "然后用一两句话总结改了什么，然后停止调用工具。"
)


def probe_for(module: str):
    """返回 (探针模块, SYSTEM 提示词, 默认输出文件名)。

    short → agent_loop：单文件单 bug，6 轮量级
    long  → agent_long ：多文件多漂移 + 对照项 + search_docs，需 20+ 轮
    """
    if module == "short":
        return G, SYSTEM, "agent_loop.json"
    import agent_long as GL
    return GL, SYSTEM_LONG, "agent_long.json"


def _api_key() -> str:
    return L._api_key()


def chat(messages: list[dict], tools: list[dict], max_tokens: int,
         think: bool = False) -> dict:
    """一次带原生 tool schema 的请求。返回 {tool_calls, content, error}。"""
    body = {
        "model": C.MODEL_ID,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_api_key()}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}",
                "wall": round(time.time() - t0, 2)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "wall": round(time.time() - t0, 2)}
    if "error" in d:
        return {"error": str(d["error"])[:300], "wall": round(time.time() - t0, 2)}
    m = d["choices"][0]["message"]
    u = d.get("usage", {})
    return {
        "tool_calls": m.get("tool_calls") or [],
        "content": m.get("content") or "",
        "completion_tokens": u.get("completion_tokens"),
        "wall": round(time.time() - t0, 2),
    }


def run_episode(scenario: dict, think: bool, max_turns: int,
                probe=None, system: str = None) -> dict:
    """跑完一整局：多轮 + 工具执行 + 结果回灌。"""
    P = probe or G
    SYS = system or SYSTEM
    vfs = P.make_vfs(scenario["files"])
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"[run/{time.time_ns()}] {scenario['task']}"},
    ]
    transcript: list[dict] = []
    total_ctok = 0
    error = None

    for turn in range(max_turns):
        r = chat(msgs, P.TOOLS, max_tokens=768, think=think)
        total_ctok += r.get("completion_tokens") or 0
        if r.get("error"):
            error = r["error"]
            break
        tcs = r.get("tool_calls") or []
        if not tcs:
            # 没有工具调用 = 模型自己收工了
            msgs.append({"role": "assistant", "content": r.get("content", "")})
            transcript.append({"name": "__final__", "args": {},
                               "result": r.get("content", "")[:200],
                               "final_text": r.get("content", "")})
            break
        msgs.append({"role": "assistant", "content": r.get("content") or "",
                     "tool_calls": tcs})
        for tc in tcs:
            fn = (tc.get("function") or {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            # agent_long 的 _mk 需要 scenario 才能跑 run_tests，签名不同；
            # 这里按探针类型分派，别让两个模块的假设互相污染
            rec = (P._mk_l(vfs, scenario, name, **args) if hasattr(P, "_mk_l")
                   else P._mk(vfs, name, **args))
            transcript.append(rec)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", "x"),
                         "name": name, "content": str(rec["result"])[:2000]})
    else:
        error = error or f"max_turns({max_turns}) reached"

    m = P.episode_metrics(transcript, vfs, scenario)
    m["max_turns"] = max_turns
    m["completion_tokens"] = total_ctok
    m["error"] = error
    m["final_text"] = (transcript[-1].get("final_text", "")[:400]
                       if transcript and transcript[-1].get("name") == "__final__"
                       else "")
    m["transcript"] = [{"name": c.get("name"),
                        "args": {k: (str(v)[:200]) for k, v in (c.get("args") or {}).items()},
                        "result": str(c.get("result"))[:300]} for c in transcript]
    m["final_vfs"] = {k: v[:400] for k, v in vfs.items()}
    return m


def task_version(probe=None, system: str = None) -> str:
    """场景 + 工具 schema + system prompt 的指纹。任一改动即换版。"""
    P = probe or G
    SYS = system or SYSTEM
    h = hashlib.sha256()
    for s in P.LONG_TASKS if hasattr(P, "LONG_TASKS") else P.SCENARIOS:
        h.update(f"{s['key']}|{s['task']}|".encode())
        for p, c in sorted(s["files"].items()):
            h.update(f"{p}={c}|".encode())
        for c in s.get("checks", []):
            h.update(f"{c['name']}={c['fn']}{tuple(c['args'])}->{c['want']}|".encode())
        for c in s.get("checks", []) or ([{"file": s["target"], "fn": s["fn_name"]}] if "target" in s else []):
            h.update(f"t={c['file']}:{c['fn']}|".encode())
    h.update(json.dumps(P.TOOLS, sort_keys=True, ensure_ascii=False).encode())
    h.update(SYS.encode())
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["short", "long"], default="short",
                    help="short=单文件单 bug(6轮)  long=多文件多漂移+对照项(20+轮)")
    ap.add_argument("--tiers", nargs="+", default=["ternary", "4bit", "6bit", "8bit"],
                    choices=list(L.TIERS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-turns", type=int, default=24,
                    help="长程场景需要更多轮，默认给到 24（短程 16 也够）")
    ap.add_argument("--settle", type=float, default=15.0)
    ap.add_argument("--no-sentinel", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    P, SYS, defname = probe_for(a.module)
    scenarios = P.LONG_TASKS if hasattr(P, "LONG_TASKS") else P.SCENARIOS
    a.out = a.out or os.path.join(RESULTS, defname)

    os.makedirs(RESULTS, exist_ok=True)
    ver = task_version(P, SYS)
    rows: list[dict] = []
    if os.path.exists(a.out):
        try:
            rows = [r for r in json.load(open(a.out)) if r.get("task_version") == ver]
        except Exception:
            rows = []
    print(f"场景版本 {ver}；沿用同版本旧结果 {len(rows)} 行")
    print(f"探针 {a.module}｜档位 {a.tiers}｜每场景每档 {a.reps} 次｜"
          f"上限 {a.max_turns} 轮｜thinking {'on' if a.think else 'off'}")

    for ti, tier in enumerate(a.tiers):
        if ti and a.settle:
            time.sleep(a.settle)
        print(f"\n{'='*70}\n档位 {tier}  ({L.TIERS[tier]['id']}, {L.TIERS[tier]['gb']} GB)\n{'='*70}")
        lad = L.Ladder(tier, think=a.think, ane=True, turboquant=0,
                       mtp=True, verbose=False)
        try:
            lad.activate()
            C.MODEL_ID = lad.model_id
            lad.assert_single_resident()
        except Exception as e:
            print(f"  激活/单驻失败：{type(e).__name__}: {e}")
            rows.append({"tier": tier, "task_version": ver, "scenario": "*",
                         "error": f"activate failed: {e}", "solved": False})
            continue

        if not a.no_sentinel:
            sent = lad.sentinel()
            if sent.failed or not sent.ok:
                print(f"  哨兵未通过：{sent.failed}")
                rows.append({"tier": tier, "task_version": ver, "scenario": "*",
                             "error": "sentinel failed", "solved": False})
                continue
            print(f"  哨兵 {len(sent.passed)}/{len(sent.passed)+len(sent.failed)} 通过")

        cfg = lad.describe()
        for rep in range(a.reps):
            for sc in scenarios:
                m = run_episode(sc, a.think, a.max_turns, probe=P, system=SYS)
                m["tier"] = tier
                m["rep"] = rep
                m["scenario"] = sc["key"]
                m["task_version"] = ver
                m["config"] = {k: cfg.get(k) for k in
                               ("model_id", "weights_gb", "bpw", "sampling",
                                "mtp_enabled", "turboquant_kv_enabled",
                                "qwen35_ane_prefill_enabled")}
                rows.append(m)
                if m.get("error"):
                    extra = f"  {m['error'][:44]}"
                    mark = "ERR "
                else:
                    mark = "OK  " if m["solved"] else "FAIL"
                    flags = []
                    if m["read_only_episode"]:
                        flags.append("只读不写")
                    if m["looped"]:
                        flags.append(f"兜圈p{m['loop_period']}")
                    # premature_done 只有短程探针有这个指标；长程探针改看
                    # 对照项是否被改坏（过度修改）—— 用 [] 而不是 m["x"]，
                    # 否则换探针就 KeyError（已踩过一次）
                    if m.get("premature_done"):
                        flags.append("早退")
                    if m.get("control_broken"):
                        flags.append(f"改坏对照{m['control_broken']}")
                    if m.get("drift_total"):
                        flags.append(f"漂移{m['drift_fixed']}/{m['drift_total']}")
                    if m.get("n_search"):
                        flags.append(f"搜{m['n_search']}")
                    if m.get("repeat_call_count"):
                        flags.append(f"重复{m['repeat_call_count']}")
                    extra = (f"  {m['turns']}轮 写{m['n_write']} 读{m['n_read']} "
                             f"测{m['n_test']}  {'/'.join(flags) if flags else '干净'}")
                print(f"  rep{rep} {mark} {sc['key']:22s}{extra}")
                sys.stdout.flush()
        json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)

    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")

    # 汇总
    tiers = [t for t in a.tiers if any(r.get("tier") == t for r in rows)]
    print(f"\n{'='*70}\n社区指控直接对照\n{'='*70}")
    head = (f"{'档位':10s}{'n':>4s}{'解决':>7s}{'写过文件':>10s}{'只读不写':>10s}"
            f"{'兜圈':>7s}{'改坏对照':>9s}{'搜索次数':>10s}{'检查通过':>12s}{'平均轮数':>9s}"
            if a.module == "long" else
            f"{'档位':10s}{'n':>4s}{'解决':>7s}{'写过文件':>10s}{'只读不写':>10s}"
            f"{'兜圈':>7s}{'早退':>7s}{'平均轮数':>9s}")
    print(head)
    for t in tiers:
        c = [r for r in rows if r.get("tier") == t and not r.get("error")]
        n = len(c)
        if not n:
            continue
        f = lambda k: sum(1 for r in c if r.get(k))
        if a.module == "long":
            chk = sum(r.get("checks_passed", 0) for r in c)
            tot = sum(r.get("checks_total", 0) for r in c)
            print(f"{t:10s}{n:>4d}{f('solved'):>7d}{f('wrote_any_file'):>10d}"
                  f"{f('read_only_episode'):>10d}{f('looped'):>7d}"
                  f"{f('control_broken'):>7d}{f('n_search') / n:>9.1f}"
                  f"{chk:>6d}/{tot:<4d}{sum(r['turns'] for r in c)/n:>7.1f}")
        else:
            print(f"{t:10s}{n:>4d}{f('solved'):>7d}{f('wrote_any_file'):>10d}"
                  f"{f('read_only_episode'):>10d}{f('looped'):>7d}"
                  f"{f('premature_done'):>7d}{sum(r['turns'] for r in c)/n:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
