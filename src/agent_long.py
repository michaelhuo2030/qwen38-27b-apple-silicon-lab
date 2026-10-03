#!/usr/bin/env python3
"""
长程 agentic 场景 harness —— 护社区指控的**量级**，而不只是形态。

指控原文
--------
* AGI Hunt：agentic loop 建 ISS 追踪器，"Bonsai 连续 114 次搜索，
  一行代码都没写"
* MindStudio："struggled with a basic self-correction loop in code generation"

G1（短程场景）实测 12/12 全通过、0 只读不写
------------------------------------------
`agent_loop.py` 的 G1 是单文件单 bug，6 轮就结束。它证明了
**指控里的「只读不写」没复现**，但它是**短程**任务，
对「114 次搜索」那个量级**没有解释力**。社区描述的失败
（长时间不收敛、反复搜索、在细节里迷路）需要**长程**任务才可能暴露。

这一版加的三样东西
----------------
1. **search_docs 工具**：跨全部文件做子串检索、返回命中路径与行。
   文件多到必须先搜才能定位，「搜索」从一个可选动作变成一个真实成本项。
2. **多文件 + 多处漂移**：三个函数的取值散落在不同文件，
   正确值只写在 docs/sla.md 里。必须先搜文档 → 再读对应源码 → 再改。
3. **对照项（control）**：其中一个函数**本来就是对的**。
   如果模型「把所有可疑的地方都改一遍」，会把对的改错 —— 测试立刻挂。
   这把任务从「会不会动」升级成「**动得准不准**」。

指标（全部由记录算出）
--------------------
    checks_passed / checks_total   几处漂移被真正修对
    control_broken                 对照项被改坏（过度修改）
    wrote_any_file / read_only     指控的字面两项
    looped / loop_period           周期性兜圈（复用 _dominant_period）
    repeat_call_count              重复调用
    distinct_files_read            实际翻了多少个文件（长程程度）
    solved                         全部修对 且 没改坏对照项 且 动过文件

纪律
----
与 lowbit_probe / agentic_probe / agent_loop 一致：
**每个指标都要证明在坏样本上会开火、好样本上不误伤**，
所以 self_check 用手写的假 transcript 驱动，不碰模型。
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_loop import (  # noqa: E402
    _norm_path, _dominant_period, make_vfs, MAX_TURNS, READ_TOOLS, WRITE_TOOLS,
    _SANDBOX,
)
from lowbit_probe import _SANDBOX as _SANDBOX2  # noqa: F401,E402  (显式再引入一次，见下)

# ---- 场景文件 ------------------------------------------------------------
# 12 个文件。docs/sla.md 是唯一写着正确值的地方；
# services/ 下三个函数里两个错一个对（对照项）。

SLA_DOC = """\
# 服务 SLA 约定

本文件是配置值的唯一权威来源。代码必须与本文件保持一致。

- auth 服务：单次请求超时 **3000** 毫秒
- payment 服务：最大重试次数 **5** 次
- search 服务：单次返回条数上限 **50** 条
- cache 服务：条目上限 **1024** 条
"""

INITIAL = {
    "/repo/README.md": "# services\n\n微服务集合。配置约定见 docs/sla.md。\n",
    "/repo/docs/sla.md": SLA_DOC,
    "/repo/docs/runbook.md": "# 值班手册\n\n值班同学看这里。\n",
    "/repo/services/auth.py": (
        "def auth_timeout():\n"
        "    \"\"\"单次请求超时（毫秒）\"\"\"\n"
        "    return 5000\n"
    ),
    "/repo/services/payment.py": (
        "def retry_max():\n"
        "    \"\"\"最大重试次数\"\"\"\n"
        "    return 2\n"
    ),
    # 对照项：本来就和文档一致（50），不该被改
    "/repo/services/search.py": (
        "def search_limit():\n"
        "    \"\"\"单次返回条数上限\"\"\"\n"
        "    return 50\n"
    ),
    "/repo/services/cache.py": (
        "def cache_entries():\n"
        "    \"\"\"条目上限\"\"\"\n"
        "    return 1024\n"
    ),
    "/repo/services/util.py": "def noop():\n    return None\n",
    "/repo/services/http.py": "def get(url):\n    raise NotImplementedError\n",
    "/repo/services/logger.py": "LOG = None\n",
    "/repo/tests/test_services.py": "# 由 run_tests 工具执行\n",
    "/repo/Makefile": "test:\n\t@echo use run_tests\n",
}

# 三个被检查的函数：前两个是漂移，第三个是对照
CHECKS = [
    {"name": "auth_timeout", "file": "/repo/services/auth.py",
     "fn": "auth_timeout", "args": [], "want": 3000, "kind": "drift"},
    {"name": "retry_max", "file": "/repo/services/payment.py",
     "fn": "retry_max", "args": [], "want": 5, "kind": "drift"},
    {"name": "search_limit", "file": "/repo/services/search.py",
     "fn": "search_limit", "args": [], "want": 50, "kind": "control"},
    {"name": "cache_entries", "file": "/repo/services/cache.py",
     "fn": "cache_entries", "args": [], "want": 1024, "kind": "control"},
]

LONG_TASKS = [
    {
        "key": "H1_config_drift",
        "files": INITIAL,
        "checks": CHECKS,
        "task": (
            "你在 /repo 工作。docs/sla.md 定义了各服务的配置值，"
            "但代码里有的和文档不一致。\n"
            "请找出**所有**不一致的地方并修好，让 run_tests 通过。\n"
            "注意：文档一致的地方不要动。改完必须调用 run_tests 确认。"
        ),
    },
]

LONG_TASK_KEYS = [t["key"] for t in LONG_TASKS]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录下的文件和子目录",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文件的完整内容",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在所有文件内容里做子串检索，返回命中的文件与行。用它来找配置值。",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件（覆盖）。修配置用这个。",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"},
                                          "content": {"type": "string"}},
                           "required": ["path", "content"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "运行测试，返回逐项通过/失败",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

READ_TOOLS_L = {"list_dir", "read_file", "search_docs"}


# ---- 工具实现 ------------------------------------------------------------

def tool_search_docs(vfs: dict, query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "查询词为空"
    hits = []
    for path in sorted(vfs):
        for i, line in enumerate(vfs[path].splitlines(), 1):
            if q in line:
                hits.append(f"{path}:{i}: {line.strip()[:120]}")
    if not hits:
        return f"没有包含 {q!r} 的行"
    return f"{len(hits)} 处命中：\n" + "\n".join(hits[:20])


def _run_checks(vfs: dict, checks: list[dict]) -> list[dict]:
    """逐项真 exec。每个检查单独起一个 subprocess，互不影响。"""
    import subprocess
    out = []
    for c in checks:
        src = vfs.get(_norm_path(c["file"]))
        if src is None:
            out.append({"name": c["name"], "kind": c["kind"], "ok": False,
                        "detail": "文件不存在"})
            continue
        script = (_SANDBOX + "\nprint(json.dumps(check("
                  f"{src!r}, {[(tuple(c['args']), c['want'])]!r}, {c['fn']!r})))\n")
        try:
            r = subprocess.run([sys.executable, "-c", script],
                               capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            out.append({"name": c["name"], "kind": c["kind"], "ok": False,
                        "detail": "timeout"})
            continue
        if r.returncode != 0:
            out.append({"name": c["name"], "kind": c["kind"], "ok": False,
                        "detail": r.stderr.strip()[:100]})
            continue
        try:
            kind, detail = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            out.append({"name": c["name"], "kind": c["kind"], "ok": False,
                        "detail": "unparseable"})
            continue
        out.append({"name": c["name"], "kind": c["kind"], "ok": kind == "ok",
                    "detail": str(detail)[:90]})
    return out


def tool_run_tests(vfs: dict, task: dict) -> str:
    res = _run_checks(vfs, task["checks"])
    n_ok = sum(1 for r in res if r["ok"])
    lines = [f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['name']}: {r['detail']}"
             for r in res]
    head = f"{n_ok}/{len(res)} 通过"
    return head + ("\n" + "\n".join(lines) if n_ok != len(res) else "")


def dispatch_l(vfs: dict, task: dict, name: str, args: dict) -> str:
    if name == "list_dir":
        from agent_loop import tool_list_dir
        return tool_list_dir(vfs, args.get("path", ""))
    if name == "read_file":
        from agent_loop import tool_read_file
        return tool_read_file(vfs, args.get("path", ""))
    if name == "search_docs":
        return tool_search_docs(vfs, args.get("query", ""))
    if name == "write_file":
        from agent_loop import tool_write_file
        return tool_write_file(vfs, args.get("path", ""), args.get("content", ""))
    if name == "run_tests":
        return tool_run_tests(vfs, task)
    return f"未知工具：{name}"


def _mk_l(vfs, task, name, **args):
    rec = {"name": name, "args": args}
    if name == "write_file":
        p = _norm_path(args.get("path", ""))
        rec["no_op"] = (vfs.get(p) == args.get("content", ""))
    rec["result"] = dispatch_l(vfs, task, name, args)
    return rec


# ---- 指标 ----------------------------------------------------------------

def episode_metrics(transcript: list[dict], vfs: dict, task: dict) -> dict:
    calls = transcript
    sig = [(c.get("name"), json.dumps(c.get("args", {}), sort_keys=True, ensure_ascii=False))
           for c in calls]

    counts: dict[tuple, int] = {}
    for s in sig:
        counts[s] = counts.get(s, 0) + 1
    repeat_call_count = sum(v - 1 for v in counts.values() if v > 1)

    max_streak = streak = 1 if sig else 0
    for i in range(1, len(sig)):
        streak = streak + 1 if sig[i] == sig[i - 1] else 1
        max_streak = max(max_streak, streak)

    period = _dominant_period([f"{n}|{a}" for n, a in sig], max_p=8, min_reps=3)
    loop_cycles = (len(sig) - period) // period if period else 0

    n_read = sum(1 for c in calls if c.get("name") in READ_TOOLS_L)
    n_write = sum(1 for c in calls if c.get("name") in WRITE_TOOLS)
    n_test = sum(1 for c in calls if c.get("name") == "run_tests")
    n_search = sum(1 for c in calls if c.get("name") == "search_docs")

    wrote = [c for c in calls if c.get("name") in WRITE_TOOLS]
    wrote_any = bool(wrote)
    touched = {_norm_path((c.get("args") or {}).get("path", "")) for c in wrote}
    files_read = {_norm_path((c.get("args") or {}).get("path", ""))
                  for c in calls if c.get("name") == "read_file"}
    drift_files = {_norm_path(c["file"]) for c in task["checks"] if c["kind"] == "drift"}
    control_files = {_norm_path(c["file"]) for c in task["checks"] if c["kind"] == "control"}

    res = _run_checks(vfs, task["checks"])
    passed = [r for r in res if r["ok"]]
    drift = [r for r in res if r["kind"] == "drift"]
    control = [r for r in res if r["kind"] == "control"]
    drift_ok = sum(1 for r in drift if r["ok"])
    control_ok = sum(1 for r in control if r["ok"])

    # 「过度修改」：改了本来正确的文件（无论改后对不对，都算动了不该动的）
    over_edit = sorted(touched & control_files)
    stall = sum(1 for c in calls if c.get("name") in WRITE_TOOLS and c.get("no_op"))

    return {
        "turns": len(calls),
        "n_read": n_read, "n_write": n_write, "n_test": n_test, "n_search": n_search,
        "wrote_any_file": wrote_any,
        "read_only_episode": n_write == 0 and n_read > 0,
        "checks_passed": len(passed), "checks_total": len(res),
        "drift_total": len(drift), "drift_fixed": drift_ok,
        "control_total": len(control), "control_ok": control_ok,
        "control_broken": len(control) - control_ok,
        "over_edited_control": over_edit,
        "distinct_files_read": len(files_read),
        "distinct_files_written": len(touched),
        "touched_drift_files": sorted(touched & drift_files),
        "repeat_call_count": repeat_call_count,
        "max_identical_streak": max_streak,
        "loop_period": period, "loop_cycles": loop_cycles, "looped": bool(period),
        "stall_turns": stall,
        "test_results": res,
        # 解决 = 漂移全修对 + 对照项全没坏 + 确实动过文件
        "solved": (drift_ok == len(drift) and control_ok == len(control)
                   and len(drift) > 0 and wrote_any),
    }


# ---- 自测：手写假 transcript 证明指标会开火 -------------------------------

def _fixed(vfs, path, old, new):
    return vfs.get(path, "").replace(old, new)


GOOD_ALL = {
    "/repo/services/auth.py": INITIAL["/repo/services/auth.py"].replace("5000", "3000"),
    "/repo/services/payment.py": INITIAL["/repo/services/payment.py"].replace("return 2", "return 5"),
}

# 只改了一处
HALF_FIX = {
    "/repo/services/auth.py": INITIAL["/repo/services/auth.py"].replace("5000", "3000"),
}

# 改对了但把对的也改坏了
OVER_EDIT = {
    **GOOD_ALL,
    "/repo/services/search.py": INITIAL["/repo/services/search.py"].replace("50", "20"),
}


def good_transcript():
    t = LONG_TASKS[0]
    v = make_vfs(t["files"])
    tr = [_mk_l(v, t, "search_docs", query="超时"),
          _mk_l(v, t, "read_file", path="/repo/docs/sla.md"),
          _mk_l(v, t, "list_dir", path="/repo/services"),
          _mk_l(v, t, "read_file", path="/repo/services/auth.py"),
          _mk_l(v, t, "read_file", path="/repo/services/payment.py"),
          _mk_l(v, t, "write_file", path="/repo/services/auth.py", content=GOOD_ALL["/repo/services/auth.py"]),
          _mk_l(v, t, "write_file", path="/repo/services/payment.py", content=GOOD_ALL["/repo/services/payment.py"]),
          _mk_l(v, t, "run_tests")]
    tr[-1]["final_text"] = "两处漂移已按 docs/sla.md 修正，测试通过。"
    return tr, v


def search_only_transcript():
    """社区指控的形态：反复搜索，一行都不改。"""
    t = LONG_TASKS[0]
    v = make_vfs(t["files"])
    tr = [_mk_l(v, t, "search_docs", query="3000")]
    for _ in range(5):
        tr += [_mk_l(v, t, "search_docs", query="3000"),
               _mk_l(v, t, "search_docs", query="5000"),
               _mk_l(v, t, "run_tests")]
    tr[-1]["final_text"] = "我还在确认各处配置。"
    return tr, v


def over_editor_transcript():
    """全改了 —— 把本来正确的也改坏。"""
    t = LONG_TASKS[0]
    v = make_vfs(t["files"])
    tr = [_mk_l(v, t, "read_file", path="/repo/docs/sla.md"),
          _mk_l(v, t, "write_file", path="/repo/services/auth.py", content=GOOD_ALL["/repo/services/auth.py"]),
          _mk_l(v, t, "write_file", path="/repo/services/payment.py", content=GOOD_ALL["/repo/services/payment.py"]),
          _mk_l(v, t, "write_file", path="/repo/services/search.py", content=OVER_EDIT["/repo/services/search.py"]),
          _mk_l(v, t, "run_tests")]
    tr[-1]["final_text"] = "统一改成文档里的值。"
    return tr, v


def half_transcript():
    t = LONG_TASKS[0]
    v = make_vfs(t["files"])
    tr = [_mk_l(v, t, "read_file", path="/repo/services/auth.py"),
          _mk_l(v, t, "write_file", path="/repo/services/auth.py", content=HALF_FIX["/repo/services/auth.py"]),
          _mk_l(v, t, "run_tests")]
    tr[-1]["final_text"] = "改好了。"
    return tr, v


def self_check(verbose: bool = True) -> int:
    fails: list[str] = []

    def ck(name, cond, detail=""):
        if cond:
            if verbose:
                print(f"  ok   {name}")
        else:
            fails.append(f"{name} {detail}")
            print(f"  FAIL {name} {detail}")

    t = LONG_TASKS[0]

    # 场景前提：初始状态必须是「2 漂移坏、2 对照好」
    r0 = _run_checks(make_vfs(t["files"]), t["checks"])
    ck("初始: 4 项里 2 项失败", sum(1 for x in r0 if not x["ok"]) == 2, r0)
    ck("初始: 漂移项全坏", all(not x["ok"] for x in r0 if x["kind"] == "drift"), r0)
    ck("初始: 对照项全好", all(x["ok"] for x in r0 if x["kind"] == "control"), r0)

    g, gv = good_transcript()
    m = episode_metrics(g, gv, t)
    ck("good: solved", m["solved"], m)
    ck("good: 4/4 通过", m["checks_passed"] == 4, m["checks_passed"])
    ck("good: 漂移 2/2 修对", m["drift_fixed"] == 2, m)
    ck("good: 对照 0 破坏", m["control_broken"] == 0, m)
    ck("good: 无过度修改", not m["over_edited_control"], m["over_edited_control"])
    ck("good: not read_only", not m["read_only_episode"])
    ck("good: not looped", not m["looped"], m)
    ck("good: 用过 search_docs", m["n_search"] >= 1, m)
    ck("good: 读了多个文件", m["distinct_files_read"] >= 3, m)

    s, sv = search_only_transcript()
    m = episode_metrics(s, sv, t)
    ck("search_only: read_only fires", m["read_only_episode"], m)
    ck("search_only: not solved", not m["solved"], m)
    ck("search_only: looped fires", m["looped"], m)
    ck("search_only: loop_period == 3", m["loop_period"] == 3, m)
    ck("search_only: 0 项修对", m["drift_fixed"] == 0, m)

    o, ov = over_editor_transcript()
    m = episode_metrics(o, ov, t)
    ck("over_edit: 漂移确实修对了", m["drift_fixed"] == 2, m)
    ck("over_edit: control_broken fires", m["control_broken"] >= 1, m)
    ck("over_edit: over_edited_control 非空", bool(m["over_edited_control"]), m)
    ck("over_edit: NOT solved（改得准才算解决）", not m["solved"], m)

    h, hv = half_transcript()
    m = episode_metrics(h, hv, t)
    ck("half: 只修 1 处", m["drift_fixed"] == 1, m)
    ck("half: not solved", not m["solved"], m)
    ck("half: wrote_any_file", m["wrote_any_file"], m)

    # 关键鉴别力：good 与 over_edit 的漂移项完全一样，只有对照项不同 → solved 必须不同
    ck("discriminates over-edit",
       episode_metrics(g, gv, t)["solved"] != episode_metrics(o, ov, t)["solved"])
    ck("discriminates completeness",
       episode_metrics(g, gv, t)["solved"] != episode_metrics(h, hv, t)["solved"])

    ck("tools: 5 个", len(TOOLS) == 5)
    ck("tools: 名字齐全",
       {x["function"]["name"] for x in TOOLS} ==
       {"list_dir", "read_file", "search_docs", "write_file", "run_tests"})
    ck("tasks: 1 unique key", len(set(LONG_TASK_KEYS)) == len(LONG_TASKS))
    ck("search_docs 真的能检索",
       "sla.md" in tool_search_docs(make_vfs(t["files"]), "3000"),
       tool_search_docs(make_vfs(t["files"]), "3000")[:120])

    print()
    if fails:
        print(f"AGENT LONG SELF-CHECK: {len(fails)} FAILED")
        for x in fails:
            print("  -", x)
        return 1
    print("AGENT LONG SELF-CHECK: all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
