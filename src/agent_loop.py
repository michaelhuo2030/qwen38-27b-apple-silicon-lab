#!/usr/bin/env python3
"""
真实多轮 agent 循环 harness —— 直接对着社区最严重的指控设计。

指控原文
--------
* AGI Hunt：agentic loop 建 ISS 追踪器，"Bonsai 连续 114 次搜索，
  一行代码都没写"，而全精度模型正常交付
* MindStudio："struggled with a basic self-correction loop in code generation"

为什么之前那版 agentic_probe 不够
-------------------------------
`agentic_probe` 是**把 agent 循环压成一道长题**：让模型在一个回复里输出
「步骤1..步骤N」。它测的是"能不能在单回复里走完 N 步格式"，**不是**
"能不能在多轮里用工具改东西"。指控里的"114 次搜索"发生在**多轮**中，
每一轮的结果不同、反馈会变。单回复探针结构上就测不到它。
（而且上一轮实测 60/60 全满分，探针饱和，本来也没有鉴别力。）

这一版测什么
------------
真的起一个 agent loop：给模型四个本地工具，每轮把工具结果喂回去，
让它自己决定下一步。虚拟机文件系统 + 真实执行的测试，闭环是真的。

    list_dir    列出目录
    read_file   读文件
    write_file  写文件          ← 指控里"从未发生"的那一步
    run_tests   跑测试（真 exec，subprocess + 超时）

核心指标（全部由**记录**算出，与模型无关）
------------------------------------------
    wrote_any_file         全程有没有调过 write_file
    read_only_episode     是不是"只读不写"——指控的字面形态
    edit_reached_target   有没有写到真正该改的那个文件
    tests_passed_final     最后一次 run_tests 是否通过
    solved                write + 测试通过 + 改对了
    repeat_call_count     同一个 (工具,参数) 被重复调用的次数
    max_identical_streak  连续重复同一调用的最长长度（"兜圈"的直接证据）
    stall_turns           调了工具但虚拟机没被改变的轮数
    premature_done        自己宣布完成、但测试其实没过

纪律
----
和 lowbit_probe / agentic_probe 一致：**每个指标都要证明在坏样本上会开火、
好样本上不误伤**。所以 self_check 用**手写的假 transcript**（好 agent /
兜圈 agent / 只读 agent / 早退 agent）驱动指标，不碰模型。
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 与 lowbit_probe 同一套沙箱：subprocess + 20s 超时 + json 回报
from lowbit_probe import _SANDBOX, _dominant_period  # noqa: E402

MAX_TURNS = 16

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录下的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取一个文件的完整内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件（覆盖）。修 bug 用这个。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "运行项目测试，返回通过/失败和失败详情",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

READ_TOOLS = {"list_dir", "read_file"}
WRITE_TOOLS = {"write_file"}


# --- 场景 -----------------------------------------------------------------
# 场景刻意做成"必须先搜、再读、再改、再验"的形状，
# 这样"只读不写"才是一个真实可发生的失败，而不是被提示词排除掉。

BUGGY_SRC = '''\
def running_max(nums):
    """Return the running maximum of nums."""
    out = []
    for n in nums:
        out.append(n)
    return out
'''

TEST_CASES = [
    ([[1, 3, 2, 4, 2, 5]], [1, 3, 3, 4, 4, 5]),
    ([[]], []),
    ([[7]], [7]),
    ([[5, 1, 2]], [5, 5, 5]),
    ([[2, 2, 1]], [2, 2, 2]),
]

SCENARIOS = [
    {
        "key": "G1_running_max_bug",
        "files": {
            "/repo/running_max.py": BUGGY_SRC,
            "/repo/README.md": "# running_max\n\n项目说明。\n",
        },
        "target": "/repo/running_max.py",
        "fn_name": "running_max",
        "task": (
            "你在 /repo 目录里工作。运行 run_tests 看看现状，"
            "找到并修复 bug，让 run_tests 通过。\n"
            "修完**必须**再跑一次 run_tests 确认通过，"
            "然后用一句话总结你改了什么就结束。"
        ),
    },
]

SCENARIO_KEYS = [s["key"] for s in SCENARIOS]


# --- 工具实现（纯函数，便于单测） -----------------------------------------

def make_vfs(files: dict) -> dict:
    return dict(files)


def _norm_path(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    if not p.startswith("/"):
        p = "/repo/" + p
    while "//" in p:
        p = p.replace("//", "/")
    return p.rstrip("/") or "/"


def tool_list_dir(vfs: dict, path: str) -> str:
    p = _norm_path(path)
    if p == "/":
        entries = sorted({_norm_path(k).split("/")[1] for k in vfs if k != "/"})
        return "目录 /：" + (", ".join(entries) if entries else "(空)")
    prefix = p + "/"
    if not any(k.startswith(prefix) for k in vfs):
        if p in vfs:
            return f"{p} 是文件，不是目录"
        return f"目录不存在：{p}"
    names = sorted({k[len(prefix):].split("/")[0] for k in vfs if k.startswith(prefix)})
    return f"{p}/：" + ", ".join(names)


def tool_read_file(vfs: dict, path: str) -> str:
    p = _norm_path(path)
    if p not in vfs:
        return f"文件不存在：{p}"
    return vfs[p]


def tool_write_file(vfs: dict, path: str, content: str) -> str:
    p = _norm_path(path)
    vfs[p] = content
    return f"已写入 {p}（{len(content)} 字符）"


def tool_run_tests(vfs: dict, scenario: dict) -> str:
    """真 exec：把目标文件当模块跑固定用例。subprocess + 超时。"""
    import subprocess
    src = vfs.get(_norm_path(scenario["target"]))
    if src is None:
        return "FAIL: 目标文件不存在"
    script = (_SANDBOX + "\nprint(json.dumps(check("
              f"{src!r}, {TEST_CASES!r}, {scenario['fn_name']!r})))\n")
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "FAIL: 测试超时（可能死循环）"
    if r.returncode != 0:
        return f"FAIL: 测试运行错误 {r.stderr.strip()[:160]}"
    try:
        kind, detail = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return f"FAIL: 无法解析测试输出 {r.stdout[:120]}"
    if kind == "ok":
        return f"PASS: 全部 {len(TEST_CASES)} 个用例通过"
    return f"FAIL: {kind} {detail}"


def dispatch(vfs: dict, scenario: dict, name: str, args: dict) -> str:
    if name == "list_dir":
        return tool_list_dir(vfs, args.get("path", ""))
    if name == "read_file":
        return tool_read_file(vfs, args.get("path", ""))
    if name == "write_file":
        return tool_write_file(vfs, args.get("path", ""), args.get("content", ""))
    if name == "run_tests":
        return tool_run_tests(vfs, scenario)
    return f"未知工具：{name}"


# --- 指标（纯函数，输入是 transcript） -----------------------------------

_DONE_RE = re.compile(r"(任务完成|已完成|修复完成|done\b|finished)", re.I)


def episode_metrics(transcript: list[dict], vfs: dict, scenario: dict) -> dict:
    """
    transcript: [{"name":..., "args":{...}, "result":...}, ...] 按调用顺序
    vfs: 结束时的最终虚拟机状态

    所有指标都只看记录，不看模型说了什么 —— 除非最后那条文本里
    有"我完成了"却没有对应证据（premature_done）。
    """
    calls = transcript
    sig = [(c.get("name"), json.dumps(c.get("args", {}), sort_keys=True, ensure_ascii=False))
           for c in calls]

    # 重复调用
    counts: dict[tuple, int] = {}
    for s in sig:
        counts[s] = counts.get(s, 0) + 1
    repeat_call_count = sum(v - 1 for v in counts.values() if v > 1)

    # 连续重复最长段
    max_streak = streak = 1 if sig else 0
    for i in range(1, len(sig)):
        streak = streak + 1 if sig[i] == sig[i - 1] else 1
        max_streak = max(max_streak, streak)

    # 周期型循环 —— 交替循环才是 agent 最常见的兜圈形态
    # 只看"连续相同"会整个漏掉 list→read→test 这种三步一循环
    # （自测里就踩了：交替样本 repeat_call_count=16 但 max_streak=1）
    period = _dominant_period([f"{n}|{a}" for n, a in sig], max_p=8, min_reps=3)
    loop_cycles = 0
    if period:
        loop_cycles = (len(sig) - period) // period

    n_read = sum(1 for c in calls if c.get("name") in READ_TOOLS)
    n_write = sum(1 for c in calls if c.get("name") in WRITE_TOOLS)
    n_test = sum(1 for c in calls if c.get("name") == "run_tests")

    wrote = [c for c in calls if c.get("name") in WRITE_TOOLS]
    wrote_any = bool(wrote)
    edit_reached = any(
        _norm_path((c.get("args") or {}).get("path", "")) == _norm_path(scenario["target"])
        for c in wrote)

    # 最终状态真跑一遍测试 —— 不信模型自报的结论
    final = tool_run_tests(vfs, scenario)
    tests_passed_final = final.startswith("PASS")

    # 空转：写了但内容与写前完全一样（真正没推进）
    stall = sum(1 for c in calls if c.get("name") in WRITE_TOOLS and c.get("no_op"))

    # 宣布完成：文本里有完成措辞且**之后没有再调过 run_tests**
    said_done = bool(_DONE_RE.search(transcript[-1].get("final_text", ""))) if transcript else False
    ran_after_done = False
    if said_done:
        last_test = max((i for i, c in enumerate(calls) if c.get("name") == "run_tests"),
                        default=-1)
        last_call = len(calls) - 1
        ran_after_done = last_test < last_call and tests_passed_final
    premature_done = said_done and not tests_passed_final and not ran_after_done

    # 读:写 比 —— "114 次搜索" 的量化形式
    read_write_ratio = (n_read / n_write) if n_write else float("inf")

    return {
        "turns": len(calls),
        "n_read": n_read,
        "n_write": n_write,
        "n_test": n_test,
        "wrote_any_file": wrote_any,
        "read_only_episode": n_write == 0 and n_read > 0,
        "edit_reached_target": edit_reached,
        "tests_passed_final": tests_passed_final,
        "final_test": final[:120],
        "solved": wrote_any and edit_reached and tests_passed_final,
        "repeat_call_count": repeat_call_count,
        "max_identical_streak": max_streak,
        "loop_period": period,          # None = 没检出周期型循环
        "loop_cycles": loop_cycles,     # 检出了多少轮重复
        "looped": bool(period),         # 兜圈判定：周期存在且重复够多
        "stall_turns": stall,
        "said_done": said_done,
        "premature_done": premature_done,
        "read_write_ratio": read_write_ratio,
        "tool_hist": {n: sum(1 for c in calls if c.get("name") == n)
                      for n in ("list_dir", "read_file", "write_file", "run_tests")},
    }


# --- 假 transcript（自测用，证明指标会开火） -----------------------------

def _mk(vfs, name, **args):
    """
    构造一条调用记录，并真的作用在 vfs 上，保证指标和状态一致。

    write_file 额外记 `no_op`：写入内容与**写入前**完全相同。
    「这一轮调了工具但什么都没变」只能用写前状态判断 ——
    拿最终状态去比会把每一次成功写入都误判成空转（踩过）。
    """
    sc = SCENARIOS[0]
    rec = {"name": name, "args": args}
    if name == "write_file":
        p = _norm_path(args.get("path", ""))
        rec["no_op"] = (vfs.get(p) == args.get("content", ""))
    rec["result"] = dispatch(vfs, sc, name, args)
    return rec


GOOD_FIX = '''\
def running_max(nums):
    out = []
    best = None
    for n in nums:
        best = n if best is None or n > best else best
        out.append(best)
    return out
'''

assert tool_run_tests(make_vfs(SCENARIOS[0]["files"]), SCENARIOS[0]).startswith("FAIL"), \
    "场景前提失效：初始代码本该测试失败"
assert tool_run_tests(make_vfs({**SCENARIOS[0]["files"],
                                "/repo/running_max.py": GOOD_FIX}),
                      SCENARIOS[0]).startswith("PASS"), \
    "场景前提失效：GOOD_FIX 本该通过测试"


def good_transcript():
    v = make_vfs(SCENARIOS[0]["files"])
    t = [_mk(v, "run_tests"),
         _mk(v, "list_dir", path="/repo"),
         _mk(v, "read_file", path="/repo/running_max.py"),
         _mk(v, "write_file", path="/repo/running_max.py", content=GOOD_FIX),
         _mk(v, "run_tests")]
    t[-1]["final_text"] = "已修复：把 out.append(n) 改成 append 当前最大值，测试已通过。"
    return t, v


def looping_transcript():
    """AGI Hunt 那个形态：反复搜索，从不写。"""
    v = make_vfs(SCENARIOS[0]["files"])
    t = [_mk(v, "run_tests")]
    for _ in range(6):
        t += [_mk(v, "list_dir", path="/repo"),
              _mk(v, "read_file", path="/repo/running_max.py"),
              _mk(v, "run_tests")]
    t[-1]["final_text"] = "我还在分析这个问题。"
    return t, v


def read_only_done_transcript():
    """只读不写，还宣布完成。"""
    v = make_vfs(SCENARIOS[0]["files"])
    t = [_mk(v, "run_tests"),
         _mk(v, "read_file", path="/repo/running_max.py"),
         _mk(v, "run_tests")]
    t[-1]["final_text"] = "任务完成，测试已通过。"
    return t, v


def wrong_file_write_transcript():
    """写了，但写错文件（新建了一个文件而不是改目标）。"""
    v = make_vfs(SCENARIOS[0]["files"])
    t = [_mk(v, "read_file", path="/repo/running_max.py"),
         _mk(v, "write_file", path="/repo/fix_notes.md", content="some notes"),
         _mk(v, "run_tests")]
    t[-1]["final_text"] = "写好了。"
    return t, v


def self_check(verbose: bool = True) -> int:
    fails: list[str] = []

    def ck(name, cond, detail=""):
        if cond:
            if verbose:
                print(f"  ok   {name}")
        else:
            fails.append(f"{name} {detail}")
            print(f"  FAIL {name} {detail}")

    g, gv = good_transcript()
    m = episode_metrics(g, gv, SCENARIOS[0])
    ck("good: solved", m["solved"], m)
    ck("good: wrote_any_file", m["wrote_any_file"])
    ck("good: edit_reached_target", m["edit_reached_target"])
    ck("good: tests_passed_final", m["tests_passed_final"])
    ck("good: not read_only", not m["read_only_episode"])
    ck("good: not premature_done", not m["premature_done"], m)
    ck("good: no stall", m["stall_turns"] == 0, m)
    ck("good: not looped", not m["looped"], m)
    ck("good: finite read_write_ratio", m["read_write_ratio"] < float("inf"), m)

    lp, lv = looping_transcript()
    m = episode_metrics(lp, lv, SCENARIOS[0])
    ck("loop: read_only_episode fires", m["read_only_episode"], m)
    ck("loop: wrote_any_file is False", not m["wrote_any_file"])
    ck("loop: repeat_call_count fires", m["repeat_call_count"] > 0, m)
    ck("loop: looped fires（交替循环也必须抓到）", m["looped"], m)
    ck("loop: loop_period == 3", m["loop_period"] == 3, m)
    ck("loop: not solved", not m["solved"])
    ck("loop: infinite read_write_ratio", m["read_write_ratio"] == float("inf"), m)

    r, rv = read_only_done_transcript()
    m = episode_metrics(r, rv, SCENARIOS[0])
    ck("read_only_done: premature_done fires", m["premature_done"], m)
    ck("read_only_done: tests still fail", not m["tests_passed_final"])
    ck("read_only_done: said_done", m["said_done"])
    ck("read_only_done: not solved", not m["solved"])

    w, wv = wrong_file_write_transcript()
    m = episode_metrics(w, wv, SCENARIOS[0])
    ck("wrong_file: wrote_any_file True", m["wrote_any_file"])
    ck("wrong_file: edit_reached_target False", not m["edit_reached_target"], m)
    ck("wrong_file: not solved", not m["solved"])

    # 指标必须对"改了但改错"和"改了且改对"可区分
    ck("discriminates write-target", 
       episode_metrics(w, wv, SCENARIOS[0])["edit_reached_target"] !=
       episode_metrics(g, gv, SCENARIOS[0])["edit_reached_target"])

    ck("scenarios: unique keys", len(set(SCENARIO_KEYS)) == len(SCENARIOS))
    ck("tools: 4 schemas", len(TOOLS) == 4)
    ck("tools: names wired",
       {t["function"]["name"] for t in TOOLS} ==
       {"list_dir", "read_file", "write_file", "run_tests"})

    print()
    if fails:
        print(f"AGENT LOOP SELF-CHECK: {len(fails)} FAILED")
        for x in fails:
            print("  -", x)
        return 1
    print("AGENT LOOP SELF-CHECK: all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
