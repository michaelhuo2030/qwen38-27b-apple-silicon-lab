#!/usr/bin/env python3
"""
Agentic 探针：测「多步动作是否收敛」，不测单轮问答质量。

为什么单独一个模块
------------------
低比特探针（lowbit_probe）测的是**单轮**能力，全部 32 次里 0 退化。
但社区对三值模型最严重的指控是 agentic 层面的：

  * AGI Hunt：agentic loop 建 ISS 追踪器，"Bonsai 连续 114 次搜索，
    一行代码都没写"，而全精度模型正常交付
  * MindStudio："struggled with a basic self-correction loop in code generation"
  * eogee：外部 drafter 模式在 prompt > 2K 接受率骤降、> 16K 直接早停（0 输出）
  * 官方口径：98.2% 是 20 项均值，medium effort 只有 96.0%，
    两项 agentic coding benchmark 只有 ~75%

**这些指控我一条都没测过。** 8 道单轮题无论过多少，都答不上
"它能不能在多轮里收敛"。所以要有一个专门测收敛的探针。

设计原则
--------
不接真的 tool-calling API（那会引入 harness 变量）。
改成**把 agent 循环展开成一道长题**，让模型必须在一个回复里
走完 N 个编号步骤。这样：

  1. 完全可机器验证（第 k 步必须产出规定的内容，答案唯一）
  2. 不依赖外部服务，不引入 harness 差异
  3. 步数可调，步数 vs 收敛失败率就是曲线

指标
----
  steps_completed / steps_required   走了几步
  converged                          是否在步数内产出全部要求
  repeated_action                    有没有重复同一个动作（"兜圈"的直接证据）
  stalled                            某一步反复出现但没进展
  premature_finish                   少几步就宣布完成（比兜圈更隐蔽的失败）
  termination_ok                     末步是否收得住（对应"给不出最终答案"）

这些指标全部在纯函数里，无模型即可自测 —— 沿用 lowbit_probe 的纪律：
**每个守卫都要证明在坏样本上会开火、好样本上不会误伤。**
"""
from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- 任务集 ---------------------------------------------------------------
# 每题给一个 step 数，模型必须逐行输出 "步骤N: <内容>"，
# 校验器逐行核对。步数越多越考验长程收敛。

A_TASKS = [
    (
        "A1_emit_3_records",
        "短程(3步)",
        "严格按下面格式输出 3 行，不要任何其它文字，不要 markdown。\n"
        "每行格式：步骤N: <值>\n"
        "N 从 1 到 3，值依次为：第一步输出 alpha，第二步输出 beta，"
        "第三步输出 gamma（都是小写英文单词）。",
        3,
    ),
    (
        "A2_emit_6_records",
        "中程(6步)",
        "严格按下面格式输出 6 行，不要任何其它文字，不要 markdown。\n"
        "每行格式：步骤N: <值>\n"
        "N 从 1 到 6。第 N 步的值是：N 为 1 时输出 v1，N 为 2 时输出 v2，"
        "以此类推直到 v6。",
        6,
    ),
    (
        "A3_emit_10_countdown",
        "长程(10步·逆序)",
        "严格按下面格式输出 10 行，不要任何其它文字，不要 markdown。\n"
        "每行格式：步骤N: <值>\n"
        "N 从 1 到 10，值必须是 10 减 N：即步骤1的值是 9，步骤2的值是 8，"
        "依次递减到步骤10的值是 0。",
        10,
    ),
    (
        "A4_self_correct",
        "自纠错循环",
        "严格按下面格式输出 5 行，不要任何其它文字，不要 markdown。\n"
        "每行格式：步骤N: <值>\n"
        "N 从 1 到 5。第 1 步的值先写成 wrong，然后第 2 步必须把它改正为 right；"
        "第 3 步写 300，第 4 步写 400，第 5 步写 500。",
        5,
    ),
    (
        "A5_dependency_chain",
        "依赖链(5步)",
        "严格按下面格式输出 5 行，不要任何其它文字，不要 markdown。\n"
        "每行格式：步骤N: <值>\n"
        "N 从 1 到 5。第 1 步的值是 2；第 N 步（N>1）的值必须等于"
        "「第 N-1 步的值乘以 2」。",
        5,
    ),
]

A_TASK_KEYS = [t[0] for t in A_TASKS]


# --- 解析 -----------------------------------------------------------------
_STEP_RE = re.compile(r"步骤\s*(\d+)\s*[:：]\s*(.*)")


def parse_steps(text: str) -> dict[int, str]:
    """抽出 步骤N: 值。返回 {N: 值}。重复的 N 保留最后一个（但由 repeated_action 单独报）。"""
    out: dict[int, str] = {}
    seen_order: list[int] = []
    for m in _STEP_RE.finditer(text):
        n = int(m.group(1))
        out[n] = m.group(2).strip()
        seen_order.append(n)
    return out, seen_order


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s.lower())


# --- 指标 -----------------------------------------------------------------
def convergence_metrics(text: str, required: int) -> dict:
    """
    全部指标都在这里，不依赖具体题目。

    emitted_steps          出现过的不同步骤号数
    max_step_seen          走到的最大步骤号（看它停在哪）
    repeated_action_count  同一个步骤号被写了两次以上的次数（"兜圈"直接证据）
    out_of_order           步骤号不是严格递增的次数
    premature_finish       最大步骤号 < required 时为 True（少几步就收工）
    termination_ok         末行是否以值结尾、不是半句话
    """
    steps, order = parse_steps(text)
    m: dict = {
        "emitted_steps": len(steps),
        "max_step_seen": max(steps) if steps else 0,
        "repeated_action_count": 0,
        "out_of_order": 0,
        "premature_finish": False,
        "termination_ok": False,
    }
    if not steps:
        return m

    # 重复动作：order 里相邻重复 / 同一 N 出现多次
    counts: dict[int, int] = {}
    for n in order:
        counts[n] = counts.get(n, 0) + 1
    m["repeated_action_count"] = sum(c - 1 for c in counts.values() if c > 1)

    # 乱序
    asc = [n for i, n in enumerate(order) if i == 0 or n > order[i - 1]]
    m["out_of_order"] = len(order) - len(asc)

    m["premature_finish"] = m["max_step_seen"] < required

    # 末步收尾
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        last = _STEP_RE.search(lines[-1])
        m["termination_ok"] = bool(last and last.group(2).strip())
    return m


def loop_flags(text: str, required: int) -> dict:
    """把收敛指标变成布尔判定。"""
    m = convergence_metrics(text, required)
    f = {
        "repeat_action": m["repeated_action_count"] >= 1,
        "out_of_order": m["out_of_order"] >= 1,
        "premature": m["premature_finish"],
        "not_terminated": not m["termination_ok"],
    }
    f["any_stall"] = any(f.values())
    f["_raw"] = m
    return f


# --- 逐题校验 -------------------------------------------------------------
def _expect_words(n: int) -> dict[int, str]:
    return {i: f"v{i}" for i in range(1, n + 1)}


def verify(task_key: str, text: str) -> tuple[bool, str]:
    steps, _ = parse_steps(text)
    required = next((t[3] for t in A_TASKS if t[0] == task_key), None)
    if required is None:
        raise KeyError(task_key)

    if task_key == "A1_emit_3_records":
        want = {1: "alpha", 2: "beta", 3: "gamma"}
        bad = [f"{n}={steps.get(n)!r}" for n, v in want.items()
               if _norm(steps.get(n, "")) != _norm(v)]
        return not bad, f"{len(steps)}/{len(want)} 步" + (f" 错:{bad}" if bad else "")

    if task_key == "A2_emit_6_records":
        want = _expect_words(6)
        bad = [n for n, v in want.items() if _norm(steps.get(n, "")) != _norm(v)]
        return not bad, f"{len(steps)}/6 步" + (f" 缺/错:{bad}" if bad else "")

    if task_key == "A3_emit_10_countdown":
        want = {i: str(10 - i) for i in range(1, 11)}
        bad = [n for n, v in want.items() if _norm(steps.get(n, "")) != v]
        return not bad, f"{len(steps)}/10 步" + (f" 缺/错:{bad[:4]}" if bad else "")

    if task_key == "A4_self_correct":
        if _norm(steps.get(1, "")) != "wrong":
            return False, f"步骤1 应为 wrong，实为 {steps.get(1)!r}"
        if _norm(steps.get(2, "")) != "right":
            return False, f"步骤2 应为 right，实为 {steps.get(2)!r}"
        want = {3: "300", 4: "400", 5: "500"}
        bad = [n for n, v in want.items() if _norm(steps.get(n, "")) != v]
        return not bad, f"纠正链 ok" + (f" 但 {bad} 错" if bad else "")

    if task_key == "A5_dependency_chain":
        want, v = {}, 2
        for i in range(1, 6):
            v = 2 if i == 1 else v * 2
            want[i] = str(v)
        bad = [n for n, x in want.items() if _norm(steps.get(n, "")) != x]
        return not bad, f"{len(steps)}/5 步" + (f" 错:{bad}" if bad else "")

    raise KeyError(task_key)


# --- 自测 -----------------------------------------------------------------
GOOD_A1 = "步骤1: alpha\n步骤2: beta\n步骤3: gamma"
GOOD_A3 = "\n".join(f"步骤{i}: {10-i}" for i in range(1, 11))
GOOD_A5 = "步骤1: 2\n步骤2: 4\n步骤3: 8\n步骤4: 16\n步骤5: 32"

# 坏样本 1：兜圈（同一动作反复做）
BAD_REPEAT = "\n".join(["步骤1: 搜索 打开文档", "步骤1: 搜索 打开文档",
                        "步骤1: 搜索 打开文档", "步骤1: 搜索 打开文档",
                        "步骤1: 搜索 打开文档"])
# 坏样本 2：中途放弃（社区说的"一行代码没写"）
BAD_STALL = "步骤1: 搜索 打开文档\n步骤2: 搜索 打开文档"
# 坏样本 3：提前收工
BAD_PREMATURE = "步骤1: alpha\n步骤2: beta"
# 坏样本 4：乱序
BAD_ORDER = "步骤3: gamma\n步骤2: beta\n步骤1: alpha"


def self_check(verbose: bool = True) -> int:
    fails: list[str] = []

    def ck(name, cond, detail=""):
        if cond:
            if verbose:
                print(f"  ok   {name}")
        else:
            fails.append(f"{name} {detail}")
            print(f"  FAIL {name} {detail}")

    # 指标必须会开火
    f = loop_flags(BAD_REPEAT, 3)
    ck("repeat_sample: repeat_action fires", f["repeat_action"], f["_raw"])
    ck("repeat_sample: any_stall", f["any_stall"], f["_raw"])

    f = loop_flags(BAD_STALL, 6)
    ck("stall_sample: premature fires", f["premature"], f["_raw"])
    ck("stall_sample: max_step_seen < required", f["_raw"]["max_step_seen"] == 2, f["_raw"])

    f = loop_flags(BAD_PREMATURE, 3)
    ck("premature_sample: fires", f["premature"], f["_raw"])

    f = loop_flags(BAD_ORDER, 3)
    ck("order_sample: out_of_order fires", f["out_of_order"], f["_raw"])

    # 好样本必须不误伤
    for name, good, req in (("A1", GOOD_A1, 3), ("A3", GOOD_A3, 10), ("A5", GOOD_A5, 5)):
        f = loop_flags(good, req)
        ck(f"good_{name}: no stall", not f["any_stall"], f["_raw"])
        ck(f"good_{name}: not premature", not f["premature"], f["_raw"])

    # 验证器正负对照
    ck("verify A1 good", verify("A1_emit_3_records", GOOD_A1)[0])
    ck("verify A1 bad", not verify("A1_emit_3_records", GOOD_A1.replace("gamma", "delta"))[0])
    ck("verify A3 good", verify("A3_emit_10_countdown", GOOD_A3)[0])
    ck("verify A3 bad (缺一步)",
       not verify("A3_emit_10_countdown", "\n".join(GOOD_A3.splitlines()[:8]))[0])
    ck("verify A5 good", verify("A5_dependency_chain", GOOD_A5)[0])
    ck("verify A5 bad (依赖算错)",
       not verify("A5_dependency_chain", "步骤1: 2\n步骤2: 5\n步骤3: 8\n步骤4: 16\n步骤5: 32")[0])
    ck("verify A4 good",
       verify("A4_self_correct", "步骤1: wrong\n步骤2: right\n步骤3: 300\n步骤4: 400\n步骤5: 500")[0])
    ck("verify A4 bad (没纠正)",
       not verify("A4_self_correct", "步骤1: wrong\n步骤2: wrong\n步骤3: 300\n步骤4: 400\n步骤5: 500")[0])

    ck("tasks: 5 unique keys", len(set(A_TASK_KEYS)) == 5, str(A_TASK_KEYS))
    for t in A_TASKS:
        try:
            verify(t[0], "")
        except KeyError:
            ck(f"verify wired: {t[0]}", False, "无校验器")

    print()
    if fails:
        print(f"AGENTIC PROBE SELF-CHECK: {len(fails)} FAILED")
        for x in fails:
            print("  -", x)
        return 1
    print("AGENTIC PROBE SELF-CHECK: all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
