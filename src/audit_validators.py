#!/usr/bin/env python3
"""
audit_validators.py — mechanical audit of the quality validators.

Run: python3 audit_validators.py

Hand-inspection found three of the four validator bugs in this project, and
missed the rest. This script exists because the bugs have a shape, and the
shape is mechanical:

  E1  a requirement the prompt allows in several forms, only one of which the
      validator accepts
  E2  a requirement stated in the prompt with no check at all
  E3  a requirement the validator checks that the prompt never stated
  E4  a test fixture that passes or fails for a reason other than the one its
      name claims

Two checks, both mechanical:

  1. LEDGER — every requirement the prompt states appears in
     VALIDATION_SPEC, and every unmeasured one carries a reason. A requirement
     may be checked or declared unmeasured; it may never be silently absent.

  2. DIFFERENTIAL — for every requirement marked checked, take a sample that
     satisfies the whole prompt, break exactly that one requirement, and
     require the validator to reject it *naming that requirement*. This is what
     catches E1 (the break is a legal alternative form), E2 (breaking it
     changes nothing) and E4 (it is rejected, but for a different reason).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_checks as Q
import tasks

FAIL = []


def bad(msg: str) -> None:
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


# --- samples that satisfy the whole prompt -----------------------------------
T2_FULL = (
    "import collections\n"
    "import time\n\n"
    "def timed(fn):\n"
    "    def wrapper(*a, **kw):\n"
    "        t0 = time.time()\n"
    "        r = fn(*a, **kw)\n"
    "        print(fn.__name__, round(time.time() - t0, 6))\n"
    "        return r\n"
    "    return wrapper\n\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        self.cap = capacity\n"
    "        self.d = collections.OrderedDict()\n"
    "    @timed\n"
    "    def get(self, key, default=None):\n"
    "        \"\"\"Return a key, or default if absent.\"\"\"\n"
    "        if key not in self.d:\n"
    "            return default\n"
    "        self.d.move_to_end(key)\n"
    "        return self.d[key]\n"
    "    @timed\n"
    "    def put(self, key, value):\n"
    "        \"\"\"Store a key, evicting the coldest when over capacity.\"\"\"\n"
    "        if key in self.d:\n"
    "            self.d.move_to_end(key)\n"
    "        self.d[key] = value\n"
    "        if len(self.d) > self.cap:\n"
    "            self.d.popitem(last=False)\n"
    "    def __len__(self):\n"
    "        return len(self.d)\n"
    "    def __repr__(self):\n"
    "        return f'LRUCache({self.cap}, keys={list(self.d)})'\n"
)

T3_FULL = (
    "import math\n\n"
    "class Shape:\n"
    "    def area(self):\n"
    "        raise NotImplementedError\n"
    "    def describe(self):\n"
    "        return f'{type(self).__name__} area={self.area():.2f}'\n\n"
    "class Circle(Shape):\n"
    "    def __init__(self, radius):\n"
    "        \"\"\"A circle of the given radius.\"\"\"\n        self.r = radius\n"
    "    def area(self):\n        return math.pi * self.r ** 2\n\n"
    "class Rectangle(Shape):\n"
    "    def __init__(self, width, height):\n"
    "        \"\"\"A width by height rectangle.\"\"\"\n"
    "        self.w, self.h = width, height\n"
    "    def area(self):\n        return self.w * self.h\n\n"
    "class Triangle(Shape):\n"
    "    def __init__(self, base, height):\n"
    "        \"\"\"A base by height triangle.\"\"\"\n"
    "        self.b, self.h = base, height\n"
    "    def area(self):\n        return self.b * self.h / 2\n\n"
    "def total_area(shapes):\n"
    "    return sum(s.area() for s in shapes)\n"
)

T6_FULL = (
    "TCP 为什么必须三次握手，这要从三个角度分点说明。"
    "第一，为什么不是两次：两次握手时，服务端发出 SYN+ACK 之后就认为连接已建立，"
    "但它无从得知客户端是否收到了这个报文。一旦该报文在网络中丢失，"
    "客户端会认为连接失败而关闭，服务端却仍持有一个半连接，"
    "其资源会被迟迟不释放的半连接队列持续占用。"
    "第二，为什么不是四次：第三次握手已经完成了双向同步，"
    "双方的序号都已被对方确认，第四次握手不再携带任何新的信息，属于纯冗余。"
    "第三，ISN（初始序列号）的作用是让序号随着连接的生命周期而变化，"
    "从而使上一条连接里延迟到达的旧报文不会因为序号巧合而被新连接误收。"
    "第四，TIME_WAIT 状态必须存在，是因为主动关闭方在发出最后一个 ACK 之后，"
    "无法确认对端是否收到了它。保留这个状态有两重作用："
    "一是万一 ACK 丢失，还能重传；二是让旧连接的迷途报文在网络中自然消散，"
    "否则它们可能被新建立的同四元组连接错误接收。"
    "第五，如果 SYN-ACK 丢失，客户端会超时重发 SYN，"
    "服务端收到后重新发送 SYN-ACK，连接最终仍然能够建立，"
    "整个过程只是多付出一个 RTT 的时延代价，并不影响正确性。"
)

T7_FULL = (
    "第一步 原价 200 元，先打 8 折：200 × 0.8 = 160 元。"
    "第二步 满 150 减 30：160 - 30 = 130 元。"
    "第三步 会员券打 85 折：130 × 0.85 = 110.5 元。"
    "第四步 积分抵扣 20 元：110.5 - 20 = 90.5 元，所以实付 90.5 元。"
    "如果把顺序改成先减 30 再打 8 折，那么 200 - 30 = 170 元，"
    "170 × 0.8 = 136 元，结果是 136 元，与原来的 90.5 元并不相同。"
    "原因是乘法与减法不满足交换律，先乘折扣再减去固定金额，"
    "被减去的 30 元本身也被打了折，所以两者结果不同。"
)

T8_FULL = ("深秋的雨像一层薄纱，把江面笼在灰白里，江风像一只凉的手慢慢推着人走。"
           "石阶被雨水洗得发亮，仿佛谁把旧信纸铺在了山脚。"
           "远处的灯一盏盏亮起来，像是被谁一个一个点燃的句子。"
           "心里的那点说不清的东西忽然松开了，像攥久的手终于放下，"
           "又像终于走完一段很长的夜路，天还没有亮，但已经不再怕黑。" * 3)

T10_FULL = (
    "import React, { useEffect, useRef } from 'react';\n"
    "type ReasoningFrame = { id: string; title: string; body: string; tokens: number };\n"
    "type Props = {\n"
    "  frames: ReasoningFrame[];\n"
    "  isStreaming: boolean;\n"
    "  onSeek?: (id: string) => void;\n"
    "};\n"
    "export default function ReasoningPanel({ frames, isStreaming, onSeek }: Props) {\n"
    "  const pinnedRef = useRef<HTMLDivElement | null>(null);\n"
    "  const manualRef = useRef(false);\n"
    "  useEffect(() => {\n"
    "    if (!manualRef.current && isStreaming) {\n"
    "      pinnedRef.current?.scrollIntoView({ behavior: 'smooth' });\n"
    "    }\n"
    "  }, [frames.length, isStreaming]);\n"
    "  const total = frames.reduce((n, f) => n + f.tokens, 0);\n"
    "  return (\n"
    "    <div className=\"panel\" onScroll={(e) => { manualRef.current = e.currentTarget.scrollTop < 1; }}>\n"
    "      {frames.map((f) => (\n"
    "        <section key={f.id} onClick={() => onSeek?.(f.id)}>\n"
    "          <h3>{f.title}</h3>\n"
    "          <p>{f.body}</p>\n"
    "          <span>{f.tokens} tokens</span>\n"
    "        </section>\n"
    "      ))}\n"
    "      <div ref={pinnedRef} />\n"
    "      <footer>total {total} tokens</footer>\n"
    "    </div>\n"
    "  );\n"
    "}\n"
)

T1_FULL = ("[" + ", ".join(
    '{"id": %d, "name": "Smart Lamp %d", "price": %d.5, "category": "Lighting",'
    ' "inStock": true, "tags": ["smart", "home"]}' % (i, i, i)
    for i in range(1, 13)) + "]")

FULL = {
    "T1_json_structured": T1_FULL,
    "T2_code_function": T2_FULL,
    "T3_code_repetitive": T3_FULL,
    "T6_qa_factual": T6_FULL,
    "T7_math_reasoning": T7_FULL,
    "T8_creative_writing": T8_FULL,
    "T10_ts_component": T10_FULL,
}

# (task, requirement id, a sample that violates ONLY that requirement,
#  a word that must appear in the rejection reason)
# The keyword column is what makes this a differential test rather than a
# boolean: a sample rejected for the wrong reason proves nothing.
DIFFERENTIAL = [
    ("T1_json_structured", "count_12", "[{\"id\": 1, \"name\": \"Lamp\", \"price\": 1.5,"
     " \"category\": \"C\", \"inStock\": true, \"tags\": [\"a\", \"b\"]}]", "spec says 12"),
    ("T1_json_structured", "inStock_bool", "[" + ", ".join(
        '{"id": %d, "name": "Lamp %d", "price": %d.5, "category": "C",'
        ' "inStock": "yes", "tags": ["a", "b"]}' % (i, i, i)
        for i in range(1, 13)) + "]", "inStock"),

    ("T2_code_function", "class_name", T2_FULL.replace("LRUCache", "Cache"),
     "LRUCache"),
    # E1: `from collections import OrderedDict` is a legal alternative form and
    # must NOT be rejected. The negative case is in IDIOM, not here.
    ("T2_code_function", "ordereddict", T2_FULL.replace(
        "import collections\n", "").replace("collections.OrderedDict()", "{}"),
     "OrderedDict"),
    ("T2_code_function", "methods_4", T2_FULL.replace(
        "    def __repr__(self):\n        return f'LRUCache({self.cap}, keys={list(self.d)})'\n", ""),
     "__repr__"),
    ("T2_code_function", "docstrings_english_triple", T2_FULL.replace(
        '        """Return a key, or default if absent."""\n', ""), "docstring"),
    ("T2_code_function", "capacity_from_init", T2_FULL.replace(
        "def __init__(self, capacity):", "def __init__(self, cap):")
     .replace("self.cap = capacity", "self.cap = cap")
     .replace("self.cap = cap)", "self.cap = cap)"), "capacity"),
    ("T2_code_function", "timed_decorator", T2_FULL.replace(
        "def timed(fn):", "def timed2(fn):").replace("@timed", "@timed2"),
     "timed"),
    ("T2_code_function", "no_markdown", "```python\n" + T2_FULL + "```", "markdown"),

    ("T3_code_repetitive", "subclasses_3", T3_FULL.replace(
        "class Triangle(Shape):\n    def __init__(self, base, height):\n"
        "        \"\"\"A base by height triangle.\"\"\"\n"
        "        self.b, self.h = base, height\n"
        "    def area(self):\n        return self.b * self.h / 2\n\n", ""),
     "subclasses"),
    ("T3_code_repetitive", "init_docstrings", T3_FULL.replace(
        '        """A circle of the given radius."""\n', ""), "docstring"),
    ("T3_code_repetitive", "ctor_params", T3_FULL.replace(
        "def __init__(self, radius):", "def __init__(self, r):")
     .replace("self.r = radius", "self.r = r"), "ctor params"),
    ("T3_code_repetitive", "total_area", T3_FULL.replace(
        "def total_area(shapes):\n    return sum(s.area() for s in shapes)\n", ""),
     "total_area"),
    ("T3_code_repetitive", "no_markdown", "```python\n" + T3_FULL + "```", "markdown"),

    ("T6_qa_factual", "why_not_four", T6_FULL.replace(
        "第二，为什么不是四次：第三次握手已经完成了双向同步，双方的序号都已被对方确认，"
        "第四次握手不再携带任何新的信息，属于纯冗余。", ""), "why-not-four"),
    ("T6_qa_factual", "time_wait", T6_FULL.replace("TIME_WAIT", "CLOSE_WAIT"),
     "TIME_WAIT"),
    ("T6_qa_factual", "synack_loss", T6_FULL.replace(
        "第五，如果 SYN-ACK 丢失，客户端会超时重发 SYN，服务端收到后重新发送 SYN-ACK，"
        "连接最终仍然能够建立，整个过程只是多付出一个 RTT 的时延代价，并不影响正确性。",
        "第五，握手过程按照标准流程顺利完成，连接随之建立。"), "SYN-ACK loss"),
    ("T6_qa_factual", "length_500", "为什么不是两次？为什么不是四次？ISN 有用。",
     "length"),

    # Every occurrence of 90.5, not just the final one: replacing only the last
    # leaves 90.5 in the text, so the sample is still correct and the
    # differential tests nothing. A too-weak mutation is itself an audit bug —
    # it reads as coverage and hides the real answer.
    ("T7_math_reasoning", "final_90_5", T7_FULL.replace("90.5", "77.5"), "90.5"),
    ("T7_math_reasoning", "reorder_addressed", T7_FULL.split("如果把顺序改成")[0],
     "reorder"),

    # Strip every marker the checker counts, not just the obvious one.
    ("T8_creative_writing", "similes_2",
     __import__("re").sub(r"像|如同|好像|仿佛|宛如|犹如|似的", "是", T8_FULL * 3),
     "simile"),
    ("T8_creative_writing", "forbidden_words", T8_FULL * 3 + "江边很寂静，夜里静谧。",
     "banned"),
    ("T8_creative_writing", "length_600", "深秋的雨像薄纱，如同旧信纸，仿佛一场旧梦。",
     "CJK"),

    ("T10_ts_component", "use_ref", T10_FULL.replace("useRef", "useMyRef"), "useRef"),
    ("T10_ts_component", "use_effect", T10_FULL.replace("useEffect", "useMyEffect"),
     "useEffect"),
    ("T10_ts_component", "component_named", T10_FULL.replace("ReasoningPanel", "Panel"),
     "ReasoningPanel"),
    ("T10_ts_component", "props_signature", T10_FULL.replace("isStreaming", "streaming"),
     "isStreaming"),
    ("T10_ts_component", "dep_frames_length", T10_FULL.replace(
        "[frames.length, isStreaming]", "[frames, isStreaming]"), "frames.length"),
    ("T10_ts_component", "no_any", T10_FULL.replace(
        "export default function ReasoningPanel({ frames, isStreaming, onSeek }: Props)",
        "export default function ReasoningPanel({ frames, isStreaming, onSeek }: any)"),
     "any"),
]


def audit_ledger() -> None:
    print("\n[1] requirement ledger")
    prompt_ids = {t[0] for t in tasks.TASKS}
    swept = set(FULL)
    for missing in prompt_ids - swept - {"T4_translation", "T5_extraction",
                                         "T9_rust_impl", "T11_doc_rewrite"}:
        bad(f"task {missing} is in tasks.py but has no ledger entry")
    s = Q.ledger_summary()
    for task, reqs in Q.VALIDATION_SPEC.items():
        for r in reqs:
            if not r["quote"]:
                bad(f"{task}/{r['id']} has no prompt_quote (E3: untraceable)")
            if r["tier"] == "semantic" and r["checked"]:
                bad(f"{task}/{r['id']} is tier=semantic but marked checked")
            if not r["checked"] and not r.get("why"):
                bad(f"{task}/{r['id']} is NOT_MEASURED with no reason "
                    f"— this is the silent gap (E2)")
    print(f"  {s['total']} requirements: {s['checked']} checked, "
          f"{s['not_measured']} declared unmeasured with reasons")


def audit_accepted() -> None:
    print("\n[2] full-prompt samples must be accepted")
    for task, text in FULL.items():
        ok, why = Q.check(task, text)
        if not ok:
            bad(f"{task}: a sample satisfying the whole prompt was rejected "
                f"— {why}")


def audit_differential() -> None:
    print(f"\n[3] {len(DIFFERENTIAL)} single-violation differential fixtures")
    for task, req, text, keyword in DIFFERENTIAL:
        # A mutation that does not change the text tests nothing, and reads as
        # a pass. This is how two T6 fixtures sat there "passing" while the
        # sample they claimed to break was untouched — the strongest argument
        # for checking the mutation rather than trusting it.
        if text == FULL[task]:
            bad(f"{task}/{req}: the mutation produced text identical to the "
                f"full-prompt sample — the fixture breaks nothing (E4)")
            continue
        if not text.strip():
            bad(f"{task}/{req}: the mutation produced an empty sample (E4: the "
                f"fixture tests nothing)")
            continue
        ok, why = Q.check(task, text)
        if ok:
            bad(f"{task}/{req}: violating this one requirement was still "
                f"accepted (E2: no check, or the check is too loose)")
        elif keyword.lower() not in why.lower():
            bad(f"{task}/{req}: rejected, but not naming '{keyword}' — "
                f"rejected for the wrong reason (E4). got: {why[:90]}")


def main() -> int:
    audit_ledger()
    audit_accepted()
    audit_differential()
    print()
    if FAIL:
        print(f"{len(FAIL)} AUDIT FAILURES")
        return 1
    print("audit clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
