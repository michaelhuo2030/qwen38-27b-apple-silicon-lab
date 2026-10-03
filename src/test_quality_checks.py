#!/usr/bin/env python3
"""
test_quality_checks.py — do the validators actually reject anything?

A quality validator that never fails is worse than none: it reports
valid_rate = 1.00 at every temperature, which looks like "temperature does not
hurt quality" and is an artifact of the instrument. So each check gets a known-
good and a known-bad sample, and the bad one must fail for the stated reason.

Run: python3 test_quality_checks.py
"""
import quality_checks as Q
import audit_validators as A

import json as _json
import re as _re


def _arr(n=12, drop=(), ntags=2, price_str=False, as_str=False):
    """A 12-object product array, with one defect injected for the bad cases."""
    out = []
    for i in range(1, n + 1):
        o = {"id": i, "name": f"P{i}", "price": f"{i}.5" if price_str else i + 0.5,
             "category": f"C{i}", "inStock": True,
             "tags": [f"a{i}", f"b{i}"][:ntags]}
        for f in drop:
            o.pop(f, None)
        out.append(_json.dumps(o, ensure_ascii=False))
    return "[" + ", ".join(out) + "]"


# One source of truth for "a sample that satisfies the whole prompt".
#
# These used to be hand-written per task, which is how they went stale twice:
# the checkers got stricter and the old fixtures started failing for reasons
# that had nothing to do with what they were named after. The full-prompt
# samples now live in audit_validators.py and both suites use the same ones.
GOOD = {task: (text, True) for task, text in A.FULL.items()}

BAD = {
    "T1_json_structured": [
        ("truncated array", '[{"id": 1, "name": "A", "price": 1.5, "category": "X"}'),
        ("wrong count", _arr(n=1)),
        ("missing inStock", _arr(drop=("inStock",))),
        ("missing tags", _arr(drop=("tags",))),
        ("tags has 1 entry", _arr(ntags=1)),
        ("price is a string", _arr(price_str=True)),
        ("not json at all", "Sure! Here are 12 products you might like:\n1. Widget..."),
    ],
    "T2_code_function": [
        ("syntax error", "class LRUCache\n    def get(self, k)\n        pass"),
        ("wrong class name", "class LRUCacheImpl:\n    def get(self,k):\n        pass\n"
                             "    def put(self,k,v):\n        pass"),
        ("missing method", "class LRUCache:\n    def get(self, k):\n        pass"),
        # The four rules the first version never checked. Each must reject on
        # its own, otherwise "100% at every temperature" is unreachable-by-luck.
        ("no OrderedDict", "def timed(fn):\n    return fn\n\n"
                           "class LRUCache:\n"
                           "    def __init__(self, capacity):\n        self.cap = capacity\n"
                           "    def get(self, key):\n        \"\"\"Get.\"\"\"\n        return 1\n"
                           "    def put(self, key, value):\n        \"\"\"Put.\"\"\"\n        pass\n"
                           "    def __len__(self):\n        return 0\n"
                           "    def __repr__(self):\n        return 'LRU'\n"),
        ("no timed decorator", "from collections import OrderedDict\n\n"
                               "class LRUCache:\n"
                               "    def __init__(self, capacity):\n        self.cap = capacity\n"
                               "    def get(self, key):\n        \"\"\"Get.\"\"\"\n        return 1\n"
                               "    def put(self, key, value):\n        \"\"\"Put.\"\"\"\n        pass\n"
                               "    def __len__(self):\n        return 0\n"
                               "    def __repr__(self):\n        return 'LRU'\n"),
        ("no docstrings", "from collections import OrderedDict\n"
                          "def timed(fn):\n    return fn\n\n"
                          "class LRUCache:\n"
                          "    def __init__(self, capacity):\n        self.cap = capacity\n"
                          "    def get(self, key):\n        return 1\n"
                          "    def put(self, key, value):\n        pass\n"
                          "    def __len__(self):\n        return 0\n"
                          "    def __repr__(self):\n        return 'LRU'\n"),
    ],
    "T3_code_repetitive": [
        ("no area()", "class Shape:\n    def describe(self):\n        return 'x'"),
        ("not a class", "def area(r):\n    return 3.14 * r * r"),
        ("missing a subclass", "class Shape:\n"
                               "    def area(self):\n        raise NotImplementedError\n"
                               "    def describe(self):\n        return 's'\n\n"
                               "class Circle(Shape):\n"
                               "    def area(self):\n        return 3.14\n\n"
                               "class Rectangle(Shape):\n"
                               "    def area(self):\n        return 1\n\n"
                               "def total_area(shapes):\n    return 0\n"),
        ("no total_area()", "class Shape:\n"
                            "    def area(self):\n        raise NotImplementedError\n"
                            "    def describe(self):\n        return 's'\n\n"
                            "class Circle(Shape):\n    def area(self):\n        return 3.14\n\n"
                            "class Rectangle(Shape):\n    def area(self):\n        return 1\n\n"
                            "class Triangle(Shape):\n    def area(self):\n        return 0.5\n"),
    ],
    "T6_qa_factual": [
        ("missing ISN", "为什么要三次：为什么不是两次会收到残留；为什么不是四次是冗余。"
                        "TIME_WAIT 等待迷途报文消散。SYN-ACK 丢失则重传。"),
        # These two were never checked before; both must reject on their own.
        ("missing TIME_WAIT", "为什么不是两次：无法确认客户端收到。为什么不是四次：冗余。"
                              "ISN 防止旧报文被误认。SYN-ACK 丢失会重传。"),
        ("missing SYN-ACK loss", "为什么不是两次：无法确认客户端收到。为什么不是四次：冗余。"
                                  "ISN 防止旧报文被误认。TIME_WAIT 让迷途报文消散。"),
        ("off topic", "TCP 是一种传输层协议，广泛用于互联网。"),
    ],
    "T7_math_reasoning": [
        ("wrong arithmetic", "200×0.8=160，160-30=130，130×0.85=110.5，最后实付 120.5 元。"),
        ("no numbers", "这道题需要分步骤计算，抱歉。"),
        ("stops before the end", "第一步 200×0.8=160。第二步 160-30=130。"),
    ],
    "T10_ts_component": [
        # The label used to say "no state hook" and the sample used useState.
        # The prompt requires useRef + useEffect, so this case is really
        # "no hooks at all" — rename it rather than leave a label that lies.
        ("no hooks at all", "export function TodoList() {\n  return <div>hi</div>;\n}"),
        ("no component", "import React from 'react';\nconsole.log('hello');"),
        # Must satisfy useRef + useEffect, or this case would pass for the wrong
        # reason (missing hooks) and never exercise the imbalance rule.
        ("brace imbalance", "import React, { useEffect, useRef } from 'react';\n"
                            "export function Panel() {\n"
                            "  const r = useRef(null);\n"
                            "  useEffect(() => { r.current = 1; }, []);\n"
                            "  return <div>{r.current}</div>;"),
        ("used any", "import React, { useEffect, useRef } from 'react';\n"
                      "export function Panel(): any {\n"
                      "  const r = useRef(null);\n"
                      "  useEffect(() => { r.current = 1; }, []);\n"
                      "  return <div>{r.current}</div>;\n"
                      "}\n"),
    ],
    "T8_creative_writing": [
        ("too short", "深秋的雨像一层薄纱。"),
        ("no simile", "深秋的雨落下。" * 30),
    ],
}


# One requirement, several legal ways to satisfy it. A validator narrower than
# the prompt it was written against is the exact failure this project has now
# hit twice (T10 wanted useState, T2 wanted `from collections import
# OrderedDict`), and it is invisible in the experiment output: a correct answer
# gets scored as a failure and the task appears to degrade with temperature.
IDIOM = [
    ("T2", "T2_code_function",
     "import collections\n"
     "def timed(fn):\n    return fn\n\n"
     "class LRUCache:\n"
     "    def __init__(self, capacity):\n        self.cap = capacity\n"
     "        self.d = collections.OrderedDict()\n"
     "    def get(self, key):\n        \"\"\"Get a key.\"\"\"\n        return 1\n"
     "    def put(self, key, value):\n        \"\"\"Put a key.\"\"\"\n        pass\n"
     "    def __len__(self):\n        return 0\n"
     "    def __repr__(self):\n        return 'LRU'\n", True,
     "`import collections` + `collections.OrderedDict()` is what the prompt "
     "literally asks for"),
    ("T2", "T2_code_function",
     "from collections import OrderedDict\n"
     "def timed(fn):\n    return fn\n\n"
     "class LRUCache:\n"
     "    def __init__(self, capacity):\n        self.cap = capacity\n"
     "        self.d = OrderedDict()\n"
     "    def get(self, key):\n        \"\"\"Get a key.\"\"\"\n        return 1\n"
     "    def put(self, key, value):\n        \"\"\"Put a key.\"\"\"\n        pass\n"
     "    def __len__(self):\n        return 0\n"
     "    def __repr__(self):\n        return 'LRU'\n", True,
     "`from collections import OrderedDict` is equally valid"),
    ("T2", "T2_code_function",
     "import collections\n"
     "def timed(fn):\n    return fn\n\n"
     "class LRUCache:\n"
     "    def __init__(self, capacity):\n        self.cap = capacity\n"
     "        self.d = {}\n"
     "    def get(self, key):\n        \"\"\"Get a key.\"\"\"\n        return 1\n"
     "    def put(self, key, value):\n        \"\"\"Put a key.\"\"\"\n        pass\n"
     "    def __len__(self):\n        return 0\n"
     "    def __repr__(self):\n        return 'LRU'\n", False,
     "importing collections but storing into a plain dict does not satisfy it"),
]


def main():
    fails = total = 0

    for task, (good_text, _) in GOOD.items():
        ok, why = Q.check(task, good_text)
        total += 1
        if not ok:
            fails += 1
        # PASS = expectation held, i.e. the good sample was accepted.
        print(f"{'PASS' if ok else 'FAIL'}  {task:<22} good accepted — {why}")

    print()
    for task, samples in BAD.items():
        for label, text in samples:
            ok, why = Q.check(task, text)
            total += 1
            if ok:                      # a bad sample that was accepted
                fails += 1
            # PASS = expectation held, i.e. the bad sample was rejected.
            print(f"{'PASS' if not ok else 'FAIL'}  {task:<22} "
                  f"bad ({label}) rejected — "
                  f"{'ACCEPTED (bad!)' if ok else why}")

    print()
    for tag, task, text, want_ok, note in IDIOM:
        ok, why = Q.check(task, text)
        total += 1
        if ok != want_ok:
            fails += 1
        print(f"{'PASS' if ok == want_ok else 'FAIL'}  {task:<22} "
              f"idiom {tag} {'accepted' if want_ok else 'rejected'} — "
              f"{why if ok != want_ok else note}")

    print()
    print(f"{total - fails}/{total} validator expectations met")
    if fails:
        print(f"{fails} FAILED — a validator that accepts bad output makes the "
              f"whole experiment\nreport valid_rate=1.00 at every temperature "
              f"and is measuring nothing.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
