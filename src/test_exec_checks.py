#!/usr/bin/env python3
"""
test_exec_checks.py — do the executable probes actually reject anything?

Run: python3 test_exec_checks.py

A behavioural probe that never fails is worse than no probe: it reports 100% at
every temperature and looks like a finding. So each probe gets a correct sample
it must accept and a set of wrong ones it must reject *for the stated reason*.

The load-bearing negative control is the FIFO cache. An LRU that never refreshes
recency and a FIFO differ in exactly one observable way — given put A, put B,
put C, get A, put D, a FIFO evicts A while an LRU evicts B — so a probe that
cannot separate those two is not testing eviction order at all, only presence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exec_probes as E

LRU_OK = (
    "import collections\n"
    "def timed(fn):\n"
    "    def w(*a, **k):\n"
    "        import time; t0=time.time(); r=fn(*a,**k)\n"
    "        print(fn.__name__, round(time.time()-t0, 6)); return r\n"
    "    return w\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        self.cap = capacity\n"
    "        self.d = collections.OrderedDict()\n"
    "    @timed\n"
    "    def get(self, key, default=None):\n"
    "        \"\"\"Get.\"\"\"\n"
    "        if key not in self.d: return default\n"
    "        self.d.move_to_end(key); return self.d[key]\n"
    "    @timed\n"
    "    def put(self, key, value):\n"
    "        \"\"\"Put.\"\"\"\n"
    "        if key in self.d: self.d.move_to_end(key)\n"
    "        self.d[key] = value\n"
    "        while len(self.d) > self.cap: self.d.popitem(last=False)\n"
    "    def __len__(self):\n        return len(self.d)\n"
    "    def __repr__(self):\n        return 'LRU'\n"
)

# Same class, no recency refresh: a FIFO wearing an LRU's name.
LRU_IS_FIFO = LRU_OK.replace("        self.d.move_to_end(key); return self.d[key]\n", "        return self.d[key]\n")
LRU_IS_FIFO = LRU_IS_FIFO.replace(
    "        if key in self.d: self.d.move_to_end(key)\n", "")

# Never evicts at all.
LRU_NEVER_EVICTS = LRU_OK.replace(
    "        while len(self.d) > self.cap: self.d.popitem(last=False)\n", "")

# Evicts the *newest* key instead of the coldest.
LRU_EVICTS_NEWEST = LRU_OK.replace(
    "        while len(self.d) > self.cap: self.d.popitem(last=False)\n",
    "        while len(self.d) > self.cap: self.d.popitem()\n")

# timed() that exists but reports nothing.
TIMED_SILENT = LRU_OK.replace(
    "        import time; t0=time.time(); r=fn(*a,**k)\n"
    "        print(fn.__name__, round(time.time()-t0, 6)); return r\n",
    "        return fn(*a, **k)\n")

SHAPES_OK = (
    "import math\n"
    "class Shape:\n"
    "    def area(self):\n        raise NotImplementedError\n"
    "    def describe(self):\n        return 's'\n"
    "class Circle(Shape):\n"
    "    def __init__(self, radius):\n        self.r = radius\n"
    "    def area(self):\n        return math.pi * self.r ** 2\n"
    "class Rectangle(Shape):\n"
    "    def __init__(self, width, height):\n        self.w, self.h = width, height\n"
    "    def area(self):\n        return self.w * self.h\n"
    "class Triangle(Shape):\n"
    "    def __init__(self, base, height):\n        self.b, self.h = base, height\n"
    "    def area(self):\n        return self.b * self.h / 2\n"
    "def total_area(shapes):\n    return sum(s.area() for s in shapes)\n"
)
# Every area present but arithmetically wrong — exactly what a presence-only
# checker scores as a pass.
WRONG_CIRCLE = SHAPES_OK.replace("math.pi * self.r ** 2", "self.r ** 2")
WRONG_RECT = SHAPES_OK.replace("return self.w * self.h", "return self.w + self.h")
WRONG_TRI = SHAPES_OK.replace("return self.b * self.h / 2", "return self.b * self.h")
# total_area returns the first shape instead of the sum.
WRONG_TOTAL = SHAPES_OK.replace("return sum(s.area() for s in shapes)",
                                "return shapes[0].area() if shapes else 0")

# (task, sample, {probe: must_pass}, keyword when a probe must reject)
#
# Expectations are per probe, not per sample. A FIFO cache is still a sample
# with a perfectly good `timed` decorator, so demanding that its `timed` probe
# also fail would be asking the fixture to test something it does not contain —
# the same mistake as a fixture rejected for the wrong reason, one level up.
CASES = [
    ("T2_code_function", LRU_OK,
     {"lru_eviction": True, "timed_prints_duration": True}, ""),
    # The load-bearing negative control: recency is never refreshed, so the
    # eviction order is FIFO. Everything else about the class is correct.
    ("T2_code_function", LRU_IS_FIFO,
     {"lru_eviction": False, "timed_prints_duration": True}, "FIFO"),
    ("T2_code_function", LRU_NEVER_EVICTS,
     {"lru_eviction": False, "timed_prints_duration": True}, ""),
    ("T2_code_function", LRU_EVICTS_NEWEST,
     {"lru_eviction": False, "timed_prints_duration": True}, ""),
    # Correct eviction, but `timed` exists and reports nothing.
    ("T2_code_function", TIMED_SILENT,
     {"lru_eviction": True, "timed_prints_duration": False}, "no output"),
    ("T2_code_function",
     "class LRUCache:\n    def get(self, k):\n        return None\n",
     {"lru_eviction": False, "timed_prints_duration": False}, ""),
    ("T3_code_repetitive", SHAPES_OK, {"areas_correct": True}, ""),
    ("T3_code_repetitive", WRONG_CIRCLE, {"areas_correct": False}, "Circle"),
    ("T3_code_repetitive", WRONG_RECT, {"areas_correct": False}, "Rectangle"),
    ("T3_code_repetitive", WRONG_TRI, {"areas_correct": False}, "Triangle"),
    ("T3_code_repetitive", WRONG_TOTAL, {"areas_correct": False}, "total_area"),
    ("T3_code_repetitive", "def area(r):\n    return 3.14*r*r\n",
     {"areas_correct": False}, ""),
]


def main() -> int:
    fails = 0
    for task, src, want, kw in CASES:
        run = E._runner()
        got = run(src, task)
        if not got:
            print(f"FAIL  {task:<20} no probe ran at all")
            fails += 1
            continue
        for probe, v in got.items():
            expect = want.get(probe)
            if expect is None:
                print(f"FAIL  {task:<20} probe {probe!r} is not declared in the "
                      f"case's expectations")
                fails += 1
                continue
            if v["ok"] != expect:
                print(f"FAIL  {task:<20} {probe}: expected "
                      f"{'accept' if expect else 'reject'}, got "
                      f"{'accepted' if v['ok'] else v['detail'][:70]}")
                fails += 1
            elif not expect and kw and kw.lower() not in v["detail"].lower():
                print(f"FAIL  {task:<20} {probe}: rejected but not naming "
                      f"'{kw}' — rejected for the wrong reason. got: "
                      f"{v['detail'][:70]}")
                fails += 1
            else:
                verdict = "accepted" if v["ok"] else f"rejected ({v['detail'][:50]})"
                print(f"PASS  {task:<20} {probe:<24} {verdict}")
    print()
    if fails:
        print(f"{fails} FAILED — a behavioural probe that cannot reject is "
              f"measuring nothing")
    else:
        print("all executable probes discriminate")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
