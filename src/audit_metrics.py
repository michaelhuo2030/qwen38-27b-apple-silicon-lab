#!/usr/bin/env python3
"""
audit_metrics.py — do the self-consistency metrics measure anything?

Run: python3 audit_metrics.py

`audit_validators.py` checks the pass/fail side. This checks the other side:
the `canonical()` projections that self-consistency is computed from. A metric
can be perfect at its job and still report a constant, and that constant looks
exactly like a finding on a chart.

Two known artifacts live here, both found by hand:

  T2/T3  canonical() keeps only the *names* of defined functions. The prompt
         names the required methods, so every correct answer projects to the
         same string and self-consistency is 100% at every temperature — a flat
         line drawn by the ruler, not by the model.
  T7     canonical() keeps the *last* number in the output. The prompt asks for
         a reordering discussion after the main calculation, so the last number
         belongs to a second derivation. T7's 20% self-consistency is that
         projection being wrong, not the model being unstable.

Both are invisible in a results table. So the metric is tested on constructed
inputs where the right answer is known by construction:

  DISCRIMINATION  a pair that must score differently must score differently
  EQUIVALENCE     a pair that means the same must score the same
  DISCRIMINATION RATE over a batch of genuinely different samples: if the
                  metric cannot separate them, it is reporting the prompt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_temp_quality import canonical

FAIL = []


def bad(msg):
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


# --- T2: two genuinely different correct LRU caches ---------------------------
T2_A = (
    "import collections\n"
    "def timed(fn):\n    return fn\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n        self.cap = capacity\n"
    "        self.d = collections.OrderedDict()\n"
    "    def get(self, key):\n        \"\"\"Get.\"\"\"\n        return self.d.get(key)\n"
    "    def put(self, key, value):\n        \"\"\"Put.\"\"\"\n"
    "        self.d[key] = value\n        self.d.move_to_end(key)\n"
    "        if len(self.d) > self.cap:\n            self.d.popitem(last=False)\n"
    "    def __len__(self):\n        return len(self.d)\n"
    "    def __repr__(self):\n        return f'LRUCache({self.cap})'\n"
)
# Same required API, genuinely different implementation and different helper
# structure. A useful self-consistency metric should notice *something*.
T2_B = T2_A.replace(
    "        self.d[key] = value\n        self.d.move_to_end(key)\n"
    "        if len(self.d) > self.cap:\n            self.d.popitem(last=False)\n",
    "        if key in self.d:\n            del self.d[key]\n"
    "        self.d[key] = value\n"
    "        while len(self.d) > self.cap:\n"
    "            self.d.popitem(last=False)\n"
)

# --- T7: one answer, two orderings -------------------------------------------
T7_MAIN = (
    "第一步 200×0.8=160。第二步 160-30=130。第三步 130×0.85=110.5。"
    "第四步 110.5-20=90.5 元。所以实付 90.5 元。"
)
T7_WITH_REORDER = T7_MAIN + (
    "如果改成先减 30 再打 8 折：200-30=170 元，170×0.8=136 元，"
    "结果是 136 元，与 90.5 元并不相同，因为乘法与减法不满足交换律。"
)

# --- T1: same data, different formatting -------------------------------------
T1_A = '[{"id":1,"name":"Lamp","price":9.5,"category":"L","inStock":true,"tags":["a","b"]}]'
T1_B = '[\n  {  "name" : "Lamp",  "id" : 1,  "category" : "L" , "inStock": true,\n     "price": 9.5,  "tags" : ["a", "b"] }\n]'


def main() -> int:
    print("\n[M1] T2 canonical must separate two different implementations")
    a, b = canonical("T2_code_function", T2_A), canonical("T2_code_function", T2_B)
    if a == b:
        bad(f"T2 canonical is identical for two different implementations: "
            f"{a!r} — self-consistency for code tasks is constant by "
            f"construction, not a finding")
    else:
        ok(f"T2 separates: {a!r} vs {b!r}")

    print("\n[M2] T3 canonical must separate two different implementations")
    t3a = ("class Shape:\n    def area(self):\n        return 0\n"
           "    def describe(self):\n        return 's'\n"
           "class Circle(Shape):\n    def __init__(self, radius):\n        self.r = radius\n"
           "    def area(self):\n        return 3.14 * self.r ** 2\n"
           "class Rectangle(Shape):\n    def __init__(self, width, height):\n"
           "        self.w, self.h = width, height\n    def area(self):\n"
           "        return self.w * self.h\n"
           "class Triangle(Shape):\n    def __init__(self, base, height):\n"
           "        self.b, self.h = base, height\n    def area(self):\n"
           "        return self.b * self.h / 2\n"
           "def total_area(shapes):\n    return sum(s.area() for s in shapes)\n")
    t3b = t3a.replace("    def area(self):\n        return 0\n",
                      "    def area(self):\n        raise NotImplementedError\n")
    a, b = canonical("T3_code_repetitive", t3a), canonical("T3_code_repetitive", t3b)
    if a == b:
        bad(f"T3 canonical is identical for two different implementations: {a!r}")
    else:
        ok(f"T3 separates: {a!r} vs {b!r}")

    print("\n[M3] T7 canonical must track the main answer, not the last number")
    c1 = canonical("T7_math_reasoning", T7_MAIN)
    c2 = canonical("T7_math_reasoning", T7_WITH_REORDER)
    if c1 != "90.50":
        bad(f"T7 canonical of a complete answer is {c1!r}, expected '90.50' — "
            f"the projection is not the main answer")
    else:
        ok("T7 canonical of the main answer is 90.50")
    if c1 == c2:
        ok("T7 canonical is stable when a correct reordering discussion is appended")
    else:
        bad(f"T7 canonical changes from {c1!r} to {c2!r} purely because the "
            f"prompt's second question was answered — self-consistency then "
            f"measures the projection, not the model")

    print("\n[M4] T1 canonical must ignore formatting, track the data")
    a, b = canonical("T1_json_structured", T1_A), canonical("T1_json_structured", T1_B)
    if a != b:
        bad(f"T1 canonical differs on identical data reformatted: {a!r} vs {b!r}")
    else:
        ok("T1 canonical is formatting-invariant")

    print("\n[M5] discrimination rate per task over a deliberately varied batch")
    # Each batch is built so that a metric with any discriminating power at all
    # must produce more than one distinct value.
    batches = {
        "T1_json_structured": [
            T1_A,
            T1_A.replace('"Lamp"', '"Fan"'),
            T1_A.replace('"id":1', '"id":2'),
            T1_A.replace('"L"', '"Kitchen"'),
        ],
        "T2_code_function": [T2_A, T2_B],
        "T6_qa_factual": [
            "为什么不是两次？为什么不是四次？ISN 有用。TIME_WAIT 等待。SYN-ACK 丢失则重传。",
            "为什么不是两次？为什么不是四次？ISN 有用。TIME_WAIT 等待。握手顺利完成。",
        ],
    }
    for task, batch in batches.items():
        vals = {canonical(task, t) for t in batch}
        if len(vals) < 2:
            bad(f"{task}: {len(batch)} deliberately different samples produced "
                f"{len(vals)} distinct canonical value(s) — the metric cannot "
                f"tell them apart")
        else:
            ok(f"{task}: {len(batch)} samples -> {len(vals)} distinct canonical values")

    print("\n[M6] code fingerprint must ignore docstring wording")
    # The fix claims rewording a docstring is not a different answer. If that
    # were not true the metric would be reporting on prose inside a code task.
    reworded = T2_A.replace('"""Get."""', '"""Look up a key, returning None if absent."""')
    a, b = canonical("T2_code_function", T2_A), canonical("T2_code_function", reworded)
    if a != b:
        bad(f"rewording a docstring changed the code fingerprint: {a} vs {b}")
    else:
        ok("docstring rewording leaves the fingerprint unchanged")

    print()
    if FAIL:
        print(f"{len(FAIL)} METRIC AUDIT FAILURES — a self-consistency number "
              f"from a constant metric looks exactly like a finding")
        return 1
    print("metric audit clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
