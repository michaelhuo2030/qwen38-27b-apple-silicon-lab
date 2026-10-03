#!/usr/bin/env python3
"""
exec_probes.py — behavioural probes for the Python tasks, as a module-level table.

    python3 exec_probes.py <temp_quality.json> [out.json]

Why execution. `AUDIT_CHECKLIST.md` records the requirements a regex cannot
decide. For Python there is no reason to leave them undecided: `exec` the
module and exercise the behaviour. A static checker can only confirm a method
*exists*, which is why 「超出时淘汰最久未使用的键」 — the entire point of an LRU
cache — sat unchecked while the task reported a perfect score.

The table is module level so `qa_audit.make_exec_runner` can import it by name
in the child process, which keeps what executes inspectable instead of pickled.

Scope, stated honestly: the React component is NOT covered. Deciding
「用户手动上滚后暂停自动滚动」 needs a JSX runtime and an effect scheduler,
there is no offline transpiler on this machine, and hand-writing one would be
building another unverified instrument inside an audit whose subject is
unverified instruments. It stays NOT_MEASURED, with the reason on the ledger.
"""
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def strip_fence(text: str) -> str:
    """Drop a markdown code fence before exec.

    The model wraps code in ``` on ~75% of generations regardless of
    temperature, and the prompt forbids it. That is scored separately as a
    formatting failure, so `compile()` must not be what notices it — a
    SyntaxError at line 1 is indistinguishable from broken code, and would
    report every fenced sample as a total failure.
    """
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", t, re.S)
    return m.group(1) if m else t


def _load(mod: str) -> dict:
    ns: dict = {}
    exec(compile(strip_fence(mod), "<gen>", "exec"), ns)
    return ns


def _close(got, want) -> bool:
    return abs(got - want) <= max(1e-9, 1e-4 * abs(want))


def probe_lruncache(mod):
    """Return (ok, detail). Exercises eviction *order*, not method presence.

    The discriminating sequence: put A,B,C; get A; put D. An LRU has just
    refreshed A, so B is coldest and B is evicted. A FIFO evicts A. A static
    presence check passes both, which is the whole reason this probe exists.
    """
    try:
        ns = _load(mod)
        C = ns.get("LRUCache")
        if C is None:
            return False, "no LRUCache class in the module namespace"
        c = C(3)
        for k in "ABC":
            c.put(k, k.lower())
        if len(c) != 3:
            return False, f"__len__() = {len(c)} after 3 puts, expected 3"
        if c.get("A") != "a":
            return False, f"get('A') returned {c.get('A')!r}, expected 'a'"
        c.put("D", "d")
        present = {k for k in "ABCD" if c.get(k) is not None}
        if "B" in present and "A" not in present:
            return False, ("evicted the key that was just read — this is FIFO, "
                           "not LRU")
        if present != {"A", "C", "D"}:
            return False, (f"after the 4th put the live keys are "
                           f"{sorted(present)}, expected A,C,D")
        if len(c) != 3:
            return False, f"capacity not enforced: __len__() = {len(c)}"
        return True, ("eviction order is LRU (the refreshed key survived and B "
                      "was evicted); capacity enforced")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def probe_timed(mod):
    """`timed(fn)` must report a duration, not merely exist."""
    try:
        ns = _load(mod)
        timed = ns.get("timed")
        if timed is None:
            return False, "no timed function"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            timed(lambda: 42)()
        out = buf.getvalue()
        if not out.strip():
            return False, "timed() ran but produced no output"
        return True, f"timed() printed {out.strip()[:40]!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def probe_shapes(mod):
    """Check the areas are arithmetically right, not merely defined."""
    try:
        ns = _load(mod)
        Circle = ns.get("Circle")
        Rectangle = ns.get("Rectangle")
        Triangle = ns.get("Triangle")
        total = ns.get("total_area")
        if not all([Circle, Rectangle, Triangle, total]):
            return False, "missing Circle/Rectangle/Triangle/total_area"
        c, r, t = Circle(2), Rectangle(3, 4), Triangle(6, 5)
        if not _close(c.area(), math.pi * 4):
            return False, f"Circle(2).area() = {c.area()}, expected {math.pi*4}"
        if not _close(r.area(), 12):
            return False, f"Rectangle(3,4).area() = {r.area()}, expected 12"
        if not _close(t.area(), 15):
            return False, f"Triangle(6,5).area() = {t.area()}, expected 15"
        if not _close(total([c, r, t]), math.pi * 4 + 27):
            return False, f"total_area returned {total([c, r, t])}"
        return True, "pi*r^2, w*h, b*h/2 all correct; total_area sums"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# Module level so the child process can import it by name.
PROBES = {
    "T2_code_function": {
        "lru_eviction": probe_lruncache,
        "timed_prints_duration": probe_timed,
    },
    "T3_code_repetitive": {
        "areas_correct": probe_shapes,
    },
}


def _runner():
    sys.path.insert(0, str(_HERE.parent / "tools"))
    from qa_audit import make_exec_runner
    return make_exec_runner(module="exec_probes", table="PROBES",
                            strip=strip_fence, sys_paths=[str(_HERE)])


def main() -> int:
    src = Path(sys.argv[1])
    blob = json.loads(src.read_text())
    run = _runner()

    for r in blob["results"]:
        if r.get("transport_error") or not r.get("text"):
            continue
        if r["task"] in PROBES:
            r["exec_checks"] = run(r["text"], r["task"])

    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=1))

    for task, names in PROBES.items():
        rows = [r for r in blob["results"]
                if r.get("task") == task and r.get("exec_checks")]
        if not rows:
            continue
        print(f"\n{task}  ({len(rows)} generations executed)")
        for n in names:
            ok = sum(1 for r in rows if r["exec_checks"].get(n, {}).get("ok"))
            print(f"  {n:26s} {ok}/{len(rows)} pass")
        for r in rows:
            for n, v in r["exec_checks"].items():
                if not v.get("ok"):
                    print(f"    T={r['temp']} rep={r['rep']} {n}: "
                          f"{v['detail'][:92]}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
