#!/usr/bin/env python3
"""
selftest.py — the audit tool, audited by the audit tool.

    python3 selftest.py

An audit library that cannot demonstrate its own guards is a library you should
not trust with a number. So each guard is exercised here against a deliberately
broken input, and the test *fails* if the guard does not fire. If a future
change makes `audit_differentials` stop catching no-op mutations, this exits
non-zero — which is the point: the guard is itself regression-protected.

Run from anywhere:  python3 tools/qa_audit/selftest.py
"""
import sys
from pathlib import Path

# This file lives *inside* the package directory, so the importable
# root is its parent: tools/, where qa_audit/ is a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qa_audit import (Differential, MetricCase, Requirement, Report,
                      audit_differentials, audit_ledger, audit_metric,
                      make_exec_runner, run_audit)

FAILURES: list[str] = []


def expect_guard_fires(guard_name: str, build_report) -> None:
    """The guard must produce at least one failure on a broken input."""
    rep = build_report()
    if rep.ok:
        FAILURES.append(f"guard {guard_name!r} did NOT fire on a broken input — "
                        f"it would pass anything")
        print(f"  FAIL  {guard_name}: guard did not fire")
    else:
        print(f"  ok    {guard_name}: fires ({len(rep.failures)} caught)")


def ok(cond, msg):
    if cond:
        print(f"  ok    {msg}")
    else:
        FAILURES.append(msg)
        print(f"  FAIL  {msg}")


# --- a tiny toy evaluator, deliberately naive, used only as a target ---------
def toy_check(task: str, text: str):
    if task == "toy" and "NEEDED" not in text:
        return False, "missing NEEDED"
    if task == "toy_formats" and text.strip() != text.strip().upper():
        return False, "must be uppercase"
    # `toy_unchecked` has a stated requirement the judge never looks at — the
    # condition the "unchecked rule" guard exists to catch. Building it needs a
    # task that genuinely ignores its own requirement; reusing `toy` here would
    # pass, because toy does check the requirement, and the guard would
    # correctly stay silent. That mistake — a fixture that does not construct
    # the condition it is named after — is the same E4 class the guard detects.
    return True, "ok"


GOOD = {"toy": "has NEEDED", "toy_formats": "LOUD", "toy_unchecked": "fine"}


def main() -> int:
    print("1. ledger guards")
    # A requirement marked checked with no mechanism must be caught.
    expect_guard_fires("checked-without-mechanism", lambda: run_audit(
        "broken", [Requirement(id="r1", tier="static", quote="q", how="")],
        GOOD, toy_check))
    # An unmeasured requirement with no reason must be caught at construction.
    try:
        Requirement(id="r2", tier="semantic", checked=False, quote="q")
        FAILURES.append("unmeasured requirement with no reason was accepted")
        print("  FAIL  silent-unmeasured: not rejected at construction")
    except ValueError:
        print("  ok    silent-unmeasured: rejected at construction")
    # tier=semantic + checked is a contradiction and must be rejected.
    try:
        Requirement(id="r3", tier="semantic", checked=True, quote="q", how="x")
        FAILURES.append("tier=semantic + checked=True was accepted")
        print("  FAIL  semantic-checked: not rejected")
    except ValueError:
        print("  ok    semantic-checked: rejected at construction")
    # An invented tier must be rejected.
    try:
        Requirement(id="r4", tier="regexish", quote="q")
        FAILURES.append("an invented tier was accepted")
        print("  FAIL  invented-tier: not rejected")
    except ValueError:
        print("  ok    invented-tier: rejected")
    # A well-formed ledger must pass.
    rep = run_audit("good", [
        Requirement(id="r1", tier="static", quote="q", how="a regex"),
        Requirement(id="r2", tier="semantic", checked=False, quote="q",
                    why="needs a human, no predicate decides it"),
    ], GOOD, toy_check)
    ok(rep.ok, f"well-formed ledger passes ({rep.checks} checks)")

    print("\n2. differential guards")
    # No-op mutation must be caught.
    expect_guard_fires("no-op mutation", lambda: run_audit(
        "noop", [Requirement(id="needed", tier="static", quote="q", how="regex")],
        GOOD, toy_check,
        differentials=[Differential(task="toy", requirement="needed",
                                    sample=GOOD["toy"], keyword="NEEDED")]))
    # Rejected for the wrong reason must be caught.
    expect_guard_fires("wrong rejection reason", lambda: run_audit(
        "reason", [Requirement(id="needed", tier="static", quote="q", how="regex")],
        GOOD, toy_check,
        differentials=[Differential(task="toy", requirement="needed",
                                    sample="nothing", keyword="BANANA")]))
    # A rule with no check must be caught.
    expect_guard_fires("unchecked rule accepts violation", lambda: run_audit(
        "gap", [Requirement(id="forbidden_word", tier="static", quote="q",
                            how="claims a regex check")],
        GOOD, toy_check,
        differentials=[Differential(task="toy_unchecked",
                                    requirement="forbidden_word",
                                    sample="uses the forbidden word",
                                    keyword="forbidden_word")]))
    # A too-narrow judge: a legal alternative form must be caught (must_reject=False).
    expect_guard_fires("judge narrower than prompt", lambda: run_audit(
        "narrow", [Requirement(id="case", tier="static", quote="q", how="upper")],
        {"toy_formats": "LOUD"}, toy_check,
        differentials=[Differential(task="toy_formats", requirement="case",
                                    sample="quiet", must_reject=False)]))
    # A well-formed differential set must pass.
    rep = run_audit("ok2",
                    [Requirement(id="needed", tier="static", quote="q",
                                 how="regex")],
                    GOOD, toy_check,
                    differentials=[Differential(task="toy",
                                                requirement="needed",
                                                sample="no marker",
                                                keyword="NEEDED")])
    ok(rep.ok, f"well-formed differentials pass ({rep.checks} checks)")

    print("\n3. metric guards")
    ident = lambda task, text: "SAME"          # noqa: E731 — a constant metric
    expect_guard_fires("constant metric", lambda: (lambda rep: (
        audit_metric(rep, "ident", ident,
                     [MetricCase(name="must differ", left="a", right="b",
                                 should_differ=True, task="t")]), rep)[1])(
        Report("m")))
    sens = lambda task, text: text              # noqa: E731 — wording-sensitive
    expect_guard_fires("wording-sensitive metric", lambda: (lambda rep: (
        audit_metric(rep, "sens", sens,
                     [MetricCase(name="must match", left="SAME THING",
                                 right="Same thing", should_differ=False,
                                 task="t")]), rep)[1])(Report("m")))
    rep = Report("m")
    good_canon = lambda task, text: text.strip().lower()   # noqa: E731
    audit_metric(rep, "good", good_canon, [
        MetricCase(name="case only", left="LOUD", right="loud",
                   should_differ=False, task="t"),
        MetricCase(name="real difference", left="a", right="b",
                   should_differ=True, task="t")])
    ok(rep.ok, f"a sound metric passes both directions ({rep.checks} checks)")

    print("\n4. exec runner isolation")
    run = make_exec_runner(module="qa_audit_selftest_probes", table="PROBES",
                           sys_paths=[str(Path(__file__).resolve().parent)])
    out = run("print('noise on stdout')\nRESULT_OK = 1 + 1", "t")
    ok(bool(out) and any(v.get("ok") for v in out.values()),
       f"runner survives artifact stdout and still reports "
       f"({list(out) or 'no probes'})")
    quick = make_exec_runner(module="qa_audit_selftest_probes", table="PROBES",
                            timeout=3.0,
                            sys_paths=[str(Path(__file__).resolve().parent)])
    out = quick("while True: pass", "hangs")
    ok(any("timed out" in v.get("detail", "") for v in out.values())
       or any("did not terminate" in v.get("detail", "")
              for v in out.values()),
       "runner reports a non-terminating artifact as a failure, not a hang")
    out = run("raise SystemExit(3)", "crashes")
    ok(bool(out) and all(v.get("ok") is False for v in out.values()),
       "runner turns an artifact crash into a reported failure")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} SELF-TEST FAILURES")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("qa_audit self-test clean: every guard fires, and every sound "
          "configuration passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
