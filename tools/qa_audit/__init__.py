"""
qa_audit — make an evaluation's *judge* auditable.

Written because a judge's bug is the most expensive kind of measurement bug:
it is invisible in the results table. A validator that rejects a correct answer
does not report "something is wrong", it reports "this task degrades with
temperature", and that reads exactly like a finding.

Every mechanism here exists because it caught a real one.

    from qa_audit import Requirement, Differential, run_audit, degeneracy

    run_audit(
        name="my-eval",
        requirements=[
            Requirement(id="uses_ordered_dict", tier="static",
                        quote="use collections.OrderedDict",
                        how="ast.ImportFrom node naming OrderedDict"),
            Requirement(id="evicts_least_recent", tier="exec",
                        quote="evict the least recently used key",
                        how="exec_checks.probe_lru"),
            Requirement(id="writes_good_prose", tier="semantic",
                        quote="有画面感",
                        why="no mechanical predicate decides this"),
        ],
        good_samples={"my_task": "a sample satisfying everything"},
        differentials=[
            Differential(task="my_task", requirement="uses_ordered_dict",
                         sample="{...uses a plain dict...}", keyword="OrderedDict"),
        ],
        metrics={"self_consistency": canonical, "discriminating_pairs": [...]},
    )

The three rules this enforces, all of them learned the hard way:

1. A requirement may be CHECKED or NOT_MEASURED. It may never be silently
   absent — a requirement nobody looks at is invisible, and invisibility is how
   a broken judge survives review.

2. A test fixture must fail *for the reason it names*. Passing is not enough:
   reject for the wrong reason and the rule is untested. A fixture can also be
   rejected for being a no-op, which is worse than useless because it is
   indistinguishable from coverage.

3. A metric is tested on constructed inputs where the right answer is known by
   construction, not on the data it will be used to score. A metric that cannot
   separate a pair that must differ is reporting its own construction, and its
   flat line looks identical to a real result.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

VALID_TIERS = ("static", "exec", "semantic")


class Report:
    """Collects failures so a whole run can be reported, not just the first."""

    def __init__(self, name: str):
        self.name = name
        self.failures: list[str] = []
        self.checks = 0

    def fail(self, msg: str) -> None:
        self.failures.append(f"[{self.name}] {msg}")

    def check(self, ok: bool, msg: str) -> bool:
        self.checks += 1
        if not ok:
            self.fail(msg)
        return ok

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        n = len(self.failures)
        head = f"{self.name}: {self.checks} checks, "
        if self.ok:
            return head + "clean"
        body = "\n".join("  FAIL  " + f for f in self.failures)
        return f"{head}{n} FAILURES\n{body}"


# --- requirement ledger -----------------------------------------------------
@dataclass
class Requirement:
    """One thing a prompt asks for, and what happens to it.

    tier
        static   a regex or AST predicate decides it
        exec     running the artifact decides it — strongest available
        semantic no mechanical or executable predicate decides it honestly
    checked
        True only if something actually decides it. For `semantic` this must be
        False, and `why` must say why not — that sentence is the deliverable.
    """

    id: str
    tier: str = "static"
    checked: bool = True
    quote: str = ""          # the phrase in the prompt, so a reviewer can
    how: str = ""            # how it is decided
    why: str = ""            # required when checked is False
    runner: str = ""         # for tier="exec"

    def __post_init__(self):
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"{self.id}: tier must be one of {VALID_TIERS}, got {self.tier!r}. "
                f"Choosing 'semantic' to skip a check is allowed; inventing a "
                f"tier to avoid one is not.")
        if self.tier == "semantic" and self.checked:
            raise ValueError(
                f"{self.id}: tier='semantic' cannot be 'checked' — by "
                f"definition nothing mechanical decides it. Either promote it to "
                f"static/exec with a real mechanism, or mark it not measured.")
        if not self.checked and not self.why:
            raise ValueError(
                f"{self.id}: an unmeasured requirement needs a reason. Silence "
                f"is how an unchecked requirement stays invisible.")


@dataclass
class Differential:
    """A sample that breaks exactly one requirement.

    `keyword` must appear in the rejection reason. A fixture that is rejected
    for a different reason proves nothing about the rule it is named after.
    """

    task: str
    requirement: str
    sample: str
    keyword: str = ""
    must_reject: bool = True
    note: str = ""


def audit_ledger(rep: Report, requirements: Sequence[Requirement]) -> dict:
    """Enforce that every requirement is decided or explicitly not measured."""
    total = sum(1 for _ in requirements)
    checked = sum(1 for r in requirements if r.checked)
    unmeasured = [r for r in requirements if not r.checked]

    for r in requirements:
        rep.check(bool(r.quote),
                  f"{r.id}: no prompt quote — the transcription cannot be "
                  f"reviewed (E3)")
        if r.checked and r.tier in ("static", "exec"):
            rep.check(bool(r.how or r.runner),
                      f"{r.id}: marked checked but names no mechanism")
    for r in unmeasured:
        rep.check(len(r.why) > 20,
                  f"{r.id}: not measured with no substantive reason")
    return {"total": total, "checked": checked,
            "not_measured": len(unmeasured)}


# --- differential fixtures --------------------------------------------------
def audit_differentials(
    rep: Report,
    differentials: Sequence[Differential],
    good_samples: dict[str, str],
    check: Callable[[str, str], tuple[bool, str]],
) -> None:
    for d in differentials:
        if d.must_reject and d.sample == good_samples.get(d.task):
            rep.fail(f"{d.task}/{d.requirement}: the mutation produced text "
                     f"identical to the known-good sample — the fixture breaks "
                     f"nothing and reads as coverage (E4)")
            continue
        if not d.sample.strip():
            rep.fail(f"{d.task}/{d.requirement}: empty sample (E4)")
            continue
        ok, why = check(d.task, d.sample)
        if d.must_reject:
            if ok:
                rep.fail(f"{d.task}/{d.requirement}: violating this one "
                         f"requirement was still accepted (E2)")
            elif d.keyword and d.keyword.lower() not in why.lower():
                rep.fail(f"{d.task}/{d.requirement}: rejected, but the reason "
                         f"does not name {d.keyword!r} — rejected for the wrong "
                         f"reason (E4). got: {why[:90]}")
        else:
            if not ok:
                rep.fail(f"{d.task}/{d.requirement}: a legal alternative form "
                         f"was rejected — the judge is narrower than the prompt "
                         f"(E1). got: {why[:90]}")


def audit_good_samples(rep: Report, good_samples: dict[str, str],
                       check: Callable[[str, str], tuple[bool, str]]) -> None:
    """A sample satisfying the whole prompt must be accepted.

    Without this, tightening a requirement can only ever lower a score, and
    there is nothing to catch a check that rejects everything.
    """
    for task, text in good_samples.items():
        ok, why = check(task, text)
        rep.check(ok, f"{task}: a sample satisfying the whole prompt was "
                      f"rejected — {why}")


# --- metric degeneracy ------------------------------------------------------
@dataclass
class MetricCase:
    """A pair whose canonical relationship is known by construction."""

    name: str
    left: str
    right: str
    should_differ: bool
    task: str = ""


def audit_metric(rep: Report, metric_name: str,
                 canonical: Callable[[str, str], str],
                 cases: Sequence[MetricCase]) -> None:
    """A metric that cannot separate what must be separated is reporting itself.

    Two failure modes, and they look identical on a chart:
      - a metric that returns a constant regardless of the input, so its flat
        line is drawn by the instrument
      - a metric that projects the wrong thing, so it is busy and confident and
        still measuring something other than the task
    """
    for c in cases:
        a, b = canonical(c.task, c.left), canonical(c.task, c.right)
        differs = a != b
        if differs != c.should_differ:
            if c.should_differ:
                rep.fail(f"metric {metric_name!r} / {c.name}: two samples that "
                         f"must differ produced {a!r} and {b!r} — the metric is "
                         f"constant by construction (E5)")
            else:
                rep.fail(f"metric {metric_name!r} / {c.name}: two samples that "
                         f"mean the same produced {a!r} and {b!r} — the metric is "
                         f"sensitive to wording it should ignore (E6)")
        else:
            rep.checks += 1


# --- executable probes ------------------------------------------------------
_CHILD = r'''
import io, json, sys
for _p in {paths!r}:
    sys.path.insert(0, _p)
from {module} import {table}
probes = {table}.get({task!r}, {{}})
mod = sys.stdin.read()

# Generated code prints. The prompt for a timing decorator asks it to, and
# anything on stdout would corrupt the JSON result sharing the stream — a bug
# that shows up as "every sample failed" and looks like a broken artifact.
_real = sys.stdout
sys.stdout = io.StringIO()
out = {{}}
for _name, _fn in probes.items():
    try:
        _ok, _why = _fn(mod)
    except BaseException as e:            # noqa: BLE001 — a probe crash is data
        _ok, _why = False, f"{{type(e).__name__}}: {{e}}"
    out[_name] = {{"ok": _ok, "detail": _why}}
sys.stdout = _real
print(json.dumps(out))
'''


def make_exec_runner(module: str, table: str, strip=lambda s: s,
                     timeout: float = 15.0, sys_paths: Sequence[str] = ()):
    """Run behavioural probes against a generated artifact, in a subprocess.

    Isolation is not paranoia. Generated code loops forever, raises, and prints
    — and the printing one silently corrupts the result if the probe shares a
    stream with the payload. An audit that hangs is worse than an audit that
    fails, and a probe that crashes must *report* a failure rather than take the
    whole run down with it.

    The child resolves its probes by importing `module` and reading `table`, so
    what it runs is inspectable source rather than pickled bytecode. `table` must
    be a module-level dict of name -> (sample) -> (ok, detail).
    """
    here = str(Path(__file__).resolve().parent)
    paths = [str(Path(p).resolve()) for p in sys_paths] or [here]

    def run(sample: str, task: str) -> dict:
        src = _CHILD.format(paths=paths, module=module, table=table, task=task)
        try:
            p = subprocess.run([sys.executable, "-c", src],
                               input=strip(sample), capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"_harness": {"ok": False,
                                 "detail": f"probe timed out after {timeout}s "
                                           f"— the artifact did not terminate"}}
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()
            return {"_harness": {"ok": False,
                                 "detail": "probe harness failed: "
                                           + (tail[-1][:140] if tail
                                              else f"exit {p.returncode}")}}
        try:
            return json.loads(p.stdout)
        except json.JSONDecodeError:
            return {"_harness": {"ok": False,
                                 "detail": "probe produced no parseable result"}}

    return run


def run_audit(name, requirements, good_samples, check, differentials=(),
              metrics=None):
    """Run the three audits in one call and return a Report.

    `metrics` is a list of (metric_name, canonical, cases) triples, where cases
    are MetricCase objects built on constructed inputs.
    """
    rep = Report(name)
    summary = audit_ledger(rep, requirements)
    audit_good_samples(rep, good_samples, check)
    audit_differentials(rep, differentials, good_samples, check)
    for mname, canonical, cases in (metrics or ()):
        audit_metric(rep, mname, canonical, cases)
    rep.summary_text = summary
    return rep
