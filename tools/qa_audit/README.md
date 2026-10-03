# qa_audit

Make an evaluation's **judge** auditable.

This library exists because of one observation: *a bug in your judge is the most
expensive kind of measurement bug, because it is invisible in the results.*

A validator that rejects a correct answer does not report "something is
wrong". It reports **"this task degrades with temperature"** — and that reads
exactly like a finding. In the project this was extracted from, a judge checked
`useState` when the prompt asked for `useRef` + `useEffect`, and for three
rounds produced a confident, wrong, publishable conclusion about a model.

```
$ python3 selftest.py
qa_audit self-test clean: every guard fires, and every sound configuration passes
```

---

## The three rules

**1. A requirement is decided or explicitly unmeasured — never silently absent.**
A requirement nobody looks at produces no error, no warning, and no number. It
just quietly inflates a pass rate. `Requirement` refuses to be constructed
without a quote (so the transcription is reviewable), refuses `tier="semantic"`
together with `checked=True` (a contradiction), and requires a substantive
reason for anything unmeasured.

**2. A fixture must fail for the reason it names.**
Passing is not enough. A differential sample that is rejected for a *different*
reason proves nothing about the rule it is named after, and a mutation that
produces text identical to the good sample tests nothing at all — while
reading, in any report, exactly like coverage.

**3. A metric is tested on constructed inputs, not on the data it will score.**
A metric that cannot separate a pair that *must* differ is reporting its own
construction; its flat line is indistinguishable from a real result. A metric
sensitive to wording it should ignore is measuring the wrong thing.

---

## Install

No dependencies. Copy the directory, or `pip install -e tools/qa_audit`.

## Use

```python
from qa_audit import Requirement, Differential, MetricCase, run_audit

rep = run_audit(
    name="my-eval",
    requirements=[
        Requirement(id="uses_ordered_dict", tier="static",
                    quote="use collections.OrderedDict",
                    how="ast.ImportFrom node naming OrderedDict"),
        Requirement(id="evicts_least_recent", tier="exec",
                    quote="evict the least recently used key",
                    how="exec_probes.probe_lru", runner="exec_probes"),
        Requirement(id="writes_well", tier="semantic", checked=False,
                    quote="有画面感",
                    why="no mechanical or executable predicate decides quality"),
    ],
    good_samples={"my_task": "...a sample satisfying everything..."},
    check=my_checker,                     # (task, text) -> (ok, reason)
    differentials=[
        # must be rejected, and the reason must name the rule
        Differential(task="my_task", requirement="uses_ordered_dict",
                     sample="...imports collections and calls .get()...",
                     keyword="OrderedDict"),
        # a legal alternative form the judge must NOT reject (E1)
        Differential(task="my_task", requirement="uses_ordered_dict",
                     sample="...from collections import OrderedDict...",
                     must_reject=False),
    ],
    metrics=[
        ("self_consistency", my_canonical, [
            MetricCase(name="two implementations", task="my_task",
                       left=impl_a, right=impl_b, should_differ=True),
        ]),
    ],
)
print(rep.summary())
raise SystemExit(0 if rep.ok else 1)
```

## Executable probes

When a requirement is about *behaviour* — eviction order, arithmetic, a state
machine — `exec` the artifact and drive it. This is strictly stronger than any
predicate, and it is often the only honest option: 「超出时淘汰最久未使用的键」
(the entire point of an LRU cache) cannot be decided by a regex, so a
presence-only checker reports a perfect score forever.

```python
from qa_audit import make_exec_runner

PROBES = {                       # module level: the child imports it by name
    "my_task": {"evicts_lru": probe_lru},
}
run = make_exec_runner(module="my_probes", table="PROBES", strip=strip_fence)
results = run(artifact_text, "my_task")
```

The child runs in a subprocess and reports three things that otherwise corrupt
or hang an audit:

- **artifact stdout** is captured, so a payload that prints — and the prompt for
  a timing decorator asks it to — cannot corrupt the result stream. This was a
  real bug: every sample reported as failed because the probe shared stdout.
- **non-termination** is a reported failure, not a hang. An audit that hangs is
  worse than an audit that fails.
- **crashes** become data, not a dead run.

Probes must have negative controls too. The load-bearing one for an LRU is a
FIFO: `put A,B,C; get A; put D` evicts **B** under LRU and **A** under FIFO, so
a probe that cannot separate them is checking presence, not behaviour.

## Calibrate thresholds from data, do not guess them

The first version of the shape probe used a `1e-6` absolute tolerance and
produced a false failure on correct code, because one generation rounded π. The
detail message (`Circle(2).area() = 12.56636, expected 12.5663706`) is what made
that obvious. When a threshold is the suspect, **print both numbers** — a vague
"failed" costs a debugging round-trip that the message would have saved.

## What this library does not do

It does not tell you whether your model is good, and it does not fix a judge
that disagrees with you. It makes the disagreement *visible*, which is the part
that is otherwise silent. A clean audit is a necessary condition for a
trustworthy number, never a sufficient one — an unmeasured requirement remains
unmeasured, and the ledger says so by name.
