#!/usr/bin/env python3
"""
run_temp_quality.py — the measurement the router was missing.

What is already known (README §6): temperature's effect on *efficiency* is small
and task-specific, and T≈2.0 collapses the output into token soup. What is not
known, and what the router's temperature values are currently resting on:
whether temperature degrades the *output*.

The trap is picking a quality metric that is really a proxy for temperature
itself. "Longer output" and "more tokens at T=1.2" are the same fact wearing
two hats. So the metrics here are chosen to be independent of the knob:

  valid_rate     — passes an objective validator written from the prompt's own
                   stated requirements (JSON schema, Python AST, the arithmetic
                   answer, required points, brace balance, stated length/simile
                   count). A validator that never rejects anything reports 1.00
                   everywhere; test_quality_checks.py proves these reject.
  self_consistency
                 — over 5 samples at one temperature, the share held by the modal
                   answer. This is the Self-Consistency idea: if sampling is
                   still on the right answer, 5 draws agree; if it has drifted,
                   they scatter. Independent of how long the output is.
  degeneracy     — distinct-2 and longest single-token run. Catches collapse;
                   cannot tell you which answer is nicer.

Not measured: whether prose is *good*. Only that it meets the length and simile
count the prompt asked for. The report says so rather than implying otherwise.

Cost: 7 tasks x (1 reference at T=0 + 5 samples x 5 warmer settings) = 182
generations. T=0 gets one run because it is deterministic; five copies of the
same answer would only be measuring the sampler.
"""
import argparse
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

import omlx_client as L
import quality_checks as Q
from tasks import TASKS

# The tasks that have a validator, so only those are run. The other four exist
# in tasks.py to shape the α / efficiency work, not to be quality-scored.
TASKS_RUN = ["T1_json_structured", "T2_code_function", "T3_code_repetitive",
             "T6_qa_factual", "T7_math_reasoning", "T10_ts_component",
             "T8_creative_writing"]

# T1 asks for 12 objects x 6 fields; T2 and T7 were 100% truncated at the
# original budgets, T10 50% — and a run that always hits the token cap measures
# the cap, not the temperature. Every budget below was set from the observed
# token count with headroom, and `truncated` is recorded per run so the analysis
# can separate the two effects rather than confuse them.
MAX_TOKENS_OVERRIDE = {
    "T1_json_structured": 1800,     # observed 659
    "T2_code_function": 1800,       # observed 600 = cap, always cut
    "T3_code_repetitive": 900,      # observed 246
    "T6_qa_factual": 1100,          # observed 384
    "T7_math_reasoning": 1600,      # observed 600 = cap, always cut
    "T10_ts_component": 1800,       # observed 742, 50% cut
    "T8_creative_writing": 1200,    # observed 441
}
TEMPS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
REPS_WARM = 5          # samples per warm temperature
REPS_T0 = 1            # T=0 is deterministic


def canonical(task: str, text: str) -> str:
    """Reduce an output to the thing the task is actually about.

    Self-consistency compares answers, not wordings. Two JSON arrays with the
    same ids and names are the same answer; two essays are the same answer only
    if they are identical, which is why prose consistency is reported but is
    expected to be near zero above T=0.
    """
    body = Q._strip_fence(text)
    if task == "T7_math_reasoning":
        nums = Q._numbers(body)
        return f"{nums[-1]:.2f}" if nums else "none"
    if task == "T1_json_structured":
        s = body.find("[")
        if s >= 0:
            # Plain bracket counting. The previous one-liner
            #   depth += ch == "[" and depth + 1 or depth - (ch == "]")
            # evaluates to depth + (depth + 1) on "[", i.e. 2*depth+1, so the
            # scan never returned to zero and *every* T1 generation fell back to
            # the raw text. T1's self-consistency was therefore an exact-text
            # comparison while claiming to be an id+name+category comparison.
            depth, end = 0, None
            for i, ch in enumerate(body[s:], s):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end:
                try:
                    arr = json.loads(body[s:end])
                    if isinstance(arr, list):
                        return json.dumps(
                            [[o.get("id"), o.get("name"), o.get("category")]
                             for o in arr if isinstance(o, dict)],
                            ensure_ascii=False, sort_keys=True)
                except Exception:
                    pass
        return body[:200]
    if task in ("T2_code_function", "T3_code_repetitive"):
        # Structural fingerprint of the parsed code, with docstrings removed.
        #
        # The previous version returned the sorted list of *defined function
        # names*. The prompt names the required methods, so every correct
        # answer projected to the same string and self-consistency was a flat
        # 100% at every temperature — a constant produced by the metric, drawn
        # on the chart as though the model had drawn it. Docstrings go because
        # rewording one is not a different answer; the statement structure is.
        import ast as _ast
        import hashlib as _hl
        try:
            tree = _ast.parse(body)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                     _ast.ClassDef, _ast.Module)):
                    if (node.body and isinstance(node.body[0], _ast.Expr)
                            and isinstance(node.body[0].value, _ast.Constant)
                            and isinstance(node.body[0].value.value, str)):
                        node.body.pop(0)
            return _hl.sha1(
                _ast.dump(tree, annotate_fields=False).encode()
            ).hexdigest()[:16]
        except Exception:
            return "unparseable"
    if task == "T10_ts_component":
        hooks = sorted(set(__import__("re").findall(r"\buse[A-Z]\w+", body)))
        return ",".join(hooks) or "none"
    if task == "T6_qa_factual":
        import re
        pts = [k for k, rx in Q.T6_POINTS.items() if re.search(rx, body, re.I)]
        return ",".join(sorted(pts))
    return body            # prose: exact text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", default=",".join(str(t) for t in TEMPS))
    ap.add_argument("--reps", type=int, default=REPS_WARM)
    ap.add_argument("--out", default="../data/temp_quality.json")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--tasks", default=",".join(TASKS_RUN),
                    help="subset, for smoke-testing the plumbing")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing output file and re-measure")
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(",") if t] or TASKS_RUN

    temps = [float(x) for x in a.temps.split(",")]
    reps_for = {t: (REPS_T0 if t == 0.0 else a.reps) for t in temps}
    n_runs = sum(reps_for.values()) * len(tasks)
    print(f"{n_runs} generations ({len(tasks)} tasks x {len(temps)} temps, "
          f"reps={a.reps} above T=0, {REPS_T0} at T=0)")

    # Interleave temperature within each replicate, so a drifting machine
    # (thermals, page cache) cannot land systematically on one temperature.
    seq = [(t, task, rep)
           for rep in range(a.reps + 1)
           for t in temps if rep < reps_for[t]
           for task in tasks]

    L.set_settings(mtp_enabled=True, mtp_fixed_depth=a.depth)
    L.wait_settled()
    L.generate("warmup", 32, 0.0)

    tmap = {t[0]: t for t in TASKS}
    # Resume: a sweep that dies at run 31 of 182 should not start over. Key on
    # (task, temp, rep) so a partially-written output file is safe to extend.
    done_keys = set()
    outp = Path(a.out)
    if outp.exists() and not a.restart:
        prev = json.loads(outp.read_text())
        results = [r for r in prev.get("results", [])
                   if not r.get("transport_error")]
        done_keys = {(r["task"], round(r["temp"], 4), r["rep"]) for r in results}
        if done_keys:
            print(f"resuming: {len(results)} runs already on disk, "
                  f"{sum(1 for _ in done_keys)} unique cells")
    else:
        results = []
    t_start = time.time()
    done = 0
    seq = [s for s in seq if (s[1], round(s[0], 4), s[2]) not in done_keys]
    skipped = n_runs - len(seq)
    n_runs = len(seq)
    if n_runs == 0:
        print("nothing to do")
        return 0
    print(f"running {n_runs} generations (skipped {skipped} already on disk)")
    for temp, task, rep in seq:
        _, _, prompt, max_tokens = tmap[task]
        max_tokens = MAX_TOKENS_OVERRIDE.get(task, max_tokens)
        r = L.generate(prompt, max_tokens, temp)
        text = r.get("text", "")
        if r.get("transport_error"):
            # A transport fault is not a quality result. Record it and move on
            # rather than scoring an empty string as "the model failed".
            results.append({"task": task, "temp": temp, "rep": rep,
                            "valid": None, "reason": r["transport_error"],
                            "transport_error": True})
            done += 1
            continue
        ok, why = Q.check(task, text)
        deg = Q.degeneracy(text)
        got = r.get("completion_tokens")
        results.append({
            "task": task, "temp": temp, "rep": rep,
            "valid": ok, "reason": why,
            "canonical": canonical(task, text),
            # Keep the raw output. Storing only `canonical` meant a later fix to
            # a validator could not be re-scored against the old generations —
            # the T10 useState/useRef bug forced a full re-run for want of this
            # field. Verbatim text is what makes a checker's verdict auditable.
            "text": text,
            "tps": r.get("gen_tps"),
            "alpha": r.get("accept_pct"),
            "completion_tokens": got,
            "max_tokens": max_tokens,
            # hit the cap → the failure is length, not quality
            "truncated": bool(got and got >= max_tokens - 2),
            "chars": len(text),
            "fenced": Q._fenced(text),
            **deg,
        })
        done += 1
        if done and (done % 10 == 0 or done == n_runs):
            el = time.time() - t_start
            print(f"  [{done}/{n_runs}] {el/60:.1f} min elapsed, "
                  f"{el/max(done,1):.1f}s/run", flush=True)
            Path(a.out).write_text(json.dumps(
                {"results": results, "temps": temps, "reps": a.reps,
                 "depth": a.depth, "prose_quality_measured":
                     Q.prose_quality_measured},
                ensure_ascii=False, indent=2))

    # summary
    print("\n" + "=" * 88)
    print(f"{'task':<22}{'T':>5}{'valid':>8}{'excl.trunc':>11}"
          f"{'selfcons':>10}{'chars':>7}{'d2':>7}{'α':>7}")
    print("-" * 88)
    for task in tasks:
        for temp in temps:
            rows = [x for x in results
                    if x["task"] == task and abs(x["temp"] - temp) < 1e-9
                    and not x.get("transport_error")]
            if not rows:
                continue
            vr = sum(1 for x in rows if x["valid"]) / len(rows)
            untr = [x for x in rows if not x["truncated"]]
            vrt = (sum(1 for x in untr if x["valid"]) / len(untr)) if untr else None
            canon = Counter(x["canonical"] for x in rows)
            sc = canon.most_common(1)[0][1] / len(rows)
            c2 = st.mean(x["distinct2"] for x in rows)
            al = [x["alpha"] for x in rows if x["alpha"] is not None]
            ch = st.mean(x["chars"] for x in rows)
            print(f"{task:<22}{temp:>5.1f}{vr:>7.0%}"
                  f"{('—' if vrt is None else f'{vrt:.0%}'):>11}"
                  f"{sc:>10.0%}{ch:>7.0f}{c2:>7.3f}"
                  f"{(st.mean(al) if al else 0):>6.1f}%")
        print("-" * 88)
    nerr = sum(1 for x in results if x.get("transport_error"))
    ntr = sum(1 for x in results if x.get("truncated"))
    print(f"truncated runs: {ntr}/{len(results)} — length failures, not quality "
          f"failures")
    print(f"transport errors (retried then abandoned): {nerr} — excluded from "
          f"every rate above")

    print("\nprose_quality_measured =", Q.prose_quality_measured,
          "(length + simile count only, not whether it reads well)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
