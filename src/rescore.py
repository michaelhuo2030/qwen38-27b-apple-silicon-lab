#!/usr/bin/env python3
"""
rescore.py — re-run the validators over stored raw text.

Every generation is stored with its verbatim `text`, so a corrected validator
can be applied to generations that already happened. This is what makes a
validator bug a five-minute fix instead of a forty-minute re-run: the T2
OrderedDict bug was caught while the sweep was still in flight and repaired
without regenerating a single token.

This exists because of a pattern that has now bitten this project repeatedly —
a validator written narrower than the prompt it checks. When that happens the
fix is cheap *if* the raw text exists, and impossible if it does not.

Usage:
  python3 rescore.py data/temp_quality_merged.json            # dry run, prints a diff
  python3 rescore.py data/temp_quality_merged.json --write    # rewrite in place
"""
import argparse
import json
import time
from pathlib import Path

import quality_checks as Q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true",
                    help="persist the new verdicts (default: report only)")
    a = ap.parse_args()

    p = Path(a.path)
    blob = json.loads(p.read_text())
    rows = blob["results"]

    n_scored = n_missing = n_flipped = 0
    by_task: dict[str, list[int]] = {}
    detail: dict[str, int] = {}

    for r in rows:
        if r.get("transport_error"):
            continue
        text = r.get("text")
        if not text:
            n_missing += 1
            continue
        ok, why = Q.check(r["task"], text)
        # Substance only, format rules skipped — the model ignores 不要 markdown
        # 标记 on ~86% of code generations at every temperature, so the two have
        # to be reported apart or the chart reads as "code quality collapses".
        ok_nf, why_nf = Q.check(r["task"], text, ignore_format=True)
        n_scored += 1
        by_task.setdefault(r["task"], []).append(int(bool(ok_nf)))
        if bool(r.get("valid")) != bool(ok):
            n_flipped += 1
            key = f"{r['task']}: {r.get('reason', '')} -> {why}"
            detail[key] = detail.get(key, 0) + 1
        r["valid"], r["reason"] = ok, why
        r["valid_noformat"], r["reason_noformat"] = ok_nf, why_nf

    print(f"re-scored {n_scored} rows; {n_missing} had no stored text and were "
          f"left untouched")
    print(f"verdicts changed: {n_flipped}\n")
    for task, vals in sorted(by_task.items()):
        print(f"  {task:<24} {sum(vals)}/{len(vals)} now valid")
    if detail:
        print("\nverdict changes:")
        for k, v in sorted(detail.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {v:3d}x {k[:130]}")

    if a.write:
        # Per-row provenance, not a blanket refusal. An earlier version refused
        # to write anything if *any* row lacked raw text, which blocked the one
        # legitimate case it was meant to protect: a merged file where the
        # freshly-measured subset has text and the older subset does not. The
        # honest outcome is to record which rows were rescored and which still
        # carry a verdict from the older validator, then let the analysis state
        # that rather than have the write blocked and the staleness hidden.
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for r in rows:
            r["verdict_source"] = "rescored" if r.get("text") else "pre-audit-validator"
        blob["rescored_at"] = now
        p.write_text(json.dumps(blob, ensure_ascii=False, indent=1))
        stale = sorted({r["task"] for r in rows if not r.get("text")})
        print(f"\nwrote {p}")
        if stale:
            print(f"WARNING: these tasks kept a pre-audit verdict and are NOT "
                  f"comparable with the rescored ones: {stale}")
            print("         A validator change means those rows must be "
                  "re-measured, not merged. Re-run those tasks to a fresh file "
                  "rather than mixing validator versions in one chart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
