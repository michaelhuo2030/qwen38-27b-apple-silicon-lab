#!/usr/bin/env python3
"""
merge_temp_quality.py — splice a re-measured subset into the main temperature file.

Why this exists: the first temperature sweep validated T2/T3/T6/T10 with
checkers that did not test what the prompts actually asked for. T10 was the
worst — it required `useState` when the prompt requires `useRef` + `useEffect`,
so every correctly-complying run was scored as a failure and the task came out
looking like the most temperature-sensitive one in the set. Those four tasks
were re-measured; T1/T7/T8 were not touched, because their checkers were
already correct and re-running them would only add noise.

The original file is left on disk untouched. It is the record of what the
broken validators claimed, and the diff between the two is the evidence for
this project's single most embarrassing lesson.

Usage:
  python3 merge_temp_quality.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "data" / "temp_quality.json"
NEW = ROOT / "data" / "temp_quality_v2.json"
OUT = ROOT / "data" / "temp_quality_merged.json"

# Checkers that were already correct; keep their original generations.
KEEP_OLD = {"T1_json_structured", "T7_math_reasoning", "T8_creative_writing"}
# Re-measured with the corrected validators.
TAKE_NEW = {"T2_code_function", "T3_code_repetitive",
            "T6_qa_factual", "T10_ts_component"}


def main() -> int:
    if not OLD.exists() or not NEW.exists():
        print(f"missing inputs: old={OLD.exists()} new={NEW.exists()}")
        return 1
    old = json.loads(OLD.read_text())
    new = json.loads(NEW.read_text())

    kept = [r for r in old["results"] if r["task"] in KEEP_OLD]
    taken = [r for r in new["results"] if r["task"] in TAKE_NEW]
    dropped = {r["task"] for r in old["results"]} - KEEP_OLD - TAKE_NEW

    if dropped:
        print(f"WARNING: unaccounted-for tasks dropped: {sorted(dropped)}")

    missing_text = [r for r in taken if not r.get("text")]
    if missing_text:
        print(f"ERROR: {len(missing_text)} new rows have no stored raw text; "
              "re-run without --restart so the field is populated")
        return 1

    merged = {
        "results": kept + taken,
        "temps": sorted({r["temp"] for r in kept + taken}),
        "reps": new.get("reps", old.get("reps")),
        "depth": new.get("depth", old.get("depth")),
        "prose_quality_measured": old.get("prose_quality_measured", False),
        "provenance": {
            "kept_from_first_sweep": sorted(KEEP_OLD),
            "remeasured_after_validator_fix": sorted(TAKE_NEW),
            "first_sweep_file": OLD.name,
            "remeasured_file": NEW.name,
        },
    }
    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1))

    by = {}
    for r in merged["results"]:
        by.setdefault(r["task"], set()).add(round(r["temp"], 3))
    print(f"wrote {OUT.name}: {len(merged['results'])} generations")
    for t in sorted(by):
        print(f"  {t:24s} {len(by[t])} temps, "
              f"{sum(1 for r in merged['results'] if r['task'] == t)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
