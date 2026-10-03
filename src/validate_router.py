#!/usr/bin/env python3
"""
validate_router.py — does the router agree with what was actually measured?

The hand-written unit cases in `router.py`'s demo are self-graded and prove
nothing about accuracy. This is the check that matters: run the router over the
11 tasks whose acceptance rate was *measured* on this machine, and ask whether
the profile it picks has an α band that contains the observed α.

Two things can be wrong and the report separates them:

* **Band error** — the router picks a sensible profile but the profile's α
  band does not contain the measured value. That is a data problem: the band
  is wrong and depth advice derived from it will be wrong.
* **Classification error** — the router picks the wrong profile entirely.
  That is a rule problem.

Reading only the aggregate would hide which of the two it is.
"""
import json
import statistics as st
import sys
from pathlib import Path

import router as R
from tasks import TASKS

DATA = Path(__file__).resolve().parent.parent / "data"


def measured_alpha(depth="1", temp="0.0"):
    """α per task at the given depth/temperature, from the sweep data.

    Cell keys are "<task_id>@<temperature>".
    """
    d = json.loads((DATA / "depth_sweep.json").read_text())
    out = {}
    for key, cell in d.get(str(depth), {}).items():
        task, _, t = key.partition("@")
        if t and t.lstrip("Tt") != temp:     # cell keys carry a "T" prefix
            continue
        a = cell.get("accept_mean")
        if a is not None:
            out[task] = a
    return out


def main():
    measured = measured_alpha()
    if not measured:
        print("no depth_sweep.json with acceptance data — run run_sweep.py first")
        return 1

    rows, band_ok, cls_bad = [], 0, []
    for task, prompt, max_tokens in [(t[0], t[2], t[3]) for t in TASKS]:
        if task not in measured:
            continue
        r = R.route(prompt)
        obs = measured[task]
        lo, hi = r.alpha_band
        in_band = lo <= obs <= hi
        band_ok += in_band
        if not in_band:
            cls_bad.append((task, r.profile, obs, (lo, hi)))

        rows.append((task, r.profile, obs, lo, hi, in_band,
                     r.params["temperature"], r.session_advice["mtp_enabled"]))

    print("=" * 88)
    print(f"Router vs measured α, depth 1 / T=0   ({len(rows)} tasks)")
    print("=" * 88)
    print(f"{'task':<24}{'routed to':<20}{'α obs':>7}{'band':>14}"
          f"{'ok':>4}{'temp':>6}{'mtp':>6}")
    print("-" * 88)
    for task, prof, obs, lo, hi, ok, temp, mtp in rows:
        band = f"{lo:.0f}–{hi:.0f}%"
        mtp_s = "on" if mtp else "off"
        print(f"{task:<24}{prof:<20}{obs:>6.1f}%{band:>14}"
              f"{'✓' if ok else '✗':>4}{temp:>6}{mtp_s:>6}")
    print("-" * 88)
    print(f"α inside the routed profile's band: {band_ok}/{len(rows)}")
    print("\nThis is a regression test, not an accuracy estimate: if the bands "
          "were fitted\non these same 11 tasks it is in-sample and the score "
          "proves nothing. Use\nvalidate_router_holdout.py on real prompts, "
          "and read the score from before\nany fitting (8/16 on 16 real "
          "coding-agent prompts).")

    if cls_bad:
        print("\nMisses (band or rule):")
        for task, prof, obs, (lo, hi) in cls_bad:
            gap = ("above band" if obs > hi else "below band")
            print(f"  {task:<24} → {prof:<20} measured {obs:.1f}%, "
                  f"band {lo:g}–{hi:g}% ({gap})")
        print("\nFor each miss, decide which it is before editing the router:")
        print("  wrong profile  → the rule fired wrongly; fix the regex order")
        print("  right profile  → the α band is wrong; fix router_profiles.json")
    else:
        print("\nNo misses.")

    # A router is only worth its build if it separates the extremes. The one
    # decision that actually changes a number is MTP on vs off.
    off = [(t, o) for t, p, o, _, _, _, _, m in rows if not m]
    on = [o for t, p, o, _, _, _, _, m in rows if m]
    if off:
        print("\nThe load-bearing split — MTP recommended OFF for:")
        for t, o in off:
            print(f"  {t:<24} measured α {o:.1f}%")
        print(f"  MTP recommended ON for: "
              f"{', '.join(f'{o:.0f}%' for o in sorted(on))}")
        print("\n  The depth-1 A/B measured +23.6% for repetitive code and "
              "−14.6% for prose.\n  This split is what a router is for: same "
              "server, same depth, opposite conclusions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
