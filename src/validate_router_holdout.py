#!/usr/bin/env python3
"""
validate_router_holdout.py — out-of-sample check for the router.

`validate_router.py` fits and tests on the same 11 tasks, so its 11/11 is
in-sample and proves nothing about accuracy. This one uses the 16 prompts
sampled from a real coding-agent history. Those were never used to set any α
band, so this is a genuine holdout.

The prompts contain project names and personal context and are **not**
redistributed. Point this script at your own copies; it prints aggregates only
and never echoes a prompt.

    python3 validate_router_holdout.py \
        --prompts  <your real_benchmark.json> \
        --results  <your results_real.json> \
        --depth 1
"""
import argparse
import json
import statistics as st
from pathlib import Path

import router as R
import tasks


def _tokens(s: str) -> set:
    import re as _re
    return set(_re.findall(r"[a-z_]{3,}|[一-鿿]", (s or "").lower()))


def _fitting_overlap(prompts: dict, threshold: float = 0.35):
    """Holdout prompts that look like the synthetic tasks the bands were fitted on.

    Returns (holdout_id, task_id, jaccard) triples above the threshold. A high
    Jaccard between a held-out prompt and a fitted synthetic prompt means the
    "holdout" reading is in-sample no matter what the ids say.
    """
    out = []
    fit = [(t[0], _tokens(t[2] if len(t) > 2 else "")) for t in tasks.TASKS]
    for pid, p in prompts.items():
        ht = _tokens(p.get("prompt", ""))
        if not ht:
            continue
        for tid, ft in fit:
            if not ft:
                continue
            j = len(ht & ft) / len(ht | ft)
            if j >= threshold:
                out.append((pid, tid, j))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                    help="JSON list of {id, cat, lang, prompt}")
    ap.add_argument("--results", required=True,
                    help="results_real.json: {depth: {'R01@T0.0': {...}}}")
    ap.add_argument("--depth", default="1")
    ap.add_argument("--temp", default="0.0")
    a = ap.parse_args()

    prompts = {p["id"]: p for p in json.loads(Path(a.prompts).read_text())}
    results = json.loads(Path(a.results).read_text()).get(str(a.depth), {})

    # A holdout that overlaps the fitting set is not a holdout. The prose
    # warning at the bottom of this file used to be the only guard, which is
    # the "print a warning and hope a human reads it" pattern: `validate_router`
    # reported 11/11 for months of work purely because it fitted and tested on
    # the same 11 tasks, and nothing in the output said so.
    #
    # Ids alone are not enough — renaming a prompt defeats an id check — so this
    # compares content fingerprints against the synthetic task prompts.
    overlap = _fitting_overlap(prompts)
    if overlap:
        print("REFUSING: these holdout prompts overlap the synthetic tasks the "
              "bands were fitted on.")
        for pid, tid, jac in overlap:
            print(f"  {pid} ~ {tid}  (token Jaccard {jac:.2f})")
        print("\nAn in-sample score is not an accuracy estimate. Point "
              "--prompts at data that was never used to fit a band.")
        return 1

    rows = []
    for key, cell in results.items():
        rid, _, t = key.partition("@")
        if t and t.lstrip("Tt") != a.temp:
            continue
        obs = cell.get("accept_mean")
        if obs is None or rid not in prompts:
            continue
        r = R.route(prompts[rid]["prompt"])
        lo, hi = r.alpha_band
        rows.append({
            "id": rid, "cat": prompts[rid].get("cat", "?"),
            "lang": prompts[rid].get("lang", "?"),
            "profile": r.profile, "obs": obs, "lo": lo, "hi": hi,
            "in_band": lo <= obs <= hi,
            "mtp": r.session_advice["mtp_enabled"],
        })

    if not rows:
        print("no matching cells — check --depth/--temp")
        return 1

    n = len(rows)
    ok = sum(x["in_band"] for x in rows)
    print("=" * 78)
    print(f"Router holdout — {n} real prompts, depth {a.depth} / T={a.temp}")
    print("=" * 78)
    print(f"{'id':<6}{'profile':<20}{'α obs':>8}{'band':>11}{'ok':>4}{'mtp':>6}")
    print("-" * 78)
    for x in sorted(rows, key=lambda r: r["obs"]):
        band = f"{x['lo']:.0f}–{x['hi']:.0f}%"
        mark = "✓" if x["in_band"] else "✗"
        mtp = "on" if x["mtp"] else "off"
        print(f"{x['id']:<6}{x['profile']:<20}{x['obs']:>7.1f}%"
              f"{band:>14}{mark:>4}{mtp:>6}")
    print("-" * 78)
    print(f"α inside the routed band: {ok}/{n}  ({100*ok/n:.0f}%)")
    print("\nThis is only an accuracy estimate if the bands were NOT fitted on "
          "these\nprompts. If fit_bands.py --results pointed here, this number "
          "is in-sample\nand meaningless — the out-of-sample reading was 8/16 "
          "before any fitting.")

    obs_all = [x["obs"] for x in rows]
    on = [x["obs"] for x in rows if x["mtp"]]
    off = [x["obs"] for x in rows if not x["mtp"]]
    print(f"\nα range: {min(obs_all):.1f}–{max(obs_all):.1f}%  "
          f"(median {st.median(obs_all):.1f}%)")
    if off:
        print(f"router says MTP off for {len(off)} prompt(s), α = "
              f"{', '.join(f'{v:.1f}%' for v in off)}")
        print(f"router says MTP on  for {len(on)}, α = "
              f"{min(on):.1f}–{max(on):.1f}%")
        print("\nThe depth-1 A/B measured −14.6% for MTP at α≈57% and +23.6% at "
              "α≈98%.\nIf every 'off' verdict here sits below every 'on' verdict, "
              "the split is consistent\nwith an independently measured result.")
    else:
        print("\nNo prompt routed to MTP off — the real workload never dips into "
              "the sub-60% band, so depth 1 with MTP on\nis right for all of "
              "them. That matches the depth sweep: depth 1 won 14/16 here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
