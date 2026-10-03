#!/usr/bin/env python3
"""
fit_bands.py — regenerate the router's α bands from measurements, not vibes.

The bands in `router_profiles.json` started as guesses and were wrong by up to
20 pt on translation and math. Hand-editing them to make a validation pass
looks like progress and is actually overfitting: the in-sample score goes to
11/11 while the real 16-prompt holdout sat at 8/16.

So fit them, from every measurement you have, and print the sample size behind
each band. A band supported by one observation is a guess with a number
attached, and the report should say so.

Inputs (any subset; missing files are skipped):
    data/depth_sweep.json   synthetic tasks, depth 1 / T=0
    --results <path>        real prompts, --depth / --temp

Usage:
    python3 fit_bands.py --results ~/lab/results_real.json --apply
"""
import argparse
import json
from pathlib import Path

import router as R

DATA = Path(__file__).resolve().parent.parent / "data"
PROFILES_PATH = DATA / "router_profiles.json"

# A band needs observations on both sides to be a band. With fewer than this
# many, fall back to a wide default and say so.
MIN_N = 3
DEFAULT_BAND = {"code_repetitive": (70, 99), "code_function": (70, 99),
                "structured_json": (70, 99), "extraction": (70, 99),
                "math_reasoning": (60, 99), "translation": (60, 99),
                "qa_factual": (55, 85), "creative": (45, 70),
                "general": (60, 92)}
# Half-width applied when there is data but too little to trust the range.
MIN_HALF_WIDTH = 8.0


def observations(real_prompts=None, real_results=None, depth="1", temp="0.0"):
    """(profile, alpha, source) for every measured prompt."""
    obs = []
    ds = DATA / "depth_sweep.json"
    if ds.exists():
        cells = json.loads(ds.read_text()).get(str(depth), {})
        for key, cell in cells.items():
            task, _, t = key.partition("@")
            if t and t.lstrip("Tt") != temp:
                continue
            if cell.get("accept_mean") is not None:
                task_prompt = _TASK_PROMPTS.get(task)
                if task_prompt:
                    obs.append((R.route(task_prompt).profile,
                                cell["accept_mean"], f"synthetic:{task}"))

    if real_prompts and real_results and Path(real_results).exists():
        prompts = {p["id"]: p for p in json.loads(Path(real_prompts).read_text())}
        cells = json.loads(Path(real_results).read_text()).get(str(depth), {})
        for key, cell in cells.items():
            rid, _, t = key.partition("@")
            if t and t.lstrip("Tt") != temp:
                continue
            if cell.get("accept_mean") is not None and rid in prompts:
                obs.append((R.route(prompts[rid]["prompt"]).profile,
                            cell["accept_mean"], f"real:{rid}"))
    return obs


def fit(obs):
    by_profile: dict[str, list[float]] = {}
    for prof, a, _ in obs:
        by_profile.setdefault(prof, []).append(a)

    out = {}
    for name in R.PROFILES:
        vals = sorted(by_profile.get(name, []))
        n = len(vals)
        if n == 0:
            lo, hi = DEFAULT_BAND[name]
            out[name] = (lo, hi, None, 0, "no data — default band")
        elif n < MIN_N:
            lo, hi = DEFAULT_BAND[name]
            out[name] = (lo, hi, vals[-1], n,
                         f"n={n} < {MIN_N} — using the default band, not the data")
        else:
            lo, hi = min(vals), max(vals)
            mid = (lo + hi) / 2
            if hi - lo < MIN_HALF_WIDTH:          # too tight to be a band
                lo, hi = mid - MIN_HALF_WIDTH / 2, mid + MIN_HALF_WIDTH / 2
                note = f"n={n}, spread <{MIN_HALF_WIDTH:g}pt — widened"
            else:
                note = f"n={n}"
            out[name] = (lo, hi, vals[-1], n, note)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-prompts")
    ap.add_argument("--results")
    ap.add_argument("--depth", default="1")
    ap.add_argument("--temp", default="0.0")
    ap.add_argument("--apply", action="store_true",
                    help="write the fitted bands back to router_profiles.json")
    a = ap.parse_args()

    obs = observations(a.real_prompts, a.results, a.depth, a.temp)
    if not obs:
        print("no measurements found — run run_sweep.py first")
        return 1

    print(f"fitting {len(obs)} observations "
          f"(depth {a.depth}, T={a.temp})\n")
    print(f"{'profile':<20}{'lo':>6}{'hi':>6}{'n':>4}  note")
    print("-" * 72)
    fitted = fit(obs)
    for name, (lo, hi, _, n, note) in sorted(fitted.items(),
                                              key=lambda kv: -kv[1][3]):
        print(f"{name:<20}{lo:>6.0f}{hi:>6.0f}{n:>4}  {note}")

    if not a.apply:
        print("\ndry run — pass --apply to write these into "
              "data/router_profiles.json")
        return 0

    table = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    for name, (lo, hi, last, n, note) in fitted.items():
        p = table["profiles"][name]
        p["alpha_lo"], p["alpha_hi"] = lo, hi
        p["alpha_n"] = n
        p["alpha_note"] = note
        if last is not None and n >= MIN_N:
            p["measured_alpha"] = last
    table["_comment"] = (
        "Router profile table. alpha bands are FITTED by src/fit_bands.py from "
        "measured acceptance rates; alpha_n says how many observations back "
        "each one, and alpha_note says when there were too few to trust. A "
        "band with n=1 is a guess. temperature/top_p are conventional "
        "defaults — this repo measured temperature's effect on efficiency and "
        "its degeneracy boundary, not a quality curve."
    )
    PROFILES_PATH.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"\nwrote {PROFILES_PATH}")
    return 0


_TASK_PROMPTS = {}


def _load_task_prompts():
    try:
        from tasks import TASKS
        _TASK_PROMPTS.update({t[0]: t[2] for t in TASKS})
    except Exception:
        pass


_load_task_prompts()

if __name__ == "__main__":
    raise SystemExit(main())
