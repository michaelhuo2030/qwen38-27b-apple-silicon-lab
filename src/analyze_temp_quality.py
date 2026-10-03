#!/usr/bin/env python3
"""
analyze_temp_quality.py — turn data/temp_quality.json into the answer the
router's temperature values were missing.

Reads the objective-validator sweep and answers three things:

  1. Where does the output start failing as temperature rises, per task?
     (This is the quality curve. The efficiency curve already existed and said
     temperature barely moves acceptance — small and task-specific.)
  2. Is that degradation a *quality* effect or a *length* effect? Every run
     records whether it hit the token cap, because a truncation that grows
     with temperature looks exactly like "high temperature hurts" while
     measuring verbosity.
  3. How consistent is the model at each temperature — the self-consistency
     share of the modal answer over 5 samples.

Writes `RESULTS_temp_quality.md` and `figures/valid_rate_vs_temperature.png`.
"""
import json
import sys
from pathlib import Path
import statistics as st
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "temp_quality.json"
FIG = ROOT / "figures"
SHORT = {
    "T1_json_structured": "JSON (12 objects, 6 fields)",
    "T2_code_function": "Python LRUCache",
    "T3_code_repetitive": "Python Shape base",
    "T6_qa_factual": "TCP explainer",
    "T7_math_reasoning": "Discount arithmetic",
    "T10_ts_component": "React component",
    "T8_creative_writing": "Prose (length+simile only)",
}


def cell(rows, task, temp):
    return [r for r in rows
            if r["task"] == task and abs(r["temp"] - temp) < 1e-9
            and not r.get("transport_error")]


def stats(rows):
    if not rows:
        return None
    vr = sum(1 for r in rows if r["valid"]) / len(rows)
    # Substance only. The model wraps code in ``` fences on ~86% of generations
    # at every temperature despite 不要 markdown 标记, so `valid` alone reads as
    # "code quality collapses" when the code is in fact fine and the wrapper is
    # not. Fall back to `valid` when the rescore has not run.
    nf = [r["valid_noformat"] for r in rows if "valid_noformat" in r]
    vrf = (sum(1 for x in nf if x) / len(nf)) if nf else None
    # Degenerate collapse, reported apart from every other failure mode.
    #
    # Note what does *not* detect this: distinct-2. An 18-token word-salad
    # sample scores distinct-2 = 1.000, because every bigram in a text that
    # short is unique. The metric is length-dependent, so it is blind to
    # precisely the case where an output has collapsed to nothing. Length is
    # the reliable signal, and the floor is calibrated from each task's own
    # observed median rather than picked.
    toks = [r.get("completion_tokens") or 0 for r in rows]
    med = st.median(toks) if toks else 0
    degen = (sum(1 for t in toks if med and t < 0.25 * med) / len(toks)) if med else 0.0
    untr = [r for r in rows if not r.get("truncated")]
    vrt = (sum(1 for r in untr if r["valid"]) / len(untr)) if untr else None
    # Self-consistency over 1 sample is trivially 100% and over 2 it is a coin
    # flip. Report it only where it means something.
    sc = (Counter(r["canonical"] for r in rows).most_common(1)[0][1] / len(rows)
          if len(rows) >= 3 else None)
    al = [r for r in rows if r.get("alpha") is not None]
    # Token-weighted α, matching the estimator used for the depth sweep.
    # A plain mean over requests weights a 200-token JSON and a 900-token
    # React component equally, which is not what "α over this cell" means —
    # α is defined over accepted speculative positions, i.e. tokens. Also
    # refuses to report a cell with too few samples to mean anything.
    aw = [(r["alpha"], r.get("completion_tokens") or 0) for r in al]
    tot = sum(w for _, w in aw)
    alpha = (sum(a * w for a, w in aw) / tot) if (tot and len(al) >= 2) else None
    return {
        "n": len(rows), "valid": vr, "valid_noformat": vrf, "degenerate": degen,
        "valid_excl_trunc": vrt, "self_cons": sc,
        "chars": st.mean(r["chars"] for r in rows),
        "d2": st.mean(r["distinct2"] for r in rows),
        "trunc": sum(1 for r in rows if r.get("truncated")) / len(rows),
        "alpha": alpha,
    }


def main():
    # Accept an override so a re-measured subset can be merged and analysed
    # without overwriting the original. The first temp_quality.json is kept as
    # the record of the run whose validators were wrong; overwriting it would
    # destroy the evidence for why the numbers changed.
    global DATA
    if len(sys.argv) > 1:
        DATA = Path(sys.argv[1])
    if not DATA.exists():
        print(f"missing {DATA}")
        return 1
    blob = json.loads(DATA.read_text())
    rows = [r for r in blob["results"] if not r.get("transport_error")]
    temps = sorted({r["temp"] for r in rows})
    tasks = [t for t in SHORT if any(r["task"] == t for r in rows)]

    n_err = sum(1 for r in blob["results"] if r.get("transport_error"))
    out = ["# Temperature vs quality\n",
           "Generated by `analyze_temp_quality.py`. Do not edit by hand.\n",
           f"{len(rows)} scored generations, {n_err} transport errors excluded.\n",
           "`valid` = passes an objective validator built from the prompt's own "
           "stated requirements.\n`valid*` = the same, over runs that did not hit "
           "the token cap.\nProse is scored on length and simile count only — "
           "that is not a quality judgement.\n"]

    # ---- per-task table
    out.append("\n## Per task\n")
    for task in tasks:
        out.append(f"\n### {SHORT[task]}\n")
        out.append("| T | n | valid | valid (no fmt) | valid* | degenerate "
                   "| self-consistency | mean chars | distinct-2 | truncated | α |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in temps:
            s = stats(cell(rows, task, t))
            if not s:
                continue
            v = "—" if s["valid_excl_trunc"] is None else f"{s['valid_excl_trunc']:.0%}"
            al = "—" if s["alpha"] is None else f"{s['alpha']:.1f}%"
            sc = "—" if s["self_cons"] is None else f"{s['self_cons']:.0%}"
            note = "  (n<3, self-consistency not meaningful)" if s["n"] < 3 else ""
            vnf = ("—" if s["valid_noformat"] is None
                   else f"{s['valid_noformat']:.0%}")
            dg = ("—" if not s["degenerate"]
                  else f"{s['degenerate']:.0%}")
            out.append(f"| {t:.1f} | {s['n']} | {s['valid']:.0%} | {vnf} | {v} | {dg} | {sc} "
                       f"| {s['chars']:.0f} | {s['d2']:.3f} "
                       f"| {s['trunc']:.0%} | {al} |{note}")

    # ---- aggregate
    out.append("\n## Aggregate across tasks\n")
    out.append("| T | mean valid | mean valid* | mean self-consistency "
               "| mean truncated | mean α |")
    out.append("|---|---|---|---|---|---|")
    agg = {}
    for t in temps:
        ss = [s for s in (stats(cell(rows, tk, t)) for tk in tasks) if s]
        if not ss:
            continue
        vt = [s["valid_excl_trunc"] for s in ss if s["valid_excl_trunc"] is not None]
        al = [s["alpha"] for s in ss if s["alpha"] is not None]
        agg[t] = st.mean(vt) if vt else None
        scs = [x["self_cons"] for x in ss if x["self_cons"] is not None]
        fmt = lambda v: "—" if v is None else f"{v:.0%}"
        out.append(f"| {t:.1f} | {st.mean(s['valid'] for s in ss):.0%} "
                   f"| {fmt(st.mean(vt) if vt else None)} "
                   f"| {fmt(st.mean(scs) if scs else None)} "
                   f"| {st.mean(s['trunc'] for s in ss):.0%} "
                   f"| {(st.mean(al) if al else 0):.1f}% |")

    # ---- where it breaks
    out.append("\n## Where each task starts to fail\n")
    out.append("First temperature at which valid* drops below 0.8, and the "
               "highest temperature\nthat still holds at 1.00.\n")
    out.append("| task | last T at 100% | first T below 80% |")
    out.append("|---|---|---|")
    for task in tasks:
        last_ok, first_bad = None, None
        for t in temps:
            s = stats(cell(rows, task, t))
            v = s["valid_excl_trunc"] if s and s["valid_excl_trunc"] is not None \
                else (s["valid"] if s else None)
            if v is None:
                continue
            if v >= 0.999:
                last_ok = t
            elif v < 0.8 and first_bad is None:
                first_bad = t
        out.append(f"| {SHORT[task]} | {last_ok if last_ok is not None else '—'} "
                   f"| {first_bad if first_bad is not None else 'never'} |")

    (ROOT / "RESULTS_temp_quality.md").write_text("\n".join(out) + "\n")
    print(f"tables -> {ROOT/'RESULTS_temp_quality.md'}")

    # ---- figure: three panels on one x. The whole point is the contrast
    # between the flat α line and the two that fall.
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for ax, key, title, ylabel in (
            (axes[0], "valid_excl_trunc", "Objective validity",
             "valid (excluding truncated runs)"),
            (axes[1], "self_cons", "Self-consistency",
             "modal answer share over 5 samples"),
            (axes[2], "alpha", "Acceptance rate α (for comparison)",
             "acceptance rate (%)")):
        # α is stored as a percentage already; the other two are fractions.
        for task in tasks:
            xs, ys = [], []
            for t in temps:
                st_ = stats(cell(rows, task, t))
                if st_ and st_[key] is not None:
                    xs.append(t)
                    ys.append(st_[key] if key == "alpha" else st_[key] * 100)
            if xs:
                # Prose "validity" only checks the length and simile count the
                # prompt asked for. Dashed, so it is not read as a quality line.
                style = (dict(ls="--", lw=1.3, ms=3)
                         if task == "T8_creative_writing"
                         else dict(lw=1.7, ms=4))
                ax.plot(xs, ys, "o-", label=SHORT[task], **style)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("temperature")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
    axes[0].set_ylim(-4, 104)
    axes[1].set_ylim(-4, 104)
    axes[2].set_ylim(40, 104)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Temperature barely moves acceptance, and moves quality a lot "
                 "— dashed prose line is length+simile only, not a judgement",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "valid_rate_vs_temperature.png", dpi=160)
    plt.close(fig)
    print(f"figure -> {FIG/'valid_rate_vs_temperature.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
