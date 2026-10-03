#!/usr/bin/env python3
"""
analyze.py — turn the raw sweep JSON into the tables and figures used in the
README and blog post.

    python3 analyze.py --data-dir ../data --figure-dir ../figures
"""
import argparse
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTHS = ["1", "2", "3", "4", "6", "8"]
TEMPS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6]
# A cell needs at least this many valid repeats to be reportable.
#
# The per-run `valid` flag (accept_den >= 25) stops one 3-token response from
# poisoning a mean, but it does not stop a cell whose only surviving sample is
# itself tiny. Rust impl at T=1.2 produced runs=[52.94] with den=51 -- one
# sample, and it read as a 39-point collapse. High temperature makes the model
# terminate early on that prompt, so there is simply nothing to measure; that
# is "insufficient", not "collapsed", and the table must say which.
MIN_VALID_REPS = 2
SHORT = {
    "T1_json_structured": "JSON output", "T2_code_function": "Python function",
    "T3_code_repetitive": "Repetitive code", "T4_translation": "Translation",
    "T5_extraction": "Info extraction", "T6_qa_factual": "Factual QA",
    "T7_math_reasoning": "Math reasoning", "T8_creative_writing": "Creative prose",
    "T9_rust_impl": "Rust impl", "T10_ts_component": "TS component",
    "T11_doc_rewrite": "Doc rewrite",
}


def load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def pooled_alpha(e):
    """Token-weighted alpha across the valid repeats of one cell.

    A simple mean of per-run ratios weights a 39-draft-token run the same as a
    574-draft-token one. That is how a single tiny sample produced a
    "-25 pt temperature effect" for Rust impl at T=1.6 that was really one
    noisy run dragging a mean. Pooling by denominator is the same principle
    already applied inside generate(), applied once more at cell level.
    """
    if not e:
        return None
    nums, dens = e.get("accept_all") or [], e.get("accept_den_all") or []
    num = den = 0.0
    for a, d in zip(nums, dens):
        if a is None or not d:
            continue
        num += a / 100.0 * d
        den += d
    return round(100.0 * num / den, 2) if den else None


def cell(d, depth, task, temp=0.0):
    e = (d.get(depth) or {}).get(f"{task}@T{temp}")
    if not e:
        return None
    # Suppress cells that survived the per-run filter but have too few
    # repeats left to carry a mean.
    if e.get("n_valid", 99) < MIN_VALID_REPS:
        return None
    return e


def a_of(d, depth, task, temp=0.0):
    """Pooled alpha for a cell, or None if the cell is under-powered."""
    return pooled_alpha(cell(d, depth, task, temp))


def thin(d, depth, task, temp=0.0):
    """Same lookup but returns the cell even if under-powered, for auditing."""
    return (d.get(depth) or {}).get(f"{task}@T{temp}")


def g(e, k):
    return None if not e else e.get(k)


def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def f(v, n=1, suf=""):
    return "—" if v is None else f"{v:.{n}f}{suf}"


def sign_test(wins, n):
    """Two-sided exact binomial p for a sign test."""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def quality_identity_tables(qi, out):
    if not qi:
        return
    diffs = qi.get("diffs", {})
    lines = ["\n### Output identity, MTP on vs off (T=0, depth 1)\n",
             "| Task | α while MTP on | chars on / off | verdict | first divergence |",
             "|---|---|---|---|---|"]
    for tk in qi.get("tasks", []):
        a, b = qi["on"].get(tk, {}), qi.get("off", {}).get(tk, {})
        d = diffs.get(tk)
        if not a or not d:
            continue
        if d["identical"]:
            verdict, where = "identical", "—"
        else:
            verdict = "differs"
            where = f"char {d['first_diff_index']}, {d['n_chars_differing']} chars after"
        lines.append(f"| {SHORT.get(tk, tk)} | {f(a.get('alpha'))}% | "
                     f"{d['len_a']} / {d['len_b']} | {verdict} | {where} |")
    lines.append("")
    n = qi.get("n_identical", 0)
    tot = len(qi.get("tasks", []))
    lines.append(f"{n} of {tot} outputs byte-identical. The MTP-off arm emitted "
                 "no MTP statistics and no acceptance figures, which is the "
                 "check that the toggle actually took.")
    lines.append("")
    lines.append(f"> {qi.get('note', '')}")
    out.append("\n".join(lines))


def ab_tables(ab, out):
    if not ab:
        return
    noise = ab.get("noise_pct", 4.5)
    lines = ["\n### MTP on vs off at depth 1 (T=0)\n",
             f"Paired, {ab.get('reps')} reps per arm. Noise band ±{noise}%. "
             "**Compound toggle** — see the note below the table.\n",
             "| Task | MTP ON tok/s | MTP OFF tok/s | Δ | verdict |",
             "|---|---|---|---|---|"]
    for tk, d in zip(ab.get("tasks", []), ab.get("deltas_pct", [])):
        a, b = ab["on"].get(tk, {}), ab.get("off", {}).get(tk, {})
        if not a:
            continue
        v = ("MTP slower" if d < -noise else
             "MTP faster" if d > noise else "within noise")
        lines.append(f"| {SHORT.get(tk, tk)} | {f(a.get('tps_mean'), 2)} | "
                     f"{f(b.get('tps_mean'), 2)} | {d:+.1f}% | {v} |")
    lines.append("")
    lines.append(f"mean {ab.get('mean_delta_pct'):+.1f}%, "
                 f"median {ab.get('median_delta_pct'):+.1f}%; "
                 f"{ab.get('n_faster')} faster / {ab.get('n_slower')} slower "
                 f"beyond the noise band. "
                 f"Toggle verified: {ab.get('toggle_verified')}.")
    lines.append("")
    lines.append(f"> **Confound.** {ab.get('confound', '')}")
    out.append("\n".join(lines))


# ---------------------------------------------------------------- tables
def depth_tables(ds, out):
    tasks = [t for t in SHORT if any(cell(ds, d, t) for d in DEPTHS)]
    lines = []

    lines.append("### Throughput vs depth (T=0)\n")
    lines.append("| Task | α@d1 | " + " | ".join(f"d{d}" for d in DEPTHS) + " | best | vs d1 |")
    lines.append("|---|---" + "|---" * (len(DEPTHS) + 2) + "|")
    opt, gains, series = {}, {}, {}
    for t in tasks:
        vals = {d: g(cell(ds, d, t), "tps_mean") for d in DEPTHS}
        ok = {d: v for d, v in vals.items() if v is not None}
        if not ok:
            continue
        best = max(ok, key=ok.get)
        gain = (vals[best] / vals["1"] - 1) * 100 if vals.get("1") and vals.get(best) else None
        opt[t], gains[t] = best, gain
        series[t] = vals
        lines.append(f"| {SHORT[t]} | {f(a_of(ds,'1',t))}% | "
                     + " | ".join(f(vals[d] or 0) for d in DEPTHS)
                     + f" | **d{best}** | {f(gain)}% |")
    out.append("\n".join(lines))

    lines = ["\n### Cost decomposition, mean over tasks (ms per decode cycle)\n",
             "| depth | backbone | MTP | cache | sample | total | tok/cycle | ms/token |",
             "|---|---|---|---|---|---|---|---|"]
    cost = {}
    for d in DEPTHS:
        bb = mean([g(cell(ds, d, t), "backbone_ms_mean") for t in tasks])
        mt = mean([g(cell(ds, d, t), "mtp_ms_mean") for t in tasks])
        ca = mean([g(cell(ds, d, t), "cache_ms_mean") for t in tasks])
        sa = mean([g(cell(ds, d, t), "sample_ms_mean") for t in tasks])
        tc = mean([g(cell(ds, d, t), "tok_per_cycle_mean") for t in tasks])
        tot = sum(x for x in (bb, mt, ca, sa) if x)
        cost[d] = (tot / tc) if tc else None
        lines.append(f"| {d} | {f(bb,2)} | {f(mt,3)} | {f(ca,3)} | {f(sa,3)} | "
                     f"{f(tot,2)} | {f(tc,2)} | {f(cost[d],2)} |")
    out.append("\n".join(lines))

    lines = ["\n### α vs depth (is acceptance depth-independent?)\n",
             "| Task | " + " | ".join(f"d{d}" for d in DEPTHS) + " | spread |",
             "|---|" + "---|" * (len(DEPTHS) + 1)]
    for t in tasks:
        a = [a_of(ds, d, t) for d in DEPTHS]
        v = [x for x in a if x is not None]
        if not v:
            continue
        lines.append(f"| {SHORT[t]} | " + " | ".join(f(x) if x else "—" for x in a)
                     + f" | {max(v)-min(v):.1f}pt |")
    out.append("\n".join(lines))

    lines = ["\n### Summary\n"]
    if opt:
        c = Counter(opt.values())
        lines.append(f"- Best depth per task: " +
                     ", ".join(f"d{k}={v}/{len(opt)}" for k, v in sorted(c.items())))
        pos = [g for g in gains.values() if g and g > 0]
        if pos:
            lines.append(f"- {len(pos)}/{len(gains)} tasks beat depth 1; "
                         f"mean gain {f(mean(pos))}%, max {f(max(pos))}%")
    best_cost = min(cost, key=lambda d: cost[d]) if cost and cost.get("1") else None
    if best_cost:
        lines.append(f"- Lowest cost per token at **depth {best_cost}** "
                     f"({f(cost[best_cost],2)} ms vs {f(cost['1'],2)} ms at depth 1)")
    out.append("\n".join(lines))
    return series, cost


def noise_floor(ts, out):
    """Pooled within-cell sigma and the resulting decision threshold."""
    groups = []
    for dep in ts.values():
        for e in dep.values():
            a = e.get("accept_all") or []
            if len(a) >= 2:
                groups.append(a)
    if not groups:
        return
    sds = [st.stdev(g) for g in groups]
    pooled = math.sqrt(sum(s * s for s in sds) / len(sds))
    half = 1.96 * pooled * math.sqrt(2 / 3)
    out.append("\n### Noise floor\n")
    out.append(f"- cells with >= 2 valid repeats: **{len(groups)}**\n"
               f"- median within-cell sigma: **{st.median(sds):.2f} pt**, "
               f"pooled sigma: **{pooled:.2f} pt**\n"
               f"- with n=3, the 95% confidence half-width is **±{half:.1f} pt**\n"
               f"- therefore differences below ~{half:.0f} pt are not findings")
    return pooled, half


def temp_tables(ts, out):
    noise_floor(ts, out)
    for depth in ("1", "2"):
        tasks = [t for t in SHORT if thin(ts, depth, t, 0.0)]
        if not tasks:
            continue
        lines = [f"\n### α vs temperature at depth {depth}\n",
                 "`ins` = under-powered cell (fewer than "
                 f"{MIN_VALID_REPS} valid repeats; the model terminates too "
                 "early at that setting to measure).\n",
                 "| Task | " + " | ".join(f"T{t}" for t in TEMPS) + " | T1.6−T0 |",
                 "|---|" + "---|" * (len(TEMPS) + 1)]
        for t in tasks:
            raw, a = [], []
            for tp in TEMPS:
                e = thin(ts, depth, t, tp)
                c = cell(ts, depth, t, tp)
                raw.append(e)
                a.append(pooled_alpha(c))
            v = [x for x in a if x is not None]
            if len(v) < 2:
                continue
            cells = []
            for x in a:
                cells.append(f"{x}" if x is not None else "ins")
            d16 = (a[5] - a[0]) if (a[0] is not None and a[5] is not None) else None
            lines.append(f"| {SHORT[t]} | " + " | ".join(cells)
                         + f" | {f(d16)}pt |")
        out.append("\n".join(lines))


def workload_tables(wl, out):
    if not wl:
        return
    lines = ["\n### Real-workload benchmark\n",
             "16 prompts taken from a real coding-agent history (aggregate only;\n"
             "raw prompts are not redistributed).\n",
             "| depth | α mean | α median | α range | tok/s mean |",
             "|---|---|---|---|---|"]
    for d in ("1", "2", "3"):
        a = [pooled_alpha(v) or v["accept_mean"] for v in wl.get("by_depth", {}).get(d, {}).values()]
        t = [v["tps_mean"] for v in wl.get("by_depth", {}).get(d, {}).values() if v.get("tps_mean")]
        if not a:
            continue
        lines.append(f"| {d} | {f(mean(a))}% | {f(st.median(a))}% | "
                     f"{f(min(a))}%–{f(max(a))}% | {f(mean(t))} |")
    out.append("\n".join(lines))

    if "sign_test" in wl:
        s = wl["sign_test"]
        lines = ["\n### Is depth 1 really better on this workload?\n",
                 f"- α: depth 1 higher on **{s['alpha_wins_d1']}/{s['n']}** prompts, "
                 f"exact sign test **p = {s['alpha_p']:.4f}**",
                 f"- throughput: depth 1 higher on {s['tps_wins_d1']}/{s['n']}",
                 f"- mean α difference (d1 − d2): **{s['alpha_delta_mean']:+.1f} pt** "
                 f"(median {s['alpha_delta_median']:+.1f}, range "
                 f"{s['alpha_delta_range'][0]:+.1f} to {s['alpha_delta_range'][1]:+.1f})"]
        out.append("\n".join(lines))


# ---------------------------------------------------------------- figures
def figures(ds, ts, wl, figdir, ab=None):
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # 1. cost per token vs depth
    if ds:
        xs, ys = [], []
        for d in DEPTHS:
            tc = mean([g(cell(ds, d, t), "tok_per_cycle_mean") for t in SHORT])
            tot = sum(x for x in (mean([g(cell(ds, d, t), "backbone_ms_mean") for t in SHORT]),
                                  mean([g(cell(ds, d, t), "mtp_ms_mean") for t in SHORT]),
                                  mean([g(cell(ds, d, t), "cache_ms_mean") for t in SHORT]))
                      if x)
            if tc and tot:
                xs.append(int(d))
                ys.append(tot / tc)
        if xs:
            # Measurement noise, so the chart does not imply that a 1.6%
            # difference between depth 1 and 2 is a result.
            allv = [v for dep in ds.values() for e in dep.values()
                    for v in (e.get("tps_all") or []) if v]
            pooled = 0.0
            groups = [e["tps_all"] for dep in ds.values() for e in dep.values()
                      if len(e.get("tps_all") or []) >= 2]
            if groups:
                pooled = math.sqrt(sum(st.stdev(g) ** 2 for g in groups) / len(groups))
            band = 1.96 * pooled * math.sqrt(2 / 3) / st.mean(allv) * 100 if allv else 4.5

            fig, ax = plt.subplots(figsize=(7, 4.4))
            ax.plot(xs, ys, "o-", lw=2, color="#c0392b", zorder=3)
            lo, hi = min(ys), max(ys)
            ax.axhspan(lo * (1 - band / 100), lo * (1 + band / 100),
                       color="#888", alpha=.22, zorder=1,
                       label=f"measurement noise (±{band:.1f}%)")
            k = ys.index(lo)
            ax.annotate(f"shallow minimum at depth {xs[k]}\n"
                        f"{lo:.2f} ms/token — but depth 1 is within noise",
                        (xs[k], lo), textcoords="offset points", xytext=(12, 26),
                        fontsize=9, color="#c0392b",
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
            ax.set_xlabel("MTP depth")
            ax.set_ylabel("ms per generated token")
            ax.set_title("Cost per token when the PLE table is offloaded to SSD")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=.3)
            ax.set_ylim(lo * (1 - band / 100) * 0.96, hi * 1.02)
            fig.tight_layout()
            fig.savefig(figdir / "cost_per_token_vs_depth.png", dpi=160)
            plt.close(fig)

    # 2. alpha vs temperature
    if ts:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, depth in zip(axes, ("1", "2")):
            for t in ["T3_code_repetitive", "T1_json_structured", "T6_qa_factual",
                      "T8_creative_writing"]:
                xs = [tp for tp in TEMPS if a_of(ts, depth, t, tp) is not None]
                ys = [a_of(ts, depth, t, tp) for tp in xs]
                if xs:
                    ax.plot(xs, ys, "o-", lw=1.6, ms=4, label=SHORT[t])
            ax.set_title(f"depth {depth}")
            ax.set_xlabel("temperature")
            ax.grid(alpha=.3)
        axes[0].set_ylabel("accept rate α (%)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Acceptance rate is nearly flat in temperature (top_p = 0.95)")
        fig.tight_layout()
        fig.savefig(figdir / "alpha_vs_temperature.png", dpi=160)
        plt.close(fig)

    # 3. alpha vs depth, per task
    if ds:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for t in ["T3_code_repetitive", "T1_json_structured", "T6_qa_factual",
                  "T8_creative_writing", "T9_rust_impl"]:
            xs = [int(d) for d in DEPTHS if a_of(ds, d, t) is not None]
            ys = [a_of(ds, d, t) for d in DEPTHS if a_of(ds, d, t) is not None]
            if xs:
                ax.plot(xs, ys, "o-", lw=1.5, ms=4, label=f"{SHORT[t]} (α@1={ys[0]:.0f}%)")
        ax.set_xlabel("MTP depth")
        ax.set_ylabel("accept rate α (%)")
        ax.set_title("Acceptance decays slowly with depth")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(figdir / "alpha_vs_depth.png", dpi=160)
        plt.close(fig)

    # 4. real workload
    if wl and wl.get("by_depth"):
        fig, ax = plt.subplots(figsize=(6.2, 4))
        depths = sorted(wl["by_depth"])
        data = [[wl["by_depth"][d][i]["accept_mean"] for i in sorted(wl["by_depth"][d])]  # already pooled upstream
                for d in depths]
        ax.boxplot(data, tick_labels=depths, showmeans=True)   # matplotlib>=3.9 renamed labels->tick_labels
        ax.set_xlabel("MTP depth")
        ax.set_ylabel("accept rate α (%)")
        ax.set_title("Real workload: depth 1 wins on 14/16 prompts")
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(figdir / "real_workload_alpha_by_depth.png", dpi=160)
        plt.close(fig)


    # 4. MTP on vs off at depth 1: the delta is monotone in acceptance
    if ab:
        pairs = [(ab["on"][t]["alpha_mean"], d, t)
                 for t, d in zip(ab.get("tasks", []), ab.get("deltas_pct", []))
                 if ab["on"].get(t, {}).get("alpha_mean") is not None]
        if pairs:
            pairs.sort()
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            band = ab.get("noise_pct", 4.5)
            ax.axhspan(-band, band, color="#888", alpha=.22, zorder=1,
                       label=f"measurement noise (±{band}%)")
            ax.plot(xs, ys, "o-", lw=2, color="#27ae60", zorder=3)
            for a_, d_, t_ in pairs:
                ax.annotate(f"{SHORT.get(t_, t_)}\n{d_:+.1f}%", (a_, d_),
                            textcoords="offset points", xytext=(0, 12),
                            ha="center", fontsize=8, color="#1e8449")
            ax.axvline(60, ls="--", lw=1, color="#c0392b", alpha=.7)
            ax.annotate("below ≈60% α, switch MTP off\nentirely — it beats any depth",
                        (61, 6), textcoords="offset points", xytext=(0, 0),
                        fontsize=8, color="#c0392b", va="center")
            ax.set_xlabel("acceptance rate α while MTP is on (%)")
            ax.set_ylabel("throughput change, MTP on vs off (%)")
            ax.set_title("At depth 1, MTP pays only when the text is predictable")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(alpha=.3)
            ax.set_ylim(min(ys) - 10, max(ys) + 12)
            fig.tight_layout()
            fig.savefig(figdir / "mtp_on_off_vs_alpha.png", dpi=160)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--figure-dir", default="../figures")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "RESULTS.md"))
    a = ap.parse_args()

    D = Path(a.data_dir)
    ds = load(D / "depth_sweep.json", {})
    ts = load(D / "temp_sweep.json", {})
    sm = load(D / "sampling.json", {})
    cx = load(D / "context.json", {})
    wl = load(D / "workload_aggregate.json", {})
    qi = load(D / "quality_identity.json", {})
    ab = load(D / "ab_mtp_on_off.json", {})

    out = ["# Measurement tables\n",
           "Generated by `analyze.py`. Do not edit by hand.\n"]
    depth_tables(ds, out)
    temp_tables(ts, out)
    workload_tables(wl, out)
    quality_identity_tables(qi, out)
    ab_tables(ab, out)

    if sm:
        lines = ["\n### Sampling control (depth 2)\n",
                 "| Config | T1 JSON | T6 QA | T8 Prose | mean | note |",
                 "|---|---|---|---|---|---|"]
        notes = {
            "T0.0_p0.95_k0_m0.0": "baseline",
            "T1.2_p0.95_k0_m0.0": "temperature up, nucleus kept",
            "T2.0_p1.0_k0_m0.0": "**degenerate output — see README**",
            "T2.0_p1.0_k1_m0.0": "top_k=1 overrides temperature → greedy",
        }
        keys = ["T1_json_structured", "T6_qa_factual", "T8_creative_writing"]
        for k, v in sm.items():
            a_ = [v[t]["accept"] for t in keys if v.get(t, {}).get("accept") is not None]
            if not a_:
                continue
            lines.append(f"| {k} | " + " | ".join(
                f"{v[t]['accept']}%" if t in v else "—" for t in keys)
                + f" | {f(mean(a_))}% | {notes.get(k,'')} |")
        out.append("\n".join(lines))

    if cx:
        lines = ["\n### Context length (depth 2, T=0)\n",
                 "| leading chars | prompt tokens | α | tok/cycle | gen tok/s |",
                 "|---|---|---|---|---|"]
        for k in sorted(cx, key=lambda x: int(x)):
            v = cx[k]
            lines.append(f"| {v.get('ctx_chars')} | {f(v.get('prompt_tokens'),0)} | "
                         f"{f(v.get('accept'))}% | {f(v.get('tok_per_cycle'),2)} | "
                         f"{f(v.get('gen_tps'))} |")
        out.append("\n".join(lines))

    Path(a.out).write_text("\n\n".join(out) + "\n")
    print(f"tables -> {a.out}")
    figures(ds, ts, wl, a.figure_dir, ab)
    print(f"figures -> {a.figure_dir}")


if __name__ == "__main__":
    main()
