#!/usr/bin/env python3
"""
A/B: on this machine, is MTP at depth 1 a speedup or a tax?

Read this docstring as a record of one wrong result and the harness bug that
produced it, because the bug is the transferable part.

The first single-shot measurement said "turning MTP off is faster" (Rust 52.3
vs 44.8 tok/s), and the reasoning behind it sounded airtight:

    at depth 1 the head drafts one character, the backbone verifies one.
    draft accepted  -> 1 character out
    draft rejected  -> 1 character out
    either way one character per cycle, so MTP can never save anything at
    depth 1, and the extra head forward is pure overhead.

A 3-replicate A/B was run to settle it. It reported MTP on is *faster* on
every task, mean +6.9%, never significantly slower -- and that number was
written up as a finding. It was an artifact. The script's `set_settings()`
logged in with `urllib`, which keeps no cookie, so the admin PUT returned
`401 {"detail":"Admin authentication required"}`; the surrounding
`except Exception: pass` swallowed it. **Both arms ran MTP depth=1.** The
+6.9% was the gap between the first arm and the second arm of an identical
configuration, read as an effect.

The tell was available the whole time and nobody looked: the "MTP OFF" arm
still emitted one MTP statistics line per request, and its acceptance rates
were identical to the ON arm's to two decimals. A disabled speculative path
cannot report 90.91% acceptance.

So `set_settings` lives in `omlx_client.py` now, carries cookies, refuses to
swallow exceptions, and asserts the readback. This script additionally asserts
the OFF arm emitted zero MTP statistics lines -- a comparison where the toggle
did not take is refused, not reported.

What remains genuinely unresolved is the mechanism. Even a correct A/B cannot
isolate the draft head, because `mtp_enabled` is a single switch with no
"draft head only" option: the two arms also differ in which decode kernel runs
(multi-row row-exact verify vs plain serial decode). And the PLE block is not
part of the toggle at all -- in the Qwen4-Exp decoder it runs as
`if "ple" in self:` inside the layer forward, adding the n-gram embedding to
hidden states every step whether MTP is on or not, so its SSD reads are in both
arms. This is not a "with PLE vs without PLE" comparison either.
"""
import json
import statistics as st
import sys

import omlx_client as L
from tasks import TASKS

TASKMAP = {t[0]: t for t in TASKS}
PICK = ["T2_code_function", "T3_code_repetitive", "T6_qa_factual", "T8_creative_writing"]
REPS = 3
# Throughput noise floor measured over 198 cells at n=3.
NOISE_PCT = 4.5
OUT = sys.argv[1] if len(sys.argv) > 1 else "ab_mtp_on_off.json"


def measure(label, expect_mtp):
    res = {}
    for tk in PICK:
        _, _, prompt, mt = TASKMAP[tk]
        runs = []
        for _ in range(REPS):
            r = L.generate(prompt, mt, 0.0)
            runs.append({"tps": r.get("gen_tps"),
                         "alpha": r.get("accept_pct"),
                         "completion_tokens": r.get("completion_tokens"),
                         "n_mtp_lines": r.get("n_mtp_lines")})
        tps = [x["tps"] for x in runs if x["tps"]]
        alpha = [x["alpha"] for x in runs if x["alpha"] is not None]
        res[tk] = {
            "runs": runs,
            "tps_mean": round(st.mean(tps), 2) if tps else None,
            "alpha_mean": round(st.mean(alpha), 2) if alpha else None,
            "mtp_lines_total": sum(x["n_mtp_lines"] or 0 for x in runs),
        }
        print(f"   {label:<9} {tk:<22} {res[tk]['tps_mean']:>6.2f} tok/s   "
              f"alpha={res[tk]['alpha_mean']}%   "
              f"mtp_lines={res[tk]['mtp_lines_total']}", flush=True)
    lines = sum(v["mtp_lines_total"] for v in res.values())
    if expect_mtp and lines == 0:
        raise RuntimeError(f"{label}: MTP enabled but no statistics were emitted")
    if not expect_mtp and lines:
        raise RuntimeError(
            f"{label}: MTP disabled but the server emitted {lines} statistics "
            f"lines. The toggle did not take, so both arms are the same "
            f"configuration and this comparison is void.")
    return res


print("=" * 76)
print(f"A/B: MTP depth=1 ON vs OFF   ({len(PICK)} tasks x {REPS} reps, T=0)")
print("=" * 76)

L.set_settings(mtp_enabled=True, mtp_fixed_depth=1)
L.wait_settled()
L.generate("warmup", 32, 0.0)
print("\n[A] MTP ON depth=1", flush=True)
on = measure("MTP ON", expect_mtp=True)

L.set_settings(mtp_enabled=False)
L.wait_settled()
L.generate("warmup", 32, 0.0)
print("\n[B] MTP OFF", flush=True)
off = measure("MTP OFF", expect_mtp=False)

print("\n" + "=" * 76)
print(f"Paired comparison (same task, same temperature)")
print("=" * 76)
print(f"{'task':<24}{'ON':>9}{'OFF':>9}{'delta':>10}{'verdict':>16}")
print("-" * 76)
diffs = []
for tk in PICK:
    a, b = on[tk]["tps_mean"], off[tk]["tps_mean"]
    pct = (a / b - 1) * 100
    diffs.append(pct)
    if pct < -NOISE_PCT:
        verdict = "MTP slower"
    elif pct > NOISE_PCT:
        verdict = "MTP faster"
    else:
        verdict = "within noise"
    print(f"{tk:<24}{a:>9.2f}{b:>9.2f}{pct:>9.1f}%{verdict:>16}")
print("-" * 76)
mean_d = st.mean(diffs)
med_d = st.median(diffs)
n_loss = sum(1 for d in diffs if d < -NOISE_PCT)
n_win = sum(1 for d in diffs if d > NOISE_PCT)
print(f"mean {mean_d:+.1f}%   median {med_d:+.1f}%")
print(f"{n_win} of {len(PICK)} tasks faster beyond the {NOISE_PCT}% noise band, "
      f"{n_loss} slower")

payload = {
    "experiment": "ab_mtp_on_off",
    "temperature": 0.0,
    "depth": 1,
    "reps": REPS,
    "tasks": PICK,
    "noise_pct": NOISE_PCT,
    "on": on,
    "off": off,
    "deltas_pct": [round(d, 2) for d in diffs],
    "mean_delta_pct": round(mean_d, 2),
    "median_delta_pct": round(med_d, 2),
    "n_faster": n_win,
    "n_slower": n_loss,
    "toggle_verified": True,
    "confound": (
        "mtp_enabled is a single switch: there is no way to disable only the "
        "draft head, so the two arms also differ in which decode kernel runs "
        "(multi-row row-exact verify vs serial decode). The PLE block is "
        "unconditional (if \"ple\" in self, inside the decoder layer), so it "
        "and its SSD reads are present in both arms. The measured delta is "
        "real; its cause is not isolated by this design."
    ),
}
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(f"\nwrote {OUT}")

print("\n[restore] MTP ON depth=1 ...", flush=True)
L.set_settings(mtp_enabled=True, mtp_fixed_depth=1)
L.wait_settled()
print("restored", flush=True)
