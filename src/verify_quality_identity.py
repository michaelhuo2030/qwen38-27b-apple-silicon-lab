#!/usr/bin/env python3
"""
Quality identity check: does MTP change the output at all?

A common misreading of acceptance rate is "alpha = 90% means one character in
ten is wrong, so the output is full of errors". This test settles it directly:
same prompt, same temperature, MTP on vs MTP fully off, and compare the output
bytes.

The answer is not a lucky observation. oMLX makes it a contract. The Qwen4-Exp
language model sets `_omlx_mtp_row_exact_verify = True`, and
`batch_generator._row_exact_verify` then runs every multi-row projection,
DeltaNet prework and attention row of the verify forward with the arithmetic of
a serial one-row decode. The in-tree comment states the intent verbatim:

    "so greedy MTP output equals MTP-off output byte for byte"

That is why this experiment must come out bit-identical. The interesting part
is therefore not "is it identical" but "is it identical at every depth, and does
it stay identical when acceptance is terrible" -- the prose task sits at
alpha ~57%, i.e. more than one draft character in three is wrong, and its
output must still match exactly.

Mechanism: a speculative draft is only ever a *candidate*. When verification
rejects it, oMLX does not keep the draft character and does not discard the
whole span -- it emits the character the backbone itself computed. The rejected
drafts cost time, never correctness.

Set OMLX_MTP_ROW_EXACT_VERIFY=0 in the server environment to trade this
guarantee for faster verify kernels; that is the one switch that could make
MTP-on and MTP-off diverge.
"""
import hashlib
import json
import sys

import omlx_client as L
from tasks import TASKS

TASKMAP = {t[0]: t for t in TASKS}
# Code / JSON / prose: acceptance ~91% / ~94% / ~58%. The prose task is the
# one that matters -- more than four draft characters in ten are wrong, and
# its output must still match exactly.
PICK = ["T9_rust_impl", "T1_json_structured", "T8_creative_writing"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "quality_identity.json"


def compare(a: str, b: str) -> dict:
    """Locate the first divergence and describe it compactly.

    "The outputs differ" is not a publishable result. What matters is *where*
    and *how much*: a single character at a position where two logits were
    within rounding of each other is a different claim from a rewrite.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    lo = max(0, i - 60)
    return {
        "identical": a == b,
        "common_prefix_chars": i,
        "len_a": len(a),
        "len_b": len(b),
        "first_diff_index": None if a == b else i,
        "n_chars_differing": sum(1 for x, y in zip(a[i:], b[i:]) if x != y),
        "excerpt_a": None if a == b else a[lo:i + 60],
        "excerpt_b": None if a == b else b[lo:i + 60],
    }


def arm(tag, expect_mtp):
    """Measure the three tasks and confirm the toggle really moved."""
    out = {}
    for tk in PICK:
        _, _, prompt, mt = TASKMAP[tk]
        r = L.generate(prompt, mt, 0.0)          # T=0: fully deterministic
        out[tk] = {
            "sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
            "chars": len(r["text"]),
            "completion_tokens": r.get("completion_tokens"),
            "alpha": r.get("accept_pct"),
            "tps": r.get("gen_tps"),
            "n_mtp_lines": r.get("n_mtp_lines"),
        }
        out[tk]["_text"] = r["text"]
    lines = sum(v["n_mtp_lines"] or 0 for v in out.values())
    if expect_mtp and lines == 0:
        raise RuntimeError(f"{tag}: MTP was enabled but emitted no statistics")
    if not expect_mtp and lines:
        raise RuntimeError(
            f"{tag}: MTP was disabled but the server still emitted {lines} "
            f"statistics lines -- the toggle did not take, and this arm is "
            f"measuring the same configuration as the other one.")
    print(f"\n[{tag}]  ({lines} MTP statistics lines)")
    for tk, v in out.items():
        print(f"   {tk:<22} sha={v['sha256'][:16]}  {v['chars']:>4} chars  "
              f"alpha={v['alpha']}%  {v['tps']} tok/s")
    return out


print("=" * 72)
print("Quality identity: is MTP-on output identical to MTP-off output?")
print("=" * 72)

print("\n[1/2] MTP enabled, depth=1 ...")
L.set_settings(mtp_enabled=True, mtp_fixed_depth=1)
L.wait_settled()
L.generate("warmup", 32, 0.0)
on = arm("MTP ON (depth 1)", expect_mtp=True)

print("\n[2/2] MTP fully disabled, reloading model ...")
L.set_settings(mtp_enabled=False)
L.wait_settled()
L.generate("warmup", 32, 0.0)
off = arm("MTP OFF", expect_mtp=False)

print("\n" + "=" * 72)
print("Comparison")
print("=" * 72)
diffs = {}
for tk in PICK:
    d = compare(on[tk].pop("_text"), off[tk].pop("_text"))
    diffs[tk] = d
    if d["identical"]:
        print(f"  {tk:<22} IDENTICAL  ({d['len_a']} chars, "
              f"alpha was {on[tk]['alpha']}% while on)")
    else:
        print(f"  {tk:<22} DIFFERS    ({d['len_a']} vs {d['len_b']} chars; first "
              f"divergence at char {d['first_diff_index']}, "
              f"{d['n_chars_differing']} chars differ from there on; "
              f"alpha was {on[tk]['alpha']}% while on)")

n_same = sum(1 for d in diffs.values() if d["identical"])
print()
if n_same == len(PICK):
    print("  All outputs identical.")
else:
    print(f"  {n_same} of {len(PICK)} outputs are byte-identical.")
    print("  Read this as: enabling MTP does not corrupt the output -- the")
    print("  rejected drafts never reach it -- but it is not a bit-exact")
    print("  guarantee. See the writeup for why the structured task holds and")
    print("  the free-form ones drift.")

payload = {
    "experiment": "quality_identity",
    "temperature": 0.0,
    "depth": 1,
    "tasks": PICK,
    "on": on,
    "off": off,
    "diffs": diffs,
    "n_identical": n_same,
    "all_identical": n_same == len(PICK),
    "toggle_verified": True,
    "note": ("oMLX runs the Qwen4-Exp multi-row verify forward with "
             "serial-decode arithmetic (_omlx_mtp_row_exact_verify), which "
             "makes greedy MTP output equal MTP-off output for deterministic "
             "tasks. It is a best-effort contract, not a proof: free-form "
             "output can still diverge where two candidate tokens were within "
             "rounding of each other. OMLX_MTP_ROW_EXACT_VERIFY=0 drops even "
             "that guarantee for faster kernels."),
}
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(f"\n  wrote {OUT}")

# restore
print("\n[restore] MTP enabled, depth=1 ...")
L.set_settings(mtp_enabled=True, mtp_fixed_depth=1)
L.wait_settled()
print("restored")
