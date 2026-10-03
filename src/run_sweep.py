#!/usr/bin/env python3
"""
run_sweep.py — depth and temperature sweeps.

Both are the same shape (one outer loop over a server-side setting, an inner
loop over prompts), so they share one driver.

    # depth sweep at T=0
    python3 run_sweep.py --depths 1,2,3,4,6,8 --temps 0.0 --reps 3 -o depth_sweep.json

    # alpha(T) curve at depth 1 and 2
    python3 run_sweep.py --depths 1,2 --temps 0.0,0.3,0.6,0.9,1.2,1.6 --reps 3 -o temp_sweep.json

Requires an oMLX server with admin access so the depth can be changed without
restarting the process by hand. See README "Changing depth without root".
"""
import argparse
import json
import time
from pathlib import Path

import omlx_client as C
from tasks import TASKS


def set_depth(d: int) -> None:
    """Change mtp_fixed_depth via the oMLX admin API and wait for the reload.

    Any settings write makes oMLX unload and reload the whole model, so this
    blocks for minutes, not seconds. That is fine: it needs no root, unlike
    restarting the service under launchd.

    The write itself is `omlx_client.set_settings`, which is not optional
    bookkeeping. An earlier version of this file logged in with a bare
    `urllib.request.urlopen`, and since urllib keeps no cookie between calls
    the PUT came back `401 {"detail":"Admin authentication required"}` -- and
    an `except Exception: pass` around it turned that into a silent no-op. The
    sweep then "measured" depths 1, 2, 3, 4, 6, 8 while the server stayed at
    depth 1 the whole time, and reported a flat curve. set_settings carries
    the session cookie, refuses to swallow the error, and asserts the
    readback, so a depth that did not take cannot be measured.
    """
    t0 = time.time()
    C.set_settings(mtp_fixed_depth=d)
    C.wait_settled(settle=45)
    print(f"  [depth={d}] reload settled in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,2,3,4,6,8")
    ap.add_argument("--temps", default="0.0")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tasks", default="", help="comma-separated subset; default all")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    temps = [float(x) for x in args.temps.split(",")]
    task_list = ([t for t in TASKS if t[0] in args.tasks.split(",")]
                 if args.tasks else list(TASKS))

    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}

    for d in depths:
        bucket = data.setdefault(str(d), {})
        todo = [t for t in task_list for tp in temps
                if f"{t[0]}@T{tp}" not in bucket]
        if not todo:
            print(f"[depth={d}] already complete, skipping", flush=True)
            continue

        print(f"\n{'='*66}\n[depth={d}] setting depth and waiting for reload", flush=True)
        set_depth(d)
        C.generate("warmup", 32, 0.0)      # exclude first-call compile

        for key, entropy, prompt, max_tokens in task_list:
            for temp in temps:
                cell = f"{key}@T{temp}"
                if cell in bucket:
                    continue
                runs = [C.generate(prompt, max_tokens, temp) for _ in range(args.reps)]
                entry = C.aggregate(runs)
                entry["entropy"] = entropy
                entry["temp"] = temp
                bucket[cell] = entry
                out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                flag = f"  !{entry['n_invalid']} invalid" if entry["n_invalid"] else ""
                print(f"  {key:<20} T={temp:<4} tps={entry['tps_mean']:<7} "
                      f"a={entry['accept_mean']}%  t/c={entry['tok_per_cycle_mean']} "
                      f"bb={entry['backbone_ms_mean']}ms{flag}", flush=True)

    print(f"\ndone -> {out}", flush=True)


if __name__ == "__main__":
    main()
