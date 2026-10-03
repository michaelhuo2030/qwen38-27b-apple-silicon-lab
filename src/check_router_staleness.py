#!/usr/bin/env python3
"""
check_router_staleness.py — flag router numbers derived under an old validator.

Run: python3 check_router_staleness.py

`router_profiles.json` carries a measured temperature ceiling per profile. That
ceiling is a number someone acts on, so it is only as good as the instrument
that produced it — and the instrument here was wrong three times.

Rather than remember to re-derive the ceilings after every validator change,
each one records the `VALIDATOR_VERSION` it was measured under, and this script
refuses to let a stale one pass unnoticed.

Exit 1 means: some ceilings are stale. That is not necessarily wrong data, it
is data whose provenance no longer matches the code that will be re-run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_checks as Q

PROFILES = Path(__file__).resolve().parent.parent / "data" / "router_profiles.json"


def main() -> int:
    blob = json.loads(PROFILES.read_text())
    profs = blob["profiles"]
    cur = Q.VALIDATOR_VERSION
    stale, unversioned, ok = [], [], []

    for name, p in profs.items():
        if p.get("temp_measured_ceiling") is None:
            continue
        v = p.get("temp_validator_version")
        if v is None:
            unversioned.append(name)
        elif v != cur:
            stale.append((name, v))
        else:
            ok.append(name)

    print(f"current validator: {cur}")
    print(f"  current : {len(ok)} {ok}")
    if unversioned:
        print(f"  UNVERSIONED (no recorded provenance): {len(unversioned)} "
              f"{unversioned}")
    for n, v in stale:
        print(f"  STALE   : {n} was measured under {v}")
    if not stale and not unversioned:
        print("\nall measured ceilings match the current validator")
        return 0
    print("\nRe-derive these ceilings from data scored by the current validator "
          "before shipping the router.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
