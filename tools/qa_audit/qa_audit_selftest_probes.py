"""Probe table for `selftest.py` — verifies the exec runner's isolation.

Kept as its own module so the child process can import it by name, which is
the same contract real evaluators use. The probes genuinely `exec` the payload,
because a harness that reports "fine" for a payload it never ran is exactly the
kind of instrument that quietly measures nothing.
"""
import io
from contextlib import redirect_stdout


def _exec_and_report(mod: str):
    """Execute the payload, absorbing its stdout, and report what happened."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(mod, "<payload>", "exec"), {})
    except BaseException as e:                      # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return True, f"executed; artifact wrote {len(buf.getvalue())} chars to stdout"


PROBES = {
    "t":       {"computes": _exec_and_report},
    "hangs":   {"computes": _exec_and_report},
    "crashes": {"computes": _exec_and_report},
}
