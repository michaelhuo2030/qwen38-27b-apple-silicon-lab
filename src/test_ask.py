#!/usr/bin/env python3
"""
test_ask.py — does the interface's fence handling do what it claims?

Run: python3 test_ask.py

Fence handling is the one part of this interface that can silently corrupt a
user's file, so it is tested against the two cases that pull in opposite
directions: a code answer that must lose its wrapper, and a document whose
interior code blocks must survive. A rule that gets either wrong is worse than
no rule, because the user only finds out when the file will not compile.

No network, no server: these are pure functions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import clean, unwrap   # noqa: E402

F = "```"

CASES = [
    # (name, input, mode, expected output, expect_changed)
    ("code, wrapper closed", f"{F}python\nimport os\nprint(1)\n{F}\n", "code",
     "import os\nprint(1)", True),
    ("code, wrapper unclosed", f"{F}python\nx = 1\n", "code", "x = 1", True),
    ("code, no wrapper", "y = 2\n", "code", "y = 2", False),
    ("code, interior fence also dropped", f"a\n{F}js\nb\n{F}\nc\n", "code",
     "a\nb\nc", True),
    ("code, bare fence language tag variants", f"{F}\nq=1\n{F}\n", "code",
     "q=1", True),

    # The case that broke the first implementation: a whole document wrapped
    # in ```markdown whose body contains a ```js block. Unwrapping must remove
    # only the outer pair.
    ("markdown doc wrapped whole", f"{F}markdown\n# T\n\n{F}js\nlet a=1\n{F}\n{F}\n",
     "markdown", f"# T\n\n{F}js\nlet a=1\n{F}", True),
    ("markdown doc, interior fences kept", f"# T\n\n{F}js\nlet a=1\n{F}\n\nend\n",
     "markdown", f"# T\n\n{F}js\nlet a=1\n{F}\n\nend", False),
    ("markdown doc with table", "| a | b |\n|---|---|\n| 1 | 2 |\n", "markdown",
     "| a | b |\n|---|---|\n| 1 | 2 |", False),
    # A lone opening fence is ambiguous, and the two readings disagree about
    # what to do. The model wrote a fence and then stopped, so by its own
    # framing the content is code inside that fence; unwrapping recovers it
    # exactly, while keeping the stray opener guarantees a file that will not
    # parse. In markdown mode a document may legitimately open with a code
    # block, but a valid one is always closed. So unwrap, and accept the
    # failure mode that leaves a usable file.
    ("starts with fence, unclosed", f"{F}js\nlet a=1\n", "markdown",
     "let a=1", True),

    ("prose wrapped", f"{F}\n一段话。\n{F}\n", "text", "一段话。", True),
    ("prose with inline ticks", "说 `code` 这个词。", "text", "说 `code` 这个词。",
     False),
]


def main() -> int:
    fails = 0
    for name, src, mode, want, want_changed in CASES:
        got, changed = clean(src, mode)
        ok = got.strip() == want.strip()
        ok_changed = changed == want_changed
        if not (ok and ok_changed):
            fails += 1
            print(f"FAIL  {name}")
            print(f"        got      : {got!r} (changed={changed})")
            print(f"        expected : {want!r} (changed={want_changed})")
        else:
            print(f"PASS  {name:42s} mode={mode:8s} changed={changed}")

    # unwrap must never raise on degenerate input.
    for bad in ["", "```", f"{F}\n{F}\n", "no fence at all", f"\n{F}js\n"]:
        try:
            unwrap(bad)
        except Exception as e:                    # noqa: BLE001
            fails += 1
            print(f"FAIL  unwrap raised on {bad!r}: {e}")

    print()
    if fails:
        print(f"{fails} FAILED — fence handling can silently corrupt a saved file")
    else:
        print(f"{len(CASES)} fence cases pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
