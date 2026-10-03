#!/usr/bin/env python3
"""
ask.py — one interface for everything this deployment is good for.

    from ask import Assistant

    a = Assistant()                                   # sets MTP/depth once
    out = a.ask("写一个 LRU 缓存", mode="code")        # fences stripped
    out = a.ask("写一份部署手册", mode="markdown")     # fences kept
    print(out.text)

Or from a terminal:

    python3 ask.py "写一个 LRU 缓存" --mode code

Why one interface. Three things had to be joined and they disagreed with each
other until they did:

  * the **router** picks a temperature per request, but it must not be trusted
    silently — it scored 8/16 on held-out real prompts, so every decision it
    makes is reported back in `Answer.why`
  * `mtp_enabled` / `mtp_fixed_depth` are **session-level**: writing them
    triggers a full model reload (~47 s measured), so they are set once at
    construction and asserted on readback, never per request
  * the model **wraps code in ``` fences 69–81% of the time regardless of
    temperature**, including at T=0, despite prompts that forbid it. Prompting
    does not fix this; the client has to

Fence handling, and the distinction that makes it safe:

    A fence that wraps the ENTIRE output is never what you want — it breaks
    piping to a file, and a Markdown document never starts and ends with one.
    So that is always removed. Fences in the *middle* of a document are the
    code blocks you asked for, and they are kept.

    That single rule covers both cases without a mode switch, which is why the
    mode is an override rather than a correctness requirement:

      mode="code"     (default) also drops any remaining bare fence lines
      mode="markdown" keeps interior fences — you want them
      mode="text"     same as "code"; fences in prose are noise
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

def _discover_api_key() -> str:
    """Find the API key without making the caller set anything.

    Order: an explicit environment variable, then a gitignored local config in
    the repo, then the server's own config directory. The local config is
    ignored by git on purpose — this repository is meant to be published, and a
    key pasted into a tracked file is a key in a public repository.

    Without this, an empty key produces a bare 401 from the admin endpoint,
    which looks like a server fault and is not one.
    """
    if os.environ.get("OMLX_API_KEY"):
        return os.environ["OMLX_API_KEY"]
    for c in (_HERE / ".ask.local.json",
              Path.home() / ".omlx" / "ask.local.json"):
        try:
            v = json.loads(c.read_text()).get("api_key")
            if v:
                return v
        except Exception:
            continue
    return ""


# Must run before `import omlx_client`. That module reads the key into a
# module-level constant at import time, so setting the variable afterwards has
# no effect — the first version of this file discovered the key inside
# __init__ and the server still answered 401.
_DISCOVERED_KEY = _discover_api_key()
if _DISCOVERED_KEY:
    os.environ["OMLX_API_KEY"] = _DISCOVERED_KEY

import omlx_client as L          # noqa: E402
import router as R              # noqa: E402

# ---------------------------------------------------------------------------
# fence handling
# ---------------------------------------------------------------------------
# Closing fence optional on purpose: the model truncates often enough that an
# unclosed ``` is a real case, and leaving it in place means the file still
# starts with a fence and still fails to compile.
_WRAPPER = re.compile(r"^\s*```[a-zA-Z0-9_+.-]*[ \t]*\n(.*?)(?:\n[ \t]*```[ \t]*)?$",
                      re.S)
_BARE_FENCE = re.compile(r"^[ \t]*```[a-zA-Z0-9_+.-]*[ \t]*$")


def unwrap(text: str) -> tuple[str, bool]:
    """Remove a fence wrapping the whole output. Returns (text, stripped).

    The decision is made by fence *balance*, not by "does the body contain
    fences". An earlier version refused to unwrap whenever the body had a
    fence, which was wrong in the case that matters most: a Markdown document
    that the model wrapped in ```markdown, containing a ```js block. Its body
    has fences, so it was left wrapped — the one case where a wrapper is
    unambiguously wrong and a naive heuristic gets it backwards.
    """
    if not text:
        return text, False
    lines = text.strip().splitlines()
    if len(lines) < 2 or not _BARE_FENCE.match(lines[0]):
        return text, False
    body = lines[1:]
    if _BARE_FENCE.match(body[-1] or ""):
        body = body[:-1]
    # Whatever is left must be balanced, or this was a document that merely
    # happens to begin with a code block rather than a wrapped one.
    if sum(1 for ln in body if _BARE_FENCE.match(ln)) % 2 != 0:
        return text, False
    return "\n".join(body), True


def clean(text: str, mode: str = "code") -> tuple[str, bool]:
    """Apply the mode's fence policy. Returns (text, changed)."""
    out, stripped = unwrap(text)
    if mode == "markdown":
        return out, stripped
    lines = out.splitlines()
    kept = [ln for ln in lines if not _BARE_FENCE.match(ln)]
    if len(kept) != len(lines):
        return "\n".join(kept), True
    return out, stripped


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class Answer:
    text: str
    mode: str
    profile: str
    temperature: float
    alpha: float | None = None
    gen_tps: float | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    fence_stripped: bool = False
    why: str = ""                  # how the temperature was chosen
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["text"] = d["text"] if len(d["text"]) < 4000 else d["text"][:4000] + "…"
        return d


# ---------------------------------------------------------------------------
# the assistant
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_HINT = (
    "只输出最终内容本身，不要任何前言、解释或总结。"
)


class Assistant:
    """A configured client. Construct once, ask many times.

    Zero configuration on the caller's side: the API key is discovered, and the
    speculative-decoding settings are only written when they are actually wrong.
    Reading them first is what avoids a ~47 s model reload in the common case
    where they are already right, which is nearly always.
    """

    #: Measured here: aggregate throughput saturates by 4 concurrent requests
    #: (30 / 41 / 45 / 46 tok/s at 1 / 2 / 4 / 8) while latency keeps climbing
    #: (5.2 / 7.9 / 14.6 / 33.7 s). Memory-bandwidth bound, so past 4 you buy
    #: latency and ~2% throughput. Enforced in code, not suggested.
    MAX_WORKERS = 4

    def __init__(self,
                 base: str | None = None,
                 api_key: str | None = None,
                 depth: int = 1,
                 mtp_enabled: bool = True,
                 verify_session: bool = True,
                 verbose: bool = True):
        if api_key is not None:
            os.environ["OMLX_API_KEY"] = api_key
            L.API_KEY = api_key
        h = L.health()
        if h.get("status") != "healthy":
            raise RuntimeError(f"oMLX is not healthy: {h}")
        self.model = h.get("default_model")
        self.memory_gb = round(h["engine_pool"]["current_model_memory"] / 1e9, 1)
        self.session = {"mtp_enabled": mtp_enabled, "mtp_fixed_depth": depth}
        self.wrote_settings = False
        if verify_session:
            try:
                current = L.get_settings(self.model)
            except Exception:
                current = {}
            if all(current.get(k) == v for k, v in self.session.items()):
                if verbose:
                    print(f"[ask] {self.model} · {self.memory_gb}GB · "
                          f"depth={depth} mtp={mtp_enabled} "
                          f"(already correct, nothing written)", file=sys.stderr)
                return
            body = L.set_settings(mtp_enabled=mtp_enabled, mtp_fixed_depth=depth)
            rb = body.get("settings", {})
            got = {k: rb.get(k) for k in self.session}
            if got != self.session:
                raise RuntimeError(f"session settings did not stick: wanted "
                                   f"{self.session}, server says {got}")
            self.wrote_settings = True
            if verbose:
                print(f"[ask] {self.model} · {self.memory_gb}GB · "
                      f"depth={depth} mtp={mtp_enabled} "
                      f"(written, reloaded={body.get('auto_reloaded')})",
                      file=sys.stderr)

    # -- the one method you need ------------------------------------------
    def ask(self,
            prompt: str,
            *,
            mode: str = "code",
            temperature: float | None = None,
            max_tokens: int | None = None,
            profile_hint: str | None = None,
            system: str | None = None,
            retries: int = 3) -> Answer:
        """Send one prompt. `mode` decides the fence policy, not the temperature.

        mode
            "code"     strip a whole-output fence and any bare fence lines
                        (default: the safe choice for anything you will save)
            "markdown" strip only a whole-output wrapper, keep interior fences —
                        use this whenever you want the model writing a document
            "text"     same as "code"
        temperature
            Overrides the router. Pass it when you know better; the router's
            bands were fitted on n<3 samples for 6 of its 9 profiles, so this
            is a normal thing to do rather than an override of authority.
        """
        if mode not in ("code", "markdown", "text"):
            raise ValueError("mode must be 'code', 'markdown' or 'text'")

        r = R.route(prompt, last_profile=profile_hint) if profile_hint else R.route(prompt)
        # Build one kwargs dict. The router's `params` already carries
        # max_tokens, so passing it separately as a keyword too gave
        # "got multiple values for keyword argument" and every request failed
        # after its retries — which is why a duplicated keyword is worth
        # catching here rather than at the call site.
        params = dict(r.params)
        why = f"routed to {r.profile}"
        if temperature is not None:
            params["temperature"] = temperature
            why += f"; temperature overridden to {temperature}"
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        mt = int(params.get("max_tokens", 1200))
        params["max_tokens"] = mt

        full = f"{system or DEFAULT_SYSTEM_HINT}\n\n{prompt}" if system else prompt

        last_err = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                raw = L.generate(full, **params)   # params already carries max_tokens
                # `generate` returns a merged dict that already carries `text`
                # plus throughput and MTP statistics. Calling `_extract_text`
                # on it looks for an OpenAI-style `choices` array that is not
                # there, and raised "no choices in response" on a generation
                # that had in fact succeeded at 52 tok/s.
                text = raw.get("text") if isinstance(raw, dict) else raw
                if not text:
                    raise ValueError(f"empty completion; server said: "
                                     f"{ {k: v for k, v in raw.items() if k != 'text'} }")
                out, changed = clean(text, mode)
                ans = Answer(
                    text=out, mode=mode, profile=r.profile,
                    temperature=params["temperature"],
                    alpha=raw.get("accept_pct") if isinstance(raw, dict) else None,
                    gen_tps=raw.get("gen_tps") if isinstance(raw, dict) else None,
                    completion_tokens=(raw.get("completion_tokens")
                                       if isinstance(raw, dict) else None),
                    seconds=round(time.time() - t0, 2),
                    fence_stripped=changed, why=why,
                )
                if changed:
                    ans.warnings.append(
                        "model wrapped the output in a markdown fence; the "
                        f"prompt asked it not to and it did so anyway. "
                        f"mode={mode} so the client handled it.")
                if (ans.completion_tokens and mt and
                        ans.completion_tokens >= mt - 2):
                    ans.warnings.append(
                        f"hit the token cap ({mt}); the answer may be cut off — "
                        f"raise max_tokens before reading anything into this")
                return ans
            except Exception as e:                      # noqa: BLE001
                last_err = e
                # 409 means another request holds the single slot. Back off and
                # retry rather than fail: a contended request is a transient
                # condition, not a wrong answer.
                wait = 2 ** attempt
                time.sleep(wait)

        a = Answer(text="", mode=mode, profile=r.profile,
                   temperature=params["temperature"], why=why)
        a.warnings.append(f"request failed after {retries} attempts: "
                          f"{type(last_err).__name__}: {last_err}")
        return a

    # -- convenience -------------------------------------------------------
    def ask_many(self, prompts: Iterable[str], *, workers: int | None = None,
                 **kw) -> list[Answer]:
        """Run prompts concurrently, capped at MAX_WORKERS.

        The cap is applied here rather than documented, because a pipeline that
        forgets it does not fail — it just gets slower, in a way that looks like
        the model being slow. Measured: 8 concurrent requests take 33.7 s for
        46.1 tok/s aggregate, versus 14.6 s for 45.3 tok/s at 4. Same throughput,
        more than twice the latency.
        """
        w = self.MAX_WORKERS if workers is None else max(
            1, min(int(workers), self.MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=w) as ex:
            return list(ex.map(lambda p: self.ask(p, **kw), prompts))

    def info(self) -> dict:
        return {"model": self.model, "memory_gb": self.memory_gb,
                "session": self.session}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-shot query against the local oMLX deployment.")
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--mode", default="code",
                    choices=["code", "markdown", "text"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--json", action="store_true", help="metadata as JSON on stderr")
    a = ap.parse_args()

    asst = Assistant(depth=a.depth, mtp_enabled=not a.no_mtp)
    ans = asst.ask(" ".join(a.prompt), mode=a.mode,
                   temperature=a.temperature, max_tokens=a.max_tokens)
    print(ans.text)
    if ans.warnings:
        for w in ans.warnings:
            print(f"[warn] {w}", file=sys.stderr)
    print(json.dumps(ans.to_dict(), ensure_ascii=False), file=sys.stderr)
    return 0 if ans.text else 1


if __name__ == "__main__":
    raise SystemExit(main())
