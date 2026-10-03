"""
omlx_client.py — minimal client for driving an oMLX server from experiments.

Deliberately dependency-free (stdlib urllib) so the lab runs anywhere with
Python 3.10+, and so the measurement path has no hidden magic.

Configuration comes from the environment:
    OMLX_BASE       default http://127.0.0.1:8091
    OMLX_API_KEY    default ""            (server may run without auth)
    OMLX_MODEL      default Qwen3.8-Flash-Next-oQ4e-mtp
    OMLX_LOG        default ~/.omlx/logs/server.log
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("OMLX_BASE", "http://127.0.0.1:8091")
API_KEY = os.environ.get("OMLX_API_KEY", "")
MODEL_ID = os.environ.get("OMLX_MODEL", "Qwen3.8-Flash-Next-oQ4e-mtp")
LOG = Path(os.environ.get(
    "OMLX_LOG", str(Path.home() / ".omlx" / "logs" / "server.log")))

# The admin API hands back a session cookie on login. urllib keeps no cookie
# state between calls, so a plain urlopen() PUT comes back 401 even though the
# equivalent `curl -c/-b` pair works. This opener is the whole fix.
_OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

# One MTP statistics line per generation, e.g.
#   MTP[7] finish=stop tokens=234 cycles=78 tok/cycle=3.00 accept=154/156 (98.7%)
#     depth[d1=78/78,d2=76/78] emits[...] timing[backbone=..ms mtp=..ms sample=..ms cache=..ms]
MTP_RE = re.compile(
    r"tokens=(\d+)\s+cycles=(\d+)\s+tok/cycle=([\d.]+)\s+"
    r"accept=(\d+)/(\d+)\s+\(([\d.]+)%\)\s+depth\[([^\]]*)\]"
    r".*?timing\[backbone=([\d.]+)ms\s+mtp=([\d.]+)ms\s+"
    r"sample=([\d.]+)ms\s+cache=([\d.]+)ms"
)

# Minimum number of drafted tokens for an alpha reading to be usable.
# A generation that stops after 3-4 tokens yields accept_den=1..2, where a
# single sample decides the ratio (often 0%). Averaging such readings in
# fabricates effects that do not replicate. See README "Methodology pitfalls".
#
# 25, not 50: the shortest task emits 82 tokens, which is accept_den~43 at
# depth 1 but ~66 at depth 2. A threshold of 50 would drop that task only at
# low depths, making the sample size vary with the very variable under study.
# 25 still rejects the pathological cases (den = 1, 2, 6) where a single
# sample decides the ratio.
MIN_ACCEPT_DEN = 25


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def health() -> dict:
    req = urllib.request.Request(f"{BASE}/health", headers=_headers())
    with _OPENER.open(req, timeout=20) as f:
        return json.load(f)


def get_settings(model_id: str | None = None) -> dict:
    """Read the model's current settings. No reload, no side effects.

    Reading before writing is what makes a client that "just works": the
    speculative-decoding switches are correct almost every time, so the
    expensive write — which unloads and reloads 69.5 GB when the value actually
    changes, ~47 s measured — can simply be skipped.
    """
    if not API_KEY:
        raise RuntimeError("OMLX_API_KEY is empty; see set_settings()")
    login = urllib.request.Request(
        f"{BASE}/admin/api/login",
        data=json.dumps({"api_key": API_KEY, "remember": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    _OPENER.open(login, timeout=30).read()
    with _OPENER.open(f"{BASE}/admin/api/models", timeout=30) as f:
        blob = json.load(f)
    items = blob if isinstance(blob, list) else blob.get("models", blob)
    if isinstance(items, dict):
        items = [items]
    want = model_id or MODEL_ID
    for m in items:
        if (m.get("id") or m.get("model_id")) == want:
            return m.get("settings", {}) or {}
    raise KeyError(f"model {want!r} not in {list(m)[:5]}")


def set_settings(**kv) -> dict:
    """Change model settings through the admin API, and confirm the write landed.

    Three traps here, all of which cost real experimental time:

    1. The admin login returns a **session cookie**, and ``urllib`` keeps no
       cookie state between calls. A login followed by a plain ``urlopen`` PUT
       fails with ``401 {"detail":"Admin authentication required"}`` even
       though the same pair works under ``curl -c ... -b ...``. This one is
       silent by construction, because nothing in the response looks like a
       config problem.
    2. The natural ``try: urlopen(...) except Exception: pass`` — written
       because the call legitimately blocks for the whole model reload and you
       expect spurious errors — turns that 401 into a no-op. The experiment
       then runs against the *unchanged* configuration. This is how a
       "MTP on vs off" comparison silently became "MTP on vs MTP on" and
       reported a 6.9% difference between two identical configs.
    3. Even a successful write triggers `auto-unload` → reload, and `/health`
       can report ``loaded_count == 1`` *before* the unload begins. Returning
       from the write is not the same as the new config being live.

    So: never swallow the write's exceptions, and assert the readback. The PUT
    response echoes the full settings blob and an ``auto_reloaded`` flag, which
    makes verification free.

    Returns the parsed response body.
    """
    if not API_KEY:
        raise RuntimeError(
            "OMLX_API_KEY is empty. This server requires auth, so an empty key "
            "returns a bare 401 from POST /admin/api/login and the traceback "
            "points at the admin API rather than at the missing environment "
            "variable. Export it before running: "
            "OMLX_API_KEY='<key>' python3 run_temp_quality.py ...")

    login = urllib.request.Request(
        f"{BASE}/admin/api/login",
        data=json.dumps({"api_key": API_KEY, "remember": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    _OPENER.open(login, timeout=30).read()

    put = urllib.request.Request(
        f"{BASE}/admin/api/models/{MODEL_ID}/settings",
        data=json.dumps(kv).encode(),
        headers={"Content-Type": "application/json"}, method="PUT")
    # Generous: this blocks for unload + reload of the whole model.
    with _OPENER.open(put, timeout=1800) as f:
        body = json.load(f)

    readback = body.get("settings", {})
    mismatched = {k: (v, readback.get(k)) for k, v in kv.items()
                  if readback.get(k) != v}
    if mismatched:
        raise RuntimeError(
            f"settings write did not stick: requested {mismatched}. "
            f"Server reported { {k: readback.get(k) for k in kv} }.")
    return body


def wait_settled(settle: float = 50.0) -> None:
    """Wait until the model is resident *and* the new config is actually live.

    `health()` can report `loaded_count == 1` before the auto-unload starts,
    so returning from a settings write is a race. Load takes ~40 s here; the
    settle is the difference between measuring the new config and measuring
    the one you just replaced.
    """
    wait_loaded()
    time.sleep(settle)


def wait_loaded(timeout: int = 1800, poll: int = 10) -> float:
    """Block until the model is resident. Returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if health().get("engine_pool", {}).get("loaded_count", 0) >= 1:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(poll)
    raise TimeoutError("model did not finish loading")


def _log_offset() -> int:
    try:
        return LOG.stat().st_size
    except FileNotFoundError:
        return 0


def _mtp_since(offset: int, settle: float = 1.2) -> list[dict]:
    """Parse MTP statistics lines appended after `offset`.

    `settle` exists because the server writes its log asynchronously; reading
    too eagerly silently yields zero lines, which is indistinguishable from
    "MTP never ran" unless you check the count.
    """
    time.sleep(settle)
    try:
        with open(LOG, "rb") as f:
            f.seek(offset)
            blob = f.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return []
    out = []
    for line in blob.splitlines():
        m = MTP_RE.search(line)
        if m:
            out.append({
                "tokens": int(m.group(1)), "cycles": int(m.group(2)),
                "tok_per_cycle": float(m.group(3)),
                "accept_num": int(m.group(4)), "accept_den": int(m.group(5)),
                "accept_pct": float(m.group(6)), "depth_hist": m.group(7),
                "backbone_ms": float(m.group(8)), "mtp_ms": float(m.group(9)),
                "sample_ms": float(m.group(10)), "cache_ms": float(m.group(11)),
            })
    return out


def _post_chat(body: dict) -> dict:
    """One POST, retrying 409 Conflict with exponential backoff."""
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(), headers=_headers(), method="POST")
    # oMLX is configured with max_concurrent_requests=1 here, so a second
    # concurrent request comes back 409 Conflict. That killed an entire
    # five-stage sweep once: a side probe shared the server, every experiment
    # request was rejected, and each stage "completed" seconds after starting
    # because it crashed on the first request. Retry instead of raising -- and
    # do not point anything else at the same server while a sweep is running.
    delay = 5.0
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=1800) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code != 409 or attempt == 7:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 60)
    raise RuntimeError("unreachable")


def _post_chat_with_deadline(body: dict, deadline_s: float = 300.0) -> dict:
    """发一个 chat 请求，并对**总耗时**设硬性上限。

    ## 为什么需要它（2026-10-04 实测，代价是 8 小时 + 122 次生成）

    `urlopen(timeout=1800)` 的 timeout 是**连接/套接字级**的，
    对「TCP 已建立、但对端既不发数据也不关连接」这种**半开连接无效**。
    实测现场：2026-10-04 00:49:12 之后 omlx-server 不再写日志，
    但 `ps` 显示服务端**健康**（healthz 秒回、手工 curl 正常出结果），
    而客户端 `sample` 出来是：

        _buffered_readline → sock_recv_into → sock_call_ex
                          → internal_select → poll      ← 永远挂在这里

    进程 CPU 时间冻结在 0:00.70，`retries=3` 永远不触发
    （没有异常抛出，重试逻辑压根没机会跑）。
    结果：一个 182 次生成的温度扫描**挂死 8 小时 41 分**，只完成 60 次。

    所以这里不能只靠 socket timeout，必须在**外层**加一道墙：
    超时就抛异常，让上层走它已有的重试/记账路径。
    """
    if deadline_s <= 0:
        return _post_chat(body)
    # ⚠️ 这里**必须**用裸 daemon 线程，不能用 `with ThreadPoolExecutor()`。
    #
    # 第一版修复用的是 ThreadPoolExecutor，结果**同样挂死**（实测 90s+ 不返回）：
    # `with` 块退出时调 `ex.shutdown(wait=True)`，又去 join 那个卡在
    # sock_recv 的僵尸线程——超时抛了异常，可主流程仍然被它拖住。
    # 执行验证（模拟半开连接）当场抓到了这个 bug。
    #
    # daemon=True 的线程不阻止解释器退出，所以放弃它之后能真正走人。
    import threading as _th
    box: dict = {}

    def _work():
        try:
            box["ok"] = _post_chat(body)
        except BaseException as e:            # noqa: BLE001
            box["err"] = e

    t = _th.Thread(target=_work, daemon=True)
    t.start()
    t.join(deadline_s)
    if t.is_alive():
        raise TimeoutError(
            f"chat 请求超过 {deadline_s:.0f}s 无响应（疑似服务端半开连接）"
        )
    if "err" in box:
        raise box["err"]
    return box["ok"]


def _extract_text(d: dict) -> str:
    """Pull the assistant text out, or raise a diagnosable error.

    A 182-generation temperature run died at run ~31 on
    `KeyError: 'content'` from d["choices"][0]["message"]["content"], with
    nothing in the server log past the previous completion — the request never
    reached the model. The shape was intermittent: the same task at the same
    temperature succeeded on a retry. So this is treated as a transport fault
    to be retried and recorded, not a modelling result to be guessed at.
    """
    choices = d.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in response: {str(d)[:300]}")
    msg = choices[0].get("message") or {}
    text = msg.get("content")
    if text is None:
        # content can legitimately be null when a response carries only
        # tool_calls or a reasoning block; capture enough to tell which.
        raise ValueError(
            f"message has no content (keys={list(msg)}, "
            f"finish_reason={choices[0].get('finish_reason')}, "
            f"raw={str(d)[:300]})")
    return text


def generate(prompt: str, max_tokens: int = 800, temperature: float = 0.0,
             top_p: float = 0.95, top_k: int = 0, min_p: float = 0.0,
             settle: float = 1.2, retries: int = 3,
             think: bool = False) -> dict:
    """One chat completion, merged with its MTP statistics.

    Returns a dict with throughput, token-weighted alpha, per-cycle timing
    breakdown, and a `valid` flag that must be honoured by the caller.

    think=False 是本仓库的主实验口径：社区实测 Qwen3.8 默认 reasoning_effort=xhigh，
    一轮能烧掉 20K thinking token（Willison 的 pelican 测试用了 21 分钟）。
    27B 官方推荐采样参数**随 thinking 模式变化**，所以 think 一旦打开，
    temperature/top_p 就不再是同一口径，跨档比较会失效 —— 那时要另开一组。
    """
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    if top_k:
        body["top_k"] = top_k
    if min_p:
        body["min_p"] = min_p

    offset = _log_offset()
    t0 = time.time()
    last_err = None
    d = None
    # 硬性总时限：实测一次温度扫描在这里挂了 8h41m（服务端半开连接，
    # socket timeout 形同虚设）。必须让单次调用有确定的上界。
    # 取值依据：正常最慢的请求是 depth sweep 的 900-token 长生成，实测 <120s；
    # 并发 4 条时单波 <480s。300s 给足余量又能兜住挂死。
    call_deadline = float(os.environ.get("OMLX_CALL_DEADLINE_S", "300"))
    for attempt in range(retries):
        try:
            d = _post_chat_with_deadline(body, call_deadline)
            text = _extract_text(d)
            break
        except Exception as e:          # transport or shape fault — retry
            last_err = f"{type(e).__name__}: {e}"
            if attempt == retries - 1:
                text = ""
                break
            time.sleep(3.0 * (attempt + 1))
    wall = time.time() - t0

    lines = _mtp_since(offset, settle)
    usage = d.get("usage", {}) if d else {}
    rec = {
        "gen_tps": usage.get("generation_tokens_per_second"),
        "ttft": usage.get("time_to_first_token"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "wall": round(wall, 2),
        "n_mtp_lines": len(lines),
        "text": text,
    }
    if last_err:
        rec["transport_error"] = last_err
        # ⚠️ `error` 键必须**同步**设置（2026-10-04 修）。
        #
        # 契约不一致的真实后果：传输失败只写在 `transport_error`，
        # 而 `bench_concurrent.py` 读的是 `a.get("error")` —— 永远读不到，
        # 于是**失败请求被当成成功计入吞吐**。更隐蔽的是 `transport_error`
        # 明明存在，下游却统计出 "n_err=0"。
        #
        # 两边都读、都写，这个键就是唯一真相。
        rec["error"] = last_err
        rec["valid"] = False
        return rec

    if not lines:
        # No MTP line at all means the MTP path never engaged for this
        # request; such a run must not be averaged into a comparison.
        rec["valid"] = False
        return rec

    accept_num = sum(x["accept_num"] for x in lines)
    accept_den = sum(x["accept_den"] for x in lines)
    cycles = [x["cycles"] for x in lines if x["cycles"]]

    rec["accept_num"] = accept_num
    rec["accept_den"] = accept_den
    rec["valid"] = accept_den >= MIN_ACCEPT_DEN
    # Token-weighted alpha. Averaging per-line percentages would give short
    # lines the same weight as long ones.
    rec["accept_pct"] = round(100.0 * accept_num / accept_den, 2) if accept_den else None
    rec["tok_per_cycle"] = round(statistics.mean(x["tok_per_cycle"] for x in lines), 3)

    def per_cycle(field: str):
        pairs = [(x[field], x["cycles"]) for x in lines if x["cycles"]]
        return round(statistics.mean(v / c for v, c in pairs), 3) if pairs else None

    rec["backbone_ms_per_cycle"] = per_cycle("backbone_ms")
    rec["mtp_ms_per_cycle"] = per_cycle("mtp_ms")
    rec["cache_ms_per_cycle"] = per_cycle("cache_ms")
    rec["sample_ms_per_cycle"] = per_cycle("sample_ms")
    rec["depth_hist_last"] = lines[-1]["depth_hist"]
    return rec


def aggregate(runs: list[dict]) -> dict:
    """Mean of the valid runs only. Invalid runs are counted, not silently mixed in."""
    good = [r for r in runs if r.get("valid")]

    def mean(field):
        v = [r[field] for r in good if r.get(field) is not None]
        return round(statistics.mean(v), 3) if v else None

    return {
        "tps_mean": mean("gen_tps"),
        "accept_mean": mean("accept_pct"),
        "tok_per_cycle_mean": mean("tok_per_cycle"),
        "backbone_ms_mean": mean("backbone_ms_per_cycle"),
        "mtp_ms_mean": mean("mtp_ms_per_cycle"),
        "cache_ms_mean": mean("cache_ms_per_cycle"),
        "sample_ms_mean": mean("sample_ms_per_cycle"),
        "out_tokens_mean": mean("completion_tokens"),
        "accept_den_all": [r.get("accept_den") for r in good],
        "tps_all": [r.get("gen_tps") for r in good],
        "accept_all": [r.get("accept_pct") for r in good],
        "n_valid": len(good),
        "n_invalid": len(runs) - len(good),
    }
