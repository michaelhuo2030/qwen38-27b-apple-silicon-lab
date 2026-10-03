"""ladder27 —— Qwen3.8-27B 四档精度阶梯的实验 harness。

为什么不用 `ask.py` 的 router
----------------------------
`ask.py` 用 `router.py` 决定采样参数，那套 band 是在 **125B Flash** 上拟合的。
Qwen3.8-27B 官方推荐的采样参数**随 thinking 模式而变**：

    thinking on : temperature 1.0  top_p 0.95 top_k 20 presence_penalty 0.0
    thinking off: temperature 0.7  top_p 0.80 top_k 20 presence_penalty 1.5

沿用 125B 的 router 会让「温度」和「thinking 模式」两个变量缠在一起，
实验结论直接失效。所以这里**不路由，采样参数显式声明且随结果一起记录**。

复用 ask.py 的哪部分
-------------------
与模型无关的部分照用（`clean`/`unwrap` 围栏处理），验证器与探针也照用
（`quality_checks` / `exec_probes` / `qa_audit`）。**清洗和判分是模型无关的，
采样策略才是模型相关的** —— 这条分界线是本文件存在的理由。

这个文件专门挡住的坑（每条都来自真实翻车，不是假想）
----------------------------------------------------
P1  两个模型服务器同驻 → N=6 崩塌 + Metal OOM（社区实测）。
    → `assert_single_resident()` 在每次测量前调用，不满足直接抛。
P2  改 settings 不重载 = 没改。写完必须回读 + 强制 reload。
P3  admin 路由认 session cookie 不认 Bearer，静默 401 被吞 → 跑在旧配置上。
    → `_put` 断言回读，**永不吞异常**。
P4  `reasoning_effort` 默认 xhigh，在简单 prompt 上狂烧 token。
    → 每个请求显式带 `enable_thinking`，并记进结果。
P5  prefix cache 让第二次请求变快，伪装成性能提升。
    → 记录 before/after cache token，并要求重复测量用**不同 prompt**。
P6  TurboQuant KV / ANE prefill 可能**静默**改变输出（社区明确 INT8 会改输出）。
    → `sentinel()` 可行性哨兵：任何配置变更后先验证输出没坏，再测性能。
P7  短输出被 TTFT 主导（v1 任务只有 11–70 token 的教训）。
    → 记录 ttft 与 generation_tokens_per_second 分开，不只看总 tok/s。
P8  `qwen35_oq_a8_enabled` 与 `qwen35_ane_prefill_enabled` 互斥。
    → 配置校验层直接拒绝非法组合。
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from http.cookiejar import CookieJar
from typing import Any, Iterable

BASE = os.environ.get("OMLX_BASE", "http://127.0.0.1:8091")
KEY_FILE = os.path.expanduser(
    "~/Documents/mtp-depth-lab/src/.ask.local.json")
LOG = os.path.expanduser("~/.omlx/logs/omlx_launchd.log")

# ---------------------------------------------------------------- 档位定义

TIERS: dict[str, dict[str, Any]] = {
    "ternary": {"id": "Qwen3.8-27B-ternary-oQ3-mtp", "gb": 13.86, "bpw": 3.70},
    # oQ3e：同一条 oQ 流水线的真 3-bit（d9beuD 从 Qwen/Qwen3.8-27B 直接量化）。
    # 与 ternary 的区别是**位宽覆盖更全**：155 个模块有逐模块位宽指定
    # （92→146 个 5-bit、8 个 4-bit、1 个 6-bit），而 TokenAI-zer 那份
    # 只有 100 个。这是检验「四档无差异是不是因为 oQ3 实际位宽不够低」的关键对照。
    "oq3e":     {"id": "Qwen3.8-27B-oQ3e-mtp",        "gb": 13.81, "bpw": 3.60},
    "4bit":    {"id": "Qwen3.8-27B-oQ4e-mtp",       "gb": 16.97, "bpw": 4.70},
    "6bit":    {"id": "Qwen3.8-27B-oQ6e-mtp",       "gb": 23.72, "bpw": 6.70},
    "8bit":    {"id": "Qwen3.8-27B-oQ8e-mtp",       "gb": 30.00, "bpw": 8.50},
}
BY_ID = {v["id"]: k for k, v in TIERS.items()}

# ---------------------------------------------------------------- 采样
# 官方推荐（Qwen3.8-27B model card）。**显式声明，不路由。**
SAMPLING: dict[str, dict[str, float]] = {
    # 决定性对照用：T=0 让同一配置重复跑必须逐字一致，任何差异都是配置漂移
    "frozen":    {"temperature": 0.0},
    # 官方推荐
    "think_on":  {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                  "presence_penalty": 0.0},
    "think_off": {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                  "presence_penalty": 1.5},
}

# 服务端日志里的 MTP 统计行
MTP_RE = re.compile(
    r"accepted[^\d]*(\d+)[^\d]*(\d+)[^\d]*([\d.]+)"          # tokens cycles tpc
    r"[^\d]*(\d+)\s*/\s*(\d+)[^\d]*([\d.]+)"                # accept_num/den pct
    r"[^\n]*?depth_hist=([^\s,]+)",                          # 各 depth 命中分布
    re.I)


# ---------------------------------------------------------------- HTTP

_op = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(CookieJar()))


def _api_key() -> str:
    with open(KEY_FILE) as f:
        return json.load(f)["api_key"]


def _login() -> None:
    """admin 路由要 session cookie。**不先做这一步，后面全是静默 401。**"""
    r = urllib.request.Request(
        BASE + "/admin/api/login",
        data=json.dumps({"api_key": _api_key(), "remember": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    _op.open(r, timeout=30).read()


def _get(path: str, timeout: int = 60) -> dict:
    r = urllib.request.Request(
        BASE + path, headers={"Authorization": f"Bearer {_api_key()}"})
    with _op.open(r, timeout=timeout) as f:
        return json.load(f)


def _put_settings(model_id: str, **kv) -> dict:
    _login()
    r = urllib.request.Request(
        f"{BASE}/admin/api/models/{model_id}/settings",
        data=json.dumps(kv).encode(),
        headers={"Content-Type": "application/json"}, method="PUT")
    with _op.open(r, timeout=3600) as f:          # 会阻塞整个 unload+reload
        body = json.load(f)
    back = body.get("settings", {})
    bad = {k: (v, back.get(k)) for k, v in kv.items() if back.get(k) != v}
    if bad:
        # 吞掉这个异常 = 整轮实验跑在未改动的配置上（125B 时踩过：
        # "MTP on vs off" 变成 "on vs on"，报出 6.9% 的假差异）
        raise RuntimeError(
            f"{model_id}: 写入未生效，回读不一致 {bad}；"
            f"这轮实验作废，不要吞掉这个异常")
    return back


def _post(path: str, body: dict, timeout: int = 3600) -> dict:
    _login() if path.startswith("/admin") else None
    r = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_api_key()}",
                 "Content-Type": "application/json"}, method="POST")
    with _op.open(r, timeout=timeout) as f:
        return json.load(f)


def status() -> dict:
    return _get("/api/status")


# ---------------------------------------------------------------- 配置校验

class ConfigError(ValueError):
    """配置本身就不合法 —— 必须在发出任何请求前拦下来。"""


def validate_config(ane: bool, oq_a8: bool, turboquant: int | None,
                    mtp: bool, dflash: bool, vlm_mtp: bool) -> None:
    """把 oMLX 里那些「互斥 / 取值范围」的约束搬到客户端。

    为什么要搬：oMLX 只在**写 settings 那一刻**抛错，而写 settings 会触发
    一次完整的模型重载。先在本地校验，坏配置根本不会碰到服务器。

    ⚠️ `turboquant` 传**原始值**，不要先过 `bool()`。
    `bool(4)` 和 `bool(8)` 都是 True，位宽信息在这一步就没了，
    于是「只允许 4 或 8」这条检查永远看不到真实值 —— 空转。
    （本仓库早期栽过同类的：配置里 0 表示"关闭"，被 `if x:` 判成"没配"，
     于是显式关闭被忽略、回落到代码默认值。）
    """
    if oq_a8 and ane:
        raise ConfigError(
            "qwen35_oq_a8_enabled 与 qwen35_ane_prefill_enabled 互斥"
            "（oMLX model_settings.py:504 会拒绝）。"
            "本机 M2 Max 无 INT8 矩阵单元（INT8 与 FP16 打平 1.01×），"
            "选 ANE，别选 a8。")
    if mtp and (dflash or vlm_mtp):
        raise ConfigError(
            "mtp_enabled 与 dflash_enabled / vlm_mtp_enabled 互斥"
            "（model_settings.py 文档明示）。")
    if turboquant is not None and turboquant not in (0, 4, 8):
        raise ConfigError(
            f"turboquant_kv_bits 只能是 0 / 4 / 8（0 与 None 都表示关闭），"
            f"收到 {turboquant!r}（{type(turboquant).__name__}）")


# ---------------------------------------------------------------- 哨兵

#: 任何配置变更后先跑这个。目的不是测质量，是**发现输出被静默改坏**。
#: TurboQuant KV 与 ANE 都可能让输出变化但不报错。
SENTINELS: list[tuple[str, str, str]] = [
    ("arithmetic", "What is 2+2*3? Reply with the number only.", "8"),
    ("json", 'Output only this JSON, no prose, no fence: {"k": 1}', '"k"'),
    ("code", "Write a Python function add(a, b) that returns a + b. "
             "Code only, no fence.", "def"),
    ("repeat", "Reply with exactly: banana", "banana"),
]


@dataclass
class SentinelReport:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


# ---------------------------------------------------------------- 结果

@dataclass
class Answer:
    text: str
    tier: str
    model_id: str
    config: dict
    sampling: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft: float | None = None            # 服务端给的
    gen_tps: float | None = None         # 服务端给的稳态 decode
    wall: float | None = None
    alpha: float | None = None           # token 加权接受率 %
    tok_per_cycle: float | None = None
    n_mtp_lines: int = 0
    cached_before: int | None = None
    cached_after: int | None = None
    truncated: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- MTP 日志

def _log_offset() -> int:
    try:
        return os.stat(LOG).st_size
    except FileNotFoundError:
        return 0


def mtp_since(offset: int, settle: float = 1.2) -> list[dict]:
    """解析 offset 之后追加的 MTP 统计行。

    `settle` 不能省：服务端异步写日志，读太早会静默拿到 0 行，
    而「0 行」和「MTP 根本没跑」长得一模一样。调用方必须看 `n_mtp_lines`。
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
                "accept_pct": float(m.group(6)),
                "depth_hist": m.group(7),
            })
    return out


def weighted_alpha(lines: Iterable[dict]) -> float | None:
    """token 加权 α。

    为什么不能简单平均：不同请求的 token 数差一个数量级，
    简单平均等于给 20-token 的回答和 700-token 的回答同等权重。
    本仓库早前就是这么算的，后来改成 token 加权 + n≥2 才站得住。
    """
    lines = list(lines)
    den = sum(x["accept_den"] for x in lines)
    if den <= 0:
        return None
    num = sum(x["accept_num"] for x in lines)
    return round(100.0 * num / den, 2)


# ---------------------------------------------------------------- 主类

class Ladder:
    """一个档位 + 一套配置的实验会话。"""

    #: 27B 的并发上限**没有实测过**。125B 上是 4（权重 78GB 摊薄不了），
    #: 而 27B 只有 13.9–30GB，omlx.ai 同型号实测 8 并发有 6.49× 加速。
    #: 所以这里**先不封顶到 4**，但默认仍保守设 4，
    #: 第一次跑时用 sweep_concurrency() 测出真实饱和点再改。
    MAX_WORKERS = 4

    def __init__(self, tier: str, *, think: bool = False,
                 ane: bool = True, turboquant: int | None = 0,
                 oq_a8: bool = False, mtp: bool = True,
                 dflash: bool = False, vlm_mtp: bool = False,
                 depth: int | None = None, ane_gdn: bool = True,
                 verbose: bool = True):
        if tier not in TIERS:
            raise KeyError(f"未知档位 {tier!r}，可选 {list(TIERS)}")
        self.tier = tier
        self.model_id = TIERS[tier]["id"]
        self.think = think
        self.ane = ane
        # 归一化：0 和 None 都表示"关闭"，统一成 None，
        # 这样下游所有 `is not None` 的判断都只对真实位宽生效。
        # TurboQuant 默认关的三条理由见 settings() 里的注释。
        self.turboquant = turboquant or None
        self.ane_gdn = ane_gdn
        self.oq_a8 = oq_a8
        self.mtp = mtp
        self.dflash = dflash
        self.vlm_mtp = vlm_mtp
        self.depth = depth
        self.verbose = verbose
        # 传原始值，不传 bool()：见 validate_config 的注释
        validate_config(ane, oq_a8, turboquant, mtp, dflash, vlm_mtp)

    # -- 配置 -------------------------------------------------------------
    @property
    def sampling_name(self) -> str:
        return "think_on" if self.think else "think_off"

    def settings(self) -> dict:
        # ⚠️ depth=0 必须表示「关闭 MTP」。原来这里无条件写
        #    mtp_enabled=self.mtp + mtp_fixed_depth=self.depth，
        #    于是 depth=0 会发出 {mtp_enabled: True, mtp_fixed_depth: 0} ——
        #    自相矛盾，而且 oMLX 0.7.0 对 mtp_fixed_depth=0 直接返 **HTTP 400**，
        #    整个 sweep 第一组就崩。之前所有实验 depth ≥ 1，所以从没触发。
        mtp_on = bool(self.mtp) and int(self.depth or 0) > 0
        s = {
            "mtp_enabled": mtp_on,
            # depth ≤ 0 时干脆不发这个键：发 0 会被服务端拒，
            # 沿用上一次的值反而可能让「关闭 MTP」这组名不副实
            **({"mtp_fixed_depth": int(self.depth)} if mtp_on else {}),
            "mtp_adaptive_max_depth": None,
            "qwen35_ane_prefill_enabled": self.ane,
            "qwen35_oq_a8_enabled": self.oq_a8,
            "turboquant_kv_enabled": self.turboquant not in (None, 0),
            "dflash_enabled": self.dflash,
            "vlm_mtp_enabled": self.vlm_mtp,
            "is_pinned": False,
        }
        if self.ane:
            s.update({
                # 同型号（M2 Max 38c/96GB）两份官方 recipe 的 ANE 参数互不相同，
                # 社区还有第三种说法，所以这几个值本身就是 E3 要 A/B 的对象，
                # 这里取 Recipe B（v0.6.2，有 batching 数据的那份）。
                # E3 要扫：fraction 0.4 vs 0.53、dual_ane、gdn on/off、
                #         sequence_length 1024 vs 2048。
                "qwen35_ane_prefill_fraction": 0.5,
                "qwen35_ane_prefill_sequence_length": 2048,
                "qwen35_ane_prefill_max_layers": 64,   # 正好 27B 的 64 层
                "qwen35_ane_prefill_dual_ane": True,
                # 旧注释说"社区最优是 False"，但那是别的机器。
                # 同型号两份 recipe 都是 gdn=True，其中 v0.6.1 那份的用户备注
                # 还明确写了 "GDN on"。默认值改回 True，由 E3 实测定夺；
                # 要做 A/B 就传 ane_gdn=False。
                "qwen35_ane_prefill_gdn": self.ane_gdn,
            })
        if self.turboquant is not None:
            s["turboquant_kv_bits"] = float(self.turboquant)
        return s

    def describe(self) -> dict:
        return {"tier": self.tier, "model_id": self.model_id,
                "weights_gb": TIERS[self.tier]["gb"],
                "bpw": TIERS[self.tier]["bpw"],
                "sampling": self.sampling_name, **self.settings()}

    # -- 前置条件 ---------------------------------------------------------
    def assert_single_resident(self, allow_unloaded: bool = False) -> None:
        """测量前的硬闸：GPU 上必须只有目标这一个模型。

        社区实测：第二个 30GB 模型常驻时 N=6 崩塌到 7.6 tok/s 并 Metal OOM；
        独占时 N=16 毫无问题。**同硬件，独占 vs 双驻的差异就是这个量级。**
        """
        st = status()
        loaded = st.get("loaded_models", [])
        if len(loaded) > 1:
            raise RuntimeError(
                f"GPU 上同时有 {len(loaded)} 个模型 {loaded}；"
                f"测量结果一律作废。先卸载到只剩 {self.model_id}")
        if loaded and loaded[0] != self.model_id:
            raise RuntimeError(
                f"当前加载的是 {loaded[0]}，不是目标 {self.model_id}")
        if not loaded and not allow_unloaded:
            raise RuntimeError("目标模型未加载")

    # -- 激活 -------------------------------------------------------------
    def activate(self, unload_others: bool = True) -> "Ladder":
        if unload_others:
            for other in set(status().get("loaded_models", [])) - {self.model_id}:
                self._log(f"  卸载 {other}")
                _post(f"/v1/models/{other}/unload", {}, timeout=1800)
            time.sleep(2)

        self._log(f"  写 settings ({self.tier})")
        _put_settings(self.model_id, **self.settings())

        if self.model_id not in status().get("loaded_models", []):
            self._log(f"  加载 {self.tier}（{TIERS[self.tier]['gb']} GB）…")
            _post(f"/v1/models/{self.model_id}/load", {}, timeout=7200)

        # 写完 settings 必须重新加载才生效；这里再断言一次单驻
        self.assert_single_resident()
        self._log(f"  就绪 {self.describe()}")
        return self

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- 请求 -------------------------------------------------------------
    def ask(self, prompt: str, *, max_tokens: int = 700,
            sampling: str | None = None, settle: float = 1.5,
            retries: int = 3) -> Answer:
        name = sampling or self.sampling_name
        if name not in SAMPLING:
            raise KeyError(f"未知采样组 {name!r}，可选 {list(SAMPLING)}")
        params = dict(SAMPLING[name])
        params["max_tokens"] = max_tokens

        before = status().get("total_cached_tokens")
        off = _log_offset()
        t0 = time.time()
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": self.think},
            **params,
        }
        text, err, usage = "", None, {}
        for attempt in range(retries):
            try:
                d = _post("/v1/chat/completions", body)
                text = (d.get("choices") or [{}])[0]
                text = (text.get("message") or {}).get("content") or ""
                usage = d.get("usage", {}) or {}
                if not text:
                    raise ValueError(f"空回复: { {k: v for k, v in d.items() if k != 'choices'} }")
                err = None
                break
            except Exception as e:                     # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        wall = time.time() - t0
        lines = mtp_since(off, settle)
        after = status().get("total_cached_tokens")

        ct = usage.get("completion_tokens")
        return Answer(
            text=text, tier=self.tier, model_id=self.model_id,
            config=self.settings(), sampling=name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=ct,
            ttft=usage.get("time_to_first_token"),
            gen_tps=usage.get("generation_tokens_per_second"),
            wall=round(wall, 2),
            alpha=weighted_alpha(lines),
            tok_per_cycle=(statistics.mean([x["tok_per_cycle"] for x in lines])
                           if lines else None),
            n_mtp_lines=len(lines),
            cached_before=before, cached_after=after,
            truncated=bool(ct and ct >= max_tokens - 2),
            error=err,
        )

    # -- 哨兵 -------------------------------------------------------------
    def sentinel(self, frozen: bool = True) -> SentinelReport:
        """配置变更后验证输出没被静默改坏。

        `frozen=True` 用 T=0 —— 同一配置下逐字可比。任何差异都说明
        TurboQuant / ANE 之类的加速改写了输出，而不是「模型随机」。
        """
        rep = SentinelReport()
        for name, prompt, needle in SENTINELS:
            a = self.ask(prompt, max_tokens=64,
                         sampling="frozen" if frozen else None, settle=0.8)
            rep.outputs[name] = a.text[:200]
            if a.error:
                rep.failed.append(f"{name}: 请求失败 {a.error}")
            elif needle.lower() not in a.text.lower():
                rep.failed.append(
                    f"{name}: 期望包含 {needle!r}，实际 {a.text[:80]!r}")
            else:
                rep.passed.append(name)
        return rep

    # -- 深度扫描 ---------------------------------------------------------
    def sweep_depth(self, depths: list[int], tasks, reps: int = 3,
                    progress=print) -> list[dict]:
        """扫 MTP depth，每档同时收 α 和 decode 吞吐。

        返回的行已按 token 加权算好 α（不是简单平均）。
        """
        self.activate()
        rows = []
        for d in depths:
            self.depth = d
            _put_settings(self.model_id, mtp_fixed_depth=d,
                          mtp_enabled=d > 0)
            _post(f"/v1/models/{self.model_id}/unload", {}, timeout=1800)
            _post(f"/v1/models/{self.model_id}/load", {}, timeout=7200)
            progress(f"  depth={d} 已加载")
            for key, ent, prompt, mt in tasks:
                lines, tps, ttfts = [], [], []
                for r in range(reps):
                    a = self.ask(prompt, max_tokens=mt)
                    if a.error:
                        continue
                    off = _log_offset()
                    lines = lines  # 已在 ask 内解析
                    if a.gen_tps:
                        tps.append(a.gen_tps)
                    if a.ttft:
                        ttfts.append(a.ttft)
                rows.append({
                    "tier": self.tier, "depth": d, "task": key,
                    "entropy": ent, "reps": reps,
                    "alpha": a.alpha, "n_mtp_lines": a.n_mtp_lines,
                    "gen_tps_mean": round(statistics.mean(tps), 2) if tps else None,
                    "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
                    "truncated": a.truncated,
                })
                progress(f"    {key:<24} α={a.alpha}  "
                         f"gen={rows[-1]['gen_tps_mean']} tok/s")
        return rows


    # -- 并发饱和点实测 ---------------------------------------------------
    def sweep_concurrency(self, prompt: str, max_tokens: int = 320,
                          sizes: tuple[int, ...] = (1, 2, 4, 8)) -> list[dict]:
        """27B 的并发饱和点**必须实测**，不能抄 125B 的 4。

        125B 权重 78GB，batch 摊薄不了多少，4 就饱和；
        27B 只有 13.9–30GB，omlx.ai 同型号实测 8 并发有 6.49× 加速。
        照抄 4 会白白浪费一半吞吐。

        每个请求的 prompt **都带不同的序号** —— 否则 prefix cache 会让
        后续请求几乎是免费的前缀命中，测出来的"加速"里混着缓存。
        （社区基准的方法论：distinct prompts, prefix cache can't mask work）
        """
        import concurrent.futures as cf
        self.assert_single_resident()
        out = []
        for n in sizes:
            t0 = time.time()
            with cf.ThreadPoolExecutor(max_workers=n) as ex:
                futs = []
                for i in range(n):
                    body = {
                        "model": self.model_id,
                        "messages": [{"role": "user", "content":
                                      f"{prompt}\n\n（这是第 {i+1} 次请求，"
                                      f"请在输出里写明 # {i+1}）"}],
                        "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": self.think},
                        **SAMPLING[self.sampling_name],
                    }
                    futs.append(ex.submit(_post, "/v1/chat/completions", body))
                res = []
                for f in futs:
                    try:
                        res.append(f.result())
                    except Exception:            # noqa: BLE001
                        pass
            dt = time.time() - t0
            toks = sum((r.get("usage") or {}).get("completion_tokens", 0)
                       for r in res)
            out.append({
                "tier": self.tier, "n_requested": n, "n_ok": len(res),
                "wall_s": round(dt, 2), "total_tokens": toks,
                "aggregate_tps": round(toks / dt, 2) if dt else None,
                "per_request_tps": round(toks / dt / n, 2) if dt and n else None,
                # 部分失败必须显式暴露，不能当成"完成了"
                "all_ok": len(res) == n,
            })
            self._log(f"    n={n:<2} 聚合 {out[-1]['aggregate_tps']} tok/s  "
                      f"每请求 {out[-1]['per_request_tps']}  "
                      f"ok={len(res)}/{n}")
        return out
