#!/usr/bin/env python3
"""
router.py — pick sampling parameters from what the prompt is asking for.

Why this exists
---------------
The measurements in this repo say the right settings are not one setting:

* Acceptance rate spans 56.7% (creative prose) to 98.3% (repetitive code),
  and the depth-1 on/off A/B moves throughput from −14.6% to +23.6% across
  that range, monotonically.
* Temperature is not neutral either: it costs 12.7 pt of acceptance on
  translation at T=1.6, and past T≈2.0 the output collapses into multilingual
  token soup.

So a fixed temperature and a fixed depth is a compromise nobody chose. Routing
is the way to stop paying that compromise.

What it can and cannot route
----------------------------
Per-request (cheap, no reload):
    temperature, top_p, top_k, max_tokens, enable_thinking

NOT per-request:
    mtp_fixed_depth, mtp_enabled — these live in the model settings, and any
    settings write makes oMLX unload and reload the whole model. Measured 47 s
    on this box. `route()` returns them as *session* advice, not per-request
    settings. Do not try to route depth per request; it will cost you a
    model reload per call.

Honesty about the numbers
-------------------------
`alpha` ranges and the `measured` flags come from this repo's experiments.
The temperature values are conventional defaults, not a measured
quality-vs-temperature curve — measuring that needs a quality judge, which
this repo does not have. `measured: false` marks profiles where the α band is
inferred rather than observed.

Usage
-----
    import router
    r = router.route("帮我写个 Python 函数解析 CSV")
    r.params          # {'temperature': 0.3, 'top_p': 0.95, ...}
    r.session_advice  # {'mtp_enabled': True, 'recommended_depth': 1, ...}
    r.explain         # why

    # or just get the request body
    body = router.route(prompt).as_request_body(model="...")

Classification is keyword-based on purpose. An LLM-based router would cost a
generation to save a few milliseconds of tuning, and would be harder to debug
when it picks wrong. The signals below are visible, and `explain()` tells you
which one fired.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "router_profiles.json"
_TABLE = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
PROFILES = _TABLE["profiles"]

# Ordered most-specific first. First match wins, so structured output and code
# are checked before the vaguer categories, and "creative" is checked before
# the catch-all "general" but after anything that is clearly not prose.
_RULES: list[tuple[str, re.Pattern, str]] = [
    ("structured_json", re.compile(
        r"```json|```yaml|返回\s*json|输出\s*json|只要\s*json|转成\s*json|转换成\s*json|"
        r"json\s*(数组|对象|格式|schema)|"
        r"\bjson\b.{0,30}\b(list|array|object|schema|format)\b|"
        r"structured\s+output|"
        # csv/toml only in an *output* context. A bare "\bcsv\b" also fires on
        # "write a Python function that parses this CSV file", which is a code
        # task, and routing it to the low-temperature profile would hurt it.
        r"(输出|导出|转成|转成表格|保存成|返回|生成).{0,10}\b(csv|tsv|toml|md\s*表格)\b|"
        r"\b(csv|tsv|toml)\s*(格式|表头|表)\b|"
        r"^\s*[\[{].*[\]}]\s*$", re.I | re.M), "JSON / 表格 / schema 类请求"),
    ("code_repetitive", re.compile(
        # Chinese alternatives must not carry \b. In Python's Unicode \b, CJK
        # characters are word characters, so "批量加版权头" has no boundary
        # after 量 and the pattern silently never matched.
        r"重构|加注释|补注释|加版权头|文件头|生成\s*(测试|test)|单元测试|"
        r"批量|逐个|逐文件|遍历所有|全部文件|脚手架|模板化|"
        r"\b(refactor|add comments|docstring|boilerplate|template|scaffold|"
        r"unit tests?|every file|all files)\b", re.I), "代码块或明显的模板/批量/重构措辞"),
    ("code_function", re.compile(
        r"```\w*\n|写\s*(个|一个|一段|个)?\s*(函数|脚本|方法|类|程序|工具)|"
        r"\b(write|create|implement|generate|refactor|fix|debug|优化|修复|"
        r"实现|改写)\b.*\b(function|script|class|program|code|method|query|sql)\b|"
        r"\b(python|rust|typescript|javascript|golang|java|sql|shell|bash|regex)\b|"
        r"正则|算法|数据结构", re.I), "函数 / 脚本 / 具体语言实现"),
    ("math_reasoning", re.compile(
        r"证明|推导|求解|计算|数学|积分|微积分|概率|统计证明|"
        r"\b(prove|proof|derive|derive|theorem|lemma|solve for|calculate|"
        r"integral|derivative|probability that)\b|[∫∑√≤≥≠π]|"
        r"how many ways|多少次", re.I), "证明 / 推导 / 计算类"),
    ("translation", re.compile(
        r"翻译|译成|译文|中译英|英译中|translate|translation|"
        r"locali[sz]e|本地化", re.I), "翻译 / 本地化"),
    ("extraction", re.compile(
        r"提取|抽取|摘出|总结|概括|归纳|改写|润色|精简|整理成|转换成表格|"
        r"\b(extract|summari[sz]e|rewrite|condense|paraphrase|list (all|out)|"
        r"pull out|tl;?dr|bullet points?|key points?)\b", re.I), "抽取 / 总结 / 改写"),
    ("creative", re.compile(
        # "写一个关于深夜便利店的短篇散文" — the genre noun is usually several
        # modifiers away from 写, so allow a bounded gap rather than requiring
        # adjacency, which silently missed most real prose requests.
        r"写[^。？！\n]{0,25}?(故事|小说|散文|诗|剧本|文案|标题|标语|"
        r"博客|短文|文章|推文|段子|童谣)|"
        r"创作|头脑风暴|创意|想象|编一|角色设定|对话剧本|"
        r"\b(story|novel|poem|poetry|essay|blog post|headline|tagline|"
        r"screenplay|creative writing|brainstorm|come up with)\b", re.I),
     "创作类措辞"),
    ("qa_factual", re.compile(
        r"什么是|是什么|为什么|怎么理解|区别|对比|哪个更|是否|解释一下|"
        r"\b(what is|what are|why |how does|explain|difference between|"
        r"compare|which is|should i|is it)\b", re.I), "问答 / 解释 / 对比类"),
]

_DEFAULT = "general"


@dataclass
class Route:
    profile: str
    label: str
    signals: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    session_advice: dict = field(default_factory=dict)
    alpha_band: tuple[float, float] | None = None
    measured_alpha: float | None = None
    measured: bool = False
    why: str = ""

    @property
    def confidence(self) -> str:
        if not self.signals:
            return "default (no signal matched)"
        if len(self.signals) > 1:
            return f"weak (signals disagree: {', '.join(self.signals)})"
        return "strong"

    def explain(self) -> str:
        head = f"[{self.profile}] {self.label}  — {self.confidence}"
        body = [head, f"  matched: {', '.join(self.signals) or '(none → default)'}"]
        if self.alpha_band:
            lo, hi = self.alpha_band
            band = f"{lo}–{hi}%"
            body.append(f"  expected α: {band}"
                        + (f" (measured {self.measured_alpha}%)"
                           if self.measured_alpha is not None else " (inferred)"))
        body.append(f"  per-request: " + ", ".join(
            f"{k}={v}" for k, v in self.params.items()))
        body.append(f"  session-level: " + ", ".join(
            f"{k}={v}" for k, v in self.session_advice.items()))
        body.append(f"  why: {self.why}")
        return "\n".join(body)

    def as_request_body(self, model: str, max_tokens: int = 2048,
                        enable_thinking: bool = False) -> dict:
        """A drop-in /v1/chat/completions body with the routed parameters."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": ""}],   # caller fills in
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            **self.params,
        }


def _profile_fields(name: str) -> Route:
    p = PROFILES[name]
    band = (p["alpha_lo"], p["alpha_hi"])
    return Route(
        profile=name,
        label=p["label"],
        params={"temperature": p["temperature"], "top_p": p["top_p"],
                "top_k": p["top_k"]},
        session_advice={"mtp_enabled": p["mtp_enabled"],
                        "recommended_depth": p["recommended_depth"]},
        alpha_band=band,
        measured_alpha=p.get("measured_alpha"),
        measured=bool(p.get("measured")),
        why=p["why"],
    )


MAX_HEAD_CHARS = 300


def split_instruction(prompt: str) -> tuple[str, str]:
    """Separate the instruction from the document it applies to.

    Prompts that ship a payload ("extract the key points from these minutes:
    ...\\n\\n2026年3月12日，产品技术周会。…核心搜索模块重构已进入联调…") are
    the normal case, and matching the whole string routes on the *document's*
    vocabulary. That is how a task about extracting meeting minutes got filed
    under repetitive-code rewriting, purely because the minutes happened to
    mention 重构.

    So: prefer matches in the instruction, and only fall back to the full text
    when the instruction itself says nothing. `returns` (head, body).
    """
    text = prompt or ""
    head = text.split("\n\n", 1)[0][:MAX_HEAD_CHARS]
    return head, text


def classify(prompt: str) -> tuple[str, list[str]]:
    """Return (profile, signals_fired). All matches are reported, not just the
    first, so a disagreement shows up in `explain()` instead of hiding."""
    head, full = split_instruction(prompt)

    def hits(text):
        return [(name, desc) for name, rx, desc in _RULES if rx.search(text)]

    head_hits = hits(head)
    # Instruction wins. Falling back to the whole prompt only when the
    # instruction is silent keeps single-paragraph prompts working.
    found = head_hits or hits(full)
    if not found:
        return _DEFAULT, []
    return found[0][0], [d for _, d in found]


def route(prompt: str, last_profile: str | None = None) -> Route:
    """Classify `prompt` and return the parameters to use for it.

    `last_profile` enables stickiness: if the same profile repeats across
    turns in a conversation, the routing is more likely to be right than a
    single-turn classification, so the caller can use this to stabilise output
    style across a multi-turn session (lower variance than re-deciding each turn).
    """
    name, signals = classify(prompt)
    r = _profile_fields(name)
    r.signals = signals
    head, _ = split_instruction(prompt)
    if len(prompt or "") > len(head):
        r.signals = signals + [f"(matched in the instruction, not the payload)"]
    if last_profile and last_profile == name and name != _DEFAULT:
        r.signals = signals + [f"same as previous turn ({last_profile})"]
    return r


def recommend_depth(measured_alpha: float | None = None) -> dict:
    """Depth advice, optionally from an acceptance rate you actually measured.

    A single request's α is available ~1 second into generation: oMLX logs
    `accept=NNN/MMM` per request. Feeding it back here beats guessing, but
    remember the switch itself costs a 47 s reload, so use it at session
    boundaries, not per turn.
    """
    if measured_alpha is None:
        return {"recommended_depth": 1, "mtp_enabled": True,
                "reason": "no measurement — default to the safe end of the tied pair"}
    if measured_alpha >= 95:
        return {"recommended_depth": 2, "mtp_enabled": True,
                "reason": f"α={measured_alpha:.1f}% ≥95%: depth 2 measured +6.6~17.9% "
                          "over depth 1, and the switch measured +23.6% vs off"}
    if measured_alpha >= 75:
        return {"recommended_depth": 1, "mtp_enabled": True,
                "reason": f"α={measured_alpha:.1f}% in the 75–95% band: depth 1, "
                          "depth 2's margin sits inside the ±4.5% noise"}
    if measured_alpha >= 60:
        return {"recommended_depth": 1, "mtp_enabled": True,
                "reason": f"α={measured_alpha:.1f}%: keep MTP on (measured −2.6%, "
                          "inside noise) but do not deepen"}
    return {"recommended_depth": 1, "mtp_enabled": False,
            "reason": f"α={measured_alpha:.1f}% <60%: measured −14.6% from having "
                      "MTP on at all — switching it off beats any depth"}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(route(" ".join(sys.argv[1:])).explain())
    else:
        for demo in [
            "帮我写个 Python 函数解析 CSV 文件",
            "把这 30 个接口按响应时间排序，输出 JSON 数组",
            "翻译成英文：今天天气不错",
            "写一个关于深夜便利店的短篇散文",
            "为什么会下雨？用简单的话解释",
            "asdf qwer zxcv",
        ]:
            print(route(demo).explain())
            print()
