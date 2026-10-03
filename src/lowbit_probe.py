"""
低比特退化探针 (low-bit degradation probe)

为什么单独写一个模块
-------------------
现有的 `quality_checks.degeneracy()` 只查词级连续重复（same word back to back）。
它抓不到社区反复报告的三值/2-bit 失败模式：

  * "repetitive loop on Tamil"          句级循环，词级看不出来
  * "stuck in a self-correction loop"   推理段自我否定、来回横跳
  * "ran the same search 114 times"     动作不收敛
  * "reasoning shown, wrong conclusion" 推理流畅但结论错 —— 循环指标全绿，质量却崩了

所以这里做两件现有工具做不到的事：

  1. **强循环检测**（句子级 / n-gram 级 / 周期检测），能抓"看起来流畅但在兜圈"
  2. **终止性检测**（有没有真的给出答案，而不是停在半路）

设计原则（写死，不许后面偷偷改）
--------------------------------
  * 每道题都有**机器可验证的答案**，不靠人打分 —— 否则分不清
    "三值差" 和 "这题本来就不会"
  * 指标先在**已知坏样本**上证明会开火，在**已知好样本**上证明不会开火。
    没证明过的守卫等于没有守卫。
  * 这些指标全部是**无模型即可运行**的纯函数，先在自测里锁死。
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# 1. 强循环检测
# ---------------------------------------------------------------------------

# 句子切分：中英文日韩都覆盖
_SENT_SPLIT = re.compile(r"[。．！？!?；;\n]+")

# 归一化：去掉空白和标点，只留"内容词"——循环检测不能被格式差异干扰
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def _content_tokens(text: str) -> list[str]:
    """切成内容 token：连续的中日韩字符各自成 token，英文按词，数字按串。"""
    toks: list[str] = []
    for chunk in re.findall(r"[一-鿿぀-ヿ가-힯]|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?", text):
        toks.append(chunk.lower())
    return toks


def _norm_sentence(s: str) -> str:
    """句子归一化：只保留内容，用于判断"是不是同一句话又说了一遍"。"""
    return _PUNCT.sub("", s.lower())


def repetition_metrics(text: str) -> dict:
    """
    返回一组循环指标。所有阈值都在下面的自测里校准过。

    ngram_repeat_rate
        3-gram 里出现过 ≥2 次的比例。正常文本 <0.02；
        兜圈文本会 >0.15。这是最灵敏的单一指标。
    max_sentence_repeat
        同一句话（归一化后）最多重复几次。正常 =1。
        抓 "repetitive loop on Tamil" 这类。
    cycle_period
        检测到的最小循环周期 p（句子数）。若文本是 ABCABCABC 则 p=3。
        None = 没检测到。抓 "来回横跳" 这类。
    distinct2
        相邻 2-gram 去重率，沿用现有指标做纵向对比。
    max_token_run
        连续相同 token 的最大长度。
    """
    body = text.strip()
    toks = _content_tokens(body)
    n = len(toks)

    out = {
        "n_tokens": n,
        "ngram_repeat_rate": 0.0,
        "max_sentence_repeat": 1,
        "cycle_period": None,
        "distinct2": 1.0,
        "max_token_run": 0,
    }
    if n == 0:
        return out

    # --- 3-gram 重复率 -------------------------------------------------
    if n >= 3:
        grams = [tuple(toks[i:i + 3]) for i in range(n - 2)]
        seen: dict[tuple, int] = {}
        for g in grams:
            seen[g] = seen.get(g, 0) + 1
        dup = sum(c for c in seen.values() if c > 1)
        out["ngram_repeat_rate"] = round(dup / len(grams), 4)

    # --- 句子级重复 --------------------------------------------------
    sents_all = [s for s in (_norm_sentence(s) for s in _SENT_SPLIT.split(body)) if len(s) >= 4]
    if sents_all:
        counts: dict[str, int] = {}
        for s in sents_all:
            counts[s] = counts.get(s, 0) + 1
        out["max_sentence_repeat"] = max(counts.values())

    # --- 周期检测 ----------------------------------------------------
    # 句子级：靠句末标点切分，能抓散文/翻译里的兜圈
    sents = sents_all
    if len(sents) >= 6:
        p = _min_period(sents)
        if p and 2 <= p <= len(sents) // 2 and len(sents) // p >= 3:
            out["cycle_period"] = p

    # token 级：**代码输出通常没有句末标点**，只做句子级会对代码循环完全失明。
    # 所以再对 token 序列做一次短周期检测，周期要长到不可能是巧合。
    if out["cycle_period"] is None and n >= 12:
        p = _dominant_period(toks)
        if p:
            out["cycle_period"] = p

    # --- 2-gram 去重率 + token run ------------------------------------
    if n >= 2:
        grams2 = [tuple(toks[i:i + 2]) for i in range(n - 1)]
        out["distinct2"] = round(len(set(grams2)) / len(grams2), 4)
    run = best = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    out["max_token_run"] = best
    return out


def _min_period(seq: list[str]) -> int | None:
    """KMP 前缀函数求最小周期（要求整除）。返回 None 表示无周期。"""
    m = len(seq)
    pi = [0] * m
    for i in range(1, m):
        j = pi[i - 1]
        while j > 0 and seq[i] != seq[j]:
            j = pi[j - 1]
        if seq[i] == seq[j]:
            j += 1
        pi[i] = j
    p = m - pi[-1]
    return p if p < m and m % p == 0 else None


def _dominant_period(seq: list[str], max_p: int = 20, min_reps: int = 4) -> int | None:
    """
    对前缀鲁棒的周期检测：候选周期 p，看 seq[i] == seq[i-p] 的比例是否 >= 0.90。

    为什么不用 KMP 的精确周期：只要开头出现**一个**不属于循环的 token
    （比如 ```` ```python ```` 这个 fence），整除关系就断了，精确周期直接失效。
    而现实里带循环的输出几乎总带 fence/前言。所以改成"分块自洽度"——
    允许前缀少数几个位置不匹配。

    要求至少 min_reps 轮，避免把短文本里的偶然重复当成循环。
    """
    n = len(seq)
    if n < 4 * min_reps:
        return None
    for p in range(2, min(max_p, n // min_reps) + 1):
        total = n - p
        if total < 3 * p:
            continue
        agree = sum(1 for i in range(p, n) if seq[i] == seq[i - p])
        if agree / total >= 0.90:
            return p
    return None


# ---------------------------------------------------------------------------
# 2. 终止性 / 是否真的交卷
# ---------------------------------------------------------------------------

# 「我改主意了 / 再想想 / 或者」这类自我纠错标记
_HESITATE = re.compile(
    r"(等等|不对|重来|再想想|让我再|或者|要不|其实不对|抱歉|重新开始|"
    r"wait|hmm|actually,? no|hold on|let me reconsider|on second thought|"
    r"or maybe|instead,?)",
    re.IGNORECASE,
)

# 明确的未完成/截断标记
_UNFINISHED = re.compile(
    r"(未完待续|待续|以下省略|此处省略|截断|TODO|继续写|还需要|让我继续|"
    r"\.\.\.\s*$|<!--)",
    re.IGNORECASE,
)


def termination_metrics(text: str) -> dict:
    """
    是否真的交卷了。社区说的 "400 token 仍在自我否定" 会被这里抓住。

    hesitate_rate   自我纠错标记密度（每 100 token）。正常中文技术写作 <1.0
    last_sentence_cut_off  末句是否残缺（不以句末标点/代码/结束括号收尾）
    committed       末段是否含明确的结论性结构（代码完整、答案行、句号收尾）
    """
    body = text.strip()
    toks = _content_tokens(body)
    n = len(toks)
    hes = len(_HESITATE.findall(body))
    out = {
        "n_tokens": n,
        "hesitate_rate": 0.0,
        "hesitate_hits": hes,
        "last_sentence_cut_off": False,
        "unfinished_markers": len(_UNFINISHED.findall(body)),
    }
    if n:
        out["hesitate_rate"] = round(hes / n * 100, 2)

    if body:
        tail = body.rstrip()[-1:]
        # 结束于中英文句号、问号、感叹号，或代码/结构收尾。
        # 数字结尾也要放过：L1 这类"只输出最终数字"的题，正确答案就是
        # "2773.3" 这种以数字收尾的串，第一版会把它误判成截断。
        ok_tail = (tail in "。．.!?！？）)]}）0123456789"
                   or body.endswith("```"))
        out["last_sentence_cut_off"] = not ok_tail
    return out


# ---------------------------------------------------------------------------
# 3. 合并视图：一句话判定
# ---------------------------------------------------------------------------

# 阈值来源：不是拍脑袋，是对 7 个 fixture 的真实指标标定的（见 self_check 末尾的矩阵）。
#
# 标定表（实测值）：
#   fixture                       ngram  sentRep  cyc  tokRun  hesitate
#   BAD_LOOP   散文兜圈            1.000      3     3     2      9.52
#   BAD_CYCLE  ABCABC              1.000      1     3     1      0.00
#   BAD_CODE   代码块重复          0.971      6     6     1      0.00
#   BAD_HES    自我纠错卡死        0.113      1   None    2     13.70
#   GOOD_CODE  正常代码            0.000      1   None    1      0.00
#   GOOD_REP   合法同构子类代码    0.283      4   None    2      0.00   <-- 合法重复的天花板
#   GOOD_MULTI 正常多语言          0.000      1   None    1      0.00
#
# 三个强信号（好坏间隔 >3 倍）：
#   ngram_repeat_rate : 合法最高 0.283 / 最差坏样本 0.971 -> 0.55 两侧各留 ~1.8 倍余量
#   cycle_period      : 三个坏样本全部检出，三个好样本全部 None -> 检出即可判
#   hesitate_rate     : 合法最高 0.00 / 最低坏样本 9.52 -> 4.0 余量极大
#
# 一个弱信号（诚实标注，不要当主判据）：
#   max_sentence_repeat : 合法代码能到 4，而 BAD_LOOP 只有 3 —— **无鉴别力**。
#   它只对"同一整句被重复 8 次以上"这种极端情况有意义，故阈值定到 8，仅作旁证。
THRESHOLDS = {
    "ngram_repeat_rate": 0.55,
    "max_sentence_repeat": 8,      # 弱信号，仅旁证
    "cycle_max_period": 24,        # 检出即循环；超过 24 token 一轮的多半是合法结构
    "max_token_run": 12,           # 病态 "aaaa..."；本批 fixture 未触发，属兜底
    "hesitate_rate": 4.0,
    "unfinished_markers": 2,
}


def loop_flags(text: str) -> dict:
    """把原始指标变成布尔判定，便于跨档位统计。"""
    rep = repetition_metrics(text)
    ter = termination_metrics(text)
    flags = {
        "loop_ngram": rep["ngram_repeat_rate"] >= THRESHOLDS["ngram_repeat_rate"],
        "loop_sentence": rep["max_sentence_repeat"] >= THRESHOLDS["max_sentence_repeat"],
        "loop_cycle": (rep["cycle_period"] is not None
                       and rep["cycle_period"] <= THRESHOLDS["cycle_max_period"]),
        "loop_token_run": rep["max_token_run"] >= THRESHOLDS["max_token_run"],
        "hesitate": ter["hesitate_rate"] >= THRESHOLDS["hesitate_rate"],
        "unfinished": ter["unfinished_markers"] >= THRESHOLDS["unfinished_markers"],
    }
    flags["any_loop"] = any(
        flags[k] for k in ("loop_ngram", "loop_sentence", "loop_cycle", "loop_token_run")
    )
    flags["any_degenerate"] = flags["any_loop"] or flags["hesitate"] or flags["unfinished"]
    flags["_raw"] = {"rep": rep, "ter": ter}
    return flags


# ---------------------------------------------------------------------------
# 4. 任务集：每题都有机器可验证答案
# ---------------------------------------------------------------------------

# (key, 熵层级, prompt, max_tokens, verify_kind, verify_arg)
L_TASKS = [
    (
        "L1_chain_math",
        "低熵·长链数值",
        # 设计说明（踩过坑才写成这样）：
        # 第一版要求"只输出最终数字"，thinking 又是关的 —— 等于让模型
        # **一次前向直接吐出算好的数，没有草稿纸**。实测 ternary/4bit/6bit
        # 分别给 1087.9 / 1083.9 / 1053.9：三档都错、且错得不一样。
        # 那种"都错且错法各异"是一次心算的正常方差，**不是量化信号**，
        # 拿来对比位宽会得出完全错误的结论。
        #
        # 现在要求先写中间步骤：既让算术可做（几行 token 就够），
        # 又让**错在哪一步可归因** —— 这比只看最终数字有用得多。
        "分三行回答，不要 markdown，不要任何其它文字。\n"
        "第一行：第 1 到 8 天每天的总耗油量是多少升？\n"
        "第二行：第 8 天结束后换成 15 辆柴油、18 辆汽油，之后每天的总耗油量是多少升？\n"
        "第三行：11 天一共多少升？\n\n"
        "车队有 37 辆车，其中 12 辆柴油、25 辆汽油。柴油车每天耗油 8.5 升，"
        "汽油车每天耗油 6.2 升。车队连续跑 11 天，第 8 天结束时 7 辆汽油车报废、"
        "换成 3 辆新柴油车（第 9 天起是 15 柴油 18 汽油）。",
        160,
        "numeric",
        2773.3,
    ),
    (
        "L2_bugfix",
        "中熵·调试",
        "只输出修复后的完整 Python 代码，不要 markdown 标记，不要解释。\n\n"
        "下面这个函数本应返回两个列表中较短的那个，但有 bug。请修复它，"
        "并保持函数名和签名不变：\n\n"
        "def shorter(a, b):\n"
        "    if len(a) < len(b):\n"
        "        return a\n"
        "    return a\n\n"
        "要求：空列表时也能正确工作。",
        300,
        "exec_shorter",
        None,
    ),
    (
        "L3_multilingual",
        "中熵·多语言",
        "依次输出四行译文，每行一种语言，不要编号，不要解释，不要 markdown。\n"
        "第一行：简体中文\n第二行：日本語\n第三行：한국어\n第四行：ไทย\n\n"
        "原文：The library was designed to be thread-safe without external locks.",
        200,
        "multilingual",
        None,
    ),
    (
        "L4_verbatim_copy",
        "低熵·精确复述",
        "只输出被引号包住的那一行，不要任何其他文字，不要 markdown。\n\n"
        "以下是配置文件的片段：\n\n"
        "server {\n"
        "    listen 8443;\n"
        "    max_connections = 4096;\n"
        "    backlog = 2048;\n"
        "    keepalive_timeout = 75;\n"
        "}\n"
        "log {\n"
        "    destination = syslog;\n"
        "    level = warn;\n"
        "}\n\n"
        "请把含有数字 4096 的那一行原文完整复述出来。",
        64,
        "verbatim",
        "max_connections = 4096;",
    ),
    (
        "L5_negation",
        "中熵·否定推理",
        "只输出 True 或 False，不要任何其他文字。\n\n"
        "以下陈述是否为真？\n\n"
        "有 8 个事件，每个事件至少发生一次，且每个事件恰好发生 2 次。"
        "因此总事件数是 17。\n\n"
        "陈述：总事件数是 17。",
        8,
        "exact_text",
        "False",
    ),
    (
        "L6_json_strict",
        "极低熵·严格结构",
        "严格输出一个 JSON 对象，不要 markdown 标记，不要任何解释文字。\n"
        "对象包含 keys 数组（3 个字符串）和 count 字段（整数），"
        "keys 必须是 [\"alpha\",\"beta\",\"gamma\"]，count 必须是这三个 key 的长度。",
        128,
        "json_exact",
        {"keys": ["alpha", "beta", "gamma"], "count": 3},
    ),
    (
        "L7_code_exec",
        "低熵·可执行",
        "只输出 Python 代码，不要 markdown 标记，不要解释。\n"
        "写一个函数 running_max(nums)，返回一个新列表，其中第 i 个元素是 nums[:i+1] 里的最大值。"
        "例如 running_max([1, 3, 2, 4, 2, 5]) 应返回 [1, 3, 3, 4, 4, 5]。空列表返回空列表。",
        260,
        "exec_running_max",
        None,
    ),
    (
        "L8_long_extract",
        "中熵·长文定位",
        # 第一版问"「蓝湾计划」的代号是什么"，但题干里已经写了"代号为「蓝湾计划」"
        # —— 自相矛盾。模型答"蓝湾计划"是合理理解，被我的校验判成错。
        # 改问它**做什么**，答案在原文里且不与题干重复。
        "阅读下面这段文字，只用一行回答：「蓝湾计划」具体要做的是什么工作？"
        "不要解释，不要 markdown。\n\n"
        "第一季度复盘。我们启动了代号为「蓝湾计划」的数据迁移工作，"
        "由基础设施组牵头。同期「磐石项目」进入灰度阶段，覆盖三个区域。"
        "「蓝湾计划」的首批迁移窗口定在四月中旬，预计涉及 1200 个服务。"
        "需要注意的是，「磐石项目」与「蓝湾计划」共享同一个回滚预案，"
        "但执行顺序不同。评审后决定「磐石项目」先行，「蓝湾计划」顺延一周。",
        48,
        "extract",
        "数据迁移",
    ),
]

L_TASK_KEYS = [t[0] for t in L_TASKS]


# ---------------------------------------------------------------------------
# 5. 逐题验证器
# ---------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    m = re.search(r"```(?:python|json)?\s*\n(.*?)(?:```|$)", text, re.DOTALL)
    return m.group(1) if m else text


def _first_json(text: str) -> dict | None:
    body = _strip_fence(text)
    start = body.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(body[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1])
                except Exception:
                    return None
    return None


import json  # noqa: E402,F811  (顶层已 import；这里保留仅为兼容旧的执行顺序)


def _num_in(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


# 目标脚本的运行沙箱（离线、纯 stdlib）
_SANDBOX = r"""
import json, sys
def check(fn_src, tests, fn_name):
    ns = {}
    try:
        exec(fn_src, ns)
    except Exception as e:
        return ["load_fail", repr(e)]
    fn = None
    if fn_name:
        cand = ns.get(fn_name)
        if callable(cand):
            fn = cand
    if fn is None:
        cands = [v for k, v in ns.items()
                 if callable(v) and not k.startswith('_')
                 and getattr(v, '__module__', None) != 'builtins']
        if not cands:
            return ["load_fail", "no callable" + (" named " + fn_name if fn_name else "")]
        fn = cands[0]
    for args, want in tests:
        try:
            got = fn(*args)
        except Exception as e:
            return ["raise", "%r on %r" % (e, args)]
        if got != want:
            return ["wrong", "%r: got %r want %r" % (args, got, want)]
    return ["ok", ""]
"""


def _run_py_checks(code: str, cases: list[tuple[tuple, object]],
                   fn_name: str | None = None) -> tuple[bool, str]:
    """
    离线执行校验：把模型写的代码塞进沙箱跑固定用例。

    fn_name 优先按名字取函数；取不到才退化成"随便挑一个可调用对象"。
    不做"任意一个能过就算过"——那会放过写了个对的 helper 但主函数是错的情况。
    """
    import subprocess
    import sys as _sys
    script = _SANDBOX + f"\nprint(json.dumps(check({code!r}, {cases!r}, {fn_name!r})))\n"
    try:
        r = subprocess.run(
            [_sys.executable, "-c", script],
            capture_output=True, text=True, timeout=20,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if r.returncode != 0:
        return False, f"runner error: {r.stderr.strip()[:160]}"
    try:
        kind, detail = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return False, f"unparseable: {r.stdout[:120]}"
    return kind == "ok", f"{kind}: {detail}"


def _script_language_ok(text: str, script: str) -> tuple[bool, str]:
    """非拉丁文字脚本：确认真的输出了目标语言，且不是同一串字符反复刷。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = [l for l in lines if script in l]
    if not hits:
        return False, f"no {script} line found"
    rep = repetition_metrics("\n".join(hits))
    if rep["max_sentence_repeat"] >= 3 or rep["ngram_repeat_rate"] > 0.30:
        return False, f"{script} line is looping ({rep['ngram_repeat_rate']})"
    return True, f"{script} ok"


def verify(task_key: str, text: str) -> tuple[bool, str]:
    """返回 (是否通过, 说明)。只判这道题自己问的东西。"""
    body = _strip_fence(text).strip()

    if task_key == "L1_chain_math":
        # 现在文本里有三个数字（每天-旧、每天-新、总计），
        # 必须取**最后一个**（总计行），不能取第一个。
        want = 2773.3
        nums = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", body.replace(",", ""))]
        if not nums:
            return False, "no number found"
        got = nums[-1]
        # 中间步骤也校验：错在哪一步比只知对错有用
        d_old, d_new = 257.0, 239.1
        s_old = any(abs(n - d_old) < 0.05 for n in nums[:-1])
        s_new = any(abs(n - d_new) < 0.05 for n in nums[:-1])
        trace = f"total={got} daily_old_ok={s_old} daily_new_ok={s_new} nums={nums[:4]}"
        return abs(got - want) < 0.05, f"{trace} want={want}"

    if task_key == "L2_bugfix":
        # 修复版对每一行：len 相同返回 a（任意一个），len 不同返回短的
        cases = [(([], [1]), []), (([1, 2], [1, 2, 3]), [1, 2]),
                 (([1, 2, 3], [1]), [1]), (([1], []), [])]
        return _run_py_checks(body, cases, fn_name="shorter")

    if task_key == "L3_multilingual":
        checks = [
            ("zh", _cjk_ok), ("ja", _kana_ok),
            ("ko", _hangul_ok), ("th", _thai_ok),
        ]
        fails = [n for n, fn in checks if not fn(body)]
        return not fails, "all 4 ok" if not fails else f"missing/wrong: {fails}"

    if task_key == "L4_verbatim_copy":
        want = "max_connections = 4096;"
        norm = _PUNCT.sub("", body).lower()
        if _PUNCT.sub("", want).lower() not in norm:
            return False, f"verbatim not found: {body[:80]!r}"
        return True, "verbatim ok"

    if task_key == "L5_negation":
        return body.strip().rstrip(".").lower() == "false", f"got {body[:40]!r}"

    if task_key == "L6_json_strict":
        got = _first_json(text)
        if got is None:
            return False, "no parseable JSON"
        if got.get("keys") != ["alpha", "beta", "gamma"]:
            return False, f"keys={got.get('keys')!r}"
        if got.get("count") != 3:
            return False, f"count={got.get('count')!r}"
        return True, "exact"

    if task_key == "L7_code_exec":
        cases = [(([1, 3, 2, 4, 2, 5],), [1, 3, 3, 4, 4, 5]),
                 (([],), []), (([7],), [7]), (([5, 4, 3],), [5, 5, 5]),
                 (([-1, -5, -3],), [-1, -1, -1])]
        return _run_py_checks(body, cases, fn_name="running_max")

    if task_key == "L8_long_extract":
        want = "数据迁移"
        return want in body, f"got {body[:60]!r}"

    raise KeyError(f"unknown task {task_key!r}")


# --- 脚本/语言检测小工具 ---------------------------------------------------

def _cjk_ok(t: str) -> bool:
    return len(re.findall(r"[一-鿿]", t)) >= 5


def _kana_ok(t: str) -> bool:
    return len(re.findall(r"[぀-ヿ]", t)) >= 5


def _hangul_ok(t: str) -> bool:
    return len(re.findall(r"[가-힯]", t)) >= 5


def _thai_ok(t: str) -> bool:
    return len(re.findall(r"[฀-๿]", t)) >= 5


# ---------------------------------------------------------------------------
# 6. 自测：证明每个指标都会开火，且不会误伤
# ---------------------------------------------------------------------------

# 坏样本 1：句级循环（模拟 "Tamil 上死循环"）
BAD_LOOP = (
    "让我再想想这个翻译。\n"
    "泰语翻译是：ไลบรารีนี้ออกแบบมาให้ปลอดภัยจากเธรด\n"
    "不对，我再试一次。\n"
    "让我再想想这个翻译。\n"
    "泰语翻译是：ไลบรารีนี้ออกแบบมาให้ปลอดภัยจากเธรด\n"
    "不对，我再试一次。\n"
    "让我再想想这个翻译。\n"
    "泰语翻译是：ไลบรารีนี้ออกแบบมาให้ปลอดภัยจากเธรด\n"
    "不对，我再试一次。\n"
)

# 坏样本 2：ABCABC 周期（模拟 "来回横跳"）
BAD_CYCLE = "A B C A B C A B C A B C A B C A B C A B C A B C A B C"

# 坏样本 3：自我纠错卡死（模拟 Ollama 1-bit 实测）
BAD_HESITATE = (
    "我需要先检查这个函数。等一下，让我再看看。\n"
    "不对，这个实现有问题。或者用另一种方式。\n"
    "其实不对，重新开始。让我再试一次。\n"
    "等等，我犯了个错误。再想想。\n"
    "或者干脆重写。抱歉，我再试一次。\n"
)

# 坏样本 4：截断
BAD_TRUNCATED = "首先我们创建一个字典，然后我们遍历这个字"

# 坏样本 5：代码循环（没有句末标点，句子级检测对它完全失明）
BAD_CODE_LOOP = """```python
def f(x):
    return x + 1

def f(x):
    return x + 1

def f(x):
    return x + 1

def f(x):
    return x + 1

def f(x):
    return x + 1

def f(x):
    return x + 1
```"""

# 好样本 1：正常技术中文，含代码
GOOD_CODE = """```python
def running_max(nums):
    out = []
    best = None
    for n in nums:
        best = n if best is None else max(best, n)
        out.append(best)
    return out
```

这个实现用一次遍历完成，best 保存到当前为止的最大值。"""

# 好样本 2：正常多语言输出
GOOD_MULTILINGUAL = (
    "这个库被设计为无需外部锁即可保证线程安全。\n"
    "このライブラリは、外部ロックなしでスレッドセーフに動作するよう設計されています。\n"
    "이 라이브러리는 외부 잠금 없이 스레드 안전하도록 설계되었습니다.\n"
    "ไลบรารีนี้ได้รับการออกแบบให้เกิดขึ้นพร้อมกันได้อย่างปลอดภัย"
)


# 坏样本 6：明写"未完待续"/省略（模拟输出被 max_tokens 截断前的自述）
BAD_UNFINISHED = (
    "这个函数首先初始化了一个缓存字典，然后遍历输入序列，"
    "接着把每一项放入缓存并更新统计信息，之后我们还需要处理边界情况，"
    "未完待续。"
    "以下省略剩下的三个分支，TODO: 补全。"
    "让我继续写剩下的部分。"
)


# 好样本 3：**合法**的结构重复 —— 三个同构子类。检测器绝不能把它当循环。
GOOD_REPETITIVE_CODE = """```python
class Shape:
    \"\"\"Base shape.\"\"\"
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    \"\"\"A circle defined by its radius.\"\"\"
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    \"\"\"A rectangle defined by width and height.\"\"\"
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class Triangle(Shape):
    \"\"\"A triangle defined by base and height.\"\"\"
    def __init__(self, base, height):
        self.base = base
        self.height = height
    def area(self):
        return self.base * self.height / 2
```"""


FIXTURES = [
    ("BAD_LOOP (prose loop)", BAD_LOOP),
    ("BAD_CYCLE (ABCABC)", BAD_CYCLE),
    ("BAD_HESITATE (self-doubt)", BAD_HESITATE),
    ("BAD_CODE_LOOP (code loop)", BAD_CODE_LOOP),
    ("BAD_UNFINISHED (truncated)", BAD_UNFINISHED),
    ("GOOD_CODE (plain code)", GOOD_CODE),
    ("GOOD_REPETITIVE (legit rep.)", GOOD_REPETITIVE_CODE),
    ("GOOD_MULTILINGUAL", GOOD_MULTILINGUAL),
]


def self_check(verbose: bool = True) -> int:
    fails: list[str] = []

    def ck(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            if verbose:
                print(f"  ok   {name}")
        else:
            fails.append(f"{name} {detail}")
            print(f"  FAIL {name} {detail}")

    # --- 坏样本必须开火 -------------------------------------------------
    f = loop_flags(BAD_LOOP)
    ck("loop_sample: ngram fires", f["loop_ngram"], f)
    ck("loop_sample: cycle fires", f["loop_cycle"], f)
    ck("loop_sample: hesitate fires", f["hesitate"], f)
    ck("loop_sample: any_loop", f["any_loop"], f)
    # 诚实断言：sentRep 是**弱信号**，阈值定到 8 之后它对 BAD_LOOP 不开火，
    # 因为合法同构代码能到 4 而真循环只有 3 —— 它本来就没有鉴别力。
    # 这里锁死这个事实，防止后人误以为它在起作用而单独依赖它。
    ck("loop_sample: sentRep 弱信号不开火（设计如此）",
       f["loop_sentence"] is False, f["_raw"]["rep"])
    ck("loop_sample: 无 sentRep 仍被抓住",
       f["any_loop"] and not f["loop_sentence"], f)

    f = loop_flags(BAD_CYCLE)
    ck("cycle_sample: cycle_period==3",
       f["_raw"]["rep"]["cycle_period"] == 3, f["_raw"]["rep"])
    ck("cycle_sample: any_loop", f["any_loop"], f)

    f = loop_flags(BAD_HESITATE)
    ck("hesitate_sample: fires", f["hesitate"], f["_raw"]["ter"])

    f = loop_flags(BAD_TRUNCATED)
    ck("truncated_sample: cut_off detected",
       f["_raw"]["ter"]["last_sentence_cut_off"], f["_raw"]["ter"])

    # 纯数字答案不能被误判成截断（L1 的正确输出就是 "2773.3" 这种）
    for num in ("2773.3", "1087.9", "42", "0.5"):
        t = termination_metrics(num)
        ck(f"numeric answer not cut_off: {num}", not t["last_sentence_cut_off"], t)

    f = loop_flags(BAD_CODE_LOOP)
    ck("code_loop: ngram fires", f["loop_ngram"], f["_raw"]["rep"])
    ck("code_loop: cycle fires", f["loop_cycle"], f["_raw"]["rep"])
    ck("code_loop: any_loop", f["any_loop"], f["_raw"]["rep"])

    f = loop_flags(BAD_UNFINISHED)
    ck("unfinished_sample: fires", f["unfinished"], f["_raw"]["ter"])

    # --- 好样本必须不误伤 ------------------------------------------------
    f = loop_flags(GOOD_CODE)
    ck("good_code: no loop", not f["any_loop"], f["_raw"]["rep"])
    ck("good_code: no hesitate", not f["hesitate"], f["_raw"]["ter"])
    ck("good_code: not truncated",
       not f["_raw"]["ter"]["last_sentence_cut_off"], f["_raw"]["ter"])

    f = loop_flags(GOOD_MULTILINGUAL)
    ck("good_multilingual: no loop", not f["any_loop"], f["_raw"]["rep"])

    f = loop_flags(GOOD_REPETITIVE_CODE)
    ck("good_repetitive_code: no cycle (合法结构重复不误伤)",
       f["_raw"]["rep"]["cycle_period"] is None, f["_raw"]["rep"])
    ck("good_repetitive_code: no loop", not f["any_loop"], f["_raw"]["rep"])
    ck("good_repetitive_code: no hesitate", not f["hesitate"], f["_raw"]["ter"])

    # --- 阈值标定：好坏两侧都要有 >=1.5 倍余量 -----------------------------
    # 用 GOOD_REPETITIVE_CODE（合法结构重复）当"好"的天花板，不是 GOOD_CODE。
    good_rate = loop_flags(GOOD_REPETITIVE_CODE)["_raw"]["rep"]["ngram_repeat_rate"]
    bad_rate = loop_flags(BAD_CODE_LOOP)["_raw"]["rep"]["ngram_repeat_rate"]
    thr = THRESHOLDS["ngram_repeat_rate"]
    ck("margin: ngram bad >= 1.5x thr", bad_rate >= thr * 1.5, f"bad={bad_rate} thr={thr}")
    ck("margin: ngram good <= 0.6x thr", good_rate <= thr * 0.6, f"good={good_rate} thr={thr}")
    # 剩下三个坏样本也必须过线（ngram 对自我纠错无效是已知局限，故只查另外两个）
    for nm, s in (("BAD_LOOP", BAD_LOOP), ("BAD_CYCLE", BAD_CYCLE)):
        r = loop_flags(s)["_raw"]["rep"]["ngram_repeat_rate"]
        ck(f"margin: {nm} ngram >= 1.5x thr", r >= thr * 1.5, f"{r} vs {thr}")

    hgood = loop_flags(GOOD_REPETITIVE_CODE)["_raw"]["ter"]["hesitate_rate"]
    hbad = loop_flags(BAD_HESITATE)["_raw"]["ter"]["hesitate_rate"]
    ck("margin: hesitate bad >> thr",
       hbad >= THRESHOLDS["hesitate_rate"] * 2, f"{hbad}")
    ck("margin: hesitate good << thr", hgood <= 0.0, f"{hgood}")

    # --- KMP 最小周期单测 ------------------------------------------------
    ck("kmp: ABCABC -> 3", _min_period(list("ABCABC")) == 3)
    ck("kmp: AAAA -> 1", _min_period(list("AAAA")) == 1)
    ck("kmp: ABCD -> None", _min_period(list("ABCD")) is None)
    ck("kmp: ABCABCX -> None", _min_period(list("ABCABCX")) is None)

    # --- 验证器必须能区分对错 -------------------------------------------
    ok, why = verify("L5_negation", "False")
    ck("verify L5 good", ok, why)
    ok, _ = verify("L5_negation", "True")
    ck("verify L5 bad", not ok)

    ok, why = verify("L1_chain_math", "257\n239.1\n2773.3")
    ck("verify L1 good", ok, why)
    ok, _ = verify("L1_chain_math", "257\n239.1\n2600.0")
    ck("verify L1 bad (total wrong)", not ok)
    # 总数对就判过 —— 中间步骤是**归因数据**不是判据（题目问的是总耗油量）。
    # 但错中间 + 对总数必须在 trace 里看得见，否则无法定位推理链在哪断的。
    ok, why = verify("L1_chain_math", "999\n888\n2773.3")
    ck("verify L1: total right -> pass (中间只作归因)", ok, why)
    ck("verify L1: 但 trace 标出中间步骤错",
       "daily_old_ok=False" in why and "daily_new_ok=False" in why, why)
    # 常见错法：忘了一次换车
    ok, _ = verify("L1_chain_math", "257\n257\n2827.0")
    ck("verify L1 catches missing swap", not ok)

    ok, why = verify("L7_code_exec", """
def running_max(nums):
    out, best = [], None
    for n in nums:
        best = n if best is None else max(best, n)
        out.append(best)
    return out
""")
    ck("verify L7 good", ok, why)
    ok, _ = verify("L7_code_exec", "def running_max(nums):\n    return nums\n")
    ck("verify L7 bad (returns input)", not ok)

    ok, why = verify("L3_multilingual", GOOD_MULTILINGUAL)
    ck("verify L3 good", ok, why)
    ok, _ = verify("L3_multilingual", "This library is thread safe.\nDone.\nOkay.\nFine.\n")
    ck("verify L3 bad (no target script)", not ok)

    ok, why = verify("L4_verbatim_copy", '`max_connections = 4096;`')
    ck("verify L4 good", ok, why)
    ok, _ = verify("L4_verbatim_copy", "max_connections = 4096;")
    ck("verify L4 good (unquoted)", ok, why)
    ok, _ = verify("L4_verbatim_copy", "backlog = 2048;")
    ck("verify L4 bad", not ok)

    ok, why = verify("L6_json_strict", '```json\n{"keys":["alpha","beta","gamma"],"count":3}\n```')
    ck("verify L6 good", ok, why)
    ok, _ = verify("L6_json_strict", '{"keys":["alpha","beta"],"count":2}')
    ck("verify L6 bad", not ok)

    ok, why = verify("L8_long_extract", "数据迁移")
    ck("verify L8 good", ok, why)
    ok, _ = verify("L8_long_extract", "磐石项目")
    ck("verify L8 bad (wrong program)", not ok)
    # 自相矛盾的旧问法（"代号是什么"而题干已给出代号）现在应当判失败，
    # 锁住这个事实：模型答"蓝湾计划"是合理理解，不能算它错。
    ok, _ = verify("L8_long_extract", "蓝湾计划")
    ck("verify L8 rejects '蓝湾计划' (不是答案)", not ok)

    # --- 任务集自洽性 ---------------------------------------------------
    ck("tasks: 8 unique keys", len(set(L_TASK_KEYS)) == 8, str(L_TASK_KEYS))
    ck("tasks: all have prompts", all(t[2].strip() for t in L_TASKS))
    ck("tasks: all have max_tokens", all(isinstance(t[3], int) and t[3] >= 8 for t in L_TASKS))
    for t in L_TASKS:
        try:
            ck(f"verify exists: {t[0]}", callable(verify))
            verify(t[0], "")
        except KeyError:
            ck(f"verify wired: {t[0]}", False, "no verifier")

    print()
    # 标定矩阵原样打出来，方便后人核对阈值不是拍脑袋
    print("calibration matrix (real measured values)")
    print(f"  {'fixture':30s} {'ngram':>7s} {'sentRep':>7s} {'cyc':>5s} {'tokRun':>6s} {'hes':>6s} {'unfin':>5s}")
    for nm, s in FIXTURES:
        fl = loop_flags(s)
        r, t = fl["_raw"]["rep"], fl["_raw"]["ter"]
        print(f"  {nm:30s} {r['ngram_repeat_rate']:7.3f} {r['max_sentence_repeat']:7d} "
              f"{str(r['cycle_period']):>5s} {r['max_token_run']:6d} "
              f"{t['hesitate_rate']:6.2f} {t['unfinished_markers']:5d}")
    print()
    if fails:
        print(f"LOWBIT PROBE SELF-CHECK: {len(fails)} FAILED")
        for x in fails:
            print("  -", x)
        return 1
    print("LOWBIT PROBE SELF-CHECK: all green")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
