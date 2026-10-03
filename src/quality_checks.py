#!/usr/bin/env python3
"""
quality_checks.py — objective validators for the temperature × quality experiment.

A temperature sweep needs a quality signal. The obvious answer is an LLM judge,
which here would mean either an external API or grading a 125B model's prose
with itself — neither is trustworthy, and both are slow.

Most of this task set does not need a judge. Each of these tasks has a
checkable property that the prompt itself specifies:

    T1  "严格输出一个 JSON 数组…12 个对象，每个对象有 id/name/price/category"
    T2  "写一个名为 LRUCache 的类，要求：…"
    T3  "写一个基类 Shape，含 area() 和 describe()"
    T6  "…分点说明：为什么不是两次，为什么不是四次；ISN（初始序列号）的作用是…"
    T7  "…计算每一步的金额"           → the final number is computable
    T10 "实现一个 React 函数组件 …"
    T8  "…有至少两处比喻，不要重复用字"  ← literally checkable

A validator only has to answer "does this output satisfy the stated
requirement", which is a much smaller and more honest job than "is this good".

What is NOT measured: whether creative prose is *good*. Degeneracy metrics can
catch a collapse but cannot tell you 0.8 is nicer than 0.4. The experiment
reports that gap rather than papering over it — see `prose_quality_measured`.
"""
import ast
import json
import re

# --- the gold answer for T7, computed by hand -------------------------------
# 200 → ×0.8 = 160 → −30 = 130 → ×0.85 = 110.5 → −20 = 90.5
T7_GOLD = 90.5
T7_INTERMEDIATES = {200, 160, 130, 110.5, 90.5}


def _strip_fence(text: str) -> str:
    """Remove a leading ```lang fence and any trailing fence.

    The prompt says "不要 markdown 标记", so a fence is already a spec
    violation — but it is a *presentation* violation, and counting it as a
    total failure would hide whether the content behind it was right.
    """
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)(?:\n```|\Z)", t, re.S)
    return m.group(1) if m else t


def _fenced(text: str) -> bool:
    return bool(re.match(r"^\s*```", text))


# --- T1: JSON array of 12 objects with fixed keys ---------------------------
# The prompt specifies six fields, not four. Getting this wrong makes the
# validator accept outputs that violate the spec.
T1_FIELDS = {
    "id": int, "name": str, "price": (int, float),
    "category": str, "inStock": bool, "tags": list,
}


def check_json_12(text: str) -> tuple[bool, str]:
    body = _strip_fence(text)
    start = body.find("[")
    if start < 0:
        return False, "no array found"
    # Trim anything the model added after the array.
    depth, end = 0, None
    for i, ch in enumerate(body[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        # Almost always a truncated generation, not a temperature effect. The
        # caller should confirm with completion_tokens vs max_tokens.
        return False, "array never closed — output truncated before the ]"
    try:
        arr = json.loads(body[start:end])
    except Exception as e:
        return False, f"json parse: {type(e).__name__}"
    if not isinstance(arr, list):
        return False, f"top level is {type(arr).__name__}, not a list"
    if len(arr) != 12:
        return False, f"{len(arr)} objects, spec says 12"
    missing, badtype, badtags = [], [], []
    for i, o in enumerate(arr):
        if not isinstance(o, dict):
            missing.append(f"#{i}: not an object")
            continue
        miss = [k for k in T1_FIELDS if k not in o]
        if miss:
            missing.append(f"#{i}: {miss}")
            continue
        for k, ty in T1_FIELDS.items():
            # bool is a subclass of int; a price of True should not pass.
            if k != "inStock" and isinstance(o[k], bool) and ty is not bool:
                badtype.append(f"#{i}.{k}")
            elif not isinstance(o[k], ty):
                badtype.append(f"#{i}.{k} is {type(o[k]).__name__}")
        if not isinstance(o["tags"], list) or len(o["tags"]) != 2:
            badtags.append(f"#{i}.tags={o['tags']!r}")
    if missing:
        return False, f"missing fields in {len(missing)} place(s): {missing[:3]}"
    if badtype:
        return False, f"wrong field type at {badtype[:4]}"
    if badtags:
        return False, f"tags must hold exactly 2 entries: {badtags[:3]}"
    return True, "12 objects, all 6 fields present with correct types"


# --- Python AST checks -------------------------------------------------------
def check_py_class(text: str, name: str, required: list[str],
                   extra: list[tuple[str, str]] = ()) -> tuple[bool, str]:
    """`required` = methods that must exist on the class.
    `extra` = (label, predicate) pairs checked against the whole module.

    The first version of this only checked the class name and two methods, and
    scored 100% at every temperature. That was the validator's fault, not the
    model's: the prompt asks for four methods, a decorator, a specific data
    structure and docstrings, and none of that was being checked. A validator
    that only tests the easy part reports 100% and measures nothing.
    """
    body = _strip_fence(text)
    try:
        tree = ast.parse(body)
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not any(c.name == name for c in classes):
        found = [c.name for c in classes] or ["<none>"]
        return False, f"no class named {name} (saw {found})"
    cls = next(c for c in classes if c.name == name)
    methods = {n.name for n in ast.walk(cls)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [m for m in required if m not in methods]
    if missing:
        return False, f"missing method(s): {missing}"
    misses = [label for label, pred in extra if not _call_extra(pred, tree, body)]
    if misses:
        return False, f"missing requirement(s): {misses}"
    return True, f"class {name}: {', '.join(required)} + all extra requirements"


def _call_extra(pred, tree, body: str) -> bool:
    """Invoke an `extra` predicate, adapting to how many arguments it takes.

    The protocol is `pred(tree, body)`, and it bit this project twice: a
    one-argument predicate raised TypeError at *scoring* time, which surfaced
    as a crash inside a 104-generation sweep rather than as a failed self-test.
    Adapting by signature means adding a predicate with the "wrong" arity is
    impossible to get wrong.
    """
    import inspect
    n = len(inspect.signature(pred).parameters)
    if n >= 2:
        return pred(tree, body)
    if n == 1:
        return pred(tree)
    return pred()


def _has_decorator(tree, name: str):
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == name for n in ast.walk(tree))


def _imports_ordereddict(tree, body: str) -> bool:
    """Accept both idioms for `collections.OrderedDict`.

    The first version only accepted `from collections import OrderedDict`. The
    model overwhelmingly writes `import collections` followed by
    `collections.OrderedDict()`, which is what the prompt literally asks for
    ("使用 collections.OrderedDict 实现") — so the validator rejected correct
    answers and turned T2 into the most temperature-sensitive task in the set.
    Second time this project shipped a checker narrower than the prompt it was
    written against. A validator has to accept every form the prompt allows,
    not the form its author happened to type.
    """
    from_import = any(
        isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "collections"
        and any(a.name == "OrderedDict" for a in n.names)
        for n in ast.walk(tree))
    if from_import:
        return True
    imports_collections = any(
        isinstance(n, ast.Import) and any(a.name == "collections" for a in n.names)
        for n in ast.walk(tree))
    # `collections.OrderedDict(...)` or `collections.OrderedDict[...]`
    qualified = re.search(r"\bcollections\s*\.\s*OrderedDict\b", body)
    return bool(imports_collections and qualified)


def _has_docstrings(tree, _body: str = "") -> bool:
    """Every public method on a class should carry a docstring; the prompt says so.

    The `_body` parameter is unused but required: `check_py_class` calls every
    `extra` predicate as `pred(tree, body)`, so a one-argument predicate would
    raise TypeError at scoring time rather than failing its own self-test.
    """
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for m in c.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not m.name.startswith("__"):
                if not ast.get_docstring(m):
                    return False
    return True


def _defines(tree, name: str):
    return _has_decorator(tree, name)


def _subclasses(tree, base: str, names: list[str]) -> bool:
    found = set()
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for b in c.bases:
            bname = b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
            if bname == base:
                found.add(c.name)
    return all(n in found for n in names)


# --- T7: the arithmetic answer ----------------------------------------------
def _numbers(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def check_math(text: str) -> tuple[bool, str]:
    nums = _numbers(text)
    if not nums:
        return False, "no numbers in output"
    tol = 0.051
    if any(abs(n - T7_GOLD) < tol for n in nums):
        steps = sum(1 for i in T7_INTERMEDIATES if any(abs(n - i) < tol for n in nums))
        return True, f"final answer {T7_GOLD} present; {steps}/5 intermediate values shown"
    # A single stray match is not an answer; require it to look like a conclusion.
    if any(abs(n - T7_GOLD) < 5.0 for n in nums[-4:]):
        return False, f"close to {T7_GOLD} but not exact: {nums[-4:]}"
    return False, f"{T7_GOLD} not among {len(nums)} numbers"


# --- T6: required points covered -------------------------------------------
# The prompt lists five things. The first version checked three of them and
# reported 100%, which read as "temperature does not touch explanation tasks"
# when two requirements had simply never been examined.
T6_POINTS = {
    "why-not-two": r"(为什么不是两次|不需要两次|两次握手)",
    "why-not-four": r"(为什么不是四次|不需要四次|四次握手)",
    "ISN": r"\bISN\b|初始序列号|初始序号",
    "TIME_WAIT": r"TIME_?WAIT|time_?wait|时间等待",
    "SYN-ACK loss": r"SYN-?ACK[^。\n]{0,25}(丢失|丢|lost)|(丢失|丢)[^。\n]{0,15}SYN",
}


def check_qa_coverage(text: str) -> tuple[bool, str]:
    miss = [k for k, rx in T6_POINTS.items() if not re.search(rx, text, re.I)]
    if miss:
        return False, f"missing point(s): {miss}"
    return True, f"all {len(T6_POINTS)} required points present"


# --- T10: React component ---------------------------------------------------
# The prompt asks for useRef + useEffect, explicitly. The first version checked
# for useState, so every run that followed the prompt correctly was scored as a
# failure -- and the task came out looking like the most temperature-sensitive
# one in the set. The checker was wrong, and it inverted the finding.
def check_ts_react(text: str) -> tuple[bool, str]:
    body = _strip_fence(text)
    has_component = re.search(r"(export\s+)?(default\s+)?function\s+\w+|"
                              r"const\s+\w+\s*[:=].*=>|export\s+const\s+\w+", body)
    if not has_component:
        return False, "no component definition found"
    need = {
        "useRef (required by the prompt)": r"\buseRef\b",
        "useEffect (required by the prompt)": r"\buseEffect\b",
        "JSX": r"<[A-Z]|<div|return\s*\(",
    }
    miss = [k for k, rx in need.items() if not re.search(rx, body)]
    if miss:
        return False, f"missing: {miss}"
    imbalance = body.count("{") - body.count("}")
    if imbalance:
        # Any imbalance at all is a truncation signal. A tolerance of 1 or 2
        # let a genuinely broken sample through in the validator self-test.
        return False, f"brace imbalance {imbalance:+d} — output looks truncated"
    if re.search(r":\s*any\b", body):
        return False, "used `any`, which the prompt forbids"
    return True, "component + useRef + useEffect + JSX, braces balanced, no `any`"


# --- T8: literal prose requirements ----------------------------------------
def check_prose(text: str) -> tuple[bool, str]:
    body = text.strip()
    n = len(re.findall(r"[一-鿿]", body))
    simile = len(re.findall(r"像|如同|好像|仿佛|宛如|犹如|似的", body))
    if n < 300:
        return False, f"{n} CJK chars, prompt asked ~600"
    if simile < 2:
        return False, f"{simile} simile marker(s), prompt asked for >=2"
    return True, f"{n} CJK chars, {simile} simile markers"


# --- degeneracy, applies to every task --------------------------------------
def degeneracy(text: str) -> dict:
    """Cheap collapse detectors. These catch 'wrote nonsense', not 'wrote well'."""
    toks = re.findall(r"\w+|[一-鿿]", text)
    n = len(toks)
    if n == 0:
        return {"distinct2": 0.0, "max_repeat_run": 0, "n_tokens": 0}
    grams = [tuple(toks[i:i + 2]) for i in range(n - 1)]
    distinct2 = len(set(grams)) / len(grams) if grams else 0.0
    run = best = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return {"distinct2": round(distinct2, 4), "max_repeat_run": best, "n_tokens": n}


# --- generic, prompt-agnostic requirement checks ----------------------------
def _no_markdown(body: str) -> bool:
    """Every prompt in this set says 不要 markdown 标记 / 不要任何解释文字."""
    if "```" in body:
        return False
    head = body.lstrip()[:60]
    # A fenced block is handled above; catch bare lead-in prose instead.
    return not re.match(r"(?i)^(sure|here('s| is)|certainly|of course|好的|以下是)"
                        r"[\s:：,，]", head)


def _ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def _docstrings_english_triple(tree) -> bool:
    """The prompt says docstring 用英文三引号格式.

    Presence is checked by `_has_docstrings`; this checks the *format* half,
    which is a separate stated requirement and was never checked at all.
    """
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for m in c.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not m.name.startswith("__"):
                d = ast.get_docstring(m)
                if not d or not _ascii(d):
                    return False
    return True


def _ctor_params(tree, cls: str, params: list[str]) -> bool:
    """Constructor parameter names, e.g. Circle(radius) / Rectangle(width, height)."""
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if c.name != cls:
            continue
        for m in c.body:
            if isinstance(m, ast.FunctionDef) and m.name == "__init__":
                got = {a.arg for a in m.args.args[1:]}   # skip self
                return set(params).issubset(got)
    return False


def _subclass_init_docstrings(tree, base: str) -> bool:
    """T3 says 各自有带 docstring 的 __init__ — stated, and previously unchecked."""
    for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if base not in {b.id if isinstance(b, ast.Name)
                        else getattr(b, "attr", "") for b in c.bases}:
            continue
        for m in c.body:
            if isinstance(m, ast.FunctionDef) and m.name == "__init__":
                if not ast.get_docstring(m):
                    return False
    return True


# --- T6: length the prompt actually asked for --------------------------------
def check_qa_full(text: str) -> tuple[bool, str]:
    body = _strip_fence(text)
    miss = [k for k, rx in T6_POINTS.items() if not re.search(rx, body, re.I)]
    # 500 字左右 — 左右 is explicitly vague, so the band is calibrated from
    # observed output rather than guessed (AUDIT_CHECKLIST E7). Measured over
    # n=26 real generations: min 299, p10 464, median 517, p90 592, max 625.
    # The band brackets that with margin so it flags gross under-delivery, not
    # ordinary variation.
    n_cjk = len(re.findall(r"[一-鿿]", body))
    if not (250 <= n_cjk <= 800):
        miss.append(f"length {n_cjk} CJK chars, prompt asked ~500")
    if miss:
        return False, f"missing: {miss}"
    return True, f"all {len(T6_POINTS)} points + length {n_cjk} CJK"


# --- T7: the reordering the prompt explicitly demands -----------------------
# T7_REORDERED = (200 - 30) * 0.8 = 136, versus 90.5 in the asked order. The
# prompt says 说明如果顺序改成「先减 30 再打 8 折」结果是否相同，为什么 — an
# explicit requirement that no checker was looking at.
T7_REORDER_VALUES = (170.0, 136.0)


def check_math_full(text: str) -> tuple[bool, str]:
    ok, why = check_math(text)
    body = _strip_fence(text)
    if not ok:
        return False, why
    # Did it actually address the reordering, or stop at the first answer?
    addressed = re.search(r"先减", body) and re.search(r"(再打|之后打|然后打)\s*8\s*折|8\s*折", body)
    if not addressed:
        return False, ("final answer correct but the reordering the prompt asks "
                       "about is absent")
    has_vals = any(abs(n - v) < 0.051 for v in T7_REORDER_VALUES
                   for n in _numbers(body))
    tail = f"; reordered values {'present' if has_vals else 'not shown (not required)'}"
    return True, why + tail


# --- T8: the words the prompt bans ------------------------------------------
T8_FORBIDDEN = ("静谧", "寂静")


def check_prose_full(text: str) -> tuple[bool, str]:
    ok, why = check_prose(text)
    body = _strip_fence(text)
    hits = [w for w in T8_FORBIDDEN if w in body]
    if hits:
        return False, f"uses banned word(s) {hits}; prompt says 不要用「静谧」「寂静」"
    return ok, why


# --- T10: the props/type/dependency the prompt spells out --------------------
T10_FRAME_FIELDS = ("id", "title", "body", "tokens")


def check_ts_react_full(text: str) -> tuple[bool, str]:
    body = _strip_fence(text)
    ok, why = check_ts_react(body)
    need: dict[str, bool] = {
        "useRef (required by the prompt)": bool(re.search(r"\buseRef\b", body)),
        "useEffect (required by the prompt)": bool(re.search(r"\buseEffect\b", body)),
        "component named ReasoningPanel":
            bool(re.search(r"ReasoningPanel", body)),
        "props include frames + isStreaming":
            bool(re.search(r"\bframes\b", body)) and bool(re.search(r"\bisStreaming\b", body)),
        "ReasoningFrame has id/title/body/tokens":
            all(re.search(rf"\b{f}\s*[?:]", body) for f in T10_FRAME_FIELDS),
        "useEffect depends on frames.length":
            bool(re.search(r"frames\s*\.\s*length", body)),
    }
    miss = [k for k, v in need.items() if not v]
    if miss:
        return False, f"missing: {miss}"
    if not ok:
        return False, why
    return True, why + " + props/type/dependency requirements"


# --- requirement ledger ----------------------------------------------------
# Every requirement these seven prompts state, and what happens to it.
#
# The rule this enforces: a requirement may be CHECKED or NOT_MEASURED, but it
# may never be silently ignored. Silence is how T2 came out "100% at every
# temperature" while the LRU eviction logic — the entire point of an LRU cache
# — was never looked at. `audit_validators.py` fails if a prompt requirement
# appears in none of the checkers' reasons, so the gap has to be written down
# to survive.
#
# tier "syntactic" = a regex or AST predicate can decide it.
# tier "semantic" = deciding it requires reading what the code actually does.
#                   These are NOT_MEASURED, and that is stated here rather than
#                   left for a reader to assume.
#
# prompt_quote is the phrase in tasks.py the row comes from, so a reviewer can
# check the transcription instead of trusting it (AUDIT_CHECKLIST E3).
VALIDATION_SPEC: dict[str, list[dict]] = {
    "T1_json_structured": [
        {"id": "array", "tier": "syntactic", "checked": True,
         "quote": "严格输出一个 JSON 数组"},
        {"id": "count_12", "tier": "syntactic", "checked": True,
         "quote": "数组包含 12 个对象"},
        {"id": "field_types_6", "tier": "syntactic", "checked": True,
         "quote": "id(整数)、name(英文产品名)、price(数字)、category(字符串)"},
        {"id": "inStock_bool", "tier": "syntactic", "checked": True,
         "quote": "inStock(布尔值)"},
        {"id": "tags_2", "tier": "syntactic", "checked": True,
         "quote": "tags(2 个标签的数组)"},
        {"id": "content_distinct_12", "tier": "semantic", "checked": False,
         "quote": "内容是 12 个不同的智能家居产品",
         "why": "distinctness is checkable but 'smart home products' is a domain "
                "judgement; mixing the two would make a domain miss look like a "
                "temperature effect. Not measured."},
    ],
    "T2_code_function": [
        {"id": "class_name", "tier": "syntactic", "checked": True,
         "quote": "写一个名为 LRUCache 的类"},
        {"id": "ordereddict", "tier": "syntactic", "checked": True,
         "quote": "使用 collections.OrderedDict 实现"},
        {"id": "methods_4", "tier": "syntactic", "checked": True,
         "quote": "get(key, default=None)、put(key, value)、__len__、__repr__"},
        {"id": "docstrings", "tier": "syntactic", "checked": True,
         "quote": "每个方法都要有完整的 docstring"},
        {"id": "docstring_english_triple", "tier": "syntactic", "checked": True,
         "quote": "docstring 用英文三引号格式"},
        {"id": "capacity_from_init", "tier": "syntactic", "checked": True,
         "quote": "容量由 __init__ 的 capacity 参数控制"},
        {"id": "timed_decorator", "tier": "syntactic", "checked": True,
         "quote": "再写一个装饰器 timed(fn)"},
        {"id": "no_markdown", "tier": "syntactic", "checked": True,
         "quote": "只输出 Python 代码，不要 markdown 标记"},
        {"id": "lru_eviction", "tier": "executable", "checked": True,
         "mechanism": "exec", "runner": "exec_checks.py:probe_lruncache",
         "quote": "超出时淘汰最久未使用的键",
         "why": "Resolved by execution rather than by pattern matching. The "
                "module is exec'd and driven through put A,B,C / get A / put D; "
                "an LRU evicts B, a FIFO evicts A, and the probe names which "
                "one it saw. A presence-only checker scores both as a pass, "
                "which is why this sat unchecked for so long."},
        {"id": "timed_prints_duration", "tier": "executable", "checked": True,
         "mechanism": "exec", "runner": "exec_checks.py:probe_timed",
         "quote": "打印函数耗时",
         "why": "Called and its stdout captured; a decorator that exists but "
                "prints nothing is rejected."},
    ],
    "T3_code_repetitive": [
        {"id": "class_shape", "tier": "syntactic", "checked": True,
         "quote": "写一个基类 Shape"},
        {"id": "methods_area_describe", "tier": "syntactic", "checked": True,
         "quote": "含 area() 和 describe() 两个方法"},
        {"id": "subclasses_3", "tier": "syntactic", "checked": True,
         "quote": "三个完全同构的子类 Circle(radius)、Rectangle(width, height)、Triangle(base, height)"},
        {"id": "ctor_params", "tier": "syntactic", "checked": True,
         "quote": "Circle(radius)、Rectangle(width, height)、Triangle(base, height)"},
        {"id": "init_docstrings", "tier": "syntactic", "checked": True,
         "quote": "各自有带 docstring 的 __init__"},
        {"id": "total_area", "tier": "syntactic", "checked": True,
         "quote": "最后写一个函数 total_area(shapes) 遍历求和"},
        {"id": "no_markdown", "tier": "syntactic", "checked": True,
         "quote": "只输出 Python 代码，不要 markdown 标记"},
        {"id": "describe_format", "tier": "semantic", "checked": False,
         "quote": "describe 返回 f\"{type} area={area}\"",
         "why": "the exact f-string shape is a formatting judgement; presence "
                "of describe() is checked, its return value is not. Not measured."},
        {"id": "areas_correct", "tier": "executable", "checked": True,
         "mechanism": "exec", "runner": "exec_checks.py:probe_shapes",
         "quote": "各自实现 area()",
         "why": "Instantiated and compared against pi*r^2, w*h and b*h/2, with "
                "total_area checked as a sum. Every area present but "
                "arithmetically wrong is rejected, which a presence-only "
                "checker passes."},
    ],
    "T6_qa_factual": [
        {"id": "why_not_two", "tier": "syntactic", "checked": True,
         "quote": "为什么不是两次"},
        {"id": "why_not_four", "tier": "syntactic", "checked": True,
         "quote": "为什么不是四次"},
        {"id": "isn", "tier": "syntactic", "checked": True,
         "quote": "ISN（初始序列号）的作用"},
        {"id": "time_wait", "tier": "syntactic", "checked": True,
         "quote": "TIME_WAIT 状态为什么必须存在"},
        {"id": "synack_loss", "tier": "syntactic", "checked": True,
         "quote": "如果 SYN-ACK 丢失会发生什么"},
        {"id": "length_500", "tier": "syntactic", "checked": True,
         "quote": "总共 500 字左右"},
        {"id": "explanations_correct", "tier": "semantic", "checked": False,
         "quote": "请系统地解释 TCP 三次握手的必要性",
         "why": "covering a point is not explaining it correctly. The checker "
                "confirms the topic appears, never that the reasoning is sound. "
                "Not measured — this is why T6's 19/26 is a coverage number."},
    ],
    "T7_math_reasoning": [
        {"id": "final_90_5", "tier": "syntactic", "checked": True,
         "quote": "实付"},
        {"id": "steps_shown", "tier": "syntactic", "checked": True,
         "quote": "请分步骤计算每一步的金额"},
        {"id": "reorder_addressed", "tier": "syntactic", "checked": True,
         "quote": "并说明如果顺序改成「先减 30 再打 8 折」结果是否相同，为什么"},
        {"id": "chinese", "tier": "syntactic", "checked": True,
         "quote": "用中文回答"},
        {"id": "reorder_correct", "tier": "semantic", "checked": False,
         "quote": "结果是否相同，为什么",
         "why": "the reordered answer is 136, not 90.5; whether the model says "
                "so correctly is not checked, only that it addresses the reorder."},
    ],
    "T8_creative_writing": [
        {"id": "length_600", "tier": "syntactic", "checked": True,
         "quote": "600 字左右的散文"},
        {"id": "similes_2", "tier": "syntactic", "checked": True,
         "quote": "有至少两处比喻"},
        {"id": "forbidden_words", "tier": "syntactic", "checked": True,
         "quote": "不要用「静谧」「寂静」这类常见词"},
        {"id": "no_repeated_chars", "tier": "semantic", "checked": False,
         "quote": "不要重复用字",
         "why": "distinct2 and max_repeat_run are recorded but no threshold is "
                "calibrated; inventing one would be exactly E7 (a guessed "
                "tolerance that passes real errors). Not measured."},
        {"id": "imagery", "tier": "semantic", "checked": False,
         "quote": "要求有画面感", "why": "not mechanically decidable. Not measured."},
        {"id": "emotional_arc", "tier": "semantic", "checked": False,
         "quote": "有情绪起伏", "why": "not mechanically decidable. Not measured."},
        {"id": "prose_quality", "tier": "semantic", "checked": False,
         "quote": "(implicit — it is a writing task)",
         "why": "nothing in this file measures whether the prose reads well."},
    ],
    "T10_ts_component": [
        {"id": "component_defined", "tier": "syntactic", "checked": True,
         "quote": "实现一个 React 函数组件"},
        {"id": "use_ref", "tier": "syntactic", "checked": True,
         "quote": "用 useRef 保存是否被用户手动滚动"},
        {"id": "use_effect", "tier": "syntactic", "checked": True,
         "quote": "用 useEffect 依赖 frames.length 触发滚动"},
        {"id": "dep_frames_length", "tier": "syntactic", "checked": True,
         "quote": "依赖 frames.length"},
        {"id": "component_named", "tier": "syntactic", "checked": True,
         "quote": "组件 ReasoningPanel"},
        {"id": "props_signature", "tier": "syntactic", "checked": True,
         "quote": "props 为 { frames: ReasoningFrame[]; isStreaming: boolean; onSeek?: ... }"},
        {"id": "frame_type_fields", "tier": "syntactic", "checked": True,
         "quote": "ReasoningFrame 是 { id; title; body; tokens }"},
        {"id": "no_any", "tier": "syntactic", "checked": True,
         "quote": "全部类型显式标注，不使用 any"},
        {"id": "braces_balanced", "tier": "syntactic", "checked": True,
         "quote": "(implicit — a truncated file is not a component)"},
        {"id": "manual_scroll_pauses", "tier": "semantic", "checked": False,
         "quote": "但用户手动上滚后暂停自动滚动",
         "why": "THE hard requirement of the task. Proving a component actually "
                "pauses on manual scroll needs execution, not a regex. Not "
                "measured."},
        {"id": "autoscroll_bottom", "tier": "semantic", "checked": False,
         "quote": "流式生成时新增的 frame 自动滚动到底部",
         "why": "scrollIntoView/ scrollTo being called is not the same as it "
                "reaching the bottom. Not measured."},
        {"id": "token_totals_shown", "tier": "semantic", "checked": False,
         "quote": "显示每个 frame 的 token 数并合计总数",
         "why": "requires reading the render output and the reduce. Not measured."},
    ],
}


def ledger_summary() -> dict:
    tot = chk = sem = 0
    for reqs in VALIDATION_SPEC.values():
        for r in reqs:
            tot += 1
            if r["checked"]:
                chk += 1
            else:
                sem += 1
    return {"total": tot, "checked": chk, "not_measured": sem}


# --- registry ---------------------------------------------------------------
CHECKS = {
    "T1_json_structured": check_json_12,
    # The prompts ask for a lot more than the class and two methods. Checking
    # only that produced a flat 100% that said nothing about temperature.
    "T2_code_function": lambda t: check_py_class(
        t, "LRUCache", ["get", "put", "__len__", "__repr__"],
        extra=[("uses OrderedDict", _imports_ordereddict),
               ("timed(fn) decorator", lambda tr, bd: _defines(tr, "timed")),
               ("docstrings on methods", _has_docstrings),
               ("docstrings in English triple quotes", _docstrings_english_triple),
               ("capacity comes from __init__ capacity",
                lambda tr, bd: _ctor_params(tr, "LRUCache", ["capacity"])),
               ("no markdown fence", lambda tr, bd: _no_markdown(bd))]),
    "T3_code_repetitive": lambda t: check_py_class(
        t, "Shape", ["area", "describe"],
        extra=[("subclasses Circle/Rectangle/Triangle",
               lambda tr, bd: _subclasses(tr, "Shape",
                                          ["Circle", "Rectangle", "Triangle"])),
               ("total_area()", lambda tr, bd: _defines(tr, "total_area")),
               ("__init__ docstring on each subclass",
                lambda tr, bd: _subclass_init_docstrings(tr, "Shape")),
               ("ctor params radius / width+height / base+height",
                lambda tr, bd: (_ctor_params(tr, "Circle", ["radius"])
                                and _ctor_params(tr, "Rectangle", ["width", "height"])
                                and _ctor_params(tr, "Triangle", ["base", "height"]))),
               ("no markdown fence", lambda tr, bd: _no_markdown(bd))]),
    "T6_qa_factual": check_qa_full,
    "T7_math_reasoning": check_math_full,
    "T10_ts_component": check_ts_react_full,
    "T8_creative_writing": check_prose_full,
}

# Which checks answer "is the CONTENT right" vs only "is the FORMAT right".
# Only the content ones are allowed to be read as a quality signal.
CONTENT_CHECKS = {"T2_code_function", "T3_code_repetitive", "T6_qa_factual",
                  "T7_math_reasoning", "T10_ts_component", "T8_creative_writing"}

prose_quality_measured = (
    False  # T8 checks length and simile count — the prompt's own stated
           # requirements. It does not measure whether the prose is good, and
           # nothing in this file does.
)


# Bump whenever a check changes meaning. `router_profiles.json` records this
# string against every measured temperature ceiling, and `check_router_staleness.py`
# flags any ceiling that was derived under a different validator. A ceiling is
# a number someone will act on, so it has to carry the instrument it came from.
VALIDATOR_VERSION = "2026-10-02.audit1"


def _derive_fence_checked() -> frozenset:
    """Which prompts forbid markdown fences — read from the prompts, not hand-listed.

    A hand-maintained list of task ids is a list that silently rots when a task
    is added. The prompt text is the source of truth, so ask it.
    """
    try:
        import tasks
    except Exception:                       # pragma: no cover
        return frozenset()
    out = set()
    for t in tasks.TASKS:
        prompt = t[2] if len(t) > 2 else ""
        if isinstance(prompt, str) and "markdown" in prompt.lower():
            out.add(t[0])
    return frozenset(out)


FENCE_CHECKED = _derive_fence_checked()


def check(task_id: str, text: str, ignore_format: bool = False) -> tuple[bool, str]:
    """Score one generation.

    `ignore_format=True` skips the output-format rules and judges only the
    substance. This exists because of a real and otherwise-hidden behaviour:
    the model wraps code in ``` fences on ~86% of generations across every
    temperature, including T=0, despite the prompt saying 不要 markdown 标记.

    Folding that into a single `valid` number reads as "code quality collapses
    with temperature", which is not what is happening — the code is fine and
    the wrapper is not. The two are reported separately so the finding stays
    legible: a formatting instruction the model ignores, versus a task it cannot
    do. Note the rule lives here rather than inside the per-task validators
    because every validator strips fences before it looks at anything, so a
    fence check downstream of `_strip_fence` is unreachable by construction.
    """
    if not ignore_format and task_id in FENCE_CHECKED and "```" in text:
        return False, "output is wrapped in a markdown fence, prompt forbids it"
    fn = CHECKS.get(task_id)
    return fn(text) if fn else (None, "no validator")
