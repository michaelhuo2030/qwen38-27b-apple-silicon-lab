"""ladder27 的自测。**不需要模型、不需要 oMLX 在跑**就能全部执行。

设计原则沿用本仓库既有的那条（qa_audit/selftest.py）：
**每条守卫必须被证明会开火，而且合法配置必须不被误伤。**
只测「坏配置被拦住」是不够的 —— 那样一个永远返回 False 的守卫也能通过。

所以每个用例都是成对的：
    assert_guard_fires(bad_input, expected_error)
    assert_guard_silent(good_input)          # 防止守卫写成永远触发

为什么这些必须在没有模型时就能测
-------------------------------
ladder27 的价值全在「别把坏数据当好数据」。而坏数据最早出现的地方是
**发请求之前**：一个互斥的 settings 组合、一次没回读就当成功的写入、
一个被 router 悄悄改掉的采样参数。等到真模型跑起来才发现这些，
一整轮实验的数据已经废了。
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ladder27 import (  # noqa: E402
    Answer, BY_ID, ConfigError, Ladder, MTP_RE, SENTINELS, SAMPLING,
    SentinelReport, TIERS, validate_config, weighted_alpha,
)

FAILURES: list[str] = []
CHECKS = 0


def ok(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(label)


def guard_fires(fn, exc, label: str) -> None:
    """守卫必须被证明会开火。"""
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except exc:
        return
    except Exception as e:                       # noqa: BLE001
        FAILURES.append(f"{label}: 抛了 {type(e).__name__} 而不是 {exc.__name__}: {e}")
        return
    FAILURES.append(f"{label}: 守卫没有开火（空操作通过）")


def guard_silent(fn, label: str) -> None:
    """合法配置必须不被误伤 —— 防止守卫写成永远触发。"""
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except Exception as e:                       # noqa: BLE001
        FAILURES.append(f"{label}: 合法配置被误伤: {type(e).__name__}: {e}")


# ======================================================================
# 1. 配置互斥
# ======================================================================
def t_config_guards() -> None:
    # P8: oq_a8 与 ane 互斥（oMLX model_settings.py:504 会拒绝）
    guard_fires(
        lambda: validate_config(ane=True, oq_a8=True, turboquant=4,
                                mtp=True, dflash=False, vlm_mtp=False),
        ConfigError, "oq_a8 + ane 互斥")
    # oq_a8 单独使用是合法的（只是本机没收益），不能误伤
    guard_silent(
        lambda: validate_config(ane=False, oq_a8=True, turboquant=4,
                                mtp=True, dflash=False, vlm_mtp=False),
        "oq_a8 单独合法")

    # dflash / vlm_mtp 与 mtp 互斥
    guard_fires(
        lambda: validate_config(ane=False, oq_a8=False, turboquant=4,
                                mtp=True, dflash=True, vlm_mtp=False),
        ConfigError, "mtp + dflash 互斥")
    guard_fires(
        lambda: validate_config(ane=False, oq_a8=False, turboquant=4,
                                mtp=True, dflash=False, vlm_mtp=True),
        ConfigError, "mtp + vlm_mtp 互斥")

    # turboquant 位数只允许 4 / 8
    guard_fires(
        lambda: validate_config(ane=False, oq_a8=False, turboquant=6,
                                mtp=True, dflash=False, vlm_mtp=False),
        ConfigError, "turboquant_kv_bits 非法值")
    guard_fires(
        lambda: validate_config(ane=False, oq_a8=False, turboquant=True,
                                mtp=True, dflash=False, vlm_mtp=False),
        ConfigError, "turboquant=True 会被当成 1bit 而非法")

    # 合法组合必须全部通过
    for kw, label in [
        (dict(ane=True, oq_a8=False, turboquant=4, mtp=True,
              dflash=False, vlm_mtp=False), "全开合法"),
        (dict(ane=False, oq_a8=False, turboquant=None, mtp=False,
              dflash=False, vlm_mtp=False), "全关合法"),
        (dict(ane=True, oq_a8=False, turboquant=8, mtp=True,
              dflash=False, vlm_mtp=False), "tqkv 8bit 合法"),
        (dict(ane=False, oq_a8=True, turboquant=4, mtp=True,
              dflash=False, vlm_mtp=False), "oq_a8 单独合法"),
        (dict(ane=False, oq_a8=False, turboquant=4, mtp=False,
              dflash=True, vlm_mtp=False), "dflash 单独合法"),
    ]:
        guard_silent(lambda kw=kw: validate_config(**kw), label)


# ======================================================================
# 2. 非法配置必须在任何 HTTP 之前被拦下
# ======================================================================
def t_no_http_before_validation() -> None:
    """Ladder 构造时校验失败，不能先去连服务器。

    这条很重要：如果校验放在 activate() 里，构造函数就会成功，
    调用方以为配置没问题，直到写 settings 触发一次**完整模型重载**
    才被服务器拒绝 —— 白白浪费几十秒和一次重载。
    """
    global CHECKS
    CHECKS += 1
    try:
        Ladder("4bit", ane=True, oq_a8=True, verbose=False)
    except ConfigError:
        return                                   # 期望路径
    except Exception as e:                       # noqa: BLE001
        FAILURES.append(
            f"非法配置在构造阶段抛了 {type(e).__name__} 而不是 ConfigError —— "
            f"说明它先尝试联网了: {e}")
        return
    FAILURES.append("Ladder 接受了互斥配置，构造阶段没拦")


# ======================================================================
# 3. 采样表
# ======================================================================
def t_sampling_table() -> None:
    """官方推荐参数不能被改错 —— 改错了实验结论就废了。"""
    ok(SAMPLING["think_off"]["temperature"] == 0.7,
       "non-thinking 温度必须是 0.7（官方推荐）")
    ok(SAMPLING["think_off"]["presence_penalty"] == 1.5,
       "non-thinking 必须带 presence_penalty=1.5")
    ok(SAMPLING["think_on"]["temperature"] == 1.0,
       "thinking 温度必须是 1.0（官方推荐）")
    ok(SAMPLING["think_on"]["presence_penalty"] == 0.0,
       "thinking 模式不能带 presence_penalty")
    ok(SAMPLING["frozen"]["temperature"] == 0.0,
       "frozen 必须是 T=0")
    # 两个模式的温度必须不同 —— 相同就说明没区分 thinking，等于没控制住变量
    ok(SAMPLING["think_on"]["temperature"] != SAMPLING["think_off"]["temperature"],
       "thinking / non-thinking 温度必须不同，否则变量没分开")
    # 守卫自检：sampling_name 必须跟着 think 走
    l_on = Ladder("4bit", think=True, verbose=False)
    l_off = Ladder("4bit", think=False, verbose=False)
    ok(l_on.sampling_name == "think_on" and l_off.sampling_name == "think_off",
       "sampling_name 必须跟随 think 标志")
    ok(l_on.settings() == l_off.settings(),
       "sampling 组不影响 settings（只在请求体里）")


# ======================================================================
# 4. settings() 生成
# ======================================================================
def t_settings_generation() -> None:
    l = Ladder("4bit", ane=True, turboquant=4, mtp=True, depth=2, verbose=False)
    s = l.settings()
    # ANE 的参数必须齐全。gdn 默认改成了 True，依据见下一段"默认值为什么这样定"。
    for k, v in [("qwen35_ane_prefill_fraction", 0.5),
                 ("qwen35_ane_prefill_sequence_length", 2048),
                 ("qwen35_ane_prefill_max_layers", 64),
                 ("qwen35_ane_prefill_dual_ane", True),
                 ("qwen35_ane_prefill_gdn", True)]:
        ok(s.get(k) == v, f"ANE 设置 {k} 应为 {v}，实际 {s.get(k)!r}")
    ok(s["qwen35_oq_a8_enabled"] is False, "默认必须关 oq_a8")
    ok(s["turboquant_kv_enabled"] is True and s["turboquant_kv_bits"] == 4.0,
       "显式传 turboquant=4 时必须开")
    ok(s["mtp_fixed_depth"] == 2, "depth 必须透传")
    ok(s["is_pinned"] is False, "实验模型不能 pin（否则自动换不出）")

    # 关掉 ANE 时：调参参数一个都不该带（避免旧值残留被误读为生效），
    # 但 **enabled 开关必须显式写 False** —— 只在关闭时不写这个键的话，
    # 上一次开着的 True 会留在 settings 里，ANE 照样在跑。
    l2 = Ladder("4bit", ane=False, turboquant=None, verbose=False)
    s2 = l2.settings()
    tuning = [k for k in s2
              if k.startswith("qwen35_ane") and k != "qwen35_ane_prefill_enabled"]
    ok(not tuning, f"ANE 关闭时不应带调参参数，实际带了 {tuning}")
    ok(s2["qwen35_ane_prefill_enabled"] is False,
       "ANE 关闭时必须显式写 enabled=False，否则上次的 True 会残留")
    ok(s2["turboquant_kv_enabled"] is False,
       "TurboQuant=None 必须显式写 enabled=False（不能只是不传 bits）")
    ok("turboquant_kv_bits" not in s2,
       "TurboQuant 关闭时不该带 bits，否则会被读成残留值")

    # describe 必须能自证身份（含 tier 与体积）
    d = l.describe()
    ok(d["tier"] == "4bit" and d["weights_gb"] == 16.97,
       f"describe 应含 tier 与体积，实际 {d.get('tier')}/{d.get('weights_gb')}")


# ======================================================================
# 4b. 默认值为什么是这样定的（2026-10-03 改，别改回去）
# ======================================================================
def t_defaults_are_deliberate() -> None:
    """这两个默认值是**刻意**改的，不是手滑。改之前先读这段注释。"""
    # --- TurboQuant 默认关 ---
    # 1. oMLX 官方 benchmark 库里那台**同型号**机器（M2 Max 38c/96GB）的
    #    两份 recipe 都是 turboquant_kv_enabled: false
    # 2. 社区一致：MTP + TQ 双开会让 verify 加速失效（omlx#2215/#2782）
    # 3. 0.7.0 仍有活 bug：TQ + MTP + 2 并发会崩（omlx#3906）。
    #    本地 turboquant_kv.py:281 的 trim() 没修，已逐行确认。
    d = Ladder("4bit", verbose=False)
    ok(d.turboquant is None, "默认 turboquant 必须归一化成 None（=关闭）")
    ok(d.settings()["turboquant_kv_enabled"] is False, "默认必须关 TurboQuant")
    # 0 和 None 等价，且不会把 0 当成位宽传下去
    z = Ladder("4bit", turboquant=0, verbose=False)
    ok(z.turboquant is None and "turboquant_kv_bits" not in z.settings(),
       "turboquant=0 必须等同关闭，且不能带 kv_bits 字段")
    # 显式开启仍然可用（E4 要测它）
    on = Ladder("4bit", turboquant=8, verbose=False)
    ok(on.settings()["turboquant_kv_enabled"] is True, "显式传 8 必须能开")
    ok(on.settings()["turboquant_kv_bits"] == 8.0, "位宽要透传")

    # --- ANE GDN 默认开 ---
    # 旧注释说"社区最优是 False"，但那是别的机器。
    # 同型号两份 recipe 都是 gdn=True，v0.6.1 那份用户备注还写了 "GDN on"。
    ok(d.settings()["qwen35_ane_prefill_gdn"] is True, "ANE GDN 默认必须是 True")
    off = Ladder("4bit", ane_gdn=False, verbose=False)
    ok(off.settings()["qwen35_ane_prefill_gdn"] is False, "要能显式关掉做 E3 A/B")


# ======================================================================
# 5. token 加权 α —— 本仓库踩过的坑
# ======================================================================
def t_weighted_alpha() -> None:
    # 构造一个「短回答错得多、长回答好得多」的组合：
    #   短：7/10   = 70%   占 10 个分母
    #   长：90/100 = 90%   占 100 个分母
    # 简单平均 = (70+90)/2 = 80.0   ← 短回答与长回答同权
    # token 加权 = 97/110 = 88.18    ← 按实际接受次数加权
    lines = [
        {"accept_num": 7, "accept_den": 10, "tokens": 1, "cycles": 1,
         "tok_per_cycle": 1.0, "accept_pct": 70.0, "depth_hist": "1:7"},
        {"accept_num": 90, "accept_den": 100, "tokens": 100, "cycles": 100,
         "tok_per_cycle": 1.0, "accept_pct": 90.0, "depth_hist": "1:90"},
    ]
    got = weighted_alpha(lines)
    ok(got == 88.18, f"token 加权 α 应为 88.18（97/110），实际 {got}")
    simple = statistics.mean([x["accept_pct"] for x in lines])
    ok(abs(simple - 80.0) < 0.01, f"对照组：简单平均应为 80.0，实际 {simple}")
    if got == simple:
        FAILURES.append("α 退化成简单平均 —— 20 次拒绝被 100 次接受掩盖，"
                        "这是本仓库早期犯过的错")
    # 反向：长回答错得多时，加权必须往下拉
    lines2 = [
        {"accept_num": 9, "accept_den": 10, "tokens": 1, "cycles": 1,
         "tok_per_cycle": 1.0, "accept_pct": 90.0, "depth_hist": "1:9"},
        {"accept_num": 10, "accept_den": 100, "tokens": 100, "cycles": 100,
         "tok_per_cycle": 1.0, "accept_pct": 10.0, "depth_hist": "1:10"},
    ]
    got2 = weighted_alpha(lines2)
    ok(got2 == 17.27, f"反向应得 17.27（19/110），实际 {got2}")
    if got2 > 50:
        FAILURES.append("长回答大量拒绝时被短回答的高 α 掩盖 —— 加权方向错了")

    # 分母为 0 必须返回 None 而不是抛异常
    ok(weighted_alpha([]) is None, "空输入必须返回 None")
    ok(weighted_alpha([{"accept_num": 0, "accept_den": 0}]) is None,
       "分母为 0 必须返回 None")
    guard_fires(lambda: weighted_alpha([{"tokens": 1}]), KeyError,
                "字段缺失必须抛而不是静默")


# ======================================================================
# 6. MTP 日志解析
# ======================================================================
def t_mtp_log_parsing() -> None:
    line = ("MTP accepted=120 cycles=40 tok/cycle=3.0 "
            "accept=117/120 (97.5%) depth_hist=[3:38,2:2,1:0]")
    m = MTP_RE.search(line)
    ok(m is not None, f"必须能解析真实格式的 MTP 行，实际没匹配: {line!r}")
    if m:
        ok(m.group(1) == "120", "tokens")
        ok(m.group(2) == "40", "cycles")
        ok(abs(float(m.group(3)) - 3.0) < 1e-6, "tok/cycle")
        ok(m.group(4) == "117" and m.group(5) == "120", "accept num/den")
        ok(abs(float(m.group(6)) - 97.5) < 1e-6, "accept pct")
        ok(m.group(7) is not None, "depth_hist")

    # 不能把非 MTP 行解析成 MTP（否则 n_mtp_lines 虚高，α 虚高）
    for junk in ("GET /v1/chat/completions 200",
                 "accepted: none",
                 "timed 3.14s"):
        ok(MTP_RE.search(junk) is None, f"不该匹配: {junk!r}")


# ======================================================================
# 7. 哨兵
# ======================================================================
def t_sentinel_report() -> None:
    good = SentinelReport(passed=["a", "b"])
    bad = SentinelReport(passed=["a"], failed=["b"])
    ok(good.ok is True, "无失败即 ok")
    ok(bad.ok is False, "有失败即 not ok")
    # 空报告必须是 ok —— 否则第一次跑还没跑就被判失败
    ok(SentinelReport().ok is True, "空哨兵报告应视为通过")

    # 哨兵必须真的带检查点，否则「哨兵」是空操作
    ok(len(SENTINELS) >= 4, f"哨兵至少 4 条，实际 {len(SENTINELS)}")
    ok(all(len(s) == 3 for s in SENTINELS), "每条哨兵要有 (名, prompt, 期望串)")
    ok(len({s[0] for s in SENTINELS}) == len(SENTINELS), "哨兵名不能重复")
    # 哨兵必须用短输出，否则每次配置验证都很慢
    for name, prompt, _ in SENTINELS:
        ok(len(prompt) < 200, f"哨兵 {name} 的 prompt 太长")


# ======================================================================
# 8. Answer 数据契约
# ======================================================================
def t_answer_contract() -> None:
    """实验数据要能自证「这次测量有效吗」。缺字段必须显式为 None，
    不能默默塞 0 —— 0 会被当成「测到了 0」，比没测更危险。"""
    a = Answer(text="x", tier="4bit", model_id="m", config={}, sampling="frozen")
    for f in ("alpha", "gen_tps", "ttft", "completion_tokens",
              "prompt_tokens", "tok_per_cycle"):
        ok(getattr(a, f) is None, f"{f} 缺省必须是 None 而不是 0")
    d = a.to_dict()
    ok("n_mtp_lines" in d and d["n_mtp_lines"] == 0, "to_dict 必须带 n_mtp_lines")
    ok(d["truncated"] is False, "truncated 缺省 False")

    # 截断必须被标出来 —— 125B 上有 1800 上限截断导致误判的先例
    a2 = Answer(text="x", tier="4bit", model_id="m", config={},
                sampling="frozen", completion_tokens=698, truncated=True)
    ok(a2.truncated is True, "截断必须可见")


# ======================================================================
# 9. 档位表
# ======================================================================
def t_tier_table() -> None:
    ok(len(TIERS) == 5, f"应为 5 档，实际 {len(TIERS)}")
    for k in ("ternary", "oq3e", "4bit", "6bit", "8bit"):
        ok(k in TIERS, f"缺档位 {k}")
    # 体积必须递增 —— 顺序反了说明填错。
    # ⚠️ oq3e 故意不参与递增断言：它和 ternary 是**同级的两个 3-bit**
    # （13.81 vs 13.86 GB），不是更低一档。把它塞进递增序列会让断言必挂，
    # 而真正该守的是「完整阶梯 3→4→6→8 单调」+「两个 3-bit 体量相当」。
    ladder = ["ternary", "4bit", "6bit", "8bit"]
    gbs = [TIERS[k]["gb"] for k in ladder]
    ok(gbs == sorted(gbs), f"阶梯体积必须递增，实际 {gbs}")
    ratio = TIERS["oq3e"]["gb"] / TIERS["ternary"]["gb"]
    ok(0.90 <= ratio <= 1.10,
       f"oq3e 与 ternary 应为同一体量级（3-bit 档），实际比值 {ratio:.3f}")
    # id 不能重复，否则切档会切错
    ids = [v["id"] for v in TIERS.values()]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    ok(not dup, f"模型 id 重复: {dup}")
    # 每档都要记 bpw
    ok(all("bpw" in v for v in TIERS.values()), "每档必须记 bpw")
    # BY_ID 必须能反查（runner 靠它把 oMLX 返回的 id 映射回档位名）
    ok(BY_ID.get(TIERS["oq3e"]["id"]) == "oq3e",
       f"BY_ID 反查失败: {BY_ID.get(TIERS['oq3e']['id'])}")
    guard_fires(lambda: Ladder("3bit", verbose=False), KeyError, "未知档位必须拒绝")
    guard_fires(lambda: Ladder("ternary", ane=True, oq_a8=True, verbose=False),
                ConfigError, "Ladder 构造必须做配置校验")


# ======================================================================
# 10. 并发上限不能是抄来的
# ======================================================================
def t_concurrency_not_hardcoded() -> None:
    """MAX_WORKERS=4 是 125B 的结论。27B 只有 13.9–30GB，
    omlx.ai 同型号实测 8 并发有 6.49× 加速。默认值可以保守，
    但必须留出可测的口子并在注释里说清它是待测的。"""
    ok(isinstance(Ladder.MAX_WORKERS, int) and Ladder.MAX_WORKERS >= 1,
       "MAX_WORKERS 必须是正整数")
    ok(hasattr(Ladder, "sweep_concurrency"),
       "必须提供并发实测入口，不能只留一个抄来的常量")
    import inspect
    src = inspect.getsource(Ladder.__init__)
    ok("MAX_WORKERS" not in src,
       "__init__ 不该把 MAX_WORKERS 写死到 session 上")


def main() -> int:
    for fn in (t_config_guards, t_no_http_before_validation, t_sampling_table,
               t_settings_generation, t_defaults_are_deliberate,
               t_weighted_alpha, t_mtp_log_parsing,
               t_sentinel_report, t_answer_contract, t_tier_table,
               t_concurrency_not_hardcoded):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            import traceback
            FAILURES.append(f"{fn.__name__} 崩了: {type(e).__name__}: {e}\n"
                            f"{traceback.format_exc()}")

    if FAILURES:
        print(f"ladder27 自测失败 {len(FAILURES)}/{CHECKS}：")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"ladder27 self-test clean: {CHECKS} checks, "
          f"每条守卫都被证明会开火，合法配置全部不被误伤")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
