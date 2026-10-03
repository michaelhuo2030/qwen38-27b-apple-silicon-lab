#!/usr/bin/env python3
"""
低比特结果对账 —— 把 task_version 指纹加进来之前跑的旧题数据剔出去。

为什么需要这个
--------------
`task_version` 指纹是 8 档对照跑完**之后**才加进 run_lowbit.py 的。
后果：lowbit.json 里 ternary/4bit/6bit 三档 96 行**全部没有指纹**，
而 rep0 每档有 16 行 = 8 个任务 × **两个题目版本**。
旧题和新题混在同一个 n 里算正确率/退化率，那正是指纹本来要防的坑。

哪两道题改过
------------
只改过两道（L1 加中间步骤以便归因、L8 修掉"题干已给代号还问代号"的自相矛盾），
其余 6 道题干一字未动。所以剔除规则可以精确到行，不需要重跑：

  L1 旧题 = 只要求输出最终数字 → 响应是**单行**（且无运算符）
  L1 新题 = 要求三行中间步骤   → 响应**≥3 行**
  L8 旧题 = 问「代号是什么」    → 响应就是代号本身（蓝湾）
  L8 新题 = 问「做什么工作」    → 响应是工作描述

判据不靠"猜"，靠响应结构本身，且对每一条被剔除/保留的行都打印出来供复核。

用法：
    python3 reconcile_lowbit.py                     # 只报告，不写
    python3 reconcile_lowbit.py --write             # 写 lowbit_clean.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

RESULTS = os.path.expanduser("~/mtp_depth_lab/results")
SOURCES = ["lowbit.json", "lowbit_8bit.json"]

# 只有这两道题改过版本；其余 6 道 rep0 的旧行与新题等价，保留
CHANGED_TASKS = {"L1_chain_math", "L8_long_extract"}

# 每题每档的目标样本数。L2–L7 的 rep0 在两个题目版本下各跑过一次，
# 是两次**独立采样**（唯一前缀不同 → prompt 不同 → 输出可以不同），
# 计数上会变成 4，和 L1/L8 剔完后的 3 不齐。要收敛到统一 n 才能逐题横比。
TARGET_N = 3


def line_count(text: str) -> int:
    return len([l for l in (text or "").splitlines() if l.strip()])


def is_old_l1(text: str) -> bool:
    """旧题只要求输出最终数字 → 单行、且没有推导痕迹。"""
    t = (text or "").strip()
    if not t:
        return False
    if line_count(t) >= 2:
        return False
    # 单行里含 ×/=/+ 等算术痕迹的，是新题被压成一行，不算旧题
    return not re.search(r"[×x*+=]", t)


def is_old_l8(text: str) -> bool:
    """旧题问代号 → 响应就是那个代号本身（无动词短语）。"""
    t = (text or "").strip()
    if not t:
        return False
    return not re.search(r"(迁移|部署|建设|开发|上线|实施|改造|建设)", t)


def classify(row: dict) -> str:
    """返回 'current' / 'old' / 'nofp'。"""
    if row.get("task_version"):
        return "current"
    task = row.get("task")
    if task not in CHANGED_TASKS:
        # 题目没改过，无指纹也等价于当前版本
        return "equivalent"
    if task == "L1_chain_math":
        return "old" if is_old_l1(row.get("text", "")) else "current"
    if task == "L8_long_extract":
        return "old" if is_old_l8(row.get("text", "")) else "current"
    return "current"


def reevaluate(row: dict) -> dict:
    """
    用**当前** lowbit_probe 重算派生字段。

    为什么必须重算：退化阈值中途调过两次（ngram 0.08→0.55、max_sentence_repeat
    4→8）。旧行里存的 `degenerate` / `flags` 是**当时那套阈值**算的，
    拿它和 rep1/rep2（当前阈值）平均就是在平均两把不同的尺子。
    存下来的 `text` 才是原始事实，派生量全部重算。
    """
    import lowbit_probe as P
    r = dict(row)
    text = r.get("text", "")
    if not text.strip():
        return r
    try:
        ok, why = P.verify(r["task"], text)
        flags = P.loop_flags(text)
    except Exception as e:
        r["reeval_error"] = f"{type(e).__name__}: {e}"
        return r
    changed = (r.get("ok") != ok) or (r.get("degenerate") != flags["any_degenerate"])
    r["ok"] = ok
    r["why"] = why
    r["degenerate"] = flags["any_degenerate"]
    r["flags"] = {k: v for k, v in flags.items() if k != "_raw"}
    r["metrics"] = flags["_raw"]
    if changed:
        r["_reeval_changed"] = True
    return r


def thin_to_target(rows: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """
    收敛到每题每档恰好 n 次，规则必须说得清、可复现：

      1. **rep1 / rep2 无条件保留** —— 它们跑在题目定稿之后，可证明是当前题版；
      2. 剩下的名额从 rep0 里取**文件里出现的最后一次**
         （结果文件是按运行顺序追加的，同一 (tier,task,rep) 的多条里
          最后一条来自后来那次跑，也就是改完题之后的那次）；
      3. 仍不够才从更早的 rep0 往前补。

    不用「按 ctok 排序取前 n」那种规则 —— 那会随机丢到 rep2，
    数字上没错但复核时说不清丢的是哪一次。
    """
    kept, dropped = [], []
    groups: dict[tuple, list] = {}
    # 记录原始出现顺序
    for i, r in enumerate(rows):
        r["_ord"] = i
        groups.setdefault((r.get("tier"), r.get("task")), []).append(r)

    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        g = groups[key]
        by_rep: dict[int, list] = {}
        for r in g:
            by_rep.setdefault(r.get("rep", 0), []).append(r)
        # 每个 rep 内部按原始顺序排好，便于「取最后一个」
        for v in by_rep.values():
            v.sort(key=lambda r: r["_ord"])

        chosen: list[dict] = []
        for rep in sorted(by_rep, reverse=True):      # rep 大 = 跑得晚 = 一定是新题
            for r in reversed(by_rep[rep]):           # 同一 rep 里从最新的开始取
                if len(chosen) < n:
                    chosen.append(r)
            if len(chosen) >= n:
                break
        chosen_ids = {id(r) for r in chosen}
        kept += chosen
        dropped += [r for r in g if id(r) not in chosen_ids]

    for r in kept + dropped:
        r.pop("_ord", None)
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "lowbit_clean.json"))
    a = ap.parse_args()

    rows: list[dict] = []
    for f in SOURCES:
        p = os.path.join(RESULTS, f)
        if not os.path.exists(p):
            print(f"缺文件：{p}")
            return 2
        rows += json.load(open(p))

    buckets: dict[str, list] = {"current": [], "equivalent": [], "old": [], "nofp": []}
    for r in rows:
        buckets[classify(r)].append(r)

    print(f"总计 {len(rows)} 行")
    for k in ("current", "equivalent", "old", "nofp"):
        print(f"  {k:11s} {len(buckets[k]):3d}")
    print()
    print("被剔除的旧题行（每条都列出以便复核）：")
    for r in buckets["old"]:
        print(f"  {r.get('tier'):8s} rep{r.get('rep')} {r.get('task'):18s} "
              f"ok={r.get('ok')} | {str(r.get('text'))[:44]!r}")
    if not buckets["old"]:
        print("  （无）")

    keep = [reevaluate(r) for r in buckets["current"] + buckets["equivalent"]]
    n_changed = sum(1 for r in keep if r.get("_reeval_changed"))
    print(f"\n剔除旧题 {len(buckets['old'])} 行后剩 {len(keep)} 行")
    print(f"用当前阈值重算派生字段，判定发生变化 {n_changed} 行")
    for r in keep:
        if r.get("_reeval_changed"):
            print(f"  变 {r.get('tier'):8s} rep{r.get('rep')} {r.get('task'):18s} "
                  f"ok={r.get('ok')} deg={r.get('degenerate')}")

    keep, dropped = thin_to_target(keep, TARGET_N)
    print(f"\n收敛到每题每档 {TARGET_N} 次：丢弃多余 {len(dropped)} 行")
    for r in dropped:
        print(f"  丢 {r.get('tier'):8s} rep{r.get('rep')} {r.get('task'):18s} "
              f"ctok={r.get('completion_tokens')} | {str(r.get('text'))[:36]!r}")

    # 收敛后每档每题应恰好 TARGET_N 次 —— 这是完整性的自证
    print(f"\n收敛后每档每题次数（应全为 {TARGET_N}）：")
    seen: dict[tuple, int] = {}
    for r in keep:
        k = (r.get("tier"), r.get("task"))
        seen[k] = seen.get(k, 0) + 1
    bad = {k: v for k, v in seen.items() if v != TARGET_N}
    tiers = sorted({k[0] for k in seen})
    tasks = sorted({k[1] for k in seen})
    for t in tiers:
        cells = " ".join(f"{tk.split('_')[0]}={seen[(t, tk)]}" for tk in tasks)
        print(f"  {t:9s} {cells}")
    if bad:
        print(f"  ⚠️ 计数异常 {len(bad)} 格：{bad}")
    else:
        print(f"  ✅ {len(tiers)} 档 × {len(tasks)} 题 × {TARGET_N} 次，全部齐平")

    if a.write:
        json.dump(keep, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
