#!/usr/bin/env python3
"""
memgate.py — 实验前的内存闸门。

## 为什么必须有这个闸门

2026-10-03 23:42 实测到的现场（不是假想）：

    PhysMem: 89G used (24G wired, 3619M compressor), 6364M unused
    vm.swapusage: total = 21504.00M  used = 20044.81M  free = 1459.19M
    Load Avg: 10.45 9.85 9.42
    omlx-server 17.58GB，其余全部进程合计 36.9GB

**swap 已经吃掉 20GB、只剩 1.4GB 空闲，load 10.45。**
在这种状态下再加并发压测 = 有真实的崩溃风险。
而且 omlx-server 是 **root 运行**，崩了 agent 自己救不回来
（`ps` 里 kill 不掉，之前踩过）。

所以并发实验**必须先过闸**：内存/换页/负载不达标就拒绝跑，
而不是"跑着看看会不会崩"。

## 三条判据（任一不过就拒绝）

1. **内存压力** `memory_pressure` 的 free percentage ≥ 25%
2. **swap 可用** ≥ 4GB 且 used < 50% total
3. **load average**（1 分钟）≤ 8

`--dry-run` 只报告不拦，用于随时体检。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as e:                      # noqa: BLE001
        return f"__ERR__ {type(e).__name__}: {e}"


def mem_free_pct() -> float | None:
    """系统级可用内存百分比。低于 25% 视为危险。"""
    out = _sh(["memory_pressure"])
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def swap_info() -> dict | None:
    """{total_mb, used_mb, free_mb, used_pct}"""
    out = _sh(["sysctl", "vm.swapusage"])
    m = re.search(r"total\s*=\s*([\d.]+)M.*?used\s*=\s*([\d.]+)M.*?"
                  r"free\s*=\s*([\d.]+)M", out, re.S)
    if not m:
        return None
    t, u, f = (float(m.group(i)) for i in (1, 2, 3))
    return {"total_mb": t, "used_mb": u, "free_mb": f,
            "used_pct": round(u / t * 100, 1) if t else 0.0}


def load_avg() -> float | None:
    out = _sh(["sysctl", "-n", "vm.loadavg"])
    m = re.search(r"\{\s*([\d.]+)", out)
    return float(m.group(1)) if m else None


def top_rss_gb(n: int = 5) -> list[dict]:
    """RSS 最大的 n 个进程，用来告诉用户是谁在占。"""
    out = _sh(["ps", "-Ao", "rss=,comm="])
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            rows.append({"rss_gb": round(int(parts[0]) / 1048576, 2),
                         "proc": parts[1]})
    rows.sort(key=lambda r: -r["rss_gb"])
    return rows[:n]


def check(strict: bool = True) -> tuple[bool, list[str], dict]:
    """返回 (是否放行, 理由列表, 原始数据)。strict=False 只报告不拦。"""
    free_pct = mem_free_pct()
    sw = swap_info()
    la = load_avg()
    reasons: list[str] = []

    if free_pct is None:
        reasons.append("读不到内存压力（memory_pressure 无输出）")
    elif free_pct < 25:
        reasons.append(f"内存可用仅 {free_pct}% < 25%")

    if sw is None:
        reasons.append("读不到 swap 状态")
    else:
        if sw["free_mb"] < 4096:
            reasons.append(f"swap 仅剩 {sw['free_mb']:.0f}MB < 4GB")
        if sw["used_pct"] > 50:
            reasons.append(f"swap 已用 {sw['used_pct']:.0f}% > 50%")

    if la is None:
        reasons.append("读不到 load average")
    elif la > 8:
        reasons.append(f"load average {la:.2f} > 8")

    data = {"mem_free_pct": free_pct, "swap": sw, "load1": la,
            "top_rss": top_rss_gb(5)}
    ok = (not reasons) or not strict
    return ok, reasons, data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只体检不拦截（退出码恒 0）")
    a = ap.parse_args()

    ok, reasons, data = check(strict=not a.dry_run)

    sw = data["swap"] or {}
    print("=== 内存体检 ===")
    print(f"  可用内存   {data['mem_free_pct']}%   (要求 ≥25%)")
    print(f"  swap       已用 {sw.get('used_mb', '?')}MB / 总 {sw.get('total_mb','?')}MB "
          f"({sw.get('used_pct','?')}%)   剩 {sw.get('free_mb','?')}MB   (要求剩≥4GB 且 用<50%)")
    print(f"  load avg   {data['load1']}   (要求 ≤8)")
    print("  内存占用 Top5:")
    for r in data["top_rss"]:
        print(f"    {r['rss_gb']:6.2f} GB  {r['proc']}")

    if a.dry_run:
        print("\n[dry-run] 不拦截。" + ("（注意：实测不达标）" if reasons else "（达标）"))
        return 0
    if ok:
        print("\n✅ 放行：可以跑并发压测")
        return 0
    print("\n❌ 拒绝启动，原因：")
    for r in reasons:
        print(f"   - {r}")
    print("\n建议：关掉占内存的聊天/浏览器/IDE，等 swap 回落再跑。")
    print("      omlx-server 是 root 运行，崩了 agent 救不回来。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
