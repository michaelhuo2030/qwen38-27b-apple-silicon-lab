# Qwen3.8-27B 社区深挖：M2 Max 96GB 实操情报

> 调研时间：2026-10-03
> 方法：4 轮、5 类通道（通用 web 搜索中英双语 / oMLX 官方 benchmark 库直取 / GitHub issues / HuggingFace datasets / 个人博客与 Reddit 转述）
> 目的：找**别人在这台机器上真正跑出来的数字**，而不是官方宣传

---

## 0. 最重要的三个发现（直接改写实验计划）

| # | 发现 | 影响 |
|---|---|---|
| **A** | **oMLX 官方 benchmark 库里有一台和我们完全同型号的机器**（M2 Max 38c / 96GB） | 我们不是在猜，有可对照的基线 |
| **B** | **同型号基线的 TG 是 23.8–27.6 tok/s，而我之前记的 19.9–21.1 是 MTP 关闭状态** | 「这台机器 MTP 没用」的旧结论**作废**。E2 depth 实验从「验证一下」升级为**高价值必做** |
| **C** | **GPU 热降频会让 back-to-back 跑分掉 25%，且 `pmset -g therm` 完全看不见** | **所有实验必须加冷却间隔**，否则测的是温度不是模型 |

---

## 1. 同型号基线：M2 Max (38c) / 96 GB

来源：[omlx.ai/benchmarks/performance/ojps1220](https://omlx.ai/benchmarks/performance/ojps1220)（2026-08-18，v0.6.1）和 [o2n98t16](https://omlx.ai/benchmarks/performance/o2n98t16)（2026-08-21，v0.6.2），均为 `Qwen3.8-27B-oQ4e-fp16-mtp`。

### 1.1 逐 context 吞吐

| Context | PP tok/s | TG tok/s | Peak mem |
|---|---|---|---|
| 1k | 195.3 | 25.3 | 23.8 GB |
| 4k | 236.4 / **222.1** | 25.9 / **25.0** | 23.3 / 25.2 GB |
| 8k | 236.9 | 23.8 | 23.6 GB |
| 16k | 230.5 | **27.6** | 24.1 GB |
| 32k | 208.7 | 25.8 | 25.3 GB |

**注意 TG 不随 context 单调下降**（16k 反而最高 27.6），说明 MTP verify 的开销被接受率抵消了。
Peak mem 从 1k 到 32k 只涨 1.5 GB —— 和「只有 16 层有 KV、64KB/token」的理论完全吻合。

### 1.2 并发（本机唯一一份 batching 数据，v0.6.2）

| Batch | TG tok/s | Speedup |
|---|---|---|
| 1× | 25.3 | 1.00× |
| 2× | 38.2 | **1.51×** |
| 4× | 62.9 | **2.49×** |

**这和 M3 Max 128GB 的数据完全相反**（M3 Max 上 2× 只有 0.36×、4× 0.59×）。
→ **我们这台机器 batching 是正收益，E7 值得认真做。**

### 1.3 两份官方 recipe（可直接 Apply custom recipe）

**Recipe A（v0.6.1, 236.9 PP）**
```json
{"turboquant_kv_enabled": false, "turboquant_kv_bits": 4, "turboquant_skip_last": true,
 "mtp_enabled": true, "vlm_mtp_enabled": false,
 "qwen35_ane_prefill_enabled": true, "qwen35_ane_prefill_sequence_length": 1024,
 "qwen35_ane_prefill_fraction": 0.53, "qwen35_ane_prefill_max_layers": 64,
 "qwen35_ane_prefill_dual_ane": false, "qwen35_ane_prefill_gdn": true,
 "qwen35_ane_prefill_gdn_fraction": 0.5, "qwen35_ane_prefill_gdn_max_layers": 48}
```
用户备注：`prompt block 1024 - dual ANE off - GDN on`

**Recipe B（v0.6.2, 222.1 PP，带 batching 数据）**
```json
{"max_context_window": 262144, "max_tokens": 8192, "temperature": 0, "top_p": 0.1,
 "enable_thinking": true, "thinking_budget_tokens": 4096,
 "turboquant_kv_enabled": false, "turboquant_kv_bits": 4, "turboquant_skip_last": true,
 "mtp_enabled": true, "vlm_mtp_enabled": false,
 "qwen35_ane_prefill_enabled": true, "qwen35_ane_prefill_sequence_length": 2048,
 "qwen35_ane_prefill_fraction": 0.4, "qwen35_ane_prefill_max_layers": 64,
 "qwen35_ane_prefill_dual_ane": true, "qwen35_ane_prefill_gdn": true,
 "qwen35_ane_prefill_gdn_fraction": 0.53, "qwen35_ane_prefill_gdn_max_layers": 48}
```

**两份 recipe 高度一致的地方（可信度高）**：
- `turboquant_kv_enabled: false` —— TurboQuant 全关
- `mtp_enabled: true` + `vlm_mtp_enabled: false` —— 用内嵌 Lightning MTP，不用外部 drafter
- `qwen35_ane_prefill_enabled: true` —— ANE prefill 开（本机 96GB 足够，社区 64GB 反噬的结论不适用）
- `qwen35_ane_prefill_max_layers: 64` / `gdn_max_layers: 48`

**两份不一致的地方（E3 要 A/B 的正是这个）**：
`sequence_length` 1024 vs 2048、`fraction` 0.53 vs 0.4、`dual_ane` false vs true、`gdn_fraction` 0.5 vs 0.53。
（我此前记录的社区最优 `fraction=0.5 / gdn=False / dual_ane=True` 又是第三种组合 —— 三个来源三个答案，只能本机实测。）

### 1.4 我们的旧基线为什么偏低

| 来源 | TG | MTP | ANE |
|---|---|---|---|
| 我此前记录（omlx.ai y474epho） | 19.9–21.1 | ✗ | ✗ |
| 同型号官方 v0.6.1 | 23.8–27.6 | ✓ | ✓ |
| llamaperf 用户实测（mlx-serve） | 20.1 长 / 13.0 短 | ✓ | — |
| 另一用户 oQ4e + mtp（无 ANE） | 18.3 | ✓ | ✗ |

**所以 19.9–21.1 是「MTP 关 + ANE 关」的裸数字，不是这台机器的上限。**
预期 E0–E3 跑完，4-bit 档应该落在 **24–28 tok/s**。

---

## 2. MTP：Metal 上的真相（比「Mac 上 MTP 无效」复杂得多）

### 2.1 「Mac 上 MTP 变慢」这句话是**有条件的**

| 场景 | 结论 | 来源 |
|---|---|---|
| **dense** Qwen3.8-27B，oMLX，内嵌 MTP | **+178% ~ +120%** | M5 Max bf16 9.3→25.9；M2 Max mlx-serve 11.2→20.1 |
| **MoE** Qwen3.6-35B-A3B，oMLX Lightning MTP | **慢 8–12%** | [jundot/omlx#2150](https://github.com/jundot/omlx/issues/2150) |
| MoE，llama.cpp Metal `--spec-type draft-mtp` n=2 | 34.6 → 46.2（**反而快 33%**） | 同上表另一人 A/B（方向相反，说明噪声大） |
| MoE，深度越深越差 | n=2/4/6/8 = 42.0/29.4/20.1/17.1 | RaynarDM HF dataset |
| M4 24GB Mac mini（Metal） | 5.8 → 5.8（打平）；code +9-10% / prose **-22~24%** 抵消 | sudoingX/qwen38-mtp 53 组 A/B |
| M4 Max + llama.cpp | MTP **不加速反而变慢** | 一手旅行实测 |

**机制解释**（社区共识，写得最清楚的是 M2 Max 那篇）：
> 「verify n+1 个 token 的成本 ≈ verify 1 个 token —— 权重只读一次，跨行共享。
> 但这一近似在不同后端成立的程度差别巨大，而这在 Apple Silicon 上就是全部。」
> 上游 llama.cpp #23752 / #23011 给的原因是**每步 Metal kernel dispatch 开销超过投机收益**。

**对我们（dense + oMLX 内嵌 MTP）的推论**：属于「dense + 内嵌 + oMLX」这一格，是社区里收益最大的一格。**但深度必须本机扫**。

### 2.2 depth 的答案：社区互相矛盾

| 机器 | 引擎 | 最优 depth | 备注 |
|---|---|---|---|
| M4 Max 64GB | oMLX | **k=3**（44.4 vs k=2 ~40） | k=4 只 code +1.2 / prose **-5.4**；agent 短输出 k=2 接受率 80.5% > k=3 71.1% |
| M5 Max 128GB | oMLX | **depth 6** | 91.9 tok/s 全套（fresh state） |
| M5 Pro 20c | MLX/MTPLX | **depth 2**（51.34） | depth 1 = 42.98 (98.2%)、depth 3 = 39.94 (98.4/90.6/81.0%) |
| M4 Max 128GB | oMLX | **k=3** | prose 53.3 / code 72.1 |
| 125B（本仓库已实测） | oMLX | **depth 1** | 真实负载 α 69.8–87.5% |

**没有一个统一答案。E2 是必做实验，不是可选项。**

### 2.3 接受率的量级参考

- 社区通用：**75–99%**（coding/agent 流量），2.5–3.6 token/cycle
- M5 Pro 实测：depth 1 = 98.2%、depth 2 = 100%/97.4%、depth 3 = 98.4%/90.6%/81.0%
- M4 Max 64GB：57–88%（长上下文 8K–32K）
- llama.cpp Q4_K_M：65%
- 我们 125B 实测：token 加权 88.18%

### 2.4 ⚠️ 两个必须知道的 MTP 陷阱

**陷阱 1：mtp.* 张量没进 checkpoint index → MTP 静默不生效。**
> oMLX 报：`"Config declares MTP layers but the weight files contain neither mtp.* tensors nor native nextn layers."`
> `mlx-community/Qwen3.8-27B-{bf16,8bit,4bit}` 都声明了 `mtp_num_hidden_layers: 1`，但转换时丢了 15 个 `mtp.*` 张量。
> 社区**曾经**因此得出「MTPLX MTP 无收益」的错误结论 —— 实际跑的是两边都是普通自回归。
> **我们的四档全部是 Jundot/TokenAI 的 `-mtp` 产物，理论上有头。但 E0 必须验证运行时那行 `Speculative backend selected: Lightning MTP`，而不是看配置文件。**

**陷阱 2：`vlm_mtp_enabled`（外部 drafter）在长上下文会早停。**
> 「外部 drafter 模式在 prompt > 2K 接受率骤降、> 16K 直接早停（0 输出）。改为内嵌后长上下文不再早停。」
> → **必须 `vlm_mtp_enabled: false`**。我们 recipe 里已经是 false，保持。

**顺带一个 norm 陷阱**（如果要自己合头）：
> EigenLabs 的 MTP 头，norm gamma **已经 +1 shift 过**（MLX 约定），**不要再 shift 一次**。oMLX 的 sanitize 会自动做。

---

## 3. ⚠️ 方法论：热降频是隐藏变量（本次最重要的一条）

来源：[jundot/omlx#2689](https://github.com/jundot/omlx/issues/2689)（M3 Max 128GB，oQ8e-mtp 16K context，MTP+TQ 全开）

| 轮次 | TG tok/s | backbone ms/cycle | avg GPU MHz | min MHz |
|---|---|---|---|---|
| R1（冷却 5 min） | **21.9** | 129.6 | 1317 | 1050 |
| R2（紧接着连跑） | **16.4** | 161.3 | 1243 | **743** |
| R3（休息 3 min） | **22.0** | 127.8 | 1294 | 958 |

- 振幅 **-25%**（M5 Max 上更狠，-32~-39%）
- R2 期间 GPU 时钟 18 个采样点内从 1368 单调掉到 743 MHz，功耗 54W → 18W
- **`pmset -g therm` 显示 `thermal=0`，完全看不见**（只有 `powermetrics` 能看到）
- 恢复 3 分钟足够

**→ 对我们的硬性要求**：
1. 每次基准之间**强制 ≥3 min 静置**，且不 sleep、不合盖
2. 报告里必须写「每组测前冷却 ≥3 min」
3. **跨组比较只信冷却过的组**；任何明显掉速先怀疑温度，别急着下结论说「这个配置更慢」
4. 长时间连续跑（比如 depth 1→2→3→4）后要重测 depth 1 作回归基线

这条实际上也解释了 M3 Max 上 batching 0.36× 的异常数据 —— 可能是热降频污染的。

---

## 4. TurboQuant KV：社区一致「默认关」

| 说法 | 证据强度 |
|---|---|
| 「MTP + TurboQuant KV 双开掉速（verify 加速失效）」 | 官方 issue #2215 / #2782，oMLX 0.6.2 修了崩溃但**没修掉速** |
| 同型号 M2 Max 两份 recipe 都 `turboquant_kv_enabled: false` | **本机同型号** |
| llama.cpp 侧：q8_0 KV cache 基本免费（f16 58.55±0.48 vs q8_0 57.25±2.07，误差棒重叠） | M2 Max 32GB 实测 |
| vLLM/Red Hat 研究：KV 3bit 明显掉点、4bit 适度掉点 | 他人研究 |

**决策：E4 保留，但降级为「低优先级的排除性实验」**。理由：我们四档的权重合计 84.55GB，Metal cap 82GiB，本来就不可能同时常驻；KV 量化省下的几 GB 在本机换不来吞吐。

**⚠️ 新发现的硬 bug**：[jundot/omlx#3906](https://github.com/jundot/omlx/issues/3906)
> `[convert] Only length-1 arrays can be converted to Python scalars`
> 复现模型就是 **Qwen3.8-27B-oQ4e-mtp**；触发条件 **2+ 并发 + TurboQuant 开启 + max_tokens ≥ ~500**。
> 根因：MTP 投机回滚把**向量** trim 长度喂给只接受标量的 `turboquant.trim()`。
> → **E7 并发实验必须默认 `turboquant_kv_enabled: false`**（本来也要关），并且如果开了 TQ 又要并发，立刻怀疑这条。

---

## 5. ANE prefill：本机该开（但比例待定）

- 64GB 机型社区结论是**反噬**（ANE banks ~13GB 与 KV 争内存 → shed + 节流，PR #3103：57K prompt 下 ANE on 217/177 vs off 326/300）
- **我们是 96GB，同型号 recipe 两份都开着 ANE，且 0.6.1 recipe 的备注明确写了 GDN on**
- 我们 ANE 只有 16 核 15.8 TOPS，预期收益远小于 M4 Max 的 +11% decode / 3.3× prefill
- oMLX 0.6.2 起有**内置 ANE/GPU split tuner**，可以在本机直接扫比例
- oMLX 0.7.0 dev 版有 `qwen35_ane_prefill_cpu_enabled`（CPU 侧补 prefill），**需要确认 0.7.0 stable 是否包含**

**E3 设计调整**：不只测 on/off，还要测 `fraction` / `dual_ane` / `sequence_length` 三个维度的两个候选值（Recipe A vs Recipe B）。

---

## 6. oMLX 0.7.0 相对 0.6.x 变了什么（我们已在 0.7.0）

来自 [v0.7.0 release notes](https://github.com/jundot/omlx/releases/tag/v0.7.0) 和 0.7.0rc1：

| 变化 | 对我们的意义 |
|---|---|
| **Exact Lightning MTP**：单请求下 verify 行与串行 decode **bit-identical** | ⭐ **E1 哨兵判据可以放宽** —— 0.7.0 上 MTP on/off 应当输出完全一致。我们的 `sentinel()` 应该测到「完全一致」，若不一致说明有问题 |
| Qwen3.8 的 Lightning MTP / DFlash2 每个 verify 周期做更少的事，**长上下文收益最大** | depth 实验要覆盖 16K/32K，不能只测短 |
| **内存守卫完全重写**：safe 留 ~20% RAM / balanced ~8% / aggressive ~2% | 我们 96GB，aggressive 可能可用；但**待测**，不要默认 |
| Partial Block Caching：不再因尾部不填满 block 而重算上千 token | 长会话 TTFT 改善 |
| 提前释放首个生成块 | TTFT 改善 |
| GPU keep-warm：请求后保持 GPU 非 idle 最多 5 分钟（`server.gpu_keep_warm_interval`） | ⭐ **这会干扰我们的热降频实验**！需要显式关掉或纳入记录 |
| 每请求 tokenizer 优化：Qwen3.8 上省 ~45ms | 计入 TTFT 口径时注意 |
| 0.7.0dev1/dev2 有个已修 bug：**静默丢弃未注册函数名的 tool call**（#3660） | 0.6.4 不受影响；我们在 0.7.0 stable，**做 agent/tool-call 测试时必须验一条真 tool call** |

**已知 0.7.0 问题**：[#3917](https://github.com/jundot/omlx/issues/3917) —— 0.6.4/macOS 26.6 → 0.7.0rc1/macOS 27，`Qwen3.8-27B-oQ6e-mtp` 最大 context 从 ~128k 掉到 ~70–90k。
我们在 **macOS 26**，大概率不受影响，但 **E4/E5 要顺手记一下 max context 边界，别等要用了才发现**。

---

## 7. Prefix Cache：真实使用中最大的那个杠杆

比任何量化选择都大（M4 Max 128GB 五方案实测）：

| 现象 | 数字 |
|---|---|
| 256K 冷启动 prefill | **~40 分钟** |
| 256K 热缓存 | **~2 分钟** |
| 128K 冷 | 68.6s → 同请求第 2 次命中 16,384 token，只重算 322 token，**2.5s** |
| **tools 段仅顺序交换** | 命中 0 → 全量重算 **81.4s** |
| 128K 中间改内容 | 只有 oMLX 能复用约一半，mlx-dspark 0.39，**MTPLX 直接归零** |

**机制**：按 token 精确匹配，粒度 = `block_size 4096`；`tools` 被渲染进 prompt 前缀（system 之后）；tools 任何变化（含顺序）→ 从该处失配。
**另一个坑**：oMLX 对「完全命中」（prompt 与缓存序列完全相等）的 stateful 缓存会**回退全量 prefill** —— agent 每轮必须有增长的新尾巴。

**对本仓库的直接影响**：`src/lab_client.py` / `src/ask.py` 的基准脚本**必须用唯一前缀**，否则测的是缓存命中率不是模型速度。上一轮 125B 的实验是「唯一前缀」做的，27B 继续保持。

---

## 8. thinking 模式：本地必须关

| 来源 | 数字 |
|---|---|
| 默认 reasoning effort = **xhigh**，think by default | Willison 的 pelican 测试：22,000 reasoning token → 3,200 output，**烧 21 分钟** |
| effort 档位本质 | 不是架构开关，是**一句话 system prompt 后缀**：`medium` 注入空（= 原生行为），`xhigh` 注入 "think carefully" |
| 实用建议 | agent loop 用 `medium`（等待减 1/3，质量无损）或 `low`；`reasoning_budget ≈ 5000 tokens` 止住失控思考 |
| Ollama 的坑 | 默认 tag `qwen3.8:27b` 带 `draft_num_predict: 4`（= MTP tag），GGUF 路径在 Apple 上是**净亏损**（5.14 vs 11.78 tok/s） |
| oMLX 的坑 | 忽略 Anthropic 的 `output_config.effort`，会 fallback 到 `reasoning_effort=xhigh` |

**→ 验证我们此前的决定**：`think_off` 为主实验是**对的**，且现在有了更强的理由（xhigh 一轮 20K thinking token 在 25 tok/s 上等于 13 分钟）。

**注意与社区的差异**：同型号 oMLX recipe B 用的是 `enable_thinking: true` + `thinking_budget_tokens: 4096`。
我们主实验用 think_off，但**应该补一组 thinking ON + budget 4096**，因为那是社区在本型号上跑出 222 PP tok/s 的配置。

---

## 9. 三值 Bonsai 2 的真实口碑：agentic 任务是重灾区

我们下的是 `TokenAI-zer/Ternary-Bonsai-2-27B-MLX-oQ3-mtp`（**不是**官方 ternary，是 stock MLX 重压版），所以下面部分不适用，但质量预期要下调：

| 独立测试结论 | 具体 |
|---|---|
| **AGI Hunt** | agentic loop 建 ISS 追踪器：**Bonsai 连续 114 次搜索，一行代码都没写**；全精度模型正常交付 |
| MindStudio | 视觉描述**通过**（明确不优于原模型）；多语言翻译**灾难性**（Tamil 上死循环，且比上一代 Bonsai 更差）；debug 能展示推理但常得错误结论 |
| blog.code-tw | 实测 14–23 tok/s（官方 143）；同任务 27 min vs 官方 5–15 min |
| modemguides | 98.2% 是 **xhigh 最高 effort 下的 20 项均值**；medium effort 是 **96.0%**；两项 agentic coding benchmark 只有 **~75%** |
| newshunt | 数学 99.5% / 代码 99.3% / 指令 +1.7% / agentic 97.3% / 知识 96.9% / 视觉 96.3% |
| HN | 「短自由输出不错，做 coding agent 令人失望」；Metal fork 有 tensor API 问题 |

**→ 对我们的 E5 质量实验的预期调整**：
1. ternary 档**大概率会在我们的 7 个代码任务上失分**（不是量化问题，是 agentic 能力问题）
2. 这**不是**「三值量化不可用」的证据 —— 官方数据说数学/代码保留 99%+。差异来自：官方是**原生三值训练**（PrismML 从头训），我们下的是**在已量化 checkpoint 上再压**的 oQ3
3. 所以 E5 的正确读法是：「**stock MLX 路线拿不到 Bonsai 2 的质量**」，而不是「三值不行」
4. **不要用 ternary 档做 agent 主力**（社区一致），它适合做常驻小模型

---

## 10. 容量与 KV（复核我们的计算，全部对得上）

| 项 | 数字 | 状态 |
|---|---|---|
| KV cache | **64 KB/token**（只有 16 层全注意力） | ✅ 已算入，官方 1k→32k 内存只涨 1.5GB 印证 |
| GDN 递归状态 | 48 层 × 1 序列 = **0.143 GiB**，与 context 无关 | ✅ |
| 8-bit 权重 | 29.50 GB → M2 Max 纯 AR 上限 **~13.6 tok/s** | 带宽 400 GB/s ÷ 29.5 GB |
| 4-bit 权重 | 16.05 GB → 纯 AR 上限 **~24.9 tok/s** | 同上 |
| 官方实测 4-bit TG | 23.8–27.6 | ⭐ **几乎打满带宽上限** |
| 官方实测 8-bit | 未见 M2 Max 数据 | 预期 ~13–18（带 MTP） |
| 三值 MLX 8.595 GB | 理论上限 ~46 tok/s | ⚠️ 官方给 M5 Max 46.8，M2 Max 需实测 |
| 262K 满 context bf16 KV | 16 GB | 与 4-bit 权重量级相当 |
| M2 Max 不分带宽档 | 64GB 版也是 400 GB/s（不像 M3/M4 Max 分档） | ✅ 我们的 409.6 GB/s 一致 |

**反推校验**：25 tok/s × 16.05 GB = 401 GB/s ≈ 我们实测的 409.6 GB/s。
**→ 4-bit 档在 M2 Max 上已经打到带宽墙了。想更快只能靠 MTP（减少访存次数），不能靠更好的 kernel。**
这也解释了为什么 8-bit 会慢一半（29.5/16.05 = 1.84×，约 0.54× 速度，和之前记录吻合）。

---

## 11. 变更清单（对 `TEST_PLAN_27B.md` 的补丁）

| 编号 | 变更 |
|---|---|
| **M1** | 所有基准之间**强制 ≥3 min 冷却**；报告注明；跨组只信冷却组 |
| **M2** | **显式关闭或记录** `server.gpu_keep_warm_interval`（0.7.0 新增，会污染热降频实验） |
| **M3** | E2 depth 升为**必做高价值**，覆盖 1/2/3/4 × {1K, 16K, 32K}，长上下文是 0.7.0 的收益区 |
| **M4** | E1 哨兵判据更新：0.7.0 的 Exact Lightning MTP 下，**MTP on/off 应 bit-identical**；不一致即报 bug |
| **M5** | E3 ANE 增加 Recipe A vs Recipe B 的 `fraction`/`dual_ane`/`sequence_length` 三维 A/B |
| **M6** | E4 TurboQuant 降级为排除性实验；**E7 并发期间强制 `turboquant_kv_enabled: false`**（规避 #3906） |
| **M7** | 新增 **E8：prefix cache 命中实验**（唯一前缀 vs 重复前缀 vs tools 重排）—— 真实使用中收益最大 |
| **M8** | 新增 thinking ON + `thinking_budget_tokens: 4096` 对照组（社区在本型号跑出最高 PP 的配置） |
| **M9** | E0 增加两条激活验证：运行时日志出现 `Speculative backend selected: Lightning MTP`；`vlm_mtp_enabled == false` |
| **M10** | E7 预期值改用同型号官方基线：**1× 25.3 / 2× 38.2 / 4× 62.9**（旧预期 21.1/69.4/136.9 来自 MTP 关闭状态） |
| **M11** | 记录 max context 边界（规避 #3917 类回归） |
| **M12** | 任何 tool-call 路径测试前，先验一条真 tool call（规避 0.7.0dev 已修的 #3660 类问题） |

---

## 12. 还没解决 / 仍然不确定

1. **三值 oQ3 在 M2 Max 的实际 tok/s** —— 带宽上限 ~46 tok/s，但 oQ3 的 decode 是否真的 memory-bound？社区在 llama.cpp 上测的 14–23 是官方三值内核，不能外推
2. **本机 GPU keep-warm 关掉后的真实热降频曲线** —— 只能自己测
3. ~~**0.7.0 是否含 `qwen35_ane_prefill_cpu_enabled`**~~ → ✅ **含**。本地源码已确认，字段全套在：
   `qwen35_ane_prefill_cpu_enabled` / `_fraction` / `_down_fraction` / `_gdn_fraction` / `_threads` / `_shared_resource`
   （出现在 `model_settings.py` / `model_profiles.py` / `admin/benchmark.py` / `admin/ane_tuning.py` / `engine/batched.py` / `engine/vlm.py`）
   → **E3 可以把「CPU 侧补 prefill」作为第三个维度纳入 A/B**
4. ~~**0.7.0 是否修了 #3906**~~ → ❌ **没修**。本地 `turboquant_kv.py:281` 的 `def trim(self, n)` 仍是
   ```python
   position = self._idx
   n = min(position, n)      # 没有标量强转，向量 n 仍会炸
   ```
   → **M6 从「预防措施」升级为「硬性禁止」：任何并发测量期间 TQ 必须关**
5. **8-bit 档在 M2 Max 的 MTP 收益** —— 带宽受限下 MTP 的杠杆应该更大（访存 29.5GB 每次），社区没数据
6. **turboquant_kv_bits 对混合架构是否正确** —— 仍是原计划的头号未知（`qwen3_5 → "gdn-qk-norm-2"` 映射错了不报错只输出变差）

### 12.1 本机已确认的环境事实（省得再查）

| 项 | 值 | 说明 |
|---|---|---|
| **服务端口** | **8091**（不是社区文档里的 8000） | 仓库里 6 个脚本全部已用 8091，无需改 |
| oMLX 版本 | 0.7.0（build 260930235148-macos15-sequoia） | |
| `wired_limit_mb` | 88000 | 与 85.9GB Metal cap 一致 |
| **`server.gpu_keep_warm_interval`** | **0.5** | ⚠️ **已启用**。做热降频实验前必须显式设 0，否则 P18 的对策失效 |
| `scheduler.max_concurrent_requests` | 8 | E7 的上限要够 |
| `memory.memory_guard_tier` | balanced（约留 8% RAM） | 0.7.0 重写后的语义；aggressive（约留 2%）可作为 E4 的一个变量 |
| 鉴权 | `/api/status` 无 key 返回 **401** | 诊断命令必须带 Bearer 或 admin cookie |

---

## 来源清单

- oMLX 官方 benchmark：<https://omlx.ai/benchmarks/performance/ojps1220>、<https://omlx.ai/benchmarks/performance/o2n98t16>
- oMLX v0.7.0 release：<https://github.com/jundot/omlx/releases/tag/v0.7.0>
- 热降频实测：<https://github.com/jundot/omlx/issues/2689>
- MoE 上 MTP 掉速：<https://github.com/jundot/omlx/issues/2150>
- TQ+MTP 崩溃/掉速：<https://github.com/jundot/omlx/issues/2782>
- TQ+MTP+并发崩溃 #3906：<https://github.com/jundot/omlx/issues/3906>
- 0.7.0 max context 回归 #3917：<https://github.com/jundot/omlx/issues/3917>
- M2 Max 负面结果集（非常有用）：<https://huggingface.co/datasets/RaynarDM/apple-silicon-llm-benchmarks>
- MTP 合头方法论：<https://huggingface.co/datasets/bluehawana/qwen3.8-27b-apple-silicon-concurrency/blob/main/MTP.md>
- M4 Max 64GB oMLX 全套实操：<https://eogee.com/article/137>
- M4 Max 128GB 五方案（prefix cache）：<https://so.html5.qq.com/page/real/search_news?docid=70000021_0536a95011b35552>
- M2 Max 96GB mlx-serve vs LM Studio：<https://llamaperf.com/gpu/m2-max-96gb>
- M2 Max 带宽/内存核算：<https://smeltcore.com/recipes/qwen3-8-27b-on-apple-m2-max-8-bit-mlx-vision-language-with-mtp-speculative-decoding>
- M5 Pro depth 扫描：<https://github.com/Upinel/UpinelAIOS/blob/main/docs/MLX-TUNING.md>
- Bonsai 2 独立评测：<https://agihunt.info/en/p/1a0e6d35dc3c21259b534f8e5c5>、<https://www.mindstudio.ai/blog/bonsai-2-27b-real-world-test>、<https://www.modemguides.com/blogs/modemguides-blog/run-bonsai-2-27b-locally>
- 过thinking 税：<https://www.promptgenius.net/blog/qwen3.8-27b-local-coding>
