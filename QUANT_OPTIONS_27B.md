# Qwen3.8-27B 量化方案全景：我们在用什么、还有什么更好的

> 查证时间：2026-10-03 17:50–18:10
> 方法：HuggingFace API 直查（`/api/models`、原始 `config.json`）+ oMLX 官方文档 + deepwiki + 社区对比
> 机器：M2 Max 96GB / Metal cap 85.9GB / oMLX 0.7.0

---

## 0. 先回答那个最基本的问题：Jundot 是什么

**「Jundot」就是 oMLX 的维护者本人，oQ 是 oMLX 自带的官方量化器。**

- `Jundot` = oMLX 的作者（github.com/jundot/omlx）
- **oQ** = **o**MLX **Q**uantization，官方文档 `docs/oQ_Quantization.md`
- 我们用的 `Qwen3.8-27B-oQ4e-mtp` 等四档，**全部是 oMLX 官方自己量化、自己发布的**

**这一点很重要，因为它改变了「要不要换量化方案」这个问题的性质**：
不是「社区第三方量化 vs 另一个第三方量化」，而是
「**oMLX 官方量化 vs 非 oQ 的其他路线**」。

### oQ 到底做了什么（不是朴素 round-to-nearest）

oQ 的核心是**数据驱动的混合精度分配**：

1. 用标定集跑一遍模型，**实测每一层对量化误差的敏感度**
   `sensitivity = MSE(float_out, quant_out) / mean(float_out²)`
2. 按敏感度分配位宽，**不是按固定规则或张量类型**
3. 强制保护：`lm_head`、MoE router、vision encoder 不降精度
4. 产出**标准 MLX 格式**，不需要任何自定义 loader

官方在 Qwen3.5-35B-A3B 上的对比（vs mlx-lm 原生 Q）：

| | 2-bit MMLU | 3-bit MMLU | 4-bit MMLU |
|---|---|---|---|
| mlx-lm 原生 | 14.0% | 76.3% | 79.7% |
| **oQ** | **64.0%** | **85.0%** | **83.3%** |

**位宽越低，oQ 的优势越大**（2-bit 差 50 个点）。这正是低比特场景的关键。

### 本机四档的真实位宽（不是标称值）

`config.json` 里 oQ 做了**逐模块位宽覆盖**，所以标称位宽不等于实际：

| 档位 | 标称 | mode | group | 逐模块覆盖 | 实测体积 |
|---|---|---|---|---|---|
| oQ3 | 3-bit | affine | 64 | **92 个模块 5-bit、7 个 4-bit、1 个 6-bit** | 13.86 GB |
| oQ4e | 4-bit | affine | 64 | **166 个模块全部 5-bit** | 16.97 GB |
| oQ6e | 6-bit | affine | 64 | 35 个模块 8-bit | 23.72 GB |
| oQ8e | 8-bit | affine→mxfp8 | 64/32 | 无 | 30.00 GB |

**⚠️ 这条修正了一个可能的误解**：oQ3 并不是全局 3-bit。
Gated DeltaNet 的 `in_proj_a/b`、`out_proj` 被钉在 5-bit（保稳定性），
`lm_head` 在 6-bit。所以**「ternary 13.86GB vs 4bit 16.97GB」的真实位宽差距，
比 3 vs 4 要小得多**。这也部分解释了为什么我们实测四档质量几乎一样。

`oq_imatrix_report.json` 显示标定用的是 `oqe_code_multilingual` 数据集、
128 样本、seq 512、adaptive 模式，504 个 entry 覆盖 615 个模块。

---

## 1. 全景：MLX 路线上的所有可选方案

按「能不能被 oMLX 加载」和「质量」两个维度整理：

| 方案 | 格式 | 位宽/组 | 体积 | 可否 oMLX | 质量定位 |
|---|---|---|---|---|---|
| **Jundot oQ4e-mtp** ✅已有 | affine | 4/64 +166@5bit | 16.97GB | ✅ 已验证 | **官方混合精度，本档最优** |
| Jundot oQ6e-mtp ✅已有 | affine | 6/64 +35@8bit | 23.72GB | ✅ 已验证 | 近无损 |
| Jundot oQ8e-mtp ✅已有 | mxfp8 | 8/32 | 30.00GB | ✅ 已验证 | 近无损 |
| TokenAI-zer oQ3 ✅已有 | affine | 3/64 | 13.86GB | ✅ 已验证 | 再压的 oQ3 |
| **d9beuD oQ3e-mtp** ⬅️ 缺档 | affine | 3/64 +146@5bit+8@4bit+1@6bit | ~14GB | 待验 | **同流水线真正的 3-bit** |
| **majentik MLX-5bit** ⬅️ 缺档 | affine | 5/64 均匀 | ~20GB | 待验 | 朴素 5-bit，无标定 |
| Jundot oQ4e-fp16-mtp | affine | 4/64 | ~17GB | 待验 | 敏感部分留 fp16 |
| lmstudio-community MLX-4bit | affine | 4/64 均匀 | 16.08GB | 大概率 ✅ | 朴素 RTN，**劣于 oQ4e** |
| lmstudio-community MLX-6bit | affine | 6/64 均匀 | 22.80GB | 大概率 ✅ | 朴素 RTN |
| mlx-community mxfp4 | **mxfp4** | 4/32 | 15.24GB | 格式支持，但本机未验 | FP4 + 共享 E8M0 指数 |
| mlx-community OptiQ-4bit | affine | 混合 237@4bit + **261@8bit** | 20.66GB | ⚠️ 官方说不保证 | ~8.6 bpw，与 oQ8e 同档 |
| nathansutton Ternary-Bonsai-2-MLX | affine | 2/128 **+ hadamard.json** | — | ❌ **陷阱** | 需 PrismML 旋转 |
| mlx-community Qwen3.8-27B-MTP-4bit | — | — | **0.27GB** | — | **只是 drafter 头，不是完整模型** |

---

## 2. 逐条排除，并说清为什么

### ❌ 换回朴素 mlx-lm Q4/Q6 —— 是降级不是升级

`lmstudio-community`（424 万下载）和 `mlx-community` 的 MLX-4bit/6bit
都是**均匀位宽、无标定**的 RTN。官方自己的数据：4-bit 时 oQ 83.3% vs mlx-lm 79.7%。
**下载量高 ≠ 质量好**，前者是 LM Studio 的默认产物。

### ❌ OptiQ-4bit —— 位宽等价，oMLX 还不保证能跑

实测 `config.json`：237 个模块 4-bit、**261 个模块 8-bit**，体积 20.66GB ≈ 8.6 bpw，
和我们的 oQ8e（30GB）基本同档。它自己的文档明确写：
> Other Mac front-ends such as mlx-vlm, LM Studio, and **oMLX** load MLX weights
> through their own stack... whether an OptiQ quant runs there depends on that stack

而且它自带 `optiq serve --mtp`，和 oMLX 的 Lightning MTP 是**竞争关系**。
**要换就得整套换出 oMLX**，不是加一个模型的事。

### ⚠️ 官方原生三值（Bonsai 2）—— 真正的陷阱，比加载失败更危险

`nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX` 的 config 是
`mode: affine, bits: 2, group: 128`——**看起来就是标准 MLX 格式，oMLX 很可能直接加载成功**。
但仓库里有 **`hadamard.json`**：PrismML 的旋转矩阵。

**风险**：oMLX 不认这个文件 → 权重按原样加载、**旋转没做** → **静默产出垃圾**，
不报错、不崩溃，只是答案变胡话。

> **这条修正我之前在 `LOWRESULTS_27B.md §2.1` 里的表述**：
> 我说「官方三值不可直接用于 oMLX」，实质结论对，但说得太轻。
> 准确说法是：**它的 config 具有欺骗性，oMLX 极可能静默加载出错结果**——
> 这比「拒绝加载」危险一个数量级。如果哪天要试，**必须先跑 4 条哨兵验证输出**。

### ❌ mlx-community 的 `-MTP-4bit` —— 它不是模型

体积 **0.27GB**。这是给**没有 MTP 头的模型**配的独立 drafter，
不能单独当模型用。名字容易误解。

---

## 3. 真正的「更好」在哪：oQ+

oQ 文档里有一条我们**没用上**的东西：

> **oQ+ (Enhanced)**: 1. Load model 2. Measure per-layer sensitivity
> 3. Build budget plan 4. **GPTQ weight optimization (all quantizable weights)**
> 5. Quantize with mixed-precision predicate 6. Save
>
> The GPTQ step uses **Hessian-based error compensation** to optimize rounding
> decisions for every quantizable weight.

**oQ+ 在相同 bpw 下严格优于 oQ**——它多做一步 GPTQ 权重优化。
oMLX 0.7.0 内置了 `OQManager`（`omlx/oq.py` + admin 面板），
**理论上可以自己把 BF16 基座量化成 oQ+**。

**但对 27B 有个硬约束**：需要先下载 `Qwen/Qwen3.8-27B` 的 BF16 原始权重（约 54GB），
本机 96GB 内存要同时装下「BF16 全精度 + 量化中间态」很吃紧，
且量化本身要跑标定 + Hessian 优化，**数小时级**。
所以：**27B 的 oQ+ 目前社区没有现成产物，自制成本过高。**

---

## 4. 结论与建议

### 我们的四档选型**没有错**，但阶梯**缺了两级**

| 档位 | 状态 | 说明 |
|---|---|---|
| oQ3（TokenAI-zer 再压） | ✅ 已有 | 覆盖模块比真 oQ3e 少（100 vs 155） |
| **oQ3e（d9beuD）** | ⬜ **建议补** | 同流水线真 3-bit，能补上「3 vs 4」的真实差距 |
| oQ4e | ✅ 已有 | 本档最优 |
| **5-bit** | ⬜ 可选 | majentik 均匀 5bit / dicksondickson oQ5e（但那个是 Swift 微调版，不可比） |
| oQ6e | ✅ 已有 | |
| oQ8e | ✅ 已有 | |

**当前实测最大的解释力缺口**：oQ3 标称 3-bit，但 92 个模块被钉在 5-bit。
**它和 oQ4e 的真实差距比标称小得多**——这很可能就是「四档质量几乎一样」
的一个未排除的原因。**补 `d9beuD/oQ3e`（155 个模块有覆盖）能直接检验这一点。**

### 三条具体建议

1. **补 `d9beuD/Qwen3.8-27B-oQ3e-mtp`**（~14GB）→ 补齐同流水线 3-bit 缺口。
   下载量只有 231，**基本没被测过**——正好是我们能提供独立数据的地方。
2. **不要换掉 oQ4e**。朴素 RTN 的 lmstudio/mlx-community 版本在官方数据里
   明确更差，换过去是降级。
3. **任何新模型入库，第一件事是跑 4 条哨兵**（`ladder27.sentinel()`）。
   尤其对带 `hadamard.json` / 旋转文件 / 自定义 config 的产物——
   **静默错误比加载失败难发现得多**。

### 一条已固化的检查项

> **看到一个「看起来是标准 MLX 格式」的模型仓库时，先 `curl` 它的文件列表，
> 看有没有 `hadamard.json` / `*_rotation*` / `dflash/` 之类的非标准文件。**
> 有 → oMLX 大概率会**静默加载但不应用**。这是比版本不兼容更危险的失败模式，
> 而且**从 config.json 看不出来**。

---

## 5. 引用来源

- oQ 官方文档：<https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md>
- oQ 实现解析：<https://deepwiki.com/jundot/omlx/11-model-quantization-(oq)>
- MLX 量化格式横评（KLD 对比）：<https://specpicks.com/reviews/llm-quantization-formats-kld-comparison-2026>
- OptiQ 兼容性自述：<https://mlx-optiq.com/docs/faq>
- 本机数据：`~/.omlx/models/*/config.json`、`oq_imatrix_report.json`
