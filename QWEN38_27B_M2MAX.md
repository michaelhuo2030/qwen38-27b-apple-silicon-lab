# Qwen3.8-27B 在 M2 Max 96GB 上：三种精度产物 + 加速方案（含社区实测）

> 调研：2026-09｜机器：**Apple M2 Max / 96GB / 38 GPU 核 / 409.6 GB/s 带宽**
> 运行时：**oMLX 0.7.0**（`/Applications/oMLX.app`），本机实测

---

## 0. 结论先行

1. **本机 oMLX 0.7.0 已内置 Qwen3.8 全套加速链路，且 native kernel 加载成功。**
   `GET /api/status` 实测：`qwen35_prefill: {available: true, import_error: null}`、`ane_prefill: {patch_available: true}`。
   **ANE 预填开箱可用**——社区教程里"要手动从 drowzeys 仓库装 ANE kernel"是 oMLX 0.6.x 的坑，0.7.0 已经不用了。

2. **同一台机器（38c / 96GB）上有社区实测基线**（omlx.ai，`Qwen3.8-27B-MLX-4bit`，oMLX v0.6.1，**MTP 关 / ANE 关**）：
   **单流 19.9–21.1 tok/s，prefill 117–126 tok/s，峰值内存 15.8–19.3GB，GPU 占用 99.9%**。
   打开 MTP 后按 tok/cycle 2.5–3.76 折算，**单流约 50–79 tok/s**。

3. **你要下的模型应该是 `mlx-community/Qwen3.8-27B-4bit`（16.05GB，147,514 下载 / 186 赞）**——
   社区标准件，也是 omlx.ai 上那条 M2 Max 基线用的同一个模型。

4. **M2 Max 没有 INT8 矩阵单元，本机 INT8 相关的一切宣传收益为 0**（M1/M2/M3 的 INT8 与 FP16 打平 1.00–1.01×，M4 才开始 1.88×）。

5. **精度怎么选（详见 §5.5）**：
   - **8-bit 质量优势在统计上不显著，速度慢一半 → 不选。**
   - **BF16 也没有可测优势**（oQ8 的 KL 已到 0.00008，"与 bf16 不可区分"）→ 不选，BF16 只是基准参照。
   - **拐点是 6-bit**：4→6 之间 KL 差 20 倍，6→8 只差 9 倍。
   - **但多模型 pipeline 会反过来**：三值（Bonsai 2 官方 MLX 仅 8.595GB）能多装 3 台、单流还快 1.88×。
   - 决定性因素是 **pipeline 串行还是真并发**——**真并发时模型越小越对，因为内存带宽只有 409.6 GB/s 一份且守恒**。

6. **三值是 Prism ML 的 Bonsai 2**（不是 penkia 的 TernaryQuench——下载量差 6500 倍）。
   本机 oMLX **已预装 bonsai native kernel**（`/api/status` 报 `available: true`），
   但官方 MLX 版需要自带 runtime（Hadamard 旋转），要走 stock oMLX 得用 `TokenAI-zer/...-oQ*-mtp` 这条再量化路线。

---

## 1. 模型本身（官方 config 逐项核对）

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)，2026-08-14 开源，Apache 2.0。

| 项 | 值 |
|---|---|
| 架构 | `Qwen3_5ForConditionalGeneration`，`model_type: qwen3_5` |
| 形态 | **27B dense（非 MoE）**，原生多模态（图 + 视频） |
| 层 | 64 层 = 16 × (3× Gated DeltaNet + 1× Gated Attention) → **48 线性 + 16 全注意力** |
| hidden / FFN | 5120 / 17408，`vocab 248320`，`tie_word_embeddings: false` |
| 全注意力 | 24 Q 头 / **4 KV 头**（GQA 6:1），head_dim 256 |
| 线性注意力 | 16 KV 头 × 128 / 48 V 头 × 128，conv kernel 4 |
| MTP | `mtp_num_hidden_layers: 1`（模型自带单步 MTP 头） |
| 上下文 | 262,144 原生，YaRN 可外推 1M |
| 思考控制 | `reasoning_effort` = xhigh（默认）/ medium / low |

**官方推荐采样参数（社区多处引用，本仓库实验需注意口径）**：
- 思考模式：`temperature 1.0 / top_p 0.95 / top_k 20 / presence_penalty 0.0`
- 非思考模式：`temperature 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5`

> ⚠️ **xhigh 是默认，且在简单 prompt 上"花很多 token"**。社区共识：`medium` 能保留大部分质量而 token 成本大幅下降。
> 本仓库那套温度实验是围绕 T=0~1.5 设计的，**接 27B 时要把 thinking 模式当作独立变量**，不能直接套用。

### 1.1 关键结构优势：只有 16 层有 KV cache

```
KV/token = 2(K,V) × 16 层 × 4 KV头 × 256 head_dim × 2 bytes = 65,536 B = 64 KB/token
```

| 上下文 | bf16 KV | 4-bit KV（TurboQuant） |
|---|---|---|
| 8K | 0.5 GB | 0.13 GB |
| 32K | 2.0 GB | 0.5 GB |
| 128K | 8.0 GB | 2.0 GB |
| **262K** | **16.0 GB** | **4.0 GB** |

线性注意力是定长递归状态，不随上下文增长。**跑满 262K 时 bf16 的 KV（16GB）和 4-bit 权重（16GB）一样大**——长上下文下 KV 量化比再降权重精度更划算。

参考：Community 实测 RTX 5090 32GB 上 Q4 权重 + Q8 KV 约 67K token 就 OOM，换 Q4 KV 才跑完约 237K 输入。**长上下文是 KV pool 决定的，不是权重决定的。**

---

## 2. 三种精度的实际产物（体积为 HF API 实测字节）

### 2.1 16-bit（BF16）

| 产物 | 体积 | 下载 / 赞 |
|---|---|---|
| `Qwen/Qwen3.8-27B` 官方 | 55.6 GB | — |
| `unsloth/Qwen3.8-27B-GGUF:BF16` | **54.66 GB**（2 卷） | 6,237,305 / 4,815 |

**M2 Max 96GB 上的社区实测：BF16 约 13–20 tok/s。**
能装下（54.66GB + KV 在 85.9GB Metal cap 内），但会挤掉当前常驻的 125B（78GB），且速度是 4-bit 的 1/3。**只适合当基准对照。**

### 2.2 4-bit —— 甜点区，证据最厚

| 产物 | 体积 | bpw | 下载 / 赞 | 备注 |
|---|---|---|---|---|
| **`mlx-community/Qwen3.8-27B-4bit`** | **16.05 GB** | 4.0 | **147,514 / 186** | ⭐ **社区标准件** |
| `Jundot/Qwen3.8-27B-oQ4e-mtp` | 16.97 GB | 4/5 混 | 10,979 / 50 | ⭐ **oQ 混合精度，见下** |
| `orcarouter/Qwen3.8-27B-MLX:4-bit` | 16.05 GB | 4.0 | 1,874 / 27 | 附 6-bit / 8-bit / mtp |
| `orcarouter:6-bit` | 24.46 GB | 6.0 | 同上 | |
| `orcarouter:8-bit` | 31.18 GB | 8.0 | 同上 | |
| `mlx-community/Qwen3.8-27B-mxfp8` | 28.66 GB | 8.0 | 1,053 / 8 | MXFP8 group_size 32 |
| `True2456/...-AWQ-4.85bpw` | 16.83 GB | 4.85 | 620 / 12 | **带 `omlx` tag** |
| `unsloth:Q4_K_M`（GGUF） | 16.06 GB | 4.5 | 6.2M / 4.8K | 社区量最大 |
| `unsloth:UD-Q4_K_XL`（GGUF） | 17.56 GB | 4.9 | 同上 | |

**⭐ Jundot 的 oQ4e 是本机最值得注意的 4-bit 产物**——它的 `config.json` 实测确认是**逐张量混合精度**：

```json
{"group_size": 64, "bits": 4, "mode": "affine",
 "language_model.model.layers.0.linear_attn.in_proj_a": {"bits": 5, ...},
 "language_model.model.layers.0.linear_attn.in_proj_b": {"bits": 5, ...},
 "language_model.model.layers.0.linear_attn.in_proj_z": {"bits": 5, ...}, ...}
```

即**基础 4-bit，敏感张量单独提到 5-bit**（社区说法是 166 个最敏感的 GatedDeltaNet 张量）。这是 oMLX 自己的量化器产物，是"stock conversion"，不掺任何 abliteration。**在 Apple Silicon 上 oMLX + oQ4e 是社区公认最快的一档。**

**orcarouter 的 KL 散度实测**（第三方，本机未复现）：

| 档位 | KL 散度 | top-1 一致率 |
|---|---|---|
| 8-bit | 0.00068 | 98.34% |
| 6-bit | 0.00216 | 96.97% |
| 4-bit | 0.02824 | **92.68%** |

**oMLX 自带 intelligence benchmark（同引擎同机 A/B）**——这条比 KL 更有说服力：

| | oQ4e（4bit） | oQ8e（8bit） |
|---|---|---|
| MMLU | 89.7% | 89.4% |
| TRUTHFULQA | 88.1% | 89.0% |
| HUMANEVAL | 14.0% | **17.1%** |
| LIVECODEBENCH | 4.7% | **7.7%** |

**4 项里 8-bit 赢 3 项**，LiveCodeBench 差 3 个百分点（相对降幅 39%）。**如果你的活是写代码，4-bit 的质量损失不是零。**

### 2.3 ternary（三值）—— **Prism ML 的 Bonsai 2**

> ⚠️ 勘误：本文早期版本误把 `penkia/TernaryQuench` 当作三值代表。**正确的是 Prism ML 的 Bonsai 系列**，
> 下载量差 6500 倍（GGUF 386 万 vs 594）。`penkia` 那份是另一条 CAT-Q 路线（3.02 bit/param，非三值），已降级为参考。

**Prism ML 两代产品**（Apache 2.0）：

| | **Bonsai 27B**（2026-07-14） | **Bonsai 2 27B**（2026-09-17） |
|---|---|---|
| 基座 | Qwen3.**6**-27B | Qwen3.**8**-27B |
| 三值 | 5.9 GB / 1.71 bpw / **94.6% 保留** | 5.9 GB / 1.76 bpw / **98.2% 保留** |
| 二值 | 3.9 GB / 1.125 bpw / 89.5% 保留 | — |
| 上下文 | 262K（4-bit KV 把 17.2GB 压到 4.3GB） | 262K |

**Bonsai 2 官方产物（HF API 实测字节）**：

| 产物 | 体积 | 下载 / 赞 |
|---|---|---|
| `prism-ml/Ternary-Bonsai-2-27B-gguf`（**PTQ1_0 / PQ2_0**） | **5.947 / 7.206 GB**（+ mmproj 0.93/0.63） | **3,869,715 / 2,359** |
| `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | **8.595 GB** | 69,192 / 417 |
| `prism-ml/Ternary-Bonsai-2-27B-gguf:F16` | 53.808 GB | 同上 |
| `prism-ml/Ternary-Bonsai-27B-*`（Qwen3.6 基座） | — | 1,090,688 / 628,497 |

**两种打包格式的取舍**：

| 格式 | 体积 | bpw | 特点 |
|---|---|---|---|
| GGUF **PTQ1_0** | 5.947 GB | 1.75 | 稠密存 trit，接近理论下界；**Ada 卡 / L4 更喜欢**（每步搬数据少） |
| GGUF **PQ2_0** | 7.206 GB | 2.13 | 每 trit 占 2-bit 槽，换取更便宜的解包；**RTX 5090 / Blackwell / H100 更喜欢** |
| **MLX 2bit** | **8.595 GB** | 2.25 | MLX 容器每组同时存 scale **和 bias**（MLX 分组低位容器的特性）→ 推高到 2.25 bpw |

**这是 Bonsai 2 最值得学的技术：blockwise Hadamard rotation。**
权重在转三值**之前**先做分块 Hadamard 旋转，推理时对激活施加逆变换（**已折叠进存储的权重，不增加体积也不增加运行时开销**）。
作用：把信息摊开，让三值舍入处理得更好——这正是低位元量化通常会推理崩塌的原因。

> ⚠️ **代价：官方 MLX 版需要专用 runtime。**
> 仓库里有独立的 `hadamard.json` / `files.json`。普通 MLX 或 llama.cpp 构建**要么加载失败，要么静默输出错误结果**（不是报错！）。
> 社区测速：M5 Max MLX 46.8–58 tok/s；RTX 5090 143 tok/s；M4 Pro 24GB 约 15–20 tok/s。

**⭐ 关键：本机 oMLX 已经预装了对口的加速 kernel**

`/Applications/oMLX.app/.../custom_kernels/bonsai/` 实测存在：
`_ext.cpython-311-darwin.so`(149KB) + `libomlx_bonsai_kernel_ops.dylib` + `omlx_bonsai_kernels.metallib`(2.1MB)
`GET /api/status` → `"bonsai": {"available": true, "import_error": null}` ✅

`patches/bonsai_qmv.py` 源码实测：拦截 `QuantizedLinear.__call__`，条件是
`bits ∈ {1,2} and mode == "affine"` 且 **batch 维 M ≤ 5（decode 区间）**——
1-bit 走 `qmv_fast`，2-bit 小 batch 走 `qmv_wide`。**decode 阶段是真 kernel 加速，不是解包后稠密乘。**

**但它加速的是「1/2-bit affine」，不是三值 {−1,0,+1}。** 所以有两条路：

| 路线 | 产物 | 能否用本机 oMLX | 说明 |
|---|---|---|---|
| **A. 官方三值** | `prism-ml/...-mlx-2bit`（8.595 GB） | ❌ | 需要 Prism ML 自带 runtime（Hadamard 旋转） |
| **B. 第三方再量化** | `TokenAI-zer/Ternary-Bonsai-2-27B-MLX-oQ*-mtp` | ✅ | **用 stock MLX 量化重压，stock oMLX / mlx-vlm ≥0.7 直接跑**，保留 vision tower(333 张量 BF16) + 嫁接的 MTP head(15 张量 0.42B) |

**路线 B 的实测阶梯**（同 5 条 prompt，last-token logits vs bf16，greedy）：

| build | bpw | 磁盘 | KL(bf16‖q) | max rel err | top-1 | top-5 | 适用 |
|---|---|---|---|---|---|---|---|
| oQ2 | 3.00 | 11.63 GB | **0.37658** | 0.2472 | **4/5** | 16/25 | 最小体积，**有可测损失** |
| oQ3 | 3.70 | 13.86 GB | 0.03478 | 0.1077 | 5/5 | 21/25 | ⭐ **体积/保真最佳比** |
| oQ4 | 4.70 | 17.02 GB | 0.01476 | 0.0519 | 5/5 | 23/25 | 均衡 |
| oQ6 | 6.70 | 23.72 GB | 0.00074 | 0.0164 | 5/5 | 24/25 | 高保真 |
| oQ8 | 8.50 | 30.00 GB | **0.00008** | 0.0083 | 5/5 | 25/25 | **与 bf16 实际不可区分** |

**这张表里最反直觉的一条**：作者自己写道——
> "a ternary model still benefits from more bits... Affine quantization builds its grid from the group minimum and maximum, so with 4 levels over a symmetric {−a, 0, +a} group the levels land at −a, −a/3, +a/3, +a — **zero itself is not representable, and zero is the most common value in a ternary tensor.**"

**三值 ≠ 最佳压缩率。** 从 2.25 bpw（官方 MLX）重新量化到 3.70 bpw（oQ3），KL 从 0.377 降到 0.035，**降了 10.8 倍，只多花了 3.1 GB**。

⚠️ **作者的诚实 caveat**：参考基准是 **bf16 转换版，不是原始 GGUF**。转换本身的误差会被所有 build 继承，**这张表看不出来**。

⚠️ **一个必须点出的矛盾**：Prism ML 官方声称 **98.2% 保留**，但 oQ3 相对 bf16 的 KL（0.035）**比 oQ4（0.015）差 2.4 倍**。
benchmark 平均分和 logits 分布是两种度量，**方向上对不上**。而本仓库的经验正落在这条线上（见 §8）：低位元量化的损失**不是平均分配的**——数学 benchmark 几乎不掉，tool calling 能掉 26%。
**所以「98.2% 保留」不能读成「你的任务只掉 1.8%」。**

---

## 3. 你的机器：不可改的物理上限（本机实测）

| 项 | 值 | 来源 |
|---|---|---|
| 芯片 / 内存 / 核 | M2 Max / 96GB / 38 GPU 核 | `sysctl` / `ioreg` |
| 内存带宽 | **409.6 GB/s** | LPDDR5-6400 512-bit 4 通道 |
| Metal cap | 85.9 GB | `sysctl iogpu.wired_limit_mb=88000`（服务启动时设置） |
| ANE | 16 核 / 15.8 TOPS | — |
| **INT8 矩阵单元** | **无** | 见下 |
| oMLX | 0.7.0，`qwen35_prefill` kernel **available** | `GET /api/status` |

### 3.1 硬证据：M2 Max 的 INT8 = FP16

跨芯片实测（`ane_int8_bench`，5 芯片 × 5 次）：

| 芯片 | FP16 TFLOPS | INT8 TOPS | **INT8 加速比** |
|---|---|---|---|
| M1 | 11.19 | 11.34 | 1.01× |
| **M2** | **16.07** | **16.27** | **1.01×** |
| M3 | 18.59 | 18.60 | 1.00× |
| M4 | 18.64 | 35.15 | **1.88×** |
| M5 | 19.31 | 36.49 | **1.89×** |

INT8 的收益不是"算得更快"，是"搬的数据更少"——只有芯片本身够快、搬数据成为瓶颈时才转化成吞吐。M2 的 ANE 算力不够，减少带宽省下的时间被更慢的 INT8 计算吃掉。

**推论**：
- oMLX 的 `qwen35_oq_a8_enabled`（INT8-activation prefill）在本机**预期收益 ≈ 0**
- 它本身也只对 **Q4/Q5 + group_size 64 affine** 生效（`patches/qwen35_oq_a8.py` 的 `_SUPPORTED_BITS = frozenset((4, 5))`），对 BF16 / 8-bit / 三值模型直接不适用
- 且与 `qwen35_ane_prefill_enabled` **互斥**（`model_settings.py:504` 显式报错）
- **本机 ANE 是可用的 → 优先开 ANE，不要开 a8。** 两者只能选一个，且 ANE 在这台机器上有实测先例。

---

## 4. 加速方案（5 层，按影响力排序）

### L0 · 硬件层：只能接受

dense 模型 decode 时每个权重字节每 token 读一遍，所以 **tok/s ≈ 内存带宽 ÷ 权重体积**。

### L1 · 权重精度：**本机唯一的一阶变量**

理论上限（409.6 GB/s ÷ 体积），实测量级见 §5：

| 精度 | 代表产物 | 体积 | 理论上限 tok/s |
|---|---|---|---|
| BF16 | unsloth BF16 | 54.66 GB | 7.5 |
| 8-bit | orcarouter 8-bit | 31.18 GB | 13.1 |
| 6-bit | orcarouter 6-bit | 24.46 GB | 16.8 |
| **4-bit** | **mlx-community 4bit** | **16.05 GB** | **25.5** |
| **ternary** | **TernaryQuench MLX** | **10.50 GB** | **39.0** |
| UD-IQ2_XXS | unsloth | 7.27 GB | 56.3 ⚠️ |
| UD-IQ1_S | unsloth | 6.19 GB | 66.2 ⚠️ |

⚠️ **IQ1_M / IQ2_S 的坑**：需要**剪掉 MTP head（block 64）**，剪了还要改 metadata 否则加载失败。等于「省内存 ↔ 丢投机解码」二选一。

> **注意理论与实测的差距**：4-bit 理论上限 25.5 tok/s，而同机 omlx.ai 实测（MTP 关）是 19.9–21.1 tok/s——**达成率约 80%**，与社区通用经验法则 `tok/s ≈ 带宽 × 0.65 ÷ 权重体积` 一致。这个 0.65–0.8 的系数可以直接拿来外推。

### L2 · oMLX 已验证的 27B 专属链路 ★

**本机源码逐条核对 + `GET /api/status` 实证**：

| 证据 | 位置 / 结果 |
|---|---|
| `Qwen3_5ForConditionalGeneration` 被直接引用 | `patches/qwen38_modelopt_mixed.py`、`patches/mlx_vlm_mtp/` |
| ANE 后端选择：`qwen3_5`/`qwen3_6`/`qwen3_8` 前缀 → qwen | `model_settings.py:77` |
| MTP 兼容 `qwen3_5*` | `mtp_enabled` 文档 |
| KV 类型映射 `mlx_lm.models.qwen3_5` → `gdn-qk-norm-2` | `cache/paged_ssd_cache.py:302` |
| **层数精确对上 27B**：`qwen35_ane_prefill_max_layers=64`、`gdn_max_layers=48` | 本机 settings 实测 |
| **native kernel 可用** | `/api/status` → `qwen35_prefill: {available: true, import_error: null}` |
| **ANE patch 可用** | `/api/status` → `ane_prefill: {patch_available: true}` |
| ModelOpt 混合加载器专写给 `unsloth/Qwen3.8-27B-NVFP4` | FP8(E4M3 per-channel) + NVFP4(E2M1) 双格式 |

`max_layers=64` / `gdn_max_layers=48` **正好等于 27B 的 64 层与 48 个 GDN 层**——这套 ANE 链路就是照着 27B 调的。

**四个开关（本机实测当前全为关）**：

**① ANE prefill** —— 把 MLP offload 到 Neural Engine
```
qwen35_ane_prefill_enabled         = False   ← 打开
qwen35_ane_prefill_sequence_length = 2048    （≥1024，64 的倍数）
qwen35_ane_prefill_fraction        = None    （未设 → 0.53；社区实测 0.5 更好）
qwen35_ane_prefill_max_layers      = 64      ← 正好覆盖 27B 全部层
qwen35_ane_prefill_gdn             = True    ← 社区最优配方是 false
qwen35_ane_prefill_dual_ane        = True
qwen35_ane_prefill_cpu_enabled     = False   ← 另一条 CPU 分流路径，8 线程
```

**② MTP 投机解码** —— 模型自带 MTP 头
```
mtp_enabled            = True
mtp_fixed_depth        = 1       ← 当前值（本仓库 125B 上实测最优）
mtp_adaptive_max_depth = None    ← 默认上限：M5 上 4，其他芯片 3
```

**③ TurboQuant KV 量化** —— 长上下文的杠杆
```
turboquant_kv_enabled = False
turboquant_kv_bits    = 4.0
```
源码确认对 **decode（L=1）**、**MTP verify 形态多行 decode（1<L≤15）**、**长 prefill（L>8192）** 三条路径都有专门 kernel。

**④ ModelOpt 混合精度加载器** —— `patches/qwen38_modelopt_mixed.py`
专写给 `unsloth/Qwen3.8-27B-NVFP4`：per-channel E4M3 **FP8**（attention/GDN/lm_head/layer 56-63 MLP）+ packed E2M1 **NVFP4**（其余 MLP），用 MLX 原生载体、matmul 后再施加原始 scale，不改动打包权重码。
**分支谓词很严格**，只认已发布的 dense 64 层 27B 几何 + 精确 config 分组目标。作者明确：**激活量化不启用，激活保持 A16。**

**其他相关开关**：`dflash_*`（与 MTP 互斥）、`specprefill_*`、`index_cache_freq`、`vlm_mtp_*`（drafter 支持 `qwen3_5_mtp`）。

### L3 · 投机解码：真正的第一杠杆

**Weschera 那个实测仓库给了社区最干净的公式：**

> **decode 上限 = 原始 decode × tokens-per-cycle。不是 prefill。**
> 原始 decode（MTP 关）在 M4 Max 上 ~24.6 tok/s。

**MTP 接受率因任务类型差 20 倍**（M4 Max 实测）：

| 任务 | 接受率 | tokens/cycle（k=3） |
|---|---|---|
| **代码** | **99.6%** | **3.76**（k=3 的近理论上限） |
| **散文** | 75–80% | 2.5–2.8 |

**所以同样的配置，代码任务比散文快约 35%。** 本仓库的 7 个任务全是代码/结构化输出——**这是好消息**。

**k 值不是越大越好**（同一仓库同日实测）：

| 配置 | 散文 | 代码 |
|---|---|---|
| k=3 | **53.3** | 72.1 |
| k=4 | 47.9 | **73.3** |
| k=3，ANE 关 | 47.9 | 47.9 |

**k=4 代码 +1.2、散文 −5.4。k=3 是全能最优。**
⚠️ 但反例也存在：M4 Mac mini 上 `n-max 4 / p-min 0.0` 反而从 baseline 5.91 掉到 3.05 tok/s。**必须自己扫。**

**⚠️ 最重要的坑：MTP 可能让机器变慢。**
社区反例：**1-bit 量化（6.7GB）跑 27 tok/s，但"难以坚持给出答案"**——速度快了但输出不可用。
本仓库那 182 次生成的实验设计（同一校验器 + 显式错误分类 E1–E10）就是为回答这类问题准备的，**接 27B 时直接复用，不要只看 tok/s。**

**MTP vs 外部 drafter（DSpark/DFlash）**：
- DFlash drafter 的单 token 成本 0.046 vs MTP 的 0.153（MTP 是内建多头，每次要过完整 vocab projection）
- 但 llama.cpp 的 DFlash2 集成**截至 2026-09-01 仍是未合并 PR，要自己编译**
- M5 Max 上实测 PR 版 11–35 tok/s，**远低于宣传的 70**；同机 Ollama MLX 不用 drafter 就有 24–56
- **判断：现在就用原生 MTP，DFlash 观望。**

**Drafter 大小会影响 MTP 行为**（M4 Max 实测）：mlx-community 把 MTP head 拆成 **239MB 独立 drafter** 后，decode 从 29.5 跳到 46.8（散文）/ 56.1（代码）。orcarouter 提供的 `mtp/` 是 **849MB**——**比 239MB 大 3.5 倍**。选带 MTP 的产物时留意 drafter 大小。

### L4 · 上下文、并发与 prefix cache

- **并发（27B，同机实测）**：oMLX 4-bit 8K 上下文下 **1×21.1 / 2×35.3(1.67×) / 4×69.4(3.29×) / 8×136.9(6.49×)**。
  → **27B 的并发扩展远好于当前 125B**（125B 在 4 并发就饱和）。因为 27B 只有 16GB 权重，batch 摊薄成本低得多。`ask.py` 的 `MAX_WORKERS=4` 上限对 27B 可以放宽，**需实测**。
- **但 64-output 短输出任务在 N=4 就到平台**（prefill 主导，prefill 在 GPU 上串行），**256-output 长输出才吃满 batching**。
- **prefix cache 的静默陷阱**：vLLM 对 hybrid attention 模型（27B 报 `is_hybrid=True`）**自动关闭 prefix caching，启动日志不提示**。手动打开后 19K 共享前缀 prefill 快 14×、53K 快 22×。
  → oMLX 侧的 `total_cached_tokens` 可以在 `/api/status` 里看（当前实例 `cache_efficiency: 76.4`）。

**❗ 测量纪律（社区方法论，与本仓库一致）**：
> "Contention invalidates everything." 第二个 30GB 模型常驻时 N=6 崩塌到 7.6 tok/s 并触发 Metal OOM；**独占时 N=16 毫无问题**。
> **绝不同时加载两个模型服务器；有别的模型占着 GPU 时不测任何东西。**

### L5 · 换运行时

| 运行时 | 判断 |
|---|---|
| **oMLX 0.7.0** | ⭐ **本机首选**：原生 MTP + vision + ANE prefill + TurboQuant + prefix cache |
| mlx-vlm | ✅ 保留 MTP 头和视觉塔；比 oMLX 轻 |
| **mlx-lm（stock）** | ❌ **不实现该架构**（社区明确：stock mlx-lm does not implement this architecture），且丢 MTP 头和视觉塔 |
| llama.cpp | ⚠️ **需 2026-05 之后版本**，旧版报 `unknown model architecture: qwen35`；主线已支持 MTP，**但 KV cache 量化在非 Q4 时会静默 fallback 到 CPU**（社区为此专门用 BeeLlama fork） |
| Ollama | `qwen3.8:27b-mlx`（~18GB）；**默认 context 只有 2048 token**，接 agent 必须写 Modelfile 设 `num_ctx` |
| vllm-metal 0.3.0 | ❌ 同机实测 **13.2 tok/s，比 oMLX 慢约 3.6×**，无 MTP 无 ANE |
| LM Studio | 兜底 |

**MLX 比 Ollama 快约 1.5×**（decode-heavy 时 2–3×，社区三方对照实测）。

---

## 5. 性能锚点：同机实测数据

### ⭐ 同一台机器（38c / 96GB），omlx.ai 社区基准

模型 `Qwen3.8-27B-MLX-4bit`，oMLX v0.6.1，macOS 26.5.1，**MTP 关 / ANE 关**：

| 上下文 | prefill tok/s | **decode tok/s** | 峰值内存 |
|---|---|---|---|
| 1K | 117.2 | **21.1** | 15.8 GB |
| 4K | 124.8 | **20.5** | 17.2 GB |
| 8K | 126.0 | **19.9** | 17.9 GB |
| 16K | 121.6 | **18.8** | 19.3 GB |

GPU 平均占用 **99.9%**，thermal Nominal，280 个采样点。
**TTFT 在 8K 上下文是 65 秒**——prefill 是 Mac 的真实瓶颈（社区语：Spark 吃 8.6K paste 只要 4 秒，Mac 要 35 秒）。

### 其他芯片的 27B 4-bit oMLX 实测（横向）

| 机器 | prefill | decode | 备注 |
|---|---|---|---|
| **M2 Max 96GB** | **126.0** | **19.9** | **本机，MTP/ANE 关** |
| M3 Max 30c（oMLX，Q4_K_XL 20.89GB） | 154–160 | 19.6 @1k / 16.7 @4k | |
| M4 Max 48GB | 256.7 | 44.9 | MTP 开 |
| M4 Pro 48GB | — | 24.8 | MTP 开 |
| M4 Max 128GB | 273.7 | 53.3 散文 / 72.1 代码 | **ANE + MTP k=3 最优** |
| M5 Pro 64GB | 368 | 23.0 | MTP 开 |
| M5 Max 128GB | 440–899 | 28–75 | |
| M5 Max 64GB | 400 | 34.6（oQ8e） | MTP 开 |
| M3 Ultra 512GB | 422–445 | 58.8–62.5 | MTP 开 |

**decode 与上下文长度的关系**（M5 Max 128GB 实测，4-bit）：
1K→66.0 / 4K→75.3 / 8K→38.8 / 16K→47.8 / 32K→42.4 / 64K→31.0 / 128K→25.2 / 195K→19.5
→ **从 128K 到 195K 掉了 22%**。长上下文是真实的成本，不是 checkbox。

**Weschera 完整实测阶梯（M4 Max 128GB，同日同 prompt，3 次重复取均值）**：

| 配置 | 散文 tok/s | 代码 tok/s | prefill 4K |
|---|---|---|---|
| **oMLX 0.6.3rc2 + ANE + native MTP, k=3** | **53.3** | **72.1** | **273.7** |
| oMLX 0.6.3rc2 + MTP k=3, ANE 关 | 47.9 | 47.9 | ~83 |
| oMLX + MTP k=4, ANE on | 47.9 | 73.3 | — |
| oMLX 0.6.3rc2 + ANE + CPU sharing + GDN, k=3 | 52.3 | 72.0 | 277.4 |
| vllm-metal 0.3.0（8-bit，无 MTP） | — | 13.2 | — |
| DFlash2（oMLX 0.6.3rc2） | 36.7 | — | — |

**ANE prefill 的真实收益**：prefill 83 → 273.7 tok/s（**3.3×**），decode **+11%**。
机理：oMLX 的 prefill 和 decode **共享同一个 engine**，prefill 在飞时 decode 得排队。ANE 把 MLP 挪走后 prefill 更快，decode 被阻塞的窗口就小了。
**CPU sharing + GDN 只换来 prefill +1.3%、decode 反而 −1.0，还要额外 21GB 存 FP16 clone → 不值得。**
**ANE fraction 0.75 比 0.5 更差**（257 vs 274）。

---

## 5.5 精度选择：8-bit 更好吗？该上 BF16 吗？多模型 pipeline 怎么排？

这一节回答三个层层递进的问题。**结论先给：**

| 问题 | 答案 |
|---|---|
| 8-bit 比 4-bit 好吗？ | **质量几乎打平，速度慢一半。不选。** |
| 那 16-bit BF16 呢？ | **质量上没有可测优势（oQ8 的 KL 已到 0.00008），只贵 1.8× 内存、慢 3.4×。不选。** |
| 拐点在哪？ | **6-bit**。4→6 之间 KL 差 **20 倍**，6→8 只差 **9 倍**。 |
| 多模型 pipeline 怎么选？ | **取决于 pipeline 的并发结构，不是"越大越好"。见 §5.5.4。** |

### 5.5.1 先看速度：8-bit 慢一半，这是硬账

M2 Max 409.6 GB/s。同机实测锚点：4-bit 16.05 GB → **19.9 tok/s**（MTP 关）。
反推有效带宽效率 = 20 × 16.05 ÷ 409.6 = **78.4%**。用它外推同机其他档位：

| 档位 | 体积 | **预测单流 tok/s** | 相对 4-bit | MTP 开后（×3.0 tok/cycle） |
|---|---|---|---|---|
| BF16 | 54.66 GB | **5.9** | **0.29×** | ~18 |
| oQ8 | 30.00 GB | **10.5** | **0.52×** | ~31 |
| **oQ6** | **23.72 GB** | **13.3** | 0.67× | ~40 |
| **4-bit** | **16.05 GB** | **19.9（实测）** | 1.00× | ~60 |
| Ternary oQ3 | 13.86 GB | 23.1 | 1.16× | ~69 |
| Ternary oQ2 | 11.63 GB | 27.6 | 1.39× | ~83 |
| **Ternary 官方 MLX** | **8.60 GB** | **37.4** | **1.88×** | **~112** |

> 标"预测"的由 §5 的 78.4% 效率外推；标"实测"的是 omlx.ai 同型号（38c/96GB）数据。MTP 那一列按代码任务 tok/cycle 3.0 折算（M4 Max 实测 3.76，散文 2.6，取保守中值）。

**所以 8-bit = 4-bit 的一半速度，换来的是——看下一节——几乎为零的质量收益。**

### 5.5.2 再看质量：8-bit 的优势在统计上站不住

**证据 A：oMLX 自带 intelligence benchmark**

| | oQ4e (4bit) | oQ8e (8bit) | 差值 | 该项 n | 二项标准误 | **显著性** |
|---|---|---|---|---|---|---|
| MMLU | 89.7% | 89.4% | −0.3 | 1000 | 1.0% | 不显著 |
| TRUTHFULQA | 88.1% | 89.0% | +0.9 | 817 | 1.1% | 不显著 |
| HUMANEVAL | 14.0% | 17.1% | +3.1 | 164 | 2.7% | **1.1σ，不显著** |
| LIVECODEBENCH | 4.7% | 7.7% | +3.0 | 300 | 1.9%（合并） | **1.6σ，不显著** |

> ⚠️ **我上一版报告写「4-bit 的质量损失看得见」，依据不足，在此更正。**
> 在它自己给的样本量下，4 项全部不显著。要靠这些数字论证"8-bit 更好"是过度解读。

**证据 B：logits 层面的 KL 散度（更硬的度量）**

| build | bpw | 磁盘 | KL(bf16‖q) | 相对 oQ4 |
|---|---|---|---|---|
| Ternary oQ2 | 3.00 | 11.63 GB | 0.37658 | **25.5× 差** |
| Ternary oQ3 | 3.70 | 13.86 GB | 0.03478 | **2.4× 差** |
| oQ4 | 4.70 | 17.02 GB | 0.01476 | 1× |
| **oQ6** | **6.70** | **23.72 GB** | **0.00074** | **20× 好** |
| oQ8 | 8.50 | 30.00 GB | 0.00008 | **185× 好** |

**这张表才是真正的图景——质量下降是指数的，不是线性的：**
- **4 → 6 bit 是一个巨大的台阶**（KL 差 20 倍）
- **6 → 8 bit 只是一点点**（KL 再降 9 倍，但绝对值已经到 8e-5）
- **oQ8 的 KL = 0.00008，等于"与 bf16 实际不可区分"**

**所以 BF16 的答案是明确的：**
既然 8-bit 的 oQ8（30GB）在 logits 层面已经与 BF16 不可区分，
**BF16（54.66GB，5.9 tok/s）就只剩成本，没有收益。选 BF16 是纯亏。**
**BF16 的唯一用途是当基准参照，不是部署选项。**

**拐点是 6-bit**：23.72 GB / 13.3 tok/s / KL 0.00074。

### 5.5.3 但对多模型 pipeline，排序会**反过来**

**硬约束（社区实测 + 本机）**：
- 96 GB 物理，`sysctl` 设的 Metal cap 85.9 GB
- **两个 30GB 模型同驻 → N=6 崩塌到 7.6 tok/s 并触发 Metal OOM**（bluehawana 实测，独占时 N=16 无问题）
- 社区 checklist 要求留 10–15% headroom
- → **实际可用于模型：约 70 GB**（96 − 15 OS/IDE/浏览器/agent tools − 缓冲）

**内存预算表**（70 GB 预算，32K 上下文 bf16 KV = 2.0 GB）：

| 档位 | 权重 + KV | **能同时装几台** | 剩余 | 单流 tok/s |
|---|---|---|---|---|
| BF16 | 56.7 GB | **1** | 13.3 | 5.9 |
| oQ8 | 32.0 GB | **2** | 6.0 | 10.5 |
| oQ6 | 25.7 GB | **2** | 18.6 | 13.3 |
| 4-bit | 18.1 GB | **3** | 15.7 | 19.9 |
| Ternary oQ3 | 15.9 GB | **4** | 6.4 | 23.1 |
| **Ternary MLX 2bit** | **10.6 GB** | **6** | **6.4** | **37.4** |

**"好几个模型同时工作"的真实代价：从 3 台掉到 1 台。**

### 5.5.4 ⚠️ 最关键的一点：内存带宽是共享且守恒的

**这是最容易被搞错的地方：**

> **N 个各自独立的小模型 ≠ 1 个 batch 的大模型。**

原因：409.6 GB/s 只有一份。Batching 的全部价值在于**权重读一次、服务 B 个 token**。
同机 omlx.ai 实测（4-bit，8K 上下文）：

| batch | 聚合 tok/s | 加速比 |
|---|---|---|
| 1× | 21.1 | 1.00× |
| 2× | 35.3 | 1.67× |
| **4×** | **69.4** | **3.29×** |
| **8×** | **136.9** | **6.49×** |

**一个 4-bit 模型 batch=8 → 136.9 tok/s 聚合。**
**四个 4-bit 模型并发（假设装得下，4×16.05=64GB）→ 总吞吐还是 ~20 tok/s，不是 80。**
**拿不到 batching 摊薄——因为它们各自只服务 1 个请求。**

**所以 pipeline 的选择完全取决于并发结构：**

| pipeline 形状 | 瓶颈 | 该选什么 |
|---|---|---|
| **A. 串行**（一次只跑一个阶段） | 单请求延迟 | **最快且质量够**：Ternary(8.6GB/37 tok/s) 或 4-bit(16GB/20 tok/s)。BF16 也可行（5.9 tok/s 交互体验差） |
| **B. 真并发**（多个不同模型同时跑） | **带宽总量守恒** | **模型越小越对**。3×4-bit 总吞吐 ≈ 6.6 tok/s 分给 3 个 → 每台 2.2 tok/s 😱；3×Ternary ≈ 12.4 → 每台 4.1 |
| **C. 多个请求喂同一个模型** | batching 摊薄 | **大模型 + 高并发最优**（8× batch = 6.49×） |
| **D. 多个阶段其实可以合并** | 内存 + 带宽双省 | ⭐ **最优**：合成一个模型的多阶段 prompt，省内存还拿 batching |

**情况 B 的量化表**（总吞吐 = 409.6 × 0.78 ÷ 总权重字节，与模型数量无关）：

| 配置 | 总权重 | **聚合 tok/s** | 每台 tok/s |
|---|---|---|---|
| 2 × oQ8 | 60.0 GB | 5.3 | 2.7 |
| 2 × 4-bit | 32.1 GB | 10.0 | 5.0 |
| 3 × 4-bit | 48.2 GB | 6.6 | 2.2 |
| 3 × Ternary MLX | 25.8 GB | 12.4 | **4.1** |
| **6 × Ternary MLX** | **51.6 GB** | **6.2** | 1.0 |

**真并发场景下，「同时装 6 台三值」的总吞吐和「3 台四比特」一样，但覆盖面广 2 倍。**

### 5.5.5 给你的决策路径

**先回答一个问题：你的 pipeline 是串行还是真并发？** 这决定一切。

- **如果是串行 / 低并发**（最常见：agent 工具链，一步步调用）→
  **选 4-bit（`mlx-community/Qwen3.8-27B-4bit`，16.05GB，19.9 tok/s）。**
  质量损失在统计上不显著，速度是 BF16 的 3.4 倍，内存 1/3.4。
  **别选 8-bit（慢一半换不到质量），更别选 BF16。**

- **如果要跑多个不同模型同时在** →
  **Ternary 值得认真考虑。** `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` 只有 8.595GB，
  **能多装 3 台，单流还快 1.88 倍**。
  ⚠️ 但要接受两个前提：(a) 需要专用 runtime（不能直接喂给 stock oMLX）；
  (b) 官方声称的 98.2% 与第三方 KL 实测方向矛盾，**必须在你的 7 个任务上实测**。
  **务实折中：`TokenAI-zer/...-oQ3-mtp`（13.86GB）——stock oMLX 直接跑，带 MTP，KL 0.035。**

- **如果要 6-bit**（质量与速度的平衡点）→ 23.72GB / 13.3 tok/s / KL 0.00074。能装 2 台。

- **BF16 什么时候才值得** → **只有在你要它当基准参照，或者 pipeline 里只有它一个模型且不在乎交互延迟时。** 生产部署不选。

**无论选哪档，都要做的验证（本仓库现成可复用）**：
1. 用 7 任务 + E1–E10 分类跑质量，**不要只看 benchmark 平均分**
2. 特别加测 **tool calling / 精确输出**（低位元量化在这里掉得最狠，参考 Qwen3.6-27B 的 TauBench 82.9→61.3）
3. 记录 MTP 接受率 α（决定 depth 取值）
4. 记录并发下的聚合吞吐，不只记单流



### ✅ 本机实测（我在这台机器上跑出来的）
- 硬件：M2 Max / 96GB / 38 核 / Metal cap 85.9GB / 409.6GB/s
- `GET /api/status` → oMLX 0.7.0，`qwen35_prefill` native kernel **available: true**，**ANE patch available: true**
- oMLX 源码中 `Qwen3_5ForConditionalGeneration` 支持、层数参数 64/48、MTP 兼容 `qwen3_5*`、a8 只支持 Q4/Q5 GS64 且与 ANE 互斥
- ModelOpt-NVFP4 加载器专写给 `unsloth/Qwen3.8-27B-NVFP4`
- 各量化产物的精确字节体积与下载/点赞数
- 当前实例运行统计：`avg_prefill_tps 63.1 / avg_generation_tps 39.3 / cache_efficiency 76.4`

### ✅ 一手来源（官方 config / HF API）
- 27B 全部结构参数
- TernaryQuench 精确 10,504,400,856 B / 3 分片 / bits=2
- Jundot oQ4e 的逐张量 5-bit 覆盖配置
- 官方推荐采样参数

### ⚠️ 社区第三方数据，同型号/同引擎但**不在本机**
- **omlx.ai 的 M2 Max 38c/96GB 基准**（同机同型号，oMLX 0.6.1，MTP+ANE 关）——**这是最可信的一条**，但版本是 0.6.1 而本机 0.7.0
- Weschera 的 M4 Max 128GB 完整阶梯（含 raw JSON，可复现）
- orcarouter KL 散度 / top-1 一致率
- oMLX 自带 intelligence benchmark 的 4bit vs 8bit
- 各 4090/5090/DGX Spark 数字
- bluehawana 并发基准（Apple Silicon 128GB，8-bit）

### ❌ 明确不可信
- `jayPark777/...-Axon-MLQT` 的 6.43GB / 55.19 tok/s / 99.2% PPL 保留
- TernaryQuench 的"1.58-bit"（磁盘实际 3.02 bit/param）

### 🔲 我仍然没验证的
1. **oMLX 0.7.0 能否真的加载并跑通任一个 27B 产物** —— 架构名在源码里、kernel 也 available，但没 load 过
2. **27B 上的 MTP 接受率 α 与最优 depth** —— M4 Max 上是 3.76(代码)/2.6(散文)，但那是 128GB M4 Max + 0.6.3rc2。oMLX 默认深度在本芯片是 3，本仓库在 125B 上是 1。**不要照抄任何一个，用本仓库的 α 方法实测。**
3. **ANE prefill 在 M2 Max 上的实际收益** —— kernel 已确认可用，但 M4 Max 上的 +11% decode / 3.3× prefill **不能直接外推**：M2 的 ANE 只有 16 核 15.8 TOPS（M4 Max 约 32 核）
4. **TernaryQuench 在本机的实际吞吐**（§2.3 的原生 2-bit kernel vs 稠密回退）
5. **27B 的质量** —— oMLX 的 4bit/8bit intelligence benchmark 是在别的芯片上跑的，本机没验
6. **27B 的并发饱和点**（同机 4-bit 实测 8 并发达 6.49×，但 `ask.py` 的 4 并发上限是照 125B 定的）

---

## 6. 可信度分层

### ✅ 本机实测（我在这台机器上跑出来的）
- 硬件：M2 Max / 96GB / 38 核 / Metal cap 85.9GB / 409.6GB/s
- `GET /api/status` → oMLX 0.7.0，`qwen35_prefill` **available: true**（ANE patch available）、**`bonsai` kernel available: true**
- `patches/bonsai_qmv.py` 源码：1/2-bit affine、decode 区间 M≤5 才走 Metal kernel
- oMLX 源码中 `Qwen3_5ForConditionalGeneration` 支持、层数参数 64/48、MTP 兼容 `qwen3_5*`、a8 只支持 Q4/Q5 GS64 且与 ANE 互斥
- ModelOpt-NVFP4 加载器专写给 `unsloth/Qwen3.8-27B-NVFP4`
- 各量化产物的精确字节体积与下载/点赞数
- `oQ3-mtp` 的 config 实测：`bits 3 / group_size 64 / affine`，**100 条逐张量覆盖**
- 当前实例运行统计：`avg_prefill_tps 63.1 / avg_generation_tps 39.3 / cache_efficiency 76.4`

### ✅ 一手来源（官方 config / HF API）
- 27B 全部结构参数
- Prism ML 全部产物尺寸（`mlx-2bit` 8.595 GB 含 `hadamard.json`；GGUF PTQ1_0 5.947 / PQ2_0 7.206 / F16 53.808 GB）
- Jundot oQ4e 的逐张量 5-bit 覆盖
- 官方推荐采样参数

### ⚠️ 社区第三方数据，同型号/同引擎但**不在本机**
- **omlx.ai 的 M2 Max 38c/96GB 基准**（同机同型号，oMLX 0.6.1，MTP+ANE 关）——**最可信的一条**，但版本 0.6.1 vs 本机 0.7.0
- Weschera 的 M4 Max 128GB 完整阶梯（含 raw JSON）
- **TokenAI-zer 的 oQ2–oQ8 KL 表**（5 条 prompt，样本小；作者自己标注参考基准是 bf16 转换版而非原始 GGUF）
- Prism ML 官方声称的 98.2% 保留（vendor claim）
- oMLX 自带 intelligence benchmark 的 4bit vs 8bit
- 各 4090/5090/DGX Spark 数字
- bluehawana 并发基准

### ❌ 明确不可信
- `jayPark777/...-Axon-MLQT` 的 6.43GB / 55.19 tok/s / 99.2% PPL 保留

### 🔲 我仍然没验证的
1. **oMLX 0.7.0 能否真的加载并跑通任一个 27B 产物**
2. **Bonsai 2 在 M2 Max 的实际吞吐**——官方只给了 M5 Max 的 46.8–58 tok/s
3. **27B 上的 MTP 接受率 α 与最优 depth**——M4 Max 上是 3.76(代码)/2.6(散文)，oMLX 默认在本芯片给 3，本仓库在 125B 上是 1。**不要照抄任何一个。**
4. **ANE prefill 在 M2 Max 的实际收益**——kernel 已就绪，但 M4 Max 的 +11% decode / 3.3× prefill 不能外推（M2 的 ANE 只有 16 核 15.8 TOPS）
5. **三值在 logits 层面和 benchmark 层面矛盾**——官方 98.2% vs oQ3 的 KL 比 oQ4 差 2.4 倍，哪个更代表你的任务未验证
6. **27B 的并发饱和点**（同机 4-bit 8 并发达 6.49×，但 `ask.py` 的 4 并发上限是照 125B 定的）
7. **§5.5.4 的决策表依赖「pipeline 是串行还是并发」** —— 这取决于你的实际用法，我目前是按两种情形分别给的

---

## 7. 建议的下一步

1. **下 `mlx-community/Qwen3.8-27B-4bit`（16.05GB）**，oMLX 加载，验证兼容性。
   ⚠️ 当前 125B 占 78GB，两者无法共存，加载前需卸载。
2. **下 `Jundot/Qwen3.8-27B-oQ4e-mtp`（16.97GB）做 A/B**——同样 4-bit 基础，敏感张量 5-bit，理论上质量更好。**这是社区在 M4 Max 上跑出 53–72 tok/s 用的那个。**
3. **开 ANE prefill**（本机 kernel 已就绪）：
   ```
   qwen35_ane_prefill_enabled = true
   qwen35_ane_prefill_fraction = 0.5      # 0.53 默认，0.5 社区实测更好
   qwen35_ane_prefill_gdn = false          # 社区最优配方
   qwen35_ane_prefill_max_layers = 64
   ```
   **不要开 `qwen35_oq_a8_enabled`**（本机 INT8 无加速单元，且与 ANE 互斥）。
   ⚠️ **改 settings 后必须重启服务才生效**——改文件不重启等于没改。
4. **开 TurboQuant KV 4-bit**（长上下文立竿见影，零风险）。
5. **用本仓库的 α 测量方法重跑 27B 的 MTP depth**，扫 k=1/2/3/4，**同时记录 tok/cycle**。
   预期：本仓库 7 个任务以代码为主 → 接受率应接近 M4 Max 实测的 99.6% 那档，**不是散文那档**。
6. **用本仓库的 7 任务 + E1–E10 分类跑质量对照**。
   ⚠️ 上一版这里写"4-bit 损失很可能看得见"，**依据不足已更正**——oMLX 的 4bit/8bit 对比在它自己的样本量下四项全不显著。
   正确的对照对象是 **4-bit vs Ternary oQ3**（KL 差 2.4 倍，这才是有信号的方向），
   并且**必测 tool calling / 精确输出**——低位元量化在这里掉得最狠（Qwen3.6-27B 前例：TauBench 82.9→61.3，−26%）。
7. **降并发上限这条可能要改** —— 27B 在同机 8 并发达 6.49×，125B 的 4 并发上限是权重量出来的，不是原理。**实测后调整 `ask.py` 的 `MAX_WORKERS`。**
8. **多模型 pipeline 的场景，追加下 `TokenAI-zer/Ternary-Bonsai-2-27B-MLX-oQ3-mtp`（13.86GB）**——
   stock oMLX 直接跑、带 MTP head、KL 0.035。**先明确 pipeline 是串行还是真并发**（§5.5.4），再决定要不要为它牺牲质量。
9. **暂不碰**：官方 Bonsai 2 MLX（需专用 runtime）、BF16（质量无优势，见 §5.5.2）、DFlash（未合并 PR）、INT8-a8（本机无收益）。

### 部署 checklist（社区共识 + 本仓库纪律合并）
- [ ] 卸载当前 125B，确认 27B 权重 + KV 全部落在 85.9GB Metal cap 内
- [ ] **只加载一个模型服务器**——有别的模型占 GPU 时不测任何东西
- [ ] `GET /api/status` 确认 `qwen35_prefill.available == true`
- [ ] 改 settings 后**重启 oMLX**（launchd: `launchctl kickstart` 或 `omlx restart`）
- [ ] 记 TTFT / TPOT / prefill tok/s / decode tok/s **分开记**，不要只记 tok/s
- [ ] 用真实 input/output 长度测，不要拿 single-stream 代替
- [ ] 保留 10–15% 内存 headroom
- [ ] thinking on / off 各跑一次（xhigh 默认太费 token，`medium` 性价比更高）

---

## 8. 社区陷阱清单（全部来自真实翻车）

| 陷阱 | 后果 |
|---|---|
| ANE kernel 没装（0.6.x） | 静默回落到 GPU-only 路径，**不报错** |
| 改 settings 不重启 | 完全不生效 |
| Ollama 默认 `num_ctx = 2048` | 接 agent 后"忘记"文件，代码重复循环 |
| vLLM 对 hybrid 模型自动关 prefix cache | 启动日志**不提示** |
| llama.cpp 非 Q4 KV cache | **静默 fallback 到 CPU**，社区要专门用 fork |
| 量化越小一定越快 | **反例：Q3 比 Q4 慢 2.7×** |
| MTP 一定加速 | **反例：M4 mini 上 n-max4 反而从 5.91 掉到 3.05** |
| 接受率越高越快 | k=7→14 接受率 98.7%→68.7%，**生成反而快 27%**；决定吞吐的是 tokens-per-forward |
| 1-bit 最快 | 27 tok/s 但"难以坚持给出答案" |
| 老 llama.cpp | `unknown model architecture: qwen35` |
| 两个模型服务器同驻 | N=6 崩塌 + Metal OOM |
| 只看模型文件大小推算显存 | 漏了 KV / buffer / MTP state / projector / OS |
| 拿 4-bit 跑 tool calling 就完事 | 前代 Qwen3.6-27B：数学 benchmark 几乎不掉，**TauBench tool calling 82.9→61.3（−26%）** |
| 沿用官方 benchmark 验收 | 社区指出 27B 与 3.6-27B **HF config 完全相同、零变更**，却宣称 DeepSWE +217% |

**最后一条最值得记住**：Qwen3.8-27B 和 Qwen3.6-27B 的 HF config **一模一样、零差异**。
所以**任何"27B 比 3.6 强很多"的说法都不能靠 config 或官方表格采信——必须用你自己的任务复现。**
本仓库那 7 任务 + 56 条要求账本 + E1–E10 错误分类，就是为这件事准备的。

---

## 9. 参考

**一手**
- 模型：https://huggingface.co/Qwen/Qwen3.8-27B
- 量化产物：https://huggingface.co/mlx-community/Qwen3.8-27B-4bit ／ https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp
- TernaryQuench：https://huggingface.co/penkia/TernaryQuench-Qwen3.8-27B-MLX
- 引擎：https://github.com/jundot/omlx

**实测仓库（本报告最有价值的两个来源）**
- ⭐ **Weschera/Qwen3.8-27B-oMLX-MTP-Mac** —— M4 Max 完整阶梯 + 原始 JSON + bench 脚本 + 踩坑清单
  https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac
- ⭐ **bluehawana** Apple Silicon 并发基准（含方法论警告）
  https://huggingface.co/datasets/bluehawana/qwen3.8-27b-apple-silicon-concurrency

**基准库**
- oMLX 社区基准（438,575 条）：https://omlx.ai/benchmarks
- ⭐ **本机型号的那条**：`Qwen3.8-27B-MLX-4bit on M2 Max (38c)` https://omlx.ai/benchmarks/performance/y474epho
- llamaperf oMLX 汇总：https://llamaperf.com/engine/omlx

**失败案例与论文**
- TQ1_0 对普通模型输出乱码：https://github.com/ggerganov/llama.cpp/issues/15193
- CAT-Q：arXiv 2606.26650｜1.58-Bit PTQ：arXiv 2608.01078
- 三值综述：http://runaihome.com/blog/ternary-llm-1.58-bit-quantization-home-lab-2026
