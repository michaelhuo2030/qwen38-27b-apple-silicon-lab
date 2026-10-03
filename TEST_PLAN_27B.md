# Qwen3.8-27B 实验规划与故障预案

> 适用机器：Apple M2 Max / 96GB / 38 GPU 核 / 409.6 GB/s，Metal cap 82 GiB
> 运行时：oMLX 0.7.0
> 配套代码：`src/ladder27.py`（harness）、`src/test_ladder27.py`（自测，82 项）、`~/mtp_depth_lab/qwen27ctl.py`（切档）
> 基线：本仓库 125B Flash 那一轮踩过的坑，全部继承（见 §1）

---

## 0. 怎么用这份文档

三条读法：

- **只想知道某个数字能不能信** → 直接看 §4「数据有效性判据」
- **要跑某个实验** → 跳到 §3 对应小节，里面有前置闸门、步骤、判据
- **出问题了要排查** → §1 按现象查表

**这份文档的立场**：一份没有判据的实验规划等于没写。所以每个实验都带
「什么情况下这轮数据作废」——作废条件比成功条件写得更细，这是有意的。

---

## 1. 通用故障模式

左边是现象，右边是根因与对策。**每一条都来自真实翻车，不是假想。**
标 ⭐ 的是这次部署 Qwen3.8-27B 时新遇到的。

### 1.1 配置与生效

| # | 现象 | 根因 | 对策（已落进代码） |
|---|---|---|---|
| ⭐P1 | 改完 settings，实验跑出"on vs off"其实都是 on | 写入被静默 401 吞掉 | `_put_settings` **断言回读**，不一致直接抛。注释里写明：本仓库曾因此报出 6.9% 的假差异 |
| ⭐P2 | admin 路由 401，但同样的 Bearer 打 `/v1/*` 正常 | `/admin/api/*` 认 **session cookie** 不认 Bearer | 所有 admin 调用前先 `POST /admin/api/login` |
| P3 | settings 写成功了但行为没变 | 改 settings 只对**下次加载**生效 | `activate()` = 卸载 → 写 → 加载 → 断言单驻，顺序固定 |
| ⭐P4 | 新模型在 `list` 里 mtp/ane 显示 `?` | 读 settings 失败被吞 | `list` 区分「未下载 / 在盘上 / 已加载」，读不到就显示 `?` 而不是假装是 off |
| P5 | `qwen35_oq_a8_enabled=True` + `ane=True` 被服务器拒绝 | 两者互斥（`model_settings.py:504`） | `validate_config` 在**构造 Ladder 时**就拦，不发任何请求 |
| P6 | 实验跑完才发现配置是旧的 | 等待太久，中间有人/别的进程改了 | 每轮实验前 `assert_single_resident()` + 回读 settings，附在结果里 |

### 1.2 静默错误（最危险的一类）

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| ⭐P7 | 开了 TurboQuant KV / ANE 之后输出变了，但没有任何报错 | 这两者的设计就是"换算法不报错"；社区明确说 INT8-activation **会改输出** | `sentinel()` 可行性哨兵：任何配置变更后先用 T=0 跑 4 个确定性探针，**输出必须逐字一致**，不一致就停下 |
| P8 | 第二次请求特别快，被当成性能提升 | prefix cache 命中 | 记录 `total_cached_tokens` 前后差；重复测量**必须用不同 prompt**（`sweep_concurrency` 每个请求带序号） |
| ⭐P9 | thinking 开与关的对比完全没意义 | `reasoning_effort` 默认 **xhigh**，在简单 prompt 上狂烧 token | 每个请求显式带 `enable_thinking`；采样组随 thinking 切换（见 §2.1） |
| P10 | 模型无视"不要 markdown 标记" | 这不是 bug，是这个模型家族的稳定行为——125B 上带围栏率 LRU 73%/Shape 81%/React 69%，**T=0 也是 100%** | 用 `mode="markdown"` 剥，而不是指望模型听话；并在结果里记 `fence_stripped` |
| P11 | 同一配置两次跑结果不同，以为是模型随机 | 采样没冻 | 做 A/B 时一律用 `sampling="frozen"`（T=0） |

### 1.3 测量方法学

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| P12 | tok/s 好看但用户体感差 | 总耗时被 TTFT 主导 | 记录 `ttft` 与 `generation_tokens_per_second` **分开**；v1 任务只有 11–70 token 时栽过 |
| P13 | 短任务的 tok/s 测出来虚高 | 输出太短，TTFT 占比过大 | 沿用 `tasks.py` v2 规则：**每个任务实际产出 ≥ 250 token** |
| P14 | distinct-2 指标显示 1.000，以为是完美重复 | 18 token 的短样本上该指标退化 | 短文本不用 distinct-2（125B 上已确认） |
| ⭐P15 | 并发一上去数据就崩 | **两个模型服务器同驻**：社区实测第二个 30GB 模型常驻时 N=6 崩塌到 7.6 tok/s 并 Metal OOM；独占时 N=16 无问题 | `assert_single_resident()` 硬闸，不满足直接抛 |
| ⭐P16 | 换了机器/模型后并发上限照抄 | 125B 的 4 是 78GB 权重摊不动的必然结果，**不是原理** | `sweep_concurrency()` 必测。27B 只有 13.9–30GB，omlx.ai 同型号实测 8 并发有 6.49× 加速 |
| P17 | 并发结果忽高忽低 | 批大小带来的 prefill 串行 vs decode 摊薄不同 | 固定 output 长度；同时报聚合 tok/s 与每请求 tok/s |
| ⭐P18 | **back-to-back 跑分莫名掉 25%，换配置又"好了"** | **GPU 热降频**。M3 Max 实测：冷却 5min → 21.9 tok/s，连着跑 → 16.4，休息 3min → 22.0。GPU 时钟 18 个采样内 1368→743 MHz。**`pmset -g therm` 显示 thermal=0 完全看不见** | **每次基准之间强制 ≥3 min 静置**（不休眠不合盖）；跨组只信冷却过的组；测完一轮 depth sweep 后**重测 depth 1 作回归** |
| ⭐P19 | 0.7.0 上"热降频"根本没发生 | 0.7.0 新增 `server.gpu_keep_warm_interval`，请求后保持 GPU 非 idle 最多 5 分钟 | 实验期间**显式设 0 或记录当前值**，否则 P18 的对策失效、热降频实验测不到东西 |
| ⭐P20 | MTP 配了但一点没提速，还以为测了 | `mtp.*` 张量没进 checkpoint index → oMLX 静默不启用 Lightning MTP，只报一行 INFO | **以运行时日志为准**：`Speculative backend selected: Lightning MTP`。看配置文件不算数 |
| ⭐P21 | 并发跑着跑着报 `[convert] Only length-1 arrays...` | [omlx#3906](https://github.com/jundot/omlx/issues/3906)：MTP 投机回滚把**向量** trim 长度喂给只吃标量的 `turboquant.trim()`。复现模型就是 Qwen3.8-27B-oQ4e-mtp，触发条件 2+ 并发 + TQ 开 + max_tokens≥500 | 并发实验期间**强制 `turboquant_kv_enabled: false`**（本来就要关）；真开了又并发就优先怀疑这条 |

### 1.4 数据本身出错

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| ⭐P18 | 分数差异只有 1–3 个点就下结论 | 样本量不够。例：oMLX 自家 benchmark 的 4bit vs 8bit，LiveCodeBench 差 3.0±1.9%（1.6σ）、HUMANEVAL 1.1σ，**四项全部不显著** | 报差异必须同时报 n 与标准误；不显著就写"不显著" |
| P19 | α（接受率）在不同任务间比较得出错误结论 | 简单平均把 20-token 和 700-token 的回答同权 | `weighted_alpha()` token 加权。97/110=88.18 而简单平均是 80.0 |
| P20 | 校验器比题目窄，结论方向反了 | 本仓库 125B 时 T10 查 `useState` 而题目要 `useRef`+`useEffect` | 沿用 `AUDIT_CHECKLIST.md` 的 E1–E10 分类；要求显式登记为"已检查"或"未测量+原因" |
| P21 | 校验器恒真 | canonical 只取函数名 → 按构造 100% 一致（125B 实测） | 改用剥 docstring 的 AST 指纹；`audit_validators.py` 守这条 |
| P22 | 指标失效但没人发现 | 指标在某种输入上退化 | `audit_metrics.py` + 差分夹具 |
| ⭐P23 | 半下载的模型被当成完整模型 | `config.json` 才 13KB 必定先下完，oMLX 的 `model_discovery` 不校验分片完整性；我加的 `.downloading` 标记它根本不认 | 先下到 `.staging/`（`model_discovery.py:1769` 跳过 `.` 开头目录），完整后 `mv` |
| ⭐P24 | 下载跑着跑着卡住不动 | **两个 `hf download` 进程抢同一个 `.lock`**，互相空等 1000 秒（实测卡死 16 分钟） | 脚本用 `mkdir` 原子锁防重复启动 |
| ⭐P25 | kill 掉了进程但没真停 | 杀的是**子进程**，父脚本 shell 还活着，在不断重启子进程 | 要杀进程树。`ps -eo pid,ppid` 确认 PPID 一起清 |
| ⭐P26 | `pkill -f "xxx"` 静默不生效 | 模式里含 CJK 时 macOS `pgrep/pkill` 报 `illegal byte sequence`（exit 3），而 `pgrep … | head -2 \|\| echo "未运行"` 把这个错误吞成"正常输出" | 用 `ps -eo pid,command \| grep` 走 awk 取 PID；或模式里用字符类 `[h]f download` 避开自匹配 |
| ⭐P27 | bash 脚本报 `bad array subscript` | macOS 自带 **bash 3.2**，负数下标 `PIDS[-1]` 是 4.2+ 才有的 | 用 `PIDS[$((${#PIDS[@]}-1))]` |
| ⭐P28 | bash 报 `LOG?: unbound variable` | 全角 `）` 紧贴 `$LOG` 时，bash 3.2 把多字节字符吃进变量名 | 一律用 `${LOG}` 花括号 |
| ⭐P29 | 国内镜像"测速同速"，装了之后下载直接失败 | hf-mirror 现在对 `/resolve/` 返回 **308 跳回官方**；用 `curl -L` 测速会跟着跳，等于拿同一个源比两遍 | 测速前先 `curl -sI` 看 `location`；有 3xx 就是代理不是源 |
| ⭐P30 | 脚本日志里 `huggingface.co`，但我明明设了 `HF_ENDPOINT` | 报错文案写死默认端点 | `HF_DEBUG=1` 看实际请求的 URL，别信文案 |

### 1.5 资源与容量

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| ⭐P31 | 四档合计 84.55GB > 82GiB Metal cap | 算好了就是装不下 | **明确按需换出**，不要设计成全常驻。每轮实验前 `assert_single_resident` |
| P32 | 8-bit 档加载失败/OOM | 30GB 权重 + KV + 激活逼近 cap | 加载前查 `model_memory_max`；必要时长上下文先降到 8K |
| P33 | 跑到一半 OOM | 长上下文的 KV 累积 | 262K 时 bf16 KV 是 16GB，和 4-bit 权重一样大 → 必须开 TurboQuant 4-bit |
| ⭐P34 | 上一档的 KV cache 残留影响下一档 | 换档时只换了权重 | 每档 `activate()` 强制 unload → load，不复用引擎 |
| P35 | 磁盘写满 | 99GB(125B) + 84.55GB(四档) + 其他 | 保持 >50GB 余量；每轮结束检查 `df` |

---

## 2. 实验规划

### 2.1 采样策略：先定死，不能路由

⚠️ **不要用 `ask.py` 的 router**。那套温度 band 是在 125B 上拟合的，而 27B 官方推荐的采样参数**随 thinking 模式而变**：

| 模式 | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| thinking **on** | 1.0 | 0.95 | 20 | 0.0 |
| thinking **off** | 0.7 | 0.80 | 20 | 1.5 |
| frozen（做 A/B 用） | 0.0 | — | — | — |

两个模式的温度不同 → **若沿用旧 router，"温度"和"thinking"两个变量会缠在一起，实验直接失效**。
这是 `ladder27.py` 不复用 router 的唯一理由（清洗与判分部分照用 `ask.py`/`quality_checks`，那些是模型无关的）。

**主实验一律 `think_off`**，理由：本仓库的 7 个任务都是代码/结构化输出，
社区实测 MTP 接受率在代码上是 99.6% vs 散文 78%，而 thinking 会烧掉大量 token 稀释这个信号。
`think_on` 留作单独一轮对照。

### 2.2 执行顺序（依赖关系决定）

```
E0 冒烟          ternary 档能否加载 + 能否生成     ← 一切的前提
  ↓
E1 正确性哨兵    每档加载后先验输出没坏            ← 挡住静默错误
  ↓
E2 MTP depth     α + tok/s，扫 k=0/1/2/3/4        ← 核心研究
  ↓
E3 ANE A/B       27B 上首次验证（M2 Max 无先例）
  ↓
E4 TurboQuant A/B 同上，且要过 E1 的哨兵
  ↓
E5 精度阶梯质量   四档 × 7 任务                    ← Ladder 的主目的
  ↓
E6 温度 × 质量    只在最优档位上做
  ↓
E7 并发饱和点     决定 ask.py 的 MAX_WORKERS
```

**E0 不过，后面全部不做。** 不要在没确认能加载的情况下先写实验脚本的"预期结果"。

---

## 3. 各实验详细方案

### E0 · 冒烟测试：能不能加载、能不能生成

**目的**：验证 oMLX 0.7.0 真的支持 `Qwen3_5ForConditionalGeneration`。
架构名在源码里出现 ≠ 能跑——**从没真的 load 过**。

**前置闸门**
- 模型目录完整：`config.json` + `model.safetensors.index.json` + ≥1 个 `model*.safetensors`
- 暂存区已清空（没有半成品）
- 125B 已卸载（当前状态：0.0/82 GiB）

**步骤**
1. `Ladder("ternary", think=False, ane=True, turboquant=4, mtp=True).activate()`
2. 记下加载耗时与 `model_memory_used`
3. 跑一次最简单的生成

**判据**
- 加载不报错，且 `loaded_models == [目标]`
- 能生成非空文本
- `GET /api/status` 的 `custom_kernels.qwen35_prefill.available == true`（本机已确认为 true）

**可能的坑**
- ⚠️ 首次加载可能因 hybrid 模型的 KV 估算触发假 OOM。oMLX 0.6.3rc2 修过一个
  "4× 高估导致假 OOM"的 bug；本机是 0.7.0 应该已修，**但如果撞上假 OOM，
  第一反应是怀疑 bug 而不是加内存**。
- ANE prefill 可能在 M2 Max 上编译失败（M2 的 ANE 只有 16 核 15.8 TOPS）。
  失败时**退回 ane=False 继续**，并把这个"未验证"明确记进结论。

**回滚**：`ane=False, turboquant=None, mtp=False` 的最小配置能跑通就算 E0 过。

---

### E1 · 正确性哨兵：挡静默错误

**目的**：TurboQuant KV / ANE 都是"换算法但不报错"的加速。
它们可能让输出**变化**而不抛异常。性能测得再好，输出错了就是废的。

**方法**：`ladder27.sentinel()`，4 条确定性探针，`sampling="frozen"`（T=0）

| 探针 | 内容 | 期望 |
|---|---|---|
| arithmetic | `2+2*3?` | 含 `8` |
| json | 输出指定 JSON | 含 `"k"` |
| code | 写 `add(a,b)` | 含 `def` |
| repeat | 复述 `banana` | 含 `banana` |

**判据**
- 全部通过 → 配置可用
- 任一失败 → **停下，先排查**。不要"先测性能，回头再看质量"。

**关键强度**：同配置下 T=0 的输出必须**逐字一致**。若两次跑出不同文本，
说明配置里还有别的变量在漂，先找到它。

---

### E2 · MTP depth sweep + α（核心）

**目的**：27B 上的最优 MTP depth。这是本次的**核心研究问题**。

**为什么不能抄现成结论**
| 来源 | 结论 | 为什么不适用 |
|---|---|---|
| oMLX 默认 | dense Qwen3.5 家族 **M5=4，其他芯片=3** | 本机是 M2 Max，所以默认会给 3 |
| 本仓库 125B 实测 | **depth=1** | 那是 125B MoE（6B 激活），27B 是 dense 27B |
| M4 Max 128GB 社区 | k=3 最优（代码 72.1 / 散文 53.3 tok/s） | 换机器换量化换版本 |
| M4 Mac mini 反例 | n-max 4 反而从 5.91 掉到 3.05 | **反例证明必须扫，不能抄** |

**方法**：`ladder27.sweep_depth(depths=[0,1,2,3,4], tasks=TASKS, reps=3)`
- k=0 即 MTP 全关，作为基线
- 每档都做 unload → load（settings 对已加载引擎不生效）
- 7 个任务覆盖不同熵层级（沿用 `tasks.py` 的分层设计）

**判据**
- α 用 **token 加权**（`weighted_alpha`），且 `n_mtp_lines > 0`（0 行和"没跑"长得一样）
- 每格 n≥3，且**看 min–max 不只看均值**
- 记录 `tok_per_cycle` —— 社区公式：**decode 上限 = 原始 decode × tokens-per-cycle**

**预期（待验证，不是结论）**
7 个任务以代码为主 → α 应接近社区实测的代码档（99.6%，tok/cycle 3.76），
**而不是散文档**（78%，2.6）。若实测落在散文档，说明有变量没控住。

**A/B 纪律**：`mtp_on_off` 这个对比必须**同一天、同一 prompt、只差一个 setting**。

---

### E3 · ANE prefill A/B

**目的**：27B 上首次验证 ANE 预填。**本机没有任何先例。**

**已知**
- kernel 已就绪（`/api/status` → `qwen35_prefill.available: true`）
- 代码里**没有 M4/M5 门控**，唯一的硬件注释是关于 4GiB 设备地址窗口的分 bank 重试
- 社区在 M4 Max 128GB 上测到 prefill 83→273.7 tok/s（3.3×）、decode +11%

**不能外推的部分**：M2 Max 的 ANE 只有 16 核 / 15.8 TOPS，M4 Max 约 32 核。
**+11% decode 和 3.3× prefill 都不能直接套到这台机器。**

**方法**
```python
for ane in (False, True):
    L = Ladder("4bit", ane=ane, ...)
    L.activate()
    assert L.sentinel().ok          # ← 先过哨兵
    # 用长 prompt 测 prefill，用长输出测 decode
```
配置用社区最优配方：`fraction=0.5`、`gdn=False`、`max_layers=64`、`dual_ane=True`。

**判据**
- prefill 吞吐与 TTFT 分开报
- **必须先过哨兵**——ANE 用近似 INT8 权重，理论上可能改输出

**已知的负面结果（照抄省事）**
| 配置 | 散文 | 代码 | prefill |
|---|---|---|---|
| ANE on + MTP k=3 | 53.3 | 72.1 | 273.7 |
| ANE off + MTP k=3 | 47.9 | 47.9 | ~83 |
| ANE fraction 0.75 | 更差 | — | 257 |
| ANE + CPU sharing + GDN | 52.3 | 72.0 | 277.4（decode 反而掉） |

→ `fraction=0.5`、`gdn=False` 是实测最优，不要改成默认值 0.53。

---

### E4 · TurboQuant KV A/B

**目的**：长上下文的唯一杠杆。262K 时 bf16 KV 是 16GB，和 4-bit 权重一样大。

**风险（这是它比 ANE 更需要哨兵的地方）**
- 27B 是**混合架构**：48 层 Gated DeltaNet **没有 KV cache**，只有 16 层全注意力有。
- oMLX 的映射是 `mlx_lm.models.qwen3_5 → "gdn-qk-norm-2"`
  （`cache/paged_ssd_cache.py:302`）——**这个映射对混合架构是否正确，没验证过**。
- 出错的话不会报错，只会输出变差。

**方法**
```python
for tq in (None, 4, 8):
    L = Ladder("6bit", turboquant=tq, ...)
    L.activate()
    rep = L.sentinel()      # ← 不可跳过
    if not rep.ok: 记录并停止
```
另外要测长上下文下的行为：8K / 32K / 64K 三点，看 KV 量化后长文是否退化。

**判据**
- 哨兵全过
- 短输出任务的实质有效率与 TurboQuant 关闭时**无显著差异**
- 长上下文下无截断率上升

---

### E5 · 精度阶梯质量对照（Ladder 的主目的）

**目的**：四档之间的质量差，决定"该选哪档"。

**已知的、必须先接受的两个事实**
1. **8-bit 相对 4-bit 的质量优势在统计上不显著。** oMLX 自家 benchmark 四项全部
   1.1–1.6σ（LiveCodeBench 4.7% vs 7.7%，n=300，差 3.0±1.9%）。而 8-bit 慢一半、内存近 2 倍。
2. **对 tool calling / 精确输出，低位元量化掉得最狠。** 前代 Qwen3.6-27B 的实测：
   数学 benchmark 几乎不掉，**TauBench tool calling 从 82.9 掉到 61.3（−26%）**。

→ **所以 E5 的判据不能是"平均分"，必须是"最容易退化的那类任务"。**

**方法**
- 复用 `tasks.py` 的 7 任务 + `quality_checks.py` 的校验器 + `exec_probes.py` 的可执行探针
- 每档每任务 n≥5（125B 那轮的标准）
- **必须包含 tool calling / 精确 JSON 输出**——那是最容易掉的地方
- 沿用 E1–E10 错误分类，每类单独计数，不只看"通过/失败"

**报告纪律**
- 报差异时**必须同时报 n 与标准误**
- 差异不显著就写"不显著"，不要挑显著的那几项
- 沿用本仓库的老规矩：要求必须显式登记为"已检查"或"未测量+原因"

**留出集**：同源验证分数无意义（同源上训的集和验证集高度相关）。若要训练/调参，
必须有 8/16 划分的不相交断言。

---

### E6 · 温度 × 质量

**目的**：温度对质量的影响，以及它对 α 的影响。

**口径警告**（125B 那轮已证实）
- **温度不是 α 的旋钮**：T=0→1.5 只让 α 从 87.1% 降到 81.5%（−5.6pt）
- **模型对"不要 markdown 标记"在所有温度下都无视**，T=0 也是 100% 带围栏
- 高温下的失败形态与低温不同：T=1.5 的 React 任务有 1 次截断 + 2 次塌成 18 token 词沙拉
- T=1 的 JSON 偶发未闭合 `]`（**不是截断**，443 token）→ 所以"闭合性"要单独查

**设计**
- 7 任务 × 6 温度 × 5 重复，T=0 只跑 1 次（确定性）= **182 次生成**（与 125B 同规模）
- 采样组用 `think_off`（0.7/0.80/20/1.5），**不是**思考模式
- 判据用**实质有效率**（剥离格式噪声），不是原始通过率

**新增的 27B 专属检查**
- 截断：`completion_tokens >= max_tokens - 2` → `Answer.truncated`，单独计数
- JSON 闭合性：不能靠"能不能 parse"一刀切——未闭合和格式错误要分开归类
- 词沙拉检测：`completion_tokens < 30` 且任务要求更长 → 单独归为 E8 类

---

### E7 · 并发饱和点

**目的**：定 `ask.py` 的 `MAX_WORKERS`。

**为什么必须测**：`MAX_WORKERS=4` 是 125B 78GB 的必然结果，**不是原理**。
27B 只有 13.9–30GB，同型号社区实测 8 并发有 6.49× 加速。照抄 4 会浪费一半吞吐。

**方法**：`ladder27.sweep_concurrency(prompt, max_tokens=320, sizes=(1,2,4,8))`
- 每个请求 prompt **带不同序号**（否则 prefix cache 让后续请求免费）
- 同时报**聚合** tok/s 与**每请求** tok/s
- `all_ok` 必须为 true——部分失败不能当"完成了"

**判据**：找 `aggregate_tps` 的拐点。拐点之后只有延迟在涨，吞吐增益 <10% 就该停。

---

## 4. 数据有效性判据

**出现以下任一情况，整轮数据作废。** 宁可重跑，不要在坏数据上分析。

### 全局作废条件

| # | 判据 | 检测方式 |
|---|---|---|
| V1 | GPU 上同时有 >1 个模型 | `assert_single_resident()` 抛异常 |
| V2 | 测量的不是目标模型 | 同上 |
| V3 | settings 回读不一致 | `_put_settings` 抛异常 |
| V4 | 哨兵未通过 | `sentinel().ok == False` |
| V5 | 目标目录只有 `config.json` 无分片 | 落地前完整性检查 |
| V6 | 磁盘余量 < 20GB | `df` |
| V7 | 同一配置 T=0 两次输出不同 | 哨兵的 repeat 探针 + 逐字比对 |

### 单次测量作废条件

| # | 判据 | 为什么 |
|---|---|---|
| V8 | `n_mtp_lines == 0` 而 MTP 开着 | 0 行和"MTP 没跑"无法区分，α 无意义 |
| V9 | `completion_tokens < 250`（长任务） | 被 TTFT 主导，v1 教训 |
| V10 | `truncated == True` 但在读质量结论 | 截断会伪装成质量问题 |
| V11 | `error is not None` | 请求失败，**不终止判失败**——重试后仍失败才记 |
| V12 | 缓存增量异常大且 prompt 重复 | prefix cache 污染 |

### 报告作废条件

| # | 判据 |
|---|---|
| V13 | 差异未报 n 与标准误 |
| V14 | 挑出显著的项报告，隐去不显著的 |
| V15 | 校验器要求未显式登记为"已检查"或"未测量+原因" |
| V16 | 同一 prompt 同时用于调参与报告 |

---

## 5. 已有的可复用资产

| 资产 | 位置 | 模型相关？ | 27B 能否直接用 |
|---|---|---|---|
| 围栏解析 `clean`/`unwrap` | `src/ask.py` | ❌ | ✅ 直接用 |
| 7 个任务 + 熵分层 | `src/tasks.py` | ⚠️ 熵分层是通用的，但 27B 可能有新的失败模式 | ✅ 用，重新校准 max_tokens |
| 质量校验器（40 条期望） | `src/quality_checks.py` | ❌ | ✅ 直接用 |
| 可执行语义探针 | `src/exec_probes.py` | ❌ | ✅ 直接用 |
| 审计库 | `tools/qa_audit/` | ❌ | ✅ 直接用 |
| 错误分类 E1–E10 | `AUDIT_CHECKLIST.md` | ❌ | ✅ 沿用，27B 可能新增类 |
| 路由器 | `src/router.py` | ✅ **是** | ❌ **不用**（采样策略随 thinking 变，见 §2.1） |
| MTP 日志解析 | `src/omlx_client.py` | ⚠️ 日志格式相关 | ✅ 格式相同，已在 `ladder27` 里复刻 |
| 并发上限 4 | `src/ask.py` | ✅ **是** | ❌ **重测**（见 E7） |

---

## 6. 复现命令

```bash
cd ~/Documents/mtp-depth-lab

# 0. 自测（不需要模型、不需要 oMLX）
python3 src/test_ladder27.py                    # 82 项
python3 src/test_ask.py && python3 src/test_exec_checks.py
python3 src/audit_validators.py && python3 src/audit_metrics.py
python3 tools/qa_audit/selftest.py

# 1. 看四档状态
python3 ~/mtp_depth_lab/qwen27ctl.py list

# 2. 切档并跑基准（逐档）
python3 ~/mtp_depth_lab/ladder_bench.py --tiers ternary

# 3. 实验（示例：MTP depth sweep）
python3 src/exp_mtp_depth_27b.py --tier 4bit --depths 0 1 2 3 4
```

---

## 7. 维护

这份文档的价值随时间衰减——**社区教程会过期，本机版本会变**。
新增故障模式请直接加到 §1 的表里，并注明是**哪次实验、哪个现象**。
没有出处的推测不要写进 §1；待验证的预期写进对应实验的「预期（待验证）」。

> 上次因为把社区仓库的 0.6.x 安装步骤当成现状，差点白下一个预编译 kernel。
> 详见 §1.4 P29。
