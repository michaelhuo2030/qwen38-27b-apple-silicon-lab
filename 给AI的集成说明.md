# 给 AI 的集成说明：把这台机器上的 Qwen3.8-Flash-Next 接进你的 pipeline

> **这份文档是自足的。** 把它整个交给 AI（连同 `src/ask.py` 和
> `examples/pipeline_integration.py`），它就能写出可用的集成代码。
> 里面没有任何"你需要注意"的事项——那些都已经做进代码了。

---

## 1. 一句话

```python
import sys
sys.path.insert(0, "/Users/a1-6/Documents/mtp-depth-lab/src")
from ask import Assistant

client = Assistant()                    # 整个进程构造一次
text = client.ask("你的 prompt", mode="code").text
```

**不需要设任何环境变量。不需要指定 depth。不需要管理并发。** 三件事都已经在代码里处理好了。

---

## 2. 完整可跑的集成模板

````python
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/Users/a1-6/Documents/mtp-depth-lab/src")
from ask import Assistant

# 整个进程一次。构造耗时 0.1s——它会先读当前配置，已经对就不写，
# 所以不会触发 47 秒的整模型重载。
client = Assistant()


def complete(prompt: str, *, mode="code", temperature=None,
             max_tokens=1200) -> str:
    """一次补全，已清理好，可直接用。

    mode="code"     剥掉模型加的 markdown 围栏（落盘/编译/执行用）
    mode="markdown" 保留中间代码块的围栏（让它写文档时用）
    mode="text"     同 code
    """
    ans = client.ask(prompt, mode=mode, temperature=temperature,
                    max_tokens=max_tokens)
    for w in ans.warnings:                 # 剥了围栏 / 撞了 token 上限等
        print(f"[warn] {w}", file=sys.stderr)
    if not ans.text:
        raise RuntimeError("补全失败，见上方 warnings")
    return ans.text


def complete_many(prompts, *, mode="code", max_tokens=1200) -> list[str]:
    """并发批量。并发数已在库内封顶为 4，这里传多少都会被压到 4。"""
    return client.ask_many(prompts, mode=mode, max_tokens=max_tokens)
````

命令行也可用（适合手工验证）：

```bash
python3 /Users/a1-6/Documents/mtp-depth-lab/src/ask.py "写个回文判断函数" --mode code
```

---

## 3. `mode` 到底是什么（唯一需要理解的一件事）

这不是质量开关，是**格式开关**。

模型在三个代码任务上都无视了题面里的「不要 markdown 标记」，
输出用三个反引号包起来：

````
```python
def f(): pass
```
````

**带围栏的比例 69–81%，在 T=0（贪心解码）下也是 100%**——不是采样噪声，
prompt 改不动它。所以客户端必须处理。

区分规则：

| 你要的 | mode | 行为 |
|---|---|---|
| 纯代码，落盘/编译/管道 | `"code"`（默认） | 剥掉包裹整个输出的围栏，顺带删残留裸围栏行 |
| Markdown 文档（README/博客/文档） | `"markdown"` | **只**剥整体包裹，**保留中间的代码块围栏** |
| 散文、抽取、问答 | `"text"` | 同 `code` |

**为什么这样切**：包裹整个输出的围栏永远不是你想要的（会让文件语法错误）；
而 Markdown 文档永远不会以 ``` 开头结尾——它的围栏在中间，那正是你要的。

判断用**围栏配平**：开头是围栏，且剥掉后剩余围栏数为偶数 → 剥。
不确定时**选"留下可用文件"那一边**（模型只吐了开围栏没闭合时，剥掉能拿回干净代码）。

11 条测试覆盖：`python3 .../src/test_ask.py`

---

## 4. 已测数据（不是估计）

### 4.1 并发

在**这台机器**上实测，聚合吞吐：

| workers | 成功 | 墙钟 | 聚合 tok/s |
|---|---|---|---|
| 1 | 1/1 | 5.2s | 30.0 |
| 2 | 2/2 | 7.9s | 40.8 |
| 4 | 4/4 | 14.6s | 45.3 |
| 8 | 8/8 | 33.7s | 46.1 |

**吞吐在 4 就饱和（45.3 → 46.1 只多 2%），延迟翻倍。** 这活儿是内存带宽瓶颈的。
所以 `Assistant.MAX_WORKERS = 4`，`ask_many` 内部强制封顶——传 16 也会被压到 4。

### 4.2 温度由路由器自动选

不需要你指定。路由器按 prompt 分类给出温度，实测有效率：

| 任务 | T≤0.6 | T=0.9 | T=1.2 | T=1.5 |
|---|---|---|---|---|
| 写 Python 代码 | 5/5 | 5/5 | 5/5 | **5/5** |
| JSON 结构化输出 | 5/5 | 5/5 | 5/5 | 4/5 |
| 概念解释（多要点） | 5/5 | 3/5 | 3/5 | **3/5** |
| React 组件 | 5/5 | 5/5 | 5/5 | **2/5** |

规律：**温度买走的是"覆盖率"，不是正确性**——退化的方式是漏掉题目点名要的某一点，
不是算错。写代码最抗温度。

**你不信路由器就直接覆盖**：`client.ask(p, temperature=0.2)`。
每个返回值都带 `ans.why`（温度怎么定的），所以决策是摊开的，不是黑箱。

### 4.3 温度不是省算力的旋钮

T=0 → T=1.5，全部 182 次生成的 α 只从 87.1% 掉到 81.5%。**别指望调温度提吞吐。**

---

## 5. 返回值：每次决策都摊开给你看

```python
ans = client.ask("写个缓存", mode="code")
ans.text             # 清理后的正文
ans.profile          # 路由器判定的任务类型
ans.temperature      # 实际温度
ans.alpha            # 本次 α（效率指标，不是质量）
ans.gen_tps          # 吞吐
ans.completion_tokens# 实际 token 数
ans.fence_stripped   # 客户端是否替你剥了围栏
ans.why              # 温度怎么决定的
ans.seconds          # 耗时
ans.warnings         # ["模型又套围栏了，已处理", "撞了 token 上限…"]
```

**两个会自动告诉你的事**：

- 模型又套围栏了 → `warnings` 明确说明"提示词要求了但它没听，客户端已处理"
- 撞到 token 上限 → `warnings` 明确说"答案可能被截断，调 max_tokens 再下结论"
  （实测 React 在 T=1.5 撞 1800 上限被截断）

---

## 6. 会自动处理、你不该再操心的事

| 事项 | 代码里怎么处理 |
|---|---|
| API key | 自动从 gitignored 的 `src/.ask.local.json` 读，**不用设环境变量** |
| depth / MTP | 固定 `depth=1, mtp=True`；**先读当前配置，已对就不写**，所以没有 47 秒重载 |
| 写入是否真的生效 | 读回断言。settings 写入静默失败过一次，把「开 vs 关」变成「同配置比同配置」，报出 6.9% 的假差异 |
| 并发上限 | `ask_many` 内部封顶 4 |
| 请求失败 | 重试 3 次 + 指数退避；耗尽后返回空文本 + `warnings`，不抛异常打断整批 |
| 围栏 | 按 mode 自动处理 |
| 截断 | 检测并 warn |

---

## 7. 硬约束（环境层面的，改不了）

- **模型常驻 78GB**（你机器可用内存 92.3GB 里的 78GB）。跑 pipeline 时别开别的重应用。
- **服务地址** `http://127.0.0.1:8091`，由 launchd 常驻（`ai.local.omlx`），重启后自动拉起。
- `Assistant()` 会探活，服务没起来会直接抛 `RuntimeError`，不会静默失败。

---

## 8. 交给 AI 时的建议

**把这份文档 + `src/ask.py` + `examples/pipeline_integration.py` 一起给它。**
只给 import 那几行的话，AI 大概率会：

- 把 `Assistant()` 写在函数体里（每次调用都重新构造）
- 手动管理 ThreadPoolExecutor 并开到 16
- 用 `response.choices[0].message.content` 取文本（**这层接口返回的是 `ans.text`**）

这三样本文档都挡掉了。

---

## 9. 自检（先跑这个再接进生产）

```bash
python3 /Users/a1-6/Documents/mtp-depth-lab/examples/pipeline_integration.py
```

它会验证：code 模式围栏被剥且 `compile()` 通过、markdown 模式围栏保留且以标题开头、
3 并发批量可用。任何一项不过就别接生产。
