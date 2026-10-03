"""
MTP depth 实验任务集 v2

v1 的缺陷：部分任务输出仅 11~70 token，测到的 tok/s 被 TTFT 和首轮主导，
不是稳态 decode 吞吐。v2 要求每个任务实际产出 >= 250 token。

分层依据（第一性原理）：
  接受率 α ≈ P(草稿头预测 == 骨干 argmax)
  语法强制位（缩进/标点/键名/常见 API）→ 近乎确定 → α 高
  知识/判断/创造位（专名/数字/比喻/措辞）→ 不确定 → α 低
"""

# (key, 熵层级, prompt, max_tokens)
TASKS = [
    (
        "T1_json_structured",
        "极低熵·结构化输出",
        '严格输出一个 JSON 数组，不要任何解释文字，不要 markdown 标记。'
        '数组包含 12 个对象，每个对象有 id(整数)、name(英文产品名)、'
        'price(数字)、category(字符串)、inStock(布尔值)、tags(2 个标签的数组) 六个字段。'
        '内容是 12 个不同的智能家居产品。',
        700,
    ),
    (
        "T2_code_function",
        "低熵·代码生成",
        "只输出 Python 代码，不要 markdown 标记，不要任何解释文字。"
        "写一个名为 LRUCache 的类，要求：使用 collections.OrderedDict 实现；"
        "包含 get(key, default=None)、put(key, value)、__len__、__repr__ 四个方法；"
        "每个方法都要有完整的 docstring，docstring 用英文三引号格式；"
        "容量由 __init__ 的 capacity 参数控制，超出时淘汰最久未使用的键；"
        "再写一个装饰器 timed(fn) 打印函数耗时。",
        600,
    ),
    (
        "T3_code_repetitive",
        "极低熵·重复模式",
        "只输出 Python 代码，不要 markdown 标记。"
        "写一个基类 Shape，含 area() 和 describe() 两个方法，describe 返回 f\"{type} area={area}\"。"
        "然后写三个完全同构的子类 Circle(radius)、Rectangle(width, height)、Triangle(base, height)，"
        "各自实现 area()，并各自有带 docstring 的 __init__。"
        "最后写一个函数 total_area(shapes) 遍历求和。",
        700,
    ),
    (
        "T4_translation",
        "低熵·翻译",
        "把下面这段技术文档翻译成中文，只输出译文，不要解释：\n\n"
        "The deployment pipeline automatically rebuilds the container image from the "
        "current commit, runs the full integration test suite in an isolated network "
        "namespace, and promotes the artifact to the staging environment only after all "
        "checks pass. If any stage fails, the pipeline rolls back to the last known-good "
        "image and notifies the on-call engineer through the incident channel. The entire "
        "process is idempotent, so re-running a failed job from the same commit produces "
        "identical results and does not require manual cleanup.",
        600,
    ),
    (
        "T5_extraction",
        "中低熵·信息抽取",
        "阅读下面这段会议纪要，抽取全部要点，用中文项目符号列表输出，每条以「- 」开头，"
        "不要加任何前言或总结：\n\n"
        "2026年3月12日，产品技术周会。参会：张明、李芳、王强、赵敏。\n"
        "议题一：Q2 排期。张明汇报核心搜索模块重构已进入联调，预计 4 月 15 日完成；"
        "李芳负责的推荐系统 A/B 实验框架延期两周，原因是数据中台接口未就绪。\n"
        "议题二：线上事故复盘。3 月 8 日 14:20 至 15:05，商品详情页出现大面积 504，"
        "根因是缓存击穿导致数据库连接池耗尽。王强提出增加多级缓存和随机过期时间，赵敏负责跟进。\n"
        "议题三：招聘。算法岗 HC 剩余 2 个，已开放 3 周，简历 12 份，安排本周四面。\n"
        "议题四：成本。云资源账单环比上升 18%，赵敏说明主要是日志存储增长过快，"
        "下周提交冷热分离方案。",
        500,
    ),
    (
        "T6_qa_factual",
        "中熵·知识问答",
        "请系统地解释 TCP 三次握手的必要性，总共 500 字左右，分点说明："
        "为什么不是两次，为什么不是四次；"
        "ISN（初始序列号）的作用；"
        "TIME_WAIT 状态为什么必须存在；"
        "以及如果 SYN-ACK 丢失会发生什么。",
        700,
    ),
    (
        "T7_math_reasoning",
        "中高熵·数学推理",
        "某电商平台做促销：原价 200 元的商品，先打 8 折得 160 元，"
        "满 150 再减 30 元得 130 元，然后用会员券打 85 折，最后用积分抵扣 20 元实付。\n"
        "请分步骤计算每一步的金额，并说明如果顺序改成「先减 30 再打 8 折」结果是否相同，"
        "为什么。用中文回答，要有计算过程。",
        600,
    ),
    (
        "T8_creative_writing",
        "高熵·创意写作",
        "写一段 600 字左右的散文，描写深秋雨后独自在江边散步的感受。"
        "要求有画面感、有情绪起伏、有至少两处比喻，"
        "不要重复用字，不要用「静谧」「寂静」这类常见词。",
        800,
    ),
    # --- 以下三条按真实工作负载补入：用户主力语言是 Rust / TypeScript，
    #     且大量产出是文档重写。合成集缺这三类会让 α 映射曲线偏向 Python 生态。
    (
        "T9_rust_impl",
        "低熵·Rust 实现",
        "只输出 Rust 代码，不要 markdown 标记，不要任何解释文字。"
        "实现一个并发安全的异步任务调度器 AsyncScheduler，要求："
        "用 tokio::sync::Semaphore 做并发上限控制；"
        "任务以 FnOnce() -> Future<Output=T> + Send + 'static 的形式注册；"
        "提供 spawn 返回 task_id、join 按 id 等待、shutdown_all 优雅关闭三个 API；"
        "任务 panic 不能拖垮调度器，需要 catch_unwind 或 JoinSet 兜住；"
        "每个公开方法都要有英文三引号 docstring，并标出 # Errors 可能返回的错误。",
        800,
    ),
    (
        "T10_ts_component",
        "低熵·TypeScript 组件",
        "只输出 TypeScript 代码，不要 markdown 标记，不要任何解释文字。"
        "实现一个 React 函数组件 ReasoningPanel，props 为 { frames: ReasoningFrame[]; "
        "isStreaming: boolean; onSeek?: (id: string) => void }，"
        "其中 ReasoningFrame 是 { id: string; title: string; body: string; tokens: number }。"
        "要求：流式生成时新增的 frame 自动滚动到底部，但用户手动上滚后暂停自动滚动；"
        "显示每个 frame 的 token 数并合计总数；"
        "用 useRef 保存是否被用户手动滚动，用 useEffect 依赖 frames.length 触发滚动；"
        "全部类型显式标注，不使用 any。",
        800,
    ),
    (
        "T11_doc_rewrite",
        "中熵·技术文档重写",
        "把下面这段部署说明重写成面向新同事的上手文档，用中文 Markdown 输出，"
        "要求包含：背景一句话说明、架构图（用文字描述）、"
        "三种常见故障及排查步骤、关键配置项表格（至少 5 项，写清默认值与含义）、"
        "以及一段可以直接复制执行的验证命令。不要编造原文没有的信息，"
        "原文没写的就标注「待补充」：\n\n"
        "服务依赖 48 层 KV cache，block_size 4096 tokens，SSD cache 上限 200GB。"
        "PLE 走 mmap，resident 会超内存上限所以强制落盘。"
        "wired_limit_mb 设为 88000。burst_decode 用 aggressive。"
        "max_concurrent_requests 限制为 1。加载失败时先看 Metal cap 日志。",
        800,
    ),
]

QUALITY_TASKS = [
    ("Q1_code", TASKS[1][2], 400),
    ("Q2_json", TASKS[0][2], 500),
    ("Q3_creative", TASKS[7][2], 500),
]

DEPTHS = [1, 2, 3, 4, 6]

# 完整实验用的 depth 列表：加 8 是为了看到凸增长在哪里真正失控，
# 而不是只测到"变慢"就下结论。
DEPTHS_FULL = [1, 2, 3, 4, 6, 8]

# 温度轴取到 1.6：社区经验说 T=1.0 时 α 掉到 40-55%，本机实测在 T≤1.2 却几乎不动。
# 探到 1.6 是为了确认"不是幅度不够，而是根本不随温度变"。
TEMPS_FULL = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6]
