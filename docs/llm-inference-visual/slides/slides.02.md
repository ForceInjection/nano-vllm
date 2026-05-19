---
layout: cover
background: /background.svg
---

<h1 class="text-4xl font-bold!">第 2 课</h1>
<h2 class="text-2xl mt-4 font-normal opacity-80">Sequence 数据结构与请求生命周期</h2>

<div class="mt-12 text-sm opacity-60">
nano-vllm 实战课程 · 从源码走读 LLM 推理引擎
</div>

---
layout: default
---

# 本课在课程中的位置

```mermaid {scale: 0.7}
flowchart LR
    L01["L01<br/>generate → step"] --> L02["<strong>L02</strong><br/>Sequence 数据结构"]
    L02 --> L03["L03<br/>调度器与抢占"]
    L03 --> L04["L04<br/>Block 管理与前缀缓存"]
    L04 --> L05["L05<br/>Prefill Batching"]
    L05 --> L06["L06<br/>Decode 与 Block Table"]
    L06 --> L07["L07<br/>Attention 与 KV Cache"]
    L07 --> L08["L08<br/>优化全景图"]
    style L02 fill:#3b82f6,color:#fff,stroke:#2563eb,stroke-width:3px
```

<div v-click class="mt-4 text-sm opacity-80">
  L01 我们追踪了 <code>generate → step</code> 的大循环。L02 打开循环中反复操作的核心对象——<strong>Sequence</strong>。
</div>

---
layout: default
---

# 1.1 课时安排

Sequence 是每个请求的"身份证"，承载推理请求的全部状态。

| 阶段 | 时长 | 内容要点 |
|------|------|----------|
| 概念回顾 | 10 min | 从 L01 step 循环引出 Sequence 作为操作对象 |
| 代码走读 | 40 min | Sequence 字段分组、计数器不变量、block_table、序列化 |
| 脚本演示 | 10 min | L02_sequence.py 的 4 个验证 section |
| 动手练习 | 15 min | 构造 Sequence 验证 num_blocks / last_block_num_tokens |
| 答疑讨论 | 15 min | 为什么 decode 只需发 last_token、block_size 设计讨论 |

---
layout: default
---

# 1.2 学习目标

<div class="mt-6 space-y-4">

<div v-click="1" class="flex items-start gap-3 p-3 bg-blue-500/10 border-l-3 border-blue-500 rounded-r">
  <span class="text-blue-400 font-bold">Q1</span>
  <span><code>Sequence</code> 保存哪三类信息？（token_ids、调度计数器、KV cache 映射）</span>
</div>

<div v-click="2" class="flex items-start gap-3 p-3 bg-blue-500/10 border-l-3 border-blue-500 rounded-r">
  <span class="text-blue-400 font-bold">Q2</span>
  <span><code>num_cached_tokens</code>、<code>num_scheduled_tokens</code>、<code>num_tokens</code> 分别表示什么？何时更新？</span>
</div>

<div v-click="3" class="flex items-start gap-3 p-3 bg-blue-500/10 border-l-3 border-blue-500 rounded-r">
  <span class="text-blue-400 font-bold">Q3</span>
  <span><code>__getstate__</code> / <code>__setstate__</code> 的作用是什么？Tensor Parallel 场景下为什么需要不同的序列化策略？</span>
</div>

</div>

---
layout: section
---

# 2. 原理说明
## Sequence 是什么，为什么需要它

---
layout: default
---

# 2.1 Sequence = 请求的「档案袋」

一个请求从进入引擎到返回结果，需要携带大量状态信息。Sequence 把这些信息打包在一起：

<div class="grid grid-cols-3 gap-4 mt-6 text-sm">
<div class="bg-blue-500/10 p-4 rounded">
  <div class="text-lg font-bold text-blue-400 mb-2">📝 Token 数据</div>
  <ul class="space-y-1">
    <li><code>token_ids</code> — prompt token 列表</li>
    <li><code>completion_token_ids</code> — 已生成的 token</li>
    <li><code>num_prompt_tokens</code> — prompt 长度（不变）</li>
  </ul>
</div>
<div class="bg-green-500/10 p-4 rounded">
  <div class="text-lg font-bold text-green-400 mb-2">⏱ 调度计数器</div>
  <ul class="space-y-1">
    <li><code>num_cached_tokens</code> — 已完成的 token</li>
    <li><code>num_scheduled_tokens</code> — 本轮要算的</li>
    <li><code>num_tokens</code> — 总 token 数（动态增长）</li>
  </ul>
</div>
<div class="bg-purple-500/10 p-4 rounded">
  <div class="text-lg font-bold text-purple-400 mb-2">🗺 KV Cache 映射</div>
  <ul class="space-y-1">
    <li><code>block_table</code> — 逻辑→物理 block 映射</li>
    <li><code>block_size</code> — 每个 block 的 token 容量</li>
    <li><code>status</code> — WAITING/RUNNING/FINISHED</li>
  </ul>
</div>
</div>

---
layout: default
---

# 2.2 状态机：WAITING → RUNNING → FINISHED

```mermaid {scale: 0.7}
stateDiagram-v2
    [*] --> WAITING: add_request
    WAITING --> RUNNING: schedule 首次选中 + allocate
    RUNNING --> RUNNING: decode 每步 may_append
    RUNNING --> WAITING: KV 不足 preempt + deallocate
    RUNNING --> FINISHED: EOS / max_tokens + deallocate
    FINISHED --> [*]
```

<div v-click class="mt-3 text-sm opacity-80">
  类比操作系统的进程三态模型：WAITING = ready、RUNNING = running、FINISHED = terminated。preempt 类似被换出（swap out）。
</div>

---
layout: default
---

# 2.2 状态转换由谁触发？

每个状态转换都对应代码中的具体调用点：

| 转换 | 触发者 | 代码位置 |
|------|--------|----------|
| `→ WAITING` | `Scheduler.add()` | `scheduler.py` |
| `WAITING → RUNNING` | `schedule()` prefill 分支分配 block | `scheduler.py:L29-L56` |
| `RUNNING → RUNNING` | decode 每步 `may_append` | `scheduler.py:L57-L73` |
| `RUNNING → WAITING` | `Scheduler.preempt()` | `scheduler.py:L75-L79` |
| `RUNNING → FINISHED` | `postprocess()` 判定 EOS/max_tokens | `scheduler.py:L81-L92` |

<div v-click class="mt-4 p-3 bg-gray-800/50 rounded text-sm">
  💡 <strong>关键</strong>：Sequence 本身不驱动状态转换——它只是状态容器。所有转换由 <code>Scheduler</code> 和 <code>BlockManager</code> 协同完成。
</div>

---
layout: default
---

# 2.3 block_table ≈ 虚拟内存页表

PagedAttention 的核心思想：将 KV cache 分页管理，类比操作系统虚拟内存。

```mermaid {scale: 0.7}
flowchart TD
    subgraph LOGIC["逻辑层 (Sequence)"]
        T["token 序列: t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9"]
    end
    subgraph MAP["block_table 页表"]
        BT["[3, 7, 2]"]
    end
    subgraph PHYS["物理层 (显存 KV cache 池)"]
        B0["Block 0"]
        B1["Block 1"]
        B2["Block 2: t8 t9"]
        B3["Block 3: t0 t1 t2 t3"]
        B7["Block 7: t4 t5 t6 t7"]
    end
    LOGIC --> MAP --> PHYS
```

<div v-click class="mt-3 text-sm opacity-80">
  好处：小粒度分配（按 block 而非整条序列）、碎片少、可动态追加、共享前缀时只需引用同一批 block
</div>

---
layout: section
---

# 3. 代码走读
## Sequence 字段与方法逐组展开

---
layout: default
---

# 3.1 Sequence 字段全景

<SourceCode file="nanovllm/engine/sequence.py" lines="14-32" />

```python
class Sequence:
    block_size: int = 256

    def __init__(self, token_ids, sampling_params, block_size=256, eos=-1):
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)      # 固定不变
        self.completion_token_ids: list[int] = []

        self.status = SequenceStatus.WAITING
        self.is_prefill = True
        self.num_cached_tokens = 0                   # 已处理完成的 token
        self.num_scheduled_tokens = 0                # 本轮要处理的 token

        self.block_table: list[int] = []             # 逻辑→物理映射
        self.block_size = block_size
        self.sampling_params = sampling_params
        ...
```

<div v-click class="mt-2 text-xs opacity-60">
  注意：<code>block_size</code> 是<strong>类变量</strong>（默认 256），所有 Sequence 共享同一个值。
</div>

---
layout: default
---

# 3.1 逐字段走读：初始化参数

<SourceCode file="nanovllm/engine/sequence.py" lines="14-32" />

```python
def __init__(self, token_ids, sampling_params, block_size=256, eos=-1):
    # ── Token 数据 ──
    self.token_ids = list(token_ids)              # ①
    self.num_prompt_tokens = len(token_ids)       # ②
    self.completion_token_ids: list[int] = []     # ③

    # ── 调度计数器 ──
    self.status = SequenceStatus.WAITING          # ④
    self.is_prefill = True                        # ⑤
    self.num_cached_tokens = 0                    # ⑥
    self.num_scheduled_tokens = 0                 # ⑦

    # ── KV Cache 映射 ──
    self.block_table: list[int] = []              # ⑧
    self.block_size = block_size                  # ⑨
    self.sampling_params = sampling_params        # ⑩
```

<div v-click="1">
<div class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mt-4">
<div><span class="text-blue-400 font-bold">①</span> <code>token_ids</code> — prompt token 列表，推理中不断追加新 token</div>
<div><span class="text-blue-400 font-bold">②</span> <code>num_prompt_tokens</code> — prompt 长度，初始化后永久不变</div>
<div><span class="text-blue-400 font-bold">③</span> <code>completion_token_ids</code> — 生成 token 记录，仅用于统计和 max_tokens 判断</div>
<div><span class="text-blue-400 font-bold">④</span> <code>status</code> — WAITING / RUNNING / FINISHED 三态</div>
<div><span class="text-blue-400 font-bold">⑤</span> <code>is_prefill</code> — 是否为 prefill 阶段，影响序列化和调度策略</div>
<div><span class="text-blue-400 font-bold">⑥</span> <code>num_cached_tokens</code> — 已写入 KV cache 的 token 数量</div>
<div><span class="text-blue-400 font-bold">⑦</span> <code>num_scheduled_tokens</code> — 本轮计划处理的 token 数量</div>
<div><span class="text-blue-400 font-bold">⑧</span> <code>block_table</code> — 逻辑 block 到物理 block 的映射数组</div>
<div><span class="text-blue-400 font-bold">⑨</span> <code>block_size</code> — 每个 KV cache block 的大小（类变量，所有实例共享）</div>
<div><span class="text-blue-400 font-bold">⑩</span> <code>sampling_params</code> — 采样参数（temperature、top_p、max_tokens 等）</div>
</div>
</div>

<div v-click="2" class="mt-3 p-3 bg-yellow-500/10 border-l-3 border-yellow-500 rounded-r text-sm">
  <strong>注意</strong>：以上 10 个字段恰好对应 2.1 节的三大分类。<code>is_prefill</code> 最容易被忽略，但它是决定序列化和调度策略的枢纽。接下来逐一展开。
</div>

---
layout: default
---

# 3.2 Token 字段：prompt 与 completion 分离

<SourceCode file="nanovllm/engine/sequence.py" lines="14-23" />

```python
# 初始化时
self.token_ids = list(token_ids)              # prompt 的 token
self.num_prompt_tokens = len(token_ids)       # 固定不变
self.completion_token_ids: list[int] = []     # 生成为空

# append_token 追加生成 token
def append_token(self, token_id: int):
    self.token_ids.append(token_id)
    self.completion_token_ids.append(token_id)
```

<div class="grid grid-cols-2 gap-4 mt-4 text-sm">
<div v-click="1" class="bg-gray-800/50 p-3 rounded">
  <strong>num_prompt_tokens</strong><br/>
  初始化后<strong>永远不变</strong><br/>
  用于区分 prompt 和 completion
</div>
<div v-click="2" class="bg-gray-800/50 p-3 rounded">
  <strong>num_completion_tokens</strong><br/>
  <code>len(completion_token_ids)</code><br/>
  每 append_token +1，决定 max_tokens 判断
</div>
</div>

---
layout: default
---

# 3.3 调度计数器的三个关键属性

<SourceCode file="nanovllm/engine/sequence.py" lines="25-27" />

```python
self.num_cached_tokens = 0       # 已处理完成、写入 KV cache 的 token 数
self.num_scheduled_tokens = 0    # 本轮 step 计划处理的 token 数
# num_tokens 是 property: len(token_ids)
```

<div class="mt-4 grid grid-cols-3 gap-3 text-sm">
<div v-click="1" class="bg-blue-500/10 p-3 rounded text-center">
  <div class="font-bold mb-1">num_cached_tokens</div>
  <div class="opacity-70">已完成 + 已写入 KV cache</div>
  <div class="text-xs mt-1 opacity-50">postprocess 中累加<br/>scheduled → cached</div>
</div>
<div v-click="2" class="bg-green-500/10 p-3 rounded text-center">
  <div class="font-bold mb-1">num_scheduled_tokens</div>
  <div class="opacity-70">本轮计划处理</div>
  <div class="text-xs mt-1 opacity-50">schedule() 中设定<br/>postprocess 中清零</div>
</div>
<div v-click="3" class="bg-purple-500/10 p-3 rounded text-center">
  <div class="font-bold mb-1">num_tokens</div>
  <div class="opacity-70">= len(token_ids)</div>
  <div class="text-xs mt-1 opacity-50">动态增长<br/>= prompt + completion</div>
</div>
</div>

<div v-click="4" class="mt-3 p-3 bg-yellow-500/10 border-l-3 border-yellow-500 rounded-r text-sm">
  <strong>不变量</strong>：<code>num_tokens = num_prompt_tokens + num_completion_tokens</code>。在 prefill 过程中：<code>num_cached_tokens ≤ num_tokens</code>，当两者相等时 prefill 完成。
</div>

---
layout: default
---

# 3.3 计数器在 postprocess 中的更新时机

<SourceCode file="nanovllm/engine/scheduler.py" lines="81-87" />

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens  # ① 累计
        seq.num_scheduled_tokens = 0                       # ② 清零
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                                       # ③ chunked-prefill 未竟
        seq.append_token(token_id)                         # ④ 追加采样 token
```

<div v-click class="mt-3 text-sm">
  <strong>生命周期</strong>：<code>schedule()</code> 设定 → <code>run()</code> 不变 → <code>postprocess()</code> 累加到 cached 然后清零。下一轮 <code>schedule()</code> 从 <code>num_cached_tokens</code> 开始取下一段。
</div>

---
layout: default
---

# 3.4 block_table 与 block 分割公式

<SourceCode file="nanovllm/engine/sequence.py" lines="55-62" />

```python
@property
def num_blocks(self) -> int:
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self) -> int:
    return self.num_tokens - (self.num_blocks - 1) * self.block_size

def block(self, i: int) -> list[int]:
    start = i * self.block_size
    end = min((i + 1) * self.block_size, self.num_tokens)
    return self.token_ids[start:end]
```

<div class="grid grid-cols-3 gap-3 mt-4 text-sm">
<div v-click="1" class="bg-blue-500/10 p-3 rounded text-center">
  <strong>num_blocks</strong><br/>
  = ⌈num_tokens / block_size⌉<br/>
  <span class="text-xs opacity-60">例: 9 tokens / 4 = 3 blocks</span>
</div>
<div v-click="2" class="bg-green-500/10 p-3 rounded text-center">
  <strong>last_block_num_tokens</strong><br/>
  = num_tokens % block_size<br/>
  <span class="text-xs opacity-60">最后一个 block 不满的情况</span>
</div>
<div v-click="3" class="bg-purple-500/10 p-3 rounded text-center">
  <strong>block(i)</strong><br/>
  取第 i 个 block 的 token_ids<br/>
  <span class="text-xs opacity-60">用于 prefix cache 哈希计算</span>
</div>
</div>

---
layout: default
---

# 3.4 示例：用具体数值走一遍公式

<SourceCode file="nanovllm/engine/sequence.py" lines="55-62" />

<div class="text-sm">

**设定**：block_size = 256，num_tokens = 1000

```python
num_blocks = (1000 + 256 - 1) // 256     # = 1255 // 256 = 4
last_block_num_tokens = 1000 - (4 - 1) * 256  # = 1000 - 768 = 232
```

</div>

<div v-click="1" class="mt-4">
<h4 class="text-sm font-bold mb-2">物理视图：各 block 包含的 token 范围</h4>

| block 索引 | token 区间 | slot 数量 |
|:---------:|:-----------:|:--------:|
| block(0)  | token_ids[0:256]   | 256（满） |
| block(1)  | token_ids[256:512] | 256（满） |
| block(2)  | token_ids[512:768] | 256（满） |
| block(3)  | token_ids[768:1000]| **232**（不满） |
</div>

<div v-click="2" class="mt-4 p-3 bg-green-500/10 border-l-3 border-green-500 rounded-r text-sm">
  <strong>观察</strong>：前 3 个 block 各装满 256 个 token，最后一块只装 232 个。<code>last_block_num_tokens = num_tokens % block_size</code>（余数为 0 时等于 block_size）。
</div>

<div v-click="3" class="mt-3">
<h4 class="text-sm font-bold mb-1">快速验证（block_size = 4）</h4>

| num_tokens | num_blocks | last_block_num_tokens | block(0) | block(1) | block(2) |
|:----------:|:----------:|:---------------------:|:--------:|:--------:|:--------:|
| 1 | 1 | 1 | [0:1] | — | — |
| 4 | 1 | 4 | [0:4] | — | — |
| 5 | 2 | 1 | [0:4] | [4:5] | — |
| 8 | 2 | 4 | [0:4] | [4:8] | — |
| 9 | 3 | 1 | [0:4] | [4:8] | [8:9] |
</div>

<div v-click="4" class="mt-3 text-xs opacity-70">
  此公式在 block_table 分配、prefix cache 哈希、以及 attention 计算 mask 时反复使用。
</div>

---
layout: default
---

# 3.4 block_size 从哪里来

<SourceCode file="nanovllm/config.py" lines="6-18" />

```python
@dataclass(slots=True)
class Config:
    model: str
    kvcache_block_size: int = 256          # block 大小，必须是 256 的倍数
    ...
    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
```

<div class="mt-4 text-sm">
  <strong>传递链</strong>：<code>Config.kvcache_block_size</code> → <code>LLMEngine.__init__</code> → <code>Sequence.block_size</code>（类变量，所有 Sequence 共享）
</div>

<div class="mt-3 p-3 bg-gray-800/50 rounded text-sm">
  💡 <strong>为什么必须是 256 的倍数？</strong> FlashAttention 的 Triton kernel 以 256 为 tile 大小读写 KV cache。block_size 对齐到这个 tile 可以避免跨 tile 的额外处理。
</div>

---
layout: default
---

# 3.4 last_token property

<SourceCode file="nanovllm/engine/sequence.py" lines="64-66" />

```python
@property
def last_token(self) -> int:
    """返回 token_ids 的最后一个元素，用于 decode 阶段的前向传播。"""
    return self.token_ids[-1]
```

<div v-click="1" class="mt-4">
<h4 class="text-sm font-bold mb-2">为什么需要 last_token？</h4>

<div class="grid grid-cols-2 gap-4 text-sm">
<div class="bg-blue-500/10 p-3 rounded">
  <div class="font-bold text-blue-400">Prefill 阶段</div>
  <ul class="mt-2 space-y-1 text-xs">
    <li>需要完整 token_ids 计算 KV cache</li>
    <li>模型输入：<code>token_ids[0:N]</code></li>
    <li>数据量：N 个整数</li>
  </ul>
</div>
<div class="bg-green-500/10 p-3 rounded">
  <div class="font-bold text-green-400">Decode 阶段</div>
  <ul class="mt-2 space-y-1 text-xs">
    <li>只需最后 1 个 token 做前向传播</li>
    <li>模型输入：<code>token_ids[-1:]</code></li>
    <li>数据量：1 个整数</li>
  </ul>
</div>
</div>
</div>

<div v-click="2" class="mt-3 p-3 bg-purple-500/10 border-l-3 border-purple-500 rounded-r text-sm">
  <strong>关键</strong>：decode 阶段每次只输入最后一个 token，KV cache 中已保存之前的全部 KV 值。<code>last_token</code> 正是为这个目的设计的快捷属性。它也是 TP 场景下 decode 序列化时传输的核心数据。
</div>

---
layout: default
---

# 3.5 序列化：为 Tensor Parallel 服务

<SourceCode file="nanovllm/engine/sequence.py" lines="72-83" />

```python
def __getstate__(self):
    if self.is_prefill:
        return (self.token_ids, self.num_prompt_tokens,
                self.block_table, ...)       # prefill: 传完整 token_ids
    else:
        return (self.last_token, ...)        # decode: 只传最后一个 token

def __setstate__(self, state):
    token_ids_or_last_token = state[0]
    if isinstance(token_ids_or_last_token, list):
        self.token_ids = token_ids_or_last_token  # prefill 恢复
    else:
        self.last_token = token_ids_or_last_token # decode 恢复
```

<div v-click class="mt-3 text-sm opacity-80">
  <strong>为什么区分？</strong>prefill 阶段子进程需要全部 prompt token 计算 KV；decode 阶段只需最后一个 token 做前向。减少 decode 的 IPC 带宽是关键优化。
</div>

---
layout: default
---

# 3.5 TP 场景下的序列化流程图

```mermaid {scale: 0.7}
flowchart TD
    R0["Rank 0: write_shm('run', seqs)"] --> P["pickle.dumps(seqs)"]
    P --> C{"seq.is_prefill?"}
    C -- Yes --> F["序列化 token_ids<br/>(完整 prompt)"]
    C -- No --> D["序列化 last_token<br/>(仅一个整数)"]
    F --> SHM["写入 SharedMemory"]
    D --> SHM
    SHM --> EV["set Events<br/>唤醒子进程"]
    EV --> R1["Rank > 0: pickle.loads"]
    R1 --> EX["执行 run()"]
```

<div v-click class="mt-3 text-sm opacity-80">
  这正是 <code>Sequence.__getstate__</code> 只传输必要字段的原因：prefill 传输完整 <code>token_ids</code> 列表，decode 只传 <code>last_token</code>（一个 int）。
</div>

---
layout: default
---

# 3.5 对比：prefill vs decode 的 IPC 数据量

<div class="text-sm">

TP 场景下，Rank 0 通过 SharedMemory 向其他 Rank 传输序列化后的 Sequence 数据。prefill 和 decode 的 IPC 负载差异巨大：

</div>

<div v-click="1" class="mt-4">
<h4 class="text-sm font-bold mb-2">序列化数据量对比（假设 prompt=1024 tokens, completion=128 tokens）</h4>

| 维度 | Prefill | Decode |
|:----|:--------|:-------|
| 序列化字段 | 完整 <code>token_ids</code> + 完整字段 | <code>last_token</code>（1 个 int）+ 必要字段 |
| 整数数量 | ~1500 个 | ~10 个 |
| 预估字节数（pickle 后） | ~12 KB | ~0.5 KB |
| 传输内容 | 全部 prompt token 用于 KV 计算 | 仅需最后 1 个 token 做单步前向 |
| 调用频率 | 每个请求仅 1 次（或 chunk-prefill 若干次） | 每个生成步调用 1 次 |
</div>

<div v-click="2" class="mt-4 grid grid-cols-2 gap-4 text-sm">
<div class="bg-blue-500/10 p-3 rounded">
  <div class="font-bold text-blue-400 mb-1">Prefill 数据量大，但次数少</div>
  <div class="text-xs">一次传完整个 prompt，后续 decode 无需重复传输。如果被抢占（preempt），重新 prefill 时会再次传输完整 token_ids。</div>
</div>
<div class="bg-green-500/10 p-3 rounded">
  <div class="font-bold text-green-400 mb-1">Decode 数据量小，但次数多</div>
  <div class="text-xs">每次 decode 只传 1 个 int，IPC 开销极低。这是 <code>__getstate__</code> 区分 prefill/decode 的核心优化动机。</div>
</div>
</div>

<div v-click="3" class="mt-3 p-3 bg-yellow-500/10 border-l-3 border-yellow-500 rounded-r text-sm">
  <strong>结论</strong>：prefill 是 I/O 密集型（大量数据传输），decode 是 compute 密集型（少量数据传输）。<code>is_prefill</code> 标志在序列化阶段实现了按需传输的优化策略。
</div>

---
layout: default
---

# 3.6 状态与 is_prefill 标志的联动

<h4 class="text-sm font-bold mb-3">两个关键标志的关系</h4>

<div class="grid grid-cols-2 gap-4 text-sm">
<div class="bg-blue-500/10 p-3 rounded">
  <div class="font-bold text-blue-400 mb-1"><code>status</code></div>
  <div>WAITING / RUNNING / FINISHED 三态，由 Scheduler 在 schedule() 和 postprocess() 中修改。</div>
</div>
<div class="bg-green-500/10 p-3 rounded">
  <div class="font-bold text-green-400 mb-1"><code>is_prefill</code></div>
  <div>True / False，表示当前是否处于 prefill 阶段，决定序列化和调度行为。</div>
</div>
</div>

<div v-click="1" class="mt-4">
<h4 class="text-sm font-bold mb-2">状态转换中的 is_prefill 变化</h4>

| 事件 | status | is_prefill | 说明 |
|:----|:------:|:----------:|:-----|
| <code>add_request()</code> 后 | WAITING | True | 新请求，尚未分配 block |
| schedule() 首次选中 | RUNNING | True | 进入 prefill 阶段 |
| prefill 完成（num_cached == num_tokens） | RUNNING | **False** | 进入 decode 阶段 |
| decode 中 append_token | RUNNING | False | 持续生成 |
| **preempt 回到 waiting** | WAITING | **True** | 被抢占后重新进入 waiting，下一轮重新 prefill |
| EOS / max_tokens | FINISHED | — | 推理结束 |
</div>

<div v-click="2" class="mt-4 p-3 bg-gray-800/50 rounded text-sm">
  <strong>联动逻辑</strong>：<code>is_prefill</code> 跟随 <code>num_cached_tokens &lt; num_tokens</code> 条件自动变化。当 <code>is_prefill = True</code> 时，<code>__getstate__</code> 序列化完整 token_ids；当 <code>is_prefill = False</code> 时，只序列化 last_token。
</div>

<div v-click="3" class="mt-3 p-3 bg-purple-500/10 border-l-3 border-purple-500 rounded-r text-sm">
  <strong>关键设计</strong>：<code>is_prefill</code> 不是由 status 推导的。preempt 时 status 变为 WAITING，同时必须显式设置 <code>is_prefill = True</code>。这是因为 WAITING 状态本身不意味着需要 prefill（新请求和抢占后的请求都需要重新 prefill，但调度逻辑不同）。
</div>

---
layout: section
---

# 4. L02 验证脚本
## L02_sequence.py 走读

---
layout: default
---

# L02_sequence.py：4 个验证 section

<SourceCode file="docs/llm-inference-visual/scripts/L02_sequence.py" lines="1-13" />

```python
"""
L02 练习：Sequence 数据结构与请求生命周期

验证要点：
- Sequence 三大类字段（token、调度计数器、KV cache 映射）
- block 分割公式：num_blocks = ⌈num_tokens / block_size⌉
- append_token 更新 completion_token_ids
- pickle 序列化：prefill 发 token_ids，decode 发 last_token

依赖：仅 CPU（nano-vllm 包），不需要模型权重
"""
```

<div class="mt-4 grid grid-cols-4 gap-2 text-xs text-center">
<div class="bg-blue-500/10 p-2 rounded">§1<br/><strong>字段分类</strong></div>
<div class="bg-green-500/10 p-2 rounded">§2<br/><strong>Block 公式验证</strong></div>
<div class="bg-purple-500/10 p-2 rounded">§3<br/><strong>append_token</strong></div>
<div class="bg-yellow-500/10 p-2 rounded">§4<br/><strong>Pickle 协议</strong></div>
</div>

---
layout: default
---

# §1：字段分类验证

```python
from nanovllm.engine.sequence import Sequence

seq = Sequence([1, 2, 3], sampling_params=None)

# token 数据
print("token_ids:", seq.token_ids)                # [1, 2, 3]
print("num_prompt_tokens:", seq.num_prompt_tokens) # 3
print("completion_token_ids:", seq.completion_token_ids)  # []

# 调度计数器
print("status:", seq.status)                      # WAITING
print("is_prefill:", seq.is_prefill)              # True
print("num_cached_tokens:", seq.num_cached_tokens) # 0
print("num_scheduled_tokens:", seq.num_scheduled_tokens) # 0

# KV Cache 映射
print("block_table:", seq.block_table)            # []
print("block_size:", seq.block_size)              # 256
```

<div v-click class="mt-3 p-3 bg-gray-800/50 rounded text-sm">
  脚本执行后会打印以上所有字段，验证 2.1 节的三大分类。<code>token_ids</code> 保存完整 token 序列，其他字段均从它衍生。
</div>

---
layout: default
---

# §2：Block 分割公式验证

```python
Sequence.block_size = 4  # 设为 4 方便手算

for n in [1, 4, 5, 8, 9]:
    seq = Sequence(list(range(n)), sampling_params=None)
    print(f"n={n}: num_blocks={seq.num_blocks}, "
          f"last_block_num_tokens={seq.last_block_num_tokens}")
    for i in range(seq.num_blocks):
        print(f"  block[{i}] = {seq.block(i)}")
```

<div v-click="1" class="mt-3">
<h4 class="text-sm font-bold mb-1">输出结果</h4>

| n | num_blocks | last_block_num_tokens | block(0) | block(1) | block(2) |
|:-:|:----------:|:---------------------:|:--------:|:--------:|:--------:|
| 1 | 1 | 1 | [0] | — | — |
| 4 | 1 | 4 | [0,1,2,3] | — | — |
| 5 | 2 | 1 | [0,1,2,3] | [4] | — |
| 8 | 2 | 4 | [0,1,2,3] | [4,5,6,7] | — |
| 9 | 3 | 1 | [0,1,2,3] | [4,5,6,7] | [8] |
</div>

<div v-click="2" class="mt-3 p-3 bg-green-500/10 border-l-3 border-green-500 rounded-r text-sm">
  <strong>公式</strong>：<code>num_blocks = (num_tokens + block_size - 1) // block_size</code>。<code>last_block_num_tokens</code> 在 num_tokens 为 block_size 的整数倍时等于 block_size（满块），否则为余数。
</div>

---
layout: default
---

# §3-4：append_token + Pickle 序列化

```python
# §3: append_token 追踪
seq = Sequence([1, 2, 3])          # prompt: [1,2,3]
seq.append_token(4)                 # 追加 completion
seq.append_token(5)
assert seq.num_completion_tokens == 2
assert seq.completion_token_ids == [4, 5]
assert seq.num_prompt_tokens == 3   # 不变！

# §4: Pickle 协议的两种模式
# Prefill pickling → 传输完整 token_ids
seq.is_prefill = True
seq2 = pickle.loads(pickle.dumps(seq))
assert seq2.token_ids == [1, 2, 3, 4, 5]  # 全部恢复

# Decode pickling → 只传输 last_token
seq.is_prefill = False
seq2 = pickle.loads(pickle.dumps(seq))
assert seq2.last_token == 5               # 只有一个 int
assert seq2.token_ids == []               # 空！
```

---
layout: default
---

# §3-4 总结：数据流一览

<div class="text-sm">

两个核心操作的效果对比表：

</div>

<div v-click="1" class="mt-3">
<h4 class="text-sm font-bold mb-2"><code>append_token</code> 操作前后</h4>

| 字段 | 操作前 | 操作后（append 5） | 说明 |
|:----|:-----:|:----------------:|:-----|
| <code>token_ids</code> | [1,2,3] | [1,2,3,5] | 追加新 token |
| <code>completion_token_ids</code> | [] | [5] | 记录生成 token |
| <code>num_prompt_tokens</code> | 3 | 3 | **不变** |
| <code>num_tokens</code> | 3 | 4 | = len(token_ids) |
</div>

<div v-click="2" class="mt-4">
<h4 class="text-sm font-bold mb-2">Pickle 序列化两种模式</h4>

| 模式 | <code>is_prefill</code> | 传输内容 | 反序列化后 <code>token_ids</code> |
|:----|:---------------------:|:-------:|:------------------------------:|
| Prefill 序列化 | True | 完整 <code>token_ids</code> 列表 | [1,2,3,5]（完整恢复） |
| Decode 序列化 | False | <code>last_token</code>（1 个 int） | []（为空！） |
</div>

<div v-click="3" class="mt-4 p-3 bg-purple-500/10 border-l-3 border-purple-500 rounded-r text-sm">
  <strong>重点</strong>：decode 序列化只传 last_token 是重要的 IPC 优化。如果子进程需要完整 token_ids（如被抢占），Rank 0 会重设 <code>is_prefill=True</code>，下一轮 prefill 自然发送完整数据。
</div>

---
layout: default
---

# 4.1 课堂练习

纯 CPU 练习：手工计算 block_size 与 num_blocks 的关系

```python
# 在 Python 交互环境中验证
from nanovllm.engine.sequence import Sequence

Sequence.block_size = 4  # 设为 4 方便手算

for n in [1, 4, 5, 8, 9]:
    seq = Sequence(list(range(n)), sampling_params=None)
    print(f"n={n}: num_blocks={seq.num_blocks}, "
          f"last_block_num_tokens={seq.last_block_num_tokens}")
    for i in range(seq.num_blocks):
        print(f"  block[{i}] = {seq.block(i)}")
```

<div v-click class="mt-3 text-sm">
  📍 验收要点：<code>num_blocks = (num_tokens + block_size - 1) // block_size</code>；<code>last_block_num_tokens = num_tokens - (num_blocks - 1) * block_size</code>（<code>sequence.py:L55-L62</code>）
</div>

---
layout: default
---

# 4.2 课后自测题

<SelfTest
  id="l02-q1"
  type="text"
  question="1. block_size 作为类变量，所有 Sequence 共享同一个值。如果多组请求需要不同的 block_size，当前设计有何问题？如何改进？"
  answer="<strong>问题</strong>：<code>block_size</code> 是类变量，修改会影响所有 Sequence 实例。如果在同一个引擎中混用不同 block_size 的 seq，block_table 的索引计算会出错。<br><strong>改进</strong>：将 block_size 改为实例变量（<code>__init__</code> 参数），每个 Sequence 持有自己的 block_size。但 BlockManager 也需要支持多 block_size 的分配，这会大幅增加 KV cache 管理复杂度——这也是为什么实际部署中通常统一 block_size。"
/>

<SelfTest
  id="l02-q2"
  type="text"
  question="2. decode 阶段 __getstate__ 只传输 last_token，丢失了完整 token_ids。在什么场景下子进程需要完整的 prompt？"
  answer="<strong>需要完整 prompt 的场景</strong>：<ol><li>子进程被抢占（preempt），需要重新 prefill——此时 rank0 会重设 <code>is_prefill=True</code>，下一轮序列化会发送完整 token_ids</li><li>需要做 prefix caching 哈希校验时，子进程需要完整的 token 序列计算哈希</li><li>如果 decode 过程中需要 logprobs 或 token 级别的调试信息</li></ol>实际上，<code>is_prefill</code> 标志决定了序列化策略：被抢占的 seq 返回 waiting 后会设置 <code>is_prefill=True</code>，下一轮 prefill 自然会发送完整 token_ids。"
/>

---
layout: default
---

# 4.2 课后自测题（续）

<SelfTest
  id="l02-q3"
  type="text"
  question="3. num_cached_tokens 和 num_scheduled_tokens 都在 postprocess 中更新。如果改为在 Sequence.append_token 中更新，需要额外传入什么信息？哪种设计更好？"
  answer="<strong>分析</strong>：如果移到 <code>append_token</code> 中更新，需要额外传入 <code>num_scheduled_tokens</code> 和 <code>is_prefill</code> 参数（判断是否 chunked prefill 未完）。<br><strong>当前设计更好</strong>：计数器更新与调度逻辑紧密相关，放在 <code>postprocess</code> 中可以集中管理。Sequence 只需提供数据存储，调度逻辑由 Scheduler 负责——这是单一职责原则的体现。如果 Sequence 自己更新计数器，它就需要理解调度语义，耦合度会增加。"
/>

---
layout: center
---

# 🎉 第 2 课完成

<div class="mt-6 text-lg opacity-80">
  掌握了 Sequence 的三类字段、状态机、block 分割逻辑
</div>

<div class="mt-4 grid grid-cols-4 gap-3 text-sm max-w-2xl mx-auto">
  <div class="bg-blue-500/10 p-3 rounded">✅ 三大类字段</div>
  <div class="bg-green-500/10 p-3 rounded">✅ 状态机转移</div>
  <div class="bg-purple-500/10 p-3 rounded">✅ block 公式</div>
  <div class="bg-yellow-500/10 p-3 rounded">✅ TP 序列化</div>
</div>

<div class="mt-10">
  <a href="#" class="text-blue-400 hover:underline text-lg">下一课：Scheduler 的队列与抢占 →</a>
</div>
