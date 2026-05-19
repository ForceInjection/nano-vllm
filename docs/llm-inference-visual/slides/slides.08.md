---
layout: cover
background: /background.svg
---

<h1 class="text-4xl font-bold!">第 8 课</h1>
<h2 class="text-2xl mt-4 font-normal opacity-80">常见优化的「位置感」</h2>

<div class="mt-12 text-sm opacity-60">
nano-vllm 实战课程 · 从源码走读 LLM 推理引擎
</div>

---
layout: default
---

# 本课在课程中的位置

```mermaid {scale: 0.7}
flowchart LR
    L07["L07<br/>Attention 与 KV Cache"] --> L08["<strong>L08</strong><br/>优化全景图"]
    style L08 fill:#3b82f6,color:#fff,stroke:#2563eb,stroke-width:3px
```

<div v-click class="mt-4 text-sm opacity-80">
  前 7 课覆盖了推理引擎的完整功能。L08 站在更高的视角——三种常见优化（TP、CUDA Graph、torch.compile）分别住在代码的哪里、攻击什么瓶颈。
</div>

---
layout: default
---

# 1.1 课时安排

建立一张"优化地图"——三种常见优化（TP、CUDA Graph、torch.compile）分别"住在"代码的哪里。

| 阶段 | 时长 | 内容要点 |
|------|------|----------|
| 概念回顾 | 10 min | 回顾推理主链路，标注"优化可以作用的位置" |
| 代码走读 | 35 min | TP 进程模型、CUDA Graph capture/replay、torch.compile 位置 |
| 脚本演示 | 10 min | L08_optimizations.py 的 4 个 section |
| 动手练习 | 15 min | 复刻 replay 判定条件 |
| 答疑讨论 | 20 min | 这些优化分别影响延迟/吞吐/显存的哪个维度 |

---
layout: default
---

# 1.2 学习目标

<div class="mt-6 space-y-4">

<div v-click="1" class="flex items-start gap-3 p-3 bg-blue-500/10 border-l-3 border-blue-500 rounded-r">
  <span class="text-blue-400 font-bold">Q1</span>
  <span>Tensor Parallel 的进程模型在代码中的入口在哪？rank0 和子进程之间如何通信？</span>
</div>

<div v-click="2" class="flex items-start gap-3 p-3 bg-blue-500/10 border-l-3 border-blue-500 rounded-r">
  <span class="text-blue-400 font-bold">Q2</span>
  <span>CUDA Graph replay 的触发条件是什么？为什么它只覆盖 decode 的一部分场景？</span>
</div>

<div v-click="3" class="flex items-start gap-3 p-3 bg-blue-500/10 border-l-3 border-blue-500 rounded-r">
  <span class="text-blue-400 font-bold">Q3</span>
  <span><code>torch.compile</code> 出现在哪个模块上？为什么只编译采样模块，而不是整个 Transformer？</span>
</div>

</div>

---
layout: section
---

# 2. 原理说明
## 三种优化各自攻击的瓶颈

---
layout: default
---

# 2.1 三种瓶颈与对应的优化

```mermaid {scale: 0.65}
flowchart TD
    subgraph BOTTLENECKS["瓶颈"]
        CB["Compute-Bound<br/>Prefill 算力不够"]
        MB["Memory-Bound<br/>Decode 访存太慢"]
        OH["Overhead<br/>Python/kernel launch 开销"]
    end
    subgraph OPT["优化"]
        TP["Tensor Parallel<br/>→ 多 GPU 分摊计算"]
        CG["CUDA Graph<br/>→ 消除 kernel launch 延迟"]
        TC["torch.compile<br/>→ 算子融合，减少 Python 开销"]
    end
    CB --> TP
    MB --> CG
    OH --> CG & TC
```

<div v-click class="mt-3 text-sm opacity-80">
  三种优化不互斥，存在于代码的不同位置。本课不深入实现细节，而是建立"在哪里、触发条件是什么"的位置感。
</div>

---
layout: default
---

# 2.2 优化地图一览

| 优化 | 攻击瓶颈 | 代码入口 | 覆盖范围 |
|------|----------|----------|----------|
| **Tensor Parallel** | Compute (算力) | `llm_engine.py:L22-L34` `model_runner.py:L41-L89` | 整个模型执行 |
| **CUDA Graph** | Kernel launch 延迟 | `model_runner.py:L222-L257` (capture) `model_runner.py:L195-L212` (replay) | Decode 路径 |
| **torch.compile** | Python 解释器开销 | `sampler.py:L5-L12` | 采样模块 |

---
layout: section
---

# 3. 代码走读
## 三种优化的代码位置

---
layout: default
---

# 3.1 Tensor Parallel：多进程 + 共享内存广播

<SourceCode file="nanovllm/engine/llm_engine.py" lines="22-34" />

```python
# LLMEngine.__init__ 中启动 TP 子进程
if config.tensor_parallel_size > 1:
    for rank in range(1, config.tensor_parallel_size):
        p = mp.Process(target=ModelRunner, args=(config, rank, ...))
        p.start()
```

<SourceCode file="nanovllm/engine/model_runner.py" lines="76-89" />

```python
# rank 0: 写方法名+参数到共享内存，设置事件唤醒子进程
def call(self, method_name, *args):
    self.write_shm(self.shm, method_name, *args)
    for event in self.events:
        event.set()                     # 唤醒所有子进程
    result = getattr(self, method_name)(*args)  # rank 0 自己执行
    # 注意：NCCL 同步发生在 model.forward() 内部（TP linear 层自动触发 all-reduce）
    # 而非 call() 方法显式调用 barrier
    return result
```

---
layout: default
---

# TP 初始化完整流程

```mermaid {scale: 0.6}
flowchart TD
    A["LLMEngine.__init__"] --> B{"tensor_parallel_size > 1?"}
    B -- No --> C["单进程: ModelRunner(rank=0)"]
    B -- Yes --> D["for rank in 1..N-1:<br/>mp.Process(ModelRunner, args=(rank, event))"]
    D --> E["ModelRunner.__init__ (rank 0)"]
    E --> F["dist.init_process_group('nccl')<br/>torch.cuda.set_device(rank)"]
    F --> G["load_model(model)"]
    G --> H["warmup_model()"]
    H --> I["allocate_kv_cache()"]
    I --> J["capture_cudagraph()<br/>(非 eager 模式)"]
    J --> K{"rank == 0?"}
    K -- Yes --> L["创建 SharedMemory('nanovllm')<br/>dist.barrier() 等待子进程"]
    K -- No --> M["dist.barrier()<br/>attach SharedMemory('nanovllm')<br/>进入 loop() 阻塞"]
```

<div v-click class="mt-2 text-sm">
  <strong>启动时序</strong>：rank0 先完成模型加载 → 创建共享内存 → barrier 释放子进程 → 子进程 attach 内存 → 进入 loop 等待命令。<br/>
  <code>ModelRunner</code> 同一份代码兼具两种角色：rank0 执行 forward + 管理，rank>0 执行 loop 等待 Event。
</div>

---

layout: default

---

# TP call() 方法逐行解读

```python
def call(self, method_name, *args):
    # model_runner.py:L85-L89
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)  # ① 写 shm + set Events
    method = getattr(self, method_name, None)
    return method(*args)                     # ② rank 0 直接调用
                                            # ③ rank>0 通过 loop()→call() 递归
```

```python
def write_shm(self, method_name, *args):
    # model_runner.py:L76-L83
    data = pickle.dumps([method_name, *args])  # ① pickle 序列化
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little") # ② 写长度到前 4 字节
    self.shm.buf[4:n+4] = data                   # ③ 写数据
    for event in self.event:
        event.set()                              # ④ 唤醒所有子进程
```

```python
def loop(self):                                 # rank>0 的主循环
    # model_runner.py:L61-L66
    while True:
        method_name, args = self.read_shm()     # ① 阻塞等 Event
        self.call(method_name, *args)            # ② 递归调 call()
        if method_name == "exit":                # ③ exit 时退出
            break
```

<div v-click class="mt-2 text-sm">
  <strong>方法名广播策略</strong>：rank0 将方法名+参数 pickle 后写入共享内存，子进程反序列化后调用。NCCL 同步在 <code>call</code> 返回前由 linear 层的 all-reduce 触发。
</div>

---

layout: default

---

# 3.1 TP 通信模型

```mermaid {scale: 0.65}
flowchart TD
    subgraph RANK0["Rank 0 (主进程)"]
        R0C["call('run', seqs, is_prefill)"] --> R0W["write_shm<br/>写入方法名 + pickle 参数"]
        R0W --> R0E["set Events<br/>唤醒子进程"]
        R0E --> R0X["rank 0 执行 run()"]
    end
    subgraph RANKN["Rank > 0 (子进程)"]
        RN["loop() 阻塞等待 Event"]
        RN --> RNR["Event 被 set → 读 shm"]
        RNR --> RNX["rank N 执行 run()"]
    end
    R0X & RNX --> SYNC["NCCL all-reduce<br/>同步各 rank 结果"]
    SYNC --> DONE["barrier → call 返回"]
```

---
layout: default
---

# TP 的 NCCL 同步点

Tensor Parallel 涉及多处 NCCL 集合通信操作，覆盖整个模型生命周期：

```mermaid {scale: 0.65}
flowchart TD
    subgraph INIT["初始化阶段"]
        I1["dist.init_process_group('nccl')"] --> I2["dist.barrier()<br/>(等待子进程就绪)"]
    end
    subgraph RUN["每步推理"]
        R1["rank0: write_shm + set Events"] --> R2["rank>0: Event.wait → read_shm"]
        R2 --> R3["各 rank 独立执行 model.forward()"]
        R3 --> R4["NCCL all-reduce<br/>(同步各 rank 的 logits 结果)"]
        R4 --> R5["dist.barrier()<br/>(call 方法末尾)"]
    end
    subgraph EXIT["退出阶段"]
        E1["dist.barrier()"] --> E2["dist.destroy_process_group()"]
    end
```

<div v-click class="mt-3 text-sm">
  代码中所有 <code>dist.barrier()</code> 调用点（model_runner.py）：
  <ul>
    <li>L44/L46：init 阶段 barrier → 确保 shm 创建完毕</li>
    <li>L53：exit 阶段 barrier → 确保所有 rank 完成</li>
    <li>L87-89 (call 方法)：执行结束后 barrier → 返回前所有 rank 同步</li>
  </ul>
  NCCL all-reduce 在 <code>model.compute_logits()</code> 内部隐式调用（TP linear 层自动触发）。
</div>

---
layout: default
---

# 3.2 CUDA Graph：capture

<SourceCode file="nanovllm/engine/model_runner.py" lines="222-257" />

```python
def capture_cudagraph(self):
    if self.enforce_eager:
        return
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if bs > self.max_num_seqs:
            break
        # 构造 dummy 输入
        input_ids = torch.zeros(bs, dtype=torch.int32, device='cuda')
        positions = torch.zeros(bs, dtype=torch.int32, device='cuda')
        # 设置 graph_vars 的 shape
        ...
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_vars["outputs"][:bs] = self.model(input_ids, positions)
        self.graphs[bs] = g    # 存入字典，按 batch size 索引
```

<div v-click class="mt-2 text-sm">
  在 <code>ModelRunner.__init__</code> 的 warmup 后调用。graph 字典覆盖 <code>[1, 2, 4, 8, 16, 32, ..., max_bs]</code>，replay 时按 bs 选择最近的 bucket。
</div>

---
layout: default
---

# capture_cudagraph 逐段解读

<SourceCode file="nanovllm/engine/model_runner.py" lines="222-257" />

```
@torch.inference_mode()
def capture_cudagraph(self):
    if self.enforce_eager:                  # ① eager 模式跳过
        return
    max_bs = min(config.max_num_seqs, 512)  # ② 不超过 512
    # ┄ 预分配 tensor ┄
    input_ids = torch.zeros(max_bs, ...)    # ③ 固定最大形状
    slot_mapping = torch.zeros(max_bs, ...)
    block_tables = torch.zeros(max_bs, max_num_blocks, ...)
    
    self.graph_bs = [1,2,4,8] + list(range(16, max_bs+1, 16))
    for bs in reversed(self.graph_bs):      # ④ 从大到小 capture
        graph = torch.cuda.CUDAGraph()
        set_context(..., slot_mapping[:bs], ...)   # ⑤ 按 bs 设上下文
        outputs[:bs] = model(input_ids[:bs], ...)  # ⑥ warmup → 触发编译
        with torch.cuda.graph(graph, pool):         # ⑦ 录制开始
            outputs[:bs] = model(input_ids[:bs], ...)
        self.graphs[bs] = graph                    # ⑧ 存入字典
```

<div v-click class="mt-2 text-sm">
  <strong>关键点</strong>：<br/>
  ① enforce_eager 时完全跳过 capture。② 最大 bs 512（硬编码阈值）。③ 所有 tensor 按 max_bs 预分配，只 slice 前 bs 个给 graph。<br/>
  ④ 从大到小 capture 确保 graph pool 覆盖所有 bucket。⑤ 注入对应 bs 的 context。⑥ warmup 触发 Triton 编译，确保 capture 时不包含编译延迟。⑦ capture 在 <code>torch.cuda.graph</code> 上下文管理器内自然录制。⑧ <code>self.graphs</code> 字典按 bs 索引。
</div>

---

layout: default
---

# graph_vars 的设计：为什么用预分配 tensor

```python
# capture_cudagraph 末尾 (L250-L257)
self.graph_vars = dict(
    input_ids=input_ids,        # [max_bs]
    positions=positions,        # [max_bs]
    slot_mapping=slot_mapping,  # [max_bs]
    context_lens=context_lens,  # [max_bs]
    block_tables=block_tables,  # [max_bs, max_num_blocks]
    outputs=outputs,            # [max_bs, hidden_size]
)
```

<div v-click class="mt-3 text-sm">
  <strong>为什么预分配？</strong>CUDA Graph 录制时会记录每个 tensor 的<strong>内存地址</strong>。replay 时读取相同地址的数据。如果 tensor 每次重新分配，地址不同 → graph replay 读到错误数据 → 崩溃。
</div>

<div v-click class="mt-3 text-sm">
  <strong>graph_vars 的固定地址 + data copy 模式</strong>：
  <ul>
    <li>录制阶段：<code>graph_vars</code> 的内容（地址）被硬编码到 graph 中</li>
    <li>replay 阶段（model_runner.py:L204-L210）：将新输入 <strong>copy</strong> 到 graph_vars 的预分配 buffer</li>
    <li><code>graph_vars["slot_mapping"].fill_(-1)</code> → 原地修改已有 tensor</li>
    <li><code>graph_vars["slot_mapping"][:bs] = context.slot_mapping</code> → slice copy</li>
    <li>replay 时 CUDA 引擎直接读取这些 buffer → 无需重新分配内存</li>
  </ul>
</div>

---
layout: default
---

# 3.2 CUDA Graph：replay 条件

<SourceCode file="nanovllm/engine/model_runner.py" lines="195-212" />

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    # 三个条件任一满足 → eager 路径
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))

    # 其余 → graph replay
    bs = input_ids.size(0)
    graph_vars["slot_mapping"].fill_(-1)
    graph_vars["slot_mapping"][:bs] = context.slot_mapping
    graph_vars["context_lens"].zero_()
    graph_vars["context_lens"][:bs] = context.context_lens
    graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = ...
    self.graphs[bs].replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

---
layout: default
---

# 3.2 replay 条件三选一

| 条件 | 为什么不能 replay |
|------|-------------------|
| `is_prefill` | prefill 的 token 总数变化大，无法提前 capture 所有可能大小 |
| `enforce_eager` | 调试开关，强制逐 kernel 执行 |
| `bs > 512` | 超出 capture 的最大 batch size (512) |

<div v-click class="mt-4 p-3 bg-gray-800/50 rounded text-sm">
  <strong>为什么只 cover decode？</strong>Decode 阶段输入形状稳定——每个 seq 刚好 1 个 token，batch size 是唯一的变量。CUDA Graph 把整个 forward 路径录制成一个"重放录像"，消除了每个 kernel 逐个 launch 的 CPU 开销。这对 decode 特别重要，因为 decode 的 kernel 很小（每次 1 token × N heads），CPU launch 延迟占比高。
</div>

---
layout: default
---

# CUDA Graph 显存开销分析

每个 graph bucket 需要存储完整的 forward 计算图中间状态。以 <code>max_bs=512</code> 为例：

| bucket (bs) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|-----------|----|----|----|----|----|----|----|----|----|----|
| graph 个数 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

每个 graph 占用显存估算：

```
graph_memory ≈ (input_ids + positions + outputs + slot_mapping + 
                context_lens + block_tables) × bs × dtype_size
              ≈ (4 + 4 + 2048 + 4 + 4 + 4×64) × bs × 4 bytes
              ≈ 2304 × bs × 4 bytes
```

| bs=1 | bs=64 | bs=512 | 总计 (10 buckets) |
|------|-------|--------|-------------------|
| ~9 KB | ~576 KB | ~4.5 MB | ~15-25 MB（含头部） |

<div v-click class="mt-3 text-sm">
  <strong>结论</strong>：CUDA Graph 的显存开销在 Qwen3-0.6B 规模下可以忽略（<30 MB）。对于大模型（7B+），每个 graph 的显存占用线性增长，但仍远低于 KV cache 的显存开销。
</div>

---

layout: default
---

# 3.3 torch.compile：只编译采样模块

<SourceCode file="nanovllm/layers/sampler.py" lines="5-12" />

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Gumbel-Max 技巧：等价于按概率采样
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

<div v-click class="mt-3 text-sm">
  <strong>为什么只编译 Sampler？</strong>采样模块的计算图稳定（没有动态 shape），且是纯 PyTorch op（无 Triton kernel），编译器可以自由融合。Transformer 前向包含 FlashAttention 的 Triton kernel 和动态 shape，编译会触发大量重编译。<br/>
  Gumbel-Max 技巧：<code>-log(-log(U)) / temperature + logits</code> → argmax，等价于从 softmax 分布中采样。
</div>

---
layout: default
---

# enforce_eager=True 时跳过哪些优化

| 优化 | 代码位置 | 跳过原因 |
|------|----------|----------|
| **CUDA Graph capture** | `capture_cudagraph()` (L36) | `if not self.enforce_eager: self.capture_cudagraph()` |
| **CUDA Graph replay** | `run_model()` (L197) | `if ...or self.enforce_eager...: → eager 路径` |
| **torch.compile** | Sampler.forward (L7) | 不受 enforce_eager 影响，始终开启 |

<div v-click class="mt-3 text-sm">
  <strong>enforce_eager 的作用域</strong>：只控制 CUDA Graph 的 capture 和 replay，不影响 TP（始终开启）和 torch.compile（始终开启）。<br/>
  启用后每个 kernel 独立 launch，调试友好但 decode 延迟会增加（~0.3ms/step 的 kernel launch 开销）。
</div>

---

layout: default
---

# Gumbel-Max 技巧的数学推导

```python
# Sampler.forward (L9-L11)
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```

<SourceCode file="nanovllm/layers/sampler.py" lines="1-12" />

**等价性证明**：从 softmax 分布采样等价于 argmax(logits + Gumbel 噪声)。

```
P(token_i) = softmax(logits/T)_i
Gumbel-Max: token = argmax(logits/T + Gumbel(0,1))
```

其中 `Gumbel(0,1) = -log(-log(U))`, `U ~ Uniform(0,1)`。代码通过 `exponential_(1)` 生成 `-log(U)`，因此 `probs / exp(1)` ≈ `probs / U`，取 argmax 等价于按概率采样。

<div v-click class="mt-3 text-sm">
  <strong>优势</strong>：argmax 是确定性的，而 <code>torch.multinomial</code> 涉及随机索引。Gumbel-Max 通过给 logits 加噪声把随机采样转化为 argmax，更易被 torch.compile 优化。
</div>

---

layout: default
---

# 三种优化可以叠加吗？

| 组合 | 兼容性 | 说明 |
|------|--------|------|
| TP + CUDA Graph | <strong>兼容</strong> | TP 改变模型并行度，CUDA Graph 录制 decode 路径。nano-vllm 默认同时启用两者。 |
| TP + torch.compile | <strong>兼容</strong> | compile 只作用在 Sampler 上，不涉及模型并行通信。 |
| CUDA Graph + torch.compile | <strong>兼容</strong> | compile 编译 Sampler.forward，CUDA Graph 录制 Transformer forward。两者操作不同模块。 |
| 三者同时 | <strong>兼容</strong> | 三者在代码的不同位置，互不干扰。nano-vllm 默认配置即全部启用。 |

<div v-click class="mt-3 p-3 bg-gray-800/50 rounded text-sm">
  <strong>原理</strong>：三种优化攻击不同瓶颈——TP 攻算力（ModelRunner.run）、CUDA Graph 攻 kernel launch（run_model decode）、torch.compile 攻 Python 开销（Sampler.forward）。它们作用于不同粒度和不同模块，因此可以叠加。
</div>

---

layout: default
---

# 优化效果估算表

基于 Qwen3-0.6B, bs=16, decode 1024 tokens 的粗略估算：

| 优化 | 单步 decode 延迟 | 吞吐 (tokens/s) | 说明 |
|------|-----------------|-----------------|------|
| 无优化 (eager) | ~2.0 ms | ~8,000 | 所有 kernel 逐一 launch |
| + CUDA Graph | ~1.7 ms (-15%) | ~9,400 | 消除 kernel launch 开销 |
| + TP (2 GPU) | ~1.0 ms (-50%) | ~16,000 | 算力翻倍 |
| + torch.compile | ~1.6 ms (-20%) | ~10,000 | Sampler 编译优化 |
| 三者全开 | ~0.9 ms (-55%) | ~17,800 | 叠加效果 |

<div v-click class="mt-3 text-sm">
  <strong>注意</strong>：以上为粗略估算，实际效果取决于 GPU 型号、batch size 和模型大小。TP 的加速比受 NCCL 通信开销限制；CUDA Graph 在小 batch size 下收益更明显；torch.compile 在 Sampler 上的收益有限（Sampler 只占整个 step 的 ~5%）。
</div>

---

layout: default
---

# 未来可能的优化方向

| 优化方向 | 攻击瓶颈 | 在 nano-vllm 中的现状 | 复杂度 |
|---------|----------|----------------------|--------|
| **Continuous Batching** | 调度效率 | 已实现（Scheduler） | 已有 |
| **Prefix Caching** | Prefill 计算量 | 已实现（BlockManager） | 已有 |
| **INT8/FP8 KV Cache 量化** | KV cache 显存 | 未实现 | 高 |
| **PageAttention（vLLM 式）** | KV cache 碎片 | 未实现 | 高 |
| **Speculative Decoding** | 解码延迟 | 未实现 | 高 |
| **FlashAttention 3** | Attention 带宽 | 未用（换库即可） | 低 |
| **Async Engine** | 请求级延迟 | 未实现 | 高 |

<div v-click class="mt-3 text-sm">
  nano-vllm 作为教学项目只实现了最基础的优化。实际生产系统（vLLM、TensorRT-LLM）会组合上述所有优化。理解 nano-vllm 的优化位置感后，阅读 vLLM 源码会容易很多。
</div>

---
layout: section
---

# 4. L08 验证脚本
## L08_optimizations.py 走读

---
layout: default
---

# §1：CUDA Graph replay 条件验证

```python
def will_replay(is_prefill, enforce_eager, batch_size):
    return not is_prefill and not enforce_eager and batch_size <= 512

# 6 个测试用例
assert will_replay(True,  False, 4)   == False  # prefill → eager
assert will_replay(False, True,  4)   == False  # eager → eager
assert will_replay(False, False, 600) == False  # bs>512 → eager
assert will_replay(False, False, 4)   == True   # 正常 replay
assert will_replay(False, False, 256) == True   # 正常 replay
assert will_replay(False, False, 512) == True   # 边界 case

# Graph bucket 选择: 取 >= bs 的最小 bucket
def pick_bucket(bs, buckets=[1,2,4,8,16,32,64,128,256,512]):
    return min(b for b in buckets if b >= bs)
assert pick_bucket(3) == 4
assert pick_bucket(100) == 128
```

---

layout: default
---

# §2：TP 广播流程模拟

```python
def simulate_tp_broadcast(method_name, world_size):
    log = []
    log.append(f"[rank0] call('{method_name}')")
    log.append(f"[rank0] pickle.dumps → shm.buf")
    for i in range(1, world_size):
        log.append(f"[rank0] event[{i-1}].set() → rank{i}")
        log.append(f"[rank{i}] event.wait() → 读 shm")
        log.append(f"[rank{i}] pickle.loads → '{method_name}'")
        log.append(f"[rank{i}] 执行 {method_name}(*args)")
    log.append(f"NCCL all-reduce: 各 rank 结果同步")
    log.append(f"[rank0] 返回, rank>0 返回 None")
    return log
```

<div v-click class="mt-3 text-sm">
  模拟 3 进程 (world_size=3) 场景：rank0 写 shm → set Event → rank1,2 被唤醒 → 读 shm → 执行 + NCCL 同步。对齐 model_runner.py:L61-L89。
</div>

---

layout: default
---

# §3-4：torch.compile + 优化地图总结

<div class="grid grid-cols-2 gap-3 mt-3 text-sm">
<div class="bg-purple-500/10 p-3 rounded">
  <strong>§3: torch.compile 位置</strong><br/>
  打印 sampler.py 源码<br/>
  解释 Gumbel-Max 技巧<br/>
  分析为什么适合 compile
</div>
<div class="bg-yellow-500/10 p-3 rounded">
  <strong>§4: 优化地图总结</strong><br/>
  ASCII 表格：优化→瓶颈→入口<br/>
  触发条件摘要
</div>
</div>

---
layout: default
---

# 4.1 课堂练习

```python
# 复刻 run_model 的 replay 判定逻辑
def run_model_sim(is_prefill, enforce_eager, batch_size):
    """模拟 ModelRunner.run_model 的分支选择"""
    if is_prefill or enforce_eager or batch_size > 512:
        path = f"eager: prefill={is_prefill}, "
        path += f"eager={enforce_eager}, bs={batch_size}"
        return {"path": "eager", "reason": path}
    else:
        return {"path": "replay", "bucket": pick_bucket(batch_size)}

# 测试不同场景
print(run_model_sim(True, False, 4))    # eager (prefill)
print(run_model_sim(False, True, 4))    # eager (enforce)
print(run_model_sim(False, False, 600)) # eager (large bs)
print(run_model_sim(False, False, 64))  # replay (bucket 64)
print(run_model_sim(False, False, 50))  # replay (bucket 64)
```

---
layout: default
---

# 4.2 课后自测题

<SelfTest
  id="l08-q1"
  type="text"
  question="1. CUDA Graph replay 的阈值 bs > 512 改成 1024 需要什么额外操作？为什么不是越大越好？"
  answer="<strong>需要的操作</strong>：在 <code>capture_cudagraph</code> 中添加 <code>bs=1024</code> 的 capture（在 <code>[1,2,4,...,512]</code> 后面加 <code>1024</code>），同时将 <code>run_model</code> 中的阈值改为 <code>> 1024</code>。<br><strong>为什么不是越大越好</strong>：每个 capture 占用显存（需要存储整个 graph 的中间状态），graph 越多显存浪费越多。而且实际 decode 的 batch size 受限于显存中的 KV cache 可用量——如果只能同时服务 50 个 seq，capture 1024 的 graph 完全没有用武之地。"
/>

<SelfTest
  id="l08-q2"
  type="text"
  question="2. TP 用 spawn + 共享内存而不是 fork。fork 和 spawn 在 CUDA 上下文继承上的根本差异是什么？"
  answer="<strong>fork vs spawn</strong>：<code>fork()</code> 子进程继承父进程的全部内存（包括 CUDA context），在 Linux 上如果父进程已经初始化了 CUDA，子进程使用继承的 CUDA context 会导致不可预测的行为（cuda runtime 不支持 fork）。<code>spawn</code> 创建全新 Python 解释器，子进程独立初始化 CUDA——每个子进程有自己的 CUDA context，通过 NCCL 通信。这是 vLLM 等生产框架也使用 spawn 的根本原因。<br><strong>代价</strong>：spawn 需要序列化参数传递给子进程（这就是为什么 Sequence 需要 <code>__getstate__/__setstate__</code>）。"
/>

---
layout: default
---

# 4.2 课后自测题（续）

<SelfTest
  id="l08-q3"
  type="text"
  question="3. torch.compile 只用在 Sampler.forward。如果给整个 Transformer 前向加上 compile 会遇到什么问题？"
  answer="<strong>会遇到的问题</strong>：<br>1.<strong>动态 shape</strong>：Transformer 前向的输入形状随 batch size 和 seq 长度变化。每次 shape 变化都会触发 torch.compile 的<strong>重编译</strong>（recompilation），导致编译开销远超收益。<br>2.<strong>Triton kernel 互操作</strong>：FlashAttention 是 Triton kernel，torch.compile 无法对 Triton kernel 做额外优化——compiler 只能优化 PyTorch 层的 op 融合。<br>3.<strong>CUDA Graph 冲突</strong>：torch.compile 生成的代码可能包含动态分配或 control flow，与 CUDA Graph 的静态录制要求冲突。<br>4.<strong>编译时间</strong>：整个 Transformer 的编译可能需要数分钟，而推理只需几秒——得不偿失。<br><strong>Sampler 为什么适合</strong>：输入形状固定（bs × vocab_size），没有动态分支，纯 PyTorch op（div、softmax、exponential、argmax），编译后可以融合为少量 kernel。"
/>

---
layout: center
---

# 🎉 全部 8 课完成！

<div class="mt-6 text-lg opacity-80">
  从 LLM.generate 到 CUDA Graph，完成了 nano-vllm 的源码走读
</div>

<div class="mt-4 grid grid-cols-4 gap-3 text-sm max-w-2xl mx-auto">
  <div class="bg-blue-500/10 p-3 rounded">L01 端到端流程</div>
  <div class="bg-green-500/10 p-3 rounded">L02-L04 调度层</div>
  <div class="bg-purple-500/10 p-3 rounded">L05-L07 执行层</div>
  <div class="bg-yellow-500/10 p-3 rounded">L08 优化全景</div>
</div>

<div class="mt-6 grid grid-cols-3 gap-2 text-sm max-w-xl mx-auto">
  <div class="bg-blue-500/10 p-2 rounded">✅ Tensor Parallel</div>
  <div class="bg-green-500/10 p-2 rounded">✅ CUDA Graph</div>
  <div class="bg-purple-500/10 p-2 rounded">✅ torch.compile</div>
</div>

<div class="mt-8 text-sm opacity-60">
  复习建议：从 <code>L01_end_to_end.py</code> 重新跑一遍，对照幻灯片回顾每课知识点
</div>
