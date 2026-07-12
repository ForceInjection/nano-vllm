# 设计文档：KV Cache CPU Offloading

> 目标：为 nano-vllm 增加 KV Cache 的 CPU 卸载能力——将被抢占序列的 KV Cache 迁移至 CPU 内存（swap out），并在重新调度时迁回 GPU（swap in），以替代当前"丢弃并重算 prefill"的做法。设计参考 vLLM，但在实现上以可读性为优先。

> 状态：P1 已实现并在 RTX 3090 上通过验证（B1/B2/B3/C）。本文兼作设计说明与实现记录，包含实现过程中经 GPU 验证与代码评审发现并修复的问题。

---

## 1. 背景：当前的抢占策略为"丢弃 + 重算"

nano-vllm 在显存不足时的唯一抢占手段是重算（recompute）：将一个正在 decode 的序列移回 `waiting`，释放其已计算的 KV，下一轮重新执行 prefill。

| 环节        | 位置                                  | 现有行为                                                                                                                                                               |
| ----------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KV 显存布局 | `nanovllm/engine/model_runner.py:115` | 单张 GPU 张量 `kv_cache[2, L, num_blocks, block_size, kv_heads, head_dim]`；`2` = K/V，每层 attention 的 `k_cache/v_cache` 是它的切片视图（`model_runner.py:117-121`） |
| 块寻址      | `nanovllm/layers/attention.py:28`     | `slot = block_id * block_size + offset`；一个物理块 = `kv_cache[:, :, block_id]`，横跨所有层                                                                           |
| 块元数据    | `nanovllm/engine/block_manager.py`    | 纯元数据：`free_block_ids` / `used_block_ids` / `hash_to_block_id`，不直接操作张量                                                                                     |
| 抢占        | `nanovllm/engine/scheduler.py:75-79`  | `preempt()` → `block_manager.deallocate()`：块归还 free 池、KV 全部丢弃；序列回 `waiting`、`is_prefill=True`，下一轮重算                                               |

`vs-vllm.md` 第 2.3 节记录了这一差距：

> - nano 里：block 不足时只能 preempt（丢弃 KV cache，下一轮 prefill 重算）
> - vLLM 里：swap out → swap in，避免重算开销

本设计补齐该能力。

## 2. vLLM 参考实现（V0）

- **配置**：`CacheConfig.swap_space`（默认 4 GiB）→ 在 CPU 侧分配 `pin_memory=True` 的 KV 张量。
- **三队列调度**：`waiting / running / swapped`。
- **两种抢占模式**：
  - `RECOMPUTE`：丢弃 KV，重算（等价于 nano 现状）。
  - `SWAP`：将 KV 块迁移至 CPU，重新调度时迁回。
- **职责分离**：`Scheduler` 每步只产出块映射 `blocks_to_swap_in / blocks_to_swap_out`（GPU 块 ↔ CPU 块）；张量拷贝由 `Worker` 侧 `cache_engine.swap_in/out` → `ops.swap_blocks` 逐块执行。

nano-vllm 沿用相同的分层与三队列结构，但将 vLLM 的 per-layer CUDA 拷贝核简化为一次张量切片 `copy_`——这得益于 nano 的单张大张量布局，代码量更小。

## 3. 目标架构：引入第三队列 `swapped`

```text
              allocate 不足（且 CPU 有空间）
   running ───────────────────────────────► swapped（KV 驻留在 CPU）
      ▲                                          │
      │        GPU 有空块 & 轮到它                 │ swap_in
      └──────────────────────────────────────────┘
                    （以 decode 身份直接续跑，无需重算）

              CPU 也满时的兜底
   running ───────────────────────────────► waiting（RECOMPUTE，= 现状）
```

与现有逻辑的对照：

- `preempt` 优先执行 swap_out 迁移至 CPU；仅当 CPU 也满时才退回 RECOMPUTE。
- GPU 有空块时优先执行 swap_in 续跑，而非从 `waiting` 重新 prefill。

## 4. 详细设计（按文件）

### 4.1 `config.py`

新增开关（默认 0 = 关闭，不影响现有行为与基准）：

```python
cpu_offload_gb: float = 0        # >0 时启用 KV Cache CPU Offloading
```

`config.py` 除该开关外，仅保留一个 `num_cpu_kvcache_blocks: int = 0` 字段作为回写通道；实际 CPU 块数由 `model_runner.allocate_kv_cache` 复用其已有的 `block_bytes` 计算（`int(cpu_offload_gb * 2**30) // block_bytes`）后写回该字段，供 `Scheduler` 读取，以避免公式重复（见 §4.4）。

### 4.2 `engine/sequence.py`

- `SequenceStatus` 增加 `SWAPPED = auto()`。
- **关键不变量**：swap 路径下 `num_cached_tokens` 不清零——这是 swap 相对 recompute 能省去重算的前提（swap_in 后 decode 从断点续跑）。
- CPU 块列表存放于 `BlockManager` 中并按 `seq_id` 索引，`Sequence` 不新增字段，以控制 `__getstate__` 的 pickle 范围（`sequence.py:72-83`）。

### 4.3 `engine/block_manager.py`（元数据层，不操作张量）

新增独立的 CPU 块池与"仅计算映射"的方法：

```python
self.free_cpu_block_ids: deque[int]              # CPU 块空闲表
self.used_cpu_block_ids: set[int]
self.swapped_block_tables: dict[int, list[int]]  # seq_id -> cpu block ids

def can_swap_out(self, seq) -> bool              # CPU 空块是否够
def swap_out(self, seq) -> dict[int, int]        # 分配 CPU 块，返回 {gpu_id: cpu_id}，释放 GPU 块
def can_swap_in(self, seq) -> bool               # GPU 空块是否够
def swap_in(self, seq) -> dict[int, int]         # 分配 GPU 块，返回 {cpu_id: gpu_id}，重建 seq.block_table，释放 CPU 块
```

**MVP 简化**：CPU 块不参与前缀哈希，仅作为临时存放位——swap_out 时记录 `gpu_id -> cpu_id` 映射并释放 GPU 块；swap_in 时分配新 GPU 块、拷回、重建 `block_table`，随后按需重新 `hash_blocks`。（vLLM 在 CPU 侧同样维护完整的 prefix-cache，本设计将其作为后续扩展。）

**共享块规则**：一个序列当且仅当其所有块 `ref_count == 1`（全部独占）时才允许 swap_out；只要含有与其他序列共享的块（`ref_count > 1`），即退回 RECOMPUTE。此规则避免了"共享块被迁至 CPU、而 swap_in 时另一持有者仍需其驻留 GPU 或该块已被逐出"的一致性问题；退回路径本已存在，代价可接受，换取实现的显著简化。`can_swap_out(seq)` 需同时校验"CPU 空块足够"与"seq 全部块独占"。

### 4.4 `engine/model_runner.py`（执行层，执行张量拷贝）

`allocate_kv_cache()` 末尾追加 CPU 镜像张量：

```python
self.cpu_kv_cache = torch.empty(
    2, L, num_cpu_blocks, block_size, kv_heads, head_dim,
    device="cpu", pin_memory=True,   # pin_memory 使 GPU<->CPU DMA 可异步
)
```

新增两个方法，经现有 `call()` 共享内存 RPC 广播至所有 TP rank 执行（块 id 各 rank 一致，无需额外同步）：

```python
def swap_out(self, mapping: dict[int, int]):     # gpu -> cpu
    for g, c in mapping.items():
        self.cpu_kv_cache[:, :, c].copy_(self.kv_cache[:, :, g], non_blocking=True)

def swap_in(self, mapping: dict[int, int]):      # cpu -> gpu
    for c, g in mapping.items():
        self.kv_cache[:, :, g].copy_(self.cpu_kv_cache[:, :, c], non_blocking=True)
```

一次 `[:, :, block_id]` 切片即覆盖该块在所有层的 K 与 V，相较 vLLM 的逐层核更为简洁——这是单张大张量布局的直接结果。TP 下每 rank 仅迁移各自的 KV 分片。

> **P1 采用同步拷贝**：先去掉 `non_blocking=True`（改为同步 `copy_`），实现更简单；CUDA stream 重叠留待 P2。上文的 `non_blocking` 为 P2 形态。
>
> `num_cpu_blocks` 的计算统一放在此处（`allocate_kv_cache` 已有 `block_bytes`），`config.py` 仅保存 `cpu_offload_gb`，避免 `block_bytes` 公式在两处重复。
>
> 此外，`allocate_kv_cache` 将正的 `config.num_kvcache_blocks` 作为 GPU 块数上限（`num_blocks = min(计算值, 该上限)`；默认 `-1` 表示不设上限，行为不变）。这为测试与压测提供了强制抢占的手段（见 §10.E 的 `cap` 参数）。

### 4.5 `engine/scheduler.py`

- `__init__` 增加 `self.swapped: deque[Sequence] = deque()`。
- `preempt(seq)` 改写：
  - `block_manager.can_swap_out(seq)` → `swap_out`，状态置 `SWAPPED`，入 `swapped` 队列，记录 `blocks_to_swap_out`。
  - 否则退回 RECOMPUTE（抽为 `_recompute()`）。
  - 发布实现在上述判断之前还有一层护栏：若待抢占序列在本 step 刚被换入，则取消其 swap-in 并退回 RECOMPUTE（见 §6）。
- `schedule()` 在 decode 阶段之前插入 swap-in 阶段：GPU 有空块时，对 `swapped` 队首执行 `swap_in`、状态置 `RUNNING`、进入 decode，并记录 `blocks_to_swap_in`。
- `schedule()` 通过实例属性携带 `blocks_to_swap_out / blocks_to_swap_in` 两个映射，供引擎在模型前向前执行拷贝。

### 4.6 `engine/llm_engine.py`

`step()` 中，在 `schedule()` 之后、`run()` 之前，若存在 swap 映射则先执行：

```python
self.model_runner.call("swap_out", blocks_to_swap_out)   # 有则调
self.model_runner.call("swap_in",  blocks_to_swap_in)
```

随后进入正常的 `run()`。

## 5. 一次 step 的完整时序（swap 场景）

```text
LLMEngine.step()
  └─ Scheduler.schedule()
       ├─ swap-in 阶段：GPU 有空块 → swapped 队首 swap_in（元数据），记 blocks_to_swap_in
       ├─ prefill / decode 组批（现有逻辑）
       └─ decode 中 allocate 不足 → preempt → swap_out（元数据），记 blocks_to_swap_out
  ├─ model_runner.call("swap_out", ...)   # GPU→CPU 张量拷贝（模型前向前）
  ├─ model_runner.call("swap_in",  ...)   # CPU→GPU 张量拷贝
  ├─ model_runner.run(seqs, is_prefill)   # 前向 + 采样
  └─ Scheduler.postprocess(...)           # 写回 token、hash 块、收尾
```

> **顺序约束**：必须严格保持 `swap_out 拷贝 → swap_in 拷贝 → run()`。原因见 §6 的"拷贝顺序"条。任何重排（swap_in 提前、或将拷贝移至 run 之后）都会导致 KV 被静默破坏。

## 6. 正确性与边界

- **拷贝顺序不可交换**：同一 decode step 内，被 `swap_out` 释放的 GPU 块可能被 `may_append` 或 swap-in 目标在同一步复用。因此必须保持 `swap_out 拷贝 → swap_in 拷贝 → run()`：① `swap_out`（GPU→CPU）须在这些块被 `run()` 覆盖之前读出；② 若某块同时是 swap_out 源与 swap_in 目标，须先 out 后 in。固定该顺序即可保证正确性。
- **换入的序列不得在同一 step 换出**：若一个序列在本 step 的 swap-in 阶段被换入、又在 decode 阶段被抢占，其 GPU 块为 swap-in 目标、KV 尚未拷回（内容无效）。此时 `swap_out` 会将无效数据拷至 CPU 并覆盖其有效 KV。`preempt()` 须检测"待抢占序列的块 ∈ `blocks_to_swap_in.values()`"，命中则取消其 swap-in 并退回 RECOMPUTE，而非 swap-out。（§5 的 out→in 顺序未覆盖此情形，此问题由 GPU 验证与代码评审发现并修复。）
- **块-哈希生命周期**：`swap_out` 释放的 GPU 块沿用 `deallocate` 的语义——保留 `.hash`，由 `_allocate_block` 在复用时清理（`block_manager.py:47-48`）。此行为与现有 preempt 一致、数据仍有效，无需特殊处理。
- **CUDA Graph 无冲突**：swap 发生在 step 之间、graph 捕获之外；swap-in 的序列以 decode 身份进 batch，走既有 graph 路径（`model_runner.py:199-212`）。
- **`num_cached_tokens` 必须保留**，否则退化为重算，功能失去意义。
- **死锁兜底**：CPU 也满时退回 RECOMPUTE，保证系统始终能前进。
- **`is_finished()` 须计入 `swapped` 队列**：`waiting`/`running` 清空但 `swapped` 非空时仍未结束，否则引擎会误判完成并丢弃已换出序列（该问题由 GPU 验证发现并修复）。
- **默认关闭**（`cpu_offload_gb=0`）：不改变现有基准、行为与显存计算。
- **TP 一致性**：swap 的块 id 由调度器统一决定并广播至所有 rank；各 rank 仅迁移各自的 KV 分片，结果一致。

## 7. 配套产出（随代码交付）

1. **课程一节 + 图示**（`docs/llm-inference-visual/`）：三队列生命周期、swap 数据流、与 vLLM 的对照。
2. **验证脚本**：以较小的 `gpu_memory_utilization` 强制抢占，打印 `swap_out N blocks → CPU` / `swap_in N blocks → GPU`，并对拍启用 swap 与走 RECOMPUTE 两条路径的最终输出逐 token 一致（验证 swap 只影响性能、不影响结果）。
3. **更新 `vs-vllm.md` 第 2.3 节**：从"未实现"改为"已实现简化版 swap，与 vLLM 的差异在于 CPU 侧不维护前缀缓存、拷贝未做 stream 重叠"。

## 8. 分阶段实现

| 阶段               | 内容                                                  | 说明                                   |
| ------------------ | ----------------------------------------------------- | -------------------------------------- |
| **P1（核心）**     | 同步 `copy_` 的 swap out/in + 三队列 + RECOMPUTE 兜底 | 以"跑通并与重算路径对拍一致"为完成标准 |
| **P2（可选优化）** | 独立 CUDA stream + event，使拷贝与计算重叠            | 更贴近生产实现                         |
| **P3（可选扩展）** | 主动分层卸载（不限于抢占时触发），接近 KV tiering     | 复杂度更高，作为进阶                   |

## 9. 关键决策（P1 已采纳）

以下取舍在 P1 落地时均已采纳：

1. **范围**：P1 仅实现"抢占时 swap"（reactive）——最小、最贴合 vLLM SWAP 模式；主动卸载（P3）作为扩展。
2. **拷贝方式**：先用同步 `copy_`，CUDA stream 重叠留待 P2。
3. **CPU 前缀缓存**：MVP 不在 CPU 侧维护前缀哈希（CPU 块作为不可共享的存放位），swap_in 后重新 hash。
4. **swap-in 优先级**：P1 不让 swapped-in 优先于 `waiting` 新请求（沿用现有"prefill 阶段先 return"结构，实现最简）。vLLM 会优先迁回在飞的 swapped 序列以更快回收显存，此项留待 P2 调优。

---

## 10. 测试方案

> **注意**：直接"开/关 swap 各跑一遍、对比输出文本"不可靠——采样为 Gumbel-Max 随机，两条路径抢占时机不同 → batch 组成不同 → RNG 抽取顺序不同 → 即使 KV 完全正确，输出 token 也可能分叉。因此对拍前须先消除采样随机性（见 B2）。

测试分层设计，区分本机（macOS，无 GPU）可运行部分与需在 GPU（3090）上运行部分。

### A. 单元测试（纯 Python，本机可运行，可进 CI）

多数缺陷位于元数据状态机，此层应覆盖充分。落地文件：`test_swap_blockmanager.py`（pytest）。

- **A1. BlockManager swap 元数据**（纯 dict/deque，不操作张量）：
  - `swap_out(seq)` 返回的 `{gpu_id: cpu_id}` 正确；GPU 块归还 `free_block_ids`、CPU 块进入 `used_cpu_block_ids`、`swapped_block_tables[seq_id]` 记录正确。
  - `swap_in(seq)` 逆操作：重建 `seq.block_table`（长度/顺序正确）、CPU 块释放、`num_cached_tokens` 保持不变（核心不变量）。
  - 往返一致：swap_out → swap_in 后逻辑块序列等价（物理 id 可不同）。
  - `can_swap_out` 在 CPU 池满时返回 False；`can_swap_in` 在 GPU 空块不足时返回 False。
- **A2. 拷贝索引正确性**：`model_runner.swap_out/in` 的切片 `dst[:,:,c].copy_(src[:,:,g])` 逻辑，用 CPU 张量测试（两侧 `device="cpu"`），验证 `[:, :, block_id]` 寻址与映射正确。可在本机拦截索引错误，无需 GPU。

### B. 集成测试（需在 3090 上运行）

落地文件：`docs/llm-inference-visual/scripts/verify_swap.py`（沿用 `verify_nanovllm.py` 风格，argv/env 传模型路径）。

- **B1. KV 逐字节往返相等（最强、完全确定）**：以已知随机值填充 `kv_cache` 若干块 → `swap_out` 至 CPU → `swap_in` 至另一批 GPU 块 → 断言 `torch.equal`。直接验证拷贝与映射无损，与采样无关。
- **B2. 单序列 swap 往返，逐 token 一致（确定性）**：注入 argmax 贪心采样（monkeypatch sampler）以消除采样随机性。以单条序列驱动 `step()`，解码若干步后强制 `swap_out` 至 CPU、再由调度器 `swap_in` 迁回，继续解码至结束；与同一序列不经 swap 的参照输出逐 token 对比。全程 batch 恒为 1，数值不受调度顺序影响，任何差异即为 KV 损坏。
  > 注：最初计划的"baseline / RECOMPUTE / SWAP 三路差分对拍"不可行——三条路径抢占时机不同导致 batch 组成不同，即便 KV 正确，跨不同 batch 规模的浮点结果也会在近似平局处使 argmax 翻转。单序列 batch=1 方案消除了该变量。
- **B3. 端到端冒烟与死锁检查**：以极小 `gpu_memory_utilization` + 长 prompt + 高并发强制大量 swap；断言可完成、无报错、无死锁、输出连贯。

### C. 可观测性断言

新增计数器 `num_swapped_out_blocks / num_swapped_in_blocks / num_recompute_preemptions`，测试断言 swap 计数 > 0——否则内存过大不会触发 swap，测试将成为空转（silent pass）。

### D. 边界用例

- **CPU 池耗尽 → 退回 RECOMPUTE**：不崩溃、能前进（验证兜底路径）。
- **共享前缀块**（`ref_count > 1`）：按 §4.3 规则，含共享块的序列整体退回 RECOMPUTE，不做部分 swap；断言此类序列不进入 `swapped` 队列、且共享块不被迁移。
- **chunked prefill 首序列**：仅对已全缓存的 decode 序列执行 swap，不涉及正在 chunk-prefill 的序列。

### E. 性能佐证（报告用，非断言）

同压力下比较 SWAP 与 RECOMPUTE 的吞吐与 prefill 工作量。脚本：`docs/llm-inference-visual/scripts/bench_swap.py`。

**实测（RTX 3090 / Qwen3-0.6B，32 序列 × (prompt≈210 + decode 256)，cap=8 强制抢占）：**

| 配置                        | 墙钟(s) | 吞吐(tok/s) | 并发(avg_batch) | prefill_tok | 重算 | swap_out/in |
| --------------------------- | ------- | ----------- | --------------- | ----------- | ---- | ----------- |
| baseline（无压力, cap=512） | 11.26   | 727.7       | 32.0            | 5792        | 0    | 0 / 0       |
| RECOMPUTE（cap=8）          | 62.73   | 130.6       | 4.7             | 9904        | 16   | 0 / 0       |
| SWAP（cap=8, offload=4）    | 62.95   | 130.1       | 4.7             | 7848        | 8    | 16 / 8      |

**结果分析：**

- **可比对象为 RECOMPUTE 与 SWAP**：二者 cap 相同、`avg_batch` 均为 4.7（并发一致），仅抢占策略不同。baseline 的 `avg_batch`=32 是无抢占参照，其吞吐更高源于并发更大（约 8 倍），不构成对 swap 策略的比较（抢占的固有代价即并发下降，无法与无抢占取得相同并发）。
- **prefill 工作量差异（与模型规模无关）**：`prefill_tok` 显示，RECOMPUTE 重跑被抢占序列的整段 prefill，累计 9904（baseline 5792 的 1.7 倍）；SWAP 不重跑，累计 7848（接近 baseline，超出部分来自 8 次退回重算）。即 SWAP 减少了约 2000 个 prefill token 的重复计算。
- **墙钟持平**（130.1 vs 130.6）：在 0.6B 规模下，SWAP 省下的重算与其引入的 PCIe 传输开销相当，二者相互抵消。
- swap 的净收益 = 省下的重算（∝ 模型规模 × prompt 长度）− PCIe 传输（∝ KV 字节数 / 带宽）。小模型下二者持平，大模型（如 14B/32B，重算长 prefill 代价高而传输量不变）下 swap 明显占优——这也是 vLLM 在大模型场景将 swap 作为默认手段的原因。

### 测试矩阵速览

| 层  | 内容                            | 运行环境  | 确定性来源                |
| --- | ------------------------------- | --------- | ------------------------- |
| A1  | BlockManager 元数据状态机       | 本机 / CI | 纯逻辑                    |
| A2  | 拷贝索引（CPU 张量）            | 本机 / CI | 纯逻辑                    |
| B1  | KV 逐字节往返相等               | 3090      | `torch.equal`，与采样无关 |
| B2  | 单序列 swap 往返逐 token 一致   | 3090      | 注入 argmax + batch=1     |
| B3  | 端到端冒烟与死锁检查            | 3090      | —                         |
| C   | swap 计数 > 0                   | 3090      | —                         |
| D   | 边界（CPU 满 / 共享块 / chunk） | 3090      | —                         |

---

## 附：改动文件速览

| 文件                               | 改动                                                                                                                       |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `nanovllm/config.py`               | 新增 `cpu_offload_gb` 开关与 `num_cpu_kvcache_blocks` 回写字段                                                             |
| `nanovllm/engine/sequence.py`      | 新增 `SWAPPED` 状态；保证 `num_cached_tokens` 在 swap 路径不清零                                                           |
| `nanovllm/engine/block_manager.py` | CPU 块池 + `can_swap_out/swap_out/can_swap_in/swap_in`（仅元数据）                                                         |
| `nanovllm/engine/kv_swap.py`（新） | `swap_blocks(src, dst, mapping)` 纯拷贝函数（无 torch import，便于单测）                                                   |
| `nanovllm/engine/model_runner.py`  | `cpu_kv_cache` 张量 + `swap_out/swap_in`（调用 `swap_blocks`）；`num_kvcache_blocks` 作 GPU 块数上限                       |
| `nanovllm/engine/scheduler.py`     | `swapped` 队列；swap 感知的 `preempt`（含换入护栏 + `_recompute`）；swap-in 阶段；观测计数器；`is_finished` 计入 `swapped` |
| `nanovllm/engine/llm_engine.py`    | `step()` 中在模型前向前执行 swap 拷贝                                                                                      |
| `nanovllm/__init__.py`             | `LLM` 改为惰性导入（PEP 562），使子模块在无 flash_attn 时可导入以跑 CPU 单测                                               |
