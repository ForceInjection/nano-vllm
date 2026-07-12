# 设计文档：KV Cache CPU Offloading（教学导向）

> 目标：给 nano-vllm 增加"把被抢占序列的 KV Cache 搬到 CPU 内存、重新调度时搬回 GPU"的能力（swap out / swap in），替代当前"直接丢弃 + 重算 prefill"的做法。参考 vLLM 的实现，但以**可读、可讲解**为第一优先级。

---

## 1. 背景：现状是"丢弃 + 重算"

nano-vllm 当前在显存不足时的唯一手段是**抢占重算（recompute）**：把某个正在 decode 的序列踢回 `waiting`，丢掉它已经算好的 KV，下一轮从头 prefill。

| 环节        | 位置                                  | 现有行为                                                                                                                                                               |
| ----------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KV 显存布局 | `nanovllm/engine/model_runner.py:115` | 单张 GPU 张量 `kv_cache[2, L, num_blocks, block_size, kv_heads, head_dim]`；`2` = K/V，每层 attention 的 `k_cache/v_cache` 是它的切片视图（`model_runner.py:117-121`） |
| 块寻址      | `nanovllm/layers/attention.py:28`     | `slot = block_id * block_size + offset`；一个物理块 = `kv_cache[:, :, block_id]`，横跨所有层                                                                           |
| 块元数据    | `nanovllm/engine/block_manager.py`    | 纯元数据：`free_block_ids` / `used_block_ids` / `hash_to_block_id`，**不碰张量本身**                                                                                   |
| 抢占        | `nanovllm/engine/scheduler.py:75-79`  | `preempt()` → `block_manager.deallocate()`：块直接还给 free 池、**KV 全部丢失**；序列回 `waiting`、`is_prefill=True`，下轮重算                                         |

`vs-vllm.md` 第 2.3 节记录的正是这个缺口：

> - nano 里：block 不足时只能 preempt（丢弃 KV cache，下一轮 prefill 重算）
> - vLLM 里：swap out → swap in，避免重算开销

本设计填补这个缺口。

## 2. vLLM 参考实现（V0，最贴合本设计）

- **配置**：`CacheConfig.swap_space`（默认 4 GiB）→ 在 CPU 侧分配 `pin_memory=True` 的 KV 张量。
- **三队列调度**：`waiting / running / swapped`。
- **两种抢占模式**：
  - `RECOMPUTE`：丢弃 KV，重算（= nano 现状）。
  - `SWAP`：把 KV 块搬到 CPU，重新调度时搬回。
- **职责分离**：`Scheduler` 每步只产出块映射 `blocks_to_swap_in / blocks_to_swap_out`（GPU 块 ↔ CPU 块）；真正的张量拷贝由 `Worker` 侧 `cache_engine.swap_in/out` → `ops.swap_blocks` 逐块完成。

nano-vllm 会**照搬这套分层与三队列**，但把 vLLM 的 per-layer CUDA 拷贝核简化成**一次张量切片 `copy_`**——这是 nano 单张大张量布局带来的红利，代码更短、更好讲。

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

- `preempt` 不再无脑丢弃，而是**优先 swap_out** 到 CPU；只有 CPU 也满时才退回 RECOMPUTE。
- GPU 有空块时**优先 swap_in** 续跑，而不是从 `waiting` 重新 prefill。

这样正好把 vLLM 的"两种抢占模式"讲清楚。

## 4. 详细设计（按文件）

### 4.1 `config.py`

新增开关（默认 0 = 关闭，**不影响任何现有行为与基准**）：

```python
cpu_offload_gb: float = 0        # >0 时启用 KV Cache CPU Offloading
```

`config.py` **只保存这个开关**；CPU 块数 `num_cpu_blocks` 由 `model_runner.allocate_kv_cache` 用其已有的 `block_bytes` 算出（`int(cpu_offload_gb * 2**30) // block_bytes`），避免公式重复（见 §4.4）。

### 4.2 `engine/sequence.py`

- `SequenceStatus` 增加 `SWAPPED = auto()`。
- **关键不变量**：swap 路径下 `num_cached_tokens` **不清零**（这正是省掉重算的原因——swap_in 后 decode 直接从断点续）。
- CPU 块列表放在 `BlockManager` 中按 `seq_id` 保存，`Sequence` 不新增字段，避免扩大 `__getstate__` 的 pickle 面（`sequence.py:72-83`）。

### 4.3 `engine/block_manager.py`（元数据层，**不碰张量**）

新增独立的 CPU 块池与"只算映射"的方法：

```python
self.free_cpu_block_ids: deque[int]              # CPU 块空闲表
self.used_cpu_block_ids: set[int]
self.swapped_block_tables: dict[int, list[int]]  # seq_id -> cpu block ids

def can_swap_out(self, seq) -> bool              # CPU 空块是否够
def swap_out(self, seq) -> dict[int, int]        # 分配 CPU 块，返回 {gpu_id: cpu_id}，释放 GPU 块
def can_swap_in(self, seq) -> bool               # GPU 空块是否够
def swap_in(self, seq) -> dict[int, int]         # 分配 GPU 块，返回 {cpu_id: gpu_id}，重建 seq.block_table，释放 CPU 块
```

**教学 MVP 简化**：CPU 块设计成"不参与前缀哈希的停车位"——swap_out 时记录 `gpu_id -> cpu_id` 映射并释放 GPU 块；swap_in 时分配新 GPU 块、拷回、重建 `block_table`，随后按需重新 `hash_blocks`。（vLLM 会在 CPU 侧也维护完整的 prefix-cache，本设计留作延伸。）

**共享块规则（重要）**：一个序列**当且仅当其所有块 `ref_count == 1`（全部独占）时才允许 swap_out**；只要含有与其他序列共享的块（`ref_count > 1`），就**回退 RECOMPUTE**。这样彻底绕开"共享块被搬到 CPU、swap_in 时另一持有者仍需它在 GPU / 或它已被逐出"的连锁难题——兜底路径本来就存在，代价可接受，换来 MVP 大幅简化。`can_swap_out(seq)` 需同时校验"CPU 空块足够"**且**"seq 全部块独占"。

### 4.4 `engine/model_runner.py`（执行层，**真正拷贝张量**）

`allocate_kv_cache()` 末尾追加 CPU 镜像张量：

```python
self.cpu_kv_cache = torch.empty(
    2, L, num_cpu_blocks, block_size, kv_heads, head_dim,
    device="cpu", pin_memory=True,   # pin_memory 使 GPU<->CPU DMA 可异步
)
```

新增两个方法，经现有 `call()` 共享内存 RPC 广播到**所有 TP rank** 执行（块 id 各 rank 一致，无需额外同步）：

```python
def swap_out(self, mapping: dict[int, int]):     # gpu -> cpu
    for g, c in mapping.items():
        self.cpu_kv_cache[:, :, c].copy_(self.kv_cache[:, :, g], non_blocking=True)

def swap_in(self, mapping: dict[int, int]):      # cpu -> gpu
    for c, g in mapping.items():
        self.kv_cache[:, :, g].copy_(self.cpu_kv_cache[:, :, c], non_blocking=True)
```

一次切片 `[:, :, block_id]` 就搬走**该块在所有层的 K 和 V**——比 vLLM 逐层核简单得多，是讲解 nano 单张大张量布局优势的好素材。TP 下每 rank 只搬自己的分片。

> **P1 用同步拷贝**：先去掉 `non_blocking=True`（改为同步 `copy_`），最好讲、零踩坑；CUDA stream 重叠留到 P2。上面的 `non_blocking` 是 P2 形态。
>
> `num_cpu_blocks` 的计算**统一放在这里**（`allocate_kv_cache` 已有 `block_bytes`），`config.py` 只保存 `cpu_offload_gb` 一个开关，避免 `block_bytes` 公式在两处重复。

### 4.5 `engine/scheduler.py`

- `__init__` 增加 `self.swapped: deque[Sequence] = deque()`。
- `preempt(seq)` 改写：
  - `block_manager.can_swap_out(seq)` → `swap_out`，状态置 `SWAPPED`，入 `swapped` 队列，记录 `blocks_to_swap_out`。
  - 否则退回原 `deallocate()`（RECOMPUTE 兜底）。
- `schedule()` 在 decode 阶段之前插入 **swap-in 阶段**：GPU 有空块时，把 `swapped` 队首 `swap_in`、状态置 `RUNNING`、直接进入 decode（不重算），记录 `blocks_to_swap_in`。
- `schedule()` 的返回值扩展为携带 `blocks_to_swap_out / blocks_to_swap_in` 两个映射，供引擎在跑模型前执行拷贝。

### 4.6 `engine/llm_engine.py`

`step()` 中，`schedule()` 之后、`run()` 之前，若存在 swap 映射，先调用：

```python
self.model_runner.call("swap_out", blocks_to_swap_out)   # 有则调
self.model_runner.call("swap_in",  blocks_to_swap_in)
```

再进入正常的 `run()`。

## 5. 一次 step 的完整时序（swap 场景）

```text
LLMEngine.step()
  └─ Scheduler.schedule()
       ├─ swap-in 阶段：GPU 有空块 → swapped 队首 swap_in（元数据），记 blocks_to_swap_in
       ├─ prefill / decode 组批（现有逻辑）
       └─ decode 中 allocate 不足 → preempt → swap_out（元数据），记 blocks_to_swap_out
  ├─ model_runner.call("swap_out", ...)   # GPU→CPU 张量拷贝（跑模型前）
  ├─ model_runner.call("swap_in",  ...)   # CPU→GPU 张量拷贝
  ├─ model_runner.run(seqs, is_prefill)   # 正常前向 + 采样
  └─ Scheduler.postprocess(...)           # 写回 token、hash 块、收尾
```

> **⚠️ 顺序不可交换（硬约束）**：必须严格 **`swap_out 拷贝 → swap_in 拷贝 → run()`**。原因见 §6 的"拷贝顺序"条。任何重排（swap_in 提前、或把拷贝挪到 run 之后）都会**静默损坏 KV**。

## 6. 正确性与边界

- **⚠️ 拷贝顺序不可交换**：同一 decode step 内，被 `swap_out` 释放的 GPU 块会被 `may_append` 或 swap-in 目标**当场复用**。因此必须 **`swap_out 拷贝 → swap_in 拷贝 → run()`**：① `swap_out`（GPU→CPU）要在这些块被 `run()` 覆盖**之前**读出；② 若某块既是 swap_out 源又是 swap_in 目标，必须先 out 后 in。保持此固定顺序即正确，重排则静默损坏 KV。
- **块-哈希生命周期**：`swap_out` 释放的 GPU 块**沿用 `deallocate` 的语义**——保留 `.hash`，由 `_allocate_block` 在复用时懒清理（`block_manager.py:47-48`）。这与现有 preempt 行为同构、数据仍有效，无需特殊处理。
- **CUDA Graph 无冲突**：swap 发生在 step 之间、graph 捕获之外；swap-in 的序列以 decode 身份进 batch，走既有 graph 路径（`model_runner.py:199-212`）。
- **`num_cached_tokens` 必须保留**，否则退化成重算，功能失去意义。
- **死锁兜底**：CPU 也满 → 回退 RECOMPUTE，保证系统始终能前进。
- **默认关闭**（`cpu_offload_gb=0`）：不改变现有基准、行为与显存计算。
- **TP 一致性**：swap 的块 id 由调度器统一决定，广播给所有 rank；各 rank 只搬各自的 KV 分片，天然一致。

## 7. 教学产出（建议与代码一起交付）

1. **课程一节 + 图**（`docs/llm-inference-visual/`）：三队列生命周期、swap 数据流、与 vLLM 的对照。
2. **verify 脚本**：用很小的 `gpu_memory_utilization` 强制抢占，打印 `swap_out N blocks → CPU` / `swap_in N blocks → GPU`，并**对拍**：开启 swap 与走 RECOMPUTE 两条路径的最终输出**逐 token 一致**（证明 swap 只影响性能、不影响结果）。
3. **更新 `vs-vllm.md` 第 2.3 节**：从"未实现"改为"已实现简化版 swap，与 vLLM 的差异在于 CPU 侧不维护前缀缓存 / 拷贝未做 stream 重叠"。

## 8. 分阶段实现

| 阶段               | 内容                                                    | 说明                               |
| ------------------ | ------------------------------------------------------- | ---------------------------------- |
| **P1（核心）**     | 同步 `copy_` 的 swap out/in + 三队列 + RECOMPUTE 兜底   | 跑通并与重算路径对拍一致，即算成功 |
| **P2（可选优化）** | 独立 CUDA stream + event，让拷贝与计算重叠              | 讲 overlap，贴近生产               |
| **P3（可选延伸）** | 主动分层卸载以"扩容"（不止抢占时触发），接近 KV tiering | 复杂度上一个台阶，作为进阶         |

## 9. 待定决策（含推荐）

1. **范围**：推荐 P1 只做**"抢占时 swap"（reactive）**——最小、最贴合 vLLM SWAP 模式、教学线索最干净；主动卸载（P3）留作延伸。
2. **拷贝方式**：推荐先用**同步 `copy_`**（可读），把 CUDA stream 重叠留到 P2。
3. **CPU 前缀缓存**：推荐 MVP 不在 CPU 侧维护前缀哈希（CPU 块作为不可共享的停车位），swap_in 后重新 hash。
4. **swap-in 优先级**：P1 暂**不**让 swapped-in 优先于 `waiting` 新请求（沿用现有"prefill 阶段先 return"结构，实现最简）。vLLM 会优先把在飞的 swapped 序列拉回以更快回收显存——留作 P2 调优。

---

## 10. 测试方案

> **先说一个坑**：最朴素的"开/关 swap 各跑一遍、对比输出文本"**不可靠**——采样用 Gumbel-Max 随机，两条路径抢占时机不同 → batch 组成不同 → RNG 抽取顺序不同 → 即使 KV 完全正确，输出 token 也可能分叉。因此对拍前必须先消除采样随机性（见 B2）。

分层设计，明确区分**本机（macOS，无 GPU）可跑** 与 **必须上 3090**。

### A. 单元测试（纯 Python，本机可跑，进 CI）

大部分 bug 在元数据状态机里，这层最该厚。落地文件：`test_swap_blockmanager.py`（pytest）。

- **A1. BlockManager swap 元数据**（纯 dict/deque，不碰张量）：
  - `swap_out(seq)` 返回的 `{gpu_id: cpu_id}` 正确；GPU 块回到 `free_block_ids`、CPU 块进 `used_cpu_block_ids`、`swapped_block_tables[seq_id]` 记录正确。
  - `swap_in(seq)` 逆操作：重建 `seq.block_table`（长度/顺序对）、CPU 块释放、`num_cached_tokens` **保持不变**（核心不变量断言）。
  - **往返一致**：swap_out → swap_in 后逻辑块序列等价（物理 id 可不同）。
  - `can_swap_out` 在 CPU 池满时返回 False；`can_swap_in` 在 GPU 空块不足时返回 False。
- **A2. 拷贝索引正确性**：`model_runner.swap_out/in` 的切片 `dst[:,:,c].copy_(src[:,:,g])` 逻辑，**用 CPU 张量**测（两侧 `device="cpu"`），验证 `[:, :, block_id]` 寻址 + 映射正确。把"索引搞反/漏层"挡在本地，不需真 GPU。

### B. 集成测试（必须在 3090 上）

落地文件：`docs/llm-inference-visual/scripts/verify_swap.py`（仿 `verify_nanovllm.py` 风格，argv/env 传模型路径）。

- **B1. KV 逐字节往返相等（最强、完全确定）**：已知随机值填 `kv_cache` 若干块 → `swap_out` 到 CPU → `swap_in` 到**另一批** GPU 块 → 断言 `torch.equal`。直接证明"拷贝 + 映射"无损，与采样无关。
- **B2. 差分对拍（消除随机性后才做）**：测试时**注入 argmax 贪心采样**（monkeypatch sampler），使输出只由 logits 决定、与 batch 顺序无关。三条路径跑同一组 prompt——① 内存充足无抢占（baseline）、② 小内存 `cpu_offload_gb=0`（RECOMPUTE）、③ 小内存 `cpu_offload_gb>0`（SWAP）——断言**逐 token 完全一致**。（生产禁贪心，测试内部用 argmax 仅为拿确定性。）
- **B3. 端到端冒烟 + 不死锁**：极小 `gpu_memory_utilization` + 长 prompt + 高并发，强制大量 swap；断言能跑完、不报错、不死锁、输出连贯。

### C. 可观测性断言（证明"真的走了 swap 路径"）

加计数器 `num_swapped_out_blocks / num_swapped_in_blocks / num_recompute_preemptions`，测试断言 **swap 计数 > 0**——否则内存设太大根本没触发，测试变成空转（silent pass）。

### D. 边界用例

- **CPU 池耗尽 → 回退 RECOMPUTE**：不崩溃、能前进（测兜底路径）。
- **共享前缀块**（`ref_count > 1`）：按 §4.3 规则，含共享块的序列**整体回退 RECOMPUTE**，不做部分 swap → 断言这类序列不进 `swapped` 队列、且共享块不被搬动。
- **chunked prefill 首序列**：只对"已全缓存的 decode 序列"swap，不动正在 chunk-prefill 的序列。

### E. 性能佐证（非断言，报告用）

同压力下 SWAP vs RECOMPUTE，统计**重算的 prefill token 数**与吞吐。swap 的卖点即重算 token↓、吞吐↑，用数据讲出来，呼应 README 的 benchmark 风格。

### 测试矩阵速览

| 层  | 内容                                 | 运行环境  | 确定性来源                |
| --- | ------------------------------------ | --------- | ------------------------- |
| A1  | BlockManager 元数据状态机            | 本机 / CI | 纯逻辑                    |
| A2  | 拷贝索引（CPU 张量）                 | 本机 / CI | 纯逻辑                    |
| B1  | KV 逐字节往返相等                    | 3090      | `torch.equal`，与采样无关 |
| B2  | RECOMPUTE / SWAP / baseline 差分对拍 | 3090      | 注入 argmax               |
| B3  | 端到端冒烟不死锁                     | 3090      | —                         |
| C   | swap 计数 > 0                        | 3090      | —                         |
| D   | 边界（CPU 满 / 共享块 / chunk）      | 3090      | —                         |

---

## 附：改动文件速览

| 文件                               | 改动                                                                |
| ---------------------------------- | ------------------------------------------------------------------- |
| `nanovllm/config.py`               | 加 `cpu_offload_gb`，算 `num_cpu_kvcache_blocks`                    |
| `nanovllm/engine/sequence.py`      | 加 `SWAPPED` 状态；保证 `num_cached_tokens` 在 swap 路径不清零      |
| `nanovllm/engine/block_manager.py` | CPU 块池 + `can_swap_out/swap_out/can_swap_in/swap_in`（仅元数据）  |
| `nanovllm/engine/model_runner.py`  | `cpu_kv_cache` 张量 + `swap_out/swap_in`（张量拷贝）                |
| `nanovllm/engine/scheduler.py`     | `swapped` 队列；swap 感知的 `preempt`；swap-in 阶段；返回 swap 映射 |
| `nanovllm/engine/llm_engine.py`    | `step()` 中在跑模型前执行 swap 拷贝                                 |
