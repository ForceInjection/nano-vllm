# nano-vllm 与真实 vLLM 的差距

nano-vllm 约为 1,400 行 Python，真实 vLLM 约为 200,000+ 行。本文标注二者的关键差异，区分"教学简化"和"真正未实现的功能"，帮助读者建立从 nano 到 real 的升级路径。

---

## 1. 教学简化（概念保留，实现精简）

这些功能真实 vLLM 有，nano-vllm 也实现了核心逻辑，但实现方式更直白。

| 功能            | nano-vLLM 做法                             | vLLM 做法                                                                     | nano 里看什么           |
| --------------- | ------------------------------------------ | ----------------------------------------------------------------------------- | ----------------------- |
| PagedAttention  | 固定 block_size=256, 手动管理 free/used 池 | 多 block_size (16/32/64/…), NuBlock 抽象, 支持 prefix caching + copy-on-write | L04 BlockManager        |
| 调度器          | 单一 `Scheduler` 类, FIFO + preempt        | `Scheduler` + `Policy` 解耦, 支持 FCFS/Priority/ChunkedPrefill 策略           | L03 schedule()          |
| Prefix Caching  | xxhash 链式哈希, hash_to_block_id 全局字典 | 相同方案, 外加完整 LRU block eviction + hash 清理                            | L04 hash_blocks()       |
| CUDA Graph      | 预 capture 几个固定 batch_size             | 动态 streaming capture, padding alignment, piecewise CUDAGraph                | L08 capture_cudagraph() |
| Tensor Parallel | spawn + SharedMemory 广播方法调用          | Ray actor 管理 + NCCL group, 支持 pipeline parallel + data parallel           | L08 TP 流程             |
| 采样            | Gumbel-Max + torch.compile                 | 插件式 sampler (top-k/top-p/min-p/beam search/penalties)                      | L08 Sampler.forward     |

---

## 2. 真正缺失的能力

这些功能 nano-vLLM 没有实现，但真实 vLLM 在生产环境中依赖它们。

### 2.1 Prefill / Decode 分离调度

真实 vLLM 将 prefill 和 decode 请求放入两个独立的调度队列，允许在 prefill batch 中插入 decode 请求（chunked prefill 的更一般形式）。

- nano 里：`Scheduler.schedule()` 先 prefill 后 decode，两者互斥
- vLLM 里：prefill 和 decode 在同一 batch 中混合调度，最大化 GPU 利用率
- 影响：长 prompt 请求会阻塞所有 decode 请求

### 2.2 连续批处理 (Continuous Batching)

真实 vLLM 的 decode 请求可以在每个 step 动态加入/离开 batch，不需要等待整个 batch 都完成。nano-vLLM 在 decode 中一旦 seq 加入 batch，就一直停留到 EOS/max_tokens。

- nano 里：decode batch 中的 seq 全部跑完才能释放资源
- vLLM 里：每步重新拼 batch，已完成的 seq 立即释放 slot

### 2.3 KV Cache Offloading

当 GPU 显存不够时，真实 vLLM 可以将不常用的 KV cache block 交换到 CPU 内存或 NVMe 磁盘。

- nano 里：block 不足时只能 preempt（丢弃 KV cache，下一轮 prefill 重算）
- vLLM 里：swap out → swap in，避免重算开销

### 2.4 Speculative Decoding

用草稿模型（draft model）先快速生成几个候选 token，再由主模型验证，一次 decode step 生成多个 token 以提高吞吐。

- 完全不涉及

### 2.5 多模态支持

vLLM 支持图像、视频等多模态输入，nano-vLLM 仅支持文本。

### 2.6 量化

vLLM 支持 AWQ、GPTQ、FP8、INT8 等多种量化方案，减少模型显存占用。nano-vLLM 只使用原始 FP16/BF16 权重。

### 2.7 异步调度与 Streaming

vLLM 的 `LLM.generate` 支持 `async` 模式和 token-level streaming。nano-vLLM 是同步生成。

---

## 3. 架构层面的差异

| 维度       | nano-vLLM        | vLLM                                   |
| ---------- | ---------------- | -------------------------------------- |
| 代码量     | ~1,400 行        | ~200,000+ 行                           |
| 模型支持   | 仅 Qwen3-0.6B    | 所有主流架构 (LLaMA/Qwen/Mistral/…)    |
| 进程管理   | 手动 spawn + shm | Ray / multiprocessing 抽象层           |
| 前后端分离 | 无               | API Server (FastAPI) + Engine 分离     |
| 日志/监控  | 仅 tqdm 进度条   | Prometheus metrics, structured logging |
| 配置系统   | dataclass Config | EngineArgs + 多层配置继承              |
| 测试覆盖   | 无自动化测试     | ~10,000+ 测试用例                      |

---

## 4. 阅读路线建议

学完 nano-vLLM 后，如果希望深入了解真实 vLLM，推荐按以下路线阅读 vLLM 源码：

1. **入口**：`vllm/entrypoints/llm.py` — 对应 nano 的 `nanovllm/llm.py`
2. **引擎**：`vllm/engine/llm_engine.py` — 对应 nano 的 `nanovllm/engine/llm_engine.py`
3. **调度器**：`vllm/core/scheduler.py` — 对应 nano 的 `nanovllm/engine/scheduler.py`（这里差异最大，可重点看）
4. **块管理器**：`vllm/core/block/` — 对应 nano 的 `nanovllm/engine/block_manager.py`
5. **模型执行**：`vllm/worker/model_runner.py` — 对应 nano 的 `nanovllm/engine/model_runner.py`
6. **注意力后端**：`vllm/attention/` — 对应 nano 的 `nanovllm/layers/attention.py`

每个模块 nano 都提供了清晰的起点，vLLM 版本则在其上叠加了生产级特性。带着 nano 的理解去看 vLLM，可以更快穿透抽象层。

---

## 5. 关键代码对应表

nano-vLLM 的层命名基本照搬 vLLM（`VocabParallelEmbedding`、`ColumnParallelLinear`、`RMSNorm`、`SiluAndMul`、`get_rope`、`Sampler` 等同名），因此文件级对应非常清晰。

> 注：vLLM 现有 **V0**（`vllm/engine`、`vllm/core`、`vllm/worker`）和 **V1**（`vllm/v1/…`，当前默认）两套引擎。下表以 V0 路径为主，V1 差异较大处一并标注。vLLM 路径基于近期版本，跨版本可能微调。

### 5.1 核心模块（建议按「学习顺序」列阅读 nano-vLLM 源码）

| 顺序 | 功能模块 | nano-vLLM 源码 | vLLM 源码 | 为什么排这里 |
| ---- | -------- | -------------- | --------- | ------------ |
| 1 | Sequence | `engine/sequence.py` (~84 行) | `vllm/sequence.py`（V1：`vllm/v1/request.py`） | 最小数据结构，后面全依赖它 |
| 2 | Scheduler | `engine/scheduler.py` (~92 行) | `vllm/core/scheduler.py`（V1：`vllm/v1/core/sched/scheduler.py`） | 编排 Sequence 的进出与抢占 |
| 3 | Block Manager | `engine/block_manager.py` (~120 行) | `vllm/core/block_manager.py`（V1：`vllm/v1/core/kv_cache_manager.py`） | 调度背后的 KV 显存分块 |
| 4 | Attention (PagedAttention) | `layers/attention.py` (~75 行，Python/Triton) | `vllm/attention/`（C++/CUDA 后端） | 消费显存块的核心算子；nano-Python vs vLLM-CUDA 对比看 |
| 5 | Model Runner | `engine/model_runner.py` (~257 行) | `vllm/worker/model_runner.py`（V1：`vllm/v1/worker/gpu_model_runner.py`） | 拼 batch → 跑模型 → 采样的执行层 |
| 6 | LLM Engine | `engine/llm_engine.py` (~90 行) | `vllm/engine/llm_engine.py`（V1：`vllm/v1/engine/`） | 顶层主循环，最后合龙 |

**为什么是自底向上：** 后者依赖前者——Scheduler 操作 Sequence（1→2），抢占/分配走 Block Manager（2→3），Attention 消费 Block Manager 分出的块（3→4），Model Runner 调用 Attention（4→5），LLM Engine 只是把上面全部装进 `while not is_finished: step()` 循环（→6）。反过来从主循环读，一路都是黑盒；自底向上则每步都建立在已懂的基础上。

### 5.2 其余对应（层 / 工具）

| nano-vLLM | vLLM 对应 | 说明 |
| --------- | --------- | ---- |
| `llm.py` → `LLM` | `vllm/entrypoints/llm.py` → `LLM` | nano 里 `LLM` 是 `LLMEngine` 别名 |
| `config.py` → `Config` | `vllm/config.py` → `ModelConfig`/`CacheConfig`/`SchedulerConfig`/… | nano 单 dataclass 塞全部配置 |
| `sampling_params.py` → `SamplingParams` | `vllm/sampling_params.py` → `SamplingParams` | nano 只保留 temperature/max_tokens/ignore_eos |
| `layers/linear.py`（各 `*ParallelLinear`） | `vllm/model_executor/layers/linear.py`（同名） | nano 去掉量化分支 |
| `layers/embed_head.py` → `VocabParallelEmbedding`/`ParallelLMHead` | `vllm/model_executor/layers/vocab_parallel_embedding.py`（同名） | TP 词表切分 |
| `layers/layernorm.py` → `RMSNorm` | `vllm/model_executor/layers/layernorm.py:RMSNorm` | vLLM 有 CUDA 融合核 |
| `layers/activation.py` → `SiluAndMul` | `vllm/model_executor/layers/activation.py:SiluAndMul` | 同名 |
| `layers/rotary_embedding.py` → `get_rope` | `vllm/model_executor/layers/rotary_embedding.py`（同名 `get_rope`） | 同名 |
| `layers/sampler.py` → `Sampler` | `vllm/model_executor/layers/sampler.py`（V1：`vllm/v1/sample/sampler.py`） | nano 只做 temperature + Gumbel-Max |
| `models/qwen3.py` → `Qwen3ForCausalLM` | `vllm/model_executor/models/qwen3.py` | 同名同结构 |
| `utils/context.py` → `Context`/`set_context` | `vllm/forward_context.py` → `ForwardContext`/`set_forward_context` | **核心对应点**：全局 forward context 传 attention metadata，避免改 forward 签名 |
| `utils/loader.py` → `load_model`/`default_weight_loader` | `vllm/model_executor/model_loader/`（`loader.py` + `weight_utils.py`） | nano 直接遍历 safetensors |
