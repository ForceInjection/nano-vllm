# tests

nano-vllm 引擎的测试与验证脚本（与课程无关；课程脚本在 `docs/llm-inference-visual/scripts/`）。

## 文件

| 文件                        | 类型            | 环境                         | 说明                                                                                                   |
| --------------------------- | --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `test_sequence.py`          | pytest 单元测试 | CPU / CI，无需 GPU、无需模型 | Sequence 的 block 切分、计数器、pickle 协议（TP 序列化）                                                |
| `test_block_manager.py`     | pytest 单元测试 | CPU / CI，无需 GPU、无需模型 | BlockManager 核心语义：前缀缓存命中/共享、引用计数回收、哈希只登记完整块、陈旧登记惰性清理、追加边界     |
| `test_scheduler.py`         | pytest 单元测试 | CPU / CI，无需 GPU、无需模型 | Scheduler 编排：批拼接预算、chunked prefill 限制、swap-in 阶段、preempt 双路径全流程、postprocess 收尾   |
| `test_swap_blockmanager.py` | pytest 单元测试 | CPU / CI，无需 GPU、无需模型 | KV Cache CPU Offloading 的元数据状态机与拷贝索引逻辑                                                   |
| `test_engine.py`            | pytest 单元测试 | CPU / CI，无需 GPU、无需模型 | LLMEngine 不变量：`exit()` 幂等（含 teardown 抛异常路径）、`step()` 的 swap 拷贝顺序（设计约束 R1）       |
| `verify_swap.py`            | GPU 验证脚本    | 需 GPU + 模型                | swap 特性端到端正确性：B1 KV 逐字节往返、B2 单序列 swap 往返逐 token 一致、B3 冒烟不死锁、B4 TP=2、C 观测计数器 |
| `bench_swap.py`             | GPU 压测脚本    | 需 GPU + 模型                | RECOMPUTE vs SWAP 吞吐 / 并发 / prefill 工作量对比                                                     |

设计与验证方案详见 [`../docs/design/kv-offload.md`](../docs/design/kv-offload.md)。

单元测试按模块分层：`sequence` → `block_manager` → `scheduler` → `engine`，每层只测自身职责（swap 的元数据状态机归 `block_manager` 层，调度编排归 `scheduler` 层）。`test_engine.py` 在导入前对 `torch` / `model_runner` 做最小桩化，使 LLMEngine 也能在无 GPU 环境被覆盖。

## 运行

### 单元测试（本机可跑，无需 GPU）

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

`pytest` 仅收集 `test_*.py`，因此 `verify_swap.py` / `bench_swap.py` 不会被自动收集，需手动运行（见下）。单元测试是 GPU-free 的——`nanovllm/__init__.py` 惰性导入 `LLM`（PEP 562），使 `nanovllm.engine.block_manager` 等轻量子模块无需 `flash_attn` 即可导入。

### GPU 验证 / 压测（需 CUDA + 模型）

模型路径经命令行参数或 `NANOVLLM_MODEL_PATH` 环境变量传入：

```bash
python tests/verify_swap.py /path/to/Qwen3-0.6B/     # 预期 B1/B2/B3/C 全部 [PASS]；有 2 张卡时含 B4(TP=2)
python tests/bench_swap.py  /path/to/Qwen3-0.6B/     # 打印 baseline / RECOMPUTE / SWAP 对比表
```

两个脚本各自在独立子进程中运行每个 `LLM`（nano-vllm 的 `exit()` 不释放 KV 显存，同进程连开多个会互相饿死），并用 `num_kvcache_blocks` 上限强制抢占。B4 需要至少 2 张空闲 GPU（TP=2 时 swap 拷贝经共享内存 RPC 广播到各 rank，各 rank 迁移自己的 KV 分片），不足时自动跳过。
