# tests

nano-vllm 引擎的测试与验证脚本（与课程无关；课程脚本在 `docs/llm-inference-visual/scripts/`）。

## 文件

| 文件                        | 类型            | 环境                         | 说明                                                                                                   |
| --------------------------- | --------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `test_swap_blockmanager.py` | pytest 单元测试 | CPU / CI，无需 GPU、无需模型 | KV Cache CPU Offloading 的元数据状态机与拷贝索引逻辑                                                   |
| `verify_swap.py`            | GPU 验证脚本    | 需 GPU + 模型                | swap 特性端到端正确性：B1 KV 逐字节往返、B2 单序列 swap 往返逐 token 一致、B3 冒烟不死锁、C 观测计数器 |
| `bench_swap.py`             | GPU 压测脚本    | 需 GPU + 模型                | RECOMPUTE vs SWAP 吞吐 / 并发 / prefill 工作量对比                                                     |

设计与验证方案详见 [`../docs/design/kv-offload.md`](../docs/design/kv-offload.md)。

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
python tests/verify_swap.py /path/to/Qwen3-0.6B/     # 预期 B1/B2/B3/C 全部 [PASS]
python tests/bench_swap.py  /path/to/Qwen3-0.6B/     # 打印 baseline / RECOMPUTE / SWAP 对比表
```

两个脚本各自在独立子进程中运行每个 `LLM`（nano-vllm 的 `exit()` 不释放 KV 显存，同进程连开多个会互相饿死），并用 `num_kvcache_blocks` 上限强制抢占。
