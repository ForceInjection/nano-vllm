# 基于 nano-vllm 的 LLM 推理可视化教程

## 1. 教程概述

本教程面向大学本科计算机专业同学，只需要具备 Python 编程基础。Transformer 与注意力机制等 LLM 原理知识将在课中逐步引入，无需提前掌握。目标是基于 nano-vllm 的真实代码路径，建立对 LLM 离线推理引擎的整体认识：从 `LLM.generate` 的入口出发，理解调度（prefill/decode）、KV cache 的 block 管理、注意力算子分支、Tensor Parallel 与 CUDA Graph 等关键机制，并用可视化图将程序逻辑、数据结构和张量（多维数组）的形状呈现出来。

### 1.1 代码与事实来源

本教程所有关键结论均以仓库代码为依据。为了兼顾阅读流畅度与可核验性，代码引用按三种形式分层呈现：

- **内联链接**：在行文首次提到某个符号时直接点击跳转，适合单一目标（例如推理主循环入口 [llm_engine.py:L49-L90](../../nanovllm/engine/llm_engine.py#L49-L90)）。
- **要点列表**：当一段逻辑涉及 ≥2 个代码位置（多文件、多行号段）时，用要点列表并列呈现。
- **嵌入片段**：对于控制流密集、一眼即懂胜过千字的小函数或关键分支，直接从源码逐字摘录一段（≤ ~17 行）嵌入正文，并以行内注释点明要观察的点。

具体使用规范与图资产组织见 [AGENTS.md](../../AGENTS.md) §4.4。

### 1.2 阅读前置

本教程对前置知识的要求很低，核心 LLM 原理会在对应课次中逐步引入：

- Python 基础：能读懂函数定义、列表操作、字典结构
- 基本数学：矩阵乘法的直觉（不需要手推公式）
- 好奇心：想知道 LLM 生成文字时底层到底发生了什么
- 不要求提前理解 FlashAttention/Triton 的实现细节（会在第 7 课以"接口与分支"为主解释）

### 1.3 原理知识分布

下表说明 LLM 原理知识在哪一课引入，帮助你提前了解知识点的位置：

| 原理主题                        | 所在课次 | 关联的工程概念                        |
| ------------------------------- | -------- | ------------------------------------- |
| Transformer 整体架构            | 第 1 课  | 理解 `LLM` 对象是什么                 |
| Tokenizer：文本 → token_ids     | 第 1 课  | 理解 `add_request` 为什么先 tokenize  |
| 自回归生成：为什么逐 token 生成 | 第 1 课  | 理解 step 循环存在的原因              |
| Self-Attention 直觉             | 第 5 课  | 理解为什么 prefill 要做变长注意力     |
| KV Cache 的数学动机             | 第 7 课  | 理解为什么 decode 只需处理 1 个 token |

### 1.4 课时安排说明

每课设计为 **90 分钟**课堂使用，包含四个阶段：

1. **原理铺垫**（或概念回顾）：建立本课所需的 LLM 原理直觉或回顾前课要点
2. **代码走读**：沿真实代码路径，逐函数讲解控制流与数据结构，必要时在正文嵌入源码片段
3. **动手练习**：完成课内"最小练习"，用代码或手算验证理解
4. **答疑讨论**：开放提问与延伸讨论

具体时间分配见各课首部的"课时安排"表。第 1 课原理内容最重；第 2–7 课以代码走读为主；第 8 课偏讨论与综合。

---

## 2. 课程目录

课程按推理链路从外到内展开。每课均包含：本章概述（含课时安排与学习目标）、原理铺垫（如适用）、关键代码锚点与嵌入片段、可视化图、最小练习与验收要点。

1. [第 1 课：从 LLM.generate 走到 step 循环](./01-llm-generate-and-step.md)
2. [第 2 课：Sequence 数据结构与请求生命周期](./02-sequence-lifecycle.md)
3. [第 3 课：Scheduler 的队列、chunked prefill 与 preempt](./03-scheduler-queues-and-preempt.md)
4. [第 4 课：BlockManager 与 prefix caching](./04-block-manager-and-prefix-cache.md)
5. [第 5 课：prefill 批构建与 context 注入](./05-prefill-batching-and-context.md)
6. [第 6 课：decode 一步生成与 block_tables](./06-decode-and-block-tables.md)
7. [第 7 课：Attention：KV 写入与算子分支](./07-attention-kv-cache-and-branches.md)
8. [第 8 课：常见优化的"位置感"（TP、CUDA Graph、torch.compile）](./08-where-optimizations-live.md)

---

## 3. 运行与验证建议

以下是最小的本地运行建议，用于把"代码阅读"与"实际运行现象"对齐。nano-vllm 运行需要本地模型权重路径，仓库提供了示例用法。

```bash
# 运行说明：安装本仓库并执行示例脚本（需要本地模型权重）。
python -m pip install -e .
python example.py
```

- 示例用法来源：[README.md:L34-L44](../../README.md#L34-L44) 与 [example.py](../../example.py)
