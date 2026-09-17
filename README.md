<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM

从零构建的轻量级 vLLM 实现。

## 关键特性

- 🚀 **快速离线推理** - 推理速度与 vLLM 相当
- 📖 **可读性强的代码** - 约 1,400 行 Python 代码，结构清晰
- ⚡ **优化套件** - 前缀缓存、张量并行、Torch 编译、CUDA Graph 等

## 实战课程

[nano-vllm 实战课程](docs/llm-inference-visual/) — 从源码走读 LLM 推理引擎：调度、KV cache、注意力、Tensor Parallel、CUDA Graph。8 课逐步展开，每课附带可运行的验证脚本。

## 安装

```bash
pip install git+https://github.com/ForceInjection/nano-vllm.git
```

## 模型下载

手动下载模型权重：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 快速开始

参考 `example.py`。API 接口与 vLLM 一致，仅 `LLM.generate` 方法有细微差异：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 性能测试

参见 `bench.py`。

**测试配置：**

- 硬件：RTX 4070 Laptop (8GB)
- 模型：Qwen3-0.6B
- 总请求数：256 条
- 输入长度：100–1024 tokens 随机采样
- 输出长度：100–1024 tokens 随机采样

**性能结果：**

| 推理引擎  | 输出 Tokens | 耗时 (s) | 吞吐 (tokens/s) |
| --------- | ----------- | -------- | --------------- |
| vLLM      | 133,966     | 98.37    | 1361.84         |
| Nano-vLLM | 133,966     | 93.41    | 1434.13         |

## 开发：验证与调试

### 验证分层

| 层级     | 命令                                                                         | 依赖                                        |
| -------- | ---------------------------------------------------------------------------- | ------------------------------------------- |
| 单元测试 | `python -m pytest tests/ -v`                                                 | 无需 GPU / 模型（纯元数据状态机与拷贝索引） |
| 功能验证 | `python docs/llm-inference-visual/scripts/verify_nanovllm.py <model>`        | GPU + 模型                                  |
| 端到端   | `python tests/verify_swap.py <model>` / `python tests/bench_swap.py <model>` | GPU + 模型                                  |

`tests/verify_swap.py` 覆盖 KV Cache CPU Offloading 的 GPU 层：B1 KV 逐字节往返相等、B2 单序列 swap 往返逐 token 一致（确定性对比）、B3 强制抢占冒烟（不死锁）、C 观测计数器断言。脚本内部每个 `LLM` 都在独立子进程中运行（引擎的 `exit()` 不释放 KV 显存，同进程连开会互相饿死），设计见 [docs/design/kv-offload.md](docs/design/kv-offload.md)。

### flash-attn 安装（常见坑）

- 预编译 wheel 按 **torch 小版本 + CUDA + cxx11 ABI** 三要素绑定（如 `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-...whl`）。不匹配的轮子能装上、import 时才报 `undefined symbol: _ZN3c10...`——下载前先核对 `torch.__version__` 与 CUDA 版本
- 没有匹配轮子时源码编译（需要 nvcc，A100 填 `8.0`，Hopper 填 `9.0`）：

```bash
pip install ninja packaging wheel
# 从 PyPI 下载 flash_attn-<ver>.tar.gz 并解包，进入目录后：
MAX_JOBS=48 TORCH_CUDA_ARCH_LIST="8.0" python setup.py bdist_wheel -d dist
pip install --no-deps dist/flash_attn-*.whl
```

- 不要用 `pip download` / `pip wheel` 处理 flash-attn 的 sdist：PEP 517 隔离构建环境里没有 torch，会报 `No module named 'torch'`
- wheel 文件名必须符合 pip 规范（`名-版本-语言标记-ABI标记-平台.whl`），随意重命名（如 `fa2.whl`）会被拒绝安装
- Python 仅支持 3.10–3.12（`pyproject.toml` 限定 `>=3.10,<3.13`，3.13 不可用）

### 容器化 GPU 验证

用任意含 torch + CUDA 工具链的镜像（如 `pytorch/pytorch` devel 系）搭建即弃验证环境，宿主机零污染：

```bash
docker run --rm --gpus all --ipc=host \
  -v $PWD:/nano-vllm -v <模型目录>:/models \
  <torch镜像> \
  bash -c "pip install -q --no-deps <flash_attn轮子> \
  && cd /nano-vllm && pip install -e . --no-build-isolation --no-deps \
  && python tests/verify_swap.py /models/Qwen3-0.6B"
```

- `--rm` 即弃：pip 装进容器层的依赖随容器销毁，镜像本体不变；`--no-deps` 保证不升级镜像里已有的 torch
- `--ipc=host`：CUDA 与多进程共享内存需要足够的 shm
- `--gpus` 不可用的环境（如驱动升级后 CDI 规格与 `--gpus` 挂载不一致）可改用 CDI 设备：`--device nvidia.com/gpu=N`，可用设备名以 `nvidia-ctk cdi list` 为准
- pip 走镜像源（如 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）可显著提速
