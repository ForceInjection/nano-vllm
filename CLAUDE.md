# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable) — flash-attn may need a pre-built wheel from GitHub releases
pip install -e .

# Run example (requires Qwen3-0.6B weights)
python example.py

# Run benchmark (256 seqs, 1024/1024 tokens, CUDA Graph on)
python bench.py

# Download model weights (one-time, via modelscope mirror in China)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen3-0.6B', cache_dir='./Qwen3-0.6B/')"
```

**Course exercise scripts** under `docs/llm-inference-visual/scripts/`:

```bash
cd docs/llm-inference-visual/scripts/

# All (GPU needed for L01)
bash run_all.sh --all

# CPU-only (L02-L08)
bash run_all.sh

# Single lesson — model path via argv or NANOVLLM_MODEL_PATH env var
python L03_scheduler.py /path/to/model

# Functional verification (6 test cases)
python verify_nanovllm.py /path/to/model

# Standalone benchmark with argparse
python benchmark.py /path/to/model --num-seqs 64 --max-input 512 --max-output 256
```

No linter or type-checker is configured. Unit tests live in `tests/` (pytest, opt-in dep): `pip install -e ".[test]"` then `python -m pytest tests/ -v` (single test: `python -m pytest tests/test_swap_blockmanager.py -v`). These are GPU-free — `nanovllm/__init__.py` imports `LLM` lazily (PEP 562) so lightweight submodules (e.g. `nanovllm.engine.block_manager`) import without `flash_attn`. They are layered by module: `test_sequence.py` → `test_block_manager.py` → `test_scheduler.py` → `test_engine.py`, plus swap-specific `test_swap_blockmanager.py`; `test_engine.py` stubs `torch`/`model_runner` at import time so `LLMEngine` (incl. the `exit()` latch and the `step()` swap-copy ordering, design R1) is covered without a GPU. GPU-level correctness/benchmark scripts for the engine also live in `tests/` (`verify_swap.py`, `bench_swap.py`; not pytest-collected, run manually with a model path). Each of those scripts runs every `LLM` in its own subprocess — `engine.exit()` doesn't free KV VRAM, so multiple engines in one process starve each other; `verify_swap.py` B4 additionally needs ≥2 GPUs (TP=2 swap broadcast) and skips otherwise. Course-provided verify/benchmark tools remain under `docs/llm-inference-visual/scripts/` (`verify_nanovllm.py`, `benchmark.py`).

The full compile/run/debug workflow for GPU verification (remote GPU box, containerized runs, source-compiled FA2 wheel, troubleshooting table) is captured in a **local-only (gitignored) project skill** at `.claude/skills/gpu-verify/SKILL.md` — if present, read it before attempting any GPU-level verification, benchmarking, or CUDA-extension compilation. It contains intranet addresses and must never be committed; the shareable, environment-agnostic version of this workflow lives in [README.md § 开发：验证与调试](README.md#开发验证与调试).

## Architecture

**`LLM` (nanovllm/llm.py) is just a class alias for `LLMEngine`** — the entire public API surface is `LLM.generate(prompts, sampling_params)`, which tokenizes inputs, loops `step()` until all sequences finish, and detokenizes outputs. Returns `list[dict]` where each dict has `{"text": str, "token_ids": list[int]}`.

### Request lifecycle

1. `LLMEngine.add_request()` tokenizes and wraps the prompt into a `Sequence`, enqueues it in `Scheduler.waiting`.
2. `Scheduler.schedule()` moves sequences through three phases (a step is either a prefill step or a decode step, never both):
   - **Prefill**: takes sequences from `waiting`, allocates KV-cache blocks (with prefix-cache reuse), schedules as many tokens as possible within `max_num_batched_tokens`. First sequence may be chunk-prefilled. If anything was scheduled here, the step returns immediately.
   - **Swap-in** (decode steps only): moves sequences from the third queue `swapped` back to `running` while GPU blocks are free — essential for liveness when every running seq was swapped out.
   - **Decode**: pops from `running` in FIFO order, allocates one new block per sequence if needed. When out of blocks it preempts (see SWAP/RECOMPUTE below). After scheduling, runs `extendleft(reversed(scheduled_seqs))` to put them back at the left of the deque — this means decoding sequences keep priority over newly started ones, creating a simple round-robin within the decoding batch.
3. `ModelRunner.run()` prepares input tensors differently for prefill vs decode, runs the model, and samples next tokens.
4. `Scheduler.postprocess()` writes token outputs back to sequences, hashes completed blocks for prefix caching, and transitions finished sequences to `FINISHED`.

### Key design decisions

- **Thread-local `Context`** (nanovllm/utils/context.py): Scheduling metadata (slot_mapping, block_tables, context_lens, cu_seqlens) is passed to attention layers via a module-level global rather than threading through model forward signatures. This avoids changing the standard Transformer forward interface.
- **Tensor parallelism via multiprocessing**: TP workers are separate processes (spawn context) communicating through NCCL for tensors and `SharedMemory` + `Event` for control. Rank 0 writes method name + pickled args to shared memory; ranks > 0 poll and execute. This is why `Sequence.__getstate__`/`__setstate__` is pickling-aware (only transmits essential fields).
- **`ModelRunner.__init__` ordering matters**: warmup model (dummy prefill) → allocate KV-cache (computed from GPU memory stats: `int(total * gpu_memory_utilization - used - peak + current) // block_bytes`) → capture CUDA graph → setup TP shared-memory IPC. During init, `torch.set_default_device("cuda")` and `torch.set_default_dtype(hf_config.dtype)` are set so all tensor creation defaults to CUDA; restored to CPU/float32 afterward.
- **CUDA graph** captured for decode at batch sizes `[1, 2, 4, 8] + range(16, max_bs+1, 16)` (max_bs = min(`max_num_seqs`, 512)), sharing one memory pool. Only used when `enforce_eager=False` and `input_ids.size(0) <= 512`. The graph captures the model forward pass; logits projection (lm_head) and sampling run outside the graph so per-request temperature can be applied.
- **Prefix caching** in `BlockManager`: each filled block is hashed (xxhash) with its prefix hash as a seed, creating a content-addressed lookup. During prefill, `can_allocate()` walks blocks checking hash matches and does reference counting for shared blocks.
- **KV-cache CPU offloading (swap-based preemption)**: when a decode step runs out of GPU blocks, `preempt()` prefers SWAP — copy the seq's KV to a CPU parking lot and send it to `scheduler.swapped` to resume as decode later — falling back to RECOMPUTE (drop KV, back to `waiting`, re-prefill) when CPU is full or the seq holds shared prefix-cache blocks (`can_swap_out` requires every block `ref_count == 1`). The CPU side is metadata + one pinned `cpu_kv_cache` tensor in `ModelRunner` (no hashing / prefix cache on CPU). Ordering is mandatory in `LLMEngine.step()`: swap_out → swap_in → run (design doc R1) — a seq swapped in earlier in the same step has garbage in its GPU blocks until the copy runs, so preempting it cancels the pending swap-in and recomputes instead. Design + verification record: `docs/design/kv-offload.md`; observability counters on the scheduler are asserted by `tests/verify_swap.py`.

### Module map

| Area                | Paths                                                                                                                                                                                                                              | Notes                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Public API          | `nanovllm/llm.py`, `nanovllm/sampling_params.py`, `nanovllm/__init__.py`                                                                                                                                                           | `LLM` = `LLMEngine`; `SamplingParams`                                                 |
| Config              | `nanovllm/config.py`                                                                                                                                                                                                               | `Config` dataclass — model path, block sizes, TP, eager; validated in `__post_init__` |
| Engine (scheduling) | `nanovllm/engine/llm_engine.py`, `nanovllm/engine/scheduler.py`, `nanovllm/engine/sequence.py`                                                                                                                                     | Request lifecycle, prefill/decode loop                                                |
| Engine (execution)  | `nanovllm/engine/model_runner.py`, `nanovllm/engine/block_manager.py`, `nanovllm/engine/kv_swap.py`                                                                                                                                | Model invocation, KV-cache allocation, GPU↔CPU block copy                             |
| Model               | `nanovllm/models/qwen3.py`                                                                                                                                                                                                         | Only Qwen3 supported; TP-aware from construction                                      |
| Layers              | `nanovllm/layers/attention.py`, `nanovllm/layers/linear.py`, `nanovllm/layers/embed_head.py`, `nanovllm/layers/rotary_embedding.py`, `nanovllm/layers/sampler.py`, `nanovllm/layers/layernorm.py`, `nanovllm/layers/activation.py` | FlashAttention, TP linear sharding, RoPE, sampling, RMSNorm, SiLU                     |
| Utilities           | `nanovllm/utils/loader.py`, `nanovllm/utils/context.py`                                                                                                                                                                            | SafeTensors loading, thread-local context                                             |

### Dependencies and constraints

- Python `>=3.10,<3.13`, torch `>=2.4.0`, triton `>=3.0.0`, transformers `>=4.51.0`, `flash-attn`, `xxhash`
- Default `Config`: `max_num_batched_tokens=16384`, `max_num_seqs=512`, `max_model_len=4096` (clamped to `hf_config.max_position_embeddings`), `gpu_memory_utilization=0.9`, `kvcache_block_size=256`, `cpu_offload_gb=0` (opt-in swap; >0 makes `ModelRunner.allocate_kv_cache` derive a pinned CPU mirror and overwrite `config.num_cpu_kvcache_blocks` in place). `num_kvcache_blocks=-1` means auto-size from free VRAM; set it >0 to cap GPU KV capacity (tests use this to force preemption). The model path must be a **local directory** (`assert os.path.isdir(self.model)`) — there is no HF-hub download at runtime; fetch weights beforehand (see Commands).
- KV-cache block size must be a multiple of 256 (`kvcache_block_size % 256 == 0`)
- Greedy sampling (temperature ≤ 1e-10) is explicitly rejected — minimum is slightly above zero
- Only Qwen3-0.6B model architecture is implemented
- `flash-attn` may fail to `pip install` from source (CPU/memory heavy). Download the pre-built wheel from GitHub releases matching the torch+CUDA version (e.g. `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`). Use `ghproxy.net` if GitHub is unreachable.
- Course exercise scripts (`docs/llm-inference-visual/scripts/`) are self-contained but share a common `show_source()`/`show_code_block()` helper for displaying nano-vllm source snippets inline. Scripts that need the model accept `sys.argv[1]` or the `NANOVLLM_MODEL_PATH` env var.
- `docs/llm-inference-visual/slides/` is a Node/Reveal.js slide SPA (served via `serve_spa.py`) and ships a committed `node_modules/`. Exclude it from code searches — it contains thousands of vendored `.py`/`.js` files unrelated to nano-vllm.

## Working conventions

- Change only the modules required for the requested behavior; avoid drive-by refactors unless they are necessary for correctness. Keep public API changes explicit and documented.
- Never log or persist prompts, completions, tokens, or model-weight paths unless a user-facing feature requires it; do not add telemetry, network calls, or background uploads.
- Avoid introducing non-determinism unless it is an intentional and documented trade-off.
- **Bug fixes**: add a runnable reproduction (script or minimal failing snippet) first, apply a targeted fix, then re-run the reproduction to confirm. If the bug affects generation, validate through the public `LLM.generate` API.
- **Performance changes**: state the hypothesis, run `bench.py` and record configuration + results (matching the README benchmark fields), and state any accuracy, determinism, or memory trade-offs.

## Documentation conventions

- Use Markdown (avoid HTML tags) when adding or editing docs. Keep headings numbered by section level, with the top-level title unnumbered. Every code block carries an in-block comment explaining what the snippet does.

### Visual course (`docs/llm-inference-visual/`)

- **Audience and language**: lessons are written in Chinese for CS undergraduates whose only prerequisite is Python. LLM-specific concepts (Transformer, attention, KV cache) must be introduced where they are first needed rather than assumed.
- **Narrative voice**: first-person plural (我们) for shared derivations, outcomes, and interpretive framings; third-person (读者) for prior-knowledge conditionals and ToC signposting (e.g. “如果读者已经了解…可以跳到…”). Avoid second-person 你/您 outside literal example strings such as tokenization inputs (`"你好"`).
- **Lesson structure**: each `Lxx-*.md` follows the canonical four-section layout — §1 本课概述 (with 1.1 课时安排 and 1.2 学习目标), §2 原理铺垫 / 原理说明, §3 代码走读, §4 练习. Keep section numbering consistent so cross-lesson references remain stable.
- **OS analogies**: when a mechanism mirrors an operating-systems concept (paging, reference counting, shared read-only pages, spawn + shared-memory IPC), name the analogy explicitly so the audience can anchor on prior coursework — as a one-line aside, not a substitute for the code walkthrough.
- **Diagram assets** live under `docs/llm-inference-visual/diagrams/`, each keeping both the draw.io source (`Lxx-*.drawio`) and the exported preview (`Lxx-*.png`); the `Lxx` prefix must match the lesson number. Labels should mirror code field names (`waiting/running/block_table/slot_mapping/block_tables`), and each diagram carries a corner note with the source file basename so readers can jump back to the implementation.
- **Code references** use three tiers, chosen per context:
  - **Inline link (Tier 1)**: link on the first mention of a symbol when one file / line range is enough.
  - **Bullet list (Tier 2)**: a dedicated bullet list when a section references ≥2 code locations, or anchors need to sit parallel to data-flow / behavior bullets.
  - **Embedded snippet (Tier 3)**: when a small function, control-flow branch, or subtle one-liner drives the conclusion, copy ≤ ~17 lines verbatim from source into the prose. The first line must be an in-block comment pointing out what to observe; keep the preceding inline anchor — the snippet is a zoom-in, not a replacement.
- Canonical example of all three tiers together: `01-llm-generate-and-step.md` §3.
