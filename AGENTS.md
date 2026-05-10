# Agents

This document defines how AI agents (and humans using them) should work in this repository, including responsibilities, change boundaries, and a repeatable workflow for code and documentation changes.

## 1. Scope

This section clarifies what “agents” means in this repository and what this document governs.

- “Agent” refers to an automated assistant used to navigate the codebase, implement changes, review diffs, and validate behavior.
- This document focuses on contribution workflow and repository conventions, not on any runtime “agent framework” inside the `nano-vllm` library.

## 2. Repository Context

This section provides the minimum technical context agents need to make correct and consistent changes.

### 2.1 Package And Entry Points

This subsection summarizes the primary package and runnable entry points for manual verification.

- Core library package: `nanovllm/`. Source: repository layout [nanovllm](./nanovllm/).
- Example usage script: `example.py`. Source: [example.py](./example.py).
- Benchmark script: `bench.py`. Source: [bench.py](./bench.py).

### 2.2 Supported Python And Dependencies

This subsection records language and dependency constraints that should influence implementation choices and compatibility decisions.

- Python: `>=3.10,<3.13`. Source: [pyproject.toml](./pyproject.toml#L13).
- Dependencies: `torch>=2.4.0`, `triton>=3.0.0`, `transformers>=4.51.0`, `flash-attn`, `xxhash`. Source: [pyproject.toml](./pyproject.toml#L14-L20).

## 3. Agent Roles

This section defines recommended “roles” so work can be split cleanly and reviewed consistently.

### 3.1 Codebase Navigator

This subsection defines the responsibilities for locating the right code and producing correct pointers for later work.

- Identify the smallest set of files that implement the behavior being changed.
- Summarize data flow boundaries between modules (for example, engine scheduling vs model execution).
- Provide file and line references for all claims.

### 3.2 Implementer

This subsection defines the responsibilities for making changes that are idiomatic to the repository and safe to maintain.

- Follow existing patterns in adjacent modules before introducing new abstractions.
- Prefer minimal diffs that preserve current public APIs unless explicitly changing the API surface.
- Ensure changes do not log prompts, model outputs, or other sensitive user data.

### 3.3 Reviewer

This subsection defines the responsibilities for validating correctness, style, and performance risk.

- Verify that changes are consistent with repository conventions and dependency constraints.
- Check for silent behavior changes (defaults, dtype, device placement, shape assumptions).
- Require a runnable reproduction for bug fixes or a benchmark note for performance changes.

### 3.4 Benchmark Runner

This subsection defines the responsibilities for making performance-related claims reproducible.

- Run `bench.py` when making performance claims. Source: [README.md](./README.md#L46-L62) and [bench.py](./bench.py).
- Record configuration (model, GPU, batch, lengths) and results in the change description, matching the README’s benchmark fields. Source: [README.md](./README.md#L50-L62).

## 4. Working Agreement

This section specifies contribution rules that agents should apply by default.

### 4.1 Minimal And Local Changes

This subsection explains how to keep changes reviewable and reduce unintended regressions.

- Change only the modules required for the requested behavior.
- Avoid “drive-by refactors” unless they are necessary for correctness.
- Keep public API changes explicit and documented.

### 4.2 Safety And Privacy

This subsection defines non-negotiable safety constraints for changes and debugging output.

- Do not print, log, or persist prompts, completions, tokens, or model weights paths unless explicitly required for a user-facing feature.
- Do not add telemetry, network calls, or background upload behavior.
- Avoid introducing non-determinism unless it is an intentional and documented trade-off.

### 4.3 Documentation Rules For This Repository

This subsection defines documentation conventions agents should follow when adding or editing Markdown files.

- Use Markdown syntax (avoid HTML tags) for formatting.
- Keep headings numbered by section level; keep the top-level document title unnumbered.
- Ensure code blocks include an in-block comment that explains what the snippet does.

### 4.4 Visual Course Doc Conventions

This subsection defines conventions that apply specifically to the tutorial under [docs/llm-inference-visual/](./docs/llm-inference-visual/). Keep reader-facing READMEs lean and record the authoring rules here.

- Audience and language: lessons are written in Chinese for CS undergraduates whose only prerequisite is Python. LLM-specific concepts (Transformer, attention, KV cache) must be introduced where they are first needed rather than assumed.
- Narrative voice: use first-person plural (我们) for shared derivations, outcomes, and interpretive framings; use third-person (读者) for prior-knowledge conditionals and ToC signposting (for example “如果读者已经了解…可以跳到…”). Avoid second-person 你/您 outside literal example strings such as tokenization inputs (`"你好"`).
- Lesson structure: each `Lxx-*.md` follows a canonical four-section layout — §1 本课概述 (with 1.1 课时安排 and 1.2 学习目标), §2 原理铺垫 / 原理说明, §3 代码走读, §4 练习. Keep section numbering consistent so cross-lesson references remain stable.
- OS analogies: when a mechanism mirrors an operating-systems concept (paging, reference counting, shared read-only pages, spawn + shared-memory IPC), name the analogy explicitly so the target audience can anchor on prior coursework. Keep the analogy a one-line aside, not a substitute for the code walkthrough.
- Diagram assets live under `docs/llm-inference-visual/diagrams/`. Every diagram keeps both the draw.io source (`Lxx-*.drawio`) and the exported preview (`Lxx-*.png`); the `Lxx` prefix must match the lesson number.
- Diagram labels should mirror code field names (for example `waiting/running/block_table/slot_mapping/block_tables`), and each diagram should carry a corner note with the source file basename so readers can jump back to the implementation.
- Code references use three tiers, chosen per context:
  - **Inline link (Tier 1)**: link on the first mention of a symbol when one file / line range is enough.
  - **Bullet list (Tier 2)**: use a dedicated bullet list when a section references ≥2 code locations or anchors need to sit parallel to data-flow / behavior bullets.
  - **Embedded snippet (Tier 3)**: when a small function, control-flow branch, or subtle one-liner drives the conclusion, copy ≤ ~17 lines verbatim from source and embed in the prose. The first line must be an in-block comment pointing out what to observe; keep the preceding inline anchor — the snippet is a zoom-in, not a replacement.
- Canonical example of all three tiers together: [01-llm-generate-and-step.md](./docs/llm-inference-visual/01-llm-generate-and-step.md) §3.

## 5. Standard Workflows

This section provides repeatable steps that agents should follow for common types of work.

### 5.1 Setup And Smoke Check

This subsection provides a minimal workflow to ensure the environment can run the core example entry point.

```bash
# Install the package from the repository checkout (editable install).
python -m pip install -e .

# Run the example script (requires local model weights; see README for download).
python example.py
```

Sources for example usage and model download guidance: [example.py](./example.py) and [README.md](./README.md#L25-L44).

### 5.2 Bug Fix Workflow

This subsection defines the minimal expected artifacts for a bug fix.

- Add a reproduction script or minimal failing snippet.
- Apply a targeted fix and avoid unrelated formatting changes.
- Re-run the reproduction to confirm the fix.
- If the bug affects generation, validate using the public `LLM.generate` API. Source: [README.md §Quick Start](./README.md#L35-L46) and [nanovllm/llm.py](./nanovllm/llm.py).

### 5.3 Performance Change Workflow

This subsection defines what is required before claiming a performance improvement or regression.

- Describe the performance hypothesis (what you changed and why it should help).
- Run the benchmark and capture configuration and results.
- State whether any accuracy, determinism, or memory trade-offs changed.

Benchmark entry point: [bench.py](./bench.py). Benchmark configuration fields reference: [README.md](./README.md#L50-L62).

## 6. Module Map

This section summarizes where to make changes for common concerns, so agents can find the right place quickly.

| Area                  | Primary Paths                                                                                      | Notes                                                                                                             |
| --------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Public API            | `nanovllm/llm.py`, `nanovllm/sampling_params.py`, `nanovllm/__init__.py`                           | User-facing entry points and parameter definitions. Source: repository layout [nanovllm](./nanovllm/).            |
| Engine                | `nanovllm/engine/llm_engine.py`, `nanovllm/engine/scheduler.py`, `nanovllm/engine/model_runner.py` | Request scheduling, execution loop, and model invocation. Source: repository layout [engine](./nanovllm/engine/). |
| Attention And Kernels | `nanovllm/layers/attention.py`, `nanovllm/layers/rotary_embedding.py`                              | Performance-sensitive components. Source: repository layout [layers](./nanovllm/layers/).                         |
| Model Definition      | `nanovllm/models/qwen3.py`                                                                         | Architecture wiring for Qwen3. Source: repository layout [models](./nanovllm/models/).                            |
| Utilities             | `nanovllm/utils/loader.py`, `nanovllm/utils/context.py`                                            | Model loading and runtime context. Source: repository layout [utils](./nanovllm/utils/).                          |

## 7. Agent Output Templates

This section provides copy-paste templates so agent-generated work is consistent and reviewable.

### 7.1 Change Summary Template

This subsection defines a standard structure for describing changes in issues, PRs, or review notes.

- Goal:
- Scope:
- Non-goals:
- Files changed:
- Verification:
- Risks and mitigations:
- Benchmarks (if applicable):

### 7.2 Reproduction Script Template

This subsection provides a minimal structure for a self-contained bug reproduction script.

```python
# Reproduction template: replace MODEL_PATH and prompts to demonstrate the bug.
import os
from nanovllm import LLM, SamplingParams


def main() -> None:
    model_path = os.path.expanduser("MODEL_PATH")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.0, max_tokens=32)

    # This prompt should trigger the buggy behavior in a deterministic way.
    outputs = llm.generate(["REPLACE_ME"], params)
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
```
