---
theme: default
title: nano-vllm 实战课程
info: |
  ## nano-vllm 实战课程
  从源码走读 LLM 推理引擎：调度、KV cache、注意力、Tensor Parallel、CUDA Graph。

  [GitHub](https://github.com/GeeeekExplorer/nano-vllm)
drawings:
  persist: false
transition: slide-left
mdc: true
overviewSnapshots: false
fonts:
  sans: "Noto Sans SC, Microsoft YaHei, PingFang SC, sans-serif"
  mono: "JetBrains Mono, Fira Code, monospace"
class: text-center
background: /background.svg
---

# nano-vllm 实战课程
## 从源码走读 LLM 推理引擎

调度 · KV Cache · 注意力 · Tensor Parallel · CUDA Graph

<div class="mt-8 grid grid-cols-4 gap-2 text-sm opacity-60 max-w-xl mx-auto">
  <div class="bg-white/5 rounded px-3 py-1">L01 端到端流程</div>
  <div class="bg-white/5 rounded px-3 py-1">L02 Sequence</div>
  <div class="bg-white/5 rounded px-3 py-1">L03 调度器</div>
  <div class="bg-white/5 rounded px-3 py-1">L04 Block 管理</div>
  <div class="bg-white/5 rounded px-3 py-1">L05 Prefill</div>
  <div class="bg-white/5 rounded px-3 py-1">L06 Decode</div>
  <div class="bg-white/5 rounded px-3 py-1">L07 Attention</div>
  <div class="bg-white/5 rounded px-3 py-1">L08 优化全景</div>
</div>

<div class="mt-8 text-sm opacity-50">
约 260 页 · 74 处源码引用 · 26 道自测题
</div>

---
src: ./slides.01.md
---

---
src: ./slides.02.md
---

---
src: ./slides.03.md
---

---
src: ./slides.04.md
---

---
src: ./slides.05.md
---

---
src: ./slides.06.md
---

---
src: ./slides.07.md
---

---
src: ./slides.08.md
---
