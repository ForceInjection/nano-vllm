# SamplingParams is dependency-light and always eagerly available.
# LLM pulls the model/GPU stack (torch, flash_attn), so it is imported lazily on first access —
# this lets lightweight submodules (e.g. nanovllm.engine.block_manager) be imported for unit
# tests on machines without flash_attn, while `from nanovllm import LLM` keeps working.
from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
