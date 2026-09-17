"""LLMEngine 不变量单测：exit() 幂等（含 teardown 抛异常路径）、step() 的 swap 拷贝顺序。

llm_engine 会导入重依赖（torch / flash_attn / model_runner），这里在导入前做最小桩化，
使其在无 GPU、无 torch 的环境（CI）也能被覆盖。桩只在本次导入期间生效，导入后立即清理，
不影响同进程中的其他测试文件。
"""
import importlib.machinery
import importlib.util
import sys
import types

import pytest

import transformers  # 先真实导入：transformers 据此缓存 torch 可用性，之后再桩 torch 也不会影响它


def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


_stubbed = []
if importlib.util.find_spec("torch") is None:          # CI / 无 torch 环境才需要桩 torch
    _torch = _fake_module("torch")
    _fake_module("torch.multiprocessing", get_context=lambda *a, **k: None)
    _torch.multiprocessing = sys.modules["torch.multiprocessing"]
    _stubbed += ["torch", "torch.multiprocessing"]
_fake_module("nanovllm.engine.model_runner", ModelRunner=object)   # 该模块依赖 flash_attn
_stubbed.append("nanovllm.engine.model_runner")

from nanovllm.engine.llm_engine import LLMEngine       # noqa: E402
from nanovllm.engine.sequence import Sequence          # noqa: E402

for _name in _stubbed:                                 # 桩只服务于上面这次导入
    sys.modules.pop(_name, None)
del _stubbed


class RecordingRunner:
    """记录 call() 顺序的假 ModelRunner（真正的 ModelRunner 需要 GPU）。"""

    def __init__(self, order=None, fail_exit=False):
        self.calls = []
        self.order = order if order is not None else []
        self.fail_exit = fail_exit

    def call(self, method, *args):
        self.calls.append(method)
        self.order.append(method)
        if method == "exit" and self.fail_exit:
            raise RuntimeError("teardown boom")
        return [7] * len(args[0]) if method == "run" else None


class FakeScheduler:
    def __init__(self, seqs, is_prefill, swap_out=None, swap_in=None, order=None):
        self._seqs, self._is_prefill = seqs, is_prefill
        self.blocks_to_swap_out = swap_out or {}
        self.blocks_to_swap_in = swap_in or {}
        self.order = order if order is not None else []
        self.postprocessed = False

    def schedule(self):
        self.order.append("schedule")
        return self._seqs, self._is_prefill

    def postprocess(self, seqs, token_ids, is_prefill):
        self.order.append("postprocess")
        self.postprocessed = True


def make_engine(runner=None, scheduler=None, order=None):
    """绕过 __init__ 构造引擎骨架（不加载模型、不 spawn 进程）。"""
    engine = LLMEngine.__new__(LLMEngine)
    engine.ps = []
    engine.model_runner = runner if runner is not None else RecordingRunner(order=order)
    if scheduler is not None:
        engine.scheduler = scheduler
    return engine


# ── exit()：幂等与异常路径 ──────────────────────────────────────────

def test_exit_is_idempotent():
    runner = RecordingRunner()
    engine = make_engine(runner=runner)

    engine.exit()
    engine.exit()                         # 显式调用后的 atexit 重入

    assert runner.calls == ["exit"]       # teardown 只跑一次
    assert not hasattr(engine, "model_runner")


def test_exit_latch_set_even_when_teardown_raises():
    runner = RecordingRunner(fail_exit=True)
    engine = make_engine(runner=runner)

    with pytest.raises(RuntimeError, match="teardown boom"):
        engine.exit()                     # 原始异常必须原样抛出

    engine.exit()                         # atexit 重入：静默返回，不重复 teardown、不掩盖上面的异常
    assert runner.calls == ["exit"]


# ── step()：swap 拷贝顺序（设计约束 R1）────────────────────────────

def test_step_orders_swap_out_then_swap_in_then_run():
    order = []
    seqs = [Sequence([1, 2, 3, 4])]
    scheduler = FakeScheduler(seqs, is_prefill=False,
                              swap_out={0: 1}, swap_in={2: 3}, order=order)
    engine = make_engine(runner=RecordingRunner(order=order), scheduler=scheduler)

    outputs, num_tokens = engine.step()

    # 顺序即正确性约束：换出（GPU→CPU）必须在换入之前，且都发生在 run() 之前
    assert order == ["schedule", "swap_out", "swap_in", "run", "postprocess"]
    assert scheduler.postprocessed
    assert outputs == [] and num_tokens == -1          # decode：返回负的 seq 数


def test_step_skips_swap_calls_when_maps_are_empty():
    order = []
    seqs = [Sequence([1, 2, 3, 4])]
    seqs[0].num_scheduled_tokens = 4
    scheduler = FakeScheduler(seqs, is_prefill=True, order=order)
    engine = make_engine(runner=RecordingRunner(order=order), scheduler=scheduler)

    outputs, num_tokens = engine.step()

    assert order == ["schedule", "run", "postprocess"]  # 未开启 swap 时不产生额外调用
    assert num_tokens == 4                             # prefill：返回本轮 token 数
