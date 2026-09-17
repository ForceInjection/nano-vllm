"""Scheduler 行为单测：批拼接规则、chunked prefill、decode+preempt、swap 全流程、postprocess。

对应课程第 3 课。用 SimpleNamespace 伪造 Config（Scheduler 只读标量字段），无需模型 / GPU。
swap 元数据状态机本身见 test_swap_blockmanager.py；这里验证的是**调度器编排**。
"""
import types

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus

BLOCK = 4


def make_scheduler(num_blocks=64, num_cpu_blocks=0, max_num_batched_tokens=64,
                   max_num_seqs=8, eos=-1) -> Scheduler:
    cfg = types.SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=BLOCK,
        num_kvcache_blocks=num_blocks,
        num_cpu_kvcache_blocks=num_cpu_blocks,
    )
    return Scheduler(cfg)


def make_seq(n_tokens: int) -> Sequence:
    Sequence.block_size = BLOCK
    return Sequence(list(range(n_tokens)))


def park_as_swapped(sched: Scheduler, seq: Sequence) -> None:
    """把 seq 手工搬进 swapped 队列（KV 元数据走真实的 block_manager.swap_out）。"""
    if seq in sched.running:
        sched.running.remove(seq)
    sched.block_manager.swap_out(seq)
    seq.status = SequenceStatus.SWAPPED
    sched.swapped.append(seq)


# ── prefill 批拼接 ──────────────────────────────────────────────────

def test_prefill_stops_at_token_budget():
    sched = make_scheduler(max_num_batched_tokens=10)
    seqs = [make_seq(4) for _ in range(3)]
    for s in seqs:
        sched.add(s)

    scheduled, is_prefill = sched.schedule()
    assert is_prefill and [s.num_scheduled_tokens for s in scheduled] == [4, 4]
    assert len(sched.waiting) == 1                    # 第三条超预算，留在 waiting
    assert seqs[2].block_table == []                  # 未被调度 → 不分配 block
    assert scheduled[0].status == SequenceStatus.RUNNING


def test_chunked_prefill_only_for_first_seq():
    sched = make_scheduler(max_num_batched_tokens=6)
    big, other = make_seq(10), make_seq(4)
    for s in (big, other):
        sched.add(s)

    scheduled, is_prefill = sched.schedule()
    assert is_prefill and scheduled == [big]          # 只切首条；后续 seq 直接 break
    assert big.num_scheduled_tokens == 6              # 被截断到预算
    assert big in sched.waiting                       # 未算完 → 仍在 waiting
    assert other.block_table == []


def test_prefill_step_skips_swap_in_even_when_blocks_are_free():
    # prefill 优先：只要产出了 prefill batch 就直接 return，swap-in 阶段不执行
    sched = make_scheduler(num_blocks=8, num_cpu_blocks=8)
    parked = make_seq(4)
    sched.add(parked)
    sched.schedule()
    sched.postprocess([parked], [7], True)
    park_as_swapped(sched, parked)

    newcomer = make_seq(4)
    sched.add(newcomer)
    scheduled, is_prefill = sched.schedule()
    assert is_prefill and scheduled == [newcomer]
    assert not sched.blocks_to_swap_in
    assert parked.status == SequenceStatus.SWAPPED and parked in sched.swapped


# ── decode：swap-in 阶段与 preempt ──────────────────────────────────

def test_swap_in_phase_precedes_decode():
    sched = make_scheduler(num_blocks=8, num_cpu_blocks=8)
    seq = make_seq(4)
    sched.add(seq)
    sched.schedule()
    sched.postprocess([seq], [7], True)               # num_cached_tokens = 4
    park_as_swapped(sched, seq)

    scheduled, is_prefill = sched.schedule()          # 无 prefill 可做 → swap-in
    assert not is_prefill
    assert sched.blocks_to_swap_in and seq.status == SequenceStatus.RUNNING
    assert scheduled == [seq]                         # 迁回后直接进 decode batch
    assert seq.num_cached_tokens == 4                 # 断点保留
    assert set(sched.blocks_to_swap_in.values()) <= set(seq.block_table)   # block_table 已重建


def test_full_swap_flow_preempt_then_resume():
    """课程 L03 §7 场景 E 的 CI 版：占满 GPU → 抢占搬出 → 等空块 → 迁回断点续跑。"""
    sched = make_scheduler(num_blocks=2, num_cpu_blocks=8, eos=0)
    a, b = make_seq(4), make_seq(4)
    for s in (a, b):
        sched.add(s)

    prefill_seqs, is_prefill = sched.schedule()
    assert is_prefill and len(prefill_seqs) == 2
    assert len(sched.block_manager.free_block_ids) == 0
    sched.postprocess(prefill_seqs, [7, 8], is_prefill)   # 两条都到 len=5 → 下次 decode 需新块

    decode_seqs, is_decode = sched.schedule()         # 队尾被抢占 → SWAP
    assert not is_decode and decode_seqs == [a]
    assert b.status == SequenceStatus.SWAPPED and b in sched.swapped
    assert sched.blocks_to_swap_out and sched.num_swapped_out_blocks == 1
    assert b.num_cached_tokens == 4                   # 核心不变量：kv 断点未被清零
    assert b.block_table == []
    sched.postprocess(decode_seqs, [9], is_decode)

    still_swapped, _ = sched.schedule()               # GPU 无空块 → B 还迁不回来
    assert not sched.blocks_to_swap_in and b in sched.swapped
    assert still_swapped == [a]
    sched.postprocess(still_swapped, [sched.eos], False)   # A 命中 EOS → 完成并释放 block

    resumed, is_decode = sched.schedule()             # swap-in 阶段把 B 迁回
    assert sched.blocks_to_swap_in and resumed == [b]
    assert b.status == SequenceStatus.RUNNING and b.num_cached_tokens == 4
    assert (sched.num_swapped_out_blocks, sched.num_swapped_in_blocks) == (1, 1)
    assert sched.num_recompute_preemptions == 0       # SWAP 路径零重算


def test_same_pressure_falls_back_to_recompute_without_cpu_pool():
    """课程 L03 §7 场景 E2 的 CI 版：同样压力、无 CPU 块 → preempt 只能丢 KV 重算。"""
    sched = make_scheduler(num_blocks=2, num_cpu_blocks=0, eos=0)
    a, b = make_seq(4), make_seq(4)
    for s in (a, b):
        sched.add(s)
    prefill_seqs, is_prefill = sched.schedule()
    sched.postprocess(prefill_seqs, [7, 8], is_prefill)

    decode_seqs, _ = sched.schedule()
    assert decode_seqs == [a]
    assert b.status == SequenceStatus.WAITING and b in sched.waiting
    assert b.num_cached_tokens == 0                   # 与 SWAP 的对照：KV 被丢弃
    assert not sched.blocks_to_swap_out and not sched.swapped
    assert sched.num_recompute_preemptions == 1


# ── postprocess ─────────────────────────────────────────────────────

def test_postprocess_finishes_on_eos_and_frees_blocks():
    sched = make_scheduler(eos=0)
    seq = make_seq(4)
    sched.add(seq)
    scheduled, is_prefill = sched.schedule()
    seq.num_scheduled_tokens = 1
    used_before = len(sched.block_manager.used_block_ids)

    sched.postprocess(scheduled, [sched.eos], False)
    assert seq.status == SequenceStatus.FINISHED and seq.is_finished
    assert seq not in sched.running
    assert len(sched.block_manager.used_block_ids) == used_before - 1


def test_postprocess_chunked_prefill_only_updates_counters():
    sched = make_scheduler(max_num_batched_tokens=6)
    seq = make_seq(10)
    sched.add(seq)
    scheduled, is_prefill = sched.schedule()          # 只算前 6 个 token
    assert is_prefill and seq.num_scheduled_tokens == 6

    sched.postprocess(scheduled, [7], is_prefill)
    assert seq.num_cached_tokens == 6
    assert seq.num_scheduled_tokens == 0
    assert seq.num_tokens == 10                       # 未算完 → 不追加采样 token
    assert seq.status == SequenceStatus.WAITING


def test_is_finished_accounts_for_waiting():
    sched = make_scheduler()
    assert sched.is_finished()
    sched.add(make_seq(4))
    assert not sched.is_finished()
    sched.waiting.clear()
    assert sched.is_finished()
