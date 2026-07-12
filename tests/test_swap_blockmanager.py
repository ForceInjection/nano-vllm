"""A-layer unit tests for KV Cache CPU Offloading (swap-based preemption).

These exercise the pure-Python swap metadata state machine in BlockManager (no GPU / no torch)
plus the torch-only copy-indexing helper. Runnable locally without a GPU:

    pip install -e ".[test]"
    python -m pytest tests/ -v

GPU-level correctness (KV byte-equality, differential parity) lives in
docs/llm-inference-visual/scripts/verify_swap.py and runs on a CUDA box.
"""
import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus

BLOCK = 4  # small block size for tests; BlockManager itself does not require %256


def make_seq(n_tokens: int) -> Sequence:
    Sequence.block_size = BLOCK
    return Sequence(list(range(n_tokens)))


def allocate_fresh(bm: BlockManager, seq: Sequence) -> None:
    n_cached = bm.can_allocate(seq)
    assert n_cached == 0
    bm.allocate(seq, n_cached)


# ── A1: BlockManager swap metadata ────────────────────────────────────────────

def test_swap_out_frees_gpu_holds_cpu_preserves_cached_tokens():
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = make_seq(12)  # 3 blocks
    allocate_fresh(bm, seq)
    assert len(seq.block_table) == 3
    gpu_blocks = list(seq.block_table)
    seq.num_cached_tokens = 8  # simulate an in-flight decode sequence

    mapping = bm.swap_out(seq)

    assert list(mapping.keys()) == gpu_blocks                  # {gpu_id: cpu_id}, order preserved
    assert seq.block_table == []                               # GPU blocks released
    assert seq.num_cached_tokens == 8                          # invariant: NOT zeroed
    assert all(b in bm.free_block_ids for b in gpu_blocks)     # returned to GPU free pool
    assert set(mapping.values()) == bm.used_cpu_block_ids      # CPU blocks now used
    assert bm.swapped_block_tables[seq.seq_id] == list(mapping.values())


def test_swap_in_restores_block_table_and_frees_cpu():
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = make_seq(12)
    allocate_fresh(bm, seq)
    seq.num_cached_tokens = 12
    out_mapping = bm.swap_out(seq)             # {gpu_id: cpu_id}

    assert bm.can_swap_in(seq)
    mapping = bm.swap_in(seq)                   # {cpu_id: gpu_id}

    assert len(seq.block_table) == 3
    assert list(mapping.keys()) == list(out_mapping.values())  # swap-in keys are the swap-out CPU ids
    assert list(mapping.values()) == seq.block_table           # values become the new GPU blocks
    assert seq.seq_id not in bm.swapped_block_tables           # bookkeeping cleared
    assert len(bm.used_cpu_block_ids) == 0                     # CPU blocks released
    assert seq.num_cached_tokens == 12


def test_swap_round_trip_logical_length_preserved():
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = make_seq(16)  # 4 blocks
    allocate_fresh(bm, seq)
    seq.num_cached_tokens = 16
    n_blocks = len(seq.block_table)

    bm.swap_out(seq)
    assert seq.block_table == []
    bm.swap_in(seq)

    assert len(seq.block_table) == n_blocks                    # physical ids may differ, count equal
    assert len(bm.used_cpu_block_ids) == 0


def test_can_swap_out_false_when_cpu_pool_too_small():
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=2)
    seq = make_seq(12)  # needs 3 CPU blocks, only 2 available
    allocate_fresh(bm, seq)
    assert bm.can_swap_out(seq) is False


def test_can_swap_out_false_for_shared_block_R2():
    # R2: a sequence with any shared block (ref_count > 1) must fall back to RECOMPUTE.
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = make_seq(12)
    allocate_fresh(bm, seq)
    bm.blocks[seq.block_table[0]].ref_count = 2                # simulate a shared prefix block
    assert bm.can_swap_out(seq) is False


def test_can_swap_in_false_when_gpu_blocks_insufficient():
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = make_seq(12)
    allocate_fresh(bm, seq)
    bm.swap_out(seq)
    bm.free_block_ids.clear()                                  # no GPU blocks free
    assert bm.can_swap_in(seq) is False


def test_swapped_seq_marked_and_queued_via_scheduler_preempt():
    # Metadata-level check that swap_out routes a seq into the swapped bookkeeping.
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = make_seq(12)
    allocate_fresh(bm, seq)

    assert bm.can_swap_out(seq)
    mapping = bm.swap_out(seq)
    seq.status = SequenceStatus.SWAPPED
    assert seq.status == SequenceStatus.SWAPPED
    assert set(mapping) and seq.seq_id in bm.swapped_block_tables


def _make_scheduler():
    import types
    from nanovllm.engine.scheduler import Scheduler
    cfg = types.SimpleNamespace(
        max_num_seqs=16, max_num_batched_tokens=1024, eos=0,
        kvcache_block_size=BLOCK, num_kvcache_blocks=10, num_cpu_kvcache_blocks=10,
    )
    return Scheduler(cfg)  # Scheduler only reads scalar config fields, no model load


def test_is_finished_accounts_for_swapped_queue():
    # Regression: a sequence parked in `swapped` means work remains — is_finished() must be False,
    # otherwise the engine abandons swapped-out sequences (never swaps them back in).
    sched = _make_scheduler()
    seq = make_seq(12)
    allocate_fresh(sched.block_manager, seq)

    seq.status = SequenceStatus.RUNNING
    sched.running.append(seq)
    assert not sched.is_finished()

    # move it to the swapped queue (running/waiting now empty)
    sched.running.clear()
    sched.swapped.append(seq)
    assert not sched.is_finished(), "is_finished() ignored the swapped queue -> swapped seqs abandoned"

    sched.swapped.clear()
    assert sched.is_finished()


def test_preempt_swaps_out_normal_running_seq():
    # A seq NOT swapped in this step is preempted via SWAP.
    sched = _make_scheduler()
    seq = make_seq(12)
    allocate_fresh(sched.block_manager, seq)
    sched.preempt(seq)
    assert seq.status == SequenceStatus.SWAPPED
    assert seq in sched.swapped
    assert sched.blocks_to_swap_out                       # swapped out
    assert sched.num_recompute_preemptions == 0


def test_preempt_recomputes_seq_swapped_in_this_step():
    # Regression for the critical aliasing bug: a seq whose GPU blocks are swap-in targets THIS
    # step (KV not yet restored) must RECOMPUTE, never swap back out — otherwise swap_out would
    # copy garbage from those fresh blocks and destroy the seq's saved KV.
    sched = _make_scheduler()
    seq = make_seq(12)
    allocate_fresh(sched.block_manager, seq)
    # simulate: this seq was just swapped in -> its blocks are values in blocks_to_swap_in
    sched.blocks_to_swap_in = {1000 + i: b for i, b in enumerate(seq.block_table)}
    sched.num_swapped_in_blocks = len(seq.block_table)

    sched.preempt(seq)

    assert seq.status == SequenceStatus.WAITING           # RECOMPUTE, not SWAP
    assert seq in sched.waiting
    assert not sched.blocks_to_swap_out                   # did NOT swap out (no garbage copy)
    assert not sched.blocks_to_swap_in                    # pending swap-in cancelled
    assert sched.num_swapped_in_blocks == 0
    assert sched.num_recompute_preemptions == 1
    # no swap_out source may ever be a swap_in target in the same step
    assert set(sched.blocks_to_swap_out).isdisjoint(sched.blocks_to_swap_in.values())


# ── A2: tensor copy indexing (torch, CPU only) ────────────────────────────────

def test_swap_blocks_copy_indexing_round_trip():
    torch = pytest.importorskip("torch")
    from nanovllm.engine.kv_swap import swap_blocks

    L, bs, kvh, hd = 2, BLOCK, 2, 5
    gpu = torch.randn(2, L, 4, bs, kvh, hd)      # 4 GPU blocks
    cpu = torch.zeros(2, L, 3, bs, kvh, hd)      # 3 CPU blocks

    # swap out: gpu block 1 -> cpu 2, gpu block 3 -> cpu 0
    swap_blocks(gpu, cpu, {1: 2, 3: 0})
    assert torch.equal(cpu[:, :, 2], gpu[:, :, 1])
    assert torch.equal(cpu[:, :, 0], gpu[:, :, 3])

    # swap in to fresh GPU blocks: cpu 2 -> gpu 1, cpu 0 -> gpu 3
    gpu2 = torch.zeros_like(gpu)
    swap_blocks(cpu, gpu2, {2: 1, 0: 3})
    assert torch.equal(gpu2[:, :, 1], gpu[:, :, 1])
    assert torch.equal(gpu2[:, :, 3], gpu[:, :, 3])
