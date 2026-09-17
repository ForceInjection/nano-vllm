"""BlockManager 核心语义单测（非 swap 部分）：前缀缓存、引用计数、追加边界、哈希生命周期。

swap 相关（can_swap_out/swap_out/can_swap_in/swap_in）见 test_swap_blockmanager.py。
纯 CPU：BlockManager 只操作元数据，不碰张量。
"""
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BLOCK = 4  # 小 block 便于手算；BlockManager 本身不要求 %256


def make_seq(n_tokens: int) -> Sequence:
    Sequence.block_size = BLOCK
    return Sequence(list(range(n_tokens)))


def allocate_fresh(bm: BlockManager, seq: Sequence) -> int:
    """为 seq 分配 block（无前缀命中），返回命中的 cached block 数（应恒为 0）。"""
    n_cached = bm.can_allocate(seq)
    assert n_cached == 0
    bm.allocate(seq, n_cached)
    return n_cached


def hash_prefill(bm: BlockManager, seq: Sequence, num_tokens: int):
    """模拟 prefill 完成后 postprocess 里的 hash_blocks（此时 num_cached_tokens 仍为 0）。"""
    seq.num_cached_tokens, seq.num_scheduled_tokens = 0, num_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens, seq.num_scheduled_tokens = num_tokens, 0


# ── 前缀缓存：命中、共享、引用计数 ──────────────────────────────────

def test_allocate_fresh_marks_all_blocks_used():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    seq = make_seq(12)                     # 3 blocks
    allocate_fresh(bm, seq)
    assert len(seq.block_table) == 3
    assert set(seq.block_table) <= bm.used_block_ids
    assert len(bm.free_block_ids) == 7
    assert seq.num_cached_tokens == 0      # 没命中前缀 → 全部要算


def test_prefix_hit_shares_blocks_and_bumps_refcount():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    a = make_seq(12)
    allocate_fresh(bm, a)
    hash_prefill(bm, a, 12)                # 注册 a 的 3 个完整块

    b = make_seq(16)                       # 4 blocks，前 3 块与 a 完全同前缀
    n_cached = bm.can_allocate(b)
    assert n_cached == 3                   # 末块永不参与命中（can_allocate 只查到 num_blocks-1）
    bm.allocate(b, n_cached)
    assert b.block_table[:3] == a.block_table
    assert all(bm.blocks[i].ref_count == 2 for i in a.block_table)
    assert b.num_cached_tokens == 3 * BLOCK
    assert len(bm.free_block_ids) == 10 - 3 - 1   # 只新分配 1 块


def test_prefix_miss_when_content_differs():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    a = make_seq(12)
    allocate_fresh(bm, a)
    hash_prefill(bm, a, 12)

    b = Sequence([100, 101, 102, 103, 104, 105, 106, 107, 0, 1, 2, 3])  # 首块内容即不同
    assert bm.can_allocate(b) == 0


def test_token_ids_check_guards_against_hash_collision():
    # 哈希命中但登记内容不符（哈希碰撞 / 陈旧登记）→ 二次校验必须拒绝，否则会复用错误的 KV
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    a = make_seq(12)
    allocate_fresh(bm, a)
    hash_prefill(bm, a, 12)
    bm.blocks[a.block_table[0]].token_ids = [999, 999, 999, 999]   # 篡改登记内容，哈希仍指向该块

    b = make_seq(16)
    assert bm.can_allocate(b) == 0


def test_deallocate_respects_refcount():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    a = make_seq(12)
    allocate_fresh(bm, a)
    hash_prefill(bm, a, 12)
    b = make_seq(16)
    bm.allocate(b, bm.can_allocate(b))

    bm.deallocate(b)
    # b 独占的第 4 块释放；共享的 3 块仍被 a 引用
    assert all(bm.blocks[i].ref_count == 1 for i in a.block_table)
    assert all(i in bm.used_block_ids for i in a.block_table)
    assert len(bm.free_block_ids) == 10 - 3

    bm.deallocate(a)
    assert not bm.used_block_ids and len(bm.free_block_ids) == 10


def test_can_allocate_returns_minus_one_when_pool_too_small():
    bm = BlockManager(num_blocks=3, block_size=BLOCK)
    big = make_seq(16)                     # 需要 4 块 > 3
    assert bm.can_allocate(big) == -1
    assert big.block_table == []           # 拒绝时不得留下半分配状态


# ── 哈希写入：只登记完整块，惰性清理 ────────────────────────────────

def test_hash_blocks_registers_complete_blocks_only():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    seq = make_seq(10)                     # 3 blocks，最后一块只有 2 token
    allocate_fresh(bm, seq)
    hash_prefill(bm, seq, 10)              # 10 // 4 = 2 → 只登记前两块

    for i in (0, 1):
        block = bm.blocks[seq.block_table[i]]
        assert block.hash != -1
        assert block.token_ids == seq.block(i)
        assert bm.hash_to_block_id[block.hash] == block.block_id
    assert bm.blocks[seq.block_table[2]].hash == -1      # 未满块不登记


def test_hash_blocks_noop_when_no_block_completed():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    seq = make_seq(8)
    allocate_fresh(bm, seq)
    seq.num_cached_tokens, seq.num_scheduled_tokens = 2, 1   # start == end == 0
    bm.hash_blocks(seq)
    assert bm.hash_to_block_id == {}


def test_hash_chain_depends_on_prefix():
    h_first = BlockManager.compute_hash([1, 2, 3, 4], -1)
    assert BlockManager.compute_hash([1, 2, 3, 4], -1) == h_first          # 确定性
    assert BlockManager.compute_hash([1, 2, 3, 4], 12345) != h_first       # 链式：带 prefix 就不同
    assert BlockManager.compute_hash([1, 2, 3, 5], -1) != h_first          # 内容不同


def test_reused_block_drops_stale_hash_registration():
    bm = BlockManager(num_blocks=2, block_size=BLOCK)     # 2 块 → 释放后可确定性地复用
    seq = make_seq(8)
    allocate_fresh(bm, seq)
    hash_prefill(bm, seq, 8)
    # deallocate 按 reversed(block_table) 归还，free 池从左端弹出 → 最后分配的那块最先被复用
    block_id = seq.block_table[-1]
    stale_hash = bm.blocks[block_id].hash

    bm.deallocate(seq)                                   # 释放但保留 hash（懒清理）
    assert bm.blocks[block_id].hash == stale_hash

    reused = make_seq(4)
    bm.allocate(reused, 0)
    assert reused.block_table[0] == block_id             # 确实复用了同一块
    assert bm.blocks[block_id].hash == -1
    assert bm.blocks[block_id].token_ids == []
    assert stale_hash not in bm.hash_to_block_id         # 陈旧登记已清除


# ── can_append / may_append：跨块边界 ───────────────────────────────

def test_may_append_allocates_only_on_block_boundary():
    bm = BlockManager(num_blocks=10, block_size=BLOCK)
    seq = make_seq(4)                     # len % block_size == 0 → 本步无需新块
    allocate_fresh(bm, seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 1

    seq.append_token(42)                  # len 变成 5 → 5 % 4 == 1 → 下一个 token 落在新块
    bm.may_append(seq)
    assert len(seq.block_table) == 2


def test_can_append_reflects_free_pool():
    # 调度器在调用 may_append 之前必须先问 can_append：池空时 may_append 没有兜底
    bm = BlockManager(num_blocks=1, block_size=BLOCK)
    seq = make_seq(4)
    allocate_fresh(bm, seq)
    assert bm.can_append(seq) is True     # len % block_size == 0 → 本步不需要新块
    seq.append_token(42)                  # len=5 → 需要新块，但池已空
    assert list(bm.free_block_ids) == []
    assert bm.can_append(seq) is False
