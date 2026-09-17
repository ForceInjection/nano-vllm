"""Sequence 数据结构单测：block 切分、计数器、TP 序列化协议。纯 CPU，无需 torch。

对应课程第 2 课。block_size 是类属性，每个用例按需改写（与 test_swap_blockmanager.py 同法）。
"""
import pickle

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK = 4


def make_seq(n_tokens: int, **sp_kwargs) -> Sequence:
    Sequence.block_size = BLOCK
    return Sequence(list(range(n_tokens)), SamplingParams(**sp_kwargs))


def test_block_math_at_boundaries():
    # 首块未满 → 1 块；恰好整除 → 块数不虚增；跨块 → 末块 token 数正确
    for n, num_blocks, last in [(1, 1, 1), (4, 1, 4), (5, 2, 1), (8, 2, 4), (9, 3, 1)]:
        seq = make_seq(n)
        assert seq.num_blocks == num_blocks, f"n={n}"
        assert seq.last_block_num_tokens == last, f"n={n}"


def test_block_slices_are_disjoint_and_complete():
    seq = make_seq(9)
    blocks = [seq.block(i) for i in range(seq.num_blocks)]
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    assert sum(len(b) for b in blocks) == seq.num_tokens


def test_prompt_and_completion_slices():
    seq = make_seq(4)
    assert seq.completion_token_ids == [] and seq.num_completion_tokens == 0
    seq.append_token(99)
    seq.append_token(100)
    assert seq.completion_token_ids == [99, 100]
    assert seq.num_completion_tokens == 2
    assert seq.prompt_token_ids == [0, 1, 2, 3]
    assert seq.last_token == 100 and len(seq) == 6
    assert seq.status == SequenceStatus.WAITING and not seq.is_finished


def test_getstate_sends_token_ids_only_for_prefill():
    seq = make_seq(5)
    seq.is_prefill = True
    assert isinstance(seq.__getstate__()[-1], list)
    seq.is_prefill = False
    assert seq.__getstate__()[-1] == seq.last_token          # decode 只传 last_token（TP IPC 优化）


def test_pickle_roundtrip_prefill_branch():
    seq = make_seq(6)
    seq.num_cached_tokens, seq.num_scheduled_tokens = 4, 2
    seq.block_table[:] = [7, 8]
    seq.is_prefill = True
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.token_ids == seq.token_ids
    assert restored.last_token == seq.last_token
    assert (restored.num_tokens, restored.num_cached_tokens) == (6, 4)
    assert restored.block_table == [7, 8]


def test_pickle_roundtrip_decode_branch_drops_token_ids():
    seq = make_seq(6)
    seq.is_prefill = False
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.token_ids == []                          # decode 分支刻意不带全量 token
    assert restored.last_token == seq.last_token
    assert restored.num_tokens == seq.num_tokens
