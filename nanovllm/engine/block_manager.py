from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, num_cpu_blocks: int = 0):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # CPU offloading: a plain parking lot of CPU block ids (no hashing / prefix-cache)
        self.free_cpu_block_ids: deque[int] = deque(range(num_cpu_blocks))
        self.used_cpu_block_ids: set[int] = set()
        self.swapped_block_tables: dict[int, list[int]] = dict()  # seq_id -> cpu block ids

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def _allocate_cpu_block(self) -> int:
        cpu_id = self.free_cpu_block_ids.popleft()
        self.used_cpu_block_ids.add(cpu_id)
        return cpu_id

    def _deallocate_cpu_block(self, cpu_id: int):
        self.used_cpu_block_ids.remove(cpu_id)
        self.free_cpu_block_ids.append(cpu_id)

    def can_swap_out(self, seq: Sequence) -> bool:
        # Only swap sequences whose blocks are all privately owned (ref_count == 1);
        # a shared block cannot be evicted while another sequence still needs it in GPU.
        if len(self.free_cpu_block_ids) < len(seq.block_table):
            return False
        return all(self.blocks[block_id].ref_count == 1 for block_id in seq.block_table)

    def swap_out(self, seq: Sequence) -> dict[int, int]:
        assert seq.seq_id not in self.swapped_block_tables
        mapping = {}
        cpu_block_ids = []
        for block_id in seq.block_table:
            cpu_id = self._allocate_cpu_block()
            mapping[block_id] = cpu_id
            cpu_block_ids.append(cpu_id)
            block = self.blocks[block_id]
            block.ref_count -= 1
            assert block.ref_count == 0
            self._deallocate_block(block_id)  # frees the GPU block, keeps block.hash (lazy cleanup)
        self.swapped_block_tables[seq.seq_id] = cpu_block_ids
        seq.block_table = []  # num_cached_tokens is intentionally preserved
        return mapping

    def can_swap_in(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= len(self.swapped_block_tables[seq.seq_id])

    def swap_in(self, seq: Sequence) -> dict[int, int]:
        cpu_block_ids = self.swapped_block_tables.pop(seq.seq_id)
        mapping = {}
        for cpu_id in cpu_block_ids:
            block_id = self._allocate_block()
            mapping[cpu_id] = block_id
            seq.block_table.append(block_id)
            self._deallocate_cpu_block(cpu_id)
        return mapping
