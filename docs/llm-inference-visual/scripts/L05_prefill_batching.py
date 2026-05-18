#!/usr/bin/env python3
"""
L05 练习：prefill 批构建与 context 注入

验证要点：
- cu_seqlens_q 为 1+bs 长度的前缀和，标记展平 input_ids 中各 seq 的边界
- positions 为展平后的 token 绝对位置，起点为 num_cached_tokens
- prefix cache 命中时 cu_seqlens_k 可能大于 cu_seqlens_q
- slot_mapping 将逻辑 token 映射到物理 KV cache 的 slot

依赖：无（纯 Python 模拟）
用法：python L05_prefill_batching.py
"""

import os


def show_source(file_path, start, end):
    # scripts/ → llm-inference-visual/ → docs/ → repo_root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    full = os.path.join(repo_root, file_path)
    if not os.path.exists(full):
        return []
    with open(full) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[start - 1:end]]


def show_code_block(title, file_path, lines):
    print(f"  // {title}  ({file_path})")
    for l in lines:
        print(f"  {l}")
    print()


# ── 模拟 prepare_prefill（对齐 model_runner.py:L129-L170）─────────────

def build_prefill_tensors(seqs):
    """
    seqs: list of (num_cached_tokens, num_scheduled_tokens)
    返回: (cu_seqlens_q, positions, input_ids 展平区间)
    """
    cu_seqlens_q = [0]
    positions = []
    token_ranges = []

    for cached, scheduled in seqs:
        start = cached
        end = cached + scheduled
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + scheduled)
        token_ranges.append(f"[{start}:{end})")

    return cu_seqlens_q, positions, token_ranges


def build_slot_mapping(block_table, start, end, block_size=256):
    """模拟 prepare_prefill 中的 slot_mapping 构造循环。"""
    slot_mapping = []
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size

    for i in range(start_block, end_block):
        slot_start = block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = block_table[i] * block_size + block_size
        else:
            slot_end = block_table[i] * block_size + end - i * block_size
        slots = list(range(slot_start, slot_end))
        slot_mapping.extend(slots)

    return slot_mapping


# ── 验证 1: cu_seqlens_q 与 positions ─────────────────────────────────

def verify_cu_seqlens():
    print("=" * 64)
    print("L05 验证：prefill 批构建 — 展平拼接 + slot_mapping + block_tables")
    print("=" * 64)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  1. cu_seqlens_q 与 positions 展平拼接                      │")
    print("│     对齐 model_runner.py:L129-L148                          │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("prepare_prefill (input_ids & positions)", "nanovllm/engine/model_runner.py",
                     show_source("nanovllm/engine/model_runner.py", 129, 149))

    # 场景 A: 无 prefix cache
    print(f"\n  场景 A: seq_a=(cached=0, scheduled=3), seq_b=(0, 2) — 无 prefix cache")
    cu_q, pos, ranges = build_prefill_tensors([(0, 3), (0, 2)])

    print(f"    逐 seq 构造:")
    print(f"      seq_a: start=0, end=3 → positions [0,1,2], query_len=3")
    print(f"      seq_b: start=0, end=2 → positions [0,1],   query_len=2")
    print(f"    展平结果:")
    print(f"      cu_seqlens_q = {cu_q}  ← 前缀和 [0, 3, 5]")
    print(f"      positions    = {pos}   ← 展平拼接 [0,1,2, 0,1]")
    print(f"    语义:")
    print(f"      seq_a 的 query 区间 = input_ids[0:3]  (cu_seqlens_q[0]:cu_seqlens_q[1])")
    print(f"      seq_b 的 query 区间 = input_ids[3:5]  (cu_seqlens_q[1]:cu_seqlens_q[2])")
    assert cu_q == [0, 3, 5]
    assert pos == [0, 1, 2, 0, 1]

    # 场景 B: prefix cache 命中
    print(f"\n  场景 B: seq_a=(cached=0, scheduled=3), seq_b=(cached=4, scheduled=2)")
    print(f"          seq_b 已缓存 4 token → positions 从 4 开始")
    cu_q, pos, ranges = build_prefill_tensors([(0, 3), (4, 2)])

    print(f"    逐 seq 构造:")
    print(f"      seq_a: start=0, end=3 → positions [0,1,2]")
    print(f"      seq_b: start=4, end=6 → positions [4,5]   ← 跳过已缓存!")
    print(f"    cu_seqlens_q = {cu_q}")
    print(f"    positions    = {pos}")
    assert cu_q == [0, 3, 5]
    assert pos == [0, 1, 2, 4, 5]
    print("  [PASS]")


# ── 验证 2: cu_seqlens_k 与 prefix cache ─────────────────────────────

def verify_cu_seqlens_k():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  2. cu_seqlens_k > cu_seqlens_q → 触发 block_tables 构造      │")
    print("│     对齐 model_runner.py:L162-L163                          │")
    print("└─────────────────────────────────────────────────────────────┘")

    # seq_a: 无 cache, prompt=3 → query 侧 3 tokens, kv 侧也是 3
    # seq_b: prefix cache 命中 6 tokens + 本轮 process 2 tokens
    #   → query 侧 2 tokens, kv 侧 8 tokens
    cu_seqlens_q = [0, 3, 5]
    cu_seqlens_k = [0, 3, 9]
    need_bt = cu_seqlens_k[-1] > cu_seqlens_q[-1]

    print(f"\n  cu_seqlens_q (query 侧)  = {cu_seqlens_q}")
    print(f"  cu_seqlens_k (kv 侧)     = {cu_seqlens_k}")
    print(f"  cu_seqlens_k[-1] ({cu_seqlens_k[-1]}) > cu_seqlens_q[-1] ({cu_seqlens_q[-1]})?")
    print(f"    → {need_bt}")
    print(f"  block_tables 是否需要构造: {'是' if need_bt else '否'}")
    print(f"  原因: seq_b 的 kv 侧长度 8 > query 侧长度 2 → 有 6 个 token 的 K/V 在 cache")
    assert need_bt
    print("  [PASS]")


# ── 验证 3: slot_mapping 构造 ────────────────────────────────────────

def verify_slot_mapping():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  3. slot_mapping: 逻辑 token → 物理 KV cache slot           │")
    print("│     对齐 model_runner.py:L149-L161                          │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("prepare_prefill (slot_mapping)", "nanovllm/engine/model_runner.py",
                     show_source("nanovllm/engine/model_runner.py", 149, 164))

    block_size = 16  # 用小值方便手算
    block_table = [5, 12, 8]
    start = 10
    end = 38

    print(f"\n  参数: block_size={block_size}, block_table={block_table}")
    print(f"  token 区间: [{start}, {end}), 共 {end - start} 个 token")
    print(f"  start_block={start // block_size}({start}//{block_size}), end_block={(end + block_size - 1) // block_size}")

    slot_mapping = build_slot_mapping(block_table, start, end, block_size)
    assert len(slot_mapping) == end - start

    # 逐块展示
    print(f"\n  逐块构造 slot_mapping:")
    for i, (bid, block_tokens) in enumerate(zip(block_table, [block_size] * 3)):
        sm_part = build_slot_mapping([bid], 0, block_tokens, block_size)
        print(f"    block[{i}]: id={bid}, slot 范围 [{sm_part[0]}, {sm_part[-1]}]")

    print(f"\n  覆盖 token [{start}:{end}) 的 slot_mapping:")
    print(f"    前 3 个 slot: {slot_mapping[:3]}")
    print(f"    后 3 个 slot: {slot_mapping[-3:]}")
    print(f"    总长度: {len(slot_mapping)} = {end - start} ✓")
    print(f"    第一个 slot = block_table[0] * block_size + start % block_size")
    print(f"                = {block_table[0]} * {block_size} + {start % block_size} = {slot_mapping[0]} ✓")
    print("  [PASS]")


def main():
    verify_cu_seqlens()
    verify_cu_seqlens_k()
    verify_slot_mapping()

    print("\n" + "=" * 64)
    print("L05 全部断言通过 ✓")
    print("=" * 64)


if __name__ == "__main__":
    main()
