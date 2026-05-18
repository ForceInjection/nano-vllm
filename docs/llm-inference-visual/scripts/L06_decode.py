#!/usr/bin/env python3
"""
L06 练习：decode 一步生成与 block_tables

验证要点：
- decode slot 公式：block_table[-1] * block_size + last_block_num_tokens - 1
- may_append 触发条件：len(seq) % block_size == 1
- block_tables padding：不同长度补齐后用 -1 标记无效 block

依赖：无（纯 Python 模拟）
用法：python L06_decode.py
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


def slot(block_table_last, block_size, last_block_num_tokens):
    """decode 写入位置的 slot 计算公式。对齐 model_runner.py:L181。"""
    return block_table_last * block_size + last_block_num_tokens - 1


def verify_slot_formula():
    """验证 decode slot 公式。"""
    print("=" * 68)
    print("L06 验证：decode 批构建 — slot_mapping 与 may_append")
    print("=" * 68)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  1. prepare_decode & slot 公式(model_runner.py:L172-L188)  │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("prepare_decode", "nanovllm/engine/model_runner.py",
                     show_source("nanovllm/engine/model_runner.py", 172, 189))

    block_size = 256

    # 场景1：新 block 的第 0 个位置(last_block_num_tokens = 1 → 新 block 起始)
    s = slot(3, block_size, 1)
    print(f"\n场景1: block_table_last=3, block_size=256, last_block_num_tokens=1")
    print(f"  slot = 3 * 256 + 1 - 1 = {s}")
    expected = 3 * 256  # 第 0 个位置
    assert s == expected, f"slot 应为 {expected}, 实际 {s}"
    print(f"  [PASS] 新 block 第 0 个位置的 slot = {expected}")

    # 场景2：block 的最后一个位置
    s = slot(3, block_size, 256)
    print(f"\n场景2: block_table_last=3, block_size=256, last_block_num_tokens=256")
    print(f"  slot = 3 * 256 + 256 - 1 = {s}")
    expected = 4 * 256 - 1  # block 3 的最后一个位置
    assert s == expected, f"slot 应为 {expected}, 实际 {s}"
    print(f"  [PASS] block 最后一个位置的 slot = {expected}")

    # 场景3：block 中间位置
    s = slot(3, block_size, 128)
    print(f"\n场景3: block_table_last=3, block_size=256, last_block_num_tokens=128")
    print(f"  slot = 3 * 256 + 128 - 1 = {s}")
    assert s == 3 * 256 + 127
    print(f"  [PASS] block 中间位置的 slot = {s}")


def verify_may_append():
    """验证 may_append 的触发条件。"""
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  2. may_append 触发条件 (block_manager.py:L103-L108)       │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("can_append / may_append", "nanovllm/engine/block_manager.py",
                     show_source("nanovllm/engine/block_manager.py", 103, 109))

    print("  >>> 验证\n")

    block_size = 4  # 小值便于手算

    print(f"  block_size = {block_size}")
    print(f"  规则：len(seq) % block_size == 1 时触发新 block 分配")
    print()

    for length in range(1, 14):
        need_new_block = (length % block_size == 1)
        marker = " ← 触发 may_append! 分配新 block" if need_new_block else ""
        print(f"    len(seq)={length:>2} → {length} % {block_size} = {length % block_size}{marker}")

    # 验证关键边界
    # len=1: 1%4=1 → True (第一个 token，需要初始 block)
    # len=4: 4%4=0 → False (刚好填满第 1 个 block)
    # len=5: 5%4=1 → True (需要第 2 个 block)
    # len=8: 8%4=0 → False (刚好填满第 2 个 block)
    # len=9: 9%4=1 → True (需要第 3 个 block)

    assert (1 % block_size == 1) == True, "len=1 应触发新 block 分配"
    assert (4 % block_size == 1) == False, "len=4 填满后不应立即分配"
    assert (5 % block_size == 1) == True, "len=5 跨 block 边界应触发"
    assert (8 % block_size == 1) == False
    assert (9 % block_size == 1) == True

    print(f"\n  [PASS] may_append 触发条件验证通过")


def verify_can_append():
    """验证 can_append 与 may_append 的关系。"""
    print("\n--- can_append 与 may_append 的配合 ---")

    block_size = 4
    print(f"  block_size = {block_size}")
    print()

    # can_append 检查下一轮是否需要新 block，且 free blocks 是否够
    for free_blocks in [0, 1]:
        for length in [1, 4, 5, 8]:
            needs_block = (length % block_size == 1)
            can = free_blocks >= needs_block
            print(f"    free_blocks={free_blocks}, len(seq)={length}, "
                  f"needs_block={needs_block}, can_append={can}")

    # 关键场景：free_blocks=0 且 len%block_size==1 → can_append=False → 触发 preempt
    assert (0 >= True) == False  # 无空闲 block，需要新 block → can_append 失败
    assert (0 >= False) == True  # 无空闲 block，不需新 block → can_append 成功
    print(f"\n  [PASS] can_append = (free_blocks >= needs_block)")
    print(f"  当 can_append=False 且无其他 seq 可 preempt 时，自身被 preempt")


def verify_block_tables_padding():
    """验证 block_tables 的 padding 逻辑。"""
    print("\n--- block_tables padding ---")

    # prepare_block_tables (model_runner.py:L123-L127):
    # max_len = max(len(seq.block_table) for seq in seqs)
    # block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]

    block_tables_raw = [
        [3, 7],           # seq_a: 2 blocks
        [5, 12, 8],       # seq_b: 3 blocks
        [1],               # seq_c: 1 block
    ]

    max_len = max(len(bt) for bt in block_tables_raw)
    padded = [bt + [-1] * (max_len - len(bt)) for bt in block_tables_raw]

    print(f"  原始 block_tables: {block_tables_raw}")
    print(f"  max_len = {max_len}")
    print(f"  padding 后:")
    for i, bt in enumerate(padded):
        print(f"    seq[{i}]: {bt}")

    assert padded == [
        [3, 7, -1],
        [5, 12, 8],
        [1, -1, -1],
    ], f"padding 结果不符合预期: {padded}"

    print(f"  [PASS] -1 哨兵标记无效 block 位置")


def main():
    verify_slot_formula()
    verify_may_append()
    verify_can_append()
    verify_block_tables_padding()

    print("\n" + "=" * 60)
    print("L06 验证完成：所有断言通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
