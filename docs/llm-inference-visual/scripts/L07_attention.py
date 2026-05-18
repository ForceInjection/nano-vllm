#!/usr/bin/env python3
"""
L07 练习：Attention — KV 写入与算子分支

验证要点：
- store_kvcache 用 slot_mapping 驱动写入，slot == -1 时跳过
- prefill 分支：flash_attn_varlen_func（变长批注意力）
- decode 分支：flash_attn_with_kvcache（增量生成注意力）
- prefix cache 命中时 K/V 直接使用 k_cache/v_cache

依赖：无（纯 Python 模拟）
用法：python L07_attention.py
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


# ── KV cache 写入模拟（对齐 attention.py:L10-L30）─────────────────────

def store_kv_sim(cache, slot_mapping, keys, values):
    """模拟 Triton kernel: slot == -1 时跳过。"""
    for idx, slot in enumerate(slot_mapping):
        if slot == -1:
            continue
        cache[slot] = (keys[idx], values[idx])
    return cache


# ── 注意力分支决策（对齐 attention.py:L59-L75）─────────────────────────

def attention_branch(context):
    """
    模拟 Attention.forward 的分支选择。
    context 包含: is_prefill, has_cache, has_block_tables
    返回调用的 API 名称和参数特征。
    """
    steps = []

    # Step 1: 是否绑定了 KV cache?
    if context["has_cache"]:
        steps.append("k_cache/v_cache 已绑定 → 调用 store_kvcache(k, v)")
    else:
        steps.append("k_cache/v_cache 未绑定 → 跳过 KV 写入（warmup 阶段）")

    # Step 2: prefill or decode?
    if context["is_prefill"]:
        if context.get("has_block_tables"):
            steps.append("is_prefill=True, block_tables≠None → prefix cache 分支")
            steps.append("  k, v = k_cache, v_cache  ← K/V 直接用缓存的")
            api = "flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, block_table=...)"
        else:
            steps.append("is_prefill=True, block_tables=None → 普通 prefill")
            steps.append("  k, v 使用本轮新计算的")
            api = "flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)"
    else:
        steps.append("is_prefill=False → decode 分支")
        steps.append("  q.unsqueeze(1)  ← 每个 seq 只有 1 token")
        api = "flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, context_lens, block_table=...)"

    return api, steps


# ── 验证 1: slot_mapping 的 -1 哨兵 ───────────────────────────────────

def verify_slot_mapping_sentinel():
    print("=" * 64)
    print("L07 验证：Attention — KV 写入 + 算子分支选择")
    print("=" * 64)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  1. store_kvcache 的 -1 哨兵（对齐 attention.py:L21-L24）   │")
    print("└─────────────────────────────────────────────────────────────┘")

    cache = {}
    store_kv_sim(cache,
        slot_mapping=[10, -1, 12, -1, 15],
        keys=["k0", "k1", "k2", "k3", "k4"],
        values=["v0", "v1", "v2", "v3", "v4"],
    )

    print(f"\n  slot_mapping = [10, -1, 12, -1, 15]")
    print(f"  写入结果:")
    for slot, (k, v) in sorted(cache.items()):
        print(f"    slot {slot:>3}: ({k}, {v})")
    print(f"  跳过的: idx=1 (slot=-1), idx=3 (slot=-1)")
    assert len(cache) == 3
    assert 10 in cache and 12 in cache and 15 in cache
    print("  [PASS]")

    # CUDA Graph 场景展示
    total = 8
    bs = 3
    slot_mapping_graph = [-1] * total
    slot_mapping_graph[:bs] = [20, 21, 22]
    print(f"\n  CUDA Graph 场景（对齐 model_runner.py:L206-L208）:")
    print(f"    total_slots={total}, bs={bs}")
    print(f"    fill_(-1) → [{', '.join(str(s) for s in slot_mapping_graph)}]")
    print(f"    slot_mapping[:bs] = [20, 21, 22]")

    cache2 = {}
    store_kv_sim(cache2, slot_mapping_graph,
                 [f"k{i}" for i in range(total)],
                 [f"v{i}" for i in range(total)])
    print(f"    写入 {len(cache2)} 个 slot: {sorted(cache2.keys())}")
    assert len(cache2) == 3
    print("  [PASS] CUDA Graph 场景: 仅前 bs 个 slot 有效")


# ── 验证 2: 注意力分支决策树 ──────────────────────────────────────────

def verify_attention_branches():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  2. Attention.forward 分支决策（对齐 attention.py:L59-L75）  │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("Attention.forward", "nanovllm/layers/attention.py",
                     show_source("nanovllm/layers/attention.py", 59, 76))

    scenarios = [
        {
            "label": "普通 prefill（warmup）",
            "ctx": {"is_prefill": True, "has_cache": False, "has_block_tables": False},
        },
        {
            "label": "普通 prefill（有 KV cache）",
            "ctx": {"is_prefill": True, "has_cache": True, "has_block_tables": False},
        },
        {
            "label": "prefix cache 命中 prefill",
            "ctx": {"is_prefill": True, "has_cache": True, "has_block_tables": True},
        },
        {
            "label": "decode",
            "ctx": {"is_prefill": False, "has_cache": True, "has_block_tables": True},
        },
    ]

    for s in scenarios:
        api, steps = attention_branch(s["ctx"])
        print(f"\n  ▸ {s['label']}")
        for step in steps:
            print(f"      {step}")
        print(f"      → {api}")

    print("\n  关系总结:")
    print("    prefill 用 flash_attn_varlen_func: 变长序列, 需要 cu_seqlens 定界")
    print("    decode  用 flash_attn_with_kvcache: 每 seq 1 token, 历史 K/V 已在 cache")
    print("    prefix cache 命中时: K/V 改用 k_cache/v_cache, block_tables 传入")


# ── 验证 3: prefix cache 触发条件 ────────────────────────────────────

def verify_prefix_cache_trigger():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  3. prefix cache 触发条件（model_runner.py:L162-L163）       │")
    print("└─────────────────────────────────────────────────────────────┘")

    cases = [
        # (cu_seqlens_q, cu_seqlens_k, expected)
        ([0, 3, 8], [0, 8, 13], True),
        ([0, 3, 8], [0, 3, 8], False),
    ]

    for cu_q, cu_k, expected in cases:
        needs_bt = cu_k[-1] > cu_q[-1]
        status = "✓ 触发" if needs_bt == expected else "✗ 错误"
        print(f"\n  cu_seqlens_q={cu_q}, cu_seqlens_k={cu_k}")
        print(f"    cu_seqlens_k[-1]({cu_k[-1]}) > cu_seqlens_q[-1]({cu_q[-1]})? {needs_bt}")
        print(f"    → {'需要构造 block_tables' if needs_bt else '不需要 block_tables'}  {status}")
        assert needs_bt == expected

    print("\n  直观理解:")
    print("    cu_seqlens_k > cu_seqlens_q 意味着某些 seq 的 cache 侧比 query 侧")
    print("    更长 → 有 prefix cache 数据 → 需要用 block_tables 查表定位")


def main():
    verify_slot_mapping_sentinel()
    verify_attention_branches()
    verify_prefix_cache_trigger()

    print("\n" + "=" * 64)
    print("L07 全部断言通过 ✓")
    print("=" * 64)


if __name__ == "__main__":
    main()
