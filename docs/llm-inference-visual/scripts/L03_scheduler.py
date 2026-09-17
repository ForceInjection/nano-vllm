#!/usr/bin/env python3
"""
L03 练习：Scheduler 的队列、chunked prefill 与 preempt

验证要点：
- prefill 从 waiting 队首取 seq，按 max_num_batched_tokens 限制拼 batch
- 除首个 seq 外不允许 chunked prefill (remaining < num_tokens and scheduled_seqs → break)
- decode 按 FIFO 从 running 取 seq，block 不够时 preempt
- preempt 双路径：SWAP 优先（KV 搬 CPU、进 swapped 队列断点续跑），否则 RECOMPUTE 兜底；
  swap_out/swap_in 元数据往返保留 num_cached_tokens（验证 6）

依赖：§1-§5 纯 Python；§6 需 nanovllm 可导入（pip install -e . 或 PYTHONPATH=仓库根）；
     §7 另需 transformers + 模型目录（argv 或 NANOVLLM_MODEL_PATH）
用法：python L03_scheduler.py
"""

from collections import deque
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


# ── prefill 模拟（对齐 scheduler.py:L40-L62）──────────────────────────

def simulate_prefill(prompt_lens, max_num_batched_tokens,
                     num_cached_tokens=None):
    if num_cached_tokens is None:
        num_cached_tokens = [0] * len(prompt_lens)

    scheduled = []
    remaining = max_num_batched_tokens

    for i, (n, cached) in enumerate(zip(prompt_lens, num_cached_tokens)):
        num_tokens = n - cached
        if remaining == 0:
            break
        if remaining < num_tokens and scheduled:
            break  # L52-L53: only chunked prefill for the first seq
        scheduled_tokens = min(num_tokens, remaining)
        scheduled.append((i, scheduled_tokens))
        remaining -= scheduled_tokens

    return scheduled, remaining


# ── decode 模拟（对齐 scheduler.py:L80-L94）───────────────────────────

def simulate_decode_step(running, free_block_ids, block_size=4):
    """
    模拟一次 decode step（对齐 scheduler.py:L80-L94）。
    关键: preempt 队尾 seq 时会释放它的 block，让当前 seq 可以继续。
    """
    scheduled = []
    preempted = []
    log = []

    free = list(free_block_ids)
    # 每个 seq 占用的 block 数: ceil(length / block_size)
    seq_blocks = {seq_id: (length + block_size - 1) // block_size
                  for seq_id, length in running}

    while running:
        seq_id, length = running.popleft()
        needs_block = (length % block_size == 1)

        # 如果 free 不够且需要新 block，尝试 preempt
        while needs_block and len(free) < 1:
            if running:
                # preempt 队尾 seq，回收它的 blocks
                victim_id, victim_len = running.pop()
                freed = seq_blocks.pop(victim_id)
                for _ in range(freed):
                    free.append(f"b{victim_id}")
                preempted.append(victim_id)
                log.append(f"    preempt seq[{victim_id}] (len={victim_len}, {freed} blocks) → "
                           f"free 增为 {len(free)}")
            else:
                # 只剩自己，自身也被 preempt
                preempted.append(seq_id)
                log.append(f"    seq[{seq_id}] len={length}: 需新 block 但无 free 且无其他 seq → 自身 preempt")
                return scheduled, preempted, log

        # can_append 成功 → 分配并调度
        if needs_block:
            blk = free.pop(0)
            log.append(f"    seq[{seq_id}] len={length}: 跨边界 → 分配 {blk} → free 剩 {len(free)}")
        else:
            log.append(f"    seq[{seq_id}] len={length}: 同 block → 无需分配 → free={len(free)}")
        new_len = length + 1
        scheduled.append((seq_id, new_len))
        seq_blocks[seq_id] = (new_len + block_size - 1) // block_size

    return scheduled, preempted, log


# ── 验证 1: 基本批拼接 ────────────────────────────────────────────────

def verify_basic_batching():
    print("=" * 64)
    print("L03 验证：Scheduler — prefill 批拼接 + chunked prefill + preempt")
    print("=" * 64)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  1. prefill 基本批拼接                                      │")
    print("│     对齐 scheduler.py:L40-L62                               │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("schedule() prefill 分支", "nanovllm/engine/scheduler.py",
                     show_source("nanovllm/engine/scheduler.py", 40, 62))

    for label, prompts, max_batch in [
        ("三条都能塞入", [300, 300, 300], 1000),
        ("第一条被 chunk", [1000, 500, 500], 800),
    ]:
        scheduled, remaining = simulate_prefill(prompts, max_batch)
        print(f"\n  {label}: prompts={prompts}, max_batch={max_batch}")
        print(f"    调度结果: scheduled={scheduled}")
        print(f"    剩余 token 预算: {remaining}")
        for i, tok in scheduled:
            print(f"      seq[{i}] 本轮处理 {tok} token")


# ── 验证 2: chunked prefill 限制 ──────────────────────────────────────

def verify_chunked_prefill_constraint():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  2. chunked prefill 限制: 仅 batch 中第一条可被切分           │")
    print("│     scheduler.py:L52-L53: remaining < num_tokens && scheduled → break │")
    print("└─────────────────────────────────────────────────────────────┘")

    scheduled, remaining = simulate_prefill([300, 800, 200], max_num_batched_tokens=1000)
    print(f"\n  prompts=[300, 800, 200], max_batch=1000")
    print(f"  逐条推演:")
    print(f"    seq[0]: 300 token ≤ 1000 → ✓ scheduled=300, remaining=700")
    print(f"    seq[1]: 800 token > remaining=700 且 scheduled 非空 → ✗ break!")
    print(f"    seq[2]: 永不检查")
    print(f"  结果: scheduled={scheduled}")
    assert scheduled == [(0, 300)]


# ── 验证 3: prefix cache 对批拼接的影响 ───────────────────────────────

def verify_prefix_cache_batching():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  3. prefix cache 命中减少本轮 token 消耗                      │")
    print("│     scheduler.py:L45-L51: num_tokens = n - cached_blocks*block_size │")
    print("└─────────────────────────────────────────────────────────────┘")

    scheduled, remaining = simulate_prefill(
        prompt_lens=[1000, 800],
        max_num_batched_tokens=1000,
        num_cached_tokens=[512, 0],
    )
    print(f"\n  seq[0]: prompt=1000, 已缓存 512 → 还需 488 token")
    print(f"  seq[1]: prompt=800,  未缓存     → 还需 800 token")
    print(f"  488 + 800 = 1288 > max_batch=1000 → seq[1] 放不下")
    print(f"  结果: scheduled={scheduled}, remaining={remaining}")
    assert scheduled == [(0, 488)]
    assert remaining == 512


# ── 验证 4: decode 调度 + preempt ────────────────────────────────────

def verify_decode_and_preempt():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  4. decode 调度 + preempt（对齐 scheduler.py:L80-L116）       │")
    print("│     can_append: free_blocks >= (len(seq) % block_size == 1) │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    show_code_block("schedule() decode 分支 + preempt()", "nanovllm/engine/scheduler.py",
                     show_source("nanovllm/engine/scheduler.py", 80, 116))

    block_size = 4
    running = deque([(0, 1), (1, 4), (2, 5)])  # (seq_id, current_length)

    # ── 场景 A: 有足够空闲 block ──
    print(f"\n  场景 A: free_blocks=[10, 11], running={list(running)}, block_size={block_size}")
    scheduled, preempted, log = simulate_decode_step(running, [10, 11], block_size)
    print("  ")
    for line in log:
        print(line)
    print(f"  调度结果: scheduled={list(scheduled)}, preempted={preempted}")
    assert len(scheduled) == 3
    assert len(preempted) == 0
    print("  [PASS] 空闲足够 → 三条全部 decode")

    # ── 场景 B: 空闲块不足，触发 preempt ──
    running = deque([(0, 1), (1, 4), (2, 8)])
    print(f"\n  场景 B: free_blocks=[], running={list(running)}, block_size={block_size}")
    print(f"    seq[2] len=8 → 占 2 blocks; seq[1] len=4 → 占 1 block; seq[0] len=1 → 占 1 block")
    scheduled, preempted, log = simulate_decode_step(running, [], block_size)
    print("")
    for line in log:
        print(line)
    print(f"  结果: scheduled={scheduled}, preempted={preempted}")
    # seq[0](len=1): needs_block, free=[] → preempt seq[2](len=8, 2 blocks) → free has 2 blocks
    # seq[0] now can allocate → scheduled
    # seq[1](len=4): no_new_block → scheduled
    # seq[2] was preempted
    assert len(scheduled) == 2, f"expected 2, got {len(scheduled)}"
    assert preempted == [2], f"expected seq[2] preempted, got {preempted}"
    print("  [PASS] seq[2] 被 preempt → 回收 blocks → seq[0]/seq[1] 继续 decode")

    # ── 场景 C: len%4 ≠ 1，不需新 block ──
    print(f"\n  场景 C: free_blocks=[], running=[(0, 6)], block_size={block_size}")
    print(f"    seq[0] len=6, 6%4=2 ≠ 1 → 不需新 block → can_append=True")
    running_c = deque([(0, 6)])
    scheduled, preempted, log = simulate_decode_step(running_c, [], block_size)
    for line in log:
        print(line)
    assert len(scheduled) == 1
    assert len(preempted) == 0
    print(f"  [PASS]")

    # ── 场景 D: 只剩自己且需要新 block，自身被 preempt ──
    print(f"\n  场景 D: free_blocks=[], running=[(0, 5)], block_size={block_size}")
    print(f"    seq[0] len=5, 5%4=1 → 需要新 block, free=[] → 自身 preempt")
    running_d = deque([(0, 5)])
    scheduled, preempted, log = simulate_decode_step(running_d, [], block_size)
    for line in log:
        print(line)
    assert len(scheduled) == 0
    assert preempted == [0]
    print("  [PASS] （边界注记：'自身抢占后空批'是模拟器的简化——真实引擎此时无可调度 seq，")
    print("         会在 scheduler.py:L93 的 assert 处中止；两者差异见第 3 课 §3.5）")


# ── 验证 5: preempt 状态机 ────────────────────────────────────────────

def verify_preempt_state_machine():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  5. preempt 双路径状态机（对齐 scheduler.py:L97-L123）        │")
    print("└─────────────────────────────────────────────────────────────┘")

    print("""
    preempt(seq) 现在是两层决策:
      ① 护栏: 本 step 刚 swap_in 的 seq → GPU 块里 KV 还没拷回（是垃圾数据）
              → 取消其 swap-in 映射，直接 _recompute
              （绝不能把垃圾拷去 CPU，覆盖掉真身）
      ② can_swap_out?  = CPU 空块够 且 所有块 ref_count==1（全部独占）
         ├─ Yes → SWAP: 记 {gpu_id: cpu_id} 映射, status=SWAPPED, 入 swapped 队列
         └─ No  → RECOMPUTE（_recompute，即旧版 preempt 四步）:
                  status=WAITING, is_prefill=True, deallocate, waiting.appendleft

    类比 OS: SWAP 路径 = 真正的换出/换入（KV 写到 CPU pinned 内存，回来断点续跑）
            RECOMPUTE = "计算换空间"——LLM 特有: KV 可由 token 序列确定性重算
    开关: cpu_offload_gb>0 启用 swap；=0（默认）时 RECOMPUTE 是唯一出路
    """)


# ── 验证 6: BlockManager swap 元数据往返 ─────────────────────────────

def verify_blockmanager_swap():
    # 断言语义与 tests/test_swap_blockmanager.py（A1/A4/A5）保持一致——两边需同步维护。
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  6. BlockManager swap 元数据往返（对齐 block_manager.py:L126-L180）│")
    print("│     cpu_offload_gb>0 时 preempt 优先走的路径；纯元数据，无需 GPU │")
    print("└─────────────────────────────────────────────────────────────┘")

    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence

    BLOCK = 4
    old_block_size = Sequence.block_size
    Sequence.block_size = BLOCK              # 进程级类属性，函数末尾恢复（§7 会按 Config 重设）
    bm = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    seq = Sequence(list(range(12)))          # 12 token → 3 个 block
    bm.allocate(seq, bm.can_allocate(seq))
    seq.num_cached_tokens = 8                # 模拟 decode 进行中：已缓存 8 token
    gpu_blocks = list(seq.block_table)
    print(f"\n  初始: seq 占 {len(gpu_blocks)} 个 GPU 块 {gpu_blocks}, num_cached_tokens={seq.num_cached_tokens}")

    # ── swap_out：GPU → CPU（记 {gpu_id: cpu_id}，GPU 块归还池，页表清空但缓存数保留）──
    assert bm.can_swap_out(seq)
    out_mapping = bm.swap_out(seq)
    print(f"  swap_out: {out_mapping}")
    print(f"    seq.block_table = {seq.block_table} (清空)")
    print(f"    seq.num_cached_tokens = {seq.num_cached_tokens} (刻意保留 → 断点续跑的凭证)")
    print(f"    GPU 块归还 free 池; CPU 块占用: {sorted(bm.used_cpu_block_ids)}; 账本 swapped_block_tables={bm.swapped_block_tables}")
    assert seq.block_table == []
    assert seq.num_cached_tokens == 8
    assert all(b in bm.free_block_ids for b in gpu_blocks)
    assert set(out_mapping.values()) == bm.used_cpu_block_ids
    assert bm.swapped_block_tables[seq.seq_id] == list(out_mapping.values())

    # ── swap_in：CPU → GPU（分新 GPU 块、重建 block_table、释放 CPU 块）──
    assert bm.can_swap_in(seq)
    in_mapping = bm.swap_in(seq)
    print(f"  swap_in : {in_mapping}")
    print(f"    seq.block_table 重建为 {seq.block_table} (物理 id 可变, 数量/顺序不变)")
    print(f"    CPU 块全部释放: used_cpu={sorted(bm.used_cpu_block_ids)}")
    assert len(seq.block_table) == len(gpu_blocks)
    assert seq.seq_id not in bm.swapped_block_tables
    assert len(bm.used_cpu_block_ids) == 0
    assert seq.num_cached_tokens == 8
    print("  [PASS] 往返后块数不变、num_cached_tokens 保留 → decode 可断点续跑，无需重算")

    # ── can_swap_out 的两条否决规则 ──
    seq2 = Sequence(list(range(12)))
    bm2 = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=2)
    bm2.allocate(seq2, bm2.can_allocate(seq2))
    assert not bm2.can_swap_out(seq2)
    print("\n  can_swap_out 否决 1: CPU 空块不足 (2 < 3) → False（CPU 满时兜底 RECOMPUTE）")

    seq3 = Sequence(list(range(12)))
    bm3 = BlockManager(num_blocks=10, block_size=BLOCK, num_cpu_blocks=10)
    bm3.allocate(seq3, bm3.can_allocate(seq3))
    bm3.blocks[seq3.block_table[0]].ref_count = 2   # 模拟共享前缀块
    assert not bm3.can_swap_out(seq3)
    print("  can_swap_out 否决 2: 含共享块 ref_count=2 → False（共享块不能搬走，整体退回 RECOMPUTE）")
    Sequence.block_size = old_block_size    # 恢复，避免向后续章节泄漏
    print("  [PASS]")


# ── 验证 7: 真实 Scheduler 对比 ──────────────────────────────────────

def verify_with_real_scheduler(model_path):
    """用真实 Scheduler 跑一次，输出和模拟结果对比。"""
    from nanovllm.config import Config
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence, SequenceStatus
    from nanovllm.sampling_params import SamplingParams

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  7. 真实 Scheduler 对比验证                                  │")
    print("│     直接调用 scheduler.add() → schedule() → postprocess()   │")
    print("└─────────────────────────────────────────────────────────────┘")

    # 构造 Config（用真实模型路径，但手动设 num_kvcache_blocks）
    config = Config(model_path, max_num_batched_tokens=1000, max_num_seqs=8,
                    kvcache_block_size=256)
    config.num_kvcache_blocks = 100  # 足够多，避免 preempt 干扰
    Sequence.block_size = config.kvcache_block_size

    scheduler = Scheduler(config)
    print(f"\n  >>> config.max_num_batched_tokens = {config.max_num_batched_tokens}")
    print(f"  >>> config.max_num_seqs = {config.max_num_seqs}")
    print(f"  >>> config.kvcache_block_size = {config.kvcache_block_size}")
    print(f"  >>> config.num_kvcache_blocks = {config.num_kvcache_blocks}")

    # ── 场景: 3 条 seq，max_batch=1000 ──
    print(f"\n  ▸ 场景: 3 条 seq (200, 300, 400 tokens), max_batch=1000")
    sp = SamplingParams(temperature=0.6, max_tokens=64)

    # 构造 prompt token_ids
    seqs = [
        Sequence(list(range(200)), sp),
        Sequence(list(range(300)), sp),
        Sequence(list(range(400)), sp),
    ]

    print(f"    创建 Sequence:")
    for i, s in enumerate(seqs):
        print(f"      seq[{i}]: num_tokens={s.num_tokens}, status={s.status.name}")

    # add 到 scheduler
    for s in seqs:
        scheduler.add(s)
    print(f"    waiting 队列长度: {len(scheduler.waiting)}, running: {len(scheduler.running)}")

    # 调用真实的 schedule()
    scheduled_seqs, is_prefill = scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in scheduled_seqs) if is_prefill else -len(scheduled_seqs)

    print(f"\n  >>> scheduled_seqs, is_prefill = scheduler.schedule()")
    print(f"      is_prefill = {is_prefill}")
    print(f"      调度了 {len(scheduled_seqs)} 条 seq:")
    for i, s in enumerate(scheduled_seqs):
        print(f"        seq[{i}]: num_tokens={s.num_tokens}, "
              f"num_scheduled_tokens={s.num_scheduled_tokens}, "
              f"num_cached_tokens={s.num_cached_tokens}, "
              f"status={s.status.name}")

    # 对比模拟
    print(f"\n  ▸ 模拟 vs 真实对比:")
    sim_result, _ = simulate_prefill([200, 300, 400], max_num_batched_tokens=1000)
    print(f"    模拟预期: scheduled={sim_result}")
    real_result = [(i, s.num_scheduled_tokens) for i, s in enumerate(scheduled_seqs)]
    print(f"    真实调度: scheduled={real_result}")

    assert len(scheduled_seqs) == 3, f"应调度 3 条, 实际 {len(scheduled_seqs)}"
    assert real_result == sim_result, f"模拟 {sim_result} ≠ 真实 {real_result}"
    print(f"    [PASS] 真实 Scheduler 输出和模拟一致")

    # ── 跑 postprocess + 下一轮 decode ──
    print(f"\n  ▸ 模拟一轮 postprocess + decode:")
    # 假 token_ids（随机）
    token_ids = [100 + i for i in range(len(scheduled_seqs))]
    scheduler.postprocess(scheduled_seqs, token_ids, is_prefill)

    print(f"    postprocess 后:")
    print(f"      waiting: {len(scheduler.waiting)}, running: {len(scheduler.running)}")
    for i, s in enumerate(seqs):
        print(f"      seq[{i}]: status={s.status.name}, num_tokens={s.num_tokens}")

    # 下一轮 decode
    scheduled_seqs2, is_prefill2 = scheduler.schedule()
    print(f"\n    下一轮 schedule(): is_prefill={is_prefill2}, seqs={len(scheduled_seqs2)}")
    for i, s in enumerate(scheduled_seqs2):
        print(f"      seq: num_scheduled_tokens={s.num_scheduled_tokens} (应为 1, decode)")
        assert s.num_scheduled_tokens == 1
    print(f"    [PASS] decode 每 seq 处理 1 token")

    # ── 场景 E: 强制抢占下的 SWAP 全流程（cpu_offload_gb>0 时 preempt 的优先出路）──
    print(f"\n  ▸ 场景 E: 开启 CPU 卸载后强制抢占 —— 真实引擎里的 swap_out → swap_in")
    config_e = Config(model_path, max_num_batched_tokens=4096, max_num_seqs=8,
                      kvcache_block_size=256)
    config_e.num_kvcache_blocks = 2       # 只给 2 个 GPU block，逼出抢占
    config_e.num_cpu_kvcache_blocks = 8   # 模拟 cpu_offload_gb>0（实际由 ModelRunner 按显存换算写回）
    config_e.eos = 0                      # 让 postprocess 能判定"完成"，从而释放 GPU block
    sched = Scheduler(config_e)
    print(f"    num_kvcache_blocks=2, num_cpu_kvcache_blocks=8（等价 cpu_offload_gb>0）, eos=0")

    a = Sequence(list(range(256)), sp)    # 恰好占满 1 个 block（256 token）
    b = Sequence(list(range(256)), sp)
    for s in (a, b):
        sched.add(s)
    p_seqs, p_flag = sched.schedule()     # prefill 两条 → 占满 2 个 GPU block
    sched.postprocess(p_seqs, [7, 8], p_flag)
    print(f"    prefill 后: running={len(sched.running)}, "
          f"free_gpu_blocks={len(sched.block_manager.free_block_ids)}")
    print(f"    两条 seq 长度都到 {len(a)}（= 256+1）：下一次 decode 需要新 block，但 GPU 已满")

    d_seqs, d_flag = sched.schedule()     # decode：队尾被抢占 → SWAP
    print(f"    decode 调度: scheduled={[s.seq_id for s in d_seqs]}, is_prefill={d_flag}")
    print(f"      blocks_to_swap_out={sched.blocks_to_swap_out} (gpu→cpu), "
          f"swapped 队列={[s.seq_id for s in sched.swapped]}")
    print(f"      被抢占 seq[{b.seq_id}]: status={b.status.name}, len={len(b)}, "
          f"num_cached_tokens={b.num_cached_tokens}（原样保留 → 回来断点续跑）")
    assert d_flag is False and [s.seq_id for s in d_seqs] == [a.seq_id]
    assert b.status == SequenceStatus.SWAPPED and b in sched.swapped
    # len(seq)=257 但只缓存了 256 个 token 的 KV（最后生成的 token 要等下一轮 decode 才算）
    assert sched.blocks_to_swap_out and b.num_cached_tokens == 256
    sched.postprocess(d_seqs, [9], d_flag)

    d2_seqs, d2_flag = sched.schedule()   # A 继续 decode；GPU 仍无空块 → B 还迁不回来
    print(f"    下一轮: scheduled={[s.seq_id for s in d2_seqs]}；B 仍在 swapped"
          f"（can_swap_in=False，GPU 无空块）: {[s.seq_id for s in sched.swapped]}")
    assert not sched.blocks_to_swap_in

    sched.postprocess(d2_seqs, [config_e.eos], d2_flag)   # A 命中 EOS → 完成，释放全部 block
    print(f"    A 命中 EOS 完成 → 释放 GPU block: "
          f"free={len(sched.block_manager.free_block_ids)}")

    d3_seqs, d3_flag = sched.schedule()   # swap-in 阶段把 B 迁回，继续 decode
    print(f"    B 迁回: blocks_to_swap_in={sched.blocks_to_swap_in} (cpu→gpu), "
          f"status={b.status.name}, num_cached_tokens={b.num_cached_tokens}")
    print(f"      计数器: swap_out={sched.num_swapped_out_blocks}, "
          f"swap_in={sched.num_swapped_in_blocks}, recompute={sched.num_recompute_preemptions}")
    assert sched.blocks_to_swap_in and b.status == SequenceStatus.RUNNING
    assert [s.seq_id for s in d3_seqs] == [b.seq_id] and b.num_cached_tokens == 256
    print("    [PASS] SWAP 全流程：抢占搬出 → 等空块 → 迁回断点续跑（零重算）")

    # ── 场景 E2（对照）: 同样的压力，关掉 CPU 卸载 → preempt 只能 RECOMPUTE ──
    print(f"\n  ▸ 场景 E2（对照）: num_cpu_kvcache_blocks=0（cpu_offload_gb=0 的真实取值）")
    config_r = Config(model_path, max_num_batched_tokens=4096, max_num_seqs=8,
                      kvcache_block_size=256)
    config_r.num_kvcache_blocks = 2
    config_r.num_cpu_kvcache_blocks = 0
    config_r.eos = 0
    sched_r = Scheduler(config_r)
    a2 = Sequence(list(range(256)), sp)
    b2 = Sequence(list(range(256)), sp)
    for s in (a2, b2):
        sched_r.add(s)
    p2_seqs, p2_flag = sched_r.schedule()
    sched_r.postprocess(p2_seqs, [7, 8], p2_flag)
    r_seqs, r_flag = sched_r.schedule()
    print(f"    被抢占 seq[{b2.seq_id}]: status={b2.status.name}, "
          f"num_cached_tokens={b2.num_cached_tokens}（归零，下一轮重算）")
    assert b2.status == SequenceStatus.WAITING and b2 in sched_r.waiting
    assert b2.num_cached_tokens == 0
    assert not sched_r.blocks_to_swap_out          # 没有 swap：KV 直接被丢弃
    assert sched_r.num_recompute_preemptions == 1
    print("    [PASS] 同一压力、两条出路：SWAP 搬 KV（保留断点）vs RECOMPUTE 丢 KV（计数归零）")


def main():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NANOVLLM_MODEL_PATH", "")
    if not model_path:
        print("用法: python L03_scheduler.py <model_path>", file=sys.stderr)
        print("或设置环境变量: export NANOVLLM_MODEL_PATH=/path/to/model", file=sys.stderr)
        sys.exit(1)
    model_path = os.path.expanduser(model_path)

    verify_basic_batching()
    verify_chunked_prefill_constraint()
    verify_prefix_cache_batching()
    verify_decode_and_preempt()
    verify_preempt_state_machine()
    verify_blockmanager_swap()
    verify_with_real_scheduler(model_path)

    print("\n" + "=" * 64)
    print("L03 全部断言通过 ✓")
    print("=" * 64)


if __name__ == "__main__":
    main()
