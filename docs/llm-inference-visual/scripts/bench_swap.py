#!/usr/bin/env python3
"""
RECOMPUTE vs SWAP 吞吐压测（需 GPU）。

用法：
  python bench_swap.py <model_path>

在显存压力下（用 num_kvcache_blocks 上限强制抢占）对比两种抢占策略：
  - RECOMPUTE：抢占丢弃 KV，下轮重算 prefill（cpu_offload_gb=0）
  - SWAP     ：抢占把 KV 搬到 CPU，空出后搬回（cpu_offload_gb>0）
外加一个"无压力"基准（大 cap，不抢占）作为吞吐上限。

诚实提示：Qwen3-0.6B 很小，prefill 重算很快，而 swap 要走 PCIe（每块 ~28MB）。
小模型上 SWAP 的墙钟未必赢 RECOMPUTE——swap 的收益随模型规模/prompt 长度增长。
因此本脚本同时报「墙钟吞吐」与「重算次数 / 搬运块数」两类指标。
"""
import os
import sys
import queue
import multiprocessing as mp


def _bench_child(model_path, label, cap, cpu_offload_gb, n_seqs, prompt_repeat, max_tokens, q):
    try:
        import time
        from nanovllm import LLM, SamplingParams
        prompt = "The history of computing began with " * prompt_repeat
        kw = dict(enforce_eager=True, tensor_parallel_size=1)
        if cap is not None:
            kw["num_kvcache_blocks"] = cap
        if cpu_offload_gb:
            kw["cpu_offload_gb"] = cpu_offload_gb
        llm = LLM(model_path, **kw)
        prompts = [prompt] * n_seqs
        sp = SamplingParams(temperature=0.7, max_tokens=max_tokens, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t
        res = dict(
            label=label,
            wall=dt,
            out_tokens=sum(len(o["token_ids"]) for o in outs),
            recompute=llm.scheduler.num_recompute_preemptions,
            swap_out=llm.scheduler.num_swapped_out_blocks,
            swap_in=llm.scheduler.num_swapped_in_blocks,
            kv_blocks=llm.model_runner.config.num_kvcache_blocks,
        )
        llm.exit()
        q.put(("ok", res))
    except Exception:
        import traceback
        q.put(("err", traceback.format_exc()))


def _run(model_path, label, cap, cpu_offload_gb, n_seqs, prompt_repeat, max_tokens):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_bench_child,
                    args=(model_path, label, cap, cpu_offload_gb, n_seqs, prompt_repeat, max_tokens, q))
    p.start()
    try:
        status, payload = q.get(timeout=1200)
    except queue.Empty:
        p.terminate()
        p.join()
        raise RuntimeError(f"[{label}] 子进程 1200s 无输出（疑似卡死/OOM）")
    p.join()
    if status == "err":
        raise RuntimeError(f"[{label}] 失败:\n" + payload)
    return payload


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NANOVLLM_MODEL_PATH", "")
    if not model_path:
        print("用法: python bench_swap.py <model_path>", file=sys.stderr)
        sys.exit(1)
    model_path = os.path.expanduser(model_path)
    assert os.path.isdir(model_path), f"模型路径不存在: {model_path}"

    # 负载：prompt≈210 tok（1 块）× 32 序列，各解码 256 token；小 cap 制造持续抢占。
    # 关键：并发工作集必须超过 cap 才会抢占——1 块的 prompt + 多序列 + 小 cap 才能持续触发
    # （2 块的长 prompt 会降低并发数，反而不抢占，见 bench 早期教训）。
    N_SEQS, PROMPT_REPEAT, MAX_TOKENS = 32, 30, 256
    PRESSURE_CAP = 8           # 强制持续抢占
    BASELINE_CAP = 512         # 充足，不抢占

    print("=" * 78)
    print(f"RECOMPUTE vs SWAP 压测  |  模型={os.path.basename(model_path.rstrip('/'))}")
    print(f"负载: {N_SEQS} 序列 × (prompt≈{PROMPT_REPEAT*7} tok + {MAX_TOKENS} decode)，"
          f"压力 cap={PRESSURE_CAP} 块 / 基准 cap={BASELINE_CAP} 块")
    print("=" * 78)

    configs = [
        ("baseline(无压力)", BASELINE_CAP, 0),
        ("RECOMPUTE",        PRESSURE_CAP, 0),
        ("SWAP",             PRESSURE_CAP, 4),
    ]
    rows = []
    for label, cap, off in configs:
        r = _run(model_path, label, cap, off, N_SEQS, PROMPT_REPEAT, MAX_TOKENS)
        r["tps"] = r["out_tokens"] / r["wall"]
        rows.append(r)
        print(f"[done] {label:18s} wall={r['wall']:6.2f}s  "
              f"tps={r['tps']:7.1f}  kv_blocks={r['kv_blocks']}  "
              f"recompute={r['recompute']}  swap_out={r['swap_out']}  swap_in={r['swap_in']}")

    print("-" * 78)
    print(f"{'配置':18s} {'墙钟(s)':>9} {'吞吐(tok/s)':>12} {'重算抢占':>9} {'swap_out块':>11} {'swap_in块':>10}")
    for r in rows:
        print(f"{r['label']:18s} {r['wall']:9.2f} {r['tps']:12.1f} "
              f"{r['recompute']:9d} {r['swap_out']:11d} {r['swap_in']:10d}")
    print("-" * 78)

    base, rec, swp = rows
    print(f"\n解读:")
    print(f"  · baseline 无抢占,是吞吐上限 ({base['tps']:.1f} tok/s)。")
    print(f"  · 压力下 RECOMPUTE 触发 {rec['recompute']} 次重算抢占；SWAP 触发 0 次重算、"
          f"改为搬运 {swp['swap_out']}/{swp['swap_in']} 块。")
    if swp['tps'] >= rec['tps']:
        print(f"  · SWAP 吞吐 {swp['tps']:.1f} ≥ RECOMPUTE {rec['tps']:.1f}：省下的重算 > PCIe 搬运成本。")
    else:
        print(f"  · SWAP 吞吐 {swp['tps']:.1f} < RECOMPUTE {rec['tps']:.1f}：本模型太小，"
              f"PCIe 搬运成本 > 省下的重算——swap 的收益需更大模型/更长 prompt 才体现。")


if __name__ == "__main__":
    main()
