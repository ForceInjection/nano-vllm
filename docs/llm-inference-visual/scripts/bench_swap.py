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
        sp = SamplingParams(temperature=0.7, max_tokens=max_tokens, ignore_eos=True)
        for _ in range(n_seqs):
            llm.add_request(prompt, sp)

        # Drive step() manually to collect concurrency + prefill work:
        #   step() returns (outputs, num_tokens); num_tokens > 0 on a prefill step (that many
        #   prefill tokens, INCLUDING recomputes), < 0 on a decode step (= -batch_size).
        prefill_tokens = decode_tokens = decode_steps = 0
        t = time.perf_counter()
        while not llm.is_finished():
            _, num_tokens = llm.step()
            if num_tokens > 0:
                prefill_tokens += num_tokens
            else:
                decode_tokens += -num_tokens
                decode_steps += 1
        dt = time.perf_counter() - t

        out_tokens = n_seqs * max_tokens                      # ignore_eos → fixed useful output
        res = dict(
            label=label,
            wall=dt,
            tps=out_tokens / dt,
            avg_batch=decode_tokens / decode_steps if decode_steps else 0.0,
            prefill_tokens=prefill_tokens,                    # total prefill work (recomputes inflate this)
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
        rows.append(r)
        print(f"[done] {label:18s} wall={r['wall']:6.2f}s  tps={r['tps']:7.1f}  "
              f"avg_batch={r['avg_batch']:5.1f}  prefill_tok={r['prefill_tokens']:7d}  "
              f"recompute={r['recompute']}  swap={r['swap_out']}/{r['swap_in']}")

    print("-" * 92)
    hdr = f"{'配置':18s} {'墙钟(s)':>8} {'吞吐(tok/s)':>11} {'并发(avg_batch)':>15} {'prefill_tok':>12} {'重算':>5} {'swap_o/i':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['label']:18s} {r['wall']:8.2f} {r['tps']:11.1f} {r['avg_batch']:15.1f} "
              f"{r['prefill_tokens']:12d} {r['recompute']:5d} {str(r['swap_out'])+'/'+str(r['swap_in']):>10}")
    print("-" * 92)

    base, rec, swp = rows
    print(f"\n解读:")
    print(f"  1. baseline 是「无抢占」参照(cap 充足 → avg_batch≈{base['avg_batch']:.0f}、prefill 只跑一遍)，"
          f"不参与 swap 对比——它并发更高,快是因为 batch 大,不是因为抢占策略。")
    print(f"  2. 公平对比 = RECOMPUTE vs SWAP:二者 **cap 相同、avg_batch≈{rec['avg_batch']:.0f} 一致**,"
          f"只差抢占策略。")
    print(f"  3. 机制层面 swap 恒赢(与模型大小无关):RECOMPUTE 因重跑 prefill,总 prefill token "
          f"= {rec['prefill_tokens']}(baseline 的 {rec['prefill_tokens']/max(base['prefill_tokens'],1):.1f}×)；"
          f"SWAP = {swp['prefill_tokens']}(≈baseline,不重跑)。")
    if swp['tps'] >= rec['tps']:
        print(f"  4. 墙钟:SWAP {swp['tps']:.1f} ≥ RECOMPUTE {rec['tps']:.1f} tok/s——省下的重算 > PCIe 搬运。")
    else:
        print(f"  4. 墙钟:SWAP {swp['tps']:.1f} ≈/< RECOMPUTE {rec['tps']:.1f} tok/s——本模型太小,"
              f"PCIe 搬运成本 ≈ 省下的重算;收益随模型规模/prompt 长度增长。")


if __name__ == "__main__":
    main()
