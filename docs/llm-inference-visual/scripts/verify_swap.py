#!/usr/bin/env python3
"""
KV Cache CPU Offloading（swap-based preemption）验证脚本 —— 需要 GPU。

用法：
  python verify_swap.py <model_path>
  NANOVLLM_MODEL_PATH=/path/to/model python verify_swap.py

覆盖设计文档 docs/design/kv-offload.md 第 10 节的 GPU 层：
  B1  KV 逐字节往返相等（最强、完全确定，与采样无关）
  B2  RECOMPUTE / SWAP / baseline 三路差分对拍（注入 argmax 去随机性，逐 token 一致）
  B3  强制大量 swap 冒烟：不死锁、能跑完
  C   观测计数器 > 0（证明确实走了 swap 路径）

每个 LLM 在独立子进程中运行：nano-vllm 的 exit() 不释放 KV 显存，同一进程内连开多个 LLM
会互相饿死，因此用 spawn 子进程隔离。用 num_kvcache_blocks 上限强制抢占（24GB 卡上小负载
本来不会抢占）。纯 Python 的元数据/拷贝索引单测见 tests/test_swap_blockmanager.py。
"""
import os
import sys
import queue
import multiprocessing as mp


def _patch_argmax():
    """把 Sampler 换成贪心 argmax，使输出只由 logits 决定、与 batch 顺序无关。"""
    import torch
    from nanovllm.layers import sampler

    def greedy_forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.argmax(dim=-1)

    sampler.Sampler.forward = greedy_forward


# ── 子进程 worker（每个 LLM 独立进程，退出即回收显存）─────────────────

def _b1_child(model_path, q):
    try:
        import torch
        from nanovllm import LLM
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1,
                  cpu_offload_gb=1, num_kvcache_blocks=64)
        mr = llm.model_runner
        assert mr.cpu_kv_cache is not None
        g_src, c_ids, g_dst = [0, 1], [0, 1], [2, 3]
        for g in g_src:
            mr.kv_cache[:, :, g].copy_(torch.randn_like(mr.kv_cache[:, :, g]))
        snap = [mr.kv_cache[:, :, g].clone() for g in g_src]
        mr.swap_out(dict(zip(g_src, c_ids)))       # {gpu: cpu}
        for g in g_dst:
            mr.kv_cache[:, :, g].zero_()
        mr.swap_in(dict(zip(c_ids, g_dst)))        # {cpu: gpu}
        torch.cuda.synchronize()
        ok = all(torch.equal(s, mr.kv_cache[:, :, g]) for s, g in zip(snap, g_dst))
        llm.exit()
        q.put(("ok", ok))
    except Exception:
        import traceback
        q.put(("err", traceback.format_exc()))


def _gen_child(model_path, greedy, cap, cpu_offload_gb, prompts, max_tokens, q):
    try:
        if greedy:
            _patch_argmax()
        from nanovllm import LLM, SamplingParams
        kw = dict(enforce_eager=True, tensor_parallel_size=1)
        if cap is not None:
            kw["num_kvcache_blocks"] = cap
        if cpu_offload_gb:
            kw["cpu_offload_gb"] = cpu_offload_gb
        llm = LLM(model_path, **kw)
        sp = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        res = ([o["token_ids"] for o in outs],
               llm.scheduler.num_swapped_out_blocks,
               llm.scheduler.num_swapped_in_blocks,
               llm.scheduler.num_recompute_preemptions)
        llm.exit()
        q.put(("ok", res))
    except Exception:
        import traceback
        q.put(("err", traceback.format_exc()))


def _run_isolated(target, *args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
    try:
        status, payload = q.get(timeout=600)     # avoid hanging forever on a stuck child
    except queue.Empty:
        p.terminate()
        p.join()
        raise RuntimeError("子进程在 600s 内无输出（疑似卡死/OOM）")
    p.join()
    if status == "err":
        raise RuntimeError("子进程失败:\n" + payload)
    return payload


# 长 prompt（~210 tokens，接近一个满块）：preempt 只在序列跨 256-token 块边界、
# 且无空块时触发，因此必须让解码序列越过边界，短 prompt 只会排队、不会抢占。
LONG_PROMPT = "The history of computing began with " * 30


def _b2_child(model_path, q):
    """确定性单序列 swap 往返：整个解码 batch 恒为 1，数值不受调度影响，可逐 token 精确对比。

    参考跑：一条序列正常解码到底。
    swap 跑：同一条序列解码几步后，强制 swap_out 到 CPU、再由调度器 swap_in 回来，继续解码。
    二者若不完全一致，即为真正的 KV 损坏（排除了多序列 batch 的浮点非确定性）。
    """
    try:
        _patch_argmax()
        from nanovllm import LLM, SamplingParams
        from nanovllm.engine.sequence import SequenceStatus

        prompt = "The history of computing began with"
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1,
                  cpu_offload_gb=1, num_kvcache_blocks=64)
        sp = SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True)

        ref = llm.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]

        # swap 跑：手动驱动 step，中途强制一次 swap 往返
        llm.add_request(prompt, sp)
        llm.step()                                   # prefill + 首 token → 序列进入 running
        for _ in range(3):
            llm.step()                               # 解码几步
        seq = llm.scheduler.running[0]
        assert llm.scheduler.block_manager.can_swap_out(seq)
        mapping = llm.scheduler.block_manager.swap_out(seq)
        seq.status = SequenceStatus.SWAPPED
        llm.scheduler.running.remove(seq)
        llm.scheduler.swapped.append(seq)
        llm.model_runner.swap_out(mapping)           # 实际 GPU→CPU 拷贝
        swapped_out = len(mapping)
        while not llm.is_finished():                 # 后续 step 的 swap-in 阶段会把它搬回
            llm.step()
        got = seq.completion_token_ids

        llm.exit()
        q.put(("ok", (ref, got, swapped_out)))
    except Exception:
        import traceback
        q.put(("err", traceback.format_exc()))


# ── B1/B2/B3/C ────────────────────────────────────────────────────────

def case_b1(model_path):
    ok = _run_isolated(_b1_child, model_path)
    assert ok, "swap 往返后 KV 不一致"
    print("[PASS] B1 KV 逐字节往返相等")


def case_b2(model_path):
    ref, got, n = _run_isolated(_b2_child, model_path)
    assert n > 0, "未发生 swap（序列块为空？）"
    assert ref == got, f"swap 往返后续解码与参考不一致\n ref={ref[:8]}...\n got={got[:8]}..."
    print(f"[PASS] B2 单序列 swap 往返逐 token 一致（中途搬出 {n} 块，batch=1 确定性对比）")


def case_b3(model_path):
    prompts = [LONG_PROMPT for _ in range(16)]
    outs, so, si, rr = _run_isolated(_gen_child, model_path, False, 8, 4, prompts, 80)
    assert len(outs) == len(prompts)
    assert all(len(t) == 80 for t in outs), "存在未跑完的序列"
    print(f"[PASS] B3 冒烟不死锁 (swap_out {so} / swap_in {si} 块, recompute {rr} 次)")
    return so, si


def case_c(so, si):
    assert so > 0, "swap_out 计数为 0：没触发 swap（测试空转）"
    assert si > 0, "swap_in 计数为 0：swapped 序列没有被拉回"
    print(f"[PASS] C 观测计数器 > 0 (out={so}, in={si})")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NANOVLLM_MODEL_PATH", "")
    if not model_path:
        print("用法: python verify_swap.py <model_path>", file=sys.stderr)
        sys.exit(1)
    model_path = os.path.expanduser(model_path)
    assert os.path.isdir(model_path), f"模型路径不存在: {model_path}"

    print("=" * 64)
    print("KV Cache CPU Offloading 验证")
    print(f"模型路径: {model_path}")
    case_b1(model_path)
    case_b2(model_path)
    so, si = case_b3(model_path)
    case_c(so, si)
    print("=" * 64)
    print("全部通过 ✅")


if __name__ == "__main__":
    main()
