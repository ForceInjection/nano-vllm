def swap_blocks(src, dst, mapping: dict[int, int]):
    """Copy KV blocks between two caches shaped (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim).

    `mapping` maps a src block id to a dst block id. A single `[:, :, block_id]` slice covers
    all layers and both K/V for that block, so one copy_ per pair moves the whole block.
    Direction is expressed purely by argument order:
      - swap out (GPU -> CPU): swap_blocks(gpu_cache, cpu_cache, {gpu_id: cpu_id})
      - swap in  (CPU -> GPU): swap_blocks(cpu_cache, gpu_cache, {cpu_id: gpu_id})

    Kept import-free (no torch) so it can be unit-tested with plain CPU tensors without pulling
    the model/flash-attn stack.
    """
    for src_id, dst_id in mapping.items():
        dst[:, :, dst_id].copy_(src[:, :, src_id])
