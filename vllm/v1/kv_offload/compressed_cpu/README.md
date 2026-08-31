# Compressed CPU offloading prototype

This package prototypes fixed-size compression inside vLLM's native
`OffloadingConnector`. It does not modify scheduler keys or the connector.

## Contract

```text
Store: paged GPU KV -> gather -> compress -> encoded D2H -> fixed CPU slot
Load:  fixed CPU slot -> encoded H2D -> decompress -> scatter -> paged GPU KV
```

`CompressedCPUOffloadingWorker` owns the common transfer and CUDA-event
lifecycle. INT4, SVD, and raw child workers implement the physical codec.
`GroupedCompressedCPUOffloadingWorker` splits one scheduler job across those
children and reports the parent job only after every participating group ends.

Each CPU object contains a versioned 64-byte header and a fixed-size payload.
`CPUOffloadingManager` continues to own keys, slot IDs, eviction, and in-flight
references. Hybrid layouts use one child manager and fixed-row mmap arena per
group. `GroupedCPULoadStoreSpec` carries `(group_idx, local_slot_id)`, so local
slot IDs can overlap safely. Capacity is calculated from the sum of encoded
group-row sizes across all workers.

`KVCompressionGroup` binds one KV cache group to its matrix layout and
`KVCompressor`. `GroupedKVCompressor` dispatches by `group_idx`, allowing, for
example, one group to use SVD and another to use INT4 without sharing an encoded
layout. Attention groups default to the selected compressed spec. Mamba/GDN
groups default to the exact `raw-bytes-v1` codec and store one boundary state
page per logical prefix object.

## Configuration

Use the INT4 worker with 16 GPU pages per offload unit:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "INT4CompressedCPUOffloadingSpec",
    "spec_module_path": "vllm.v1.kv_offload.compressed_cpu.spec",
    "blocks_per_chunk": 16,
    "cpu_bytes_to_use": 68719476736,
    "int4_group_size": 64,
    "compression_log_operations": true
  }
}
```

Override individual group codecs when needed:

```json
{
  "compression_groups": {
    "0": {"codec": "raw"},
    "3": {"codec": "svd", "svd_rank": 64, "svd_factor_dtype": "fp16"}
  }
}
```

Valid codec names are `raw`, `int4`, and `svd`. Group indices and cache kinds
are printed as `[compressed-offload][group-plan]` during startup.

For SVD, replace the algorithm-specific fields:

```json
{
  "spec_name": "SVDCompressedCPUOffloadingSpec",
  "spec_module_path": "vllm.v1.kv_offload.compressed_cpu.spec",
  "svd_rank": 32
}
```

The token span is `blocks_per_chunk * tokens_per_block`. Qwen3.5-9B uses
528-token aligned pages, so 16 pages represent 8,448 tokens. Incomplete tails
remain in vLLM's GPU cache.

## Current boundary

- Non-packed multi-group cache layouts are supported. Packed hybrid layouts
  fail closed because their bytes cannot yet be assigned to isolated arenas.
- INT4 and SVD sources must be FP16, BF16, or FP32. Raw state sources use uint8
  views over their exact physical bytes.
- Stores must contain complete chunk-aligned objects. Loads may restore only a
  suffix of an object, but the worker still transfers and decompresses the full
  compressed object.
- The native scheduler requests an external load only when at least one whole
  offload chunk remains after the GPU prefix-cache hit. For an 8K chunk, a GPU
  hit ending inside that same 8K range therefore suppresses the external load,
  even though the worker's scatter path accepts a page-aligned suffix. A fully
  evicted 8K range is loadable without scheduler changes.
- The default matrix count is the number of layers in each group. A worker
  accepts either one canonical reference per matrix or one reference containing
  all matrices contiguously.
- INT4 uses groupwise symmetric quantization with one FP16 scale per group.
- SVD uses `torch.linalg.svd`, FP8 U/B factors, FP16 singular values, and an
  unfused FP32 reconstruction. It is a correctness baseline, not a performance
  implementation.
- Scratch tensors are allocated per submitted job. A bounded persistent scratch
  ring and fused kernels are follow-up work.
- Tiering and Mooncake are not integrated in this prototype.

## Validation

Run the focused codec, capacity, mmap, and CUDA worker tests with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  --confcutdir=tests/v1/kv_offload \
  tests/v1/kv_offload/test_compressed_cpu.py
```

The tests cover INT4, SVD, and raw round trips, physical header validation,
group routing, overlapping local slot IDs, fixed-slot capacity, full loads,
partial suffix loads, and mmap-backed hybrid transfers. A Qwen3.5-9B TP4
server test exercised a full 8,448-token store, GPU eviction, external hit,
H2D, decode, and scatter across three raw GDN groups plus one INT4 attention
group. A controlled quality benchmark is still required before treating lossy
codecs as production-safe.
