import functools
import math
import time
import torch
import flashinfer
import pandas as pd
from typing import List, Tuple, Dict, Optional
from flashinfer.utils import MaskMode

# Add triton import for benchmarking
try:
    from triton.testing import do_bench
except ImportError:
    # Fallback timing function if triton is not available
    import time
    def do_bench(fn, warmup=25, rep=100):
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(rep):
            fn()
        torch.cuda.synchronize()
        end = time.time()
        
        return (end - start) * 1000 / rep  # Convert to ms


def get_variant_decl_dump_logits(num_qo_heads, fixed_max_kv_len):
    """
    Generate variant declaration with a fixed maximum KV length.
    This allows reusing the same compiled kernel across different batches.
    """
    variant_decl_dump_logits = f"""
    struct DumpLogits : AttentionVariantBase {{
    static constexpr bool use_softmax = true;

    uint32_t window_left, qo_len, kv_len;
    float sm_scale_log2;

    // Create closure
    template <typename Params>
    __device__ __host__ DumpLogits(const Params& params, uint32_t batch_idx,
                                    uint8_t* smem_ptr) {{
        qo_len = params.get_qo_len(batch_idx);
        kv_len = params.get_kv_len(batch_idx);
        window_left = kv_len;
        sm_scale_log2 = params.sm_scale * math::log2e;
    }}

    REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {{
        if (qo_idx == (qo_len-1) && kv_idx < kv_len) {{
        // Note: Using fixed_max_kv_len here instead of actual max_kv_len
        params.output_logits[batch_idx * {num_qo_heads} * {fixed_max_kv_len} +
                            qo_head_idx * {fixed_max_kv_len} +
                            kv_idx] = logits;
        }}
        return logits;
    }});
    }};"""
    return variant_decl_dump_logits


def benchmark_batch_dump_logits(
    seq_len_configs, 
    num_qo_heads=32, 
    num_kv_heads=8, 
    head_dim=128, 
    page_block_size=1, 
    device="cuda",
    fixed_max_kv_len: Optional[int] = None
):
    """
    Benchmark flashinfer.BatchPrefillWithPagedKVCacheWrapper with and without JIT args.
    
    Args:
        seq_len_configs: List of (qo_len, kv_len) tuples representing requests to batch together
        num_qo_heads: Number of query heads
        num_kv_heads: Number of key-value heads  
        head_dim: Dimension of each head
        page_block_size: Size of each page block
        device: Device to run on
        fixed_max_kv_len: Fixed maximum KV length to use for buffer allocation.
                         If None, uses the actual max from the batch.
    
    Returns:
        Dict with timing results for both configurations
    """
    dev = torch.device(device)
    
    # Extract sequence lengths
    qo_lens = [config[0] for config in seq_len_configs]
    kv_lens = [config[1] for config in seq_len_configs]
    actual_max_kv_len = max(kv_lens)
    batch_size = len(seq_len_configs)
    
    # Use fixed max if provided, otherwise use actual max
    buffer_max_kv_len = fixed_max_kv_len if fixed_max_kv_len is not None else actual_max_kv_len
    
    # Validate that fixed max is sufficient
    if fixed_max_kv_len is not None and actual_max_kv_len > fixed_max_kv_len:
        raise ValueError(f"Actual max KV length ({actual_max_kv_len}) exceeds fixed max ({fixed_max_kv_len})")
    
    # Setup indices and data structures
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
    
    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0).int()
    
    total_blocks = kv_indptr[-1].item()
    
    # Create query data
    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, dtype=torch.float16, device=dev
    )
    
    # Create KV data
    kv_data = torch.randn(
        total_blocks,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=dev,
    )
    
    last_page_len = (seq_lens - 1) % page_block_size + 1
    kv_indices = torch.arange(total_blocks, device=dev).int()
    
    # Define JIT args for dump logits variant with fixed max
    variant_decl = get_variant_decl_dump_logits(num_qo_heads, buffer_max_kv_len)
    
    jit_args = (
        "batch_prefill_dump_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        head_dim,  # hidden_dim_qk
        head_dim,  # hidden_dim_vo
        ["output_logits"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale", "max_kv_len"],  # additional_scalar_names
        ["double", "int64_t"],  # additional_scalar_dtypes
        "DumpLogits",
        variant_decl,
    )
    
    # Setup workspace buffers
    workspace_size = 128 * 1024 * 1024
    float_workspace_buffer_jit = torch.empty(workspace_size, dtype=torch.uint8, device=dev)
    float_workspace_buffer_baseline = torch.empty(workspace_size, dtype=torch.uint8, device=dev)
    
    # === JIT Version ===
    wrapper_jit = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer_jit, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    
    wrapper_jit.plan(
        q_indptr.to(dev),
        kv_indptr.to(dev),
        kv_indices,
        last_page_len.to(dev),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
    )
    
    # Prepare logits buffer with fixed max size
    logits = torch.empty(batch_size * num_qo_heads, buffer_max_kv_len, dtype=torch.float32, device=dev)
    scale = 1.0 / math.sqrt(head_dim)
    
    # === Baseline Version ===
    wrapper_baseline = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer_baseline, kv_layout="NHD", backend="fa2"
    )
    
    wrapper_baseline.plan(
        q_indptr.to(dev),
        kv_indptr.to(dev),
        kv_indices,
        last_page_len.to(dev),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
    )
    
    # Warmup both versions
    for _ in range(5):
        # Pass buffer_max_kv_len to the kernel
        _ = wrapper_jit.run(q, kv_data, logits, scale, buffer_max_kv_len, return_lse=True)
        _ = wrapper_baseline.run(q, kv_data, return_lse=True)
    
    torch.cuda.synchronize()
    
    # Benchmark JIT version
    jit_time = do_bench(
        lambda: wrapper_jit.run(q, kv_data, logits, scale, buffer_max_kv_len, return_lse=True)
    )
    
    # Benchmark baseline version
    baseline_time = do_bench(
        lambda: wrapper_baseline.run(q, kv_data, return_lse=True)
    )
    
    # Calculate speedup
    speedup = baseline_time / jit_time if jit_time > 0 else float('inf')
    
    # Results
    results = {
        'jit_time_ms': jit_time,
        'baseline_time_ms': baseline_time,
        'speedup': speedup,
        'batch_size': batch_size,
        'num_verify_requests': batch_size,
        'actual_max_kv_len': actual_max_kv_len,
        'buffer_max_kv_len': buffer_max_kv_len,
        'fixed_max_used': fixed_max_kv_len is not None,
        'kv_len': kv_lens[0] if len(set(kv_lens)) == 1 else 'mixed',
        'qo_len': qo_lens[0] if len(set(qo_lens)) == 1 else 'mixed',
    }
    
    return results


def extract_logits_for_request(logits_buffer, batch_idx, num_qo_heads, kv_len, buffer_max_kv_len):
    """
    Extract the actual logits for a specific request from the fixed-size buffer.
    
    Args:
        logits_buffer: Full logits buffer with shape (batch_size * num_qo_heads, buffer_max_kv_len)
        batch_idx: Index of the request in the batch
        num_qo_heads: Number of query heads
        kv_len: Actual KV length for this request
        buffer_max_kv_len: Fixed maximum KV length used for buffer allocation
    
    Returns:
        Tensor with shape (num_qo_heads, kv_len) containing the actual logits
    """
    start_idx = batch_idx * num_qo_heads
    end_idx = (batch_idx + 1) * num_qo_heads
    return logits_buffer[start_idx:end_idx, :kv_len]


def run_comprehensive_benchmark(fixed_max_kv_len: Optional[int] = 65536):
    """
    Run comprehensive benchmark suite with optional fixed maximum KV length.
    
    Args:
        fixed_max_kv_len: Fixed maximum KV length to use across all tests.
                         If None, uses dynamic max for each batch.
    """
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE BENCHMARK SUITE")
    if fixed_max_kv_len:
        print(f"📌 Using fixed max KV length: {fixed_max_kv_len}")
    else:
        print("📌 Using dynamic max KV length (will cause recompilations)")
    print("="*80)
    
    all_results = []
    
    # Test different batch sizes and KV lengths
    for num_verify_requests in [1, 4, 8, 16]:
        for kv_len in [4096, 8192, 16384, 32768]:
            config = [(16, kv_len)] * num_verify_requests
            
            print(f"\n🔍 Testing: {num_verify_requests} requests, KV length {kv_len}")
            
            try:
                results = benchmark_batch_dump_logits(
                    config, 
                    fixed_max_kv_len=fixed_max_kv_len
                )
                all_results.append(results)
                
                # Print immediate results
                print(f"  ✅ JIT: {results['jit_time_ms']:.3f}ms")
                print(f"  ✅ Baseline: {results['baseline_time_ms']:.3f}ms") 
                print(f"  🏁 Speedup: {results['speedup']:.3f}x")
                if fixed_max_kv_len:
                    print(f"  📊 Buffer efficiency: {results['actual_max_kv_len']}/{results['buffer_max_kv_len']} "
                          f"({results['actual_max_kv_len']/results['buffer_max_kv_len']*100:.1f}%)")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
    # Convert to pandas DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Add derived columns for analysis
        df['overhead_pct'] = ((df['jit_time_ms'] - df['baseline_time_ms']) / df['baseline_time_ms']) * 100
        df['jit_slower'] = df['jit_time_ms'] > df['baseline_time_ms']
        df['buffer_efficiency'] = df['actual_max_kv_len'] / df['buffer_max_kv_len'] * 100
        
        print("\n" + "="*80)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Display summary statistics
        print(f"\nOverall Statistics:")
        print(f"  Average JIT time: {df['jit_time_ms'].mean():.3f}ms")
        print(f"  Average Baseline time: {df['baseline_time_ms'].mean():.3f}ms")
        print(f"  Average speedup: {df['speedup'].mean():.3f}x")
        print(f"  Average buffer efficiency: {df['buffer_efficiency'].mean():.1f}%")
        print(f"  Cases where JIT is slower: {df['jit_slower'].sum()}/{len(df)}")
        
        return df
    else:
        print("❌ No successful benchmark results collected!")
        return None


def test_batch_dump_logits_with_fixed_max():
    """Test with fixed maximum KV length and verify correctness"""
    print("\n" + "="*80)
    print("🧪 TESTING CORRECTNESS WITH FIXED MAX KV LENGTH")
    print("="*80)
    
    kv_lens = [1024, 512]  # Different KV lengths
    fixed_max_kv_len = 2048  # Fixed maximum that's larger than both
    actual_max_kv_len = max(kv_lens)
    qo_lens = [16, 1]
    batch_size = len(kv_lens)
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_block_size = 1
    dev = torch.device("cuda")
    
    print(f"Test config:")
    print(f"  KV lengths: {kv_lens}")
    print(f"  Fixed max KV length: {fixed_max_kv_len}")
    print(f"  QO lengths: {qo_lens}")
    
    torch.manual_seed(42)
    variant_decl = get_variant_decl_dump_logits(num_qo_heads, fixed_max_kv_len)

    jit_args = (
        "batch_prefill_dump_logits",
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        128,
        128,
        ["output_logits"],
        ["float"],
        ["sm_scale", "max_kv_len"],
        ["double", "int64_t"],
        "DumpLogits",
        variant_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / 1).int()
    
    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat([torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0).int()

    total_blocks = kv_indptr[-1].item()

    # Create query data
    q = torch.rand(
        q_indptr[-1].item(), num_qo_heads, head_dim, dtype=torch.float16, device=dev
    )
    # Create KV data
    kv_data = torch.randn(
        total_blocks,
        2,
        page_block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=dev,
    )
    
    last_page_len = (seq_lens - 1) % page_block_size + 1
    kv_indices = torch.arange(total_blocks, device=dev).int()

    wrapper.plan(
        q_indptr.to(dev),
        kv_indptr.to(dev),
        kv_indices,
        last_page_len.to(dev),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
    )
    
    sm_scale = 1.0 / math.sqrt(128)
    # Allocate with fixed max size
    logits = torch.empty(batch_size * num_qo_heads, fixed_max_kv_len, dtype=torch.float32, device="cuda")
    out, lse = wrapper.run(q, kv_data, logits, sm_scale, fixed_max_kv_len, return_lse=True)
    print("Lse of shape: ", lse.shape)
    logits = logits * sm_scale
    
    # Extract actual logits for each request
    logits_req1 = extract_logits_for_request(logits, 0, num_qo_heads, kv_lens[0], fixed_max_kv_len)
    logits_req2 = extract_logits_for_request(logits, 1, num_qo_heads, kv_lens[1], fixed_max_kv_len)
    
    print(f"\nExtracted logits shapes:")
    print(f"  Request 1: {logits_req1.shape} (expected: ({num_qo_heads}, {kv_lens[0]}))")
    print(f"  Request 2: {logits_req2.shape} (expected: ({num_qo_heads}, {kv_lens[1]}))")
    
    # Generate reference
    kv_req1 = kv_data[kv_indptr[0]:kv_indptr[1]]
    kv_req2 = kv_data[kv_indptr[1]:kv_indptr[2]]
    k_req1, _ = kv_req1.split(1, dim=1)
    k_req2, _ = kv_req2.split(1, dim=1)
    k_req1 = k_req1.squeeze(1).squeeze(1)
    k_req2 = k_req2.squeeze(1).squeeze(1)
    q_req1 = q[q_indptr[0]:q_indptr[1]]
    q_req2 = q[q_indptr[1]:q_indptr[2]]

    heads_per_kv = num_qo_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    def qk_logits(q_last: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        kv_len = k.shape[0]
        out = torch.empty(num_qo_heads, kv_len,
                          dtype=torch.float32, device=q_last.device)
        for h in range(num_qo_heads):
            kh = h // heads_per_kv
            out[h] = (q_last[h] * k[:, kh]).sum(-1) * scale
        return out

    logits_ref_req1 = qk_logits(q_req1[-1], k_req1)
    logits_ref_req2 = qk_logits(q_req2[-1], k_req2)
    
    print("\nCorrectness check:")
    match1 = torch.allclose(logits_req1, logits_ref_req1, atol=1e-3, rtol=1e-3)
    match2 = torch.allclose(logits_req2, logits_ref_req2, atol=1e-3, rtol=1e-3)
    print(f"  Request 1 matches reference: {match1}")
    print(f"  Request 2 matches reference: {match2}")
    print(f"  Overall match: {match1 and match2}")
    
    return match1 and match2


# Example usage
if __name__ == "__main__":
    # Test correctness first
    test_batch_dump_logits_with_fixed_max()
    
    # Run benchmarks with fixed max
    print("\n" + "="*80)
    print("Running benchmarks with fixed max KV length = 65536")
    print("="*80)
    results_fixed = run_comprehensive_benchmark(fixed_max_kv_len=65536)
    print(results_fixed)