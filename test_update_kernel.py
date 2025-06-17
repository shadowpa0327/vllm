#!/usr/bin/env python3
"""
Standalone test script for update_selected_indices_kernel
"""

import torch
import triton
import triton.language as tl
import numpy as np

# Copy the kernel from the original code
@triton.jit
def update_selected_indices_kernel(
    all_kv_indices,
    sinks_indices_offsets,
    len_sinks_indices,
    recent_indices_offsets,
    len_recent_indices,
    selected_indices_ptrs,
    selected_indices_ptrs_stride,
):
    head_id = tl.program_id(0)
    req_id = tl.program_id(1)
    #l.device_print("head_id: ", head_id)
    #tl.device_print("req_id: ", req_id)
    l_sinks = tl.load(len_sinks_indices + req_id)
    l_recent = tl.load(len_recent_indices + req_id)
    total = l_sinks + l_recent

    sinks_off = tl.load(sinks_indices_offsets + req_id)
    recent_off = tl.load(recent_indices_offsets + req_id)
    out_addr = tl.load(selected_indices_ptrs + req_id)
    out_ptr = out_addr.to(tl.pointer_type(tl.int32))

    pid_inner = tl.program_id(2)
    stride = tl.num_programs(2)
    for i in range(pid_inner, total, stride):
        val = (
            tl.load(all_kv_indices + sinks_off + i)
            if i < l_sinks
            else tl.load(all_kv_indices + recent_off + (i - l_sinks))
        )
        tl.store(out_ptr + head_id * selected_indices_ptrs_stride + i, val)

def test_update_kernel_simple():
    """Simplified test with smaller dimensions for easier debugging"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    print("\n" + "="*60)
    print("SIMPLIFIED TEST")
    print("="*60)
    
    # Simplified parameters
    num_layers = 2
    num_kv_heads = 2
    num_heads = num_layers * num_kv_heads  # Total: 4 heads
    MAX_LEN_KV_CACHE = 100
    num_requests = 1
    
    # Create simple test data
    all_kv_indices_size = 20
    all_kv_indices_gpu = torch.arange(all_kv_indices_size, dtype=torch.int32, device=device)
    
    selected_buffer = torch.zeros(
        (num_layers, num_kv_heads, MAX_LEN_KV_CACHE), 
        dtype=torch.int32, 
        device=device
    )
    
    # Simple split: 3 sinks + 5 recent = 8 total
    sink_len = 3
    recent_len = 5
    sinks_offset = 0      # Indices 0, 1, 2
    recent_offset = 10    # Indices 10, 11, 12, 13, 14
    
    print(f"Simple test: {sink_len} sinks + {recent_len} recent = {sink_len + recent_len} total")
    print(f"Sinks from indices {sinks_offset}-{sinks_offset + sink_len - 1}")
    print(f"Recent from indices {recent_offset}-{recent_offset + recent_len - 1}")
    
    # Create parameter tensors
    sinks_indices_offsets = torch.tensor([sinks_offset], dtype=torch.int32, device=device)
    len_sinks_indices = torch.tensor([sink_len], dtype=torch.int32, device=device)
    recent_indices_offsets = torch.tensor([recent_offset], dtype=torch.int32, device=device)
    len_recent_indices = torch.tensor([recent_len], dtype=torch.int32, device=device)
    selected_indices_ptrs = torch.tensor([selected_buffer.data_ptr()], dtype=torch.int64, device=device)
    
    # Launch kernel
    grid = (num_heads, num_requests)
    update_selected_indices_kernel[grid](
        all_kv_indices_gpu,
        sinks_indices_offsets,
        len_sinks_indices,
        recent_indices_offsets,
        len_recent_indices,
        selected_indices_ptrs,
        MAX_LEN_KV_CACHE,
    )
    
    # Verify
    expected = [0, 1, 2, 10, 11, 12, 13, 14]  # sinks + recent
    
    print("\nSimple test results:")
    for head_idx in range(num_heads):
        layer = head_idx // num_kv_heads
        head = head_idx % num_kv_heads
        actual = selected_buffer[layer, head, :len(expected)].cpu().tolist()
        match = actual == expected
        print(f"  Head {head_idx}: expected={expected}, actual={actual}, match={match}")
    
    return True


def test_update_kernel_mul_reqs():
    """Test the update_selected_indices_kernel with multiple requests"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    print("\n" + "="*60)
    print("MULTIPLE REQUESTS TEST")
    print("="*60)
    
    # Test parameters for 2 requests
    num_layers = 2
    num_kv_heads = 2
    num_heads = num_layers * num_kv_heads  # Total: 4 heads
    MAX_LEN_KV_CACHE = 100
    num_requests = 2
    
    print(f"Test configuration:")
    print(f"  num_layers: {num_layers}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  total heads: {num_heads}")
    print(f"  num_requests: {num_requests}")
    
    # Create concatenated all_kv_indices for multiple requests
    # Request 0: indices 0-14 (15 indices)
    # Request 1: indices 100-119 (20 indices)
    req0_indices = torch.arange(0, 15, dtype=torch.int32, device=device)  # [0, 1, 2, ..., 14]
    req1_indices = torch.arange(100, 120, dtype=torch.int32, device=device)  # [100, 101, ..., 119]
    
    # Concatenate all indices
    all_kv_indices_gpu = torch.cat([req0_indices, req1_indices], dim=0)
    print(f"  all_kv_indices_gpu: {all_kv_indices_gpu}")
    
    # Create indptr to separate requests
    # all_kv_indptr[0] = 0 (start of req0)
    # all_kv_indptr[1] = 15 (start of req1, end of req0)
    # all_kv_indptr[2] = 35 (end of req1)
    all_kv_indptr = torch.tensor([0, 15, 35], dtype=torch.int32, device=device)
    print(f"  all_kv_indptr: {all_kv_indptr}")
    
    # Verify the split
    req0_data = all_kv_indices_gpu[all_kv_indptr[0]:all_kv_indptr[1]]
    req1_data = all_kv_indices_gpu[all_kv_indptr[1]:all_kv_indptr[2]]
    print(f"  Request 0 data: {req0_data}")
    print(f"  Request 1 data: {req1_data}")
    
    # Create selected indices buffer for both requests
    # Shape: (num_requests, num_layers, num_kv_heads, MAX_LEN_KV_CACHE)
    selected_buffer = torch.zeros(
        (num_requests, num_layers, num_kv_heads, MAX_LEN_KV_CACHE), 
        dtype=torch.int32, 
        device=device
    )
    
    # Request 0: 3 sinks + 5 recent = 8 total
    # Sinks: indices 0, 1, 2 (positions 0-2 in req0_data)
    # Recent: indices 10, 11, 12, 13, 14 (positions 10-14 in req0_data)
    req0_sink_len = 3
    req0_recent_len = 5
    req0_sinks_offset = all_kv_indptr[0] + 0   # Start of req0 + 0
    req0_recent_offset = all_kv_indptr[0] + 10  # Start of req0 + 10
    
    # Request 1: 4 sinks + 6 recent = 10 total
    # Sinks: indices 100, 101, 102, 103 (positions 0-3 in req1_data)
    # Recent: indices 114, 115, 116, 117, 118, 119 (positions 14-19 in req1_data)
    req1_sink_len = 4
    req1_recent_len = 6
    req1_sinks_offset = all_kv_indptr[1] + 0   # Start of req1 + 0
    req1_recent_offset = all_kv_indptr[1] + 14  # Start of req1 + 14
    
    print(f"\nRequest 0:")
    print(f"  sink_len: {req0_sink_len}, recent_len: {req0_recent_len}")
    print(f"  sinks_offset: {req0_sinks_offset}, recent_offset: {req0_recent_offset}")
    print(f"  Expected sinks: {all_kv_indices_gpu[req0_sinks_offset:req0_sinks_offset+req0_sink_len]}")
    print(f"  Expected recent: {all_kv_indices_gpu[req0_recent_offset:req0_recent_offset+req0_recent_len]}")
    
    print(f"\nRequest 1:")
    print(f"  sink_len: {req1_sink_len}, recent_len: {req1_recent_len}")
    print(f"  sinks_offset: {req1_sinks_offset}, recent_offset: {req1_recent_offset}")
    print(f"  Expected sinks: {all_kv_indices_gpu[req1_sinks_offset:req1_sinks_offset+req1_sink_len]}")
    print(f"  Expected recent: {all_kv_indices_gpu[req1_recent_offset:req1_recent_offset+req1_recent_len]}")
    
    # Create parameter tensors for both requests
    sinks_indices_offsets = torch.tensor([req0_sinks_offset, req1_sinks_offset], dtype=torch.int32, device=device)
    len_sinks_indices = torch.tensor([req0_sink_len, req1_sink_len], dtype=torch.int32, device=device)
    recent_indices_offsets = torch.tensor([req0_recent_offset, req1_recent_offset], dtype=torch.int32, device=device)
    len_recent_indices = torch.tensor([req0_recent_len, req1_recent_len], dtype=torch.int32, device=device)
    
    # Create pointers to each request's buffer
    req0_ptr = selected_buffer[0].data_ptr()
    req1_ptr = selected_buffer[1].data_ptr()
    selected_indices_ptrs = torch.tensor([req0_ptr, req1_ptr], dtype=torch.int64, device=device)
    selected_indices_ptrs_stride = MAX_LEN_KV_CACHE
    
    print(f"\nKernel parameters:")
    print(f"  sinks_indices_offsets: {sinks_indices_offsets}")
    print(f"  len_sinks_indices: {len_sinks_indices}")
    print(f"  recent_indices_offsets: {recent_indices_offsets}")
    print(f"  len_recent_indices: {len_recent_indices}")
    print(f"  selected_indices_ptrs: {selected_indices_ptrs}")
    print(f"  selected_indices_ptrs_stride: {selected_indices_ptrs_stride}")
    
    # Launch kernel with 2D grid (heads, requests)
    grid = (num_heads, num_requests)
    print(f"  grid: {grid}")
    
    print("\nLaunching kernel...")
    update_selected_indices_kernel[grid](
        all_kv_indices_gpu,
        sinks_indices_offsets,
        len_sinks_indices,
        recent_indices_offsets,
        len_recent_indices,
        selected_indices_ptrs,
        selected_indices_ptrs_stride,
    )
    
    print("Kernel completed!")
    
    # Verify results for both requests
    print("\n=== VERIFICATION ===")
    
    # Expected results
    req0_expected_sinks = [0, 1, 2]
    req0_expected_recent = [10, 11, 12, 13, 14]
    req0_expected_combined = req0_expected_sinks + req0_expected_recent
    
    req1_expected_sinks = [100, 101, 102, 103]
    req1_expected_recent = [114, 115, 116, 117, 118, 119]
    req1_expected_combined = req1_expected_sinks + req1_expected_recent
    
    all_correct = True
    
    # Check each request
    for req_id in range(num_requests):
        print(f"\n--- Request {req_id} ---")
        
        if req_id == 0:
            expected_combined = req0_expected_combined
            total_len = req0_sink_len + req0_recent_len
        else:
            expected_combined = req1_expected_combined
            total_len = req1_sink_len + req1_recent_len
        
        print(f"Expected combined: {expected_combined}")
        
        # Check a few representative heads for this request
        test_heads = [0, num_heads-1]  # Test first and last head
        
        for head_idx in test_heads:
            layer = head_idx // num_kv_heads
            head = head_idx % num_kv_heads
            
            print(f"  Head {head_idx} (layer={layer}, head={head}):")
            
            # Get actual results for this head and request
            actual_results = selected_buffer[req_id, layer, head, :total_len].cpu().tolist()
            
            match = actual_results == expected_combined
            print(f"    Expected: {expected_combined}")
            print(f"    Actual:   {actual_results}")
            print(f"    Match:    {match}")
            
            if not match:
                all_correct = False
                # Show mismatches
                for i, (exp, act) in enumerate(zip(expected_combined, actual_results)):
                    if exp != act:
                        print(f"      MISMATCH at position {i}: expected {exp}, got {act}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Multiple requests test: {'PASSED' if all_correct else 'FAILED'}")
    
    if all_correct:
        print("✅ All requests and heads correctly updated selected indices!")
        print("✅ Request 0: Sinks [0,1,2] + Recent [10,11,12,13,14]")
        print("✅ Request 1: Sinks [100,101,102,103] + Recent [114,115,116,117,118,119]")
    else:
        print("❌ Some requests/heads had incorrect results")
    
    # Additional verification: check that unused positions are still zero
    max_total_len = max(req0_sink_len + req0_recent_len, req1_sink_len + req1_recent_len)
    unused_positions = selected_buffer[:, :, :, max_total_len:max_total_len+5].cpu()
    if torch.all(unused_positions == 0):
        print("✅ Unused positions remain zero (no buffer overflow)")
    else:
        print("❌ Some unused positions were modified (potential buffer overflow)")
        all_correct = False
    
    return all_correct


if __name__ == "__main__":
    print("Testing update_selected_indices_kernel...")
    
    # Run simplified test first
    #test_update_kernel_simple()
    
    # Run multiple requests test
    test_update_kernel_mul_reqs()
    