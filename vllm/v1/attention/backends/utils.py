# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
import numpy as np

@dataclass
class CommonAttentionMetadata:
    """
    Attention metadata attributes that can be shared by layers in different KV
    cache groups and thus having different block table.
    """

    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""
    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""



import triton
import triton.language as tl
import torch

@triton.jit
def concat_kv_indices_kernel(
    selected_indices_ptr,
    selected_indices_stride,
    len_selected_indices,
    all_indices_ptr,
    all_indices_offsets,
    len_all_indices,
    output_ptr,
    output_offsets,
    num_requests,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Concatenate selected KV indices with new KV indices for each request.
    
    This kernel processes multiple requests in parallel, concatenating their
    selected indices with new indices without considering head dimensions.
    
    Args:
        selected_indices_ptr: Pointer to selected indices buffer [num_requests, max_selected]
        selected_indices_stride: Stride for selected indices (max_selected_indices)
        len_selected_indices: Number of selected indices per request [num_requests]
        all_indices_ptr: Pointer to all KV indices buffer (flattened)
        all_indices_offsets: Start offset in all_indices for each request [num_requests]
        len_all_indices: Number of new indices per request [num_requests]
        output_ptr: Output buffer pointer (flattened)
        output_offsets: Start offset in output for each request [num_requests]
        num_requests: Number of requests to process
        BLOCK_SIZE: Block size for vectorized operations
    """
    # Each program handles multiple requests in a loop
    for req_id in range(tl.program_id(0), num_requests, tl.num_programs(0)):
        # Load lengths for this request
        l_sel = tl.load(len_selected_indices + req_id)
        l_all = tl.load(len_all_indices + req_id)
        
        # Load offsets for this request
        all_off = tl.load(all_indices_offsets + req_id)
        out_off = tl.load(output_offsets + req_id)
        
        # Calculate pointers for this request
        req_sel_ptr = selected_indices_ptr + req_id * selected_indices_stride
        req_out_ptr = output_ptr + out_off
        
        # Copy selected indices first (vectorized)
        for i in range(0, l_sel, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < l_sel
            sel_vals = tl.load(req_sel_ptr + offs, mask=mask, other=0)
            tl.store(req_out_ptr + offs, sel_vals, mask=mask)
        
        # Copy new indices after selected indices (vectorized)
        for i in range(0, l_all, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < l_all
            all_vals = tl.load(all_indices_ptr + all_off + offs, mask=mask, other=0)
            tl.store(req_out_ptr + offs + l_sel, all_vals, mask=mask) 