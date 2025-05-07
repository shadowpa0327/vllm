import os
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def quantize_tensor(w: torch.Tensor, n_bits: int, sym: bool, clip_ratio: float = 1.0) -> torch.Tensor:
    """
    Quantize a 2D tensor [num_groups, group_size].
    
    Args:
        w: Weight tensor to quantize
        n_bits: Number of bits for quantization
        sym: Whether to use symmetric quantization
        clip_ratio: Ratio to clip the maximum values
        
    Returns:
        Quantized tensor with the same shape as input
    """
    #assert w.dim() == 2, "Weight format should be: [num_groups, group_size]"
    #assert n_bits < 16, "n_bits must be less than 16."
    
    if sym:
        # Symmetric quantization
        w_max = w.abs().amax(dim=-1, keepdim=True)
        q_max = (2 ** (n_bits - 1) - 1)
        q_min = (-2 ** (n_bits - 1))
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        scales = scales.clamp(min=1e-5, max=1e4)
        base = torch.zeros_like(scales)
    else:
        # Asymmetric quantization
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        q_max = (2 ** n_bits - 1)
        q_min = 0
        if clip_ratio < 1.0:
            w_max *= clip_ratio
            w_min *= clip_ratio
        scales = (w_max - w_min) / q_max
        scales = scales.clamp(min=1e-5, max=1e4)
        base = -w_min

    # Round, clamp, then "dequantize"
    w = torch.clamp(torch.round((w + base) / scales), q_min, q_max) * scales - base
    return w


@torch.no_grad()
def fake_quantize_weight(
    w: torch.Tensor,
    n_bits: int = 8,
    sym: bool = True,
    group_size: int = 128,
    clip_ratio: float = 1.0,
    group_dim: int = 0
) -> torch.Tensor:
    """
    Fake-quantize a weight tensor by grouping and calling quantize_tensor.
    If n_bits is 16, return the weight without quantization.

    Args:
        w: Weight tensor, typically [out_features, in_features]
        n_bits: Number of bits for quantization
        sym: Whether to use symmetric quantization
        group_size: Size of each group. If None, use row/column-based grouping
        clip_ratio: Ratio to clip the maximum absolute value
        group_dim: Dimension along which to group (0=row-based, 1=column-based)
    
    Returns:
        Fake-quantized tensor with the same shape as input
    """
    if n_bits == 16: #Torch.compile is non happy with this line.....
        return w
    original_shape = w.shape
    
    # Handle column-based grouping by transposing if needed
    if group_dim == 1:
        w = w.transpose(0, 1)  # [in_features, out_features]

    # Handle row-based grouping (no reshaping needed)
    if group_size is None:
        w_reshaped = w
        assert w_reshaped.dim() == 2, "Expected 2D tensor after grouping or transposing"
    else:
        # Reshape into [num_groups, group_size]
        assert w.shape[1] % group_size == 0, "group_size must divide total elements"
        w_reshaped = w.view(-1, group_size)
    
    # Perform quantization
    w_quant = quantize_tensor(w_reshaped, n_bits, sym, clip_ratio)

    # Reshape back to original dimensions
    if group_size is not None:
        w_quant = w_quant.view(*w.shape)

    # Transpose back if needed
    if group_dim == 1:
        w_quant = w_quant.transpose(0, 1)

    return w_quant.view(original_shape)