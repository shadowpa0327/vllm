import functools
import math

import pytest
import torch

import flashinfer
import flashinfer.jit
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer.jit.attention import (
    gen_customize_single_decode_module,
    gen_customize_single_prefill_module,
)
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer.utils import MaskMode, is_sm90a_supported

def test_batch_dump_logits():
    torch.manual_seed(42)
    variant_decl = r"""
struct DumpLogits : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ DumpLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (qo_idx == (qo_len-1) && kv_idx < kv_len) {
      params.output_logits[batch_idx * params.num_qo_heads * params.max_kv_len +
                          qo_head_idx * params.max_kv_len +
                          kv_idx] = logits * params.sm_scale;
    }
    return logits;
  });
};"""
    jit_args = (
        "batch_prefill_dump_logits",  # uri
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,  # idtype
        128,  # hidden_dim_qk
        128,  # hidden_dim_vo
        ["output_logits"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale", "max_kv_len"],  # additional_scalar_names
        ["double", "int64_t"],  # additional_scalar_dtypes
        "DumpLogits",
        variant_decl,
    )

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )

    kv_lens = [16, 16]
    max_kv_len = max(kv_lens)
    qo_lens = [16, 1]
    batch_size = len(kv_lens)
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_block_size = 1
    dev = torch.device("cuda")

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

    logits = torch.empty(batch_size*num_qo_heads, max_kv_len, dtype=torch.float32, device="cuda")
    wrapper.run(q, kv_data, logits, 1.0 / math.sqrt(128), max_kv_len)

    kv_req1 = kv_data[kv_indptr[0]:kv_indptr[1]]
    kv_req2 = kv_data[kv_indptr[1]:kv_indptr[2]]
    k_req1, _ = kv_req1.split(1, dim=1)
    k_req2, _ = kv_req2.split(1, dim=1)
    k_req1 = k_req1.squeeze(1).squeeze(1)
    k_req2 = k_req2.squeeze(1).squeeze(1) #(kv_len, num_kv_heads, head_dim)
    q_req1 = q[q_indptr[0]:q_indptr[1]] #(qo_len, num_qo_heads, head_dim)
    q_req2 = q[q_indptr[1]:q_indptr[2]] #(qo_len, num_qo_heads, head_dim)


    # ------------------  replace the TODO with this  ------------------

    heads_per_kv = num_qo_heads // num_kv_heads      # 4  (GQA ratio)
    scale         = 1.0 / math.sqrt(head_dim)        # same scale as the kernel

    def qk_logits(q_last: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q_last : (num_qo_heads, head_dim)          – last query token of the request
        k      : (kv_len, num_kv_heads, head_dim) – full key matrix for the request
        returns: (num_qo_heads, kv_len)            – un-normalized logits
        """
        kv_len = k.shape[0]
        out = torch.empty(num_qo_heads, kv_len,
                          dtype=torch.float32, device=q_last.device)
        for h in range(num_qo_heads):
            kh = h // heads_per_kv                 # map query-head → kv-head
            out[h] = (q_last[h] * k[:, kh]).sum(-1) * scale
        return out

    # ------------ request-1 (batch 0) ------------
    logits_req1 = qk_logits(q_req1[-1], k_req1)      # only last token matters

    # ------------ request-2 (batch 1) ------------
    logits_req2 = qk_logits(q_req2[-1], k_req2)      # single-token request

    logits_ref = torch.zeros_like(logits)
    logits_ref[0:num_qo_heads, :logits_req1.shape[1]] = logits_req1
    logits_ref[num_qo_heads:2*num_qo_heads, :logits_req2.shape[1]] = logits_req2
    print("Ref and Flashinfer match: ", torch.allclose(logits_ref, logits, atol=1e-3, rtol=1e-3))
    
if __name__ == "__main__":
    test_batch_dump_logits()