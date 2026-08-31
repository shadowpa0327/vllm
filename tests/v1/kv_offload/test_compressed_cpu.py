# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior tests for experimental compressed CPU offloading."""

import inspect
import numpy as np
from types import SimpleNamespace
from typing import cast
import uuid
from collections.abc import Callable

import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.compressed_cpu import (
    CompressedCPUOffloadingSpec,
    CompressedCPUOffloadingWorker,
    GroupedCompressedCPUOffloadingWorker,
    GroupedCPULoadStoreSpec,
    GroupedCPUOffloadingManager,
    GroupedKVCompressor,
    INT4CompressedCPUOffloadingSpec,
    INT4CompressedCPUOffloadingWorker,
    INT4KVCompressor,
    KVCompressionGroup,
    KVCompressionSourceLayout,
    KVSplitKVCompressor,
    SVDAlgorithm,
    KVCompressor,
    RawCompressedCPUOffloadingWorker,
    RawKVCompressor,
    SVDCompressedCPUOffloadingSpec,
    SVDCompressedCPUOffloadingWorker,
    SVDFactorDType,
    SVDKVCompressor,
)
from vllm.v1.kv_offload.compressed_cpu.rope import RoPEKeyTransform
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory


def _low_rank_matrices(
    layout: KVCompressionSourceLayout,
    rank: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    return tuple(
        (
            torch.randn(layout.rows, rank, device=device)
            @ torch.randn(rank, layout.columns, device=device)
        ).to(layout.dtype)
        for _ in range(layout.matrix_count)
    )


def _round_trip(
    compressor: KVCompressor,
    source: tuple[torch.Tensor, ...],
    layout: KVCompressionSourceLayout,
) -> tuple[torch.Tensor, ...]:
    encoded_layout = compressor.build_layout(layout)
    encoded = torch.empty(encoded_layout.storage_nbytes, dtype=torch.uint8)
    compressor.compress_into(source, encoded, encoded_layout)
    compressor.validate_blob(encoded, encoded_layout)
    return compressor.decompress(encoded, encoded_layout)


def _make_offloading_config(
    spec_name: str,
    extra_config: dict[str, object],
    engine_id: str = "compressed-test",
) -> OffloadingConfig:
    normalized_extra_config = {
        "spec_name": spec_name,
        "spec_module_path": "vllm.v1.kv_offload.compressed_cpu.spec",
        "cpu_bytes_to_use": 4 * 4096,
        **extra_config,
    }
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(4, ("layer.0", "layer.1")),),
        worker_kv_bytes_per_block=1024,
        enable_kv_cache_events=False,
        extra_config=normalized_extra_config,
        engine_id=engine_id,
        model=OffloadingModelConfig(name="test-model", dtype="bfloat16"),
        cache=OffloadingCacheConfig(tokens_per_hash=4, blocks_per_chunk=2),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )


def _make_hybrid_offloading_config(engine_id: str = "hybrid-test") -> OffloadingConfig:
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=4,
                layer_names=("attn.0", "attn.1"),
                cache_kind="attention",
                worker_kv_bytes_per_block=1024,
            ),
            OffloadingGroupConfig(
                tokens_per_block=4,
                layer_names=("state.0", "state.1"),
                cache_kind="mamba",
                worker_kv_bytes_per_block=256,
            ),
        ),
        worker_kv_bytes_per_block=1280,
        enable_kv_cache_events=False,
        extra_config={
            "spec_name": "INT4CompressedCPUOffloadingSpec",
            "spec_module_path": "vllm.v1.kv_offload.compressed_cpu.spec",
            "cpu_bytes_to_use": 8 * 4096 * 4,
            "blocks_per_chunk": 2,
            "compression_log_operations": True,
        },
        engine_id=engine_id,
        model=OffloadingModelConfig(name="hybrid-model", dtype="bfloat16"),
        cache=OffloadingCacheConfig(tokens_per_hash=4, blocks_per_chunk=2),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )


def _canonical_gpu_caches(
    source_layout: KVCompressionSourceLayout,
    blocks_per_chunk: int,
    num_gpu_blocks: int,
) -> tuple[CanonicalKVCaches, list[torch.Tensor]]:
    tokens_per_block = source_layout.rows // blocks_per_chunk
    page_nbytes = (
        tokens_per_block * source_layout.columns * source_layout.dtype.itemsize
    )
    matrices = [
        torch.zeros(
            num_gpu_blocks,
            tokens_per_block,
            source_layout.columns,
            dtype=source_layout.dtype,
            device="cuda:0",
        )
        for _ in range(source_layout.matrix_count)
    ]
    canonical_tensors = [
        CanonicalKVCacheTensor(
            tensor=matrix.view(torch.int8).view(num_gpu_blocks, page_nbytes),
            page_size_bytes=page_nbytes,
        )
        for matrix in matrices
    ]
    refs = [
        CanonicalKVCacheRef(tensor_idx=index, page_size_bytes=page_nbytes)
        for index in range(source_layout.matrix_count)
    ]
    return CanonicalKVCaches(canonical_tensors, [refs]), matrices


def _hybrid_canonical_gpu_caches() -> tuple[
    CanonicalKVCaches,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    num_gpu_blocks = 8
    attention = [
        torch.zeros(
            num_gpu_blocks,
            4,
            64,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        for _ in range(2)
    ]
    states = [
        torch.zeros(
            num_gpu_blocks,
            1,
            128,
            dtype=torch.uint8,
            device="cuda:0",
        )
        for _ in range(2)
    ]
    tensors = [
        CanonicalKVCacheTensor(
            tensor=matrix.view(torch.int8).view(num_gpu_blocks, 512),
            page_size_bytes=512,
        )
        for matrix in attention
    ]
    tensors.extend(
        CanonicalKVCacheTensor(
            tensor=matrix.view(torch.int8).view(num_gpu_blocks, 128),
            page_size_bytes=128,
        )
        for matrix in states
    )
    return (
        CanonicalKVCaches(
            tensors=tensors,
            group_data_refs=[
                [
                    CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512),
                    CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=512),
                ],
                [
                    CanonicalKVCacheRef(tensor_idx=2, page_size_bytes=128),
                    CanonicalKVCacheRef(tensor_idx=3, page_size_bytes=128),
                ],
            ],
        ),
        attention,
        states,
    )


def _wait_for_one_result(
    worker: CompressedCPUOffloadingWorker,
    job_id: int,
) -> None:
    worker.wait({job_id})
    results = worker.get_finished()
    assert len(results) == 1
    assert results[0].job_id == job_id
    assert results[0].success
    assert results[0].transfer_size == worker.layout.storage_nbytes


def test_compression_workers_are_an_explicit_abc_hierarchy() -> None:
    assert inspect.isabstract(CompressedCPUOffloadingWorker)
    assert inspect.isabstract(CompressedCPUOffloadingSpec)
    assert issubclass(INT4CompressedCPUOffloadingWorker, CompressedCPUOffloadingWorker)
    assert issubclass(SVDCompressedCPUOffloadingWorker, CompressedCPUOffloadingWorker)


def test_int4_compresses_then_reconstructs_the_object() -> None:
    layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=2,
        rows=32,
        columns=64,
    )
    torch.manual_seed(1)
    source = tuple(
        torch.randn(layout.rows, layout.columns, dtype=layout.dtype)
        for _ in range(layout.matrix_count)
    )
    compressor = INT4KVCompressor(group_size=64)
    reconstructed = _round_trip(compressor, source, layout)

    encoded_layout = compressor.build_layout(layout)
    assert encoded_layout.storage_nbytes < layout.raw_nbytes
    for actual, expected in zip(reconstructed, source, strict=True):
        relative_error = (actual.float() - expected.float()).norm()
        relative_error /= expected.float().norm()
        assert relative_error.item() < 0.16


def test_svd_compresses_then_reconstructs_low_rank_matrices() -> None:
    retained_rank = 4
    layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=2,
        rows=32,
        columns=64,
    )
    source = _low_rank_matrices(layout, retained_rank)
    compressor = SVDKVCompressor(rank=retained_rank)
    reconstructed = _round_trip(compressor, source, layout)

    encoded_layout = compressor.build_layout(layout)
    assert encoded_layout.storage_nbytes < layout.raw_nbytes
    for actual, expected in zip(reconstructed, source, strict=True):
        relative_error = (actual.float() - expected.float()).norm()
        relative_error /= expected.float().norm()
        assert relative_error.item() < 0.05


def test_svd_fp16_factors_remove_fp8_quantization_error_at_full_rank() -> None:
    layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=2,
        rows=32,
        columns=16,
    )
    torch.manual_seed(29)
    source = tuple(
        torch.randn(layout.rows, layout.columns, dtype=layout.dtype)
        for _ in range(layout.matrix_count)
    )

    fp8_reconstructed = _round_trip(
        SVDKVCompressor(rank=16, factor_dtype="fp8"),
        source,
        layout,
    )
    fp16_compressor = SVDKVCompressor(rank=16, factor_dtype="fp16")
    fp16_reconstructed = _round_trip(fp16_compressor, source, layout)

    fp8_errors = []
    fp16_errors = []
    for fp8, fp16, expected in zip(
        fp8_reconstructed,
        fp16_reconstructed,
        source,
        strict=True,
    ):
        expected_norm = expected.float().norm()
        fp8_errors.append(
            ((fp8.float() - expected.float()).norm() / expected_norm).item()
        )
        fp16_errors.append(
            ((fp16.float() - expected.float()).norm() / expected_norm).item()
        )

    assert fp16_compressor.codec_id == "svd-fp16-v1"
    assert max(fp16_errors) < 0.001
    assert max(fp16_errors) < max(fp8_errors) / 100


def test_svd_rejects_an_unknown_factor_dtype() -> None:
    with pytest.raises(ValueError, match="factor_dtype must be 'fp8' or 'fp16'"):
        SVDKVCompressor(rank=4, factor_dtype="float32")  # type: ignore[arg-type]


def test_encoded_header_rejects_a_different_codec_contract() -> None:
    layout = KVCompressionSourceLayout(torch.bfloat16, 1, 32, 64)
    compressor = INT4KVCompressor(group_size=64)
    encoded_layout = compressor.build_layout(layout)
    encoded = torch.empty(encoded_layout.storage_nbytes, dtype=torch.uint8)
    source = (torch.randn(layout.rows, layout.columns, dtype=layout.dtype),)
    compressor.compress_into(source, encoded, encoded_layout)
    encoded[0] ^= 1

    with pytest.raises(ValueError, match="header does not match"):
        compressor.validate_blob(encoded, encoded_layout)


def test_raw_codec_preserves_state_bytes_exactly() -> None:
    layout = KVCompressionSourceLayout(torch.uint8, 2, 1, 128)
    source = tuple(torch.randint(0, 256, (1, 128), dtype=torch.uint8) for _ in range(2))
    reconstructed = _round_trip(RawKVCompressor(), source, layout)

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(reconstructed, source, strict=True)
    )


def test_both_codecs_define_a_smaller_fixed_object_for_an_8k_prefix() -> None:
    source_layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=16,
        rows=8192,
        columns=256,
    )

    int4_layout = INT4KVCompressor(group_size=64).build_layout(source_layout)
    svd_layout = SVDKVCompressor(rank=32).build_layout(source_layout)

    assert source_layout.raw_nbytes == 64 * 1024 * 1024
    assert int4_layout.storage_nbytes < source_layout.raw_nbytes
    assert svd_layout.storage_nbytes < source_layout.raw_nbytes
    assert int4_layout.source.rows == 8192
    assert svd_layout.source.rows == 8192


def test_grouped_compressor_dispatches_distinct_group_layouts_and_algorithms() -> None:
    int4_source_layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=2,
        rows=16,
        columns=32,
    )
    svd_source_layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=1,
        rows=24,
        columns=16,
    )
    grouped = GroupedKVCompressor(
        (
            KVCompressionGroup(
                group_idx=0,
                source_layout=int4_source_layout,
                compressor=INT4KVCompressor(group_size=32),
            ),
            KVCompressionGroup(
                group_idx=1,
                source_layout=svd_source_layout,
                compressor=SVDKVCompressor(rank=3, factor_dtype="fp16"),
            ),
        )
    )

    torch.manual_seed(41)
    int4_source = tuple(
        torch.randn(
            int4_source_layout.rows,
            int4_source_layout.columns,
            dtype=int4_source_layout.dtype,
        )
        for _ in range(int4_source_layout.matrix_count)
    )
    svd_source = _low_rank_matrices(svd_source_layout, rank=3)
    sources = (int4_source, svd_source)

    for group_idx, source in enumerate(sources):
        group = grouped.get_group(group_idx)
        encoded = torch.empty(
            group.encoded_layout.storage_nbytes,
            dtype=torch.uint8,
        )
        grouped.compress_into(group_idx, source, encoded)
        grouped.validate_blob(group_idx, encoded)
        reconstructed = grouped.decompress(group_idx, encoded)

        assert len(reconstructed) == group.source_layout.matrix_count
        assert all(
            matrix.shape == (group.source_layout.rows, group.source_layout.columns)
            for matrix in reconstructed
        )

    assert grouped.get_group(0).encoded_layout.codec_id == "int4-groupwise-v1"
    assert grouped.get_group(1).encoded_layout.codec_id == "svd-fp16-v1"


def test_grouped_compressor_rejects_ambiguous_or_unknown_groups() -> None:
    source_layout = KVCompressionSourceLayout(torch.bfloat16, 1, 8, 16)
    group = KVCompressionGroup(
        group_idx=2,
        source_layout=source_layout,
        compressor=INT4KVCompressor(group_size=16),
    )

    with pytest.raises(ValueError, match="indices must be unique"):
        GroupedKVCompressor((group, group))

    grouped = GroupedKVCompressor((group,))
    with pytest.raises(KeyError, match="no compressor configured"):
        grouped.get_group(3)


def test_hybrid_spec_isolates_attention_and_state_storage() -> None:
    spec = OffloadingSpecFactory.create_spec(_make_hybrid_offloading_config())

    assert isinstance(spec, INT4CompressedCPUOffloadingSpec)
    assert len(spec.group_plans) == 2
    assert spec.group_plans[0].compression.encoded_layout.codec_id == (
        "int4-groupwise-v1"
    )
    assert spec.group_plans[0].pages_per_object == 2
    assert spec.group_plans[1].compression.encoded_layout.codec_id == "raw-bytes-v1"
    assert spec.group_plans[1].pages_per_object == 1
    assert spec.num_blocks == 16


def test_grouped_manager_routes_overlapping_local_slot_ids() -> None:
    manager = GroupedCPUOffloadingManager((2, 2))
    req_context = ReqContext("request")
    key0 = make_offload_key(b"attention", 0)
    key1 = make_offload_key(b"state", 1)

    output = manager.prepare_store([key0, key1], req_context)
    assert output is not None
    assert isinstance(output.store_spec, GroupedCPULoadStoreSpec)
    assert output.store_spec.block_ids.tolist() == [0, 0]
    assert output.store_spec.group_indices.tolist() == [0, 1]
    manager.complete_store(output.keys_to_store, req_context)

    assert manager.lookup(key0, req_context) is LookupResult.HIT
    assert manager.lookup(key1, req_context) is LookupResult.HIT
    load_spec = manager.prepare_load([key1, key0], req_context)
    assert isinstance(load_spec, GroupedCPULoadStoreSpec)
    assert load_spec.block_ids.tolist() == [0, 0]
    assert load_spec.group_indices.tolist() == [1, 0]
    manager.complete_load([key1, key0], req_context)


@pytest.mark.parametrize(
    ("spec_name", "extra_config", "expected_type"),
    [
        (
            "INT4CompressedCPUOffloadingSpec",
            {"int4_group_size": 64},
            INT4CompressedCPUOffloadingSpec,
        ),
        (
            "SVDCompressedCPUOffloadingSpec",
            {"svd_rank": 4},
            SVDCompressedCPUOffloadingSpec,
        ),
    ],
)
def test_specs_load_out_of_tree_and_charge_capacity_by_encoded_bytes(
    spec_name: str,
    extra_config: dict[str, object],
    expected_type: type[CompressedCPUOffloadingSpec],
) -> None:
    spec = OffloadingSpecFactory.create_spec(
        _make_offloading_config(spec_name, extra_config)
    )

    assert isinstance(spec, expected_type)
    assert spec.encoded_bytes_per_worker == spec.compressed_layout.storage_nbytes
    assert spec.encoded_bytes_per_worker < spec.source_layout.raw_nbytes
    assert spec.num_blocks == 4


def test_svd_fp16_storage_expansion_requires_explicit_opt_in() -> None:
    extra_config: dict[str, object] = {
        "svd_rank": 64,
        "svd_factor_dtype": "fp16",
    }
    config = _make_offloading_config(
        "SVDCompressedCPUOffloadingSpec",
        extra_config,
    )
    with pytest.raises(ValueError, match="choose a stronger compression setting"):
        OffloadingSpecFactory.create_spec(config)

    extra_config["allow_compression_expansion"] = True
    spec = OffloadingSpecFactory.create_spec(
        _make_offloading_config(
            "SVDCompressedCPUOffloadingSpec",
            extra_config,
        )
    )

    assert isinstance(spec, SVDCompressedCPUOffloadingSpec)
    assert spec.compressed_layout.codec_id == "svd-fp16-v1"
    assert spec.compressed_layout.storage_nbytes > spec.source_layout.raw_nbytes
    assert spec.num_blocks == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_spec_created_worker_round_trips_through_shared_encoded_slots() -> None:
    spec = OffloadingSpecFactory.create_spec(
        _make_offloading_config(
            "INT4CompressedCPUOffloadingSpec",
            {"int4_group_size": 64},
            engine_id=f"compressed-test-{uuid.uuid4()}",
        )
    )
    assert isinstance(spec, INT4CompressedCPUOffloadingSpec)
    kv_caches, gpu_matrices = _canonical_gpu_caches(
        spec.source_layout,
        blocks_per_chunk=spec.blocks_per_chunk,
        num_gpu_blocks=8,
    )
    source_block_ids = [1, 3]
    destination_block_ids = [5, 6]
    torch.manual_seed(23)
    for gpu_matrix in gpu_matrices:
        gpu_matrix[source_block_ids] = torch.randn_like(gpu_matrix[source_block_ids])
    expected = tuple(
        gpu_matrix[source_block_ids].clone() for gpu_matrix in gpu_matrices
    )

    worker = spec.get_worker(kv_caches)
    assert isinstance(worker, INT4CompressedCPUOffloadingWorker)
    try:
        assert worker.submit_store(
            1,
            GPULoadStoreSpec(
                source_block_ids,
                group_sizes=(spec.blocks_per_chunk,),
                block_indices=(0,),
            ),
            CPULoadStoreSpec([0]),
        )
        _wait_for_one_result(worker, 1)

        for gpu_matrix in gpu_matrices:
            gpu_matrix[destination_block_ids] = 0
        assert worker.submit_load(
            2,
            CPULoadStoreSpec([0]),
            GPULoadStoreSpec(
                destination_block_ids,
                group_sizes=(spec.blocks_per_chunk,),
                block_indices=(0,),
            ),
        )
        _wait_for_one_result(worker, 2)

        for gpu_matrix, expected_pages in zip(gpu_matrices, expected, strict=True):
            actual = gpu_matrix[destination_block_ids]
            relative_error = (actual.float() - expected_pages.float()).norm()
            relative_error /= expected_pages.float().norm()
            assert relative_error.item() < 0.16
    finally:
        worker.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hybrid_worker_compresses_attention_and_preserves_state() -> None:
    spec = OffloadingSpecFactory.create_spec(
        _make_hybrid_offloading_config(engine_id=f"hybrid-test-{uuid.uuid4()}")
    )
    assert isinstance(spec, INT4CompressedCPUOffloadingSpec)
    kv_caches, attention, states = _hybrid_canonical_gpu_caches()
    attention_source_ids = [1, 2]
    state_source_id = 3
    attention_destination_ids = [5, 6]
    state_destination_id = 7
    torch.manual_seed(53)
    for matrix in attention:
        matrix[attention_source_ids] = torch.randn_like(matrix[attention_source_ids])
    for state in states:
        state[state_source_id] = torch.randint(
            0,
            256,
            state[state_source_id].shape,
            dtype=torch.uint8,
            device="cuda:0",
        )
    expected_attention = tuple(
        matrix[attention_source_ids].clone() for matrix in attention
    )
    expected_states = tuple(state[state_source_id].clone() for state in states)

    worker = spec.get_worker(kv_caches)
    assert isinstance(worker, GroupedCompressedCPUOffloadingWorker)
    cpu_spec = GroupedCPULoadStoreSpec([0, 0], [0, 1])
    try:
        assert worker.submit_store(
            1,
            GPULoadStoreSpec(
                [*attention_source_ids, state_source_id],
                group_sizes=(2, 1),
                block_indices=(0, 1),
            ),
            cpu_spec,
        )
        worker.wait({1})
        results = worker.get_finished()
        assert [result.job_id for result in results] == [1]
        expected_transfer_size = sum(
            plan.compression.encoded_layout.storage_nbytes for plan in spec.group_plans
        )
        assert results[0].transfer_size == expected_transfer_size

        for matrix in attention:
            matrix[attention_destination_ids] = 0
        for state in states:
            state[state_destination_id] = 0
        assert worker.submit_load(
            2,
            cpu_spec,
            GPULoadStoreSpec(
                [*attention_destination_ids, 0, state_destination_id],
                group_sizes=(2, 2),
                block_indices=(0, 0),
            ),
        )
        worker.wait({2})
        results = worker.get_finished()
        assert [result.job_id for result in results] == [2]

        for matrix, expected in zip(attention, expected_attention, strict=True):
            actual = matrix[attention_destination_ids]
            relative_error = (actual.float() - expected.float()).norm()
            relative_error /= expected.float().norm()
            assert relative_error.item() < 0.16
        for state, expected in zip(states, expected_states, strict=True):
            assert torch.equal(state[state_destination_id], expected)
    finally:
        worker.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("worker_factory", "max_relative_error"),
    [
        (
            lambda **kwargs: INT4CompressedCPUOffloadingWorker(group_size=64, **kwargs),
            0.16,
        ),
        (
            lambda **kwargs: SVDCompressedCPUOffloadingWorker(rank=2, **kwargs),
            0.05,
        ),
    ],
)
def test_worker_store_then_load_restores_paged_gpu_kv(
    worker_factory: Callable[..., CompressedCPUOffloadingWorker],
    max_relative_error: float,
) -> None:
    blocks_per_chunk = 2
    tokens_per_block = 4
    source_layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=2,
        rows=blocks_per_chunk * tokens_per_block,
        columns=16,
    )
    if max_relative_error < 0.1:
        source_matrices = _low_rank_matrices(source_layout, rank=2, device="cuda:0")
        compressor: KVCompressor = SVDKVCompressor(rank=2)
    else:
        torch.manual_seed(11)
        source_matrices = tuple(
            torch.randn(
                source_layout.rows,
                source_layout.columns,
                dtype=source_layout.dtype,
                device="cuda:0",
            )
            for _ in range(source_layout.matrix_count)
        )
        compressor = INT4KVCompressor(group_size=64)
    layout = compressor.build_layout(source_layout)
    kv_caches, gpu_matrices = _canonical_gpu_caches(
        source_layout,
        blocks_per_chunk,
        num_gpu_blocks=8,
    )
    source_block_ids = [3, 1]
    destination_block_ids = [5, 6]
    for gpu_matrix, source_matrix in zip(gpu_matrices, source_matrices, strict=True):
        gpu_matrix[source_block_ids] = source_matrix.view(
            blocks_per_chunk,
            tokens_per_block,
            source_layout.columns,
        )

    worker = worker_factory(
        kv_caches=kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        tokens_per_block=tokens_per_block,
        num_cpu_blocks=4,
        layout=layout,
    )
    store_gpu_spec = GPULoadStoreSpec(
        source_block_ids,
        group_sizes=(blocks_per_chunk,),
        block_indices=(0,),
    )
    assert worker.submit_store(1, store_gpu_spec, CPULoadStoreSpec([0]))
    _wait_for_one_result(worker, 1)

    for gpu_matrix in gpu_matrices:
        gpu_matrix[destination_block_ids] = 0
    load_gpu_spec = GPULoadStoreSpec(
        destination_block_ids,
        group_sizes=(blocks_per_chunk,),
        block_indices=(0,),
    )
    assert worker.submit_load(2, CPULoadStoreSpec([0]), load_gpu_spec)
    _wait_for_one_result(worker, 2)

    for gpu_matrix, expected in zip(gpu_matrices, source_matrices, strict=True):
        actual = gpu_matrix[destination_block_ids].reshape_as(expected)
        relative_error = (actual.float() - expected.float()).norm()
        relative_error /= expected.float().norm()
        assert relative_error.item() < max_relative_error
    worker.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_worker_load_can_scatter_only_a_cached_chunk_suffix() -> None:
    blocks_per_chunk = 2
    tokens_per_block = 4
    source_layout = KVCompressionSourceLayout(
        dtype=torch.bfloat16,
        matrix_count=1,
        rows=blocks_per_chunk * tokens_per_block,
        columns=32,
    )
    compressor = INT4KVCompressor(group_size=64)
    layout = compressor.build_layout(source_layout)
    kv_caches, gpu_matrices = _canonical_gpu_caches(
        source_layout,
        blocks_per_chunk,
        num_gpu_blocks=8,
    )
    source_block_ids = [1, 2]
    destination_block_id = 7
    torch.manual_seed(19)
    source_pages = torch.randn_like(gpu_matrices[0][source_block_ids])
    gpu_matrices[0][source_block_ids] = source_pages
    expected_suffix = gpu_matrices[0][source_block_ids[1]].clone()

    worker = INT4CompressedCPUOffloadingWorker(
        group_size=64,
        kv_caches=kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        tokens_per_block=tokens_per_block,
        num_cpu_blocks=4,
        layout=layout,
    )
    assert worker.submit_store(
        1,
        GPULoadStoreSpec(
            source_block_ids,
            group_sizes=(blocks_per_chunk,),
            block_indices=(0,),
        ),
        CPULoadStoreSpec([0]),
    )
    _wait_for_one_result(worker, 1)

    gpu_matrices[0][destination_block_id] = 0
    assert worker.submit_load(
        2,
        CPULoadStoreSpec([0]),
        GPULoadStoreSpec(
            [destination_block_id],
            group_sizes=(1,),
            block_indices=(1,),
        ),
    )
    _wait_for_one_result(worker, 2)

    actual = gpu_matrices[0][destination_block_id]
    relative_error = (actual.float() - expected_suffix.float()).norm()
    relative_error /= expected_suffix.float().norm()
    assert relative_error.item() < 0.16
    worker.shutdown()


def test_svd_worker_preserves_every_compressor_setting() -> None:
    """The worker rebuilds its own compressor, so it must carry every knob.

    Regression: the worker used to construct ``SVDKVCompressor(rank, factor_dtype)``
    and silently drop the algorithm, so a spec configured for the rank-aware
    ``"lowrank"`` path still ran the full ``torch.linalg.svd`` -- 12.7 s per
    256 MiB chunk instead of 1.1 s, with no error and no log line.
    """
    configured = SVDKVCompressor(
        rank=8,
        factor_dtype="fp8",
        algorithm="lowrank",
        lowrank_niter=3,
        lowrank_oversample=4,
        batch_size=2,
    )
    for field in ("rank", "factor_dtype", "algorithm", "lowrank_niter",
                  "lowrank_oversample", "batch_size"):
        assert hasattr(configured, field), field

    layout = configured.build_layout(
        KVCompressionSourceLayout(
            dtype=torch.float16, matrix_count=2, rows=16, columns=16
        )
    )
    worker_kwargs = inspect.signature(
        SVDCompressedCPUOffloadingWorker.__init__
    ).parameters
    for field in ("algorithm", "lowrank_niter", "lowrank_oversample", "batch_size"):
        assert field in worker_kwargs, (
            f"SVDCompressedCPUOffloadingWorker cannot receive {field!r}; "
            "a spec-level codec setting would be silently dropped"
        )
    assert layout.parameter == 8


def test_kv_split_gives_k_and_v_independent_ratios() -> None:
    """K and V can carry different codecs and different compression ratios.

    vLLM's FlashAttention cache packs K and V into the content dimension, so one
    gathered matrix reads [K_h0 | V_h0 | K_h1 | V_h1 | ...].  The split codec
    separates those spans, which matters because K and V do not behave alike: V
    sits much closer to full rank, while K carries the larger magnitude.
    """
    head_dim, heads, rows, matrices = 8, 4, 32, 2
    source = KVCompressionSourceLayout(
        dtype=torch.float16,
        matrix_count=matrices,
        rows=rows,
        columns=2 * heads * head_dim,
    )
    coarse = KVSplitKVCompressor(
        INT4KVCompressor(group_size=128), INT4KVCompressor(group_size=16), head_dim
    )
    fine = KVSplitKVCompressor(
        INT4KVCompressor(group_size=16), INT4KVCompressor(group_size=128), head_dim
    )
    assert coarse.build_layout(source).storage_nbytes == (
        fine.build_layout(source).storage_nbytes
    ), "mirrored group sizes must cost the same bytes"

    mixed = KVSplitKVCompressor(
        SVDKVCompressor(rank=4, algorithm="lowrank"),
        INT4KVCompressor(group_size=32),
        head_dim,
    )
    assert "svd" in mixed.codec_id and "int4" in mixed.codec_id

    torch.manual_seed(3)
    for compressor in (coarse, fine, mixed):
        layout = compressor.build_layout(source)
        sources = tuple(
            torch.randn(rows, source.columns, dtype=torch.float16)
            for _ in range(matrices)
        )
        blob = torch.zeros(layout.storage_nbytes, dtype=torch.uint8)
        compressor.compress_into(sources, blob, layout)
        compressor.validate_blob(blob, layout)
        restored = compressor.decompress(blob, layout)
        assert len(restored) == matrices
        for original, out in zip(sources, restored, strict=True):
            assert out.shape == original.shape
            assert out.dtype == original.dtype

    # A K-only change must leave the V half's bytes untouched.
    layout = coarse.build_layout(source)
    base = tuple(
        torch.randn(rows, source.columns, dtype=torch.float16) for _ in range(matrices)
    )
    perturbed = tuple(m.clone() for m in base)
    for m in perturbed:
        m.view(rows, heads, 2, head_dim)[:, :, 0, :] += 5.0  # K spans only
    blobs = []
    for mats in (base, perturbed):
        blob = torch.zeros(layout.storage_nbytes, dtype=torch.uint8)
        coarse.compress_into(mats, blob, layout)
        blobs.append(blob)
    _, v_layout, _, v_offset = coarse._plan(layout)
    v_bytes = slice(v_offset, v_offset + v_layout.storage_nbytes)
    assert torch.equal(blobs[0][v_bytes], blobs[1][v_bytes]), (
        "perturbing only K changed the V region; the column split is wrong"
    )


@pytest.mark.parametrize(
    "factor_dtype,algorithm,max_rel_err",
    [
        ("fp16", "exact", 5e-3),
        ("fp16", "lowrank", 5e-3),
    ],
)
def test_svd_at_full_rank_is_near_lossless(
    factor_dtype: str, algorithm: str, max_rel_err: float
) -> None:
    """Full-rank SVD must round-trip a matrix, whatever the accuracy story is.

    This is the identity sanity check: with rank == min(rows, columns) nothing is
    truncated, so the only error left is factor precision.  If this fails, the
    SVD codec has an implementation bug and no accuracy number from it means
    anything.  If it passes, the collapse observed at lower ranks really is
    information loss.

    Run with fp16 factors so that factor precision is not a confound; the
    resulting object is larger than its source, which is fine — this
    configuration exists to answer a correctness question, not to save bytes.
    """
    rows = columns = 64
    source = KVCompressionSourceLayout(
        dtype=torch.float32, matrix_count=2, rows=rows, columns=columns
    )
    compressor = SVDKVCompressor(
        rank=min(rows, columns),
        factor_dtype=cast(SVDFactorDType, factor_dtype),
        algorithm=cast(SVDAlgorithm, algorithm),
        lowrank_oversample=0,
    )
    layout = compressor.build_layout(source)
    assert layout.parameter == min(rows, columns), "full rank must survive the layout"

    torch.manual_seed(11)
    sources = tuple(
        torch.randn(rows, columns, dtype=torch.float32) for _ in range(2)
    )
    blob = torch.zeros(layout.storage_nbytes, dtype=torch.uint8)
    compressor.compress_into(sources, blob, layout)
    restored = compressor.decompress(blob, layout)

    for original, out in zip(sources, restored, strict=True):
        rel = ((out - original).norm() / original.norm()).item()
        assert rel < max_rel_err, (
            f"full-rank SVD ({factor_dtype}/{algorithm}) reconstructed with "
            f"relative error {rel:.4f}; nothing is truncated, so this is a bug"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group", [1, 2, 4])
def test_cross_layer_grouping_round_trips_and_widens_the_matrix(group: int) -> None:
    """One matrix may span several adjacent layers (xKV-style grouping).

    Each layer's page is already token-major, so a grouped matrix is those pages
    concatenated along the channel axis.  This checks both halves of that: the
    matrix really is `group` layers wide, and gather/scatter remain exact
    inverses under a byte-identity codec.
    """
    num_layers, blocks_per_chunk, tokens_per_block, layer_columns = 4, 2, 8, 6
    matrix_count = num_layers // group
    layout_source = KVCompressionSourceLayout(
        dtype=torch.float16,
        matrix_count=num_layers,
        rows=blocks_per_chunk * tokens_per_block,
        columns=layer_columns,
    )
    kv_caches, matrices = _canonical_gpu_caches(layout_source, blocks_per_chunk, 4)

    grouped_source = KVCompressionSourceLayout(
        dtype=torch.float16,
        matrix_count=matrix_count,
        rows=blocks_per_chunk * tokens_per_block,
        columns=layer_columns * group,
    )
    compressor = RawKVCompressor()
    worker = RawCompressedCPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=blocks_per_chunk,
        tokens_per_block=tokens_per_block,
        num_cpu_blocks=2,
        layout=compressor.build_layout(grouped_source),
        layers_per_matrix=group,
    )

    torch.manual_seed(5)
    for matrix in matrices:
        matrix.copy_(torch.randn_like(matrix))
    original = [matrix.clone() for matrix in matrices]

    gathered, _ = worker._gather_matrices(np.arange(blocks_per_chunk))
    assert len(gathered) == matrix_count
    assert gathered[0].shape == (grouped_source.rows, layer_columns * group)

    # column block l of a grouped matrix must be layer l of that group
    for m, group_matrix in enumerate(gathered):
        for l in range(group):
            layer = original[m * group + l][:blocks_per_chunk].reshape(
                grouped_source.rows, layer_columns
            )
            got = group_matrix[:, l * layer_columns : (l + 1) * layer_columns]
            assert torch.equal(got, layer), (
                f"matrix {m} column block {l} is not layer {m * group + l}"
            )

    for matrix in matrices:
        matrix.zero_()
    worker._scatter_matrices(gathered, np.arange(blocks_per_chunk), 0)
    for restored, expected in zip(matrices, original, strict=True):
        assert torch.equal(
            restored[:blocks_per_chunk], expected[:blocks_per_chunk]
        ), "gather/scatter are not inverses under cross-layer grouping"


def test_rope_key_transform_is_invertible_and_leaves_values_alone() -> None:
    """Un-rotating keys must be exact and must not touch value columns.

    vLLM stores keys post-RoPE, which roughly quadruples the rank a key matrix
    needs.  RoPE is orthogonal and fixed by token position, and a prefix-cache
    chunk always covers a known token range, so the worker can un-rotate before
    compressing and re-rotate on load -- keeping the pre-RoPE accuracy gain on
    the keys the model actually attends to.
    """
    head_dim, heads, rows, start = 8, 3, 16, 32
    transform = RoPEKeyTransform(
        head_dim=head_dim, max_position=128, theta=10000.0, device="cpu"
    )

    mask = transform.key_column_mask(2 * heads * head_dim)
    assert mask.sum().item() == heads * head_dim, "half the columns are keys"
    # per-head layout is [K | V]: the first head_dim columns are keys
    assert bool(mask[0]) and not bool(mask[head_dim])

    torch.manual_seed(2)
    original = torch.randn(rows, 2 * heads * head_dim)
    work = original.clone()

    transform.apply_(work, start, inverse=True)
    values = ~mask
    assert torch.equal(work[:, values], original[:, values]), (
        "value columns were modified by the key rotation"
    )
    assert not torch.allclose(work[:, mask], original[:, mask]), (
        "key columns were not rotated at all"
    )

    transform.apply_(work, start, inverse=False)
    assert torch.allclose(work, original, atol=1e-5), (
        "forward after inverse must return the original keys"
    )

    # the rotation is orthogonal, so it must preserve the norm of any error
    error = torch.randn_like(original) * 0.01
    perturbed = original + error
    rotated_error = perturbed.clone()
    transform.apply_(rotated_error, start, inverse=True)
    base = original.clone()
    transform.apply_(base, start, inverse=True)
    assert torch.allclose(
        (rotated_error - base).norm(), error.norm(), rtol=1e-4
    ), "rotation changed the error norm; relative error would not be preserved"


def test_int4_dequant_scratch_is_bounded_by_slice_not_object_size() -> None:
    """Dequantising must not materialise the whole object in FP32 at once.

    A 32768-token chunk of 4 grouped layers is ~1.07e9 elements; expanding all of
    it to FP32 asks for a single 4 GiB allocation, which killed the engine on
    long prompts while every short-prompt test passed.  Slicing bounds the
    scratch regardless of object size, and must not change the result.
    """
    source = KVCompressionSourceLayout(
        dtype=torch.float16, matrix_count=2, rows=64, columns=128
    )
    torch.manual_seed(9)
    matrices = tuple(
        torch.randn(64, 128, dtype=torch.float16) for _ in range(2)
    )

    whole = INT4KVCompressor(group_size=64, dequant_slice_elements=1 << 24)
    sliced = INT4KVCompressor(group_size=64, dequant_slice_elements=128)
    assert sliced.dequant_slice_elements < source.elements_per_matrix, (
        "the sliced case must actually take more than one pass"
    )

    outputs, blobs = [], []
    for compressor in (whole, sliced):
        layout = compressor.build_layout(source)
        blob = torch.zeros(layout.storage_nbytes, dtype=torch.uint8)
        compressor.compress_into(matrices, blob, layout)
        blobs.append(blob)
        outputs.append(compressor.decompress(blob, layout))

    # both directions must slice: the encoded bytes themselves must match
    assert torch.equal(blobs[0], blobs[1]), (
        "slicing changed the encoded object; quantisation must be exact"
    )
    for one, many in zip(outputs[0], outputs[1], strict=True):
        assert torch.equal(one, many), (
            "slicing changed the dequantised values; it must be exact"
        )

    with pytest.raises(ValueError):
        INT4KVCompressor(group_size=64, dequant_slice_elements=8)
