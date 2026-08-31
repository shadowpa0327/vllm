# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compression-aware worker implementations for CPU KV offloading."""

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.compressed_cpu.rope import RoPEKeyTransform
from vllm.v1.kv_offload.compressed_cpu.codec import (
    CompressedKVLayout,
    INT4KVCompressor,
    KVCompressor,
    RawKVCompressor,
    SVDAlgorithm,
    SVDFactorDType,
    SVDKVCompressor,
)
from vllm.v1.kv_offload.compressed_cpu.manager import GroupedCPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


@dataclass(frozen=True)
class _MatrixBinding:
    tensor: torch.Tensor
    byte_offset: int
    page_nbytes: int
    # Canonical per-block trailing shape when it is heads-major (e.g.
    # FlashAttention's (num_kv_heads, block_size, 2 * head_size)).  Gather
    # permutes such pages to token-major rows; None means already token-major.
    page_shape: tuple[int, ...] | None = None


@dataclass
class _CompressedTransfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    transfer_nbytes: int
    keepalive: list[torch.Tensor]


@dataclass
class _GroupedTransfer:
    job_id: int
    pending_groups: set[int]
    success: bool = True
    transfer_nbytes: int = 0
    transfer_time: float = 0.0


class CompressedCPUOffloadingWorker(OffloadingWorker, ABC):
    """ABC for compressed GPU-to-CPU offloading.

    The base class owns block gathering, CPU slot I/O, CUDA event ordering, and
    scattering. Concrete subclasses define the physical compression algorithm
    through :meth:`compress` and :meth:`decompress`.

    Args:
        kv_caches: Canonical vLLM KV cache tensors.
        blocks_per_chunk: GPU pages represented by one compressed CPU object.
        tokens_per_block: Logical tokens represented by one GPU page.
        num_cpu_blocks: Number of compressed CPU slots.
        layout: Fixed encoded object layout.
        pages_per_object: Physical GPU pages gathered into one encoded object.
            Defaults to ``blocks_per_chunk``. State groups may use one page for
            each logically chunk-sized checkpoint object.
        mmap_region: Optional shared CPU slot region.
        log_operations: Emit per-job compression lifecycle logs.

    Notes:
        This prototype supports one uniform KV cache group. Its source matrices
        must either have one canonical reference per matrix or one concatenated
        reference containing every matrix.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        tokens_per_block: int,
        num_cpu_blocks: int,
        layout: CompressedKVLayout,
        pages_per_object: int | None = None,
        mmap_region: SharedOffloadRegion | None = None,
        log_operations: bool = False,
        layers_per_matrix: int = 1,
        rope_transform: "RoPEKeyTransform | None" = None,
    ) -> None:
        # Keys are stored post-RoPE, but RoPE inflates their rank sharply.  When
        # a transform is supplied the worker un-rotates keys before compressing
        # and re-rotates after decompressing, so the codec sees pre-RoPE keys.
        self.rope_transform = rope_transform
        if layers_per_matrix < 1 or layout.source.columns % layers_per_matrix:
            raise ValueError(
                f"{layout.source.columns} columns cannot be split across "
                f"layers_per_matrix={layers_per_matrix}"
            )
        self.layers_per_matrix = layers_per_matrix
        if blocks_per_chunk < 1 or tokens_per_block < 1:
            raise ValueError("blocks_per_chunk and tokens_per_block must be positive")
        if pages_per_object is None:
            pages_per_object = blocks_per_chunk
        if pages_per_object < 1 or layout.source.rows % pages_per_object:
            raise ValueError(
                "compression rows must divide evenly across the physical "
                "pages in one encoded object"
            )
        source_rows_per_page = layout.source.rows // pages_per_object
        if (
            pages_per_object == blocks_per_chunk
            and source_rows_per_page != tokens_per_block
        ):
            raise ValueError(
                f"compression rows per page={source_rows_per_page} do not match "
                f"tokens_per_block={tokens_per_block}"
            )
        if len(kv_caches.group_data_refs) != 1:
            raise NotImplementedError(
                "compressed CPU offloading currently supports one KV cache group"
            )

        self.layout = layout
        self.blocks_per_chunk = blocks_per_chunk
        self.tokens_per_block = tokens_per_block
        self.pages_per_object = pages_per_object
        self.source_rows_per_page = source_rows_per_page
        self.log_operations = log_operations
        self._device: torch.device
        self._matrix_bindings = self._build_matrix_bindings(kv_caches)
        self._mmap_region = mmap_region

        if mmap_region is not None and PIN_MEMORY:
            pin_mmap_region(mmap_region)
        if mmap_region is not None:
            compressed_cache = mmap_region.create_next_view(layout.storage_nbytes)
            self._compressed_cache = compressed_cache.view(torch.uint8)
            self._host_non_blocking = mmap_region.is_pinned
        else:
            started = time.monotonic()
            self._compressed_cache = torch.zeros(
                (num_cpu_blocks, layout.storage_nbytes),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            self._host_non_blocking = PIN_MEMORY
            logger.info(
                "Allocated %d compressed CPU slots of %d bytes in %.3f s",
                num_cpu_blocks,
                layout.storage_nbytes,
                time.monotonic() - started,
            )

        self._transfer_events: dict[int, torch.Event] = {}
        self._transfers: deque[_CompressedTransfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    @abstractmethod
    def compress(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Compress gathered GPU matrices into an encoded GPU object.

        Args:
            source_matrices: Matrices matching :attr:`layout`.
            destination: GPU byte tensor with exactly one encoded slot.
        """

    @abstractmethod
    def decompress(self, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reconstruct GPU matrices from an encoded GPU object.

        Args:
            source: GPU byte tensor containing exactly one encoded slot.

        Returns:
            Reconstructed matrices matching :attr:`layout`.
        """

    @property
    @abstractmethod
    def compressor(self) -> KVCompressor:
        """Return the concrete physical compressor."""

    @override
    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        """Gather, compress, and asynchronously copy GPU KV objects to CPU.

        Args:
            job_id: Unique transfer identifier.
            src_spec: Full, aligned GPU pages to encode.
            dst_spec: Compressed CPU slot IDs.

        Returns:
            True after the asynchronous work is submitted.

        Raises:
            ValueError: If the store does not contain complete aligned chunks.
            TypeError: If ``dst_spec`` is not a CPU slot specification.
        """
        if not isinstance(dst_spec, CPULoadStoreSpec):
            raise TypeError("compressed CPU stores require CPULoadStoreSpec")
        self._ensure_new_job(job_id)
        self._validate_gpu_spec(src_spec)
        gpu_block_ids = src_spec.block_ids
        cpu_block_ids = dst_spec.block_ids
        logical_start = int(src_spec.block_indices[0])
        if (
            self.pages_per_object == self.blocks_per_chunk
            and logical_start % self.blocks_per_chunk
        ):
            raise ValueError("compressed stores must start at a chunk boundary")
        expected_gpu_blocks = len(cpu_block_ids) * self.pages_per_object
        if len(gpu_block_ids) != expected_gpu_blocks:
            raise ValueError(
                f"compressed store has {len(gpu_block_ids)} GPU pages; "
                f"expected {expected_gpu_blocks} for {len(cpu_block_ids)} slots"
            )
        for cpu_block_id in cpu_block_ids:
            self._cpu_slot(int(cpu_block_id))

        stream, start_event, end_event = self._new_transfer_resources(
            wait_for_compute=True
        )
        keepalive: list[torch.Tensor] = []
        with current_platform.stream(stream):
            start_event.record(stream)
            for index, cpu_block_id in enumerate(cpu_block_ids):
                begin = index * self.pages_per_object
                end = begin + self.pages_per_object
                matrices, gpu_indices = self._gather_matrices(gpu_block_ids[begin:end])
                if os.environ.get("COMPRESSED_DEBUG_ROUNDTRIP"):
                    # post-RoPE snapshot for the load-time comparison
                    self._dbg_rows = [m[:2048].clone() for m in matrices]
                if self.rope_transform is not None:
                    chunk_start = (
                        logical_start + index * self.pages_per_object
                    ) * self.tokens_per_block
                    matrices = tuple(
                        self.rope_transform.apply_(m, chunk_start, inverse=True)
                        for m in matrices
                    )
                encoded = torch.empty(
                    self.layout.storage_nbytes,
                    dtype=torch.uint8,
                    device=matrices[0].device,
                )
                self.compress(matrices, encoded)
                self._cpu_slot(int(cpu_block_id)).copy_(
                    encoded,
                    non_blocking=self._host_non_blocking,
                )
                keepalive.extend((*matrices, gpu_indices, encoded))
            end_event.record(stream)

        self._track_transfer(
            job_id,
            stream,
            start_event,
            end_event,
            len(cpu_block_ids) * self.layout.storage_nbytes,
            keepalive,
        )
        if self.log_operations:
            logger.info(
                "[compressed-offload][store] job=%d codec=%s objects=%d "
                "gpu_pages=%d encoded_bytes=%d",
                job_id,
                self.layout.codec_id,
                len(cpu_block_ids),
                len(gpu_block_ids),
                len(cpu_block_ids) * self.layout.storage_nbytes,
            )
        return True

    @override
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        """Copy, decompress, and scatter CPU objects back into paged GPU KV.

        Args:
            job_id: Unique transfer identifier.
            src_spec: Compressed CPU slot IDs.
            dst_spec: Destination GPU pages, optionally covering a chunk suffix.

        Returns:
            True after the asynchronous work is submitted.

        Raises:
            ValueError: If CPU slots and the requested GPU page range disagree.
            TypeError: If ``src_spec`` is not a CPU slot specification.
        """
        if not isinstance(src_spec, CPULoadStoreSpec):
            raise TypeError("compressed CPU loads require CPULoadStoreSpec")
        self._ensure_new_job(job_id)
        self._validate_gpu_spec(dst_spec)
        cpu_block_ids = src_spec.block_ids
        gpu_block_ids = dst_spec.block_ids
        page_offset = (
            int(dst_spec.block_indices[0]) % self.blocks_per_chunk
            if self.pages_per_object == self.blocks_per_chunk
            else 0
        )
        expected_cpu_blocks = (
            page_offset + len(gpu_block_ids) + self.pages_per_object - 1
        ) // self.pages_per_object
        if len(cpu_block_ids) != expected_cpu_blocks:
            raise ValueError(
                f"compressed load has {len(cpu_block_ids)} CPU slots; "
                f"expected {expected_cpu_blocks} for page offset {page_offset} "
                f"and {len(gpu_block_ids)} GPU pages"
            )

        for cpu_block_id in cpu_block_ids:
            self.compressor.validate_blob(
                self._cpu_slot(int(cpu_block_id)), self.layout
            )

        stream, start_event, end_event = self._new_transfer_resources(
            wait_for_compute=False
        )
        keepalive: list[torch.Tensor] = []
        gpu_cursor = 0
        with current_platform.stream(stream):
            start_event.record(stream)
            for index, cpu_block_id in enumerate(cpu_block_ids):
                source_slot = self._cpu_slot(int(cpu_block_id))
                encoded = torch.empty(
                    self.layout.storage_nbytes,
                    dtype=torch.uint8,
                    device=self._device,
                )
                encoded.copy_(source_slot, non_blocking=self._host_non_blocking)
                matrices = self.decompress(encoded)
                if self.rope_transform is not None:
                    chunk_base = (
                        int(dst_spec.block_indices[0])
                        - page_offset
                        + index * self.pages_per_object
                    )
                    matrices = tuple(
                        self.rope_transform.apply_(
                            m, chunk_base * self.tokens_per_block, inverse=False
                        )
                        for m in matrices
                    )
                if os.environ.get("COMPRESSED_DEBUG_ROUNDTRIP") and getattr(
                    self, "_dbg_rows", None
                ):
                    for mi, m in enumerate(matrices):
                        ref = self._dbg_rows[mi].float()
                        got = m[:2048].float()
                        rel = ((got - ref).norm() / ref.norm().clamp(min=1e-6)).item()
                        logger.info(
                            "[compressed-offload][dbg-roundtrip] job=%d matrix=%d "
                            "rel_err=%.4f ref_norm=%.1f got_norm=%.1f",
                            job_id, mi, rel, ref.norm().item(), got.norm().item(),
                        )
                chunk_page_offset = page_offset if index == 0 else 0
                page_count = min(
                    self.pages_per_object - chunk_page_offset,
                    len(gpu_block_ids) - gpu_cursor,
                )
                destination_ids = gpu_block_ids[gpu_cursor : gpu_cursor + page_count]
                gpu_indices = self._scatter_matrices(
                    matrices,
                    destination_ids,
                    chunk_page_offset,
                )
                keepalive.extend((encoded, *matrices, gpu_indices))
                gpu_cursor += page_count
            end_event.record(stream)
        if gpu_cursor != len(gpu_block_ids):
            raise ValueError("compressed load did not consume every GPU page")

        self._track_transfer(
            job_id,
            stream,
            start_event,
            end_event,
            len(cpu_block_ids) * self.layout.storage_nbytes,
            keepalive,
        )
        if self.log_operations:
            logger.info(
                "[compressed-offload][load] job=%d codec=%s objects=%d "
                "gpu_pages=%d encoded_bytes=%d first_page_offset=%d",
                job_id,
                self.layout.codec_id,
                len(cpu_block_ids),
                len(gpu_block_ids),
                len(cpu_block_ids) * self.layout.storage_nbytes,
                page_offset,
            )
        return True

    @override
    def get_finished(self) -> list[TransferResult]:
        """Return completed compressed transfers in submission order."""
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.transfer_nbytes,
                    transfer_time=transfer_time,
                )
            )
            self._stream_pool.append(transfer.stream)
            self._event_pool.extend((transfer.start_event, transfer.end_event))
            del self._transfer_events[transfer.job_id]
        return results

    @override
    def wait(self, job_ids: set[int]) -> None:
        """Synchronize selected compressed transfer jobs.

        Args:
            job_ids: Job IDs whose completion events must be synchronized.
        """
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    @override
    def shutdown(self) -> None:
        """Complete transfers and release CPU/GPU worker resources."""
        while self._transfers:
            self._transfers.popleft().end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._matrix_bindings.clear()
        self._compressed_cache = torch.empty(0, dtype=torch.uint8)
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None

    def _build_matrix_bindings(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> list[tuple[_MatrixBinding, ...]]:
        """Bind each compression matrix to the layer pages that compose it.

        With ``layers_per_matrix > 1`` one matrix spans several adjacent layers.
        Each layer's page is already token-major, so the grouped matrix is just
        those pages concatenated along the channel axis -- no transpose needed.
        """
        refs = kv_caches.group_data_refs[0]
        source = self.layout.source
        group = self.layers_per_matrix
        layer_columns = source.columns // group
        layer_nbytes = self.source_rows_per_page * layer_columns * source.dtype.itemsize
        num_layers = source.matrix_count * group

        def byte_tensor(tensor_index: int) -> torch.Tensor:
            canonical = kv_caches.tensors[tensor_index]
            return canonical.tensor.view(torch.int8).view(-1, canonical.page_size_bytes)

        def block_shape(tensor_index: int) -> tuple[int, ...] | None:
            """Trailing per-block shape when the page is heads-major.

            The codec factorises (tokens x channels) matrices and the RoPE
            transform maps rows to positions, so pages must be gathered
            token-major.  FlashAttention stores
            (num_blocks, num_kv_heads, block_size, 2 * head_size): tokens are
            the SECOND trailing dim, and reading the raw bytes row-major would
            hand the codec head-major rows.  Returns the trailing shape when
            that permutation is required, None when rows are already tokens.
            """
            forced = os.environ.get("COMPRESSED_FORCE_PAGE_SHAPE")
            if forced:
                return tuple(int(x) for x in forced.split(","))
            trail = tuple(kv_caches.tensors[tensor_index].tensor.shape[1:])
            if len(trail) > 1:
                if trail[0] == self.source_rows_per_page:
                    return None
                if trail[1] == self.source_rows_per_page:
                    return trail
                raise ValueError(
                    f"cannot locate the token dim in canonical page shape "
                    f"{trail} (tokens_per_page={self.source_rows_per_page})"
                )
            # Canonical tensors are flattened to (num_blocks, page_bytes), so
            # the byte order must come from the attention backend's declared
            # layout.  HND pages are heads-major and need the permutation.
            try:
                from vllm.v1.attention.backends.utils import get_kv_cache_layout

                cache_layout = get_kv_cache_layout()
            except Exception:
                return None
            if cache_layout != "HND":
                return None
            if self.rope_transform is None:
                logger.warning(
                    "KV cache layout is HND but head_dim is unknown; SVD "
                    "codecs will factorise head-major rows"
                )
                return None
            packed = 2 * self.rope_transform.head_dim
            return (layer_columns // packed, self.source_rows_per_page, packed)

        logger.info(
            "[compressed-offload][bindings] refs=%d canonical_shapes=%s",
            len(refs),
            [tuple(t.tensor.shape) for t in kv_caches.tensors[:2]],
        )
        parts: list[_MatrixBinding] = []
        if len(refs) == num_layers:
            for ref in refs:
                if ref.page_size_bytes != layer_nbytes:
                    raise ValueError(
                        f"canonical page has {ref.page_size_bytes} bytes; "
                        f"expected {layer_nbytes} for one layer"
                    )
                parts.append(
                    _MatrixBinding(
                        tensor=byte_tensor(ref.tensor_idx),
                        byte_offset=0,
                        page_nbytes=layer_nbytes,
                        page_shape=block_shape(ref.tensor_idx),
                    )
                )
        elif len(refs) == 1:
            ref = refs[0]
            expected_nbytes = num_layers * layer_nbytes
            if ref.page_size_bytes != expected_nbytes:
                raise ValueError(
                    f"concatenated canonical page has {ref.page_size_bytes} bytes; "
                    f"expected {expected_nbytes}"
                )
            trail = tuple(kv_caches.tensors[ref.tensor_idx].tensor.shape[1:])
            if len(trail) > 1 and trail[0] != self.source_rows_per_page:
                raise NotImplementedError(
                    "concatenated canonical pages with non-token-major layout "
                    f"{trail} are not supported"
                )
            tensor = byte_tensor(ref.tensor_idx)
            parts.extend(
                _MatrixBinding(
                    tensor=tensor,
                    byte_offset=index * layer_nbytes,
                    page_nbytes=layer_nbytes,
                )
                for index in range(num_layers)
            )
        else:
            raise NotImplementedError(
                "compressed CPU offloading requires one canonical reference "
                f"per layer ({num_layers}) or one concatenated reference; "
                f"got {len(refs)}"
            )

        devices = {part.tensor.device for part in parts}
        if len(devices) != 1:
            raise ValueError("all compression matrices must reside on one device")
        self._device = devices.pop()
        return [
            tuple(parts[i * group : (i + 1) * group])
            for i in range(source.matrix_count)
        ]

    def _gather_matrices(
        self,
        block_ids: np.ndarray,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        gpu_indices = torch.as_tensor(
            block_ids.astype(np.int64, copy=False),
            dtype=torch.long,
            device=self._device,
        )
        source = self.layout.source
        layer_columns = source.columns // self.layers_per_matrix
        matrices: list[torch.Tensor] = []
        pages = len(gpu_indices)
        for group_parts in self._matrix_bindings:
            pieces = []
            for part in group_parts:
                raw = part.tensor[
                    gpu_indices,
                    part.byte_offset : part.byte_offset + part.page_nbytes,
                ].contiguous()
                elems = raw.view(source.dtype)
                if part.page_shape is None:
                    piece = elems.view(source.rows, layer_columns)
                else:
                    heads, tokens = part.page_shape[0], part.page_shape[1]
                    piece = (
                        elems.view(pages, heads, tokens, -1)
                        .permute(0, 2, 1, 3)
                        .reshape(source.rows, layer_columns)
                    )
                pieces.append(piece)
            matrices.append(
                pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
            )
        return tuple(matrices), gpu_indices

    def _scatter_matrices(
        self,
        matrices: tuple[torch.Tensor, ...],
        block_ids: np.ndarray,
        chunk_page_offset: int,
    ) -> torch.Tensor:
        if len(matrices) != len(self._matrix_bindings):
            raise ValueError("decompress returned an unexpected matrix count")
        gpu_indices = torch.as_tensor(
            block_ids.astype(np.int64, copy=False),
            dtype=torch.long,
            device=self._device,
        )
        row_start = chunk_page_offset * self.source_rows_per_page
        row_count = len(block_ids) * self.source_rows_per_page
        layer_columns = self.layout.source.columns // self.layers_per_matrix
        for matrix, group_parts in zip(matrices, self._matrix_bindings, strict=True):
            rows = matrix[row_start : row_start + row_count]
            for index, part in enumerate(group_parts):
                piece = rows[:, index * layer_columns : (index + 1) * layer_columns]
                if part.page_shape is not None:
                    heads = part.page_shape[0]
                    piece = (
                        piece.reshape(
                            len(block_ids), self.source_rows_per_page, heads, -1
                        )
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                matrix_bytes = (
                    piece.contiguous()
                    .view(torch.int8)
                    .view(len(block_ids), part.page_nbytes)
                )
                destination = part.tensor[
                    :, part.byte_offset : part.byte_offset + part.page_nbytes
                ]
                destination.index_copy_(0, gpu_indices, matrix_bytes)
        return gpu_indices

    def _validate_gpu_spec(self, spec: GPULoadStoreSpec) -> None:
        if len(spec.group_sizes) != 1 or len(spec.block_indices) != 1:
            raise NotImplementedError(
                "compressed CPU offloading currently supports one KV cache group"
            )
        if int(spec.group_sizes[0]) != len(spec.block_ids):
            raise ValueError("GPU group size does not match its block ID count")

    def _cpu_slot(self, block_id: int) -> torch.Tensor:
        if block_id < 0 or block_id >= self._compressed_cache.shape[0]:
            raise ValueError(f"compressed CPU slot {block_id} is out of range")
        return self._compressed_cache[block_id]

    def _ensure_new_job(self, job_id: int) -> None:
        if job_id in self._transfer_events:
            raise ValueError(f"compressed transfer job {job_id} already exists")

    def _new_transfer_resources(
        self,
        *,
        wait_for_compute: bool,
    ) -> tuple[torch.cuda.Stream, torch.Event, torch.Event]:
        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        if wait_for_compute:
            stream.wait_stream(current_platform.current_stream())
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)
        return stream, start_event, end_event

    def _track_transfer(
        self,
        job_id: int,
        stream: torch.cuda.Stream,
        start_event: torch.Event,
        end_event: torch.Event,
        transfer_nbytes: int,
        keepalive: list[torch.Tensor],
    ) -> None:
        if job_id in self._transfer_events:
            raise ValueError(f"compressed transfer job {job_id} already exists")
        self._transfer_events[job_id] = end_event
        self._transfers.append(
            _CompressedTransfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                transfer_nbytes=transfer_nbytes,
                keepalive=keepalive,
            )
        )


class DirectCompressedCPUOffloadingWorker(CompressedCPUOffloadingWorker):
    """Worker that uses the compressor the spec already built.

    Every other worker in this module re-derives its compressor from loose
    scalars, so any codec setting the spec knows about is lost unless
    ``_create_group_worker`` is edited to forward it -- silently, with no error.
    Composite codecs cannot be expressed that way at all.  This worker takes the
    compressor object itself and is the preferred path for new codecs.

    Args:
        compressor: Compressor whose ``codec_id`` matches ``layout``.
        kv_caches: Canonical vLLM KV cache tensors.
        blocks_per_chunk: Logical GPU pages represented by one scheduler key.
        tokens_per_block: Logical tokens represented by one GPU page.
        num_cpu_blocks: Number of CPU slots.
        layout: Fixed encoded-object layout.
        pages_per_object: Physical GPU pages gathered for one object.
        mmap_region: Optional shared CPU slot region.
        log_operations: Emit per-job compression lifecycle logs.
    """

    def __init__(
        self,
        *,
        compressor: KVCompressor,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        tokens_per_block: int,
        num_cpu_blocks: int,
        layout: CompressedKVLayout,
        pages_per_object: int | None = None,
        mmap_region: SharedOffloadRegion | None = None,
        log_operations: bool = False,
        layers_per_matrix: int = 1,
        rope_transform: "RoPEKeyTransform | None" = None,
    ) -> None:
        if layout.codec_id != compressor.codec_id:
            raise ValueError(
                f"worker received layout {layout.codec_id!r} for compressor "
                f"{compressor.codec_id!r}"
            )
        self._compressor = compressor
        super().__init__(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            tokens_per_block=tokens_per_block,
            num_cpu_blocks=num_cpu_blocks,
            layout=layout,
            pages_per_object=pages_per_object,
            mmap_region=mmap_region,
            log_operations=log_operations,
            layers_per_matrix=layers_per_matrix,
            rope_transform=rope_transform,
        )

    @property
    @override
    def compressor(self) -> KVCompressor:
        """Return the compressor supplied by the spec."""
        return self._compressor

    @override
    def compress(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Encode gathered matrices with the supplied compressor.

        Args:
            source_matrices: Gathered source matrices.
            destination: Encoded GPU byte destination.
        """
        self._compressor.compress_into(source_matrices, destination, self.layout)

    @override
    def decompress(self, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reconstruct matrices with the supplied compressor.

        Args:
            source: Encoded GPU byte object.

        Returns:
            Matrices matching the worker's source layout.
        """
        return self._compressor.decompress(source, self.layout)


class INT4CompressedCPUOffloadingWorker(CompressedCPUOffloadingWorker):
    """Compressed worker using groupwise symmetric INT4.

    Args:
        kv_caches: Canonical vLLM KV cache tensors.
        blocks_per_chunk: GPU pages represented by one compressed object.
        tokens_per_block: Tokens represented by one GPU page.
        num_cpu_blocks: Number of compressed CPU slots.
        layout: Fixed encoded object layout.
        pages_per_object: Physical GPU pages in one object.
        group_size: Number of source values sharing one FP16 scale.
        mmap_region: Optional shared CPU slot region.
        log_operations: Emit per-job compression lifecycle logs.
    """

    def __init__(
        self,
        *,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        tokens_per_block: int,
        num_cpu_blocks: int,
        layout: CompressedKVLayout,
        group_size: int = 64,
        pages_per_object: int | None = None,
        mmap_region: SharedOffloadRegion | None = None,
        log_operations: bool = False,
        layers_per_matrix: int = 1,
        rope_transform: "RoPEKeyTransform | None" = None,
    ) -> None:
        self._compressor = INT4KVCompressor(group_size=group_size)
        if layout.codec_id != self._compressor.codec_id:
            raise ValueError("INT4 worker received an incompatible layout")
        if layout.parameter != group_size:
            raise ValueError("INT4 worker group size does not match its layout")
        super().__init__(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            tokens_per_block=tokens_per_block,
            num_cpu_blocks=num_cpu_blocks,
            layout=layout,
            pages_per_object=pages_per_object,
            mmap_region=mmap_region,
            log_operations=log_operations,
            layers_per_matrix=layers_per_matrix,
            rope_transform=rope_transform,
        )

    @property
    @override
    def compressor(self) -> KVCompressor:
        """Return the INT4 compressor."""
        return self._compressor

    @override
    def compress(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Compress gathered matrices with INT4.

        Args:
            source_matrices: Gathered source matrices.
            destination: Encoded GPU byte destination.
        """
        self._compressor.compress_into(source_matrices, destination, self.layout)

    @override
    def decompress(self, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decompress an INT4 GPU object.

        Args:
            source: Encoded GPU byte object.

        Returns:
            Reconstructed source matrices.
        """
        return self._compressor.decompress(source, self.layout)


class SVDCompressedCPUOffloadingWorker(CompressedCPUOffloadingWorker):
    """Compressed worker using per-matrix SVD factors.

    Args:
        kv_caches: Canonical vLLM KV cache tensors.
        blocks_per_chunk: GPU pages represented by one compressed object.
        tokens_per_block: Tokens represented by one GPU page.
        num_cpu_blocks: Number of compressed CPU slots.
        layout: Fixed encoded object layout.
        pages_per_object: Physical GPU pages in one object.
        rank: Maximum retained SVD rank.
        factor_dtype: Physical dtype for the left and right SVD factors.
        algorithm: SVD algorithm, ``"exact"`` or ``"lowrank"``.
        lowrank_niter: Power iterations for the ``"lowrank"`` algorithm.
        lowrank_oversample: Extra probe columns for the ``"lowrank"`` algorithm.
        batch_size: Matrices decomposed per batched call.
        mmap_region: Optional shared CPU slot region.
        log_operations: Emit per-job compression lifecycle logs.
    """

    def __init__(
        self,
        *,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        tokens_per_block: int,
        num_cpu_blocks: int,
        layout: CompressedKVLayout,
        rank: int = 32,
        factor_dtype: SVDFactorDType = "fp8",
        algorithm: SVDAlgorithm = "exact",
        lowrank_niter: int = 2,
        lowrank_oversample: int = 16,
        batch_size: int = 8,
        pages_per_object: int | None = None,
        mmap_region: SharedOffloadRegion | None = None,
        log_operations: bool = False,
        layers_per_matrix: int = 1,
        rope_transform: "RoPEKeyTransform | None" = None,
    ) -> None:
        self._compressor = SVDKVCompressor(
            rank=rank,
            factor_dtype=factor_dtype,
            algorithm=algorithm,
            lowrank_niter=lowrank_niter,
            lowrank_oversample=lowrank_oversample,
            batch_size=batch_size,
        )
        expected_rank = min(rank, layout.source.rows, layout.source.columns)
        if layout.codec_id != self._compressor.codec_id:
            raise ValueError("SVD worker received an incompatible layout")
        if layout.parameter != expected_rank:
            raise ValueError("SVD worker rank does not match its layout")
        super().__init__(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            tokens_per_block=tokens_per_block,
            num_cpu_blocks=num_cpu_blocks,
            layout=layout,
            pages_per_object=pages_per_object,
            mmap_region=mmap_region,
            log_operations=log_operations,
            layers_per_matrix=layers_per_matrix,
            rope_transform=rope_transform,
        )

    @property
    @override
    def compressor(self) -> KVCompressor:
        """Return the SVD compressor."""
        return self._compressor

    @override
    def compress(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Compress gathered matrices with SVD.

        Args:
            source_matrices: Gathered source matrices.
            destination: Encoded GPU byte destination.
        """
        self._compressor.compress_into(source_matrices, destination, self.layout)

    @override
    def decompress(self, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decompress an SVD GPU object.

        Args:
            source: Encoded GPU byte object.

        Returns:
            Reconstructed source matrices.
        """
        return self._compressor.decompress(source, self.layout)


class RawCompressedCPUOffloadingWorker(CompressedCPUOffloadingWorker):
    """Compressed-tier worker that preserves one group's bytes exactly.

    Args:
        kv_caches: Canonical vLLM cache tensors for one group.
        blocks_per_chunk: Logical GPU pages represented by one scheduler key.
        tokens_per_block: Logical tokens represented by one GPU page.
        num_cpu_blocks: Number of raw CPU slots.
        layout: Fixed raw encoded-object layout.
        pages_per_object: Physical GPU pages gathered for one object.
        mmap_region: Optional shared CPU slot region.
        log_operations: Emit per-job lifecycle logs.
    """

    def __init__(
        self,
        *,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        tokens_per_block: int,
        num_cpu_blocks: int,
        layout: CompressedKVLayout,
        pages_per_object: int | None = None,
        mmap_region: SharedOffloadRegion | None = None,
        log_operations: bool = False,
        layers_per_matrix: int = 1,
        rope_transform: "RoPEKeyTransform | None" = None,
    ) -> None:
        self._compressor = RawKVCompressor()
        if layout.codec_id != self._compressor.codec_id:
            raise ValueError("raw worker received an incompatible layout")
        super().__init__(
            kv_caches=kv_caches,
            blocks_per_chunk=blocks_per_chunk,
            tokens_per_block=tokens_per_block,
            num_cpu_blocks=num_cpu_blocks,
            layout=layout,
            pages_per_object=pages_per_object,
            mmap_region=mmap_region,
            log_operations=log_operations,
            layers_per_matrix=layers_per_matrix,
            rope_transform=rope_transform,
        )

    @property
    @override
    def compressor(self) -> KVCompressor:
        """Return the raw identity compressor."""
        return self._compressor

    @override
    def compress(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Copy gathered matrices into a raw encoded object.

        Args:
            source_matrices: Gathered source matrices.
            destination: Encoded GPU byte destination.
        """
        self._compressor.compress_into(source_matrices, destination, self.layout)

    @override
    def decompress(self, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Expose exact matrices from a raw encoded object.

        Args:
            source: Encoded GPU byte object.

        Returns:
            Exact source matrix views.
        """
        return self._compressor.decompress(source, self.layout)


class GroupedCompressedCPUOffloadingWorker(OffloadingWorker):
    """Route one scheduler transfer across independent group workers.

    Args:
        workers: One single-group worker per KV cache group, in group order.

    Notes:
        Every child owns a distinct CPU arena and CUDA stream chain. A parent
        job completes only after every participating group child completes.
    """

    def __init__(
        self,
        workers: tuple[CompressedCPUOffloadingWorker, ...],
    ) -> None:
        if not workers:
            raise ValueError("at least one compressed group worker is required")
        self.workers = workers
        self._transfers: deque[_GroupedTransfer] = deque()
        self._transfers_by_id: dict[int, _GroupedTransfer] = {}

    @override
    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        """Split a GPU-to-CPU transfer by KV cache group.

        Args:
            job_id: Unique parent transfer identifier.
            src_spec: Group-major GPU pages from the scheduler.
            dst_spec: Group-qualified CPU slot IDs.

        Returns:
            True after all participating child transfers are submitted.
        """
        if not isinstance(dst_spec, GroupedCPULoadStoreSpec):
            raise TypeError("grouped stores require GroupedCPULoadStoreSpec")
        gpu_specs = self._split_gpu_spec(src_spec, filter_null_state_pages=False)
        cpu_specs = self._split_cpu_spec(dst_spec)
        groups = self._active_groups(gpu_specs, cpu_specs)
        self._start_transfer(job_id, groups)
        for group_idx in groups:
            self.workers[group_idx].submit_store(
                job_id,
                gpu_specs[group_idx],
                cpu_specs[group_idx],
            )
        logger.info(
            "[compressed-offload][grouped-store] job=%d groups=%s gpu_pages=%s "
            "objects=%s",
            job_id,
            groups,
            tuple(len(gpu_specs[idx].block_ids) for idx in groups),
            tuple(len(cpu_specs[idx].block_ids) for idx in groups),
        )
        return True

    @override
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        """Split a CPU-to-GPU transfer by KV cache group.

        Args:
            job_id: Unique parent transfer identifier.
            src_spec: Group-qualified CPU slot IDs.
            dst_spec: Group-major destination GPU pages.

        Returns:
            True after all participating child transfers are submitted.
        """
        if not isinstance(src_spec, GroupedCPULoadStoreSpec):
            raise TypeError("grouped loads require GroupedCPULoadStoreSpec")
        gpu_specs = self._split_gpu_spec(dst_spec, filter_null_state_pages=True)
        cpu_specs = self._split_cpu_spec(src_spec)
        groups = self._active_groups(gpu_specs, cpu_specs)
        self._start_transfer(job_id, groups)
        for group_idx in groups:
            self.workers[group_idx].submit_load(
                job_id,
                cpu_specs[group_idx],
                gpu_specs[group_idx],
            )
        logger.info(
            "[compressed-offload][grouped-load] job=%d groups=%s gpu_pages=%s "
            "objects=%s",
            job_id,
            groups,
            tuple(len(gpu_specs[idx].block_ids) for idx in groups),
            tuple(len(cpu_specs[idx].block_ids) for idx in groups),
        )
        return True

    @override
    def get_finished(self) -> list[TransferResult]:
        """Return parent jobs whose participating groups all completed.

        Returns:
            Completed parent transfers in submission order.
        """
        for group_idx, worker in enumerate(self.workers):
            for result in worker.get_finished():
                transfer = self._transfers_by_id.get(result.job_id)
                if transfer is None or group_idx not in transfer.pending_groups:
                    raise RuntimeError(
                        "compressed group worker returned an unknown transfer result"
                    )
                transfer.pending_groups.remove(group_idx)
                transfer.success &= result.success
                transfer.transfer_nbytes += result.transfer_size or 0
                transfer.transfer_time = max(
                    transfer.transfer_time,
                    result.transfer_time or 0.0,
                )

        results: list[TransferResult] = []
        while self._transfers and not self._transfers[0].pending_groups:
            transfer = self._transfers.popleft()
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=transfer.success,
                    transfer_size=transfer.transfer_nbytes,
                    transfer_time=transfer.transfer_time,
                )
            )
            del self._transfers_by_id[transfer.job_id]
        return results

    @override
    def wait(self, job_ids: set[int]) -> None:
        """Synchronize selected parent transfers in every child.

        Args:
            job_ids: Parent transfer IDs to wait for.
        """
        for worker in self.workers:
            worker.wait(job_ids)

    @override
    def shutdown(self) -> None:
        """Release all child worker resources."""
        for worker in self.workers:
            worker.shutdown()
        self._transfers.clear()
        self._transfers_by_id.clear()

    def _split_gpu_spec(
        self,
        spec: GPULoadStoreSpec,
        *,
        filter_null_state_pages: bool,
    ) -> tuple[GPULoadStoreSpec, ...]:
        if len(spec.group_sizes) != len(self.workers):
            raise ValueError("GPU spec group count does not match compressed workers")
        specs: list[GPULoadStoreSpec] = []
        cursor = 0
        for group_idx, (group_size, block_index) in enumerate(
            zip(spec.group_sizes, spec.block_indices, strict=True)
        ):
            block_ids = spec.block_ids[cursor : cursor + group_size]
            cursor += group_size
            worker = self.workers[group_idx]
            if (
                filter_null_state_pages
                and worker.pages_per_object != worker.blocks_per_chunk
            ):
                block_ids = block_ids[block_ids != 0]
                block_index = 0
            specs.append(
                GPULoadStoreSpec(
                    block_ids.tolist(),
                    group_sizes=(len(block_ids),),
                    block_indices=(int(block_index),),
                )
            )
        if cursor != len(spec.block_ids):
            raise ValueError("GPU spec contains unassigned block IDs")
        return tuple(specs)

    def _split_cpu_spec(
        self,
        spec: GroupedCPULoadStoreSpec,
    ) -> tuple[CPULoadStoreSpec, ...]:
        specs = []
        for group_idx in range(len(self.workers)):
            block_ids = spec.block_ids[spec.group_indices == group_idx]
            specs.append(CPULoadStoreSpec(block_ids.tolist()))
        known = (spec.group_indices >= 0) & (spec.group_indices < len(self.workers))
        if not bool(known.all()):
            raise ValueError("CPU spec references an unknown compression group")
        return tuple(specs)

    def _active_groups(
        self,
        gpu_specs: tuple[GPULoadStoreSpec, ...],
        cpu_specs: tuple[CPULoadStoreSpec, ...],
    ) -> tuple[int, ...]:
        groups = tuple(
            idx
            for idx, (gpu_spec, cpu_spec) in enumerate(
                zip(gpu_specs, cpu_specs, strict=True)
            )
            if len(gpu_spec.block_ids) or len(cpu_spec.block_ids)
        )
        if not groups:
            raise ValueError("grouped transfer contains no cache objects")
        for idx in groups:
            if not len(gpu_specs[idx].block_ids) or not len(cpu_specs[idx].block_ids):
                raise ValueError(
                    f"group {idx} has CPU objects without GPU pages or vice versa"
                )
        return groups

    def _start_transfer(self, job_id: int, groups: tuple[int, ...]) -> None:
        if job_id in self._transfers_by_id:
            raise ValueError(f"compressed transfer job {job_id} already exists")
        transfer = _GroupedTransfer(job_id=job_id, pending_groups=set(groups))
        self._transfers.append(transfer)
        self._transfers_by_id[job_id] = transfer
