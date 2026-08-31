# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offloading specs for experimental group-isolated compressed CPU KV storage."""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.compressed_cpu.codec import (
    GroupedKVCompressor,
    INT4KVCompressor,
    KVCompressionGroup,
    KVSplitKVCompressor,
    KVCompressionSourceLayout,
    KVCompressor,
    RawKVCompressor,
    SVDAlgorithm,
    SVDFactorDType,
    SVDKVCompressor,
)
from vllm.v1.kv_offload.compressed_cpu.manager import (
    GroupedCPUOffloadingManager,
)
from vllm.v1.kv_offload.compressed_cpu.rope import RoPEKeyTransform
from vllm.v1.kv_offload.compressed_cpu.worker import (
    DirectCompressedCPUOffloadingWorker,
    CompressedCPUOffloadingWorker,
    GroupedCompressedCPUOffloadingWorker,
    INT4CompressedCPUOffloadingWorker,
    RawCompressedCPUOffloadingWorker,
    SVDCompressedCPUOffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig, OffloadingGroupConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

logger = init_logger(__name__)


def _parse_int_option(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class CompressedGroupPlan:
    """Resolved physical storage and codec contract for one KV cache group.

    Args:
        group_idx: vLLM KV cache group index.
        cache_kind: Physical cache family from normalized configuration.
        compression: Source layout, codec, and encoded-object layout.
        pages_per_object: Physical GPU pages gathered for one encoded object.
        aggregate_slot_stride: Page-aligned bytes reserved across all workers.
        num_blocks: Number of local object slots in this group's arena.
    """

    group_idx: int
    cache_kind: str
    compression: KVCompressionGroup
    pages_per_object: int
    aggregate_slot_stride: int
    num_blocks: int


class CompressedCPUOffloadingSpec(OffloadingSpec, ABC):
    """Base spec for fixed-size, group-isolated compressed CPU storage.

    Args:
        config: Normalized native offloading configuration.

    Notes:
        One scheduler chunk size is retained across groups. Each group may use
        an independent codec and CPU arena. Attention groups default to the
        concrete spec's codec; Mamba/GDN state groups default to exact raw
        boundary-state storage.
    """

    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    @override
    def build_metric_definitions(
        cls,
        extra_config: dict[str, Any],
    ) -> dict[str, OffloadingMetricMetadata]:
        """Return the CPU-tier metrics shared with raw offloading.

        Args:
            extra_config: Connector extra configuration.

        Returns:
            Metric definitions emitted by the CPU slot managers.
        """
        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)
        if not config.groups:
            raise ValueError("compressed CPU offloading requires a KV cache group")
        if config.parallel.world_size < 1:
            raise ValueError("compressed CPU offloading requires world_size >= 1")
        self._layers_per_matrix: dict[int, int] = {}
        # Optional pre-RoPE key handling.  RoPE roughly quadruples the rank a key
        # matrix needs, so a low-rank codec should see un-rotated keys; RoPE is
        # orthogonal, so the accuracy gain survives re-rotation on load.
        rope_cfg = self.extra_config.get("prerope", None)
        self._rope_transform: RoPEKeyTransform | None = None
        if rope_cfg:
            if not isinstance(rope_cfg, Mapping):
                raise TypeError("prerope must be a mapping of RoPE parameters")
            self._rope_transform = RoPEKeyTransform(
                head_dim=int(rope_cfg["head_dim"]),
                max_position=int(rope_cfg.get("max_position", 131072)),
                theta=float(rope_cfg.get("theta", 500000.0)),
                factor=float(rope_cfg.get("factor", 1.0)),
                low_freq_factor=float(rope_cfg.get("low_freq_factor", 1.0)),
                high_freq_factor=float(rope_cfg.get("high_freq_factor", 4.0)),
                original_max_position=int(
                    rope_cfg.get("original_max_position", 8192)
                ),
                device=f"cuda:{torch.accelerator.current_device_index()}"
                if torch.cuda.is_available() else "cpu",
            )
        if config.packed_layout and len(config.groups) > 1:
            raise NotImplementedError(
                "group-isolated compression cannot split a packed hybrid KV layout"
            )

        cpu_bytes_to_use = int(self.extra_config.get("cpu_bytes_to_use", 0))
        if cpu_bytes_to_use <= 0:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        alignment = int(self.extra_config.get("compression_alignment", 64))
        unresolved = []
        for group_idx, group_config in enumerate(config.groups):
            source_layout, pages_per_object = self._build_source_layout(
                group_idx,
                group_config,
            )
            compression = KVCompressionGroup(
                group_idx=group_idx,
                source_layout=source_layout,
                compressor=self._create_group_compressor(group_idx, group_config),
                alignment=alignment,
            )
            aggregate_slot_stride = round_up(
                compression.encoded_layout.storage_nbytes * config.parallel.world_size,
                self.BLOCK_SIZE_ALIGNMENT,
            )
            unresolved.append(
                (
                    group_idx,
                    group_config,
                    compression,
                    pages_per_object,
                    aggregate_slot_stride,
                )
            )

        bytes_per_group_set = sum(item[4] for item in unresolved)
        self.num_blocks = cpu_bytes_to_use // bytes_per_group_set
        self.group_plans = tuple(
            CompressedGroupPlan(
                group_idx=group_idx,
                cache_kind=group_config.cache_kind,
                compression=compression,
                pages_per_object=pages_per_object,
                aggregate_slot_stride=aggregate_slot_stride,
                num_blocks=self.num_blocks,
            )
            for (
                group_idx,
                group_config,
                compression,
                pages_per_object,
                aggregate_slot_stride,
            ) in unresolved
        )
        self.compression_groups = GroupedKVCompressor(
            plan.compression for plan in self.group_plans
        )
        self._validate_compression_ratios()

        # Compatibility aliases for the original single-group prototype.
        first = self.group_plans[0]
        self.source_layout = first.compression.source_layout
        self.compressor = first.compression.compressor
        self.compressed_layout = first.compression.encoded_layout
        self.encoded_bytes_per_worker = self.compressed_layout.storage_nbytes
        self.cpu_page_size_per_worker = self.encoded_bytes_per_worker
        self.kv_bytes_per_chunk = first.aggregate_slot_stride

        self.replicated_layout = False
        self.eviction_policy = str(self.extra_config.get("eviction_policy", "lru"))
        self.cache_policy_module_path = self.extra_config.get(
            "cache_policy_module_path"
        )
        self.log_operations = bool(
            self.extra_config.get("compression_log_operations", False)
        )
        self._manager: OffloadingManager | None = None
        self._worker: OffloadingWorker | None = None

        logger.info(
            "Configured group-isolated compressed CPU offloading groups=%d "
            "slots_per_group=%d blocks_per_chunk=%d logical_tokens=%s "
            "bytes_per_group_set=%d",
            len(self.group_plans),
            self.num_blocks,
            self.blocks_per_chunk,
            tuple(
                group.tokens_per_block * self.blocks_per_chunk
                for group in config.groups
            ),
            bytes_per_group_set,
        )
        for plan in self.group_plans:
            layout = plan.compression.encoded_layout
            logger.info(
                "[compressed-offload][group-plan] group=%d kind=%s codec=%s "
                "pages_per_object=%d raw_bytes=%d encoded_bytes=%d ratio=%.3fx "
                "arena_stride=%d slots=%d",
                plan.group_idx,
                plan.cache_kind,
                layout.codec_id,
                plan.pages_per_object,
                layout.source.raw_nbytes,
                layout.storage_nbytes,
                layout.compression_ratio,
                plan.aggregate_slot_stride,
                plan.num_blocks,
            )

    @abstractmethod
    def create_compressor(self) -> KVCompressor:
        """Create the concrete spec's default attention compressor."""

    @override
    def get_manager(self) -> OffloadingManager:
        """Return fixed-slot CPU management for one or many group arenas.

        Returns:
            A native CPU manager for one group or a group-routing manager for
            hybrid cache layouts.
        """
        if self._manager is None:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            if len(self.group_plans) == 1:
                self._manager = CPUOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
            else:
                self._manager = GroupedCPUOffloadingManager(
                    num_blocks_per_group=tuple(
                        plan.num_blocks for plan in self.group_plans
                    ),
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        """Return compression-aware workers bound to canonical GPU caches.

        Args:
            kv_caches: Registered canonical GPU KV caches.

        Returns:
            One compression worker or a group-routing composite worker.

        Raises:
            RuntimeError: If the current platform is not CUDA-like.
            ValueError: If canonical cache groups do not match the plan.
        """
        if self._worker is None:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "compressed CPU offloading currently requires a CUDA-like GPU"
                )
            if len(kv_caches.group_data_refs) != len(self.group_plans):
                raise ValueError(
                    "canonical KV cache group count does not match compression plans"
                )
            workers = tuple(
                self._create_group_worker(
                    plan,
                    CanonicalKVCaches(
                        tensors=kv_caches.tensors,
                        group_data_refs=[kv_caches.group_data_refs[plan.group_idx]],
                    ),
                    self._create_mmap_region(plan),
                )
                for plan in self.group_plans
            )
            self._worker = (
                workers[0]
                if len(workers) == 1
                else GroupedCompressedCPUOffloadingWorker(workers)
            )
        return self._worker

    def _build_source_layout(
        self,
        group_idx: int,
        group: OffloadingGroupConfig,
    ) -> tuple[KVCompressionSourceLayout, int]:
        group_options = self._group_options(group_idx)
        # Layers per compression matrix.  >1 concatenates that many adjacent
        # layers along the channel axis so one factorisation is shared across
        # them (xKV, arXiv 2503.18893).  Grouping only pays when the chunk's
        # token count exceeds the grouped channel count -- see CODEC_ROADMAP.md.
        layers_per_matrix = _parse_int_option(
            group_options.get(
                "layers_per_matrix",
                self.extra_config.get("compression_layers_per_matrix", 1),
            ),
            "layers_per_matrix",
        )
        num_layers = len(group.layer_names)
        if layers_per_matrix < 1 or num_layers % layers_per_matrix:
            raise ValueError(
                f"group {group_idx} has {num_layers} layers, which is not "
                f"divisible by layers_per_matrix={layers_per_matrix}"
            )
        matrix_count = _parse_int_option(
            group_options.get(
                "matrix_count",
                self.extra_config.get(
                    "compression_matrix_count",
                    num_layers // layers_per_matrix,
                ),
            ),
            "matrix_count",
        )
        worker_bytes_per_block = group.worker_kv_bytes_per_block
        if worker_bytes_per_block <= 0:
            if len(self.config.groups) != 1:
                raise ValueError(
                    f"KV cache group {group_idx} is missing per-group byte metadata"
                )
            worker_bytes_per_block = self.config.worker_kv_bytes_per_block

        if group.cache_kind == "mamba":
            dtype = torch.uint8
            pages_per_object = 1
            rows = 1
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE.get(self.config.model.dtype)
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    "compressed attention offloading requires a floating KV cache "
                    f"dtype, got {self.config.model.dtype}"
                )
            pages_per_object = self.blocks_per_chunk
            rows = group.tokens_per_block * pages_per_object

        raw_nbytes = worker_bytes_per_block * pages_per_object
        denominator = matrix_count * rows * dtype.itemsize
        if raw_nbytes <= 0 or raw_nbytes % denominator:
            raise ValueError(
                f"group {group_idx} KV bytes per object ({raw_nbytes}) cannot be "
                f"represented as {matrix_count} matrices with {rows} rows and "
                f"dtype {dtype}"
            )
        self._layers_per_matrix[group_idx] = layers_per_matrix
        return (
            KVCompressionSourceLayout(
                dtype=dtype,
                matrix_count=matrix_count,
                rows=rows,
                columns=raw_nbytes // denominator,
            ),
            pages_per_object,
        )

    def _create_group_compressor(
        self,
        group_idx: int,
        group: OffloadingGroupConfig,
    ) -> KVCompressor:
        options = self._group_options(group_idx)
        if options.get("codec") is None:
            if group.cache_kind == "mamba":
                return RawKVCompressor()
            return self.create_compressor()
        return self._codec_from_options(options, group_idx)

    def _codec_from_options(
        self,
        options: Mapping[str, object],
        group_idx: int,
    ) -> KVCompressor:
        """Build one compressor from a codec options mapping.

        Recurses for composite codecs, so ``kvsplit`` can nest any two codecs.

        Args:
            options: Mapping carrying ``codec`` and its parameters.
            group_idx: KV cache group index, used in error messages.

        Returns:
            The configured compressor.
        """
        codec_name = str(options.get("codec", "")).lower()

        def scalar(key: str, default: object) -> int:
            return _parse_int_option(
                options.get(key, self.extra_config.get(key, default)), key
            )

        if codec_name == "raw":
            return RawKVCompressor()
        if codec_name == "int4":
            return INT4KVCompressor(group_size=scalar("int4_group_size", 64))
        if codec_name == "svd":
            return SVDKVCompressor(
                rank=scalar("svd_rank", 32),
                factor_dtype=cast(
                    SVDFactorDType,
                    str(
                        options.get(
                            "svd_factor_dtype",
                            self.extra_config.get("svd_factor_dtype", "fp8"),
                        )
                    ),
                ),
                algorithm=cast(
                    SVDAlgorithm,
                    str(
                        options.get(
                            "svd_algorithm",
                            self.extra_config.get("svd_algorithm", "exact"),
                        )
                    ),
                ),
                lowrank_niter=scalar("svd_lowrank_niter", 2),
                lowrank_oversample=scalar("svd_lowrank_oversample", 16),
                batch_size=scalar("svd_batch_size", 8),
            )
        if codec_name == "kvsplit":
            halves = []
            for half in ("k", "v"):
                sub = options.get(half)
                if not isinstance(sub, Mapping):
                    raise TypeError(
                        f"compression group {group_idx} kvsplit needs a {half!r} "
                        "mapping naming that half's codec"
                    )
                halves.append(self._codec_from_options(sub, group_idx))
            return KVSplitKVCompressor(
                halves[0],
                halves[1],
                head_dim=scalar("kv_split_head_dim", 128),
            )
        raise ValueError(
            f"compression group {group_idx} has unsupported codec {codec_name!r}"
        )

    def _create_group_worker(
        self,
        plan: CompressedGroupPlan,
        kv_caches: CanonicalKVCaches,
        mmap_region: SharedOffloadRegion | None,
    ) -> CompressedCPUOffloadingWorker:
        compressor = plan.compression.compressor
        tokens_per_block = self.config.groups[plan.group_idx].tokens_per_block
        layout = plan.compression.encoded_layout
        if isinstance(compressor, INT4KVCompressor):
            return INT4CompressedCPUOffloadingWorker(
                layers_per_matrix=self._layers_per_matrix.get(plan.group_idx, 1),
                rope_transform=self._rope_transform,
                group_size=compressor.group_size,
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                tokens_per_block=tokens_per_block,
                num_cpu_blocks=plan.num_blocks,
                layout=layout,
                pages_per_object=plan.pages_per_object,
                mmap_region=mmap_region,
                log_operations=self.log_operations,
            )
        if isinstance(compressor, SVDKVCompressor):
            return SVDCompressedCPUOffloadingWorker(
                layers_per_matrix=self._layers_per_matrix.get(plan.group_idx, 1),
                rope_transform=self._rope_transform,
                rank=compressor.rank,
                factor_dtype=compressor.factor_dtype,
                algorithm=compressor.algorithm,
                lowrank_niter=compressor.lowrank_niter,
                lowrank_oversample=compressor.lowrank_oversample,
                batch_size=compressor.batch_size,
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                tokens_per_block=tokens_per_block,
                num_cpu_blocks=plan.num_blocks,
                layout=layout,
                pages_per_object=plan.pages_per_object,
                mmap_region=mmap_region,
                log_operations=self.log_operations,
            )
        if isinstance(compressor, KVSplitKVCompressor):
            return DirectCompressedCPUOffloadingWorker(
                layers_per_matrix=self._layers_per_matrix.get(plan.group_idx, 1),
                rope_transform=self._rope_transform,
                compressor=compressor,
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                tokens_per_block=tokens_per_block,
                num_cpu_blocks=plan.num_blocks,
                layout=layout,
                pages_per_object=plan.pages_per_object,
                mmap_region=mmap_region,
                log_operations=self.log_operations,
            )
        if isinstance(compressor, RawKVCompressor):
            return RawCompressedCPUOffloadingWorker(
                layers_per_matrix=self._layers_per_matrix.get(plan.group_idx, 1),
                rope_transform=self._rope_transform,
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                tokens_per_block=tokens_per_block,
                num_cpu_blocks=plan.num_blocks,
                layout=layout,
                pages_per_object=plan.pages_per_object,
                mmap_region=mmap_region,
                log_operations=self.log_operations,
            )
        raise TypeError(f"unsupported KV compressor type {type(compressor).__name__}")

    def _create_mmap_region(
        self,
        plan: CompressedGroupPlan,
    ) -> SharedOffloadRegion | None:
        if plan.num_blocks == 0:
            return None
        world_size = self.config.parallel.world_size
        rank = torch.accelerator.current_device_index() % world_size
        digest = hashlib.sha256(plan.compression.encoded_layout.header).hexdigest()[:12]
        return SharedOffloadRegion(
            engine_id=f"{self.config.engine_id}_g{plan.group_idx}_{digest}",
            num_blocks=plan.num_blocks,
            rank=rank,
            kv_bytes_per_block=plan.aggregate_slot_stride,
            cpu_page_size=plan.compression.encoded_layout.storage_nbytes,
        )

    def _group_options(self, group_idx: int) -> Mapping[str, object]:
        configured = self.extra_config.get("compression_groups", {})
        if not isinstance(configured, Mapping):
            raise TypeError("compression_groups must be a mapping")
        options = configured.get(str(group_idx), configured.get(group_idx, {}))
        if not isinstance(options, Mapping):
            raise TypeError(f"compression group {group_idx} options must be a mapping")
        return options

    def _validate_compression_ratios(self) -> None:
        allow_expansion = bool(
            self.extra_config.get("allow_compression_expansion", False)
        )
        for plan in self.group_plans:
            layout = plan.compression.encoded_layout
            if isinstance(plan.compression.compressor, RawKVCompressor):
                continue
            if (
                layout.storage_nbytes >= layout.source.raw_nbytes
                and not allow_expansion
            ):
                raise ValueError(
                    f"group {plan.group_idx} {layout.codec_id} produces "
                    f"{layout.storage_nbytes} bytes for a "
                    f"{layout.source.raw_nbytes}-byte source; choose a stronger "
                    "compression setting"
                )
            if layout.storage_nbytes >= layout.source.raw_nbytes:
                logger.warning(
                    "Allowing diagnostic encoded-object expansion group=%d codec=%s "
                    "raw_bytes=%d encoded_bytes=%d",
                    plan.group_idx,
                    layout.codec_id,
                    layout.source.raw_nbytes,
                    layout.storage_nbytes,
                )


class INT4CompressedCPUOffloadingSpec(CompressedCPUOffloadingSpec):
    """Compressed CPU spec whose attention groups default to symmetric INT4."""

    @override
    def create_compressor(self) -> KVCompressor:
        """Create the configured default INT4 compressor.

        Returns:
            A groupwise symmetric INT4 compressor.
        """
        return INT4KVCompressor(
            group_size=int(self.extra_config.get("int4_group_size", 64))
        )


class SVDCompressedCPUOffloadingSpec(CompressedCPUOffloadingSpec):
    """Compressed CPU spec whose attention groups default to SVD factors."""

    @override
    def create_compressor(self) -> KVCompressor:
        """Create the configured default SVD compressor.

        Returns:
            A per-matrix SVD compressor with configured rank and factor dtype.
        """
        factor_dtype = cast(
            SVDFactorDType,
            str(self.extra_config.get("svd_factor_dtype", "fp8")),
        )
        return SVDKVCompressor(
            rank=int(self.extra_config.get("svd_rank", 32)),
            factor_dtype=factor_dtype,
            algorithm=cast(
                SVDAlgorithm, str(self.extra_config.get("svd_algorithm", "exact"))
            ),
            lowrank_niter=int(self.extra_config.get("svd_lowrank_niter", 2)),
            lowrank_oversample=int(
                self.extra_config.get("svd_lowrank_oversample", 16)
            ),
            batch_size=int(self.extra_config.get("svd_batch_size", 8)),
        )
