# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-size codecs for experimental compressed CPU KV offloading."""

import hashlib
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, override

import torch

_ALIGNMENT = 64
_HEADER_MAGIC = b"VLLMCMP\0"
_HEADER_VERSION = 1
_HEADER_STRUCT = struct.Struct("<8sHHHHIIIQQ16s4x")
_HEADER_SIZE = _HEADER_STRUCT.size
_INT4_CODEC_CODE = 1
_SVD_FP8_CODEC_CODE = 2
_SVD_FP16_CODEC_CODE = 3
_RAW_CODEC_CODE = 4
_KVSPLIT_CODEC_CODE = 5
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_SCALE_DTYPE = torch.float16
_SIGMA_DTYPE = torch.float16
SVDFactorDType = Literal["fp8", "fp16"]
SVDAlgorithm = Literal["exact", "lowrank"]
_SVD_ALGORITHMS: frozenset[str] = frozenset(("exact", "lowrank"))
_SVD_FACTOR_DTYPES: dict[SVDFactorDType, torch.dtype] = {
    "fp8": _FP8_DTYPE,
    "fp16": torch.float16,
}
_DTYPE_CODES = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.uint8: 4,
}
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def _layout_digest(
    codec_id: str,
    parameter: int,
    source: "KVCompressionSourceLayout",
) -> bytes:
    canonical = (
        f"{codec_id}:{parameter}:{source.dtype}:{source.matrix_count}:"
        f"{source.rows}:{source.columns}"
    )
    return hashlib.sha256(canonical.encode("ascii")).digest()[:16]


def _copy_tensor_bytes(
    destination: torch.Tensor,
    offset: int,
    source: torch.Tensor,
) -> None:
    source_bytes = source.contiguous().view(torch.uint8).reshape(-1)
    destination[offset : offset + source_bytes.numel()].copy_(source_bytes)


def _quantize_fp8_factor(factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    factor32 = factor.to(torch.float32)
    amax = factor32.abs().amax()
    scale32 = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax))
    stored_scale = scale32.to(_SCALE_DTYPE)
    safe_scale = torch.where(
        stored_scale.to(torch.float32) > 0,
        stored_scale.to(torch.float32),
        torch.ones_like(scale32),
    )
    quantized = torch.clamp(
        factor32 / safe_scale,
        min=-_FP8_MAX,
        max=_FP8_MAX,
    ).to(_FP8_DTYPE)
    return quantized, stored_scale.reshape(1)


@dataclass(frozen=True)
class KVCompressionSourceLayout:
    """Uniform matrix layout presented to a KV compressor.

    Each matrix represents one layer-like KV payload. Its row axis spans every
    token in one offload chunk and its column axis contains that matrix's KV
    features for one token.

    Args:
        dtype: Source KV dtype. Compression codecs require floating point;
            the raw codec also accepts uint8 physical state pages.
        matrix_count: Number of equally shaped matrices in one object.
        rows: Token rows in each matrix.
        columns: Feature columns in each matrix.
    """

    dtype: torch.dtype
    matrix_count: int
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_CODES:
            raise ValueError(
                "compressed CPU offloading supports float16, bfloat16, "
                f"float32, and uint8 sources, got {self.dtype}"
            )
        if self.matrix_count < 1 or self.rows < 1 or self.columns < 1:
            raise ValueError("matrix_count, rows, and columns must be positive")

    @property
    def elements_per_matrix(self) -> int:
        """Return the source element count in one matrix."""
        return self.rows * self.columns

    @property
    def raw_nbytes(self) -> int:
        """Return the total uncompressed byte size."""
        return self.matrix_count * self.elements_per_matrix * self.dtype.itemsize


@dataclass(frozen=True)
class CompressedKVLayout:
    """Immutable physical contract for one compressed CPU object.

    Args:
        codec_id: Versioned codec identifier.
        codec_code: Numeric codec identifier stored in the header.
        parameter: Codec parameter, such as INT4 group size or SVD rank.
        source: Source matrix layout.
        payload_nbytes: Header plus meaningful encoded payload bytes.
        storage_nbytes: Aligned bytes reserved for one CPU slot.
        header: Exact version and layout header stored in every object.
    """

    codec_id: str
    codec_code: int
    parameter: int
    source: KVCompressionSourceLayout
    payload_nbytes: int
    storage_nbytes: int
    header: bytes

    @property
    def compression_ratio(self) -> float:
        """Return raw bytes divided by encoded slot bytes."""
        return self.source.raw_nbytes / self.storage_nbytes


def _build_layout(
    *,
    codec_id: str,
    codec_code: int,
    parameter: int,
    source: KVCompressionSourceLayout,
    payload_nbytes: int,
    alignment: int,
) -> CompressedKVLayout:
    storage_nbytes = _align_up(payload_nbytes, alignment)
    header = _HEADER_STRUCT.pack(
        _HEADER_MAGIC,
        _HEADER_VERSION,
        codec_code,
        _DTYPE_CODES[source.dtype],
        source.matrix_count,
        parameter,
        source.rows,
        source.columns,
        source.raw_nbytes,
        storage_nbytes,
        _layout_digest(codec_id, parameter, source),
    )
    return CompressedKVLayout(
        codec_id=codec_id,
        codec_code=codec_code,
        parameter=parameter,
        source=source,
        payload_nbytes=payload_nbytes,
        storage_nbytes=storage_nbytes,
        header=header,
    )


class KVCompressor(ABC):
    """Physical compression contract used by compressed offloading workers."""

    @property
    @abstractmethod
    def codec_id(self) -> str:
        """Return a versioned identifier for the physical representation."""

    @abstractmethod
    def build_layout(
        self,
        source: KVCompressionSourceLayout,
        alignment: int = _ALIGNMENT,
    ) -> CompressedKVLayout:
        """Build the fixed-size encoded layout for a source object.

        Args:
            source: Uniform source matrix layout.
            alignment: Encoded CPU slot alignment.

        Returns:
            The immutable physical layout used by encode, decode, and storage.
        """

    @abstractmethod
    def compress_into(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        """Compress matrices into a same-device byte destination.

        Args:
            source_matrices: Floating matrices matching ``layout.source``.
            destination: Contiguous one-dimensional uint8 encoded slot.
            layout: Physical layout returned by :meth:`build_layout`.
        """

    @abstractmethod
    def decompress(
        self,
        source: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> tuple[torch.Tensor, ...]:
        """Reconstruct source matrices from a same-device encoded object.

        Args:
            source: Contiguous one-dimensional uint8 encoded slot.
            layout: Physical layout returned by :meth:`build_layout`.

        Returns:
            Reconstructed matrices in the source dtype.
        """

    def validate_blob(
        self,
        blob: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        """Validate an encoded object's size and static header.

        Args:
            blob: Encoded byte tensor on CPU or an accelerator.
            layout: Expected physical layout.

        Raises:
            ValueError: If the tensor shape, size, or header is incompatible.

        Notes:
            Validating an accelerator tensor copies its header to CPU and
            synchronizes. Workers validate the CPU object before H2D instead.
        """
        self._validate_blob_shape(blob, layout)
        actual_header = bytes(blob[:_HEADER_SIZE].detach().cpu().tolist())
        if actual_header != layout.header:
            raise ValueError(
                f"{layout.codec_id} object header does not match the expected layout"
            )

    def _validate_source_matrices(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        source = layout.source
        if len(source_matrices) != source.matrix_count:
            raise ValueError(
                f"expected {source.matrix_count} source matrices, "
                f"got {len(source_matrices)}"
            )
        for index, matrix in enumerate(source_matrices):
            if matrix.shape != (source.rows, source.columns):
                raise ValueError(
                    f"source matrix {index} has shape {tuple(matrix.shape)}; "
                    f"expected {(source.rows, source.columns)}"
                )
            if matrix.dtype != source.dtype:
                raise ValueError(
                    f"source matrix {index} has dtype {matrix.dtype}; "
                    f"expected {source.dtype}"
                )
            if matrix.device != destination.device:
                raise ValueError("source matrices and encoded slot must share a device")
        self._validate_blob_shape(destination, layout)

    def _validate_blob_shape(
        self,
        blob: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        if blob.dtype != torch.uint8 or blob.ndim != 1 or not blob.is_contiguous():
            raise ValueError(
                "encoded object must be a contiguous one-dimensional uint8"
            )
        if blob.numel() != layout.storage_nbytes:
            raise ValueError(
                f"encoded object has {blob.numel()} bytes; "
                f"expected {layout.storage_nbytes}"
            )


@dataclass(frozen=True)
class KVCompressionGroup:
    """Bind one KV cache group to its source layout and compressor.

    Args:
        group_idx: vLLM KV cache group index.
        source_layout: Matrix grouping presented to the compressor.
        compressor: Compression algorithm used only for this group.
        alignment: Alignment of this group's encoded objects in bytes.

    Notes:
        Each group owns an independent encoded layout. Groups can therefore
        use different matrix groupings, encoded sizes, and algorithms.
    """

    group_idx: int
    source_layout: KVCompressionSourceLayout
    compressor: KVCompressor
    alignment: int = _ALIGNMENT
    encoded_layout: CompressedKVLayout = field(init=False)

    def __post_init__(self) -> None:
        if self.group_idx < 0:
            raise ValueError("compression group index must be non-negative")
        object.__setattr__(
            self,
            "encoded_layout",
            self.compressor.build_layout(self.source_layout, self.alignment),
        )

    def compress_into(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Compress this group's source matrices into an encoded object.

        Args:
            source_matrices: Matrices matching this group's source layout.
            destination: Contiguous uint8 destination matching the encoded layout.
        """
        self.compressor.compress_into(
            source_matrices,
            destination,
            self.encoded_layout,
        )

    def decompress(self, source: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reconstruct this group's source matrices.

        Args:
            source: Encoded object matching this group's encoded layout.

        Returns:
            Reconstructed matrices matching this group's source layout.
        """
        return self.compressor.decompress(source, self.encoded_layout)

    def validate_blob(self, blob: torch.Tensor) -> None:
        """Validate an encoded object against this group's contract.

        Args:
            blob: Encoded byte tensor on CPU or an accelerator.
        """
        self.compressor.validate_blob(blob, self.encoded_layout)


class GroupedKVCompressor:
    """Dispatch independent compression contracts by KV cache group.

    Args:
        groups: Non-empty collection of uniquely indexed compression groups.

    Notes:
        This class does not combine groups into one encoded object. It keeps
        group layouts independent so storage can allocate a separate arena for
        every encoded size.
    """

    def __init__(self, groups: Iterable[KVCompressionGroup]) -> None:
        self.groups = tuple(groups)
        if not self.groups:
            raise ValueError("at least one compression group is required")
        group_indices = tuple(group.group_idx for group in self.groups)
        if len(set(group_indices)) != len(group_indices):
            raise ValueError("compression group indices must be unique")
        self._groups_by_idx = {group.group_idx: group for group in self.groups}

    def get_group(self, group_idx: int) -> KVCompressionGroup:
        """Return the compression contract for a KV cache group.

        Args:
            group_idx: vLLM KV cache group index.

        Returns:
            The matching group compression contract.

        Raises:
            KeyError: If no compressor is configured for ``group_idx``.
        """
        try:
            return self._groups_by_idx[group_idx]
        except KeyError:
            raise KeyError(
                f"no compressor configured for KV cache group {group_idx}"
            ) from None

    def compress_into(
        self,
        group_idx: int,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
    ) -> None:
        """Compress matrices with the algorithm selected for a group.

        Args:
            group_idx: vLLM KV cache group index.
            source_matrices: Matrices matching the selected group's layout.
            destination: Contiguous uint8 destination for the selected group.
        """
        self.get_group(group_idx).compress_into(source_matrices, destination)

    def decompress(
        self,
        group_idx: int,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Reconstruct matrices with the algorithm selected for a group.

        Args:
            group_idx: vLLM KV cache group index.
            source: Encoded object for the selected group.

        Returns:
            Reconstructed matrices matching the selected group's layout.
        """
        return self.get_group(group_idx).decompress(source)

    def validate_blob(self, group_idx: int, blob: torch.Tensor) -> None:
        """Validate an encoded object against its group contract.

        Args:
            group_idx: vLLM KV cache group index.
            blob: Encoded byte tensor on CPU or an accelerator.
        """
        self.get_group(group_idx).validate_blob(blob)


class RawKVCompressor(KVCompressor):
    """Header-validated identity codec for group-specific raw state pages."""

    @property
    def codec_id(self) -> str:
        """Return the versioned raw representation identifier."""
        return "raw-bytes-v1"

    def build_layout(
        self,
        source: KVCompressionSourceLayout,
        alignment: int = _ALIGNMENT,
    ) -> CompressedKVLayout:
        """Build a fixed-size header followed by unmodified source bytes.

        Args:
            source: Uniform source matrix layout.
            alignment: Encoded CPU slot alignment.

        Returns:
            The immutable raw-object layout.
        """
        return _build_layout(
            codec_id=self.codec_id,
            codec_code=_RAW_CODEC_CODE,
            parameter=0,
            source=source,
            payload_nbytes=_HEADER_SIZE + source.raw_nbytes,
            alignment=alignment,
        )

    def compress_into(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        """Copy source matrices behind the validated object header.

        Args:
            source_matrices: Matrices matching ``layout.source``.
            destination: Contiguous one-dimensional uint8 encoded slot.
            layout: Raw layout returned by :meth:`build_layout`.
        """
        self._validate_layout(layout)
        self._validate_source_matrices(source_matrices, destination, layout)
        destination.zero_()
        destination[:_HEADER_SIZE].copy_(
            torch.tensor(
                tuple(layout.header),
                dtype=torch.uint8,
                device=destination.device,
            )
        )
        offset = _HEADER_SIZE
        for matrix in source_matrices:
            _copy_tensor_bytes(destination, offset, matrix)
            offset += matrix.numel() * matrix.element_size()

    def decompress(
        self,
        source: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> tuple[torch.Tensor, ...]:
        """Return matrix views over a raw encoded object.

        Args:
            source: Contiguous one-dimensional uint8 encoded slot.
            layout: Raw layout returned by :meth:`build_layout`.

        Returns:
            Matrix views with the source shape and dtype.
        """
        self._validate_layout(layout)
        self._validate_blob_shape(source, layout)
        matrix_nbytes = layout.source.elements_per_matrix * layout.source.dtype.itemsize
        matrices = []
        for index in range(layout.source.matrix_count):
            offset = _HEADER_SIZE + index * matrix_nbytes
            matrices.append(
                source[offset : offset + matrix_nbytes]
                .view(layout.source.dtype)
                .view(layout.source.rows, layout.source.columns)
            )
        return tuple(matrices)

    def _validate_layout(self, layout: CompressedKVLayout) -> None:
        if layout.codec_id != self.codec_id or layout.parameter != 0:
            raise ValueError("raw compressor and encoded layout are incompatible")


class INT4KVCompressor(KVCompressor):
    """Groupwise symmetric INT4 compressor with FP16 scales.

    Args:
        group_size: Number of flattened KV values sharing one quantization scale.
    """

    def __init__(
        self,
        group_size: int = 64,
        dequant_slice_elements: int = 1 << 24,
    ) -> None:
        if group_size < 1:
            raise ValueError("INT4 group_size must be positive")
        if dequant_slice_elements < group_size:
            raise ValueError(
                "dequant_slice_elements must be at least one group"
            )
        self.group_size = group_size
        # Elements dequantised per pass.  Bounds peak scratch to 4 bytes each,
        # independent of how many tokens or layers one object holds.
        self.dequant_slice_elements = dequant_slice_elements

    @property
    def codec_id(self) -> str:
        """Return the versioned INT4 representation identifier."""
        return "int4-groupwise-v1"

    def build_layout(
        self,
        source: KVCompressionSourceLayout,
        alignment: int = _ALIGNMENT,
    ) -> CompressedKVLayout:
        """Build a fixed-size packed-INT4 layout.

        Args:
            source: Uniform source matrix layout.
            alignment: Encoded CPU slot alignment.

        Returns:
            Layout containing FP16 scales followed by packed nibbles.
        """
        if source.dtype not in _FLOAT_DTYPES:
            raise ValueError("INT4 compression requires floating-point source data")
        group_count = source.elements_per_matrix * source.matrix_count
        group_count = (group_count + self.group_size - 1) // self.group_size
        scale_nbytes = group_count * _SCALE_DTYPE.itemsize
        packed_offset = _align_up(_HEADER_SIZE + scale_nbytes, 16)
        padded_elements = group_count * self.group_size
        packed_nbytes = (padded_elements + 1) // 2
        return _build_layout(
            codec_id=self.codec_id,
            codec_code=_INT4_CODEC_CODE,
            parameter=self.group_size,
            source=source,
            payload_nbytes=packed_offset + packed_nbytes,
            alignment=alignment,
        )

    def compress_into(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        """Quantize and pack source matrices into INT4 bytes.

        Args:
            source_matrices: Floating matrices matching ``layout.source``.
            destination: Contiguous one-dimensional uint8 encoded slot.
            layout: INT4 layout returned by :meth:`build_layout`.
        """
        self._validate_layout(layout)
        self._validate_source_matrices(source_matrices, destination, layout)
        destination.zero_()
        destination[:_HEADER_SIZE].copy_(
            torch.tensor(
                tuple(layout.header), dtype=torch.uint8, device=destination.device
            )
        )

        # Quantise in bounded slices, for the same reason decompress does:
        # expanding every matrix to FP32 at once is a single 4 GiB allocation on
        # a 32768-token chunk of grouped layers, which kills the engine on long
        # prompts while short-prompt tests pass.
        device = destination.device
        total_elements = sum(matrix.numel() for matrix in source_matrices)
        group_count = (total_elements + self.group_size - 1) // self.group_size
        padded_elements = group_count * self.group_size
        flat = [matrix.reshape(-1) for matrix in source_matrices]
        bounds, running = [], 0
        for piece in flat:
            bounds.append((running, running + piece.numel()))
            running += piece.numel()

        def gather(begin: int, end: int) -> torch.Tensor:
            """Elements [begin, end) across the concatenated matrices, as FP32."""
            out = torch.zeros(end - begin, dtype=torch.float32, device=device)
            for piece, (lo, hi) in zip(flat, bounds, strict=True):
                if hi <= begin or lo >= end:
                    continue
                a, b = max(lo, begin), min(hi, end)
                out[a - begin : b - begin] = piece[a - lo : b - lo].to(torch.float32)
            return out

        stored_scales = torch.empty(group_count, dtype=_SCALE_DTYPE, device=device)
        # Pack per slice: a full-object nibble tensor plus its packed copy is
        # ~6 GB of transient for a 64K-token object and OOMs next to a live
        # engine.  group_size is even, and slices are group-aligned, so every
        # slice packs independently and bit-identically.
        assert self.group_size % 2 == 0
        packed = torch.empty(
            (padded_elements + 1) // 2, dtype=torch.uint8, device=device
        )
        groups_per_slice = max(1, self.dequant_slice_elements // self.group_size)
        for first in range(0, group_count, groups_per_slice):
            last = min(first + groups_per_slice, group_count)
            begin, end = first * self.group_size, last * self.group_size
            grouped = gather(begin, min(end, total_elements))
            if grouped.numel() < end - begin:
                grouped = torch.cat(
                    (grouped, grouped.new_zeros(end - begin - grouped.numel()))
                )
            grouped = grouped.view(last - first, self.group_size)
            amax = grouped.abs().amax(dim=1)
            scale32 = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
            slice_scales = scale32.to(_SCALE_DTYPE)
            stored_scales[first:last] = slice_scales
            safe = torch.where(
                slice_scales.to(torch.float32) > 0,
                slice_scales.to(torch.float32),
                torch.ones_like(scale32),
            )
            quantized = torch.round(grouped / safe[:, None]).clamp(-7, 7)
            nibbles = (quantized.to(torch.int16).reshape(-1) + 8).to(torch.uint8)
            packed[begin // 2 : end // 2] = nibbles[0::2] | (nibbles[1::2] << 4)

        scale_offset = _HEADER_SIZE
        scale_nbytes = stored_scales.numel() * _SCALE_DTYPE.itemsize
        packed_offset = _align_up(scale_offset + scale_nbytes, 16)
        _copy_tensor_bytes(destination, scale_offset, stored_scales)
        destination[packed_offset : packed_offset + packed.numel()].copy_(packed)

    def decompress(
        self,
        source: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> tuple[torch.Tensor, ...]:
        """Unpack and dequantize an INT4 object.

        Args:
            source: Contiguous one-dimensional uint8 encoded slot.
            layout: INT4 layout returned by :meth:`build_layout`.

        Returns:
            Reconstructed source matrices.
        """
        self._validate_layout(layout)
        self._validate_blob_shape(source, layout)
        source_layout = layout.source
        total_elements = source_layout.elements_per_matrix * source_layout.matrix_count
        group_count = (total_elements + self.group_size - 1) // self.group_size
        padded_elements = group_count * self.group_size
        scale_nbytes = group_count * _SCALE_DTYPE.itemsize
        scales = source[_HEADER_SIZE : _HEADER_SIZE + scale_nbytes].view(_SCALE_DTYPE)
        packed_offset = _align_up(_HEADER_SIZE + scale_nbytes, 16)
        packed_nbytes = (padded_elements + 1) // 2
        packed = source[packed_offset : packed_offset + packed_nbytes]

        # Dequantise in bounded slices.  Materialising the whole object at once
        # costs 4 bytes per element in FP32 -- 4 GiB for a 32768-token chunk of
        # grouped layers -- which is enough to kill the engine on a long prompt
        # while every unit test, running short prompts, passes.
        values = torch.empty(
            total_elements, dtype=torch.float32, device=source.device
        )
        groups_per_slice = max(1, self.dequant_slice_elements // self.group_size)
        for first in range(0, group_count, groups_per_slice):
            last = min(first + groups_per_slice, group_count)
            begin = first * self.group_size
            end = last * self.group_size
            chunk = packed[begin // 2 : (end + 1) // 2]
            nibbles = torch.empty(
                chunk.numel() * 2, dtype=torch.uint8, device=source.device
            )
            nibbles[0::2] = chunk & 0x0F
            nibbles[1::2] = chunk >> 4
            quantized = nibbles[: end - begin].to(torch.int16) - 8
            slice_values = (
                quantized.view(last - first, self.group_size).to(torch.float32)
                * scales[first:last].to(torch.float32)[:, None]
            ).reshape(-1)
            keep = min(end, total_elements) - begin
            if keep <= 0:
                break
            values[begin : begin + keep] = slice_values[:keep]

        matrix_size = source_layout.elements_per_matrix
        return tuple(
            values[index * matrix_size : (index + 1) * matrix_size]
            .view(source_layout.rows, source_layout.columns)
            .to(source_layout.dtype)
            for index in range(source_layout.matrix_count)
        )

    def _validate_layout(self, layout: CompressedKVLayout) -> None:
        if layout.codec_id != self.codec_id or layout.parameter != self.group_size:
            raise ValueError("INT4 compressor and encoded layout are incompatible")


@dataclass(frozen=True)
class _SVDMatrixLayout:
    u_offset: int
    u_scale_offset: int | None
    sigma_offset: int
    right_offset: int
    right_scale_offset: int | None


def _build_svd_matrix_layouts(
    source: KVCompressionSourceLayout,
    rank: int,
    factor_dtype: SVDFactorDType,
) -> tuple[tuple[_SVDMatrixLayout, ...], int]:
    factor_itemsize = _SVD_FACTOR_DTYPES[factor_dtype].itemsize
    stores_scales = factor_dtype == "fp8"
    cursor = _HEADER_SIZE
    matrices: list[_SVDMatrixLayout] = []
    for _ in range(source.matrix_count):
        cursor = _align_up(cursor, _ALIGNMENT)
        u_offset = cursor
        cursor += source.rows * rank * factor_itemsize
        u_scale_offset = None
        if stores_scales:
            cursor = _align_up(cursor, _SCALE_DTYPE.itemsize)
            u_scale_offset = cursor
            cursor += _SCALE_DTYPE.itemsize
        cursor = _align_up(cursor, _SIGMA_DTYPE.itemsize)
        sigma_offset = cursor
        cursor += rank * _SIGMA_DTYPE.itemsize
        cursor = _align_up(cursor, 16)
        right_offset = cursor
        cursor += source.columns * rank * factor_itemsize
        right_scale_offset = None
        if stores_scales:
            cursor = _align_up(cursor, _SCALE_DTYPE.itemsize)
            right_scale_offset = cursor
            cursor += _SCALE_DTYPE.itemsize
        matrices.append(
            _SVDMatrixLayout(
                u_offset=u_offset,
                u_scale_offset=u_scale_offset,
                sigma_offset=sigma_offset,
                right_offset=right_offset,
                right_scale_offset=right_scale_offset,
            )
        )
    return tuple(matrices), _align_up(cursor, _ALIGNMENT)


class SVDKVCompressor(KVCompressor):
    """Per-matrix truncated SVD compressor with selectable factor precision.

    Args:
        rank: Maximum retained SVD rank for each matrix.
        factor_dtype: Physical dtype for the left and right SVD factors.
        algorithm: ``"exact"`` runs a full :func:`torch.linalg.svd` and truncates;
            ``"lowrank"`` runs :func:`torch.svd_lowrank`, whose cost scales with
            the retained rank instead of the full matrix dimension.
        lowrank_niter: Power iterations used by the ``"lowrank"`` algorithm.
        lowrank_oversample: Extra probe columns beyond ``rank`` used by the
            ``"lowrank"`` algorithm to sharpen the retained subspace.
        batch_size: Matrices decomposed per batched call.  Batching keeps the
            decomposition off the Python loop; lower it to bound peak memory.
    """

    def __init__(
        self,
        rank: int = 32,
        factor_dtype: SVDFactorDType = "fp8",
        algorithm: SVDAlgorithm = "exact",
        lowrank_niter: int = 2,
        lowrank_oversample: int = 16,
        batch_size: int = 8,
    ) -> None:
        if rank < 1 or rank > 65535:
            raise ValueError("SVD rank must be in [1, 65535]")
        if factor_dtype not in _SVD_FACTOR_DTYPES:
            raise ValueError("SVD factor_dtype must be 'fp8' or 'fp16'")
        if algorithm not in _SVD_ALGORITHMS:
            raise ValueError("SVD algorithm must be 'exact' or 'lowrank'")
        if lowrank_niter < 0:
            raise ValueError("SVD lowrank_niter must be non-negative")
        if lowrank_oversample < 0:
            raise ValueError("SVD lowrank_oversample must be non-negative")
        if batch_size < 1:
            raise ValueError("SVD batch_size must be positive")
        self.rank = rank
        self.factor_dtype = factor_dtype
        self.algorithm = algorithm
        self.lowrank_niter = lowrank_niter
        self.lowrank_oversample = lowrank_oversample
        self.batch_size = batch_size

    def _decompose(
        self,
        stacked: torch.Tensor,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose a batch of matrices into rank-truncated SVD factors.

        Args:
            stacked: Float32 batch of shape ``(batch, rows, columns)``.
            rank: Retained rank.

        Returns:
            ``(u, sigma, right)`` shaped ``(batch, rows, rank)``,
            ``(batch, rank)`` and ``(batch, columns, rank)``, so that
            ``(u * sigma) @ right.transpose(-2, -1)`` reconstructs the input.
        """
        if self.algorithm == "lowrank":
            probe = min(
                rank + self.lowrank_oversample,
                stacked.shape[-2],
                stacked.shape[-1],
            )
            u, sigma, v = torch.svd_lowrank(
                stacked, q=probe, niter=self.lowrank_niter
            )
            return u[..., :rank], sigma[..., :rank], v[..., :rank]
        u, sigma, vh = torch.linalg.svd(stacked, full_matrices=False)
        return (
            u[..., :rank],
            sigma[..., :rank],
            vh[..., :rank, :].transpose(-2, -1),
        )

    @property
    def codec_id(self) -> str:
        """Return the versioned, precision-specific SVD identifier."""
        return f"svd-{self.factor_dtype}-v1"

    def build_layout(
        self,
        source: KVCompressionSourceLayout,
        alignment: int = _ALIGNMENT,
    ) -> CompressedKVLayout:
        """Build a fixed-size FP8-factor SVD layout.

        Args:
            source: Uniform source matrix layout.
            alignment: Encoded CPU slot alignment.

        Returns:
            Layout containing U/B factors in the configured precision and
            FP16 singular values.
        """
        if source.dtype not in _FLOAT_DTYPES:
            raise ValueError("SVD compression requires floating-point source data")
        retained_rank = min(self.rank, source.rows, source.columns)
        _, payload_nbytes = _build_svd_matrix_layouts(
            source,
            retained_rank,
            self.factor_dtype,
        )
        return _build_layout(
            codec_id=self.codec_id,
            codec_code=(
                _SVD_FP8_CODEC_CODE
                if self.factor_dtype == "fp8"
                else _SVD_FP16_CODEC_CODE
            ),
            parameter=retained_rank,
            source=source,
            payload_nbytes=payload_nbytes,
            alignment=alignment,
        )

    def compress_into(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        """Decompose and encode source matrices into FP8 SVD factors.

        Args:
            source_matrices: Floating matrices matching ``layout.source``.
            destination: Contiguous one-dimensional uint8 encoded slot.
            layout: SVD layout returned by :meth:`build_layout`.
        """
        self._validate_layout(layout)
        self._validate_source_matrices(source_matrices, destination, layout)
        destination.zero_()
        destination[:_HEADER_SIZE].copy_(
            torch.tensor(
                tuple(layout.header), dtype=torch.uint8, device=destination.device
            )
        )
        matrix_layouts, _ = _build_svd_matrix_layouts(
            layout.source,
            layout.parameter,
            self.factor_dtype,
        )
        rank = layout.parameter
        for start in range(0, len(source_matrices), self.batch_size):
            batch = source_matrices[start : start + self.batch_size]
            stacked = torch.stack([matrix.to(torch.float32) for matrix in batch])
            batch_u, batch_sigma, batch_right = self._decompose(stacked, rank)
            batch_layouts = matrix_layouts[start : start + len(batch)]
            for index, matrix_layout in enumerate(batch_layouts):
                u = batch_u[index]
                sigma = batch_sigma[index]
                right = batch_right[index].contiguous()
                if self.factor_dtype == "fp8":
                    u_stored, u_scale = _quantize_fp8_factor(u.contiguous())
                    right_stored, right_scale = _quantize_fp8_factor(right)
                else:
                    u_stored = u.to(torch.float16)
                    right_stored = right.to(torch.float16)
                sigma_stored = sigma.to(_SIGMA_DTYPE)
                _copy_tensor_bytes(destination, matrix_layout.u_offset, u_stored)
                if matrix_layout.u_scale_offset is not None:
                    _copy_tensor_bytes(
                        destination,
                        matrix_layout.u_scale_offset,
                        u_scale,
                    )
                _copy_tensor_bytes(
                    destination, matrix_layout.sigma_offset, sigma_stored
                )
                _copy_tensor_bytes(
                    destination, matrix_layout.right_offset, right_stored
                )
                if matrix_layout.right_scale_offset is not None:
                    _copy_tensor_bytes(
                        destination,
                        matrix_layout.right_scale_offset,
                        right_scale,
                    )

    def decompress(
        self,
        source: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> tuple[torch.Tensor, ...]:
        """Reconstruct matrices from FP8 SVD factors.

        Args:
            source: Contiguous one-dimensional uint8 encoded slot.
            layout: SVD layout returned by :meth:`build_layout`.

        Returns:
            Reconstructed source matrices.
        """
        self._validate_layout(layout)
        self._validate_blob_shape(source, layout)
        source_layout = layout.source
        matrix_layouts, _ = _build_svd_matrix_layouts(
            source_layout,
            layout.parameter,
            self.factor_dtype,
        )
        matrices: list[torch.Tensor] = []
        for matrix_layout in matrix_layouts:
            rank = layout.parameter
            factor_dtype = _SVD_FACTOR_DTYPES[self.factor_dtype]
            factor_itemsize = factor_dtype.itemsize
            u_count = source_layout.rows * rank
            right_count = source_layout.columns * rank
            u_stored = (
                source[
                    matrix_layout.u_offset : matrix_layout.u_offset
                    + u_count * factor_itemsize
                ]
                .view(factor_dtype)
                .view(source_layout.rows, rank)
            )
            sigma = source[
                matrix_layout.sigma_offset : matrix_layout.sigma_offset
                + rank * _SIGMA_DTYPE.itemsize
            ].view(_SIGMA_DTYPE)
            right_stored = (
                source[
                    matrix_layout.right_offset : matrix_layout.right_offset
                    + right_count * factor_itemsize
                ]
                .view(factor_dtype)
                .view(source_layout.columns, rank)
            )
            u = u_stored.to(torch.float32)
            right = right_stored.to(torch.float32)
            if matrix_layout.u_scale_offset is not None:
                u_scale = source[
                    matrix_layout.u_scale_offset : matrix_layout.u_scale_offset
                    + _SCALE_DTYPE.itemsize
                ].view(_SCALE_DTYPE)[0]
                u *= u_scale.to(torch.float32)
            if matrix_layout.right_scale_offset is not None:
                right_scale = source[
                    matrix_layout.right_scale_offset : matrix_layout.right_scale_offset
                    + _SCALE_DTYPE.itemsize
                ].view(_SCALE_DTYPE)[0]
                right *= right_scale.to(torch.float32)
            matrix = (u * sigma.to(torch.float32)) @ right.transpose(0, 1)
            matrices.append(matrix.to(source_layout.dtype))
        return tuple(matrices)

    def _validate_layout(self, layout: CompressedKVLayout) -> None:
        expected_rank = min(
            self.rank,
            layout.source.rows,
            layout.source.columns,
        )
        if layout.codec_id != self.codec_id or layout.parameter != expected_rank:
            raise ValueError("SVD compressor and encoded layout are incompatible")


class KVSplitKVCompressor(KVCompressor):
    """Compress the K and V halves of a KV object with independent codecs.

    vLLM's FlashAttention cache packs K and V into the content dimension, so one
    gathered matrix has columns laid out per head as
    ``[K_h0 | V_h0 | K_h1 | V_h1 | ...]``, each span ``head_dim`` wide.  This
    compressor separates those spans and hands each half to its own compressor,
    which lets K and V carry different compression ratios -- useful because they
    do not behave alike: V is far closer to full rank than K, while K carries the
    larger magnitude and therefore dominates a shared error budget.

    Args:
        k_compressor: Compressor applied to the K columns.
        v_compressor: Compressor applied to the V columns.
        head_dim: Width in elements of one head's K (or V) span.
    """

    def __init__(
        self,
        k_compressor: KVCompressor,
        v_compressor: KVCompressor,
        head_dim: int,
    ) -> None:
        if head_dim < 1:
            raise ValueError("kv-split head_dim must be positive")
        self.k_compressor = k_compressor
        self.v_compressor = v_compressor
        self.head_dim = head_dim

    @property
    @override
    def codec_id(self) -> str:
        """Return an identifier naming both halves' codecs."""
        return (
            f"kvsplit[{self.k_compressor.codec_id}|"
            f"{self.v_compressor.codec_id}]-v1"
        )

    def _half_source(
        self,
        source: KVCompressionSourceLayout,
    ) -> KVCompressionSourceLayout:
        if source.columns % (2 * self.head_dim):
            raise ValueError(
                f"kv-split needs columns divisible by 2*head_dim; got "
                f"{source.columns} columns and head_dim {self.head_dim}"
            )
        return KVCompressionSourceLayout(
            dtype=source.dtype,
            matrix_count=source.matrix_count,
            rows=source.rows,
            columns=source.columns // 2,
        )

    def _plan(
        self,
        layout: CompressedKVLayout,
        alignment: int = _ALIGNMENT,
    ) -> tuple[CompressedKVLayout, CompressedKVLayout, int, int]:
        """Recompute both halves' layouts and their byte offsets."""
        half = self._half_source(layout.source)
        k_layout = self.k_compressor.build_layout(half, alignment)
        v_layout = self.v_compressor.build_layout(half, alignment)
        k_offset = _align_up(_HEADER_SIZE, alignment)
        v_offset = _align_up(k_offset + k_layout.storage_nbytes, alignment)
        return k_layout, v_layout, k_offset, v_offset

    @override
    def build_layout(
        self,
        source: KVCompressionSourceLayout,
        alignment: int = _ALIGNMENT,
    ) -> CompressedKVLayout:
        """Build a layout holding both halves' encoded objects back to back.

        Args:
            source: Source matrix layout for the combined K and V columns.
            alignment: Encoded CPU slot alignment.

        Returns:
            Layout whose payload is the K object followed by the V object.
        """
        half = self._half_source(source)
        k_layout = self.k_compressor.build_layout(half, alignment)
        v_layout = self.v_compressor.build_layout(half, alignment)
        k_offset = _align_up(_HEADER_SIZE, alignment)
        v_offset = _align_up(k_offset + k_layout.storage_nbytes, alignment)
        return _build_layout(
            codec_id=self.codec_id,
            codec_code=_KVSPLIT_CODEC_CODE,
            parameter=self.head_dim,
            source=source,
            payload_nbytes=v_offset + v_layout.storage_nbytes,
            alignment=alignment,
        )

    def _split(
        self,
        matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, columns = matrix.shape
        heads = columns // (2 * self.head_dim)
        view = matrix.view(rows, heads, 2, self.head_dim)
        return (
            view[:, :, 0, :].reshape(rows, heads * self.head_dim).contiguous(),
            view[:, :, 1, :].reshape(rows, heads * self.head_dim).contiguous(),
        )

    @override
    def compress_into(
        self,
        source_matrices: tuple[torch.Tensor, ...],
        destination: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> None:
        """Compress the K and V halves into their own regions.

        Args:
            source_matrices: Floating matrices matching ``layout.source``.
            destination: Contiguous one-dimensional uint8 encoded slot.
            layout: Layout returned by :meth:`build_layout`.
        """
        self._validate_source_matrices(source_matrices, destination, layout)
        k_layout, v_layout, k_offset, v_offset = self._plan(layout)
        destination[:k_offset].zero_()
        destination[:_HEADER_SIZE].copy_(
            torch.tensor(
                tuple(layout.header), dtype=torch.uint8, device=destination.device
            )
        )
        halves = tuple(self._split(matrix) for matrix in source_matrices)
        self.k_compressor.compress_into(
            tuple(h[0] for h in halves),
            destination[k_offset : k_offset + k_layout.storage_nbytes],
            k_layout,
        )
        self.v_compressor.compress_into(
            tuple(h[1] for h in halves),
            destination[v_offset : v_offset + v_layout.storage_nbytes],
            v_layout,
        )

    @override
    def decompress(
        self,
        source: torch.Tensor,
        layout: CompressedKVLayout,
    ) -> tuple[torch.Tensor, ...]:
        """Reconstruct matrices by interleaving both halves' columns.

        Args:
            source: Contiguous one-dimensional uint8 encoded slot.
            layout: Layout returned by :meth:`build_layout`.

        Returns:
            Matrices matching ``layout.source``.
        """
        k_layout, v_layout, k_offset, v_offset = self._plan(layout)
        k_mats = self.k_compressor.decompress(
            source[k_offset : k_offset + k_layout.storage_nbytes], k_layout
        )
        v_mats = self.v_compressor.decompress(
            source[v_offset : v_offset + v_layout.storage_nbytes], v_layout
        )
        rows, columns = layout.source.rows, layout.source.columns
        heads = columns // (2 * self.head_dim)
        out: list[torch.Tensor] = []
        for k_mat, v_mat in zip(k_mats, v_mats, strict=True):
            merged = torch.empty(
                rows, heads, 2, self.head_dim,
                dtype=layout.source.dtype, device=k_mat.device,
            )
            merged[:, :, 0, :] = k_mat.view(rows, heads, self.head_dim)
            merged[:, :, 1, :] = v_mat.view(rows, heads, self.head_dim)
            out.append(merged.view(rows, columns))
        return tuple(out)
