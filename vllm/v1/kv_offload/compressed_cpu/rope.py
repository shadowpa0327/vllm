# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Un-rotate stored keys so a codec can compress pre-RoPE keys.

vLLM stores keys *after* RoPE, but RoPE inflates their rank sharply: on
Llama-3.1-8B the rank needed for 95% of the spectral energy rises roughly 4-8x
across the rotation, and a rank-384 reconstruction is about twice as wrong.
Low-rank methods for KV therefore target the pre-RoPE keys.

RoPE is an orthogonal linear map fixed entirely by token position, so a codec can
recover the pre-RoPE keys itself:

    store:  K_pre = RoPE^-1(K_post, pos) -> compress
    load:   decompress -> K_post = RoPE(K_pre_hat, pos)

Orthogonality means the relative error passes through the rotation unchanged, so
the pre-RoPE gain is kept in full on the keys the model actually attends to.  A
prefix-cache chunk always covers a fixed token range, so `pos` is known.
"""

import math

import torch


def _llama3_inv_freq(
    head_dim: int,
    theta: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position: int,
) -> torch.Tensor:
    """Inverse frequencies under Llama-3 style frequency scaling."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )
    if factor == 1.0:
        return inv_freq
    low_wavelen = original_max_position / low_freq_factor
    high_wavelen = original_max_position / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (original_max_position / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RoPEKeyTransform:
    """Apply or undo RoPE on the key columns of a gathered matrix.

    Args:
        head_dim: Channels per attention head.
        max_position: Largest token position the tables must cover.
        theta: RoPE base frequency.
        factor: Llama-3 frequency-scaling factor; 1.0 disables scaling.
        low_freq_factor: Llama-3 scaling low-frequency cutoff.
        high_freq_factor: Llama-3 scaling high-frequency cutoff.
        original_max_position: Context length the scaling is defined against.
        device: Device the tables live on.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        theta: float = 500000.0,
        factor: float = 1.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
        device: torch.device | str = "cuda",
    ) -> None:
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        self.head_dim = head_dim
        self.max_position = max_position
        inv_freq = _llama3_inv_freq(
            head_dim, theta, factor, low_freq_factor,
            high_freq_factor, original_max_position,
        ).to(device)
        positions = torch.arange(max_position, dtype=torch.float64, device=device)
        angles = torch.outer(positions, inv_freq)
        self._cos = torch.cat((angles.cos(), angles.cos()), dim=-1).float()
        self._sin = torch.cat((angles.sin(), angles.sin()), dim=-1).float()

    def key_column_mask(self, columns: int) -> torch.Tensor:
        """Columns holding keys, given the per-head ``[K | V]`` interleave."""
        head_index = torch.arange(columns, device=self._cos.device) // self.head_dim
        return head_index % 2 == 0

    def apply_(
        self,
        matrix: torch.Tensor,
        start_position: int,
        inverse: bool,
    ) -> torch.Tensor:
        """Rotate the key columns of ``matrix`` in place.

        Args:
            matrix: ``(rows, columns)`` gathered matrix; rows are tokens.
            start_position: Token position of row 0.
            inverse: Undo the rotation instead of applying it.

        Returns:
            The same tensor, with key columns rotated.
        """
        rows, columns = matrix.shape
        end = start_position + rows
        if end > self.max_position:
            raise ValueError(
                f"RoPE tables cover {self.max_position} positions; "
                f"chunk needs {end}"
            )
        mask = self.key_column_mask(columns)
        keys = matrix[:, mask]
        heads = keys.shape[1] // self.head_dim
        view = keys.view(rows, heads, self.head_dim)
        cos = self._cos[start_position:end].unsqueeze(1).to(view.dtype)
        sin = self._sin[start_position:end].unsqueeze(1).to(view.dtype)
        if inverse:
            sin = -sin
        half = self.head_dim // 2
        x1, x2 = view[..., :half], view[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        matrix[:, mask] = (view * cos + rotated * sin).reshape(rows, -1)
        return matrix
