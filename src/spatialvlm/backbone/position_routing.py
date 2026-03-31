"""Position routing utilities for Stage 4.

Text tokens keep standard M-RoPE position indexing.
Spatial tokens carry GridCellRoPE3D rotations and should not use text sequence positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class RoutedPositionBatch:
    """Position-routing outputs for mixed text+spatial sequences."""

    combined_tokens: torch.Tensor  # [B, T+N, D]
    is_spatial_mask: torch.Tensor  # [B, T+N] bool
    text_mrope_position_ids: torch.Tensor  # [B, 3, T]
    spatial_rope3d: torch.Tensor  # [B, N, 64]


class PositionRouter:
    """Routes text and spatial positional representations before LLM attention."""

    def __init__(self, mrope_section: Sequence[int], expected_spatial_rotary_dim: int = 64) -> None:
        self.mrope_section = [int(x) for x in mrope_section]
        self.expected_spatial_rotary_dim = expected_spatial_rotary_dim
        self.rotary_pairs = int(sum(self.mrope_section))
        if self.rotary_pairs != expected_spatial_rotary_dim:
            raise ValueError(
                "GridCellRoPE3D rotary dim must match M-RoPE rotary pairs. "
                f"Got sum(mrope_section)={self.rotary_pairs}, "
                f"expected_spatial_rotary_dim={expected_spatial_rotary_dim}."
            )

    def build_text_mrope_position_ids(
        self,
        batch_size: int,
        text_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build text position IDs for M-RoPE: [B, 3, T]."""
        if device is None:
            device = torch.device("cpu")
        seq = torch.arange(text_len, device=device, dtype=torch.long)  # [T]
        temporal = seq.unsqueeze(0).expand(batch_size, -1)  # [B, T]
        height = torch.zeros(batch_size, text_len, device=device, dtype=torch.long)  # [B, T]
        width = torch.zeros(batch_size, text_len, device=device, dtype=torch.long)  # [B, T]
        return torch.stack([temporal, height, width], dim=1)  # [B, 3, T]

    def route(
        self,
        text_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        spatial_rope3d: torch.Tensor,
        text_mrope_position_ids: torch.Tensor | None = None,
    ) -> RoutedPositionBatch:
        """Create a routed batch for mixed text+spatial processing.

        Parameters
        ----------
        text_tokens : Tensor[B, T, D]
        spatial_tokens : Tensor[B, N, D]
        spatial_rope3d : Tensor[B, N, 64]
        text_mrope_position_ids : Tensor[B, 3, T] | Tensor[B, T] | None
            If [B, T], it is promoted to [B, 3, T] with zero height/width channels.
        """
        if text_tokens.ndim != 3 or spatial_tokens.ndim != 3:
            raise ValueError(
                "text_tokens and spatial_tokens must both be rank-3 tensors [B, L, D]."
            )

        b_text, t_len, d_text = text_tokens.shape
        b_spatial, n_spatial, d_spatial = spatial_tokens.shape
        if b_text != b_spatial:
            raise ValueError(f"Batch mismatch: text B={b_text}, spatial B={b_spatial}.")
        if d_text != d_spatial:
            raise ValueError(f"Hidden-size mismatch: text D={d_text}, spatial D={d_spatial}.")

        if spatial_rope3d.shape != (b_spatial, n_spatial, self.expected_spatial_rotary_dim):
            raise ValueError(
                "spatial_rope3d must be "
                f"[B, N, {self.expected_spatial_rotary_dim}], got {tuple(spatial_rope3d.shape)}."
            )

        if text_mrope_position_ids is None:
            text_mrope_position_ids = self.build_text_mrope_position_ids(
                batch_size=b_text,
                text_len=t_len,
                device=text_tokens.device,
            )
        elif text_mrope_position_ids.ndim == 2:
            if text_mrope_position_ids.shape != (b_text, t_len):
                raise ValueError(
                    f"text_mrope_position_ids [B, T] expected {(b_text, t_len)}, "
                    f"got {tuple(text_mrope_position_ids.shape)}."
                )
            temporal = text_mrope_position_ids
            zeros = torch.zeros_like(temporal)
            text_mrope_position_ids = torch.stack([temporal, zeros, zeros], dim=1)  # [B, 3, T]
        elif text_mrope_position_ids.ndim == 3:
            if text_mrope_position_ids.shape != (b_text, 3, t_len):
                raise ValueError(
                    f"text_mrope_position_ids [B,3,T] expected {(b_text, 3, t_len)}, "
                    f"got {tuple(text_mrope_position_ids.shape)}."
                )
        else:
            raise ValueError("text_mrope_position_ids must be None, [B,T], or [B,3,T].")

        combined_tokens = torch.cat([text_tokens, spatial_tokens], dim=1)  # [B, T+N, D]
        is_spatial_mask = torch.zeros(
            b_text,
            t_len + n_spatial,
            device=text_tokens.device,
            dtype=torch.bool,
        )
        is_spatial_mask[:, t_len:] = True

        return RoutedPositionBatch(
            combined_tokens=combined_tokens,
            is_spatial_mask=is_spatial_mask,
            text_mrope_position_ids=text_mrope_position_ids,
            spatial_rope3d=spatial_rope3d,
        )
