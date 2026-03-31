"""GATr wrapper for Stage 2 geometric branch.

This module wraps the reference `geometric-algebra-transformer` implementation and exposes a
SpatialVLM-oriented interface:

  1. Input 3D points `[B, N, 3]` (from depth backprojection)
  2. Embed into PGA multivectors `[B, N, 1, 16]`
  3. Run GATr blocks with improved PGA bilinears (join + geometric product)
  4. Extract invariants `[B, N, 48] = scalar channels (32) + MV norms (16)`
  5. Project to LLM hidden size `[B, N, 4096]`

Compatibility note:
  In some modern Python/Numpy stacks, GATr's cached `opt_einsum` path planner can fail at runtime.
  We disable cached einsum by default (`enable_cached_einsum(False)`), which keeps the same model
  computation while using regular `torch.einsum`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

from spatialvlm.encoders.projector import MLPProjector


class GATrWrapper(nn.Module):
    """SpatialVLM wrapper around GATr for geometric token features."""

    def __init__(
        self,
        num_blocks: int = 8,
        gatr_mv_channels: int = 16,
        gatr_s_channels: int = 32,
        projector_out_dim: int = 4096,
        join_reference: Literal["data", "canonical"] = "data",
        checkpoint_blocks: bool = True,
        normalize_inputs: bool = True,
        disable_cached_einsum: bool = True,
        eps: float = 1e-8,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        try:
            from gatr import GATr, MLPConfig, SelfAttentionConfig
            from gatr.interface import embed_point
            from gatr.utils.einsum import enable_cached_einsum
        except ImportError as exc:  # pragma: no cover - exercised only on missing dependency
            raise ImportError(
                "GATr dependency not found. Install with "
                "`pip install -e REPOS/geometric-algebra-transformer --no-deps`."
            ) from exc

        if disable_cached_einsum:
            enable_cached_einsum(False)

        self.gatr_mv_channels = gatr_mv_channels
        self.gatr_s_channels = gatr_s_channels
        self._join_reference = join_reference
        self._normalize_inputs = normalize_inputs
        self._eps = eps
        self._embed_point: Callable[[torch.Tensor], torch.Tensor] = embed_point

        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=gatr_mv_channels,
            hidden_mv_channels=gatr_mv_channels,
            in_s_channels=1,
            out_s_channels=gatr_s_channels,
            hidden_s_channels=gatr_s_channels,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            checkpoint=["block"] if checkpoint_blocks else None,
        )
        self.projector = MLPProjector(
            in_dim=self.invariant_dim,
            out_dim=projector_out_dim,
        )
        self.to(device)

    @property
    def invariant_dim(self) -> int:
        """Invariant feature width: scalar channels + MV-channel norms."""
        return self.gatr_s_channels + self.gatr_mv_channels

    def uses_improved_pga(self) -> bool:
        """Returns True if each block MLP starts with `GeometricBilinear` (improved PGA path)."""
        from gatr.layers.mlp.geometric_bilinears import GeometricBilinear

        for block in self.gatr.blocks:
            if len(block.mlp.layers) == 0:
                return False
            if not isinstance(block.mlp.layers[0], GeometricBilinear):
                return False
        return True

    def forward(
        self,
        points_3d: torch.Tensor,
        return_invariants: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run GATr and return projected features.

        Parameters
        ----------
        points_3d : Tensor[B, N, 3]
            Per-token 3D points in camera/world coordinates (metres).
        return_invariants : bool
            If True, also return the pre-projector invariants `[B, N, 48]`.

        Returns
        -------
        projected : Tensor[B, N, projector_out_dim]
            LLM-width geometric token features.
        invariants : Tensor[B, N, 48], optional
            Scalar channels + multivector channel norms.
        """
        if points_3d.ndim != 3 or points_3d.shape[-1] != 3:
            raise ValueError(f"Expected points_3d shape [B, N, 3], got {tuple(points_3d.shape)}")

        coords = points_3d
        if self._normalize_inputs:
            radius = torch.linalg.vector_norm(coords, dim=-1)  # [B, N]
            max_radius = radius.amax(dim=-1, keepdim=True).clamp_min(self._eps)  # [B, 1]
            coords = coords / max_radius.unsqueeze(-1)  # [B, N, 3]

        mv_in = self._embed_point(coords).unsqueeze(-2)  # [B, N, 1, 16]
        scalar_in = torch.linalg.vector_norm(coords, dim=-1, keepdim=True)  # [B, N, 1]

        mv_out, scalar_out = self.gatr(  # [B, N, C_mv, 16], [B, N, C_s]
            mv_in,
            scalars=scalar_in,
            join_reference=self._join_reference,
        )
        if scalar_out is None:
            raise RuntimeError("GATr returned no scalar channels; expected out_s_channels > 0.")

        mv_norms = torch.linalg.vector_norm(mv_out, dim=-1)  # [B, N, C_mv]
        invariants = torch.cat([scalar_out, mv_norms], dim=-1)  # [B, N, C_s + C_mv]
        projected = self.projector(invariants)  # [B, N, projector_out_dim]

        if return_invariants:
            return projected, invariants
        return projected
