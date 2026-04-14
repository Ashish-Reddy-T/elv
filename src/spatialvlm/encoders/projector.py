"""Shared MLP projector for bridging vision encoder outputs to LLM hidden size.

A simple two-layer MLP: Linear → GELU → Linear.

Instantiated three times with different input dimensions:
  SigLIP2 projector:  MLPProjector(3456, 4096)  ≈ 31M params
  DINOv2 projector:   MLPProjector(3072, 4096)  ≈ 29M params
  GATr projector:     MLPProjector(48,   4096)  ≈ 17M params  (Phase 3)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """Two-layer MLP projector: Linear(in_dim, hidden_dim) → GELU → Linear(hidden_dim, out_dim).

    Parameters
    ----------
    in_dim : int
        Input feature dimension (encoder output width).
    out_dim : int
        Output dimension (LLM hidden size).
    hidden_dim : int | None
        Hidden layer dimension. Defaults to out_dim if not specified.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project token features to LLM hidden size.

        Parameters
        ----------
        x : Tensor[B, N, in_dim]
            Encoder output tokens.

        Returns
        -------
        Tensor[B, N, out_dim]
        """
        # Frozen vision encoders often stay fp32 while projectors are cast to bf16/fp16
        # (see scripts/train.py). Align activations to parameter dtype for matmul.
        first = self.net[0]
        assert isinstance(first, nn.Linear)
        w_dtype = first.weight.dtype
        if x.dtype != w_dtype:
            x = x.to(dtype=w_dtype)
        return self.net(x)  # [B, N, out_dim]
