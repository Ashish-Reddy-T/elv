"""Gated cross-attention block for LLM injection (Stage 4 bridge).

Pattern follows Flamingo-style gated residual injection:
  x <- x + tanh(alpha_attn) * CrossAttention(x, vision)
  x <- x + tanh(alpha_ff)   * FeedForward(x)

Both gates are initialized to 0, so the block is an exact passthrough at init.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo-style gated cross-attention block with zero-init gates."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))

        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim, bias=False),
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        text_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject visual context into language tokens.

        Parameters
        ----------
        text_tokens : Tensor[B, T, D]
        vision_tokens : Tensor[B, N, D]
        vision_key_padding_mask : Tensor[B, N] | None
            True means token should be ignored (PyTorch MHA semantics).
        """
        # text_tokens: [B, T, D], vision_tokens: [B, N, D]
        q = self.q_norm(text_tokens)  # [B, T, D]
        kv = self.kv_norm(vision_tokens)  # [B, N, D]
        attn_out, _ = self.cross_attn(  # [B, T, D]
            q,
            kv,
            kv,
            key_padding_mask=vision_key_padding_mask,
            need_weights=False,
        )
        x = text_tokens + torch.tanh(self.attn_gate) * attn_out  # [B, T, D]

        ff_out = self.ff(self.ff_norm(x))  # [B, T, D]
        x = x + torch.tanh(self.ff_gate) * ff_out  # [B, T, D]

        return x  # [B, T, D]
