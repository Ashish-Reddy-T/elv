"""Gated cross-attention block for LLM injection (Stage 4 bridge).

Pattern follows Flamingo-style gated residual injection:
  x <- x + tanh(alpha_attn) * CrossAttention(x, vision)
  x <- x + tanh(alpha_ff)   * FeedForward(x)

Both gates are initialized to 0, so the block is an exact passthrough at init.

Cross-attention uses Grouped Query Attention (GQA) to match Qwen3-VL's
architecture: 32 query heads, 8 KV heads (4:1 ratio).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo-style gated cross-attention block with zero-init gates and GQA."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.num_groups = num_heads // num_kv_heads  # heads per KV group
        self.dropout = dropout

        # Layer norms
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        # Q/K/V/O projections (K and V use num_kv_heads, not num_heads)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)        # → num_heads * head_dim
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)       # → num_kv_heads * head_dim
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)       # → num_kv_heads * head_dim
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

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
            True means token should be **ignored** (PyTorch MHA semantics).
        """
        # text_tokens: [B, T, D], vision_tokens: [B, N, D]
        bsz, t_len, _ = text_tokens.shape
        _, n_vis, _ = vision_tokens.shape

        q = self.q_proj(self.q_norm(text_tokens))    # [B, T, num_heads * head_dim]
        kv_in = self.kv_norm(vision_tokens)
        k = self.k_proj(kv_in)                        # [B, N, num_kv_heads * head_dim]
        v = self.v_proj(kv_in)                        # [B, N, num_kv_heads * head_dim]

        # Reshape for multi-head attention
        q = q.view(bsz, t_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, T, head_dim]

        k = k.view(bsz, n_vis, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n_vis, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # [B, num_kv_heads, N, head_dim]

        # GQA: expand KV heads to match query heads by repeating each KV head
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            k = k.reshape(bsz, self.num_heads, n_vis, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_groups, -1, -1)
            v = v.reshape(bsz, self.num_heads, n_vis, self.head_dim)
        # k, v now: [B, num_heads, N, head_dim]

        # Build attention mask from padding mask
        attn_mask: torch.Tensor | None = None
        if vision_key_padding_mask is not None:
            # True = ignore → -inf, False = attend → 0
            neg_inf = torch.finfo(q.dtype).min
            attn_mask = torch.where(
                vision_key_padding_mask[:, None, None, :],  # [B, 1, 1, N]
                torch.full((1,), neg_inf, device=q.device, dtype=q.dtype),
                torch.zeros(1, device=q.device, dtype=q.dtype),
            )

        attn_out = functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, num_heads, T, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t_len, self.hidden_dim)
        attn_out = self.o_proj(attn_out)  # [B, T, D]

        x = text_tokens + torch.tanh(self.attn_gate) * attn_out  # [B, T, D]

        ff_out = self.ff(self.ff_norm(x))  # [B, T, D]
        x = x + torch.tanh(self.ff_gate) * ff_out  # [B, T, D]

        return x  # [B, T, D]
