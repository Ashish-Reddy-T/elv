"""Spatial Vision Aggregator (SVA) for Stage 3 fusion.

SVA uses 1369 query tokens (DINOv2-based, 37x37 grid) that cross-attend to
concatenated visual key/value tokens:
  - SigLIP tokens (semantic):   [B, 576, D]
  - DINOv2 tokens (structural): [B, 1369, D]
  - GATr tokens (geometric):    [B, 1369, D]

Total KV tokens = 3314.

DINOv2 as query base:
  The structural encoder owns the spatial layout. Its 37x37 tokens form the
  skeleton onto which SigLIP semantics and GATr geometry are fused via KV
  cross-attention. This eliminates the 37x37 -> 24x24 compression bottleneck
  of the previous 576-query design, giving 1:1 position-to-query mapping.

Typed attention bias:
  Optional learned 3x3 matrix over token-source types:
    0: SigLIP, 1: DINOv2, 2: GATr
  Bias is added to attention logits as an additive mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SVACrossAttentionLayer(nn.Module):
    """Single cross-attention layer for SVA."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        use_typed_attention_bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.typed_attention_bias: nn.Parameter | None
        if use_typed_attention_bias:
            self.typed_attention_bias = nn.Parameter(torch.zeros(3, 3))
        else:
            self.typed_attention_bias = None

    def build_typed_attention_mask(
        self,
        query_type_ids: torch.Tensor,
        kv_type_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build additive attention bias mask from the learned 3x3 type matrix."""
        if self.typed_attention_bias is None:
            return None

        # query_type_ids: [Nq], kv_type_ids: [Nk]
        typed_bias = self.typed_attention_bias[query_type_ids][:, kv_type_ids]  # [Nq, Nk]
        return typed_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, Nq, Nk]

    def _attention_with_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_type_ids: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unfused attention that exposes per-source-type attention mass.

        Runs only on the diagnostic path (`return_attention_stats=True`).
        Numerically equivalent to the fused `scaled_dot_product_attention`
        call used on the default path.
        """
        # q: [B, H, Nq, Dh], k/v: [B, H, Nk, Dh]
        scale = self.head_dim**-0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Nq, Nk]
        if attn_mask is not None:
            logits = logits + attn_mask
        weights = functional.softmax(logits, dim=-1)  # [B, H, Nq, Nk]
        attn_out = torch.matmul(weights, v)  # [B, H, Nq, Dh]

        stats: dict[str, torch.Tensor] = {}
        for type_id, key in (
            (0, "head_mean_to_siglip"),
            (1, "head_mean_to_dino"),
            (2, "head_mean_to_gatr"),
        ):
            type_mask = kv_type_ids == type_id  # [Nk]
            if type_mask.any():
                mass_per_query = weights[..., type_mask].sum(dim=-1)  # [B, H, Nq]
                head_mean = mass_per_query.mean(dim=(0, 2))  # [H]
            else:
                head_mean = torch.zeros(self.num_heads, device=weights.device, dtype=weights.dtype)
            stats[key] = head_mean.detach()

        if self.typed_attention_bias is not None:
            stats["typed_bias"] = self.typed_attention_bias.detach().clone()

        return attn_out, stats

    def forward(
        self,
        queries: torch.Tensor,
        kv_tokens: torch.Tensor,
        query_type_ids: torch.Tensor,
        kv_type_ids: torch.Tensor,
        kv_padding_mask: torch.Tensor | None = None,
        return_attention_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cross-attend queries over visual key/value tokens.

        Parameters
        ----------
        queries : Tensor[B, Nq, D]
        kv_tokens : Tensor[B, Nk, D]
        query_type_ids : Tensor[Nq]
            Integer token types in {0, 1, 2}.
        kv_type_ids : Tensor[Nk]
            Integer token types in {0, 1, 2}.
        kv_padding_mask : Tensor[B, Nk] | None
            True = token valid, False = masked out.
        return_attention_stats : bool
            If True, also return a dict of per-head attention mass on each
            source type (siglip/dino/gatr) + the typed bias snapshot. Uses
            an unfused attention path and is intended for diagnostics only.
        """
        # queries: [B, Nq, D], kv_tokens: [B, Nk, D]
        residual = queries
        bsz, n_q, _ = queries.shape
        _, n_kv, _ = kv_tokens.shape

        q = self.q_proj(self.q_norm(queries))  # [B, Nq, D]
        k = self.k_proj(self.kv_norm(kv_tokens))  # [B, Nk, D]
        v = self.v_proj(self.kv_norm(kv_tokens))  # [B, Nk, D]

        q = q.view(bsz, n_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nq, Dh]
        k = k.view(bsz, n_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nk, Dh]
        v = v.view(bsz, n_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Nk, Dh]

        typed_mask = self.build_typed_attention_mask(query_type_ids, kv_type_ids)
        if typed_mask is not None:
            typed_mask = typed_mask.to(device=queries.device, dtype=queries.dtype)

        padding_mask: torch.Tensor | None = None
        if kv_padding_mask is not None:
            if kv_padding_mask.shape != (bsz, n_kv):
                raise ValueError(
                    f"kv_padding_mask must be [B, Nk] = {(bsz, n_kv)}, got {kv_padding_mask.shape}"
                )
            # True(valid)/False(mask) -> additive mask [B,1,1,Nk] with 0 or -inf.
            neg_inf = torch.finfo(queries.dtype).min
            padding_mask = torch.where(
                kv_padding_mask[:, None, None, :],
                torch.zeros(1, device=queries.device, dtype=queries.dtype),
                torch.full((1,), neg_inf, device=queries.device, dtype=queries.dtype),
            )

        attn_mask: torch.Tensor | None = None
        if typed_mask is not None and padding_mask is not None:
            attn_mask = typed_mask + padding_mask
        elif typed_mask is not None:
            attn_mask = typed_mask
        elif padding_mask is not None:
            attn_mask = padding_mask

        stats: dict[str, torch.Tensor] | None = None
        if return_attention_stats:
            attn_out, stats = self._attention_with_stats(q, k, v, kv_type_ids, attn_mask)
        else:
            if attn_mask is not None:
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            attn_out = functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0
            )  # [B, H, Nq, Dh]

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, n_q, self.hidden_dim)  # [B,Nq,D]
        attn_out = self.o_proj(attn_out)  # [B, Nq, D]

        out = self.out_norm(residual + attn_out)  # [B, Nq, D]
        if return_attention_stats:
            assert stats is not None
            return out, stats
        return out


class SpatialVisionAggregator(nn.Module):
    """SVA module: 1369 DINOv2-based queries over 3314 visual KV tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 1369,
        num_layers: int = 2,
        num_heads: int = 32,
        use_typed_attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        scale = hidden_dim**-0.5
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim) * scale)
        self.layers = nn.ModuleList(
            [
                SVACrossAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    use_typed_attention_bias=use_typed_attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        siglip_tokens: torch.Tensor,
        dinov2_tokens: torch.Tensor,
        gatr_tokens: torch.Tensor,
        queries: torch.Tensor | None = None,
        query_type_ids: torch.Tensor | None = None,
        kv_padding_mask: torch.Tensor | None = None,
        return_attention_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Aggregate visual streams into 1369 fused query tokens.

        Parameters
        ----------
        siglip_tokens : Tensor[B, 576, D]
        dinov2_tokens : Tensor[B, 1369, D]
        gatr_tokens : Tensor[B, 1369, D]
        queries : Tensor[B, 1369, D] | None
            If None, uses DINOv2 tokens as query base (structural anchor).
        query_type_ids : Tensor[1369] | None
            Type IDs in {0,1,2}. If None, all queries are type 1 (DINOv2).
        kv_padding_mask : Tensor[B, 3314] | None
            True(valid)/False(masked).
        return_attention_stats : bool
            If True, also return a dict keyed by `layer_{i}` with per-layer
            per-head attention mass on each source type. Diagnostic use only;
            activates an unfused attention path.
        """
        # siglip_tokens: [B, Ns, D], dinov2_tokens: [B, Nd, D], gatr_tokens: [B, Ng, D]
        bsz, n_sig, dim = siglip_tokens.shape
        _, n_dino, dim_dino = dinov2_tokens.shape
        _, n_gatr, dim_gatr = gatr_tokens.shape
        if dim != self.hidden_dim or dim_dino != self.hidden_dim or dim_gatr != self.hidden_dim:
            raise ValueError(
                "All token streams must match hidden_dim. "
                f"Got siglip={dim}, dino={dim_dino}, gatr={dim_gatr}, expected={self.hidden_dim}."
            )
        if n_dino != self.num_queries:
            raise ValueError(
                f"DINOv2 token count must match num_queries={self.num_queries}, got {n_dino}."
            )

        kv_tokens = torch.cat([siglip_tokens, dinov2_tokens, gatr_tokens], dim=1)  # [B, 3314, D]
        kv_type_ids = torch.cat(
            [
                torch.zeros(n_sig, dtype=torch.long, device=kv_tokens.device),
                torch.ones(n_dino, dtype=torch.long, device=kv_tokens.device),
                torch.full((n_gatr,), 2, dtype=torch.long, device=kv_tokens.device),
            ],
            dim=0,
        )  # [3314]

        if query_type_ids is None:
            # DINOv2-based queries get type 1 by default
            query_type_ids = torch.ones(self.num_queries, dtype=torch.long, device=kv_tokens.device)
        if query_type_ids.shape != (self.num_queries,):
            raise ValueError(
                "query_type_ids must be "
                f"[num_queries]={self.num_queries}, got {query_type_ids.shape}."
            )

        if queries is None:
            queries = dinov2_tokens  # [B, 1369, D] — structural anchor
        if queries.shape != (bsz, self.num_queries, self.hidden_dim):
            raise ValueError(
                f"queries must be [B, {self.num_queries}, {self.hidden_dim}], got {queries.shape}."
            )
        queries = queries + self.query_embed.unsqueeze(0).to(
            device=queries.device, dtype=queries.dtype
        )

        all_stats: dict[str, dict[str, torch.Tensor]] | None = (
            {} if return_attention_stats else None
        )
        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(
                queries=queries,
                kv_tokens=kv_tokens,
                query_type_ids=query_type_ids,
                kv_type_ids=kv_type_ids,
                kv_padding_mask=kv_padding_mask,
                return_attention_stats=return_attention_stats,
            )
            if return_attention_stats:
                assert isinstance(layer_out, tuple)
                queries, layer_stats = layer_out
                assert all_stats is not None
                all_stats[f"layer_{layer_idx}"] = layer_stats
            else:
                assert isinstance(layer_out, torch.Tensor)
                queries = layer_out  # [B, 1369, D]

        if return_attention_stats:
            assert all_stats is not None
            return queries, all_stats
        return queries  # [B, 1369, D]
