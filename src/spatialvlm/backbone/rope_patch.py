"""RoPE monkey-patch: inject IcosahedralRoPE3D into Qwen3-VL's positional encoding.

This module patches two methods in Qwen3-VL at runtime:
1. **The Catcher** — patches Qwen3VLForConditionalGeneration.forward() to intercept
   spatial_coords_3d and mm_token_type_ids from kwargs, stashing them on the
   rotary embedding module for downstream use.

2. **The Math** — patches Qwen3VLTextRotaryEmbedding.forward() to:
   a. Compute standard M-RoPE cos/sin for all tokens (unchanged for text)
   b. Retrieve stashed 3D coordinates
   c. Compute IcosahedralRoPE3D (6 dirs, 8 freqs) -> [B, N_spatial, 96]
   d. Build cos/sin from the icosahedral angles, pad to 128 with identity
   e. Overwrite cos/sin at spatial token positions

Design decisions:
- Text tokens keep standard M-RoPE [24,20,20] = 64 pairs (128 dims)
- Spatial tokens get icosahedral 48 pairs (96 dims) + 16 identity pairs (32 dims)
- Identity pairs (cos=1, sin=0) make those dims position-agnostic for spatial tokens
- During autoregressive generation, spatial tokens are in the KV cache from prefill;
  only new text tokens are processed, so the patch is a no-op during decode.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from spatialvlm.geometry.gridcell_rope3d import IcosahedralRoPE3D

# Module-level singleton — created once, moved to device as needed
_icosahedral_rope3d: IcosahedralRoPE3D | None = None

# Spatial token type ID used in mm_token_type_ids to identify our tokens
# Qwen3-VL uses: 0=text, 1=image, 2=video. We use 3=spatial.
SPATIAL_TOKEN_TYPE_ID: int = 3


def _get_icosahedral_rope(device: torch.device) -> IcosahedralRoPE3D:
    """Get or create the IcosahedralRoPE3D singleton on the correct device."""
    global _icosahedral_rope3d
    if _icosahedral_rope3d is None:
        _icosahedral_rope3d = IcosahedralRoPE3D()
    return _icosahedral_rope3d.to(device)


def _build_icosahedral_cos_sin(
    positions_3d: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin from IcosahedralRoPE3D and pad to head_dim.

    Parameters
    ----------
    positions_3d : Tensor[B, N_spatial, 3]
        3D coordinates in metres.
    head_dim : int
        Full head dimension (128 for Qwen3-VL-8B).
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    cos, sin : Tensor[B, N_spatial, head_dim]
        Ready to overwrite into the full cos/sin tensors at spatial positions.
    """
    device = positions_3d.device
    rope3d = _get_icosahedral_rope(device)

    # IcosahedralRoPE3D returns interleaved [sin, cos, sin, cos, ...] angles
    angles = rope3d(positions_3d)  # [B, N_spatial, 96]
    bsz, n_spatial, rope_dims = angles.shape
    assert rope_dims == 96, f"Expected 96 icosahedral dims, got {rope_dims}"

    # Split interleaved sin/cos pairs: angles layout is [sin, cos, sin, cos, ...]
    # Each pair occupies 2 consecutive dims
    n_pairs = rope_dims // 2  # 48 pairs
    angles_reshaped = angles.view(bsz, n_spatial, n_pairs, 2)  # [B, N, 48, 2]
    sin_vals = angles_reshaped[..., 0]  # [B, N, 48] — the sin values
    cos_vals = angles_reshaped[..., 1]  # [B, N, 48] — the cos values

    # Pad to head_dim // 2 pairs with identity (cos=1, sin=0)
    total_pairs = head_dim // 2  # 64 for head_dim=128
    pad_pairs = total_pairs - n_pairs  # 16 identity pairs

    if pad_pairs > 0:
        pad_sin = torch.zeros(bsz, n_spatial, pad_pairs, device=device, dtype=dtype)
        pad_cos = torch.ones(bsz, n_spatial, pad_pairs, device=device, dtype=dtype)
        sin_vals = torch.cat([sin_vals, pad_sin], dim=-1)  # [B, N, 64]
        cos_vals = torch.cat([cos_vals, pad_cos], dim=-1)  # [B, N, 64]

    # Duplicate to match RoPE convention: [cos, cos] for full head_dim
    # Qwen3-VL does: emb = cat((freqs, freqs), dim=-1) -> cos/sin of shape [B, S, head_dim]
    cos_out = torch.cat([cos_vals, cos_vals], dim=-1)  # [B, N, 128]
    sin_out = torch.cat([sin_vals, sin_vals], dim=-1)  # [B, N, 128]

    return cos_out.to(dtype=dtype), sin_out.to(dtype=dtype)


def patch_rope_forward(
    original_forward: Any,
    rotary_emb_self: Any,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched Qwen3VLTextRotaryEmbedding.forward().

    Computes standard M-RoPE, then overwrites spatial token positions with
    icosahedral cos/sin. No-op if no spatial coordinates are stashed.

    Parameters
    ----------
    original_forward : callable
        The original Qwen3VLTextRotaryEmbedding.forward method.
    rotary_emb_self : Qwen3VLTextRotaryEmbedding
        The rotary embedding module instance (has stashed attributes).
    x : Tensor
        Hidden states (for dtype/device reference).
    position_ids : Tensor[3, B, seq_len]
        M-RoPE position IDs.

    Returns
    -------
    cos, sin : Tensor[B, seq_len, head_dim]
        Position embeddings with spatial positions overwritten.
    """
    # Step 1: Standard M-RoPE for all tokens
    cos, sin = original_forward(x, position_ids)  # [B, seq_len, head_dim]

    # Step 2: Check for stashed spatial data
    spatial_coords = getattr(rotary_emb_self, "_spatial_coords_3d", None)
    spatial_mask = getattr(rotary_emb_self, "_spatial_token_mask", None)

    if spatial_coords is None or spatial_mask is None:
        return cos, sin

    # spatial_mask: [B, seq_len] bool — True at spatial token positions
    # spatial_coords: [B, N_spatial, 3] — 3D coordinates for spatial tokens
    if not spatial_mask.any():
        return cos, sin

    # Step 3: Compute icosahedral cos/sin for spatial positions
    head_dim = cos.shape[-1]
    ico_cos, ico_sin = _build_icosahedral_cos_sin(
        spatial_coords, head_dim=head_dim, dtype=cos.dtype
    )  # [B, N_spatial, head_dim]

    # Step 4: Overwrite spatial positions in the full cos/sin tensors
    # spatial_mask: [B, seq_len] -> expand to [B, seq_len, head_dim]
    cos = cos.clone()
    sin = sin.clone()
    mask_expanded = spatial_mask.unsqueeze(-1).expand_as(cos)  # [B, seq_len, head_dim]
    cos[mask_expanded] = ico_cos.reshape(-1)
    sin[mask_expanded] = ico_sin.reshape(-1)

    # Step 5: Clear stashed data to avoid stale references
    rotary_emb_self._spatial_coords_3d = None
    rotary_emb_self._spatial_token_mask = None

    return cos, sin


def patch_model_forward(
    original_forward: Any,
    model_self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Patched Qwen3VLForConditionalGeneration.forward().

    Intercepts spatial_coords_3d and spatial_token_mask from kwargs,
    stashes them on the rotary embedding module, then calls the original forward.
    """
    # Extract our custom kwargs (not part of Qwen3-VL's signature)
    spatial_coords_3d = kwargs.pop("spatial_coords_3d", None)
    spatial_token_mask = kwargs.pop("spatial_token_mask", None)

    # Fallback: derive spatial_token_mask from mm_token_type_ids if not provided
    if spatial_token_mask is None and spatial_coords_3d is not None:
        mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            spatial_token_mask = mm_token_type_ids == SPATIAL_TOKEN_TYPE_ID

    # Stash on the rotary embedding module for the RoPE patch to consume
    rotary_emb = model_self.model.language_model.rotary_emb
    rotary_emb._spatial_coords_3d = spatial_coords_3d
    rotary_emb._spatial_token_mask = spatial_token_mask

    # Call original forward
    return original_forward(*args, **kwargs)


def apply_rope_patch(model: Any) -> None:
    """Apply the RoPE monkey-patch to a Qwen3VLForConditionalGeneration model.

    This modifies the model in-place by wrapping two methods:
    1. model.forward() — catches spatial_coords_3d from kwargs
    2. model.model.language_model.rotary_emb.forward() — injects icosahedral RoPE

    Parameters
    ----------
    model : Qwen3VLForConditionalGeneration
        The Qwen3-VL model instance to patch. Must have the standard
        model.model.language_model.rotary_emb attribute chain.

    Raises
    ------
    AttributeError
        If the model doesn't have the expected attribute structure.
    """
    # Validate model structure
    if not hasattr(model, "model"):
        raise AttributeError("Model must have a 'model' attribute (Qwen3VLModel).")
    if not hasattr(model.model, "language_model"):
        raise AttributeError("Model.model must have a 'language_model' attribute.")
    if not hasattr(model.model.language_model, "rotary_emb"):
        raise AttributeError("Language model must have a 'rotary_emb' attribute.")

    rotary_emb = model.model.language_model.rotary_emb

    # Initialize stash attributes
    rotary_emb._spatial_coords_3d = None
    rotary_emb._spatial_token_mask = None

    # Patch 1: RoPE forward (The Math)
    original_rope_forward = rotary_emb.forward
    rotary_emb.forward = functools.partial(
        patch_rope_forward, original_rope_forward, rotary_emb
    )

    # Patch 2: Model forward (The Catcher)
    original_model_forward = model.forward
    model.forward = functools.partial(
        patch_model_forward, original_model_forward, model
    )
