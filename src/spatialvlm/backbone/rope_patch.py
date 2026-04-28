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
    # cos_out = torch.cat([cos_vals, cos_vals], dim=-1)  # [B, N, 128]
    # sin_out = torch.cat([sin_vals, sin_vals], dim=-1)  # [B, N, 128]

    cos_out = torch.repeat_interleave(cos_vals, 2, dim=-1) # [B, N, 128]
    sin_out = torch.repeat_interleave(sin_vals, 2, dim=-1) # [B, N, 128]

    return cos_out.to(dtype=dtype), sin_out.to(dtype=dtype)


def patch_rope_forward(
    original_forward: Any,
    rotary_emb_self: Any,
    x: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched Qwen3VLTextRotaryEmbedding.forward().
    
    Ensures 3D icosahedral RoPE is applied to all layers during prefill
    without clearing stashed data prematurely.
    """
    # 1. Run the original M-RoPE first
    cos, sin = original_forward(x, *args, **kwargs)

    # 2. Retrieve stashed data
    spatial_coords = getattr(rotary_emb_self, "_spatial_coords_3d", None)
    spatial_mask = getattr(rotary_emb_self, "_spatial_token_mask", None)

    # 3. Check for presence of data
    if spatial_coords is None or spatial_mask is None:
        # print(spatial_coords is None, spatial_mask is None)
        # print("earliest exit")
        return cos, sin

    # 4. Determine Phase: Prefill vs. Decoding
    # cos shape is [B, seq_len, head_dim]
    current_seq_len = cos.shape[1]
    stashed_seq_len = spatial_mask.shape[1]

    # Only apply icosahedral logic if we are in the Prefill phase 
    # (where sequence lengths match). During decoding, current_seq_len is 1.
    if current_seq_len == stashed_seq_len:
        if not spatial_mask.any():
            # print('returning because not spatial_mask.any()')
            return cos, sin

        # # Compute icosahedral cos/sin
        head_dim = cos.shape[-1]
        ico_cos, ico_sin = _build_icosahedral_cos_sin(
            spatial_coords * 0.01, # <--- Scale here
            head_dim=head_dim, 
            dtype=cos.dtype
        )
        # Overwrite spatial positions
        cos = cos.clone()
        sin = sin.clone()
        
        # Expand mask to [B, seq_len, head_dim]
        mask_expanded = spatial_mask.unsqueeze(-1).expand_as(cos)
        
        # In-place update for the prefill tokens
        cos[mask_expanded] = ico_cos.reshape(-1)
        sin[mask_expanded] = ico_sin.reshape(-1)

        # print(f"DEBUG: Max Ico Theta: {torch.acos(ico_cos).max().item():.4f}")
        # print(f"DEBUG: Max Orig Theta: {torch.acos(cos[mask_expanded]).max().item():.4f}")

    # CRITICAL: We do NOT clear rotary_emb_self._spatial_coords_3d here.
    # Because this module is shared across layers, clearing it here would
    # prevent Layer 1, 2, 3... from seeing the data.
    # The attributes should be cleared manually after model.generate() returns.

    # print('returning end of function')
    return cos, sin

def _deepstack_embedding_hook(
    module: Any,
    input: tuple,  # noqa: ARG001
    output: torch.Tensor,
) -> torch.Tensor:
    """Forward hook on the embedding layer: splice fused visual tokens at spatial positions."""
    visual_embeds = getattr(module, "_deepstack_visual_embeds", None)
    spatial_mask = getattr(module, "_deepstack_spatial_mask", None)

    if visual_embeds is not None and spatial_mask is not None:
        # output: [B, seq_len, hidden_dim]  spatial_mask: [B, seq_len]
        flat = visual_embeds.reshape(-1, output.shape[-1]).to(output.dtype)
        output = output.clone()
        output[spatial_mask] = flat
        # Clear — one-shot per forward
        module._deepstack_visual_embeds = None
        module._deepstack_spatial_mask = None

    return output


def stash_spatial_forward_kwargs(model_self: Any, kwargs: dict[str, Any]) -> None:
    """
    Robustly stashes spatial data onto all relevant submodules.
    Uses .get() instead of .pop() to ensure data persists across multiple 
    internal calls during the generation loop.
    """
    # 1. Extract data without destroying the dictionary
    new_coords = kwargs.get("spatial_coords_3d", None)
    new_mask = kwargs.get("spatial_token_mask", None)
    new_visual_embeds = kwargs.get("deepstack_visual_embeds", None)

    # 2. Guard Clause: If this is a subsequent call with no new data,
    # do NOT proceed, as we don't want to overwrite valid data with None.
    if new_coords is None and new_mask is None and new_visual_embeds is None:
        return 

    # 3. Derive mask if it's missing but coords exist (fallback)
    if new_mask is None and new_coords is not None:
        mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            # SPATIAL_TOKEN_TYPE_ID should be defined in your global scope (e.g., 3)
            new_mask = (mm_token_type_ids == SPATIAL_TOKEN_TYPE_ID)

    # 4. Global Broadcast: Iterate through all submodules
    # This ensures PEFT wrappers or nested language models are all tagged.
    count = 0
    for m in model_self.modules():
        c_name = m.__class__.__name__.lower()
        # Look for anything that resembles a Rotary Embedding
        if "rotary" in c_name or "rope" in c_name:
            if new_coords is not None:
                m._spatial_coords_3d = new_coords
            if new_mask is not None:
                m._spatial_token_mask = new_mask
            count += 1
            
    # 5. Handle Word Embeddings (for the DeepStack/VLM hook)
    if new_visual_embeds is not None:
        try:
            embed_module = _find_embed_tokens(model_self)
            embed_module._deepstack_visual_embeds = new_visual_embeds
            if new_mask is not None:
                embed_module._deepstack_spatial_mask = new_mask
        except Exception as e:
            print(f"DEBUG: Failed to stash visual embeds on embedding layer: {e}")

    print(f"DEBUG: Stashed spatial data on {count} RoPE modules.")
    print(f"DEBUG: Mask provided: {new_mask is not None}, Coords provided: {new_coords is not None}")

def patch_model_forward(
    original_forward: Any,
    model_self: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Patched Qwen3VLForConditionalGeneration.forward().

    Intercepts spatial_coords_3d, spatial_token_mask, and deepstack_visual_embeds
    from kwargs.  Stashes RoPE data on the rotary embedding module and visual
    embeddings on the word-embedding module (consumed by a registered hook).
    """
    stash_spatial_forward_kwargs(model_self, kwargs)
    return original_forward(*args, **kwargs)


def _find_embed_tokens(model: Any) -> Any:
    """Locate the word-embedding module (works through PEFT wrapping)."""
    # Standard HuggingFace method (works for PeftModel too)
    embed = getattr(model, "get_input_embeddings", None)
    if callable(embed):
        module = embed()
        if module is not None:
            return module

    # Fallback: traverse known Qwen3-VL paths
    for path in [
        ("model", "language_model", "embed_tokens"),
        ("model", "embed_tokens"),
        ("base_model", "model", "model", "language_model", "embed_tokens"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue

    raise AttributeError("Cannot find embed_tokens module in model.")


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
    rotary_emb.forward = functools.partial(patch_rope_forward, original_rope_forward, rotary_emb)

    # Patch 2: Model forward (The Catcher + DeepStack stash)
    original_model_forward = model.forward
    model.forward = functools.partial(patch_model_forward, original_model_forward, model)

    # Patch 3: Embedding hook (DeepStack injection — splices visual tokens into embeddings)
    embed_tokens = _find_embed_tokens(model)
    embed_tokens._deepstack_visual_embeds = None
    embed_tokens._deepstack_spatial_mask = None
    embed_tokens.register_forward_hook(_deepstack_embedding_hook)
