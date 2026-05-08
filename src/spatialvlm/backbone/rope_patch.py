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
    """Compute spatial cos/sin tensors for Qwen's rotary convention.

    ``IcosahedralRoPE3D`` returns interleaved sin/cos values for 48 rotary
    pairs. Qwen expects full-head tensors laid out as ``[pairs, pairs]`` for
    ``rotate_half``. The remaining 16 pairs are identity padded.
    """
    device = positions_3d.device
    rope3d = _get_icosahedral_rope(device)

    angles = rope3d(positions_3d)  # [B, N_spatial, 96]
    if angles.ndim != 3 or angles.shape[-1] != 96:
        raise ValueError(f"Expected IcosahedralRoPE3D output [B, N, 96], got {angles.shape}.")

    bsz, n_spatial, rope_dims = angles.shape
    n_pairs = rope_dims // 2
    total_pairs = head_dim // 2
    if n_pairs > total_pairs:
        raise ValueError(
            f"Icosahedral pairs ({n_pairs}) exceed target rotary pairs ({total_pairs})."
        )

    angles = angles.view(bsz, n_spatial, n_pairs, 2)
    sin_vals = angles[..., 0].to(dtype=dtype)
    cos_vals = angles[..., 1].to(dtype=dtype)

    pad_pairs = total_pairs - n_pairs
    if pad_pairs:
        pad_sin = torch.zeros(bsz, n_spatial, pad_pairs, device=device, dtype=dtype)
        pad_cos = torch.ones(bsz, n_spatial, pad_pairs, device=device, dtype=dtype)
        sin_vals = torch.cat([sin_vals, pad_sin], dim=-1)
        cos_vals = torch.cat([cos_vals, pad_cos], dim=-1)

    cos_out = torch.cat([cos_vals, cos_vals], dim=-1)
    sin_out = torch.cat([sin_vals, sin_vals], dim=-1)
    return cos_out.contiguous(), sin_out.contiguous()


def patch_rope_forward(
    original_forward: Any,
    rotary_emb_self: Any,
    x: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched Qwen3-VL RoPE.

    Spatial RoPE is injected only during the prefill call, where the current
    sequence length matches the stashed full prompt mask. Decode calls process
    one new text token and should use native M-RoPE unchanged.
    """
    cos, sin = original_forward(x, *args, **kwargs)

    spatial_coords = getattr(rotary_emb_self, "_spatial_coords_3d", None)
    spatial_mask = getattr(rotary_emb_self, "_spatial_token_mask", None)
    if spatial_coords is None or spatial_mask is None:
        return cos, sin

    current_seq_len = cos.shape[1]
    stashed_seq_len = spatial_mask.shape[1]
    if current_seq_len != stashed_seq_len or not spatial_mask.any():
        return cos, sin

    if spatial_coords.shape[0] != spatial_mask.shape[0]:
        raise ValueError(
            f"spatial_coords batch ({spatial_coords.shape[0]}) != mask batch "
            f"({spatial_mask.shape[0]})."
        )

    head_dim = cos.shape[-1]
    ico_cos, ico_sin = _build_icosahedral_cos_sin(
        spatial_coords,
        head_dim=head_dim,
        dtype=cos.dtype,
    )

    new_cos = cos.clone()
    new_sin = sin.clone()
    for batch_idx in range(spatial_mask.shape[0]):
        mask_indices = spatial_mask[batch_idx].nonzero(as_tuple=True)[0]
        if mask_indices.numel() != ico_cos.shape[1]:
            raise ValueError(
                f"Batch {batch_idx} has {mask_indices.numel()} spatial mask positions, "
                f"but {ico_cos.shape[1]} spatial coordinates."
            )
        new_cos[batch_idx, mask_indices, :] = ico_cos[batch_idx]
        new_sin[batch_idx, mask_indices, :] = ico_sin[batch_idx]

    rotary_emb_self._spatial_coords_3d = None
    rotary_emb_self._spatial_token_mask = None
    return new_cos, new_sin


_HOOK_CALL_COUNTER: int = 0  # incremented on every embed_tokens call during generation


def _deepstack_embedding_hook(
    module: Any,
    input: tuple,  # noqa: ARG001
    output: torch.Tensor,
) -> torch.Tensor:
    """Forward hook on the embedding layer: splice fused visual tokens at spatial positions."""
    global _HOOK_CALL_COUNTER
    _HOOK_CALL_COUNTER += 1
    call_n = _HOOK_CALL_COUNTER

    visual_embeds = getattr(module, "_deepstack_visual_embeds", None)
    spatial_mask = getattr(module, "_deepstack_spatial_mask", None)
    blend_alpha = float(getattr(module, "_deepstack_blend_alpha", 1.0))

    # print(
    #     f"[HOOK fire #{call_n}] module_id={id(module)} "
    #     f"output.shape={output.shape} "
    #     f"has_visual_embeds={visual_embeds is not None} "
    #     f"has_spatial_mask={spatial_mask is not None}",
    #     flush=True,
    # )

    if visual_embeds is None or spatial_mask is None:
        return output

    # print(
    #     f"[HOOK fire #{call_n}] mask.shape={spatial_mask.shape} "
    #     f"mask_true_count={spatial_mask.sum().item()} "
    #     f"output_seq_len={output.shape[1]} mask_seq_len={spatial_mask.shape[1]}",
    #     flush=True,
    # )

    # Only inject during prefill: seq_len must match the stashed mask length.
    if output.shape[1] != spatial_mask.shape[1]:
        # print(
        #     f"[HOOK fire #{call_n}] SKIP — seq_len mismatch "
        #     f"({output.shape[1]} != {spatial_mask.shape[1]})",
        #     flush=True,
        # )
        return output

    # Stats before injection
    # print(
    #     f"[HOOK fire #{call_n}] visual_embeds stats: "
    #     f"shape={visual_embeds.shape} "
    #     f"mean={visual_embeds.float().mean().item():.4f} "
    #     f"std={visual_embeds.float().std().item():.4f} "
    #     f"norm={visual_embeds.float().norm().item():.4f} "
    #     f"has_nan={visual_embeds.isnan().any().item()} "
    #     f"has_inf={visual_embeds.isinf().any().item()}",
    #     flush=True,
    # )
    spatial_positions = output[spatial_mask]
    target_norm = spatial_positions.float().norm().item()
    current_norm = visual_embeds.float().norm().item()
    scale = target_norm / (current_norm + 1e-6)

    # output: [B, seq_len, hidden_dim]  spatial_mask: [B, seq_len]
    # APPLY THE SCALE HERE
    flat = (visual_embeds * scale).reshape(-1, output.shape[-1]).to(output.dtype)
    
    output = output.clone()
    if blend_alpha >= 1.0:
        output[spatial_mask] = flat
    elif blend_alpha > 0.0:
        output[spatial_mask] = output[spatial_mask] * (1.0 - blend_alpha) + flat * blend_alpha

    return output


def stash_spatial_forward_kwargs(model_self: Any, kwargs: dict[str, Any]) -> None:
    """
    Robustly stashes spatial data onto all relevant submodules.
    Pops SpatialVLM-only kwargs so HF/PEFT forward methods never receive them.
    """
    # 1. Extract and remove custom SpatialVLM kwargs before HF/PEFT sees them.
    new_coords = kwargs.pop("spatial_coords_3d", None)
    new_mask = kwargs.pop("spatial_token_mask", None)
    new_visual_embeds = kwargs.pop("deepstack_visual_embeds", None)
    new_blend_alpha = kwargs.pop("deepstack_blend_alpha", None)
    mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)

    # 2. Guard Clause: If this is a subsequent call with no new data,
    # do NOT proceed, as we don't want to overwrite valid data with None.
    if new_coords is None and new_mask is None and new_visual_embeds is None:
        return

    # 3. Derive mask if it's missing but coords exist (fallback)
    if new_mask is None and new_coords is not None:
        if mm_token_type_ids is not None:
            # SPATIAL_TOKEN_TYPE_ID should be defined in your global scope (e.g., 3)
            new_mask = (mm_token_type_ids == SPATIAL_TOKEN_TYPE_ID)

    # 4. Global Broadcast: Iterate through all submodules
    # This ensures PEFT wrappers or nested language models are all tagged.
    count = 0
    if hasattr(model_self, "modules") and callable(model_self.modules):
        candidate_modules = list(model_self.modules())
    else:
        candidate_modules = []
        for path in (
            ("model", "language_model", "rotary_emb"),
            ("model", "model", "language_model", "rotary_emb"),
        ):
            obj = model_self
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                candidate_modules.append(obj)
            except AttributeError:
                continue

    for m in candidate_modules:
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
        except AttributeError:
            embed_module = None
        if embed_module is not None:
            embed_module._deepstack_visual_embeds = new_visual_embeds
            if new_mask is not None:
                embed_module._deepstack_spatial_mask = new_mask
            if new_blend_alpha is not None:
                embed_module._deepstack_blend_alpha = float(new_blend_alpha)

_SPATIAL_KWARGS = frozenset(
    (
        "spatial_coords_3d",
        "spatial_token_mask",
        "deepstack_visual_embeds",
        "deepstack_blend_alpha",
        "mm_token_type_ids",
    )
)


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
    # Remove our custom keys before passing to HF forward: Qwen3VLModel.forward()
    # passes **kwargs through to self.language_model() alongside its own explicit
    # deepstack_visual_embeds=None, causing "multiple values" TypeError if our key
    # is still present.  Stashing is already complete at this point.
    for k in _SPATIAL_KWARGS:
        kwargs.pop(k, None)
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
