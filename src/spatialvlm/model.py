"""Integrated SpatialVLM model wiring all 5 architectural stages.

Pipeline:
  Stage 1: Dual Vision Encoding
    SigLIP2-SO400M/16 @ 384px  -> [B, 576, 3456] -> MLP -> [B, 576, 4096]
    DINOv2-L/14 @ 518px        -> [B, 1369, 3072] -> MLP -> [B, 1369, 4096]

  Stage 2: Geometric Branch (parallel to Stage 1)
    GT Depth @ 518x518 -> backproject -> 15th-percentile aggregation -> [B, 1369, 3]
    GATr (8 equivariant blocks, PGA)                                 -> [B, 1369, 4096]

  Stage 3: Fusion
    SVA cross-attention: 1369 DINOv2-based queries attend 3314 KV tokens -> [B, 1369, 4096]
    RMS norm matching: scale to text-token magnitude                     -> [B, 1369, 4096]

  Stage 4: LLM Backbone
    Qwen3-VL-8B with LoRA rank-32
    RoPE monkey-patch: text -> M-RoPE, spatial -> IcosahedralRoPE3D
    DeepStack: hidden_states[spatial_mask] += fused_visual_embeds

  Stage 5: Training (external — see training/ modules)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone
from spatialvlm.config.model import SpatialVLMConfig
from spatialvlm.encoders.dinov2 import DINOv2Encoder
from spatialvlm.encoders.projector import MLPProjector
from spatialvlm.encoders.siglip import SigLIP2Encoder
from spatialvlm.fusion.norm_matching import RMSNormMatching
from spatialvlm.fusion.sva import SpatialVisionAggregator
from spatialvlm.geometry.backproject import aggregate_patches_percentile, backproject_depth_map
from spatialvlm.geometry.gatr_wrapper import GATrWrapper
from spatialvlm.utils.camera import CameraIntrinsics


class SpatialVLM(nn.Module):
    """Integrated SpatialVLM: 5-stage pipeline for spatial intelligence recovery.

    This module wires together all architectural components and manages the
    data flow from raw inputs (images, depth, camera intrinsics, text) through
    to LLM output logits.

    Parameters
    ----------
    config : SpatialVLMConfig
        Top-level configuration composing all stage sub-configs.
    device : torch.device | None
        Target device. If None, defaults to CPU.
    lazy_load_encoders : bool
        If True, vision encoders are loaded on first forward() call.
    lazy_load_backbone : bool
        If True, LLM backbone is loaded on first forward() call.
    """

    def __init__(
        self,
        config: SpatialVLMConfig | None = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        lazy_load_encoders: bool = True,
        lazy_load_backbone: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if config is None:
            config = SpatialVLMConfig()
        if device is None:
            device = torch.device("cpu")

        self.config = config
        self._device = device

        # ── Stage 1: Dual Vision Encoders (frozen) ──────────────────────
        self.siglip_encoder = SigLIP2Encoder(
            model_id=config.encoder.siglip_model_id,
            extract_layers=config.encoder.siglip_extract_layers,
            device=device,
            lazy_load=lazy_load_encoders,
            local_files_only=local_files_only,
        )
        self.dinov2_encoder = DINOv2Encoder(
            model_id=config.encoder.dinov2_model_id,
            extract_layers=config.encoder.dinov2_extract_layers,
            device=device,
            lazy_load=lazy_load_encoders,
            local_files_only=local_files_only,
        )

        # MLP projectors: encoder output dim -> LLM hidden size
        # SigLIP: 3 layers × 1152 = 3456 -> 4096
        self.siglip_projector = MLPProjector(
            in_dim=self.siglip_encoder.out_dim,
            out_dim=config.encoder.proj_output_dim,
        )
        # DINOv2: 3 layers × 1024 = 3072 -> 4096
        self.dinov2_projector = MLPProjector(
            in_dim=self.dinov2_encoder.out_dim,
            out_dim=config.encoder.proj_output_dim,
        )

        # ── Stage 2: Geometric Branch ───────────────────────────────────
        self.gatr = GATrWrapper(
            num_blocks=config.geometry.gatr_blocks,
            gatr_mv_channels=config.geometry.gatr_mv_channels,
            gatr_s_channels=config.geometry.gatr_s_channels,
            projector_out_dim=config.encoder.proj_output_dim,
            device=device,
        )

        # ── Stage 3: Fusion ─────────────────────────────────────────────
        self.sva = SpatialVisionAggregator(
            hidden_dim=config.encoder.proj_output_dim,
            num_queries=config.fusion.sva_num_queries,
            num_layers=config.fusion.sva_num_layers,
            use_typed_attention_bias=config.fusion.use_typed_attention_bias,
        )
        self.norm_matching = RMSNormMatching(
            ema_momentum=config.fusion.norm_ema_momentum,
        )

        # ── Stage 4: LLM Backbone ──────────────────────────────────────
        self.backbone = Qwen3VLBackbone(
            model_id=config.backbone.model_id,
            lora_rank=config.backbone.lora_rank,
            lora_alpha=config.backbone.lora_alpha,
            lazy_load=lazy_load_backbone,
            local_files_only=local_files_only,
            device=device,
            torch_dtype=torch_dtype,
        )

        self.to(device)

    def encode_vision(
        self,
        siglip_pixels: torch.Tensor,
        dinov2_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Run dual vision encoders and project to LLM hidden size.

        Parameters
        ----------
        siglip_pixels : Tensor[B, 3, 384, 384]
            Preprocessed images for SigLIP2.
        dinov2_pixels : Tensor[B, 3, 518, 518]
            Preprocessed images for DINOv2.

        Returns
        -------
        siglip_tokens : Tensor[B, 576, 4096]
            Projected SigLIP2 features.
        dinov2_tokens : Tensor[B, 1369, 4096]
            Projected DINOv2 features.
        """
        siglip_feats = self.siglip_encoder(siglip_pixels)  # [B, 576, 3456]
        dinov2_feats = self.dinov2_encoder(dinov2_pixels)  # [B, 1369, 3072]

        siglip_tokens = self.siglip_projector(siglip_feats)  # [B, 576, 4096]
        dinov2_tokens = self.dinov2_projector(dinov2_feats)  # [B, 1369, 4096]

        return siglip_tokens, dinov2_tokens

    def encode_geometry(
        self,
        depth: torch.Tensor,
        intrinsics: CameraIntrinsics,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 2: Backproject depth, aggregate to patches, run GATr.

        Parameters
        ----------
        depth : Tensor[B, 518, 518]
            GT depth map in metres.
        intrinsics : CameraIntrinsics
            Camera intrinsics for the depth map.

        Returns
        -------
        gatr_tokens : Tensor[B, 1369, 4096]
            Projected GATr geometric features.
        positions_3d : Tensor[B, 1369, 3]
            Per-patch 3D positions (for RoPE injection).
        """
        point_map = backproject_depth_map(depth, intrinsics)  # [B, 518, 518, 3]
        positions_3d = aggregate_patches_percentile(
            point_map,
            depth,
            patch_size=self.config.encoder.dinov2_patch_size,
            percentile=self.config.geometry.depth_percentile,
        )  # [B, 1369, 3]

        gatr_tokens = self.gatr(positions_3d)  # [B, 1369, 4096]

        return gatr_tokens, positions_3d

    def fuse(
        self,
        siglip_tokens: torch.Tensor,
        dinov2_tokens: torch.Tensor,
        gatr_tokens: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 3: SVA cross-attention + RMS norm matching.

        Parameters
        ----------
        siglip_tokens : Tensor[B, 576, 4096]
        dinov2_tokens : Tensor[B, 1369, 4096]
        gatr_tokens : Tensor[B, 1369, 4096]
        text_tokens : Tensor[B, T, 4096] | None
            Text token embeddings for RMS norm matching EMA update (training only).

        Returns
        -------
        fused_tokens : Tensor[B, 1369, 4096]
            Norm-matched spatial tokens ready for DeepStack injection.
        """
        # SVA: 1369 DINOv2-based queries attend to 3314 KV tokens
        fused = self.sva(
            siglip_tokens=siglip_tokens,
            dinov2_tokens=dinov2_tokens,
            gatr_tokens=gatr_tokens,
        )  # [B, 1369, 4096]

        # RMS norm matching: scale to text-token magnitude
        fused = self.norm_matching(fused, text_tokens=text_tokens)  # [B, 1369, 4096]

        return fused

    def build_deepstack_inputs(
        self,
        fused_tokens: torch.Tensor,
        positions_3d: torch.Tensor,
        input_ids: torch.Tensor,
        spatial_start_idx: int,
    ) -> dict[str, Any]:
        """Build kwargs for Qwen3-VL forward with DeepStack and spatial RoPE.

        DeepStack mechanism (native Qwen3-VL):
            hidden_states[spatial_mask, :] += fused_visual_embeds
        at early LLM layers. This is a residual addition with zero parameters.

        Parameters
        ----------
        fused_tokens : Tensor[B, 1369, 4096]
            Norm-matched spatial tokens from Stage 3.
        positions_3d : Tensor[B, 1369, 3]
            Per-token 3D positions for IcosahedralRoPE3D.
        input_ids : Tensor[B, seq_len]
            Token IDs for the LLM (text + placeholder spatial tokens).
        spatial_start_idx : int
            Index in the sequence where spatial tokens begin.

        Returns
        -------
        kwargs : dict
            Extra kwargs to pass to the backbone's forward():
            - spatial_coords_3d: Tensor[B, N_spatial, 3]
            - spatial_token_mask: Tensor[B, seq_len] bool
            - deepstack_visual_embeds: Tensor[B, N_spatial, 4096]
        """
        bsz, seq_len = input_ids.shape
        n_spatial = fused_tokens.shape[1]  # 1369

        # Build spatial token mask: True at positions where spatial tokens are injected
        spatial_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=input_ids.device)
        spatial_end_idx = spatial_start_idx + n_spatial
        spatial_mask[:, spatial_start_idx:spatial_end_idx] = True

        return {
            "spatial_coords_3d": positions_3d,  # for RoPE monkey-patch
            "spatial_token_mask": spatial_mask,  # for RoPE monkey-patch
            "deepstack_visual_embeds": fused_tokens,  # for DeepStack injection
        }

    def forward(
        self,
        siglip_pixels: torch.Tensor,
        dinov2_pixels: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: CameraIntrinsics,
        input_ids: torch.Tensor,
        spatial_start_idx: int = 0,
        text_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> Any:
        """Full forward pass through all 5 stages.

        Parameters
        ----------
        siglip_pixels : Tensor[B, 3, 384, 384]
            Preprocessed images for SigLIP2.
        dinov2_pixels : Tensor[B, 3, 518, 518]
            Preprocessed images for DINOv2.
        depth : Tensor[B, 518, 518]
            GT depth map in metres.
        intrinsics : CameraIntrinsics
            Camera intrinsics for the depth map.
        input_ids : Tensor[B, seq_len]
            Token IDs for the LLM. Must include placeholder tokens for spatial injection.
        spatial_start_idx : int
            Index in input_ids where the 1369 spatial tokens begin.
        text_tokens : Tensor[B, T, 4096] | None
            Text token embeddings for RMS norm matching (training only).
        attention_mask : Tensor[B, seq_len] | None
            Attention mask for the LLM.
        labels : Tensor[B, seq_len] | None
            Labels for causal LM loss computation.
        **backbone_kwargs
            Additional kwargs passed through to the backbone.

        Returns
        -------
        output : CausalLMOutputWithPast or similar
            LLM output with loss (if labels provided) and logits.
        """
        # Stage 1: Dual vision encoding (parallel in practice, sequential here)
        siglip_tokens, dinov2_tokens = self.encode_vision(
            siglip_pixels, dinov2_pixels
        )  # [B, 576, 4096], [B, 1369, 4096]

        # Stage 2: Geometric branch
        gatr_tokens, positions_3d = self.encode_geometry(
            depth, intrinsics
        )  # [B, 1369, 4096], [B, 1369, 3]

        # Stage 3: Fusion — provide text tokens for norm matching.
        # If not supplied by the caller, extract them from input_ids to avoid
        # stale EMA values causing 40-100x vision/text norm mismatch.
        if text_tokens is None and input_ids is not None:
            with torch.no_grad():
                embed_layer = self.backbone.model.get_input_embeddings()
                all_embeds = embed_layer(input_ids)  # [B, seq_len, D]
                n_spatial = siglip_tokens.shape[1]
                spatial_end = spatial_start_idx + n_spatial
                text_mask = torch.ones(input_ids.shape[1], dtype=torch.bool,
                                       device=input_ids.device)
                text_mask[spatial_start_idx:spatial_end] = False
                text_tokens = all_embeds[:, text_mask]  # [B, T_text, D]

        fused_tokens = self.fuse(
            siglip_tokens, dinov2_tokens, gatr_tokens, text_tokens=text_tokens
        )  # [B, 1369, 4096]

        # Stage 4: Build DeepStack inputs and run LLM backbone
        deepstack_kwargs = self.build_deepstack_inputs(
            fused_tokens, positions_3d, input_ids, spatial_start_idx
        )

        # Merge with any additional backbone kwargs
        forward_kwargs = {
            "input_ids": input_ids,
            **deepstack_kwargs,
            **backbone_kwargs,
        }
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        if labels is not None:
            forward_kwargs["labels"] = labels

        return self.backbone(**forward_kwargs)

    def generate(
        self,
        siglip_pixels: torch.Tensor,
        dinov2_pixels: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: CameraIntrinsics,
        input_ids: torch.Tensor,
        spatial_start_idx: int,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        """Autoregressive generation with spatial context.

        Visual encoding is run once (no grad) then spatial embeddings are stashed
        for the prefill step. The RoPE patch and DeepStack hook are self-clearing,
        so they fire exactly once during prefill and are no-ops for decode steps.
        """
        from spatialvlm.backbone.rope_patch import stash_spatial_forward_kwargs

        self.backbone.load_model()  # no-op if already loaded; triggers lazy init

        # Check if it's a wrapper (like a functools.partial or a decorated function)
        import inspect

        with torch.no_grad():
            siglip_tokens, dinov2_tokens = self.encode_vision(siglip_pixels, dinov2_pixels)
            gatr_tokens, positions_3d = self.encode_geometry(depth, intrinsics)

            # DEBUG 1: Visual/Geo stats
            # print(f"DEBUG: siglip_tokens mean: {siglip_tokens.mean().item():.4f}, has_nan: {torch.isnan(siglip_tokens).any()}")
            # print(f"DEBUG: gatr_tokens mean: {gatr_tokens.mean().item():.4f}, has_nan: {torch.isnan(gatr_tokens).any()}")
            
            # Extract text embeddings from the prompt so norm_matching can scale
            # visual tokens to match text token magnitude. Without this, stale EMA
            # values cause 40-100x norm mismatch → model outputs garbage.
            embed_layer = self.backbone.model.get_input_embeddings()
            all_embeds = embed_layer(input_ids)  # [B, seq_len, D]
            n_spatial = siglip_tokens.shape[1]  # 1369
            spatial_end = spatial_start_idx + n_spatial
            # Mask out spatial placeholder positions; use remaining text tokens
            text_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
            text_mask[spatial_start_idx:spatial_end] = False
            text_tokens_for_norm = all_embeds[:, text_mask]  # [B, T_text, D]

            fused_tokens = self.fuse(siglip_tokens, dinov2_tokens, gatr_tokens,
                                     text_tokens=text_tokens_for_norm)
            # DEBUG 2: Compare Norms
            text_norm = text_tokens_for_norm.norm(p=2, dim=-1).mean().item()
            fused_norm = fused_tokens.norm(p=2, dim=-1).mean().item()
            # print(f"DEBUG: Avg Text Norm: {text_norm:.4f} | Avg Fused Norm: {fused_norm:.4f}")
            if abs(text_norm - fused_norm) > (text_norm * 10):
                print("WARNING: Massive norm mismatch detected!")

            deepstack_kwargs = self.build_deepstack_inputs(
                fused_tokens, positions_3d, input_ids, spatial_start_idx
            )

            # DEBUG 3: Inspect hook data
            # Assuming deepstack_kwargs contains the indices/embeddings to be swapped
            # print(f"DEBUG: deepstack_kwargs keys: {deepstack_kwargs.keys()}")
            # print(f"DEBUG: spatial_start_idx: {spatial_start_idx}")
            # print(f"DEBUG: fused_tokens shape: {fused_tokens.shape}")


        # Stash on the base model (not the PeftModel wrapper).
        # apply_rope_patch patches base_model.model.language_model.rotary_emb and
        # registers the DeepStack hook on base_model's embed_tokens — both require
        # the un-wrapped Qwen3VLForConditionalGeneration, not the PeftModel.
        try:
            from peft import PeftModel as _PeftModel
            _hf = self.backbone.model
            _base = _hf.model if isinstance(_hf, _PeftModel) else _hf
            # --- THE FIX START ---
            # If we are in PEFT mode, the 'active' embed tokens might be inside 
            # the PeftModel wrapper, not the base model.
            _active_model = self.backbone.model 
        except ImportError:
            _base = self.backbone.model
            _active_model = _base

        # 1. Stash on base as usual (for safety/consistency)
        # 1. Create the mask manually based on the same logic you used for embeddings
        spatial_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        n_spatial = fused_tokens.shape[1]
        spatial_token_mask[:, spatial_start_idx : spatial_start_idx + n_spatial] = True

        # 2. Add it to the deepstack_kwargs so stash_spatial_forward_kwargs finds it
        deepstack_kwargs["spatial_token_mask"] = spatial_token_mask

        # 3. Call the stash function (ensure this is the PeftModel or Base model)
        stash_spatial_forward_kwargs(_base, deepstack_kwargs)
        
        # 2. ALSO stash on the active model if it's different (the PEFT wrapper)
        if _active_model is not _base:
            # Re-build kwargs because stash_spatial_forward_kwargs pops keys
            # We need to pass the data again to the wrapper's modules
            retry_kwargs = {
                "spatial_coords_3d": positions_3d,
                "spatial_token_mask": deepstack_kwargs.get("spatial_token_mask"),
                "deepstack_visual_embeds": fused_tokens
            }
            stash_spatial_forward_kwargs(_active_model, retry_kwargs)
        # --- THE FIX END ---

        # use_cache=True so the KV cache is built during prefill and decode steps only
        # process new tokens — without this, gradient_checkpointing_enable() sets
        # config.use_cache=False, causing full re-forward at every decode step and
        # clearing the one-shot DeepStack hook after the first token.
        gen_kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs.update(generate_kwargs)

        from spatialvlm.backbone.rope_patch import _find_embed_tokens
        embed_module = _find_embed_tokens(_base)
        # print(f"DEBUG [PRE-GEN]: Embed Module ID: {id(embed_module)}")
        # print(f"DEBUG [PRE-GEN]: Has visual embeds: {hasattr(embed_module, '_deepstack_visual_embeds')}")

        rotary_emb = _base.model.language_model.rotary_emb
        # print(f"DEBUG [PRE-GEN]: Has 3D coords: {hasattr(rotary_emb, '_spatial_coords_3d')}")

        active_embed = _find_embed_tokens(_active_model)
        # print(f"DEBUG [PRE-GEN]: Active Embed ID: {id(active_embed)}")
        # print(f"DEBUG [PRE-GEN]: Active Has visual embeds: {hasattr(active_embed, '_deepstack_visual_embeds')}")
        
        # 1. Manual Embedding Injection
        embed_layer = self.backbone.model.get_input_embeddings()
        all_embeds = embed_layer(input_ids)
        n_spatial = fused_tokens.shape[1]
        all_embeds[:, spatial_start_idx : spatial_start_idx + n_spatial] = fused_tokens.to(all_embeds.dtype)  

        # 2. FORCE-BROADCAST 3D COORDS
        # We iterate through the whole model to find ANY rotary_emb module
        # This fixes the "POST-GEN: True" issue by ensuring the active layer is hit.
        found_rope = False
        for name, module in self.backbone.model.named_modules():
            if "rotary_emb" in name.lower():
                module._spatial_coords_3d = positions_3d
                module._spatial_token_mask = deepstack_kwargs.get("spatial_token_mask")
                found_rope = True
                # print(f"DEBUG: Successfully tagged RoPE module: {name}")

        if not found_rope:
            print("WARNING: No Rotary Embedding modules found! Spatial logic will fail.")


        # 3. Generation with Metadata
        # Qwen-VL often expects 'image_grid_thw' to trigger the M-RoPE logic
        grid_thw = torch.tensor([[1, 37, 37]], device=all_embeds.device)
        
        gen_kwargs = {
            "inputs_embeds": all_embeds,
            "image_grid_thw": grid_thw, # Tell the model it's a 37x37 grid
            "use_cache": True,
            "attention_mask": attention_mask,
            "max_new_tokens": 128,
        }
        gen_kwargs.update(generate_kwargs)

        out = self.backbone.model.generate(**gen_kwargs)
        
        # 4. Manual Post-Gen Cleanup (Self-clearing)
        for name, module in self.backbone.model.named_modules():
            if hasattr(module, "_spatial_coords_3d"):
                del module._spatial_coords_3d
                
        return out
    
    ######################################3
        # 3. Prepare generation kwargs
        # We MUST remove 'input_ids' when passing 'inputs_embeds'
        grid_thw = torch.tensor([[1, 37, 37]], device=all_embeds.device)

        gen_kwargs: dict[str, Any] = {
            "inputs_embeds": all_embeds,
            "use_cache": True,
            "image_grid_thw": grid_thw, # CRITICAL for Qwen-VL models
        }
        
        # Ensure the attention mask allows text to see all 1369 spatial tokens
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        
        gen_kwargs.update(generate_kwargs)

        # Re-stash just in case the RoPE hook is actually looking for the attributes
        stash_spatial_forward_kwargs(_base, deepstack_kwargs)

        # 4. Still stash for RoPE (Geometric) logic
        # Even if the embedding hook fails, the RoPE hook might still work if we 
        # can get the model to actually start processing the sequence.
        stash_spatial_forward_kwargs(_base, deepstack_kwargs)

        # 5. Generate
        gk_max_tokens = gen_kwargs["max_new_tokens"]
        gen_kwargs["max_new_tokens"] = 1
        
        tokenizer = None
        for attr in ['tokenizer', 'processor', 'text_tokenizer']:
            if hasattr(self.backbone, attr):
                obj = getattr(self.backbone, attr)
                tokenizer = getattr(obj, 'tokenizer', obj)
                break
        # 1. Prediction Check
        debug_out = self.backbone.model.generate(
            **gen_kwargs, 
            output_scores=True, 
            return_dict_in_generate=True
        )
        top_token = torch.argmax(debug_out.scores[0], dim=-1)

        if tokenizer is None:
            # If the backbone doesn't have it, check the model's internal config 
            # or just use the ID for now.
            print(f"DEBUG: Could not find tokenizer. First token ID: {top_token.item()}")
        else:
            print(f"DEBUG: Token string: '{tokenizer.decode(top_token)}'")


        gen_kwargs["max_new_tokens"] = gk_max_tokens
        out = self.backbone.model.generate(**gen_kwargs)
        
        # Debugging Post-Gen
        print(f"DEBUG [POST-GEN]: 3D coords still stashed: {hasattr(_base.model.language_model.rotary_emb, '_spatial_coords_3d')}")
        
        return out