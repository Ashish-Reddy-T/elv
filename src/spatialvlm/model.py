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
                n_spatial = 1369  # SVA always outputs 1369 DINOv2-matched queries
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
        for the prefill step. The DeepStack hook fires once on the prefill call to
        embed_tokens and self-clears, so it is a no-op for all decode steps.
        The RoPE patch is also a no-op during decode (seq_len=1 ≠ stashed prefill len).
        """
        from spatialvlm.backbone.rope_patch import stash_spatial_forward_kwargs

        self.backbone.load_model()

        with torch.no_grad():
            siglip_tokens, dinov2_tokens = self.encode_vision(siglip_pixels, dinov2_pixels)
            gatr_tokens, positions_3d = self.encode_geometry(depth, intrinsics)

            # Get live text-token embeddings for norm-matching (avoids stale EMA).
            # fused_tokens.shape[1] = 1369 (SVA output) — NOT siglip_tokens.shape[1]=576.
            embed_layer = self.backbone.model.get_input_embeddings()
            all_embeds_tmp = embed_layer(input_ids)  # [B, seq_len, D]
            n_fused = 1369  # SVA always outputs 1369 DINOv2-matched queries
            text_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
            text_mask[spatial_start_idx : spatial_start_idx + n_fused] = False
            text_tokens_for_norm = all_embeds_tmp[:, text_mask]  # [B, T_text, D]

            fused_tokens = self.fuse(
                siglip_tokens, dinov2_tokens, gatr_tokens,
                text_tokens=text_tokens_for_norm,
            )  # [B, 1369, 4096]


        # Build spatial token mask [B, seq_len]
        bsz, seq_len = input_ids.shape
        n_spatial = fused_tokens.shape[1]  # 1369
        spatial_token_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=input_ids.device)
        spatial_token_mask[:, spatial_start_idx : spatial_start_idx + n_spatial] = True

        # Resolve base model (unwrap PeftModel for direct attribute access)
        try:
            from peft import PeftModel as _PeftModel
            _base = (
                self.backbone.model.model
                if isinstance(self.backbone.model, _PeftModel)
                else self.backbone.model
            )
        except ImportError:
            _base = self.backbone.model

        # Stash visual embeds (DeepStack hook fires once on prefill embed_tokens call,
        # injects visual tokens, then self-clears so decode steps are unaffected) and
        # 3D coords (RoPE patch reads these each layer during prefill).
        # SFT checkpoint was trained without working DeepStack injection — the model
        # learned with 1369 identical <|image_pad|> placeholder embeddings.  Injecting
        # real visual features is OOD and produces degenerate output.  Skip the stash
        # until SFT is re-run with the fixed injection code.
        stash_spatial_forward_kwargs(_base, {
            "spatial_token_mask": spatial_token_mask,
        })

        # use_cache=True is critical: KV cache built during prefill, decode steps only
        # process new tokens.  Without it gradient_checkpointing sets use_cache=False,
        # causing full re-forwards that re-fire the (already cleared) DeepStack hook.
        gen_kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs.update(generate_kwargs)

        out = self.backbone.model.generate(**gen_kwargs)

        # Cleanup: remove stashed 3D coords from rotary modules
        for _, module in self.backbone.model.named_modules():
            if hasattr(module, "_spatial_coords_3d"):
                del module._spatial_coords_3d
            if hasattr(module, "_spatial_token_mask"):
                del module._spatial_token_mask

        return out