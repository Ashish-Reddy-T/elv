"""Model and training configuration dataclasses.

All pre-trained model constants are marked ⚠ VERIFY — they must be confirmed
by loading the model and reading its config:

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    print(cfg.to_dict())

Never copy constants from documentation into production code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    """Configuration for the dual vision encoder stage (Stage 1)."""

    # ⚠ VERIFY exact HuggingFace model ID before using
    siglip_model_id: str = "google/siglip2-so400m-patch16-naflex"
    # ⚠ VERIFY exact HuggingFace model ID before using
    dinov2_model_id: str = "facebook/dinov2-large"

    # Input resolutions chosen for exact integer patch counts
    siglip_image_size: int = 384   # 384 / 16 = 24.0 → exactly 576 patches
    dinov2_image_size: int = 518   # 518 / 14 = 37.0 → exactly 1369 patches

    # ⚠ VERIFY from cfg.vision_config.patch_size (SigLIP2) at runtime
    siglip_patch_size: int = 16
    # ⚠ VERIFY from cfg.patch_size (DINOv2) at runtime
    dinov2_patch_size: int = 14

    # Multi-layer extraction — evenly-spaced thirds of total depth
    # ⚠ VERIFY indices match actual model depth (SigLIP2: 27 layers, DINOv2: 24 layers)
    siglip_extract_layers: List[int] = field(default_factory=lambda: [9, 18, 27])
    dinov2_extract_layers: List[int] = field(default_factory=lambda: [8, 16, 24])

    # Projector output dim — must equal LLM hidden size
    # ⚠ VERIFY from Qwen3-VL cfg.hidden_size
    proj_output_dim: int = 4096


@dataclass
class GeometryConfig:
    """Configuration for the geometric branch (Stage 2)."""

    # Depth rendering — must match dinov2_image_size for pixel-perfect alignment
    depth_image_size: int = 518
    # 15th-percentile depth aggregation: nearest foreground bias per patch
    depth_percentile: float = 0.15

    # GATr configuration
    # ⚠ VERIFY against REPOS/geometric-algebra-transformer defaults
    gatr_blocks: int = 8
    gatr_mv_channels: int = 16   # multivector channels per token
    gatr_s_channels: int = 32    # scalar channels per token

    # PGA basis dimension (mathematical constant: 2^4 = 16 for 3D PGA)
    pga_dim: int = 16

    # Derived: invariant features extracted from GATr output
    # = gatr_s_channels (scalar) + gatr_mv_channels (MV norms)
    @property
    def gatr_invariant_dim(self) -> int:
        return self.gatr_s_channels + self.gatr_mv_channels  # 48

    # GridCellRoPE3D — our design, no pre-trained model
    n_tetra_dirs: int = 4                      # regular tetrahedron (isotropic 3D frame)
    n_freqs: int = 8                           # golden-ratio spaced frequencies
    base_freq: float = 10.0                    # f_k = base_freq × phi^k
    golden_ratio: float = 1.618033988749895    # phi

    @property
    def rope3d_dims(self) -> int:
        """Output dims: n_tetra_dirs × n_freqs × 2 (sin/cos) = 64."""
        return self.n_tetra_dirs * self.n_freqs * 2  # must equal Qwen3's rotary pairs


@dataclass
class FusionConfig:
    """Configuration for the fusion stage (Stage 3)."""

    # SVA (Spatial Vision Aggregator)
    sva_num_queries: int = 576    # matches SigLIP patch count (24×24 grid)
    # ⚠ VERIFY: 576 (SigLIP) + 1369 (DINOv2) + 1369 (GATr) = 3314
    sva_kv_tokens: int = 3314
    sva_num_layers: int = 2

    # Gated cross-attention injection — every 4th of 36 Qwen3 layers
    # ⚠ VERIFY: must match actual Qwen3 num_hidden_layers
    cross_attn_layers: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32, 36]
    )

    # RMS norm matching EMA
    norm_ema_momentum: float = 0.99

    # Typed attention bias: 3×3 learned matrix (9 scalar params, negligible cost)
    use_typed_attention_bias: bool = True


@dataclass
class BackboneConfig:
    """Configuration for the LLM backbone (Stage 4)."""

    # ⚠ VERIFY exact HuggingFace model ID
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"

    # All values below: ⚠ VERIFY from AutoConfig.from_pretrained(model_id)
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8    # GQA 4:1
    head_dim: int = 128             # hidden_size // num_attention_heads

    # M-RoPE section: [time, height, width] → 64 rotary pairs total
    # ⚠ VERIFY from cfg.rope_scaling.mrope_section
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64            # effective scale = lora_alpha / lora_rank = 2.0


@dataclass
class SpatialVLMConfig:
    """Top-level config composing all stage sub-configs."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
