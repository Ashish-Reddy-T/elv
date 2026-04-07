"""Stage 4: Qwen3-VL-8B backbone with LoRA and RoPE monkey-patch."""

from spatialvlm.backbone.qwen3_vl import Qwen3BackboneStats, Qwen3VLBackbone
from spatialvlm.backbone.rope_patch import SPATIAL_TOKEN_TYPE_ID, apply_rope_patch

__all__ = [
    "Qwen3BackboneStats",
    "Qwen3VLBackbone",
    "SPATIAL_TOKEN_TYPE_ID",
    "apply_rope_patch",
]
