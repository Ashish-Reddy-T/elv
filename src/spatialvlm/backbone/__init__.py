"""Stage 4: Qwen3-VL-8B backbone with LoRA and position routing."""

from spatialvlm.backbone.position_routing import PositionRouter, RoutedPositionBatch
from spatialvlm.backbone.qwen3_vl import Qwen3BackboneStats, Qwen3VLBackbone

__all__ = [
    "PositionRouter",
    "Qwen3BackboneStats",
    "Qwen3VLBackbone",
    "RoutedPositionBatch",
]
