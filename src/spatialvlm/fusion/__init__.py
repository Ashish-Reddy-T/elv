"""Stage 3: Norm-balanced cross-attention fusion (SVA + gated cross-attn)."""

from spatialvlm.fusion.gated_cross_attn import GatedCrossAttentionBlock
from spatialvlm.fusion.norm_matching import RMSNormMatching
from spatialvlm.fusion.sva import SpatialVisionAggregator

__all__ = ["GatedCrossAttentionBlock", "RMSNormMatching", "SpatialVisionAggregator"]
