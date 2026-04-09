"""Stage 3: Norm-balanced cross-attention fusion (SVA + RMS norm matching)."""

from spatialvlm.fusion.norm_matching import RMSNormMatching
from spatialvlm.fusion.sva import SpatialVisionAggregator

__all__ = ["RMSNormMatching", "SpatialVisionAggregator"]
