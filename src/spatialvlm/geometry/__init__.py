"""Stage 2: Geometric branch (backprojection, GATr, IcosahedralRoPE3D)."""

from spatialvlm.geometry.backproject import aggregate_patches_percentile, backproject_depth_map
from spatialvlm.geometry.gatr_wrapper import GATrWrapper
from spatialvlm.geometry.gridcell_rope3d import GridCellRoPE3D, IcosahedralRoPE3D

__all__ = [
    "aggregate_patches_percentile",
    "backproject_depth_map",
    "GATrWrapper",
    "GridCellRoPE3D",
    "IcosahedralRoPE3D",
]
