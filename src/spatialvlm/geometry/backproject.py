"""Depth map backprojection and patch-level 3D aggregation.

The 15th-percentile aggregation strategy:
  For each spatial patch, rank all pixels by depth (ascending = nearest first).
  Select the pixel at the 15th-percentile rank. This captures the nearest
  foreground surface rather than the mean depth, which is biased toward
  distant backgrounds (doorways, ceilings). Robust to ~29 outlier/invalid
  pixels out of 196 per patch (14×14 DINOv2 patch).
"""

from __future__ import annotations

import torch

from spatialvlm.utils.camera import CameraIntrinsics, backproject_pixel, make_pixel_grid


def backproject_depth_map(
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    """Backproject a batch of depth maps into 3D point maps.

    Parameters
    ----------
    depth : Tensor[B, H, W]
        Metric depth in metres. Zero values are treated as invalid/missing.
    intrinsics : CameraIntrinsics
        Shared camera intrinsics for the batch.

    Returns
    -------
    point_map : Tensor[B, H, W, 3]
        3D camera-space coordinates [X, Y, Z] per pixel.
        Invalid pixels (depth == 0) produce (0, 0, 0).
    """
    # depth: [B, H, W]
    b, h, w = depth.shape
    grid = make_pixel_grid(w, h, device=depth.device, dtype=depth.dtype)  # [H, W, 2]
    u = grid[..., 0].unsqueeze(0).expand(b, -1, -1)   # [B, H, W]
    v = grid[..., 1].unsqueeze(0).expand(b, -1, -1)   # [B, H, W]
    return backproject_pixel(u, v, depth, intrinsics)  # [B, H, W, 3]


def aggregate_patches_percentile(
    point_map: torch.Tensor,
    depth: torch.Tensor,
    patch_size: int = 14,
    percentile: float = 0.15,
) -> torch.Tensor:
    """Aggregate 3D points to patch-level using the k-th percentile depth.

    For each spatial patch of (patch_size × patch_size) pixels, this function
    identifies the pixel at the given depth percentile (nearest foreground)
    and returns its 3D coordinates. This is the hypothesis H2e module: we claim
    15th-percentile > mean aggregation for spatial task performance.

    Parameters
    ----------
    point_map : Tensor[B, H, W, 3]
        Full 3D point map from backproject_depth_map.
    depth : Tensor[B, H, W]
        Depth values for percentile ranking (same H, W as point_map).
    patch_size : int
        Spatial patch size in pixels. Use 14 for DINOv2.
    percentile : float
        Depth percentile to select. 0.15 = 15th percentile (nearest foreground).

    Returns
    -------
    patch_points : Tensor[B, n_patches, 3]
        One 3D point per patch.
        n_patches = (H // patch_size) * (W // patch_size).
        For H=W=518, patch_size=14: n_patches = 37 * 37 = 1369.
    """
    b, h, w, _ = point_map.shape
    ph = h // patch_size   # number of patches along height (37 for 518/14)
    pw = w // patch_size   # number of patches along width  (37 for 518/14)
    n = ph * pw            # total patches (1369 for 518×518 / 14)
    px = patch_size * patch_size  # pixels per patch (196 for 14×14)

    # Reshape point_map to [B, n_patches, pixels_per_patch, 3]
    # Step 1: [B, ph, patch_size, pw, patch_size, 3]
    pm = point_map.view(b, ph, patch_size, pw, patch_size, 3)
    # Step 2: [B, ph, pw, patch_size, patch_size, 3]
    pm = pm.permute(0, 1, 3, 2, 4, 5).contiguous()
    # Step 3: [B, n, px, 3]
    pm = pm.view(b, n, px, 3)
    assert pm.shape == (b, n, px, 3), f"Reshape failed: expected {(b, n, px, 3)}, got {pm.shape}"

    # Reshape depth to [B, n_patches, pixels_per_patch]
    dp = depth.view(b, ph, patch_size, pw, patch_size)
    dp = dp.permute(0, 1, 3, 2, 4).contiguous()
    dp = dp.view(b, n, px)  # [B, n, px]

    # Replace invalid depths (zero or NaN) with inf so they sort last (farthest)
    dp_valid = dp.clone()
    dp_valid[(dp_valid == 0) | torch.isnan(dp_valid)] = float("inf")

    # Sort ascending: smallest (nearest) depth first → [B, n, px]
    sorted_idx = torch.argsort(dp_valid, dim=-1)

    # Select pixel at the percentile rank
    # Clamp to valid range in case percentile * px rounds to px
    k = max(0, min(int(percentile * px), px - 1))

    # Index of the selected pixel per patch: [B, n]
    sel_idx = sorted_idx[:, :, k]

    # Gather 3D point at selected index
    # pm: [B, n, px, 3], sel_idx: [B, n]
    # expand sel_idx to [B, n, 1, 3] for gather on dim=2
    sel_idx_exp = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    patch_points = torch.gather(pm, dim=2, index=sel_idx_exp).squeeze(2)  # [B, n, 3]
    assert patch_points.shape == (b, n, 3), f"Expected {(b, n, 3)}, got {patch_points.shape}"

    return patch_points  # [B, n_patches, 3]


def pool_positions_to_sva_grid(
    positions: torch.Tensor,
    source_h: int = 37,
    source_w: int = 37,
    target_h: int = 24,
    target_w: int = 24,
) -> torch.Tensor:
    """DEPRECATED: No longer needed — SVA now uses 1369 DINOv2-based queries.

    With the icosahedral redesign, 1369 backprojected positions map 1:1 to
    1369 SVA queries. No spatial pooling is required. This function is kept
    for backward compatibility and ablation testing (H1d: 1369 vs 576 queries).

    Pool 3D positions from the DINOv2 patch grid to the SVA query grid.

    Each SVA query at grid cell (i, j) corresponds to a spatial sub-region of
    the source (DINOv2/GATr) grid.  The 3D position assigned to that query is
    the mean of the source positions that fall within its region.

    This ensures geometric consistency: the fused token's content (from SVA
    cross-attention over a spatial sub-region) and its positional encoding
    (GridCellRoPE3D) describe the **same** physical area of the scene.

    The mapping is identical to ``F.adaptive_avg_pool2d`` with output size
    ``(target_h, target_w)``, which partitions the source grid into
    ``target_h × target_w`` non-overlapping bins of size
    ``floor(source/target)`` or ``ceil(source/target)`` and averages within
    each bin.  This is the same spatial grouping that SVA's learnable queries
    implicitly cover.

    Parameters
    ----------
    positions : Tensor[B, N, 3]
        Per-patch 3D positions on the source grid, where N = source_h * source_w.
    source_h, source_w : int
        Spatial dimensions of the source grid (37×37 for DINOv2).
    target_h, target_w : int
        Spatial dimensions of the SVA query grid (24×24 for SigLIP).

    Returns
    -------
    pooled : Tensor[B, target_h * target_w, 3]
        One representative 3D position per SVA query cell.
    """
    b, n, c = positions.shape
    if n != source_h * source_w:
        raise ValueError(
            f"positions token count ({n}) != source_h*source_w ({source_h * source_w})."
        )
    if c != 3:
        raise ValueError(f"positions last dim must be 3, got {c}.")

    # Reshape to spatial grid: [B, 3, source_h, source_w]
    spatial = positions.permute(0, 2, 1).reshape(b, 3, source_h, source_w)

    # Adaptive average pool — same bin assignment as SVA's spatial sub-regions
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        spatial, (target_h, target_w)
    )  # [B, 3, target_h, target_w]

    # Reshape back to token sequence: [B, target_h * target_w, 3]
    return pooled.reshape(b, 3, target_h * target_w).permute(0, 2, 1).contiguous()
