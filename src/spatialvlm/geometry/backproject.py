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

    return patch_points  # [B, n_patches, 3]
