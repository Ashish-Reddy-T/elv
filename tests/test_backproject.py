"""Unit tests for depth backprojection and patch-level aggregation."""

import torch
import pytest

from spatialvlm.utils.camera import CameraIntrinsics
from spatialvlm.geometry.backproject import (
    aggregate_patches_percentile,
    backproject_depth_map,
)

# Default Habitat-like intrinsics for testing
INTRINSICS = CameraIntrinsics(fx=320.0, fy=320.0, cx=259.0, cy=259.0, width=518, height=518)


class TestBackprojectDepthMap:
    def test_output_shape(self):
        depth = torch.ones(2, 518, 518)
        pts = backproject_depth_map(depth, INTRINSICS)
        assert pts.shape == (2, 518, 518, 3), f"Expected [2,518,518,3], got {pts.shape}"

    def test_z_equals_depth(self):
        """Z coordinate must equal the input depth everywhere."""
        depth = torch.rand(1, 518, 518) + 0.1  # avoid zero
        pts = backproject_depth_map(depth, INTRINSICS)
        assert torch.allclose(pts[..., 2], depth, atol=1e-5)

    def test_zero_depth_gives_zero_point(self):
        """Zero-depth pixels should produce the zero 3D point."""
        depth = torch.zeros(1, 518, 518)
        pts = backproject_depth_map(depth, INTRINSICS)
        assert torch.all(pts == 0.0)

    def test_principal_point_pixel_has_zero_xy(self):
        """The pixel at (cx, cy) projects to X=0, Y=0 regardless of depth."""
        depth = torch.ones(1, 518, 518) * 3.0
        pts = backproject_depth_map(depth, INTRINSICS)
        # cx=259, cy=259 → row=259, col=259
        cx, cy = int(INTRINSICS.cx), int(INTRINSICS.cy)
        xy = pts[0, cy, cx, :2]  # [X, Y]
        assert torch.allclose(xy, torch.zeros(2), atol=1e-4), f"Expected (0,0), got {xy}"

    def test_device_consistency(self):
        depth = torch.ones(1, 518, 518, device="cpu")
        pts = backproject_depth_map(depth, INTRINSICS)
        assert pts.device.type == "cpu"

    def test_batch_independent(self):
        """Same depth map in two batch slots should produce identical points."""
        single = torch.rand(1, 518, 518) + 0.5
        batch = single.expand(3, -1, -1)
        pts = backproject_depth_map(batch, INTRINSICS)
        assert torch.allclose(pts[0], pts[1]) and torch.allclose(pts[1], pts[2])


class TestAggregatePatchesPercentile:
    def test_output_shape_518x518(self):
        """Standard DINOv2 input: 518×518, patch_size=14 → 1369 patches."""
        depth = torch.ones(2, 518, 518)
        pts = backproject_depth_map(depth, INTRINSICS)
        patches = aggregate_patches_percentile(pts, depth, patch_size=14)
        assert patches.shape == (2, 1369, 3), f"Expected [2,1369,3], got {patches.shape}"

    def test_output_shape_arbitrary(self):
        """Test with smaller image: 28×28, patch_size=14 → 4 patches."""
        cam = CameraIntrinsics(fx=10.0, fy=10.0, cx=14.0, cy=14.0, width=28, height=28)
        depth = torch.ones(1, 28, 28) * 2.0
        pts = backproject_depth_map(depth, cam)
        patches = aggregate_patches_percentile(pts, depth, patch_size=14)
        assert patches.shape == (1, 4, 3)

    def test_constant_depth_plane(self):
        """All pixels at depth d → all patch z-coords should equal d."""
        d = 3.5
        depth = torch.ones(1, 518, 518) * d
        pts = backproject_depth_map(depth, INTRINSICS)
        patches = aggregate_patches_percentile(pts, depth, patch_size=14)
        assert torch.allclose(patches[..., 2], torch.tensor(d), atol=1e-4), (
            f"Z-coords should all be {d}, got range "
            f"[{patches[..., 2].min():.4f}, {patches[..., 2].max():.4f}]"
        )

    def test_zero_depth_gives_zero_patch(self):
        """When all pixels are zero-depth (invalid), patch point should be (0,0,0)."""
        depth = torch.zeros(1, 518, 518)
        pts = backproject_depth_map(depth, INTRINSICS)
        patches = aggregate_patches_percentile(pts, depth, patch_size=14)
        assert torch.all(patches == 0.0)

    def test_nan_depth_no_crash(self):
        """NaN pixels should not cause a crash; they sort to the end (farther than valid)."""
        depth = torch.ones(1, 518, 518) * 2.0
        # Inject NaNs into the first few pixels of each patch
        depth[0, ::14, ::14] = float("nan")
        pts = backproject_depth_map(depth, INTRINSICS)
        # Should not raise
        patches = aggregate_patches_percentile(pts, depth, patch_size=14)
        assert patches.shape == (1, 1369, 3)
        # Valid patches (those without all-NaN) should have finite values
        valid_mask = ~torch.isnan(patches).any(dim=-1)
        assert valid_mask.sum() > 0

    def test_percentile_selects_nearest(self):
        """Lower percentile → selects pixels closer to camera (smaller Z)."""
        depth = torch.ones(1, 518, 518) * 5.0
        # Set first pixel of first patch to a closer depth
        depth[0, 0, 0] = 1.0  # very close
        pts = backproject_depth_map(depth, INTRINSICS)

        # 15th percentile: with 196 pixels, rank 0 is the closest
        # 15th pct → index ~29. Since only 1 pixel is close, 15th pct picks
        # the close pixel if it's at rank < 0.15*196=29 → yes (rank 0)
        patches_15 = aggregate_patches_percentile(pts, depth, patch_size=14, percentile=0.0)
        patches_75 = aggregate_patches_percentile(pts, depth, patch_size=14, percentile=0.75)

        # First patch: 0th percentile should pick the close pixel (Z≈1.0)
        # 75th percentile should pick a farther pixel (Z≈5.0)
        z_close = patches_15[0, 0, 2]
        z_far = patches_75[0, 0, 2]
        assert z_close < z_far, (
            f"0th-pct Z={z_close:.3f} should be < 75th-pct Z={z_far:.3f}"
        )

    def test_percentile_rank_correctness(self):
        """Synthetic patch: verify the selected pixel matches expected rank."""
        # Single-batch, 1-patch image: 14×14 pixels all at unique depths
        cam = CameraIntrinsics(fx=10.0, fy=10.0, cx=7.0, cy=7.0, width=14, height=14)
        # Depths: 1.0, 2.0, ..., 196.0 arranged in row-major order
        depth_vals = torch.arange(1, 197, dtype=torch.float32).view(1, 14, 14)
        pts = backproject_depth_map(depth_vals, cam)

        # 15th percentile of 196 pixels → rank k = int(0.15 * 196) = 29
        # Depth at rank 29 (0-indexed) = 30.0 (since depths are 1..196)
        patches = aggregate_patches_percentile(pts, depth_vals, patch_size=14, percentile=0.15)
        k = int(0.15 * 196)
        expected_z = float(k + 1)  # depths are 1-indexed
        assert torch.allclose(patches[0, 0, 2], torch.tensor(expected_z), atol=1e-4), (
            f"Expected Z={expected_z} at rank {k}, got {patches[0, 0, 2]:.4f}"
        )
