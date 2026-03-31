"""Unit tests for camera intrinsics and backprojection utilities."""

import torch
import pytest

from spatialvlm.utils.camera import CameraIntrinsics, backproject_pixel, make_pixel_grid


class TestMakePixelGrid:
    def test_shape(self):
        grid = make_pixel_grid(width=518, height=518)
        assert grid.shape == (518, 518, 2)

    def test_shape_rectangular(self):
        grid = make_pixel_grid(width=640, height=480)
        assert grid.shape == (480, 640, 2)

    def test_u_axis_is_columns(self):
        # u (col) should increment along dim=1 (W axis)
        grid = make_pixel_grid(width=4, height=3)
        # grid[row, col, 0] = u = col
        for col in range(4):
            assert (grid[:, col, 0] == col).all(), f"u should equal col index at col={col}"

    def test_v_axis_is_rows(self):
        # v (row) should increment along dim=0 (H axis)
        grid = make_pixel_grid(width=4, height=3)
        # grid[row, col, 1] = v = row
        for row in range(3):
            assert (grid[row, :, 1] == row).all(), f"v should equal row index at row={row}"

    def test_device_propagation(self):
        grid = make_pixel_grid(width=8, height=8, device=torch.device("cpu"))
        assert grid.device.type == "cpu"

    def test_dtype(self):
        grid = make_pixel_grid(width=8, height=8, dtype=torch.float32)
        assert grid.dtype == torch.float32


class TestBackprojectPixel:
    """Test the pinhole camera backprojection math."""

    def test_center_pixel_identity_intrinsics(self):
        """Center pixel with unit focal length → (0, 0, d)."""
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=4.0, cy=4.0, width=8, height=8)
        u = torch.tensor(4.0)   # center col
        v = torch.tensor(4.0)   # center row
        d = torch.tensor(5.0)

        pt = backproject_pixel(u, v, d, intrinsics)
        assert pt.shape == (3,)
        assert torch.allclose(pt, torch.tensor([0.0, 0.0, 5.0]))

    def test_known_projection(self):
        """Verify X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d."""
        intrinsics = CameraIntrinsics(fx=2.0, fy=4.0, cx=10.0, cy=20.0, width=100, height=100)
        u = torch.tensor(14.0)  # u - cx = 4.0
        v = torch.tensor(28.0)  # v - cy = 8.0
        d = torch.tensor(3.0)

        pt = backproject_pixel(u, v, d, intrinsics)
        expected = torch.tensor([
            (14.0 - 10.0) * 3.0 / 2.0,   # X = 6.0
            (28.0 - 20.0) * 3.0 / 4.0,   # Y = 6.0
            3.0,                           # Z = 3.0
        ])
        assert torch.allclose(pt, expected)

    def test_batch_shape(self):
        """Backprojection should broadcast over arbitrary batch dims."""
        intrinsics = CameraIntrinsics(fx=320.0, fy=320.0, cx=259.0, cy=259.0, width=518, height=518)
        B, H, W = 2, 518, 518
        grid = make_pixel_grid(W, H)
        u = grid[..., 0].unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        v = grid[..., 1].unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        depth = torch.ones(B, H, W) * 2.0

        pts = backproject_pixel(u, v, depth, intrinsics)
        assert pts.shape == (B, H, W, 3)

    def test_zero_depth_gives_zero_xyz(self):
        """Zero depth → all-zero 3D point (not visible)."""
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=4, height=4)
        u = torch.tensor(2.0)
        v = torch.tensor(3.0)
        d = torch.tensor(0.0)
        pt = backproject_pixel(u, v, d, intrinsics)
        assert torch.allclose(pt, torch.zeros(3))

    def test_depth_z_equals_input(self):
        """Z coordinate must equal the input depth value."""
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)
        depths = torch.tensor([0.5, 1.0, 2.5, 10.0])
        u = torch.zeros_like(depths) + 50.0
        v = torch.zeros_like(depths) + 50.0
        pts = backproject_pixel(u, v, depths, intrinsics)
        assert torch.allclose(pts[..., 2], depths)
