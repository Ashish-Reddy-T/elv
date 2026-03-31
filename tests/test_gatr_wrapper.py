"""Tests for Stage 2 GATr wrapper."""

import math

import pytest
import torch

from spatialvlm.geometry.gatr_wrapper import GATrWrapper


@pytest.fixture
def gatr_wrapper():
    return GATrWrapper(
        num_blocks=1,
        gatr_mv_channels=16,
        gatr_s_channels=32,
        projector_out_dim=4096,
        checkpoint_blocks=True,
        disable_cached_einsum=True,
        device=torch.device("cpu"),
    )


class TestGATrWrapperShapes:
    def test_standard_shape(self, gatr_wrapper):
        points = torch.randn(1, 1369, 3)
        projected, invariants = gatr_wrapper(points, return_invariants=True)
        assert invariants.shape == (1, 1369, 48), (
            f"Expected invariants [1,1369,48], got {invariants.shape}"
        )
        assert projected.shape == (1, 1369, 4096), (
            f"Expected projected [1,1369,4096], got {projected.shape}"
        )

    def test_output_device_matches_input(self, gatr_wrapper):
        points = torch.randn(2, 64, 3)
        projected = gatr_wrapper(points)
        assert projected.device == points.device

    def test_bad_input_shape_raises(self, gatr_wrapper):
        with pytest.raises(ValueError, match="Expected points_3d shape"):
            _ = gatr_wrapper(torch.randn(2, 64, 4))


class TestGATrWrapperNumerics:
    def test_outputs_are_finite_for_edge_values(self, gatr_wrapper):
        points = torch.tensor(
            [[[0.0, 0.0, 0.0], [1e6, -1e6, 5e5], [1e-9, -1e-9, 0.0], [2.0, 3.0, 4.0]]],
            dtype=torch.float32,
        )
        projected, invariants = gatr_wrapper(points, return_invariants=True)
        assert torch.isfinite(projected).all()
        assert torch.isfinite(invariants).all()

    def test_eval_mode_deterministic(self, gatr_wrapper):
        gatr_wrapper.eval()
        points = torch.randn(1, 64, 3)
        with torch.no_grad():
            out1 = gatr_wrapper(points)
            out2 = gatr_wrapper(points)
        assert torch.allclose(out1, out2)


class TestGATrWrapperArchitecture:
    def test_improved_pga_path_is_present(self, gatr_wrapper):
        from gatr.layers.mlp.geometric_bilinears import GeometricBilinear

        assert gatr_wrapper.uses_improved_pga()
        for block in gatr_wrapper.gatr.blocks:
            assert isinstance(block.mlp.layers[0], GeometricBilinear)

    def test_invariant_features_rotation_invariant(self, gatr_wrapper):
        """Rotate inputs around Z-axis and verify invariant features remain unchanged."""
        theta = 0.7
        c = math.cos(theta)
        s = math.sin(theta)
        rot_z = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        points = torch.randn(1, 64, 3)
        rotated_points = points @ rot_z.T

        gatr_wrapper.eval()
        with torch.no_grad():
            _, inv_a = gatr_wrapper(points, return_invariants=True)
            _, inv_b = gatr_wrapper(rotated_points, return_invariants=True)

        assert torch.allclose(inv_a, inv_b, atol=3e-3, rtol=3e-3)
