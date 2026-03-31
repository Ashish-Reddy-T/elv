"""Tests for Stage 4 position routing."""

import pytest
import torch

from spatialvlm.backbone.position_routing import PositionRouter


class TestPositionRouter:
    def test_init_requires_matching_rotary_dims(self):
        with pytest.raises(ValueError, match="must match M-RoPE rotary pairs"):
            _ = PositionRouter(mrope_section=[24, 20, 10], expected_spatial_rotary_dim=64)

    def test_build_text_mrope_position_ids_shape(self):
        router = PositionRouter(mrope_section=[24, 20, 20], expected_spatial_rotary_dim=64)
        pos = router.build_text_mrope_position_ids(batch_size=2, text_len=5)
        assert pos.shape == (2, 3, 5)
        # Temporal channel should be monotonic [0..T-1].
        assert torch.equal(pos[0, 0], torch.tensor([0, 1, 2, 3, 4]))
        assert torch.equal(pos[0, 1], torch.zeros(5, dtype=torch.long))
        assert torch.equal(pos[0, 2], torch.zeros(5, dtype=torch.long))

    def test_route_outputs_shapes(self):
        router = PositionRouter(mrope_section=[24, 20, 20], expected_spatial_rotary_dim=64)
        text = torch.randn(2, 7, 32)
        spatial = torch.randn(2, 4, 32)
        rope3d = torch.randn(2, 4, 64)

        routed = router.route(text, spatial, rope3d)
        assert routed.combined_tokens.shape == (2, 11, 32)
        assert routed.is_spatial_mask.shape == (2, 11)
        assert routed.text_mrope_position_ids.shape == (2, 3, 7)
        assert routed.spatial_rope3d.shape == (2, 4, 64)
        assert routed.is_spatial_mask[:, :7].sum().item() == 0
        assert routed.is_spatial_mask[:, 7:].all()

    def test_route_promotes_2d_text_position_ids(self):
        router = PositionRouter(mrope_section=[24, 20, 20], expected_spatial_rotary_dim=64)
        text = torch.randn(1, 3, 16)
        spatial = torch.randn(1, 2, 16)
        rope3d = torch.randn(1, 2, 64)
        ids_2d = torch.tensor([[10, 11, 12]], dtype=torch.long)

        routed = router.route(text, spatial, rope3d, text_mrope_position_ids=ids_2d)
        assert routed.text_mrope_position_ids.shape == (1, 3, 3)
        assert torch.equal(routed.text_mrope_position_ids[0, 0], ids_2d[0])
        assert torch.equal(routed.text_mrope_position_ids[0, 1], torch.zeros(3, dtype=torch.long))
        assert torch.equal(routed.text_mrope_position_ids[0, 2], torch.zeros(3, dtype=torch.long))

    def test_route_rejects_bad_spatial_rope_shape(self):
        router = PositionRouter(mrope_section=[24, 20, 20], expected_spatial_rotary_dim=64)
        text = torch.randn(1, 3, 16)
        spatial = torch.randn(1, 2, 16)
        bad_rope3d = torch.randn(1, 2, 32)
        with pytest.raises(ValueError, match="spatial_rope3d must be"):
            _ = router.route(text, spatial, bad_rope3d)
