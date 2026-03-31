"""Tests for Spatial Vision Aggregator (SVA)."""

import torch

from spatialvlm.fusion.sva import SpatialVisionAggregator


class TestSVA:
    def test_output_shape_with_full_kv_count(self):
        """SVA should map [B,576,D] queries over 3314 KV tokens to [B,576,D]."""
        model = SpatialVisionAggregator(
            hidden_dim=128,
            num_queries=576,
            num_layers=2,
            num_heads=8,
            use_typed_attention_bias=True,
        )
        siglip = torch.randn(1, 576, 128)
        dinov2 = torch.randn(1, 1369, 128)
        gatr = torch.randn(1, 1369, 128)
        out = model(siglip, dinov2, gatr)
        assert out.shape == (1, 576, 128)

    def test_typed_attention_bias_matrix_shape(self):
        """Each SVA layer should hold a learned 3x3 typed bias matrix."""
        model = SpatialVisionAggregator(
            hidden_dim=64,
            num_queries=576,
            num_layers=2,
            num_heads=8,
            use_typed_attention_bias=True,
        )
        for layer in model.layers:
            assert layer.typed_attention_bias is not None
            assert layer.typed_attention_bias.shape == (3, 3)

    def test_no_typed_bias_when_disabled(self):
        model = SpatialVisionAggregator(
            hidden_dim=64,
            num_queries=576,
            num_layers=1,
            num_heads=8,
            use_typed_attention_bias=False,
        )
        assert model.layers[0].typed_attention_bias is None

    def test_attention_mask_builder_shape(self):
        model = SpatialVisionAggregator(
            hidden_dim=32,
            num_queries=8,
            num_layers=1,
            num_heads=8,
            use_typed_attention_bias=True,
        )
        layer = model.layers[0]
        q_types = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0], dtype=torch.long)
        kv_types = torch.tensor([0, 0, 1, 2, 1, 2, 2], dtype=torch.long)
        mask = layer.build_typed_attention_mask(q_types, kv_types)
        assert mask is not None
        assert mask.shape == (1, 1, 8, 7)

    def test_padding_mask_accepts_boolean_valid_mask(self):
        model = SpatialVisionAggregator(
            hidden_dim=64,
            num_queries=576,
            num_layers=1,
            num_heads=8,
            use_typed_attention_bias=True,
        )
        siglip = torch.randn(1, 576, 64)
        dinov2 = torch.randn(1, 1369, 64)
        gatr = torch.randn(1, 1369, 64)
        kv_mask = torch.ones(1, 3314, dtype=torch.bool)
        kv_mask[:, -50:] = False
        out = model(siglip, dinov2, gatr, kv_padding_mask=kv_mask)
        assert out.shape == (1, 576, 64)

    def test_bad_query_shape_raises(self):
        model = SpatialVisionAggregator(hidden_dim=32, num_queries=8, num_layers=1, num_heads=8)
        siglip = torch.randn(1, 8, 32)
        dinov2 = torch.randn(1, 5, 32)
        gatr = torch.randn(1, 5, 32)
        bad_queries = torch.randn(1, 7, 32)
        try:
            _ = model(siglip, dinov2, gatr, queries=bad_queries)
            assert False, "Expected ValueError for bad query shape"
        except ValueError as exc:
            assert "queries must be" in str(exc)
