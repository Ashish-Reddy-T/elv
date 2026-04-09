"""Tests for the RoPE monkey-patch (IcosahedralRoPE3D injection into Qwen3-VL).

Tests the building blocks of the monkey-patch without requiring
a full Qwen3-VL model (which needs @pytest.mark.slow).
"""

import math

import torch

from spatialvlm.backbone.rope_patch import (
    SPATIAL_TOKEN_TYPE_ID,
    _build_icosahedral_cos_sin,
    _get_icosahedral_rope,
)


class TestBuildIcosahedralCosSin:
    """Tests for _build_icosahedral_cos_sin."""

    def test_output_shape(self):
        """cos/sin should be [B, N_spatial, head_dim]."""
        positions = torch.randn(2, 1369, 3)
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float32)
        assert cos.shape == (2, 1369, 128)
        assert sin.shape == (2, 1369, 128)

    def test_identity_padding(self):
        """Last 32 dims (16 pairs) should have cos=1, sin=0 for any position."""
        positions = torch.randn(1, 10, 3) * 5.0
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float32)

        # The layout after cat((freqs, freqs), dim=-1):
        # cos has shape [B, N, 128]. The first 64 dims are the cos values for pairs 0-63,
        # and the second 64 dims are the same cos values repeated.
        # Pairs 48-63 (indices 48:64 in first half, 112:128 in second half) are identity.
        # Check that identity pairs have cos=1
        cos_first_half_pad = cos[:, :, 48:64]  # 16 identity pairs in first half
        cos_second_half_pad = cos[:, :, 112:128]  # same 16 pairs repeated
        assert torch.allclose(cos_first_half_pad, torch.ones_like(cos_first_half_pad), atol=1e-6), (
            "Padded cos dims should be 1.0"
        )
        assert torch.allclose(cos_second_half_pad, torch.ones_like(cos_second_half_pad), atol=1e-6)

        # Check that identity pairs have sin=0
        sin_first_half_pad = sin[:, :, 48:64]
        sin_second_half_pad = sin[:, :, 112:128]
        assert torch.allclose(
            sin_first_half_pad, torch.zeros_like(sin_first_half_pad), atol=1e-6
        ), "Padded sin dims should be 0.0"
        assert torch.allclose(sin_second_half_pad, torch.zeros_like(sin_second_half_pad), atol=1e-6)

    def test_zero_position_identity(self):
        """At origin (0,0,0), ALL icosahedral projections are 0 -> cos=1, sin=0 everywhere."""
        positions = torch.zeros(1, 1, 3)
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float32)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)
        assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-6)

    def test_nonzero_position_differs(self):
        """Non-origin position should produce non-trivial cos/sin."""
        positions = torch.tensor([[[1.0, 2.0, 3.0]]])
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float32)
        # First 96 dims (48 pairs) should NOT be all 1/0
        assert not torch.allclose(cos[:, :, :48], torch.ones(1, 1, 48), atol=0.1)
        assert not torch.allclose(sin[:, :, :48], torch.zeros(1, 1, 48), atol=0.1)

    def test_output_bounded(self):
        """cos/sin values must be in [-1, 1]."""
        positions = torch.randn(4, 100, 3) * 10.0
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float32)
        assert cos.abs().max() <= 1.0 + 1e-5
        assert sin.abs().max() <= 1.0 + 1e-5

    def test_dtype_propagation(self):
        """Output dtype should match requested dtype."""
        positions = torch.randn(1, 5, 3)
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float16)
        assert cos.dtype == torch.float16
        assert sin.dtype == torch.float16

    def test_batch_independence(self):
        """Each batch element should produce independent cos/sin."""
        positions = torch.randn(3, 10, 3)
        cos, sin = _build_icosahedral_cos_sin(positions, head_dim=128, dtype=torch.float32)
        for i in range(3):
            cos_i, sin_i = _build_icosahedral_cos_sin(
                positions[i : i + 1], head_dim=128, dtype=torch.float32
            )
            assert torch.allclose(cos[i : i + 1], cos_i, atol=1e-4)
            assert torch.allclose(sin[i : i + 1], sin_i, atol=1e-4)


class TestIcosahedralRopeSingleton:
    """Tests for the IcosahedralRoPE3D singleton."""

    def test_singleton_consistency(self):
        """_get_icosahedral_rope should return the same module each call."""
        device = torch.device("cpu")
        rope1 = _get_icosahedral_rope(device)
        rope2 = _get_icosahedral_rope(device)
        assert rope1 is rope2

    def test_correct_dims(self):
        """Singleton should have 6 directions and 8 frequencies."""
        rope = _get_icosahedral_rope(torch.device("cpu"))
        assert rope.directions.shape == (6, 3)
        assert rope.freqs.shape == (8,)


class TestSpatialTokenTypeId:
    def test_value(self):
        """Spatial token type ID should be 3 (after text=0, image=1, video=2)."""
        assert SPATIAL_TOKEN_TYPE_ID == 3


class TestGenerationMode:
    """Tests for generation-mode behavior of the RoPE patch."""

    def test_no_stashed_data_returns_unchanged(self):
        """When no spatial data is stashed, patch_rope_forward returns original cos/sin."""
        from spatialvlm.backbone.rope_patch import patch_rope_forward

        # Mock rotary_emb with no stashed data
        class MockRotaryEmb:
            _spatial_coords_3d = None
            _spatial_token_mask = None

        cos_orig = torch.randn(1, 20, 128)
        sin_orig = torch.randn(1, 20, 128)

        def mock_original_forward(x, position_ids):
            return cos_orig, sin_orig

        rotary = MockRotaryEmb()
        cos_out, sin_out = patch_rope_forward(
            mock_original_forward, rotary, torch.randn(1, 20, 64), torch.zeros(3, 1, 20)
        )
        assert torch.equal(cos_out, cos_orig)
        assert torch.equal(sin_out, sin_orig)

    def test_empty_mask_returns_unchanged(self):
        """When spatial_token_mask is all False, cos/sin should be unchanged."""
        from spatialvlm.backbone.rope_patch import patch_rope_forward

        class MockRotaryEmb:
            _spatial_coords_3d = torch.randn(1, 0, 3)  # no spatial tokens
            _spatial_token_mask = torch.zeros(1, 20, dtype=torch.bool)

        cos_orig = torch.randn(1, 20, 128)
        sin_orig = torch.randn(1, 20, 128)

        def mock_original_forward(x, position_ids):
            return cos_orig.clone(), sin_orig.clone()

        rotary = MockRotaryEmb()
        cos_out, sin_out = patch_rope_forward(
            mock_original_forward, rotary, torch.randn(1, 20, 64), torch.zeros(3, 1, 20)
        )
        assert torch.allclose(cos_out, cos_orig)
        assert torch.allclose(sin_out, sin_orig)

    def test_stashed_data_cleared_after_use(self):
        """After patch_rope_forward consumes spatial data, stash should be cleared."""
        from spatialvlm.backbone.rope_patch import patch_rope_forward

        class MockRotaryEmb:
            _spatial_coords_3d = torch.randn(1, 5, 3)
            _spatial_token_mask = torch.zeros(1, 20, dtype=torch.bool)

        # Mark positions 10-14 as spatial
        MockRotaryEmb._spatial_token_mask[0, 10:15] = True

        cos_orig = torch.ones(1, 20, 128)
        sin_orig = torch.zeros(1, 20, 128)

        def mock_original_forward(x, position_ids):
            return cos_orig.clone(), sin_orig.clone()

        rotary = MockRotaryEmb()
        patch_rope_forward(
            mock_original_forward, rotary, torch.randn(1, 20, 64), torch.zeros(3, 1, 20)
        )
        # After consumption, stash should be cleared
        assert rotary._spatial_coords_3d is None
        assert rotary._spatial_token_mask is None


class TestConfigAlignment:
    """Verify alignment with GeometryConfig defaults."""

    def test_rope_dims_match_config(self):
        """IcosahedralRoPE3D output (96) should match config.rope3d_dims."""
        from spatialvlm.config.model import GeometryConfig

        cfg = GeometryConfig()
        assert cfg.rope3d_dims == 96

    def test_freq_ratio_is_e_one_third(self):
        """Config freq_ratio should be e^(1/3)."""
        from spatialvlm.config.model import GeometryConfig

        cfg = GeometryConfig()
        assert abs(cfg.freq_ratio - math.exp(1.0 / 3.0)) < 1e-10
