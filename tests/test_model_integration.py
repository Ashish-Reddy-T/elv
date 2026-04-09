"""Tests for the integrated SpatialVLM model.

These tests verify the wiring between stages without loading real pretrained
models. Each encoder/backbone is mocked to return tensors of the correct shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from spatialvlm.config.model import SpatialVLMConfig
from spatialvlm.model import SpatialVLM
from spatialvlm.utils.camera import CameraIntrinsics


class TestSpatialVLMInit:
    """Test that SpatialVLM initializes with default config."""

    def test_default_config(self):
        """SpatialVLM should accept default SpatialVLMConfig."""
        # Patch encoders and backbone to avoid loading real models
        with (
            patch("spatialvlm.model.SigLIP2Encoder") as mock_siglip,
            patch("spatialvlm.model.DINOv2Encoder") as mock_dinov2,
            patch("spatialvlm.model.GATrWrapper") as mock_gatr,
            patch("spatialvlm.model.Qwen3VLBackbone") as mock_backbone,
        ):
            # Configure mock encoder properties
            mock_siglip_inst = MagicMock()
            mock_siglip_inst.out_dim = 3456
            mock_siglip.return_value = mock_siglip_inst

            mock_dinov2_inst = MagicMock()
            mock_dinov2_inst.out_dim = 3072
            mock_dinov2.return_value = mock_dinov2_inst

            mock_gatr.return_value = MagicMock()
            mock_backbone.return_value = MagicMock()

            model = SpatialVLM(lazy_load_encoders=True, lazy_load_backbone=True)
            assert model.config is not None
            assert model.config.fusion.sva_num_queries == 1369

    def test_custom_config(self):
        """SpatialVLM should accept custom config overrides."""
        with (
            patch("spatialvlm.model.SigLIP2Encoder") as mock_siglip,
            patch("spatialvlm.model.DINOv2Encoder") as mock_dinov2,
            patch("spatialvlm.model.GATrWrapper") as mock_gatr,
            patch("spatialvlm.model.Qwen3VLBackbone") as mock_backbone,
        ):
            mock_siglip_inst = MagicMock()
            mock_siglip_inst.out_dim = 3456
            mock_siglip.return_value = mock_siglip_inst

            mock_dinov2_inst = MagicMock()
            mock_dinov2_inst.out_dim = 3072
            mock_dinov2.return_value = mock_dinov2_inst

            mock_gatr.return_value = MagicMock()
            mock_backbone.return_value = MagicMock()

            config = SpatialVLMConfig()
            config.backbone.lora_rank = 16
            model = SpatialVLM(config=config, lazy_load_encoders=True, lazy_load_backbone=True)
            assert model.config.backbone.lora_rank == 16


class TestBuildDeepstackInputs:
    """Test the DeepStack input construction."""

    def test_deepstack_kwargs_shape(self):
        """build_deepstack_inputs should produce correctly shaped tensors."""
        with (
            patch("spatialvlm.model.SigLIP2Encoder") as mock_siglip,
            patch("spatialvlm.model.DINOv2Encoder") as mock_dinov2,
            patch("spatialvlm.model.GATrWrapper") as mock_gatr,
            patch("spatialvlm.model.Qwen3VLBackbone") as mock_backbone,
        ):
            mock_siglip_inst = MagicMock()
            mock_siglip_inst.out_dim = 3456
            mock_siglip.return_value = mock_siglip_inst

            mock_dinov2_inst = MagicMock()
            mock_dinov2_inst.out_dim = 3072
            mock_dinov2.return_value = mock_dinov2_inst

            mock_gatr.return_value = MagicMock()
            mock_backbone.return_value = MagicMock()

            model = SpatialVLM(lazy_load_encoders=True, lazy_load_backbone=True)

        bsz, seq_len, n_spatial, hidden = 2, 2048, 1369, 4096
        fused = torch.randn(bsz, n_spatial, hidden)
        positions = torch.randn(bsz, n_spatial, 3)
        input_ids = torch.zeros(bsz, seq_len, dtype=torch.long)
        start_idx = 100

        kwargs = model.build_deepstack_inputs(fused, positions, input_ids, start_idx)

        assert kwargs["spatial_coords_3d"].shape == (bsz, n_spatial, 3)
        assert kwargs["spatial_token_mask"].shape == (bsz, seq_len)
        assert kwargs["deepstack_visual_embeds"].shape == (bsz, n_spatial, hidden)

        # Verify mask is True only at spatial positions
        mask = kwargs["spatial_token_mask"]
        assert mask[:, start_idx : start_idx + n_spatial].all()
        assert not mask[:, :start_idx].any()
        assert not mask[:, start_idx + n_spatial :].any()

    def test_spatial_mask_dtype(self):
        """Spatial token mask should be boolean."""
        with (
            patch("spatialvlm.model.SigLIP2Encoder") as mock_siglip,
            patch("spatialvlm.model.DINOv2Encoder") as mock_dinov2,
            patch("spatialvlm.model.GATrWrapper") as mock_gatr,
            patch("spatialvlm.model.Qwen3VLBackbone") as mock_backbone,
        ):
            mock_siglip_inst = MagicMock()
            mock_siglip_inst.out_dim = 3456
            mock_siglip.return_value = mock_siglip_inst

            mock_dinov2_inst = MagicMock()
            mock_dinov2_inst.out_dim = 3072
            mock_dinov2.return_value = mock_dinov2_inst

            mock_gatr.return_value = MagicMock()
            mock_backbone.return_value = MagicMock()

            model = SpatialVLM(lazy_load_encoders=True, lazy_load_backbone=True)

        fused = torch.randn(1, 1369, 4096)
        positions = torch.randn(1, 1369, 3)
        input_ids = torch.zeros(1, 2048, dtype=torch.long)

        kwargs = model.build_deepstack_inputs(fused, positions, input_ids, 0)
        assert kwargs["spatial_token_mask"].dtype == torch.bool


class TestFuseMethod:
    """Test the fusion stage (SVA + norm matching)."""

    def test_fuse_output_shape(self):
        """fuse() should return [B, 1369, 4096]."""
        with (
            patch("spatialvlm.model.SigLIP2Encoder") as mock_siglip,
            patch("spatialvlm.model.DINOv2Encoder") as mock_dinov2,
            patch("spatialvlm.model.GATrWrapper") as mock_gatr,
            patch("spatialvlm.model.Qwen3VLBackbone") as mock_backbone,
        ):
            mock_siglip_inst = MagicMock()
            mock_siglip_inst.out_dim = 3456
            mock_siglip.return_value = mock_siglip_inst

            mock_dinov2_inst = MagicMock()
            mock_dinov2_inst.out_dim = 3072
            mock_dinov2.return_value = mock_dinov2_inst

            mock_gatr.return_value = MagicMock()
            mock_backbone.return_value = MagicMock()

            model = SpatialVLM(lazy_load_encoders=True, lazy_load_backbone=True)

        bsz, hidden = 2, 4096
        siglip = torch.randn(bsz, 576, hidden)
        dinov2 = torch.randn(bsz, 1369, hidden)
        gatr = torch.randn(bsz, 1369, hidden)

        fused = model.fuse(siglip, dinov2, gatr)
        assert fused.shape == (bsz, 1369, hidden)


class TestExports:
    """Test that SpatialVLM is importable from the package."""

    def test_import_from_package(self):
        from spatialvlm import SpatialVLM, SpatialVLMConfig

        assert SpatialVLM is not None
        assert SpatialVLMConfig is not None
