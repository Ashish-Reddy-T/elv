"""Tests for MLPProjector (fast) and vision encoders (slow — require model downloads).

Fast tests (no model loading):
  - MLPProjector shape + trainability

Slow tests (marked @pytest.mark.slow, require HuggingFace downloads):
  - SigLIP2Encoder: output shape, frozen params, CLS handling
  - DINOv2Encoder: output shape, frozen params, CLS stripping
"""

import pytest
import torch

from spatialvlm.encoders.projector import MLPProjector

# ──────────────────────────────────────────────────────────────────────────────
# MLPProjector — fast tests, no model deps
# ──────────────────────────────────────────────────────────────────────────────


class TestMLPProjector:
    def test_siglip_shape(self):
        """SigLIP projector: [B, 576, 3456] → [B, 576, 4096]."""
        proj = MLPProjector(in_dim=3456, out_dim=4096)
        x = torch.randn(2, 576, 3456)
        out = proj(x)
        assert out.shape == (2, 576, 4096), f"Expected (2,576,4096), got {out.shape}"

    def test_dinov2_shape(self):
        """DINOv2 projector: [B, 1369, 3072] → [B, 1369, 4096]."""
        proj = MLPProjector(in_dim=3072, out_dim=4096)
        x = torch.randn(2, 1369, 3072)
        out = proj(x)
        assert out.shape == (2, 1369, 4096), f"Expected (2,1369,4096), got {out.shape}"

    def test_gatr_shape(self):
        """GATr projector: [B, 1369, 48] → [B, 1369, 4096]."""
        proj = MLPProjector(in_dim=48, out_dim=4096)
        x = torch.randn(1, 1369, 48)
        out = proj(x)
        assert out.shape == (1, 1369, 4096)

    def test_custom_hidden_dim(self):
        """Custom hidden_dim should be used."""
        proj = MLPProjector(in_dim=64, out_dim=128, hidden_dim=256)
        x = torch.randn(1, 10, 64)
        out = proj(x)
        assert out.shape == (1, 10, 128)

    def test_default_hidden_equals_out(self):
        """Default hidden_dim = out_dim."""
        proj = MLPProjector(in_dim=32, out_dim=64)
        # net[0] is Linear(32, 64), net[2] is Linear(64, 64)
        assert proj.net[0].out_features == 64
        assert proj.net[2].out_features == 64

    def test_all_params_trainable(self):
        """All MLPProjector parameters must require grad."""
        proj = MLPProjector(in_dim=32, out_dim=64)
        for name, param in proj.named_parameters():
            assert param.requires_grad, f"Parameter {name} should require grad"

    def test_param_count_reasonable(self):
        """Verify approximate parameter count for SigLIP projector."""
        # Linear(3456, 4096) + Linear(4096, 4096) ≈ 30.8M
        proj = MLPProjector(in_dim=3456, out_dim=4096)
        n_params = sum(p.numel() for p in proj.parameters())
        assert 25_000_000 < n_params < 40_000_000, (
            f"SigLIP projector params {n_params:,} outside expected range"
        )

    def test_gradient_flows(self):
        """Gradients should flow through the projector."""
        proj = MLPProjector(in_dim=16, out_dim=8)
        x = torch.randn(1, 4, 16, requires_grad=True)
        out = proj(x).sum()
        out.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# ──────────────────────────────────────────────────────────────────────────────
# SigLIP2Encoder — slow tests (require HuggingFace download)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSigLIP2Encoder:
    """Requires: google/siglip2-so400m-patch16-naflex download (~400MB)."""

    @pytest.fixture(scope="class")
    def encoder(self):
        from spatialvlm.encoders.siglip import SigLIP2Encoder
        return SigLIP2Encoder(device=torch.device("cpu"))

    def test_output_shape(self, encoder):
        """Standard input: [B, 3, 384, 384] → [B, 576, 3456]."""
        x = torch.randn(1, 3, 384, 384)
        out = encoder(x)
        # n_patches and out_dim are introspected from config at init
        assert out.shape[0] == 1
        assert out.shape[1] == encoder.n_patches
        assert out.shape[2] == encoder.out_dim
        # For default config: 576 patches, 3456 = 3 × 1152
        assert out.shape == (1, 576, 3456), (
            f"Expected [1, 576, 3456] but got {out.shape}. "
            "If shape differs, model config has different hidden_size or patch_size — "
            "update EncoderConfig accordingly."
        )

    def test_batch_shape(self, encoder):
        x = torch.randn(2, 3, 384, 384)
        out = encoder(x)
        assert out.shape[0] == 2

    def test_all_params_frozen(self, encoder):
        """Every parameter in the underlying SigLIP2 model must be frozen."""
        for name, param in encoder._model.named_parameters():
            assert not param.requires_grad, (
                f"Parameter {name} should be frozen (requires_grad=False)"
            )

    def test_no_cls_token_in_output(self, encoder):
        """Output must have exactly n_patches tokens (no CLS token)."""
        x = torch.randn(1, 3, 384, 384)
        out = encoder(x)
        assert out.shape[1] == encoder.n_patches

    def test_output_is_deterministic(self, encoder):
        x = torch.randn(1, 3, 384, 384)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_extract_layer_count(self, encoder):
        """Should extract from exactly 3 layers by default."""
        assert len(encoder._extract_layers) == 3

    def test_patch_count_matches_config(self, encoder):
        """n_patches = (image_size // patch_size)² — verified from loaded config."""
        expected = (encoder._image_size // encoder._patch_size) ** 2
        assert encoder.n_patches == expected


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2Encoder — slow tests (require HuggingFace download)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDINOv2Encoder:
    """Requires: facebook/dinov2-large download (~300MB)."""

    @pytest.fixture(scope="class")
    def encoder(self):
        from spatialvlm.encoders.dinov2 import DINOv2Encoder
        return DINOv2Encoder(device=torch.device("cpu"))

    def test_output_shape(self, encoder):
        """Standard input: [B, 3, 518, 518] → [B, 1369, 3072]."""
        x = torch.randn(1, 3, 518, 518)
        out = encoder(x)
        assert out.shape[0] == 1
        assert out.shape[1] == encoder.n_patches
        assert out.shape[2] == encoder.out_dim
        assert out.shape == (1, 1369, 3072), (
            f"Expected [1, 1369, 3072] but got {out.shape}. "
            "If shape differs, model config differs from expected — "
            "update EncoderConfig accordingly."
        )

    def test_batch_shape(self, encoder):
        x = torch.randn(2, 3, 518, 518)
        out = encoder(x)
        assert out.shape[0] == 2

    def test_all_params_frozen(self, encoder):
        """Every parameter in the underlying DINOv2 model must be frozen."""
        for name, param in encoder._model.named_parameters():
            assert not param.requires_grad, (
                f"Parameter {name} should be frozen"
            )

    def test_cls_token_stripped(self, encoder):
        """DINOv2 prepends a CLS token — output must NOT include it."""
        x = torch.randn(1, 3, 518, 518)
        out = encoder(x)
        # If CLS were included, shape[1] would be n_patches + 1 = 1370
        assert out.shape[1] == encoder.n_patches, (
            f"CLS token not stripped: got {out.shape[1]} tokens, "
            f"expected {encoder.n_patches}"
        )

    def test_output_is_deterministic(self, encoder):
        x = torch.randn(1, 3, 518, 518)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_patch_count_1369(self, encoder):
        """For 518×518 input and patch_size=14: exactly 37×37=1369 patches."""
        assert encoder.n_patches == 1369, (
            f"Expected 1369 patches (37×37), got {encoder.n_patches}. "
            "Check that DINOv2 patch_size=14 from config."
        )

    def test_hidden_size_from_config(self, encoder):
        """Hidden size should be read from config (expected 1024 for dinov2-large)."""
        assert encoder._hidden_size == 1024, (
            f"Expected hidden_size=1024 for dinov2-large, got {encoder._hidden_size}. "
            "Update EncoderConfig.proj_output_dim accordingly if different."
        )
