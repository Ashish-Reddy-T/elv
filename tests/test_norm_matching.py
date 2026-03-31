"""Unit tests for RMS norm matching module."""

import torch
import pytest

from spatialvlm.fusion.norm_matching import RMSNormMatching


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute mean RMS norm over all tokens: scalar."""
    return x.float().pow(2).mean(dim=-1).sqrt().mean()


@pytest.fixture
def module():
    return RMSNormMatching(ema_momentum=0.99)


class TestOutputShape:
    def test_shape_preserved(self, module):
        vis = torch.randn(2, 576, 4096)
        out = module(vis)
        assert out.shape == (2, 576, 4096)

    def test_dtype_preserved(self, module):
        vis = torch.randn(2, 576, 4096, dtype=torch.float32)
        out = module(vis)
        assert out.dtype == torch.float32

    def test_shape_with_text(self, module):
        module.train()
        vis = torch.randn(2, 576, 4096)
        txt = torch.randn(2, 64, 4096)
        out = module(vis, text_tokens=txt)
        assert out.shape == (2, 576, 4096)


class TestNoLearnableParameters:
    def test_zero_parameters(self, module):
        params = list(module.parameters())
        assert len(params) == 0, f"Expected 0 learnable params, found {len(params)}"

    def test_has_ema_buffer(self, module):
        buffers = dict(module.named_buffers())
        assert "text_rms_ema" in buffers

    def test_ema_initialised_to_one(self, module):
        assert torch.allclose(module.text_rms_ema, torch.ones(1))


class TestNormScaling:
    def test_scales_toward_text_norm(self):
        """After seeing text tokens, vision output RMS should approach text RMS."""
        module = RMSNormMatching(ema_momentum=0.0)  # instant update for testing
        module.train()

        # Vision tokens with large norms (10× text)
        vis = torch.randn(2, 576, 64) * 10.0   # RMS ≈ 10.0
        txt = torch.randn(2, 64, 64) * 1.0     # RMS ≈ 1.0

        out = module(vis, text_tokens=txt)

        text_rms = float(rms_norm(txt))
        out_rms = float(rms_norm(out))

        assert abs(out_rms - text_rms) < 0.5, (
            f"Output RMS {out_rms:.3f} should be close to text RMS {text_rms:.3f}"
        )

    def test_no_op_when_norms_match(self):
        """If vision and text norms are already equal, scaling should be ~1."""
        module = RMSNormMatching(ema_momentum=0.0)
        module.train()

        vis = torch.randn(2, 576, 64)
        txt = vis[:, :64, :].clone()  # same statistics

        # Force EMA to match vision RMS
        vis_rms = float(rms_norm(vis))
        module.text_rms_ema = torch.tensor([vis_rms])

        out = module(vis)  # no text tokens → uses stored EMA
        ratio = float(rms_norm(out)) / float(rms_norm(vis))
        assert abs(ratio - 1.0) < 0.05, f"Scale ratio should be ≈1.0, got {ratio:.4f}"


class TestEMAUpdate:
    def test_ema_moves_toward_text_rms(self):
        """EMA should shift toward the current batch text RMS."""
        module = RMSNormMatching(ema_momentum=0.5)
        module.train()

        initial_ema = float(module.text_rms_ema)

        # Text tokens with high RMS
        txt = torch.randn(2, 32, 64) * 5.0     # RMS ≈ 5.0
        vis = torch.randn(2, 10, 64)
        _ = module(vis, text_tokens=txt)

        new_ema = float(module.text_rms_ema)
        # With momentum=0.5, new_ema = 0.5 * 1.0 + 0.5 * ~5.0 ≈ 3.0
        assert new_ema > initial_ema, (
            f"EMA should increase toward text RMS, was {initial_ema:.3f}, now {new_ema:.3f}"
        )

    def test_ema_not_updated_at_inference(self):
        """EMA should not change when module is in eval mode (no text tokens)."""
        module = RMSNormMatching(ema_momentum=0.5)
        module.eval()

        initial_ema = float(module.text_rms_ema)

        txt = torch.randn(2, 32, 64) * 5.0
        vis = torch.randn(2, 10, 64)
        _ = module(vis, text_tokens=txt)  # eval mode → EMA should NOT update

        new_ema = float(module.text_rms_ema)
        assert new_ema == initial_ema, "EMA should not update in eval mode"

    def test_ema_not_updated_without_text_tokens(self):
        """EMA should not change when text_tokens=None, even in train mode."""
        module = RMSNormMatching(ema_momentum=0.5)
        module.train()

        initial_ema = float(module.text_rms_ema)
        vis = torch.randn(2, 10, 64)
        _ = module(vis, text_tokens=None)

        new_ema = float(module.text_rms_ema)
        assert new_ema == initial_ema, "EMA should not update without text_tokens"

    def test_repeated_updates_converge(self):
        """After many batches with consistent text RMS, EMA should converge."""
        module = RMSNormMatching(ema_momentum=0.9)
        module.train()

        target_rms = 3.0
        vis = torch.randn(2, 10, 64)

        for _ in range(200):
            # Text tokens crafted to have a specific RMS
            # RMS of randn is 1.0, so scale by target_rms
            txt = torch.randn(2, 32, 64) * target_rms
            _ = module(vis, text_tokens=txt)

        converged_ema = float(module.text_rms_ema)
        assert abs(converged_ema - target_rms) < 0.2, (
            f"EMA should converge to ≈{target_rms}, got {converged_ema:.4f}"
        )


class TestInferenceMode:
    def test_no_text_uses_ema(self):
        """At inference without text tokens, module should use stored EMA."""
        module = RMSNormMatching(ema_momentum=0.0)
        module.eval()

        # Manually set EMA to a known value
        target = 2.0
        module.text_rms_ema = torch.tensor([target])

        vis = torch.randn(1, 10, 64) * 10.0  # large norms
        out = module(vis, text_tokens=None)

        # Output RMS should be close to target
        out_rms = float(rms_norm(out))
        assert abs(out_rms - target) < 0.5, (
            f"Inference output RMS {out_rms:.3f} should match stored EMA {target:.3f}"
        )
