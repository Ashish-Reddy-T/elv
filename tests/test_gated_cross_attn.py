"""Tests for Flamingo-style gated cross-attention block."""

import torch

from spatialvlm.fusion.gated_cross_attn import GatedCrossAttentionBlock


class TestGatedCrossAttention:
    def test_shape(self):
        block = GatedCrossAttentionBlock(hidden_dim=64, num_heads=8)
        text = torch.randn(2, 32, 64)
        vision = torch.randn(2, 576, 64)
        out = block(text, vision)
        assert out.shape == (2, 32, 64)

    def test_zero_init_is_exact_passthrough(self):
        """At init, tanh(0)=0 so output must equal input exactly."""
        block = GatedCrossAttentionBlock(hidden_dim=64, num_heads=8)
        block.eval()
        text = torch.randn(1, 16, 64)
        vision = torch.randn(1, 32, 64)
        out = block(text, vision)
        assert torch.allclose(out, text, atol=0.0, rtol=0.0)

    def test_nonzero_gate_changes_output(self):
        block = GatedCrossAttentionBlock(hidden_dim=64, num_heads=8)
        with torch.no_grad():
            block.attn_gate.fill_(2.0)
        text = torch.randn(1, 16, 64)
        vision = torch.randn(1, 32, 64)
        out = block(text, vision)
        assert not torch.allclose(out, text)

    def test_key_padding_mask(self):
        block = GatedCrossAttentionBlock(hidden_dim=32, num_heads=8)
        text = torch.randn(1, 8, 32)
        vision = torch.randn(1, 10, 32)
        # True means "ignore" for nn.MultiheadAttention.
        key_padding_mask = torch.zeros(1, 10, dtype=torch.bool)
        key_padding_mask[:, -3:] = True
        out = block(text, vision, vision_key_padding_mask=key_padding_mask)
        assert out.shape == text.shape
