"""DEPRECATED: Tests for Flamingo-style gated cross-attention block with GQA.

Gated cross-attention was replaced by Qwen3-VL's native DeepStack (2026-04-05).
These tests are kept for ablation reference but skipped in the main test suite.
"""

import pytest
import torch

pytestmark = pytest.mark.skip(reason="Gated cross-attention deprecated — replaced by DeepStack")

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
        # True means "ignore" in our semantics
        key_padding_mask = torch.zeros(1, 10, dtype=torch.bool)
        key_padding_mask[:, -3:] = True
        out = block(text, vision, vision_key_padding_mask=key_padding_mask)
        assert out.shape == text.shape


class TestGQA:
    """Verify Grouped Query Attention with fewer KV heads than query heads."""

    def test_gqa_4_to_1_ratio(self):
        """32 query heads, 8 KV heads (Qwen3 ratio)."""
        block = GatedCrossAttentionBlock(hidden_dim=256, num_heads=32, num_kv_heads=8)
        text = torch.randn(1, 16, 256)
        vision = torch.randn(1, 64, 256)
        out = block(text, vision)
        assert out.shape == (1, 16, 256)

    def test_kv_projection_shapes(self):
        """K and V projections should use num_kv_heads, not num_heads."""
        block = GatedCrossAttentionBlock(hidden_dim=128, num_heads=16, num_kv_heads=4)
        head_dim = 128 // 16  # = 8
        expected_kv_dim = 4 * head_dim  # = 32
        assert block.k_proj.out_features == expected_kv_dim
        assert block.v_proj.out_features == expected_kv_dim
        assert block.q_proj.out_features == 128  # full hidden_dim

    def test_gqa_fewer_kv_params(self):
        """GQA block with 8 KV heads should have fewer params than 32 KV heads."""
        full = GatedCrossAttentionBlock(hidden_dim=256, num_heads=32, num_kv_heads=32)
        gqa = GatedCrossAttentionBlock(hidden_dim=256, num_heads=32, num_kv_heads=8)
        full_kv = full.k_proj.weight.numel() + full.v_proj.weight.numel()
        gqa_kv = gqa.k_proj.weight.numel() + gqa.v_proj.weight.numel()
        assert gqa_kv < full_kv, f"GQA KV params ({gqa_kv}) should be < full ({full_kv})"
        assert gqa_kv == full_kv // 4  # 4:1 ratio

    def test_gqa_zero_init_passthrough(self):
        """GQA block at init should also be exact passthrough."""
        block = GatedCrossAttentionBlock(hidden_dim=128, num_heads=16, num_kv_heads=4)
        block.eval()
        text = torch.randn(1, 8, 128)
        vision = torch.randn(1, 32, 128)
        out = block(text, vision)
        assert torch.allclose(out, text, atol=0.0, rtol=0.0)

    def test_num_heads_not_divisible_by_kv_heads_raises(self):
        """Should reject if num_heads is not a multiple of num_kv_heads."""
        try:
            GatedCrossAttentionBlock(hidden_dim=64, num_heads=8, num_kv_heads=3)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_default_kv_heads_equals_num_heads(self):
        """When num_kv_heads is not specified, should default to standard MHA."""
        block = GatedCrossAttentionBlock(hidden_dim=64, num_heads=8)
        assert block.num_kv_heads == 8
        assert block.num_groups == 1
