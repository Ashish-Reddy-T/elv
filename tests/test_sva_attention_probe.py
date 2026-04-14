"""Tests for the SVA attention-mass probe (`return_attention_stats`).

The probe reports per-layer, per-head attention mass on each KV source
(SigLIP / DINOv2 / GATr). Because attention weights sum to 1 across the
KV dimension, the three per-type head means must also sum to ~1 per
(layer, head).
"""

import pytest
import torch

from spatialvlm.fusion.sva import SpatialVisionAggregator


def _make_model(
    hidden_dim: int = 64,
    num_queries: int = 1369,
    num_layers: int = 2,
    num_heads: int = 8,
    use_typed_attention_bias: bool = True,
) -> SpatialVisionAggregator:
    return SpatialVisionAggregator(
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
        use_typed_attention_bias=use_typed_attention_bias,
    )


def test_probe_returns_tuple_with_stats_dict():
    model = _make_model()
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    out, stats = model(siglip, dinov2, gatr, return_attention_stats=True)

    assert out.shape == (1, 1369, 64)
    assert set(stats.keys()) == {"layer_0", "layer_1"}
    for layer_stats in stats.values():
        assert "head_mean_to_siglip" in layer_stats
        assert "head_mean_to_dino" in layer_stats
        assert "head_mean_to_gatr" in layer_stats


def test_probe_head_means_have_correct_shape():
    model = _make_model(hidden_dim=128, num_heads=16, num_layers=3)
    siglip = torch.randn(2, 576, 128)
    dinov2 = torch.randn(2, 1369, 128)
    gatr = torch.randn(2, 1369, 128)

    _, stats = model(siglip, dinov2, gatr, return_attention_stats=True)

    assert len(stats) == 3
    for layer_stats in stats.values():
        for key in ("head_mean_to_siglip", "head_mean_to_dino", "head_mean_to_gatr"):
            assert layer_stats[key].shape == (16,)


def test_probe_fractions_sum_to_one_per_layer_per_head():
    """Per-head attention mass must sum to ~1 across the three source types."""
    torch.manual_seed(0)
    model = _make_model(num_heads=4, num_layers=2)
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    _, stats = model(siglip, dinov2, gatr, return_attention_stats=True)

    for layer_name, layer_stats in stats.items():
        total = (
            layer_stats["head_mean_to_siglip"]
            + layer_stats["head_mean_to_dino"]
            + layer_stats["head_mean_to_gatr"]
        )
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5), (
            f"{layer_name} per-head mass sum = {total.tolist()}, expected ~1.0"
        )


def test_probe_typed_bias_snapshot_matches_parameter():
    model = _make_model(use_typed_attention_bias=True)
    with torch.no_grad():
        model.layers[0].typed_attention_bias.copy_(torch.arange(9).float().view(3, 3))
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    _, stats = model(siglip, dinov2, gatr, return_attention_stats=True)

    bias_snapshot = stats["layer_0"]["typed_bias"]
    assert bias_snapshot.shape == (3, 3)
    assert torch.allclose(bias_snapshot, torch.arange(9).float().view(3, 3))


def test_probe_omits_typed_bias_when_disabled():
    model = _make_model(use_typed_attention_bias=False)
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    _, stats = model(siglip, dinov2, gatr, return_attention_stats=True)

    assert "typed_bias" not in stats["layer_0"]


def test_probe_off_is_default_and_returns_tensor_only():
    """Default path must be a no-op: no tuple, no stats, no behavioural change."""
    torch.manual_seed(1)
    model = _make_model()
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    out = model(siglip, dinov2, gatr)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1369, 64)


def test_probe_and_default_paths_agree_numerically():
    """The unfused probe path must be numerically equivalent to fused SDPA."""
    torch.manual_seed(2)
    model = _make_model(num_heads=4, num_layers=2)
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    with torch.no_grad():
        out_fast = model(siglip, dinov2, gatr)
        out_probe, _ = model(siglip, dinov2, gatr, return_attention_stats=True)

    assert torch.allclose(out_fast, out_probe, atol=1e-5, rtol=1e-5)


def test_probe_fractions_respect_padding_mask():
    """Masked KV tokens should not carry attention mass."""
    torch.manual_seed(3)
    model = _make_model(num_heads=4, num_layers=1)
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)
    # Mask out the entire GATr span.
    kv_mask = torch.ones(1, 3314, dtype=torch.bool)
    kv_mask[:, 576 + 1369 :] = False

    _, stats = model(
        siglip,
        dinov2,
        gatr,
        kv_padding_mask=kv_mask,
        return_attention_stats=True,
    )

    gatr_mass = stats["layer_0"]["head_mean_to_gatr"]
    assert torch.allclose(gatr_mass, torch.zeros_like(gatr_mass), atol=1e-6), (
        f"Masked-out GATr tokens still received attention: {gatr_mass.tolist()}"
    )


def test_probe_gradients_still_flow():
    """Using the probe path must not block backprop through the SVA output."""
    model = _make_model(num_heads=4, num_layers=1)
    siglip = torch.randn(1, 576, 64, requires_grad=True)
    dinov2 = torch.randn(1, 1369, 64, requires_grad=True)
    gatr = torch.randn(1, 1369, 64, requires_grad=True)

    out, _ = model(siglip, dinov2, gatr, return_attention_stats=True)
    out.sum().backward()

    assert siglip.grad is not None and siglip.grad.abs().sum() > 0
    assert dinov2.grad is not None and dinov2.grad.abs().sum() > 0
    assert gatr.grad is not None and gatr.grad.abs().sum() > 0


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_probe_layer_count_matches_module(num_layers: int):
    model = _make_model(num_layers=num_layers, num_heads=4)
    siglip = torch.randn(1, 576, 64)
    dinov2 = torch.randn(1, 1369, 64)
    gatr = torch.randn(1, 1369, 64)

    _, stats = model(siglip, dinov2, gatr, return_attention_stats=True)
    assert len(stats) == num_layers
    for i in range(num_layers):
        assert f"layer_{i}" in stats
