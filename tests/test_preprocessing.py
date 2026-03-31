"""Tests for RGB/depth preprocessing helpers."""

from __future__ import annotations

import pytest
import torch

from spatialvlm.data.preprocessing import (
    normalize_depth_bhw,
    preprocess_rgb_depth,
    resize_depth_bhw,
    resize_rgb_bchw,
    to_float_rgb01,
)


def test_to_float_rgb01_uint8():
    rgb = torch.full((1, 3, 4, 4), fill_value=255, dtype=torch.uint8)
    out = to_float_rgb01(rgb)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out))


def test_to_float_rgb01_float255():
    rgb = torch.full((1, 3, 4, 4), fill_value=128.0, dtype=torch.float32)
    out = to_float_rgb01(rgb)
    expected = torch.full_like(out, 128.0 / 255.0)
    assert torch.allclose(out, expected)


def test_resize_rgb_bchw_shape():
    rgb = torch.randn(2, 3, 32, 40)
    out = resize_rgb_bchw(rgb, size=(518, 518))
    assert out.shape == (2, 3, 518, 518)


def test_resize_rgb_bchw_rejects_invalid_shape():
    rgb = torch.randn(2, 32, 40)
    with pytest.raises(ValueError, match="RGB must be"):
        resize_rgb_bchw(rgb)


def test_resize_depth_bhw_shape():
    depth = torch.randn(2, 30, 50)
    out = resize_depth_bhw(depth, size=(518, 518))
    assert out.shape == (2, 518, 518)


def test_normalize_depth_handles_nan_and_zero():
    depth = torch.tensor([[[float("nan"), 0.0], [2.0, 4.0]]], dtype=torch.float32)
    out = normalize_depth_bhw(depth, percentile=100.0)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out[0, 0, 0] == 0.0  # NaN replaced
    assert out[0, 0, 1] == 0.0  # zero depth stays zero


def test_normalize_depth_with_max_depth():
    depth = torch.tensor([[[0.0, 2.0], [5.0, 20.0]]], dtype=torch.float32)
    out = normalize_depth_bhw(depth, max_depth=10.0)
    assert torch.allclose(out[0, 0, 1], torch.tensor(0.2))
    assert torch.allclose(out[0, 1, 0], torch.tensor(0.5))
    assert out[0, 1, 1] == 1.0  # clipped


def test_preprocess_rgb_depth_end_to_end():
    rgb = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
    depth = torch.rand(1, 32, 32) * 5.0
    rgb_out, depth_out = preprocess_rgb_depth(rgb, depth, size=(518, 518), depth_percentile=99.0)
    assert rgb_out.shape == (1, 3, 518, 518)
    assert depth_out.shape == (1, 518, 518)
    assert rgb_out.dtype == torch.float32
    assert depth_out.dtype == torch.float32
    assert rgb_out.min() >= 0.0
    assert rgb_out.max() <= 1.0
    assert depth_out.min() >= 0.0
    assert depth_out.max() <= 1.0
