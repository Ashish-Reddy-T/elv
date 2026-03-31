"""Image/depth preprocessing utilities for SpatialVLM data pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def to_float_rgb01(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to float in [0,1].

    Supports:
    - uint8 [0,255]
    - float [0,255] or [0,1]
    """
    if rgb.dtype == torch.uint8:
        return rgb.float() / 255.0
    out = rgb.float()
    if out.numel() > 0 and out.max() > 1.0:
        out = out / 255.0
    return out


def resize_rgb_bchw(
    rgb: torch.Tensor,
    size: tuple[int, int] = (518, 518),
) -> torch.Tensor:
    """Resize RGB tensor [B,3,H,W] with bilinear interpolation."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}.")
    return functional.interpolate(
        rgb,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def resize_depth_bhw(
    depth: torch.Tensor,
    size: tuple[int, int] = (518, 518),
) -> torch.Tensor:
    """Resize depth tensor [B,H,W] using nearest-neighbor interpolation."""
    if depth.ndim != 3:
        raise ValueError(f"Depth must be [B,H,W], got {tuple(depth.shape)}.")
    depth_4d = depth.unsqueeze(1)  # [B,1,H,W]
    out = functional.interpolate(depth_4d, size=size, mode="nearest")
    return out[:, 0]  # [B,H,W]


def normalize_depth_bhw(
    depth: torch.Tensor,
    max_depth: float | None = None,
    percentile: float = 99.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize depth [B,H,W] into [0,1], robust to zeros/NaNs."""
    if depth.ndim != 3:
        raise ValueError(f"Depth must be [B,H,W], got {tuple(depth.shape)}.")

    d = depth.float().clone()
    d[~torch.isfinite(d)] = 0.0
    d = torch.clamp(d, min=0.0)

    if max_depth is not None:
        denom = max(max_depth, eps)
        return torch.clamp(d / denom, 0.0, 1.0)

    bsz = d.shape[0]
    out = torch.zeros_like(d)
    for i in range(bsz):
        flat = d[i].reshape(-1)
        valid = flat[flat > 0]
        if valid.numel() == 0:
            continue
        scale = torch.quantile(valid, q=percentile / 100.0).item()
        scale = max(scale, eps)
        out[i] = torch.clamp(d[i] / scale, 0.0, 1.0)
    return out


def preprocess_rgb_depth(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    size: tuple[int, int] = (518, 518),
    depth_percentile: float = 99.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess RGB + depth into model-ready tensors.

    Parameters
    ----------
    rgb : Tensor[B, 3, H, W]
    depth : Tensor[B, H, W]
    """
    rgb_float = to_float_rgb01(rgb)
    rgb_resized = resize_rgb_bchw(rgb_float, size=size)
    depth_resized = resize_depth_bhw(depth, size=size)
    depth_norm = normalize_depth_bhw(depth_resized, percentile=depth_percentile)
    return rgb_resized, depth_norm
