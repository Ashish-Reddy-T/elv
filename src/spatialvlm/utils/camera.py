"""Camera intrinsics and projection utilities.

Implements the standard pinhole camera model:
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

where (u, v) are pixel coordinates (col, row) and (cx, cy) is the principal point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Follows OpenCV convention: u = col index (x), v = row index (y).

    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    width, height : int
        Image dimensions in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def make_pixel_grid(
    width: int,
    height: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a dense grid of pixel (u, v) coordinates.

    Parameters
    ----------
    width, height : int
        Image dimensions.
    device : torch.device | None
    dtype : torch.dtype

    Returns
    -------
    grid : Tensor[H, W, 2]
        Last dimension is (u, v) = (col, row), both 0-indexed.
    """
    u = torch.arange(width, device=device, dtype=dtype)   # [W]
    v = torch.arange(height, device=device, dtype=dtype)  # [H]
    # indexing='xy': first arg varies along W (u), second arg along H (v)
    uu, vv = torch.meshgrid(u, v, indexing="xy")           # both [H, W]
    return torch.stack([uu, vv], dim=-1)                   # [H, W, 2]


def backproject_pixel(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    """Backproject pixels with given depths into 3D camera-space coordinates.

    Parameters
    ----------
    u : Tensor[...]
        Pixel column coordinates (float).
    v : Tensor[...]
        Pixel row coordinates (float).
    depth : Tensor[...]
        Metric depth values (same shape as u, v).
        Zero or NaN values produce (0, 0, 0) or (NaN, NaN, NaN) respectively.
    intrinsics : CameraIntrinsics

    Returns
    -------
    points : Tensor[..., 3]
        3D coordinates [X, Y, Z] in camera space. Shape = (*u.shape, 3).
    """
    # X = (u - cx) * d / fx
    # Y = (v - cy) * d / fy
    # Z = d
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    z = depth
    return torch.stack([x, y, z], dim=-1)  # [..., 3]
