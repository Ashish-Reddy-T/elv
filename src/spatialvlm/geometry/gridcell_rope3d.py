"""IcosahedralRoPE3D: icosahedral Fourier basis rotary position encoding for 3D coordinates.

Design rationale (extends GridPE 2D — Li et al. AAAI 2025 — to 3D):

1. Icosahedral directions (6 directions):
   The regular icosahedron has 12 vertices forming 6 antipodal pairs on the unit sphere.
   These 6 unique unit vectors are the optimal packing of 6 directions on S^2 and satisfy
   the isotropy condition:
       sum_i d_i (x) d_i^T = 2 I_3
   This is the 3D analogue of the hexagonal (3-direction) set used in GridPE 2D,
   which achieves optimal uniform coverage on S^1.

2. e^(1/3) frequency spacing (8 frequencies):
   f_k = 10.0 * e^(k/3), where e^(1/3) = 1.3956...
   The GridPE paper proves (via the economy principle, Wei et al. 2015) that
   the optimal scale ratio for p-dimensional space is r = e^(1/p).
   For p=3: r = e^(1/3), which minimizes the total encoding dimensions needed
   to unambiguously represent positions in 3D space.

   Note: For p=2, e^(1/2) = 1.649 ≈ phi (golden ratio). Our previous golden-ratio
   choice was an unwitting approximation of the theoretically optimal 2D ratio.

3. Output dimension:
   6 directions x 8 frequencies x 2 (sin/cos) = 96 dimensions.
   This exceeds Qwen3-VL-8B's 64 rotary pairs (128 dims), so at the RoPE
   injection point we pad with 16 identity pairs (cos=1, sin=0) to reach 128.
   The 96 real dims encode richer 3D position than the old 64-dim tetrahedral version.

4. No learnable parameters:
   The encoding is a deterministic function of 3D position. All state is
   stored in register_buffer (moved with .to(device) but not trained).

Usage:
    rope3d = IcosahedralRoPE3D()
    positions = ...  # [B, N, 3] in metres
    angles = rope3d(positions)  # [B, N, 96]
    # Pad to 128 and inject via RoPE monkey-patch (replaces M-RoPE for spatial tokens)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _build_icosahedral_directions() -> torch.Tensor:
    """Compute 6 normalized icosahedral direction vectors.

    The regular icosahedron has 12 vertices at (0, +/-1, +/-phi), (+/-1, +/-phi, 0),
    (+/-phi, 0, +/-1) where phi = (1+sqrt(5))/2. We take 6 antipodal pairs,
    selecting one from each pair (positive-first convention).

    Returns
    -------
    dirs : Tensor[6, 3]
        Normalized icosahedral direction vectors on the unit sphere.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    norm = math.sqrt(1.0 + phi * phi)  # sqrt(1 + phi^2) = sqrt(3.618...)

    # 6 antipodal pairs from icosahedron vertices (one from each pair)
    dirs = (
        torch.tensor(
            [
                [0.0, +1.0, +phi],
                [0.0, +1.0, -phi],
                [+1.0, +phi, 0.0],
                [+1.0, -phi, 0.0],
                [+phi, 0.0, +1.0],
                [+phi, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        / norm
    )  # [6, 3], each row is a unit vector

    return dirs


class IcosahedralRoPE3D(nn.Module):
    """Rotary position encoding for 3D spatial coordinates.

    Implements an icosahedral Fourier basis inspired by neuroscience grid cell theory
    and the GridPE paper (Li et al., AAAI 2025). Zero learnable parameters; all
    constants stored as buffers.
    """

    N_DIRS: int = 6  # icosahedral directions (antipodal pairs)
    N_FREQS: int = 8  # e^(1/3)-spaced frequencies
    BASE_FREQ: float = 10.0  # f_0 in rad/m
    FREQ_RATIO: float = math.exp(1.0 / 3.0)  # e^(1/3), optimal for 3D

    def __init__(self) -> None:
        super().__init__()

        # 6 icosahedral unit-vectors — antipodal pairs from regular icosahedron
        # Satisfy sum_i d_i (x) d_i^T = 2 I_3 (isotropic coverage)
        icosa_dirs = _build_icosahedral_directions()  # [6, 3]
        self.register_buffer("directions", icosa_dirs)  # [6, 3]

        # 8 frequencies: f_k = 10.0 * e^(k/3)
        # Optimal scale ratio r = e^(1/3) for p=3 (economy principle)
        freqs = torch.tensor(
            [self.BASE_FREQ * (self.FREQ_RATIO**k) for k in range(self.N_FREQS)],
            dtype=torch.float32,
        )  # [8]
        self.register_buffer("freqs", freqs)  # [8]

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute 3D rotary position encoding.

        Parameters
        ----------
        positions : Tensor[B, N, 3]
            3D coordinates in camera/world space (metres).

        Returns
        -------
        angles : Tensor[B, N, 96]
            Rotary encoding for 3D spatial tokens.
            Dimension 96 = 6 dirs x 8 freqs x 2 (sin interleaved with cos).
            Layout: [sin(d_1.p.f_0), cos(d_1.p.f_0), sin(d_1.p.f_1), cos(d_1.p.f_1), ...]
        """
        # positions: [B, N, 3]
        # directions: [6, 3]

        # Scalar projections: dot each position with each icosahedral direction
        # projections[b, n, i] = d_i . p_{b,n}
        projections = torch.einsum("bnd,fd->bnf", positions, self.directions)  # [B, N, 6]

        # Scale by frequencies: [B, N, 6, 8]
        scaled = projections.unsqueeze(-1) * self.freqs  # [B, N, 6, 8]

        # Compute sin and cos: both [B, N, 6, 8]
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)

        # Interleave: [B, N, 6, 8, 2] -> flatten last 3 dims -> [B, N, 96]
        enc = torch.stack([sin_enc, cos_enc], dim=-1)  # [B, N, 6, 8, 2]
        enc = enc.reshape(*positions.shape[:2], self.N_DIRS * self.N_FREQS * 2)  # [B, N, 96]

        return enc


# Backward-compatible alias for code that references the old name
GridCellRoPE3D = IcosahedralRoPE3D
