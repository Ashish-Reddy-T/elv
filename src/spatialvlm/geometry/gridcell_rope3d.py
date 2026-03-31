"""GridCellRoPE3D: tetrahedral Fourier basis rotary position encoding for 3D coordinates.

Design rationale:

1. Tetrahedral directions (4 directions):
   The 4 vertices of a regular tetrahedron inscribed in the unit sphere are the unique
   set of 4 unit vectors satisfying isotropy:
       Σᵢ dᵢ ⊗ dᵢ = (4/3) I₃
   This means no direction in 3D space is privileged — the encoding covers all
   orientations equally. Any other set of 4 directions breaks this symmetry.

2. Golden-ratio frequency spacing (8 frequencies):
   fₖ = 10.0 × φᵏ, where φ = 1.618... (golden ratio)
   The golden ratio is maximally irrational: no two frequencies share a common period,
   minimising aliasing and maximising coverage of the frequency domain.
   Range: f₀ = 10.0 rad/m (10cm resolution) to f₇ ≈ 76.0 rad/m (2.9m room-scale).

3. Output dimension:
   4 directions × 8 frequencies × 2 (sin/cos) = 64 dimensions.
   This matches Qwen3-VL-8B's M-RoPE exactly: head_dim=128 → 64 rotary pairs per head.
   No projection layer needed — direct compatibility.

4. No learnable parameters:
   The encoding is a deterministic function of 3D position. All state is
   stored in register_buffer (moved with .to(device) but not trained).

Usage:
    rope3d = GridCellRoPE3D()
    positions = ...  # [B, N, 3] in metres
    angles = rope3d(positions)  # [B, N, 64]
    # Apply as rotary embedding in attention computation (replaces M-RoPE for spatial tokens)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GridCellRoPE3D(nn.Module):
    """Rotary position encoding for 3D spatial coordinates.

    Implements a tetrahedral Fourier basis inspired by neuroscience grid cell theory.
    Zero learnable parameters; all constants stored as buffers.

    Parameters
    ----------
    None — all hyperparameters are fixed by design (see module docstring).
    """

    PHI: float = 1.618033988749895  # golden ratio φ
    N_DIRS: int = 4                 # tetrahedral directions
    N_FREQS: int = 8                # golden-ratio frequencies
    BASE_FREQ: float = 10.0        # f₀ in rad/m (10cm resolution)

    def __init__(self) -> None:
        super().__init__()

        # 4 tetrahedral unit-vectors — vertices of a regular tetrahedron on unit sphere
        # Unique up to global rotation; satisfy Σᵢ dᵢ⊗dᵢ = (4/3) I₃ (isotropic coverage)
        s = 1.0 / math.sqrt(3.0)
        tetra = torch.tensor(
            [
                [+s, +s, +s],  # d₁
                [+s, -s, -s],  # d₂
                [-s, +s, -s],  # d₃
                [-s, -s, +s],  # d₄
            ],
            dtype=torch.float32,
        )  # [4, 3]
        self.register_buffer("tetra_dirs", tetra)  # [4, 3]

        # 8 golden-ratio spaced frequencies: fₖ = 10.0 × φᵏ
        # Spans 10cm (f₀ = 10.0 rad/m → λ = 0.63m) to 2.91m room scale
        freqs = torch.tensor(
            [self.BASE_FREQ * (self.PHI**k) for k in range(self.N_FREQS)],
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
        angles : Tensor[B, N, 64]
            Rotary encoding ready for use in RoPE attention.
            Dimension 64 = 4 dirs × 8 freqs × 2 (sin interleaved with cos).
            Layout: [sin(d₁·p·f₀), cos(d₁·p·f₀), sin(d₁·p·f₁), cos(d₁·p·f₁), ...]
        """
        # positions: [B, N, 3]
        # tetra_dirs: [4, 3]

        # Scalar projections: dot each position with each tetrahedral direction
        # projections[b, n, i] = dᵢ · p_{b,n}
        projections = torch.einsum("bnd,fd->bnf", positions, self.tetra_dirs)  # [B, N, 4]

        # Scale by frequencies: [B, N, 4, 8]
        scaled = projections.unsqueeze(-1) * self.freqs  # [B, N, 4, 8]

        # Compute sin and cos: both [B, N, 4, 8]
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)

        # Interleave: [B, N, 4, 8, 2] → flatten last 3 dims → [B, N, 64]
        enc = torch.stack([sin_enc, cos_enc], dim=-1)   # [B, N, 4, 8, 2]
        enc = enc.reshape(*positions.shape[:2], self.N_DIRS * self.N_FREQS * 2)  # [B, N, 64]

        return enc
