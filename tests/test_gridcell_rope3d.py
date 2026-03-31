"""Unit tests for GridCellRoPE3D — tetrahedral Fourier basis rotary encoding.

Critical properties verified:
  1. Output shape: [B, N, 3] → [B, N, 64]
  2. Zero learnable parameters
  3. Tetrahedral directions are unit vectors
  4. Isotropy: Σᵢ dᵢ⊗dᵢ = (4/3)I₃  (equal coverage in all 3D directions)
  5. Golden-ratio frequency spacing: f[k+1]/f[k] ≈ φ
  6. Correct number of frequencies and output dims
  7. Deterministic: same input → same output
  8. Distinctness: different positions → different encodings
  9. Output is in [-1, 1] (sin/cos)
"""

import math

import torch
import pytest

from spatialvlm.geometry.gridcell_rope3d import GridCellRoPE3D


@pytest.fixture
def rope3d():
    return GridCellRoPE3D()


class TestOutputShape:
    def test_standard_shape(self, rope3d):
        pos = torch.randn(2, 576, 3)
        out = rope3d(pos)
        assert out.shape == (2, 576, 64), f"Expected [2,576,64], got {out.shape}"

    def test_dinov2_tokens(self, rope3d):
        pos = torch.randn(1, 1369, 3)
        out = rope3d(pos)
        assert out.shape == (1, 1369, 64)

    def test_single_token(self, rope3d):
        pos = torch.randn(4, 1, 3)
        out = rope3d(pos)
        assert out.shape == (4, 1, 64)

    def test_output_dim_is_64(self, rope3d):
        """64 = 4 dirs × 8 freqs × 2 (sin/cos) — must match Qwen3's rotary pairs."""
        pos = torch.randn(1, 10, 3)
        out = rope3d(pos)
        assert out.shape[-1] == 64


class TestNoLearnableParameters:
    def test_zero_parameters(self, rope3d):
        params = list(rope3d.parameters())
        assert len(params) == 0, f"Expected 0 learnable params, found {len(params)}"

    def test_has_buffers(self, rope3d):
        buffers = dict(rope3d.named_buffers())
        assert "tetra_dirs" in buffers
        assert "freqs" in buffers


class TestTetrahedralDirections:
    def test_unit_vectors(self, rope3d):
        """Each tetrahedral direction must be a unit vector."""
        dirs = rope3d.tetra_dirs  # [4, 3]
        norms = torch.norm(dirs, dim=-1)  # [4]
        assert torch.allclose(norms, torch.ones(4), atol=1e-6), (
            f"Tetrahedral directions not unit: norms = {norms}"
        )

    def test_count_is_4(self, rope3d):
        assert rope3d.tetra_dirs.shape == (4, 3)

    def test_isotropy(self, rope3d):
        """Σᵢ dᵢ ⊗ dᵢ should equal (4/3) I₃ for a regular tetrahedron.

        This is the defining property of the tetrahedral frame: all 3D directions
        are covered equally. Any other set of 4 directions breaks this symmetry.
        """
        dirs = rope3d.tetra_dirs.float()  # [4, 3]
        # Outer product sum: [3, 3]
        outer_sum = torch.einsum("nd,ne->de", dirs, dirs)   # [3, 3]
        expected = (4.0 / 3.0) * torch.eye(3)
        assert torch.allclose(outer_sum, expected, atol=1e-6), (
            f"Isotropy violation. Got:\n{outer_sum}\nExpected:\n{expected}"
        )

    def test_specific_values(self, rope3d):
        """Verify exact tetrahedral vertices."""
        s = 1.0 / math.sqrt(3.0)
        expected = torch.tensor([
            [+s, +s, +s],
            [+s, -s, -s],
            [-s, +s, -s],
            [-s, -s, +s],
        ], dtype=torch.float32)
        assert torch.allclose(rope3d.tetra_dirs.float(), expected, atol=1e-6)


class TestFrequencies:
    def test_count_is_8(self, rope3d):
        assert rope3d.freqs.shape == (8,)

    def test_base_frequency(self, rope3d):
        """First frequency should be BASE_FREQ = 10.0."""
        assert torch.allclose(
            rope3d.freqs[0], torch.tensor(10.0), atol=1e-5
        ), f"Base freq expected 10.0, got {rope3d.freqs[0].item()}"

    def test_golden_ratio_spacing(self, rope3d):
        """Consecutive frequencies should have ratio ≈ φ = 1.618..."""
        freqs = rope3d.freqs.float()  # [8]
        ratios = freqs[1:] / freqs[:-1]  # [7]
        phi = torch.tensor(1.618033988749895)
        assert torch.allclose(ratios, phi.expand(7), atol=1e-5), (
            f"Frequency ratios should all be φ={phi.item():.6f}, got {ratios}"
        )

    def test_frequencies_ascending(self, rope3d):
        freqs = rope3d.freqs.float()
        diffs = freqs[1:] - freqs[:-1]
        assert (diffs > 0).all(), "Frequencies should be strictly ascending"


class TestEncoding:
    def test_output_bounded(self, rope3d):
        """sin/cos outputs must lie in [-1, 1]."""
        pos = torch.randn(4, 100, 3) * 10.0
        out = rope3d(pos)
        assert out.abs().max() <= 1.0 + 1e-5

    def test_deterministic(self, rope3d):
        """Same input always produces same output."""
        torch.manual_seed(42)
        pos = torch.randn(2, 50, 3)
        out1 = rope3d(pos)
        out2 = rope3d(pos)
        assert torch.allclose(out1, out2)

    def test_distinct_positions(self, rope3d):
        """Different 3D positions should produce different encodings."""
        pos1 = torch.zeros(1, 1, 3)
        pos2 = torch.ones(1, 1, 3) * 0.1
        out1 = rope3d(pos1)
        out2 = rope3d(pos2)
        assert not torch.allclose(out1, out2), "Different positions gave identical encodings"

    def test_zero_position(self, rope3d):
        """Origin (0,0,0): all projections = 0 → sin=0, cos=1, alternating."""
        pos = torch.zeros(1, 1, 3)
        out = rope3d(pos)  # [1, 1, 64]
        # Encoding layout: [sin, cos, sin, cos, ...] for each (dir, freq) pair
        sins = out[0, 0, 0::2]  # indices 0, 2, 4, ... = sin terms
        coss = out[0, 0, 1::2]  # indices 1, 3, 5, ... = cos terms
        assert torch.allclose(sins, torch.zeros(32), atol=1e-6), "sin(0) should be 0"
        assert torch.allclose(coss, torch.ones(32), atol=1e-6), "cos(0) should be 1"

    def test_device_propagation(self, rope3d):
        pos = torch.randn(1, 10, 3, device="cpu")
        out = rope3d(pos)
        assert out.device.type == "cpu"

    def test_batch_independence(self, rope3d):
        """Each batch element should be encoded independently."""
        pos = torch.randn(3, 100, 3)
        out = rope3d(pos)
        # Manually encode each element
        for i in range(3):
            out_i = rope3d(pos[i : i + 1])
            assert torch.allclose(out[i : i + 1], out_i)

    def test_output_layout_sin_cos_interleaved(self, rope3d):
        """Verify the (sin, cos) interleaving at position p=(1,0,0).

        For p=(1,0,0), projection onto d₁=[s,s,s] = s, d₂=[s,-s,-s] = s, etc.
        Then encoding[0] = sin(proj_d₁ × f₀), encoding[1] = cos(proj_d₁ × f₀), ...
        """
        pos = torch.tensor([[[1.0, 0.0, 0.0]]])   # [1, 1, 3]
        out = rope3d(pos)                           # [1, 1, 64]
        enc = out[0, 0]  # [64]

        s = 1.0 / math.sqrt(3.0)
        dirs = torch.tensor([[+s, +s, +s], [+s, -s, -s], [-s, +s, -s], [-s, -s, +s]])
        freqs = torch.tensor([10.0 * (1.618033988749895**k) for k in range(8)])

        idx = 0
        for di, d in enumerate(dirs):
            proj = float((d * pos[0, 0]).sum())
            for fk in freqs:
                expected_sin = math.sin(proj * float(fk))
                expected_cos = math.cos(proj * float(fk))
                assert abs(float(enc[idx]) - expected_sin) < 1e-5, (
                    f"sin mismatch at dir={di}, freq idx: expected {expected_sin:.6f}, "
                    f"got {float(enc[idx]):.6f}"
                )
                assert abs(float(enc[idx + 1]) - expected_cos) < 1e-5, (
                    f"cos mismatch at dir={di}, freq idx: expected {expected_cos:.6f}, "
                    f"got {float(enc[idx+1]):.6f}"
                )
                idx += 2
