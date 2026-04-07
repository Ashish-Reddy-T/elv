"""Unit tests for IcosahedralRoPE3D — icosahedral Fourier basis rotary encoding.

Critical properties verified:
  1. Output shape: [B, N, 3] -> [B, N, 96]
  2. Zero learnable parameters
  3. Icosahedral directions are unit vectors (6 directions)
  4. Isotropy: sum_i d_i (x) d_i^T = 2*I_3  (equal coverage in all 3D directions)
  5. e^(1/3) frequency spacing: f[k+1]/f[k] = e^(1/3) (optimal for 3D)
  6. Correct number of frequencies and output dims
  7. Deterministic: same input -> same output
  8. Distinctness: different positions -> different encodings
  9. Output is in [-1, 1] (sin/cos)
  10. Backward-compatible alias: GridCellRoPE3D == IcosahedralRoPE3D
"""

import math

import pytest
import torch

from spatialvlm.geometry.gridcell_rope3d import GridCellRoPE3D, IcosahedralRoPE3D


@pytest.fixture
def rope3d():
    return IcosahedralRoPE3D()


class TestOutputShape:
    def test_standard_shape(self, rope3d):
        pos = torch.randn(2, 1369, 3)
        out = rope3d(pos)
        assert out.shape == (2, 1369, 96), f"Expected [2,1369,96], got {out.shape}"

    def test_arbitrary_tokens(self, rope3d):
        pos = torch.randn(1, 500, 3)
        out = rope3d(pos)
        assert out.shape == (1, 500, 96)

    def test_single_token(self, rope3d):
        pos = torch.randn(4, 1, 3)
        out = rope3d(pos)
        assert out.shape == (4, 1, 96)

    def test_output_dim_is_96(self, rope3d):
        """96 = 6 dirs x 8 freqs x 2 (sin/cos)."""
        pos = torch.randn(1, 10, 3)
        out = rope3d(pos)
        assert out.shape[-1] == 96


class TestNoLearnableParameters:
    def test_zero_parameters(self, rope3d):
        params = list(rope3d.parameters())
        assert len(params) == 0, f"Expected 0 learnable params, found {len(params)}"

    def test_has_buffers(self, rope3d):
        buffers = dict(rope3d.named_buffers())
        assert "directions" in buffers
        assert "freqs" in buffers


class TestIcosahedralDirections:
    def test_unit_vectors(self, rope3d):
        """Each icosahedral direction must be a unit vector."""
        dirs = rope3d.directions  # [6, 3]
        norms = torch.norm(dirs, dim=-1)  # [6]
        assert torch.allclose(norms, torch.ones(6), atol=1e-6), (
            f"Icosahedral directions not unit: norms = {norms}"
        )

    def test_count_is_6(self, rope3d):
        assert rope3d.directions.shape == (6, 3)

    def test_isotropy(self, rope3d):
        """sum_i d_i (x) d_i^T should equal 2*I_3 for 6 icosahedral directions.

        This is the defining property of the icosahedral frame: all 3D directions
        are covered equally. The icosahedron is a Platonic solid whose vertex
        directions satisfy isotropy with factor n/3 = 6/3 = 2.
        """
        dirs = rope3d.directions.float()  # [6, 3]
        # Outer product sum: [3, 3]
        outer_sum = torch.einsum("nd,ne->de", dirs, dirs)   # [3, 3]
        expected = 2.0 * torch.eye(3)
        assert torch.allclose(outer_sum, expected, atol=1e-5), (
            f"Isotropy violation. Got:\n{outer_sum}\nExpected:\n{expected}"
        )

    def test_directions_from_icosahedron_vertices(self, rope3d):
        """Verify directions come from icosahedron vertices (0, +/-1, +/-phi)."""
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        norm = math.sqrt(1.0 + phi * phi)
        # Check that each direction's components are {0, 1/norm, phi/norm}
        dirs = rope3d.directions.float()
        abs_dirs = dirs.abs()
        component_vals = torch.tensor([0.0, 1.0 / norm, phi / norm])
        for i in range(6):
            for j in range(3):
                val = abs_dirs[i, j].item()
                assert any(abs(val - c) < 1e-5 for c in component_vals), (
                    f"Direction {i} component {j} = {val} not in icosahedral set"
                )


class TestFrequencies:
    def test_count_is_8(self, rope3d):
        assert rope3d.freqs.shape == (8,)

    def test_base_frequency(self, rope3d):
        """First frequency should be BASE_FREQ = 10.0."""
        assert torch.allclose(
            rope3d.freqs[0], torch.tensor(10.0), atol=1e-5
        ), f"Base freq expected 10.0, got {rope3d.freqs[0].item()}"

    def test_e_one_third_spacing(self, rope3d):
        """Consecutive frequencies should have ratio = e^(1/3) = 1.3956..."""
        freqs = rope3d.freqs.float()  # [8]
        ratios = freqs[1:] / freqs[:-1]  # [7]
        e_one_third = torch.tensor(math.exp(1.0 / 3.0))
        assert torch.allclose(ratios, e_one_third.expand(7), atol=1e-5), (
            f"Frequency ratios should all be e^(1/3)={e_one_third.item():.6f}, got {ratios}"
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
        """Origin (0,0,0): all projections = 0 -> sin=0, cos=1, alternating."""
        pos = torch.zeros(1, 1, 3)
        out = rope3d(pos)  # [1, 1, 96]
        # Encoding layout: [sin, cos, sin, cos, ...] for each (dir, freq) pair
        sins = out[0, 0, 0::2]  # indices 0, 2, 4, ... = sin terms (48 values)
        coss = out[0, 0, 1::2]  # indices 1, 3, 5, ... = cos terms (48 values)
        assert torch.allclose(sins, torch.zeros(48), atol=1e-6), "sin(0) should be 0"
        assert torch.allclose(coss, torch.ones(48), atol=1e-6), "cos(0) should be 1"

    def test_device_propagation(self, rope3d):
        pos = torch.randn(1, 10, 3, device="cpu")
        out = rope3d(pos)
        assert out.device.type == "cpu"

    def test_batch_independence(self, rope3d):
        """Each batch element should be encoded independently."""
        pos = torch.randn(3, 100, 3)
        out = rope3d(pos)
        for i in range(3):
            out_i = rope3d(pos[i : i + 1])
            assert torch.allclose(out[i : i + 1], out_i)

    def test_output_layout_sin_cos_interleaved(self, rope3d):
        """Verify (sin, cos) interleaving at position p=(1,0,0)."""
        pos = torch.tensor([[[1.0, 0.0, 0.0]]])   # [1, 1, 3]
        out = rope3d(pos)                           # [1, 1, 96]
        enc = out[0, 0]  # [96]

        dirs = rope3d.directions.float()
        freqs = rope3d.freqs.float()

        idx = 0
        for di in range(6):
            d = dirs[di]
            proj = float((d * pos[0, 0]).sum())
            for fk in freqs:
                expected_sin = math.sin(proj * float(fk))
                expected_cos = math.cos(proj * float(fk))
                assert abs(float(enc[idx]) - expected_sin) < 1e-5, (
                    f"sin mismatch at dir={di}, freq: expected {expected_sin:.6f}, "
                    f"got {float(enc[idx]):.6f}"
                )
                assert abs(float(enc[idx + 1]) - expected_cos) < 1e-5, (
                    f"cos mismatch at dir={di}, freq: expected {expected_cos:.6f}, "
                    f"got {float(enc[idx+1]):.6f}"
                )
                idx += 2


class TestBackwardCompatibility:
    def test_alias(self):
        """GridCellRoPE3D should be an alias for IcosahedralRoPE3D."""
        assert GridCellRoPE3D is IcosahedralRoPE3D
