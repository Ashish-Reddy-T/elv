"""Tests for permutation diagnostic."""

from __future__ import annotations

import torch

from spatialvlm.eval.permutation_test import permute_tokens, run_permutation_test


def test_permute_tokens_all_positions():
    x = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    out = permute_tokens(x)
    assert out.shape == x.shape
    assert torch.equal(torch.sort(out.flatten()).values, torch.sort(x.flatten()).values)


def test_permute_tokens_spatial_mask_only():
    x = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    mask = torch.tensor([[False, False, True, True, True, False, False, False]])
    out = permute_tokens(x, spatial_mask=mask)
    # Non-masked tokens should remain unchanged.
    assert torch.equal(out[:, [0, 1, 5, 6, 7]], x[:, [0, 1, 5, 6, 7]])


def test_run_permutation_test_detects_drop():
    tokens = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
    target = torch.arange(12, dtype=torch.float32)

    def scoring_fn(batch):
        t = batch["vision_tokens"][0, :, 0]
        # Perfect ordering gives 1.0, permutations reduce score.
        return 1.0 - (t - target).abs().mean() / 12.0

    result = run_permutation_test(
        scoring_fn=scoring_fn,
        batch={"vision_tokens": tokens},
        num_permutations=32,
        seed=7,
    )
    assert result.baseline_score == 1.0
    assert result.permuted_mean < result.baseline_score
    assert result.relative_drop > 0.0
    assert result.num_permutations == 32
