"""Permutation diagnostic for spatial grounding claims."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PermutationTestResult:
    """Summary of permutation test outcomes."""

    baseline_score: float
    permuted_mean: float
    permuted_std: float
    absolute_drop: float
    relative_drop: float
    empirical_pvalue: float
    num_permutations: int

    def is_spatially_grounded(self, min_relative_drop: float = 0.15) -> bool:
        return self.relative_drop >= min_relative_drop


def permute_tokens(
    tokens: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Permute token order per sample.

    tokens: [B, N, D]
    spatial_mask: [B, N] (optional). If provided, only masked positions are permuted.
    """
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
    batch, num_tokens = tokens.shape[:2]
    out = tokens.clone()

    for b in range(batch):
        if spatial_mask is None:
            idx = torch.randperm(num_tokens, generator=generator, device=tokens.device)
            out[b] = out[b, idx]
            continue

        if spatial_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"spatial_mask must match [B,N], got {tuple(spatial_mask.shape)} vs "
                f"{tuple(tokens.shape[:2])}"
            )
        indices = torch.nonzero(spatial_mask[b], as_tuple=False).flatten()
        if indices.numel() <= 1:
            continue
        perm = indices[torch.randperm(indices.numel(), generator=generator, device=tokens.device)]
        out[b, indices] = tokens[b, perm]
    return out


def _as_float_score(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Metric function must return a scalar tensor or float.")
        return float(value.detach().cpu().item())
    return float(value)


def run_permutation_test(
    scoring_fn: Callable[[Mapping[str, Any]], torch.Tensor | float],
    batch: Mapping[str, Any],
    token_key: str = "vision_tokens",
    spatial_mask_key: str | None = None,
    num_permutations: int = 64,
    seed: int = 0,
    eps: float = 1e-8,
) -> PermutationTestResult:
    """Run baseline-vs-permuted score comparison."""
    if token_key not in batch:
        raise KeyError(f"Batch must contain `{token_key}` for permutation test.")
    if num_permutations <= 0:
        raise ValueError(f"num_permutations must be > 0, got {num_permutations}.")

    tokens = batch[token_key]
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Batch `{token_key}` must be a torch.Tensor.")

    generator = torch.Generator(device=tokens.device if tokens.is_cuda else torch.device("cpu"))
    generator.manual_seed(seed)

    baseline = _as_float_score(scoring_fn(batch))
    permuted_scores: list[float] = []

    for _ in range(num_permutations):
        permuted_batch = dict(batch)
        mask = None
        if spatial_mask_key is not None:
            mask = permuted_batch.get(spatial_mask_key)
            if mask is not None and not isinstance(mask, torch.Tensor):
                raise TypeError("spatial mask must be a torch.Tensor when provided.")
        permuted_batch[token_key] = permute_tokens(
            tokens=tokens,
            spatial_mask=mask,
            generator=generator,
        )
        permuted_scores.append(_as_float_score(scoring_fn(permuted_batch)))

    permuted = torch.tensor(permuted_scores, dtype=torch.float32)
    perm_mean = float(permuted.mean().item())
    perm_std = float(permuted.std(unbiased=False).item())
    abs_drop = baseline - perm_mean
    rel_drop = max(0.0, abs_drop / max(abs(baseline), eps))

    # One-sided empirical p-value for "permuted >= baseline" under null.
    count = int((permuted >= baseline).sum().item())
    pvalue = (count + 1.0) / (num_permutations + 1.0)

    return PermutationTestResult(
        baseline_score=baseline,
        permuted_mean=perm_mean,
        permuted_std=perm_std,
        absolute_drop=abs_drop,
        relative_drop=rel_drop,
        empirical_pvalue=pvalue,
        num_permutations=num_permutations,
    )
