"""Core evaluation metrics for navigation and spatial reasoning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationEpisodeResult:
    """Single navigation episode summary."""

    success: bool
    path_length: float
    shortest_path_length: float


@dataclass(frozen=True)
class MetricBundle:
    """Standardized metric tuple for experiment reporting."""

    success_rate: float
    spl: float
    permutation_sensitivity: float
    composite: float


def success_rate(successes: Sequence[bool]) -> float:
    if len(successes) == 0:
        return 0.0
    return sum(1.0 for s in successes if s) / float(len(successes))


def spl(episodes: Sequence[NavigationEpisodeResult], eps: float = 1e-8) -> float:
    """Compute Success weighted by Path Length (SPL)."""
    if len(episodes) == 0:
        return 0.0

    vals: list[float] = []
    for ep in episodes:
        if not ep.success:
            vals.append(0.0)
            continue
        denom = max(ep.path_length, ep.shortest_path_length, eps)
        vals.append(ep.shortest_path_length / denom)
    return sum(vals) / float(len(vals))


def permutation_sensitivity_index(
    baseline_score: float,
    permuted_score: float,
    eps: float = 1e-8,
) -> float:
    """Relative drop from baseline under token permutation.

    PSI = max(0, (baseline - permuted) / max(|baseline|, eps))
    """
    denom = max(abs(baseline_score), eps)
    return max(0.0, (baseline_score - permuted_score) / denom)


def weighted_composite(
    metrics: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted average of scalar metrics."""
    if len(metrics) == 0:
        return 0.0

    if weights is None:
        return sum(metrics.values()) / float(len(metrics))

    total_weight = 0.0
    weighted_sum = 0.0
    for key, value in metrics.items():
        w = float(weights.get(key, 0.0))
        if w <= 0.0:
            continue
        weighted_sum += w * float(value)
        total_weight += w
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def compute_metric_bundle(
    episodes: Sequence[NavigationEpisodeResult],
    baseline_score: float,
    permuted_score: float,
    composite_weights: Mapping[str, float] | None = None,
) -> MetricBundle:
    """Compute SR, SPL, PSI, and a weighted composite metric."""
    sr = success_rate([ep.success for ep in episodes])
    spl_score = spl(episodes)
    psi = permutation_sensitivity_index(
        baseline_score=baseline_score,
        permuted_score=permuted_score,
    )
    cmb = weighted_composite(
        metrics={
            "success_rate": sr,
            "spl": spl_score,
            "permutation_sensitivity": psi,
        },
        weights=composite_weights,
    )
    return MetricBundle(
        success_rate=sr,
        spl=spl_score,
        permutation_sensitivity=psi,
        composite=cmb,
    )
