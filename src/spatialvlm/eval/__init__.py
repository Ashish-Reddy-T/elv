"""Evaluation benchmarks, metrics, ablation orchestration."""

from .ablations import AblationOrchestrator, AblationResult, AblationSpec, default_ablation_specs
from .benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSpec,
    default_benchmark_suite,
    validate_primary_suite_is_indoor,
)
from .metrics import (
    MetricBundle,
    NavigationEpisodeResult,
    compute_metric_bundle,
    permutation_sensitivity_index,
    spl,
    success_rate,
    weighted_composite,
)
from .permutation_test import PermutationTestResult, permute_tokens, run_permutation_test

__all__ = [
    "NavigationEpisodeResult",
    "MetricBundle",
    "success_rate",
    "spl",
    "permutation_sensitivity_index",
    "weighted_composite",
    "compute_metric_bundle",
    "PermutationTestResult",
    "permute_tokens",
    "run_permutation_test",
    "BenchmarkSpec",
    "BenchmarkResult",
    "BenchmarkRunner",
    "default_benchmark_suite",
    "validate_primary_suite_is_indoor",
    "AblationSpec",
    "AblationResult",
    "AblationOrchestrator",
    "default_ablation_specs",
]
