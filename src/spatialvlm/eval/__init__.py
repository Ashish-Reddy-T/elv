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
from .paper_assets import (
    load_phase9_results,
    render_ablation_table_tex,
    render_main_results_table_tex,
    write_paper_assets,
    write_permutation_csv,
)
from .permutation_test import PermutationTestResult, permute_tokens, run_permutation_test
from .phase9 import (
    Phase9RunSpec,
    missing_phase9_runs,
    permutation_smoking_gun_pass,
    phase9_coverage_complete,
    phase9_run_specs,
)

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
    "Phase9RunSpec",
    "phase9_run_specs",
    "missing_phase9_runs",
    "phase9_coverage_complete",
    "permutation_smoking_gun_pass",
    "load_phase9_results",
    "render_ablation_table_tex",
    "render_main_results_table_tex",
    "write_paper_assets",
    "write_permutation_csv",
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
