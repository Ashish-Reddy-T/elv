"""Benchmark suite definitions and execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata for a benchmark target."""

    benchmark_id: str
    display_name: str
    task_family: str
    primary: bool
    indoor: bool
    requires_gt_depth: bool
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    """Single benchmark evaluation result."""

    benchmark_id: str
    score: float
    metrics: dict[str, float]
    metadata: dict[str, Any]


def default_benchmark_suite(include_supplementary: bool = False) -> list[BenchmarkSpec]:
    """Default suite with indoor-first primary evaluations."""
    primary = [
        BenchmarkSpec(
            benchmark_id="vlnce-r2r",
            display_name="VLN-CE R2R",
            task_family="navigation",
            primary=True,
            indoor=True,
            requires_gt_depth=True,
            notes="Primary Habitat benchmark.",
        ),
        BenchmarkSpec(
            benchmark_id="vlnce-rxr",
            display_name="VLN-CE RxR",
            task_family="navigation",
            primary=True,
            indoor=True,
            requires_gt_depth=True,
            notes="Primary Habitat benchmark.",
        ),
        BenchmarkSpec(
            benchmark_id="objectnav-hm3d",
            display_name="ObjectNav HM3D",
            task_family="navigation",
            primary=True,
            indoor=True,
            requires_gt_depth=True,
            notes="Primary Habitat benchmark.",
        ),
        BenchmarkSpec(
            benchmark_id="sqa3d",
            display_name="SQA3D",
            task_family="spatial_qa",
            primary=True,
            indoor=True,
            requires_gt_depth=True,
            notes="Indoor reconstructions with depth availability.",
        ),
        BenchmarkSpec(
            benchmark_id="vsi-bench",
            display_name="VSI-Bench",
            task_family="spatial_reasoning",
            primary=True,
            indoor=True,
            requires_gt_depth=True,
            notes="Primary spatial intelligence benchmark.",
        ),
        BenchmarkSpec(
            benchmark_id="navtrust",
            display_name="NavTrust",
            task_family="navigation",
            primary=True,
            indoor=True,
            requires_gt_depth=True,
            notes="Robustness and trustworthiness evaluation.",
        ),
    ]
    if not include_supplementary:
        return primary

    supplementary = [
        BenchmarkSpec(
            benchmark_id="whats-up",
            display_name="What's Up",
            task_family="spatial_reasoning",
            primary=False,
            indoor=False,
            requires_gt_depth=False,
            notes="Real-image benchmark: supplementary only for GT-depth pipeline.",
        ),
        BenchmarkSpec(
            benchmark_id="cv-bench",
            display_name="CV-Bench",
            task_family="vision_language",
            primary=False,
            indoor=False,
            requires_gt_depth=False,
            notes="General VLM benchmark: supplementary context.",
        ),
    ]
    return primary + supplementary


def validate_primary_suite_is_indoor(specs: Sequence[BenchmarkSpec]) -> bool:
    """Ensure primary benchmark rows remain aligned with indoor thesis."""
    primary = [s for s in specs if s.primary]
    return all(s.indoor for s in primary)


class BenchmarkRunner:
    """Registry-driven benchmark runner."""

    def __init__(
        self,
        specs: Sequence[BenchmarkSpec],
        evaluators: Mapping[str, Callable[[BenchmarkSpec], BenchmarkResult]],
    ) -> None:
        self.specs = list(specs)
        self.evaluators = dict(evaluators)

    def run(
        self,
        benchmark_ids: Sequence[str] | None = None,
    ) -> dict[str, BenchmarkResult]:
        selected = set(benchmark_ids) if benchmark_ids is not None else None
        out: dict[str, BenchmarkResult] = {}
        for spec in self.specs:
            if selected is not None and spec.benchmark_id not in selected:
                continue
            if spec.benchmark_id not in self.evaluators:
                raise KeyError(f"Missing evaluator for benchmark `{spec.benchmark_id}`.")
            result = self.evaluators[spec.benchmark_id](spec)
            out[spec.benchmark_id] = result
        return out
