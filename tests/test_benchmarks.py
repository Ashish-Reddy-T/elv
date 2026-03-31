"""Tests for benchmark registry and suite defaults."""

from __future__ import annotations

from spatialvlm.eval.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    default_benchmark_suite,
    validate_primary_suite_is_indoor,
)


def test_default_suite_is_indoor_primary_only():
    suite = default_benchmark_suite(include_supplementary=False)
    ids = {x.benchmark_id for x in suite}
    assert "whats-up" not in ids
    assert validate_primary_suite_is_indoor(suite)
    assert all(x.primary for x in suite)


def test_default_suite_with_supplementary_includes_whats_up():
    suite = default_benchmark_suite(include_supplementary=True)
    ids = {x.benchmark_id for x in suite}
    assert "whats-up" in ids
    assert "cv-bench" in ids


def test_benchmark_runner_executes_registered_functions():
    suite = default_benchmark_suite(include_supplementary=False)

    def make_eval(spec):
        return BenchmarkResult(
            benchmark_id=spec.benchmark_id,
            score=0.5,
            metrics={"score": 0.5},
            metadata={"name": spec.display_name},
        )

    evaluators = {spec.benchmark_id: make_eval for spec in suite}
    runner = BenchmarkRunner(specs=suite, evaluators=evaluators)
    out = runner.run()
    assert len(out) == len(suite)
    assert all(v.score == 0.5 for v in out.values())
