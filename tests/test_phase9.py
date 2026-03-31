"""Tests for Phase 9 run specification helpers."""

from __future__ import annotations

from spatialvlm.eval.phase9 import (
    missing_phase9_runs,
    permutation_smoking_gun_pass,
    phase9_coverage_complete,
    phase9_run_specs,
)


def test_phase9_specs_include_all_todo_runs():
    specs = phase9_run_specs()
    ids = {s.run_id for s in specs}
    assert "full-model" in ids
    assert "permutation-test" in ids
    assert "dense-vs-sparse-rewards" in ids
    assert len(ids) == 16


def test_missing_phase9_runs_and_coverage():
    partial = {"full-model": {"score": 0.8}}
    missing = missing_phase9_runs(partial)
    assert "no-gatr" in missing
    assert not phase9_coverage_complete(partial)

    complete = {spec.run_id: {} for spec in phase9_run_specs()}
    assert missing_phase9_runs(complete) == []
    assert phase9_coverage_complete(complete)


def test_permutation_smoking_gun_pass():
    assert permutation_smoking_gun_pass(ours_relative_drop=0.2, baseline_relative_drop=0.01)
    assert not permutation_smoking_gun_pass(ours_relative_drop=0.1, baseline_relative_drop=0.01)
    assert not permutation_smoking_gun_pass(ours_relative_drop=0.2, baseline_relative_drop=0.05)
