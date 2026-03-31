"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from spatialvlm.eval.metrics import (
    NavigationEpisodeResult,
    compute_metric_bundle,
    permutation_sensitivity_index,
    spl,
    success_rate,
    weighted_composite,
)


def test_success_rate():
    assert success_rate([True, False, True, True]) == pytest.approx(0.75)


def test_spl():
    episodes = [
        NavigationEpisodeResult(success=True, path_length=10.0, shortest_path_length=8.0),
        NavigationEpisodeResult(success=False, path_length=5.0, shortest_path_length=4.0),
        NavigationEpisodeResult(success=True, path_length=6.0, shortest_path_length=6.0),
    ]
    score = spl(episodes)
    assert score == pytest.approx((0.8 + 0.0 + 1.0) / 3.0)


def test_permutation_sensitivity_index():
    psi = permutation_sensitivity_index(baseline_score=0.8, permuted_score=0.6)
    assert psi == pytest.approx(0.25)


def test_weighted_composite():
    cmb = weighted_composite(
        metrics={"a": 0.5, "b": 1.0},
        weights={"a": 1.0, "b": 3.0},
    )
    assert cmb == pytest.approx(0.875)


def test_compute_metric_bundle():
    episodes = [
        NavigationEpisodeResult(success=True, path_length=10.0, shortest_path_length=8.0),
        NavigationEpisodeResult(success=True, path_length=5.0, shortest_path_length=5.0),
    ]
    bundle = compute_metric_bundle(
        episodes=episodes,
        baseline_score=0.8,
        permuted_score=0.6,
        composite_weights={"success_rate": 0.4, "spl": 0.4, "permutation_sensitivity": 0.2},
    )
    assert bundle.success_rate == pytest.approx(1.0)
    assert bundle.spl == pytest.approx((0.8 + 1.0) / 2.0)
    assert bundle.permutation_sensitivity == pytest.approx(0.25)
    assert 0.0 <= bundle.composite <= 1.0
