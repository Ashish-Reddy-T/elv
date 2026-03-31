"""Tests for paper asset generation utilities."""

from __future__ import annotations

import json
from pathlib import Path

from spatialvlm.eval.paper_assets import (
    load_phase9_results,
    render_ablation_table_tex,
    render_main_results_table_tex,
    write_paper_assets,
)


def _sample_runs() -> dict[str, dict[str, object]]:
    return {
        "full-model": {
            "title": "Full model",
            "score": 0.8123,
            "delta_vs_full_model": 0.0,
        },
        "no-gatr": {
            "title": "No GATr",
            "score": 0.7311,
            "delta_vs_full_model": -0.0812,
        },
        "permutation-test": {
            "title": "Permutation test",
            "score": None,
            "relative_drop": 0.17,
            "baseline_relative_drop": 0.02,
        },
    }


def test_load_phase9_results(tmp_path: Path):
    payload = {"runs": _sample_runs()}
    f = tmp_path / "results.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    runs = load_phase9_results(f)
    assert "full-model" in runs
    assert runs["no-gatr"]["score"] == 0.7311


def test_render_tables():
    runs = _sample_runs()
    ablation_tex = render_ablation_table_tex(runs)
    main_tex = render_main_results_table_tex(runs)
    assert "No GATr" in ablation_tex
    assert "SpatialVLM (Full)" in main_tex


def test_write_paper_assets(tmp_path: Path):
    paper_dir = tmp_path / "paper"
    write_paper_assets(_sample_runs(), paper_dir)
    assert (paper_dir / "tables" / "ablation_results.tex").exists()
    assert (paper_dir / "tables" / "main_results.tex").exists()
    assert (paper_dir / "figures" / "permutation_curve.csv").exists()
